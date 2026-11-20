"""
Score the SEAS5 rollout arms against the ParFlow truth and against each other.

THE POINT IS THE LADDER, NOT ANY ONE ARM'S NUMBER. Each arm differs from the one
above it in exactly one respect, so the DROP between consecutive arms prices
that one thing:

  A   NLDAS-2 at 1 km                     emulator error alone
  B      "     coarsened to 1 deg         -> the gap A-B is the RESOLUTION LOSS
  B2  ERA5 through the whole chain        -> the gap B-B2 is the CORRECTION CHAIN
  C   SEAS5 bias-corrected                -> the gap B2-C is FORECAST ERROR
  D   SEAS5 raw                           -> the gap C-D is what BIAS CORRECTION BUYS
  E   NLDAS-2 climatology                 the no-forecast-information reference

Reported with two references, because they answer different questions:
  * vs PERSISTENCE -- holds the day-0 state for 210 days. The right yardstick for
    a short forecast of a slow field, and the one the published report uses.
  * vs CLIMATOLOGY (arm E) -- the right yardstick for a SEASONAL forecast.
    Beating persistence over seven months is nearly automatic for a field with a
    seasonal cycle; beating climatology is the actual claim. An arm can be well
    ahead of persistence and still behind climatology, and that distinction is
    the headline result of this experiment, so do not report only the first.

CRPS uses the sorted-member identity rather than the O(M^2) double loop in
analyze_ensemble.py:
    sum_ij |x_i - x_j| = 2 * sum_i (2i - M + 1) * x_(i)      (x sorted ascending)
which makes the fair estimator O(M log M). With 100 members over 210 days and
4.2e5 cells the double loop is ~4e11 operations; this is ~1e9. The identity is
verified against analyze_ensemble.crps_fair at import time by `_selftest`, so a
mistake in it cannot pass silently.

SPREAD-TO-ERROR IS THE SECOND RESULT. The emulator is under-dispersed
(spread-to-error 0.43 with sampler noise as its only perturbation). Arms C and D
add a 25-member FORCING ensemble -- the physically correct missing source. The
decomposition printed here separates the two contributions by comparing the
spread within one forcing member (diffusion only) against the spread across the
full ensemble, which is why run_seas5_rollout.py keeps the two member dimensions
distinct on disk.

Usage:
    python analyze_seas5.py --arms B E --out-dir ../report_seas5/figures
    python analyze_seas5.py --arms B B2 C D E --arm-a Results/groundwater_f15_200ep_fold1/rollout
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

import viz_style as vs
from analyze_ensemble import crps_fair, rank_of_truth, VARS, PRETTY, UNIT, DATA, RAW

ARM_ORDER = ["A", "B", "B2", "C", "D", "E"]
ARM_LABEL = {
    "A": "A  NLDAS-2 at 1 km (emulator error alone)",
    "B": "B  NLDAS-2 coarsened to the SEAS5 grid (+ resolution loss)",
    "B2": "B2  ERA5 through the full chain (+ correction chain)",
    "C": "C  SEAS5 bias-corrected (+ forecast error)",
    "D": "D  SEAS5 raw (no bias correction)",
    "E": "E  NLDAS-2 climatology (no forecast information)",
}
ARM_COLOR = {"A": vs.C_ACTUAL, "B": vs.C_F10, "B2": "#2E8B57",
             "C": vs.C_PRED, "D": "#8E44AD", "E": vs.C_PERSIST}


def crps_fair_sorted(ens, obs):
    """Fair CRPS, O(M log M). ens (M, ...), obs (...) -> obs's shape."""
    M = ens.shape[0]
    if M < 2:
        return np.abs(ens[0] - obs)
    t1 = np.abs(ens - obs[None]).mean(axis=0)
    s = np.sort(ens, axis=0)
    w = (2 * np.arange(M) - M + 1).reshape((M,) + (1,) * (ens.ndim - 1))
    return t1 - (s * w).sum(axis=0) / (M * (M - 1))


def _selftest():
    rng = np.random.default_rng(0)
    for M in (2, 5, 16):
        e = rng.normal(size=(M, 7, 3))
        o = rng.normal(size=(7, 3))
        a, b = crps_fair(e, o), crps_fair_sorted(e, o)
        if not np.allclose(a, b, atol=1e-10):
            raise AssertionError(f"CRPS identity disagrees at M={M}: "
                                 f"max |diff| {np.abs(a - b).max():.3e}")


_selftest()


def load_arm(arm, out_root, arm_a_dir=None):
    """-> {var: (M_total, n_days, H, W) float32}, global_days, per-forcing-member
    member counts. Arm A comes from the existing ensemble rollout."""
    if arm == "A":
        if not arm_a_dir:
            return None
        p = os.path.join(arm_a_dir, "ensemble_state.npz")
        if not os.path.exists(p):
            return None
        d = np.load(p)
        gd = d["global_days"].astype(int)
        return ({v: d[v].astype(np.float32) for v in VARS}, gd, [d[VARS[0]].shape[0]])
    files = sorted(glob.glob(os.path.join(out_root, f"arm{arm}", "member_*.npz")))
    if not files:
        return None
    gd, per, stacks = None, [], {v: [] for v in VARS}
    for f in files:
        d = np.load(f)
        g = d["global_days"].astype(int)
        gd = g if gd is None else gd
        if not np.array_equal(g, gd):
            raise SystemExit(f"{f} spans different days than its siblings")
        per.append(d[VARS[0]].shape[0])
        for v in VARS:
            stacks[v].append(d[v].astype(np.float32))
    return ({v: np.concatenate(stacks[v], axis=0) for v in VARS}, gd, per)


def truth_for(gdays, test_year):
    day_index = json.load(open(os.path.join(DATA, "tiles", "day_index.json")))
    day0 = min(i for i, e in enumerate(day_index) if e["water_year"] == test_year)
    t2 = xr.open_dataset(os.path.join(RAW, f"target2d_wy{test_year}.nc"))
    out = {v: t2[v].values[[g - day0 for g in gdays]].astype(np.float32)
           for v in VARS}
    t2.close()
    dates = [day_index[int(g)]["date"] for g in gdays]
    return out, dates


def rmse_t(a, y):
    return np.sqrt(np.nanmean((a - y) ** 2, axis=(1, 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["B", "E"], choices=ARM_ORDER)
    ap.add_argument("--rollout-root", default="Results/seas5")
    ap.add_argument("--arm-a", default="Results/groundwater_f15_200ep_fold1/rollout")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--test-year", type=int, default=2006)
    args = ap.parse_args()
    vs.apply_style()
    os.makedirs(args.out_dir, exist_ok=True)

    arms = [a for a in ARM_ORDER if a in args.arms]
    loaded = {}
    for a in arms:
        got = load_arm(a, args.rollout_root, args.arm_a)
        if got is None:
            print(f"arm {a}: no output found, skipping")
            continue
        loaded[a] = got
        print(f"arm {a}: {got[0][VARS[0]].shape[0]} total members "
              f"({len(got[2])} forcing x {got[2][0]} diffusion), "
              f"{len(got[1])} days")
    if not loaded:
        raise SystemExit("no arm output found; run the rollouts first")

    # Every arm is scored on the common day window -- arm A's 360-day rollout is
    # truncated to the 210 days one SEAS5 initialisation reaches, so the ladder
    # compares like with like.
    common = sorted(set.intersection(*[set(v[1].tolist()) for v in loaded.values()]))
    common = np.array(common)
    print(f"\ncommon day window: {len(common)} days")
    truth, dates = truth_for(common, args.test_year)

    summary = {"test_year": args.test_year, "n_days": len(common),
               "arms": {}, "ladder": {}}
    curves = {}

    clim_mean = None
    if "E" in loaded:
        f, gd, _ = loaded["E"]
        sel = np.searchsorted(gd, common)
        clim_mean = {v: f[v][:, sel].mean(0) for v in VARS}

    for a in arms:
        if a not in loaded:
            continue
        fields, gd, per = loaded[a]
        sel = np.searchsorted(gd, common)
        entry, cv = {}, {}
        for v in VARS:
            ens = fields[v][:, sel]                       # (M, n_days, H, W)
            y = truth[v]
            mean = ens.mean(0)
            pers = np.broadcast_to(y[0], y.shape)
            r_mean, r_pers = rmse_t(mean, y), rmse_t(pers, y)
            sd = ens.std(0, ddof=1) if ens.shape[0] > 1 else np.zeros_like(mean)

            crps = np.empty(len(common), dtype=np.float64)
            for j in range(len(common)):
                crps[j] = np.nanmean(crps_fair_sorted(ens[:, j], y[j]))
            mae_pers = np.nanmean(np.abs(pers - y), axis=(1, 2))

            e = {"rmse_ensemble_mean": float(np.sqrt(np.nanmean((mean - y) ** 2))),
                 "rmse_persistence": float(np.sqrt(np.nanmean((pers - y) ** 2))),
                 "crps": float(crps.mean()),
                 "mae_persistence": float(mae_pers.mean()),
                 "mean_spread": float(np.nanmean(sd)),
                 "n_members": int(ens.shape[0])}
            e["skill_vs_persistence"] = 1 - e["rmse_ensemble_mean"] / e["rmse_persistence"]
            e["crps_skill_vs_persistence"] = 1 - e["crps"] / e["mae_persistence"]
            e["spread_to_error"] = e["mean_spread"] / e["rmse_ensemble_mean"]
            if clim_mean is not None:
                rc = float(np.sqrt(np.nanmean((clim_mean[v] - y) ** 2)))
                e["rmse_climatology"] = rc
                e["skill_vs_climatology"] = 1 - e["rmse_ensemble_mean"] / rc
            # Spread decomposition: within one forcing member the only
            # perturbation is sampler noise; across the full ensemble the forcing
            # ensemble is added. The ratio says which source dominates.
            if len(per) > 1:
                k = per[0]
                within = np.mean([fields[v][i * k:(i + 1) * k][:, sel].std(0, ddof=1).mean()
                                  for i in range(len(per))])
                e["spread_within_forcing_member"] = float(within)
                e["spread_total"] = e["mean_spread"]
                e["forcing_share_of_variance"] = float(
                    max(0.0, 1 - (within ** 2) / max(e["mean_spread"] ** 2, 1e-12)))
            entry[v] = e
            cv[v] = {"rmse": r_mean, "rmse_persistence": r_pers, "crps": crps,
                     "spread": np.nanmean(sd, axis=(1, 2))}
            print(f"  arm {a} {v:18s} RMSE {e['rmse_ensemble_mean']:.4f} "
                  f"| skill vs pers {e['skill_vs_persistence']:+.3f}"
                  + (f" | vs clim {e['skill_vs_climatology']:+.3f}"
                     if "skill_vs_climatology" in e else "")
                  + f" | CRPS {e['crps']:.4f} | spread/err {e['spread_to_error']:.2f}")
        summary["arms"][a] = entry
        curves[a] = cv

    # the ladder: the drop between consecutive arms is what each one prices
    gaps = [("A", "B", "resolution loss"), ("B", "B2", "correction chain"),
            ("B2", "C", "forecast error"), ("C", "D", "value of bias correction")]
    for lo, hi, what in gaps:
        if lo in summary["arms"] and hi in summary["arms"]:
            summary["ladder"][f"{lo}->{hi}"] = {
                "prices": what,
                **{v: {"d_rmse": summary["arms"][hi][v]["rmse_ensemble_mean"]
                               - summary["arms"][lo][v]["rmse_ensemble_mean"],
                       "d_skill": summary["arms"][hi][v]["skill_vs_persistence"]
                                - summary["arms"][lo][v]["skill_vs_persistence"]}
                   for v in VARS}}

    for name, key, ylab in [("seas5_ladder_rmse", "rmse", "RMSE"),
                            ("seas5_ladder_crps", "crps", "CRPS")]:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=140, sharex=True)
        T = np.arange(len(common))
        for ax, v in zip(axes, VARS):
            for a in arms:
                if a not in curves:
                    continue
                ax.plot(T, curves[a][v][key], color=ARM_COLOR[a],
                        linewidth=vs.LW_MAIN if a in ("A", "C") else vs.LW_SECONDARY,
                        linestyle="--" if a in ("D", "E") else "-",
                        label=ARM_LABEL[a])
            if key == "rmse":
                any_arm = next(iter(curves))
                ax.plot(T, curves[any_arm][v]["rmse_persistence"], color="k",
                        linewidth=vs.LW_FAINT + 0.8, linestyle=":",
                        label="persistence (day-0 state held)")
            vs.style_axes(ax, ylabel=f"{PRETTY[v]}\n{ylab} ({UNIT[v]})")
        ti = np.linspace(0, len(common) - 1, 8).astype(int)
        axes[-1].set_xticks(ti)
        axes[-1].set_xticklabels([dates[i] for i in ti], rotation=30, ha="right")
        h, l = axes[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.055),
                   ncol=2, frameon=False, fontsize=vs.FS_LEGEND - 2)
        fig.suptitle(f"SEAS5 attribution ladder -- {ylab} by lead day, "
                     f"WY{args.test_year}, fold 1", fontsize=vs.FS_SUPTITLE)
        fig.subplots_adjust(top=0.92, bottom=0.26, hspace=0.12,
                            left=0.08, right=0.98)
        vs.save_both(fig, os.path.join(args.out_dir, name))
        plt.close(fig)
        print(f"wrote {name}.pdf and .png")

    with open(os.path.join(args.out_dir, "seas5_summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)

    lines = ["arm,variable,rmse,skill_vs_persistence,skill_vs_climatology,"
             "crps,crps_skill_vs_persistence,spread_to_error,n_members"]
    for a in arms:
        if a not in summary["arms"]:
            continue
        for v in VARS:
            e = summary["arms"][a][v]
            lines.append(f"{a},{v},{e['rmse_ensemble_mean']:.5f},"
                         f"{e['skill_vs_persistence']:+.4f},"
                         f"{e.get('skill_vs_climatology', float('nan')):+.4f},"
                         f"{e['crps']:.5f},{e['crps_skill_vs_persistence']:+.4f},"
                         f"{e['spread_to_error']:.3f},{e['n_members']}")
    with open(os.path.join(args.out_dir, "seas5_ladder.csv"), "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n== the ladder ==")
    for k, g in summary["ladder"].items():
        d = g[VARS[0]]["d_rmse"]
        print(f"  {k:7s} prices {g['prices']:24s} "
              f"d(WTD RMSE) {d:+.4f} m, "
              f"d(SM RMSE) {g[VARS[1]]['d_rmse']:+.5f}")
    print(f"\nwrote {args.out_dir}/seas5_summary.json and seas5_ladder.csv")


if __name__ == "__main__":
    main()
