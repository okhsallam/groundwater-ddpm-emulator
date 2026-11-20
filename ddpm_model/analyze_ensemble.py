"""
Probabilistic verification of the ensemble rollout.

A deterministic rollout can only be scored on how close one trajectory is to the
truth. An ensemble can be asked the questions that actually matter for a
forecast: is the mean better than any single draw, does the spread grow with lead
time, and is the spread an honest statement of the error?

Scores computed here:

  ENSEMBLE MEAN RMSE vs. a single member vs. persistence. Averaging independent
  draws cancels noise, so the mean should beat a single member; by how much says
  how much of the single-member error was sampling noise rather than bias.

  CRPS, the continuous ranked probability score. The probabilistic analogue of
  MAE: it reduces to MAE when the ensemble collapses to a point, so a
  deterministic forecast can be scored on the same axis and the comparison is
  fair. Computed with the fair (unbiased) estimator,
      CRPS = mean|x_i - y| - 1/(2 M (M-1)) * sum_ij |x_i - x_j|,
  which removes the small-ensemble bias of the plug-in form.

  SPREAD-SKILL. For a calibrated ensemble the RMSE of the mean should match the
  ensemble standard deviation (up to the sqrt((M+1)/M) small-ensemble factor).
  Spread below error means overconfidence, the usual failure mode.

  RANK HISTOGRAM. Where the truth falls among the sorted members. Flat means
  calibrated; U-shaped means under-dispersed; dome-shaped means over-dispersed.

Usage:
    python analyze_ensemble.py --rollout-dir ../data/rollout_wy2006_fstr_f15_ens \
        --out-dir ../data/rollout_wy2006_fstr_f15_ens/analysis
"""
import argparse
import json
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import viz_style as vs

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "data")
RAW = os.path.join(DATA, "parflow_raw")
VARS = ("water_table_depth", "soil_moisture")
PRETTY = {"water_table_depth": "Water table depth", "soil_moisture": "Soil moisture"}
UNIT = {"water_table_depth": "m", "soil_moisture": "-"}
SUB = 6          # spatial stride for the O(M^2) and rank statistics


def crps_fair(ens, obs):
    """Fair CRPS. ens: (M, ...), obs: (...). Returns the same shape as obs."""
    M = ens.shape[0]
    t1 = np.abs(ens - obs[None]).mean(axis=0)
    t2 = np.zeros_like(t1)
    for i in range(M):                       # O(M^2) but M is small
        t2 += np.abs(ens[i][None] - ens).sum(axis=0)
    return t1 - t2 / (2.0 * M * (M - 1))


def rank_of_truth(ens, obs):
    """Index of the truth among the members, 0..M. Ties broken at random."""
    M = ens.shape[0]
    below = (ens < obs[None]).sum(axis=0)
    ties = (ens == obs[None]).sum(axis=0)
    jitter = (np.random.random(below.shape) * (ties + 1)).astype(int)
    return np.clip(below + jitter, 0, M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    vs.apply_style()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(0)

    d = np.load(os.path.join(args.rollout_dir, "ensemble_state.npz"))
    meta = json.loads(open(os.path.join(args.rollout_dir, "rollout_meta.json")).read())
    gd = d["global_days"].tolist()
    M = d["water_table_depth"].shape[0]
    label = args.label or f"FSTR split, {meta['future_len']}-day window, {M} members"

    day_index = json.loads(open(os.path.join(DATA, "tiles", "day_index.json")).read())
    yr = meta["test_year"]
    day0 = min(i for i, e in enumerate(day_index) if e["water_year"] == yr)
    dates = [day_index[int(g)]["date"] for g in gd]
    t2 = xr.open_dataset(os.path.join(RAW, f"target2d_wy{yr}.nc"))
    truth = {v: t2[v].values[[g - day0 for g in gd]] for v in VARS}
    t2.close()

    summary = {"label": label, "n_members": M, "test_year": yr,
               "future_len": meta["future_len"], "clamped": meta["clamped"]}

    # ---- 1. ensemble mean vs single member vs persistence -----------------
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=140, sharex=True,
                             constrained_layout=True)
    T = np.arange(len(dates))
    for ax, var in zip(axes, VARS):
        y = truth[var]
        ens = d[var]
        mean = ens.mean(0)
        pers = np.broadcast_to(y[0], y.shape)
        r = lambda a: np.sqrt(np.nanmean((a - y) ** 2, axis=(1, 2)))
        r_mean, r_pers = r(mean), r(pers)
        r_mem = np.stack([r(ens[m]) for m in range(M)])

        ax.fill_between(T, r_mem.min(0), r_mem.max(0), color=vs.C_F15, alpha=0.18,
                        label=f"individual members (range of {M})")
        ax.plot(T, r_mem.mean(0), color=vs.C_F15, linewidth=vs.LW_SECONDARY,
                linestyle=":", label="mean over members of member RMSE")
        ax.plot(T, r_mean, color=vs.C_PRED, linewidth=vs.LW_MAIN + 0.4,
                label="RMSE of the ENSEMBLE MEAN")
        ax.plot(T, r_pers, color=vs.C_PERSIST, linewidth=vs.LW_SECONDARY,
                linestyle="--", label="persistence")
        vs.style_axes(ax, ylabel=f"{PRETTY[var]}\nRMSE ({UNIT[var]})")
        ax.legend(fontsize=vs.FS_LEGEND - 1, frameon=False, loc="upper left")
        ax.set_ylim(0, r_pers.max() * 1.35)

        s = summary.setdefault(var, {})
        s.update(rmse_ensemble_mean=float(np.sqrt(np.nanmean((mean - y) ** 2))),
                 rmse_single_member_mean=float(np.mean([
                     np.sqrt(np.nanmean((ens[m] - y) ** 2)) for m in range(M)])),
                 rmse_persistence=float(np.sqrt(np.nanmean((pers - y) ** 2))))
        s["skill_ensemble_mean"] = 1 - s["rmse_ensemble_mean"] / s["rmse_persistence"]
        s["skill_single_member"] = 1 - s["rmse_single_member_mean"] / s["rmse_persistence"]
    ti = np.linspace(0, len(dates) - 1, 8).astype(int)
    axes[-1].set_xticks(ti)
    axes[-1].set_xticklabels([dates[i] for i in ti], rotation=25, ha="right")
    axes[-1].set_xlabel("date")
    fig.suptitle(f"Ensemble mean against individual members and persistence -- {label}\n"
                 "averaging independent draws cancels sampling noise; the gap to the "
                 "member band is how much of the single-run error was noise",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    plt.savefig(os.path.join(args.out_dir, "ens_mean_vs_members.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 2. spread growth and spread-skill --------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=140, constrained_layout=True)
    for col, var in enumerate(VARS):
        y, ens = truth[var], d[var]
        sd = ens.std(0, ddof=1)
        err = ens.mean(0) - y
        sd_t = np.sqrt(np.nanmean(sd ** 2, axis=(1, 2)))
        rmse_t = np.sqrt(np.nanmean(err ** 2, axis=(1, 2)))
        # a calibrated M-member ensemble satisfies RMSE = sqrt((M+1)/M) * spread
        cal = np.sqrt((M + 1) / M)

        ax = axes[0, col]
        ax.plot(T, rmse_t, color=vs.C_PRED, linewidth=vs.LW_MAIN,
                label="RMSE of ensemble mean")
        ax.plot(T, cal * sd_t, color=vs.C_F10, linewidth=vs.LW_MAIN, linestyle="--",
                label=r"ensemble spread $\times\sqrt{(M{+}1)/M}$")
        vs.style_axes(ax, ylabel=f"{UNIT[var]}", title=PRETTY[var])
        ax.set_xticks(ti)
        ax.set_xticklabels([dates[i][:7] for i in ti], rotation=25, ha="right")
        ax.legend(fontsize=vs.FS_LEGEND - 1, frameon=False)
        ax.set_ylim(bottom=0)

        # binned spread-skill: does a wider forecast actually mean a worse one?
        ax = axes[1, col]
        f_sd = sd[:, ::SUB, ::SUB].ravel()
        f_er = np.abs(err[:, ::SUB, ::SUB]).ravel()
        ok = np.isfinite(f_sd) & np.isfinite(f_er)
        f_sd, f_er = f_sd[ok], f_er[ok]
        qs = np.quantile(f_sd, np.linspace(0, 1, 11))
        cx, cy = [], []
        for lo, hi in zip(qs[:-1], qs[1:]):
            m = (f_sd >= lo) & (f_sd < hi)
            if m.sum() > 50:
                cx.append(f_sd[m].mean())
                cy.append(np.sqrt((f_er[m] ** 2).mean()))
        ax.plot(cx, cy, color=vs.C_PRED, linewidth=vs.LW_MAIN, marker="o",
                markersize=11, markeredgecolor="white", markeredgewidth=1.8,
                label="binned by forecast spread")
        lim = max(max(cx), max(cy))
        ax.plot([0, lim], [0, lim], color="#333333", linewidth=2.0, linestyle="--",
                label="perfect calibration (1:1)")
        vs.style_axes(ax, xlabel=f"ensemble spread ({UNIT[var]})",
                      ylabel=f"RMSE of ensemble mean ({UNIT[var]})")
        ax.legend(fontsize=vs.FS_LEGEND - 1, frameon=False, loc="upper left")
        summary[var]["mean_spread"] = float(np.nanmean(sd))
        summary[var]["spread_skill_ratio"] = float(cal * np.nanmean(sd_t) / np.nanmean(rmse_t))
    fig.suptitle(f"Is the spread an honest statement of the error?  {label}\n"
                 "top: growth through the rollout.  bottom: binned spread against "
                 "realised error -- points below the 1:1 line mean the ensemble is "
                 "overconfident",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    plt.savefig(os.path.join(args.out_dir, "ens_spread_skill.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- 3. CRPS and rank histogram ---------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 11), dpi=140, constrained_layout=True)
    for col, var in enumerate(VARS):
        y, ens = truth[var], d[var]
        ys, es = y[:, ::SUB, ::SUB], ens[:, :, ::SUB, ::SUB]
        crps_t = np.nanmean(crps_fair(es, ys), axis=(1, 2))
        # a deterministic forecast scored on the same axis: CRPS reduces to MAE
        mae_mean = np.nanmean(np.abs(es.mean(0) - ys), axis=(1, 2))
        mae_pers = np.nanmean(np.abs(np.broadcast_to(ys[0], ys.shape) - ys), axis=(1, 2))

        ax = axes[0, col]
        ax.plot(T, crps_t, color=vs.C_PRED, linewidth=vs.LW_MAIN, label="ensemble CRPS")
        ax.plot(T, mae_mean, color=vs.C_F10, linewidth=vs.LW_SECONDARY, linestyle="--",
                label="MAE of ensemble mean (deterministic)")
        ax.plot(T, mae_pers, color=vs.C_PERSIST, linewidth=vs.LW_SECONDARY,
                linestyle=":", label="MAE of persistence")
        vs.style_axes(ax, ylabel=f"{UNIT[var]}", title=PRETTY[var])
        ax.set_xticks(ti)
        ax.set_xticklabels([dates[i][:7] for i in ti], rotation=25, ha="right")
        ax.legend(fontsize=vs.FS_LEGEND - 1, frameon=False)
        ax.set_ylim(bottom=0)

        ax = axes[1, col]
        np.random.seed(0)
        ranks = rank_of_truth(es, ys).ravel()
        ax.hist(ranks, bins=np.arange(M + 2) - 0.5, color=vs.C_F15,
                edgecolor="white", linewidth=1.2, density=True)
        ax.axhline(1.0 / (M + 1), color="#333333", linewidth=2.4, linestyle="--",
                   label="flat = calibrated")
        vs.style_axes(ax, xlabel="rank of the truth among the members",
                      ylabel="frequency")
        ax.legend(fontsize=vs.FS_LEGEND - 1, frameon=False)
        summary[var]["crps_mean"] = float(np.nanmean(crps_t))
        summary[var]["mae_ensemble_mean"] = float(np.nanmean(mae_mean))
        summary[var]["mae_persistence"] = float(np.nanmean(mae_pers))
        summary[var]["crps_skill_vs_persistence"] = float(
            1 - np.nanmean(crps_t) / np.nanmean(mae_pers))
        edges = np.histogram(ranks, bins=np.arange(M + 2) - 0.5, density=True)[0]
        summary[var]["rank_hist_deviation"] = float(
            np.abs(edges - 1.0 / (M + 1)).sum() / 2)
    fig.suptitle(f"Probabilistic scores -- {label}\n"
                 "CRPS reduces to MAE for a point forecast, so the two are directly "
                 "comparable.  A flat rank histogram means the truth is "
                 "indistinguishable from a member",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    plt.savefig(os.path.join(args.out_dir, "ens_crps_rank.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)

    json.dump(summary, open(os.path.join(args.out_dir, "ensemble_summary.json"), "w"),
              indent=2)
    print(f"wrote ensemble analysis -> {args.out_dir}\n")
    for var in VARS:
        s = summary[var]
        print(f"{var}:")
        print(f"  RMSE  ensemble mean {s['rmse_ensemble_mean']:.4f} | "
              f"single member {s['rmse_single_member_mean']:.4f} | "
              f"persistence {s['rmse_persistence']:.4f}")
        print(f"  skill ensemble mean {s['skill_ensemble_mean']:+.4f} | "
              f"single member {s['skill_single_member']:+.4f}")
        print(f"  CRPS {s['crps_mean']:.4f} vs persistence MAE {s['mae_persistence']:.4f} "
              f"-> CRPS skill {s['crps_skill_vs_persistence']:+.4f}")
        print(f"  spread/error ratio {s['spread_skill_ratio']:.3f} "
              f"({'under-dispersed' if s['spread_skill_ratio'] < 0.9 else 'over-dispersed' if s['spread_skill_ratio'] > 1.1 else 'well calibrated'})")


if __name__ == "__main__":
    main()
