"""
Main-report-style figure suite for the SEAS5-forced rollouts, across all folds.

Mirrors the conventions of make_refined_figures.py so the two reports can sit
side by side: north-up maps via viz_style, blue for truth, orange for the
forecast-driven run, grey for the references, and relative error leading the
cross-fold comparison because absolute RMSE cannot compare metres with a
dimensionless fraction, nor test years that differ in how much the state moves.

One fold is held in memory at a time. Forty members x 210 days x 637 x 665 is
14 GB per fold in float32, so the per-fold pass reduces to daily statistics and
a few map snapshots before releasing the array.

Usage:
    python make_seas5_figures.py --arm Dp --out-dir ../report_seas5/figures
"""
import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from matplotlib.colors import ListedColormap, BoundaryNorm

import viz_style as vs
import plot_fold_evolution as evo
from analyze_seas5 import crps_fair_sorted
from analyze_ensemble import VARS, PRETTY, UNIT, DATA, RAW

FOLD_WY = {1: 2006, 2: 2005, 3: 2004, 4: 2003}
FOLD_TRAIN = {1: [2003, 2004, 2005], 2: [2003, 2004, 2006],
              3: [2003, 2005, 2006], 4: [2004, 2005, 2006]}
# Chronological, matching every other multi-year figure in this study.
PLOT_ORDER = sorted(FOLD_WY, key=lambda f: FOLD_WY[f])
MAP_DAYS = [14, 69, 139, 209]          # ~month 1, 3, 5, 7 of the forecast
CAL = {f"{m:02d}": a for m, a in zip(range(1, 13),
       ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))}


ARM_LABEL = {
    "D":  "raw SEAS5",
    "C":  "bias-corrected SEAS5",
    "B":  "reanalysis at SEAS5 resolution",
    "B2": "reanalysis, full correction chain",
    "E":  "climatological forcing",
}


def arm_label(arm):
    """What the reader sees. The arm letter is an internal path component."""
    return ARM_LABEL.get(arm, f"SEAS5 arm {arm}")


def monlab(d):
    """"2002-10-05" -> "Oct 02", matching the Part I convention."""
    return f"{CAL[d[5:7]]} {d[2:4]}"


def load_fold(root, arm, fold):
    d = os.path.join(root, f"arm{arm}_fold{fold}")
    files = sorted(glob.glob(os.path.join(d, "member_*.npz")))
    if not files:
        return None
    gd, parts = None, {v: [] for v in VARS}
    for f in files:
        z = np.load(f)
        g = z["global_days"].astype(int)
        gd = g if gd is None else gd
        for v in VARS:
            parts[v].append(z[v].astype(np.float32))
    return {v: np.concatenate(parts[v], 0) for v in VARS}, gd


def references(fold, gd):
    di = json.load(open(os.path.join(DATA, "tiles", "day_index.json")))
    wy = FOLD_WY[fold]
    day0 = min(i for i, e in enumerate(di) if e["water_year"] == wy)
    t2 = xr.open_dataset(os.path.join(RAW, f"target2d_wy{wy}.nc"))
    truth = {v: t2[v].values[[g - day0 for g in gd]].astype(np.float32) for v in VARS}
    t2.close()
    doy = pd.to_datetime([di[int(g)]["date"] for g in gd]).dayofyear.values
    acc = {v: np.zeros((len(gd),) + truth[VARS[0]].shape[1:]) for v in VARS}
    cnt = np.zeros(len(gd))
    for y in FOLD_TRAIN[fold]:
        ds = xr.open_dataset(os.path.join(RAW, f"target2d_wy{y}.nc"))
        pos = {d: i for i, d in enumerate(pd.DatetimeIndex(ds["time"].values).dayofyear.values)}
        idx = [pos.get(d) for d in doy]
        ok = np.array([i is not None for i in idx])
        sel = np.array([i if i is not None else 0 for i in idx])
        for v in VARS:
            acc[v][ok] += ds[v].values[sel][ok]
        cnt += ok
        ds.close()
    # Day-of-year 366 exists in a leap verifying year but in none of the three
    # training years, leaving that day with no climatological sample at all.
    # Zero-filling it would put a spurious spike in every climatology-referenced
    # score, so the neighbouring day is carried in instead.
    miss = np.where(cnt == 0)[0]
    cnt = np.where(cnt == 0, 1, cnt)
    clim = {v: (acc[v] / cnt[:, None, None]).astype(np.float32) for v in VARS}
    for i in miss:
        j = i - 1 if i > 0 else i + 1
        for v in VARS:
            clim[v][i] = clim[v][j]
    pers = {v: np.broadcast_to(truth[v][0], truth[v].shape) for v in VARS}
    dates = [di[int(g)]["date"] for g in gd]
    return truth, pers, clim, dates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="Dp")
    ap.add_argument("--rollout-root", default="Results/seas5")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    vs.apply_style()
    os.makedirs(args.out_dir, exist_ok=True)
    S = {}

    cache_only = os.environ.get("SEAS5_FIG_CACHE_ONLY") == "1"
    for fold in ([] if cache_only else PLOT_ORDER):
        got = load_fold(args.rollout_root, args.arm, fold)
        if got is None:
            print(f"  fold {fold}: nothing to load"); continue
        ens, gd = got
        truth, pers, clim, dates = references(fold, gd)
        M = ens[VARS[0]].shape[0]
        rec = {"n_members": M, "dates": dates, "wy": FOLD_WY[fold]}
        for v in VARS:
            y, e = truth[v], ens[v]
            mean = e.mean(0); sd = e.std(0, ddof=1)
            r = lambda a: np.sqrt(np.nanmean((a - y) ** 2, axis=(1, 2)))
            # p95-p5 of the truth: the scale that makes folds comparable
            rng = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
            crps = np.array([np.nanmean(crps_fair_sorted(e[:, j], y[j]))
                             for j in range(len(gd))])
            # rank of the truth among members, pooled over a thinned grid
            sub = (slice(None), slice(None, None, 8), slice(None, None, 8))
            ranks = (e[(slice(None),) + sub[1:]][:, ::6] <
                     y[sub[1:]][::6][None]).sum(0).ravel()
            rec[v] = dict(rmse=r(mean), rmse_pers=r(pers[v]), rmse_clim=r(clim[v]),
                          spread=np.nanmean(sd, axis=(1, 2)), crps=crps, range=rng,
                          ranks=ranks,
                          maps={d: (y[d], mean[d], sd[d]) for d in MAP_DAYS
                                if d < len(gd)})
        S[fold] = rec
        print(f"  fold {fold} (WY{FOLD_WY[fold]}): {M} members, {len(gd)} days", flush=True)
        del ens

    cache = os.path.join(args.out_dir, f"_stats_arm{args.arm}.pkl")
    if S:
        import pickle
        with open(cache, "wb") as fh:
            pickle.dump(S, fh)
        print(f"  cached reduced statistics -> {os.path.basename(cache)}")
    elif os.path.exists(cache):
        import pickle
        S = pickle.load(open(cache, "rb"))
        print(f"  loaded cached statistics from {os.path.basename(cache)}")
    if not S:
        raise SystemExit("no folds loaded")
    folds = [f for f in PLOT_ORDER if f in S]
    T = np.arange(len(S[folds[0]]["dates"]))
    tick = np.linspace(0, len(T) - 1, 6).astype(int)

    def datelab(f):
        return [S[f]["dates"][i][:7] for i in tick]

    # ---- 1. error evolution, absolute ------------------------------------
    for key, lab, fname in (("rmse", "RMSE", "fig_seas5_error_evolution"),
                            ("rel", "relative error", "fig_seas5_error_relative")):
        # sharex="col", NOT True: True shares one x axis across every panel, so
        # per-column date labels all collapse onto the last fold's dates and
        # three of four panels are silently mislabelled.
        fig, axes = plt.subplots(2, len(folds), figsize=(4.6*len(folds), 7.6),
                                 squeeze=False, sharex="col")
        for j, f in enumerate(folds):
            for i, v in enumerate(VARS):
                ax = axes[i][j]; d = S[f][v]
                sc = d["range"] if key == "rel" else 1.0
                ax.plot(T, d["rmse_clim"]/sc, color=vs.C_PERSIST, lw=vs.LW_SECONDARY,
                        ls="--", label="state climatology")
                ax.plot(T, d["rmse_pers"]/sc, color="k", lw=vs.LW_FAINT+0.7,
                        ls=":", label="persistence")
                ax.plot(T, d["rmse"]/sc, color=vs.C_PRED, lw=vs.LW_MAIN,
                        label=arm_label(args.arm))
                ax.fill_between(T, (d["rmse"]-d["spread"])/sc,
                                (d["rmse"]+d["spread"])/sc, color=vs.C_PRED, alpha=.15, lw=0)
                unit = "" if key == "rel" else f" ({UNIT[v]})"
                vs.style_axes(ax, ylabel=(f"{PRETTY[v]}\n{lab}{unit}" if j == 0 else None),
                              title=(f"WY{S[f]['wy']} · fold {f}" if i == 0 else None))
                ax.set_ylim(bottom=0)
        for j, f in enumerate(folds):
            tk = np.linspace(0, len(T)-1, 4).astype(int)
            axes[-1][j].set_xticks(tk)
            axes[-1][j].set_xticklabels([monlab(S[f]["dates"][i]) for i in tk],
                                        rotation=30, ha="right",
                                        fontsize=vs.FS_TICK-5)
        h, l = axes[0][0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(.5, .055), ncol=3,
                   frameon=False, fontsize=vs.FS_LEGEND-2)
        fig.suptitle(f"{arm_label(args.arm).capitalize()}: {lab} through the 210-day chain",
                     fontsize=vs.FS_SUPTITLE, fontweight="bold", y=.985,
                     va="top")
        # top was .90, which put the suptitle on the per-column water-year
        # titles; the panels start lower now
        fig.subplots_adjust(top=.855, bottom=.20, hspace=.14, wspace=.24,
                            left=.085, right=.985)
        vs.save_both(fig, os.path.join(args.out_dir, fname)); plt.close(fig)
        print(f"  wrote {fname}")

    # ---- 2. spread-skill: the headline ------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4))
    for ax, v in zip(axes, VARS):
        for f in folds:
            d = S[f][v]
            ax.plot(T, d["spread"]/d["rmse"], lw=vs.LW_MAIN,
                    label=f"WY{S[f]['wy']} (fold {f})")
        ax.axhline(1.0, color="k", lw=vs.LW_FAINT+0.7, ls=":")
        ax.text(len(T)*0.70, 1.04, "calibrated", fontsize=vs.FS_ANNOT-4,
                color="#4A555F", fontweight="bold")
        vs.style_axes(ax, xlabel="forecast day", ylabel="spread / error",
                      title=PRETTY[v])
        ax.set_xticks(np.linspace(0, len(T)-1, 5).astype(int))
        ax.set_xticklabels([f"{i+1}" for i in np.linspace(0, len(T)-1, 5).astype(int)])
        ax.set_ylim(0, 1.35)
    # a four-line in-axes legend crossed every curve; it goes under the panels
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(.5, .0), ncol=4,
               frameon=False, fontsize=vs.FS_LEGEND-3)
    fig.suptitle("Ensemble dispersion: the forcing ensemble supplies the "
                 "perturbation sampler noise alone could not",
                 fontsize=vs.FS_SUPTITLE-2, fontweight="bold")
    fig.tight_layout(rect=(0, .085, 1, .93))
    vs.save_both(fig, os.path.join(args.out_dir, "fig_seas5_spread_skill")); plt.close(fig)
    print("  wrote fig_seas5_spread_skill")

    # ---- 3. CRPS and rank histogram ---------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.8))
    for i, v in enumerate(VARS):
        for f in folds:
            axes[i][0].plot(T, S[f][v]["crps"], lw=vs.LW_SECONDARY,
                            label=f"WY{S[f]['wy']}")
        vs.style_axes(axes[i][0], ylabel=f"{PRETTY[v]}\nCRPS ({UNIT[v]})",
                      xlabel="forecast day" if i else None)
        tk = np.linspace(0, len(T)-1, 5).astype(int)
        axes[i][0].set_xticks(tk); axes[i][0].set_xticklabels([f"{j+1}" for j in tk])
        allr = np.concatenate([S[f][v]["ranks"] for f in folds])
        M = S[folds[0]]["n_members"]
        axes[i][1].hist(allr, bins=np.arange(M+2)-.5, color=vs.C_PRED,
                        edgecolor="white", linewidth=.4)
        axes[i][1].axhline(len(allr)/(M+1), color="k", ls=":", lw=vs.LW_FAINT+0.7)
        vs.style_axes(axes[i][1], ylabel="count",
                      xlabel="rank of truth among members" if i else None,
                      title="flat = calibrated" if i == 0 else None)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(.5, .0), ncol=4,
               frameon=False, fontsize=vs.FS_LEGEND-3)
    fig.suptitle("Probabilistic verification, all folds pooled",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    fig.tight_layout(rect=(0, .06, 1, .945))
    vs.save_both(fig, os.path.join(args.out_dir, "fig_seas5_crps_rank")); plt.close(fig)
    print("  wrote fig_seas5_crps_rank")

    # ---- 4. maps at four lead times, water table depth --------------------
    for v in VARS:
        days = [d for d in MAP_DAYS if d in S[folds[0]][v]["maps"]]
        fig, axes = plt.subplots(len(folds), len(days)*2,
                                 figsize=(2.5*len(days)*2, 2.7*len(folds)), squeeze=False)
        sp = evo.SPEC[v]
        scm = ListedColormap(sp["colors"])
        snorm = BoundaryNorm(sp["bounds"], scm.N)
        for i, f in enumerate(folds):
            for j, d in enumerate(days):
                y, mean, sd = S[f][v]["maps"][d]
                lim = float(np.nanpercentile(np.abs(mean-y), 98))
                sim = vs.north_up_imshow(axes[i][2*j], mean, cmap=scm, norm=snorm)
                vs.map_axes_off(axes[i][2*j],
                                title=(f"day {d+1}" if i == 0 else None))
                vs.north_up_imshow(axes[i][2*j+1], mean-y, cmap="RdBu_r",
                                   vmin=-lim, vmax=lim)
                vs.map_axes_off(axes[i][2*j+1],
                                title=(f"error, day {d+1}" if i == 0 else None))
            axes[i][0].set_ylabel(f"WY{S[f]['wy']}", fontsize=vs.FS_LABEL-3,
                                  fontweight="bold")
            axes[i][0].yaxis.set_visible(True)
            axes[i][0].set_yticks([])
        fig.suptitle(f"{PRETTY[v]}: ensemble mean and error, red = forecast too high",
                     fontsize=vs.FS_SUPTITLE-2, fontweight="bold")
        fig.tight_layout(rect=(0, .085, 1, .95))
        cax = fig.add_axes([0.14, 0.035, 0.34, 0.022])
        cb = fig.colorbar(sim, cax=cax, orientation="horizontal",
                          boundaries=sp["bounds"], ticks=sp["bounds"],
                          spacing="uniform")
        cb.ax.set_xticklabels(sp["ticklabels"])
        cb.set_label(vs.axis_label(f"{PRETTY[v]}, ensemble mean", UNIT[v]),
                     fontsize=vs.FS_CBAR_LABEL-3, fontweight="bold", labelpad=6)
        cb.ax.tick_params(labelsize=vs.FS_CBAR_TICK-4)
        vs.save_both(fig, os.path.join(args.out_dir, f"fig_seas5_maps_{v}")); plt.close(fig)
        print(f"  wrote fig_seas5_maps_{v}")

    # ---- 5. cross-fold summary -------------------------------------------
    rows = []
    for f in folds:
        for v in VARS:
            d = S[f][v]
            rows.append(dict(fold=f, wy=S[f]["wy"], variable=v,
                             rmse=float(np.sqrt(np.mean(d["rmse"]**2))),
                             nrmse=float(np.sqrt(np.mean(d["rmse"]**2))/d["range"]),
                             skill_pers=float(1-np.sqrt(np.mean(d["rmse"]**2))/np.sqrt(np.mean(d["rmse_pers"]**2))),
                             skill_clim=float(1-np.sqrt(np.mean(d["rmse"]**2))/np.sqrt(np.mean(d["rmse_clim"]**2))),
                             crps=float(d["crps"].mean()),
                             spread_err=float(np.mean(d["spread"])/np.sqrt(np.mean(d["rmse"]**2)))))
    with open(os.path.join(args.out_dir, f"seas5_arm{args.arm}_summary.json"), "w") as fh:
        json.dump(rows, fh, indent=1)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))
    x = np.arange(len(folds)); w = .36
    for ax, key, lab in ((axes[0], "nrmse", "relative error"),
                         (axes[1], "skill_clim", "skill vs climatology"),
                         (axes[2], "spread_err", "spread / error")):
        for k, v in enumerate(VARS):
            vals = [next(r[key] for r in rows if r["fold"] == f and r["variable"] == v) for f in folds]
            ax.bar(x + (k-.5)*w, vals, w, color=(vs.C_ACTUAL if k == 0 else vs.C_PRED),
                   label=PRETTY[v])
        if key == "spread_err": ax.axhline(1, color="k", ls=":", lw=vs.LW_FAINT+0.7)
        if key == "skill_clim": ax.axhline(0, color="k", lw=vs.LW_FAINT+0.7)
        vs.style_axes(ax, ylabel=lab)
        ax.set_xticks(x)
        ax.set_xticklabels([str(S[f]["wy"]) for f in folds], fontsize=vs.FS_TICK-3)
        ax.set_xlabel("verifying water year   (fold 4 \u2192 1)", fontsize=vs.FS_LABEL-5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(.5, .055), ncol=2,
               frameon=False, fontsize=vs.FS_LEGEND-2)
    fig.suptitle(f"{arm_label(args.arm).capitalize()}: cross-fold summary",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    fig.tight_layout(rect=(0, .10, 1, .925))
    vs.save_both(fig, os.path.join(args.out_dir, "fig_seas5_summary")); plt.close(fig)
    print("  wrote fig_seas5_summary")
    print(f"\n  summary -> {args.out_dir}/seas5_arm{args.arm}_summary.json")


if __name__ == "__main__":
    main()
