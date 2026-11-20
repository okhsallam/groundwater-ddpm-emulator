"""
Cross-fold comparison, led by RELATIVE error.

Absolute RMSE cannot be compared between water table depth (metres) and soil
moisture (dimensionless), nor fairly between test years that differ in how much
the state actually moves. Both problems are solved by the same normalisation:

    nRMSE = RMSE / (p95 - p5 of the truth)

the error expressed as a fraction of the range the variable actually covers.
Percentiles rather than min-max because water table depth is exactly zero in
4-5% of cell-days and the extremes are not representative. A pointwise ratio
|pred-truth|/truth is not used at all: it is undefined wherever the truth is
zero, which is precisely where the field is most interesting.

Two forms are computed:
  domain      denominator is the domain-wide p5-p95 over the test year; used for
              the headline numbers and the evolution curves
  per pixel   denominator is each pixel's own temporal p5-p95, excluding the
              0.7% of cells that barely move at all (range < 0.01); used for the
              maps and the median statistic

Outputs into --out-dir, each as PDF and PNG.

Usage:
    python compare_folds_relative.py --out-dir ../report_folds/figures
"""
import argparse
import json
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import viz_style as vs
from analyze_rollout import rolling_mean

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "parflow_raw")
RES = os.path.join(HERE, "Results")

# Insertion order is the plotting order: chronological by verifying water year,
# so WY2003 is leftmost and WY2006 rightmost in every panel of this study. The
# fold numbers run the other way; the tables keep them, the figures do not.
FOLDS = {4: ([2004, 2005, 2006], 2003), 3: ([2003, 2005, 2006], 2004),
         2: ([2003, 2004, 2006], 2005), 1: ([2003, 2004, 2005], 2006)}
VARS = ("water_table_depth", "soil_moisture")
NAME = {"water_table_depth": "Water table depth", "soil_moisture": "Soil moisture"}
UNIT = {"water_table_depth": "m", "soil_moisture": "-"}
FLOOR = {"water_table_depth": 0.01, "soil_moisture": 0.005}
FC = {1: "#1F6FB4", 2: "#E8702A", 3: "#2E9E5B", 4: "#8B5FBF"}
MONTH = {f"{m:02d}": n for m, n in zip(range(1, 13),
         ("Oct", "Nov", "Dec", "Jan", "Feb", "Mar",
          "Apr", "May", "Jun", "Jul", "Aug", "Sep"))}
CAL = {f"{m:02d}": n for m, n in zip(range(1, 13),
       ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))}


def load_fold(f):
    train, yr = FOLDS[f]
    R = os.path.join(RES, f"groundwater_f15_200ep_fold{f}", "rollout")
    mn = np.load(os.path.join(R, "ensemble_mean.npz"))
    gd = mn["global_days"].tolist()
    di = json.loads(open(os.path.join(DATA, "tiles", "day_index.json")).read())
    day0 = min(i for i, e in enumerate(di) if e["water_year"] == yr)
    dates = [di[int(g)]["date"] for g in gd]
    t2 = xr.open_dataset(os.path.join(RAW, f"target2d_wy{yr}.nc"))
    truth = {v: t2[v].values[[g - day0 for g in gd]] for v in VARS}
    t2.close()
    summ = json.loads(open(os.path.join(R, "analysis", "ensemble_summary.json")).read())
    return yr, dates, truth, mn, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    vs.apply_style()
    os.makedirs(args.out_dir, exist_ok=True)
    O = lambda n: os.path.join(args.out_dir, n)

    R = {}          # fold -> everything needed downstream
    for f in FOLDS:
        print(f"fold{f} ...")
        yr, dates, truth, mn, summ = load_fold(f)
        d = dict(yr=yr, dates=dates, summ=summ)
        for v in VARS:
            y = truth[v]
            den = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
            err_t = np.sqrt(np.nanmean((mn[v] - y) ** 2, axis=(1, 2)))
            pers_t = np.sqrt(np.nanmean(
                (np.broadcast_to(y[0], y.shape) - y) ** 2, axis=(1, 2)))
            pix_rmse = np.sqrt(np.nanmean((mn[v] - y) ** 2, axis=0))
            pix_rng = np.nanpercentile(y, 95, axis=0) - np.nanpercentile(y, 5, axis=0)
            ok = pix_rng > FLOOR[v]
            pix_n = np.where(ok, pix_rmse / np.where(ok, pix_rng, np.nan), np.nan)
            d[v] = dict(den=den, err_t=err_t, pers_t=pers_t,
                        rmse=float(np.sqrt(np.nanmean((mn[v] - y) ** 2))),
                        pers=float(np.sqrt(np.nanmean(
                            (np.broadcast_to(y[0], y.shape) - y) ** 2))),
                        nrmse_t=100 * err_t / den, pers_nrmse_t=100 * pers_t / den,
                        pix_n=100 * pix_n,
                        first30=float(np.sqrt(np.nanmean((mn[v][:30] - y[:30]) ** 2))),
                        last30=float(np.sqrt(np.nanmean((mn[v][-30:] - y[-30:]) ** 2))))
            d[v]["nrmse"] = 100 * d[v]["rmse"] / den
            d[v]["pers_nrmse"] = 100 * d[v]["pers"] / den
            d[v]["nrmse_first30"] = 100 * d[v]["first30"] / den
            d[v]["nrmse_last30"] = 100 * d[v]["last30"] / den
        R[f] = d
        del mn, truth

    T = np.arange(len(R[1]["dates"]))
    tick = lambda ax, dts: (ax.set_xticks(np.linspace(0, len(dts) - 1, 7).astype(int)),
                            ax.set_xticklabels(
                                [CAL[dts[i][5:7]] for i in
                                 np.linspace(0, len(dts) - 1, 7).astype(int)]))

    # ---- 1. relative error through the year, the headline figure ----------
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=140, constrained_layout=True)
    for ax, v in zip(axes, VARS):
        for f in FOLDS:
            ax.plot(T, rolling_mean(R[f][v]["nrmse_t"]), color=FC[f],
                    linewidth=vs.LW_MAIN + 0.4,
                    label=f"Fold {f} (test WY{R[f]['yr']})")
            ax.plot(T, rolling_mean(R[f][v]["pers_nrmse_t"]), color=FC[f],
                    linewidth=1.4, linestyle=":", alpha=0.65)
        tick(ax, R[1]["dates"])
        vs.style_axes(ax, xlabel="Month of the water year",
                      ylabel="Relative error  nRMSE [%]", title=NAME[v])
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=vs.FS_LEGEND - 1, frameon=False, loc="upper left")
    fig.suptitle("Relative error through the one-year rollout\n"
                 "solid: emulator   dotted: persistence, same colour",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    vs.save_both(fig, O("fig_relative_error_evolution"))
    plt.close(fig)

    # ---- 2. the two variables on ONE axis, which is the point of nRMSE ----
    fig, ax = plt.subplots(figsize=(13, 7.5), dpi=140, constrained_layout=True)
    x = np.arange(len(FOLDS))
    w = 0.34
    for j, v in enumerate(VARS):
        vals = [R[f][v]["nrmse"] for f in FOLDS]
        ax.bar(x + (j - 0.5) * w, vals, w * 0.92,
               color=vs.C_ACTUAL if j == 0 else vs.C_PRED,
               edgecolor="white", linewidth=1.5, label=NAME[v])
        for xi, val in zip(x + (j - 0.5) * w, vals):
            ax.text(xi, val, f"{val:.1f}%", ha="center", va="bottom",
                    fontsize=vs.FS_ANNOT - 1, fontweight="bold")
    for j, v in enumerate(VARS):
        vals = [R[f][v]["pers_nrmse"] for f in FOLDS]
        ax.plot(x + (j - 0.5) * w, vals, "_", markersize=34, markeredgewidth=3.5,
                color="#555555", label="Persistence" if j == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {f}\ntest WY{R[f]['yr']}" for f in FOLDS])
    vs.style_axes(ax, ylabel="Annual relative error  nRMSE [%]")
    ax.legend(fontsize=vs.FS_LEGEND, frameon=False, loc="upper left")
    ax.set_title("Relative error puts both variables on one scale",
                 fontsize=vs.FS_TITLE + 3, fontweight="bold")
    vs.save_both(fig, O("fig_relative_error_summary"))
    plt.close(fig)

    # ---- 3. absolute RMSE, for reference ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(18, 7.5), dpi=140, constrained_layout=True)
    for ax, v in zip(axes, VARS):
        for f in FOLDS:
            ax.plot(T, rolling_mean(R[f][v]["err_t"]), color=FC[f],
                    linewidth=vs.LW_MAIN + 0.4, label=f"Fold {f}")
            ax.plot(T, rolling_mean(R[f][v]["pers_t"]), color=FC[f],
                    linewidth=1.4, linestyle=":", alpha=0.65)
        tick(ax, R[1]["dates"])
        vs.style_axes(ax, xlabel="Month of the water year",
                      ylabel=f"RMSE [{UNIT[v]}]", title=NAME[v])
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=vs.FS_LEGEND - 1, frameon=False, loc="upper left")
    fig.suptitle("Absolute error through the rollout\n"
                 "solid: emulator   dotted: persistence, same colour",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    vs.save_both(fig, O("fig_rmse_evolution"))
    plt.close(fig)

    # ---- 4. where the relative error sits, spatially -----------------------
    fig, axes = plt.subplots(2, 4, figsize=(21, 11), dpi=140)
    for r, v in enumerate(VARS):
        hi = float(np.nanpercentile(
            np.concatenate([R[f][v]["pix_n"][np.isfinite(R[f][v]["pix_n"])].ravel()
                            for f in FOLDS]), 95))
        for c, f in enumerate(FOLDS):
            ax = axes[r, c]
            im = vs.north_up_imshow(ax, R[f][v]["pix_n"], cmap="YlOrRd", vmin=0, vmax=hi)
            med = np.nanmedian(R[f][v]["pix_n"])
            vs.map_axes_off(ax, (f"Fold {f} — WY{R[f]['yr']}\nmedian {med:.0f}%"
                                 if r == 0 else f"median {med:.0f}%"))
            if c == 0:
                ax.set_ylabel(NAME[v], fontsize=vs.FS_LABEL + 2, fontweight="bold")
                ax.yaxis.set_visible(True)
                ax.set_yticks([])
        cb = fig.colorbar(im, ax=axes[r, :], fraction=0.016, pad=0.01)
        cb.set_label("nRMSE [%]", fontsize=vs.FS_CBAR_LABEL, fontweight="bold")
        cb.ax.tick_params(labelsize=vs.FS_CBAR_TICK)
    fig.suptitle("Relative error per cell, normalised by that cell's own annual range",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    vs.save_both(fig, O("fig_relative_error_maps"))
    plt.close(fig)

    # ---- table ------------------------------------------------------------
    rows = []
    print(f"\n{'fold':5s}{'test':7s}{'variable':20s}{'RMSE':>9s}{'nRMSE%':>9s}"
          f"{'pers%':>8s}{'skill':>8s}{'CRPSsk':>8s}{'first30%':>10s}{'last30%':>9s}")
    for f in FOLDS:
        for v in VARS:
            d, s = R[f][v], R[f]["summ"][v]
            rows.append(dict(fold=f, test_year=R[f]["yr"], variable=v,
                             rmse=d["rmse"], nrmse_pct=d["nrmse"],
                             persistence_nrmse_pct=d["pers_nrmse"],
                             skill=s["skill_ensemble_mean"],
                             crps_skill=s["crps_skill_vs_persistence"],
                             spread_error=s["spread_skill_ratio"],
                             nrmse_first30_pct=d["nrmse_first30"],
                             nrmse_last30_pct=d["nrmse_last30"],
                             median_pixel_nrmse_pct=float(np.nanmedian(d["pix_n"]))))
            print(f"{f:<5d}{R[f]['yr']:<7d}{v:20s}{d['rmse']:9.4f}{d['nrmse']:9.1f}"
                  f"{d['pers_nrmse']:8.1f}{s['skill_ensemble_mean']:+8.3f}"
                  f"{s['crps_skill_vs_persistence']:+8.3f}"
                  f"{d['nrmse_first30']:10.1f}{d['nrmse_last30']:9.1f}")
    json.dump(rows, open(O("fold_comparison.json"), "w"), indent=1)
    print(f"\nwrote figures and fold_comparison.json -> {args.out_dir}")


if __name__ == "__main__":
    main()
