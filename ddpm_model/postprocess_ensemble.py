"""
Presentation post-processing for an ensemble rollout.

Produces the same figure set as the cross-validation work, with the ensemble
carried through as a band rather than collapsed to its mean:

    field_evolution_<var>.pdf   ParFlow vs ensemble mean across the year
    spread_evolution_<var>.pdf  where and when the members disagree
    maps_<var>.pdf              per-pixel RMSE / correlation / skill
    rmse_evolution.pdf          daily RMSE with the member envelope
    wells_map.pdf               the eight reference wells
    wells_water_table.pdf       observed vs ParFlow vs ensemble (banded)
    wells_soil_moisture.pdf     ParFlow vs ensemble (banded)

The member arrays are 9.8 GB per variable, so each variable is loaded once, the
small derived quantities are extracted, and the array is released before the
next one. Nothing holds two variables at full size simultaneously.

Usage:
    python postprocess_ensemble.py --rollout-dir <ensemble dir> \
        --out-dir <ensemble dir>/analysis --label "NLDAS-2 forcing only"
"""
import argparse
import csv
import gc
import json
import os
from collections import defaultdict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import viz_style as vs
import plot_fold_evolution as evo
from analyze_rollout import per_pixel_rmse, per_pixel_corr, skill_score, rolling_mean, map_triptych

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "data")
RAW = os.path.join(DATA, "parflow_raw")
VARS = ("water_table_depth", "soil_moisture")
PRETTY = {"water_table_depth": "Water table depth", "soil_moisture": "Soil moisture"}
UNIT = {"water_table_depth": "m", "soil_moisture": "-"}
C_OBS = "#1B7A3D"
MONTHS = {f"{m:02d}": n for m, n in zip(range(1, 13), (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))}


def date_ticks(ax, dates, n=6, short=True):
    ti = np.linspace(0, len(dates) - 1, n).astype(int)
    ax.set_xticks(ti)
    ax.set_xticklabels([f"{MONTHS[dates[i][5:7]]} {dates[i][2:4]}" for i in ti]
                       if short else [dates[i] for i in ti],
                       rotation=0 if short else 25, ha="center" if short else "right")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    vs.apply_style()
    os.makedirs(args.out_dir, exist_ok=True)
    O = lambda n: os.path.join(args.out_dir, n)

    meta = json.loads(open(os.path.join(args.rollout_dir, "rollout_meta.json")).read())
    yr, M = meta["test_year"], meta["n_members"]
    label = args.label or f"{meta['future_len']}-day window, {M} members"

    day_index = json.loads(open(os.path.join(DATA, "tiles", "day_index.json")).read())
    mean_npz = np.load(os.path.join(args.rollout_dir, "ensemble_mean.npz"))
    gd = mean_npz["global_days"].tolist()
    day0 = min(i for i, e in enumerate(day_index) if e["water_year"] == yr)
    dates = [day_index[int(g)]["date"] for g in gd]
    idx = [g - day0 for g in gd]
    t2 = xr.open_dataset(os.path.join(RAW, f"target2d_wy{yr}.nc"))
    truth = {v: t2[v].values[idx] for v in VARS}
    t2.close()

    wells = json.loads(open(os.path.join(DATA, "reference_wells.json")).read())["wells"]
    obs = defaultdict(dict)
    with open(os.path.join(DATA, "usgs_wells", "observations.csv")) as f:
        for r in csv.DictReader(f):
            obs[r["site_no"]][r["date"]] = float(r["depth_to_water_m"])

    # ---- heavy pass: one variable at a time -------------------------------
    per_day = {}     # var -> (rmse of mean, per-member rmse)
    well_ens = {}    # var -> (M, n_days, n_wells)
    src = np.load(os.path.join(args.rollout_dir, "ensemble_state.npz"))
    for var in VARS:
        print(f"loading members for {var} ...")
        ens = src[var]                              # (M, D, H, W), ~9.8 GB
        y = truth[var]
        mean = mean_npz[var]
        per_day[var] = (
            np.sqrt(np.nanmean((mean - y) ** 2, axis=(1, 2))),
            np.stack([np.sqrt(np.nanmean((ens[m] - y) ** 2, axis=(1, 2)))
                      for m in range(M)]),
        )
        well_ens[var] = np.stack(
            [ens[:, :, w["y"], w["x"]] for w in wells], axis=-1)   # (M, D, n_wells)
        del ens
        gc.collect()
        print(f"  done {var}")

    # ---- field evolution and spread ---------------------------------------
    for var in VARS:
        for _ext in (".pdf", ".png"):
            evo.plot("run", var, meta["future_len"], dates, truth[var], mean_npz[var],
                     O(f"field_evolution_{var}{_ext}"), year=yr)
        sd = mean_npz[f"{var}_sd"]
        n = 7
        sel = np.linspace(0, len(dates) - 1, n).astype(int)
        fig, axes = plt.subplots(1, n, figsize=(3.5 * n, 5.2), dpi=140)
        vmax = float(np.nanpercentile(sd, 98))
        for ax, i in zip(axes, sel):
            im = vs.north_up_imshow(ax, sd[i], cmap="magma_r", vmin=0, vmax=vmax)
            vs.map_axes_off(ax, f"{MONTHS[dates[i][5:7]]} {dates[i][2:4]}")
        cax = fig.add_axes([0.3, 0.10, 0.4, 0.035])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(f"ensemble standard deviation ({UNIT[var]})",
                     fontsize=vs.FS_CBAR_LABEL, fontweight="bold", labelpad=10)
        cb.ax.tick_params(labelsize=vs.FS_CBAR_TICK)
        fig.suptitle(f"{PRETTY[var]}: where the {M} members disagree -- {label}\n"
                     "dark = the model is uncertain there",
                     fontsize=vs.FS_SUPTITLE, fontweight="bold", y=0.99)
        fig.subplots_adjust(left=0.02, right=0.99, top=0.80, bottom=0.20, wspace=0.05)
        vs.save_both(fig, O(f"spread_evolution_{var}"))
        plt.close(fig)

    # ---- per-pixel maps ----------------------------------------------------
    for var in VARS:
        y = truth[var]
        mean = mean_npz[var]
        rmse = per_pixel_rmse(mean, y)
        sk = skill_score(rmse, per_pixel_rmse(np.broadcast_to(y[0], y.shape), y))
        for _ext in (".pdf", ".png"):
            map_triptych(rmse, per_pixel_corr(mean, y), sk, var,
                         f"{PRETTY[var]} (ensemble mean)",
                         O(f"maps_{var}{_ext}"))

    # ---- RMSE evolution, with the member envelope --------------------------
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), dpi=140, sharex=True,
                             constrained_layout=True)
    T = np.arange(len(dates))
    for ax, var in zip(axes, VARS):
        y = truth[var]
        pers = np.sqrt(np.nanmean(
            (np.broadcast_to(y[0], y.shape) - y) ** 2, axis=(1, 2)))
        r_mean, r_mem = per_day[var]
        ax.fill_between(T, r_mem.min(0), r_mem.max(0), color=vs.C_F15, alpha=0.20,
                        label=f"individual members (envelope of {M})")
        for m in range(M):
            ax.plot(T, r_mem[m], color=vs.C_F15, linewidth=0.6, alpha=0.30)
        ax.plot(T, rolling_mean(r_mean), color=vs.C_PRED, linewidth=vs.LW_MAIN + 0.5,
                label=f"ensemble mean (annual RMSE "
                      f"{np.sqrt(np.nanmean((mean_npz[var] - y) ** 2)):.4f})")
        ax.plot(T, pers, color=vs.C_PERSIST, linewidth=vs.LW_SECONDARY,
                linestyle="--", label="persistence (day-0 state held)")
        vs.style_axes(ax, ylabel=f"{PRETTY[var]}\nRMSE ({UNIT[var]})")
        ax.legend(fontsize=vs.FS_LEGEND, frameon=False, loc="upper left")
        ax.set_ylim(0, pers.max() * 1.35)
    date_ticks(axes[-1], dates, 8, short=False)
    axes[-1].set_xlabel("date")
    fig.suptitle(f"Error through the chained WY{yr} rollout -- {label}\n"
                 "the band is the spread across members; the ensemble mean is "
                 "below every individual member",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    vs.save_both(fig, O("rmse_evolution"))
    plt.close(fig)

    # ---- wells -------------------------------------------------------------
    import plot_well_timeseries as wts
    for _ext in (".pdf", ".png"):
        wts.plot_map("run", yr, truth, {15: {v: mean_npz[v] for v in VARS}},
                     wells, obs, dates, O(f"wells_map{_ext}"))

    NROW, NCOL = 2, 4
    for var in VARS:
        fig, axes = plt.subplots(NROW, NCOL, figsize=(8.2 * NCOL, 6.6 * NROW), dpi=130)
        ew = well_ens[var]                                    # (M, D, n_wells)
        for j, (ax, w) in enumerate(zip(axes.ravel(), wells)):
            tr = truth[var][:, w["y"], w["x"]].astype(float)
            lo, hi = ew[:, :, j].min(0), ew[:, :, j].max(0)
            ax.fill_between(T, lo, hi, color=vs.C_PRED, alpha=0.22,
                            label=f"ensemble envelope ({M} members)")
            ax.plot(T, ew[:, :, j].mean(0), color=vs.C_PRED, linewidth=vs.LW_MAIN,
                    linestyle="--", label="ensemble mean")
            ax.plot(T, tr, color=vs.C_ACTUAL, linewidth=vs.LW_MAIN + 0.3,
                    label="ParFlow (emulator target)")
            if var == "water_table_depth":
                o = np.array([obs[w["site_no"]].get(d, np.nan) for d in dates])
                ok = np.isfinite(o)
                shift = float(tr[ok].mean() - o[ok].mean())
                ax.plot(T, o + shift, color=C_OBS, linewidth=vs.LW_MAIN + 0.5,
                        label="observed (USGS well), shifted onto ParFlow's mean")
                ax.invert_yaxis()
                ylab = "Depth to water below\nland surface (m)\n↑ wetter"
                ax.text(0.015, 0.03, f"well shifted {shift:+.2f} m",
                        transform=ax.transAxes, fontsize=vs.FS_ANNOT - 2,
                        fontweight="bold", color="#333333",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  edgecolor="#CCCCCC", alpha=0.9))
            else:
                ylab = "Soil moisture (-)"
            vs.style_axes(ax, ylabel=ylab)
            wts.panel_title(ax, w)
            date_ticks(ax, dates, 5)
        handles, labels = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
                   ncol=len(labels), frameon=False, fontsize=vs.FS_LEGEND + 1)
        extra = ("" if var == "soil_moisture" else
                 "\nonly the observations are shifted; ParFlow and the ensemble are unmoved")
        note = ("\nNO OBSERVATION CURVE: no soil-moisture measurement exists in this "
                "domain for WY2003-2006" if var == "soil_moisture" else extra)
        fig.suptitle(f"{PRETTY[var]} at the 8 fixed USGS reference wells -- {label}"
                     + note, fontsize=vs.FS_SUPTITLE, fontweight="bold", y=0.985)
        fig.subplots_adjust(left=0.045, right=0.995, top=0.815, bottom=0.115,
                            wspace=0.20, hspace=0.46)
        vs.save_both(fig, O(f"wells_{var}"))
        plt.close(fig)

    print(f"\nwrote post-processing -> {args.out_dir}")
    for f in sorted(os.listdir(args.out_dir)):
        if f.endswith((".pdf", "")):
            print("   ", f)


if __name__ == "__main__":
    main()
