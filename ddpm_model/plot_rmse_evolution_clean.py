"""
Error growth through the rollout: ensemble mean with a +- spread band.

The band is the ensemble standard deviation about its mean -- the model's own
predictive uncertainty -- not the spread of the member RMSE scores. The latter is
an average over ~4x10^5 cells per member and is a tenth of a percent wide, which
tells the reader nothing. Only ensemble_mean.npz is needed, so this is quick: the
16 GB member array is not touched.

Usage:
    python plot_rmse_evolution_clean.py --rollout-dir <dir> --out-dir <dir>
"""
import argparse
import json
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import viz_style as vs
from analyze_rollout import rolling_mean

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "data")
RAW = os.path.join(DATA, "parflow_raw")
VARS = ("water_table_depth", "soil_moisture")
NAME = {"water_table_depth": "Water table depth", "soil_moisture": "Soil moisture"}
UNIT = {"water_table_depth": "m", "soil_moisture": "-"}
MONTH = {f"{m:02d}": n for m, n in zip(range(1, 13),
         ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--name", default="rmse_evolution")
    args = ap.parse_args()
    vs.apply_style()
    os.makedirs(args.out_dir, exist_ok=True)

    meta = json.loads(open(os.path.join(args.rollout_dir, "rollout_meta.json")).read())
    M, yr = meta["n_members"], meta["test_year"]
    mn = np.load(os.path.join(args.rollout_dir, "ensemble_mean.npz"))
    gd = mn["global_days"].tolist()
    di = json.loads(open(os.path.join(DATA, "tiles", "day_index.json")).read())
    day0 = min(i for i, e in enumerate(di) if e["water_year"] == yr)
    dates = [di[int(g)]["date"] for g in gd]
    t2 = xr.open_dataset(os.path.join(RAW, f"target2d_wy{yr}.nc"))
    truth = {v: t2[v].values[[g - day0 for g in gd]] for v in VARS}
    t2.close()
    T = np.arange(len(dates))

    fig, axes = plt.subplots(2, 1, figsize=(15, 10.5), dpi=140, sharex=True,
                             constrained_layout=True)
    for ax, var in zip(axes, VARS):
        y = truth[var]
        r_mean = rolling_mean(np.sqrt(np.nanmean((mn[var] - y) ** 2, axis=(1, 2))))
        spread = rolling_mean(np.sqrt(np.nanmean(mn[f"{var}_sd"] ** 2, axis=(1, 2))))
        pers = np.sqrt(np.nanmean((np.broadcast_to(y[0], y.shape) - y) ** 2, axis=(1, 2)))

        ax.fill_between(T, np.maximum(r_mean - spread, 0.0), r_mean + spread,
                        color=vs.C_PRED, alpha=0.25,
                        label=f"Ensemble mean $\\pm$ spread ({M} members)")
        ax.plot(T, r_mean, color=vs.C_PRED, linewidth=vs.LW_MAIN + 0.6,
                label="Ensemble mean")
        ax.plot(T, pers, color=vs.C_PERSIST, linewidth=vs.LW_SECONDARY,
                linestyle="--", label="Persistence")
        vs.style_axes(ax, ylabel=f"RMSE [{UNIT[var]}]", title=NAME[var])
        ax.legend(fontsize=vs.FS_LEGEND, frameon=False, loc="upper left")
        ax.set_ylim(0, max(pers.max(), (r_mean + spread).max()) * 1.22)

    ti = np.linspace(0, len(dates) - 1, 8).astype(int)
    axes[-1].set_xticks(ti)
    axes[-1].set_xticklabels([f"{MONTH[dates[i][5:7]]} {dates[i][2:4]}" for i in ti])
    axes[-1].set_xlabel("Date")
    fig.suptitle("Error growth through the one-year rollout",
                 fontsize=vs.FS_SUPTITLE, fontweight="bold")
    out = vs.save_both(fig, os.path.join(args.out_dir, args.name))
    plt.close(fig)
    print(f"wrote {out}.pdf and .png")
    for var in VARS:
        y = truth[var]
        rm = np.sqrt(np.nanmean((mn[var] - y) ** 2))
        sp = np.sqrt(np.nanmean(mn[f"{var}_sd"] ** 2))
        print(f"  {NAME[var]:20s} annual RMSE {rm:.4f} {UNIT[var]} | "
              f"mean spread {sp:.4f} ({100*sp/rm:.0f}% of RMSE)")


if __name__ == "__main__":
    main()
