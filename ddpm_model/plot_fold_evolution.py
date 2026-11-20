"""
Per-fold temporal evolution maps: ParFlow on top, DDPM underneath.

One figure per (fold, variable, forecast window): a row of dates spread across
the rolled-out water year, with the ParFlow truth along the top and the
emulator's chained prediction directly below it, so the two can be compared
column by column as the year progresses.

The colour scale is DISCRETE -- five classes, not a continuous ramp. A
continuous map of a field this skewed (half the domain sits above a water table
shallower than 1 m, while the tail reaches tens of metres) compresses almost
everything into one or two shades. Five bright, well-separated classes make the
wet/dry pattern and its drift legible from across a room, which is what these
figures are for. The class boundaries are fixed, not derived per fold, so the
same colour means the same depth in every figure.

Bin choice comes from the pooled WY2003-2006 distribution:
  water table depth  0-0.5 / 0.5-1 / 1-2 / 2-5 / >5 m   (37 / 15 / 33 / 13 / 2 %)
  soil moisture      <0.20 / -0.28 / -0.36 / -0.43 / >0.43

Usage:
    python plot_fold_evolution.py --all
    python plot_fold_evolution.py --fold fold1 --out-dir ../data/comparisons/fold1
"""
import argparse
import json
import os

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

import viz_style as vs
from compare_folds import FOLDS, LEADS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# shallow/wet -> deep/dry, five saturated classes
WTD_COLORS = ["#08519C", "#2E9BD6", "#3FAE5A", "#F2C230", "#C0392B"]
WTD_BOUNDS = [0.0, 0.5, 1.0, 2.0, 5.0, 1e6]
WTD_TICKLABELS = ["0", "0.5", "1", "2", "5", ""]

# dry -> wet
SM_COLORS = ["#A0522D", "#E0A62E", "#8FCF6B", "#2E9BD6", "#08519C"]
SM_BOUNDS = [0.0, 0.20, 0.28, 0.36, 0.43, 1.0]
SM_TICKLABELS = ["0", "0.20", "0.28", "0.36", "0.43", "1"]

SPEC = {
    "water_table_depth": dict(colors=WTD_COLORS, bounds=WTD_BOUNDS,
                              ticklabels=WTD_TICKLABELS, unit="m",
                              pretty="Water table depth",
                              note="dark blue = water table at the surface, "
                                   "red = 5 m or deeper"),
    "soil_moisture": dict(colors=SM_COLORS, bounds=SM_BOUNDS,
                          ticklabels=SM_TICKLABELS, unit="-",
                          pretty="Soil moisture",
                          note="brown = dry, dark blue = near saturation"),
}
MONTHS = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May",
          "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct",
          "11": "Nov", "12": "Dec"}


def nice_date(d):
    return f"{MONTHS[d[5:7]]} {int(d[8:10])}, {d[:4]}"


def plot(fold, var, lead, dates, truth, pred, save_path, n_cols=7, year=None,
         pred_label=None, subtitle=None):
    sp = SPEC[var]
    cmap = ListedColormap(sp["colors"])
    norm = BoundaryNorm(sp["bounds"], cmap.N)
    yr = year if year is not None else FOLDS[fold]["test"]
    idx = np.linspace(0, len(dates) - 1, n_cols).astype(int)

    fig, axes = plt.subplots(2, n_cols, figsize=(3.5 * n_cols, 8.6), dpi=140)
    for c, i in enumerate(idx):
        for r, (field, row_name) in enumerate([(truth[i], "ParFlow"), (pred[i], "DDPM")]):
            ax = axes[r, c]
            im = vs.north_up_imshow(ax, field, cmap=cmap, norm=norm)
            vs.map_axes_off(ax)
            if r == 0:
                ax.set_title(nice_date(dates[i]), fontsize=vs.FS_TITLE - 1,
                             fontweight="bold", pad=8)
    bottom = pred_label if pred_label else f"DDPM\n({lead}-day window)"
    for r, lab, color in [(0, "ParFlow\n(truth)", vs.C_ACTUAL),
                          (1, bottom, vs.C_PRED)]:
        axes[r, 0].set_ylabel(lab, fontsize=vs.FS_LABEL + 2, fontweight="bold",
                              color=color)
        axes[r, 0].yaxis.set_visible(True)
        axes[r, 0].set_yticks([])

    cax = fig.add_axes([0.25, 0.085, 0.5, 0.028])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      boundaries=sp["bounds"], ticks=sp["bounds"], spacing="uniform")
    cb.ax.set_xticklabels(sp["ticklabels"])
    cb.set_label(f"{sp['pretty']} ({sp['unit']})   --   {sp['note']}",
                 fontsize=vs.FS_CBAR_LABEL, fontweight="bold", labelpad=12)
    cb.ax.tick_params(labelsize=vs.FS_CBAR_TICK)

    # `fold` is a free-text label; the default placeholder reads as an artefact
    _tag = f" --- {fold}" if fold and str(fold).lower() not in ("run", "none") else ""
    head = subtitle if subtitle else (
        f"{sp['pretty']} through the WY{yr} rollout{_tag}, "
        f"{lead}-day forecast window\n"
        f"top: ParFlow.  bottom: the emulator, chained from day 0 with no "
        f"further sight of truth.")
    fig.suptitle(head, fontsize=vs.FS_SUPTITLE, fontweight="bold", y=0.995,
                 va="top")
    # top was 0.855, which left the second suptitle line sitting on the column
    # date labels; the panels start lower now
    fig.subplots_adjust(left=0.035, right=0.995, top=0.80, bottom=0.155,
                        wspace=0.04, hspace=0.06)
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(fold, out_dir, day_index):
    os.makedirs(out_dir, exist_ok=True)
    yr = year if year is not None else FOLDS[fold]["test"]
    day0 = min(i for i, e in enumerate(day_index) if e["water_year"] == yr)
    t2 = xr.open_dataset(os.path.join(DATA_DIR, "parflow_raw", f"target2d_wy{yr}.nc"))
    for lead in LEADS:
        pr = np.load(os.path.join(
            DATA_DIR, f"rollout_wy{yr}_{fold}_f{lead}_lean", "predicted_state.npz"))
        gd = pr["global_days"].tolist()
        dates = [day_index[int(g)]["date"] for g in gd]
        idx = [g - day0 for g in gd]
        for var in SPEC:
            plot(fold, var, lead, dates, t2[var].values[idx], pr[var],
                 os.path.join(out_dir, f"evolution_{var}_f{lead}.png"))
    t2.close()
    print(f"  {fold} (test WY{yr}) -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold")
    ap.add_argument("--out-dir")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    vs.apply_style()
    day_index = json.loads(open(os.path.join(DATA_DIR, "tiles", "day_index.json")).read())
    if args.all:
        for fold in FOLDS:
            run(fold, os.path.join(DATA_DIR, "comparisons", fold), day_index)
    else:
        if not (args.fold and args.out_dir):
            raise SystemExit("give --fold and --out-dir, or --all")
        run(args.fold, args.out_dir, day_index)


if __name__ == "__main__":
    main()
