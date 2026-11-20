"""
Per-fold time series at the fixed USGS reference wells, one figure per variable.

For each fold (each held-out water year) this writes:
    well_timeseries_water_table_depth.png
    well_timeseries_soil_moisture.png
    well_locations_map.png

The wells come from data/reference_wells.json and are the SAME eight sites in
every fold, so panels can be read across folds. Each panel is titled with what
the site actually is -- name, state, depth class, coordinates -- rather than a
PLSS code.

Water table depth carries three or four curves: the USGS observation, ParFlow
(the emulator's target), and the DDPM at each forecast window.

Soil moisture carries NO observation curve, and the figure says so on its face.
There is no soil-moisture measurement anywhere in this domain for WY2003-2006:
every USGS soil-moisture site in the region begins in 2023. Those panels are
therefore a ParFlow-vs-emulator comparison at the well locations, which is still
worth showing -- it is the same eight places, so the two variables can be read
side by side -- but it is not validation against data and is not labelled as if
it were.

Observed and modelled depths do not share a datum (a point well in one screened
interval vs. a 1 km cell average over a 5-layer column), so the water-table
panels plot each series centred on its own mean for the year. These are not
climatological anomalies -- four water years is not a climatology. Depth is
positive downward, so the axis is inverted: UP IS WETTER.

Usage:
    python plot_well_timeseries.py --fold fold1 --out-dir ../data/comparisons/fold1
    python plot_well_timeseries.py --all
"""
import argparse
import csv
import json
import os
from collections import defaultdict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import viz_style as vs
from compare_folds import FOLDS, LEADS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
WELL_DIR = os.path.join(DATA_DIR, "usgs_wells")
C_OBS = "#1B7A3D"
NCOL, NROW = 4, 2


def load():
    wells = json.loads(open(os.path.join(DATA_DIR, "reference_wells.json")).read())["wells"]
    obs = defaultdict(dict)
    with open(os.path.join(WELL_DIR, "observations.csv")) as f:
        for r in csv.DictReader(f):
            obs[r["site_no"]][r["date"]] = float(r["depth_to_water_m"])
    day_index = json.loads(open(os.path.join(DATA_DIR, "tiles", "day_index.json")).read())
    return wells, obs, day_index


def gather(fold, day_index):
    """dates plus, per lead, the full predicted field; and the ParFlow truth."""
    yr = FOLDS[fold]["test"]
    day0 = min(i for i, e in enumerate(day_index) if e["water_year"] == yr)
    preds = {}
    dates = None
    for lead in LEADS:
        pr = np.load(os.path.join(
            DATA_DIR, f"rollout_wy{yr}_{fold}_f{lead}_lean", "predicted_state.npz"))
        gd = pr["global_days"].tolist()
        dates = [day_index[int(g)]["date"] for g in gd]
        idx = [g - day0 for g in gd]
        preds[lead] = {v: pr[v] for v in ("water_table_depth", "soil_moisture")}
    t2 = xr.open_dataset(os.path.join(DATA_DIR, "parflow_raw", f"target2d_wy{yr}.nc"))
    truth = {v: t2[v].values[idx] for v in ("water_table_depth", "soil_moisture")}
    t2.close()
    return yr, dates, truth, preds


MONTHS = {"01": "Jan", "02": "Feb", "03": "Mar", "04": "Apr", "05": "May",
          "06": "Jun", "07": "Jul", "08": "Aug", "09": "Sep", "10": "Oct",
          "11": "Nov", "12": "Dec"}


def date_ticks(ax, dates, n=5):
    """Short, non-overlapping labels: 'Oct 05' rather than '2005-10-02'."""
    ti = np.linspace(0, len(dates) - 1, n).astype(int)
    ax.set_xticks(ti)
    ax.set_xticklabels([f"{MONTHS[dates[i][5:7]]} {dates[i][2:4]}" for i in ti],
                       rotation=0, ha="center")


def panel_title(ax, w):
    """Title and description both ABOVE the axes -- nothing overlays the curves.

    An earlier version put the description in a box inside the axes, which sat
    on top of the data. It now goes in the margin between the title and the
    plot, with the title padded far enough to clear it.
    """
    ax.set_title(w["title"], fontsize=vs.FS_TITLE, fontweight="bold", pad=46,
                 loc="left")
    ax.text(0.0, 1.015, w["subtitle"], transform=ax.transAxes,
            fontsize=vs.FS_ANNOT - 2, color="#444444", va="bottom", ha="left",
            linespacing=1.4)


def finish(fig, axes, suptitle, legend_ax):
    """Explicit spacing -- constrained_layout cannot cope with eight panels that
    each carry a multi-line title, and silently collapses the axes to zero."""
    handles, labels = legend_ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.005),
               ncol=len(labels), frameon=False, fontsize=vs.FS_LEGEND + 2)
    fig.suptitle(suptitle, fontsize=vs.FS_SUPTITLE, fontweight="bold", y=0.985)
    # generous top/hspace: each panel carries a title plus a two-line
    # description in the margin above its axes
    fig.subplots_adjust(left=0.045, right=0.995, top=0.815, bottom=0.115,
                        wspace=0.20, hspace=0.46)


def plot_water_table(fold, yr, dates, truth, preds, wells, obs, save_path,
                     mode="wellshift"):
    """Two ways of putting a well and the model on one axis.

    mode="raw"       -- nothing moved. Shows the truth: ParFlow-CONUS1 puts the
        water table metres shallower than the wells measure, right across this
        domain. Honest, but the curves are so far apart that their shapes cannot
        be compared.

    mode="wellshift" -- ONLY the observations are moved, by a single constant, so
        their mean matches ParFlow's mean at that cell over the same days. The
        model and the emulator are left exactly where they are, so ParFlow-vs-DDPM
        stays a true absolute comparison and the well becomes readable against
        it. This is the right default: a well and a 1 km cell average share no
        datum, so the offset is not information about the emulator, whereas the
        ParFlow-DDPM gap is.

    Centring every series on its own mean was the earlier default and is worse:
    it moves the two things that DO share a datum.
    """
    fig, axes = plt.subplots(NROW, NCOL, figsize=(8.2 * NCOL, 6.6 * NROW), dpi=130)
    t = np.arange(len(dates))
    for ax, w in zip(axes.ravel(), wells):
        y, x = w["y"], w["x"]
        o = np.array([obs[w["site_no"]].get(d, np.nan) for d in dates])
        tr = truth["water_table_depth"][:, y, x].astype(float)
        ok = np.isfinite(o)
        shift = float(tr[ok].mean() - o[ok].mean()) if mode == "wellshift" else 0.0
        o_plot = o + shift

        curves = [(o_plot, C_OBS,
                   "observed (USGS well)" + (", shifted onto ParFlow's mean"
                                             if mode == "wellshift" else ""),
                   "-", vs.LW_MAIN + 0.8),
                  (tr, vs.C_ACTUAL, "ParFlow (emulator target)", "-", vs.LW_MAIN + 0.3)]
        for lead in LEADS:
            curves.append((preds[lead]["water_table_depth"][:, y, x].astype(float),
                           vs.C_F10 if lead == 10 else vs.C_F15,
                           f"DDPM {lead}-day window", "--", vs.LW_MAIN))
        for arr, color, lab, ls, lw in curves:
            ax.plot(t, np.asarray(arr, float), color=color, linewidth=lw,
                    linestyle=ls, label=lab)
        ax.invert_yaxis()          # depth is positive downward: up = wetter

        gap = float(o[ok].mean() - tr[ok].mean())
        note = (f"well shifted {shift:+.2f} m onto ParFlow's mean"
                if mode == "wellshift"
                else f"well sits {gap:+.1f} m deeper than ParFlow on average")
        ax.text(0.015, 0.03, note, transform=ax.transAxes,
                fontsize=vs.FS_ANNOT - 2, fontweight="bold", color="#333333",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))
        vs.style_axes(ax, ylabel="Depth to water below\nland surface (m)\n\u2191 wetter")
        panel_title(ax, w)
        date_ticks(ax, dates)

    if mode == "wellshift":
        sub = ("ParFlow and the emulator are NOT moved -- their gap is real emulator "
               "error.  Only the well is shifted, by one constant per site, onto "
               "ParFlow's mean,\nbecause a point well and a 1 km cell average share no "
               "datum.  Depth axis inverted so UP = WETTER.")
    else:
        sub = ("RAW depths, nothing moved.  ParFlow-CONUS1 puts the water table metres "
               "shallower than the wells measure, right across this domain.\n"
               "That offset is a ParFlow bias, not an emulator error -- the companion "
               "figure removes it from the well only.  Depth axis inverted so UP = WETTER.")
    finish(fig, axes,
           f"Water table depth at the 8 fixed USGS reference wells -- {fold}, test WY{yr}\n"
           + sub, axes.ravel()[0])
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_soil_moisture(fold, yr, dates, truth, preds, wells, save_path):
    fig, axes = plt.subplots(NROW, NCOL, figsize=(8.2 * NCOL, 6.6 * NROW), dpi=130)
    t = np.arange(len(dates))
    for ax, w in zip(axes.ravel(), wells):
        y, x = w["y"], w["x"]
        ax.plot(t, truth["soil_moisture"][:, y, x], color=vs.C_ACTUAL,
                linewidth=vs.LW_MAIN + 0.3, label="ParFlow (emulator target)")
        for lead in LEADS:
            ax.plot(t, preds[lead]["soil_moisture"][:, y, x],
                    color=vs.C_F10 if lead == 10 else vs.C_F15, linestyle="--",
                    linewidth=vs.LW_MAIN, label=f"DDPM {lead}-day window")
        vs.style_axes(ax, ylabel="Soil moisture (-)")
        panel_title(ax, w)
        date_ticks(ax, dates)
    finish(fig, axes,
           f"Soil moisture at the same 8 well locations -- {fold}, test WY{yr}\n"
           "NO OBSERVATION CURVE -- there is no soil-moisture measurement anywhere in "
           "this domain for WY2003-2006 (every USGS site in the region begins in 2023).\n"
           "ParFlow vs. emulator at the well locations, not validation against data. "
           "Absolute values, not deviations.",
           axes.ravel()[0])
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_map(fold, yr, truth, preds, wells, obs, dates, save_path):
    """Where the wells are, with each fold's own agreement written on."""
    static = xr.open_dataset(os.path.join(DATA_DIR, "parflow_raw", "static_conditions.nc"))
    wtd_mean = np.nanmean(truth["water_table_depth"], axis=0)
    fig, ax = plt.subplots(figsize=(13.5, 12.5), dpi=140, constrained_layout=True)
    im = vs.north_up_imshow(ax, wtd_mean, cmap=vs.CMAP_STATE, vmin=0,
                            vmax=np.nanpercentile(wtd_mean, 95))
    vs.add_colorbar(fig, im, ax, f"ParFlow mean water table depth, WY{yr} (m)")

    placed = []
    H, W = wtd_mean.shape
    for i, w in enumerate(wells):
        y, x = w["y"], w["x"]
        c = vs.WELL_COLORS[i % len(vs.WELL_COLORS)]
        ax.scatter([x], [y], s=460, marker="o", facecolor=c,
                   edgecolor="white", linewidth=3.2, zorder=5)
        ax.text(x, y, w["short_id"][1:], color="white", fontsize=vs.FS_ANNOT - 1,
                fontweight="bold", ha="center", va="center", zorder=6)
        dx = 22 if x < W - 110 else -22
        dy = 24 if y < 80 else (-34 if y > H - 80 else 20)
        if any((y - py) ** 2 + (x - px) ** 2 < 70 ** 2 for py, px in placed):
            dy, dx = -dy, -dx
        placed.append((y, x))
        ax.annotate(w["short_id"], (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=vs.FS_ANNOT + 3, color=c, fontweight="bold",
                    ha="left" if dx > 0 else "right",
                    path_effects=[__import__("matplotlib.patheffects", fromlist=["x"])
                                  .withStroke(linewidth=4.0, foreground="white")])
    vs.map_axes_off(ax)
    # `fold` is a free-text label; when it is the default placeholder it reads
    # as an artefact in the title, so it is only shown when it says something
    _tag = f" --- {fold}" if fold and str(fold).lower() not in ("run", "none") else ""
    ax.set_title(f"USGS reference wells on the WY{yr} water table{_tag}\n"
                 "the same eight wells appear in every fold",
                 fontsize=vs.FS_TITLE + 3, fontweight="bold")

    handles = []
    for i, w in enumerate(wells):
        o = np.array([obs[w["site_no"]].get(d, np.nan) for d in dates])
        tr = truth["water_table_depth"][:, w["y"], w["x"]].astype(float)
        pd15 = preds[15]["water_table_depth"][:, w["y"], w["x"]].astype(float)
        ok = np.isfinite(o)
        bias = float(np.nanmean(pd15[ok]) - np.nanmean(tr[ok]))
        handles.append(plt.Line2D(
            [], [], marker="o", linestyle="none", markersize=15,
            markerfacecolor=vs.WELL_COLORS[i % len(vs.WELL_COLORS)],
            markeredgecolor="white", markeredgewidth=2,
            label=(f"{w['title']}  |  observed median {w['median_depth_m']:.1f} m, "
                   f"DDPM bias vs ParFlow {bias:+.2f} m")))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.015),
              frameon=False, fontsize=vs.FS_LEGEND, ncol=1)
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run(fold, out_dir, wells, obs, day_index):
    os.makedirs(out_dir, exist_ok=True)
    yr, dates, truth, preds = gather(fold, day_index)
    plot_water_table(fold, yr, dates, truth, preds, wells, obs,
                     os.path.join(out_dir, "well_timeseries_water_table_depth_raw.png"),
                     mode="raw")
    plot_water_table(fold, yr, dates, truth, preds, wells, obs,
                     os.path.join(out_dir, "well_timeseries_water_table_depth_wellshifted.png"),
                     mode="wellshift")
    plot_soil_moisture(fold, yr, dates, truth, preds, wells,
                       os.path.join(out_dir, "well_timeseries_soil_moisture.png"))
    plot_map(fold, yr, truth, preds, wells, obs, dates,
             os.path.join(out_dir, "well_locations_map.png"))
    print(f"  {fold} (test WY{yr}) -> {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold")
    ap.add_argument("--out-dir")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    vs.apply_style()
    wells, obs, day_index = load()
    print(f"{len(wells)} fixed reference wells")
    if args.all:
        for fold in FOLDS:
            run(fold, os.path.join(DATA_DIR, "comparisons", fold), wells, obs, day_index)
    else:
        if not (args.fold and args.out_dir):
            raise SystemExit("give --fold and --out-dir, or --all")
        run(args.fold, args.out_dir, wells, obs, day_index)


if __name__ == "__main__":
    main()
