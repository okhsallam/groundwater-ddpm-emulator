"""
Evolution maps: every fold, truth and emulator, one figure per variable.

Eight rows by seven columns. The seven columns are seven dates spread evenly
through the rolled-out water year; the eight rows are the four folds in
chronological order of verifying water year (WY2003 first), each fold giving
two rows -- ParFlow truth on top, DDPM ensemble mean directly beneath it. One
shared discrete colourbar at the bottom.

Each fold carries its OWN dates: the folds verify different water years, so the
column dates differ row-pair by row-pair. The single header row is therefore
labelled month + day-of-water-year rather than a calendar date -- the day
positions are identical across folds by construction and the month at each of
them is the same in every fold, whereas the calendar dates drift by up to a day
because WY2004 is a leap year, so dated headers would be true only for the top
row. The dates actually used by every fold are printed to stdout so the
alignment can be audited. A previous version of a similar figure shared one
date axis across panels drawn from different years and so mislabelled three of
four.

The prediction is the member mean of the 16-member rollout ensemble. That mean
is already stored, per day, in ensemble_mean.npz (written as wtd.mean(0) beside
ensemble_state.npz by run_ensemble_rollout.py, with an identical global_days
array), so it is read from there rather than decompressing the 16 GB ensemble.

The large open-water bodies are blanked in every panel (see lake_mask.py).
That is a DRAWING choice only -- no score, table or error series anywhere in
the paper excludes those cells.

Usage:
    python fig_evolution_maps.py
"""
import gc
import json
import os
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P
from lake_mask import blank_lakes

N_COLS = 7

# Open water is blanked in the map panels (see lake_mask.py): Lake Michigan
# carries a water table and a soil moisture like any other cell, but they are
# not about soil, and at this colour scale the lake is the loudest thing in
# the figure. A blanked cell is drawn in this light neutral grey. Grey is not
# used by either colour scale -- both run through saturated hues and both end
# in dark blue -- so a blanked cell cannot be read as a data class, and the
# grey is dark enough against the white page that the lake reads as a filled
# area of the panel rather than a hole cut out of it. DRAWING ONLY: every
# number in this script and in the paper is still computed over the full
# domain.
LAKE_FILL = "#D9D9D9"


def nice_date(d):
    """'2005-10-02' -> 'Oct 2, 2005'."""
    return f"{P.MONTH[d[5:7]]} {int(d[8:10])}, {d[:4]}"


def day_index():
    with open(os.path.join(P.DATA, "tiles", "day_index.json")) as fh:
        return json.load(fh)


def load_fold(fold, di):
    """Seven evenly spread days of truth and ensemble mean for one fold.

    Alignment follows plot_fold_evolution.py / make_seas5_figures.py exactly:
    day0 is the first day_index entry whose water_year is the verifying year,
    and the truth index of a predicted global day g is g - day0.
    """
    wy = P.FOLD_WY[fold]
    day0 = min(i for i, e in enumerate(di) if e["water_year"] == wy)

    z = np.load(P.rollout_path(fold).replace("ensemble_state.npz",
                                             "ensemble_mean.npz"))
    gd = z["global_days"].astype(int)
    pick = np.linspace(0, len(gd) - 1, N_COLS).astype(int)
    gsel = gd[pick]
    dates = [di[int(g)]["date"] for g in gsel]
    tidx = [int(g) - day0 for g in gsel]

    out = {"wy": wy, "dates": dates, "doy": tidx, "truth": {}, "pred": {}}
    ds = xr.open_dataset(os.path.join(P.RAW, f"target2d_wy{wy}.nc"))
    # the netCDF time axis is the independent check on the alignment above
    nc_dates = [str(t)[:10] for t in
                ds["time"].isel(time=tidx).values.astype("datetime64[D]")]
    if nc_dates != dates:
        raise SystemExit(f"fold {fold}: day_index dates {dates} do not match "
                         f"the target file's own time axis {nc_dates}")
    for var in P.VARS:
        out["truth"][var] = ds[var].isel(time=tidx).values.astype(np.float32)
        full = z[var]                      # (days, 637, 665) member mean
        out["pred"][var] = full[pick].astype(np.float32)
        del full
        gc.collect()
    ds.close()
    z.close()

    print(f"  fold {fold} (WY{wy}): day0={day0}, n_days={len(gd)}, "
          f"ens shape={(len(gd),) + out['truth'][P.VARS[0]].shape[1:]}")
    print(f"    global_days {list(map(int, gsel))}")
    print(f"    truth idx   {tidx}")
    print(f"    dates       {[nice_date(d) for d in dates]}")
    return out


def headers(folds):
    """Month + day-of-water-year, valid for every fold.

    The day positions are identical across folds by construction; assert that
    the month at each position is too, so one header row cannot mislabel a
    fold the way per-fold calendar dates would.
    """
    ref = folds[P.FOLDS[0]]
    for f in P.FOLDS[1:]:
        d = folds[f]
        if d["doy"] != ref["doy"]:
            raise SystemExit(f"fold {f}: day-of-water-year positions differ")
        for a, b in zip(d["dates"], ref["dates"]):
            if a[5:7] != b[5:7]:
                raise SystemExit(f"fold {f}: month at a shared day position "
                                 f"differs ({a} vs {b})")
    return [f"{P.MONTH[d[5:7]]}\nday {n}"
            for d, n in zip(ref["dates"], ref["doy"])]


def figure(var, folds, heads):
    sp = P.SPEC[var]
    cmap, norm = P.cmap_for(var)
    cmap = cmap.with_extremes(bad=LAKE_FILL)   # blanked open water
    # panels are 637 x 665, so the axes box is sized 0.958 high per unit wide;
    # the figure height follows from the margins below to leave no interior gaps
    fig_w = 11.8
    fig = plt.figure(figsize=(fig_w, fig_w * 1.1232), dpi=160)
    axes = fig.subplots(8, N_COLS)
    im = None
    for k, fold in enumerate(P.FOLDS):
        d = folds[fold]
        for half, kind in enumerate(("truth", "pred")):
            r = 2 * k + half
            for c in range(N_COLS):
                ax = axes[r, c]
                im = P.north_up(ax, blank_lakes(d[kind][var][c]),
                                cmap=cmap, norm=norm)
                P.map_off(ax)
                if r == 0:
                    ax.set_title(heads[c], fontsize=P.FS_TITLE,
                                 fontweight="bold", linespacing=1.25, pad=5)
            # stacked on two lines: a single line of 14 characters is as
            # tall as the panel and runs into the row above and below
            lab = f"WY{d['wy']}\n{'ParFlow' if kind == 'truth' else 'DDPM'}"
            ax0 = axes[r, 0]
            ax0.set_ylabel(lab, fontsize=P.FS_TITLE, fontweight="bold",
                           color=P.C_TRUTH if kind == "truth" else P.C_PRED,
                           labelpad=5, linespacing=1.3, ha="center")
            ax0.yaxis.set_visible(True)
            ax0.set_yticks([])

    fig.subplots_adjust(left=0.072, right=0.995, top=0.955, bottom=0.055,
                        wspace=0.025, hspace=0.025)
    # box each water-year block, so the ParFlow/DDPM pair of one year reads
    # as one object; this also opens the gaps BETWEEN years and grows the
    # figure to suit, so it must run before the colourbar axes is placed
    P.year_boxes(fig, axes)
    cax = fig.add_axes([0.30, 0.028, 0.40, 0.0125])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal",
                      boundaries=sp["bounds"], ticks=sp["bounds"],
                      spacing="uniform")
    cb.ax.set_xticklabels(sp["ticklabels"])
    cb.ax.tick_params(labelsize=P.FS_LABEL, pad=2)
    cb.set_label(f"{sp['pretty']} ({sp['unit']}), {sp['note']}",
                 fontsize=P.FS_TITLE, fontweight="bold", labelpad=8)
    P.save(fig, f"fig_evolution_{var}")


def main():
    P.apply_style()
    di = day_index()
    folds = {}
    for fold in P.FOLDS:                      # chronological: WY2003 .. WY2006
        folds[fold] = load_fold(fold, di)
        gc.collect()
    heads = headers(folds)
    print(f"  column headers: {[h.replace(chr(10), ' ') for h in heads]}")
    for var in P.VARS:
        figure(var, folds, heads)


if __name__ == "__main__":
    main()
