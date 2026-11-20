"""
fig_forcing_comparison -- what the forcing substitution actually changes.

Every result in the paper turns on replacing NLDAS-2 with raw SEAS5 and
changing nothing else, so the reader should be able to see the two forcings
side by side before being shown what the emulator does with them. Four water
years, two rows: accumulated precipitation above, temperature below.

WHY ACCUMULATED PRECIPITATION BUT INSTANTANEOUS TEMPERATURE. Precipitation is
a flux whose hydrological effect is its integral -- the water table responds
to the season's total and to when it arrived, not to any one day -- and an
accumulation curve shows total, timing and dry spells at once. Temperature is
a state: accumulating it would produce a quantity with no physical meaning.
It is drawn as a 15-day running mean of the daily mean (tmin+tmax)/2, because
360 raw daily values in a 1.5 in panel is a solid band.

WHAT IS AVERAGED. Both series are domain means. NLDAS-2 is averaged over the
637x665 1 km grid; SEAS5 over its native ~1 degree grid (10 x 13 cells
covering the domain), area-weighted by cos(latitude). They are therefore the
same quantity computed on the two grids the model actually sees, not one
regridded onto the other.

WHICH MEMBERS. The first 10 of SEAS5's 25 hindcast members -- the ones that
drove the rollouts (rollout_meta_*.json: members [0]..[9]) -- so the spread
drawn here is the spread the emulator was actually given.

THE SPLICE. A single initialisation reaches 215 days, so each water year is
assembled by date routing: the 1 October initialisation until the 1 May one
starts, then the May one, exactly as seas5_chained does for the rollouts. The
changeover is marked.
"""
import datetime as dt
import json
import os
import sys

import netCDF4 as nc
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P
P.apply_style()
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_MEM = 10            # the members the rollouts used
N_DAYS = 360
SMOOTH = 15           # days, for the temperature panel only
CACHE = os.path.join(P.FIGS, "_forcing_comparison.npz")


def _dates(d):
    v = d.variables["valid_date"] if "valid_date" in d.variables else d.variables["time"]
    base = dt.datetime.strptime(v.units.split("since")[1].strip()[:19],
                                "%Y-%m-%d %H:%M:%S")
    return [(base + dt.timedelta(days=float(x))).date() for x in v[:]]


def seas5(wy):
    """(members, 360) precipitation and mean temperature, spliced by date."""
    by_day = {}
    for f in (f"seas5/forcing_coarse_seas5_{wy-1}10.nc",
              f"seas5/forcing_coarse_seas5_{wy}05.nc"):   # later init wins
        d = nc.Dataset(os.path.join(ROOT, "data", f))
        days = _dates(d)
        w = np.cos(np.deg2rad(d.variables["lat"][:]))[None, None, :, None]
        pr = np.asarray(d.variables["raw_precipitation"][:N_MEM], float)
        tm = (np.asarray(d.variables["raw_tmin"][:N_MEM], float)
              + np.asarray(d.variables["raw_tmax"][:N_MEM], float)) / 2 - 273.15
        wb = np.broadcast_to(w, pr.shape)
        pm = np.average(pr, axis=(2, 3), weights=wb)
        tt = np.average(tm, axis=(2, 3), weights=wb)
        for i, day in enumerate(days):
            by_day[day] = (pm[:, i], tt[:, i])
    start = dt.date(wy - 1, 10, 2)
    days = [start + dt.timedelta(days=k) for k in range(N_DAYS)]
    miss = [d for d in days if d not in by_day]
    if miss:
        raise ValueError(f"WY{wy}: {len(miss)} days have no SEAS5 forcing, "
                         f"first {miss[0]} -- the splice does not cover the year")
    return (np.array([by_day[d][0] for d in days]).T,
            np.array([by_day[d][1] for d in days]).T,
            days)


def nldas(wy):
    d = nc.Dataset(os.path.join(ROOT, "data", "parflow_raw", f"forcing_wy{wy}.nc"))
    days = _dates(d)
    i0 = days.index(dt.date(wy - 1, 10, 2))
    sl = slice(i0, i0 + N_DAYS)
    pr = np.asarray(d.variables["precipitation"][sl], float)
    tm = ((np.asarray(d.variables["tmin"][sl], float)
           + np.asarray(d.variables["tmax"][sl], float)) / 2 - 273.15)
    return pr.reshape(N_DAYS, -1).mean(1), tm.reshape(N_DAYS, -1).mean(1)


def load():
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        return {int(w): dict(sp=z[f"sp{w}"], st=z[f"st{w}"],
                             np_=z[f"np{w}"], nt=z[f"nt{w}"],
                             splice=int(z[f"sx{w}"])) for w in P.WATER_YEARS}
    out, store = {}, {}
    for wy in P.WATER_YEARS:
        sp, st, days = seas5(wy)
        npr, ntm = nldas(wy)
        sx = days.index(dt.date(wy, 5, 2))      # the fold's own 2 May
        out[wy] = dict(sp=sp, st=st, np_=npr, nt=ntm, splice=sx)
        store.update({f"sp{wy}": sp, f"st{wy}": st, f"np{wy}": npr,
                      f"nt{wy}": ntm, f"sx{wy}": sx})
        print(f"    WY{wy}: NLDAS {npr.sum():6.0f} mm, SEAS5 "
              f"{sp.sum(1).mean():6.0f} mm "
              f"({100*(sp.sum(1).mean()/npr.sum()-1):+5.1f}%), splice day {sx}")
    np.savez(CACHE, **store)
    return out


def smooth(a, k=SMOOTH):
    pad = k // 2
    b = np.pad(np.asarray(a, float), (pad, pad), mode="edge")
    return np.convolve(b, np.ones(k) / k, mode="valid")[:len(a)]


def main():
    d = load()
    FS = 1.18                                  # this figure is read closely
    fig, axes = plt.subplots(2, len(P.WATER_YEARS), figsize=(7.6, 4.8),
                             sharex=True, sharey="row")
    x = np.arange(1, N_DAYS + 1)
    for j, wy in enumerate(P.WATER_YEARS):
        e = d[wy]
        # ---- accumulated precipitation ----------------------------------
        ax = axes[0][j]
        cs = np.cumsum(e["sp"], axis=1)
        ax.fill_between(x, cs.min(0), cs.max(0), color=P.C_SEAS5, alpha=.18, lw=0)
        ax.plot(x, cs.mean(0), color=P.C_SEAS5, lw=1.5)
        ax.plot(x, np.cumsum(e["np_"]), color=P.C_PRED, lw=1.7)
        ax.set_title(f"WY{wy}", fontsize=P.FS_TITLE * FS, fontweight="bold",
                     pad=3)
        ax.text(.04, .93, f"{np.cumsum(e['np_'])[-1]:.0f} mm",
                transform=ax.transAxes, color=P.C_PRED, fontsize=P.FS_ANNOT*FS,
                fontweight="bold", va="top")
        ax.text(.04, .80, f"{cs.mean(0)[-1]:.0f} mm",
                transform=ax.transAxes, color=P.C_SEAS5, fontsize=P.FS_ANNOT*FS,
                fontweight="bold", va="top")
        # ---- temperature -------------------------------------------------
        ax = axes[1][j]
        ts = np.array([smooth(m) for m in e["st"]])
        ax.fill_between(x, ts.min(0), ts.max(0), color=P.C_SEAS5, alpha=.18, lw=0)
        ax.plot(x, ts.mean(0), color=P.C_SEAS5, lw=1.5)
        ax.plot(x, smooth(e["nt"]), color=P.C_PRED, lw=1.7)
        for ax in (axes[0][j], axes[1][j]):
            ax.axvline(e["splice"] + 1, color="#2B2B2B", lw=1.0, ls=(0, (4, 3)),
                       zorder=1)
            ax.grid(True, lw=.45, color=P.C_GRID)
            ax.set_axisbelow(True)
            ax.set_xlim(1, N_DAYS)
            ax.set_xticks([1, 90, 180, 270, 360])
            ax.tick_params(labelsize=P.FS_TICK * FS)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[1][j].set_xlabel("Lead day", fontsize=P.FS_LABEL * FS)
    axes[0][0].set_ylabel("Accumulated\nprecipitation [mm]",
                          fontsize=P.FS_LABEL * FS)
    axes[1][0].set_ylabel("Temperature [${}^{\\circ}$C]\n15-day mean",
                          fontsize=P.FS_LABEL * FS)

    import matplotlib.lines as ml
    import matplotlib.patches as mp
    h = [ml.Line2D([], [], color=P.C_PRED, lw=1.7, label="NLDAS-2 reanalysis"),
         ml.Line2D([], [], color=P.C_SEAS5, lw=1.5,
                   label="Raw SEAS5, ensemble mean"),
         mp.Patch(facecolor=P.C_SEAS5, alpha=.18,
                  label="SEAS5 ensemble range (10 members)"),
         ml.Line2D([], [], color="#2B2B2B", lw=1.0, ls=(0, (4, 3)),
                   label="Forcing splice, 2 May")]
    # TWO ROWS, NOT ONE, AND THE REASON IS NOT TASTE.
    # paper_style sets savefig.bbox = "tight", so anything that sticks out
    # past the canvas does not get clipped -- it ENLARGES the saved canvas.
    # These four labels on one row are 11.7 in wide, so the file was written
    # 11.7 in across instead of the 7.6 in asked for here, and \textwidth
    # then scaled the whole thing by 0.56 instead of 0.86. The panels lost a
    # third of their size and the apparently empty left margin on the page was
    # the padding tight bbox added to reach the legend. At ncol=2 the legend
    # is narrower than the axes, nothing overhangs, and the figure prints at
    # the size it was designed for. Check the saved pixel width after any
    # change to these labels: it must stay near 7.6 x 200 px.
    fig.legend(handles=h, loc="lower center", ncol=2, frameon=False,
               fontsize=P.FS_LEGEND * FS, bbox_to_anchor=(.5, .004),
               columnspacing=2.2, handlelength=1.9, labelspacing=.45)
    # The title is measured too, for the same reason as the legend: at
    # FS_SUP*FS the longer wording "... against raw seasonal forecast" sets
    # 8.7 in wide and was itself pushing the saved canvas past 7.6 in even
    # after the legend was folded. "Seasonal forecast" is what SEAS5 is, and
    # the caption says so in full, so the title carries the short name and
    # keeps its size -- which is what holds the hierarchy over the panel
    # titles at FS_TITLE*FS.
    fig.suptitle("Atmospheric forcing: reanalysis against raw SEAS5",
                 fontsize=P.FS_SUP * FS, fontweight="bold", y=.995)
    # left clears the two-line y-labels and the 4-digit tick labels; right
    # leaves room for the last panel's "360" tick, which hung over the edge
    # and dragged the canvas with it
    fig.subplots_adjust(left=.125, right=.985, top=.895, bottom=.225,
                        wspace=.16, hspace=.17)
    P.save(fig, "fig_forcing_comparison")


if __name__ == "__main__":
    main()
