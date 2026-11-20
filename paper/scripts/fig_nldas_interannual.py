"""
fig_nldas_interannual -- how different the four water years are from each other.

The companion to fig_forcing_comparison. That figure holds the year fixed and
varies the forcing; this one holds the forcing fixed -- NLDAS-2 reanalysis
only -- and varies the year. Together they separate the two sources of
variation the cross-validation has to contend with.

Left: accumulated precipitation, so the end point is the season total, the
slope is the daily rate and flat stretches are dry spells. Right: temperature
as a 15-day running mean, because 360 raw daily values x 4 years in one panel
is a solid band.

THE MARKERS ARE RAW DAILY VALUES. The single wettest day of each year is
marked on the left; the hottest and coldest days on the right. On the left the
marker lands on the drawn curve, because the curve is the accumulation of the
same daily series. On the right it cannot: the hottest day is near 31 C while
the 15-day mean through it peaks near 26, so a marker drawn on the smoothed
line alone floats several degrees clear of it and reads as a plotting fault.
Each marker is therefore joined to its own curve by a thin drop line at the
date of the extreme, so the reader is told what the marker is attached to and
reads its distance from the curve as information: that distance is exactly the
amount of extreme the 15-day mean removes. A faint full daily trace was tried
instead and does not work -- a one-day cold spike is a hairline V that
disappears at print size, leaving the marker looking detached after all.

COLOUR MEANS YEAR HERE, AND ONLY HERE. Everywhere else in the paper navy is
ParFlow, orange the reanalysis-forced emulator and green the seasonal-forced
one. Re-using those three for water years -- which an earlier version of this
figure did -- invites a reader who has learned the vocabulary to misread the
panel, so the years get four colours of their own and the caption says so.

The four were chosen by measurement, not by eye. A sequential purple ramp was
tried first: it carries the ordering of the years, but four steps of one hue
are hard to tell apart in a panel where the curves cross repeatedly, which is
exactly what this figure asks the reader to do. The replacement is qualitative
and was picked by maximising the smallest CIE76 separation inside the set
while keeping every member clear of the paper's reserved colours. It scores
dE 72.7 between the closest two years against 59.2 and 44.4 for the runners-up
(red/blue/purple/brown and red/blue/purple/teal), no member is closer than
dE 33.1 to navy, orange, green or grey, and under simulated deuteranopia --
the common form, and the one that defeats red/purple sets -- the closest pair
is still dE 24.5 where both runners-up fall below 10. Ordering is given up in
the trade and now lives only in the key, which is the right way round: the
reader needs to separate the curves more than to rank them.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P
P.apply_style()
import matplotlib.pyplot as plt

CACHE = os.path.join(P.FIGS, "_forcing_comparison.npz")
SMOOTH = 15
# qualitative, maximally separated, and clear of the paper's reserved colours
# (navy ParFlow, orange reanalysis, green seasonal, grey persistence).
# See the module docstring for the measured separations; do not substitute a
# single-hue ramp here, and do not reach for red or a warm brown -- both sit
# close to the emulator orange, and red/purple pairs collapse under the common
# form of colour blindness.
YEAR_C = {2003: "#C2185B",   # magenta
          2004: "#0097A7",   # cyan
          2005: "#B8860B",   # gold
          2006: "#4527A0"}   # indigo


def smooth(a, k=SMOOTH):
    pad = k // 2
    b = np.pad(np.asarray(a, float), (pad, pad), mode="edge")
    return np.convolve(b, np.ones(k) / k, mode="valid")[:len(a)]


def main():
    if not os.path.exists(CACHE):
        raise SystemExit("run fig_forcing_comparison.py first: it builds "
                         f"{os.path.basename(CACHE)}, which this figure reuses "
                         "so the two cannot disagree")
    z = np.load(CACHE, allow_pickle=True)
    FS = 1.15
    fig, (axp, axt) = plt.subplots(1, 2, figsize=(7.4, 3.3))
    x = np.arange(1, 361)

    for wy in P.WATER_YEARS:
        c = YEAR_C[wy]
        p = z[f"np{wy}"]
        t = z[f"nt{wy}"]
        # ---- precipitation ------------------------------------------------
        cum = np.cumsum(p)
        axp.plot(x, cum, color=c, lw=1.8, label=f"WY{wy}", zorder=3)
        iw = int(np.argmax(p))
        axp.plot(iw + 1, cum[iw], "o", mfc="none", mec=c, mew=1.6, ms=6,
                 zorder=4)
        # ---- temperature ---------------------------------------------------
        ts = smooth(t)
        axt.plot(x, ts, color=c, lw=1.8, label=f"WY{wy}", zorder=3)
        ih, ic = int(np.argmax(t)), int(np.argmin(t))
        for i in (ih, ic):
            # drop line from the smoothed curve to the raw daily extreme: it
            # says which curve the marker belongs to, and its length is the
            # peak the 15-day mean has averaged away
            axt.plot([i + 1, i + 1], [ts[i], t[i]], color=c, lw=.9,
                     alpha=.55, zorder=2, solid_capstyle="butt")
        axt.plot([ih + 1, ic + 1], [t[ih], t[ic]], "o", mfc="none", mec=c,
                 mew=1.6, ms=6, ls="none", zorder=4)

    axp.set_ylabel("Accumulated precipitation [mm]", fontsize=P.FS_LABEL * FS)
    axt.set_ylabel("Temperature [${}^{\\circ}$C]",
                   fontsize=P.FS_LABEL * FS)
    axp.set_title("Precipitation", fontsize=P.FS_TITLE * FS, fontweight="bold",
                  pad=4)
    axt.set_title("Temperature", fontsize=P.FS_TITLE * FS, fontweight="bold",
                  pad=4)
    axp.annotate("wettest day", xy=(0.03, 0.93), xycoords="axes fraction",
                 fontsize=P.FS_ANNOT * FS, color="#4A4A4A", va="top")
    axt.annotate("hottest / coldest day,\njoined to its 15-day mean",
                 xy=(0.03, 0.96), xycoords="axes fraction",
                 fontsize=P.FS_ANNOT * FS, color="#4A4A4A", va="top",
                 linespacing=1.3)
    for ax in (axp, axt):
        ax.set_xlabel("Day of water year", fontsize=P.FS_LABEL * FS)
        ax.set_xlim(1, 360)
        ax.set_xticks([1, 90, 180, 270, 360])
        ax.grid(True, lw=.45, color=P.C_GRID)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=P.FS_TICK * FS)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axp.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False,
               fontsize=P.FS_LEGEND * FS, handlelength=1.6, labelspacing=.25)
    fig.suptitle("Interannual variability of the reanalysis forcing",
                 fontsize=P.FS_SUP * FS, fontweight="bold", y=.995)
    fig.subplots_adjust(left=.088, right=.995, top=.845, bottom=.155,
                        wspace=.34)
    P.save(fig, "fig_nldas_interannual")


if __name__ == "__main__":
    main()
