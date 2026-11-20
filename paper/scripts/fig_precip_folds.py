"""fig_precip_folds -- accumulated domain-mean precipitation for the four water years.

ONE set of axes, four curves. Plotting the four seasonal accumulations together
rather than one panel per year lets the years be read against each other
directly: the end points are the season totals, the slope at any point is the
domain-mean daily rate, flat stretches are dry spells, and the fan between the
curves is the interannual spread that the cross-validation design has to
contend with.

The daily series itself is deliberately NOT drawn. At \\textwidth = 6.5 in,
four daily strips side by side leave ~1.4 in for 365 bars -- about 0.27 pt per
day -- so the bars merge into a solid block and carry no readable information.
The same content is already available twice over: the curve slope here is the
daily rate, and tables/precip_stats.tex reports the daily distribution
(wet-day fraction, wet-day intensity, 95th percentile) numerically, which is
more precise than anything that could be eyeballed off a 0.27 pt bar. The one
property of the daily series that does not survive accumulation -- how extreme
the biggest event of each year is -- is marked on the curves.

Numbers come from paper/tables/precip_stats.json, written by
compute_precip_stats.py, so the figure and the table can never disagree.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P

P.apply_style()
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

TABLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tables")

# Drawn at its printed size: \textwidth in a 1 in margin article.
FIGSIZE = (6.5, 3.9)


def month_axis(ax, dates, n_days):
    """Tick the first of each month, labelled Oct..Sep, on day-of-water-year.

    Positions are taken from a 365-day water year; in WY2004 every month from
    March on sits one day later, which is 1/365 of the axis and invisible.
    """
    starts = [(i, d[5:7]) for i, d in enumerate(dates) if d[8:10] == "01"]
    ax.set_xticks([i for i, _ in starts])
    ax.set_xticklabels([P.MONTH[m] for _, m in starts], fontsize=P.FS_TICK)
    ax.set_xlim(0, n_days)


def main():
    with open(os.path.join(TABLES, "precip_stats.json")) as f:
        rows = {r["water_year"]: r for r in json.load(f)["rows"]}

    fig, ax = plt.subplots(figsize=FIGSIZE)
    handles = []

    for wy in P.WATER_YEARS:                       # chronological: 2003..2006
        r = rows[wy]
        pr = np.asarray(r["daily_domain_mean_mm"])
        cum = np.concatenate([[0.0], np.cumsum(pr)])   # day 0 .. day N
        t = np.arange(cum.size)
        c = P.WY_COLOR[wy]

        ax.plot(t, cum, color=c, lw=P.LW_MAIN, solid_capstyle="round",
                zorder=3)
        ax.plot([t[-1]], [cum[-1]], marker="o", ms=4.5, color=c, zorder=4)

        # The one feature accumulation hides: the biggest single-day event.
        k = int(np.argmax(pr)) + 1
        ax.plot([k], [cum[k]], marker="o", ms=6.5, mfc="white", mec=c,
                mew=1.4, zorder=5)

        handles.append(Line2D([], [], color=c, lw=P.LW_MAIN,
                              label=f"WY{wy}   {r['season_total_mm']:.0f} mm"))

    handles.append(Line2D([], [], color="#666666", lw=0, marker="o", ms=6.5,
                          mfc="white", mec="#666666", mew=1.4,
                          label="Largest single-day event"))

    n_days = max(len(rows[wy]["dates"]) for wy in P.WATER_YEARS)
    month_axis(ax, rows[2003]["dates"], n_days)
    ax.set_ylim(0, max(r["season_total_mm"] for r in rows.values()) * 1.06)
    ax.set_xlabel("Month of water year")
    ax.set_ylabel("Cumulative precipitation (mm)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=P.C_GRID, lw=0.6)
    ax.set_title("Accumulated domain-mean precipitation by water year",
                 fontsize=P.FS_TITLE, fontweight="bold", loc="left", pad=6)

    leg = ax.legend(handles=handles, loc="upper left", fontsize=P.FS_LEGEND,
                    handlelength=1.6, labelspacing=0.42, borderpad=0.2,
                    handletextpad=0.6)
    for txt in leg.get_texts():
        txt.set_fontfamily("DejaVu Sans")

    fig.tight_layout()
    P.save(fig, "fig_precip_folds")

    for wy in P.WATER_YEARS:                       # cross-check, printed
        r = rows[wy]
        end = float(np.cumsum(r["daily_domain_mean_mm"])[-1])
        print(f"  WY{wy}: curve end {end:8.2f} mm   table total "
              f"{r['season_total_mm']:8.2f} mm   diff {end - r['season_total_mm']:+.2e}")


if __name__ == "__main__":
    main()
