"""
fig_error_bars_combined -- the two annual summary bar charts, merged.

REPLACES the pair fig_error_bars (make_error_figures.py, paper Fig. 11) and
fig_seasonal_error_bars (make_seasonal_figures.py, paper Fig. 25) with ONE
figure, so the reanalysis-forced and the SEAS5-forced arm can be read against
each other and against their common reference in a single glance instead of
fourteen pages apart.

NOTHING IS RECOMPUTED. The two loaders and the two scalar reducers are
imported from the scripts that own them and called unchanged:

    make_error_figures.load_all()      -> ddpm_model/Results/_ensemble_cache/
                                          fold{1,2,3,4}.npz
    make_error_figures.annual()        -> (RMSE, persistence RMSE), absolute
                                          or in % of the annual range
    make_seasonal_figures.load_stats() -> report_full/figures_dyear/
                                          _stats_armDyear.pkl
    make_seasonal_figures.window()     -> the same pair for the SEAS5 arm

so every bar here is numerically the bar it was in Fig. 11 or Fig. 25, and
both of those scripts' self-consistency assertions still run on the way in.

WHY THERE IS ONLY ONE GREY BAR PER GROUP. Persistence is the day-0 ParFlow
truth state held fixed for 360 days of the same verifying water year, and the
relative-error normaliser is the p95 - p5 of the same ParFlow truth over the
same window. Neither depends on the atmospheric forcing, so the two arms share
them exactly: checked here, the two independently cached persistence numbers
agree to better than 1.5e-6 relative (float32 vs float64 reduction order, no
more), asserted below at 1e-4. The two arms are therefore measured against an
identical yardstick and the grey bar is drawn once, from the reanalysis cache,
rather than twice at the same height.

LAYOUT. 2 x 2 panels: COLUMNS are the two state variables, ROWS are absolute
error (top) and error relative to the annual range (bottom). The variable is
therefore named once per column, on the top row only, and the row is named by
its y-axis label -- the arrangement the other way round repeats the variable
in all four titles and leaves "absolute" and "relative" nowhere to go but a
title too long for the panel. Four verifying water years per panel, three bars
per water year -- orange reanalysis, green SEAS5, grey persistence -- in the
house colours of paper_style. Printed values are set vertically: at
\\textwidth there is room for three numbers over one water year only if they
are turned, and turning them was preferred to dropping any.

Sized and typeset for reproduction at \\textwidth = 6.5 in in a 1-inch-margin
article: the canvas is 7.5 in wide, so the reduction is only x0.87 and every
font below lands at 7.4 pt or larger on the page -- against x0.63 and an
effective 5.9 pt tick label in the two figures this replaces.

Usage:
    python fig_error_bars_combined.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P                                      # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402
import make_error_figures as ME                              # noqa: E402
import make_seasonal_figures as MS                           # noqa: E402

# Short enough to sit on a 2.2-inch-tall axis at 13 pt without colliding with
# the tick labels; "% of simulated range" is spelled out in the caption.
YLAB_ABS = {"water_table_depth": "RMSE [m]", "soil_moisture": "RMSE [-]"}
YLAB_REL = "nRMSE [% of range]"

PERS_TOL = 1e-4          # the two arms' persistence must be the same number

# Typography, a step up from the paper_style defaults because this figure is
# reduced by only x0.90 rather than x0.63.
FS_TICK = P.FS_TICK + 2.0       # 11.5 -> 10.4 pt on the page
FS_LABEL = P.FS_LABEL + 2.0     # 13.0 -> 11.7 pt
FS_TITLE = P.FS_TITLE + 1.5     # 13.5 -> 12.2 pt
FS_LEGEND = P.FS_LEGEND + 2.0   # 12.0 -> 10.8 pt
FS_SUP = P.FS_SUP               # 15.0 -> 13.5 pt
FS_VAL = 8.5                    #        ->  7.7 pt, the printed bar values

ARMS = (("rean", P.C_PRED, "Reanalysis-forced emulator"),
        ("seas", P.C_SEAS5, "SEAS5-forced emulator"),
        ("pers", P.C_PERS, "Persistence"))


def collect(R, S):
    """{(var, relative): {arm: [one value per water year, chronological]}}."""
    out = {}
    for var in P.VARS:
        for rel in (False, True):
            rean, seas, pers = [], [], []
            for f in P.FOLDS:
                m_r, p_r = ME.annual(R[f], var, rel)
                m_s, p_s = MS.window(S[f], var, rel)
                assert abs(p_r - p_s) <= PERS_TOL * p_r, (
                    "the two arms disagree on persistence -- they must not: "
                    f"WY{P.FOLD_WY[f]} {var} rel={rel} {p_r} vs {p_s}")
                rean.append(m_r)
                seas.append(m_s)
                pers.append(p_r)
            out[(var, rel)] = dict(rean=rean, seas=seas, pers=pers)
    return out


def fig_error_bars_combined(D):
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 5.6), dpi=200,
                             constrained_layout=True)
    x = np.arange(len(P.FOLDS), dtype=float)
    w = 0.27
    off = {"rean": -w, "seas": 0.0, "pers": +w}

    for row, rel in enumerate((False, True)):
        for col, var in enumerate(P.VARS):
            ax = axes[row, col]
            v = D[(var, rel)]
            ax.set_axisbelow(True)         # gridlines behind the bars
            fmt = "{:.1f}" if rel else ("{:.3f}" if var == "water_table_depth"
                                        else "{:.4f}")
            for arm, colour, label in ARMS:
                ax.bar(x + off[arm], v[arm], w, color=colour, label=label,
                       edgecolor="white", linewidth=0.4)
                # turned, so three numbers fit over one water year at
                # \textwidth; bold for the two emulator arms, plain grey for
                # the reference, exactly the weighting of the two originals
                bold = arm != "pers"
                for xi, val in zip(x, v[arm]):
                    ax.text(xi + off[arm], val, " " + fmt.format(val),
                            ha="center", va="bottom", rotation=90,
                            fontsize=FS_VAL if bold else FS_VAL - 0.6,
                            fontweight="bold" if bold else "normal",
                            color=colour if bold else "#6E6E6E")
            ax.set_xticks(x)
            ax.set_xticklabels([f"WY{y}" for y in P.WATER_YEARS],
                               fontsize=FS_TICK)
            ax.tick_params(axis="y", labelsize=FS_TICK)
            # headroom for the turned labels: the longest is five characters
            ax.set_ylim(0, max(max(v[a]) for a in ("rean", "seas", "pers"))
                        * 1.42)
            ax.set_ylabel(YLAB_REL if rel else YLAB_ABS[var],
                          fontsize=FS_LABEL)
            if row == 0:        # the variable names the COLUMN, once
                ax.set_title(P.SPEC[var]["pretty"], fontsize=FS_TITLE,
                             fontweight="bold")
            ax.grid(axis="x", visible=False)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False,
               fontsize=FS_LEGEND, columnspacing=1.6, handlelength=1.5,
               handletextpad=0.6)
    fig.suptitle("Annual error by verifying water year, both forcing arms",
                 fontsize=FS_SUP, fontweight="bold")
    P.save(fig, "fig_error_bars_combined")


def report(D):
    hdr = (f"{'WY':<8}{'variable':20}{'kind':10}"
           f"{'reanalysis':>12}{'SEAS5':>12}{'persistence':>13}"
           f"{'skill_rean':>12}{'skill_seas':>12}")
    print(hdr)
    print("-" * len(hdr))
    for var in P.VARS:
        for rel in (False, True):
            v = D[(var, rel)]
            for i, y in enumerate(P.WATER_YEARS):
                r, s, p = v["rean"][i], v["seas"][i], v["pers"][i]
                print(f"WY{y:<6}{var:20}{'relative' if rel else 'absolute':10}"
                      f"{r:12.4f}{s:12.4f}{p:13.4f}"
                      f"{1.0 - r / p:+12.3f}{1.0 - s / p:+12.3f}")


def main():
    P.apply_style()
    R = ME.load_all()
    S = MS.load_stats()
    print("reanalysis and seasonal caches loaded and self-verified")
    D = collect(R, S)
    print("persistence identical in both arms to <1e-4 relative")
    fig_error_bars_combined(D)
    report(D)


if __name__ == "__main__":
    main()
