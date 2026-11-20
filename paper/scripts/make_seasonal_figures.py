"""
Seasonal half of the results: the emulator driven by RAW (uncorrected) ECMWF
SEAS5, the FULL WATER YEAR -- 360 days from 1 October -- four folds, 40-member
ensembles. One SEAS5 forecast does not reach that far, so the forcing is
SPLICED: days 1 to 211 of the water year come from the preceding 1 October
initialisation and the rest from the 1 May one, the changeover falling on
2 May. This is the rollout arm called "Dyear"; it REPLACES the 210-day,
single-initialisation arm "D" the paper first reported, whose caches under
report_full/figures/ must not be mixed in here.

Exactly the vocabulary of the reanalysis half (make_error_figures.py,
fig_evolution_maps.py), now over the same full water year. Every output is
prefixed "fig_seasonal_".

  fig_seasonal_evolution_water_table_depth   8 rows x 7 columns of maps
  fig_seasonal_evolution_soil_moisture
  fig_seasonal_rmse_water_table_depth        4 rows, one per fold
  fig_seasonal_rmse_soil_moisture
  fig_seasonal_nrmse_water_table_depth       the same series, relative
  fig_seasonal_nrmse_soil_moisture
  fig_seasonal_error_bars                    2x2 summary vs persistence

and, via make_error_figures.make_combined (which owns them), the two figures
that REPLACE the four per-fold error figures above together with the four
reanalysis ones:

  fig_error_water_table_depth    RMSE above, relative RMSE below, three
  fig_error_soil_moisture        series (persistence, reanalysis-forced,
                                 SEAS5-forced), the four folds averaged
                                 against lead time

  paper/tables/seasonal_metrics.tex   booktabs tabular, no table wrapper
  paper/tables/seasonal_metrics.json

CONVENTIONS

  * Daily series come from the cached per-fold reduction
    report_full/figures_dyear/_stats_armDyear.pkl, written by
    ddpm_model/make_seas5_figures.py --arm Dyear: "rmse" is the daily domain
    RMSE of the 40-member ENSEMBLE MEAN, "rmse_pers" the same for the day-0
    state held fixed, "spread" the daily domain-mean ensemble standard
    deviation, and "range" the p95 - p5 of the ParFlow truth over the 360-day
    window.
    "crps" and "ranks" are in the cache but are NOT used by the paper.
  * THE BAND IS THE ENSEMBLE SPREAD ABOUT THE MEAN, exactly as in the
    reanalysis half and as defined by
    ddpm_model/make_refined_figures.py:fig_evolution_single --

        band = [max(mean - spread, 0), mean + spread]

    with "spread" the cached per-cell ensemble standard deviation averaged
    over the domain, and the mean running down its MIDDLE. CAREFUL: the
    reanalysis half's "sd_field" is the same per-cell field aggregated as a
    ROOT-MEAN-SQUARE, not as an arithmetic mean, so the two are NOT the same
    quantity and the bands of these per-fold figures are not strictly
    comparable with those of fig_rmse_* / fig_nrmse_*. The root-mean-square is
    never smaller -- here 18 to 29 % larger -- and the combined figures
    fig_error_* put BOTH arms on the arithmetic mean so that their two bands
    do mean the same thing. The earlier min-max envelope over the per-member error
    curves is NOT plotted: it was a hairline that also sat entirely above the
    mean. paper/figures/_seasonal_member_rmse.npz, built by
    seasonal_prep_member_rmse.py, is untouched and is still loaded as a
    consistency check on the cached ensemble-mean series.
  * THE SPLICE IS MARKED. A thin vertical line on each error panel sits at the
    fold's own 2 May -- plot position 212, or 213 in the leap water year 2004 --
    the day the atmospheric forcing is re-drawn from a fresh initialisation.
    Its position is read from each fold's stored dates, never hard-coded, so a
    change of calendar cannot silently move it.
  * mean and spread are smoothed with analyze_rollout.rolling_mean (7-day,
    edge-padded) before plotting, matching the report figure the reviewer
    referred to. Persistence is drawn RAW, as it is there.
  * The headline RMSE of a fold is the ROOT-MEAN-SQUARE over the 360 daily
    values (not their average) -- the true RMSE over the whole window, and the
    aggregation already used for this arm in seas5_armDyear_summary.json, which
    the numbers below reproduce exactly.
  * Relative error is 100 * absolute / "range".

Map days: see seasonal_prep_evolution.py, which writes
paper/figures/_seasonal_evolution.npz.

Usage:
    python seasonal_prep_evolution.py      # once, ~15 min, builds the map cache
    python make_seasonal_figures.py
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P                                      # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402
from analyze_rollout import rolling_mean                      # noqa: E402
from lake_mask import blank_lakes                             # noqa: E402

TABLES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tables")
STATS = os.path.join(P.ROOT, "report_full", "figures_dyear",
                     "_stats_armDyear.pkl")
EVO = os.path.join(P.FIGS, "_seasonal_evolution.npz")
MEM = os.path.join(P.FIGS, "_seasonal_member_rmse.npz")

N_DAYS = 360
N_MEMBERS = 40
N_COLS = 7

# Open water is blanked in the map panels (see lake_mask.py). DRAWING ONLY:
# every score, table and error series in this script is still computed over
# the full domain. Neither colour scale contains grey, so a blanked cell
# cannot be read as a data class.
LAKE_FILL = "#D9D9D9"

# The SEAS5 forcing is spliced on 2 May, when the October initialisation runs
# out and the May one takes over. Marked, not merely mentioned: the reader has
# to be able to see whether the error does anything there.
SPLICE_MMDD = "05-02"
SPLICE_LABEL = ("Forcing splice, 2 May "
                "(October $\\rightarrow$ May initialisation)")
C_SPLICE = "#1A1A1A"
YLAB_ABS = {"water_table_depth": "RMSE [m]", "soil_moisture": "RMSE [-]"}
YLAB_REL = "Relative error [% of simulated range]"


# ------------------------------------------------------------------ data ---
def load_stats():
    with open(STATS, "rb") as fh:
        S = pickle.load(fh)
    for f in P.FOLDS:
        d = S[f]
        assert int(d["wy"]) == P.FOLD_WY[f], (f, d["wy"])
        assert int(d["n_members"]) == N_MEMBERS, (f, d["n_members"])
        assert len(d["dates"]) == N_DAYS, (f, len(d["dates"]))
        for v in P.VARS:
            for k in ("rmse", "rmse_pers", "spread"):
                assert len(d[v][k]) == N_DAYS, (f, v, k)
            assert float(d[v]["range"]) > 0
            assert float(d[v]["rmse_pers"][0]) == 0.0, (f, v)   # day 0 = truth
    return S


def splice_index(d):
    """Plot position of the 2 May splice, taken from the fold's OWN dates.

    Day 1 of the forecast is 2 October (the rollout emits days 1..360 of the
    water year), so 2 May lands at position 212 -- 213 in WY2004, where
    29 February pushes the spring one day later. Read, not assumed.
    """
    dates = [str(x) for x in d["dates"]]
    hit = [i for i, s in enumerate(dates) if s[5:10] == SPLICE_MMDD]
    assert len(hit) == 1, (dates[0], dates[-1], hit)
    assert hit[0] in (212, 213), (dates[hit[0]], hit[0])
    return hit[0]


def series(d, var, relative):
    """mean, mean +- ensemble spread, persistence -- absolute or in % of range."""
    s = 100.0 / float(d[var]["range"]) if relative else 1.0
    mean = rolling_mean(np.asarray(d[var]["rmse"], np.float64)) * s
    sdf = rolling_mean(np.asarray(d[var]["spread"], np.float64)) * s
    return (mean, np.maximum(mean - sdf, 0.0), mean + sdf,
            np.asarray(d[var]["rmse_pers"], np.float64) * s)


def window(d, var, relative):
    """Scalar RMSE and persistence RMSE over the 360-day window."""
    s = 100.0 / float(d[var]["range"]) if relative else 1.0
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)))
    return rms(d[var]["rmse"]) * s, rms(d[var]["rmse_pers"]) * s


def splice_check(S, half_window=15):
    """Is the error step across the 2 May splice bigger than an ordinary day?

    The forcing is re-drawn there, so a jump would be the one thing to look
    for. Quantified on the RAW daily domain-mean RMSE -- never the smoothed
    curve, which spreads a one-day step over seven days and would hide it:

      step   = rmse[splice] - rmse[splice - 1]
      normal = the |day-to-day change| on every OTHER day of the surrounding
               month (splice +- half_window), as a median and a 95th percentile

    and the step is reported as a multiple of that median. A splice that is
    indistinguishable from ordinary variability sits around 1x and below the
    95th percentile of the neighbourhood.

    A one-day step is not the only way a re-drawn atmosphere could show, so a
    second, slower test runs beside it: the LEVEL SHIFT, the 14-day mean after
    the splice minus the 14-day mean before it, ranked against the same
    statistic computed at every other day of the forecast. That catches a
    change of regime that a one-day difference would miss.
    """
    print(f"  splice diagnostic: step across 2 May vs the day-to-day change "
          f"over the surrounding {2 * half_window + 1} days")
    out = []
    for f in P.FOLDS:
        d = S[f]
        i = splice_index(d)
        for var in P.VARS:
            r = np.asarray(d[var]["rmse"], np.float64)
            step = r[i] - r[i - 1]
            lo, hi = max(i - half_window, 1), min(i + half_window + 1, len(r))
            nb = np.array([abs(r[j] - r[j - 1]) for j in range(lo, hi)
                           if j != i])
            med, p95 = float(np.median(nb)), float(np.percentile(nb, 95))
            rel = abs(step) / med if med > 0 else float("inf")
            # where the step ranks among the neighbourhood's own daily changes
            pct = 100.0 * float(np.mean(nb <= abs(step)))

            # the slower test: a 14-day level shift, ranked against the same
            # statistic at every other day the window is wide enough for
            w = 14
            lev = lambda k: float(r[k:k + w].mean() - r[k - w:k].mean())
            shift = lev(i)
            cand = np.array([abs(lev(k))
                             for k in range(w + 6, len(r) - w - 6)])
            lev_pct = 100.0 * float(np.mean(cand <= abs(shift)))

            out.append(dict(fold=int(f), water_year=int(P.FOLD_WY[f]),
                            variable=var, splice_date=str(d["dates"][i]),
                            splice_position=int(i), step=float(step),
                            neighbour_median_abs_step=med,
                            neighbour_p95_abs_step=p95, ratio_to_median=rel,
                            percentile_in_neighbourhood=pct,
                            distinguishable=bool(abs(step) > p95),
                            level_shift_14day=shift,
                            level_shift_percentile=lev_pct,
                            level_shift_n_compared=int(cand.size)))
            u = "m" if var == "water_table_depth" else ""
            print(f"    WY{P.FOLD_WY[f]} {var:19} step {step:+.5f}{u:>2}"
                  f"   median |step| {med:.5f} p95 {p95:.5f}"
                  f"   {rel:4.2f}x, {pct:5.1f}th pct"
                  f"  -> {'JUMP' if abs(step) > p95 else 'ordinary'}"
                  f"   | 14-day shift {shift:+.5f} ({lev_pct:4.1f}th pct)")
    n = sum(r["distinguishable"] for r in out)
    m = sum(r["level_shift_percentile"] > 95.0 for r in out)
    print(f"    {n} of {len(out)} one-day splice steps exceed the 95th "
          f"percentile of their own neighbourhood")
    print(f"    {m} of {len(out)} 14-day level shifts exceed the 95th "
          f"percentile of the same statistic over the rest of the forecast")
    return out


# -------------------------------------------------------- figures 1 to 4 ---
def fig_rollout(S, var, relative):
    """One row per fold, chronological, each row labelled with its own dates."""
    sp = P.SPEC[var]
    fig, axes = plt.subplots(len(P.FOLDS), 1, figsize=(8.6, 10.2), dpi=200,
                             constrained_layout=True)
    band_label = f"Ensemble mean $\\pm$ spread ({N_MEMBERS} members)"
    # the band and persistence must both fit; 8% headroom keeps the panel full
    top = max(max(series(S[f], var, relative)[2].max(),
                  series(S[f], var, relative)[3].max())
              for f in P.FOLDS)
    for ax, f in zip(axes, P.FOLDS):
        d = S[f]
        mean, lo, hi, pers = series(d, var, relative)
        t = np.arange(len(mean))
        P.ensemble_panel(ax, t, mean, lo, hi, pers=pers, color=P.C_SEAS5,
                         band_label=band_label,
                         label="DDPM ensemble mean")
        sp_i = splice_index(d)
        ax.axvline(sp_i, color=C_SPLICE, lw=1.0, ls="-", zorder=3.5,
                   label=SPLICE_LABEL)
        ax.set_xlim(0, len(mean) - 1)
        ax.set_ylim(0, top * 1.08)
        P.date_ticks(ax, [str(x) for x in d["dates"]], n=7)

        rmse, prs = window(d, var, relative)
        fmt = ("{:.1f}%" if relative else
               ("{:.3f} m" if var == "water_table_depth" else "{:.4f}"))
        ax.text(0.012, 0.94,
                f"WY{P.FOLD_WY[f]}   " + ("nRMSE " if relative else "RMSE ")
                + fmt.format(rmse) + "    persistence " + fmt.format(prs)
                + f"    skill {1.0 - rmse / prs:+.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=P.FS_ANNOT, fontweight="bold", color="#333333")
    axes[-1].set_xlabel("Date")
    fig.supylabel(YLAB_REL if relative else YLAB_ABS[var], fontsize=P.FS_LABEL)

    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(k) for k in ("DDPM ensemble mean", band_label,
                                  "Persistence", SPLICE_LABEL)]
    lab = dict(zip(l, l))
    lab["Persistence"] = "Persistence (first forecast day held fixed)"
    fig.legend([h[i] for i in order], [lab[l[i]] for i in order],
               loc="outside lower center", ncol=4, frameon=False)
    fig.suptitle(f"{sp['pretty']}: "
                 + ("relative error" if relative else "error")
                 + " growth through the 360-day SEAS5-forced water year",
                 fontsize=P.FS_SUP, fontweight="bold")
    # what was actually PLOTTED, not what was intended: the horizon is the
    # whole point of this revision, so it is read back off the axes
    for ax, f in zip(axes, P.FOLDS):
        d = S[f]
        x0, x1 = ax.get_xlim()
        assert (x0, x1) == (0.0, float(N_DAYS - 1)), (f, x0, x1)
        vlines = [ln for ln in ax.lines
                  if len(set(np.asarray(ln.get_xdata(), float))) == 1]
        assert len(vlines) == 1, (f, len(vlines))
        xv = float(np.asarray(vlines[0].get_xdata(), float)[0])
        assert xv == splice_index(d), (f, xv)
        print(f"    WY{P.FOLD_WY[f]}: x-axis {x0:.0f}..{x1:.0f} = "
              f"{int(x1 - x0) + 1} days ({d['dates'][0]} to "
              f"{d['dates'][-1]}); splice line at x={xv:.0f} "
              f"({d['dates'][int(xv)]})")
    P.save(fig, f"fig_seasonal_{'nrmse' if relative else 'rmse'}_{var}")


# --------------------------------------------------------------- figure 5 ---
def fig_error_bars(S):
    """2x2 window summary: emulator bar (value printed) + grey persistence bar."""
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4), dpi=200,
                             constrained_layout=True)
    panels = [(v, rel) for v in P.VARS for rel in (False, True)]
    x = np.arange(len(P.FOLDS))
    w = 0.38
    for ax, (var, rel) in zip(axes.ravel(), panels):
        sp = P.SPEC[var]
        ax.set_axisbelow(True)
        mod = [window(S[f], var, rel)[0] for f in P.FOLDS]
        prs = [window(S[f], var, rel)[1] for f in P.FOLDS]
        ax.bar(x - w / 2, mod, w, color=P.C_SEAS5, label="DDPM ensemble mean")
        ax.bar(x + w / 2, prs, w, color=P.C_PERS, label="Persistence")
        fmt = "{:.1f}" if rel else ("{:.3f}" if var == "water_table_depth"
                                    else "{:.4f}")
        for xi, m, p in zip(x, mod, prs):
            ax.text(xi - w / 2, m, fmt.format(m), ha="center", va="bottom",
                    fontsize=P.FS_ANNOT - 0.5, fontweight="bold",
                    color=P.C_SEAS5)
            ax.text(xi + w / 2, p, fmt.format(p), ha="center", va="bottom",
                    fontsize=P.FS_ANNOT - 1.5, color="#6E6E6E")
        ax.set_xticks(x)
        ax.set_xticklabels([f"WY{y}" for y in P.WATER_YEARS])
        ax.set_ylim(0, max(max(prs), max(mod)) * 1.26)
        ax.set_ylabel(YLAB_REL if rel else YLAB_ABS[var])
        ax.set_title(f"{sp['pretty']} — "
                     + ("relative error" if rel else "absolute error"),
                     fontweight="bold")
        ax.grid(axis="x", visible=False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Full-water-year (360-day) error by verifying water year, "
                 "SEAS5-forced emulator against persistence",
                 fontsize=P.FS_SUP, fontweight="bold")
    P.save(fig, "fig_seasonal_error_bars")


# --------------------------------------------------------- evolution maps ---
def evolution_cache():
    """The seven map days per fold, with their day-of-water-year positions.

    Days 1, 60, 120, 180, 240, 300 and 360 of the water year, so the seven
    columns span the whole 360-day forecast.

    The positions are recovered from day_index.json the way
    ddpm_model/make_seas5_figures.py:references() aligns truth: day0 is the
    first entry of the verifying water year, and a predicted global day g sits
    at truth index g - day0. seasonal_prep_evolution.py already checked each
    stored date against the target file's own time axis.
    """
    Z = np.load(EVO)
    with open(os.path.join(P.DATA, "tiles", "day_index.json")) as fh:
        di = json.load(fh)
    pos = {e["date"]: i for i, e in enumerate(di)}
    out = {}
    for f in P.FOLDS:
        wy = P.FOLD_WY[f]
        day0 = min(i for i, e in enumerate(di) if e["water_year"] == wy)
        dates = [str(d) for d in Z[f"f{f}_dates"]]
        doy = [pos[d] - day0 for d in dates]
        assert int(Z[f"f{f}_nmem"]) == N_MEMBERS
        assert doy == [int(k) + 1 for k in Z[f"f{f}_days"]], (f, doy)
        out[f] = dict(wy=wy, dates=dates, doy=doy,
                      truth={v: Z[f"f{f}_{v}_truth"] for v in P.VARS},
                      pred={v: Z[f"f{f}_{v}_mean"] for v in P.VARS})
        print(f"  fold {f} (WY{wy}): day0={day0}, days of water year {doy}")
        print(f"    dates       {dates}")
    return out


def headers(folds):
    """Month + day-of-water-year, valid for every fold.

    Calendar dates would be true only for the top row: WY2004 is a leap year,
    so the same day of the water year falls one calendar day earlier there.
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


def figure_maps(var, folds, heads):
    sp = P.SPEC[var]
    cmap, norm = P.cmap_for(var)
    cmap = cmap.with_extremes(bad=LAKE_FILL)   # blanked open water
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
            lab = f"WY{d['wy']}\n{'ParFlow' if kind == 'truth' else 'DDPM'}"
            ax0 = axes[r, 0]
            ax0.set_ylabel(lab, fontsize=P.FS_TITLE, fontweight="bold",
                           color=P.C_TRUTH if kind == "truth" else P.C_SEAS5,
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
    P.save(fig, f"fig_seasonal_evolution_{var}")


# ----------------------------------------------------------------- tables ---
def tables(S):
    rows = []
    for f in P.FOLDS:
        d = S[f]
        for var in P.VARS:
            rmse, pers = window(d, var, False)
            nrmse, pers_n = window(d, var, True)
            rows.append(dict(
                fold=int(f), water_year=int(P.FOLD_WY[f]), variable=var,
                variable_pretty=P.SPEC[var]["pretty"], unit=P.SPEC[var]["unit"],
                n_members=int(d["n_members"]), n_days=len(d["dates"]),
                date_first=str(d["dates"][0]), date_last=str(d["dates"][-1]),
                seasonal_range_den=float(d[var]["range"]),
                rmse=rmse, nrmse_pct=nrmse,
                persistence_rmse=pers, persistence_nrmse_pct=pers_n,
                skill_vs_persistence=1.0 - rmse / pers))

    os.makedirs(TABLES, exist_ok=True)
    meta = dict(
        source="report_full/figures_dyear/_stats_armDyear.pkl "
               "(ddpm_model/Results/seas5/armDyear_fold{1,2,3,4})",
        forcing="raw (uncorrected) ECMWF SEAS5, two initialisations spliced "
                "on 2 May (October initialisation to day 211 of the water "
                "year, May initialisation thereafter)",
        n_members=N_MEMBERS,
        n_days=N_DAYS, lead="360 days from 1 October (the full water year)",
        splice_day_of_water_year={str(P.FOLD_WY[f]): int(splice_index(S[f]))
                                  for f in P.FOLDS},
        rmse_definition="root-mean-square over the 360 daily domain RMSE "
                        "values of the ensemble mean",
        nrmse_definition="100 * RMSE / (p95 - p5 of the ParFlow truth over the "
                         "360-day window, domain-wide)",
        skill_definition="1 - RMSE_emulator / RMSE_persistence",
        persistence_definition="day-0 truth state held fixed for 360 days",
        verified="reproduces report_full/figures_dyear/"
                 "seas5_armDyear_summary.json exactly "
                 "(rmse, nrmse, skill_pers)",
        rows=rows)
    with open(os.path.join(TABLES, "seasonal_metrics.json"), "w") as fh:
        json.dump(meta, fh, indent=1)

    def get(y, v):
        return next(r for r in rows if r["water_year"] == y
                    and r["variable"] == v)

    L = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r" & \multicolumn{4}{c}{Water table depth [m]}"
         r" & \multicolumn{4}{c}{Soil moisture [--]} \\",
         r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
         r"Water year & RMSE & nRMSE & Pers. & Skill"
         r" & RMSE & nRMSE & Pers. & Skill \\",
         r" & & [\%] & RMSE & [--] & & [\%] & RMSE & [--] \\",
         r"\midrule"]
    for y in P.WATER_YEARS:
        a, b = get(y, "water_table_depth"), get(y, "soil_moisture")
        L.append(
            f"WY{y} & {a['rmse']:.3f} & {a['nrmse_pct']:.1f} & "
            f"{a['persistence_rmse']:.3f} & {a['skill_vs_persistence']:+.3f} & "
            f"{b['rmse']:.4f} & {b['nrmse_pct']:.1f} & "
            f"{b['persistence_rmse']:.4f} & {b['skill_vs_persistence']:+.3f} \\\\")
    L.append(r"\midrule")
    m = {v: {k: float(np.mean([get(y, v)[k] for y in P.WATER_YEARS]))
             for k in ("rmse", "nrmse_pct", "persistence_rmse",
                       "skill_vs_persistence")} for v in P.VARS}
    a, b = m["water_table_depth"], m["soil_moisture"]
    L.append(
        f"Mean & {a['rmse']:.3f} & {a['nrmse_pct']:.1f} & "
        f"{a['persistence_rmse']:.3f} & {a['skill_vs_persistence']:+.3f} & "
        f"{b['rmse']:.4f} & {b['nrmse_pct']:.1f} & "
        f"{b['persistence_rmse']:.4f} & {b['skill_vs_persistence']:+.3f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TABLES, "seasonal_metrics.tex"), "w") as fh:
        fh.write("\n".join(L) + "\n")

    hdr = (f"{'WY':<7}{'variable':20}{'RMSE':>10}{'nRMSE%':>8}"
           f"{'persRMSE':>10}{'persN%':>8}{'skill':>8}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"WY{r['water_year']:<5}{r['variable']:20}{r['rmse']:10.4f}"
              f"{r['nrmse_pct']:8.2f}{r['persistence_rmse']:10.4f}"
              f"{r['persistence_nrmse_pct']:8.2f}"
              f"{r['skill_vs_persistence']:+8.3f}")
    print(f"  wrote seasonal_metrics.tex / .json in {TABLES}")
    return rows


def check_against_report(rows):
    """The published arm-Dyear summary is the independent check on the scalars."""
    ref = os.path.join(P.ROOT, "report_full", "figures_dyear",
                       "seas5_armDyear_summary.json")
    if not os.path.exists(ref):
        print("  (no seas5_armDyear_summary.json to check against)")
        return
    want = {(r["fold"], r["variable"]): r for r in json.load(open(ref))}
    for r in rows:
        w = want[(r["fold"], r["variable"])]
        for mine, theirs, sc in ((r["rmse"], w["rmse"], 1.0),
                                 (r["nrmse_pct"], w["nrmse"], 100.0),
                                 (r["skill_vs_persistence"],
                                  w["skill_pers"], 1.0)):
            assert abs(mine - theirs * sc) <= 1e-6 * max(1.0, abs(mine)), \
                (r["fold"], r["variable"], mine, theirs * sc)
    print("  scalars match "
          "report_full/figures_dyear/seas5_armDyear_summary.json")


def load_members(S):
    """Per-member daily RMSE, (40, 360) per fold and variable.

    No longer plotted -- the band is the ensemble spread about the mean -- but
    still loaded as the consistency check on the cached series: it re-asserts
    here what seasonal_prep_member_rmse.py asserted when it built the file,
    that the ensemble-mean series it stores is the cached one and that it lies
    below the mean of the member series (averaging the members, not their
    errors).
    """
    if not os.path.exists(MEM):
        raise SystemExit(f"missing {MEM}: run seasonal_prep_member_rmse.py")
    Z = np.load(MEM)
    M, beat, tot = {}, 0, 0
    for f in P.FOLDS:
        assert int(Z[f"f{f}_nmem"]) == N_MEMBERS
        M[f] = {}
        for v in P.VARS:
            mem = np.asarray(Z[f"f{f}_{v}_mem"], np.float64)
            ens = np.asarray(Z[f"f{f}_{v}_ens"], np.float64)
            assert mem.shape == (N_MEMBERS, N_DAYS), (f, v, mem.shape)
            want = np.asarray(S[f][v]["rmse"], np.float64)
            assert np.max(np.abs(ens - want) / np.maximum(want, 1e-12)) <= 1e-3
            assert np.all(ens < mem.mean(0)), (f, v)
            beat += int(np.sum(ens < mem.min(0)))
            tot += N_DAYS
            M[f][v] = mem
    print(f"  per-member envelope: the ensemble mean is more accurate than the "
          f"BEST individual member on {beat} of {tot} "
          f"fold-variable-day combinations")
    return M


def main():
    P.apply_style()
    S = load_stats()
    print(f"cached daily series verified ({N_DAYS} days, {N_MEMBERS} members, "
          f"{len(P.FOLDS)} folds)")
    for f in P.FOLDS:
        d = S[f]
        print(f"  WY{P.FOLD_WY[f]}: {d['dates'][0]} .. {d['dates'][-1]}, "
              f"splice {d['dates'][splice_index(d)]} at plot position "
              f"{splice_index(d)}")
    load_members(S)
    spl = splice_check(S)
    rows = tables(S)
    check_against_report(rows)
    with open(os.path.join(TABLES, "seasonal_metrics.json")) as fh:
        meta = json.load(fh)
    meta["splice_diagnostic"] = spl
    with open(os.path.join(TABLES, "seasonal_metrics.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    for var in P.VARS:
        for rel in (False, True):
            fig_rollout(S, var, rel)
    fig_error_bars(S)
    if not os.path.exists(EVO):
        raise SystemExit(f"missing {EVO}: run seasonal_prep_evolution.py first")
    folds = evolution_cache()
    heads = headers(folds)
    print(f"  column headers: {[h.replace(chr(10), ' ') for h in heads]}")
    for var in P.VARS:
        figure_maps(var, folds, heads)
    # The two COMBINED error figures -- fig_error_water_table_depth and
    # fig_error_soil_moisture -- carry this arm beside the reanalysis one and
    # replace fig_seasonal_rmse_* / fig_seasonal_nrmse_* in the paper. They
    # live in make_error_figures.py, which owns the reanalysis cache, and are
    # produced from here as well so that either script on its own leaves the
    # paper's figure set complete. The already-loaded seasonal stats are
    # handed over, so nothing is read twice.
    import make_error_figures as ME               # local: no import cycle
    ME.make_combined(S=S)


if __name__ == "__main__":
    main()
