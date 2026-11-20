"""
Error-evolution figures and the headline metric table (NLDAS-forced emulator).

Five figures, all drawn with the single line-plot vocabulary of paper_style:
ensemble-mean error, a shaded band for the ensemble spread about that mean,
grey dashed persistence. Nothing else -- no climatology, no single member, no
second lead time.

  fig_error_water_table_depth   THE COMBINED FIGURE: RMSE above, relative
  fig_error_soil_moisture       RMSE below, three series in each panel --
                                persistence, the emulator under NLDAS-2
                                reanalysis forcing, the emulator under raw
                                SEAS5 seasonal forcing -- the four water years
                                averaged, against LEAD TIME. These two replace
                                the eight per-water-year error figures listed
                                next (four here, four in
                                make_seasonal_figures.py), which are still
                                drawn but are no longer used by the paper.

  fig_rmse_water_table_depth    absolute error, 4 rows (WY2003 .. WY2006)
  fig_rmse_soil_moisture
  fig_nrmse_water_table_depth   same series divided by the annual range
  fig_nrmse_soil_moisture
  fig_error_bars                2x2 annual summary, emulator vs persistence

and two deliverables that carry the same numbers:

  paper/tables/nldas_metrics.tex   booktabs tabular, no table wrapper
  paper/tables/nldas_metrics.json

CONVENTIONS, fixed by the cache and by ddpm_model/make_refined_figures.py:

  * "<var>/r_mem" (16, 360), "<var>/r_mean" (360,) and "<var>/r_pers" (360,)
    are ABSOLUTE daily domain RMSE in native units (m for water table depth,
    dimensionless for soil moisture). Verified here: the RMS over the 360 days
    of r_mean reproduces the cached scalar "<var>/rmse" to <1e-4 relative, and
    likewise r_pers vs "<var>/pers".
  * Relative error is 100 * absolute / "<var>/den", where den is the domain-wide
    p95 - p5 of the ParFlow truth over the verifying water year -- exactly the
    normaliser used by compare_folds_relative.py and make_refined_figures.py.
  * THE BAND IS THE ENSEMBLE SPREAD ABOUT THE MEAN, exactly as defined by
    ddpm_model/make_refined_figures.py:fig_evolution_single --

        band = [max(mean - sd_field, 0), mean + sd_field]

    where "<var>/sd_field" is the per-cell ensemble standard deviation
    aggregated over the domain as a ROOT-MEAN-SQUARE (see
    ddpm_model/cache_ensemble_scores.py), i.e. the model's own predictive
    uncertainty. NOTE that the seasonal cache aggregates the same per-cell
    field as an ARITHMETIC MEAN, so the two are NOT the same quantity; the
    combined figures below deal with that explicitly. The mean
    therefore runs down the MIDDLE of a wide band. The earlier min-max envelope
    over the per-member error curves is NOT used: those curves are aggregates
    over ~4x10^5 cells and agree to within a couple of per cent by
    construction, so the envelope was a hairline that also sat entirely above
    the mean. "<var>/r_mem" stays in the cache, untouched, but is not plotted.
  * mean and sd_field are smoothed with analyze_rollout.rolling_mean (7-day,
    edge-padded) before plotting, so the band matches the report figure the
    reviewer referred to. Persistence is drawn RAW, as it is there.

THE COMBINED FIGURES, which carry what eight figures carried before:

  * THE X AXIS IS LEAD TIME, 0 to 359 days from the 1 October initial state,
    because the four folds verify in four DIFFERENT water years and lead time
    is the only coordinate they share. Day 0 is the initial state, where
    persistence is exact by construction; the first forecast day, 2 October,
    is lead day 1. A second axis along the top carries the months, which the
    four folds agree on to within the one day 29 February moves them in
    WY2004.
  * AGGREGATION OVER THE FOLDS: the ARITHMETIC MEAN of the four folds' daily
    domain RMSE at each lead day -- the same aggregation the Mean row of
    nldas_metrics.tex and seasonal_metrics.tex uses. For the relative panel
    each fold is divided by ITS OWN p95 - p5 range BEFORE the four are
    averaged, because the four ranges differ. The ensemble spread is
    aggregated the SAME way and the band is the fold-mean error plus and minus
    the fold-mean spread, so the band is the typical one-year ensemble spread
    rather than a pooled-variance quantity. The 7-day rolling mean is applied
    after the fold average rather than before; both operations are linear and
    edge-padded, so this is exact.
  * ONE DEFINITION OF SPREAD FOR BOTH ARMS. The band would otherwise compare
    two different quantities: this cache's "sd_field" is a ROOT-MEAN-SQUARE
    over the domain, the seasonal cache's "spread" an ARITHMETIC MEAN, and the
    root-mean-square is 18 to 29 % larger here. Both bands therefore use the
    ARITHMETIC MEAN, which is the only one of the two recoverable for both
    arms from what is on disk: the seasonal cache stores nothing else, while
    the reanalysis per-cell standard deviation is still in each fold's
    rollout/ensemble_mean.npz and is re-reduced by nldas_mean_spread() below
    (a re-reduction, not a re-run). The per-fold figures above are NOT changed
    and still show the root-mean-square aggregation.
  * PERSISTENCE IS DRAWN ONCE. The two caches' persistence series, their
    nRMSE denominators and their 360 dates are identical bit for bit, which
    cross_check() asserts rather than assumes.
  * The SEAS5 forcing splice is marked at lead day 212, its position in
    WY2003, WY2005 and WY2006; in the leap water year WY2004 it falls one
    position later, at 213, which is 1/360 of the axis and narrower than the
    line itself.

Usage:
    python make_error_figures.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P                                      # noqa: E402
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.legend_handler import HandlerTuple            # noqa: E402
from matplotlib.lines import Line2D                           # noqa: E402
from matplotlib.patches import Patch                          # noqa: E402
from analyze_rollout import rolling_mean                      # noqa: E402

TABLES = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tables")

YLAB_ABS = {"water_table_depth": "RMSE [m]", "soil_moisture": "RMSE [-]"}
YLAB_REL = "Relative error [% of simulated range]"
TOL = 1e-4          # relative tolerance for the cache self-consistency check


# ------------------------------------------------------------------ data ---
def load_all():
    """Cache for every fold, chronologically keyed, with the scalars verified."""
    R = {}
    for f in P.FOLDS:
        d = P.load_fold_cache(f)
        assert int(d["test_year"]) == P.FOLD_WY[f], (f, d["test_year"])
        for v in P.VARS:
            for series, scalar in (("r_mean", "rmse"), ("r_pers", "pers")):
                x = np.asarray(d[f"{v}/{series}"], dtype=np.float64)
                got = float(np.sqrt(np.mean(x ** 2)))
                want = float(d[f"{v}/{scalar}"])
                assert abs(got - want) <= TOL * want, (f, v, scalar, got, want)
        R[f] = d
    return R


def series(d, var, relative):
    """mean, mean +- ensemble spread, persistence -- absolute or in % of range."""
    s = 100.0 / float(d[f"{var}/den"]) if relative else 1.0
    mean = rolling_mean(np.asarray(d[f"{var}/r_mean"], dtype=np.float64)) * s
    sdf = rolling_mean(np.asarray(d[f"{var}/sd_field"], dtype=np.float64)) * s
    return (mean, np.maximum(mean - sdf, 0.0), mean + sdf,
            np.asarray(d[f"{var}/r_pers"], dtype=np.float64) * s)


def annual(d, var, relative):
    s = 100.0 / float(d[f"{var}/den"]) if relative else 1.0
    return float(d[f"{var}/rmse"]) * s, float(d[f"{var}/pers"]) * s


# -------------------------------------------------------- figures 1 to 4 ---
def fig_rollout(R, var, relative):
    """One row per fold, chronological, each row labelled with its own dates."""
    sp = P.SPEC[var]
    fig, axes = plt.subplots(len(P.FOLDS), 1, figsize=(8.6, 10.2), dpi=200,
                             constrained_layout=True)

    nmem = int(R[P.FOLDS[0]]["n_members"])
    band_label = f"Ensemble mean $\\pm$ spread ({nmem} members)"
    # the band and persistence must both fit; 8% headroom keeps the panel full
    top = max(max(series(R[f], var, relative)[2].max(),
                  series(R[f], var, relative)[3].max()) for f in P.FOLDS)
    for ax, f in zip(axes, P.FOLDS):
        d = R[f]
        mean, lo, hi, pers = series(d, var, relative)
        t = np.arange(len(mean))
        P.ensemble_panel(ax, t, mean, lo, hi, pers=pers,
                         band_label=band_label,
                         label="DDPM ensemble mean")
        ax.set_ylim(0, top * 1.08)
        P.date_ticks(ax, d["dates"], n=7)

        rmse, prs = annual(d, var, relative)
        # soil moisture is dimensionless: print the bare number, not "0.043 -"
        fmt = ("{:.1f}%" if relative else
               ("{:.3f} m" if var == "water_table_depth" else "{:.4f}"))
        ax.text(0.012, 0.94,
                f"WY{P.FOLD_WY[f]}   " + ("nRMSE " if relative else "RMSE ")
                + fmt.format(rmse) + "    persistence " + fmt.format(prs)
                + f"    skill {1.0 - rmse / prs:+.2f}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=P.FS_ANNOT, fontweight="bold", color="#333333")
    axes[-1].set_xlabel("Date")
    # one shared y-label: the relative-error label is too long to repeat on
    # four rows without colliding with the neighbouring tick labels
    fig.supylabel(YLAB_REL if relative else YLAB_ABS[var],
                  fontsize=P.FS_LABEL)

    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(k) for k in (l[-1], band_label, "Persistence")]
    lab = dict(zip(l, l))
    lab["Persistence"] = "Persistence (first forecast day held fixed)"
    fig.legend([h[i] for i in order], [lab[l[i]] for i in order],
               loc="outside lower center", ncol=3, frameon=False)
    fig.suptitle(f"{sp['pretty']}: "
                 + ("relative error" if relative else "error")
                 + " growth through the one-year rollout",
                 fontsize=P.FS_SUP, fontweight="bold")
    P.save(fig, f"fig_{'nrmse' if relative else 'rmse'}_{var}")


# --------------------------------------------------------------- figure 5 ---
def fig_error_bars(R):
    """2x2 annual summary: emulator bar (value printed) + grey persistence bar."""
    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4), dpi=200,
                             constrained_layout=True)
    panels = [(v, rel) for v in P.VARS for rel in (False, True)]
    x = np.arange(len(P.FOLDS))
    w = 0.38
    for ax, (var, rel) in zip(axes.ravel(), panels):
        sp = P.SPEC[var]
        ax.set_axisbelow(True)          # gridlines behind the bars, not over them
        mod = [annual(R[f], var, rel)[0] for f in P.FOLDS]
        prs = [annual(R[f], var, rel)[1] for f in P.FOLDS]
        ax.bar(x - w / 2, mod, w, color=P.C_PRED, label="DDPM ensemble mean")
        ax.bar(x + w / 2, prs, w, color=P.C_PERS, label="Persistence")
        fmt = "{:.1f}" if rel else ("{:.3f}" if var == "water_table_depth"
                                    else "{:.4f}")
        for xi, m, p in zip(x, mod, prs):
            ax.text(xi - w / 2, m, fmt.format(m), ha="center", va="bottom",
                    fontsize=P.FS_ANNOT - 0.5, fontweight="bold",
                    color=P.C_PRED)
            ax.text(xi + w / 2, p, fmt.format(p), ha="center", va="bottom",
                    fontsize=P.FS_ANNOT - 1.5, color="#6E6E6E")
        ax.set_xticks(x)
        ax.set_xticklabels([f"WY{y}" for y in P.WATER_YEARS])
        ax.set_ylim(0, max(prs) * 1.26)
        ax.set_ylabel(YLAB_REL if rel else YLAB_ABS[var])
        ax.set_title(f"{sp['pretty']} — "
                     + ("relative error" if rel else "absolute error"),
                     fontweight="bold")
        ax.grid(axis="x", visible=False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Annual error by verifying water year, emulator against "
                 "persistence", fontsize=P.FS_SUP, fontweight="bold")
    P.save(fig, "fig_error_bars")


# ----------------------------------------------------------------- tables ---
def tables(R):
    rows = []
    for f in P.FOLDS:
        d = R[f]
        for var in P.VARS:
            rmse, pers = annual(d, var, False)
            nrmse, pers_n = annual(d, var, True)
            rows.append(dict(
                fold=int(f), water_year=int(P.FOLD_WY[f]), variable=var,
                variable_pretty=P.SPEC[var]["pretty"], unit=P.SPEC[var]["unit"],
                n_members=int(d["n_members"]),
                date_first=str(d["dates"][0]), date_last=str(d["dates"][-1]),
                annual_range_den=float(d[f"{var}/den"]),
                rmse=rmse, nrmse_pct=nrmse,
                persistence_rmse=pers, persistence_nrmse_pct=pers_n,
                skill_vs_persistence=1.0 - rmse / pers))

    os.makedirs(TABLES, exist_ok=True)
    meta = dict(
        source="ddpm_model/Results/_ensemble_cache/fold{1,2,3,4}.npz",
        forcing="NLDAS reanalysis", n_members=16, n_days=360,
        nrmse_definition="100 * RMSE / (p95 - p5 of the ParFlow truth over the "
                         "verifying water year, domain-wide)",
        skill_definition="1 - RMSE_emulator / RMSE_persistence",
        persistence_definition="day-0 truth state held fixed for 360 days",
        verified="cached scalars reproduced from the daily series to <1e-4 rel.",
        rows=rows)
    with open(os.path.join(TABLES, "nldas_metrics.json"), "w") as fh:
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
    with open(os.path.join(TABLES, "nldas_metrics.tex"), "w") as fh:
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
    print(f"  wrote nldas_metrics.tex / .json in {TABLES}")
    return rows


# ================================================================ #
# THE COMBINED ERROR FIGURES -- what figures 7-10 and 21-24 became. #
# ================================================================ #
# Two figures, one per variable, two stacked panels each (RMSE above,
# relative RMSE below), three series per panel: persistence, the emulator
# under NLDAS-2 reanalysis forcing, the emulator under raw SEAS5 seasonal
# forcing. The four verifying water years no longer occupy four panels; they
# are AVERAGED, and the x axis is therefore LEAD TIME, the only coordinate the
# four folds share. See the module docstring for the whole derivation.

SPREAD_CACHE = os.path.join(P.FIGS, "_nldas_domain_mean_spread.npz")
FS = 1.30                    # type scale for the two combined figures
N_DAYS = 360
N_MEM_NLDAS = 16
N_MEM_SEAS5 = 40
C_SPLICE = "#1A1A1A"
# 2 May is plot position 212 in WY2003, WY2005 and WY2006 and 213 in the leap
# water year WY2004; the line is drawn at the position three of the four folds
# share. One day is 1/360 of the axis, far below the width of the line itself.
SPLICE_POS = 212
YLAB_REL_SHORT = "Relative RMSE [%]"


def nldas_mean_spread(R):
    """Daily domain ARITHMETIC MEAN of the per-cell ensemble standard deviation.

    WHY THIS EXISTS. The reanalysis cache stores "<var>/sd_field", which
    aggregates the per-cell ensemble standard deviation over the domain as a
    ROOT-MEAN-SQUARE; the seasonal cache stores "spread", which aggregates the
    same per-cell field as an ARITHMETIC MEAN. The two are not the same
    quantity -- the RMS is never smaller, and here it is 18 to 29 % larger --
    so a figure that drew one band from each cache would be comparing two
    different definitions of dispersion. The combined figures therefore use
    the ARITHMETIC MEAN for BOTH arms, because that is the only one of the two
    that can be recovered for both from what is already on disk.

    This is a re-reduction, not a re-run: the per-cell standard deviation
    itself is in each fold's rollout/ensemble_mean.npz as "<var>_sd" (360 x
    637 x 665), so the arithmetic mean is read straight off it (~40 s for the
    four folds) and cached in paper/figures/. The ROOT-MEAN-SQUARE is
    recomputed from the same array at the same time and must reproduce
    "<var>/sd_field" exactly -- the check that the right array was read, and
    the check that is re-run every time the cache is loaded.
    """
    if os.path.exists(SPREAD_CACHE):
        Z = np.load(SPREAD_CACHE)
        src = "cached"
    else:
        out = {}
        print(f"  building {os.path.basename(SPREAD_CACHE)} from the stored "
              f"per-cell ensemble standard deviations")
        for f in P.FOLDS:
            z = np.load(os.path.join(
                P.DDPM, "Results", f"groundwater_f15_200ep_fold{f}",
                "rollout", "ensemble_mean.npz"))
            for v in P.VARS:
                sd = z[f"{v}_sd"]
                assert sd.shape[0] == N_DAYS, (f, v, sd.shape)
                out[f"f{f}_{v}_mean"] = np.nanmean(
                    sd, axis=(1, 2)).astype(np.float32)
                out[f"f{f}_{v}_rms"] = np.sqrt(np.nanmean(
                    sd ** 2, axis=(1, 2))).astype(np.float32)
            del z
        os.makedirs(P.FIGS, exist_ok=True)
        np.savez_compressed(SPREAD_CACHE, **out)
        Z = np.load(SPREAD_CACHE)
        src = "rebuilt"

    S = {}
    for f in P.FOLDS:
        S[f] = {}
        for v in P.VARS:
            rms = np.asarray(Z[f"f{f}_{v}_rms"], np.float64)
            want = np.asarray(R[f][f"{v}/sd_field"], np.float64)
            rel = float(np.max(np.abs(rms - want) / np.maximum(want, 1e-12)))
            assert rel <= 1e-6, (f, v, rel)
            S[f][v] = np.asarray(Z[f"f{f}_{v}_mean"], np.float64)
    r = [float(np.mean(S[f][v]) / np.mean(R[f][f"{v}/sd_field"]))
         for f in P.FOLDS for v in P.VARS]
    print(f"  reanalysis ensemble spread re-aggregated as the domain "
          f"arithmetic mean ({src}); it is {100 * min(r):.0f}-{100 * max(r):.0f} % "
          f"of the root-mean-square the cache stores, and the "
          f"root-mean-square recomputed from the same field reproduces "
          f"<var>/sd_field exactly")
    return S


def load_seasonal():
    """The seasonal arm's cached daily series, with its own assertions."""
    import make_seasonal_figures as MS          # local: avoids an import cycle
    S = MS.load_stats()
    for f in P.FOLDS:
        assert MS.splice_index(S[f]) in (SPLICE_POS, SPLICE_POS + 1)
    return S


def cross_check(R, S):
    """What the combined figures may legitimately draw ONCE, not twice.

    Both caches carry a persistence series and a normalising range for every
    fold and variable. The paper asserts that they are the same numbers. They
    are, bit for bit, so persistence is drawn once and the relative axis means
    the same thing for both arms.
    """
    for f in P.FOLDS:
        a = [str(x) for x in R[f]["dates"]]
        b = [str(x) for x in S[f]["dates"]]
        assert a == b, (f, a[0], b[0], a[-1], b[-1])
        for v in P.VARS:
            p, q = (np.asarray(R[f][f"{v}/r_pers"], np.float64),
                    np.asarray(S[f][v]["rmse_pers"], np.float64))
            assert np.array_equal(p, q), (f, v, float(np.abs(p - q).max()))
            assert float(R[f][f"{v}/den"]) == float(S[f][v]["range"]), (f, v)
    print("  the two caches agree EXACTLY on the persistence series, on the "
          "nRMSE denominator (p95 - p5 of the ParFlow truth over the 360-day "
          "window) and on the 360 dates of every fold: persistence is drawn "
          "once")


def combined_series(R, S, SPR, var, relative):
    """The four folds collapsed onto one curve per series, against lead time.

    AGGREGATION, stated once and used for everything in these two figures:
    each fold's daily domain RMSE is first put on the figure's own scale
    (divided by that fold's own p95 - p5 range for the relative panel, because
    the four folds have different ranges and must be normalised BEFORE they
    are pooled), and the four are then averaged day by day -- an ARITHMETIC
    MEAN over the folds, the same aggregation the paper's tables use for their
    Mean column. The ensemble spread is aggregated the SAME way, and the band
    is mean +- that averaged spread, so the band is the typical one-year
    ensemble spread and not a pooled-variance quantity.

    The 7-day rolling mean is applied to the fold-averaged curve rather than
    to each fold: both operations are linear and edge-padded, so they commute
    exactly. Persistence is drawn RAW, as in the figures these replace.
    """
    sc = {f: (100.0 / float(R[f][f"{var}/den"]) if relative else 1.0)
          for f in P.FOLDS}
    agg = lambda get: np.mean(
        [np.asarray(get(f), np.float64) * sc[f] for f in P.FOLDS], axis=0)

    nm = rolling_mean(agg(lambda f: R[f][f"{var}/r_mean"]))
    ns = rolling_mean(agg(lambda f: SPR[f][var]))
    sm = rolling_mean(agg(lambda f: S[f][var]["rmse"]))
    ss = rolling_mean(agg(lambda f: S[f][var]["spread"]))
    pers = agg(lambda f: R[f][f"{var}/r_pers"])
    for a in (nm, ns, sm, ss, pers):
        assert a.shape == (N_DAYS,), a.shape
    return dict(
        t=np.arange(N_DAYS, dtype=float), pers=pers,
        nldas=(nm, np.maximum(nm - ns, 0.0), nm + ns),
        seas5=(sm, np.maximum(sm - ss, 0.0), sm + ss))


def combined_scalars(R, S, var, relative):
    """Four-year mean window RMSE and skill for each arm, as in the tables."""
    out = {}
    rms = lambda a: float(np.sqrt(np.mean(np.asarray(a, np.float64) ** 2)))
    for arm in ("nldas", "seas5"):
        e, p = [], []
        for f in P.FOLDS:
            s = 100.0 / float(R[f][f"{var}/den"]) if relative else 1.0
            if arm == "nldas":
                e.append(float(R[f][f"{var}/rmse"]) * s)
                p.append(float(R[f][f"{var}/pers"]) * s)
            else:
                e.append(rms(S[f][var]["rmse"]) * s)
                p.append(rms(S[f][var]["rmse_pers"]) * s)
        out[arm] = (float(np.mean(e)),
                    float(np.mean([1.0 - a / b for a, b in zip(e, p)])))
        out[f"pers_{arm}"] = float(np.mean(p))
    # The two arms' persistence is the SAME daily series, bit for bit
    # (asserted in cross_check). The two window scalars nevertheless differ in
    # the seventh significant figure, because the reanalysis one is the cached
    # "<var>/pers" -- the RMSE over the whole 4-D residual -- while the
    # seasonal one is the root-mean-square of the float32 daily series. Checked
    # to 1e-5 relative, the same kind of tolerance load_all() uses, and the
    # reanalysis number is the one reported.
    assert abs(out["pers_nldas"] - out["pers_seas5"]) <= 1e-5 * max(
        1.0, out["pers_nldas"]), out
    out["pers"] = out["pers_nldas"]
    return out


def fig_error_combined(R, S, SPR, var):
    """ONE figure per variable: RMSE above, relative RMSE below, three series."""
    sp = P.SPEC[var]
    rc = {"font.size": P.FS_TICK * FS,
          "axes.titlesize": P.FS_TITLE * FS,
          "axes.labelsize": P.FS_LABEL * FS,
          "xtick.labelsize": P.FS_TICK * FS,
          "ytick.labelsize": P.FS_TICK * FS,
          "legend.fontsize": P.FS_LEGEND * 1.22}
    fmt = ("{:.3f} m" if var == "water_table_depth" else "{:.4f}")

    with plt.rc_context(rc):
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 7.9), dpi=200,
                                 sharex=True, constrained_layout=True)
        for ax, relative in zip(axes, (False, True)):
            d = combined_series(R, S, SPR, var, relative)
            k = combined_scalars(R, S, var, relative)
            t = d["t"]

            # the wider (seasonal) band goes down first; both bands carry a
            # thin edge in their own colour, so the overlap still reads as two
            for arm, col in (("seas5", P.C_SEAS5), ("nldas", P.C_PRED)):
                m, lo, hi = d[arm]
                ax.fill_between(t, lo, hi, color=col, alpha=0.20, lw=0,
                                zorder=2.0 if arm == "seas5" else 2.1)
                for y in (lo, hi):
                    ax.plot(t, y, color=col, lw=0.7, alpha=0.75, zorder=2.2)
            ax.axvline(SPLICE_POS, color=C_SPLICE, lw=1.3, ls=(0, (6, 3)),
                       zorder=2.6)
            ax.plot(t, d["pers"], color=P.C_PERS, lw=P.LW_REF * FS, ls="--",
                    zorder=2.8)
            ax.plot(t, d["seas5"][0], color=P.C_SEAS5, lw=P.LW_MAIN * FS,
                    zorder=3.0)
            ax.plot(t, d["nldas"][0], color=P.C_PRED, lw=P.LW_MAIN * FS,
                    zorder=3.1)

            top = max(d["seas5"][2].max(), d["nldas"][2].max(),
                      d["pers"].max())
            # ticks BEFORE the limits: set_xticks autoscales the view out
            # to the last tick, so the limits have to be the final word
            ax.set_xticks(np.arange(0, N_DAYS + 1, 60))
            ax.set_xticks(np.arange(0, N_DAYS + 1, 30), minor=True)
            ax.set_xlim(0, N_DAYS - 1)
            ax.set_ylim(0, top * 1.30)      # headroom for the two text lines
            ax.set_ylabel(YLAB_REL_SHORT if relative else YLAB_ABS[var])
            ax.set_title(("(b)  Relative error, % of each year's own "
                          "simulated range" if relative else
                          "(a)  Absolute error"),
                         fontsize=P.FS_TITLE * 1.18, fontweight="bold",
                         loc="left", pad=5)

            # the four-year mean of the window scalars, in the empty top
            # left corner, on a white backing so the dashed splice line and
            # the bands never run through the text
            f = "{:.1f}%" if relative else fmt
            tag = "nRMSE " if relative else "RMSE "
            rows = [(P.C_PRED, f"NLDAS-2 reanalysis ({N_MEM_NLDAS} members)",
                     k["nldas"]),
                    (P.C_SEAS5, f"Raw SEAS5 seasonal ({N_MEM_SEAS5} members)",
                     k["seas5"]),
                    ("#6E6E6E", "Persistence", (k["pers"], None))]
            for i, (col, name, (rmse, skill)) in enumerate(rows):
                ax.text(0.014, 0.972 - 0.079 * i,
                        f"{name}:   " + tag + f.format(rmse)
                        + ("" if skill is None else f"   skill {skill:+.2f}"),
                        transform=ax.transAxes, ha="left", va="top",
                        fontsize=P.FS_ANNOT * 1.22, fontweight="bold",
                        color=col, zorder=4.0,
                        bbox=dict(facecolor="white", edgecolor="none",
                                  alpha=0.78, pad=1.4))

        axes[-1].set_xlabel("Lead time [days from the 1 October initial "
                            "state]")
        # months, for a reader who thinks in seasons rather than in lead days.
        # Valid for every fold: the four folds' month boundaries agree to
        # within the one day that 29 February moves them in WY2004.
        top_ax = axes[0].secondary_xaxis("top")
        starts = [0, 30, 60, 91, 122, 150, 181, 211, 242, 272, 303, 334, 359]
        top_ax.set_xticks(starts[:-1], minor=True)
        top_ax.set_xticks([(a + b) / 2 for a, b in zip(starts, starts[1:])])
        top_ax.set_xticklabels([P.MONTH[f"{m:02d}"] for m in
                                (10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9)],
                               fontsize=P.FS_TICK * 1.12)
        top_ax.tick_params(which="major", length=0, pad=2)
        top_ax.tick_params(which="minor", length=3.5)

        lw = P.LW_MAIN * FS
        handles = [
            (Patch(color=P.C_PRED, alpha=0.20),
             Line2D([], [], color=P.C_PRED, lw=lw)),
            (Patch(color=P.C_SEAS5, alpha=0.20),
             Line2D([], [], color=P.C_SEAS5, lw=lw)),
            Line2D([], [], color=P.C_PERS, lw=P.LW_REF * FS, ls="--"),
            Line2D([], [], color=C_SPLICE, lw=1.3, ls=(0, (6, 3)))]
        labels = ["DDPM, NLDAS-2 reanalysis forcing",
                  "DDPM, raw SEAS5 seasonal forcing",
                  "Persistence (identical for both)",
                  "SEAS5 forcing splice, 2 May"]
        # the note is the legend's TITLE, not a loose fig.text: only then
        # does constrained_layout reserve room for it and keep it off the
        # legend entries
        note = ("Solid line: ensemble mean.   Shaded band: ensemble mean "
                "$\\pm$ 1 ensemble spread, the per-cell standard\ndeviation "
                "averaged over the domain.   Every series is the mean of the "
                "four verifying\nwater years (WY2003\u2013WY2006) at the same "
                "lead time.")
        leg = fig.legend(handles, labels, loc="outside lower center", ncol=2,
                         frameon=False, handlelength=2.8, columnspacing=2.2,
                         handletextpad=0.8, title=note,
                         # ndivide=1 draws BOTH artists over the whole
                         # handle box, so the handle is the line on its own
                         # band; the default (None) would divide the box
                         # between them and show them side by side
                         handler_map={tuple: HandlerTuple(ndivide=1,
                                                          pad=0.0)})
        leg.get_title().set_fontsize(P.FS_ANNOT * 1.16)
        leg.get_title().set_color("#444444")
        leg.get_title().set_linespacing(1.4)
        fig.suptitle(f"{sp['pretty']}: error growth through the 360-day "
                     f"rollout,\nreanalysis and seasonal forcing against "
                     f"persistence",
                     fontsize=P.FS_SUP * 1.06, fontweight="bold",
                     linespacing=1.25)

        # read back what was PLOTTED, not what was intended
        for ax in axes:
            x0, x1 = ax.get_xlim()
            assert (x0, x1) == (0.0, float(N_DAYS - 1)), (x0, x1)
            v = [ln for ln in ax.lines
                 if len(set(np.asarray(ln.get_xdata(), float))) == 1]
            assert len(v) == 1 and float(v[0].get_xdata()[0]) == SPLICE_POS
        print(f"    {var}: x-axis 0..{N_DAYS - 1} lead days, splice line at "
              f"x={SPLICE_POS} (2 May), {len(labels)} legend entries, "
              f"canvas 7.5 x 7.9 in")
        P.save(fig, f"fig_error_{var}")


def make_combined(R=None, S=None):
    """Entry point for both scripts: the two figures and nothing else."""
    R = load_all() if R is None else R
    S = load_seasonal() if S is None else S
    SPR = nldas_mean_spread(R)
    cross_check(R, S)
    for var in P.VARS:
        fig_error_combined(R, S, SPR, var)


def main():
    P.apply_style()
    R = load_all()
    print("cache scalars verified against the daily series")
    for var in P.VARS:
        for rel in (False, True):
            fig_rollout(R, var, rel)
    fig_error_bars(R)
    tables(R)
    # the two figures that replace the eight per-water-year error panels
    make_combined(R=R)


if __name__ == "__main__":
    main()
