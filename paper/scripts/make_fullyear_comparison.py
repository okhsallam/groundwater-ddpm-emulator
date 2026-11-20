"""
The full-water-year reanalysis-vs-seasonal comparison table.

One table that answers a single question: over the whole 360-day water year,
what is LOST by replacing the NLDAS-2 reanalysis forcing (perfect knowledge of
the weather) with the raw ECMWF SEAS5 seasonal forecast (what would genuinely
have been available on 1 October)? The emulator, its weights and its
normalisation are identical in the two arms; only the 45 forcing channels
differ, so every difference below is a property of the forcing.

  paper/tables/fullyear_comparison.tex    booktabs tabular, no table wrapper
  paper/tables/fullyear_comparison.json   the same numbers with provenance

NOTHING IS RE-ROLLED OUT. Every number comes from one of two cached daily
reductions, both of which the paper already uses:

  reanalysis   ddpm_model/Results/_ensemble_cache/fold{1,2,3,4}.npz
               "<var>/r_mean" (360,)  daily domain RMSE of the 16-member
                                      ensemble mean
               "<var>/r_pers" (360,)  the same for persistence
               "<var>/den"            p95 - p5 of the ParFlow truth over the
                                      verifying water year
               -> this is the cache behind make_error_figures.py and
                  paper/tables/nldas_metrics.{tex,json}

  seasonal     report_full/figures_dyear/_stats_armDyear.pkl
               "<var>/rmse" (360,)      daily domain RMSE of the 40-member
                                        ensemble mean
               "<var>/rmse_pers" (360,) persistence
               "<var>/rmse_clim" (360,) the day-of-year climatology of the
                                        SIMULATOR STATE
               "<var>/range"            the same p95 - p5
               -> this is the cache behind make_seasonal_figures.py and
                  paper/tables/seasonal_metrics.{tex,json}

THE TWO BASELINES ARE SHARED, AND THAT IS NOT AN ASSUMPTION.

  Persistence (day-0 ParFlow state held fixed for 360 days) and the day-of-year
  climatology of the ParFlow state (the mean of the three TRAINING water years
  at the same day of year, built by ddpm_model/make_seas5_figures.py:
  references()) are both functions of the truth alone. They do not know which
  forcing drove the emulator. check_shared_baselines() below therefore verifies,
  day by day and fold by fold, that

      pkl["<var>/rmse_pers"] == npz["<var>/r_pers"]   exactly
      pkl["<var>/range"]     == npz["<var>/den"]      exactly
      pkl["dates"]           == npz["dates"]          exactly

  and only then uses the seasonal cache's "rmse_clim" as the climatology
  baseline for the REANALYSIS arm as well. The reanalysis cache has no
  climatology series of its own; this is the only quantity in the table that is
  not already published for the arm it is applied to, and it is exact, not
  estimated.

AGGREGATION, fixed by the two existing tables so this one cannot disagree:

  * RMSE over a window = root-mean-square of the daily domain RMSE values in
    that window (NOT their average).
  * nRMSE = 100 * RMSE / den, with den the p95 - p5 of the truth over the
    whole verifying water year -- the same denominator at every horizon, so
    the short- and long-lead rows are on one scale.
  * Skill = 1 - RMSE_emulator / RMSE_baseline, computed inside the window.
  * The Mean column is the arithmetic mean of the four water years, exactly as
    in nldas_metrics.tex and seasonal_metrics.tex. The derived "Ratio" and
    "Delta" rows are formed from the entries directly above them in the SAME
    column, so the table reads correctly down a column.

Usage:
    python make_fullyear_comparison.py
"""
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P                                      # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PAPER = os.path.dirname(HERE)
TABLES = os.path.join(PAPER, "tables")
NPZ = os.path.join(P.ROOT, "ddpm_model", "Results", "_ensemble_cache",
                   "fold{}.npz")
PKL = os.path.join(P.ROOT, "report_full", "figures_dyear",
                   "_stats_armDyear.pkl")

N_DAYS = 360
# Early vs late horizon. Day 1 of the rollout is 2 October, so "days 1-90" is
# October to December and "days 271-360" is July to September -- 9 to 12 months
# of lead under the seasonal forcing.
WINDOWS = (("full", 0, 360, "days 1--360"),
           ("early", 0, 90, "days 1--90"),
           ("late", 270, 360, "days 271--360"))


# ------------------------------------------------------------------ data ---
def load():
    """Both caches, keyed by fold, with the shared baselines verified."""
    with open(PKL, "rb") as fh:
        S = pickle.load(fh)
    A = {}
    for f in P.FOLDS:
        Z = np.load(NPZ.format(f), allow_pickle=True)
        d = S[f]
        wy = P.FOLD_WY[f]
        assert int(d["wy"]) == wy and int(Z["test_year"]) == wy
        dates = [str(x) for x in d["dates"]]
        assert dates == [str(x) for x in Z["dates"]], f
        assert len(dates) == N_DAYS, (f, len(dates))
        rec = dict(wy=wy, fold=int(f), dates=dates,
                   n_members_nldas=int(Z["n_members"]),
                   n_members_seas5=int(d["n_members"]))
        for v in P.VARS:
            nld = np.asarray(Z[f"{v}/r_mean"], np.float64)
            se = np.asarray(d[v]["rmse"], np.float64)
            pers_n = np.asarray(Z[f"{v}/r_pers"], np.float64)
            pers_s = np.asarray(d[v]["rmse_pers"], np.float64)
            clim = np.asarray(d[v]["rmse_clim"], np.float64)
            den_n, den_s = float(Z[f"{v}/den"]), float(d[v]["range"])
            # the baselines are functions of the truth, so the two caches must
            # agree on them bit for bit; verified, not assumed
            assert np.array_equal(pers_n, pers_s), (f, v, "persistence differs")
            assert den_n == den_s, (f, v, den_n, den_s)
            assert pers_n[0] == 0.0 and clim.shape == (N_DAYS,)
            assert nld.shape == se.shape == (N_DAYS,)
            rec[v] = dict(nldas=nld, seas5=se, pers=pers_n, clim=clim,
                          den=den_n)
        A[f] = rec
    print(f"loaded {len(A)} folds; persistence series, p95-p5 denominators and "
          f"dates agree exactly between the reanalysis and seasonal caches")
    return A


def rms(a, i, j):
    return float(np.sqrt(np.mean(np.asarray(a[i:j], np.float64) ** 2)))


def cell(rec, var, i, j):
    """Every scalar of one fold/variable/window."""
    d = rec[var]
    den = d["den"]
    out = {}
    for arm in ("nldas", "seas5"):
        r = rms(d[arm], i, j)
        out[arm] = dict(rmse=r, nrmse_pct=100.0 * r / den)
    for base in ("pers", "clim"):
        out[base] = dict(rmse=rms(d[base], i, j),
                         nrmse_pct=100.0 * rms(d[base], i, j) / den)
    for arm in ("nldas", "seas5"):
        for base in ("pers", "clim"):
            out[arm][f"skill_{base}"] = 1.0 - out[arm]["rmse"] / out[base]["rmse"]
    out["ratio_rmse"] = out["seas5"]["rmse"] / out["nldas"]["rmse"]
    out["den"] = den
    return out


def compute(A):
    """R[window][variable][water year] -> the scalars; plus a 'mean' pseudo-year."""
    R = {}
    for name, i, j, _ in WINDOWS:
        R[name] = {}
        for v in P.VARS:
            per = {P.FOLD_WY[f]: cell(A[f], v, i, j) for f in P.FOLDS}
            # arithmetic mean over the four water years, as in the two
            # published tables; ratio and delta are rebuilt from the means so
            # that the Mean column reads consistently down the page
            m = {}
            for arm in ("nldas", "seas5", "pers", "clim"):
                m[arm] = {k: float(np.mean([per[y][arm][k]
                                            for y in P.WATER_YEARS]))
                          for k in per[P.WATER_YEARS[0]][arm]}
            m["ratio_rmse"] = m["seas5"]["rmse"] / m["nldas"]["rmse"]
            m["den"] = float(np.mean([per[y]["den"] for y in P.WATER_YEARS]))
            per["mean"] = m
            R[name][v] = per
    return R


# ------------------------------------------------------------------ check ---
def check_published(R):
    """The full-year block must reproduce the paper's own two tables exactly."""
    for fn, arm, key in (("nldas_metrics.json", "nldas", "skill_vs_persistence"),
                         ("seasonal_metrics.json", "seas5",
                          "skill_vs_persistence")):
        path = os.path.join(TABLES, fn)
        pub = json.load(open(path))["rows"]
        worst = 0.0
        for row in pub:
            y, v = row["water_year"], row["variable"]
            mine = R["full"][v][y]
            for a, b in ((mine[arm]["rmse"], row["rmse"]),
                         (mine[arm]["nrmse_pct"], row["nrmse_pct"]),
                         (mine["pers"]["rmse"], row["persistence_rmse"]),
                         (mine[arm]["skill_pers"], row[key])):
                worst = max(worst, abs(a - b) / max(1e-9, abs(b)))
        assert worst < 1e-5, (fn, worst)
        print(f"  full-year {arm} block reproduces tables/{fn} "
              f"(worst relative difference {worst:.2e})")

    ref = os.path.join(P.ROOT, "report_full", "figures_dyear",
                       "seas5_armDyear_summary.json")
    worst = 0.0
    for row in json.load(open(ref)):
        mine = R["full"][row["variable"]][row["wy"]]
        for a, b in ((mine["seas5"]["rmse"], row["rmse"]),
                     (mine["seas5"]["nrmse_pct"], 100.0 * row["nrmse"]),
                     (mine["seas5"]["skill_pers"], row["skill_pers"]),
                     (mine["seas5"]["skill_clim"], row["skill_clim"])):
            worst = max(worst, abs(a - b) / max(1e-9, abs(b)))
    assert worst < 1e-5, ("armDyear summary", worst)
    print(f"  full-year seas5 skill against BOTH baselines reproduces "
          f"report_full/figures_dyear/seas5_armDyear_summary.json "
          f"(worst relative difference {worst:.2e})")

    # the two headline sentences of Section 'seasonal'
    for v, want_p, want_c in (("water_table_depth", 0.267, 0.017),
                              ("soil_moisture", 0.354, 0.040)):
        m = R["full"][v]["mean"]["seas5"]
        assert abs(round(m["skill_pers"], 3) - want_p) < 5e-4, (v, m)
        assert abs(round(m["skill_clim"], 3) - want_c) < 5e-4, (v, m)
    for v, want in (("water_table_depth", 0.464), ("soil_moisture", 0.565)):
        m = R["full"][v]["mean"]["nldas"]
        assert abs(round(m["skill_pers"], 3) - want) < 5e-4, (v, m)
    print("  the four-year means quoted in paper.tex "
          "(+0.267/+0.017, +0.354/+0.040, +0.464, +0.565) are reproduced")


# ------------------------------------------------------------------- tex ---
COLS = ("WY2003", "WY2004", "WY2005", "WY2006", "mean")
YEARKEY = dict(zip(COLS, list(P.WATER_YEARS) + ["mean"]))


def row(label, sub, vals, fmt, bold=False):
    body = " & ".join(fmt(v) for v in vals)
    lab = f"\\textbf{{{label}}}" if bold and label else (label or "")
    return f"{lab} & {sub} & {body} \\\\"


def block(R, var):
    sp = P.SPEC[var]
    wide = var == "water_table_depth"
    fr = (lambda x: f"{x:.3f}") if wide else (lambda x: f"{x:.4f}")
    fn = lambda x: f"{x:.1f}"
    fs = lambda x: f"${x:+.3f}$"
    ft = lambda x: f"{x:.2f}"
    fd = lambda x: f"${x:+.3f}$"

    def g(win, arm, key):
        return [R[win][var][YEARKEY[c]][arm][key] for c in COLS]

    def diff(win, key):
        return [b - a for a, b in zip(g(win, "nldas", key),
                                      g(win, "seas5", key))]

    unit = "[m]" if wide else "[--]"
    NL, SE = "NLDAS-2", "SEAS5 (raw)"
    L = [f"\\multicolumn{{7}}{{@{{}}l}}{{\\textbf{{{sp['pretty']}}} {unit}}} \\\\",
         "\\addlinespace[1.5pt]"]
    L.append(row("RMSE", NL, g("full", "nldas", "rmse"), fr))
    L.append(row("", SE, g("full", "seas5", "rmse"), fr))
    L.append(row("", "Ratio SEAS5/NLDAS",
                 [R["full"][var][YEARKEY[c]]["ratio_rmse"] for c in COLS], ft))
    L.append("\\addlinespace[2pt]")
    L.append(row("nRMSE [\\%]", NL, g("full", "nldas", "nrmse_pct"), fn))
    L.append(row("", SE, g("full", "seas5", "nrmse_pct"), fn))
    L.append("\\addlinespace[2pt]")
    L.append(row("Skill vs.\\ persistence", NL,
                 g("full", "nldas", "skill_pers"), fs))
    L.append(row("", SE, g("full", "seas5", "skill_pers"), fs))
    L.append(row("", "$\\Delta$ (SEAS5$-$NLDAS)", diff("full", "skill_pers"), fd))
    L.append("\\addlinespace[2pt]")
    L.append(row("Skill vs.\\ climatology", NL,
                 g("full", "nldas", "skill_clim"), fs))
    L.append(row("", SE, g("full", "seas5", "skill_clim"), fs))
    L.append(row("", "$\\Delta$ (SEAS5$-$NLDAS)", diff("full", "skill_clim"), fd))
    L.append("\\addlinespace[2pt]")
    L.append(row("Baseline RMSE", "Persistence", g("full", "pers", "rmse"), fr))
    L.append(row("(identical both arms)", "Climatology",
                 g("full", "clim", "rmse"), fr))
    L.append("\\addlinespace[3pt]")
    L.append("\\multicolumn{7}{@{}l}{\\textit{Lead-time split: the same "
             "quantities over the first and last 90 days}} \\\\")
    L.append("\\addlinespace[1.5pt]")
    L.append(row("nRMSE [\\%], days 1--90", NL,
                 g("early", "nldas", "nrmse_pct"), fn))
    L.append(row("", SE, g("early", "seas5", "nrmse_pct"), fn))
    L.append(row("nRMSE [\\%], days 271--360", NL,
                 g("late", "nldas", "nrmse_pct"), fn))
    L.append(row("", SE, g("late", "seas5", "nrmse_pct"), fn))
    L.append("\\addlinespace[2pt]")
    L.append(row("Skill vs.\\ clim., days 1--90", NL,
                 g("early", "nldas", "skill_clim"), fs))
    L.append(row("", SE, g("early", "seas5", "skill_clim"), fs))
    L.append(row("Skill vs.\\ clim., days 271--360", NL,
                 g("late", "nldas", "skill_clim"), fs))
    L.append(row("", SE, g("late", "seas5", "skill_clim"), fs))
    return L


def write_tex(R):
    # 47 body rows: \footnotesize and tabcolsep 4.2pt bring the natural width
    # to about 433 pt against a 470 pt \textwidth, and arraystretch 0.86 keeps
    # the height inside one float page.
    L = [r"{\footnotesize\setlength{\tabcolsep}{4.2pt}"
         r"\renewcommand{\arraystretch}{0.82}",
         r"\begin{tabular}{@{}l l rrrr @{\hspace{1.1em}} r@{}}", r"\toprule",
         r"Metric & Forcing & WY2003 & WY2004 & WY2005 & WY2006 "
         r"& \textbf{Mean} \\",
         r"\midrule"]
    L += block(R, "water_table_depth")
    L += [r"\midrule"]
    L += block(R, "soil_moisture")
    L += [r"\bottomrule", r"\end{tabular}}"]
    path = os.path.join(TABLES, "fullyear_comparison.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"  wrote {path}")
    return L


def write_json(R, A):
    def pack(win, var, y):
        c = R[win][var][y]
        return dict(
            rmse_nldas=c["nldas"]["rmse"], rmse_seas5=c["seas5"]["rmse"],
            rmse_ratio_seas5_over_nldas=c["ratio_rmse"],
            nrmse_pct_nldas=c["nldas"]["nrmse_pct"],
            nrmse_pct_seas5=c["seas5"]["nrmse_pct"],
            rmse_persistence=c["pers"]["rmse"],
            rmse_climatology=c["clim"]["rmse"],
            nrmse_pct_persistence=c["pers"]["nrmse_pct"],
            nrmse_pct_climatology=c["clim"]["nrmse_pct"],
            skill_vs_persistence_nldas=c["nldas"]["skill_pers"],
            skill_vs_persistence_seas5=c["seas5"]["skill_pers"],
            delta_skill_vs_persistence=(c["seas5"]["skill_pers"]
                                        - c["nldas"]["skill_pers"]),
            skill_vs_climatology_nldas=c["nldas"]["skill_clim"],
            skill_vs_climatology_seas5=c["seas5"]["skill_clim"],
            delta_skill_vs_climatology=(c["seas5"]["skill_clim"]
                                        - c["nldas"]["skill_clim"]),
            normaliser_p95_minus_p5=c["den"])

    meta = dict(
        what="Full-water-year (360-day) comparison of the ParFlow-CLM emulator "
             "driven by NLDAS-2 reanalysis forcing and by raw ECMWF SEAS5 "
             "seasonal forcing. Leave-one-water-year-out, WY2003-2006.",
        recomputed_from_rollouts=False,
        provenance=dict(
            nldas_daily_series="ddpm_model/Results/_ensemble_cache/"
                               "fold{1,2,3,4}.npz, keys '<var>/r_mean' "
                               "(16-member ensemble-mean daily domain RMSE), "
                               "'<var>/r_pers', '<var>/den'; the cache behind "
                               "paper/scripts/make_error_figures.py and "
                               "paper/tables/nldas_metrics.{tex,json}",
            seas5_daily_series="report_full/figures_dyear/_stats_armDyear.pkl, "
                               "keys '<var>/rmse' (40-member ensemble-mean "
                               "daily domain RMSE), '<var>/rmse_pers', "
                               "'<var>/rmse_clim', '<var>/range'; the cache "
                               "behind paper/scripts/make_seasonal_figures.py "
                               "and paper/tables/seasonal_metrics.{tex,json}",
            climatology_baseline="report_full/figures_dyear/"
                                 "_stats_armDyear.pkl, key '<var>/rmse_clim'. "
                                 "Built by ddpm_model/make_seas5_figures.py:"
                                 "references() as the day-of-year mean of the "
                                 "ParFlow state over the three TRAINING water "
                                 "years of each fold. It is a function of the "
                                 "truth alone and so is the SAME baseline for "
                                 "both forcings; the reanalysis cache does not "
                                 "store it, which is why it is read from the "
                                 "seasonal cache for both arms.",
            shared_baseline_check="verified bit-for-bit in load(): the daily "
                                  "persistence series, the p95-p5 normaliser "
                                  "and the 360 dates are identical between the "
                                  "two caches in every fold and variable",
            cross_check="the full-year block reproduces "
                        "paper/tables/nldas_metrics.json, "
                        "paper/tables/seasonal_metrics.json and "
                        "report_full/figures_dyear/seas5_armDyear_summary.json "
                        "to <1e-5 relative, and the four-year means quoted in "
                        "paper.tex (+0.267, +0.017, +0.354, +0.040, +0.464, "
                        "+0.565) exactly"),
        definitions=dict(
            rmse="root-mean-square of the daily domain RMSE values of the "
                 "ensemble mean over the window (not their average)",
            nrmse="100 * RMSE / (p95 - p5 of the ParFlow truth over the "
                  "verifying water year, domain-wide) -- the project's "
                  "existing definition, used unchanged at every horizon so "
                  "the short- and long-lead rows share one scale",
            skill="1 - RMSE_emulator / RMSE_baseline, computed inside the "
                  "window",
            persistence="day-0 ParFlow state held fixed for 360 days",
            climatology="day-of-year mean of the ParFlow state over the three "
                        "training water years of the fold",
            mean_column="arithmetic mean over the four water years of each "
                        "quantity, as in nldas_metrics.tex and "
                        "seasonal_metrics.tex; the Ratio and Delta rows of "
                        "the Mean column are formed from the Mean entries "
                        "above them, so each column reads consistently",
            windows={w[0]: dict(days=f"{w[1] + 1}-{w[2]}", label=w[3])
                     for w in WINDOWS}),
        n_members=dict(nldas=16, seas5=40),
        n_days=N_DAYS,
        folds={f"WY{P.FOLD_WY[f]}": dict(
            fold=int(f), date_first=A[f]["dates"][0],
            date_last=A[f]["dates"][-1]) for f in P.FOLDS},
        results={win: {v: {(f"WY{y}" if y != "mean" else "mean"):
                           pack(win, v, y)
                           for y in list(P.WATER_YEARS) + ["mean"]}
                       for v in P.VARS}
                 for win, _, _, _ in WINDOWS})
    path = os.path.join(TABLES, "fullyear_comparison.json")
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"  wrote {path}")


# ----------------------------------------------------------------- report ---
def report(R):
    for win, i, j, lab in WINDOWS:
        print(f"\n=== {lab} (index {i}..{j - 1}) ===")
        hdr = (f"{'variable':18}{'metric':26}" +
               "".join(f"{c:>10}" for c in COLS))
        print(hdr); print("-" * len(hdr))
        for v in P.VARS:
            w = v == "water_table_depth"
            def line(name, arm, key, f):
                print(f"{v[:17]:18}{name:26}" +
                      "".join(f"{f(R[win][v][YEARKEY[c]][arm][key]):>10}"
                              for c in COLS))
            fr = (lambda x: f"{x:.3f}") if w else (lambda x: f"{x:.4f}")
            line("RMSE reanalysis", "nldas", "rmse", fr)
            line("RMSE seas5", "seas5", "rmse", fr)
            print(f"{v[:17]:18}{'RMSE ratio':26}" +
                  "".join(f"{R[win][v][YEARKEY[c]]['ratio_rmse']:10.2f}"
                          for c in COLS))
            line("nRMSE% reanalysis", "nldas", "nrmse_pct",
                 lambda x: f"{x:.1f}")
            line("nRMSE% seas5", "seas5", "nrmse_pct", lambda x: f"{x:.1f}")
            line("skill_pers reanalysis", "nldas", "skill_pers",
                 lambda x: f"{x:+.3f}")
            line("skill_pers seas5", "seas5", "skill_pers",
                 lambda x: f"{x:+.3f}")
            line("skill_clim reanalysis", "nldas", "skill_clim",
                 lambda x: f"{x:+.3f}")
            line("skill_clim seas5", "seas5", "skill_clim",
                 lambda x: f"{x:+.3f}")
            line("RMSE persistence", "pers", "rmse", fr)
            line("RMSE climatology", "clim", "rmse", fr)
            print()


def main():
    A = load()
    R = compute(A)
    check_published(R)
    report(R)
    L = write_tex(R)
    write_json(R, A)
    print("\n".join(L))


if __name__ == "__main__":
    main()
