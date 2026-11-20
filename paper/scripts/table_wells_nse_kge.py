"""
Table: NSE and KGE at the 45 observation wells, for both comparisons.

TWO QUESTIONS, NOT ONE. The emulator is asked to reproduce ParFlow, and
ParFlow is asked to reproduce the real world. Scoring the emulator against the
observation alone confounds the two, so every well here carries both:

  (a) EMULATOR vs PARFLOW at the same grid cell -- how well the emulator does
      the job it was trained to do. ParFlow is the reference, so the whole
      360-day water year is available at every well and all 45 score.
  (b) EMULATOR vs OBSERVATION -- how well the emulator tracks the aquifer,
      which it can only do as far as ParFlow does. The reference is the well
      record, which at most of the ISWS network is monthly or quarterly, so
      this comparison is limited by the sampling and not every well scores.

Each under both forcings: reanalysis (NLDAS-2) and raw seasonal (SEAS5, the
360-day "Dyear" arm).

DEFINITIONS.

    NSE = 1 - sum((sim - ref)^2) / sum((ref - mean(ref))^2)

    KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
    alpha = sd(sim) / sd(ref)      beta = mean(sim) / mean(ref)

This is KLING-GUPTA 2009, the original. The 2012 revision replaces beta with
gamma = CV(sim)/CV(ref), which divides by the mean twice and is therefore
strictly less stable on a quantity that can sit near zero, which is exactly
the hazard here.

BETA IS THE FRAGILE TERM. It is a ratio of means, and these are depths to
water that have had one constant per site removed and are then held at or
above zero: a well whose shifted record hovers just below the land surface has
a mean of a few centimetres, and beta -- and with it KGE -- is then a division
artefact rather than a bias ratio. wells_common.kge therefore refuses the
division unless |mean(ref)| exceeds BETA_MIN_SD = 0.25 times sd(ref): the
denominator must be large compared with the variability the metric is trying
to describe. Where it is not, KGE is "not scored" and the reason is recorded
in the JSON; NSE, which has no ratio of means in it, is still reported.

SPARSE RECORDS. The paper's rule is used unchanged: a minimum of 30 MATCHED
DAYS, under a 3-day tolerance against the observation and 1 day against
ParFlow. Below that a well is reported as not scored rather than given a
number, because a handful of quarterly readings can produce an NSE of any
value at all. Twelve of the 45 wells are read too sparsely to reach 30 matched
days in four water years, and they are the twelve the roster table already
shows with n < 10.

Writes paper/tables/wells_nse_kge.tex (a bare booktabs tabular, no table
wrapper, to match the other files in paper/tables/) and
paper/tables/wells_nse_kge.json, which carries every number including the
soil-moisture-against-ParFlow comparison the printed table has no room for.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P
P.apply_style()

import wells_common as C

TAB = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "tables")
LO, HI = pd.Timestamp("2002-10-01"), pd.Timestamp("2006-09-30")
NMIN = 30                 # the paper's rule, unchanged
TOL_OBS = "3D"            # an observation is never carried more than 3 days
TOL_SIM = "1D"            # ParFlow and the emulator are both daily


def _win(s):
    return None if s is None else s[(s.index >= LO) & (s.index <= HI)]


def _fmt(v, nd=2):
    """A metric cell, or an en dash where the record cannot support one.

    A value that rounds to zero is printed as 0.00 whatever its sign: an
    "$-$0.00" in a column of efficiencies reads as a typesetting error, and
    the sign of a quantity four thousandths from zero carries no meaning.
    """
    if v is None or not np.isfinite(v):
        return "--"
    if abs(v) < 0.5 * 10 ** (-nd):
        return f"{0.0:.{nd}f}"
    return f"{v:.{nd}f}".replace("-", "$-$")


def main():
    os.makedirs(TAB, exist_ok=True)
    ws = C.load_wells()
    obs = C.observations(ws)
    mods = C.available(ws)
    keys = [m["key"] for m in mods]
    print(f"  forcings present: {', '.join(keys)}")
    ref_mod = next((m for m in mods if m["key"] == "nldas"), mods[0])

    # the observation the paper plots and scores: shifted, capped, floored
    offs, obs_sh = C.shifted_observations(ws, obs, ref_mod["wtd"])
    pf_wtd = C.parflow_truth("water_table_depth")
    pf_sm = C.parflow_truth("soil_moisture")

    rows = []
    for w in ws:
        sid = w["short_id"]
        rec = dict(no=w["no"], short_id=sid, name=w["pretty"],
                   network=w["net"], offset_applied_m=round(offs[sid], 3))
        o = _win(obs_sh.get(sid))
        p = _win(pf_wtd.get(sid))
        s = _win(pf_sm.get(sid))
        for m in mods:
            k = m["key"]
            # (a) against the simulator, water table depth
            e = C.efficiency(p, m["wtd"].get(sid), tol=TOL_SIM, nmin=NMIN)
            rec[f"sim_wtd_{k}"] = e
            # (a') against the simulator, soil moisture -- JSON only
            rec[f"sim_sm_{k}"] = C.efficiency(
                s, m["sm"].get(sid), tol=TOL_SIM, nmin=NMIN)
            # (b) against the observation, water table depth
            rec[f"obs_wtd_{k}"] = C.efficiency(
                o, m["wtd"].get(sid), tol=TOL_OBS, nmin=NMIN)
        rows.append(rec)

    # ------------------------------------------------------------ summary --
    COLS = [(f"{cmp_}_wtd_{k}", f"{cmp_}/{k}")
            for cmp_ in ("sim", "obs") for k in keys]

    def col(field, metric):
        return np.array([r[field][metric] for r in rows], float)

    summary = {}
    for field, lab in COLS:
        for metric in ("nse", "kge", "r", "alpha", "beta"):
            v = col(field, metric)
            v = v[np.isfinite(v)]
            summary[f"{field}_{metric}_median"] = (round(float(np.median(v)), 3)
                                                   if v.size else None)
            summary[f"{field}_{metric}_n_wells"] = int(v.size)
        summary[f"{field}_n_scored"] = int(sum(r[field]["scored"]
                                               for r in rows))
        summary[f"{field}_n_beta_refused"] = int(
            sum(r[field]["scored"] and not r[field]["beta_ok"] for r in rows))

    # also the soil-moisture arm, JSON only
    for k in keys:
        f = f"sim_sm_{k}"
        for metric in ("nse", "kge"):
            v = np.array([r[f][metric] for r in rows], float)
            v = v[np.isfinite(v)]
            summary[f"{f}_{metric}_median"] = (round(float(np.median(v)), 3)
                                               if v.size else None)
        summary[f"{f}_n_scored"] = int(sum(r[f]["scored"] for r in rows))

    unscored = {f: [r["no"] for r in rows if not r[f]["scored"]]
                for f, _ in COLS}

    json.dump(dict(
        definitions=dict(
            nse="1 - sum((sim-ref)^2) / sum((ref-mean(ref))^2)",
            kge="Kling-Gupta 2009: 1 - sqrt((r-1)^2+(alpha-1)^2+(beta-1)^2), "
                "alpha = sd(sim)/sd(ref), beta = mean(sim)/mean(ref). NOT the "
                "2012 revision.",
            beta_guard=(
                "beta is a ratio of means and is a division artefact when the "
                "reference mean is near zero, which a shifted depth to water "
                "can be. KGE is reported only where |mean(ref)| >= "
                f"{C.BETA_MIN_SD} * sd(ref); otherwise kge and beta are null, "
                "beta_ok is false, and NSE, r and alpha are still given."),
            comparisons=dict(
                sim_wtd="emulator water table depth vs ParFlow at the same "
                        "grid cell, 1-day matching",
                sim_sm="emulator soil moisture vs ParFlow at the same grid "
                       "cell, 1-day matching",
                obs_wtd="emulator water table depth vs the well observation, "
                        "3-day matching, observation shifted by one capped "
                        "constant per site and floored at zero for the "
                        "ISWS/UIUC network"),
            nmin=NMIN,
            nmin_note=("a well with fewer than %d matched days is reported as "
                       "not scored (scored=false) rather than given a number; "
                       "every metric is then null and n records how many days "
                       "were in fact matched" % NMIN),
            window=[str(LO.date()), str(HI.date())]),
        forcings=keys, summary=summary, unscored_well_numbers=unscored,
        wells=rows),
        open(os.path.join(TAB, "wells_nse_kge.json"), "w"), indent=1,
        default=lambda x: None if isinstance(x, float) and not np.isfinite(x)
        else (bool(x) if isinstance(x, (bool, np.bool_)) else float(x)))

    # -------------------------------------------------------------- latex --
    # ELEVEN COLUMNS BY FORTY-FIVE ROWS, AND IT IS THE HEIGHT THAT BINDS.
    # The width is comfortable: the widest cell is "$-$11.76" and the whole
    # tabular comes to about 5.5 in against a 6.5 in text block. The height
    # does not. Forty-five body rows plus four header rows, two summary rows
    # and the booktabs rules come to about 8.3 in at \footnotesize (9 pt on an
    # 11 pt baseline), which with even a three-line caption overruns the
    # 9.0 in text block. At \scriptsize (8 pt on 9.5 pt) the same table is
    # about 7.2 in and fits a page with room for the caption. The alternative
    # -- dropping the twelve wells that are too sparsely read to score
    # against the observation -- would also drop their emulator-vs-ParFlow
    # numbers, which are perfectly well determined, so the type size gives
    # way instead of the data.
    #
    # The two comparisons are separated by a vertical rule because they
    # answer different questions and must not be read as one block.
    L = [r"{\scriptsize\setlength{\tabcolsep}{3.2pt}"
         # 48 rows: 45 fitted at the default row height, 48 overran the text
         # block by 11 pt, so the rows are tightened rather than the type.
         r"\renewcommand{\arraystretch}{0.92}",
         r"\begin{tabular}{r l r @{\hspace{0.7em}} r r r r "
         r"@{\hspace{0.7em}} | @{\hspace{0.7em}} r r r r}",
         r"\toprule",
         r"& & & \multicolumn{4}{c}{Emulator vs.\ ParFlow} & "
         r"\multicolumn{4}{c}{Emulator vs.\ observation} \\",
         r"\cmidrule(lr){4-7} \cmidrule(lr){8-11}",
         r"& & & \multicolumn{2}{c}{Reanalysis} & "
         r"\multicolumn{2}{c}{Seasonal} & \multicolumn{2}{c}{Reanalysis} & "
         r"\multicolumn{2}{c}{Seasonal} \\",
         r"\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9} "
         r"\cmidrule(lr){10-11}",
         r"\# & Well & $n_{\mathrm{match}}$ & NSE & KGE & NSE & KGE & NSE & KGE & NSE & KGE \\",
         r"\midrule"]
    order = [f"sim_wtd_{k}" for k in keys] + [f"obs_wtd_{k}" for k in keys]
    for r in rows:
        n_obs = r[f"obs_wtd_{keys[0]}"]["n"]
        cells = [str(r["no"]), r["name"], str(int(n_obs))]
        for f in order:
            cells += [_fmt(r[f]["nse"]), _fmt(r[f]["kge"])]
        L.append(" & ".join(cells) + r" \\")
    L.append(r"\midrule")
    med = ["", r"\textbf{Median}", ""]
    for f in order:
        med += [_fmt(summary[f"{f}_nse_median"]),
                _fmt(summary[f"{f}_kge_median"])]
    L.append(" & ".join(med) + r" \\")
    nsc = ["", "Wells scored", ""]
    for f in order:
        nsc += [str(summary[f"{f}_n_scored"]),
                str(summary[f"{f}_kge_n_wells"])]
    L.append(" & ".join(nsc) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    open(os.path.join(TAB, "wells_nse_kge.tex"), "w").write("\n".join(L) + "\n")
    print(f"  wrote {TAB}/wells_nse_kge.tex and wells_nse_kge.json")

    # ------------------------------------------------------------- stdout --
    show = pd.DataFrame([
        dict(no=r["no"], name=r["name"], net=r["network"],
             n_obs=r[f"obs_wtd_{keys[0]}"]["n"],
             **{f"{m}_{f.replace('_wtd','')}": r[f][m]
                for f in order for m in ("nse", "kge")})
        for r in rows])
    pd.set_option("display.width", 250)
    print(show.to_string(index=False, float_format=lambda v: f"{v:+.3f}",
                         na_rep="  --"))
    print()
    for f, lab in COLS:
        print(f"  {lab:<12} NSE median {summary[f'{f}_nse_median']}"
              f"  (n wells {summary[f'{f}_nse_n_wells']})"
              f"   KGE median {summary[f'{f}_kge_median']}"
              f"  (n wells {summary[f'{f}_kge_n_wells']})"
              f"   scored {summary[f'{f}_n_scored']}/45"
              f"   KGE refused on beta {summary[f'{f}_n_beta_refused']}")
    for k in keys:
        f = f"sim_sm_{k}"
        print(f"  sim_sm/{k:<6} NSE median {summary[f'{f}_nse_median']}"
              f"   KGE median {summary[f'{f}_kge_median']}"
              f"   scored {summary[f'{f}_n_scored']}/45   [JSON only]")
    print()
    for f, lab in COLS:
        u = unscored[f]
        print(f"  not scored, {lab}: {len(u)} wells {u}")


if __name__ == "__main__":
    main()
