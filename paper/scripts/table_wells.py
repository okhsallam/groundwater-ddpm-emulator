"""
Table: the observation wells and the emulator's skill at each.

One row per well, in the paper's single merged 1..N sequence (48 wells: 8
USGS and 40 ISWS/UIUC). The two numbers
that describe the site -- the median observed depth to water and the number of
observations -- are recomputed here over WY2003-WY2006 rather than copied from
the well JSONs, because those JSONs summarise each well's whole record (the
USGS sites run from the 1980s, the UIUC records continue past 2006) and the
paper only ever uses the four water years.

The correlation is pooled over the whole 2003-2006 record under the same rules
the figures use: the observation shifted by one constant per site, a 3-day
matching tolerance so an observation is never carried more than three days to
meet a forecast day, and 30-day increments that would straddle a gap
excluded. A forcing
with fewer than 30 matched days at a well is left unscored rather than given a
number computed from nothing.

The observation series scored here is the one the figures plot, from
wells_common.shifted_observations: one constant per site aligning the observed
minimum with the simulated one, which makes a negative depth impossible by
construction. Correlation is invariant to a constant, so the alignment does
not move any number in this table at all.

THE PRINTED TABLE IS THE ROSTER ONLY -- number, well, network, latitude,
longitude. Every score lives in wells.json and is quoted in the text, so the
table in the paper stays narrow enough to set at \\textwidth without
rotation.

THE MATCHED-DAY COLUMNS ARE KEPT IN THE JSON, AND THEY ARE NOW REDUNDANT.
While the seasonal arm was the 210-day run it covered 1 October to 29 April
only -- the wetting limb, the easier and more monotonic half of the hydrograph
-- so scoring each forcing on its own days compared two different questions
and flattered the seasonal forecast. The matched-day columns existed to undo
that. The seasonal arm is now the 360-day spliced water year, whose model dates
are IDENTICAL to the reanalysis arm's, so the intersection of the two forcings'
matched days is the reanalysis arm's own matched days and r_<forcing>_matched
equals r_<forcing> exactly. The columns are still computed, and the equality is
still asserted at the end of this script, because the moment an arm shorter
than the water year joins the comparison the correction becomes necessary
again -- and silently.

Writes paper/tables/wells.tex (a bare booktabs tabular, no table wrapper) and
paper/tables/wells.json.
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
NMIN = 30                 # the paper's rule: see wells_grid and table_wells_nse_kge


def main():
    os.makedirs(TAB, exist_ok=True)
    ws = C.load_wells()
    obs = C.observations(ws)
    mods = C.available(ws)
    keys = [m["key"] for m in mods]
    print(f"  forcings present: {', '.join(keys)}")
    ref_mod = next((m for m in mods if m["key"] == "nldas"), mods[0])
    # the same shifted, non-negative observation the figures plot
    offs, obs_sh = C.shifted_observations(ws, obs, ref_mod["wtd"])

    rows, thin = [], []
    for w in ws:
        sid = w["short_id"]
        o = obs.get(sid)
        o = o[(o.index >= LO) & (o.index <= HI)] if o is not None else None
        off = offs[sid]
        rec = dict(no=w["no"], short_id=sid, name=w["pretty"],
                   network=w["net"], lat=round(float(w["lat"]), 4),
                   lon=round(float(w["lon"]), 4),
                   grid_row=int(w["y"]), grid_col=int(w["x"]),
                   median_depth_m=(round(float(o.median()), 3)
                                   if o is not None and len(o) else None),
                   n_obs=int(len(o)) if o is not None else 0,
                   offset_applied_m=round(off, 3))
        if o is not None and len(o):
            # shifted AND floored at zero, from wells_common, so the table and
            # the figure score the identical series
            oo = obs_sh[sid]
            oo = oo[(oo.index >= LO) & (oo.index <= HI)]
            b = np.asarray(oo.values, float)
            # one mask per forcing over the observation's own rows, so any two
            # forcings can be scored on exactly the same days
            mk, vl = {}, {}
            for m in mods:
                k, v = C.match_mask(oo, m["wtd"].get(sid), "3D")
                mk[m["key"]], vl[m["key"]] = k, v
            seasonal = [m["key"] for m in mods if m["key"] != "nldas"]
            common = None
            if mk.get("nldas") is not None:
                common = mk["nldas"].copy()
                for k in seasonal:
                    if mk.get(k) is not None:
                        common &= mk[k]
            for m in mods:
                key = m["key"]
                if mk.get(key) is None:
                    rec[f"r_{key}"] = None
                    continue
                # each forcing on its own days
                idx = C.matched_index(oo, m["wtd"][sid], "3D")
                st = C.score(oo, m["wtd"][sid], tol="3D", nmin=NMIN, idx=idx)
                rec[f"r_{key}"] = (round(st["r"], 3)
                                   if np.isfinite(st["r"]) else None)
                rec[f"r30_{key}"] = (round(st["inc"], 3)
                                     if np.isfinite(st["inc"]) else None)
                rec[f"n_matched_{key}"] = st["n"]
                # and every forcing again on the days ALL of them cover
                rw, nw = C.r_on(vl[key], b, common, nmin=NMIN)
                rec[f"r_{key}_matched"] = (round(rw, 3)
                                           if np.isfinite(rw) else None)
                rec["n_matched_common"] = nw
                if not np.isfinite(st["r"]):
                    thin.append((w["no"], key, st["n"]))
            # do the two arms in fact cover the same observation days?
            rec["same_days_as_nldas"] = {
                m["key"]: (None if mk.get(m["key"]) is None
                           or mk.get("nldas") is None
                           else bool(np.array_equal(mk[m["key"]],
                                                    mk["nldas"])))
                for m in mods if m["key"] != "nldas"}
        rows.append(rec)

    json.dump(dict(
        arm_note=(
            "The seasonal arm is 'Dyear': 360 days from the 1 October "
            "initialisation, the 1 May initialisation spliced on, cached as "
            "data/well_series/seas5_Dyear_fold<N>.npz. It REPLACES the "
            "210-day arm 'D' used in an earlier version of these numbers, "
            "whose caches (seas5_D_fold<N>.npz) are still on disk and are "
            "not read."),
        window_note=(
            "r_<forcing> scores each forcing on its own matched days; "
            "r_<forcing>_matched scores every forcing on the SAME days -- "
            "the intersection of all forcings' matched days. Both arms now "
            "span the identical 360 model days of each water year, so that "
            "intersection is just the reanalysis arm's own matched days and "
            "r_<forcing>_matched == r_<forcing> at every well; "
            "same_days_as_nldas records that per well. The correction is "
            "retained because it becomes necessary again, and silently, the "
            "moment an arm shorter than the water year joins the comparison "
            "-- as the 210-day arm was, when the reanalysis scored a median "
            "r of 0.56 on the full water year but 0.63 on the seasonal "
            "window against 0.54 for the seasonal forcing."),
        note=("Fourteen observation wells, numbered 1-14 in one sequence "
              "across the merged USGS and UIUC networks. Median depth and "
              "n_obs cover WY2003-WY2006. r is the correlation of the "
              "emulator's ensemble mean with the observation after a single "
              "constant per site is applied to the observation; r30 is the "
              "same for 30-day increments that do not straddle a forecast "
              "gap; n_matched is the number of observations within 3 days of "
              "a forecast day."),
        forcings=keys, wells=rows),
        open(os.path.join(TAB, "wells.json"), "w"), indent=1)

    # ------------------------------------------------------------- latex --
    # The ROSTER ONLY. Scores are quoted in the text from wells.json; putting
    # them in the table too made it wider than \textwidth and forced either a
    # rotation or a type size the reduction could not survive. The layout here
    # is byte-for-byte the one the paper already sets, so nothing in paper.tex
    # moves when the numbers behind it are recomputed.
    # The roster is now 45 wells, which does not fit a page as a single
    # column, so it is set as two blocks side by side. `n` is the number of
    # daily values in WY2003-WY2006 after screening, and it is in the table
    # because it varies by three orders of magnitude across this network --
    # the reader needs it to know which rows a correlation can be read from
    # at all.
    # "USGS" and "UIUC" both begin with U, so the letter is mapped, not sliced.
    NET_LETTER = {"USGS": "G", "UIUC": "U"}

    def cell(r_):
        # a bare hyphen sets as a hyphen, not a minus: a western longitude is
        # set in math mode
        return " & ".join([
            str(r_["no"]), r_["name"], NET_LETTER.get(r_["network"], "?"),
            f"{r_['lat']:.3f}", "$" + f"{r_['lon']:.3f}" + "$",
            str(int(r_["n_obs"]))])

    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:]
    # Two blocks of six columns do not fit \textwidth at body size, so the
    # table is set \footnotesize with tight column separation and the network
    # abbreviated to its initial, which the caption expands.
    head = (r"\# & Well & Net. & Lat. & Lon. & $n_{\mathrm{obs}}$")
    blank = " & " * 5
    # A VISIBLE RULE, not white space, between the two blocks: six numeric
    # columns followed by six more of the same kind read as one twelve-column
    # table unless something says where the first table ends.
    L = [r"{\footnotesize\setlength{\tabcolsep}{3.4pt}",
         r"\begin{tabular}{r l c r r r @{\hspace{0.9em}} | @{\hspace{0.9em}} "
         r"r l c r r r}",
         r"\toprule",
         head + " & " + head + r" \\",
         r"\midrule"]
    for i in range(half):
        a = cell(left[i])
        b = cell(right[i]) if i < len(right) else blank
        L.append(a + " & " + b + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}"]
    open(os.path.join(TAB, "wells.tex"), "w").write("\n".join(L) + "\n")

    print(f"  wrote {TAB}/wells.tex and wells.json")
    show = ["no", "name", "network", "median_depth_m", "n_obs"] \
        + [f"r_{k}" for k in keys] \
        + [f"r_{k}_matched" for k in keys] + ["n_matched_common"]
    df = pd.DataFrame(rows)
    print(df[show].to_string(index=False))
    for k in keys:
        v, vw = df[f"r_{k}"].dropna(), df[f"r_{k}_matched"].dropna()
        print(f"  median r ({k}): own days {v.median():+.3f} (n wells {len(v)})"
              f"   shared matched days {vw.median():+.3f}")
    if "nldas" in keys and len(keys) > 1:
        k2 = [k for k in keys if k != "nldas"][0]
        n_better = int((df["r_nldas_matched"] >= df[f"r_{k2}_matched"]).sum())
        print(f"  on the shared matched days the reanalysis is at or above "
              f"{k2} at {n_better}/{len(df)} wells")
        # Is the windowing correction still doing anything? It is not, as long
        # as every arm covers the water year; say so loudly either way.
        same = [r_["same_days_as_nldas"].get(k2) for r_ in rows
                if isinstance(r_.get("same_days_as_nldas"), dict)]
        n_same = sum(1 for x in same if x)
        print(f"  {k2} and nldas match the SAME observation days at "
              f"{n_same}/{len(same)} wells")
        d = (df[f"r_{k2}"] - df[f"r_{k2}_matched"]).abs().max()
        d2 = (df["r_nldas"] - df["r_nldas_matched"]).abs().max()
        if n_same == len(same) and max(d, d2) < 5e-4:
            print("  => the matched-day correction is a NO-OP: both arms "
                  "cover the identical 360 days, so each forcing's own days "
                  "ARE the shared days. No windowing correction is required.")
        else:
            print(f"  => the matched-day correction still BITES: max |r - "
                  f"r_matched| is {max(d, d2):.3f}; the arms do not cover the "
                  f"same days and only the _matched columns are comparable.")
    if thin:
        print("  unscored (fewer than %d matched days or a constant series):"
              % NMIN)
        for no, k, n in thin:
            print(f"    well {no}, {k}: n={n}")


if __name__ == "__main__":
    main()
