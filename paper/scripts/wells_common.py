"""
The fourteen observation wells, as the paper uses them.

Shared by the map, the two hydrograph grids and the table so that a well's
number, its name and its score mean the same thing in all four.

Three conventions are inherited from ddpm_model/plot_wells_combined.py, which
is imported here rather than re-implemented:

  * ONE CONSTANT PER SITE, applied to the OBSERVATION only. ParFlow puts the
    water table systematically shallower than the wells log it; that is a
    model-vs-reality offset, not an emulator error, so it is removed before the
    shapes are compared and it is removed from the observation so the emulator
    is never moved to fit. The constant ALIGNS THE MINIMA, which makes a
    negative depth to water impossible by construction and treats every well
    in both networks identically -- see the note above ``offset`` below.
  * A 3-DAY MATCHING TOLERANCE. Every arm now spans the whole water year, so
    the only gaps left are the handful of days between one fold's rollout
    ending in late September and the next beginning on 2 October. Nearest-
    neighbour matching alone would reach across a gap of any width; the
    tolerance caps how far an observation may be carried to meet a forecast
    day. It is kept at 3 days, unchanged, so the scores stay comparable with
    the earlier 210-day run.
  * 30-DAY INCREMENTS THAT STRADDLE A GAP ARE EXCLUDED (span <= 45 days).

The one convention added here: the two networks are MERGED. The paper numbers
the wells 1..14 in a single sequence -- 1-8 USGS, 9-14 UIUC, the order the
series caches already use -- and never splits them.

THE 360-DAY GUARD. The seasonal arm is now "Dyear", the 360-day full water
year spliced from the 1 October and 1 May initialisations; it REPLACED the
210-day single-initialisation arm "D" everywhere in the paper. The superseded
210-day caches (seas5_D_fold*.npz) are still on disk beside the new ones, and a
figure built from the wrong pair is not visibly broken -- the seasonal line
simply stops in April, which is precisely what the previous version of these
figures did. Every fold this module loads is therefore checked BY SHAPE against
N_DAYS before any of it is plotted or scored.
"""
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paper_style as P                                        # noqa: E402

sys.path.insert(0, P.DDPM)
import plot_wells_combined as W                                # noqa: E402

WTD_CACHE = os.path.join(P.DATA, "well_series")
SM_CACHE = os.path.join(P.DATA, "well_series_sm")
N_DAYS = 360                 # every arm spans the whole water year
FORCINGS = (
    # key, cache prefix, colour, legend label
    ("nldas", "nldas", P.C_PRED, "Emulator, reanalysis forcing"),
    # "Dyear": 360 days from 1 October, the 1 May initialisation spliced on.
    # NOT seas5_D, which is the superseded 210-day arm.
    ("seas5", "seas5_Dyear", P.C_SEAS5, "Emulator, raw seasonal forcing"),
)
WATER_YEARS = P.WATER_YEARS          # (2003, 2004, 2005, 2006), chronological


# ------------------------------------------------------------------ wells ---
def pretty_name(name):
    """The USGS 'name' field carries the site code and the network tag for the
    Iowa well; the UIUC names carry the ICN prefix. Neither belongs on a map."""
    n = re.sub(r"\s*\(.*?\)", "", str(name))                 # (LK 13)
    n = re.sub(r"^\d{3}[NSEW]\d{2}[A-Z0-9]+\s*", "", n)      # 080N05W22CBCB2
    n = re.sub(r"^\d{4}", "", n)                             # the 1941 prefix
    n = re.sub(r"\s+[A-Z]{2}CRN-\d+$", "", n)                # IACRN-4
    n = n.replace("ICN ", "").strip()
    # Title-case the words but leave the ISWS site codes alone: HRY-91B and
    # MTOW-02 are identifiers, and "Hry-91B" is simply the wrong name for a
    # well the network calls HRY-91B.
    return " ".join(t if (t.isupper() and any(c.isdigit() for c in t))
                    else t.title() for t in n.split())


def load_wells():
    """plot_wells_combined's loader, numbered 1..14 in one sequence."""
    ws = W.load_wells()
    for i, w in enumerate(ws, 1):
        w["no"] = i
        w["pretty"] = pretty_name(w["name"])
    return ws


# ------------------------------------------------------------ despiking ---
# The raw well records carry three defects that are artefacts of the logging
# and not of the aquifer, and all three distort a hydrograph comparison:
#
#   1. EXACTLY REPEATED ROWS. 1156 of the 19023 UIUC readings are a verbatim
#      repeat of another row, same timestamp and same value.
#   2. TWO DIFFERENT READINGS ON ONE DAY. After the repeats are dropped, four
#      days across the fourteen wells still carry readings that disagree -- by
#      up to 3.07 m at Snicarte 1 on 2003-07-01, where the logger reports both
#      7.84 m and 10.91 m against a surrounding level of 10.9 m. A median of
#      the day's readings would invent a value (9.38 m) that was never
#      measured and that no neighbouring day supports.
#   3. ISOLATED SPIKES. A handful of short excursions depart from the local
#      level and return to it within a day or two.
#
# Defect 3 is the one that needs care, because the thing it most resembles is
# a real recharge event: both are sudden and large. They are separated by what
# happens AFTERWARDS. A recharge event is a step -- the water table rises and
# the series continues from the new level, then recesses. A logging artefact
# reverts -- the level before and the level after the excursion agree, and
# only the excursion itself is out of line. Only reverting excursions are
# removed here; steps are kept however large, which is why the sharp summer
# rises at Stelle and the 1.6 m December drop at Monmouth all survive.
#
# The size threshold is per well, not a single depth in metres, because the
# wells differ by two orders of magnitude in how much they move from day to
# day: the 99th percentile of the absolute daily change is 0.040 m at Snicarte
# 1 and 0.446 m at Bellville. A departure is large when it is large FOR THAT
# WELL. The floor keeps the threshold off the quantisation noise of the
# flattest records.
SPIKE_K = 2.0          # multiples of the well's 99th-percentile daily change
SPIKE_FLOOR = 0.25     # metres, never flag anything smaller than this
SPIKE_MAXRUN = 3       # days; a longer excursion is treated as real
SPIKE_CTX = 4          # readings of context taken on each side
SPIKE_MAXSPAN = 120    # days; context further away than this is not used
SPIKE_CTX_MIN = 2      # readings a side below which no judgement is made
SPIKE_SPARSE_K = 1.5   # x the well's IQR, for records with no consecutive days
SPIKE_REVERT = 0.5     # before/after must agree to within this x the departure
OBS_SPAN = ("2002-10-01", "2006-10-01")


def _daily(s):
    """One value per day, with the two duplicate defects resolved.

    Exact repeats are dropped first. If a day still carries readings that
    disagree, the one CLOSEST TO THE SURROUNDING DAYS is kept rather than an
    average of them, so the value plotted is always a value that was actually
    measured.
    """
    s = s[(s.index >= OBS_SPAN[0]) & (s.index < OBS_SPAN[1])]
    s = s[~pd.Series(list(zip(s.index, s.values)), dtype=object).duplicated().values]
    day = s.index.normalize()
    prov = s.groupby(day).median()                      # provisional context
    spread = s.groupby(day).max() - s.groupby(day).min()
    out = prov.copy()
    for d in spread.index[spread > 0]:
        cand = s[day == d].values
        nb = prov[(prov.index >= d - pd.Timedelta("3D")) &
                  (prov.index <= d + pd.Timedelta("3D")) & (prov.index != d)]
        if len(nb):
            out[d] = cand[np.argmin(np.abs(cand - np.median(nb.values)))]
    return out.reindex(pd.date_range(out.index.min(), out.index.max(), freq="D"))


def _spike_scale(d, valid):
    """The size a departure has to beat, in metres, for THIS well.

    Two sampling regimes have to be handled by one rule. The fifteen
    near-daily records give the natural yardstick directly: the 99th
    percentile of the absolute change from one calendar day to the next. The
    thirty monthly and quarterly records have no consecutive days at all --
    np.diff over a daily axis is NaN everywhere -- and an earlier version of
    this function returned no mask for them at all, so most of the ISWS
    network went unscreened while appearing to have been screened. For those,
    the yardstick is the change from one READING to the next, taken at the
    75th percentile rather than the 99th so that the excursion being tested
    cannot inflate the threshold meant to catch it.
    """
    dv = np.diff(d.values)
    dv = dv[np.isfinite(dv)]                       # consecutive calendar days
    if len(dv) >= 20:
        return max(SPIKE_FLOOR, SPIKE_K * np.percentile(np.abs(dv), 99))
    # For a monthly or quarterly record the change between readings is not a
    # yardstick for a bad reading: a month is long enough to contain a real
    # rise and fall, so that scale flags ordinary seasonal movement. The
    # yardstick used instead is the well's own interquartile range -- how far
    # it normally travels over the record -- which a single bad reading
    # cannot inflate the way it inflates a spread of differences. Only an
    # excursion larger than one and a half times that range is rejected, so
    # what goes is the reading that leaves the well's own operating envelope
    # and comes straight back.
    if len(valid) >= 8:
        iqr = float(valid.quantile(.75) - valid.quantile(.25))
        if iqr > 0:
            return max(SPIKE_FLOOR, SPIKE_SPARSE_K * iqr)
    return np.nan                                  # too short to judge


def _spike_mask(d):
    """True where a run of <= SPIKE_MAXRUN readings departs AND reverts.

    The context is taken as the nearest READINGS on each side, not the nearest
    calendar days. Several records are dense for most of a year and then drop
    to one reading a month, and a window measured in days finds no neighbours
    at all in the sparse stretches -- which is where a single bad reading is
    least likely to be noticed by eye.

    SPIKE_MAXSPAN is what keeps this honest on a sparsely sampled well. The
    reverting test infers "the level came back" from the readings either side,
    and that inference is only worth making when those readings are close
    enough in time to be about the same hydrological situation. The context is
    therefore narrowed until it fits inside SPIKE_MAXSPAN, and if even two
    readings a side will not fit, the point is left alone. A monthly well is
    screened where its neighbours are weeks away and not screened where they
    are seasons away.
    """
    valid = d.dropna()
    v = valid.values
    t = valid.index
    n = len(v)
    bad_c = np.zeros(n, bool)
    if n < 5:
        return np.zeros(len(d), bool), np.nan
    thr = _spike_scale(d, valid)
    if not np.isfinite(thr):
        return np.zeros(len(d), bool), np.nan
    for run in range(1, SPIKE_MAXRUN + 1):
        for i in range(n - run + 1):
            if bad_c[i:i + run].any():
                continue
            seg = v[i:i + run]
            for ctx in range(SPIKE_CTX, 1, -1):     # narrow until it fits
                lo, hi = max(0, i - ctx), min(n, i + run + ctx)
                if (hi - lo) - run < 4:             # need >= 2 a side
                    continue
                if (t[hi - 1] - t[lo]).days <= SPIKE_MAXSPAN:
                    break
            else:
                continue
            before, after = v[lo:i], v[i + run:hi]
            if len(before) < 2 or len(after) < 2:
                continue
            if (t[hi - 1] - t[lo]).days > SPIKE_MAXSPAN:
                continue
            mb, ma = np.median(before), np.median(after)
            lvl = (mb + ma) / 2.0
            dev = seg[np.argmax(np.abs(seg - lvl))] - lvl
            if np.abs(seg - lvl).min() < thr:                # every day far out
                continue
            if (np.sign(seg - lvl) != np.sign(dev)).any():   # ... same side
                continue
            if abs(mb - ma) < SPIKE_REVERT * abs(dev):       # ... and reverts
                bad_c[i:i + run] = True
    return d.index.isin(t[bad_c]), thr


def clean_observations(obs, report=False):
    """Daily, de-duplicated, de-spiked observations. See the notes above."""
    out, rows = {}, []
    for sid, s in obs.items():
        n_raw = len(s)
        d = _daily(s)
        # Single pass, deliberately. Re-running the detector on its own
        # output re-derives the threshold from data the detector has already
        # thinned, which shrinks it and makes the next pass reject more: a
        # cascade that ends by rejecting ordinary variation. The threshold is
        # computed once, from the record as measured.
        bad, thr = _spike_mask(d)
        d = d.mask(pd.Series(bad, index=d.index))
        out[sid] = d.dropna()
        rows.append(dict(sid=sid, n_raw=n_raw, n_daily=int(d.notna().sum() + bad.sum()),
                         n_spikes=int(bad.sum()), threshold_m=thr,
                         dates=[str(x.date()) for x in d.index[bad]]))
    return (out, pd.DataFrame(rows)) if report else out


def observations(ws):
    """The observations the paper uses: cleaned, never raw."""
    return clean_observations(W.observations(ws))


def raw_observations(ws):
    return W.observations(ws)


# ----------------------------------------------------------------- series ---
def check_full_year(cache, prefix):
    """Assert the 360-day contract, and return [(fold, n_days), ...].

    The check is on the npz's own shape, not on the dates the plot ends up
    drawing, because that is the only place a stale arm declares itself before
    it has been averaged, re-indexed and drawn. A cache that is absent is not
    an error -- a variable a forcing was never run for simply has no series --
    but a cache that is PRESENT and short is.
    """
    seen = []
    for fold in P.FOLDS:
        p = os.path.join(cache, f"{prefix}_fold{fold}.npz")
        if not os.path.exists(p):
            continue
        with np.load(p) as z:
            n = int(z["series"].shape[1])
        seen.append((fold, n))
        if n != N_DAYS:
            raise ValueError(
                f"{p}: {n} days, expected {N_DAYS}. The paper's rollouts span "
                f"the whole water year; a {n}-day series is the superseded "
                f"210-day arm (seas5_D) and must not be plotted or scored.")
    return seen


def model_series(prefix, ws):
    """Water table depth. Missing cache -> every well None, no exception."""
    if not check_full_year(WTD_CACHE, prefix):
        return {w["short_id"]: None for w in ws}
    return W.model_series(prefix, ws)


def model_series_sm(prefix, ws):
    """Soil moisture, same layout, from data/well_series_sm/."""
    import json
    check_full_year(SM_CACHE, prefix)
    di = json.load(open(os.path.join(P.DATA, "tiles", "day_index.json")))
    out = {w["short_id"]: [] for w in ws}
    found = False
    for fold in P.FOLDS:
        p = os.path.join(SM_CACHE, f"{prefix}_fold{fold}.npz")
        if not os.path.exists(p):
            continue
        found = True
        z = np.load(p)
        s = z["series"]
        dates = pd.DatetimeIndex([di[int(g)]["date"]
                                  for g in z["global_days"].astype(int)])
        for j, k in enumerate([str(x) for x in z["sid"]]):
            if k in out:
                out[k].append(pd.DataFrame(
                    dict(mean=s[:, :, j].mean(0),
                         lo=np.percentile(s[:, :, j], 10, axis=0),
                         hi=np.percentile(s[:, :, j], 90, axis=0)),
                    index=dates))
    if not found:
        return {w["short_id"]: None for w in ws}

    def _assemble(v):
        if not v:
            return None
        df = pd.concat(v).sort_index()
        # a complete daily axis turns the few days between one fold's last
        # forecast day and the next fold's first into NaN, which breaks the
        # line instead of drawing a straight segment across the fold boundary
        return df.reindex(pd.date_range(df.index.min(), df.index.max(),
                                        freq="D"))
    return {k: _assemble(v) for k, v in out.items()}


def parflow_truth(var="soil_moisture"):
    """{short_id: Series} of the ParFlow target at the same fourteen cells."""
    p = os.path.join(SM_CACHE, "parflow_truth.npz")
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p}: run paper/scripts/extract_parflow_wells.py first")
    z = np.load(p)
    idx = pd.DatetimeIndex(z["dates"].astype(str))
    return {str(s): pd.Series(z[var][:, j].astype(float), index=idx)
            for j, s in enumerate(z["sid"])}


def coverage(mods, key, sid=None):
    """Per forcing, per water year: the days the drawn series actually covers.

    The 360-day guard in check_full_year catches a stale cache; this catches
    everything downstream of it -- a wrong splice, a fold that failed, a
    re-index that dropped the tail. It reports what the FIGURE will show, so
    "the seasonal line reaches September" is a printed fact and not an
    impression of a PNG.
    """
    out = []
    for m in mods:
        for wy in WATER_YEARS:
            d = None
            for k, v in m[key].items():
                if sid is not None and k != sid:
                    continue
                if v is None:
                    continue
                d = clip(v, wy).dropna(subset=["mean"])
                break
            if d is None or not len(d):
                out.append((m["key"], wy, None, None, 0))
                continue
            out.append((m["key"], wy, d.index.min().date(),
                        d.index.max().date(), len(d)))
    return out


def print_coverage(mods, key):
    print("  days covered per water year (the figure's x extent):")
    for k, wy, lo, hi, n in coverage(mods, key):
        if lo is None:
            print(f"    {k:<7} WY{wy}  no data")
            continue
        print(f"    {k:<7} WY{wy}  {lo} .. {hi}  {n} days"
              + ("" if hi.month == 9 else "   <-- does NOT reach September"))


def available(ws):
    """The forcings whose cache exists, in the paper's fixed order, with both
    variables. A forcing absent from disk is simply skipped."""
    out = []
    for key, prefix, color, label in FORCINGS:
        wtd = model_series(prefix, ws)
        sm = model_series_sm(prefix, ws)
        if all(v is None for v in wtd.values()) and \
           all(v is None for v in sm.values()):
            continue
        out.append(dict(key=key, prefix=prefix, color=color, label=label,
                        wtd=wtd, sm=sm))
    return out


# -------------------------------------------------- the datum alignment ---
# ParFlow places the water table systematically shallower than the wells
# record it. That is a model-vs-reality discrepancy the emulator did not
# create and must not be charged with, so each observed series is moved onto
# the simulator's datum by ONE CONSTANT PER SITE and the comparison that
# remains is one of SHAPE. Only the observation moves; ParFlow and the
# emulator are always drawn on their own true scale.
#
# THE CONSTANT ALIGNS THE MINIMA, not the means.
#
#     off = min(simulated) - min(observed)
#
# Depth to water is positive downward, so a negative value would put the
# water table above the land surface. Aligning minima makes that impossible
# by construction: the projection of the rollout already guarantees the
# simulated depth is non-negative, so the shifted observation's minimum
# lands exactly on the simulated minimum and nothing can fall below it.
# No cap, no clipping, no special case, and every well is treated the same
# way -- which is the reason for preferring it.
#
# The alternative, matching means, is a better level estimator but cannot
# satisfy the constraint on its own: it drives 1810 values below zero across
# 26 of the 45 wells. Repairing that requires off = max(mean-match,
# -min(obs)), which binds at some wells and not others, so the network ends
# up with two different estimators and the reader cannot tell which well got
# which. Aligning minima binds everywhere, identically.
#
# WHAT IT COSTS, measured rather than asserted. Correlation is invariant to
# a constant, so every correlation in the paper is unchanged. NSE and KGE
# are not: the median NSE against the observations moves from -0.78 to -1.09
# and the median KGE from +0.244 to +0.220, because a minimum is a noisier
# anchor than a mean.
#
# AND THE CAVEAT. The minimum of a short record is systematically less
# extreme than the minimum of a long one, simply because it was sampled
# less, and these records run from 1 to 1716 readings. The constant
# therefore carries a bias that tracks record length. It is second order
# against the 0.8-3.4 m datum differences being removed, but it is real and
# the Methods says so.


def offset(ob, mod):
    """ONE constant per site: min(simulated) - min(observed). See the note
    above for why the minima and not the means. Applied to the observation
    only.

    The observed minimum is taken over the WHOLE record being plotted, not
    only over the days the simulation covers. Those are not the same set: the
    simulated year runs 2 October to late September, while the observations
    span the water year exactly, so a handful of early-October and
    late-September readings fall outside it. Using the narrower window would
    leave the constant too small to lift those few readings, and a value
    outside the scored window would plot below zero -- which the guard in
    shifted_observations duly caught the first time this was written that way.
    """
    if ob is None or mod is None or not len(ob):
        return 0.0
    v = np.asarray(ob.values, float)
    v = v[np.isfinite(v)]
    m = np.asarray(mod["mean"].values, float)
    m = m[np.isfinite(m)]
    if not v.size or not m.size:
        return 0.0
    return float(m.min() - v.min())


def shift_obs(ob, off):
    """The observation series the paper plots and scores. One function, so
    the figures, the roster table and the metric tables cannot drift apart."""
    if ob is None or not len(ob):
        return ob
    return pd.Series(np.asarray(ob.values, float) + off, index=ob.index)


def shifted_observations(ws, obs, ref_mod, report=False):
    """(offsets, shifted observations) for the whole roster.

    ``ref_mod`` is the forcing the constant is measured against -- the
    reanalysis arm, the only one covering the whole record -- so the seasonal
    arm is never scored against an observation tuned to it.

    The non-negativity of the result is ASSERTED, not hoped for: if a shifted
    series ever dipped below zero the alignment above would be wrong, and a
    silent negative depth is exactly the failure this rule exists to prevent.
    """
    offs, out, rows = {}, {}, []
    for w in ws:
        sid = w["short_id"]
        o = obs.get(sid)
        off = offset(o, ref_mod.get(sid))
        offs[sid] = off
        sh = shift_obs(o, off)
        out[sid] = sh
        if sh is not None and len(sh):
            v = np.asarray(sh.values, float)
            v = v[np.isfinite(v)]
            if v.size and v.min() < -1e-9:
                raise ValueError(
                    f"{sid}: shifted depth {v.min():.4f} m is below zero; "
                    "min-alignment should make that impossible")
        if report:
            raw = (np.asarray(o.values, float) if o is not None and len(o)
                   else np.zeros(0))
            raw = raw[np.isfinite(raw)]
            shv = np.asarray(sh.values, float) if sh is not None else np.zeros(0)
            shv = shv[np.isfinite(shv)]
            rows.append(dict(
                no=w["no"], name=w["pretty"], net=w["net"], n=int(raw.size),
                raw_min_m=float(raw.min()) if raw.size else np.nan,
                offset_m=off,
                shifted_min_m=float(shv.min()) if shv.size else np.nan,
                n_negative=int((shv < -1e-9).sum()),
                n_clipped=0))
    if report:
        return offs, out, pd.DataFrame(rows)
    return offs, out


def match(ref, mod, tol="3D"):
    """Pair a reference series with a model series without crossing a gap."""
    if mod is None or ref is None or not len(ref):
        return None, None
    m = mod["mean"].reindex(ref.index, method="nearest",
                            tolerance=pd.Timedelta(tol))
    keep = m.notna().values & np.isfinite(np.asarray(ref.values, float))
    if keep.sum() == 0:
        return None, None
    return (np.asarray(m[keep].values, float),
            np.asarray(ref[keep].values, float))


# A flat record carries no phase information, so a correlation against it is
# meaningless and the panel is marked "no score" instead. The test is against a
# tolerance rather than exact zero: a constant series that has been through a
# groupby-median comes back with a standard deviation of order 1e-17 rather
# than 0, which an equality test reads as varying and then reports as a
# correlation of 0.00. FLAT_STD is far below the millimetre resolution of the
# loggers, so it can only ever catch arithmetic noise.
FLAT_STD = 1e-6


def score(ref, mod, tol="3D", nmin=30, idx=None):
    """r, 30-day-increment r and n, under the paper's matching rules."""
    a, b = match(ref, mod, tol)
    if (a is None or len(a) < nmin
            or np.std(a) < FLAT_STD or np.std(b) < FLAT_STD):
        return dict(r=np.nan, inc=np.nan, n=0 if a is None else len(a))
    r = float(np.corrcoef(a, b)[0, 1])
    inc = np.nan
    if idx is not None and len(a) > 40:
        k = 30
        span = (idx[k:] - idx[:-k]).days.values
        near = span <= 45          # an increment must not straddle a gap
        da, db = (a[k:] - a[:-k])[near], (b[k:] - b[:-k])[near]
        if near.sum() > 10 and da.std() > 0 and db.std() > 0:
            inc = float(np.corrcoef(da, db)[0, 1])
    return dict(r=r, inc=inc, n=int(len(a)))


def match_mask(ref, mod, tol="3D"):
    """(boolean mask over ref's rows, model values aligned to ref).

    The mask is what makes a LIKE-FOR-LIKE comparison possible: two forcings
    can be scored on the intersection of their masks, i.e. on exactly the same
    observation days, instead of each on whatever days it happens to cover.
    """
    if mod is None or ref is None or not len(ref):
        return None, None
    m = mod["mean"].reindex(ref.index, method="nearest",
                            tolerance=pd.Timedelta(tol))
    keep = m.notna().values & np.isfinite(np.asarray(ref.values, float))
    return keep, np.asarray(m.values, float)


def r_on(vals, ref_vals, mask, nmin=30):
    """Correlation on a given mask, or NaN if the sample cannot support one."""
    if mask is None or vals is None or mask.sum() < nmin:
        return np.nan, 0 if mask is None else int(mask.sum())
    a, b = vals[mask], np.asarray(ref_vals, float)[mask]
    if np.std(a) < FLAT_STD or np.std(b) < FLAT_STD:
        return np.nan, int(mask.sum())
    return float(np.corrcoef(a, b)[0, 1]), int(mask.sum())


# ------------------------------------------------------------ NSE and KGE ---
# NSE = 1 - SSE / SS_obs, the Nash-Sutcliffe efficiency, with the OBSERVATION
# (or, against the simulator, the ParFlow series) as the reference whose
# variance sets the denominator.
#
# KGE is KLING-GUPTA 2009, the original formulation:
#
#     KGE = 1 - sqrt((r - 1)^2 + (alpha - 1)^2 + (beta - 1)^2)
#     alpha = sd(sim) / sd(obs)        beta = mean(sim) / mean(obs)
#
# NOT the 2012 revision, whose gamma = CV(sim)/CV(obs) divides by the mean
# twice and is therefore even less stable here.
#
# BETA IS A RATIO OF MEANS, and these are shifted depths to water that can sit
# arbitrarily close to zero: at Stelle the shifted observation has a mean of
# about 0.9 m but at a well whose record is a few centimetres of water above a
# shallow table the mean can be small enough that beta explodes. A beta
# computed from a near-zero denominator is not a bias ratio, it is a division
# artefact, and a single such well would dominate any median taken over the
# network. So beta is required to rest on a denominator that is large compared
# with the variability of the reference itself: |mean(obs)| must exceed
# BETA_MIN_SD times sd(obs). Where it does not, KGE is reported as "not
# scored" (NaN) and the reason is recorded, rather than a number being printed
# that the data cannot support. r and alpha, which are both invariant to the
# location of the series, are unaffected and NSE is still reported.
BETA_MIN_SD = 0.25


def nse(sim, ref):
    """1 - sum((sim-ref)^2) / sum((ref-mean(ref))^2). NaN on a flat ref."""
    sim = np.asarray(sim, float)
    ref = np.asarray(ref, float)
    den = float(((ref - ref.mean()) ** 2).sum())
    if den <= 0 or not np.isfinite(den):
        return np.nan
    return float(1.0 - ((sim - ref) ** 2).sum() / den)


def kge(sim, ref):
    """Kling-Gupta 2009, and its three components, as a dict.

    ``beta`` and therefore ``kge`` are NaN when the reference mean is too
    close to zero to divide by (see BETA_MIN_SD); ``r`` and ``alpha`` are
    still returned, and ``beta_ok`` says which happened.
    """
    sim = np.asarray(sim, float)
    ref = np.asarray(ref, float)
    sd_r, sd_s = float(ref.std()), float(sim.std())
    out = dict(r=np.nan, alpha=np.nan, beta=np.nan, kge=np.nan, beta_ok=False)
    if sd_r < FLAT_STD or sd_s < FLAT_STD:
        return out
    out["r"] = float(np.corrcoef(sim, ref)[0, 1])
    out["alpha"] = sd_s / sd_r
    mu_r = float(ref.mean())
    if abs(mu_r) < BETA_MIN_SD * sd_r:
        return out                      # beta would be a division artefact
    out["beta_ok"] = True
    out["beta"] = float(sim.mean()) / mu_r
    out["kge"] = float(1.0 - np.sqrt((out["r"] - 1) ** 2
                                     + (out["alpha"] - 1) ** 2
                                     + (out["beta"] - 1) ** 2))
    return out


def efficiency(ref, mod, tol="3D", nmin=30):
    """NSE and KGE of a model series against a reference, matched under the
    paper's rules. ``n`` below ``nmin`` -> every metric NaN, n reported."""
    a, b = match(ref, mod, tol)          # a = model, b = reference
    out = dict(n=0 if a is None else int(len(a)), nse=np.nan, kge=np.nan,
               r=np.nan, alpha=np.nan, beta=np.nan, beta_ok=False,
               scored=False)
    if a is None or len(a) < nmin:
        return out
    out["scored"] = True
    out["nse"] = nse(a, b)
    out.update({k: v for k, v in kge(a, b).items()})
    out["n"] = int(len(a))
    return out


def matched_index(ref, mod, tol="3D"):
    m = mod["mean"].reindex(ref.index, method="nearest",
                            tolerance=pd.Timedelta(tol))
    keep = m.notna().values & np.isfinite(np.asarray(ref.values, float))
    return ref.index[keep]


def wy_window(wy):
    """The water year, 1 Oct to 30 Sep -- the x axis of every panel, whether
    or not the forcing in it reaches the end."""
    return pd.Timestamp(f"{wy-1}-10-01"), pd.Timestamp(f"{wy}-09-30")


def clip(s, wy):
    if s is None:
        return None
    lo, hi = wy_window(wy)
    return s[(s.index >= lo) & (s.index <= hi)]
