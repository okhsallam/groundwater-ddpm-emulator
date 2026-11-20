"""
Build the forcing tables that drive the SEAS5 rollout arms.

Everything here lands on the 1 deg SEAS5 frame or as a 1 km monthly correction
field; the 1 km daily forcing is composed on the fly inside the rollout by
seas5_forcing.compose. That is deliberate: materialising arm C at 1 km would be
25 members x 210 days x 3 vars x 637 x 665 = 27 GB, against ~100 MB for the
tables, and the composition is a bilinear interpolation of 45 small fields per
hop -- negligible next to a diffusion sampling call.

THE ARMS (see seas5_forcing.py for why they must share one interpolation path):

  A   NLDAS-2 at 1 km                         emulator error alone; already run
  B   NLDAS-2 coarsened to 1 deg, back to 1 km      + resolution loss alone
  B2  ERA5 coarsened to 1 deg, back to 1 km,
      with the stage-2 delta                        + the whole correction chain,
                                                      driven by a perfect forecast
  C   SEAS5, bias-corrected                         + genuine forecast error
  D   SEAS5, raw                                    what the correction buys
  E   NLDAS-2 day-of-year climatology at 1 km       the no-forecast-information
                                                      reference

Arm B carries no delta, because a coarsened NLDAS-2 field is already in NLDAS-2
space; arms B2, C and D do. Arm B2 is the one that makes arm C interpretable:
without it, a weak arm C cannot be distinguished from a lossy pipeline.

THE TWO-STAGE CORRECTION

Stage 1, SEAS5 -> ERA5, on the 1 deg frame, per cell and per lead window, by
empirical quantile mapping over the C3S reference period with the target years
withheld. This is where forecast bias and its growth with lead time are removed.
Five lead windows of 43 days: enough to resolve the growth, while leaving ~950
ERA5 days and ~24,000 SEAS5 member-days per cell per window to estimate 101
quantile knots from.

  Temperature is mapped additively outside the fitted range, not clipped to it,
  so a forecast colder than any day in the sample stays colder rather than being
  flattened onto the sample minimum. Precipitation gets a wet-day frequency
  correction first -- the SEAS5 threshold matching ERA5's dry-day fraction is
  found and everything below it set to zero -- because quantile-mapping a field
  with a large atom at zero otherwise smears drizzle across the whole lower tail,
  and the emulator reads precipitation as the event that drives recharge. Its
  heavy tail is extrapolated multiplicatively, again so extremes are not capped.

Stage 2, ERA5 -> NLDAS-2 at 1 km, as a per-cell per-calendar-month delta,
additive in K for the temperatures and multiplicative for precipitation,
estimated over fold 1's TRAINING years only (WY2003-2005). This restores
NLDAS-2's 1 km climatological spatial detail and the offset between the two
reanalyses. It does NOT restore daily spatial detail -- see
seas5_forcing.compose for the honest statement of that limit.

LEAK DISCIPLINE, since the whole result turns on it:
  * stage 1 is fitted on 1993-2016 excluding 2005 (the init year) and 2006 (the
    verification year);
  * stage 2 is fitted on WY2003-2005, fold 1's training years, so WY2006 never
    enters;
  * arm E's climatology likewise comes from WY2003-2005 only;
  * arm B2 uses verification-period ERA5 as forcing INPUT, never as a fitting
    sample.
Nothing fitted here has seen the period being forecast.

TIMESTAMP AND UNIT CONVENTIONS, all verified against the delivered files:
  * SEAS5 leadtime_hour H = 24k describes calendar day init_date + (k-1), since
    the k-th 24-hour window runs from (k-1)*24 h to k*24 h after a 00 UTC init.
  * total_precipitation is accumulated from the forecast start in metres:
    daily_mm[k] = (tp[k] - tp[k-1]) * 1000, with daily_mm[1] = tp[1] * 1000.
  * mn2t24/mx2t24 are K already, and are the min/max over that same window.
  * ERA5 daily_sum of total_precipitation is in metres; x1000 for mm.
  * NLDAS-2 here is K and mm/day (data/parflow_raw/forcing_wy*.nc).

Usage:
    python prepare_seas5_domain.py              # every stage whose inputs exist
    python prepare_seas5_domain.py --stages stage2 armB armE
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from seas5_forcing import (COARSE_LAT, COARSE_LON, COARSE_RES, FORCING_VARS,
                           NX, NY, WATER_YEARS, CoarseMap, assert_matches_seas5)

HERE = Path(__file__).resolve().parent
PARFLOW = HERE / "data" / "parflow_raw"
SEAS5_RAW = HERE / "data" / "seas5" / "raw"
ERA5_RAW = HERE / "data" / "era5" / "raw"
OUT = HERE / "data" / "seas5"

TARGET_INIT = (2005, 10)           # 2005-10-01 -> the first 210 days of WY2006
TARGET_WY = 2006                   # fold 1's test year

# One 1 October initialisation per water year, which is also one per fold -- see
# download_seas5.py for why that alignment matters (each forecast is scored by a
# model that never saw the year it forecasts).
TARGET_INITS = [(2002, 10), (2003, 10), (2004, 10), (2005, 10)]
INIT_TO_FOLD = {2002: (2003, 4), 2003: (2004, 3),
                2004: (2005, 2), 2005: (2006, 1)}
TRAIN_WYS = [2003, 2004, 2005]     # fold 1's training years -- stage 2 and arm E
QM_YEARS = [y for y in range(1993, 2017) if y not in (2005, 2006)]

N_LEAD = 215
LEAD_EDGES = [1, 44, 87, 130, 173, N_LEAD + 1]     # 5 windows of 43 days
N_QUANTILE = 101
WET_MM = 0.1                       # ERA5 dry-day threshold, mm/day

NC_ENC = dict(zlib=True, complevel=4)


# ---------------------------------------------------------------- readers

def _pick(ds, *names):
    for n in names:
        if n in ds.variables or n in ds.coords:
            return n
    raise KeyError(f"none of {names} in {list(ds.variables)}")


def coarsen_latlon(arr, lat, lon):
    """(..., nlat, nlon) on 1-D lat/lon -> (..., 10, 13) on the coarse frame.

    Plain mean of the source cells whose centres fall in each coarse cell. A
    source point sitting exactly on a coarse cell boundary is assigned by
    numpy's round-half-to-even, which is arbitrary but deterministic; for the
    climatological means this is used for, a 4x4 versus 5x4 block at an edge is
    immaterial.
    """
    ny_c, nx_c = COARSE_LAT.size, COARSE_LON.size
    iy = np.rint((np.asarray(lat, float) - COARSE_LAT[0]) / COARSE_RES).astype(int)
    ix = np.rint((np.asarray(lon, float) - COARSE_LON[0]) / COARSE_RES).astype(int)
    oky, okx = (iy >= 0) & (iy < ny_c), (ix >= 0) & (ix < nx_c)
    a = np.asarray(arr, float)[..., oky, :][..., :, okx]
    flat = (iy[oky][:, None] * nx_c + ix[okx][None, :]).ravel()
    counts = np.bincount(flat, minlength=ny_c * nx_c)
    lead = a.shape[:-2]
    a2 = a.reshape(-1, flat.size)
    out = np.empty((a2.shape[0], ny_c * nx_c))
    for i in range(a2.shape[0]):
        s = np.bincount(flat, weights=a2[i], minlength=ny_c * nx_c)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = s / counts
    return out.reshape(*lead, ny_c, nx_c)


def load_seas5(year, month):
    """-> (member, lead 1..215, 10, 13) per variable, K and mm/day."""
    path = SEAS5_RAW / f"seas5_{year}{month:02d}.nc"
    if not path.exists():
        return None
    # decode_timedelta=False is REQUIRED, not a preference. The CDS writes a
    # `dtype` attribute on `forecast_period`, which collides with xarray's
    # timedelta decoder and makes a plain open_dataset raise
    # "failed to prevent overwriting existing key dtype in attrs". Verified
    # against the delivered 2002-10 file on 2026-10-21.
    ds = xr.open_dataset(path, decode_timedelta=False)
    la, lo = _pick(ds, "latitude", "lat"), _pick(ds, "longitude", "lon")
    lat, lon = ds[la].values, ds[lo].values
    assert_matches_seas5(lat, lon)
    mdim = _pick(ds, "number", "realization", "member")
    ldim = _pick(ds, "forecastMonth", "step", "forecast_period", "leadtime_hour")
    names = {"tmin": ("mn2t24", "mn2t", "tmin"),
             "tmax": ("mx2t24", "mx2t", "tmax"),
             "precipitation": ("tp", "total_precipitation")}
    out = {}
    for v, cands in names.items():
        da = ds[_pick(ds, *cands)]
        # The delivered arrays are 5-D: a length-1 `forecast_reference_time`
        # sits between member and lead. Squeeze every size-1 dimension that is
        # not one of the four we index, otherwise the transpose below is given
        # four names for a five-axis array.
        for d in list(da.dims):
            if d not in (mdim, ldim, la, lo) and da.sizes[d] == 1:
                da = da.squeeze(d, drop=True)
        extra = [d for d in da.dims if d not in (mdim, ldim, la, lo)]
        if extra:
            raise SystemExit(f"{path.name}: unexpected dimension(s) {extra} on "
                             f"{v}; expected only member/lead/lat/lon")
        da = da.transpose(mdim, ldim, la, lo)
        a = np.asarray(da.values, dtype=np.float64)
        if a.shape[1] != N_LEAD:
            raise SystemExit(f"{path.name}: expected {N_LEAD} lead steps, "
                             f"got {a.shape[1]}")
        if v == "precipitation":
            # accumulated from forecast start, in metres -> per-day mm
            a = np.diff(a, axis=1, prepend=0.0) * 1000.0
            np.maximum(a, 0.0, out=a)
        out[v] = _ascending(a, lat, lon)
    ds.close()
    return out


def _ascending(a, lat, lon):
    """Reorder the last two axes onto ascending lat then ascending lon."""
    if lat[0] > lat[-1]:
        a = a[..., ::-1, :]
    if lon[0] > lon[-1]:
        a = a[..., :, ::-1]
    return a


def load_era5(years):
    """-> (DatetimeIndex, {var: (n_days, 10, 13)}) on the coarse frame."""
    frames = {}
    for v in FORCING_VARS:
        parts = []
        for y in years:
            path_y = ERA5_RAW / f"era5_{v}_{y}.nc"
            if not path_y.exists():
                return None, None
            ds = xr.open_dataset(path_y)
            la, lo = _pick(ds, "latitude", "lat"), _pick(ds, "longitude", "lon")
            td = _pick(ds, "valid_time", "time", "date")
            name = _pick(ds, "t2m", "tp", "2m_temperature", "total_precipitation")
            da = ds[name].transpose(td, la, lo)
            a = np.asarray(da.values, dtype=np.float64)
            if v == "precipitation":
                # The two acquisition paths deliver different units, so convert
                # on the UNITS ATTRIBUTE, never on the variable name: the derived
                # daily-statistics dataset gives metres, while
                # download_era5_hourly.py has already summed hourly metres into
                # mm. Keying off the name would double-convert whichever path is
                # not the one assumed -- a factor of 1000 in the channel that
                # drives recharge, which the bias correction would then quietly
                # absorb into its quantile map instead of failing.
                u = str(da.attrs.get("units", "")).strip().lower()
                if u in ("m", "metre", "meters", "metres"):
                    a = a * 1000.0
                elif u not in ("mm", "kg m-2", "kg/m2"):
                    raise SystemExit(
                        f"{path_y.name}: unrecognised ERA5 precipitation units "
                        f"{u!r}; expected m or mm")
            c = coarsen_latlon(_ascending(a, ds[la].values, ds[lo].values),
                               np.sort(ds[la].values), np.sort(ds[lo].values))
            parts.append((pd.to_datetime(ds[td].values), c))
            ds.close()
        dates = pd.DatetimeIndex(np.concatenate([p[0].values for p in parts]))
        arr = np.concatenate([p[1] for p in parts], axis=0)
        o = np.argsort(dates.values)
        frames[v] = (dates[o], arr[o])
    d0 = frames[FORCING_VARS[0]][0]
    for v in FORCING_VARS[1:]:
        if not frames[v][0].equals(d0):
            raise SystemExit("ERA5 variables span different dates")
    return d0, {v: frames[v][1] for v in FORCING_VARS}


def load_nldas(wys):
    """-> (DatetimeIndex, {var: (n_days, 637, 665)}) at 1 km, as downloaded."""
    dates, out = [], {v: [] for v in FORCING_VARS}
    for wy in wys:
        ds = xr.open_dataset(PARFLOW / f"forcing_wy{wy}.nc")
        dates.append(pd.to_datetime(ds["time"].values))
        for v in FORCING_VARS:
            out[v].append(np.asarray(ds[v].values, dtype=np.float32))
        ds.close()
    d = pd.DatetimeIndex(np.concatenate([x.values for x in dates]))
    return d, {v: np.concatenate(out[v], axis=0) for v in FORCING_VARS}


# ---------------------------------------------------------- quantile mapping

def fit_qm(S, E, var):
    qs = np.linspace(0.0, 1.0, N_QUANTILE)
    if var != "precipitation":
        return dict(kind="temp",
                    sq=np.maximum.accumulate(np.quantile(S, qs)),
                    eq=np.maximum.accumulate(np.quantile(E, qs)))
    p_dry = float(np.mean(E <= WET_MM))
    thresh = float(np.quantile(S, p_dry)) if p_dry > 0 else -1.0
    Sw, Ew = S[S > thresh], E[E > WET_MM]
    if Sw.size < 50 or Ew.size < 50:
        return dict(kind="precip", thresh=thresh, sq=None, eq=None)
    return dict(kind="precip", thresh=thresh,
                sq=np.maximum.accumulate(np.quantile(Sw, qs)),
                eq=np.maximum.accumulate(np.quantile(Ew, qs)))


def apply_qm(x, k):
    x = np.asarray(x, dtype=np.float64)
    if k["kind"] == "temp":
        sq, eq = k["sq"], k["eq"]
        y = np.interp(x, sq, eq)
        lo, hi = x < sq[0], x > sq[-1]
        y[lo] = x[lo] + (eq[0] - sq[0])        # additive, not clipped
        y[hi] = x[hi] + (eq[-1] - sq[-1])
        return y
    y = np.zeros_like(x)
    wet = x > k["thresh"]
    if k["sq"] is None:
        y[wet] = x[wet]
        return y
    sq, eq = k["sq"], k["eq"]
    xv = x[wet]
    yv = np.interp(xv, sq, eq)
    hi = xv > sq[-1]
    if sq[-1] > 0:
        yv[hi] = xv[hi] * (eq[-1] / sq[-1])    # multiplicative tail
    y[wet] = yv
    return y


def lead_windows():
    return [np.arange(a, b) for a, b in zip(LEAD_EDGES[:-1], LEAD_EDGES[1:])]


# ------------------------------------------------------------------ stages

def stage_seas5(cmap):
    """Raw and bias-corrected SEAS5 on the coarse frame, for the target init."""
    y0, m0 = TARGET_INIT
    target = load_seas5(y0, m0)
    if target is None:
        print(f"  SKIP seas5: {SEAS5_RAW}/seas5_{y0}{m0:02d}.nc missing "
              f"(accept the SEAS5 licence, then run download_seas5.py)")
        return
    clim = {}
    for y in QM_YEARS:
        c = load_seas5(y, m0)
        if c is not None:
            clim[y] = c
    if len(clim) < 10:
        print(f"  SKIP seas5 bias correction: only {len(clim)} hindcast inits "
              f"present, need >=10 (run download_seas5.py --climatology)")
        return
    era_dates, era = load_era5(sorted(clim))
    if era is None:
        print("  SKIP seas5 bias correction: ERA5 reference incomplete "
              "(run download_era5_daily.py)")
        return

    init = pd.Timestamp(f"{y0}-{m0:02d}-01")
    n_mem = target[FORCING_VARS[0]].shape[0]
    ny_c, nx_c = COARSE_LAT.size, COARSE_LON.size
    windows = lead_windows()
    diag = {"n_members": int(n_mem), "n_clim_inits": len(clim),
            "clim_years": sorted(clim), "lead_edges": LEAD_EDGES,
            "n_quantile": N_QUANTILE, "wet_mm": WET_MM, "vars": {}}

    # ERA5 sample aligned to lead index: era_s[year_i, lead-1] is the ERA5 value
    # on the calendar day that lead describes for that year's initialisation.
    years = sorted(clim)
    era_pos = {d: i for i, d in enumerate(era_dates)}
    lead_rows = []
    for y in years:
        base = pd.Timestamp(f"{y}-{m0:02d}-01")
        idx = [era_pos.get(base + pd.Timedelta(days=k)) for k in range(N_LEAD)]
        if any(i is None for i in idx):
            raise SystemExit(f"ERA5 missing days for the {y}-{m0:02d} init; "
                             "extend download_era5_daily.py's year list")
        lead_rows.append(idx)
    lead_rows = np.asarray(lead_rows)                            # (n_years, 215)

    out = {}
    for v in FORCING_VARS:
        s_clim = np.stack([clim[y][v] for y in years], axis=0)   # (Y,M,215,ny,nx)
        e_clim = era[v][lead_rows]                               # (Y,215,ny,nx)
        raw = target[v]                                          # (M,215,ny,nx)
        bc = np.empty_like(raw)
        before = after = 0.0
        for w in windows:
            li = w - 1
            for i in range(ny_c):
                for j in range(nx_c):
                    S = s_clim[:, :, li, i, j].ravel()
                    E = e_clim[:, li, i, j].ravel()
                    k = fit_qm(S, E, v)
                    bc[:, li, i, j] = apply_qm(raw[:, li, i, j], k).reshape(
                        raw.shape[0], li.size)
            cov = cmap.covered.reshape(ny_c, nx_c)
            before += float(np.mean(s_clim[:, :, li][..., cov])
                            - np.mean(e_clim[:, li][..., cov]))
            after += float(np.mean(bc[:, li][..., cov])
                           - np.mean(e_clim[:, li][..., cov]))
        out[f"raw_{v}"] = raw.astype(np.float32)
        out[f"bc_{v}"] = bc.astype(np.float32)
        cov = cmap.covered.reshape(ny_c, nx_c)
        diag["vars"][v] = {
            "clim_bias_vs_era5_mean_over_windows": before / len(windows),
            "target_bias_after_correction": after / len(windows),
            "raw_domain_mean": float(np.mean(raw[..., cov])),
            "bc_domain_mean": float(np.mean(bc[..., cov])),
        }
        if v == "precipitation":
            diag["vars"][v]["wet_day_frac_raw"] = float(
                np.mean(raw[..., cov] > WET_MM))
            diag["vars"][v]["wet_day_frac_bc"] = float(
                np.mean(out["bc_precipitation"][..., cov] > WET_MM))
            diag["vars"][v]["wet_day_frac_era5_clim"] = float(
                np.mean(e_clim[..., cov] > WET_MM))
        print(f"    {v}: raw mean {diag['vars'][v]['raw_domain_mean']:.4g} -> "
              f"bc {diag['vars'][v]['bc_domain_mean']:.4g}", flush=True)

    dates = pd.date_range(init, periods=N_LEAD, freq="D")
    ds = xr.Dataset(
        {k: (("member", "lead_day", "lat", "lon"), a) for k, a in out.items()},
        coords={"member": np.arange(n_mem), "lead_day": np.arange(1, N_LEAD + 1),
                "lat": COARSE_LAT, "lon": COARSE_LON,
                "valid_date": ("lead_day", dates)},
        attrs={"init": str(init.date()), "source": "ECMWF SEAS5 (C3S system 5)",
               "correction": "stage 1 quantile mapping to ERA5; stage 2 delta "
                             "applied at composition time, not here",
               "note": "lead_day k describes calendar day init + (k-1) days"})
    for v in FORCING_VARS:
        u = "mm" if v == "precipitation" else "K"
        ds[f"raw_{v}"].attrs["units"] = u
        ds[f"bc_{v}"].attrs["units"] = u
    p = OUT / f"forcing_coarse_seas5_{y0}{m0:02d}.nc"
    ds.to_netcdf(p, encoding={k: NC_ENC for k in out})
    (OUT / f"qm_diag_{y0}{m0:02d}.json").write_text(json.dumps(diag, indent=1))
    print(f"  wrote {p.name} ({p.stat().st_size/1e6:.1f} MB) and its diagnostics")


# Initialisations to prepare, settable from the command line. The default is
# the four October starts; the full-year experiment also needs the 1 May
# re-initialisations, which are not in TARGET_INITS. Leaving this hardcoded
# meant the May raw files were copied into place and silently never converted,
# and the rollout then failed looking for a coarse file nobody had written.
EXTRA_INITS = []


def stage_seas5_raw(cmap):
    """Raw SEAS5 on the 1 deg frame, one file per initialisation -- arm D/Dp input.

    No ERA5, no hindcast climatology, no quantile map: this is the uncorrected
    path, which is what makes it runnable from a single CDS request per forecast.
    The file carries the same `raw_*` variable names the corrected stage writes,
    so ArmForcing reads either without caring which produced it; `corrected` in
    the attributes records which it was.

    What this cannot do is hide a bias. A raw seasonal forecast carries the
    forecast system's systematic error straight into the conditioning, and over
    14 chained calls that error compounds. Reporting arm D is therefore a
    measurement of that, not a workaround for it -- see SEAS5.md section 7a.
    """
    wrote = 0
    for (y0, m0) in list(TARGET_INITS) + list(EXTRA_INITS):
        target = load_seas5(y0, m0)
        if target is None:
            print(f"  SKIP {y0}-{m0:02d}: "
                  f"{SEAS5_RAW}/seas5_{y0}{m0:02d}.nc missing")
            continue
        n_mem = target[FORCING_VARS[0]].shape[0]
        n_lead = target[FORCING_VARS[0]].shape[1]
        init = pd.Timestamp(f"{y0}-{m0:02d}-01")
        wy, fold = INIT_TO_FOLD.get(y0, (None, None))
        cov = cmap.covered.reshape(COARSE_LAT.size, COARSE_LON.size)
        out = {f"raw_{v}": target[v].astype(np.float32) for v in FORCING_VARS}

        ds = xr.Dataset(
            {k: (("member", "lead_day", "lat", "lon"), a) for k, a in out.items()},
            coords={"member": np.arange(n_mem),
                    "lead_day": np.arange(1, n_lead + 1),
                    "lat": COARSE_LAT, "lon": COARSE_LON,
                    "valid_date": ("lead_day",
                                   pd.date_range(init + pd.Timedelta(days=1),
                                                 periods=n_lead, freq="D"))},
            attrs={"init": str(init.date()), "water_year": str(wy),
                   "fold": str(fold), "corrected": "false",
                   "source": "ECMWF SEAS5 (C3S system 5), uncorrected",
                   "note": "lead_day k describes calendar day init + (k-1); "
                           "total_precipitation de-accumulated to mm/day"})
        for v in FORCING_VARS:
            ds[f"raw_{v}"].attrs["units"] = "mm" if v == "precipitation" else "K"
        path = OUT / f"forcing_coarse_seas5_{y0}{m0:02d}.nc"
        ds.to_netcdf(path, encoding={k: NC_ENC for k in out})
        wrote += 1
        pm = float(np.mean(out["raw_precipitation"][..., cov]))
        wet = float(np.mean(out["raw_precipitation"][..., cov] > WET_MM))
        print(f"  {init.date()} -> WY{wy} (fold {fold}): {n_mem} members x "
              f"{n_lead} lead days | precip mean {pm:.3f} mm/d, wet-day frac "
              f"{wet:.2f} | tmin mean {np.mean(out['raw_tmin'][..., cov]):.2f} K"
              f" -> {path.name}", flush=True)
    if wrote:
        print(f"  wrote {wrote} initialisation(s); nothing is bias-corrected")


def stage_stage2(cmap):
    """The ERA5 -> NLDAS-2 1 km monthly delta, from fold 1's training years."""
    nd, nl = load_nldas(TRAIN_WYS)
    ed, er = load_era5(sorted({d.year for d in
                               pd.date_range(WATER_YEARS[TRAIN_WYS[0]][0],
                                             WATER_YEARS[TRAIN_WYS[-1]][1])}))
    if er is None:
        print("  SKIP stage2: ERA5 for 2002-2005 incomplete "
              "(run download_era5_daily.py)")
        return
    keep = nd.isin(ed)
    if not keep.all():
        raise SystemExit(f"{int((~keep).sum())} NLDAS-2 days have no ERA5 day")
    pos = pd.Index(ed).get_indexer(nd)
    delta = {}
    for v in FORCING_VARS:
        e_fine = cmap.to_1km(er[v][pos])                     # (n_days,637,665)
        a = nl[v]
        d = np.empty((12, NY, NX), dtype=np.float32)
        for m in range(1, 13):
            sel = nd.month == m
            n_mean = a[sel].mean(axis=0)
            e_mean = e_fine[sel].mean(axis=0)
            if v == "precipitation":
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.where(e_mean > 1e-6, n_mean / e_mean, 1.0)
                d[m - 1] = np.clip(r, 0.25, 4.0)
            else:
                d[m - 1] = n_mean - e_mean
        delta[v] = d
        lab = "ratio" if v == "precipitation" else "K"
        print(f"    {v}: monthly delta range {d.min():.3f} .. {d.max():.3f} [{lab}]",
              flush=True)
    ds = xr.Dataset({v: (("month", "y", "x"), delta[v]) for v in FORCING_VARS},
                    coords={"month": np.arange(1, 13)},
                    attrs={"fitted_on": f"WY{TRAIN_WYS[0]}-{TRAIN_WYS[-1]} "
                                        "(fold 1 training years only)",
                           "tmin_tmax": "additive, K",
                           "precipitation": "multiplicative ratio, clipped 0.25-4"})
    p = OUT / "downscale_delta.nc"
    ds.to_netcdf(p, encoding={v: NC_ENC for v in FORCING_VARS})
    print(f"  wrote {p.name} ({p.stat().st_size/1e6:.1f} MB)")


def stage_armB(cmap):
    """NLDAS-2 for the target water year, coarsened to 1 deg (arm B's input)."""
    nd, nl = load_nldas([TARGET_WY])
    coarse = {v: cmap.coarsen(nl[v]).astype(np.float32) for v in FORCING_VARS}
    ds = xr.Dataset({v: (("time", "lat", "lon"), coarse[v]) for v in FORCING_VARS},
                    coords={"time": nd, "lat": COARSE_LAT, "lon": COARSE_LON},
                    attrs={"arm": "B", "source": f"NLDAS-2 WY{TARGET_WY} at 1 km, "
                                                 "averaged onto the SEAS5 frame",
                           "note": "no stage-2 delta: already in NLDAS-2 space"})
    p = OUT / f"forcing_coarse_armB_wy{TARGET_WY}.nc"
    ds.to_netcdf(p, encoding={v: NC_ENC for v in FORCING_VARS})
    for v in FORCING_VARS:
        print(f"    {v}: 1 km mean {nl[v].mean():.4g} -> coarse mean "
              f"{coarse[v][:, cmap.covered.reshape(COARSE_LAT.size, -1)].mean():.4g}")
    print(f"  wrote {p.name} ({p.stat().st_size/1e6:.1f} MB)")


def stage_armB2(cmap):
    """ERA5 over the verification water year, coarsened (arm B2's input)."""
    yrs = sorted({pd.Timestamp(WATER_YEARS[TARGET_WY][0]).year,
                  pd.Timestamp(WATER_YEARS[TARGET_WY][1]).year})
    ed, er = load_era5(yrs)
    if er is None:
        print(f"  SKIP armB2: ERA5 for {yrs} incomplete")
        return
    lo, hi = (pd.Timestamp(x) for x in WATER_YEARS[TARGET_WY])
    sel = (ed >= lo) & (ed <= hi)
    ds = xr.Dataset({v: (("time", "lat", "lon"), er[v][sel].astype(np.float32))
                     for v in FORCING_VARS},
                    coords={"time": ed[sel], "lat": COARSE_LAT, "lon": COARSE_LON},
                    attrs={"arm": "B2", "source": f"ERA5 daily, WY{TARGET_WY}, "
                                                  "averaged onto the SEAS5 frame",
                           "note": "forcing INPUT standing in for a perfect "
                                   "forecast; never a bias-correction sample. "
                                   "Compose WITH the stage-2 delta."})
    p = OUT / f"forcing_coarse_armB2_wy{TARGET_WY}.nc"
    ds.to_netcdf(p, encoding={v: NC_ENC for v in FORCING_VARS})
    print(f"  wrote {p.name} ({p.stat().st_size/1e6:.1f} MB)")


def stage_armE(cmap):
    """NLDAS-2 day-of-year climatology at 1 km, from the training years only."""
    nd, nl = load_nldas(TRAIN_WYS)
    doy = nd.dayofyear.values
    clim = {}
    for v in FORCING_VARS:
        c = np.zeros((366, NY, NX), dtype=np.float32)
        for d in range(1, 367):
            sel = doy == d
            if sel.any():
                c[d - 1] = nl[v][sel].mean(axis=0)
            else:
                c[d - 1] = nl[v][doy == 365].mean(axis=0)   # 29 Feb fallback
        clim[v] = c
        print(f"    {v}: climatology mean {c.mean():.4g}")
    ds = xr.Dataset({v: (("dayofyear", "y", "x"), clim[v]) for v in FORCING_VARS},
                    coords={"dayofyear": np.arange(1, 367)},
                    attrs={"arm": "E", "fitted_on": f"WY{TRAIN_WYS[0]}-"
                                                    f"{TRAIN_WYS[-1]} NLDAS-2",
                           "note": "the no-forecast-information reference that "
                                   "seasonal skill is scored against"})
    p = OUT / "nldas_climatology_1km.nc"
    ds.to_netcdf(p, encoding={v: NC_ENC for v in FORCING_VARS})
    print(f"  wrote {p.name} ({p.stat().st_size/1e6:.1f} MB)")


FOLD_TEST = {1: 2006, 2: 2005, 3: 2004, 4: 2003}
FOLD_TRAIN = {1: [2003, 2004, 2005], 2: [2003, 2004, 2006],
              3: [2003, 2005, 2006], 4: [2004, 2005, 2006]}
WY_INIT = {wy: (y, 10) for y, wy in INIT_TO_FOLD.items()
           for wy in (INIT_TO_FOLD[y][0],)}


def _coarse_nldas(wy, cmap, n_lead=N_LEAD):
    """NLDAS-2 for the first n_lead days of a water year, on the coarse frame."""
    ds = xr.open_dataset(PARFLOW / f"forcing_wy{wy}.nc")
    out = {}
    for v in FORCING_VARS:
        fine = np.asarray(ds[v].values[:n_lead], dtype=np.float32)
        out[v] = cmap.coarsen(fine)                      # (n_lead, nlat, nlon)
    dates = pd.DatetimeIndex(ds["time"].values)[:n_lead]
    ds.close()
    return dates, out


def stage_bcsd(cmap):
    """BCSD against NLDAS-2 alone -- no ERA5. One correction per fold.

    BC: quantile mapping of SEAS5 onto NLDAS-2, fitted on the fold's THREE
    TRAINING water years and POOLED ACROSS THE 59 COVERED COARSE CELLS.

      Pooling is what makes three years enough. Per cell per 43-day lead window
      three years give only ~129 NLDAS-2 days; pooled over the covered cells
      that is ~7,600, which is ample for a quantile map. The assumption it buys
      is that the bias is similar across the domain -- a 6 x 9 degree box, so
      not a wild one, and `pooling_diagnostic` below reports the per-cell spread
      so the assumption can be checked rather than believed. The alternative, a
      per-cell map, is what the abandoned ERA5 bridge was for.

      Precipitation gets the wet-day frequency correction first, then the
      quantile map on wet values, because the measured bias is in BOTH
      components -- 1.22x too many wet days and 1.38x too much rain on them --
      so neither a frequency fix nor a mean scaling alone is adequate.

    SD: spatial disaggregation by the ratio of NLDAS-2's own 1 km monthly
    climatology to that same climatology coarsened and interpolated back. This
    restores the 1 km climatological pattern (lake shore, terrain, land cover)
    that a 1 degree field cannot carry, and uses nothing but the fold's training
    years.

    LEAK DISCIPLINE: every statistic comes from the fold's training water years.
    The test year's NLDAS-2 appears only in the after-the-fact diagnostic
    printed at the end, which is verification, not fitting.

    Applied PER MEMBER, so all 25 survive the correction.
    """
    windows = lead_windows()
    ny_c, nx_c = COARSE_LAT.size, COARSE_LON.size
    cov = cmap.covered.reshape(ny_c, nx_c)
    report = {}

    for fold in sorted(FOLD_TEST):
        test_wy = FOLD_TEST[fold]
        train_wys = FOLD_TRAIN[fold]
        t_init = WY_INIT.get(test_wy)
        tr_inits = [WY_INIT[w] for w in train_wys if w in WY_INIT]
        if t_init is None or len(tr_inits) < len(train_wys):
            print(f"  fold {fold}: need inits for WY{test_wy} and {train_wys}")
            continue
        target = load_seas5(*t_init)
        fit_s5 = {w: load_seas5(*WY_INIT[w]) for w in train_wys}
        if target is None or any(v is None for v in fit_s5.values()):
            have = [w for w, v in fit_s5.items() if v is not None]
            print(f"  SKIP fold {fold}: SEAS5 missing (test "
                  f"{'ok' if target is not None else 'missing'}, "
                  f"train present for {have})")
            continue

        fit_nl = {}
        for w in train_wys:
            _, c = _coarse_nldas(w, cmap)
            fit_nl[w] = c

        # ---- BC: pooled quantile map, fitted on training years only ----------
        out = {f"raw_{v}": target[v].astype(np.float32) for v in FORCING_VARS}
        diag = {"fold": fold, "test_wy": test_wy, "train_wys": train_wys,
                "pooled_cells": int(cov.sum()), "vars": {}}
        for v in FORCING_VARS:
            bc = np.empty_like(target[v])
            knots = {}
            for wi, w in enumerate(windows):
                li = w - 1
                S = np.concatenate([fit_s5[y][v][:, li][..., cov].ravel()
                                    for y in train_wys])
                E = np.concatenate([fit_nl[y][v][li][..., cov].ravel()
                                    for y in train_wys])
                k = fit_qm(S, E, v)
                knots[wi] = dict(n_seas5=int(S.size), n_nldas=int(E.size))
                flat = target[v][:, li].reshape(-1)
                bc[:, li] = apply_qm(flat, k).reshape(target[v][:, li].shape)
            out[f"bc_{v}"] = bc.astype(np.float32)
            diag["vars"][v] = {
                "samples_per_window": knots[0],
                "raw_mean": float(np.mean(target[v][..., cov])),
                "bc_mean": float(np.mean(bc[..., cov])),
                "train_nldas_mean": float(np.mean(
                    [np.mean(fit_nl[y][v][..., cov]) for y in train_wys])),
            }
            if v == "precipitation":
                diag["vars"][v]["wetfrac_raw"] = float(
                    np.mean(target[v][..., cov] > WET_MM))
                diag["vars"][v]["wetfrac_bc"] = float(
                    np.mean(bc[..., cov] > WET_MM))
                diag["vars"][v]["wetfrac_train_nldas"] = float(np.mean(
                    [np.mean(fit_nl[y][v][..., cov] > WET_MM) for y in train_wys]))

        # PHYSICAL CONSISTENCY OF THE TEMPERATURE PAIR. tmin and tmax get
        # separate transfer functions, which does not by itself guarantee
        # tmax >= tmin: a day at a high tmin percentile and a low tmax
        # percentile could cross. In practice it does not, because each is
        # mapped onto NLDAS-2's own internally-consistent distribution and the
        # two are strongly rank-correlated on a given day (a synthetic test with
        # SEAS5's diurnal range compressed to 7.2 K against NLDAS-2's 10.2 K
        # produced zero inversions in 60,000 samples). But an impossible pair
        # reaching the model would be out of distribution in a way nothing else
        # would catch, so count them, and swap rather than clamp -- a swap keeps
        # both values and the diurnal range, and only fixes which is labelled
        # which.
        bad = out["bc_tmax"] < out["bc_tmin"]
        n_bad = int(bad.sum())
        if n_bad:
            lo = np.minimum(out["bc_tmin"], out["bc_tmax"])
            hi = np.maximum(out["bc_tmin"], out["bc_tmax"])
            out["bc_tmin"], out["bc_tmax"] = lo, hi
            print(f"    NOTE fold {fold}: tmax < tmin after mapping in {n_bad} "
                  f"of {bad.size} values ({100*n_bad/bad.size:.4f}%); swapped. "
                  f"If this is ever more than a rounding-level fraction, switch "
                  f"to mapping mean temperature and diurnal range instead.",
                  flush=True)
        diag["tmax_lt_tmin_after_bc"] = n_bad
        diag["tmax_lt_tmin_frac"] = float(n_bad / bad.size)

        n_mem, n_lead = target[FORCING_VARS[0]].shape[:2]
        init = pd.Timestamp(f"{t_init[0]}-{t_init[1]:02d}-01")
        ds = xr.Dataset(
            {k: (("member", "lead_day", "lat", "lon"), a) for k, a in out.items()},
            coords={"member": np.arange(n_mem),
                    "lead_day": np.arange(1, n_lead + 1),
                    "lat": COARSE_LAT, "lon": COARSE_LON,
                    "valid_date": ("lead_day",
                                   pd.date_range(init, periods=n_lead, freq="D"))},
            attrs={"init": str(init.date()), "water_year": str(test_wy),
                   "fold": str(fold), "corrected": "true",
                   "method": "BCSD; BC = quantile map onto NLDAS-2 pooled over "
                             f"{int(cov.sum())} coarse cells, fitted on "
                             f"WY{train_wys} only; SD applied at composition "
                             "time from downscale_delta_fold%d.nc" % fold,
                   "source": "ECMWF SEAS5 (C3S system 5)"})
        for v in FORCING_VARS:
            u = "mm" if v == "precipitation" else "K"
            ds[f"raw_{v}"].attrs["units"] = u
            ds[f"bc_{v}"].attrs["units"] = u
        path = OUT / f"forcing_coarse_seas5_{t_init[0]}{t_init[1]:02d}.nc"
        ds.to_netcdf(path, encoding={k: NC_ENC for k in out})

        # ---- SD: per-fold 1 km monthly disaggregation field -------------------
        acc = {v: np.zeros((12, NY, NX)) for v in FORCING_VARS}
        cnt = np.zeros(12)
        for w in train_wys:
            z = xr.open_dataset(PARFLOW / f"forcing_wy{w}.nc")
            mon = pd.DatetimeIndex(z["time"].values).month
            for m in range(1, 13):
                sel = mon == m
                if not sel.any():
                    continue
                for v in FORCING_VARS:
                    acc[v][m - 1] += np.asarray(z[v].values[sel],
                                                dtype=np.float64).sum(axis=0)
                cnt[m - 1] += int(sel.sum())
            z.close()
        cnt = np.where(cnt == 0, 1, cnt)
        delta = {}
        for v in FORCING_VARS:
            fine = (acc[v] / cnt[:, None, None]).astype(np.float32)   # (12,Y,X)
            smooth = cmap.to_1km(cmap.coarsen(fine))                  # same, via 1 deg
            if v == "precipitation":
                with np.errstate(invalid="ignore", divide="ignore"):
                    r = np.where(smooth > 1e-6, fine / smooth, 1.0)
                delta[v] = np.clip(r, 0.25, 4.0).astype(np.float32)
            else:
                delta[v] = (fine - smooth).astype(np.float32)
        dsd = xr.Dataset({v: (("month", "y", "x"), delta[v]) for v in FORCING_VARS},
                         coords={"month": np.arange(1, 13)},
                         attrs={"fold": str(fold),
                                "fitted_on": f"NLDAS-2 WY{train_wys}",
                                "tmin_tmax": "additive, K",
                                "precipitation": "multiplicative, clipped 0.25-4",
                                "note": "BCSD spatial disaggregation: ratio of the "
                                        "1 km monthly climatology to itself passed "
                                        "through the 1 deg frame"})
        dpath = OUT / f"downscale_delta_fold{fold}.nc"
        dsd.to_netcdf(dpath, encoding={v: NC_ENC for v in FORCING_VARS})

        # ---- diagnostic against the TEST year (verification, never fitted) ----
        _, test_nl = _coarse_nldas(test_wy, cmap)
        for v in FORCING_VARS:
            act = float(np.mean(test_nl[v][..., cov]))
            diag["vars"][v]["test_nldas_actual"] = act
            diag["vars"][v]["raw_vs_actual_pct"] = 100 * (
                diag["vars"][v]["raw_mean"] / act - 1) if act else float("nan")
            diag["vars"][v]["bc_vs_actual_pct"] = 100 * (
                diag["vars"][v]["bc_mean"] / act - 1) if act else float("nan")
        report[fold] = diag
        pv = diag["vars"]["precipitation"]
        print(f"  fold {fold} (test WY{test_wy}, fit WY{train_wys}): "
              f"precip raw {pv['raw_mean']:.3f} -> bc {pv['bc_mean']:.3f} mm/d "
              f"(actual {pv['test_nldas_actual']:.3f}); "
              f"wet-day frac {pv['wetfrac_raw']:.3f} -> {pv['wetfrac_bc']:.3f} "
              f"(train NLDAS-2 {pv['wetfrac_train_nldas']:.3f})", flush=True)
        print(f"           -> {path.name}, {dpath.name}", flush=True)

    if report:
        (OUT / "bcsd_report.json").write_text(json.dumps(report, indent=1))
        print(f"  wrote {OUT / 'bcsd_report.json'}")


STAGES = {"stage2": stage_stage2, "armB": stage_armB, "armB2": stage_armB2,
          "armE": stage_armE, "seas5": stage_seas5, "seas5_raw": stage_seas5_raw,
          "bcsd": stage_bcsd}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inits", nargs="*", default=[],
                    help="extra initialisations to prepare, YYYYMM, e.g. 200605")
    ap.add_argument("--stages", nargs="+", default=list(STAGES),
                    choices=list(STAGES))
    args = ap.parse_args()
    global EXTRA_INITS
    EXTRA_INITS = [(int(t[:4]), int(t[4:6])) for t in args.inits]
    if EXTRA_INITS:
        print(f"  also preparing: {EXTRA_INITS}")
    OUT.mkdir(parents=True, exist_ok=True)
    cmap = CoarseMap()
    print(f"coarse frame {COARSE_LAT.size} x {COARSE_LON.size}; "
          f"{int(cmap.covered.sum())} cells cover the domain "
          f"(NLDAS-2 at 1/8 deg would supply roughly 3,300)", flush=True)
    for s in args.stages:
        print(f"\n== {s} ==", flush=True)
        STAGES[s](cmap)


if __name__ == "__main__":
    main()
