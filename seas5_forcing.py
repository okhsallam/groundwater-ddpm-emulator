"""
Shared grid and composition machinery for driving the groundwater emulator with
SEAS5 forcing. Imported by prepare_seas5_domain.py (which builds the correction
tables) and by ddpm_model/run_seas5_rollout.py (which consumes them).

It lives in one module on purpose. The experiment's whole interpretive value
rests on four rollout arms differing in exactly one thing each:

  A  NLDAS-2 at 1 km                              emulator error alone
                                                  (already run: fold 1's rollout)
  B  NLDAS-2 coarsened to the SEAS5 grid and
     interpolated back to 1 km                    + resolution loss alone
  C  SEAS5, bias-corrected                        + genuine forecast error
  D  SEAS5, raw                                   what the correction buys
  E  NLDAS-2 day-of-year climatology at 1 km      the "no forecast information"
                                                  reference that seasonal skill
                                                  is properly scored against

If arms B, C and D did not pass through bit-identical coarsening and
interpolation code, a difference between them could be an artefact of two
interpolators rather than the thing each arm is supposed to isolate. Hence one
module, one `interp_coarse_to_1km`, one `coarsen_1km`.

WHY BILINEAR HERE, WHEN prepare_glofas_domain.py ARGUED FOR NEAREST NEIGHBOUR.
That argument was about a 0.05 deg source going to a 1 km target: the source
held real structure (a channel network, runoff peaks) that bilinear would smear,
and a visible 5 km staircase was the honest representation. Here the source is
1 deg -- about 100x COARSER than the target, roughly 88 cells over the whole
domain. There is no fine structure to preserve, so the choice is purely about
what artefact to introduce, and nearest neighbour would stamp 100 km blocks with
step discontinuities into a field the network reads as spatial texture. Bilinear
is the lesser distortion. The honest statement of the limit is in `compose`.

THE RESOLUTION GAP IS THE DEEPEST CAVEAT OF THE WHOLE EXPERIMENT. The model was
trained on NLDAS-2 at 1/8 deg (~0.125 deg) resampled to 1 km -- order 49 x 68
distinct forcing values across the domain. SEAS5 delivers order 6 x 9. Even with
a perfect bias correction the forcing is out of distribution in spatial texture,
which is why arm B exists and why it must be reported alongside arm C.
"""
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

HERE = Path(__file__).resolve().parent
LATLON_CACHE = HERE / "data" / "glofas" / "conus1_ceser_latlon.npz"

NY, NX = 637, 665
GRID = "conus1"
GRID_BOUNDS = [2053, 565, 2718, 1202]

# The 1 deg grid SEAS5 is delivered on, as implied by the download's
# area = [46, -96, 37, -84] request. ECMWF's 1 deg grid has points at whole
# degrees, each standing for a cell centred there and spanning +/- 0.5 deg.
# Defined canonically here rather than read from the SEAS5 file so that arm B is
# buildable before SEAS5 has downloaded; `assert_matches_seas5` then fails loudly
# if the delivered grid ever differs, instead of silently regridding onto the
# wrong frame.
COARSE_LAT = np.arange(37.0, 46.0 + 1e-9, 1.0)        # ascending
COARSE_LON = np.arange(-96.0, -84.0 + 1e-9, 1.0)      # ascending
COARSE_RES = 1.0

FORCING_VARS = ["tmin", "tmax", "precipitation"]

# ---------------------------------------------------------------------------
# THE BIAS-CORRECTION REFERENCE PERIOD -- ONE DEFINITION, USED EVERYWHERE.
#
# This lived in two files once (download_seas5.py fetched 1981-2022 while the
# ERA5 downloader covered 1993-2017) and the two drifted. The failure was
# silent and total: stage 1 pairs every SEAS5 hindcast init with ERA5 days, so a
# single init year without ERA5 made `load_era5` return None and the whole bias
# correction was skipped with a one-line "reference incomplete" message. A
# correction that quietly does not happen is worse than one that crashes.
#
# 1993-2016 is the C3S standard reference window for SEAS5 hindcast bias
# adjustment. SEAS5's own hindcast reaches back to 1981, but there is no point
# fetching inits whose verifying ERA5 we do not hold.
QM_REFERENCE_WINDOW = (1993, 2016)


def qm_years(exclude=()):
    """Hindcast init years the quantile map is fitted on.

    `exclude` drops the target initialisation year and the verification year, so
    neither the forecast's own year nor the year it is scored against informs
    its correction.
    """
    lo, hi = QM_REFERENCE_WINDOW
    drop = set(exclude)
    return [y for y in range(lo, hi + 1) if y not in drop]


def era5_years_needed(qm, stage2=(), verif=()):
    """Calendar years of ERA5 that a given set of init years requires.

    An October initialisation runs 215 days and so reaches into MAY OF THE
    FOLLOWING YEAR -- hence every init year's successor. Stage 2 and arm B2 add
    their own whole water years on top.
    """
    years = set(qm) | {y + 1 for y in qm} | set(stage2) | set(verif)
    return sorted(years)

# Water years, as everywhere else in this project: WY Y runs 1 Oct Y-1 to 30 Sep Y.
WATER_YEARS = {2003: ("2002-10-01", "2003-09-30"),
               2004: ("2003-10-01", "2004-09-30"),
               2005: ("2004-10-01", "2005-09-30"),
               2006: ("2005-10-01", "2006-09-30")}


def target_latlon():
    """(637, 665) lat and lon of every conus1 cell centre in the CESER window.

    Reuses the cache prepare_glofas_domain.py already wrote; falls back to
    hf_hydrodata.grid.to_latlon if it is missing.
    """
    if LATLON_CACHE.exists():
        z = np.load(LATLON_CACHE)
        return z["lat"], z["lon"]
    from hf_hydrodata import grid
    x0, y0, x1, y1 = GRID_BOUNDS
    X, Y = np.meshgrid(np.arange(x0, x1, dtype=float),
                       np.arange(y0, y1, dtype=float))
    flat = np.empty(X.size * 2)
    flat[0::2], flat[1::2] = X.ravel(), Y.ravel()
    pairs = np.asarray(grid.to_latlon(GRID, *flat.tolist())).reshape(-1, 2)
    lat, lon = pairs[:, 0].reshape(NY, NX), pairs[:, 1].reshape(NY, NX)
    LATLON_CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(LATLON_CACHE, lat=lat, lon=lon)
    return lat, lon


def assert_matches_seas5(lat, lon, tol=1e-6):
    """Fail loudly if a delivered SEAS5 file is not on the assumed 1 deg frame."""
    got_lat = np.sort(np.asarray(lat, dtype=float))
    got_lon = np.sort(np.asarray(lon, dtype=float))
    ok = (got_lat.shape == COARSE_LAT.shape and got_lon.shape == COARSE_LON.shape
          and np.allclose(got_lat, COARSE_LAT, atol=tol)
          and np.allclose(got_lon, COARSE_LON, atol=tol))
    if not ok:
        raise SystemExit(
            "SEAS5 grid differs from the frame seas5_forcing.py assumes.\n"
            f"  expected lat {COARSE_LAT[0]}..{COARSE_LAT[-1]} "
            f"({COARSE_LAT.size}), lon {COARSE_LON[0]}..{COARSE_LON[-1]} "
            f"({COARSE_LON.size})\n"
            f"  got      lat {got_lat[0]}..{got_lat[-1]} ({got_lat.size}), "
            f"lon {got_lon[0]}..{got_lon[-1]} ({got_lon.size})\n"
            "Update COARSE_LAT/COARSE_LON (and re-run prepare_seas5_domain.py, "
            "since every correction table is tied to that frame).")


class CoarseMap:
    """Index map between the conus1 1 km CESER window and the 1 deg SEAS5 frame.

    Built once (a few hundred ms) and cheap to reuse. Holds both directions:
      * `coarsen`  1 km field -> 1 deg, by averaging the 1 km cells whose centres
        fall in each 1 deg cell. At 1 km the cells are equal-area to well within
        the precision that matters here, so a plain mean IS the area average.
      * `to_1km`   1 deg field -> 1 km, bilinear at the 1 km cell centres.
    """

    def __init__(self):
        self.lat_t, self.lon_t = target_latlon()
        iy = np.rint((self.lat_t - COARSE_LAT[0]) / COARSE_RES).astype(int)
        ix = np.rint((self.lon_t - COARSE_LON[0]) / COARSE_RES).astype(int)
        if iy.min() < 0 or iy.max() >= COARSE_LAT.size or \
           ix.min() < 0 or ix.max() >= COARSE_LON.size:
            raise SystemExit("the CESER domain falls outside the coarse frame; "
                             "widen the download AREA and COARSE_LAT/LON")
        self.flat = (iy * COARSE_LON.size + ix).ravel()
        self.n_coarse = COARSE_LAT.size * COARSE_LON.size
        self.counts = np.bincount(self.flat, minlength=self.n_coarse)
        # The domain is a rotated Lambert window, so the coarse cells it touches
        # are a ragged subset of the frame. Cells with no 1 km centre inside them
        # cannot be coarsened to and are filled by nearest neighbour before any
        # interpolation back -- otherwise a NaN at a corner of a bilinear stencil
        # would poison a strip of the 1 km field along the domain edge.
        self.covered = self.counts > 0
        self._fill_src = self._nearest_covered()
        self._pts = np.stack([self.lat_t.ravel(), self.lon_t.ravel()], axis=-1)

    def _nearest_covered(self):
        gy, gx = np.meshgrid(np.arange(COARSE_LAT.size),
                             np.arange(COARSE_LON.size), indexing="ij")
        gy, gx = gy.ravel(), gx.ravel()
        src = np.arange(self.n_coarse)
        cov = np.flatnonzero(self.covered)
        if cov.size == 0:
            raise SystemExit("no coarse cell contains any domain cell")
        miss = np.flatnonzero(~self.covered)
        d = ((gy[miss][:, None] - gy[cov][None, :]) ** 2
             + (gx[miss][:, None] - gx[cov][None, :]) ** 2)
        src[miss] = cov[np.argmin(d, axis=1)]
        return src

    def coarsen(self, field_1km):
        """(..., 637, 665) -> (..., nlat, nlon). Uncovered cells are filled by
        nearest covered neighbour, so the result is interpolable everywhere."""
        a = np.asarray(field_1km, dtype=np.float64)
        lead = a.shape[:-2]
        flat = a.reshape(-1, NY * NX)
        out = np.empty((flat.shape[0], self.n_coarse))
        for i in range(flat.shape[0]):
            s = np.bincount(self.flat, weights=flat[i], minlength=self.n_coarse)
            with np.errstate(invalid="ignore", divide="ignore"):
                m = s / self.counts
            out[i] = m[self._fill_src]
        return out.reshape(*lead, COARSE_LAT.size, COARSE_LON.size)

    def to_1km(self, field_coarse, method="linear"):
        """(..., nlat, nlon) on the ascending coarse frame -> (..., 637, 665)."""
        a = np.asarray(field_coarse, dtype=np.float64)
        lead = a.shape[:-2]
        flat = a.reshape(-1, COARSE_LAT.size, COARSE_LON.size)
        out = np.empty((flat.shape[0], NY * NX))
        for i in range(flat.shape[0]):
            f = RegularGridInterpolator((COARSE_LAT, COARSE_LON), flat[i],
                                        method=method, bounds_error=False,
                                        fill_value=None)
            out[i] = f(self._pts)
        return out.reshape(*lead, NY, NX).astype(np.float32)


def compose(coarse_fields, delta, months, cmap, apply_delta=True):
    """Build 1 km forcing from a 1 deg field set, optionally with the stage-2
    downscaling delta.

    coarse_fields : {var: (n_days, nlat, nlon)} on the coarse frame
    delta         : {"tmin": (12, 637, 665), "tmax": ..., "precipitation": ...}
                    additive in K for the temperatures, multiplicative for
                    precipitation; indexed by calendar month 1-12
    months        : (n_days,) calendar month of each day, 1-12
    cmap          : a CoarseMap
    apply_delta   : False reproduces the raw arm (D) -- interpolate and stop

    WHAT THE DELTA DOES AND DOES NOT DO. It restores NLDAS-2's CLIMATOLOGICAL
    spatial detail at 1 km -- the lake-shore gradient, the orographic and
    land-cover signal -- and the mean offset between the two reanalyses. It does
    NOT restore daily spatial detail: the day-to-day anomaly stays as smooth as
    the 1 deg source made it. A field composed this way has the right 1 km
    climatology and a 1 deg weather pattern laid over it. That is the honest
    description, and it is exactly what arm B is built to price.
    """
    months = np.asarray(months)
    out = {}
    for v in FORCING_VARS:
        fine = cmap.to_1km(coarse_fields[v])                     # (n_days,637,665)
        if apply_delta:
            d = delta[v][months - 1]                             # (n_days,637,665)
            fine = fine * d if v == "precipitation" else fine + d
        if v == "precipitation":
            np.maximum(fine, 0.0, out=fine)
        out[v] = fine.astype(np.float32)
    return out


def day_index():
    """data/tiles/day_index.json -- the project's global day numbering."""
    with open(HERE / "data" / "tiles" / "day_index.json") as f:
        return json.load(f)


class ArmForcing:
    """Reader for one rollout arm's forcing, in physical units at 1 km.

    Hides which arm needs which correction, so the rollout script contains no
    per-arm branching and cannot accidentally give one arm a correction another
    does not get:

      B   coarsened NLDAS-2, no delta   -- already in NLDAS-2 space
      B2  coarsened ERA5,    delta      -- perfect forecast, full chain
      C   SEAS5 bias-corrected, delta   -- the experiment
      D   SEAS5 raw,         no delta   -- interpolate and normalise, nothing else
      Dp  SEAS5 raw PRECIPITATION only, with NLDAS-2 tmin/tmax
      E   NLDAS-2 1 km day-of-year climatology, no interpolation at all

    WHY Dp EXISTS. Precipitation is what drives recharge, and it is also the
    variable a seasonal system predicts worst. Arm Dp forecasts precipitation and
    holds temperature at the truth, so the skill it loses relative to arm B is
    attributable to PRECIPITATION forecast error alone. Arm D forecasts all three
    at once and cannot separate them. Run both and the difference prices the
    temperature forecast.

    Keeping NLDAS-2 temperature is not a free pass: tmin/tmax are the channels a
    seasonal system predicts comparatively well (they inherit the predictable
    part of the climate signal), so Dp is the optimistic end of a realistic
    range, not a cheat. State it that way in any write-up.

    `n_members` is the forcing ensemble size: 25 for C, D and Dp; 1 for the rest.
    """

    _DELTA_ARMS = {"B2", "C"}
    _SEAS5_ARMS = {"C", "D", "Dp"}
    # Dyear: raw SEAS5 spliced across two initialisations to span a water year.
    # One initialisation reaches 215 days -- verified, not assumed: 365-day
    # requests for all four October starts returned forecast_period = 215 -- so
    # a single forecast cannot cover Oct-Sep. Dyear takes the 1 October forecast
    # to the start of May and the 1 May re-forecast to the end of the year.
    # It is a SEQUENCE OF TWO FORECASTS, not one annual outlook: the atmosphere
    # is re-drawn on 1 May and members are paired by index across that boundary.
    _CHAIN_ARMS = {"Dyear"}

    def __init__(self, arm, cmap=None, root=None, init=None, fold=None):
        import pandas as pd
        self.pd = pd
        if arm not in {"B", "B2", "C", "D", "Dp", "E", "Dyear"}:
            raise SystemExit(f"unknown arm {arm!r}; expected B, B2, C, D, Dp, "
                             "E or Dyear")
        self.arm = arm
        self.cmap = cmap or CoarseMap()
        self.root = Path(root) if root else HERE / "data" / "seas5"
        self.di = day_index()
        self.delta = None
        if arm in self._DELTA_ARMS:
            # Prefer the PER-FOLD disaggregation field. Each fold's field is
            # fitted on that fold's training water years only, so using one
            # fold's field on another would quietly leak the other fold's test
            # year into the forcing. The unsuffixed file is the older
            # ERA5-bridge artefact and is accepted only as a fallback.
            cands = ([self.root / f"downscale_delta_fold{fold}.nc"] if fold
                     else []) + [self.root / "downscale_delta.nc"]
            p = next((c for c in cands if c.exists()), None)
            if p is None:
                raise SystemExit(
                    f"arm {arm} needs {cands[0]}; run "
                    "prepare_seas5_domain.py --stages bcsd")
            if fold and p.name == "downscale_delta.nc":
                print(f"  warning: fold {fold} requested but only "
                      f"{p.name} exists; it was fitted on WY2003-2005")
            d = xr.open_dataset(p)
            self.delta = {v: np.asarray(d[v].values, dtype=np.float32)
                          for v in FORCING_VARS}
            d.close()

        if arm == "E":
            p = self.root / "nldas_climatology_1km.nc"
            self._req(p, "armE")
            d = xr.open_dataset(p)
            self._clim = {v: np.asarray(d[v].values, dtype=np.float32)
                          for v in FORCING_VARS}
            d.close()
            self.n_members = 1
            self.label = "NLDAS-2 day-of-year climatology (WY2003-2005)"
        elif arm in ("B", "B2"):
            cand = sorted(self.root.glob(f"forcing_coarse_arm{arm}_wy*.nc"))
            if not cand:
                raise SystemExit(f"arm {arm} needs "
                                 f"{self.root}/forcing_coarse_arm{arm}_wy*.nc; run "
                                 f"prepare_seas5_domain.py --stages arm{arm}")
            d = xr.open_dataset(cand[-1])
            self._time = pd.DatetimeIndex(d["time"].values)
            self._src = {v: np.asarray(d[v].values, dtype=np.float32)
                         for v in FORCING_VARS}
            self.label = d.attrs.get("source", cand[-1].name)
            d.close()
            self.n_members = 1
        elif arm in self._CHAIN_ARMS:
            from seas5_chained import ChainedForcing
            if init is None:
                raise SystemExit("arm Dyear needs init= (the October start)")
            tag = init if isinstance(init, str) else f"{init[0]}{init[1]:02d}"
            iy, im = int(str(tag)[:4]), int(str(tag)[4:6])
            if im != 10:
                raise SystemExit(f"arm Dyear expects an October init, got {tag}")
            may = f"{iy + 1}05"
            parts = []
            for t in (tag, may):
                sub = ArmForcing("D", cmap=self.cmap, root=self.root,
                                 init=t, fold=fold)
                parts.append((sub._init, sub))
            counts = {p[1].n_members for p in parts}
            if len(counts) != 1:
                raise SystemExit(
                    f"member counts differ across the splice: "
                    f"{[p[1].n_members for p in parts]}. Pairing members by "
                    f"index requires the same ensemble size in both forecasts.")
            self._chain = ChainedForcing(parts)
            self.n_members = parts[0][1].n_members
            self._init = parts[0][0]
            self.label = (f"SEAS5 raw, spliced {self._chain.label}, "
                          f"{self.n_members} members")
            self._nldas = None
        else:
            if init is not None:
                tag = init if isinstance(init, str) else f"{init[0]}{init[1]:02d}"
                cand = [self.root / f"forcing_coarse_seas5_{tag}.nc"]
                if not cand[0].exists():
                    raise SystemExit(f"arm {arm} needs {cand[0]}; run "
                                     "download_seas5.py then "
                                     "prepare_seas5_domain.py --stages seas5_raw")
            else:
                cand = sorted(self.root.glob("forcing_coarse_seas5_*.nc"))
                if not cand:
                    raise SystemExit(
                        f"arm {arm} needs {self.root}/forcing_coarse_seas5_*.nc; "
                        "accept the SEAS5 licence, run download_seas5.py, then "
                        "prepare_seas5_domain.py --stages seas5_raw")
                if len(cand) > 1:
                    raise SystemExit(
                        f"{len(cand)} initialisations present; pass init= to say "
                        f"which: {[c.name for c in cand]}. Guessing would score a "
                        "forecast against the wrong water year.")
            d = xr.open_dataset(cand[0])
            pre = "bc_" if arm == "C" else "raw_"
            missing = [v for v in FORCING_VARS if pre + v not in d.variables]
            if missing:
                raise SystemExit(
                    f"{cand[0].name} has no {pre}{missing[0]} -- it was written "
                    f"{'without' if pre == 'bc_' else 'with'} correction. "
                    f"Arm {arm} needs the {pre}* fields.")
            self._src = {v: np.asarray(d[pre + v].values, dtype=np.float32)
                         for v in FORCING_VARS}
            self._init = pd.Timestamp(d.attrs["init"])
            self.n_members = int(self._src[FORCING_VARS[0]].shape[0])
            kind = {"C": "bias-corrected", "D": "raw",
                    "Dp": "raw precipitation only"}[arm]
            self.label = (f"SEAS5 {kind}, init {self._init.date()}, "
                          f"{self.n_members} members")
            d.close()

            # Arm Dp substitutes NLDAS-2 truth for the two temperature channels,
            # read per call rather than held in memory: a water year of tmin+tmax
            # at 1 km is 1.2 GB and only 15 days are needed per hop.
            self._nldas = None
            if arm == "Dp":
                wy = int(xr.open_dataset(cand[0]).attrs.get("water_year", 0)) \
                     or (self._init.year + 1)
                f = HERE / "data" / "parflow_raw" / f"forcing_wy{wy}.nc"
                if not f.exists():
                    raise SystemExit(f"arm Dp needs NLDAS-2 temperatures at {f}")
                self._nldas = xr.open_dataset(f)
                self._nldas_time = pd.DatetimeIndex(self._nldas["time"].values)
                self.label += " + NLDAS-2 tmin/tmax"

    @staticmethod
    def _req(p, stage):
        if not p.exists():
            raise SystemExit(f"missing {p}; run prepare_seas5_domain.py "
                             f"--stages {stage}")

    def dates(self, gdays):
        return self.pd.to_datetime([self.di[g]["date"] for g in gdays])

    def fields(self, gdays, member=0):
        """{var: (len(gdays), 637, 665)} in K and mm/day, physical units."""
        if self.arm in self._CHAIN_ARMS:
            return self._chain.fields(gdays, member=member)
        dates = self.dates(gdays)
        if self.arm == "E":
            doy = dates.dayofyear.values
            return {v: self._clim[v][doy - 1] for v in FORCING_VARS}
        if self.arm in ("B", "B2"):
            idx = self._time.get_indexer(dates)
            if (idx < 0).any():
                missing = [str(d.date()) for d, i in zip(dates, idx) if i < 0]
                raise SystemExit(f"arm {self.arm} has no forcing for {missing}")
            coarse = {v: self._src[v][idx] for v in FORCING_VARS}
        else:
            lead = np.asarray((dates - self._init).days) + 1
            n_lead = self._src[FORCING_VARS[0]].shape[1]
            if lead.min() < 1 or lead.max() > n_lead:
                raise SystemExit(
                    f"requested days fall outside the forecast: lead "
                    f"{lead.min()}..{lead.max()} against 1..{n_lead}. One SEAS5 "
                    f"initialisation reaches {n_lead} days, not a water year.")
            coarse = {v: self._src[v][member, lead - 1] for v in FORCING_VARS}
        out = compose(coarse, self.delta, dates.month.values, self.cmap,
                      apply_delta=self.delta is not None)
        if getattr(self, "_nldas", None) is not None:
            # Arm Dp: keep SEAS5's precipitation, take temperature from NLDAS-2.
            idx = self._nldas_time.get_indexer(dates)
            if (idx < 0).any():
                bad = [str(d.date()) for d, i in zip(dates, idx) if i < 0]
                raise SystemExit(f"arm Dp: no NLDAS-2 temperature for {bad}")
            for v in ("tmin", "tmax"):
                out[v] = np.asarray(self._nldas[v].values[idx], dtype=np.float32)
        return out
