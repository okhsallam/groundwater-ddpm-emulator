"""Download the ParFlow-CONUS1 data needed to replicate Bennett et al. (2024)
"Spatio-Temporal Machine Learning for Regional to Continental Scale
Terrestrial Hydrology" (the FSTR paper), clipped to the CESER domain, for
training our own diffusion model as a groundwater forecaster.

Scope (decided 2026-09-14):
- Domain: CESER AOI only (~1km conus1 grid), not full CONUS. Same
  grid_bounds used in ../ceser_ddpm_GloFAS/explore_parflow_data.
- Water years 2003-2006 (2002-10-01 -> 2006-09-30) -- the only period
  conus1_baseline_mod covers, and exactly the paper's train (2003-2005) /
  test (2006) split.
- Inputs/outputs matched to the paper's Section 2.1/2.2 as closely as the
  hf_hydrodata catalog allows:
    * Target: 5-layer pressure_head (the paper's fundamental target), plus
      the ready-made water_table_depth and (single-layer) soil_moisture the
      paper derives from it -- kept for direct comparison, not re-derived.
    * Forcing ("action" tensors): daily min/max air_temp, total
      precipitation, snowmelt, bulk evapotranspiration.
      - snowmelt is NOT a direct hf_hydrodata variable. Derived here as
        max(0, SWE[t-1] - SWE[t]) from daily end-of-day SWE (decided with
        user 2026-09-14). The very first day of WY2003 has no prior SWE
        (conus1_baseline_mod starts exactly there) and is left NaN.
    * Static parameters: porosity, permeability, van_genuchten_alpha,
      van_genuchten_n (all 5-layer), topographic_index, elevation.
      - distance-to-nearest-stream is dropped (decided with user
        2026-09-14): not a direct hf_hydrodata variable, and not worth a
        custom distance-transform derivation for this pass.
- float64, as returned by the API (no downcast).
- Output: one NetCDF per (group, water-year) under
  data/parflow_raw/ -- resumable, since each file is written only after a
  full successful pull for that year.

Run with `nohup` / in the background -- this is network-bound and pulls
several years of daily gridded data.
"""
from pathlib import Path
import datetime as dt
import json
import time

import numpy as np
import xarray as xr
import hf_hydrodata as hf

GRID = "conus1"
# CESER domain, see ../ceser_ddpm_GloFAS/explore_parflow_data/README.md
GRID_BOUNDS = [2053, 565, 2718, 1202]
LATLNG_BOUNDS = [-94.05003238261448, 38.62888854853616, -85.5671330592881, 43.968394539679124]
OUTDIR = Path(__file__).parent / "data" / "parflow_raw"
OUTDIR.mkdir(parents=True, exist_ok=True)

WATER_YEARS = {
    2003: ("2002-10-01", "2003-09-30"),
    2004: ("2003-10-01", "2004-09-30"),
    2005: ("2004-10-01", "2005-09-30"),
    2006: ("2005-10-01", "2006-09-30"),
}

TARGET_3D_VAR = "pressure_head"  # dataset=conus1_baseline_mod, daily, 5 z-layers, units m
# z-layer ordering confirmed 2026-09-14 by cross-checking pressure_head sign
# against water_table_depth: z is ParFlow's bottom-up index, so z=0 is the
# deepest 100m layer and z=4 is the shallowest 0.1m surface layer -- the
# REVERSE of the paper's surface-to-bottom listing (0.1, 0.3, 0.6, 1, 100m).
# Same ordering applies to the static 3D vars below (permeability, porosity,
# van_genuchten_alpha/n).
TARGET_2D_VARS = {  # dataset=conus1_baseline_mod, daily
    "water_table_depth": "m",
    "soil_moisture": "kg/m2",  # single layer despite has_z flag
}
FORCING_VARS_DIRECT = {  # pulled directly, then snowmelt is derived below
    ("air_temp", "min"): ("tmin", "K"),
    ("air_temp", "max"): ("tmax", "K"),
    ("precipitation", None): ("precipitation", "mm"),
}
ET_VAR = "evapotranspiration"  # dataset=conus1_baseline_mod, daily sum, mm/d ("et")
SWE_VAR = "swe"  # dataset=conus1_baseline_mod, daily eod, mm -- used only to derive snowmelt

STATIC_2D_VARS = {"elevation": "m", "topographic_index": "-"}
STATIC_3D_VARS = {  # 5 z-layers
    "permeability": "m/h", "porosity": "-",
    "van_genuchten_alpha": "1/m", "van_genuchten_n": "-",
}


def date_range(start, end_inclusive):
    d = dt.date.fromisoformat(start)
    end = dt.date.fromisoformat(end_inclusive)
    out = []
    while d <= end:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def pull_var(dataset, variable, date_start, date_end_excl, z=None, **kw):
    kwargs = dict(dataset=dataset, variable=variable, grid=GRID, grid_bounds=GRID_BOUNDS,
                  date_start=date_start, date_end=date_end_excl, **kw)
    if z is not None:
        kwargs["z"] = z
    return hf.get_gridded_data(**kwargs)


def pull_var_chunked(dataset, variable, date_start, date_end_excl, chunk_days, **kw):
    """Like pull_var, but splits into <=chunk_days windows and concatenates
    along time, to stay under the API's 2 GB-per-request cap."""
    d = dt.date.fromisoformat(date_start)
    end = dt.date.fromisoformat(date_end_excl)
    chunks = []
    while d < end:
        chunk_end = min(d + dt.timedelta(days=chunk_days), end)
        chunks.append(pull_var(dataset, variable, d.isoformat(), chunk_end.isoformat(), **kw))
        d = chunk_end
    return np.concatenate(chunks, axis=0)


def squeeze_singleton_z(arr, varname):
    if arr.ndim == 4:
        assert arr.shape[1] == 1, f"{varname}: expected singleton z, got shape {arr.shape}"
        arr = arr[:, 0]
    return arr


def download_target2d(wy, y_start, y_end):
    outfile = OUTDIR / f"target2d_wy{wy}.nc"
    if outfile.exists():
        print(f"  [skip] {outfile.name} already exists")
        return
    date_end_excl = (dt.date.fromisoformat(y_end) + dt.timedelta(days=1)).isoformat()
    dates = date_range(y_start, y_end)

    data_vars = {}
    for var, units in TARGET_2D_VARS.items():
        t0 = time.time()
        arr = pull_var_chunked("conus1_baseline_mod", var, y_start, date_end_excl,
                                chunk_days=90, temporal_resolution="daily")
        arr = squeeze_singleton_z(arr, var)
        assert arr.shape[0] == len(dates), f"{var}: got {arr.shape[0]} days, expected {len(dates)}"
        data_vars[var] = (("time", "y", "x"), arr, {"units": units})
        print(f"    target2d/{var} wy{wy}: shape={arr.shape} ({time.time()-t0:.0f}s)")

    coords = {"time": [np.datetime64(d) for d in dates]}
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update(dataset="conus1_baseline_mod", grid=GRID, grid_bounds=str(GRID_BOUNDS), source="hf_hydrodata")
    tmp = outfile.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp)
    tmp.rename(outfile)
    print(f"  [done] wrote {outfile.name} ({outfile.stat().st_size / 1e9:.2f} GB)")


def download_target3d(wy, y_start, y_end):
    outfile = OUTDIR / f"target3d_wy{wy}.nc"
    if outfile.exists():
        print(f"  [skip] {outfile.name} already exists")
        return
    date_end_excl = (dt.date.fromisoformat(y_end) + dt.timedelta(days=1)).isoformat()
    dates = date_range(y_start, y_end)

    t0 = time.time()
    # 5-layer pressure_head for a full water year is ~6.2 GB, over the API's
    # 2 GB/request cap -- pull in <=90-day chunks and concatenate.
    arr = pull_var_chunked("conus1_baseline_mod", TARGET_3D_VAR, y_start, date_end_excl,
                            chunk_days=90, temporal_resolution="daily")
    assert arr.shape[0] == len(dates), f"{TARGET_3D_VAR}: got {arr.shape[0]} days, expected {len(dates)}"
    print(f"    target3d/{TARGET_3D_VAR} wy{wy}: shape={arr.shape} ({time.time()-t0:.0f}s)")

    ds = xr.Dataset(
        {TARGET_3D_VAR: (("time", "z", "y", "x"), arr, {"units": "m"})},
        coords={"time": [np.datetime64(d) for d in dates], "z": np.arange(arr.shape[1])},
    )
    ds.attrs.update(dataset="conus1_baseline_mod", grid=GRID, grid_bounds=str(GRID_BOUNDS), source="hf_hydrodata")
    tmp = outfile.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp)
    tmp.rename(outfile)
    print(f"  [done] wrote {outfile.name} ({outfile.stat().st_size / 1e9:.2f} GB)")


def download_forcing(wy, y_start, y_end):
    outfile = OUTDIR / f"forcing_wy{wy}.nc"
    if outfile.exists():
        print(f"  [skip] {outfile.name} already exists")
        return
    date_end_excl = (dt.date.fromisoformat(y_end) + dt.timedelta(days=1)).isoformat()
    dates = date_range(y_start, y_end)

    data_vars = {}

    # tmin, tmax, precipitation -- direct pulls from NLDAS2
    for (var, agg), (outname, units) in FORCING_VARS_DIRECT.items():
        t0 = time.time()
        kw = {"temporal_resolution": "daily"}
        if agg is not None:
            kw["aggregation"] = agg
        arr = pull_var_chunked("NLDAS2", var, y_start, date_end_excl, chunk_days=90, **kw)
        arr = squeeze_singleton_z(arr, outname)
        assert arr.shape[0] == len(dates), f"{outname}: got {arr.shape[0]} days, expected {len(dates)}"
        data_vars[outname] = (("time", "y", "x"), arr, {"units": units})
        print(f"    forcing/{outname} wy{wy}: shape={arr.shape} ({time.time()-t0:.0f}s)")

    # bulk evapotranspiration -- direct pull from conus1_baseline_mod (daily sum, mm/d)
    t0 = time.time()
    et = pull_var_chunked("conus1_baseline_mod", ET_VAR, y_start, date_end_excl,
                           chunk_days=90, temporal_resolution="daily")
    et = squeeze_singleton_z(et, ET_VAR)
    assert et.shape[0] == len(dates), f"{ET_VAR}: got {et.shape[0]} days, expected {len(dates)}"
    data_vars[ET_VAR] = (("time", "y", "x"), et, {"units": "mm/d"})
    print(f"    forcing/{ET_VAR} wy{wy}: shape={et.shape} ({time.time()-t0:.0f}s)")

    # snowmelt -- derived from SWE[t-1] - SWE[t], clipped at 0. Need one extra
    # day before y_start to compute melt on the first day of the water year.
    t0 = time.time()
    prev_day = (dt.date.fromisoformat(y_start) - dt.timedelta(days=1)).isoformat()
    has_prev_day = prev_day >= "2002-10-01"  # conus1_baseline_mod starts here
    swe_start = prev_day if has_prev_day else y_start
    swe = pull_var_chunked("conus1_baseline_mod", SWE_VAR, swe_start, date_end_excl,
                            chunk_days=90, temporal_resolution="daily", aggregation="eod")
    swe = squeeze_singleton_z(swe, SWE_VAR)
    if has_prev_day:
        melt = np.clip(swe[:-1] - swe[1:], a_min=0, a_max=None)
    else:
        melt = np.full_like(swe, np.nan)
        melt[1:] = np.clip(swe[:-1] - swe[1:], a_min=0, a_max=None)
    assert melt.shape[0] == len(dates), f"snowmelt: got {melt.shape[0]} days, expected {len(dates)}"
    data_vars["snowmelt"] = (("time", "y", "x"), melt, {
        "units": "mm/d",
        "note": "derived as max(0, SWE[t-1]-SWE[t]); NaN on first day of WY2003 (no prior SWE available)",
    })
    print(f"    forcing/snowmelt wy{wy}: shape={melt.shape} ({time.time()-t0:.0f}s)")

    coords = {"time": [np.datetime64(d) for d in dates]}
    ds = xr.Dataset(data_vars, coords=coords)
    ds.attrs.update(grid=GRID, grid_bounds=str(GRID_BOUNDS), source="hf_hydrodata",
                     forcing_datasets="NLDAS2 (tmin/tmax/precipitation); conus1_baseline_mod (et, snowmelt derivation)")
    tmp = outfile.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp)
    tmp.rename(outfile)
    print(f"  [done] wrote {outfile.name} ({outfile.stat().st_size / 1e9:.2f} GB)")


def download_static():
    outfile = OUTDIR / "static_conditions.nc"
    if outfile.exists():
        print(f"[skip] {outfile.name} already exists")
        return
    data_vars = {}
    for var, units in STATIC_2D_VARS.items():
        arr = hf.get_gridded_data(dataset="conus1_domain", variable=var, grid=GRID, grid_bounds=GRID_BOUNDS)
        data_vars[var] = (("y", "x"), arr, {"units": units})
        print(f"  static/{var}: shape={arr.shape}")
    for var, units in STATIC_3D_VARS.items():
        arr = hf.get_gridded_data(dataset="conus1_domain", variable=var, grid=GRID, grid_bounds=GRID_BOUNDS)
        data_vars[var] = (("z", "y", "x"), arr, {"units": units})
        print(f"  static/{var}: shape={arr.shape}")
    ds = xr.Dataset(data_vars, coords={"z": np.arange(5)})
    ds.attrs.update(dataset="conus1_domain", grid=GRID, grid_bounds=str(GRID_BOUNDS), source="hf_hydrodata")
    tmp = outfile.with_suffix(".nc.tmp")
    ds.to_netcdf(tmp)
    tmp.rename(outfile)
    print(f"[done] wrote {outfile.name} ({outfile.stat().st_size / 1e9:.2f} GB)")


def main():
    meta = {
        "domain": "CESER AOI (subset of ParFlow-CONUS1)",
        "grid": GRID, "grid_bounds_ij": GRID_BOUNDS, "latlng_bounds": LATLNG_BOUNDS,
        "water_years": {str(k): v for k, v in WATER_YEARS.items()},
        "paper": "Bennett et al. 2024, doi:10.1029/2023MS004095 (FSTR)",
        "train_water_years": [2003, 2004, 2005], "test_water_year": 2006,
        "target_3d_var": TARGET_3D_VAR, "target_2d_vars": TARGET_2D_VARS,
        "forcing_vars": ["tmin", "tmax", "precipitation", "snowmelt", ET_VAR],
        "static_2d_vars": STATIC_2D_VARS, "static_3d_vars": STATIC_3D_VARS,
        "deviations_from_paper": {
            "snowmelt": "derived as max(0, SWE[t-1]-SWE[t]) from daily SWE; not a direct hf_hydrodata variable",
            "distance_to_stream": "dropped entirely; not a direct hf_hydrodata variable and not derived this pass",
        },
        "source": "hf_hydrodata (https://github.com/hydroframe/hf_hydrodata), conus1_baseline_mod + conus1_domain + NLDAS2",
        "citation": "O'Neill et al. 2021, doi:10.5194/gmd-14-7223-2021 (conus1_baseline_mod)",
    }
    (OUTDIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    print("=== static conditions ===")
    download_static()

    for wy, (y_start, y_end) in WATER_YEARS.items():
        print(f"=== water year {wy} ({y_start} -> {y_end}) ===")
        print(" -- target (pressure_head, 3D daily) --")
        download_target3d(wy, y_start, y_end)
        print(" -- target (water_table_depth, soil_moisture, 2D daily) --")
        download_target2d(wy, y_start, y_end)
        print(" -- forcing (tmin, tmax, precipitation, snowmelt, et, daily) --")
        download_forcing(wy, y_start, y_end)

    print("\nAll done. Files in", OUTDIR)


if __name__ == "__main__":
    main()
