"""
Download ECMWF SEAS5 (C3S system 5) seasonal forecasts over the CESER domain,
for driving the trained groundwater emulator with forecast rather than
reanalysis forcing.

WHY THIS IS A DROP-IN. The current model (ddpm_model/config_f15_fold1.yaml,
forcing_set: "nldas") conditions on exactly three forcing channels -- tmin,
tmax, precipitation -- all of them genuine atmospheric variables. Everything
else in the 69-channel conditioning is static terrain or the model's own
prediction. So substituting a seasonal forecast changes three arrays and
nothing else; no retraining, no re-tiling.

VERIFIED AGAINST THE CDS CATALOGUE (2026-10-08), not assumed:
  * originating_centre "ecmwf", system "5"
  * minimum/maximum_2m_temperature_in_the_last_24_hours and total_precipitation
    all exist at 24-hour leadtime steps
  * leadtime_hour runs 24 .. 5160 by 24  ->  exactly 215 forecast days
  * initialisations are the 1st of the month only (day: ['01'])
  * hindcast years for these variables: 1981-2022
  * the form exposes no `grid` key, so delivery is on the archive's standard
    1 deg lat/lon grid

215 days = 14 hops of the model's 15-day window = 210 forecast days. One
initialisation therefore covers roughly Oct-Apr, NOT a water year; covering a
full water year needs two separate forecasts (Oct 1 and Apr 1 inits).

TWO UNIT/CONVENTION TRAPS, both handled downstream in prepare_seas5_domain.py,
flagged here because they are the GloFAS-timestamp class of error:

  1. `total_precipitation` is ACCUMULATED FROM THE FORECAST START, in metres.
     Daily mm is (tp[lead] - tp[lead-1]) * 1000, with lead 24 h taken as-is.
     Treating the raw field as a daily total gives a monotonically increasing
     "precipitation" channel that grows to metres by month seven.
  2. mn2t24/mx2t24 at leadtime_hour H describe the 24 hours ENDING at H, so
     leadtime 24 is forecast day 1. Inits are at 00 UTC, so the windows are
     00-00 UTC days, which is the convention NLDAS-2 daily tmin/tmax uses.

Two sets of initialisations are downloaded:

  TARGET_INITS  the forecasts actually rolled out.
  CLIM_INITS    the same init month across the whole hindcast record, used to
                build the SEAS5 side of the quantile-mapping climatology in
                prepare_seas5_domain.py. The target years are EXCLUDED from it,
                so the bias correction never sees the year it corrects.

The domain subset is ~88 cells at 1 deg, so one init is a few MB and the whole
download is well under a gigabyte -- unlike GloFAS, size is not a constraint
here.

Credentials are never stored in this file. They are read, in order, from
  1. the CDSAPI_URL / CDSAPI_KEY environment variables
  2. ~/.cdsapirc
  3. .cdsapirc in this directory (keep it out of version control)
Note the endpoint differs from GloFAS: SEAS5 is on the CDS, not the EWDS.

Usage:
    export CDSAPI_URL="https://cds.climate.copernicus.eu/api"
    export CDSAPI_KEY="<your personal access token>"
    python download_seas5.py                 # target inits only
    python download_seas5.py --climatology   # the hindcast record as well
"""
import argparse
import os
import sys
import time
from pathlib import Path

import cdsapi

from seas5_forcing import QM_REFERENCE_WINDOW, qm_years

HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "seas5" / "raw"

DATASET = "seasonal-original-single-levels"
CENTRE, SYSTEM = "ecmwf", "5"
# All three forcing variables in one request. Precipitation is the one that
# matters most -- it is what drives recharge, and the experiment asked for is a
# precipitation forecast -- but the request cost is per field, the domain subset
# is ~88 cells, and a whole initialisation is a few MB either way. Fetching the
# temperatures at the same time costs essentially nothing and avoids a second
# round trip through the CDS queue if the full-forcing arm is ever wanted. Which
# variables are actually FED to the model is an inference-time choice (see the
# D / Dp arms in seas5_forcing.py), not a download-time one.
VARIABLES = ["minimum_2m_temperature_in_the_last_24_hours",
             "maximum_2m_temperature_in_the_last_24_hours",
             "total_precipitation"]
LEADTIME_HOURS = [str(h) for h in range(24, 5161, 24)]      # 215 days

# CESER domain is lat 38.24-44.37, lon -94.06 to -85.59. Buffered out to whole
# degrees plus one cell so that every conus1 cell has four 1-deg neighbours for
# bilinear interpolation, including at the corners.
AREA = [46, -96, 37, -84]                                   # N, W, S, E

# ONE FORECAST PER WATER YEAR, WHICH IS ALSO ONE PER FOLD.
#
# A water year runs 1 Oct (Y-1) to 30 Sep Y, so the 1 October initialisation of
# year Y-1 opens water year Y and its 215-day reach covers Oct->early May. That
# lines each forecast up with exactly the fold whose held-out year it verifies
# against, which is what makes this a cross-validated seasonal evaluation rather
# than four unrelated case studies:
#
#     init 2002-10-01  ->  WY2003  ->  fold 4 (trained 2004-2006)
#     init 2003-10-01  ->  WY2004  ->  fold 3 (trained 2003, 2005, 2006)
#     init 2004-10-01  ->  WY2005  ->  fold 2 (trained 2003, 2004, 2006)
#     init 2005-10-01  ->  WY2006  ->  fold 1 (trained 2003-2005)
#
# Each forecast is therefore scored by a model that never saw the year being
# forecast, in training or in normalisation.
TARGET_INITS = [(2002, 10), (2003, 10), (2004, 10), (2005, 10)]

# init year -> (water year it opens, fold whose test year that is)
INIT_TO_FOLD = {2002: (2003, 4), 2003: (2004, 3),
                2004: (2005, 2), 2005: (2006, 1)}

# The hindcast reaches 1981-2022 per the catalogue, but the climatology window
# comes from seas5_forcing.QM_REFERENCE_WINDOW -- the SINGLE definition shared
# with the ERA5 downloader and with prepare_seas5_domain.py. These were once two
# independent lists (1981-2022 here against 1993-2017 for ERA5) and the mismatch
# would have skipped the entire bias correction silently: stage 1 pairs every
# init with ERA5 days, so one init year lacking ERA5 makes the reference load
# return None. Fetching inits whose verifying ERA5 we do not hold buys nothing.
DEFAULT_URL = "https://cds.climate.copernicus.eu/api"


def credentials():
    url, key = os.environ.get("CDSAPI_URL"), os.environ.get("CDSAPI_KEY")
    if url and key:
        return url, key
    for rc in (Path.home() / ".cdsapirc", HERE / ".cdsapirc"):
        if rc.exists():
            kv = {}
            for line in rc.read_text().splitlines():
                if ":" in line:
                    k, v = line.split(":", 1)
                    kv[k.strip()] = v.strip()
            if "key" in kv:
                return kv.get("url", DEFAULT_URL), kv["key"]
    sys.exit("No CDS credentials. Set CDSAPI_URL and CDSAPI_KEY, or create "
             "~/.cdsapirc. See the module docstring.")


def climatology_inits(target_inits):
    """Every reference-window year for each target init month, targets excluded.

    Excluding them is what keeps the bias correction honest: the statistics used
    to correct the 2005-10 forecast are estimated without 2005 or 2006 in the
    sample, so no part of the period being forecast informs its own correction.
    """
    months = sorted({m for _, m in target_inits})
    drop = {y for y, _ in target_inits} | {y + 1 for y, _ in target_inits}
    return [(y, m) for m in months for y in qm_years(exclude=drop)]


def fetch(client, year, month):
    out = RAW / f"seas5_{year}{month:02d}.nc"
    if out.exists():
        return False
    tmp = out.with_suffix(".part")
    client.retrieve(DATASET, {
        "originating_centre": CENTRE,
        "system": SYSTEM,
        "variable": VARIABLES,
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": ["01"],
        "leadtime_hour": LEADTIME_HOURS,
        "area": AREA,
        "data_format": "netcdf",
    }, str(tmp))
    tmp.rename(out)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--climatology", action="store_true",
                    help="also fetch the full hindcast record for the target "
                         "init months (the bias-correction sample)")
    ap.add_argument("--years", default=None,
                    help="comma-separated hindcast init years to fetch instead "
                         "of the whole reference window. The calibration is "
                         "staged, so that a 12-year refit can be scored before "
                         "committing to the remaining years.")
    args = ap.parse_args()

    RAW.mkdir(parents=True, exist_ok=True)
    url, key = credentials()
    client = cdsapi.Client(url=url, key=key, quiet=True, wait_until_complete=True)

    inits = list(TARGET_INITS)
    if args.climatology:
        inits += climatology_inits(TARGET_INITS)
    if args.years:
        want = {int(y) for y in args.years.split(",")}
        inits = [(y, m) for y, m in inits if y in want]
    todo = [(y, m) for y, m in inits if not (RAW / f"seas5_{y}{m:02d}.nc").exists()]
    print(f"{len(inits)} initialisations requested, {len(todo)} still to fetch",
          flush=True)

    for i, (y, m) in enumerate(todo, 1):
        t0 = time.time()
        try:
            fetch(client, y, m)
        except Exception as e:
            print(f"[{i}/{len(todo)}] {y}-{m:02d}  FAILED  {type(e).__name__}: {e}",
                  flush=True)
            continue
        path = RAW / f"seas5_{y}{m:02d}.nc"
        print(f"[{i}/{len(todo)}] {y}-{m:02d}  "
              f"{path.stat().st_size / 1e6:.1f} MB  {time.time() - t0:.0f}s",
              flush=True)

    have = sorted(RAW.glob("seas5_*.nc"))
    print(f"\n{len(have)} files in {RAW}")


if __name__ == "__main__":
    main()
