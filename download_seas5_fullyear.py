"""
SEAS5 forcing for the four folds over a FULL water year.

A monthly SEAS5 initialisation is documented to reach 215 days, which is why
the existing data stops in late April. Covering 1 Oct to 30 Sep therefore needs
one of two things, and which one is available is an empirical question the CDS
constraints do not answer (they report 5160 h for every month, yet a longer
request is still accepted, and CDS accepts requests it later truncates):

  OCT  1 October initialisation, lead days 1-365.
       Preferred: the forecast starts exactly where the rollout starts, so
       lead time and forecast day coincide.

  AUG  1 August initialisation, lead days 1-427. The water year then occupies
       lead days 62-427, so the whole year is covered at 2-14 months lead.
       Fallback: only valid if SEAS5's extended 13-month range (issued from
       Feb/May/Aug/Nov) is served by this dataset.

Both are submitted together. Whichever returns a usable number of lead steps is
the one we use; the other is discarded. Submitting both costs only queue slots
and saves a full ~2 h queue cycle if OCT turns out to be truncated.

The ParFlow initial state is 1 October in either case. Only the FORCING origin
differs, so an August initialisation changes the lead time of the forecast, not
the experiment's initial condition.

Resumable: a file that already exists and carries enough lead steps is skipped.

Usage:
    python download_seas5_fullyear.py --mode oct   [--days 365]
    python download_seas5_fullyear.py --mode aug   [--days 427]
"""
import argparse
import os
import sys
import time

import cdsapi
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download_seas5 as D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "seas5", "raw_fullyear")
# init year -> the water year it is meant to force
OCT_INITS = [(2002, 10), (2003, 10), (2004, 10), (2005, 10)]
AUG_INITS = [(2002, 8), (2003, 8), (2004, 8), (2005, 8)]


def have_enough(path, need):
    if not os.path.exists(path):
        return False
    try:
        import xarray as xr
        d = xr.open_dataset(path, decode_timedelta=False)
        n = max(v for v in d.sizes.values())
        d.close()
        return n >= need
    except Exception:
        return False


def fetch(client, year, month, days):
    name = f"seas5_{year}{month:02d}_{days}d.nc"
    out = os.path.join(OUT, name)
    if have_enough(out, days):
        print(f"  {name}: already complete", flush=True)
        return out
    tmp = out + ".part"
    lead = [str(h) for h in range(24, days * 24 + 1, 24)]
    t0 = time.time()
    client.retrieve(D.DATASET, {
        "originating_centre": D.CENTRE,
        "system": D.SYSTEM,
        "variable": D.VARIABLES,          # tmin, tmax, total precipitation
        "year": [str(year)],
        "month": [f"{month:02d}"],
        "day": ["01"],
        "leadtime_hour": lead,            # all 25 hindcast members come with it
        "area": D.AREA,
        "data_format": "netcdf",
    }, tmp)
    os.replace(tmp, out)
    print(f"  {name}: {os.path.getsize(out)/1e6:.1f} MB in {time.time()-t0:.0f}s",
          flush=True)
    return out


def report(path):
    import xarray as xr
    d = xr.open_dataset(path, decode_timedelta=False)
    print(f"    -> dims {dict(d.sizes)}", flush=True)
    d.close()


def adopt_probe():
    """The standalone 365-day probe requested exactly the WY2006 file. Reuse it
    rather than queueing a duplicate against a two-slot limit."""
    src = os.path.join(os.path.dirname(OUT), "raw365", "seas5_200510_365d.nc")
    dst = os.path.join(OUT, "seas5_200510_365d.nc")
    if os.path.exists(src) and not os.path.exists(dst):
        os.makedirs(OUT, exist_ok=True)
        os.replace(src, dst)
        print(f"  adopted the probe download -> {os.path.basename(dst)}",
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("oct", "aug"), required=True)
    ap.add_argument("--days", type=int, default=None)
    a = ap.parse_args()
    days = a.days or (365 if a.mode == "oct" else 427)
    inits = OCT_INITS if a.mode == "oct" else AUG_INITS
    os.makedirs(OUT, exist_ok=True)
    adopt_probe()
    url, key = D.credentials()
    client = cdsapi.Client(url=url, key=key, quiet=True, wait_until_complete=True)
    print(f"{a.mode.upper()}: {len(inits)} initialisations x {days} lead days",
          flush=True)

    # CDS allows only TWO queued requests per dataset: submitting four at once
    # got three rejected with "Maximum number queued requests for ...". So the
    # pool is two wide, and a worker that is refused backs off and retries
    # rather than giving up, because a refusal means "come back later", not
    # "this request is invalid".
    def one(ym):
        y, m = ym
        for attempt in range(40):
            try:
                c = cdsapi.Client(url=url, key=key, quiet=True,
                                  wait_until_complete=True)
                p = fetch(c, y, m, days)
                report(p)
                return True
            except Exception as e:
                msg = str(e)
                if "Maximum number queued" in msg or "rejected" in msg:
                    wait = 120 + 60 * min(attempt, 8)
                    print(f"  {y}-{m:02d} queue full, retry in {wait}s "
                          f"(attempt {attempt+1})", flush=True)
                    time.sleep(wait)
                    continue
                print(f"  {y}-{m:02d} FAILED {type(e).__name__}: "
                      f"{msg[:200]}", flush=True)
                return False
        print(f"  {y}-{m:02d} gave up after repeated queue refusals", flush=True)
        return False

    with ThreadPoolExecutor(2) as ex:          # two slots, the CDS limit
        ok = list(ex.map(one, inits))
    print(f"  {sum(ok)}/{len(inits)} succeeded", flush=True)
    print("done ->", OUT, flush=True)


if __name__ == "__main__":
    main()
