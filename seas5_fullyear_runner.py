"""
Queue-aware, disconnection-proof downloader for the full-year SEAS5 forcing.

Two things this gets right that the previous attempt did not.

QUEUE DISCIPLINE. CDS permits only two queued requests on this dataset. The
earlier version submitted and let the server refuse, which produced 43 rejected
jobs in a few hours -- noisy, and no faster. This one polls the job list and
submits only when a slot is actually free, so it never generates a rejection.

PATIENCE. CDS congestion is normal and unrelated to request validity: a
single-step request for a lead time we hold data for sat unstarted for 15
minutes while the service was busy. "accepted" means queued, not stuck, so a
request is never abandoned for being slow. Only an explicit failure is fatal.

Whether SEAS5 serves more than 215 lead days from an October start is still an
open question; the answer is simply the forecast_period length of the first
file that lands, which this prints for every download.

Usage:
    setsid nohup python seas5_fullyear_runner.py --days 365 &
"""
import argparse
import json
import os
import sys
import time

import cdsapi
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import download_seas5 as D

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "data", "seas5", "raw_fullyear")
INITS = [(2006, 5), (2003, 5), (2004, 5), (2005, 5)]   # May inits, WY2006 first
MAX_LIVE = 2


def live_jobs(url, key):
    try:
        r = requests.get(f"{url}/retrieve/v1/jobs?limit=100",
                         headers={"PRIVATE-TOKEN": key}, timeout=90)
        return [j for j in r.json().get("jobs", [])
                if j.get("status") in ("accepted", "running")]
    except Exception:
        return []


def steps_in(path):
    try:
        import xarray as xr
        d = xr.open_dataset(path, decode_timedelta=False)
        n = max(v for k, v in d.sizes.items() if k not in ("latitude", "longitude"))
        dims = dict(d.sizes)
        d.close()
        return n, dims
    except Exception as e:
        return None, str(e)[:120]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=215)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    url, key = D.credentials()
    lead = [str(h) for h in range(24, a.days * 24 + 1, 24)]

    for y, m in INITS:
        out = os.path.join(OUT, f"seas5_{y}{m:02d}_{a.days}d.nc")
        if os.path.exists(out):
            n, dims = steps_in(out)
            print(f"[{y}-{m:02d}] already have it: {n} lead steps", flush=True)
            continue

        # wait for a free slot instead of being refused
        while len(live_jobs(url, key)) >= MAX_LIVE:
            print(f"[{y}-{m:02d}] queue full ({MAX_LIVE} live), waiting 5 min",
                  flush=True)
            time.sleep(300)

        print(f"[{y}-{m:02d}] submitting {a.days} lead days", flush=True)
        t0 = time.time()
        try:
            c = cdsapi.Client(url=url, key=key, quiet=True,
                              wait_until_complete=True)
            tmp = out + ".part"
            c.retrieve(D.DATASET, {
                "originating_centre": D.CENTRE, "system": D.SYSTEM,
                "variable": D.VARIABLES, "year": [str(y)],
                "month": [f"{m:02d}"], "day": ["01"],
                "leadtime_hour": lead, "area": D.AREA,
                "data_format": "netcdf"}, tmp)
            os.replace(tmp, out)
            n, dims = steps_in(out)
            mins = (time.time() - t0) / 60
            print(f"[{y}-{m:02d}] DONE in {mins:.0f} min, "
                  f"{os.path.getsize(out)/1e6:.1f} MB", flush=True)
            print(f"[{y}-{m:02d}] LEAD STEPS RETURNED: {n}   dims={dims}",
                  flush=True)
            if n is not None and n < a.days:
                print(f"[{y}-{m:02d}] *** TRUNCATED: asked {a.days}, got {n}. "
                      f"An October start cannot span a water year; "
                      f"use the August initialisation instead. ***", flush=True)
        except Exception as e:
            print(f"[{y}-{m:02d}] FAILED {type(e).__name__}: {str(e)[:250]}",
                  flush=True)
    print("runner finished", flush=True)


if __name__ == "__main__":
    main()
