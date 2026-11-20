"""
Download real groundwater observations for the CESER domain, WY2003-2006.

This replaces the UIUC/ISWS export, which could not be used at all: it carried
no coordinates for any well, its records started in 2014 (no overlap with the
simulation), and its depths were feet-below-surface in a deep confined aquifer.
USGS NWIS solves all three -- every site has a published latitude/longitude, the
daily record covers the simulation period, and the network includes shallow
water-table wells whose depths are in ParFlow's range.

What it pulls: NWIS parameter 72019, "depth to water level, feet below land
surface", daily values (statistic 00003), for every groundwater site inside the
domain bounding box with data overlapping WY2003-2006. Depths are converted to
metres. Each site's lat/lon is converted to the conus1 grid and then to a local
(y, x) in our 637x665 domain.

A caution that shapes what these data can be used for: many NWIS "depth to
water" wells are screened in DEEP CONFINED aquifers and report potentiometric
head, not the shallow water table ParFlow-CONUS1 resolves. In this domain 16 of
the 80 wells that return data sit below 30 m, one of them at 53 m -- those are
not comparable to a ParFlow water table whose domain median is ~1 m. The
`--max-median-depth` filter (default 10 m) marks which wells are comparable;
all wells are saved, with the flag, so the choice stays visible rather than
silently applied.

Outputs into data/usgs_wells/:
    wells.json        -- site metadata, local (y, x), depth statistics, usable flag
    observations.csv  -- site_no, date, depth_to_water_m

Usage:
    python fetch_usgs_wells.py
    python fetch_usgs_wells.py --max-median-depth 15
"""
import argparse
import csv
import json
import os
import subprocess
from collections import defaultdict

import numpy as np
import xarray as xr

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DIR = os.path.join(DATA_DIR, "usgs_wells")
# every state the domain bounding box touches
STATES = ["il", "ia", "mo", "in", "wi", "mn", "ky", "mi", "oh"]
PARAM = "72019"          # depth to water level, feet below land surface
STAT = "00003"           # daily mean
START, END = "2002-10-01", "2006-09-30"   # WY2003 through WY2006
FT_TO_M = 0.3048
SITE_URL = ("https://waterservices.usgs.gov/nwis/site/?format=rdb&stateCd={st}"
            f"&siteType=GW&outputDataTypeCd=dv&parameterCd={PARAM}&siteStatus=all")
DV_URL = ("https://waterservices.usgs.gov/nwis/dv/?format=rdb&sites={sites}"
          f"&startDT={START}&endDT={END}&parameterCd={PARAM}&siteStatus=all")


def curl(url, timeout=300, retries=3):
    """A silently-empty response is indistinguishable from 'no data', so retry
    before accepting emptiness -- two chunks came back blank on the first run
    and their wells vanished without any error."""
    for attempt in range(retries):
        r = subprocess.run(["curl", "-s", "--max-time", str(timeout), url],
                           capture_output=True, text=True)
        if r.stdout.strip():
            return r.stdout
        print(f"    empty response, retry {attempt + 1}/{retries}")
    return ""


def parse_rdb(text):
    """NWIS RDB: '#' comments, a header row, a format row, then data.

    A multi-site response concatenates several such blocks, each with its own
    header, so the header must be re-latched whenever one appears -- otherwise
    the second block's header row is parsed as data and yields a row whose
    site_no is literally "site_no".
    """
    out, hdr = [], None
    for l in text.splitlines():
        if not l or l.startswith("#"):
            continue
        v = l.split("\t")
        if v[0] == "agency_cd":       # a (new) header block
            hdr = v
            continue
        if hdr is None:
            continue
        if all(c and c[-1] in "sdn" and c[:-1].isdigit() for c in v[:2]):
            continue                   # the format row (e.g. '5s', '15s', '20d')
        if len(v) == len(hdr):
            out.append(dict(zip(hdr, v)))
    return out


def domain_bbox(static):
    from hf_hydrodata import grid
    gb = static.attrs["grid_bounds"]
    if isinstance(gb, str):
        gb = [int(v) for v in gb.strip("[]").replace(",", " ").split()]
    imin, jmin, imax, jmax = (int(v) for v in gb)
    corners = [grid.to_latlon("conus1", i, j)
               for i, j in [(imin, jmin), (imax, jmin), (imin, jmax), (imax, jmax)]]
    lats = [c[0] for c in corners]
    lons = [c[1] for c in corners]
    return (min(lons), min(lats), max(lons), max(lats)), (imin, jmin, imax, jmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-median-depth", type=float, default=10.0,
                    help="metres; wells deeper than this are flagged not comparable "
                         "to the ParFlow water table (default 10)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    static = xr.open_dataset(os.path.join(DATA_DIR, "parflow_raw", "static_conditions.nc"))
    (lon0, lat0, lon1, lat1), (imin, jmin, imax, jmax) = domain_bbox(static)
    H, W = static.sizes["y"], static.sizes["x"]
    print(f"domain bbox lon [{lon0:.3f}, {lon1:.3f}]  lat [{lat0:.3f}, {lat1:.3f}]")

    sites = {}
    for st in STATES:
        rows = parse_rdb(curl(SITE_URL.format(st=st)))
        n_st = 0
        for r in rows:
            try:
                lat, lon = float(r["dec_lat_va"]), float(r["dec_long_va"])
            except (KeyError, ValueError):
                continue
            if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
                continue
            b, e = r.get("begin_date", ""), r.get("end_date", "")
            if not (b and e and b <= END and e >= START):
                continue
            s = sites.setdefault(r["site_no"], {
                "site_no": r["site_no"], "name": r.get("station_nm", ""),
                "state": st.upper(),   # which state inventory returned it; used for
                                       # human-readable figure titles
                "lat": lat, "lon": lon, "begin": b, "end": e,
                "alt_va": r.get("alt_va", ""), "huc_cd": r.get("huc_cd", "")})
            s["begin"], s["end"] = min(s["begin"], b), max(s["end"], e)
            n_st += 1
        print(f"  {st}: {n_st} candidate site-series in domain and period")
    print(f"unique candidate wells: {len(sites)}")

    obs = []
    ids = list(sites)
    CHUNK = 20      # 4 years of daily values x 40 sites overran the service
    for k in range(0, len(ids), CHUNK):
        chunk = ids[k:k + CHUNK]
        before = len(obs)
        for r in parse_rdb(curl(DV_URL.format(sites=",".join(chunk)))):
            col = [c for c in r if c.endswith(f"_{PARAM}_{STAT}")]
            if not col:
                continue
            raw = r[col[0]]
            if raw in ("", "Ice", "Eqp", "***"):
                continue
            try:
                obs.append((r["site_no"], r["datetime"], float(raw) * FT_TO_M))
            except ValueError:
                continue
        got = len(obs) - before
        print(f"  fetched {min(k + CHUNK, len(ids))}/{len(ids)} wells, "
              f"+{got} values ({len(obs)} total)"
              + ("   <-- WARNING: chunk returned nothing" if got == 0 else ""))

    from hf_hydrodata import grid
    per = defaultdict(list)
    for sid, _, v in obs:
        per[sid].append(v)

    kept = {}
    for sid, vals in per.items():
        a = np.asarray(vals)
        s = dict(sites[sid])
        i, j = grid.from_latlon("conus1", s["lat"], s["lon"])
        y, x = int(round(j)) - jmin, int(round(i)) - imin
        if not (0 <= y < H and 0 <= x < W):
            continue          # rounds outside the clipped domain
        s.update(y=y, x=x, n_days=int(a.size),
                 median_depth_m=float(np.median(a)),
                 min_depth_m=float(a.min()), max_depth_m=float(a.max()),
                 seasonal_range_m=float(np.percentile(a, 95) - np.percentile(a, 5)),
                 comparable=bool(np.median(a) <= args.max_median_depth))
        kept[sid] = s

    with open(os.path.join(OUT_DIR, "wells.json"), "w") as f:
        json.dump({"bbox_lon_lat": [lon0, lat0, lon1, lat1],
                   "max_median_depth_m": args.max_median_depth,
                   "parameter": f"USGS NWIS {PARAM} daily mean depth to water (ft) -> m",
                   "period": [START, END], "wells": kept}, f, indent=1)
    with open(os.path.join(OUT_DIR, "observations.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["site_no", "date", "depth_to_water_m"])
        for sid, d, v in obs:
            if sid in kept:
                w.writerow([sid, d, f"{v:.4f}"])

    comp = [s for s in kept.values() if s["comparable"]]
    print(f"\nwells with data inside the grid : {len(kept)}")
    print(f"comparable (median <= {args.max_median_depth:g} m) : {len(comp)}")
    print(f"deeper than 30 m (confined, not comparable): "
          f"{len([s for s in kept.values() if s['median_depth_m'] > 30])}")
    comp.sort(key=lambda s: -s["n_days"])
    print(f"\n{'site_no':18s} {'y,x':>10s} {'n':>5s} {'med_m':>6s} {'range_m':>8s}  name")
    for s in comp[:15]:
        print(f"{s['site_no']:18s} {s['y']:4d},{s['x']:<5d} {s['n_days']:5d} "
              f"{s['median_depth_m']:6.2f} {s['seasonal_range_m']:8.2f}  {s['name'][:34]}")
    print(f"\nwrote {OUT_DIR}/wells.json and observations.csv")


if __name__ == "__main__":
    main()
