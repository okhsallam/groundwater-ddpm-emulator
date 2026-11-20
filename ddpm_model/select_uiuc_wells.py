"""
Choose reference wells from the UIUC/ISWS unconfined network, 2003-2006.

WHY THIS NETWORK IS USABLE WHERE THE EARLIER UIUC DATA WAS NOT. The project
previously recorded UIUC wells as unusable: no coordinates, records starting in
2014, and screens in a deep confined aquifer. This file
(DOE_GW_OBV_unconfined_2003_2006_depth.csv) resolves all three -- it carries WGS84
coordinates, spans 2003-01-01 to 2006-12-31, and every record is flagged
UNCONFINED. It is therefore a genuinely new observational constraint rather than
a re-reading of data already rejected.

It is also a better constraint than the USGS set for this purpose: seven wells
carry 800-1700 daily observations across the whole four-year period, where the
USGS picks were chosen partly for having any coverage at all.

SELECTION, fixed before any model output is consulted, exactly as
select_reference_wells.py did for USGS:
  * inside the domain and within ~2 km of a model cell centre;
  * at least MIN_DAYS observations, spread over all four water years;
  * median depth inside ParFlow's representable range;
  * spatially separated, so the panel does not show one aquifer four times.
No model field enters the choice.

Output: data/reference_wells_uiuc.json, same schema as reference_wells.json so
the existing plotting code can consume either.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CSV = os.path.join(ROOT, "data", "UIUC", "DOE_GW_OBV_unconfined_2003_2006_depth.csv")
OUT = os.path.join(ROOT, "data", "reference_wells_uiuc.json")
MIN_DAYS = 300
MIN_SEP_CELLS = 40
MAX_MEDIAN_DEPTH = 20.0          # ParFlow's range here is 0-53 m; 20 is generous
WATER_YEARS = [2003, 2004, 2005, 2006]


def water_year(ts):
    return ts.year + (ts.month >= 10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--out", default=OUT)
    # The ISWS asked for their whole network to be evaluated, not a thinned
    # sample of it. --all keeps every well the model grid can actually be
    # sampled at and drops only the three screening rules that exist to pick a
    # SMALL representative panel: the minimum record length, the requirement
    # that all four water years be covered, and the spatial thinning. The
    # in-domain test is NOT dropped, because a well outside the domain has no
    # model cell to compare against; neither is the depth ceiling.
    ap.add_argument("--all", action="store_true",
                    help="keep every in-domain well, unthinned")
    args = ap.parse_args()

    import sys
    sys.path.insert(0, ROOT)
    from seas5_forcing import target_latlon
    lat, lon = target_latlon()
    ny, nx = lat.shape
    tree = cKDTree(np.column_stack([lat.ravel(), lon.ravel()]))

    d = pd.read_csv(CSV, parse_dates=["date_time"])
    d = d.dropna(subset=["depth_to_water_m", "latitude_wgs84", "longitude_wgs84"])
    d["wy"] = d["date_time"].map(water_year)

    rows = []
    for pid, g in d.groupby("p_number"):
        la, lo = float(g.latitude_wgs84.iloc[0]), float(g.longitude_wgs84.iloc[0])
        dist, idx = tree.query([la, lo])
        if dist > 0.02:                      # ~2 km; outside the domain
            continue
        y, x = divmod(int(idx), nx)
        per_wy = {int(w): int((g.wy == w).sum()) for w in WATER_YEARS}
        rows.append(dict(p_number=int(pid), name=str(g.well_name.iloc[0]),
                         aq_class=str(g.aq_class.iloc[0]),
                         lat=la, lon=lo, y=y, x=x,
                         n_days=int(len(g)),
                         days_per_water_year=per_wy,
                         covers_all_years=all(v > 0 for v in per_wy.values()),
                         median_depth_m=float(g.depth_to_water_m.median()),
                         min_depth_m=float(g.depth_to_water_m.min()),
                         max_depth_m=float(g.depth_to_water_m.max()),
                         seasonal_range_m=float(g.depth_to_water_m.max()
                                                - g.depth_to_water_m.min())))
    if args.all:
        cand = [r for r in rows if r["median_depth_m"] <= MAX_MEDIAN_DEPTH]
        print(f"  {len(rows)} wells on the model grid; {len(cand)} within the "
              f"{MAX_MEDIAN_DEPTH} m depth ceiling (no coverage or spacing "
              f"filter applied)")
    else:
        cand = [r for r in rows
                if r["n_days"] >= MIN_DAYS and r["covers_all_years"]
                and r["median_depth_m"] <= MAX_MEDIAN_DEPTH]
        print(f"  {len(rows)} wells on the model grid; {len(cand)} meet coverage "
              f"(>= {MIN_DAYS} obs, all four water years) and depth criteria")

    # Depth order throughout: the figures then read as a gradient from the
    # shallow unconfined wells the emulator represents best to the deep ones it
    # does not, instead of an arbitrary sequence.
    cand.sort(key=lambda r: r["median_depth_m"])
    if args.all:
        picks = cand
    else:
        # Spatial separation, so the panel is not one aquifer shown repeatedly.
        picks = []
        for r in cand:
            if all((r["y"]-p["y"])**2 + (r["x"]-p["x"])**2 >= MIN_SEP_CELLS**2
                   for p in picks):
                picks.append(r)
        if len(picks) > args.n:
            take = np.linspace(0, len(picks)-1, args.n).round().astype(int)
            picks = [picks[i] for i in sorted(set(take))]

    for i, r in enumerate(picks, 1):
        r["short_id"] = f"U{i}"
        r["title"] = f"U{i}  {r['name']} (UIUC {r['p_number']})"
        r["subtitle"] = (
            f"median {r['median_depth_m']:.1f} m below surface, "
            f"seasonal range {r['seasonal_range_m']:.1f} m; "
            f"{r['n_days']} observations\n"
            f"{r['lat']:.3f}°N, {abs(r['lon']):.3f}°W  "
            f"(grid row {r['y']}, col {r['x']})")

    out = dict(
        source="UIUC/ISWS unconfined observation network, "
               "DOE_GW_OBV_unconfined_2003_2006_depth.csv",
        selection_rule=(
            ("Every UNCONFINED well within ~2 km of a model cell centre, "
             f"median depth <= {MAX_MEDIAN_DEPTH} m. No record-length, "
             "water-year-coverage or spatial-thinning filter: the whole "
             "network is evaluated. Ordered by median observed depth."
             if args.all else
             f"UNCONFINED wells within ~2 km of a model cell centre, "
             f">= {MIN_DAYS} observations covering all four water years, "
             f"median depth <= {MAX_MEDIAN_DEPTH} m, picks spread along the "
             f"observed-depth distribution and >= {MIN_SEP_CELLS} grid cells "
             f"apart.")
            + " Observation coverage and observed depth only -- no model output."),
        variable="depth to water below land surface (m)",
        n_candidates=len(cand), wells=picks)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"  selected {len(picks)}:")
    for r in picks:
        print(f"    {r['short_id']}  {r['name'][:26]:26s} "
              f"median {r['median_depth_m']:5.2f} m  n={r['n_days']:5d}  "
              f"({r['lat']:.2f}, {r['lon']:.2f})  cell ({r['y']},{r['x']})")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
