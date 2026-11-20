"""
Pick the fixed set of USGS wells shown in every fold's figures.

Same principle as select_reference_points.py: chosen ONCE, from observation
coverage and observed depth alone -- never from model output -- so the same
places appear in every fold and every lead time and the panels can be compared.

The rule, in order:
  1. keep only wells flagged `comparable` by fetch_usgs_wells.py (median depth to
     water within ParFlow's range; the deep confined-aquifer wells measure
     potentiometric head, not a water table);
  2. require at least MIN_DAYS_PER_YEAR observations in EVERY one of the four
     water years, so one well set works for all four folds;
  3. walk the observed-depth distribution from shallowest to deepest and take
     evenly spaced picks, so the set spans the hydrologic gradient rather than
     clustering at whatever depth happens to be common;
  4. enforce MIN_SEPARATION between picks so they are not all the same place --
     the Indiana observation-well network dominates the domain by count and
     would otherwise supply nearly every slot;
  5. drop wells within EDGE_MARGIN cells of the domain edge, where the tile
     mosaic is measurably noisier (see the constant below for the numbers).

Output: data/reference_wells.json, with a human-readable `title` and `subtitle`
for each well so the figures can say what a site IS rather than printing a PLSS
code like "080N05W22CBCB2".

NOTE: these wells carry DEPTH TO WATER only. No soil-moisture observations exist
anywhere in this domain for WY2003-2006 -- every USGS soil-moisture site in the
region begins in 2023 -- so the soil-moisture figures at these locations compare
ParFlow with the emulator and carry no observation curve. That is stated on the
figures rather than left as an apparent gap.
"""
import json
from pathlib import Path
from collections import defaultdict
import csv

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
WELL_DIR = PROJECT_ROOT / "data" / "usgs_wells"
OUT_PATH = PROJECT_ROOT / "data" / "reference_wells.json"
WATER_YEARS = [2003, 2004, 2005, 2006]
MIN_DAYS_PER_YEAR = 300
MIN_SEPARATION = 60          # grid cells (~60 km)
# The outermost cells are mosaicked from fewer overlapping tiles and measurably
# noisier: median per-pixel water-table RMSE is 0.646 m in the outer 4 cells
# against 0.555 m in the interior (fold1 f15), converging by ~8 cells. A well
# sitting there would be scored against the weakest part of the reconstruction.
EDGE_MARGIN = 8
N_WELLS = 8

STATE_NAMES = {"IL": "Illinois", "IA": "Iowa", "MO": "Missouri", "IN": "Indiana",
               "WI": "Wisconsin", "MN": "Minnesota", "KY": "Kentucky",
               "MI": "Michigan", "OH": "Ohio"}


def depth_class(median_m):
    if median_m < 1.5:
        return "very shallow water table"
    if median_m < 3.5:
        return "shallow water table"
    if median_m < 6.5:
        return "intermediate water table"
    return "deeper water table"


def clean_name(raw):
    """USGS station names mix a readable label with a PLSS code; keep the label."""
    raw = raw.strip()
    parts = [p.strip() for p in raw.split("  ") if p.strip()]
    if len(parts) > 1:
        # e.g. "080N05W22CBCB2      1941Elmira Depot 5" -> "Elmira Depot 5"
        tail = parts[-1]
        while tail and tail[0].isdigit():
            tail = tail[1:]
        if len(tail) >= 3:
            return tail.strip()
    return raw


def main():
    meta = json.loads((WELL_DIR / "wells.json").read_text())
    counts = defaultdict(lambda: defaultdict(int))
    with open(WELL_DIR / "observations.csv") as f:
        for r in csv.DictReader(f):
            wy = int(r["date"][:4]) + (1 if r["date"][5:7] >= "10" else 0)
            counts[r["site_no"]][wy] += 1

    H, W = 637, 665
    cands = []
    n_edge = 0
    for sid, w in meta["wells"].items():
        if not w["comparable"]:
            continue
        if not (EDGE_MARGIN <= w["y"] < H - EDGE_MARGIN
                and EDGE_MARGIN <= w["x"] < W - EDGE_MARGIN):
            n_edge += 1
            continue
        per_year = [counts[sid].get(y, 0) for y in WATER_YEARS]
        if min(per_year) < MIN_DAYS_PER_YEAR:
            continue
        w = dict(w)
        w["days_per_water_year"] = {str(y): n for y, n in zip(WATER_YEARS, per_year)}
        cands.append(w)
    if len(cands) < N_WELLS:
        raise SystemExit(f"only {len(cands)} wells meet the coverage rule")
    print(f"{len(cands)} wells have >= {MIN_DAYS_PER_YEAR} days in all four water years "
          f"({n_edge} dropped for sitting within {EDGE_MARGIN} cells of the domain edge)")

    cands.sort(key=lambda w: w["median_depth_m"])
    targets = np.linspace(0, len(cands) - 1, N_WELLS)
    chosen = []
    for t in targets:
        # nearest unused candidate to this point on the depth ladder that is far
        # enough from everything already picked
        order = sorted(range(len(cands)), key=lambda i: abs(i - t))
        for i in order:
            w = cands[i]
            if any(c["site_no"] == w["site_no"] for c in chosen):
                continue
            if any((w["y"] - c["y"]) ** 2 + (w["x"] - c["x"]) ** 2 < MIN_SEPARATION ** 2
                   for c in chosen):
                continue
            chosen.append(w)
            break
    chosen.sort(key=lambda w: w["median_depth_m"])

    for k, w in enumerate(chosen, start=1):
        state = STATE_NAMES.get(w.get("state", ""), w.get("state", ""))
        w["short_id"] = f"W{k}"
        w["title"] = f"W{k}  {clean_name(w['name'])} -- {state}"
        w["subtitle"] = (f"{depth_class(w['median_depth_m'])}: "
                         f"median {w['median_depth_m']:.1f} m below surface, "
                         f"seasonal range {w['seasonal_range_m']:.1f} m\n"
                         f"{w['lat']:.3f}°N, {abs(w['lon']):.3f}°W  "
                         f"(grid row {w['y']}, col {w['x']})")

    OUT_PATH.write_text(json.dumps({
        "selection_rule": (f"USGS NWIS wells with median depth to water within ParFlow's "
                           f"range and >= {MIN_DAYS_PER_YEAR} daily values in every water "
                           f"year 2003-2006; {N_WELLS} picks spaced evenly along the "
                           f"observed-depth distribution, >= {MIN_SEPARATION} grid cells apart. "
                           f"Observation coverage and observed depth only -- no model output."),
        "variable": "depth to water below land surface (m), USGS NWIS parameter 72019",
        "soil_moisture_note": ("No soil-moisture observations exist in this domain for "
                               "WY2003-2006; every USGS soil-moisture site in the region "
                               "begins in 2023. Soil-moisture figures at these wells "
                               "compare ParFlow with the emulator only."),
        "wells": chosen,
    }, indent=1))

    print(f"\nwrote {OUT_PATH}")
    for w in chosen:
        print(f"  {w['title']}")
        print(f"      {w['subtitle'].splitlines()[0]}")
        print(f"      {w['subtitle'].splitlines()[1]}")


if __name__ == "__main__":
    main()
