"""
Compute per-variable normalization stats (mean/std) from the TRAIN water years
only (WY2003-2004, see ddpm_model/dataset_groundwater.py's split convention),
so val/test never leak into normalization the way they would if stats were
computed over the full 4-year record.

Computed directly from the raw data/parflow_raw/ NetCDFs rather than from the
tiled npz files, since the tiles have 25% spatial overlap and would otherwise
double-count/bias the pixels sitting in overlap regions.

5-layer variables (pressure_head, permeability, porosity, van_genuchten_alpha,
van_genuchten_n) get one mean/std per z-layer -- see README's z-layer-ordering
section for why: the deepest and shallowest layers are on very different
physical scales.

Output: data/normalization_stats.json (or --out), consumed by
dataset_groundwater.py's `normalization_stats` argument. --train-years lets
the 2026-09-17 4-fold cross-validation (each water year takes a turn as test)
compute fold-specific stats instead of always using WY2003-2004.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "data" / "parflow_raw"

STATIC_VARS_1LAYER = ["elevation", "topographic_index"]
STATIC_VARS_5LAYER = ["permeability", "porosity", "van_genuchten_alpha", "van_genuchten_n"]
FORCING_VARS = ["tmin", "tmax", "precipitation", "evapotranspiration", "snowmelt"]
TARGET2D_VARS = ["water_table_depth", "soil_moisture"]
TARGET3D_VARS_5LAYER = ["pressure_head"]


def stats_1layer(arr):
    return {"mean": float(np.nanmean(arr)), "std": float(np.nanstd(arr))}


def stats_5layer(arr):
    """arr: (..., z, y, x) -- returns per-z mean/std lists."""
    flat_by_z = np.moveaxis(arr, -3, 0).reshape(arr.shape[-3], -1)
    means = np.nanmean(flat_by_z, axis=1)
    stds = np.nanstd(flat_by_z, axis=1)
    return {"mean": [float(m) for m in means], "std": [float(s) for s in stds]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-years", type=int, nargs="+", default=[2003, 2004])
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    out_path = Path(args.out) if args.out else PROJECT_ROOT / "data" / "normalization_stats.json"

    stats = {}

    static = xr.open_dataset(RAW_DIR / "static_conditions.nc")
    for v in STATIC_VARS_1LAYER:
        stats[v] = stats_1layer(static[v].values)
    for v in STATIC_VARS_5LAYER:
        stats[v] = stats_5layer(static[v].values)
    static.close()
    print("static done:", list(stats.keys()))

    forcing_accum = {v: [] for v in FORCING_VARS}
    target2d_accum = {v: [] for v in TARGET2D_VARS}
    target3d_accum = {v: [] for v in TARGET3D_VARS_5LAYER}

    for wy in args.train_years:
        forcing = xr.open_dataset(RAW_DIR / f"forcing_wy{wy}.nc")
        target2d = xr.open_dataset(RAW_DIR / f"target2d_wy{wy}.nc")
        target3d = xr.open_dataset(RAW_DIR / f"target3d_wy{wy}.nc")
        for v in FORCING_VARS:
            forcing_accum[v].append(forcing[v].values)
        for v in TARGET2D_VARS:
            target2d_accum[v].append(target2d[v].values)
        for v in TARGET3D_VARS_5LAYER:
            target3d_accum[v].append(target3d[v].values)
        forcing.close(); target2d.close(); target3d.close()
        print(f"loaded WY{wy}")

    for v in FORCING_VARS:
        stats[v] = stats_1layer(np.concatenate(forcing_accum[v], axis=0))
    for v in TARGET2D_VARS:
        stats[v] = stats_1layer(np.concatenate(target2d_accum[v], axis=0))
    for v in TARGET3D_VARS_5LAYER:
        stats[v] = stats_5layer(np.concatenate(target3d_accum[v], axis=0))

    with open(out_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"wrote {out_path}")
    for k, s in stats.items():
        print(f"  {k}: {s}")


if __name__ == "__main__":
    main()
