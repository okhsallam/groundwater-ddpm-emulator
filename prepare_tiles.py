"""
Build tiled, per-day training data for the CESER groundwater diffusion model.

Reads the raw per-water-year NetCDFs under data/parflow_raw/ (see README.md
for the full variable/dimension dictionary) and writes, for each requested
tile size, one static npz per tile plus one dynamic npz per (tile, day) --
day indexed globally across all 4 water years (0..1460, 2002-10-01 ->
2006-09-30). Storing at day granularity (rather than pre-baking a fixed
forecast window like ceser_ddpm_GloFAS's prepare_patches.py does) lets the
dataloader pick any rollout/forecast length later without re-prepping data.

Resumable: skips any (tile, day) npz that already exists, same convention as
download_paper_data.py.

Usage:
    python prepare_tiles.py                       # full run, both tile sizes
    python prepare_tiles.py --tile-sizes 128       # just one size
    python prepare_tiles.py --max-days 3           # smoke test
"""
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).parent
RAW_DIR = PROJECT_ROOT / "data" / "parflow_raw"
OUT_ROOT = PROJECT_ROOT / "data" / "tiles"

WATER_YEARS = [2003, 2004, 2005, 2006]
DOMAIN_SHAPE = (637, 665)  # (y, x)
OVERLAP_FRAC = 0.25

FORCING_VARS = ["tmin", "tmax", "precipitation", "evapotranspiration", "snowmelt"]
TARGET2D_VARS = ["water_table_depth", "soil_moisture"]
TARGET3D_VARS = ["pressure_head"]


def compute_tile_grid(domain_shape, tile_size, overlap_frac=OVERLAP_FRAC):
    """Overlapping tiles covering (H, W) exactly, last tile flush to the far edge."""
    H, W = domain_shape
    stride = max(1, round(tile_size * (1 - overlap_frac)))

    def axis_starts(size):
        starts = list(range(0, size - tile_size, stride))
        starts.append(size - tile_size)
        return sorted(set(starts))

    y0s = axis_starts(H)
    x0s = axis_starts(W)
    tiles = []
    for y0 in y0s:
        for x0 in x0s:
            tiles.append({"y0": y0, "y1": y0 + tile_size, "x0": x0, "x1": x0 + tile_size})
    return tiles


def build_day_index(max_days=None):
    """Global day index 0..1460 -> (water_year, local_idx, date)."""
    index = []
    for wy in WATER_YEARS:
        ds = xr.open_dataset(RAW_DIR / f"forcing_wy{wy}.nc")
        dates = ds.time.values
        n = len(dates) if max_days is None else min(len(dates), max_days)
        for local_idx in range(n):
            index.append({
                "water_year": wy,
                "local_idx": local_idx,
                "date": str(dates[local_idx])[:10],
            })
        ds.close()
        if max_days is not None:
            # smoke test: only take the first `max_days` of the first water year
            break
    return index


def prepare_static_tiles(tiles, out_dir):
    static_dir = out_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)
    ds = xr.open_dataset(RAW_DIR / "static_conditions.nc")
    arrays = {v: ds[v].values.astype(np.float32) for v in ds.data_vars}
    ds.close()
    for tile_id, t in enumerate(tiles):
        out_path = static_dir / f"tile_{tile_id:03d}.npz"
        if out_path.exists():
            continue
        sl_y, sl_x = slice(t["y0"], t["y1"]), slice(t["x0"], t["x1"])
        crop = {v: a[..., sl_y, sl_x] for v, a in arrays.items()}
        np.savez(out_path, y0=t["y0"], y1=t["y1"], x0=t["x0"], x1=t["x1"], **crop)


def prepare_dynamic_tiles(tiles, out_dir, day_index, max_days=None):
    dyn_dir = out_dir / "dynamic"
    for tile_id in range(len(tiles)):
        (dyn_dir / f"tile_{tile_id:03d}").mkdir(parents=True, exist_ok=True)

    global_day = 0
    water_years = sorted({d["water_year"] for d in day_index})
    for wy in water_years:
        forcing = xr.open_dataset(RAW_DIR / f"forcing_wy{wy}.nc")
        target2d = xr.open_dataset(RAW_DIR / f"target2d_wy{wy}.nc")
        target3d = xr.open_dataset(RAW_DIR / f"target3d_wy{wy}.nc")
        n_days = sum(1 for d in day_index if d["water_year"] == wy)

        # Pull the whole year into memory once (float32) -- far fewer, much
        # larger reads than doing random per-tile access against the netCDF.
        forcing_arr = {v: forcing[v].isel(time=slice(0, n_days)).values.astype(np.float32) for v in FORCING_VARS}
        target2d_arr = {v: target2d[v].isel(time=slice(0, n_days)).values.astype(np.float32) for v in TARGET2D_VARS}
        target3d_arr = {v: target3d[v].isel(time=slice(0, n_days)).values.astype(np.float32) for v in TARGET3D_VARS}
        forcing.close(); target2d.close(); target3d.close()

        for local_idx in range(n_days):
            date = day_index[global_day]["date"]
            for tile_id, t in enumerate(tiles):
                out_path = dyn_dir / f"tile_{tile_id:03d}" / f"day_{global_day:04d}.npz"
                if out_path.exists():
                    continue
                sl_y, sl_x = slice(t["y0"], t["y1"]), slice(t["x0"], t["x1"])
                payload = {"date": date, "global_day": global_day}
                for v in FORCING_VARS:
                    payload[v] = forcing_arr[v][local_idx, sl_y, sl_x]
                for v in TARGET2D_VARS:
                    payload[v] = target2d_arr[v][local_idx, sl_y, sl_x]
                for v in TARGET3D_VARS:
                    payload[v] = target3d_arr[v][local_idx, :, sl_y, sl_x]
                np.savez(out_path, **payload)
            global_day += 1

        print(f"finished WY{wy} ({n_days} days, global_day now {global_day})", flush=True)
        if max_days is not None:
            break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-sizes", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--max-days", type=int, default=None, help="smoke-test: only prep the first N days of WY2003")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    day_index = build_day_index(max_days=args.max_days)
    with open(OUT_ROOT / "day_index.json", "w") as f:
        json.dump(day_index, f, indent=2)
    print(f"day_index: {len(day_index)} days")

    for tile_size in args.tile_sizes:
        tiles = compute_tile_grid(DOMAIN_SHAPE, tile_size)
        out_dir = OUT_ROOT / f"tile{tile_size}"
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "tile_grid.json", "w") as f:
            json.dump({
                "tile_size": tile_size,
                "overlap_frac": OVERLAP_FRAC,
                "domain_shape": DOMAIN_SHAPE,
                "tiles": tiles,
            }, f, indent=2)
        print(f"tile_size={tile_size}: {len(tiles)} tiles -> {out_dir}")

        prepare_static_tiles(tiles, out_dir)
        prepare_dynamic_tiles(tiles, out_dir, day_index, max_days=args.max_days)

    print("Done.")


if __name__ == "__main__":
    main()
