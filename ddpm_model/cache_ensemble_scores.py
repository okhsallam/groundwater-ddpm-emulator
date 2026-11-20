"""
Cache the per-member error statistics that the ensemble-shaded figures need.

The member arrays are 9.8 GB per variable per fold, so they are read exactly
once here and reduced to a few kilobytes: for each fold and variable, the RMSE
time series of every individual member, the ensemble-mean RMSE, the aggregated
per-cell ensemble spread, the persistence baseline, the per-cell temporal
correlation of the ensemble mean against the truth, and the normalisation
denominator p95-p5 that converts any of the RMSE series to a relative error.

Everything downstream reads only the cache.

Usage:
    python cache_ensemble_scores.py --folds 1 2 3 4
"""
import argparse
import gc
import json
import os

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "parflow_raw")
RES = os.path.join(HERE, "Results")
CACHE = os.path.join(HERE, "Results", "_ensemble_cache")

FOLDS = {1: 2006, 2: 2005, 3: 2004, 4: 2003}
VARS = ("water_table_depth", "soil_moisture")
STD_FLOOR = {"water_table_depth": 1e-3, "soil_moisture": 1e-4}
RNG_FLOOR = {"water_table_depth": 0.01, "soil_moisture": 0.005}


def pearson_map(a, b, floor):
    """Per-cell temporal Pearson r, NaN where either series is near-constant."""
    a = a - a.mean(0, keepdims=True)
    b = b - b.mean(0, keepdims=True)
    sa, sb = np.sqrt((a ** 2).sum(0)), np.sqrt((b ** 2).sum(0))
    ok = (sa > floor) & (sb > floor)
    r = np.full(sa.shape, np.nan, dtype=np.float32)
    np.divide((a * b).sum(0), sa * sb, out=r, where=ok)
    r[~ok] = np.nan
    return r


def run_fold(f):
    yr = FOLDS[f]
    R = os.path.join(RES, f"groundwater_f15_200ep_fold{f}", "rollout")
    meta = json.loads(open(os.path.join(R, "rollout_meta.json")).read())
    M = meta["n_members"]
    mn = np.load(os.path.join(R, "ensemble_mean.npz"))
    gd = mn["global_days"].tolist()
    di = json.loads(open(os.path.join(DATA, "tiles", "day_index.json")).read())
    day0 = min(i for i, e in enumerate(di) if e["water_year"] == yr)
    dates = [di[int(g)]["date"] for g in gd]

    t2 = xr.open_dataset(os.path.join(RAW, f"target2d_wy{yr}.nc"))
    out = {"dates": np.array(dates), "n_members": M, "test_year": yr}
    src = np.load(os.path.join(R, "ensemble_state.npz"))
    for v in VARS:
        y = t2[v].values[[g - day0 for g in gd]].astype(np.float32)
        p = mn[v]
        print(f"  fold{f} {v}: members ...", flush=True)
        ens = src[v]                                   # (M, T, H, W)
        r_mem = np.stack([np.sqrt(np.nanmean((ens[m] - y) ** 2, axis=(1, 2)))
                          for m in range(M)])          # (M, T)
        del ens
        gc.collect()

        rng = float(np.nanpercentile(y, 95) - np.nanpercentile(y, 5))
        pix_rng = (np.nanpercentile(y, 95, axis=0)
                   - np.nanpercentile(y, 5, axis=0))
        ok = pix_rng > RNG_FLOOR[v]
        pix_rmse = np.sqrt(np.nanmean((p - y) ** 2, axis=0))
        out[f"{v}/r_mem"] = r_mem.astype(np.float32)
        out[f"{v}/r_mean"] = np.sqrt(
            np.nanmean((p - y) ** 2, axis=(1, 2))).astype(np.float32)
        out[f"{v}/r_pers"] = np.sqrt(np.nanmean(
            (np.broadcast_to(y[0], y.shape) - y) ** 2, axis=(1, 2))).astype(np.float32)
        out[f"{v}/sd_field"] = np.sqrt(
            np.nanmean(mn[f"{v}_sd"] ** 2, axis=(1, 2))).astype(np.float32)
        out[f"{v}/den"] = rng
        out[f"{v}/corr"] = pearson_map(p, y, STD_FLOOR[v])
        out[f"{v}/pix_nrmse"] = np.where(
            ok, 100 * pix_rmse / np.where(ok, pix_rng, np.nan), np.nan
        ).astype(np.float32)
        out[f"{v}/rmse"] = float(np.sqrt(np.nanmean((p - y) ** 2)))
        out[f"{v}/pers"] = float(np.sqrt(np.nanmean(
            (np.broadcast_to(y[0], y.shape) - y) ** 2)))
        del y, p
        gc.collect()
    t2.close()
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, f"fold{f}.npz"), **out)
    print(f"  fold{f} cached", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()
    for f in args.folds:
        run_fold(f)


if __name__ == "__main__":
    main()
