"""
Cache the seven map days needed by the seasonal evolution figures.

A fold of the SEAS5-forced ensemble is 40 members x 360 days x 637 x 665.  That
is ~24 GB in float32, so a fold is never held whole: one member file (4 members)
is read at a time, only the seven wanted days are kept, and a running sum gives
the ensemble mean.  Peak memory is one member file plus a (7, 637, 665) buffer.

Truth and day alignment follow ddpm_model/make_seas5_figures.py:references()
exactly -- global_days index data/tiles/day_index.json, the water year's first
entry gives day0, and target2d_wy<WY>.nc is indexed at (global_day - day0).

Output: paper/figures/_seasonal_evolution.npz
"""
import glob
import json
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P

# Seven days evenly spaced over the 360-day forecast, endpoints included:
# lead day 1 (the first day the rollout predicts) through lead day 360, i.e.
# days 1, 60, 120, 180, 240, 300, 360 of the water year -- Oct, Nov, Jan, Mar,
# May, Jul, Sep in every fold, so one set of column headers serves all four.
DAYS = np.linspace(0, 359, 7).astype(int)

ROLL = os.path.join(P.DDPM, "Results", "seas5")
OUT = os.path.join(P.FIGS, "_seasonal_evolution.npz")


def member_files(fold):
    return sorted(glob.glob(os.path.join(ROLL, f"armDyear_fold{fold}", "member_*.npz")))


def truth_for(fold, gd, days):
    """ParFlow truth at the wanted rollout days, make_seas5_figures convention."""
    di = json.load(open(os.path.join(P.DATA, "tiles", "day_index.json")))
    wy = P.FOLD_WY[fold]
    day0 = min(i for i, e in enumerate(di) if e["water_year"] == wy)
    idx = [int(gd[d]) - day0 for d in days]
    ds = xr.open_dataset(os.path.join(P.RAW, f"target2d_wy{wy}.nc"))
    out = {v: ds[v].isel(time=idx).values.astype(np.float32) for v in P.VARS}
    stamp = [str(s)[:10] for s in ds["time"].isel(time=idx).values]
    ds.close()
    dates = [di[int(gd[d])]["date"] for d in days]
    assert stamp == dates, (stamp, dates)       # truth row == labelled date
    return out, dates


def main():
    store = {}
    for fold in P.FOLDS:
        files = member_files(fold)
        if not files:
            print(f"  fold {fold}: no member files", flush=True)
            continue
        gd = None
        acc = {v: np.zeros((len(DAYS), 637, 665), np.float64) for v in P.VARS}
        n = 0
        for fp in files:
            z = np.load(fp)
            g = z["global_days"].astype(int)
            if gd is None:
                gd = g
            else:
                assert np.array_equal(gd, g), f"global_days differ in {fp}"
            for v in P.VARS:
                a = z[v]                                 # (4, 360, 637, 665) f16
                acc[v] += a[:, DAYS].astype(np.float32).sum(0)
                del a
            n += z[P.VARS[0]].shape[0]
            z.close()
            print(f"    fold {fold}: {os.path.basename(fp)} -> {n} members",
                  flush=True)
        mean = {v: (acc[v] / n).astype(np.float32) for v in P.VARS}
        truth, dates = truth_for(fold, gd, DAYS)
        store[f"f{fold}_dates"] = np.array(dates)
        store[f"f{fold}_days"] = DAYS
        store[f"f{fold}_nmem"] = np.array(n)
        for v in P.VARS:
            store[f"f{fold}_{v}_truth"] = truth[v]
            store[f"f{fold}_{v}_mean"] = mean[v]
        print(f"  fold {fold} (WY{P.FOLD_WY[fold]}): {n} members, days "
              f"{list(DAYS)} = {dates}", flush=True)
    os.makedirs(P.FIGS, exist_ok=True)
    np.savez_compressed(OUT, **store)
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
