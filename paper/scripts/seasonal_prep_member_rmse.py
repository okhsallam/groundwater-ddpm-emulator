"""
Per-member daily error series for the SEAS5-forced ensemble.

The seasonal statistics cache (_stats_armDyear.pkl) stores only the
ensemble-mean daily RMSE and the domain-mean ensemble spread, so the seasonal
figures could not draw the min-max envelope over member error curves that the
reanalysis half uses. This builds the missing quantity: for every fold,
variable and member, the daily domain RMSE against the ParFlow truth, a
(40, 360) array.

A fold is 40 members x 360 days x 637 x 665, ~24 GB in float32, so it is never
held whole: one member file (4 members) is read at a time, each member reduced
to its 360 daily RMSEs immediately, and a running sum over members gives the
ensemble mean, which is reduced the same way at the end. Peak memory is the
truth field plus one member file plus one accumulator, ~12 GB. The four folds
are independent, so they run as four processes.

Truth and day alignment follow ddpm_model/make_seas5_figures.py:references()
exactly -- global_days index data/tiles/day_index.json, the water year's first
entry gives day0, and target2d_wy<WY>.nc is indexed at (global_day - day0).

Two checks are asserted per fold and variable:
  * it is strictly BELOW the mean of the per-member daily RMSEs (if the two
    agreed, the reduction would have averaged in the wrong place) -- asserted
    as each fold is reduced;
  * the ensemble-mean daily RMSE reproduces the cached "rmse" in
    _stats_armDyear.pkl (the check on truth alignment and member ordering) --
    asserted in a final pass, AFTER the npz is written, so this reduction can
    run alongside the scoring job that writes that cache and the hours of
    reading are not lost if the cache is not there yet.

Output: paper/figures/_seasonal_member_rmse.npz
"""
import glob
import json
import multiprocessing
import os
import pickle
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P

ROLL = os.path.join(P.DDPM, "Results", "seas5")
STATS = os.path.join(P.ROOT, "report_full", "figures_dyear",
                     "_stats_armDyear.pkl")
OUT = os.path.join(P.FIGS, "_seasonal_member_rmse.npz")
TOL = 1e-3          # relative; the cache reduced float16 members in float32


def daily_rmse(pred, truth):
    """Domain RMSE per day; nanmean, as the cached reduction uses."""
    return np.sqrt(np.nanmean((pred - truth) ** 2, axis=(1, 2)))


def truth_series(fold, gd):
    di = json.load(open(os.path.join(P.DATA, "tiles", "day_index.json")))
    wy = P.FOLD_WY[fold]
    day0 = min(i for i, e in enumerate(di) if e["water_year"] == wy)
    idx = [int(g) - day0 for g in gd]
    ds = xr.open_dataset(os.path.join(P.RAW, f"target2d_wy{wy}.nc"))
    stamp = [str(s)[:10] for s in ds["time"].isel(time=idx).values]
    dates = [di[int(g)]["date"] for g in gd]
    assert stamp == dates, (fold, stamp[:3], dates[:3])
    out = {v: ds[v].isel(time=idx).values.astype(np.float32) for v in P.VARS}
    ds.close()
    return out, dates


def check_against_cache(store):
    """The cached ensemble-mean series is the check on truth alignment.

    Separate from the reduction, and run last: the cache is written by
    ddpm_model/make_seas5_figures.py --arm Dyear, which may still be running
    when this reduction starts.
    """
    if not os.path.exists(STATS):
        raise SystemExit(f"\n  WROTE {OUT}, but {STATS} does not exist yet, so "
                         f"the cached-rmse check could not run. Once the "
                         f"scoring job has written it, run\n"
                         f"      python {os.path.basename(__file__)} --check-only\n"
                         f"  which redoes the check from the npz without "
                         f"re-reading the members.")
    S = pickle.load(open(STATS, "rb"))
    for fold in P.FOLDS:
        assert len(S[fold]["dates"]) == 360, (fold, len(S[fold]["dates"]))
        for v in P.VARS:
            ens = np.asarray(store[f"f{fold}_{v}_ens"], np.float64)
            want = np.asarray(S[fold][v]["rmse"], np.float64)
            assert ens.shape == want.shape, (fold, v, ens.shape, want.shape)
            rel = np.abs(ens - want) / np.maximum(want, 1e-12)
            assert rel.max() <= TOL, (fold, v, "cached rmse", float(rel.max()),
                                      float(ens[int(rel.argmax())]),
                                      float(want[int(rel.argmax())]))
            print(f"  fold {fold} (WY{P.FOLD_WY[fold]}) {v}: max rel. dev. "
                  f"from cached rmse {rel.max():.2e}", flush=True)


def reduce_fold(fold):
    """One fold, start to finish. Independent of every other fold, so the
    four run as four processes: the reduction is single-threaded numpy over
    ~17 GB of compressed member files per fold, and serialising them took
    hours for no reason."""
    files = sorted(glob.glob(os.path.join(ROLL, f"armDyear_fold{fold}",
                                          "member_*.npz")))
    assert files, f"fold {fold}: no member files"
    assert len(files) == 10, (fold, len(files))
    gd = np.load(files[0])["global_days"].astype(int)
    assert len(gd) == 360, (fold, len(gd))   # the full water year, not 210
    truth, dates = truth_series(fold, gd)
    nd = len(gd)

    per = {v: [] for v in P.VARS}
    acc = {v: np.zeros((nd,) + truth[v].shape[1:], np.float64)
           for v in P.VARS}
    n = 0
    for fp in files:
        z = np.load(fp)
        assert np.array_equal(z["global_days"].astype(int), gd), fp
        for v in P.VARS:
            a = z[v]                              # (4, 360, 637, 665) f16
            for m in range(a.shape[0]):
                x = a[m].astype(np.float32)       # (360, 637, 665)
                per[v].append(daily_rmse(x, truth[v]))
                acc[v] += x
                del x
            del a
        n += 4
        z.close()
        print(f"    fold {fold}: {os.path.basename(fp)} -> {n} members",
              flush=True)

    out = {f"f{fold}_dates": np.array(dates), f"f{fold}_nmem": np.array(n)}
    for v in P.VARS:
        mem = np.asarray(per[v], np.float64)              # (40, 360)
        assert mem.shape == (n, nd), (fold, v, mem.shape)
        ens = daily_rmse((acc[v] / n).astype(np.float32), truth[v])

        # averaging the members must beat averaging their errors
        assert np.all(ens < mem.mean(0)), (fold, v, "mean-of-errors")

        out[f"f{fold}_{v}_mem"] = mem.astype(np.float32)
        out[f"f{fold}_{v}_ens"] = ens.astype(np.float32)
        beat = int(np.sum(ens < mem.min(0)))
        print(f"  fold {fold} (WY{P.FOLD_WY[fold]}) {v}: ensemble mean "
              f"below the best member on {beat}/{nd} days", flush=True)
    return out


def main():
    # --check-only: the reduction is already cached; just run the comparison
    # against the scoring job's statistics, which may not have existed when
    # the reduction finished.
    if "--check-only" in sys.argv:
        Z = np.load(OUT)
        print(f"  checking {OUT} against {STATS}")
        check_against_cache({k: Z[k] for k in Z.files})
        return
    with multiprocessing.get_context("spawn").Pool(len(P.FOLDS)) as pool:
        parts = pool.map(reduce_fold, list(P.FOLDS))
    store = {}
    for d in parts:
        store.update(d)
    os.makedirs(P.FIGS, exist_ok=True)
    np.savez_compressed(OUT, **store)
    print(f"  wrote {OUT}")
    check_against_cache(store)


if __name__ == "__main__":
    main()
