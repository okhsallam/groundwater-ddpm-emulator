"""
Extract modelled water-table depth at observation-well cells, and cache it.

The rollouts are 15-16 GB per fold and the well comparison needs fourteen grid
cells from each. Decompressing the whole archive to read fourteen columns is
unavoidable once, but it should not be repeated for every figure tweak, so the
extracted series are cached as a few kilobytes per source.

Handles both storage layouts:
  * NLDAS-driven folds  -- Results/<run>/rollout/ensemble_state.npz, one archive
    with (members, days, y, x)
  * SEAS5 arms          -- Results/seas5/arm<A>_fold<N>/member_*.npz, one archive
    per forcing member, each (diffusion members, days, y, x)

Output: data/well_series/<source>.npz with, per well, an array of shape
(members, days) plus the global day indices.
"""
import argparse
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CACHE = os.path.join(ROOT, "data", "well_series")
VAR = "water_table_depth"


def wells():
    out = []
    for f, net in ((os.path.join(ROOT, "data", "reference_wells.json"), "USGS"),
                   (os.path.join(ROOT, "data", "reference_wells_uiuc.json"), "UIUC")):
        if not os.path.exists(f):
            print(f"  (no {f}, skipping {net})"); continue
        d = json.load(open(f))
        for w in d["wells"]:
            out.append(dict(net=net, sid=w["short_id"], y=int(w["y"]), x=int(w["x"]),
                            title=w.get("title", w["short_id"]),
                            subtitle=w.get("subtitle", ""),
                            lat=w["lat"], lon=w["lon"],
                            median_depth_m=w.get("median_depth_m")))
    return out


def extract(paths, ws):
    """-> (n_members_total, n_days, n_wells) float32, and the day index."""
    ys = np.array([w["y"] for w in ws]); xs = np.array([w["x"] for w in ws])
    chunks, gd = [], None
    for p in paths:
        z = np.load(p)
        a = z[VAR]                       # (M, days, y, x)
        chunks.append(np.asarray(a[:, :, ys, xs], dtype=np.float32))
        g = z["global_days"].astype(int)
        gd = g if gd is None else gd
        del a, z
    return np.concatenate(chunks, axis=0), gd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True,
                    help="e.g. nldas_fold1 seas5_Dp_fold1")
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    ws = wells()
    print(f"  {len(ws)} wells "
          f"({sum(w['net']=='USGS' for w in ws)} USGS, "
          f"{sum(w['net']=='UIUC' for w in ws)} UIUC)")

    for src in args.sources:
        out = os.path.join(CACHE, f"{src}.npz")
        if os.path.exists(out):
            print(f"  {src}: cached"); continue
        if src.startswith("nldas_fold"):
            f = src.replace("nldas_fold", "")
            paths = [os.path.join(HERE, "Results",
                                  f"groundwater_f15_200ep_fold{f}", "rollout",
                                  "ensemble_state.npz")]
        else:                                   # seas5_<arm>_fold<N>
            _, arm, fold = src.split("_")
            paths = sorted(glob.glob(os.path.join(
                HERE, "Results", "seas5", f"arm{arm}_{fold}", "member_*.npz")))
        paths = [p for p in paths if os.path.exists(p)]
        if not paths:
            print(f"  {src}: no files, skipping"); continue
        series, gd = extract(paths, ws)
        np.savez_compressed(out, series=series, global_days=gd,
                            sid=np.array([w["sid"] for w in ws]),
                            net=np.array([w["net"] for w in ws]))
        print(f"  {src}: {series.shape[0]} members x {series.shape[1]} days "
              f"x {series.shape[2]} wells -> {os.path.basename(out)}", flush=True)


if __name__ == "__main__":
    main()
