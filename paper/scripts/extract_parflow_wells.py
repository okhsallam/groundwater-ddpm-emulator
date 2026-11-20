"""
ParFlow truth at the fourteen observation-well cells, WY2003-WY2006.

The soil-moisture figure has no observations to compare against -- no well in
either network logs soil moisture -- so the reference line is ParFlow itself,
read at exactly the cells the emulator series were extracted from. Water table
depth is cached alongside it, which is what makes the shift convention of the
water-table figure auditable.

The targets are contiguous and uncompressed, so fourteen columns cost fourteen
strided reads per day rather than a 1.2 GB load.

Output: data/well_series_sm/parflow_truth.npz
"""
import json
import os

import netCDF4 as nc
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RAW = os.path.join(ROOT, "data", "parflow_raw")
OUT = os.path.join(ROOT, "data", "well_series_sm")
WY = (2003, 2004, 2005, 2006)


def wells():
    out = []
    for f, net in ((os.path.join(ROOT, "data", "reference_wells.json"), "USGS"),
                   (os.path.join(ROOT, "data", "reference_wells_uiuc.json"), "UIUC")):
        for w in json.load(open(f))["wells"]:
            out.append(dict(net=net, sid=w["short_id"],
                            y=int(w["y"]), x=int(w["x"])))
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    ws = wells()
    ys = [w["y"] for w in ws]; xs = [w["x"] for w in ws]
    sm, wtd, dates = [], [], []
    for wy in WY:
        d = nc.Dataset(os.path.join(RAW, f"target2d_wy{wy}.nc"))
        t = nc.num2date(d["time"][:], d["time"].units,
                        only_use_cftime_datetimes=False)
        dates += [np.datetime64(str(x)[:10]) for x in t]
        a = np.empty((len(t), len(ws)), np.float32)
        b = np.empty((len(t), len(ws)), np.float32)
        for j, (y, x) in enumerate(zip(ys, xs)):
            a[:, j] = d["soil_moisture"][:, y, x]
            b[:, j] = d["water_table_depth"][:, y, x]
        sm.append(a); wtd.append(b)
        print(f"  WY{wy}: {len(t)} days  {str(t[0])[:10]}..{str(t[-1])[:10]}",
              flush=True)
        d.close()
    sm = np.concatenate(sm); wtd = np.concatenate(wtd)
    np.savez_compressed(os.path.join(OUT, "parflow_truth.npz"),
                        soil_moisture=sm, water_table_depth=wtd,
                        dates=np.array([str(x) for x in dates]),
                        sid=np.array([w["sid"] for w in ws]),
                        net=np.array([w["net"] for w in ws]))
    print(f"  wrote parflow_truth.npz  {sm.shape}  "
          f"sm range {sm.min():.3f}-{sm.max():.3f}")


if __name__ == "__main__":
    main()
