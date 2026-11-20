"""
Soil moisture at the fourteen observation-well cells, for every rollout.

data/well_series/*.npz carries only water_table_depth: extract_well_series.py
pulled one variable. The soil-moisture figure needs the same fourteen columns
from the same archives, and those archives are 10-16 GB each because they hold
the full (members, days, 637, 665) field.

Rather than np.load the whole 9.8 GB array to keep 14 cells of it, this walks
the .npy stream inside the zip one (member, day) frame at a time -- 1.7 MB of
decompressed buffer -- and gathers the cells as it goes. Same bytes read, 4000x
less memory, so the 44 archives can run in parallel.

Output: data/well_series_sm/<source>.npz, series (members, days, 14) float32,
laid out exactly like the water-table cache so the plotting code is shared.
"""
import glob
import json
import os
import sys
import zipfile
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
DDPM = os.path.join(ROOT, "ddpm_model")
OUT = os.path.join(ROOT, "data", "well_series_sm")


# Which rollout arms to extract. "Dyear" is the 360-day spliced water year
# that replaced the 210-day single-initialisation run.
ARMS = ("Dyear",)

def wells():
    out = []
    for f, net in ((os.path.join(ROOT, "data", "reference_wells.json"), "USGS"),
                   (os.path.join(ROOT, "data", "reference_wells_uiuc.json"), "UIUC")):
        for w in json.load(open(f))["wells"]:
            out.append(dict(net=net, sid=w["short_id"],
                            y=int(w["y"]), x=int(w["x"])))
    return out


def stream_cells(path, var, ys, xs):
    """(members, days, n_cells) float32 from <var>.npy inside the npz."""
    with zipfile.ZipFile(path) as z:
        with z.open(var + ".npy") as fh:
            ver = np.lib.format.read_magic(fh)
            shape, fortran, dtype = np.lib.format._read_array_header(fh, ver)
            assert not fortran, path
            m, nd, ny, nx = shape
            lin = ys * nx + xs                      # cell offsets in one frame
            n = ny * nx
            itemsize = dtype.itemsize
            buf = bytearray(n * itemsize)
            mv = memoryview(buf)
            res = np.empty((m, nd, len(lin)), np.float32)
            for i in range(m):
                for d in range(nd):
                    got = 0
                    while got < len(buf):
                        k = fh.readinto(mv[got:])
                        if not k:
                            raise EOFError(f"{path} truncated at {i},{d}")
                        got += k
                    frame = np.frombuffer(buf, dtype=dtype, count=n)
                    res[i, d] = frame[lin]
            return res, shape


def job(args):
    src, paths, var = args
    ws = wells()
    ys = np.array([w["y"] for w in ws]); xs = np.array([w["x"] for w in ws])
    chunks, gd = [], None
    for p in paths:
        a, shape = stream_cells(p, var, ys, xs)
        chunks.append(a)
        with zipfile.ZipFile(p) as z:
            with z.open("global_days.npy") as fh:
                g = np.lib.format.read_array(fh).astype(int)
        gd = g if gd is None else gd
    series = np.concatenate(chunks, 0)
    out = os.path.join(OUT, f"{src}.npz")
    np.savez_compressed(out, series=series, global_days=gd,
                        sid=np.array([w["sid"] for w in ws]),
                        net=np.array([w["net"] for w in ws]))
    return f"{src}: {series.shape} -> {os.path.basename(out)}"


def sources():
    tasks = []
    for f in (4, 3, 2, 1):
        p = os.path.join(DDPM, "Results", f"groundwater_f15_200ep_fold{f}",
                         "rollout", "ensemble_state.npz")
        if os.path.exists(p):
            tasks.append((f"nldas_fold{f}", [p], "soil_moisture"))
    for arm in ARMS:
        for f in (4, 3, 2, 1):
            ms = sorted(glob.glob(os.path.join(
                DDPM, "Results", "seas5", f"arm{arm}_fold{f}", "member_*.npz")))
            # one task per member: 40 small jobs parallelise far better than 4
            for k, p in enumerate(ms):
                tasks.append((f"seas5_{arm}_fold{f}_m{k:03d}", [p],
                              "soil_moisture"))
    return tasks


def merge_members():
    """Fuse the per-member seas5 archives into one per fold, member axis first,
    so the cache matches the water-table layout (40, days, 14)."""
    for arm in ARMS:
        for f in (4, 3, 2, 1):
            ps = sorted(glob.glob(os.path.join(OUT, f"seas5_{arm}_fold{f}_m*.npz")))
            if not ps:
                continue
            zs = [np.load(p) for p in ps]
            series = np.concatenate([z["series"] for z in zs], 0)
            np.savez_compressed(
                os.path.join(OUT, f"seas5_{arm}_fold{f}.npz"),
                series=series, global_days=zs[0]["global_days"],
                sid=zs[0]["sid"], net=zs[0]["net"])
            print(f"  merged seas5_{arm}_fold{f}: {series.shape}")


def main():
    os.makedirs(OUT, exist_ok=True)
    if "--merge-only" in sys.argv:
        merge_members(); return
    if "--validate" in sys.argv:
        # the streaming reader must reproduce the existing water-table cache
        ws = wells()
        ys = np.array([w["y"] for w in ws]); xs = np.array([w["x"] for w in ws])
        p = os.path.join(DDPM, "Results", "groundwater_f15_200ep_fold4",
                         "rollout", "ensemble_state.npz")
        ref = np.load(os.path.join(ROOT, "data", "well_series",
                                   "nldas_fold4.npz"))["series"]
        with zipfile.ZipFile(p) as z:
            with z.open("water_table_depth.npy") as fh:
                ver = np.lib.format.read_magic(fh)
                shape, fo, dt = np.lib.format._read_array_header(fh, ver)
                print("  archive shape", shape, dt, "fortran", fo)
                m, nd, ny, nx = shape
                lin = ys * nx + xs
                n = ny * nx
                buf = bytearray(n * dt.itemsize); mv = memoryview(buf)
                got_rows = []
                for d in range(5):                 # first member, 5 days
                    g = 0
                    while g < len(buf):
                        k = fh.readinto(mv[g:])
                        g += k
                    got_rows.append(np.frombuffer(buf, dt, n)[lin].astype(np.float32))
        a = np.array(got_rows)
        print("  streamed :", np.round(a[0], 4))
        print("  cached   :", np.round(ref[0, :5][0], 4))
        print("  max |diff| over 5 days x 14 wells:",
              float(np.abs(a - ref[0, :5]).max()))
        return
    tasks = [t for t in sources()
             if not os.path.exists(os.path.join(OUT, f"{t[0]}.npz"))]
    print(f"  {len(tasks)} archives to read", flush=True)
    with Pool(min(22, max(1, len(tasks)))) as pool:
        for msg in pool.imap_unordered(job, tasks):
            print("  " + msg, flush=True)
    merge_members()


if __name__ == "__main__":
    main()
