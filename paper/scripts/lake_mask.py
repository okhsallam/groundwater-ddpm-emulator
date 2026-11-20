"""
A mask for the large open-water bodies in the domain.

Why this exists: Lake Michigan occupies the north-east corner of the study
domain and ParFlow carries a water table and a soil moisture there like any
other cell. Those values are not wrong, they are simply not about soil --
a "water table depth" under 150 m of lake is a bookkeeping entry -- and
colouring them alongside the land makes the lake the most eye-catching
feature of every state map while saying nothing. The cells are therefore
blanked in the map figures.

NOTHING ELSE USES THIS. The mask is applied at draw time only: every score,
table and error series in the paper is computed over the full domain exactly
as before, so the figures and the numbers cannot drift apart. If the masked
cells were ever excluded from the statistics too, that would be a different
experiment and would have to be said.

The polygons are Natural Earth 1:50m lakes, read from the cartopy cache that
is already on this machine, so this works offline. Within the domain they
resolve to Lake Michigan and Lake Winnebago, 6.98% of the 423605 cells.
The rasterised mask is cached beside this file because the point-in-polygon
pass costs about a minute.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CACHE = os.path.join(ROOT, "data", "norm", "lake_mask.npy")


def _build():
    import sys
    sys.path.insert(0, ROOT)
    import cartopy
    cartopy.config["pre_existing_data_dir"] = cartopy.config["data_dir"]
    import cartopy.io.shapereader as shpreader
    from shapely.geometry import Point, box
    from shapely.ops import unary_union
    from shapely.prepared import prep
    from seas5_forcing import target_latlon

    lat, lon = target_latlon()
    shp = shpreader.natural_earth(resolution="50m", category="physical",
                                  name="lakes")
    dom = box(lon.min() - 0.2, lat.min() - 0.2, lon.max() + 0.2, lat.max() + 0.2)
    geoms = [r.geometry for r in shpreader.Reader(shp).records()
             if r.geometry.intersects(dom)]
    if not geoms:
        return np.zeros(lat.shape, bool)
    g = prep(unary_union(geoms))
    m = np.zeros(lat.shape, bool)
    for j in range(lat.shape[0]):
        for i in range(lat.shape[1]):
            if g.contains(Point(lon[j, i], lat[j, i])):
                m[j, i] = True
    return m


def lake_mask(shape=None):
    """True where a cell is open water. Cached; built on first use."""
    if os.path.exists(CACHE):
        m = np.load(CACHE)
    else:
        m = _build()
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        np.save(CACHE, m)
    if shape is not None and m.shape != tuple(shape):
        raise ValueError(f"lake mask is {m.shape}, field is {tuple(shape)}")
    return m


def blank_lakes(field):
    """A copy of `field` with the open-water cells set to NaN, for drawing."""
    a = np.array(field, dtype=float)
    m = lake_mask(a.shape[-2:])
    a[..., m] = np.nan
    return a


if __name__ == "__main__":
    m = lake_mask()
    print(f"lake mask {m.shape}: {m.sum()} cells, {100*m.mean():.2f}% of domain")
    print(f"cached at {CACHE}")
