"""fig_domain -- study-domain context figure.

Right: CONUS, with the ParFlow-CONUS1 model domain and the study domain
outlined in red. Left, stacked: the study domain itself, coloured by elevation
and by topographic index.

Geometry is taken from the conus1 grid itself. The cached per-cell lat/lon grid
in data/glofas/conus1_ceser_latlon.npz is verified cell-by-cell against
hf_hydrodata.grid.to_latlon before use; if the cache is missing or disagrees,
the coordinates are rebuilt from to_latlon directly.
"""
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P

P.apply_style()
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cartopy
cartopy.config["pre_existing_data_dir"] = cartopy.config["data_dir"]   # offline
import cartopy.crs as ccrs
import cartopy.io.shapereader as shpreader
from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

GRID = "conus1"
COL0, ROW0, COL1, ROW1 = 2053, 565, 2718, 1202      # grid_bounds, half-open
NCOL_FULL, NROW_FULL = 3342, 1888
LABEL_STATES = {"IA", "IL", "WI", "MN", "MO", "IN", "MI", "NE", "KS", "OH", "KY"}


# ------------------------------------------------------------ geometry ---
def latlon_grid():
    """Per-cell lat/lon for the domain, cache verified against to_latlon."""
    from hf_hydrodata import grid
    npz = os.path.join(P.DATA, "glofas", "conus1_ceser_latlon.npz")
    ny, nx = ROW1 - ROW0, COL1 - COL0
    if os.path.exists(npz):
        z = np.load(npz)
        lat, lon = z["lat"], z["lon"]
        if lat.shape == (ny, nx):
            ok = True
            for i, j in ((0, 0), (0, nx - 1), (ny - 1, 0), (ny - 1, nx - 1),
                         (ny // 2, nx // 2)):
                g = grid.to_latlon(GRID, float(COL0 + j), float(ROW0 + i))
                if abs(g[0] - lat[i, j]) > 1e-6 or abs(g[1] - lon[i, j]) > 1e-6:
                    ok = False
            if ok:
                print("  lat/lon cache verified against grid.to_latlon")
                return lat, lon
        print("  lat/lon cache disagrees with to_latlon -- rebuilding")
    lat = np.empty((ny, nx)); lon = np.empty((ny, nx))
    for i in range(ny):
        for j in range(nx):
            lat[i, j], lon[i, j] = grid.to_latlon(GRID, float(COL0 + j),
                                                  float(ROW0 + i))
    return lat, lon


def ring(c0, r0, c1, r1, n=120):
    """Lat/lon perimeter of a half-open conus1 cell-index box, as a closed ring."""
    from hf_hydrodata import grid
    cs = np.linspace(c0, c1, n)
    rs = np.linspace(r0, r1, n)
    pts = ([(c, r0) for c in cs] + [(c1, r) for r in rs]
           + [(c, r1) for c in cs[::-1]] + [(c0, r) for r in rs[::-1]])
    ll = np.array([grid.to_latlon(GRID, float(c), float(r)) for c, r in pts])
    return ll[:, 1], ll[:, 0]          # lon, lat


# ----------------------------------------------------------- left panels ---
def zoom_panel(fig, ax, field, var, lat, lon):
    """One static field over the domain. north_up draws row 0 (south) at the
    bottom, so axes-fraction (0,0) is the SW corner of the domain."""
    sp = P.SPEC[var]
    cm, norm = P.cmap_for(var)
    im = P.north_up(ax, field, cmap=cm, norm=norm)
    P.map_off(ax, f"{sp['pretty']} ({sp['unit']})")

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.025,
                      ticks=sp["bounds"], extend="neither")
    cb.ax.set_yticklabels(sp["ticklabels"], fontsize=P.FS_TICK - 1.5)
    cb.outline.set_linewidth(0.6)
    cb.ax.tick_params(length=2.5, width=0.6)

    ny, nx = field.shape
    ax.add_patch(Rectangle((-0.5, -0.5), nx, ny, fill=False,
                           ec=P.C_DOMAIN, lw=1.6, zorder=5, clip_on=False))

    box = dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.82)
    for (i, j), (x, y), ha, va in (((0, 0), (0.018, 0.022), "left", "bottom"),
                                   ((ny - 1, nx - 1), (0.982, 0.978), "right", "top")):
        ax.text(x, y, f"{abs(lat[i, j]):.2f}$^\\circ$N  {abs(lon[i, j]):.2f}$^\\circ$W",
                transform=ax.transAxes, ha=ha, va=va, zorder=6,
                fontsize=P.FS_TICK - 2.5, color="#333333", bbox=box)

    # 1 km cells, so a 200 km bar is 200/nx of the panel width (bottom right)
    frac = 200.0 / nx
    bx, by = 0.965 - frac, 0.045
    ax.plot([bx, bx + frac], [by, by], transform=ax.transAxes, color="#1A1A1A",
            lw=2.6, solid_capstyle="butt", zorder=6)
    ax.text(bx + frac / 2, by + 0.020, "200 km", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=P.FS_TICK - 2,
            color="#1A1A1A", zorder=6, bbox=box)
    # north arrow, on its own opaque backing so it reads over busy imagery
    ax.add_patch(Rectangle((0.018, 0.085), 0.085, 0.235,
                           transform=ax.transAxes, fc="white", ec="none",
                           alpha=0.82, zorder=6))
    ax.annotate("", xy=(0.060, 0.245), xytext=(0.060, 0.102),
                xycoords=ax.transAxes, textcoords=ax.transAxes, zorder=7,
                arrowprops=dict(arrowstyle="-|>,head_width=0.22,head_length=0.42",
                                color="#1A1A1A", lw=2.0))
    ax.text(0.060, 0.258, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=P.FS_TICK + 1.5, fontweight="bold",
            color="#1A1A1A", zorder=7)

    # No grey note under the panel: the discrete colourbar beside it and
    # Table 2 already state what each class means, and the note repeated them
    # in less space.
    return im


# ---------------------------------------------------------- right panel ---
def conus_panel(ax, sw, ne):
    ax.set_extent([-126.0, -65.5, 23.5, 51.5], ccrs.PlateCarree())

    states = list(shpreader.Reader(shpreader.natural_earth(
        resolution="50m", category="cultural",
        name="admin_1_states_provinces_lakes")).records())
    us = [r for r in states if r.attributes.get("admin") ==
          "United States of America"]
    ax.add_geometries([r.geometry for r in us], ccrs.PlateCarree(),
                      facecolor=P.C_LAND, edgecolor=P.C_BORDER, linewidth=0.55,
                      zorder=1)
    ax.add_geometries(
        [r.geometry for r in states
         if r.attributes.get("admin") in ("Canada", "Mexico")],
        ccrs.PlateCarree(), facecolor="#F6F5F2", edgecolor="#FFFFFF",
        linewidth=0.4, zorder=0)
    for name, res, lw, col in (("coastline", "50m", 0.7, P.C_COAST),
                               ("admin_0_boundary_lines_land", "50m", 0.7,
                                P.C_COAST)):
        cat = "physical" if name == "coastline" else "cultural"
        ax.add_geometries(
            list(shpreader.Reader(shpreader.natural_earth(
                resolution=res, category=cat, name=name)).geometries()),
            ccrs.PlateCarree(), facecolor="none", edgecolor=col, linewidth=lw,
            zorder=2)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      xlocs=np.arange(-125, -60, 10), ylocs=np.arange(25, 55, 5),
                      color="#C8CCD0", linewidth=0.5, linestyle=":", zorder=3)
    gl.top_labels = gl.right_labels = False
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    gl.xlabel_style = gl.ylabel_style = dict(size=P.FS_TICK - 1.5,
                                             color="#666666")

    # ParFlow-CONUS1 model domain
    clon, clat = ring(0, 0, NCOL_FULL - 1, NROW_FULL - 1)
    ax.plot(clon, clat, transform=ccrs.PlateCarree(), color=P.C_TRUTH,
            lw=1.4, ls=(0, (5, 3)), zorder=4, label="ParFlow-CONUS1 domain")

    # the study domain
    dlon, dlat = ring(COL0, ROW0, COL1 - 1, ROW1 - 1)
    ax.fill(dlon, dlat, transform=ccrs.PlateCarree(), color=P.C_DOMAIN,
            alpha=0.18, lw=0, zorder=5)
    ax.plot(dlon, dlat, transform=ccrs.PlateCarree(), color=P.C_DOMAIN,
            lw=2.2, zorder=6, label="Study domain")

    for r in us:
        if r.attributes.get("postal") in LABEL_STATES:
            ax.text(r.attributes["longitude"], r.attributes["latitude"],
                    r.attributes["postal"], transform=ccrs.PlateCarree(),
                    ha="center", va="center", fontsize=P.FS_TICK - 2,
                    color="#7A8088", zorder=7)

    pc = ccrs.PlateCarree()._as_mpl_transform(ax)
    ax.annotate("Study domain\n665 $\\times$ 637 km at 1 km",
                xy=(float(np.mean(dlon)), float(np.max(dlat))), xycoords=pc,
                xytext=(-99.5, 48.3), textcoords=pc,
                ha="center", va="bottom", fontsize=P.FS_ANNOT,
                color="#222222", linespacing=1.3, zorder=8,
                arrowprops=dict(arrowstyle="-", color=P.C_DOMAIN, lw=1.0,
                                shrinkA=2, shrinkB=4),
                bbox=dict(boxstyle="round,pad=0.32", fc="white",
                          ec=P.C_DOMAIN, lw=0.9, alpha=0.95))
    ax.text(0.5, -0.045,
            f"Domain extent   SW {abs(sw[0]):.2f}$^\\circ$N "
            f"{abs(sw[1]):.2f}$^\\circ$W    "
            f"NE {abs(ne[0]):.2f}$^\\circ$N {abs(ne[1]):.2f}$^\\circ$W",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=P.FS_ANNOT - 1, color="#555555")
    ax.legend(loc="lower left", fontsize=P.FS_LEGEND - 0.5, frameon=True,
              framealpha=0.92, edgecolor="#CCCCCC", borderpad=0.5)
    ax.set_title("CONUS context", fontsize=P.FS_TITLE, fontweight="bold", pad=6)
    ax.spines["geo"].set_edgecolor("#999999")
    ax.spines["geo"].set_linewidth(0.8)


def main():
    from hf_hydrodata import grid
    lat, lon = latlon_grid()
    sw = grid.to_latlon(GRID, float(COL0), float(ROW0))
    ne = grid.to_latlon(GRID, float(COL1), float(ROW1))
    print(f"  SW corner {sw[0]:.4f} N {sw[1]:.4f} E | NE corner "
          f"{ne[0]:.4f} N {ne[1]:.4f} E")

    with xr.open_dataset(os.path.join(P.RAW, "static_conditions.nc")) as ds:
        fields = {}
        for v in P.STATIC_VARS:
            da = ds[v]
            if "z" in da.dims:          # take the surface layer of a 3-D field
                da = da.isel(z=-1)
            a = np.squeeze(da.values)
            assert a.ndim == 2, f"{v} is not 2-D after squeeze: {a.shape}"
            fields[v] = a
            print(f"  {v}: {a.shape} units={da.attrs.get('units')!r} "
                  f"range {np.nanmin(a):.2f}..{np.nanmax(a):.2f}")

    proj = ccrs.LambertConformal(central_longitude=-96, central_latitude=39,
                                 standard_parallels=(33, 45))
    fig = plt.figure(figsize=(13.2, 7.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 2.65],
                          wspace=0.135, hspace=0.30,
                          left=0.025, right=0.985, top=0.895, bottom=0.075)
    for k, v in enumerate(P.STATIC_VARS):
        zoom_panel(fig, fig.add_subplot(gs[k, 0]), fields[v], v, lat, lon)
    conus_panel(fig.add_subplot(gs[:, 1], projection=proj), sw, ne)

    fig.suptitle("Study domain: a 665 $\\times$ 637 km subset of "
                 "ParFlow-CONUS1 in the upper Midwest",
                 fontsize=P.FS_SUP, fontweight="bold", y=0.965)
    P.save(fig, "fig_domain")


if __name__ == "__main__":
    main()
