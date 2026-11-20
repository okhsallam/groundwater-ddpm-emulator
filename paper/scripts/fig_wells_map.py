"""
Figure: the domain's topography with every observation well on it.

The roster is whatever wells_common.load_wells() returns -- 48 at the time of
writing -- and NOTHING here is hard-coded to a count: the title, the roster
block and its column layout, the cluster insets and their magnifications are
all derived from the wells themselves, so the figure follows the network when
it grows again.

The two networks are merged -- the paper treats them as one body of evidence --
so the wells are numbered in a single sequence, the USGS sites first and the
ISWS/UIUC unconfined network after them, the same order the series caches and
the two well tables use.

The background is the ParFlow static elevation field, shaded by a low-angle
light source. The Midwest's relief is 115-500 m over 665 km, so an unshaded
ramp shows almost nothing; the hillshade is what makes the Illinois and
Wabash valleys legible, and the wells sit in them.

Geolocation is exact rather than approximate: PF-CONUS1 is a Lambert conformal
grid, and the projection below reproduces every (lat, lon) -> (row, col)
mapping in the well JSONs to within one 1 km cell, which is how the state
boundaries can be trusted against the marker positions.

WHY THERE ARE INSETS, which the fourteen-well version of this figure did not
need. The ISWS network is not spread evenly: single-linkage clustering of the
48 wells finds two tight groups -- 12 wells inside 46 x 28 km and 10 inside
45 x 52 km -- and the result is stable for any link distance from 18 to 30 km,
so it is a property of the network and not of a threshold. Inside those groups
the nearest neighbours are 5.8 and 7.3 km apart while the marker disc is
9.8 km across at domain scale, so THE DISCS THEMSELVES MERGE. No amount of
leader-line work can fix that: a leader can only point at a pile. Those two
groups are therefore drawn again, magnified, in the two insets at lower right,
and on the main map their wells carry a disc and a keyed box but no number.
Every well is numbered exactly once across the main map and the two insets,
which is asserted below rather than left to inspection.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P
import wells_common as C
P.apply_style()

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.patches import Rectangle
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr

ROOT = P.ROOT
# PF-CONUS1 grid, validated against the well roster (max error 1.1 km)
PROJ4 = ("+proj=lcc +lat_1=33 +lat_2=45 +lat_0=39 +lon_0=-96 +x_0=0 +y_0=0 "
         "+a=6378137 +b=6356752.31 +units=m +no_defs")
ORIGIN = (-1885055.4994, -604245.0001)      # global cell (0,0) corner, metres
BOUNDS = (2053, 565)                        # this domain's lower-left cell
CELL = 1000.0

FIG_W, FIG_H = 12.4, 8.2

MARK_S = 56          # marker area; the number is set beside the disc, not in it
FS_NUM = P.FS_TICK - 1.5     # the well number drawn on the map
NUM_STROKE = 2.2             # white halo around it, points
LABEL_RINGS = 14     # how far out the search for a free label slot may go

# The clearance a number needs is a property of the TYPE, not of the map, so
# it is derived from the font size and the marker size at draw time (see
# _label_geometry) instead of being carried as a pair of hand-tuned distances
# in metres. The earlier fixed SEP = 14 km was tuned when the network had 14
# wells and a roomier canvas; at 48 wells it let numbers sit on discs they did
# not belong to, because 0.6 x 14 km is less than half a two-digit number plus
# a marker radius. These are multipliers on the measured type box instead, so
# they hold on the main map and inside an inset at three times the scale.
SEP_PAD = 1.10       # label-to-label clearance, x the type box
DISC_PAD = 1.05      # label-to-disc clearance, x (half type box + disc radius)
STEP_PAD = 1.18      # first search ring, x the disc clearance
LEADER_AMBIG = 1.35  # draw a leader if another disc is this close, relatively

# Cluster detection. Single linkage, and the answer does not depend on the
# link distance anywhere in 18-30 km (checked), so 24 km is not a tuned value.
CLUSTER_LINK = 24000.0
CLUSTER_MIN = 4             # fewer than this is not worth an inset
CLUSTER_MARGIN = 9000.0     # context around a cluster in its inset

# The roster block, in FIGURE fractions. Everything about it is derived from
# these numbers and len(ws): the number of columns is whatever is needed to
# keep the list between ROSTER_TOP and ROSTER_FLOOR, and the column pitch is
# whatever divides ROSTER_X0..ROSTER_X1 into that many columns. A roster that
# grew from 14 to 48 entries under a FIXED rows-per-column therefore cannot
# silently walk off the right-hand edge of the canvas, which is what the
# hard-coded 23 rows would have done here: ceil(48/23) is three columns, and
# the third would have started at x = 1.004.
ROSTER_X0 = 0.755    # left edge of the block (the colourbar ends near 0.70)
ROSTER_X1 = 0.997    # nothing may be drawn to the right of this
ROSTER_TOP = 0.905   # baseline of the first entry
ROSTER_FLOOR = 0.515  # the insets own everything below this
ROSTER_DY = 0.0245   # line pitch

# The two inset panels, lower right, below the roster.
INSET_BOXES = ((0.755, 0.280, 0.242, 0.176),
               (0.755, 0.050, 0.242, 0.176))
MAP_BOX = (0.045, 0.045, 0.60, 0.885)


# --------------------------------------------------------------- clusters ---
def _clusters(px, py, link=CLUSTER_LINK, min_n=CLUSTER_MIN):
    """Single-linkage groups of wells, largest first, then north to south."""
    n = len(px)
    par = list(range(n))

    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]
            a = par[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(px[i] - px[j], py[i] - py[j]) < link:
                par[find(i)] = find(j)
    g = {}
    for i in range(n):
        g.setdefault(find(i), []).append(i)
    out = [sorted(v) for v in g.values() if len(v) >= min_n]
    return sorted(out, key=lambda v: -np.mean([py[i] for i in v]))


# ------------------------------------------------------------ label logic ---
def _label_geometry(ax_w_in, span_m, n_digits=2):
    """How much room a well number actually takes, in map metres.

    Everything downstream is measured, not guessed, so that changing the
    canvas size, the number font or the inset scale cannot quietly
    reintroduce overlaps:

      m_per_pt   the map scale, from the axes width in inches and the
                 east-west span of the view in metres
      box        the white-haloed two-digit number, width x height
      disc       the marker radius plus its white rim

    Returns (sep_x, sep_y, disc_x, disc_y, step, disc_r) in metres. The x and
    y clearances differ by nearly a factor of two -- a two-digit number is far
    wider than it is tall -- which the single isotropic SEP could not express,
    and which is why numbers used to collide side-to-side while leaving
    daylight above and below.
    """
    m_per_pt = span_m / (ax_w_in * 72.0)
    box_w = (n_digits * FS_NUM * 0.64 + NUM_STROKE) * m_per_pt
    box_h = (FS_NUM * 0.98 + NUM_STROKE) * m_per_pt
    disc_r = (np.sqrt(MARK_S) / 2.0 + 1.0) * m_per_pt
    return (SEP_PAD * box_w, SEP_PAD * box_h,
            DISC_PAD * (box_w / 2.0 + disc_r),
            DISC_PAD * (box_h / 2.0 + disc_r),
            STEP_PAD * max(DISC_PAD * (box_w / 2.0 + disc_r),
                           DISC_PAD * (box_h / 2.0 + disc_r)),
            disc_r)


def _label_offsets(px, py, which, geom):
    """Place the numbers in ``which`` clear of EVERY disc and of each other.

    ``px``/``py`` are all the wells in the panel, so a number never lands on a
    marker even when that marker is one of the unnumbered cluster members.
    Each label is searched for on rings of increasing radius and the first
    free position is taken; a fixed nudge direction only moves a collision.
    """
    sep_x, sep_y, disc_x, disc_y, step, _ = geom
    lab = {}
    rings = {}
    # densest first: the wells with the most neighbours get the pick of the
    # free space, rather than being painted into a corner by isolated ones
    crowd = {i: sum(abs(px[i] - px[j]) < 4 * sep_x
                    and abs(py[i] - py[j]) < 4 * sep_y for j in range(len(px)))
             for i in which}
    for i in sorted(which, key=lambda i: (-crowd[i], px[i], py[i])):
        best = None
        for ring in range(1, LABEL_RINGS + 1):
            r = step * ring
            # 16 directions, offset on alternate rings, so a crowded panel has
            # twice the candidate slots the old 12 gave it
            for k in range(16):
                th = (np.pi / 8.0) * k + (np.pi / 16.0 if ring % 2 else 0.0)
                cx, cy = px[i] + r * np.cos(th), py[i] + r * np.sin(th)
                if any(abs(cx - a) < sep_x and abs(cy - b) < sep_y
                       for a, b in lab.values()):
                    continue
                if any(abs(cx - px[j]) < disc_x and abs(cy - py[j]) < disc_y
                       for j in range(len(px))):
                    continue
                best = (cx, cy)
                break
            if best:
                rings[i] = ring
                break
        # A label with no free slot would sit on another one and the map would
        # assert a well number at a place it does not mean. That is a
        # correctness failure, so it stops the figure rather than being drawn.
        assert best, (f"well index {i}: no free label position within "
                      f"{LABEL_RINGS} rings; raise LABEL_RINGS or FS_NUM")
        lab[i] = best

    # Independent re-check of the FINISHED layout, not of the search.
    for i in which:
        for j in which:
            if j > i:
                assert (abs(lab[i][0] - lab[j][0]) >= sep_x
                        or abs(lab[i][1] - lab[j][1]) >= sep_y), ("labels", i, j)
        for j in range(len(px)):
            assert (abs(lab[i][0] - px[j]) >= disc_x
                    or abs(lab[i][1] - py[j]) >= disc_y), ("on disc", i, j)
    return lab, rings


def _draw_wells(ax, crs, ws, px, py, show, number, geom):
    """Discs for ``show``, numbers for ``number``, leaders where ambiguous."""
    sx = [px[i] for i in show]
    sy = [py[i] for i in show]
    ax.scatter(sx, sy, s=MARK_S, transform=crs, zorder=8,
               facecolor=P.C_PRED, edgecolor="white", linewidth=1.0, alpha=.97)
    ax.scatter(sx, sy, s=MARK_S, transform=crs, zorder=9, facecolor="none",
               edgecolor="#2B2B2B", linewidth=0.5)

    lab, rings = _label_offsets(px, py, number, geom)
    step, disc_r = geom[4], geom[5]
    n_lead = 0
    for i in number:
        a, b = lab[i]
        d_own = float(np.hypot(a - px[i], b - py[i]))
        # The test is AMBIGUITY, not distance: a leader is drawn whenever some
        # other well's disc is within LEADER_AMBIG times the distance to its
        # own, so that "nearest disc" is no longer a safe reading. Out in open
        # country that is none of them, and those numbers sit quietly beside
        # their marker with no line at all.
        d_other = min([float(np.hypot(a - px[j], b - py[j]))
                       for j in show if j != i] or [np.inf])
        if d_other < LEADER_AMBIG * d_own or d_own > 2.0 * step:
            t = disc_r / max(d_own, 1.0)        # start at the rim, not the hub
            ax.plot([px[i] + (a - px[i]) * t, a], [py[i] + (b - py[i]) * t, b],
                    transform=crs, color="#2B2B2B", lw=0.6, zorder=9.4,
                    solid_capstyle="round",
                    path_effects=[pe.withStroke(linewidth=1.9,
                                                foreground="white", alpha=.95)])
            n_lead += 1
        ax.text(a, b, str(ws[i]["no"]), transform=crs, ha="center",
                va="center", fontsize=FS_NUM, fontweight="bold",
                color="#12395B", zorder=10,
                path_effects=[pe.withStroke(linewidth=NUM_STROKE,
                                            foreground="white")])
    return n_lead, max(rings.values()) if rings else 0


def _scalebar(ax, crs, win, length_m, label):
    bx = win[0] + 0.06 * (win[1] - win[0])
    by = win[2] + 0.085 * (win[3] - win[2])
    ax.plot([bx, bx + length_m], [by, by], color="#1A1A1A", lw=3.2, zorder=11,
            transform=crs, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=6, foreground="white")])
    ax.text(bx + length_m / 2, by + 0.035 * (win[3] - win[2]), label,
            transform=crs, ha="center", fontsize=P.FS_TICK - 2.0,
            fontweight="bold", color="#1A1A1A", zorder=11,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])


def _fit_window(px, py, idx, aspect, margin=CLUSTER_MARGIN):
    """Bounding box of ``idx`` + margin, grown to the panel's aspect ratio."""
    xs = [px[i] for i in idx]
    ys = [py[i] for i in idx]
    x0, x1 = min(xs) - margin, max(xs) + margin
    y0, y1 = min(ys) - margin, max(ys) + margin
    w, h = x1 - x0, y1 - y0
    if w / h < aspect:                      # too tall: widen it
        extra = (h * aspect - w) / 2.0
        x0, x1 = x0 - extra, x1 + extra
    else:                                   # too wide: heighten it
        extra = (w / aspect - h) / 2.0
        y0, y1 = y0 - extra, y1 + extra
    return (x0, x1, y0, y1)


# ------------------------------------------------------------------ figure ---
def main():
    ws = C.load_wells()
    dem = xr.open_dataset(os.path.join(P.RAW, "static_conditions.nc")).elevation.values
    ny, nx = dem.shape
    x0 = ORIGIN[0] + BOUNDS[0] * CELL
    y0 = ORIGIN[1] + BOUNDS[1] * CELL
    extent = (x0, x0 + nx * CELL, y0, y0 + ny * CELL)
    pad = 18 * CELL          # so a marker on the domain edge is not half cut
    view = (extent[0] - pad, extent[1] + pad, extent[2] - pad, extent[3] + pad)
    crs = ccrs.LambertConformal(
        central_longitude=-96, central_latitude=39,
        standard_parallels=(33, 45),
        globe=ccrs.Globe(semimajor_axis=6378137, semiminor_axis=6356752.31))
    # identical to PROJ4 above to within 1e-7 m

    # truncated terrain: the blue end of matplotlib's "terrain" reads as water
    base = plt.get_cmap("terrain")
    cm = ListedColormap(base(np.linspace(0.22, 0.95, 256)))
    vmin, vmax = 110.0, 500.0
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(dem, cmap=cm, blend_mode="soft", vert_exag=35,
                   dx=CELL, dy=CELL, vmin=vmin, vmax=vmax)

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes(list(MAP_BOX), projection=crs)
    P.north_up(ax, rgb, extent=extent, transform=crs, zorder=1)
    ax.set_extent(view, crs=crs)

    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "lakes", "50m"), facecolor="#AFCFE8",
        edgecolor="#5B87A8", linewidth=0.6, zorder=2)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "physical", "rivers_lake_centerlines", "50m"), facecolor="none",
        edgecolor="#6FA3C7", linewidth=0.55, alpha=.85, zorder=3)
    ax.add_feature(cfeature.NaturalEarthFeature(
        "cultural", "admin_1_states_provinces_lines", "50m"), facecolor="none",
        edgecolor="#333333", linewidth=0.9, alpha=.75, zorder=4)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="#FFFFFF",
                      alpha=.5, linestyle=":", zorder=5,
                      x_inline=False, y_inline=False,
                      xlocs=np.arange(-94, -84, 2),
                      ylocs=np.arange(39, 45, 1))
    gl.top_labels = gl.right_labels = False
    gl.rotate_labels = False
    gl.xlabel_style = gl.ylabel_style = dict(size=P.FS_TICK - 1,
                                             color="#555555")

    # state names, placed in open country rather than over a well.
    # ILLINOIS was at (-89.55, 40.35), which is the middle of the ten-well
    # MTOW cluster and is now inside its keyed box; -88.75 then landed on the
    # box's right-hand edge. It moves to the open country north-east of it.
    for name, lon, lat in (("ILLINOIS", -88.45, 41.35), ("INDIANA", -86.6, 40.6),
                           ("IOWA", -92.6, 42.4), ("MISSOURI", -92.6, 39.3),
                           ("WISCONSIN", -89.9, 43.4), ("MICHIGAN", -86.0, 42.6)):
        ax.text(lon, lat, name, transform=ccrs.PlateCarree(), ha="center",
                va="center", fontsize=P.FS_TICK - 1.5, color="#3A3A3A",
                alpha=.8, fontweight="bold", zorder=6,
                path_effects=[pe.withStroke(linewidth=2.6, foreground="white",
                                            alpha=.7)])

    # wells: grid cell -> projected metre, cell centre
    px = [x0 + (w["x"] + 0.5) * CELL for w in ws]
    py = [y0 + (w["y"] + 0.5) * CELL for w in ws]
    n = len(ws)

    # ------------------------------------------------------------ insets ---
    groups = _clusters(px, py)
    assert len(groups) <= len(INSET_BOXES), (
        f"{len(groups)} clusters but only {len(INSET_BOXES)} inset panels")
    keys = "ABCDEFGH"
    ax_w_in = MAP_BOX[2] * FIG_W
    main_m_per_pt = (view[1] - view[0]) / (ax_w_in * 72.0)

    # Every well that FALLS IN an inset window is drawn and numbered there and
    # nowhere else -- not merely the cluster members -- so a bystander well
    # inside the box is not silently numbered twice. Admitting a bystander
    # moves the window, which can admit another, so the window is re-fitted to
    # whatever it now contains until it stops changing; otherwise the last
    # well let in sits half-clipped on the frame with no margin of its own.
    windows, members = [], []
    for g, box in zip(groups, INSET_BOXES):
        aspect = (box[2] * FIG_W) / (box[3] * FIG_H)
        inside = list(g)
        for _ in range(6):
            win = _fit_window(px, py, inside, aspect)
            got = [i for i in range(n)
                   if win[0] <= px[i] <= win[1] and win[2] <= py[i] <= win[3]]
            if got == inside:
                break
            inside = got
        else:
            raise AssertionError("inset window did not settle")
        windows.append(win)
        members.append(inside)

    in_inset = {}
    for k, mem in enumerate(members):
        for i in mem:
            assert i not in in_inset, (
                f"well {ws[i]['no']} falls in two inset windows")
            in_inset[i] = k
    on_main = [i for i in range(n) if i not in in_inset]

    # keyed boxes on the main map, drawn where the inset actually looks
    for k, win in enumerate(windows):
        ax.add_patch(Rectangle((win[0], win[2]), win[1] - win[0],
                               win[3] - win[2], transform=crs, zorder=9.6,
                               facecolor="none", edgecolor="#1A1A1A",
                               linewidth=1.1))
        ax.text(win[1] + 0.012 * (view[1] - view[0]), win[3],
                keys[k], transform=crs, ha="left", va="center",
                fontsize=P.FS_LABEL, fontweight="bold", color="#1A1A1A",
                zorder=11,
                path_effects=[pe.withStroke(linewidth=2.8, foreground="white")])

    geom = _label_geometry(ax_w_in, view[1] - view[0])
    n_lead, ring = _draw_wells(ax, crs, ws, px, py, list(range(n)), on_main,
                               geom)
    print(f"  main map: {len(on_main)} of {n} wells numbered, {n_lead} with a "
          f"leader, deepest search ring {ring}")

    for k, (box, win) in enumerate(zip(INSET_BOXES, windows)):
        iax = fig.add_axes(list(box), projection=crs)
        P.north_up(iax, rgb, extent=extent, transform=crs, zorder=1)
        iax.set_extent(win, crs=crs)
        iax.add_feature(cfeature.NaturalEarthFeature(
            "physical", "rivers_lake_centerlines", "10m"), facecolor="none",
            edgecolor="#6FA3C7", linewidth=0.6, alpha=.9, zorder=3)
        iax.add_feature(cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces_lines", "10m"),
            facecolor="none", edgecolor="#333333", linewidth=0.9, alpha=.75,
            zorder=4)
        mem = sorted(members[k])
        igeom = _label_geometry(box[2] * FIG_W, win[1] - win[0])
        # re-index onto the members only: ws, px and py must be sliced
        # TOGETHER or the inset prints a local index as the well number
        mw = [ws[i] for i in mem]
        mx = [px[i] for i in mem]
        my = [py[i] for i in mem]
        loc = list(range(len(mem)))
        nl, rg = _draw_wells(iax, crs, mw, mx, my, loc, loc, igeom)
        assert [w["no"] for w in mw] == [ws[i]["no"] for i in mem]
        _scalebar(iax, crs, win, 20000.0, "20 km")
        iax.set_xticks([]); iax.set_yticks([]); iax.grid(False)
        for s in iax.spines.values():
            s.set_visible(True); s.set_edgecolor("#1A1A1A"); s.set_linewidth(1.1)

        mag = main_m_per_pt / ((win[1] - win[0]) / (box[2] * FIG_W * 72.0))
        gw = (max(px[i] for i in mem) - min(px[i] for i in mem)) / 1000.0
        gh = (max(py[i] for i in mem) - min(py[i] for i in mem)) / 1000.0
        fig.text(box[0], box[1] + box[3] + 0.008, keys[k],
                 fontsize=P.FS_LABEL, fontweight="bold", color="#1A1A1A",
                 va="bottom", ha="left")
        fig.text(box[0] + 0.016, box[1] + box[3] + 0.009,
                 f"{len(mem)} wells in {gw:.0f} $\\times$ {gh:.0f} km, "
                 f"at {mag:.1f}$\\times$ map scale",
                 fontsize=P.FS_TICK - 2.0, color="#2B2B2B", va="bottom",
                 ha="left")
        print(f"  inset {keys[k]}: {len(mem)} wells, "
              f"{(win[1] - win[0]) / 1000:.0f} x {(win[3] - win[2]) / 1000:.0f} km "
              f"window, {mag:.2f}x, {nl} leaders, deepest ring {rg}")

    # every well numbered exactly once, on the main map or in one inset
    drawn = sorted(on_main + list(in_inset))
    assert drawn == list(range(n)), "a well is numbered twice or not at all"

    sm = plt.cm.ScalarMappable(cmap=cm,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    cax = fig.add_axes([0.665, 0.13, 0.016, 0.62])
    cb = fig.colorbar(sm, cax=cax, ticks=np.arange(150, 501, 50))
    cb.set_label("Land surface elevation [m]", fontsize=P.FS_LABEL,
                 fontweight="bold")
    cb.outline.set_linewidth(0.6)

    # 1 km cells, so a scale bar is exact by construction
    bx, by, L = x0 + 24 * CELL, y0 + 26 * CELL, 100 * CELL
    ax.plot([bx, bx + L], [by, by], color="#1A1A1A", lw=3.2, zorder=11,
            transform=crs, solid_capstyle="butt",
            path_effects=[pe.withStroke(linewidth=6, foreground="white")])
    ax.text(bx + L / 2, by + 11 * CELL, "100 km", transform=crs, ha="center",
            fontsize=P.FS_TICK - 0.5, fontweight="bold", color="#1A1A1A",
            zorder=11,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax.annotate("", xy=(0.052, 0.965), xytext=(0.052, 0.885),
                xycoords="axes fraction", zorder=11,
                arrowprops=dict(arrowstyle="-|>", color="#1A1A1A", lw=2.0,
                                shrinkA=0, shrinkB=0))
    ax.text(0.052, 0.975, "N", transform=ax.transAxes, ha="center",
            va="bottom", fontsize=P.FS_LABEL, fontweight="bold",
            color="#1A1A1A", zorder=11,
            path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])

    P.map_off(ax)
    gl.bottom_labels = gl.left_labels = True

    # The roster, in the single merged sequence the paper uses throughout.
    # Rows per column and the column pitch are SOLVED for, not assumed: fit as
    # many entries into one column as ROSTER_TOP..ROSTER_FLOOR allows, take
    # however many columns that needs, and share ROSTER_X0..ROSTER_X1 between
    # them. At 48 wells this is three columns of 16; the block cannot overrun
    # the canvas, reach back into the colourbar, or drop into the insets,
    # whatever the roster becomes.
    fig.text(ROSTER_X0, 0.945, "Observation wells", fontsize=P.FS_LABEL,
             fontweight="bold", va="top", ha="left")
    rows_max = int((ROSTER_TOP - ROSTER_FLOOR) / ROSTER_DY) + 1
    ncol = int(np.ceil(len(ws) / rows_max))
    nrow = int(np.ceil(len(ws) / ncol))          # balance the last column
    cw = (ROSTER_X1 - ROSTER_X0) / ncol
    # Fit the type to the column rather than guessing at it: measure the
    # widest "NN  Name" the roster actually contains and scale the font so it
    # just fills the pitch. Text width is linear in point size, so one
    # measurement gives the answer. Capped so a short roster does not set in
    # display type, and floored so an impossible one fails loudly instead of
    # shrinking to nothing.
    gap = 0.005
    rend = fig.canvas.get_renderer()
    fig_px = FIG_W * fig.dpi

    def row_width(fs_try):
        probe = [fig.text(0, 0, f"{max(w['no'] for w in ws)}", fontsize=fs_try,
                          fontweight="bold"),
                 fig.text(0, 0, max((w["pretty"] for w in ws), key=len),
                          fontsize=fs_try)]
        wpx = sum(t.get_window_extent(rend).width for t in probe)
        for t in probe:
            t.remove()
        return wpx / fig_px + gap

    fs0 = 8.0
    fs = min(P.FS_TICK, fs0 * (cw * 0.92) / row_width(fs0))  # 8% gutter
    assert fs >= 5.5, f"roster will not fit: best font {fs:.1f} pt"
    for k, w in enumerate(ws):
        col, row = divmod(k, nrow)
        x = ROSTER_X0 + 0.011 + col * cw
        y = ROSTER_TOP - row * ROSTER_DY
        fig.text(x, y, f"{w['no']}", fontsize=fs, fontweight="bold",
                 va="center", ha="right", color="#12395B")
        fig.text(x + gap, y, w["pretty"], fontsize=fs,
                 va="center", ha="left", color="#2B2B2B")
    assert ROSTER_TOP - (nrow - 1) * ROSTER_DY >= ROSTER_FLOOR, nrow
    print(f"  roster: {len(ws)} wells in {ncol} column(s) of {nrow} at "
          f"{fs:.1f} pt, bottom entry at y={ROSTER_TOP - (nrow - 1) * ROSTER_DY:.3f}, "
          f"widest row {row_width(fs):.4f} of {cw:.4f} column pitch")

    fig.text(0.02, 0.975, "Study domain: land surface elevation and the "
             f"{len(ws)} observation wells", fontsize=P.FS_TITLE + 1,
             fontweight="bold", va="top", ha="left")
    print(f"  drew {len(ws)} wells "
          f"({sum(w['net'] == 'USGS' for w in ws)} USGS, "
          f"{sum(w['net'] == 'UIUC' for w in ws)} ISWS/UIUC)")
    P.save(fig, "fig_wells_map")


if __name__ == "__main__":
    main()
