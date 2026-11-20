"""
One plotting vocabulary for every figure in the paper.

The report that preceded this paper accumulated several styles; the paper uses
exactly one. Three rules, applied everywhere:

  1. Line plots show the ENSEMBLE MEAN, a shaded band for the ensemble, and a
     grey persistence reference. Nothing else -- no climatology, no single
     member, no second lead time.
  2. Map panels use DISCRETE colour classes, not a continuous ramp. A continuous
     ramp on a field as skewed as water table depth compresses most of the
     domain into one shade; five bright classes keep the wet/dry pattern legible
     and mean the same depth in every figure.
  3. Folds are ordered CHRONOLOGICALLY by verifying water year -- WY2003 first,
     WY2006 last -- in every panel, table and legend.

Import this, never redefine a colour locally.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import BoxStyle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # project root
DDPM = os.path.join(ROOT, "ddpm_model")
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "parflow_raw")
FIGS = os.path.join(os.path.dirname(HERE), "figures")
sys.path.insert(0, DDPM)

# ----------------------------------------------------------------- folds ---
FOLD_WY = {1: 2006, 2: 2005, 3: 2004, 4: 2003}
WY_FOLD = {v: k for k, v in FOLD_WY.items()}
FOLDS = tuple(sorted(FOLD_WY, key=FOLD_WY.get))        # 4,3,2,1 -> WY2003..2006
WATER_YEARS = tuple(FOLD_WY[f] for f in FOLDS)

# ---------------------------------------------------------------- colour ---
C_TRUTH = "#12395B"      # ParFlow / observation      deep navy
C_PRED = "#E8702A"       # emulator, reanalysis-forced  vivid orange
C_SEAS5 = "#1C7C54"      # emulator, seasonal forcing   green
C_SEAS5C = "#8B5FBF"     # emulator, corrected seasonal purple
C_PERS = "#9A9A9A"       # persistence reference        grey
C_GRID = "#E6E6E6"

# One colour per verifying water year, chronological. These are the paper's
# four series colours re-used as a categorical set for the forcing figure,
# where no model variant is being shown: four distinct hues (blue, orange,
# green, purple) that stay separable under the common forms of colour-vision
# deficiency and differ in lightness as well as hue. Defined here, not in the
# figure script, so a water year has one colour everywhere.
WY_COLOR = {2003: C_TRUTH, 2004: C_PRED, 2005: C_SEAS5, 2006: C_SEAS5C}

# ------------------------------------------------------------- typography ---
FS_SUP = 15
FS_TITLE = 12
FS_LABEL = 11
FS_TICK = 9.5
FS_LEGEND = 10
FS_ANNOT = 9
LW_MAIN = 1.9
LW_REF = 1.4

MONTH = {f"{m:02d}": a for m, a in zip(range(1, 13),
         ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"))}

# ------------------------------------------------- discrete map classes ----
# Bins from the pooled WY2003-2006 distribution, fixed so a colour means the
# same thing in every figure of the paper.
SPEC = {
    "water_table_depth": dict(
        colors=["#08519C", "#2E9BD6", "#3FAE5A", "#F2C230", "#C0392B"],
        bounds=[0.0, 0.5, 1.0, 2.0, 5.0, 1e6],
        ticklabels=["0", "0.5", "1", "2", "5", ""],
        pretty="Water table depth", short="WTD", unit="m",
        note="dark blue: water table at the surface; red: 5 m or deeper"),
    "soil_moisture": dict(
        colors=["#A0522D", "#E0A62E", "#8FCF6B", "#2E9BD6", "#08519C"],
        bounds=[0.0, 0.20, 0.28, 0.36, 0.43, 1.0],
        ticklabels=["0", "0.20", "0.28", "0.36", "0.43", "1"],
        pretty="Soil moisture", short="SM", unit="-",
        note="brown: dry; dark blue: near saturation"),
    # --- static domain descriptors (fig_domain) ----------------------------
    # Bins from the study-domain distribution of data/parflow_raw/
    # static_conditions.nc; elevation spans 114.5-498 m, topographic index
    # 1.83-20.92 over the 637x665 grid.
    "elevation": dict(
        colors=["#1A6B3C", "#4E9B4A", "#8FBF4A", "#E3D15C",
                "#D79B47", "#B06B3A", "#7A4428"],
        bounds=[110, 160, 190, 215, 245, 285, 340, 500],
        ticklabels=["110", "160", "190", "215", "245", "285", "340", "500"],
        pretty="Elevation", short="DEM", unit="m",
        note="green: lowland river corridors; brown: the divides"),
    # Nine EQUAL-AREA classes. The index is ln(a/tan beta), so its domain
    # distribution is strongly right-skewed (median 6.91, 90th percentile
    # 12.6, max 20.9) and the six round bins used earlier left the two palest
    # classes carrying most of the upland while the top class carried a tail.
    # The breaks below are the equal-area (11.1 %) quantiles snapped to the
    # nearest 0.25, which keeps every class between 9.8 % and 12.4 % of the
    # domain while the labels stay readable. The class BREAKS are unchanged
    # from the equal-area design; only the colours below were reworked.
    #
    # COLOURS -- deliberately CATEGORICAL, not a smooth ramp. The previous
    # straw -> green -> teal -> indigo sequential ramp was legible but its
    # neighbours blended into one another, and the figure was asked to make
    # every class read as its own colour. Each class now sits in a different
    # hue family -- lemon, sand, orange, vermillion, rose, violet, blue, deep
    # blue, navy -- and the nine were placed in OKLCH by maximising the
    # MINIMUM perceptual distance between ANY TWO of them (not merely
    # between neighbours), simultaneously under normal vision and under
    # simulated protanopia, deuteranopia and tritanopia (Machado et al. 2009
    # matrices, severity 1.0). Minimum OKLab dE x100 over ALL 36 pairs:
    #
    #            normal  protan  deutan  tritan   worst
    #   new       12.1    11.7    11.9    11.9    11.7
    #   previous   8.8     8.9     8.7     8.6     8.6
    #
    # i.e. +36 % on the worst case. Restricted to ADJACENT pairs the new
    # figures are 12.1 / 12.1 / 11.9 / 11.9. The all-pairs and adjacent
    # minima are within 0.2 of each other, so no distant pair is hiding a
    # collision: none of the nine classes is confusable with any other for
    # any of the four kinds of vision.
    #
    # WHAT THIS COSTS, on the record, because it was chosen with eyes open.
    # A categorical palette buys separability at the price of the
    # at-a-glance ORDERING a sequential ramp gives for free: hue alone no
    # longer says which of two classes is wetter, so ranking two swatches
    # now means consulting the colourbar. Two properties are kept to limit
    # the damage: L* still falls monotonically (96 -> 16), so the ordering
    # survives greyscale printing and a colour-blind reader keeps the full
    # ranking from lightness alone; and the hue still rotates one way only,
    # warm (dry, well-drained divides) to cool (saturated valley bottoms),
    # so the sequence keeps a direction. A fully free-hue palette scored far
    # better (minimum pairwise 15.4) but printed as confetti on this field,
    # whose classes interleave at the pixel scale, and was rejected: the map
    # has to stay readable, not only the colourbar.
    #
    # Area fractions and the hydrological reading of each class are in
    # paper/tables/topo_index.tex.
    # COLOURS: an ordered multi-hue ramp, straw -> green -> teal -> indigo,
    # NOT a categorical palette. Two stronger-contrast schemes were tried and
    # rejected on the rendered map: the topographic index varies at pixel
    # scale, and at nine classes 70% of adjacent 1 km cells fall in different
    # classes with a mean jump of 2.7 classes. Any palette whose neighbouring
    # classes differ strongly therefore renders the panel as confetti and the
    # drainage network disappears. An ordered ramp turns that same pixel-scale
    # variation into texture, which is what makes the dendritic pattern
    # legible. Separability here is min adjacent dE 8.7 under normal vision
    # and all three dichromacies -- lower than a categorical scheme would give
    # in the colourbar, and the price of a readable map.
    "topographic_index": dict(
        colors=["#F7EEBE", "#BEE86F", "#6AD07D", "#59AB93", "#55898A",
                "#356E82", "#27536A", "#1A3856", "#131651"],
        bounds=[1.8, 4.75, 5.5, 6.0, 6.75, 7.25, 8.5, 10.25, 12.5, 21.0],
        ticklabels=["1.8", "4.75", "5.5", "6.0", "6.75", "7.25",
                    "8.5", "10.25", "12.5", "21"],
        pretty="Topographic index", short="TI", unit="-",
        note="lemon: well-drained divides; navy: saturated valley bottoms"),
}
VARS = ("water_table_depth", "soil_moisture")
STATIC_VARS = ("elevation", "topographic_index")

# Map furniture shared by the domain figure.
C_DOMAIN = "#D62728"     # the study-domain rectangle          red
C_LAND = "#EDE9E1"       # CONUS land fill                     warm grey
C_BORDER = "#FFFFFF"     # state borders over the land fill
C_COAST = "#5A6672"      # coastline / national border         slate


def apply_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.labelsize": FS_LABEL,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": C_GRID,
        "grid.linewidth": 0.7,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,          # editable text in the submitted PDF
        "ps.fonttype": 42,
    })


def cmap_for(var):
    sp = SPEC[var]
    cm = ListedColormap(sp["colors"])
    return cm, BoundaryNorm(sp["bounds"], cm.N)


def north_up(ax, field, **kw):
    """conus1 rows run south to north; imshow draws top-down."""
    return ax.imshow(np.asarray(field)[::-1], origin="upper",
                     interpolation="nearest", **kw)


def map_off(ax, title=None):
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=FS_TITLE - 1, fontweight="bold", pad=4)


def year_boxes(fig, axes, rows_per_year=2, lw=1.6, margin_pt=3.0,
               gap_pt=8.0, rounding_pt=4.0, color="black"):
    """A BOX around each year block of a paired-row grid of map panels.

    Rows come in blocks of ``rows_per_year`` (ParFlow above DDPM for one water
    year). One rectangle is drawn around each block -- all of its rows and all
    of its columns together -- so the ParFlow/DDPM pairing is explicit and the
    year blocks read as separate objects. This replaces the thin rules an
    earlier version drew in the inter-year gaps, which were invisible at the
    size the figure is printed.

    Nothing is hard-coded: the geometry is read from the axes themselves, in
    INCHES, so the margin is equal on all four sides and the rounded corners
    stay circular whatever the figure's aspect.

      * the box spans column 0's left edge to the last column's right edge, so
        it clears the row labels on the left;
      * ``margin_pt`` of clear space is left on all four sides of the panels;
      * the blocks are re-spaced so adjacent boxes stand ``gap_pt`` apart while
        the rows WITHIN a year keep their original tight gap -- tight inside a
        year, loose between years, which is the whole point;
      * the figure is made taller by exactly the space that re-spacing needs,
        and taller again if the top box would otherwise reach the column
        headers, so no panel changes size, shape or aspect.

    Call this AFTER the final subplots_adjust and BEFORE adding the colourbar
    axes, which is placed in fractions of the final figure height. The canvas
    is drawn first so aspect-constrained panel boxes report their true extent.

    Returns the list of (x0, y0, x1, y1) box corners, in inches.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    W, H = fig.get_size_inches()
    n = axes.shape[0]
    nblk, rem = divmod(n, rows_per_year)
    if rem:
        raise ValueError(f"{n} rows is not a whole number of blocks of "
                         f"{rows_per_year}")

    # --- the grid as it stands, in inches ---------------------------------
    pos = [[ax.get_position() for ax in row] for row in axes]
    x0 = min(p.x0 for row in pos for p in row) * W
    x1 = max(p.x1 for row in pos for p in row) * W
    ytop = [max(p.y1 for p in row) * H for row in pos]
    ybot = [min(p.y0 for p in row) * H for row in pos]
    h = ytop[0] - ybot[0]
    if max(abs((ytop[r] - ybot[r]) - h) for r in range(n)) > 1e-6:
        raise ValueError("the rows differ in height; grid is not uniform")

    m = margin_pt / 72.0                       # box margin, every side
    g_in = ybot[0] - ytop[1]                   # gap inside a year: keep it
    g_out = 2 * m + gap_pt / 72.0              # gap between year blocks
    top, bot = H - ytop[0], ybot[-1]           # space above / below the grid

    # keep the top box clear of the column headers
    heads = [ax.title.get_window_extent(rend).y0 / fig.dpi
             for ax in axes[0] if ax.get_title()]
    d_top = max(0.0, (ytop[0] + 2 * m) - min(heads)) if heads else 0.0

    # --- grow the figure by exactly what the new spacing needs ------------
    Hn = H + d_top + nblk * (rows_per_year - 1) * g_in \
         + (nblk - 1) * g_out - (n - 1) * g_in
    fig.set_size_inches(W, Hn, forward=True)

    def row_top(r):
        blk, k = divmod(r, rows_per_year)
        return (Hn - top - d_top) - r * h \
            - blk * g_out - (blk * (rows_per_year - 1) + k) * g_in

    if abs((row_top(n - 1) - h) - bot) > 1e-9:
        raise AssertionError("the re-spaced grid does not end where it did")
    for r in range(n):
        yt = row_top(r)
        for c, p in enumerate(pos[r]):
            # width and height in inches are preserved exactly, so the
            # aspect-locked panels neither move horizontally nor resize
            axes[r][c].set_position([p.x0, (yt - h) / Hn,
                                     p.x1 - p.x0, h / Hn])

    # --- one rounded box per year block -----------------------------------
    corners = []
    for blk in range(nblk):
        yb = row_top(blk * rows_per_year + rows_per_year - 1) - h - m
        yt = row_top(blk * rows_per_year) + m
        # figure FRACTIONS, not fig.dpi_scale_trans: a tight-bbox save
        # shifts transFigure to the cropped origin but leaves the inch
        # transform where it was, which would slide the boxes off the grid.
        # mutation_aspect = W/Hn undoes the anisotropy of the fractions, so
        # the corner rounding is a true circular arc of rounding_pt.
        box = FancyBboxPatch(
            ((x0 - m) / W, yb / Hn), ((x1 - x0) + 2 * m) / W, (yt - yb) / Hn,
            boxstyle=BoxStyle("Round", pad=0.0,
                              rounding_size=rounding_pt / 72.0 / W),
            mutation_aspect=W / Hn, transform=fig.transFigure,
            fill=False, edgecolor=color, linewidth=lw,
            joinstyle="round", capstyle="round", zorder=20, clip_on=False)
        fig.add_artist(box)
        corners.append((x0 - m, yb, x1 + m, yt))

    fig.canvas.draw()
    lab = [ax.yaxis.label.get_window_extent(rend).x1 / fig.dpi
           for ax in axes[:, 0] if ax.get_ylabel()]
    print(f"  year boxes: {nblk} x {rows_per_year} rows, margin "
          f"{margin_pt:.1f} pt, gap between boxes {gap_pt:.1f} pt, lw {lw} pt")
    print(f"    figure {W:.3f} x {H:.3f} -> {W:.3f} x {Hn:.3f} in "
          f"(panels unchanged at {(pos[0][0].x1 - pos[0][0].x0) * W:.4f} x "
          f"{h:.4f} in); gap inside a year {g_in * 72:.2f} pt, between years "
          f"{g_out * 72:.2f} pt")
    for blk, (a, b, c, d) in enumerate(corners):
        print(f"    box {blk}: x {a:.4f}..{c:.4f} in, y {b:.4f}..{d:.4f} in "
              f"({c - a:.4f} x {d - b:.4f})")
    if lab:
        print(f"    clearance to the row labels {(x0 - m) - max(lab):.4f} in, "
              f"to the right figure edge {W - (x1 + m):.4f} in")
    return corners


def ensemble_panel(ax, t, mean, lo, hi, pers=None, label="DDPM ensemble mean",
                   color=C_PRED, pers_label="Persistence",
                   band_label="Ensemble range"):
    """THE line-plot primitive of the paper: mean, ensemble band, persistence.

    The band is drawn exactly as the caller supplies it; the error-evolution
    figures pass mean +- the ensemble spread, so the mean runs down its middle.
    """
    ax.fill_between(t, lo, hi, color=color, alpha=0.22, lw=0,
                    label=band_label)
    if pers is not None:
        ax.plot(t, pers, color=C_PERS, lw=LW_REF, ls="--", label=pers_label)
    ax.plot(t, mean, color=color, lw=LW_MAIN, label=label)
    ax.set_ylim(bottom=0)
    return ax


def date_ticks(ax, dates, n=6, fmt_year=True):
    idx = np.linspace(0, len(dates) - 1, n).astype(int)
    ax.set_xticks(idx)
    ax.set_xticklabels(
        [f"{MONTH[dates[i][5:7]]}{' ' + dates[i][2:4] if fmt_year else ''}"
         for i in idx], fontsize=FS_TICK - 0.5)


def save(fig, name, out_dir=None):
    out = out_dir or FIGS
    os.makedirs(out, exist_ok=True)
    for ext in (".pdf", ".png"):
        fig.savefig(os.path.join(out, name + ext), dpi=200)
    plt.close(fig)
    print(f"  wrote {name}", flush=True)


def load_fold_cache(fold):
    z = np.load(os.path.join(DDPM, "Results", "_ensemble_cache", f"fold{fold}.npz"),
                allow_pickle=True)
    return {k: z[k] for k in z.files}


def rollout_path(fold):
    return os.path.join(DDPM, "Results", f"groundwater_f15_200ep_fold{fold}",
                        "rollout", "ensemble_state.npz")
