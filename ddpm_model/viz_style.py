"""
Shared presentation-quality plotting style for the CESER groundwater figures.

Two things every figure here must get right:

1. NORTH-UP ORIENTATION. The conus1 grid's y index increases NORTHWARD (verified
   2026-09-23: the flat ~176 m Lake Michigan surface sits at high row index, and
   the lake is geographically NE of the CESER domain). matplotlib's imshow
   defaults to origin="upper", which puts row 0 at the top and therefore flips
   the map upside down -- Lake Michigan ends up at the bottom. Always draw maps
   through `north_up_imshow()`, never a bare ax.imshow().

2. LEGIBILITY AT PRESENTATION SIZE. Big bold type, thick saturated lines,
   colorbars with labelled, readable ticks -- these figures get projected on a
   screen and dropped into slides/web pages, not squinted at in a notebook.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- type scale (points) ---------------------------------------------------
# Raised 2026-09-30: the earlier scale was legible on a laptop but small once a
# multi-panel figure is scaled into a paper column. Dropping some annotation is
# preferred to text nobody can read.
FS_SUPTITLE = 26
FS_TITLE = 21
FS_LABEL = 20
FS_TICK = 18
FS_LEGEND = 18
FS_CBAR_LABEL = 19
FS_CBAR_TICK = 17
FS_ANNOT = 18

# --- line weights ----------------------------------------------------------
LW_MAIN = 3.2
LW_SECONDARY = 2.4
LW_FAINT = 1.1

# --- palette: fixed order, never cycled ------------------------------------
C_ACTUAL = "#12395B"      # deep navy  - ground truth
C_PRED = "#E8702A"        # vivid orange - model prediction
C_PERSIST = "#7A7A7A"     # grey - persistence baseline
C_F10 = "#1F6FB4"         # blue  - 10-day lead
C_F15 = "#C2185B"         # magenta - 15-day lead
FOLD_COLORS = {"fold1": "#1F6FB4", "fold2": "#E8702A", "fold3": "#2E9E5B", "fold4": "#8B5FBF"}
# one distinct hue per reference point -- six of them, so P5/P6 do not reuse
# P1/P2's colour and get mistaken for the same site across panels
POINT_COLORS = ["#1F6FB4", "#E8702A", "#2E9E5B", "#C2185B", "#7B3FB5", "#0F8B8D"]
# eight for the USGS reference wells -- cycling the six above made W7/W8
# indistinguishable from W1/W2 on the location map
WELL_COLORS = ["#1F6FB4", "#E8702A", "#2E9E5B", "#C2185B",
               "#7B3FB5", "#0F8B8D", "#8C6D1F", "#6B4423"]

# --- colormaps: one hue for magnitude, diverging w/ neutral midpoint --------
CMAP_ERROR = "YlOrRd"      # RMSE / |error| : sequential, light -> dark
CMAP_STATE = "viridis_r"   # absolute state (depth)
# correlation and skill share a sense: BLUE = good, RED = bad, pale at the
# neutral value -- so a viewer never has to re-read the colorbar per panel.
CMAP_CORR = "RdYlBu"       # Pearson r: r=+1 blue, r=0 pale, r=-1 red
CMAP_SKILL = "RdBu"        # 1 - RMSE_model/RMSE_persist: >0 blue (beats persistence)
CMAP_DIFF = "RdBu_r"       # signed error (pred - truth): red = over-predicted
GRID_ALPHA = 0.3


def apply_style():
    """Global rcParams for every figure in the suite."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "font.size": FS_TICK,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.labelsize": FS_LABEL,
        "axes.labelweight": "bold",
        "axes.edgecolor": "#444444",
        "axes.linewidth": 1.4,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "legend.fontsize": FS_LEGEND,
        "figure.titlesize": FS_SUPTITLE,
        "figure.titleweight": "bold",
        "lines.linewidth": LW_MAIN,
        "grid.alpha": GRID_ALPHA,
    })


def save_both(fig, path_no_ext, **kwargs):
    """Write the same figure as PDF (for LaTeX) and PNG (for slides and review).

    Takes a path with no extension; both are produced side by side so the two
    can never drift apart.
    """
    kwargs.setdefault("bbox_inches", "tight")
    kwargs.setdefault("facecolor", "white")
    path_no_ext = str(path_no_ext)
    if path_no_ext.endswith((".pdf", ".png")):
        path_no_ext = path_no_ext[:-4]
    fig.savefig(path_no_ext + ".pdf", **kwargs)
    fig.savefig(path_no_ext + ".png", dpi=200, **kwargs)
    return path_no_ext


def axis_label(name, unit):
    """Axis text in the form the report uses: 'Water table depth [m]'."""
    return f"{name} [{unit}]" if unit else name


def north_up_imshow(ax, field, **kwargs):
    """imshow with origin='lower' so the map is drawn north-up.

    The conus1 y index increases northward; the matplotlib default
    (origin='upper') would render the domain upside down.
    """
    kwargs.setdefault("origin", "lower")
    kwargs.setdefault("interpolation", "nearest")
    return ax.imshow(field, **kwargs)


def add_colorbar(fig, im, ax, label, fraction=0.046, pad=0.04):
    cb = fig.colorbar(im, ax=ax, fraction=fraction, pad=pad)
    cb.set_label(label, fontsize=FS_CBAR_LABEL, fontweight="bold")
    cb.ax.tick_params(labelsize=FS_CBAR_TICK)
    return cb


def style_axes(ax, xlabel=None, ylabel=None, title=None, grid=True, legend=False):
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=FS_LABEL, fontweight="bold")
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=FS_LABEL, fontweight="bold")
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    if grid:
        ax.grid(True, alpha=GRID_ALPHA, linewidth=1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(width=1.4, length=6, labelsize=FS_TICK)
    if legend:
        ax.legend(fontsize=FS_LEGEND, frameon=False)
    return ax


def map_axes_off(ax, title=None):
    """Maps have no meaningful pixel-index axes -- hide them, keep the title."""
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=FS_TITLE, fontweight="bold")
    return ax
