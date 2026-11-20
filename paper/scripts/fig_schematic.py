"""
The autoregressive rollout, drawn so the three things that matter are visible:
what the network is conditioned on, what it emits, and what is fed back.

The report's version showed the chain but not the provenance rule -- which
channels are allowed to come from where. That rule is the whole reason the
result counts as emulation rather than interpolation of the simulator's own
diagnostics, so the figure makes it explicit with colour.

FORMALISATION (this revision).  The first version of this figure was drawn for
a slide: saturated filled boxes, bold white text, one decorative colour per
box. It is reproduced here as a journal figure instead.

  * The canvas is drawn at the width it is PRINTED at, 6.5 in (\\textwidth
    in a 1-inch-margin article), so every font size below is, to within half a
    percent, the size the reader sees on paper: the smallest type in the
    figure prints at 7 pt, not at whatever 7 pt becomes after a reduction.
    Two things are needed to keep that true, and both are in `main`: a white
    patch over the whole canvas, so the paper-wide tight bbox cannot crop to
    the artists and change the aspect; and savefig.pad_inches cut to 0.02 in,
    because the default 0.1 in would widen the file to 6.7 in and cost 3 %.
  * Panels are white with a thin rule; provenance is carried by a coloured
    4 pt bar on the panel's leading edge and by the legend, not by a saturated
    fill. Colour therefore encodes the argument (where a channel came from)
    and nothing else.
  * Labels are the Methods' own notation, not prose paraphrases: the
    conditioning stack is $\\mathbf{c}$ (Sect. 2.2.1), the state $\\mathbf{s}$
    and the residual $\\mathbf{r}_\\ell$ are Eq. (4), the projection is Eq. (5),
    the lead index $\\ell$ up to $L$ and the diffusion step count $T$ are as
    defined there. The only symbol the figure introduces is the hop index
    $k$ up to $K$, which the Methods leave unnamed; it and every other symbol
    are defined in the two-line key between the panels.
  * Two labelled panels, (a) one call and (b) the chain, so the reading order
    is stated rather than inferred.
  * Panel (b) is a row of DISCRETE call boxes, $k = 1, \\ldots, 5$, an
    ellipsis, $k = 23, 24$, joined by arrows -- not a continuous 24-segment
    strip. A strip is more faithful to the arithmetic but it reads as a bar,
    not as a sequence of calls, and the point of the panel is that the year is
    produced by repeated calls rather than by one long integration. The boxes
    are drawn to the same rules as (a): white, thin rule, provenance in a
    4 pt leading-edge bar. The provenance claim is carried by the ARROW
    colours -- exactly one navy arrow, at the head of the chain, and orange
    everywhere after it -- so "the day-0 ParFlow state is the only ParFlow
    field in the chain" is visible before it is read. The calendar lives on a
    broken day-of-water-year axis under the boxes, broken in the same place
    as the ellipsis so the axis never implies days that are not drawn.

Colours all come from paper_style, so a colour means the same thing here as in
the result figures: navy = ParFlow, orange = emulator state, grey = neutral.

BUGS ALREADY FIXED -- DO NOT REINTRODUCE
  1. Arrow heads hidden under box padding. FancyBboxPatch's `pad` inflates the
     PAINTED box beyond the nominal (x, y, w, h) on every side, so an arrow
     drawn to the nominal edge ends underneath the neighbouring box. Every box
     here is drawn with pad=0 (see `panel`), so the painted edge IS the
     nominal edge and a shrink of ~1 pt is all that is needed.
  2. Z-order. Arrows are drawn at zorder=5, ABOVE the boxes at zorder=2, so a
     head that lands on a border is still visible.
  3. Chain arrows consumed by padding. The chain IS a row of discrete boxes
     with an arrow in every gap, and the gap is the thing that fails first: a
     padded box on each side eats it from both ends and the arrow disappears.
     Two defences, both load-bearing. (i) Every chain box goes through
     `panel`, i.e. pad=0, so a box occupies exactly its nominal width.
     (ii) CHAIN_GAP is 0.028 axes units = 13 pt of real, empty space, which is
     wide enough for a 1 pt shrink at each end and a 4 pt arrow head and still
     leaves shaft. Narrowing the gap to fit another box is the regression to
     watch for: drop a box instead.
  4. The feedback arc drawn over the boxes. connectionstyle arc3 bends to the
     LEFT of the a -> b direction, and the feedback runs right to left, so a
     POSITIVE rad lifts it across the three panels it is meant to pass under.
     It takes rad < 0 to dip into the clear band below them.

Matplotlib's mathtext, which renders every $...$ here, is not LaTeX: \\big and
\\le are not defined in it and raise at save time. Plain parentheses and an
explicit range ($t = T, \\ldots, 1$) are used instead.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

# ------------------------------------------------------------- provenance ---
# Three categories, because the figure's argument is a three-way partition of
# the conditioning channels. Taken from paper_style so that navy still means
# ParFlow and orange still means the emulator, exactly as in every result
# figure of the paper.
C_STATIC = "#7C8894"        # static parameters: ParFlow INPUTS, never outputs
C_FORCE = "#2E9BD6"         # atmosphere: the paper's mid-blue (P.SPEC ramps)
C_STATE = P.C_PRED          # the model's own state                  orange
C_PARFLOW = P.C_TRUTH       # ParFlow state -- day 0, and nowhere else  navy

INK = "#23303C"             # body text
INK_SOFT = "#5A6672"        # secondary text
RULE = "#AAB4BE"            # panel borders
ARROW = "#4A5663"

FIG_W, FIG_H = 6.5, 4.85    # inches; 6.5 in == \textwidth, i.e. PRINTED SIZE

# Type sizes, in points at the printed size. Anchored to the paper's scale
# (P.FS_LABEL = 11, P.FS_ANNOT = 9) and stepped down from there; nothing is
# smaller than 6.4 pt.
FS_PANEL = P.FS_LABEL           # 11.0  panel letters
FS_HEAD = P.FS_ANNOT - 0.4      #  8.6  box headings
FS_MATH = P.FS_ANNOT - 0.4      #  8.6  display notation
FS_BODY = P.FS_ANNOT - 1.4      #  7.6  box detail
FS_SUB = P.FS_ANNOT - 2.0       #  7.0  the second and third line of a box
FS_KEY = P.FS_ANNOT - 2.0       #  7.0  legend and symbol key

LW_RULE = 0.7                   # panel borders
LW_ARROW = 0.9
BAR = 0.0085                    # width of the provenance bar, axes units


def panel(ax, x, y, w, h, bar=None, ec=RULE, fc="white", lw=LW_RULE, r=0.008):
    """A light panel with a thin rule and, optionally, a provenance bar.

    pad=0 is deliberate: the painted edge is then the nominal edge, so an
    arrow drawn to (x + w, y + h/2) is not swallowed (see bug 1 in the
    module docstring).
    """
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    if bar:
        ax.add_patch(Rectangle((x, y + 0.004), BAR, h - 0.008,
                               facecolor=bar, edgecolor="none", zorder=3))


def head(ax, x, y, title, count=None, xr=None):
    ax.text(x, y, title, ha="left", va="center", fontsize=FS_HEAD,
            color=INK, zorder=4)
    if count is not None:
        ax.text(xr, y, count, ha="right", va="center", fontsize=FS_BODY,
                color=INK_SOFT, zorder=4)


def arrow(ax, a, b, color=ARROW, lw=LW_ARROW, rad=0.0, ms=9,
          shrinkA=1.0, shrinkB=1.0, ls="-", zorder=5):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", lw=lw, color=color, mutation_scale=ms,
        shrinkA=shrinkA, shrinkB=shrinkB, zorder=zorder, linestyle=ls,
        connectionstyle=f"arc3,rad={rad}"))


def main():
    P.apply_style()
    # savefig.bbox is "tight" paper-wide, and a tight bbox adds pad_inches on
    # every side -- 0.1 in by default, which would widen the file to 6.7 in
    # and so shrink every type size by 3 % when LaTeX fits it to \textwidth.
    # 0.02 in keeps the file at 6.54 in: a 0.6 % reduction, i.e. none.
    plt.rcParams["savefig.pad_inches"] = 0.02
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    # ...and the tight bbox would otherwise crop to the drawn artists, which
    # silently changes the aspect ratio the type sizes were chosen for. This
    # white patch covers the whole canvas, so the tight bbox is the canvas.
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor="none",
                           zorder=0))

    # ================================================== (a) a single call ===
    ax.text(0.000, 0.984, "(a)", ha="left", va="center",
            fontsize=FS_PANEL, fontweight="bold", color=INK)
    ax.text(0.042, 0.984, "One call of the emulator (hop $k$)", ha="left",
            va="center", fontsize=P.FS_ANNOT, color=INK)

    # Four stations, read left to right and then down the right-hand column:
    # conditioning -> denoiser -> projection -> predicted state -> feedback.
    cx, cw = 0.000, 0.350          # conditioning stack
    nx, nw = 0.384, 0.232          # denoiser
    rx, rw = 0.660, 0.340          # projection (upper) and output (lower)
    ytop = 0.902                   # the top edge shared by all three columns

    # ------------------------------------------ conditioning stack, left ---
    ax.text(cx, 0.932, "Conditioning $\\mathbf{c}$", ha="left", va="bottom",
            fontsize=FS_HEAD, color=INK)
    ax.text(cx + cw, 0.932, "69 channels", ha="right", va="bottom",
            fontsize=FS_BODY, color=INK_SOFT)
    ax.plot([cx, cx + cw], [0.924, 0.924], color=INK, lw=0.8, zorder=4)

    rh, rgap = 0.105, 0.021
    rows = (
        (C_STATIC, "Static parameters", "22",
         ("terrain and soil hydraulic properties",
          "a ParFlow $\\it{input}$, never its output")),
        (C_FORCE, "Atmospheric forcing", "45",
         ("$P$, $T_\\mathrm{min}$, $T_\\mathrm{max}$ over the window, $3 \\times L$",
          "operationally available (NLDAS-2, SEAS5)")),
        (C_STATE, "Initial state $\\mathbf{s}_{t_0}$", "2",
         ("water table depth, soil moisture",
          "$k = 1$: ParFlow;  $k > 1$: emulator")),
    )
    ymid = ytop - (3 * rh + 2 * rgap) / 2          # the flow's centre line
    y_r3 = ytop - 3 * rh - 2 * rgap                # bottom row's lower edge
    for i, (bar, title, nch, lines) in enumerate(rows):
        y0 = ytop - (i + 1) * rh - i * rgap
        panel(ax, cx, y0, cw, rh, bar=bar)
        head(ax, cx + 0.020, y0 + 0.080, title, nch + " ch", cx + cw - 0.010)
        for j, ln in enumerate(lines):
            ax.text(cx + 0.020, y0 + 0.051 - 0.028 * j, ln, ha="left",
                    va="center", fontsize=FS_SUB, color=INK_SOFT, zorder=4)
        arrow(ax, (cx + cw, y0 + rh / 2), (nx, ymid))

    # ----------------------------------------------- the denoiser, centre ---
    panel(ax, nx, y_r3, nw, ytop - y_r3, ec=INK, lw=0.9)
    ax.add_patch(Rectangle((nx, ytop - 0.040), nw, 0.040, facecolor="#EDF1F4",
                           edgecolor="none", zorder=2))
    ax.plot([nx, nx + nw], [ytop - 0.040] * 2, color=INK, lw=0.7, zorder=4)
    ax.text(nx + nw / 2, ytop - 0.020, "Conditional DDPM", ha="center",
            va="center", fontsize=FS_HEAD, color=INK, zorder=4)
    for dy, t, fs, col in (
            (0.072, "U-Net $\\hat{v}_\\theta(\\mathbf{x}_t, t, \\mathbf{c})$",
             FS_MATH, INK),
            (0.101, "9,284,226 parameters", FS_SUB, INK_SOFT),
            (0.128, "$T = 100$ diffusion steps", FS_SUB, INK_SOFT),
            (0.155, "$v$-prediction", FS_SUB, INK_SOFT),
            (0.199, "$\\mathbf{x}_T \\sim \\mathcal{N}(0, \\mathbf{I})"
                    " \\longrightarrow \\mathbf{x}_0$", FS_MATH, INK),
            (0.252, "target is the $\\it{residual}$", FS_SUB, INK_SOFT),
            (0.285, "$\\mathbf{r}_\\ell = \\mathbf{s}_{t_0+\\ell}"
                    " - \\mathbf{s}_{t_0}$,  Eq. (4)", FS_MATH, INK),
            (0.316, "$\\ell = 1, \\ldots, L$;  $L = 15$ d", FS_SUB, INK_SOFT),
            (0.343, "not the absolute state", FS_SUB, INK_SOFT)):
        ax.text(nx + nw / 2, ytop - dy, t, ha="center", va="center",
                fontsize=fs, color=col, zorder=4)
    for dy in (0.176, 0.228):
        ax.plot([nx + 0.030, nx + nw - 0.030], [ytop - dy] * 2, color=RULE,
                lw=0.6, zorder=4)

    # ------------------------------- the projection, top of the right column
    ph = 0.135
    panel(ax, rx, ytop - ph, rw, ph, ec=INK, lw=0.9)
    head(ax, rx + 0.020, ytop - 0.027, "Projection $\\Pi$", "Eq. (5)",
         rx + rw - 0.012)
    ax.text(rx + rw / 2, ytop - 0.063,
            "$\\Pi(\\mathbf{s}) = \\min(\\max(\\mathbf{s},"
            "\\mathbf{s}_\\mathrm{min}), \\mathbf{s}_\\mathrm{max})$",
            ha="center", va="center", fontsize=FS_MATH, color=INK, zorder=4)
    for j, t in enumerate(("onto the physically admissible set,",
                           "inside the loop and after $\\it{every}$ call")):
        ax.text(rx + rw / 2, ytop - 0.093 - 0.026 * j, t, ha="center",
                va="center", fontsize=FS_SUB, color=INK_SOFT, zorder=4)
    arrow(ax, (nx + nw, ytop - ph / 2), (rx, ytop - ph / 2))

    # ---------------------------- the predicted state, below the projection
    oh = 0.150
    panel(ax, rx, y_r3, rw, oh, bar=C_STATE)
    head(ax, rx + 0.020, y_r3 + oh - 0.027, "Predicted state", "30 ch",
         rx + rw - 0.012)
    ax.text(rx + 0.020, y_r3 + oh - 0.062,
            "$\\mathbf{s}_{t_0+\\ell}, \\;\\; \\ell = 1, \\ldots, L$",
            ha="left", va="center", fontsize=FS_MATH, color=INK, zorder=4)
    for j, t in enumerate(("2 variables $\\times$ 15 lead days",
                           "the emulator's own field, not ParFlow's")):
        ax.text(rx + 0.020, y_r3 + oh - 0.093 - 0.028 * j, t, ha="left",
                va="center", fontsize=FS_SUB, color=INK_SOFT, zorder=4)
    arrow(ax, (rx + rw / 2, ytop - ph), (rx + rw / 2, y_r3 + oh))

    # ------------------------------------------------------- the feedback ---
    arrow(ax, (rx + 0.30 * rw, y_r3), (cx + 0.45 * cw, y_r3), color=C_STATE,
          lw=1.1, rad=-0.22, ms=11)
    ax.text(0.500, y_r3 - 0.108,
            "feedback: $\\mathbf{s}^{(k+1)}_{t_0} = \\mathbf{s}_{t_0+L}$, "
            "the last predicted day, is the next call's initial state",
            ha="center", va="center", fontsize=FS_SUB, color=C_STATE,
            zorder=6)

    # ---------------------------------------------------- legend and key ---
    ax.plot([0.0, 1.0], [0.398, 0.398], color=RULE, lw=0.6, zorder=1)
    keys = ((C_STATIC, "static parameter (a ParFlow input)", 0.000),
            (C_FORCE, "atmospheric forcing", 0.290),
            (C_STATE, "emulator state", 0.485),
            (C_PARFLOW, "ParFlow state (day 0 only)", 0.650))
    for col, lab, x in keys:
        ax.add_patch(Rectangle((x, 0.362), 0.015, 0.017, facecolor=col,
                               edgecolor="none", zorder=4))
        ax.text(x + 0.021, 0.370, lab, ha="left", va="center",
                fontsize=FS_KEY, color=INK, zorder=4)

    ax.text(0.000, 0.330,
            "$\\mathbf{s}$ state (water table depth, soil moisture);  "
            "$\\mathbf{c}$ conditioning stack;  $\\ell = 1,\\ldots,L$ lead "
            "day, $L = 15$;",
            ha="left", va="center", fontsize=FS_KEY, color=INK_SOFT)
    ax.text(0.000, 0.303,
            "$k = 1,\\ldots,K$ hop index, $K = 24$;  $t_0 = (k-1)L$ the "
            "initial day of hop $k$;  $t = T,\\ldots,1$ the diffusion step.",
            ha="left", va="center", fontsize=FS_KEY, color=INK_SOFT)

    # ===================================================== (b) the chain ====
    ax.text(0.000, 0.248, "(b)", ha="left", va="center",
            fontsize=FS_PANEL, fontweight="bold", color=INK)
    ax.text(0.042, 0.248,
            "The chain: $K = 24$ calls span one water year, and no ParFlow "
            "output re-enters it after day 0",
            ha="left", va="center", fontsize=P.FS_ANNOT, color=INK)

    # Seven boxes out of twenty-four: k = 1..5, an ellipsis, k = 23, 24 --
    # the same selection the slide version used, because it is the one that
    # shows both ends of the chain and makes the omission explicit.
    K_TOTAL = 24                    # calls in a water year
    L = 15                          # days emitted by one call
    SHOWN = (1, 2, 3, 4, 5, 23, 24)
    NB = len(SHOWN)
    CHAIN_GAP = 0.028               # arrow gap between adjacent calls (13 pt)
    ELL_GAP = 0.088                 # the wider gap that holds the ellipsis
    S0_W = 0.125                    # the one simulator state in the whole year
    S0_GAP = 0.030
    X_END = 0.986                   # right edge; the "360" label must fit in

    bh = 0.050                      # call-box height
    byt = 0.206                     # top edge of the row of boxes
    by = byt - bh                   # bottom edge
    ymid_b = by + bh / 2
    sx = S0_W + S0_GAP
    bw = (X_END - sx - (NB - 2) * CHAIN_GAP - ELL_GAP) / NB

    xs, cur = [], sx                # left edge of each drawn call box
    for i in range(NB):
        xs.append(cur)
        cur += bw + (ELL_GAP if i == 4 else CHAIN_GAP)

    # ---------------------------------------- the one ParFlow field, day 0 ---
    panel(ax, 0.000, by, S0_W, bh, bar=C_PARFLOW)
    ax.text(0.020, ymid_b, "$\\mathbf{s}_0$  ParFlow", ha="left",
            va="center", fontsize=FS_SUB, color=INK, zorder=4)
    # Navy, and the only navy arrow in the panel: this is the provenance rule
    # drawn rather than asserted.
    arrow(ax, (S0_W, ymid_b), (sx, ymid_b), color=C_PARFLOW, lw=1.1, ms=9)

    # ------------------------------------------------------- the 24 calls ---
    for i, k in enumerate(SHOWN):
        panel(ax, xs[i], by, bw, bh, bar=C_STATE)
        ax.text(xs[i] + (bw + BAR) / 2, ymid_b, f"$k = {k}$", ha="center",
                va="center", fontsize=FS_MATH, color=INK, zorder=4)
        if i == NB - 1:
            continue
        x0, x1 = xs[i] + bw, xs[i + 1]
        if i == 4:                                  # across the ellipsis
            # Three plotted dots, not mathtext \cdots: a marker sits exactly
            # on ymid_b, whereas \cdots sits on the math axis and rides a
            # fraction of an em below the arrows it is continuing.
            xm = (x0 + x1) / 2
            arrow(ax, (x0, ymid_b), (xm - 0.019, ymid_b), color=C_STATE, ms=8)
            ax.plot([xm - 0.010, xm, xm + 0.010], [ymid_b] * 3, ls="none",
                    marker="o", ms=2.3, mfc=C_STATE, mec="none", zorder=4)
            arrow(ax, (xm + 0.019, ymid_b), (x1, ymid_b), color=C_STATE, ms=8)
        else:
            arrow(ax, (x0, ymid_b), (x1, ymid_b), color=C_STATE, ms=8)

    ax.text(sx, byt + 0.016, "each box is one call of (a)", ha="left",
            va="center", fontsize=FS_SUB, color=INK_SOFT)
    n_hidden = K_TOTAL - NB
    ax.text((xs[4] + bw + xs[5]) / 2, byt + 0.016,
            f"{n_hidden} further calls", ha="center", va="center",
            fontsize=FS_SUB, color=INK_SOFT)

    # ----------------------------- day of the water year, broken like the row
    # Ticks sit on the call boundaries, so the axis and the boxes are the same
    # partition of the year; the axis is cut where the ellipsis is, so it
    # never draws a day that is not in a box.
    ya = by - 0.019
    for a, b in ((0, 4), (5, NB - 1)):
        xl, xr = xs[a], xs[b] + bw
        ax.plot([xl, xr], [ya, ya], color=INK_SOFT, lw=0.6, zorder=3)
        bnds = [xs[i] if i == a else xs[i] - CHAIN_GAP / 2
                for i in range(a, b + 1)] + [xr]
        days = [(SHOWN[i] - 1) * L for i in range(a, b + 1)] + [SHOWN[b] * L]
        for xb, d in zip(bnds, days):
            ax.plot([xb, xb], [ya, ya - 0.009], color=INK_SOFT, lw=0.6,
                    zorder=3)
            ax.text(xb, ya - 0.025, str(d), ha="center", va="center",
                    fontsize=FS_KEY - 0.4, color=INK_SOFT)
    ax.text((sx + X_END) / 2, ya - 0.052, "day of the water year",
            ha="center", va="center", fontsize=FS_KEY, color=INK_SOFT)

    ax.text(0.000, 0.049,
            "$K \\times L = 24 \\times 15 = 360$ days, one water year; the "
            "chain is identical under reanalysis and under seasonal forcing.",
            ha="left", va="center", fontsize=FS_KEY, color=INK)
    ax.text(0.000, 0.018,
            "Only the navy arrow carries a ParFlow field; every later call "
            "is started by the preceding call's own prediction, as in (a).",
            ha="left", va="center", fontsize=FS_KEY, color=INK)

    P.save(fig, "fig_rollout_schematic")
    print(f"  canvas {FIG_W} x {FIG_H} in + 0.02 in pad -> 6.54 x 4.89 in; "
          f"at \\textwidth = 6.5 in that is a 0.6 % reduction, so the "
          f"smallest type prints at 7.0 pt")


if __name__ == "__main__":
    main()
