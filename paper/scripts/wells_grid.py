"""
The wells-by-water-years panel grid, shared by the water-table and the
soil-moisture figure.

Rows are the fourteen wells in the paper's single 1..14 sequence, columns are
the four water years in chronological order, WY2003 first.

The x axis of every panel is the WHOLE water year, 1 October to 30 September,
and BOTH forcings now fill it: the seasonal arm is the 360-day spliced water
year ("Dyear"), not the 210-day run that used to stop in late April. The series
are still carried on a complete daily index, so any day a rollout does not
cover -- now only the few days around each fold boundary -- is NaN and the line
breaks there instead of a straight segment being drawn across days no one
forecast. If a seasonal line in the rendered figure stops in April, the stale
210-day cache has been read; wells_common.check_full_year exists to make that
impossible.

THE REFERENCE AND THE SIMULATOR ARE TWO DIFFERENT LINES. ``ref`` is the series
the scores are computed against and it is the ONLY series the per-site constant
in ``offs`` is applied to -- though the water-table figure now hands in an
observation that has ALREADY been shifted and floored at zero by
wells_common.shifted_observations, and passes ``offs`` as all zeros, so that
the series the figure draws and the series the tables score are one object.
``extra`` is an optional second reference drawn on
its OWN, TRUE SCALE with NO offset: in the water-table figure that is ParFlow,
the simulation the emulator is trained to reproduce, which must not be moved to
meet the observation. Correlation is invariant to the shift, so ParFlow is
still scored against the same observation the forcings are scored against.

PAGE FIT. Fifteen rows by four columns go into a one-page, \textwidth figure
in a 1-inch-margin article, so the figure is drawn at close to its final
printed width (FIG_W below, against a 6.5 in text block) and its height is set
by what the caption leaves of the 9.0 in text block (FIG_H below). Drawing
near final size is what keeps the type legible: a 13 in wide figure scaled to
6.5 in halves every font, and 7 pt ticks become 3 pt. Nothing here may grow
the figure's WIDTH without shrinking every font in it to match.

PER-PANEL Y LIMITS. Each panel is scaled to the data actually in it --
reference, simulator, every model mean and every model band -- not to a range
shared down the row. Fourteen wells sit at fourteen different depths and a
single well's water table can move by metres between water years; a shared
range spends most of every panel on blank paper. The cost is that the four
years of a row no longer share a scale, so every panel carries its own y ticks.

The scores sit in fixed left-to-right slots -- the simulator first when there
is one, then reanalysis and raw seasonal -- so a row does not reflow if a
forcing is missing. There is NO reserved slot for an arm that does not exist:
the legend lists the series that are drawn and nothing more. Because both
forcings cover the same 360 days, the reanalysis and seasonal figures in a
panel are read on the same sample of observations, and the panel numbers no
longer need the matched-window caveat the 210-day version carried.
"""
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator, FuncFormatter

import paper_style as P
import wells_common as C

# ---------------------------------------------------------------- page fit --
# The figure is included at \textwidth in a 1-inch-margin article, i.e. 6.5 in,
# on a page whose text height is 9.00 in.
#
# THE HEIGHT IS SET BY WHAT IS LEFT AFTER THE CAPTION. The well captions were
# cut to four lines, and four lines of a \small caption in an 11 pt document
# is 4 x 12 pt = 48 pt; add \abovecaptionskip (10 pt) and the \intextsep above
# and below a [H] float (12 pt each) and the furniture costs 82 pt = 1.13 in.
# That leaves 7.87 in, and 7.70 in is taken, which still fits a fifth caption
# line if one appears. Included at \textwidth the figure is therefore
# 6.50 x 7.70 in on the page, up from 6.435 x 7.31 in under the old ten-line
# caption: 5 % more height for the hydrographs.
#
# THE SAVED WIDTH IS UNCHANGED, DELIBERATELY. The reduction from the saved
# width to the page width is what sets the final type size (6.50/8.015 = 0.81),
# so growing the height at a fixed width cannot shrink a single font; all of
# the extra height goes into the panels, because M_TOP and M_BOT below are in
# inches and do not scale.
TEXT_W = 6.5          # \textwidth
TEXT_H = 9.0          # \textheight
CAP_IN = 1.13         # caption + float separation, four \small lines
FIG_W = 8.05          # saved 8.015 in; type reduces by only 6.50/8.015
FIG_H = 9.40          # saved 8.015 x 9.50 -> ratio 1.185 -> 7.70 in on the
                      # page at \textwidth, under TEXT_H - CAP_IN = 7.87 in
# Margins are set EXPLICITLY, in inches, and tight_layout is not used: the row
# labels are horizontal y labels, which tight_layout does not measure, so it
# let them run off the canvas and bbox_inches="tight" then grew the saved
# figure to 11 in wide -- which is the one thing that must not happen, because
# the page reduction is what shrinks the type.
M_LEFT = 2.02         # supylabel + row label + y tick labels
M_RIGHT = 0.04
M_TOP = 1.12          # suptitle + three legend rows + the WY column titles.
                      # It was 1.42 while the legend carried a sixth,
                      # "pending" entry; with that gone the legend is three
                      # rows and the 0.30 in it no longer needs goes to the
                      # panels, which is why the rows are taller than FIG_H
                      # alone would make them.
M_BOT = 0.20          # x tick labels
WSPACE = .25

# The ISWS asked for their whole network, which takes the panel from 14 wells
# to 45. They do not fit on one page at a legible row height, so the grid is
# paginated: the canvas and every font stay exactly as they were and only the
# number of pages grows. 15 rows is the most that keeps a row tall enough to
# read a hydrograph in, and 45 wells divide into three equal pages.
PAGE_ROWS = 16
SPARSE_MARKERS = 60   # a record with fewer points per panel is marked, not just joined
DOTS_ONLY = 6         # below this, points only: a line would invent a hydrograph
SHOW_PANEL_SCORES = False   # print r into each panel; the scores are
                            # computed and returned either way


def _fmt(v, _):
    """Compact y tick: 0.4 not 0.40, 2 not 2.0, -1.5 kept."""
    s = f"{v:.2f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _rfmt(r):
    """A correlation in the least ink that still reads: .27, -.46, 1.00.

    Four numbers now share one panel of a fourteen-by-four grid, so the
    leading zero of a quantity that cannot exceed one is dropped rather than
    the type size.
    """
    s = f"{r:.2f}"
    if s.startswith("0."):
        return s[1:]
    if s.startswith("-0."):
        return "-" + s[2:]
    return s


def _span(vals):
    """(lo, hi) of everything finite in vals, or None."""
    if not vals:
        return None
    v = np.concatenate([np.asarray(a, float).ravel() for a in vals])
    v = v[np.isfinite(v)]
    if not v.size:
        return None
    lo, hi = float(v.min()), float(v.max())
    if hi - lo < 1e-6:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def _panel_span(o, ds, sim=None):
    """The data actually drawn in ONE panel: reference, simulator, and every
    model mean and band. Returns None when the panel is empty."""
    v = []
    if o is not None and len(o):
        v.append(np.asarray(o, float))
    if sim is not None and len(sim):
        v.append(np.asarray(sim, float))
    for d in ds:
        if d is None or not len(d):
            continue
        for c in ("lo", "hi", "mean"):
            v.append(np.asarray(d[c].values, float))
    return _span(v)


def build(ws, ref, mods, key, *, invert, offs, value_label, ref_label,
          title, name, tol, out_dir=None, extra=None):
    """Draw the grid and return the per-panel scores.

    ``extra``, when given, is dict(key, color, label, short, series), a second
    reference drawn unshifted. ``offs`` never touches it.
    """
    nr, nc = len(ws), len(C.WATER_YEARS)
    fig, axes = plt.subplots(nr, nc, figsize=(FIG_W, FIG_H),
                             squeeze=False, sharex="col", sharey=False)
    rows = []

    # the fixed left-to-right score slots: the simulator, then the forcings in
    # the paper's order, whether or not each one's cache is on disk
    slot_keys = ([extra["key"]] if extra else []) + [k for k, _, _, _
                                                     in C.FORCINGS]
    slot = {k: i for i, k in enumerate(slot_keys)}
    dx = .955 / len(slot_keys)
    fs_annot = P.FS_ANNOT - (1.0 if extra else 0.5)

    for i, w in enumerate(ws):
        sid = w["short_id"]
        off = offs.get(sid, 0.0)
        o_all = ref.get(sid)
        # the simulator: its own true scale, the offset deliberately not used
        x_all = extra["series"].get(sid) if extra else None
        # a row-wide span, used only to give an EMPTY panel a sane axis
        row_sp = _panel_span(
            None if o_all is None or not len(o_all)
            else np.asarray(o_all.values, float) + off,
            [m[key].get(sid) for m in mods],
            None if x_all is None else np.asarray(x_all.values, float)
        ) or (0.0, 1.0)

        for j, wy in enumerate(C.WATER_YEARS):
            ax = axes[i][j]
            t0, t1 = C.wy_window(wy)
            o = C.clip(o_all, wy)
            ov = (None if o is None or not len(o)
                  else np.asarray(o.values, float) + off)
            xs = C.clip(x_all, wy)
            xv = (None if xs is None or not len(xs)
                  else np.asarray(xs.values, float))
            drew = False
            cut = []
            for m in mods:
                d = m[key].get(sid)
                if d is None:
                    cut.append(None)
                    continue
                d = d[(d.index >= t0) & (d.index <= t1)]
                cut.append(d)
                if not len(d) or not np.isfinite(d["mean"].values).any():
                    continue
                ax.fill_between(d.index, d["lo"], d["hi"], color=m["color"],
                                alpha=.18, lw=0, zorder=2)
                ax.plot(d.index, d["mean"], color=m["color"], lw=1.0,
                        zorder=3, solid_capstyle="round")
                drew = True
            if xv is not None and np.isfinite(xv).any():
                ax.plot(xs.index, xv, color=extra["color"], lw=1.1, zorder=4,
                        solid_capstyle="round")
                drew = True
            if ov is not None:
                # Most of the ISWS network is read monthly or quarterly, not
                # daily. A polyline through four points a year is nearly
                # invisible and a single reading draws nothing at all, so a
                # sparse record is marked as well as joined; the reader can
                # then see how much evidence a panel actually rests on.
                nfin = int(np.isfinite(ov).sum())
                if nfin <= DOTS_ONLY:
                    # three readings in a year joined by straight segments
                    # draw a hydrograph that was never observed; the points
                    # are shown as points
                    mk = dict(marker="o", ms=2.6, mew=0, ls="none")
                elif nfin <= SPARSE_MARKERS:
                    mk = dict(marker="o", ms=2.2, mew=0)
                else:
                    mk = {}
                ax.plot(o.index, ov, color=P.C_TRUTH if extra is None
                        else P.C_PERS, lw=1.1, zorder=5, **mk)
            if not drew and ov is None:
                ax.text(.5, .5, "no data", ha="center", va="center",
                        transform=ax.transAxes, fontsize=P.FS_ANNOT - 1.5,
                        color="#AAAAAA")

            # ------------------------------------------- limits, per panel --
            sp = _panel_span(ov, cut, xv) or row_sp
            lo_v, hi_v = sp
            rng = hi_v - lo_v
            # a small margin at the top, and one line's worth of room at the
            # bottom for the single row of scores
            pad_lo, pad_hi = .08 * rng, .30 * rng
            ax.set_xlim(t0, t1)
            if invert:
                ax.set_ylim(hi_v + pad_hi, lo_v - pad_lo)
            else:
                ax.set_ylim(lo_v - pad_hi, hi_v + pad_lo)

            ax.grid(True, lw=.45, color=P.C_GRID)
            ax.set_axisbelow(True)
            ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(10, 1, 4, 7)))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=3, prune="lower"))
            ax.yaxis.set_major_formatter(FuncFormatter(_fmt))
            ax.tick_params(labelsize=P.FS_TICK - 1.0, length=2.0, pad=1.0)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)

            # one score per series, in a fixed LEFT-TO-RIGHT slot so the row
            # does not reflow when a forcing is missing; the order is the
            # legend's order and the legend says so
            scored = []
            if extra is not None and xs is not None and len(xs):
                scored.append((extra["key"], extra["color"],
                               pd.DataFrame({"mean": xs})))
            scored += [(m["key"], m["color"], d) for m, d in zip(mods, cut)]
            for k_, color_, d in scored:
                if d is None or ov is None or not len(d):
                    continue
                oo = pd.Series(ov, index=o.index)
                st = C.score(oo, d, tol=tol, nmin=30)
                # The scores are still COMPUTED for every panel -- the CSV,
                # the wells table and every correlation quoted in the text
                # come from `rows` below -- they are simply no longer printed
                # into the panels, which the four-series grid had become too
                # dense to carry legibly.
                if SHOW_PANEL_SCORES:
                    x = .022 + slot[k_] * dx
                    if np.isfinite(st["r"]):
                        ax.text(x, .055, _rfmt(st["r"]), transform=ax.transAxes,
                                ha="left", va="bottom", color=color_,
                                fontsize=fs_annot, fontweight="bold", zorder=6)
                    else:
                        ax.text(x, .055, "n/a", transform=ax.transAxes,
                                ha="left", va="bottom", color="#BBBBBB",
                                fontsize=fs_annot, zorder=6)
                rows.append(dict(forcing=k_, well=w["no"], short_id=sid,
                                 water_year=wy, r=st["r"], inc=st["inc"],
                                 n=st["n"]))
            if i == 0:
                ax.set_title(f"WY{wy}", fontsize=P.FS_TITLE - 2,
                             fontweight="bold", pad=3)

        # the row label is the well's number and its name, and nothing else
        axes[i][0].set_ylabel(f"{w['no']}  {w['pretty']}",
                              fontsize=P.FS_LABEL - 2, fontweight="bold",
                              rotation=0, ha="right", va="center",
                              labelpad=12)

    # ------------------------------------------------------------- legend --
    handles = [mlines.Line2D([], [], color=P.C_TRUTH if extra is None
                             else P.C_PERS, lw=1.6, label=ref_label)]
    if extra is not None:
        handles.append(mlines.Line2D([], [], color=extra["color"], lw=1.8,
                                     label=extra["label"]))
    # Only the forcings that are actually drawn. There is no reserved slot for
    # an arm that does not exist: the figure legend describes what is in the
    # figure and nothing else.
    have = {m["key"] for m in mods}
    for key_, _, color, label in C.FORCINGS:
        if key_ in have:
            handles.append(mlines.Line2D([], [], color=color, lw=1.8,
                                         label=label))
    # One band handle PER ARM, in that arm's own colour. A single grey swatch
    # was wrong twice over: there are two bands in every panel, and grey is
    # already the observation's colour in this same legend.
    for key_, _, color, label in C.FORCINGS:
        if key_ in have:
            handles.append(mpatches.Patch(facecolor=color, alpha=.18,
                                          edgecolor="none",
                                          label=f"10th-90th percentile, "
                                                f"{label.split(', ')[-1]}"))
    # No per-panel score legend: with SHOW_PANEL_SCORES off there is nothing
    # in the panels for it to explain.
    if SHOW_PANEL_SCORES:
        if extra is None:
            note = ("Panel figures: correlation $r$ with the reference, one per\n"
                    "forcing, left to right in the order of this legend")
        else:
            note = (f"Panel figures: correlation $r$ with the observation, one\n"
                    f"per series, left to right in the order of this legend\n"
                    f"({extra['short']} first, then each forcing)")
        handles.append(mlines.Line2D([], [], color="none", label=note))
    fig.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(.5, 1 - .30 / FIG_H),
               ncol=2, frameon=False,
               fontsize=P.FS_LEGEND - 2.0, handlelength=1.8,
               columnspacing=1.4, labelspacing=.3, borderpad=0, handletextpad=.6)
    fig.suptitle(title, fontsize=P.FS_SUP - 2, fontweight="bold",
                 y=1 - .06 / FIG_H)
    fig.supylabel(value_label, fontsize=P.FS_LABEL, fontweight="bold",
                  x=.024)
    fig.subplots_adjust(left=M_LEFT / FIG_W, right=1 - M_RIGHT / FIG_W,
                        top=1 - M_TOP / FIG_H, bottom=M_BOT / FIG_H,
                        hspace=.20, wspace=WSPACE)
    P.save(fig, name, out_dir)
    return pd.DataFrame(rows)


def build_paged(ws, *args, name, title, page_rows=PAGE_ROWS, **kw):
    """`build` over consecutive slices of `ws`, one figure per page.

    Returns the concatenated per-panel scores, so every caller downstream --
    the CSV, the table, the medians quoted in the text -- sees one table over
    all wells and cannot silently end up reporting a single page.
    """
    pages = [ws[i:i + page_rows] for i in range(0, len(ws), page_rows)]
    out = []
    for k, chunk in enumerate(pages, 1):
        suffix = f" ({k} of {len(pages)})" if len(pages) > 1 else ""
        out.append(build(chunk, *args,
                         name=f"{name}_p{k}" if len(pages) > 1 else name,
                         title=title + suffix, **kw))
        print(f"    page {k}/{len(pages)}: wells "
              f"{chunk[0]['no']}-{chunk[-1]['no']}", flush=True)
    return pd.concat(out, ignore_index=True)
