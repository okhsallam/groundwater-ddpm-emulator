"""table_topo_index -- what each topographic-index class in Figure 1 means.

The topographic (wetness) index is ln(a / tan beta): the upslope contributing
area per unit contour length divided by the local slope. It is large where a
cell drains a lot of ground, where the ground is flat, or both -- that is, it
measures a cell's TENDENCY TO SATURATE, not its wetness on any given day.

The nine classes of P.SPEC["topographic_index"] are equal-area quantiles of the
study-domain distribution snapped to the nearest 0.25. This script re-derives
the area fraction of each class from the same file the figure reads, so the
table cannot drift from the map, and writes

    paper/tables/topo_index.tex    bare booktabs tabular, no table wrapper
    paper/tables/topo_index.json   the same numbers, machine-checkable

Median elevation per class is reported in the JSON (and summarised in the .tex
footnote) as the evidence that the classes really do run from the divides to
the valley floors: it falls monotonically across the nine classes.
"""
import json
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P

VAR = "topographic_index"
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "tables")

# One plain sentence per class, in the same order as SPEC["bounds"].
MEANING = [
    "Steepest, most convex ground: moraine crests and the interfluve "
    "divides. A small contributing area drained by a comparatively strong "
    "slope, so water leaves about as fast as it arrives; the deepest "
    "unsaturated zone in the domain.",
    "Upper hillslope shoulders just below the divides. Still freely "
    "drained, with no sustained convergence of upslope water.",
    "Mid-hillslope backslopes of the till plain. Drainage keeps pace with "
    "recharge except in the wettest spells.",
    "Lower hillslopes and gently rolling till plain, the modal landform of "
    "the domain. The slope is slight, but the contributing area is still "
    "small.",
    "Flat till and lacustrine plain, straddling the domain median. Slope is "
    "nearly absent, so after a wet period the water table rises into reach "
    "of the root zone.",
    "Footslopes and shallow swales, where convergence begins to outweigh "
    "slope. The soil stays wet well after an event.",
    "Convergent hollows and headwater swales, much of it artificially "
    "drained cropland. Seasonally saturated, and the first ground to "
    "produce saturation-excess runoff.",
    "Valley floors and broad floodplains: a large contributing area over "
    "very low slope. The water table is shallow for most of the year.",
    "Channel corridors, floodplain wetlands and open water, Lake Michigan "
    "included. Saturated or ponded much of the time; these are the cells "
    "where surface and subsurface flow are most strongly coupled.",
]


def main():
    sp = P.SPEC[VAR]
    b = list(sp["bounds"])
    n = len(b) - 1
    assert len(sp["colors"]) == n and len(MEANING) == n, (
        f"{n} classes, {len(sp['colors'])} colours, {len(MEANING)} meanings")

    with xr.open_dataset(os.path.join(P.RAW, "static_conditions.nc")) as ds:
        ti = np.squeeze(ds[VAR].values).ravel()
        el = np.squeeze(ds["elevation"].values).ravel()
    ok = np.isfinite(ti)
    ti, el = ti[ok], el[ok]
    N = ti.size
    lo_data, hi_data = float(ti.min()), float(ti.max())
    assert b[0] <= lo_data and hi_data <= b[-1], \
        f"data {lo_data:.3f}..{hi_data:.3f} falls outside bounds {b[0]}..{b[-1]}"

    rows, covered = [], 0
    for i in range(n):
        m = (ti >= b[i]) & (ti < b[i + 1]) if i < n - 1 else (ti >= b[i])
        k = int(m.sum())
        covered += k
        lo = lo_data if i == 0 else b[i]
        hi = hi_data if i == n - 1 else b[i + 1]
        rows.append(dict(
            klass=i + 1, color=sp["colors"][i],
            lower=round(lo, 2), upper=round(hi, 2),
            bin_lower=b[i], bin_upper=b[i + 1],
            n_cells=k, area_fraction_percent=round(100 * k / N, 1),
            median_index=round(float(np.median(ti[m])), 2),
            median_elevation_m=round(float(np.median(el[m])), 1),
            meaning=MEANING[i]))
    assert covered == N, f"{covered} of {N} cells classified"
    fr = [r["area_fraction_percent"] for r in rows]
    med_el = [r["median_elevation_m"] for r in rows]
    assert all(med_el[i] > med_el[i + 1] for i in range(n - 1)), \
        "median elevation is not monotonically decreasing across the classes"

    # ------------------------------------------------------------- .tex ---
    L = [r"{\small\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{l r p{0.635\textwidth}}", r"\toprule",
         r"Topographic index & Domain & "
         r"What the class represents \\",
         r"$\ln(a/\tan\beta)$ & area (\%) & \\", r"\midrule"]
    for i, r in enumerate(rows):
        L.append(f"{r['lower']:.2f}--{r['upper']:.2f} & "
                 f"{r['area_fraction_percent']:.1f} & {r['meaning']} \\\\"
                 + ("" if i == n - 1 else r" \addlinespace[2.5pt]"))
    L += [r"\bottomrule", r"\end{tabular}}", "%",
          "% Classes are the equal-area (11.1 %) quantiles of the topographic",
          "% index over the 637 x 665 study-domain grid, chosen on a 0.25 grid to",
          f"% every class holds between {min(fr):.1f} % and {max(fr):.1f} % of"
          " the domain.",
          "% The index is ln(a / tan beta): upslope contributing area per unit",
          "% contour length over local slope, so a high value means a large",
          "% contributing area, flat ground, or both -- a tendency to saturate.",
          f"% Median cell elevation falls monotonically from {med_el[0]:.0f}"
          f" m in class 1 to {med_el[-1]:.0f} m in class {n}, confirming",
          "% that the sequence runs from the divides to the valley floors.",
          "% Source: data/parflow_raw/static_conditions.nc, variable",
          "% topographic_index (units attribute: -), the same field plotted in",
          "% Figure 1; generated by paper/scripts/table_topo_index.py.", ""]
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "topo_index.tex"), "w") as f:
        f.write("\n".join(L))

    # ------------------------------------------------------------ .json ---
    doc = dict(
        variable=VAR, definition="ln(a / tan beta)",
        description="upslope contributing area per unit contour length "
                    "divided by local slope; high values mean a large "
                    "contributing area and/or flat ground, i.e. a tendency "
                    "to saturate",
        source="data/parflow_raw/static_conditions.nc",
        grid=dict(rows=637, cols=665, cell_km=1.0, n_cells=int(N)),
        data_range=[round(lo_data, 4), round(hi_data, 4)],
        bounds=b, colors=list(sp["colors"]), ticklabels=list(sp["ticklabels"]),
        n_classes=n,
        class_construction="equal-area (11.1 %) quantiles snapped to the "
                           "nearest 0.25",
        area_fraction_percent_range=[min(fr), max(fr)],
        classes=rows)
    with open(os.path.join(OUT, "topo_index.json"), "w") as f:
        json.dump(doc, f, indent=2)
        f.write("\n")

    print(f"  {n} classes over {N} cells, index {lo_data:.2f}..{hi_data:.2f}")
    for r in rows:
        print(f"    {r['klass']}  {r['lower']:>5.2f}-{r['upper']:>5.2f}  "
              f"{r['area_fraction_percent']:>5.1f} %  "
              f"median index {r['median_index']:>5.2f}  "
              f"median elevation {r['median_elevation_m']:>6.1f} m  "
              f"{r['color']}")
    print(f"  area fractions span {min(fr):.1f}-{max(fr):.1f} % "
          f"(target {100 / n:.1f} %)")
    print("  wrote tables/topo_index.tex and tables/topo_index.json")


if __name__ == "__main__":
    main()
