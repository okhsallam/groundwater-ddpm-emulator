"""
Figure: ParFlow soil moisture against the emulator at the fourteen well cells.

The same grid as the water-table figure -- fourteen wells by four water years,
chronological -- for the second predicted field.

Neither network logs soil moisture, so there is no observation to compare
against and none is invented: the reference line is ParFlow itself, read from
data/parflow_raw/target2d_wy<YYYY>.nc at exactly the cells the emulator series
were extracted from. This panel therefore asks a different question from the
water-table panel. There, the emulator is judged against the real world and
ParFlow's own bias is removed first; here it is judged against the simulation
it was trained to emulate, so no shift is applied and none is needed.

As in the water-table figure, the seasonal arm is the 360-day spliced water
year ("Dyear") and fills the whole x axis; the day span of every drawn series
is printed so a stale 210-day cache cannot pass unnoticed.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as P
P.apply_style()

import wells_common as C
import wells_grid as G


def main():
    ws = C.load_wells()
    truth = C.parflow_truth("soil_moisture")
    mods = C.available(ws)
    print(f"  forcings present: {', '.join(m['key'] for m in mods)}")
    for m in mods:
        n = sum(1 for v in m["sm"].values() if v is not None)
        print(f"    {m['key']}: soil moisture at {n}/14 wells")
    C.print_coverage(mods, "sm")

    stats = G.build_paged(
        ws, truth, mods, "sm", invert=False,
        offs={w["short_id"]: 0.0 for w in ws},
        value_label="Soil moisture [-]",
        ref_label="ParFlow (the simulation being emulated)",
        title=f"ParFlow soil moisture against the emulator at {len(ws)} "
              "well cells",
        name="fig_wells_soil_moisture",
        tol="1D")
    stats.to_csv(os.path.join(P.FIGS, "wells_sm_scores_by_year.csv"),
                 index=False)
    print(stats.groupby(["forcing", "water_year"])["r"].mean().round(3)
          .to_string())
    miss = stats[~np.isfinite(stats["r"])]
    if len(miss):
        print("  unscored panels:")
        print(miss[["forcing", "well", "water_year", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
