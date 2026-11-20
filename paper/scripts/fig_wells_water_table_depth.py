"""
Figure: observed water table against the emulator, every well, every year.

One row per observation well in the paper's merged sequence -- the eight USGS
wells, then the whole ISWS unconfined network ordered by median depth -- and
four columns, the four water years in chronological order. The grid is
paginated at wells_grid.PAGE_ROWS rows a page; the scores returned span every
page, so nothing downstream reports a single page by accident. Each panel
carries four series: the observed hydrograph, ParFlow -- the simulation the
emulator is trained to reproduce -- and the emulator under reanalysis forcing
and under raw seasonal forcing, with the 10th-90th percentile of each ensemble
shaded and the correlation of each modelled series with the observation.

The seasonal arm is the 360-day spliced water year ("Dyear"), so it fills the
same x axis as the reanalysis arm and the two panel correlations are read on
the same sample of observations. The day span of every drawn series is printed
to stdout for exactly that reason: a seasonal line stopping in April means the
superseded 210-day cache was read.

ParFlow belongs in this panel because without it the reader cannot tell an
emulator error from an error it inherited: the emulator can only be as right as
the simulation it reproduces.

The observation is shifted by ONE CONSTANT PER SITE and NOTHING ELSE IS MOVED
-- ParFlow least of all, which is drawn on its own true scale. ParFlow holds
the water table systematically shallower than the wells record it, which is a
model-vs-reality discrepancy the emulator did not create and must not be
charged with. The constant is the difference of means over the reanalysis-
forced period -- the one series that covers the whole record, so the same
number applies to every panel in the row -- and it is printed to stdout and
carried in the wells table rather than in the row label.

THE SHIFT CANNOT PUSH AN ISWS WELL BELOW ZERO. Depth to water is positive
downward, so a negative value puts the water table above the land surface. The
constant is a difference of MEANS and knows nothing about the minimum, so it
could and did push the wet end of a shallow record metres into the air. It is
now capped -- shrunk toward zero by exactly as much as the minimum needs, and
no further -- and whatever the cap cannot reach, because the raw record is
itself negative, is clipped at zero. Both stages are audited to stdout and to
figures/wells_offset_audit.csv. See wells_common.NONNEG_NETS.

COLOUR. Navy (P.C_TRUTH) is the simulator in both well figures, so ParFlow
takes it here as it already does in the soil-moisture figure; the observation,
which no model is fitted to, takes the paper's neutral reference grey
(P.C_PERS).
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
    obs = C.observations(ws)
    mods = C.available(ws)
    have = [m["key"] for m in mods]
    print(f"  forcings present: {', '.join(have)}")
    C.print_coverage(mods, "wtd")

    # One constant per site, from the reanalysis-forced series, aligning the
    # observed minimum with the simulated one. Depth to water is positive
    # downward, so this makes a negative plotted depth impossible by
    # construction rather than by repair. The shifted series is built ONCE,
    # here, and handed to the grid as the reference -- so `offs` passed below
    # is all zeros and the grid applies nothing further.
    ref_mod = next((m for m in mods if m["key"] == "nldas"), mods[0])
    offs, obs_sh, audit = C.shifted_observations(ws, obs, ref_mod["wtd"],
                                                 report=True)
    print("  observation shift [m]: min(simulated) - min(observed)")
    for _, r in audit.iterrows():
        print(f"    {r['no']:>2}  {r['name']:<18} {r['net']:<4} "
              f"raw min {r['raw_min_m']:+7.3f}  offset {r['offset_m']:+7.3f}  "
              f"shifted min {r['shifted_min_m']:+7.3f}")
    print(f"  offsets span {audit.offset_m.min():+.3f} to "
          f"{audit.offset_m.max():+.3f} m; "
          f"{int(audit.n_negative.sum())} plotted values below zero "
          f"(must be 0); nothing clipped")
    audit.to_csv(os.path.join(P.FIGS, "wells_offset_audit.csv"), index=False)

    # The simulator itself, at the same cells. It is passed as
    # `extra`, NOT as `ref`, which is the only series `offs` is applied to:
    # ParFlow is drawn on its own true scale.
    truth = C.parflow_truth("water_table_depth")

    stats = G.build_paged(
        ws, obs_sh, mods, "wtd", invert=True,
        # the shift is already in obs_sh; nothing further is applied
        offs={w["short_id"]: 0.0 for w in ws},
        value_label="Depth to water below land surface [m]",
        ref_label="Observation",
        # the title is kept inside FIG_W: a suptitle wider than the canvas
        # makes bbox_inches="tight" grow the saved figure, and the page
        # reduction then shrinks every font in it
        title=f"Water table at {len(ws)} wells: observation, ParFlow "
              "and the emulator",
        name="fig_wells_water_table_depth",
        tol="3D",
        extra=dict(key="parflow", color=P.C_TRUTH, short="ParFlow",
                   label="ParFlow (the simulation being emulated, unshifted)",
                   series=truth))
    stats.to_csv(os.path.join(P.FIGS, "wells_wtd_scores_by_year.csv"),
                 index=False)
    t = stats.groupby(["forcing", "water_year"])["r"].mean().round(3)
    print(t.to_string())
    miss = stats[~np.isfinite(stats["r"])]
    if len(miss):
        print("  unscored panels:")
        print(miss[["forcing", "well", "water_year", "n"]].to_string(index=False))


if __name__ == "__main__":
    main()
