"""
A continuous water year of SEAS5 forcing, spliced from successive initialisations.

WHY THIS EXISTS. One SEAS5 initialisation reaches 215 lead days. This was
verified rather than assumed: 365-day requests for all four October starts were
accepted by the CDS and every one returned forecast_period = 215. An October
initialisation therefore cannot span a water year, and neither can an August
one, because the 215-day ceiling applies to every initialisation month in this
dataset.

WHAT IT DOES INSTEAD. The year is covered by two forecasts: the 1 October
initialisation to the start of May, and the 1 May re-initialisation to the end
of the water year. For any requested date the MOST RECENT initialisation that
covers it is used, which is what an operational forecaster would have in hand on
that date.

WHAT THIS CHANGES SCIENTIFICALLY, and it must be stated wherever these runs are
reported: the composite is a SEQUENCE OF TWO SEASONAL FORECASTS, not a single
annual outlook. Member m of the October forecast and member m of the May
forecast are independent realisations; the atmosphere is re-drawn on 1 May.
Pairing them by index preserves 25 distinct trajectories but introduces a
discontinuity in the forcing realisation at the splice, and any feature in the
error curves at that date should be read with that in mind rather than smoothed
over.

The emulator's STATE is never reinitialised. The chain runs unbroken for all 24
hops; only the forcing's origin changes.

ON REDUCING THE SPLICE DISCONTINUITY. Pairing by index is arbitrary, so the
assignment of May members to October members is free to choose. If the splice
turns out to leave a visible artefact, the fix is an OPTIMAL ASSIGNMENT: the
one-to-one permutation of the 25 May members onto the 25 October members that
minimises total discontinuity, solved with the Hungarian algorithm
(scipy.optimize.linear_sum_assignment).

It must be a bijection, not a nearest-neighbour choice. Selecting, for each
October member, whichever May member continues it best would reuse some May
members and drop others; the composite would then no longer sample the May
forecast distribution, and its spread -- and every probabilistic score built on
that spread -- would be biased. A permutation leaves the marginal distribution
exactly intact.

The cost function must use only the two forecasts, never the verifying truth.
On 1 May a forecaster holds both runs, so an assignment made from them is
operationally available; one made from the observed outcome is not.

MEASURE FIRST. Precipitation is a flux with short memory, not a state, so there
is no physical requirement of continuity across 30 April / 2 May, and both
draws come from much the same early-May climatology. Quantify the jump in
domain-mean forcing and in the groundwater state at the splice against ordinary
day-to-day variability before correcting anything.
"""
import numpy as np
import pandas as pd


class ChainedForcing:
    """Routes each requested date to the most recent initialisation covering it.

    `sources` is a list of (init_timestamp, ArmForcing) in any order. Each
    ArmForcing must already be built on its own initialisation.
    """

    def __init__(self, sources):
        if not sources:
            raise SystemExit("ChainedForcing needs at least one initialisation")
        self.sources = sorted(sources, key=lambda s: pd.Timestamp(s[0]))
        self.inits = [pd.Timestamp(s[0]) for s in self.sources]
        self.label = " + ".join(f"init {i.date()}" for i in self.inits)

    def reach(self, k):
        """Lead days available from source k."""
        src = self.sources[k][1]
        return src._src[list(src._src)[0]].shape[1]

    def _route(self, dates):
        """For each date, the index of the initialisation that should supply it.

        The most recent init whose forecast covers the date wins. A date that no
        initialisation reaches is an error rather than a silent gap: a missing
        forcing day would be interpolated over by the rollout and would not show
        up until the results looked subtly wrong.
        """
        dates = pd.DatetimeIndex(dates)
        pick = np.full(len(dates), -1, dtype=int)
        for k, init in enumerate(self.inits):
            lead = np.asarray((dates - init).days)
            ok = (lead >= 1) & (lead <= self.reach(k))
            pick[ok] = k                      # later inits overwrite earlier
        if (pick < 0).any():
            bad = [str(d.date()) for d, p in zip(dates, pick) if p < 0]
            raise SystemExit(
                f"no initialisation covers {len(bad)} requested day(s): "
                f"{bad[:5]}{' ...' if len(bad) > 5 else ''}. "
                f"Available: {self.label}, each reaching "
                f"{[self.reach(k) for k in range(len(self.sources))]} days.")
        return pick

    def splice_dates(self, dates):
        """The dates at which the supplying initialisation changes."""
        pick = self._route(dates)
        d = pd.DatetimeIndex(dates)
        return [d[i] for i in range(1, len(pick)) if pick[i] != pick[i - 1]]

    def fields(self, gdays, member=0):
        """Same contract as ArmForcing.fields: {var: (len(gdays), 637, 665)}."""
        ref = self.sources[0][1]
        dates = ref.dates(gdays)
        pick = self._route(dates)
        gdays = np.asarray(gdays)
        out = None
        for k in np.unique(pick):
            sel = np.flatnonzero(pick == k)
            part = self.sources[int(k)][1].fields(gdays[sel], member=member)
            if out is None:
                out = {v: np.empty((len(gdays),) + a.shape[1:], dtype=a.dtype)
                       for v, a in part.items()}
            for v, a in part.items():
                out[v][sel] = a
        return out
