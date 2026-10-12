"""Paired explanation seasons using the original simulator's random stream.

The unchanged simulator supplies the baseline and input/failure checks. Probes
replay its event randomness, sharing a changed qualifying grid between the
qualifying-only and combined interventions. Probe championship countback is
unneeded: only the intervened driver's season points enter the summary.
"""
import numpy as np

from f1rank.overall.order import numpy_perturbed_order
from f1rank.overall.scoring import score_grand_prix
from f1rank.overall.simulation import QUALIFYING_FIELDS, qualifying_grid, simulate

PROBE_ORDER = ("qualifying_pace", "total_race_pace", "all")


class _RecordedQualifyingNoise:
    """Record the existing qualifying kernel's calls without changing its RNG."""

    def __init__(self, rng):
        self.rng, self.calls = rng, []

    def normal(self, *, size):
        value = self.rng.normal(size=size)
        value.flags.writeable = False
        self.calls.append(("normal", value, None))
        return value

    def standard_t(self, df, *, size):
        value = self.rng.standard_t(df, size=size)
        value.flags.writeable = False
        self.calls.append(("standard_t", value, np.asarray(df).copy()))
        return value


class _RepeatedQualifyingNoise:
    """Repeat each draw lane for its interventions; never generate new noise."""

    def __init__(self, recorded, repeats):
        if type(repeats) is not int or repeats < 1:
            raise ValueError("Need a positive number of qualifying interventions")
        self.calls, self.repeats, self.position = tuple(recorded.calls), repeats, 0

    def _read(self, method, size, df=None):
        if self.position >= len(self.calls):
            raise ValueError("Qualifying replay requested unrecorded randomness")
        expected, value, original_df = self.calls[self.position]
        shape = (size,) if np.isscalar(size) else tuple(size)
        if (expected != method or shape != (value.shape[0] * self.repeats, *value.shape[1:])
                or (method == "standard_t" and not np.array_equal(df, np.repeat(original_df, self.repeats, axis=0)))):
            raise ValueError("Qualifying replay method, axes or degrees of freedom disagree")
        self.position += 1
        return np.repeat(value, self.repeats, axis=0)

    def normal(self, *, size):
        return self._read("normal", size)

    def standard_t(self, df, *, size):
        return self._read("standard_t", size, df)

    def finish(self):
        if self.position != len(self.calls):
            raise ValueError("Qualifying replay did not consume every recorded request")


def paired_seasons(draws, scenarios, *, replicates, seed, advancing):
    """Exact baseline and own-driver points for each field-mean intervention.

Call through the bounded summary generator, which validates the stream address,
block dimensions, definition, contract and dependency fingerprints. Baseline
simulation here retains all original scenario and failure validation too.
"""
    baseline = simulate(draws, scenarios, replicates=replicates, seed=seed, advancing=advancing)
    ix = np.argsort(draws.drivers)
    reverse = np.argsort(ix)
    q, r = draws.qualifying[:, ix], draws.race[:, ix]
    nd, n = q.shape
    if advancing is None:
        advancing = QUALIFYING_FIELDS[n]
    q_mean, r_mean = q.mean(axis=-1), r.mean(axis=-1)
    # Axes: draw, intervened driver, competitor. The diagonal is each
    # intervened driver's own state, with every competitor retained.
    drivers = np.arange(n)
    q_probe = np.broadcast_to(q[:, None, :], (nd, n, n)).copy()
    r_probe = np.broadcast_to(r[:, None, :], (nd, n, n)).copy()
    q_probe[:, drivers, drivers] = q_mean[:, None]
    r_probe[:, drivers, drivers] = r_mean[:, None]
    q_parameters = [np.repeat(value, n) for value in
                    (draws.q_scale, draws.q_log_scale_sd, draws.q_df)]
    probes = np.zeros((nd, replicates, n, len(PROBE_ORDER)))
    rng = np.random.default_rng(seed)
    for rep in range(replicates):
        for scenario in scenarios:
            qday = rng.normal(size=q.shape) * draws.q_weekend_sd[:, None]
            rday = (draws.shared_weekend[:, None] * qday
                    + rng.normal(size=q.shape) * draws.r_weekend_sd[:, None])
            recorded = _RecordedQualifyingNoise(rng)
            grid = qualifying_grid(q, qday, draws.q_scale, draws.q_log_scale_sd,
                                   draws.q_df, recorded, advancing)
            gumbels = rng.gumbel(size=q.shape)
            hazard = draws.retirement_hazard[:, scenario.context, None]
            uniform = rng.uniform(size=q.shape)
            failed_after = np.floor(np.divide(np.log1p(-uniform), np.log1p(-hazard),
                out=np.full(q.shape, np.inf), where=hazard > 0))
            laps = np.minimum(failed_after, scenario.laps).astype(int)
            if (laps.max(axis=-1) < scenario.laps).any():
                raise ValueError("No full-distance finisher: shortened-race model required")
            replay = _RepeatedQualifyingNoise(recorded, n)
            changed_grid = qualifying_grid(q_probe.reshape(nd*n, n), np.repeat(qday, n, axis=0),
                *q_parameters, replay, advancing).reshape(nd, n, n)
            replay.finish()
            pace = draws.pace_coefficient[:, scenario.context, None, None]
            grid_effect = draws.grid_coefficient[:, scenario.context, None, None]
            many_gumbels = np.broadcast_to(gumbels[:, None, :], (nd, n, n))
            many_laps = np.broadcast_to(laps[:, None, :], (nd, n, n))
            # Keep the original expression/order of operations. Adding a
            # strength delta to a rounded baseline need not be bit-identical.
            for k, (race_state, qualifying) in enumerate(
                    ((r[:, None, :], changed_grid), (r_probe, grid[:, None, :]), (r_probe, changed_grid))):
                strength = pace * (race_state + rday[:, None, :]) - grid_effect * np.log(qualifying)
                line = np.argsort(numpy_perturbed_order(strength, many_gumbels), axis=-1) + 1
                _, points = score_grand_prix(many_laps, line, scheduled_laps=scenario.laps,
                    leader_laps=scenario.laps, consecutive_racing_laps=scenario.consecutive_racing_laps)
                probes[:, rep, :, k] += points[:, drivers, drivers]
    return baseline, probes[:, :, reverse]
