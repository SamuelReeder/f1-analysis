"""Paired explanation seasons using the original simulator's random stream.

The unchanged simulator supplies the baseline and input/failure checks. Probes
replay its event randomness, sharing a changed qualifying grid between the
qualifying-only and combined interventions. Probe championship countback is
unneeded: only the intervened driver's season points enter the summary.
"""
import copy

import numpy as np

from f1rank.overall.order import numpy_perturbed_order
from f1rank.overall.scoring import score_grand_prix
from f1rank.overall.simulation import qualifying_grid, simulate

PROBE_ORDER = ("qualifying_pace", "total_race_pace", "all")


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
    q_mean, r_mean = q.mean(axis=-1), r.mean(axis=-1)
    probes = np.zeros((nd, replicates, n, len(PROBE_ORDER)))
    rng = np.random.default_rng(seed)
    probe_rng = np.random.default_rng(seed)
    for rep in range(replicates):
        for scenario in scenarios:
            qday = rng.normal(size=q.shape) * draws.q_weekend_sd[:, None]
            rday = (draws.shared_weekend[:, None] * qday
                    + rng.normal(size=q.shape) * draws.r_weekend_sd[:, None])
            qualifying_state = copy.deepcopy(rng.bit_generator.state)
            grid = qualifying_grid(q, qday, draws.q_scale, draws.q_log_scale_sd,
                                   draws.q_df, rng, advancing)
            gumbels = rng.gumbel(size=q.shape)
            hazard = draws.retirement_hazard[:, scenario.context, None]
            uniform = rng.uniform(size=q.shape)
            failed_after = np.floor(np.divide(np.log1p(-uniform), np.log1p(-hazard),
                out=np.full(q.shape, np.inf), where=hazard > 0))
            laps = np.minimum(failed_after, scenario.laps).astype(int)
            if (laps.max(axis=-1) < scenario.laps).any():
                raise ValueError("No full-distance finisher: shortened-race model required")
            pace = draws.pace_coefficient[:, scenario.context, None]
            grid_effect = draws.grid_coefficient[:, scenario.context, None]
            for driver in range(n):
                q_probe, r_probe = q.copy(), r.copy()
                q_probe[:, driver], r_probe[:, driver] = q_mean, r_mean
                probe_rng.bit_generator.state = copy.deepcopy(qualifying_state)
                changed_grid = qualifying_grid(q_probe, qday, draws.q_scale,
                    draws.q_log_scale_sd, draws.q_df, probe_rng, advancing)
                # Keep the original expression/order of operations. Adding a
                # strength delta to a rounded baseline need not be bit-identical.
                for k, (race_state, qualifying) in enumerate(
                        ((r, changed_grid), (r_probe, grid), (r_probe, changed_grid))):
                    strength = pace * (race_state + rday) - grid_effect * np.log(qualifying)
                    line = np.argsort(numpy_perturbed_order(strength, gumbels), axis=-1) + 1
                    _, points = score_grand_prix(laps, line, scheduled_laps=scenario.laps,
                        leader_laps=scenario.laps, consecutive_racing_laps=scenario.consecutive_racing_laps)
                    probes[:, rep, driver, k] += points[:, driver]
    return baseline, probes[:, :, reverse]
