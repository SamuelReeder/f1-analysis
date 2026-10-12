"""Development condition frequencies and joint draws; no fit or acceptance claim."""
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import gammaln

from analysis.overall_forecast_catalog import _event
from f1rank.overall.design import FIRST_LAP_SEASON
from f1rank.overall.fitting import arrays_digest


POLICY_PATH = Path(__file__).resolve().parents[1] / "docs/overall_condition_policy.json"
POLICY = json.loads(POLICY_PATH.read_text())
STATES = tuple(POLICY["states"])


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _indices(value, size, name):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in "iu" or (a < 0).any() or (a >= size).any():
        raise ValueError(f"Invalid {name} indices")
    return a


def training_history(design):
    """Deduplicate risk observations to events, preserving unknown exclusions."""
    q, o = design.qualifying, design.outcomes
    cutoff = _event(design.catalog["cutoff"])
    if (q.events.event_id.duplicated().any() or q.events.empty
            or q.entries.duplicated(["event_id", "driver_id"]).any()
            or not np.array_equal(q.entries.entry_idx, np.arange(len(q.entries)))):
        raise ValueError("Nonunique or misindexed training catalogue")
    for frame in (q.events, q.entries):
        if any(_event(e) > cutoff for e in frame.event_id):
            raise ValueError("Condition history reaches beyond the training cutoff")
    circuits = design.catalog["circuits"]
    if (len(set(circuits)) != len(circuits) or len(circuits) != o["n_circuits"]
            or any(not isinstance(c, str) or not c for c in circuits)):
        raise ValueError("Invalid circuit catalogue")
    event_circuit = {}
    for row in q.events.itertuples():
        if (not isinstance(row.circuit_idx, (int, np.integer)) or not 0 <= row.circuit_idx < len(circuits)
                or circuits[row.circuit_idx] != row.circuit_id):
            raise ValueError("Event circuit disagrees with the training catalogue")
        event_circuit[row.event_id] = row.circuit_id
    context_circuit = _indices(o["context_circuit"], len(circuits), "context circuit")
    wet, neutral = np.asarray(o["context_wet"]), np.asarray(o["context_sc"])
    if (wet.shape != context_circuit.shape or neutral.shape != wet.shape
            or not np.isin(wet, [0, 1]).all() or not np.isin(neutral, [0, 1]).all()):
        raise ValueError("Invalid observed condition labels")
    entry = _indices(o["risk_entry"], len(q.entries), "risk entry")
    context = _indices(o["risk_context"], len(context_circuit), "risk context")
    if len(entry) != len(context) or len(np.unique(entry)) != len(entry):
        raise ValueError("Missing or duplicate risk observations")
    observed = {}
    for index, k in zip(entry, context):
        event = str(q.entries.iloc[index].event_id)
        circuit = circuits[context_circuit[k]]
        state = int(wet[k]) * 2 + int(neutral[k])
        if event < f"{FIRST_LAP_SEASON}-01" or event_circuit.get(event) != circuit:
            raise ValueError("Risk context is outside the outcome window or has the wrong circuit")
        if event in observed and observed[event] != (circuit, state):
            raise ValueError("Conflicting condition observations within an event")
        observed[event] = (circuit, state)
    excluded = sorted(design.catalog["excluded_context_events"])
    if (len(set(excluded)) != len(excluded) or any(_event(e) > cutoff for e in excluded)
            or set(excluded) & set(observed)):
        raise ValueError("Unknown context cannot contribute condition counts")
    counts = {c: [0] * len(STATES) for c in circuits}
    rows = []
    for event, (circuit, state) in sorted(observed.items()):
        counts[circuit][state] += 1
        rows.append(dict(event_id=event, circuit_id=circuit, state=STATES[state]))
    eligible = {e for e in event_circuit if e >= f"{FIRST_LAP_SEASON}-01"}
    history = dict(training_cutoff=cutoff,
        training_arrays_key=arrays_digest(dict(qualifying=q.arrays(), laps=design.laps, outcomes=o)),
        states=list(STATES), event_contexts=rows, counts=counts,
        excluded_context_events=excluded, events_without_risk_context=sorted(eligible - set(observed)),
        policy_id=POLICY["id"], policy_sha256=_hash(POLICY),
        source_availability_verified=False, independent_evidence_established=False,
        publication_allowed=False)
    return dict(history, history_sha256=_hash(history))


class ConditionLaw:
    """Conjugate circuit probabilities, with persistent uncertainty across visits."""
    def __init__(self, design):
        self.history = training_history(design)
        self._alpha = {c: np.asarray(v, float) + POLICY["prior_alpha"]
                       for c, v in self.history["counts"].items()}

    def alpha(self, circuit):
        if not isinstance(circuit, str) or not circuit:
            raise ValueError("An explicit circuit identity is required")
        return self._alpha.get(circuit, np.asarray(POLICY["prior_alpha"], float)).copy()

    def mean(self, circuit):
        a = self.alpha(circuit)
        return a / a.sum()

    def sequence_log_probability(self, circuit, states):
        """Probability of an ordered condition sequence, integrating shared phi."""
        states = _indices(states, len(STATES), "condition state")
        a = self.alpha(circuit)
        counts = np.bincount(states, minlength=len(STATES))
        return float(gammaln(a.sum()) - gammaln(a.sum() + len(states))
                     + (gammaln(a + counts) - gammaln(a)).sum())

    def draw(self, catalog, *, seed, shape, draw_range):
        from analysis.overall_observation_kernels import ObservationNoise
        ObservationNoise(seed, shape, draw_range)  # common strict axis/seed validation
        if (catalog["training_cutoff"] != self.history["training_cutoff"]
                or catalog["training_arrays_key"] != self.history["training_arrays_key"]
                or _event(catalog["event_id"]) <= self.history["training_cutoff"]
                or catalog["future_weather"] != "unknown" or catalog["future_neutralisation"] != "unknown"):
            raise ValueError("Conditions require the matching bounded history and an unknown future event")
        circuit, event = catalog["circuit_id"], catalog["event_id"]
        def rng(site, identity):
            key = _hash(["overall_conditions", POLICY["id"], seed, site, identity])
            return np.random.default_rng(int(key, 16))
        # Replaying the full small probability stream gives exact batch invariance.
        # No event key here: visits share a circuit draw in each joint trajectory.
        probabilities = rng("circuit_probabilities", circuit).dirichlet(self.alpha(circuit), size=tuple(shape))
        u = rng("event_state", [circuit, event]).random(size=tuple(shape))
        cdf = probabilities.cumsum(axis=-1)
        cdf[..., -1] = 1.  # exact categorical endpoint, not smoothing probabilities
        state = (u[..., None] >= cdf).sum(axis=-1)
        start, stop = draw_range
        state, probabilities = state[:, start:stop], probabilities[:, start:stop]
        return dict(state=state, wet=state // 2 == 1, neutralised=state % 2 == 1,
                    probabilities=probabilities,
                    log_probability=np.log(np.take_along_axis(probabilities, state[..., None], axis=-1)[..., 0]))


def iter_forecast_weekends(design, snapshots, posterior, scenarios, *, advancing, seed,
                           training_arrays_key, draw_batch_size=128, unseen_circuit_policy=None,
                           unseen_track_policy=None):
    """Unknown conditions integrated by joint simulation, still conditional on operations.

    Scenarios supply only exposure_laps and grid_policy. No observed future
    wetness or neutralisation is accepted. This does not establish calibrated
    forecasts, independent evidence, official classification or publication.
    """
    from analysis.overall_observation_kernels import _iter_weekends
    law = ConditionLaw(design)
    if law.history["training_arrays_key"] != training_arrays_key:
        raise ValueError("Condition history and forecast training arrays disagree")
    yield from _iter_weekends(design, snapshots, posterior, scenarios,
        advancing=advancing, seed=seed, training_arrays_key=training_arrays_key,
        draw_batch_size=draw_batch_size, unseen_circuit_policy=unseen_circuit_policy,
        unseen_track_policy=unseen_track_policy, condition_law=law)
