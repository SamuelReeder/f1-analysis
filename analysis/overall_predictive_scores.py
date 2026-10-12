"""Blind scoring primitives for the registered Overall evaluation architecture.

This module does not fit, choose folds, set calibration gates or publish ranks.
The scoring contract alone is not a complete experimental registration.
"""
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = json.loads((ROOT / "docs/overall_scoring_contract.json").read_text())


def log_predictive(log_probabilities):
    """Integrate likelihood over draws, rather than average their log scores."""
    values = np.asarray(log_probabilities, dtype=float)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError("Expected nonempty draw-by-event log probabilities")
    if np.isnan(values).any() or np.isposinf(values).any():
        raise ValueError("Invalid log probabilities")
    # logaddexp handles all-impossible events without subtracting -inf from -inf.
    return np.logaddexp.reduce(values, axis=0) - np.log(len(values))


def joint_log_predictive(components):
    """Combine conditional factors within each SAME joint draw before integrating.

Identity checks protect the interface. Producer provenance must independently
establish that those identities came from the asserted posterior and targets.
"""
    if not components:
        raise ValueError("No predictive components")
    first = components[0]
    identities = ("posterior_id", "draw_ids", "event_ids")
    if not isinstance(first.get("posterior_id"), str) or not first["posterior_id"]:
        raise ValueError("Missing posterior identity")
    for key in ("draw_ids", "event_ids"):
        values = first.get(key)
        if not isinstance(values, list) or not values or not all(isinstance(v, str) and v for v in values) or len(set(values)) != len(values):
            raise ValueError("Missing or duplicate predictive identities")
    shape = (len(first["draw_ids"]), len(first["event_ids"]))
    total = np.zeros(shape)
    for component in components:
        if any(component.get(k) != first[k] for k in identities):
            raise ValueError("Predictive factors must share posterior, draws and event order")
        values = np.asarray(component["log_probabilities"], dtype=float)
        if values.shape != shape:
            raise ValueError("Predictive factor axes do not match their identities")
        log_predictive(values)  # validate each factor before summing
        total += values
    return log_predictive(total)


def _discrete(probabilities, observed):
    p = np.asarray(probabilities, dtype=float)
    y = np.asarray(observed)
    if p.ndim != 2 or min(p.shape) == 0 or y.shape != (len(p),):
        raise ValueError("Expected observation-by-category probabilities and outcomes")
    if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(
            p.sum(axis=1), 1., rtol=0., atol=CONTRACT["probability_sum_absolute_tolerance"]):
        raise ValueError("Invalid probability distribution; do not silently normalize")
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError("Observed category outside the declared support")
    return p, y


def ordinal_scores(probabilities, observed):
    """Zero-based, ordered categories; RPS is the unnormalised CDF squared error."""
    p, y = _discrete(probabilities, observed)
    mass = p[np.arange(len(p)), y]
    with np.errstate(divide="ignore"):
        log_score = np.log(mass)
    target_cdf = np.arange(p.shape[1] - 1)[None, :] >= y[:, None]
    rps = ((p.cumsum(axis=1)[:, :-1] - target_cdf) ** 2).sum(axis=1)
    return dict(log_probability=log_score, ranked_probability_score=rps)


def randomized_pit(probabilities, observed, uniforms):
    """Caller supplies and records the randomization; no seed search here."""
    p, y = _discrete(probabilities, observed)
    u = np.asarray(uniforms, dtype=float)
    if u.shape != y.shape or not np.isfinite(u).all() or (u < 0).any() or (u > 1).any():
        raise ValueError("Invalid PIT randomization")
    mass = p[np.arange(len(p)), y]
    lower = (p * (np.arange(p.shape[1])[None, :] < y[:, None])).sum(axis=1)
    return lower + u * mass


def retirement_scores(probability, retired):
    p, y = np.asarray(probability, dtype=float), np.asarray(retired)
    if p.ndim != 1 or not len(p) or y.shape != p.shape or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected aligned retirement probabilities and binary outcomes")
    probs, outcome = _discrete(np.column_stack([1 - p, p]), y.astype(int))
    return dict(log_probability=ordinal_scores(probs, outcome)["log_probability"],
                brier_score=(p - y) ** 2)


def _event_id(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}", value) or value.endswith("-00"):
        raise ValueError("Invalid event identity")
    return value


def _field(value):
    if not isinstance(value, list) or len(value) < 2 or not all(isinstance(v, str) and v for v in value) or len(value) != len(set(value)):
        raise ValueError("Invalid field")
    return sorted(value)


def _paired(differences, seasons):
    policy = CONTRACT["paired_comparison"]
    years = sorted({int(y) for y in seasons})
    with np.errstate(over="ignore", invalid="ignore"):
        sums = np.array([differences[seasons == y].sum() for y in years])
    counts = np.array([(seasons == y).sum() for y in years])
    picks = np.random.default_rng(policy["seed"]).integers(len(years), size=(policy["bootstrap_replicates"], len(years)))
    with np.errstate(over="ignore", invalid="ignore"):
        boot = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
        total = sums.sum()
    if not np.isfinite(sums).all() or not np.isfinite(boot).all() or not np.isfinite(total):
        raise ValueError("nonfinite paired aggregation; comparison cannot pass")
    low, high = np.percentile(boot, policy["percentiles"])
    return dict(mean_difference_per_race=float(total / counts.sum()),
                interval=[float(low), float(high)], seasons=years,
                n_races=int(counts.sum()),
                comparison_rule_passed=bool(len(years) >= policy["minimum_seasons"] and low > 0))


def compare_arms(records, inventory):
    """Require a complete, common event inventory for this ONE forecast origin.

Inventory records identify event, strictly earlier training cutoff, complete
field, information set and target. This function cannot certify a registration
date or prove input provenance: the evaluation runner must do those checks.
"""
    if not inventory:
        raise ValueError("Empty evaluation inventory")
    expected = {}
    for event in inventory:
        key = _event_id(event["event_id"])
        if key in expected or _event_id(event["training_cutoff"]) >= key:
            raise ValueError("Duplicate event or training reaching the test event")
        _field(event["field"])
        if event["information_set"] not in ("prequalifying", "actual_grid") or event["target"] != CONTRACT["primary_target"]:
            raise ValueError("Unsupported comparison origin or target")
        expected[key] = event
    if len({e["information_set"] for e in inventory}) != 1:
        raise ValueError("Do not pool different forecast origins")
    arms = [CONTRACT["candidate_arm"], CONTRACT["primary_baseline"], *CONTRACT["secondary_baselines"]]
    found, errors = {}, []
    for row in records:
        event, arm = row.get("event_id"), row.get("arm")
        key = (event, arm)
        if event not in expected or arm not in arms or key in found:
            errors.append(dict(event_id=event, arm=arm, reason="unexpected or duplicate result"))
            continue
        found[key] = row
        target = expected[event]
        try:
            if any(row.get(k) != target[k] for k in ("training_cutoff", "information_set", "target")) or _field(row.get("field")) != _field(target["field"]):
                raise ValueError("mismatched cutoff, field, origin or target")
            if row.get("status") != "scored":
                raise ValueError("required fit or score failed")
            score = row.get("log_probability")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not np.isfinite(score) or score > 0:
                raise ValueError("invalid or nonfinite log probability; infinite comparisons cannot pass")
        except ValueError as exc:
            errors.append(dict(event_id=event, arm=arm, reason=str(exc)))
    for event in expected:
        for arm in arms:
            if (event, arm) not in found:
                errors.append(dict(event_id=event, arm=arm, reason="missing required result"))
    result = dict(complete=not errors, errors=errors, required_events=sorted(expected),
                  required_arms=arms, comparison_rule_passed=False, publication_allowed=False,
                  comparisons={})
    if errors:
        return result
    events = sorted(expected)
    seasons = np.array([int(event[:4]) for event in events])
    candidate = np.array([found[(e, arms[0])]["log_probability"] for e in events])
    for arm in arms[1:]:
        baseline = np.array([found[(e, arm)]["log_probability"] for e in events])
        try:
            result["comparisons"][arm] = _paired(candidate - baseline, seasons)
        except ValueError as exc:
            result.update(complete=False, comparisons={})
            result["errors"].append(dict(arm=arm, reason=str(exc)))
            return result
    result["comparison_rule_passed"] = result["comparisons"][CONTRACT["primary_baseline"]]["comparison_rule_passed"]
    return result
