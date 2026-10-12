"""Historical condition law: exact probabilities and full paired-draw integration."""
import copy
import itertools

import numpy as np
import pytest
from scipy.stats import dirichlet

from analysis import overall_future_conditions as conditions
from analysis import overall_observation_kernels as kernels
from analysis import overall_predictive_scores as scores
from analysis.overall_forecast_catalog import build_catalog
from analysis.overall_forecast_states import UNSEEN_TRACK_POLICY
from f1rank.overall.design import build
from test_overall_forecast_states import fixture, future_path
from test_overall_observation_kernels import observation_posterior, scenarios


def operations(path):
    return {event: {k: v for k, v in scenario.items() if k in ("exposure_laps", "grid_policy")}
            for event, scenario in scenarios(path).items()}


def predict(design, path, posterior, batch_size=3, conditional=None, **kwargs):
    return list(conditions.iter_forecast_weekends(design, path, posterior,
        operations(path) if conditional is None else conditional, advancing=(4, 2), seed=41,
        training_arrays_key=build_catalog(design, path[0])["training_arrays_key"],
        draw_batch_size=batch_size, **kwargs))


def test_history_counts_events_not_drivers_and_keeps_wet_events_without_finishers():
    source, clean, _, _ = fixture()
    source["race_weather"].loc[lambda x: x.event_id >= "2022-01", "rainfall"] = True
    source["race_track_status"].loc[lambda x: x.event_id == "2022-02", "status"] = "7"
    source["race"].loc[lambda x: x.event_id == "2022-02", "status"] = "Engine"
    design = build("2022-02", tables=source, clean=clean)
    law = conditions.ConditionLaw(design)
    assert law.history["counts"] == {"track": [1, 0, 1, 1]}
    assert "2022-02" not in design.catalog["outcome_events"]
    assert law.history["event_contexts"][-1]["state"] == "wet_neutralised"
    np.testing.assert_array_equal(law.alpha("track"), [1.5, .5, 1.5, 1.5])
    assert len(law.history["event_contexts"]) == 3


def test_future_rows_never_enter_history_and_unknown_history_is_preserved():
    source, clean, original, _ = fixture()
    before = conditions.ConditionLaw(original).history
    source["race_weather"].loc[lambda x: x.event_id > "2022-02", "rainfall"] = True
    source["race_track_status"].loc[lambda x: x.event_id > "2022-02", "status"] = "7"
    later = build("2022-02", tables=source, clean=clean)
    assert conditions.ConditionLaw(later).history == before
    source["race_track_status"].loc[lambda x: x.event_id == "2022-02", "status"] = None
    law = conditions.ConditionLaw(build("2022-02", tables=source, clean=clean))
    assert law.history["counts"] == {"track": [2, 0, 0, 0]}
    assert law.history["excluded_context_events"] == ["2022-02"]
    assert law.history["events_without_risk_context"] == ["2022-02"]


def test_prior_and_shared_circuit_sequence_probabilities_are_exact():
    _, _, design, _ = fixture()
    law = conditions.ConditionLaw(design)
    np.testing.assert_array_equal(law.alpha("new"), [.5] * 4)
    np.testing.assert_array_equal(law.mean("new"), [.25] * 4)
    # Shared uncertainty makes repeated identical conditions more likely than
    # multiplying posterior-mean probabilities independently across visits.
    np.testing.assert_allclose(np.exp(law.sequence_log_probability("new", [0, 0])), 1 / 8)
    np.testing.assert_allclose(np.exp(law.sequence_log_probability("new", [0, 1])), 1 / 24)
    assert np.exp(law.sequence_log_probability("new", [0, 0])) > law.mean("new")[0] ** 2
    total = sum(np.exp(law.sequence_log_probability("track", list(s)))
                for s in itertools.product(range(4), repeat=3))
    np.testing.assert_allclose(total, 1., atol=1e-14)


def test_drawn_circuit_means_match_analytic_posterior_and_visits_share_draws():
    _, _, design, snap = fixture()
    law, path = conditions.ConditionLaw(design), future_path(snap)
    catalogs = [build_catalog(design, s) for s in path]
    shape = (2, 16384)
    a, b = [law.draw(c, seed=41, shape=shape, draw_range=(0, shape[1])) for c in catalogs]
    np.testing.assert_array_equal(a["probabilities"], b["probabilities"])
    assert np.any(a["state"] != b["state"])
    reference = dirichlet(law.alpha("track"))
    error = abs(a["probabilities"].mean(axis=(0, 1)) - reference.mean())
    assert np.all(error <= 6 * np.sqrt(reference.var() / np.prod(shape)))
    # One state per event/draw, with no driver axis or independently sampled flags.
    assert a["wet"].shape == shape
    np.testing.assert_array_equal(a["state"], 2 * a["wet"] + a["neutralised"])


@pytest.mark.parametrize("fault", ["future", "duplicate_risk", "conflicting_context", "wrong_circuit", "excluded", "bad_label"])
def test_corrupt_context_history_is_rejected(fault):
    _, _, design, _ = fixture()
    design = copy.deepcopy(design)
    if fault == "future":
        design.catalog["cutoff"] = "2022-01"
    elif fault == "duplicate_risk":
        design.outcomes["risk_entry"][1] = design.outcomes["risk_entry"][0]
    elif fault == "conflicting_context":
        design.outcomes["risk_context"][1] = 1
    elif fault == "wrong_circuit":
        design.qualifying.events.loc[0, "circuit_id"] = "wrong"
    elif fault == "excluded":
        design.catalog["excluded_context_events"] = ["2022-02"]
    else:
        design.outcomes["context_wet"][0] = 2
    with pytest.raises(ValueError):
        conditions.ConditionLaw(design)


def test_full_weekend_uses_same_conditional_kernels_and_joint_parameter_draws():
    _, _, design, snap = fixture()
    p = observation_posterior(design)
    # Deliberately distinguish draws and conditions to expose marginal averaging.
    p["overall_pace_wet"][:] = .8
    p["overall_grid_sc"][:] = -.9
    actual = predict(design, [snap], p, batch_size=7)[0]
    event = actual["events"][0]
    for state in range(4):
        explicit = scenarios([snap])
        explicit[snap["event"]["event_id"]].update(wet=bool(state // 2), neutralised=bool(state % 2))
        reference = list(kernels.iter_conditional_weekends(design, [snap], p, explicit,
            advancing=(4, 2), seed=41, training_arrays_key=actual["training_arrays_key"], draw_batch_size=7))[0]["events"][0]
        selected = event["conditions"]["state"] == state
        for key in ("strength", "grid_feature_slot", "running_positions", "joint_order_completion_log_probability"):
            np.testing.assert_array_equal(event[key][selected], reference[key][selected])
        for key in ("completed", "failed", "log_probability"):
            np.testing.assert_array_equal(event["completion"][key][selected], reference["completion"][key][selected])
    np.testing.assert_allclose(event["joint_conditions_order_completion_log_probability"],
        event["joint_order_completion_log_probability"] + event["conditions"]["log_probability"])
    assert actual["conditions_forecast"]
    assert not any(actual[k] for k in ("publication_allowed", "convergence_verified", "independent_evidence_established"))


def test_batches_field_permutation_prefix_and_unseen_circuit_preserve_joint_trajectories():
    _, _, design, snap = fixture()
    path, p = future_path(snap), observation_posterior(design)
    for s in path:
        s["event"]["circuit_id"] = "new"
    kw = dict(unseen_circuit_policy=kernels.UNSEEN_CIRCUIT_POLICY, unseen_track_policy=UNSEEN_TRACK_POLICY)
    whole = predict(design, path, p, batch_size=7, **kw)[0]
    batched = predict(design, path, p, batch_size=3, **kw)
    shuffled = copy.deepcopy(path)
    for s in shuffled:
        s["entrants"].reverse()
    permuted = predict(design, shuffled, p, batch_size=7, **kw)[0]
    prefix = predict(design, path[:1], p, batch_size=7, **kw)[0]
    for i, event in enumerate(whole["events"]):
        for key in ("state", "probabilities", "wet", "neutralised", "log_probability"):
            value = event["conditions"][key]
            np.testing.assert_array_equal(value, np.concatenate([b["events"][i]["conditions"][key] for b in batched], axis=1))
            np.testing.assert_array_equal(value, permuted["events"][i]["conditions"][key])
            if i == 0:
                np.testing.assert_array_equal(value, prefix["events"][0]["conditions"][key])
        for key in ("strength", "running_positions", "joint_order_completion_log_probability"):
            np.testing.assert_array_equal(event[key], np.concatenate([b["events"][i][key] for b in batched], axis=1))
            np.testing.assert_array_equal(event[key], permuted["events"][i][key])
    np.testing.assert_array_equal(whole["events"][0]["conditions"]["probabilities"],
                                  whole["events"][1]["conditions"]["probabilities"])


@pytest.mark.parametrize("fault", ["observed_weather", "bad_final_draw", "wrong_history", "known_future"])
def test_future_conditions_cannot_be_injected_or_mismatched(fault):
    _, _, design, snap = fixture()
    p, ops = observation_posterior(design), operations([snap])
    if fault == "observed_weather":
        ops[snap["event"]["event_id"]]["wet"] = True
    elif fault == "bad_final_draw":
        p["overall_hazard_wet"][-1, -1] = np.nan
    else:
        catalog = build_catalog(design, snap)
        if fault == "wrong_history":
            catalog["training_arrays_key"] = "wrong"
        else:
            catalog["future_weather"] = "wet"
        with pytest.raises(ValueError):
            conditions.ConditionLaw(design).draw(catalog, seed=41, shape=(2, 7), draw_range=(0, 7))
        return
    with pytest.raises(ValueError):
        predict(design, [snap], p, conditional=ops)


def mixture_fixture():
    identity = dict(posterior_id="paired", draw_ids=["draw-a", "draw-b"], event_ids=["future"],
                    condition_ids=["dry", "wet"])
    target = dict(identity, log_probabilities=np.log([[[.9, .2]], [[.3, .8]]]))
    panel = dict(identity, log_probabilities=np.log([[[.8, .1]], [[.2, .7]]]))
    weights = dict(event_ids=["future"], condition_ids=["dry", "wet"],
                   condition_history_sha256="fixture", probabilities=[[.75, .25]])
    return target, panel, weights


def test_condition_mixture_reweights_both_parameter_draws_and_conditions():
    target, panel, weights = mixture_fixture()
    expected = (.75 * (.8 * .9 + .2 * .3) + .25 * (.1 * .2 + .7 * .8)) / (.75 * (.8 + .2) + .25 * (.1 + .7))
    actual = scores.condition_mixture_log_predictive(target, panel, weights)
    np.testing.assert_allclose(np.exp(actual), [expected])
    naive = .75 * (.8 * .9 + .2 * .3) / (.8 + .2) + .25 * (.1 * .2 + .7 * .8) / (.1 + .7)
    assert not np.isclose(expected, naive)
    # The complement must still normalize after conditioning, even for rare panels.
    complement = dict(target, log_probabilities=np.log1p(-np.exp(target["log_probabilities"])))
    other = scores.condition_mixture_log_predictive(complement, panel, weights)
    np.testing.assert_allclose(np.exp(actual) + np.exp(other), [1.])
    panel["log_probabilities"] -= 1e6
    np.testing.assert_allclose(scores.condition_mixture_log_predictive(target, panel, weights), actual, atol=1e-10)


@pytest.mark.parametrize("fault", ["condition_order", "draw_order", "weight_order", "bad_sum", "positive_mass", "impossible_panel", "nan", "axes"])
def test_mixture_rejects_invalid_factors_without_smoothing(fault):
    target, panel, weights = mixture_fixture()
    if fault == "condition_order": panel["condition_ids"] = ["wet", "dry"]
    elif fault == "draw_order": panel["draw_ids"] = ["draw-b", "draw-a"]
    elif fault == "weight_order": weights["condition_ids"] = ["wet", "dry"]
    elif fault == "bad_sum": weights["probabilities"] = [[.75, .75]]
    elif fault == "positive_mass": target["log_probabilities"][0, 0, 0] = 1.
    elif fault == "impossible_panel": panel["log_probabilities"][:] = -np.inf
    elif fault == "nan": panel["log_probabilities"][0, 0, 0] = np.nan
    else: target["log_probabilities"] = target["log_probabilities"][:, 0, :]
    with pytest.raises(ValueError):
        scores.condition_mixture_log_predictive(target, panel, weights)


def test_zero_condition_weight_and_impossible_target_are_retained():
    target, panel, weights = mixture_fixture()
    weights["probabilities"] = [[1., 0.]]
    target["log_probabilities"][:, :, 0] = -np.inf
    assert np.isneginf(scores.condition_mixture_log_predictive(target, panel, weights)).all()
