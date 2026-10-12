import copy
import json

import numpy as np
import pytest

from analysis import overall_predictive_scores as scores


def component(probabilities):
    return dict(posterior_id="joint-fixture", draw_ids=["chain0-draw0", "chain0-draw1"],
                event_ids=["2001-01"], log_probabilities=np.log(probabilities).reshape(2, 1))


def inventory_and_records():
    inventory = [dict(event_id=event, training_cutoff=f"{int(event[:4]) - 1}-01",
                      field=["a", "b"], information_set="actual_grid",
                      target=scores.CONTRACT["primary_target"])
                 for event in ["2001-01", "2002-01", "2002-02", "2003-01"]]
    arms = [scores.CONTRACT["candidate_arm"], scores.CONTRACT["primary_baseline"],
            *scores.CONTRACT["secondary_baselines"]]
    rows = [dict(**event, arm=arm, status="scored", log_probability=-1. if arm == arms[0] else -2.)
            for event in inventory for arm in arms]
    return inventory, rows


def test_probability_is_integrated_before_taking_its_log():
    value = scores.log_predictive(np.log([[.9], [.1]]))
    np.testing.assert_allclose(value, np.log([.5]))
    assert not np.isclose(value, np.log([.9, .1]).mean())


def test_joint_scoring_preserves_dependence_between_factors():
    a = component([.9, .1])
    joint = scores.joint_log_predictive([a, copy.deepcopy(a)])
    np.testing.assert_allclose(joint, np.log([.41]))
    assert not np.isclose(joint, 2 * scores.log_predictive(a["log_probabilities"]))


def test_conditioning_reweights_joint_draws_before_integrating_the_target():
    target, panel = component([.9, .1]), component([.9, .1])
    result = scores.conditional_log_predictive(target, panel)
    np.testing.assert_allclose(result, np.log([.82]))
    assert not np.isclose(result, scores.log_predictive(target["log_probabilities"]))
    complement = component([.1, .9])
    np.testing.assert_allclose(np.exp(result) + np.exp(scores.conditional_log_predictive(complement, panel)), 1.)


def test_conditioning_weights_have_no_effect_when_independent_of_parameters():
    target, panel = component([.9, .1]), component([1., 1.])
    for shift in (0., -1000., -2e20):
        panel["log_probabilities"][:] = shift
        np.testing.assert_allclose(scores.conditional_log_predictive(target, panel), np.log([.5]), atol=1e-14)


def test_conditional_score_keeps_zero_mass_and_rejects_impossible_conditioning():
    target, panel = component([.9, .1]), component([1., 1.])
    panel["log_probabilities"][0] = -np.inf
    np.testing.assert_allclose(scores.conditional_log_predictive(target, panel), np.log([.1]))
    target["log_probabilities"][:] = -np.inf
    assert np.isneginf(scores.conditional_log_predictive(target, panel)).all()
    panel["log_probabilities"][:] = -np.inf
    with pytest.raises(ValueError, match="impossible conditioning"):
        scores.conditional_log_predictive(target, panel)


@pytest.mark.parametrize("key,value", [("posterior_id", "another-fit"),
    ("draw_ids", ["chain0-draw1", "chain0-draw0"]), ("event_ids", ["2001-02"])])
def test_conditional_score_rejects_independent_or_reordered_factors(key, value):
    target, panel = component([.9, .1]), component([.9, .1])
    panel[key] = value
    with pytest.raises(ValueError, match="share"):
        scores.conditional_log_predictive(target, panel)


def test_conditional_mass_does_not_accept_positive_log_densities():
    target, panel = component([.9, .1]), component([.9, .1])
    for which in (target, panel):
        other = copy.deepcopy(which["log_probabilities"])
        which["log_probabilities"][0] = .1
        with pytest.raises(ValueError, match="discrete log masses"):
            scores.conditional_log_predictive(target, panel)
        which["log_probabilities"] = other


@pytest.mark.parametrize("key,value", [("posterior_id", "other-fit"),
    ("draw_ids", ["chain0-draw1", "chain0-draw0"]), ("event_ids", ["2001-02"])])
def test_independent_or_misaligned_factors_cannot_be_paired(key, value):
    a, b = component([.9, .1]), component([.9, .1])
    b[key] = value
    with pytest.raises(ValueError, match="share"):
        scores.joint_log_predictive([a, b])


def test_impossible_prediction_is_retained_without_smoothing_or_nan():
    value = scores.log_predictive([[-np.inf, -10000], [-np.inf, -10000]])
    assert np.isneginf(value[0]) and value[1] == -10000
    for bad in [[[np.inf]], [[np.nan]], []]:
        with pytest.raises(ValueError):
            scores.log_predictive(bad)


def test_ordinal_scores_have_hand_computable_values():
    result = scores.ordinal_scores([[1., 0., 0.], [0., 0., 1.], [0., 1., 0.]], [0, 0, 0])
    np.testing.assert_array_equal(result["ranked_probability_score"], [0, 2, 1])
    assert result["log_probability"][0] == 0
    assert np.isneginf(result["log_probability"][1:]).all()
    np.testing.assert_allclose(scores.randomized_pit([[.2, .3, .5]], [1], [.4]), [.32])


@pytest.mark.parametrize("probabilities,observed", [([[.4, .4]], [0]),
    ([[1.1, -.1]], [0]), ([[.5, .5]], [2]), ([[.5, .5]], [.5]), ([[np.nan, .5]], [0])])
def test_invalid_distributions_are_not_renormalized_or_dropped(probabilities, observed):
    with pytest.raises(ValueError):
        scores.ordinal_scores(probabilities, observed)


def test_retirement_scoring_uses_the_binary_event_and_keeps_impossible_outcomes():
    result = scores.retirement_scores([0, .25, 1], [0, 1, 0])
    np.testing.assert_allclose(result["brier_score"], [0, .5625, 1])
    np.testing.assert_allclose(result["log_probability"][:2], [0, np.log(.25)])
    assert np.isneginf(result["log_probability"][2])


def test_complete_primary_success_is_not_itself_publication_permission():
    inventory, rows = inventory_and_records()
    result = scores.compare_arms(rows, inventory)
    assert result["complete"] and result["comparison_rule_passed"]
    assert not result["publication_allowed"]
    assert result["comparisons"][scores.CONTRACT["primary_baseline"]]["interval"] == [1., 1.]
    json.dumps(result, allow_nan=False)
    assert scores.compare_arms(rows[::-1], inventory[::-1]) == result


def test_season_block_comparison_matches_the_inherited_registered_rule():
    from f1rank.championship import N_BOOT, MIN_SELECTION_SEASONS, paired_seasons
    assert scores.CONTRACT["paired_comparison"]["bootstrap_replicates"] == N_BOOT
    assert scores.CONTRACT["paired_comparison"]["minimum_seasons"] == MIN_SELECTION_SEASONS
    differences = {2001: np.array([.2]), 2002: np.array([-.2, .6]), 2003: np.array([.4])}
    inherited = paired_seasons(differences)
    actual = scores._paired(np.concatenate(list(differences.values())), np.array([2001, 2002, 2002, 2003]))
    np.testing.assert_allclose(actual["interval"], inherited["ci95"], rtol=0, atol=0)
    assert actual["mean_difference_per_race"] == inherited["mean_diff_per_race"]
    assert actual["comparison_rule_passed"] == inherited["enters"]


def test_secondary_success_cannot_rescue_primary_failure():
    inventory, rows = inventory_and_records()
    for row in rows:
        if row["arm"] == scores.CONTRACT["primary_baseline"]:
            row["log_probability"] = -.5
    result = scores.compare_arms(rows, inventory)
    assert result["complete"] and not result["comparison_rule_passed"]
    assert result["comparisons"]["grid_only"]["comparison_rule_passed"]


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "failed", "field", "origin", "cutoff", "target", "infinite", "positive", "unknown"])
def test_no_failed_or_mismatched_event_can_disappear_from_the_comparison(mutation):
    inventory, rows = inventory_and_records()
    event = rows[0]["event_id"]
    if mutation == "missing":
        rows.pop(0)
    elif mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "unknown":
        rows[0]["event_id"] = "2004-01"
    else:
        key, value = {"failed": ("status", "failed"), "field": ("field", ["a", "c"]),
                      "origin": ("information_set", "prequalifying"), "cutoff": ("training_cutoff", event),
                      "target": ("target", "grid_rank"), "infinite": ("log_probability", -np.inf),
                      "positive": ("log_probability", .1)}[mutation]
        rows[0][key] = value
    result = scores.compare_arms(rows, inventory)
    assert not result["complete"] and not result["comparison_rule_passed"] and result["errors"]
    assert event in result["required_events"]
    assert not result["comparisons"]
    json.dumps(result, allow_nan=False)


def test_insufficient_season_blocks_cannot_pass_a_good_mean_difference():
    inventory, rows = inventory_and_records()
    inventory = inventory[:-1]
    rows = [row for row in rows if row["event_id"] != "2003-01"]
    result = scores.compare_arms(rows, inventory)
    assert result["complete"] and not result["comparison_rule_passed"]


def test_future_training_and_mixed_origins_in_inventory_are_rejected():
    inventory, rows = inventory_and_records()
    inventory[0]["training_cutoff"] = inventory[0]["event_id"]
    with pytest.raises(ValueError, match="training"):
        scores.compare_arms(rows, inventory)
    inventory, rows = inventory_and_records()
    inventory[0]["information_set"] = "prequalifying"
    with pytest.raises(ValueError, match="pool"):
        scores.compare_arms(rows, inventory)


def test_overflowing_comparison_remains_unresolved_and_json_serializable():
    inventory, rows = inventory_and_records()
    for row in rows:
        if row["arm"] != scores.CONTRACT["candidate_arm"]:
            row["log_probability"] = -1e308
    result = scores.compare_arms(rows, inventory)
    assert not result["complete"] and not result["comparison_rule_passed"]
    assert "nonfinite" in result["errors"][0]["reason"]
    json.dumps(result, allow_nan=False)
