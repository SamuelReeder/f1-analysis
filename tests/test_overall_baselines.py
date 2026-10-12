"""Baseline identity and conditional predictive arithmetic on synthetic fields."""
import copy
import itertools

import numpy as np
import numpyro
import pytest

from analysis.overall_baselines import (
    legacy_feature_medians, log_probabilities, ordered_arrays, race_model,
)
from analysis.overall_predictive_scores import log_predictive
from f1rank.championship import race_model as legacy_model


@pytest.fixture
def events():
    return [dict(event_id="synthetic-a", drivers=["a", "b", "c", "d"],
        grid_slot=[1, 2, 4, 4], running_position=[2, 0, 1, 0],
        car=[.3, .3, -.3, -.3], driver=[.1, .4, -.1, -.4]),
        dict(event_id="synthetic-b", drivers=["a", "b"],
        grid_slot=[1, 2], running_position=[1, 2], car=[0., 0.], driver=[.1, -.1])]


@pytest.fixture
def posterior():
    return dict(c_grid=np.array([[.7, 1.2], [.5, .9]]),
                a_car=np.array([[.3, -.1], [.6, .2]]),
                b_driver=np.array([[2., 1.], [1.5, -.3]]))


def test_qualifying_features_use_median_of_paired_sums():
    skill = np.array([[0., 0.], [10., 100.]])[..., None]
    compat = np.array([[100., 10.], [0., 0.]])[..., None]
    post = dict(skill=skill, compat=compat, car=np.concatenate([skill, skill + 2], axis=-1),
                car_track=np.concatenate([compat, compat], axis=-1))
    result = legacy_feature_medians(post)
    np.testing.assert_array_equal(result["driver"], [55.])
    np.testing.assert_array_equal(result["car"], [55., 57.])
    assert result["driver"][0] != np.median(skill) + np.median(compat)


def test_feature_axis_mismatch_cannot_pair_unrelated_draws():
    post = {name: np.zeros((2, 3, 4)) for name in ("skill", "compat", "car", "car_track")}
    post["compat"] = np.zeros((3, 2, 4))
    with pytest.raises(ValueError, match="paired"):
        legacy_feature_medians(post)


@pytest.mark.parametrize("arm,variant", [("legacy_grid_qualifying", "grid_ratings"), ("grid_only", "grid")])
def test_prior_sites_and_ordinary_order_likelihood_match_existing_baseline(events, posterior, arm, variant):
    panel = ordered_arrays(events, arm=arm)
    coefficients = {name: value[0, 0] for name, value in posterior.items()}
    new = numpyro.handlers.trace(numpyro.handlers.seed(
        numpyro.handlers.substitute(race_model, data=coefficients), 0)).get_trace(panel["arrays"], arm=arm)
    old = numpyro.handlers.trace(numpyro.handlers.seed(
        numpyro.handlers.substitute(legacy_model, data=coefficients), 0)).get_trace(panel["arrays"], variant=variant)
    assert set(new) == set(old)
    for name in ("c_grid", "a_car", "b_driver"):
        if name in new:
            np.testing.assert_array_equal(new[name]["fn"].loc, old[name]["fn"].loc)
            np.testing.assert_array_equal(new[name]["fn"].scale, old[name]["fn"].scale)
    np.testing.assert_allclose(new["ll_race"]["value"], old["ll_race"]["value"], rtol=1e-6)
    np.testing.assert_allclose(new["ll"]["fn"].log_factor, old["ll"]["fn"].log_factor, rtol=1e-6)


@pytest.mark.parametrize("arm", ["legacy_grid_qualifying", "grid_only"])
def test_predictive_draws_match_likelihood_and_integrate_probabilities(events, posterior, arm):
    panel = ordered_arrays(events, arm=arm)
    logs = log_probabilities(posterior, panel)
    assert logs.shape == (2, 2, 2)
    for chain, draw in itertools.product(range(2), repeat=2):
        params = {name: value[chain, draw] for name, value in posterior.items()}
        trace = numpyro.handlers.trace(numpyro.handlers.substitute(race_model, data=params)).get_trace(
            panel["arrays"], arm=arm)
        np.testing.assert_allclose(logs[chain, draw], trace["ll_race"]["value"], rtol=1e-6)
    integrated = log_predictive(logs.reshape(-1, 2))
    np.testing.assert_allclose(integrated, np.log(np.exp(logs).mean(axis=(0, 1))))
    assert not np.allclose(integrated, logs.mean(axis=(0, 1)))


def test_field_is_preserved_and_pit_lane_feature_precedes_running_selection(events):
    panel = ordered_arrays(events, arm="legacy_grid_qualifying")
    assert panel["fields"][0] == ["a", "b", "c", "d"]
    assert panel["ordered_fields"][0] == ["c", "a", "b", "d"]
    np.testing.assert_array_equal(panel["arrays"]["mask"][0], [True, True, False, False])
    np.testing.assert_allclose(panel["arrays"]["log_grid"][0, 0], np.log(4))
    events[0]["grid_slot"][2] = 0
    with pytest.raises(ValueError, match="complete declared field"):
        ordered_arrays(events, arm="grid_only")


def test_empty_and_singleton_running_panels_are_retained(events, posterior):
    events[0]["running_position"] = [0, 0, 0, 0]
    events[1]["running_position"] = [0, 1]
    panel = ordered_arrays(events, arm="legacy_grid_qualifying")
    assert panel["event_ids"] == ["synthetic-a", "synthetic-b"]
    np.testing.assert_array_equal(log_probabilities(posterior, panel), np.zeros((2, 2, 2)))


def test_grid_only_has_no_qualifying_feature_dependency(events, posterior):
    before = log_probabilities(posterior, ordered_arrays(events, arm="grid_only"))
    for event in events:
        del event["car"], event["driver"]
    panel = ordered_arrays(events, arm="grid_only")
    np.testing.assert_array_equal(before, log_probabilities({"c_grid": posterior["c_grid"]}, panel))
    with pytest.raises(ValueError, match="finite and numeric"):
        ordered_arrays(events, arm="legacy_grid_qualifying")


def test_input_field_permutation_does_not_change_probability(events, posterior):
    first = log_probabilities(posterior, ordered_arrays(events, arm="legacy_grid_qualifying"))
    permuted = copy.deepcopy(events)
    for event in permuted:
        for key in ("drivers", "grid_slot", "running_position", "car", "driver"):
            event[key].reverse()
    second = log_probabilities(posterior, ordered_arrays(permuted, arm="legacy_grid_qualifying"))
    np.testing.assert_array_equal(first, second)


def test_small_field_orders_normalize(posterior):
    fields = []
    for i, order in enumerate(itertools.permutations(range(3))):
        fields.append(dict(event_id=str(i), drivers=["a", "b", "c"], grid_slot=[1, 2, 3],
            running_position=(np.argsort(order) + 1).tolist(), car=[-.5, -.5, .5], driver=[.3, .1, -.4]))
    logs = log_probabilities(posterior, ordered_arrays(fields, arm="legacy_grid_qualifying"))
    np.testing.assert_allclose(np.exp(logs).sum(axis=-1), 1., rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("change", ["duplicate_event", "duplicate_driver", "rank_gap", "rank_tie", "missing_feature", "nan_feature"])
def test_incomplete_or_ambiguous_inputs_stop_instead_of_dropping_entries(events, change):
    if change == "duplicate_event": events[1]["event_id"] = events[0]["event_id"]
    elif change == "duplicate_driver": events[0]["drivers"][1] = "a"
    elif change == "rank_gap": events[0]["running_position"][0] = 3
    elif change == "rank_tie": events[0]["running_position"][0] = 1
    elif change == "missing_feature": events[0]["driver"].pop()
    elif change == "nan_feature": events[0]["driver"][1] = np.nan
    with pytest.raises(ValueError):
        ordered_arrays(events, arm="legacy_grid_qualifying")


def test_race_only_arm_cannot_silently_become_an_existing_baseline(events):
    with pytest.raises(ValueError, match="separate model"):
        ordered_arrays(events, arm="current_race_only")


def test_bad_final_draw_is_not_skipped(events, posterior):
    panel = ordered_arrays(events, arm="legacy_grid_qualifying")
    posterior["a_car"][-1, -1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        log_probabilities(posterior, panel)


def test_coefficient_chain_axes_cannot_be_flattened_or_transposed(events, posterior):
    panel = ordered_arrays(events, arm="legacy_grid_qualifying")
    posterior["a_car"] = posterior["a_car"].reshape(4, 1)
    with pytest.raises(ValueError, match="paired chain/draw"):
        log_probabilities(posterior, panel)
