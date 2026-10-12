"""Unseen-circuit prediction retains the fitted reference and joint uncertainty."""
import copy

import numpy as np
import pytest

from analysis import overall_observation_kernels as kernels
from analysis.overall_forecast_catalog import build_catalog
from analysis.overall_forecast_states import UNSEEN_TRACK_POLICY
from test_overall_forecast_states import fixture, future_path, trace_model
from test_overall_observation_kernels import observation_posterior, scenarios


def predict(design, path, posterior, *, batch=7, policy=kernels.UNSEEN_CIRCUIT_POLICY):
    return list(kernels.iter_conditional_weekends(design, path, posterior, scenarios(path),
        advancing=(4, 2), seed=41, training_arrays_key=build_catalog(design, path[0])["training_arrays_key"],
        draw_batch_size=batch, unseen_circuit_policy=policy, unseen_track_policy=UNSEEN_TRACK_POLICY))


def test_extending_the_reference_in_a_model_trace_preserves_all_fitted_coefficients():
    _, _, design, _ = fixture()
    p = observation_posterior(design)
    for i, name in enumerate(kernels.CONTEXT_NAMES):
        p[f"overall_{name}_circuit_z"] += i + 2.
    coordinates = {name: np.full(p["nu"].shape, i - 1.)
                   for i, name in enumerate(kernels.CONTEXT_NAMES)}
    extended = copy.deepcopy(design)
    count = design.outcomes["n_circuits"]
    extended.outcomes["n_circuits"] += 1
    for key, tail in (("context_circuit", [count] * 4), ("context_wet", [0, 0, 1, 1]),
                      ("context_sc", [0, 1, 0, 1])):
        extended.outcomes[key] = np.concatenate([extended.outcomes[key], tail])
    expanded = {key: value.copy() for key, value in p.items()}
    for name in kernels.CONTEXT_NAMES:
        raw = p[f"overall_{name}_circuit_z"]
        added = np.concatenate([raw, coordinates[name][..., None]], axis=-1)
        expanded[f"overall_{name}_circuit_z"] = added
        # Coordinate change only: do not claim an unchanged independent base prior.
        expanded[f"overall_{name}_base"] += p[f"overall_sd_{name}_circuit"] * (
            added.mean(axis=-1) - raw.mean(axis=-1))
    before, after = trace_model(design, p), trace_model(extended, expanded)
    for name in kernels.CONTEXT_NAMES:
        np.testing.assert_allclose(np.asarray(after[f"overall_{name}"]["value"])[:count * 4],
            np.asarray(before[f"overall_{name}"]["value"]), rtol=2e-6)
    for wet, neutralised in ((False, False), (False, True), (True, False), (True, True)):
        actual = kernels.context_coefficients(p, None, wet=wet, neutralised=neutralised,
                                             new_circuit_z=coordinates)
        index = count * 4 + int(wet) * 2 + int(neutralised)
        for name in ("grid", "pace"):
            np.testing.assert_allclose(actual[name][0, 0],
                np.asarray(after[f"overall_{name}"]["value"])[index], rtol=2e-6)
        np.testing.assert_allclose(np.exp(actual["log_hazard"])[0, 0],
            np.asarray(after["overall_hazard"]["value"])[index], rtol=2e-6)


@pytest.mark.parametrize("count", [1, 3, 9])
def test_exact_gaussian_moments_retain_training_mean_and_new_circuit_covariance(count):
    # Symmetric cubature integrates every quadratic of independent unit Normals
    # exactly. There is no random tolerance or fitted-result selection here.
    dimension = count + 2
    nodes = np.stack([np.eye(dimension), -np.eye(dimension)]) * np.sqrt(dimension)
    shape = nodes.shape[:2]
    scale = .7
    p = {}
    for name in kernels.CONTEXT_NAMES:
        p.update({f"overall_{name}_{part}": np.zeros(shape) for part in ("base", "wet", "sc")})
        p[f"overall_{name}_circuit_z"] = nodes[..., :count]
        p[f"overall_sd_{name}_circuit"] = np.full(shape, scale)
    new = []
    for column in (count, count + 1):
        result = kernels.context_coefficients(p, None, wet=False, neutralised=False,
            new_circuit_z={name: nodes[..., column] for name in kernels.CONTEXT_NAMES})
        new.append(result["hazard_logit"])
        for name in ("grid", "pace"):
            np.testing.assert_allclose(np.log(result[name]), new[-1], atol=1e-15)
    known = kernels.context_coefficients(p, 0, wet=False, neutralised=False)["hazard_logit"]
    for value in new:
        np.testing.assert_allclose(value.mean(), 0., atol=1e-15)
        np.testing.assert_allclose(np.mean(value ** 2), scale ** 2 * (1 + 1/count), atol=1e-15)
        np.testing.assert_allclose(np.mean((value - known) ** 2), 2 * scale ** 2, atol=1e-15)
    np.testing.assert_allclose(np.mean(new[0] * new[1]), scale ** 2/count, atol=1e-15)
    np.testing.assert_allclose(np.mean((new[0] - new[1]) ** 2), 2 * scale ** 2, atol=1e-15)


def test_circuit_innovations_persist_across_visits_batches_fields_and_path_prefixes():
    _, _, design, snap = fixture()
    path, p = future_path(snap), observation_posterior(design)
    for snapshot in path:
        snapshot["event"]["circuit_id"] = "unseen"
    whole = predict(design, path, p)[0]
    permuted = copy.deepcopy(path)
    for snapshot in permuted:
        snapshot["entrants"].reverse()
    batches = predict(design, permuted, p, batch=3)
    prefix = predict(design, path[:1], p)[0]
    distinct = copy.deepcopy(path)
    distinct[1]["event"]["circuit_id"] = "different_unseen"
    other = predict(design, distinct, p)[0]
    assert whole["unseen_circuit_policy"] == kernels.UNSEEN_CIRCUIT_POLICY
    assert not whole["publication_allowed"] and not whole["convergence_verified"]
    for key in whole["events"][0]["context"]:
        left = whole["events"][0]["context"][key]
        np.testing.assert_array_equal(left, whole["events"][1]["context"][key])
        np.testing.assert_array_equal(left, prefix["events"][0]["context"][key])
        np.testing.assert_array_equal(left, other["events"][0]["context"][key])
        assert not np.array_equal(left, other["events"][1]["context"][key])
        for event in range(len(path)):
            np.testing.assert_array_equal(whole["events"][event]["context"][key],
                np.concatenate([b["events"][event]["context"][key] for b in batches], axis=1))
    assert not np.array_equal(whole["events"][0]["context"]["grid"],
                              whole["events"][0]["context"]["pace"])
    for event in range(len(path)):
        for key in ("strength", "running_positions", "joint_order_completion_log_probability"):
            np.testing.assert_allclose(whole["events"][event][key],
                np.concatenate([b["events"][event][key] for b in batches], axis=1), atol=1e-12)


def test_declaring_extension_leaves_known_circuit_predictions_unchanged():
    _, _, design, snap = fixture()
    p = observation_posterior(design)
    default, explicit = predict(design, [snap], p, policy=None), predict(design, [snap], p)
    for key in ("strength", "running_positions", "joint_order_completion_log_probability"):
        np.testing.assert_array_equal(default[0]["events"][0][key], explicit[0]["events"][0][key])


@pytest.mark.parametrize("fault", ["omitted_policy", "wrong_policy", "last_draw_nonfinite"])
def test_unseen_circuit_rejects_missing_policy_and_bad_joint_parameters_before_yield(fault):
    _, _, design, snap = fixture()
    snap["event"]["circuit_id"] = "unseen"
    p, policy = observation_posterior(design), kernels.UNSEEN_CIRCUIT_POLICY
    if fault == "omitted_policy":
        policy = None
    elif fault == "wrong_policy":
        policy = "zero_effect"
    else:
        p["overall_hazard_circuit_z"][-1, -1, 0] = np.nan
    with pytest.raises(ValueError):
        predict(design, [snap], p, batch=1, policy=policy)


@pytest.mark.parametrize("fault", ["missing", "extra", "axes", "nonfinite", "known_circuit"])
def test_population_coordinates_cannot_be_missing_misaligned_or_replace_known_effects(fault):
    _, _, design, _ = fixture()
    p = observation_posterior(design)
    coordinates = {name: np.zeros(p["nu"].shape) for name in kernels.CONTEXT_NAMES}
    circuit = None
    if fault == "missing":
        coordinates.pop("hazard")
    elif fault == "extra":
        coordinates["other"] = np.zeros(p["nu"].shape)
    elif fault == "axes":
        coordinates["pace"] = coordinates["pace"][:, 0]
    elif fault == "nonfinite":
        coordinates["grid"][-1, -1] = np.nan
    else:
        circuit = 0
    with pytest.raises(ValueError):
        kernels.context_coefficients(p, circuit, wet=False, neutralised=False,
                                     new_circuit_z=coordinates)
