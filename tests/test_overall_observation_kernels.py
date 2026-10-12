"""Conditional forecast kernels: model equivalence and known probabilities."""
import copy
import itertools
import math

import numpy as np
import pytest

from analysis import overall_observation_kernels as kernels
from analysis.overall_forecast_catalog import build_catalog
from analysis.overall_forecast_states import forecast_states
from f1rank.overall.design import build
from test_overall_forecast_states import (expanded_parameters, fixture, future_path,
                                         posterior, trace_model)


def observation_posterior(design, shape=(2, 7)):
    p = posterior(design, shape)
    for name, value in dict(sigma0=.4, sigma_tau=.3, nu=5., sd_car_segment=.2).items():
        p[name] = np.full(shape, value)
    p["nu"] += np.arange(math.prod(shape)).reshape(shape) / 10
    for i, name in enumerate(kernels.CONTEXT_NAMES):
        for part, value in dict(base=-3. if name == "hazard" else .2, wet=.1, sc=-.2).items():
            p[f"overall_{name}_{part}"] = np.full(shape, value)
        p[f"overall_sd_{name}_circuit"] = np.full(shape, .3)
        z = np.arange(design.outcomes["n_circuits"]) * (i + 1.)
        p[f"overall_{name}_circuit_z"] = np.broadcast_to(z, shape + z.shape).copy()
    return p


def scenarios(path):
    return {s["event"]["event_id"]:dict(wet=False, neutralised=True,
        grid_policy="actual_grid" if s["information_set"] == "actual_grid" else "qualifying_order_no_penalties",
        exposure_laps={r["driver_id"]:s["event"]["scheduled_laps"] for r in s["entrants"]}) for s in path}


def predict(design, path, p, batch_size=3, conditional=None):
    return list(kernels.iter_conditional_weekends(design, path, p, conditional or scenarios(path),
        advancing=(4, 2), seed=41, training_arrays_key=build_catalog(design, path[0])["training_arrays_key"],
        draw_batch_size=batch_size))


def test_qualifying_observation_location_scale_and_df_match_the_joint_model():
    source, clean, design, snap = fixture()
    full, p = build(snap["event"]["event_id"], tables=source, clean=clean), observation_posterior(design)
    key = build_catalog(design, snap)["training_arrays_key"]
    event = forecast_states(design, [snap], p, seed=41, training_arrays_key=key)["events"][0]
    shape = p["nu"].shape
    noise = kernels.ObservationNoise(41, shape, (0, shape[1]))
    generated = kernels.qualifying_segments(event, p, noise, advancing=(), full_df=p["nu"])
    matched = expanded_parameters(design, full, p, snap, 41)
    q = full.qualifying.arrays()
    matched["mu"] = np.zeros(shape + (q["n_sessions"],))
    matched["sigma_u"] = np.zeros(shape + (q["n_sessions"],))
    matched["car_segment_z"] = np.zeros(shape + (q["n_car_segments"],))
    obs = full.qualifying.obs
    future = obs[(obs.event_id == snap["event"]["event_id"]) & (obs.segment == "Q1")].sort_values("driver_id")
    session = int(future.session_idx.iloc[0])
    identity = [event["event_id"], 0]
    matched["sigma_u"][..., session] = noise.draw("qualifying_scale", identity)
    for row in future.itertuples():
        matched["car_segment_z"][..., row.car_segment_idx] = noise.draw("qualifying_car_segment", [*identity, row.team])
    trace = trace_model(full, matched)
    # The existing qualifying site wraps its Student-t in the training mask.
    likelihood = trace["y"]["fn"].base_dist
    segment = generated["segments"][0]
    np.testing.assert_allclose(segment["location"][0, 0], np.asarray(likelihood.loc)[future.index], rtol=3e-6, atol=3e-6)
    np.testing.assert_allclose(segment["scale"][0, 0], np.asarray(likelihood.scale)[future.index], rtol=3e-6)
    np.testing.assert_allclose(p["nu"][0, 0], np.asarray(likelihood.df), rtol=3e-6)
    # Also compare the resulting transformed observation densities, not just shape.
    from scipy.stats import t
    import jax.numpy as jnp
    y = np.asarray(trace["y"]["value"]).copy()
    y[future.index] = segment["pace"][0, 0]
    reference = np.asarray(likelihood.log_prob(jnp.asarray(y)))[future.index]
    actual = t.logpdf(segment["pace"][0, 0], p["nu"][0, 0],
                     loc=segment["location"][0, 0], scale=segment["scale"][0, 0])
    np.testing.assert_allclose(actual, reference, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("wet,neutralised", list(itertools.product([False, True], repeat=2)))
def test_context_coefficients_match_joint_model_across_circuits_and_conditions(wet, neutralised):
    source, clean, _, snap = fixture()
    source["events"].loc[source["events"].event_id == "2022-01", "circuit_id"] = "another_track"
    design = build("2022-02", tables=source, clean=clean)
    p, trace = observation_posterior(design), None
    trace = trace_model(design, p)
    for circuit in range(design.outcomes["n_circuits"]):
        actual = kernels.context_coefficients(p, circuit, wet=wet, neutralised=neutralised)
        context = circuit * 4 + int(wet) * 2 + int(neutralised)
        for name in ("grid", "pace"):
            np.testing.assert_allclose(actual[name][0, 0], np.asarray(trace["overall_" + name]["value"])[context], rtol=2e-6)
        np.testing.assert_allclose(np.exp(actual["log_hazard"])[0, 0], np.asarray(trace["overall_hazard"]["value"])[context], rtol=2e-6)


def test_order_and_completion_factors_match_the_joint_likelihood():
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    from f1rank.overall.model import model
    _, _, design, _ = fixture()
    p = observation_posterior(design)
    o = copy.deepcopy(design.outcomes)
    o["completed"][:3], o["failed"][:3] = [0, 7, 12], True
    arrays = lambda block: {k:jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in block.items()}
    conditioned = handlers.condition(model, data={k:jnp.asarray(v[0, 0]) for k,v in p.items()})
    trace = handlers.trace(handlers.seed(conditioned, jax.random.PRNGKey(19))).get_trace(
        arrays(design.qualifying.arrays()), arrays(design.laps), arrays(o))
    total = sum(np.asarray(trace[name]["value"]) for name in ("overall_race_skill", "overall_team", "overall_day"))
    total += np.asarray(trace["overall_car"]["value"])[o["entry_car"]]
    strength = (np.asarray(trace["overall_pace"]["value"])[o["context"], None] * total[o["entry"]]
                - np.asarray(trace["overall_grid"]["value"])[o["context"], None] * o["log_grid"])
    positions = np.where(o["mask"], np.arange(1, o["mask"].shape[1] + 1), 0)
    np.testing.assert_allclose(kernels.order_log_probability(strength, positions).sum(),
        np.asarray(trace["overall_finish_order"]["fn"].log_factor), atol=5e-6, rtol=5e-6)
    log_completion = 0.
    for i, context in enumerate(o["risk_context"]):
        coeff = kernels.context_coefficients(p, int(context // 4), wet=bool(context % 4 // 2), neutralised=bool(context % 2))
        shape = p["nu"].shape + (1,)
        log_completion += kernels.completion_log_probability(coeff,
            np.full(shape, o["completed"][i]), np.full(shape, o["failed"][i]))[0, 0, 0]
    np.testing.assert_allclose(log_completion, np.asarray(trace["overall_completion"]["fn"].log_factor), atol=1e-5, rtol=1e-5)


def test_pl_probabilities_normalise_for_every_running_panel_and_keep_impossible_inputs_out():
    strength = np.log([1., 2., 4.])
    for size in range(4):
        for panel in itertools.combinations(range(3), size):
            probabilities = []
            for order in itertools.permutations(panel):
                positions = np.zeros(3, int)
                positions[list(order)] = np.arange(1, size + 1)
                probabilities.append(np.exp(kernels.order_log_probability(strength, positions)))
            np.testing.assert_allclose(sum(probabilities), 1., atol=1e-14)
    np.testing.assert_allclose(np.exp(kernels.order_log_probability(strength, [1, 2, 3])), 1/7 * 2/6)
    for ranks in ([1, 1, 0], [0, 2, 3], [-1, 1, 2], [1.1, 2, 3]):
        with pytest.raises(ValueError):
            kernels.order_log_probability(strength, ranks)


def test_joint_draw_indices_and_conditional_factors_stay_paired_in_the_pipeline():
    _, _, design, snap = fixture()
    p = observation_posterior(design)
    index = np.arange(p["nu"].size).reshape(p["nu"].shape)
    p["overall_pace_base"] = index / 5.
    p["overall_hazard_base"] = -index / 3.
    p["overall_qualifying_link"] = -index / 2.
    key = build_catalog(design, snap)["training_arrays_key"]
    latent = forecast_states(design, [snap], p, seed=41, training_arrays_key=key)["events"][0]["states"]
    event = predict(design, [snap], p, batch_size=7)[0]["events"][0]
    pace = np.exp(index / 5. - .2)  # explicitly neutralised, dry scenario
    grid = np.exp(p["overall_grid_base"] - .2)
    expected = pace[..., None] * latent["race_total"] - grid[..., None] * np.log(event["grid_feature_slot"])
    np.testing.assert_allclose(event["strength"], expected, atol=1e-12)
    hazard_logit = -index / 3. - .2
    completion = (event["completion"]["completed"] * -np.logaddexp(0., hazard_logit)[..., None]
        + np.where(event["completion"]["failed"], -np.logaddexp(0., -hazard_logit)[..., None], 0.)).sum(axis=-1)
    order = kernels.order_log_probability(expected, event["running_positions"])
    np.testing.assert_allclose(event["joint_order_completion_log_probability"], completion + order)


def test_geometric_censoring_probabilities_normalise_and_keep_extreme_logits_finite():
    _, _, design, _ = fixture()
    p = observation_posterior(design)
    coeff = kernels.context_coefficients(p, 0, wet=False, neutralised=False)
    shape = p["nu"].shape + (1,)
    exposure = 9
    masses = [np.exp(kernels.completion_log_probability(coeff, np.full(shape, k), np.ones(shape, bool)))
              for k in range(exposure)]
    masses.append(np.exp(kernels.completion_log_probability(coeff, np.full(shape, exposure), np.zeros(shape, bool))))
    np.testing.assert_allclose(sum(masses), 1., atol=1e-14)
    event = dict(event_id="synthetic", drivers=["a", "b"])
    for eta, failed in ((-1000., False), (1000., True)):
        p["overall_hazard_base"][:] = eta
        coeff = kernels.context_coefficients(p, 0, wet=False, neutralised=False)
        completion = kernels.draw_completion(event, coeff, np.array([1, 60]),
            kernels.ObservationNoise(41, p["nu"].shape, (0, p["nu"].shape[1])))
        assert (completion["failed"] == failed).all()
        assert np.isfinite(completion["log_probability"]).all()
        if failed:
            assert (completion["completed"] == 0).all()
        else:
            np.testing.assert_array_equal(completion["completed"], np.broadcast_to([1, 60], completion["completed"].shape))


def test_knockout_keeps_earlier_eliminations_and_teammates_share_segment_effects():
    _, _, design, snap = fixture()
    p = observation_posterior(design)
    # Negligible observation noise compared with the deliberately ordered field.
    p["sigma0"][:] = 1e-12
    p["sd_car_segment"][:] = 0.
    key = build_catalog(design, snap)["training_arrays_key"]
    event = forecast_states(design, [snap], p, seed=41, training_arrays_key=key)["events"][0]
    event["states"]["qualifying_total"][:] = np.arange(6, 0, -1)
    noise = kernels.ObservationNoise(41, p["nu"].shape, (0, p["nu"].shape[1]))
    q = kernels.qualifying_segments(event, p, noise, advancing=(4, 2), full_df=p["nu"])
    np.testing.assert_array_equal(q["positions"], np.broadcast_to(np.arange(1, 7), q["positions"].shape))
    for segment, count in zip(q["segments"], (6, 4, 2)):
        assert (segment["active"].sum(axis=-1) == count).all()
    p["sd_car_segment"][:] = .4
    q = kernels.qualifying_segments(event, p, noise, advancing=(4, 2), full_df=p["nu"])
    for segment in q["segments"]:
        delta = segment["location"] - event["states"]["qualifying_total"]
        for car in range(len(event["cars"])):
            drivers = np.where(event["entry_car"] == car)[0]
            for i in drivers[1:]:
                np.testing.assert_allclose(delta[..., i], delta[..., drivers[0]], atol=1e-14)
        np.testing.assert_allclose((segment["car_effect"] * segment["car_active"]).sum(axis=-1), 0., atol=1e-14)


def test_generated_orders_and_retirements_match_known_distribution():
    shape = (2, 12000)
    event = dict(event_id="synthetic", drivers=["a", "b", "c"])
    noise = kernels.ObservationNoise(79, shape, (0, shape[1]))
    strength = np.broadcast_to(np.log([1., 2., 4.]), shape + (3,))
    order = kernels.draw_running_order(event, strength, np.ones(strength.shape, bool), noise)
    for i, prob in enumerate((1/7, 2/7, 4/7)):
        observed = (order[..., i] == 1).mean()
        assert abs(observed - prob) < 6 * np.sqrt(prob * (1-prob) / math.prod(shape))
    h = .2
    coefficients = dict(log_hazard=np.full(shape, np.log(h)), log_survival=np.full(shape, np.log1p(-h)))
    completion = kernels.draw_completion(event, coefficients, np.array([1, 3, 8]), noise)
    for i, distance in enumerate([1, 3, 8]):
        prob = 1 - (1 - h)**distance
        assert abs(completion["failed"][..., i].mean() - prob) < 6 * np.sqrt(prob * (1-prob) / math.prod(shape))


def test_identity_noise_is_permutation_invariant_and_stronger_running_driver_cannot_lose_places():
    shape = (2, 7)
    event = dict(event_id="synthetic", drivers=["a", "b", "c"])
    noise = kernels.ObservationNoise(41, shape, (0, shape[1]))
    strength = np.broadcast_to([.3, -.4, .1], shape + (3,)).copy()
    running = np.ones(strength.shape, bool)
    running[0, :, 1] = False
    base = kernels.draw_running_order(event, strength, running, noise)
    strength[..., 0] += 1.
    better = kernels.draw_running_order(event, strength, running, noise)
    assert (better[..., 0] <= base[..., 0]).all()
    ix = [2, 0, 1]
    permuted = dict(event, drivers=[event["drivers"][i] for i in ix])
    actual = kernels.draw_running_order(permuted, strength[..., ix], running[..., ix], noise)
    np.testing.assert_array_equal(actual, better[..., ix])
    p = dict(log_survival=np.full(shape, np.log(.9)), log_hazard=np.full(shape, np.log(.1)))
    before = kernels.draw_completion(event, p, np.array([20, 10, 30]), noise)
    after = kernels.draw_completion(permuted, p, np.array([30, 20, 10]), noise)
    np.testing.assert_array_equal(after["completed"], before["completed"][..., ix])
    np.testing.assert_array_equal(after["failed"], before["failed"][..., ix])


@pytest.mark.parametrize("distribution", ["normal", "student", "uniform", "gumbel"])
def test_observation_streams_are_exactly_batch_invariant(distribution):
    shape = (2, 7)
    kwargs = dict(distribution=distribution, df=np.arange(14).reshape(shape) + 1.)
    full = kernels.ObservationNoise(41, shape, (0, 7)).draw("test", ["event", "driver"], **kwargs)
    parts = [kernels.ObservationNoise(41, shape, (start, min(start + 3, 7))).draw("test", ["event", "driver"], **kwargs)
             for start in range(0, 7, 3)]
    np.testing.assert_array_equal(full, np.concatenate(parts, axis=1))


def test_complete_pipeline_preserves_every_draw_field_permutation_and_prefix():
    _, _, design, snap = fixture()
    path, p = future_path(snap), observation_posterior(design)
    whole = predict(design, path, p, batch_size=7)
    shuffled = copy.deepcopy(path)
    for snap in shuffled:
        snap["entrants"].reverse()
    other = predict(design, shuffled, p, batch_size=3)
    prefix = predict(design, path[:1], p, batch_size=7)
    assert [b["draw_range"] for b in other] == [[0, 3], [3, 6], [6, 7]]
    assert not whole[0]["publication_allowed"] and not whole[0]["conditions_forecast"]
    for j in range(len(path)):
        for name in ("running_positions", "strength", "order_log_probability", "joint_order_completion_log_probability"):
            np.testing.assert_allclose(whole[0]["events"][j][name], np.concatenate([b["events"][j][name] for b in other], axis=1), atol=1e-12)
            if j == 0:
                np.testing.assert_array_equal(whole[0]["events"][0][name], prefix[0]["events"][0][name])
        for block, name in (("qualifying", "positions"), ("completion", "completed"), ("completion", "failed"), ("context", "log_hazard")):
            np.testing.assert_array_equal(whole[0]["events"][j][block][name], np.concatenate([b["events"][j][block][name] for b in other], axis=1))


def test_actual_grid_is_an_explicit_covariate_without_replacing_the_joint_states():
    _, _, design, snap = fixture()
    snap["information_set"] = "actual_grid"
    for i, row in enumerate(snap["entrants"]):
        row["grid_position"] = i  # includes the pit-lane fallback
    p = observation_posterior(design)
    event = predict(design, [snap], p)[0]["events"][0]
    np.testing.assert_array_equal(event["grid_feature_slot"], [6, 1, 2, 3, 4, 5])
    assert event["catalog"]["information_set"] == "actual_grid"
    np.testing.assert_allclose(event["joint_order_completion_log_probability"],
        event["order_log_probability"] + event["completion"]["log_probability"].sum(axis=-1))


@pytest.mark.parametrize("fault", ["missing_parameter", "bad_final_df", "negative_scale", "axes", "unknown_circuit", "unknown_weather", "missing_exposure", "grid_policy"])
def test_invalid_parameters_or_unresolved_conditions_fail_before_any_prediction(fault):
    _, _, design, snap = fixture()
    p = observation_posterior(design)
    condition = scenarios([snap])
    event = snap["event"]["event_id"]
    if fault == "missing_parameter":
        del p["sd_car_segment"]
    elif fault == "bad_final_df":
        p["nu"][-1, -1] = np.nan
    elif fault == "negative_scale":
        p["sigma_tau"][-1, -1] = -1.
    elif fault == "axes":
        p["overall_grid_base"] = p["overall_grid_base"][:, 0]
    elif fault == "unknown_circuit":
        snap["event"]["circuit_id"] = "unseen"
    elif fault == "unknown_weather":
        condition[event]["wet"] = None
    elif fault == "missing_exposure":
        condition[event]["exposure_laps"].pop("d0")
    elif fault == "grid_policy":
        condition[event]["grid_policy"] = "actual_grid"
    with pytest.raises(ValueError):
        predict(design, [snap], p, conditional=condition)
