"""An unseen track has one shared factor, carried through paired car effects."""
import copy

import numpy as np
import pytest

from analysis import overall_forecast_states as states
from analysis.overall_forecast_catalog import build_catalog
from analysis.overall_race_only import build_race_only
from f1rank.overall.design import build
from test_overall_forecast_states import (expanded_parameters, fixture, future_path,
                                         posterior, trace_model)
from test_overall_observation_kernels import observation_posterior, scenarios
from analysis import overall_observation_kernels as kernels


def forecast(design, path, p, *, batch=3, policy=states.UNSEEN_TRACK_POLICY):
    return states.forecast_states(design, path, p, seed=41,
        training_arrays_key=build_catalog(design, path[0])["training_arrays_key"],
        draw_batch_size=batch, unseen_track_policy=policy)


@pytest.mark.parametrize("policy", [None, "fixed_zero", True])
def test_new_joint_model_circuit_requires_explicit_declared_policy(policy):
    _, _, design, snap = fixture()
    snap["event"]["circuit_id"] = "new"
    with pytest.raises(ValueError, match="policy"):
        forecast(design, [snap], posterior(design), policy=policy)


def test_known_circuit_fixed_factor_is_unchanged_when_policy_is_declared():
    _, _, design, snap = fixture()
    design.qualifying.circuit_factor[:] = 1.7
    p = posterior(design)
    a, b = [forecast(design, [snap], p, policy=policy) for policy in (None, states.UNSEEN_TRACK_POLICY)]
    assert a["events"][0]["circuit_factor"] == 1.7
    for key, value in a["events"][0]["states"].items():
        np.testing.assert_array_equal(value, b["events"][0]["states"][key])


def test_shared_factor_covariance_and_paired_car_link_have_exact_moments(monkeypatch):
    _, _, design, snap = fixture("2022-01", "2022-02")
    p = posterior(design, (2, 4))
    p["track_load_z"][:] = np.arange(p["track_load_z"].shape[-1]) + 1
    p["sd_track_load"][:] = .5
    p["overall_car_link"][:] = np.arange(8).reshape(2, 4) / 3
    original = states._Noise.draw
    def nodes(self, site, identity, **kwargs):
        if site != "qualifying_circuit_factor":
            return original(self, site, identity, **kwargs)
        value = np.broadcast_to([[-1.], [1.]], self.shape)
        return value if self.draw_slice is None else value[:, self.draw_slice].copy()
    monkeypatch.setattr(states._Noise, "draw", nodes)
    baseline = forecast(design, [snap], p)["events"][0]
    snap["event"]["circuit_id"] = "new"
    event = forecast(design, [snap], p)["events"][0]
    rows = design.qualifying.cars.groupby("team_season_idx").first()
    ix = [rows[(rows.team == team) & (rows.season == 2022)].index.item() for team in event["cars"]]
    load = p["sd_track_load"][..., None] * p["track_load_z"][..., ix]
    load -= load.mean(axis=-1, keepdims=True)
    expected = event["circuit_factor"][..., None] * load
    np.testing.assert_allclose(event["states"]["car_track"], expected, atol=1e-15)
    np.testing.assert_allclose(expected.mean(axis=(0, 1)), 0., atol=1e-15)
    covariance = np.einsum("cdi,cdj->ij", expected, expected) / 8
    np.testing.assert_allclose(covariance, np.outer(load[0, 0], load[0, 0]), atol=1e-15)
    pair = expected[..., 0] - expected[..., 1]
    np.testing.assert_allclose(np.mean(pair ** 2), (load[0, 0, 0] - load[0, 0, 1]) ** 2)
    for name, multiplier in (("qualifying_total", 1.), ("race_total", p["overall_car_link"][..., None])):
        np.testing.assert_allclose(event["states"][name] - baseline["states"][name],
                                  expected[..., event["entry_car"]] * multiplier, atol=1e-14)


def test_new_track_forecast_matches_joint_trace_conditioned_on_its_same_factor():
    source, clean, design, snap = fixture()
    target = snap["event"]["event_id"]
    source["events"].loc[source["events"].event_id == target, "circuit_id"] = "new"
    snap["event"]["circuit_id"] = "new"
    p = posterior(design)
    event = forecast(design, [snap], p)["events"][0]
    full = build(target, tables=source, clean=clean)
    circuits = full.qualifying.events.groupby("circuit_id").circuit_idx.first()
    full.qualifying.circuit_factor[circuits["new"]] = event["circuit_factor"][0, 0]
    trace = trace_model(full, expanded_parameters(design, full, p, snap, 41))
    index = full.qualifying.events.loc[lambda x: x.event_id == target, "event_idx"].item()
    cars = full.qualifying.cars.loc[lambda x: x.event_idx == index].sort_values("team").car_idx.to_numpy()
    for key, reference in (("car_track", np.asarray(trace["car_track"]["value"])[cars]),
        ("race_car", np.asarray(trace["overall_car"]["value"])[cars]),
        ("qualifying_car", sum(np.asarray(trace[site]["value"])[cars] for site in ("car", "car_event", "car_track")))):
        np.testing.assert_allclose(event["states"][key][0, 0], reference, rtol=3e-6, atol=3e-6)


def test_same_circuit_persists_across_visits_and_collector_keeps_every_draw():
    _, _, design, snap = fixture()
    p, path = posterior(design, (2, 7)), future_path(snap)
    for s in path:
        s["event"]["circuit_id"] = "new"
    whole, batched = forecast(design, path, p, batch=7), forecast(design, path, p, batch=3)
    permutation = copy.deepcopy(path)
    for s in permutation:
        s["entrants"].reverse()
    permuted, prefix = forecast(design, permutation, p), forecast(design, path[:1], p)
    different = copy.deepcopy(path)
    different[1]["event"]["circuit_id"] = "another"
    other = forecast(design, different, p)
    np.testing.assert_array_equal(whole["events"][0]["circuit_factor"], whole["events"][1]["circuit_factor"])
    assert not np.array_equal(whole["events"][0]["circuit_factor"], other["events"][1]["circuit_factor"])
    for i, event in enumerate(whole["events"]):
        assert event["circuit_factor"].shape == (2, 7)
        assert event["circuit_factor_policy"] == states.UNSEEN_TRACK_POLICY
        for comparison in (batched, permuted, prefix) if i == 0 else (batched, permuted):
            np.testing.assert_array_equal(event["circuit_factor"], comparison["events"][i]["circuit_factor"])
            for name, value in event["states"].items():
                np.testing.assert_allclose(value, comparison["events"][i]["states"][name], atol=1e-14)


def test_race_only_preserves_removed_axis_without_a_qualifying_policy():
    source, clean, _, snap = fixture()
    design = build_race_only("2022-02", tables=source, clean=clean)
    snap["event"]["circuit_id"] = "new"
    event = forecast(design, [snap], posterior(design), policy=None)["events"][0]
    assert event["circuit_factor_policy"] == "race_only_axis_removed"
    assert event["circuit_factor"] == 0.
    assert not np.any(event["states"]["car_track"])


def test_weekend_records_factor_and_requires_track_and_outcome_policies():
    _, _, design, snap = fixture()
    snap["event"]["circuit_id"] = "new"
    p = observation_posterior(design)
    kwargs = dict(advancing=(4, 2), seed=41,
        training_arrays_key=build_catalog(design, snap)["training_arrays_key"],
        unseen_circuit_policy=kernels.UNSEEN_CIRCUIT_POLICY)
    with pytest.raises(ValueError, match="track factor"):
        next(kernels.iter_conditional_weekends(design, [snap], p, scenarios([snap]), **kwargs))
    result = list(kernels.iter_conditional_weekends(design, [snap], p, scenarios([snap]),
        unseen_track_policy=states.UNSEEN_TRACK_POLICY, **kwargs))[0]
    reference = forecast(design, [snap], p, batch=7)["events"][0]
    np.testing.assert_array_equal(result["events"][0]["circuit_factor"], reference["circuit_factor"])
    assert result["unseen_track_policy"] == states.UNSEEN_TRACK_POLICY
    assert result["track_policy_sha256"]
    assert not result["publication_allowed"]
