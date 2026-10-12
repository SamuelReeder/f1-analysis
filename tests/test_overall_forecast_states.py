"""Future latent states checked against the existing joint probability model."""
import copy
import datetime as dt

import numpy as np
import pytest

from analysis.overall_forecast_catalog import build_catalog
from analysis.overall_forecast_states import (RACE_CAR_SD, _Noise, forecast_states,
                                              reconstruct_history)
from f1rank.overall.design import build
from test_overall import tables


def fixture(cutoff="2022-02", target="2026-01"):
    source, clean = tables()
    dates = {"2021-22":"2021-12-01", "2022-01":"2022-01-10",
             "2022-02":"2022-02-10", "2026-01":"2026-03-10"}
    source["events"]["date"] = source["events"].event_id.map(dates)
    design = build(cutoff, tables=source, clean=clean)
    origin = dates[cutoff] + "T20:00:00Z"
    race_date = dt.date.fromisoformat(dates[target])
    field = source["race"].loc[source["race"].event_id == target].sort_values("driver_id")
    snap = dict(schema_version=1, information_set="prequalifying", origin_at=origin,
        source=dict(kind="archived_asof", captured_at=origin, reference="synthetic announced field"),
        event=dict(event_id=target, circuit_id="track", scheduled_laps=60,
                   qualifying_start=str(race_date - dt.timedelta(days=1)) + "T14:00:00Z",
                   race_start=str(race_date) + "T14:00:00Z"),
        calendar_after_cutoff=[target], entrants=[dict(driver_id=r.driver_id, team=r.team,
            prior_entries_after_cutoff=0) for r in field.itertuples()])
    return source, clean, design, snap


def posterior(design, shape=(2, 3)):
    q = design.qualifying.arrays()
    rng = np.random.default_rng(37)
    p = {key:np.full(shape, .15) for key in (
        "sd_level", "sd_skill_race", "sd_skill_season", "overall_sd_level", "overall_sd_drift_race",
        "overall_sd_drift_season", "sd_car_race", "sd_car_season", "sd_car_reset", "sd_car_event",
        "sd_track_load", "sd_compat", "overall_sd_team", "overall_sd_car_day", "sd_driver_form", "overall_sd_day")}
    for key, value in dict(exp_gain=.4, exp_scale=20., age_slope=-.02, overall_qualifying_link=.8,
        overall_team_link=.6, overall_car_link=1.1, overall_shared_weekend=.7, rho=.8, rho_reset=.4).items():
        p[key] = np.full(shape, value)
    for key, n in dict(level_z=q["n_drivers"], skill_z=q["n_entries"],
        overall_level_z=q["n_drivers"], overall_walk_z=q["n_entries"],
        car_step_z=q["n_cars"], car_jump_z=q["n_cars"], track_load_z=q["n_team_seasons"],
        compat_z=q["n_era_stints"], overall_team_z=q["n_era_stints"],
        overall_car_contrast=design.outcomes["car_basis"].shape[1]).items():
        p[key] = rng.normal(size=shape + (n,)) * (RACE_CAR_SD if key == "overall_car_contrast" else 1.)
    return p


def forecast(design, snapshots, p, seed=41):
    key = build_catalog(design, snapshots[0])["training_arrays_key"]
    return forecast_states(design, snapshots, p, seed=seed, training_arrays_key=key)


def trace_model(design, p):
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    from f1rank.overall.model import model
    arrays = lambda block: {k:jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in block.items()}
    data = {k:jnp.asarray(v[0, 0]) for k, v in p.items()}
    run = handlers.seed(handlers.condition(model, data=data), jax.random.PRNGKey(19))
    return handlers.trace(run).get_trace(arrays(design.qualifying.arrays()), arrays(design.laps),
                                         arrays(design.outcomes), generate=True)


def test_reconstructed_history_matches_joint_model_deterministics():
    _, _, design, _ = fixture()
    p = posterior(design)
    history, trace = reconstruct_history(design, p), trace_model(design, p)
    for output, site, groups in (
        ("qualifying_skill", "skill", design.qualifying.entries.event_idx),
        ("race_skill", "overall_race_skill", design.qualifying.entries.event_idx),
        ("qualifying_team", "compat", design.qualifying.entries.event_idx),
        ("race_team", "overall_team", design.qualifying.entries.event_idx),
        ("car_persistent", "car", design.qualifying.cars.event_idx)):
        actual = history[output][0, 0].copy()
        for event in np.unique(groups):
            actual[groups == event] -= actual[groups == event].mean()
        np.testing.assert_allclose(actual, np.asarray(trace[site]["value"]), atol=3e-6, rtol=3e-6)
    assert np.all(np.asarray(trace["overall_car_contrast"]["fn"].variance) == RACE_CAR_SD**2)


def expanded_parameters(design, full, p, snap, seed):
    """Attach the same innovations to a full model trace, without fitting it."""
    noise = _Noise(seed, p["sd_level"].shape)
    result = {k:v.copy() for k, v in p.items() if v.ndim == 2}
    result.update(level_z=p["level_z"].copy(), overall_level_z=p["overall_level_z"].copy())
    event = snap["event"]["event_id"]
    # These fixtures retain driver identities. Newcomer uncertainty has separate
    # checks below; every future row here is a new state in the full joint model.
    for key, site in (("skill_z", "driver_step"), ("overall_walk_z", "race_step")):
        old = {(r.driver_id, r.event_id):r.entry_idx for r in design.qualifying.entries.itertuples()}
        result[key] = np.stack([p[key][..., old[(r.driver_id, r.event_id)]] if (r.driver_id, r.event_id) in old
            else noise.draw(site, [r.driver_id, r.event_id]) for r in full.qualifying.entries.itertuples()], axis=-1)
    for key, site in (("car_step_z", "car_step"), ("car_jump_z", "car_jump")):
        old = {(r.team, r.event_idx):r.car_idx for r in design.qualifying.cars.itertuples()}
        result[key] = np.stack([p[key][..., old[(r.team, r.event_idx)]] if (r.team, r.event_idx) in old
            else noise.draw("car_first" if key == "car_jump_z" and r.first else site,
                [r.team, event], student=key == "car_step_z")
            for r in full.qualifying.cars.itertuples()], axis=-1)
    for key, site in (("compat_z", "compatibility"), ("overall_team_z", "race_team")):
        old = {(r.driver_id, r.team, r.era):r.era_stint_idx for r in design.qualifying.entries.itertuples()}
        rows = full.qualifying.entries.groupby("era_stint_idx").first()
        result[key] = np.stack([p[key][..., old[(r.driver_id, r.team, r.era)]] if (r.driver_id, r.team, r.era) in old
            else noise.draw(site, (r.driver_id, r.team, r.era)) for r in rows.itertuples()], axis=-1)
    old = {(r.team, r.season):i for i, r in design.qualifying.cars.groupby("team_season_idx").first().iterrows()}
    rows = full.qualifying.cars.groupby("team_season_idx").first()
    result["track_load_z"] = np.stack([p["track_load_z"][..., old[(r.team, r.season)]] if (r.team, r.season) in old
        else noise.draw("track_load", (r.team, r.season)) for r in rows.itertuples()], axis=-1)
    contrasts = reconstruct_history(design, p)["car_season_contrast"]
    counts = design.qualifying.cars.groupby("team_season_idx").first().groupby("season").size()
    raw_car = np.stack([contrasts[..., old[(r.team, r.season)]] + RACE_CAR_SD / np.sqrt(counts[r.season]) * noise.draw("race_car_mean", r.season)
        if (r.team, r.season) in old else RACE_CAR_SD * noise.draw("race_car", (r.team, r.season))
        for r in rows.itertuples()], axis=-1)
    result["overall_car_contrast"] = raw_car @ full.outcomes["car_basis"]
    for key, site in (("driver_form_z", "qualifying_day"), ("overall_day_z", "race_day")):
        result[key] = np.stack([noise.draw(site, [r.driver_id, r.event_id]) for r in full.qualifying.entries.itertuples()], axis=-1)
    for key, site in (("car_total_decentered", "car_qualifying_day"), ("overall_car_day_z", "car_race_day")):
        result[key] = np.stack([noise.draw(site, [r.team, str(full.qualifying.events.iloc[r.event_idx].event_id)])
            for r in full.qualifying.cars.itertuples()], axis=-1)
    return result


@pytest.mark.parametrize("cutoff,target,new_team", [("2022-01", "2022-02", False),
    ("2022-02", "2026-01", False), ("2022-01", "2022-02", True)])
def test_future_states_match_conditioned_joint_model_with_identical_innovations(cutoff, target, new_team):
    source, clean, design, snap = fixture(cutoff, target)
    if new_team:
        for row in snap["entrants"][:2]: row["team"] = "new_team"
        for frame in (source["entries"], source["race"], clean):
            frame.loc[(frame.event_id == target) & frame.driver_id.isin(["d0", "d1"]), "team"] = "new_team"
    full = build(target, tables=source, clean=clean)
    p = posterior(design)
    states = forecast(design, [snap], p)["events"][0]["states"]
    trace = trace_model(full, expanded_parameters(design, full, p, snap, 41))
    entries = full.qualifying.entries.query("event_id == @target").sort_values("driver_id").entry_idx.to_numpy()
    event_index = int(full.qualifying.events.query("event_id == @target").event_idx.iloc[0])
    cars = full.qualifying.cars.query("event_idx == @event_index").sort_values("team").car_idx.to_numpy()
    for name, site, ix in (("qualifying_skill", "skill", entries), ("race_skill", "overall_race_skill", entries),
        ("qualifying_team", "compat", entries), ("race_team", "overall_team", entries),
        ("qualifying_day", "driver_form", entries), ("race_day", "overall_day", entries),
        ("race_car", "overall_car", cars), ("car_persistent", "car", cars), ("car_track", "car_track", cars)):
        np.testing.assert_allclose(states[name][0, 0], np.asarray(trace[site]["value"])[ix], atol=3e-6, rtol=3e-6, err_msg=name)
    qcar = sum(np.asarray(trace[site]["value"])[cars] for site in ("car", "car_event", "car_track"))
    np.testing.assert_allclose(states["qualifying_car"][0, 0], qcar, atol=3e-6, rtol=3e-6)


def future_path(snap):
    first = copy.deepcopy(snap)
    first["event"].update(event_id="2022-03", qualifying_start="2022-03-09T14:00:00Z", race_start="2022-03-10T14:00:00Z")
    first["calendar_after_cutoff"] = ["2022-03"]
    second = copy.deepcopy(first)
    second["event"].update(event_id="2022-04", qualifying_start="2022-04-09T14:00:00Z", race_start="2022-04-10T14:00:00Z")
    second["calendar_after_cutoff"].append("2022-04")
    for row in second["entrants"]:
        row["prior_entries_after_cutoff"] = 1
    return [first, second]


def test_state_path_is_prefix_consistent_and_invariant_to_field_permutation():
    _, _, design, snap = fixture()
    p, path = posterior(design), future_path(snap)
    alone = forecast(design, path[:1], p)
    whole = forecast(design, path, p)
    shuffled = copy.deepcopy(path)
    for item in shuffled: item["entrants"].reverse()
    other = forecast(design, shuffled, p)
    for i, event in enumerate(whole["events"]):
        for key, value in event["states"].items():
            np.testing.assert_array_equal(value, other["events"][i]["states"][key])
            if i == 0: np.testing.assert_array_equal(value, alone["events"][i]["states"][key])
    assert whole["latent_states_propagated"] and not whole["outcome_prediction_available"]
    assert not whole["publication_allowed"] and not whole["convergence_verified"]


def test_adding_unseen_driver_and_team_retains_population_and_shared_uncertainty():
    _, _, design, snap = fixture()
    for name in ("new_a", "new_b"):
        snap["entrants"].append(dict(driver_id=name, team="new_team", dob="2003-01-01",
            debut_season=2026, experience_before_cutoff=0, prior_entries_after_cutoff=0))
    p = posterior(design)
    event = forecast(design, [snap], p)["events"][0]
    a, b = [event["drivers"].index(k) for k in ("new_a", "new_b")]
    assert event["entry_car"][a] == event["entry_car"][b]
    assert np.var(event["states"]["qualifying_skill"][..., a] - event["states"]["qualifying_skill"][..., b]) > 0
    assert np.var(event["states"]["race_car"][..., event["entry_car"][a]]) > 0
    assert event["states"]["race_total"].shape == (2, 3, 8)


def test_joint_draw_axes_are_preserved_without_pairing_independent_samples():
    _, _, design, snap = fixture()
    p = posterior(design)
    baseline = forecast(design, [snap], p)["events"][0]["states"]
    p["level_z"][1, 2, 0] += 1
    changed = forecast(design, [snap], p)["events"][0]["states"]
    delta = changed["qualifying_skill"] - baseline["qualifying_skill"]
    np.testing.assert_array_equal(delta[0], 0.)
    np.testing.assert_array_equal(delta[1, :2], 0.)
    assert np.any(delta[1, 2] != 0)
    np.testing.assert_allclose(changed["race_skill"] - baseline["race_skill"], p["overall_qualifying_link"][..., None] * delta, atol=1e-15)


def test_centring_uses_future_field_instead_of_comparing_old_centred_entries():
    _, _, design, snap = fixture()
    p = posterior(design)
    # Suppress new innovations and trend so the exact known portable levels apply.
    for key in ("sd_skill_race", "sd_skill_season", "exp_gain", "age_slope"):
        p[key][...] = 0
    snap["entrants"] = snap["entrants"][:3]
    result = forecast(design, [snap], p)["events"][0]["states"]["qualifying_skill"]
    expected = p["sd_level"][..., None] * p["level_z"][..., :3]
    expected -= expected.mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(result, expected)


def test_new_team_relative_variance_restores_missing_common_season_mean():
    _, _, design, snap = fixture()
    # A same-season new team tests the lost common mean in old zero-sum contrasts.
    snap = future_path(snap)[0]
    for row in snap["entrants"][:2]: row["team"] = "new_team"
    shape = (2, 12000)
    p = posterior(design, shape)
    for key in ("overall_car_link", "overall_sd_car_day"):
        p[key][...] = 0
    result = forecast(design, [snap], p)["events"][0]
    cars = result["states"]["race_car"]
    delta = cars[..., result["cars"].index("new_team")] - cars[..., result["cars"].index("t2")]
    # Independent iid Normal team effects have difference variance 2*s^2.
    # Without restoring the old mean the variance is smaller by s^2 / n_old.
    expected = 2 * RACE_CAR_SD**2
    assert abs(delta.var() - expected) < expected * .05


def test_absence_accumulates_driver_uncertainty_across_the_calendar_gap():
    _, _, design, snap = fixture()
    p, path = posterior(design), future_path(snap)
    path[0]["entrants"] = [r for r in path[0]["entrants"] if r["driver_id"] != "d0"]
    path[1]["entrants"][0]["prior_entries_after_cutoff"] = 0
    for key in ("level_z", "skill_z", "exp_gain", "age_slope"):
        p[key][...] = 0
    result = forecast(design, path, p)["events"][-1]
    noise = _Noise(41, p["sd_level"].shape)
    # The returning driver advances over the gap once; its continuously entered
    # comparator advances at each intervening event. Centring cancels in a pair.
    a = np.sqrt(2) * p["sd_skill_race"] * noise.draw("driver_step", ["d0", "2022-04"])
    b = p["sd_skill_race"] * (noise.draw("driver_step", ["d1", "2022-03"])
        + noise.draw("driver_step", ["d1", "2022-04"]))
    np.testing.assert_allclose(result["states"]["qualifying_skill"][..., 0]
        - result["states"]["qualifying_skill"][..., 1], a - b, atol=1e-15)


def test_team_effect_is_retained_across_events_in_the_same_era():
    _, _, design, snap = fixture()
    p, path = posterior(design), future_path(snap)
    path[0]["entrants"][0]["team"] = "new_team"
    path[1]["entrants"][0]["team"] = "new_team"
    result = forecast(design, path, p)["events"]
    # The stint is shared across events, while weekend deviations are fresh.
    np.testing.assert_array_equal(result[0]["states"]["qualifying_team"], result[1]["states"]["qualifying_team"])
    np.testing.assert_array_equal(result[0]["states"]["race_team"], result[1]["states"]["race_team"])
    assert np.any(result[0]["states"]["qualifying_day"] != result[1]["states"]["qualifying_day"])


@pytest.mark.parametrize("change", ["missing_state", "flat_draws", "mismatched_draws", "nan", "negative_scale",
    "bad_seed", "wrong_training", "missing_field", "different_origin", "wrong_participation", "metadata_change"])
def test_invalid_sources_axes_and_incomplete_paths_fail(change):
    _, _, design, snap = fixture()
    p, path, seed = posterior(design), future_path(snap), 41
    key = build_catalog(design, path[0])["training_arrays_key"]
    if change == "missing_state": del p["skill_z"]
    elif change == "flat_draws": p["sd_level"] = p["sd_level"].reshape(-1)
    elif change == "mismatched_draws": p["overall_level_z"] = p["overall_level_z"][:1]
    elif change == "nan": p["rho"][0, 0] = np.nan
    elif change == "negative_scale": p["sd_level"][0, 0] = -1
    elif change == "bad_seed": seed = True
    elif change == "wrong_training": key = "wrong"
    elif change == "missing_field": path = path[1:]
    elif change == "different_origin": path[1]["origin_at"] = "2022-02-11T20:00:00Z"
    elif change == "wrong_participation": path[1]["entrants"][0]["prior_entries_after_cutoff"] = 0
    elif change == "metadata_change": path[1]["entrants"][0]["dob"] = "2000-01-01"
    with pytest.raises(ValueError):
        forecast_states(design, path, p, seed=seed, training_arrays_key=key)
