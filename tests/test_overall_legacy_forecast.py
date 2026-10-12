"""Legacy forecast semantics checked against the original qualifying likelihood."""
import copy

import numpy as np
import pandas as pd
import pytest

from analysis import overall_legacy_forecast as legacy
from analysis.overall_forecast_catalog import build_qualifying_catalog
from f1rank.design import build_design
from test_overall_forecast_states import fixture, future_path, posterior as joint_fixture


def setup(cutoff="2022-02", target="2026-01", shape=(2, 7)):
    source, _, shared, snap = fixture(cutoff, target)
    design = build_design(2010, end_event=cutoff, tables=source)
    # Only the common qualifying priors are borrowed from this synthetic fixture;
    # this is not a fitted joint posterior or an input path for real baselines.
    template = joint_fixture(shared, shape)
    p = {k:v.copy() for k,v in template.items() if not k.startswith("overall_") and v.ndim == 2}
    q = design.arrays()
    rng = np.random.default_rng(73)
    for name, size in dict(level_z=q["n_drivers"], skill_z=q["n_entries"],
        car_step_z=q["n_cars"], car_jump_z=q["n_cars"], track_load_z=q["n_team_seasons"], compat_z=q["n_stints"]).items():
        p[name] = rng.normal(size=shape + (size,))
    return source, design, snap, p


def predict(design, path, p, *, batch=3, cutoff=None, metadata=None, seed=41):
    cutoff = cutoff or str(design.events.event_id.max())
    return legacy.forecast_legacy_features(design, path, p, cutoff=cutoff,
        fit_metadata=dict(start_season=2010, model_kw={}) if metadata is None else metadata,
        seed=seed, training_arrays_key=build_qualifying_catalog(design, path[0], cutoff=cutoff)["training_arrays_key"],
        draw_batch_size=batch)


def full_fixture(source, design, snap, event, p, *, keep_future_times=False):
    """Condition the original model on matching coordinates and the announced field."""
    source = copy.deepcopy(source)
    target = snap["event"]["event_id"]
    source["events"].loc[source["events"].event_id == target, "circuit_id"] = snap["event"]["circuit_id"]
    rows = [dict(event_id=target, driver_id=r["driver_id"], team=r["team"], constructor_id=r["team"],
                 constructor_name=r["team"], quali_position=np.nan) for r in snap["entrants"]]
    source["entries"] = pd.concat([source["entries"].loc[lambda x: x.event_id < target], pd.DataFrame(rows)], ignore_index=True)
    for r in snap["entrants"]:
        if r["driver_id"] not in set(source["drivers"].driver_id):
            source["drivers"] = pd.concat([source["drivers"], pd.DataFrame([dict(driver_id=r["driver_id"],
                dob=r["dob"], debut_season=r["debut_season"])])], ignore_index=True)
    # Future times do not define the training factor or the conditioned states.
    if not keep_future_times:
        source["quali_times"] = source["quali_times"].loc[lambda x: x.event_id < target]
    full = build_design(2010, end_event=target, tables=source)
    old_circuit = design.events.groupby("circuit_id").circuit_idx.first()
    for name, index in full.events.groupby("circuit_id").circuit_idx.first().items():
        full.circuit_factor[index] = design.circuit_factor[old_circuit[name]] if name in old_circuit else 0.
    noise = legacy.LegacyNoise(41, p["sd_level"].shape)
    result = {k:v.copy() for k,v in p.items() if v.ndim == 2}
    shape = p["sd_level"].shape
    old_driver = design.drivers.set_index("driver_id").driver_idx
    result["level_z"] = np.stack([p["level_z"][..., old_driver[r.driver_id]] if r.driver_id in old_driver
        else noise.draw("driver_level", r.driver_id) for r in full.drivers.itertuples()], axis=-1)
    old_entries = {(r.driver_id, r.event_id):r.entry_idx for r in design.entries.itertuples()}
    result["skill_z"] = np.stack([p["skill_z"][..., old_entries[(r.driver_id, r.event_id)]]
        if (r.driver_id, r.event_id) in old_entries else noise.draw("driver_step", [r.driver_id, r.event_id])
        for r in full.entries.itertuples()], axis=-1)
    old_cars = {(r.team, str(design.events.iloc[r.event_idx].event_id)):r.car_idx for r in design.cars.itertuples()}
    for name in ("car_step_z", "car_jump_z"):
        result[name] = np.stack([p[name][..., old_cars[(r.team, str(full.events.iloc[r.event_idx].event_id))]]
            if (r.team, str(full.events.iloc[r.event_idx].event_id)) in old_cars else
            noise.draw("car_step" if name == "car_step_z" else "car_first" if r.first else "car_jump",
                       [r.team, str(full.events.iloc[r.event_idx].event_id)], student=name == "car_step_z")
            for r in full.cars.itertuples()], axis=-1)
    old_pairs = {(r.driver_id, r.team):r.stint_idx for r in design.entries.drop_duplicates("stint_idx").itertuples()}
    result["compat_z"] = np.stack([p["compat_z"][..., old_pairs[(r.driver_id, r.team)]]
        if (r.driver_id, r.team) in old_pairs else noise.draw("compatibility", (r.driver_id, r.team))
        for r in full.entries.groupby("stint_idx").first().itertuples()], axis=-1)
    old_loads = {(r.team, int(r.season)):index for index,r in design.cars.groupby("team_season_idx").first().iterrows()}
    result["track_load_z"] = np.stack([p["track_load_z"][..., old_loads[(r.team, r.season)]]
        if (r.team, r.season) in old_loads else noise.draw("track_load", (r.team, int(r.season)))
        for r in full.cars.groupby("team_season_idx").first().itertuples()], axis=-1)
    q = full.arrays()
    result.update(car_total=np.zeros(shape+(q["n_cars"],)), driver_form_z=np.zeros(shape+(q["n_entries"],)),
        mu=np.zeros(shape+(q["n_sessions"],)), sigma_u=np.zeros(shape+(q["n_sessions"],)),
        car_segment_z=np.zeros(shape+(q["n_car_segments"],)))
    for name,value in dict(sigma0=.4, sigma_tau=.3, nu=5., sd_car_segment=.2).items():
        result[name] = np.full(shape,value)
    for row in full.entries[full.entries.event_id == target].itertuples():
        result["driver_form_z"][..., row.entry_idx] = noise.draw("qualifying_day", [row.driver_id, target])
    event_index = full.events.loc[full.events.event_id == target, "event_idx"].item()
    for row in full.cars[full.cars.event_idx == event_index].itertuples():
        car = event["cars"].index(row.team)
        result["car_total"][..., row.car_idx] = event["states"]["car"][..., car] + event["states"]["car_event"][..., car]
    return full, result


@pytest.mark.parametrize("cutoff,target,new_team,new_driver", [
    ("2022-01", "2022-02", False, False),
    ("2022-02", "2026-01", False, False),
    ("2022-02", "2026-01", True, False),
    ("2022-02", "2026-01", True, True),
])
def test_legacy_future_states_match_original_model_with_same_innovations(cutoff,target,new_team,new_driver):
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    from f1rank.model import model
    source, design, snap, p = setup(cutoff,target)
    design.circuit_factor[:] = .8
    if new_team:
        snap["entrants"][0]["team"] = "new_team"
    if new_driver:
        snap["entrants"][0].update(driver_id="new_driver", dob="2004-01-01", debut_season=2026, experience_before_cutoff=0)
    event = predict(design,[snap],p)["events"][0]
    full, expanded = full_fixture(source,design,snap,event,p)
    run = handlers.seed(handlers.condition(model, data={k:jnp.asarray(v[0,0]) for k,v in expanded.items()}), jax.random.PRNGKey(19))
    trace = handlers.trace(run).get_trace({k:jnp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in full.arrays().items()})
    entries = full.entries.loc[lambda x: x.event_id == target].sort_values("driver_id").entry_idx.to_numpy()
    event_index = full.events.loc[lambda x: x.event_id == target,"event_idx"].item()
    cars = full.cars.loc[lambda x: x.event_idx == event_index].sort_values("team").car_idx.to_numpy()
    for name in ("skill","compat","driver_form"):
        np.testing.assert_allclose(event["states"][name][0,0],np.asarray(trace[name]["value"])[entries],rtol=3e-6,atol=3e-6)
    for name in ("car","car_track","car_event"):
        np.testing.assert_allclose(event["states"][name][0,0],np.asarray(trace[name]["value"])[cars],rtol=3e-6,atol=3e-6)
    assert len(event["drivers"]) == len(snap["entrants"])
    assert event["catalog"]["likelihood_arm"] == "legacy_qualifying"


def test_lineage_compatibility_persists_across_reset_and_return():
    _,design,snap,p = setup()
    path = future_path(snap)
    # d0 used t0 before the reset and subsequently moved to t1; returning to t0
    # must reuse the original lineage effect, not an era-specific replacement.
    path[0]["entrants"][0]["team"] = "t0"
    path[1]["entrants"][0]["team"] = "t0"
    result = predict(design,path,p)
    raw = {(r.driver_id,r.team):p["sd_compat"]*p["compat_z"][...,r.stint_idx]
           for r in design.entries.drop_duplicates("stint_idx").itertuples()}
    for event in result["events"]:
        expected = np.stack([raw[(r["driver_id"],r["team"])] for r in event["catalog"]["entries"]],axis=-1)
        expected -= expected.mean(axis=-1,keepdims=True)
        np.testing.assert_allclose(event["states"]["compat"],expected,atol=1e-15)


def test_future_transients_do_not_change_legacy_race_features():
    _,design,snap,p = setup()
    before = predict(design,[snap],p)["events"][0]
    changed = {k:v.copy() for k,v in p.items()}
    changed["sd_driver_form"] *= 100
    changed["sd_car_event"] *= 100
    after = predict(design,[snap],changed)["events"][0]
    for name in ("driver","car"):
        np.testing.assert_array_equal(before["race_features"][name],after["race_features"][name])
    assert not np.array_equal(before["states"]["qualifying_total"],after["states"]["qualifying_total"])
    expected_driver = np.median(before["states"]["skill"]+before["states"]["compat"],axis=(0,1))
    expected_car = np.median(before["states"]["car"]+before["states"]["car_track"],axis=(0,1))
    np.testing.assert_array_equal(before["race_features"]["driver"],expected_driver)
    np.testing.assert_array_equal(before["race_features"]["car"],expected_car[before["entry_car"]])


def test_batches_fields_and_prefixes_preserve_all_legacy_draws():
    _,design,snap,p = setup()
    path = future_path(snap)
    whole = predict(design,path,p,batch=7)
    shuffled = copy.deepcopy(path)
    for s in shuffled:
        s["entrants"].reverse()
    batched,prefix = predict(design,shuffled,p,batch=3),predict(design,path[:1],p,batch=2)
    for i,event in enumerate(whole["events"]):
        for other in (batched,prefix) if i==0 else (batched,):
            for name,value in event["states"].items():
                np.testing.assert_array_equal(value,other["events"][i]["states"][name])
            for name,value in event["race_features"].items():
                np.testing.assert_array_equal(value,other["events"][i]["race_features"][name])
    assert not whole["publication_allowed"] and not whole["convergence_verified"]


def test_unseen_circuit_records_original_legacy_no_adjustment_policy():
    _,design,snap,p = setup()
    snap["event"]["circuit_id"] = "new"
    event = predict(design,[snap],p)["events"][0]
    assert event["circuit_factor_policy"] == "legacy_unseen_no_adjustment"
    assert not event["states"]["car_track"].any()


@pytest.mark.parametrize("fault",["successor","era_metadata","wrong_start","missing","axes","last_nonfinite","negative","future","wrong_history","missing_path"])
def test_legacy_adapter_rejects_incompatible_or_incomplete_inputs(fault):
    _,design,snap,p = setup()
    metadata = dict(start_season=2010,model_kw={})
    cutoff = None
    path = [snap]
    if fault == "successor": p["overall_qualifying_link"] = np.ones((2,7))
    elif fault == "era_metadata": metadata["model_kw"] = {"compat_unit":"era"}
    elif fault == "wrong_start": metadata["start_season"] = 2018
    elif fault == "missing": p.pop("compat_z")
    elif fault == "axes": p["compat_z"] = p["compat_z"][:1]
    elif fault == "last_nonfinite": p["skill_z"][-1,-1,-1] = np.nan
    elif fault == "negative": p["sd_level"][-1,-1] = -1.
    elif fault == "future": cutoff = "2022-01"
    elif fault == "missing_path": path = future_path(snap)[1:]
    else:
        with pytest.raises(ValueError,match="training arrays"):
            next(legacy.iter_legacy_qualifying_states(design,path,p,cutoff="2022-02",fit_metadata=metadata,
                seed=41,training_arrays_key="wrong"))
        return
    with pytest.raises(ValueError):
        predict(design,path,p,cutoff=cutoff,metadata=metadata)


def test_future_qualifying_results_do_not_change_training_or_forecast_features():
    source,design,snap,p = setup()
    original = predict(design,[snap],p)
    source["quali_times"].loc[lambda x:x.event_id>"2022-02","time_s"] = .01
    other = build_design(2010,end_event="2022-02",tables=source)
    future = predict(other,[snap],p)
    assert original["training_arrays_key"] == future["training_arrays_key"]
    for name,value in original["events"][0]["race_features"].items():
        np.testing.assert_array_equal(value,future["events"][0]["race_features"][name])


def race_case():
    _,design,snap,p = setup()
    snap["information_set"] = "actual_grid"
    for i,row in enumerate(snap["entrants"]):
        row["grid_position"] = i + 1 if i < 5 else 0
    forecast = predict(design,[snap],p)
    panel = {snap["event"]["event_id"]:{r["driver_id"]:i+1 for i,r in enumerate(snap["entrants"])}}
    coefficients = dict(c_grid=np.ones((2,3)),a_car=np.zeros((2,3)),b_driver=np.zeros((2,3)))
    return forecast,panel,coefficients


def test_race_adapter_uses_origin_grid_and_independent_coefficient_draw_axes():
    forecast,panel,coefficients = race_case()
    result = legacy.forecast_legacy_race(forecast,coefficients,panel)
    # PL weights are reciprocal grid slots here; the pit-lane slot is full field.
    weights = 1 / np.arange(1,7)
    expected = sum(np.log(weights[i]/weights[i:].sum()) for i in range(6))
    np.testing.assert_allclose(result["log_probabilities"],np.full((2,3,1),expected))
    assert result["qualifying_posterior_shape"] == [2,7]
    assert result["coefficient_posterior_shape"] == [2,3]
    assert result["origins"][0]["snapshot_sha256"] == forecast["events"][0]["catalog"]["snapshot_sha256"]
    assert not result["publication_allowed"] and not result["source_availability_verified"]


@pytest.mark.parametrize("fault",["missing_event","missing_driver","prequalifying","incomplete_features","invalid_order"])
def test_race_adapter_cannot_substitute_response_data_for_forecast_inputs(fault):
    forecast,panel,coefficients = race_case()
    event = forecast["events"][0]
    if fault == "missing_event": panel.clear()
    elif fault == "missing_driver": panel[event["event_id"]].pop(event["drivers"][0])
    elif fault == "prequalifying": event["catalog"]["information_set"] = "prequalifying"
    elif fault == "incomplete_features": forecast.pop("feature_policy")
    else: panel[event["event_id"]][event["drivers"][0]] = 2
    with pytest.raises(ValueError):
        legacy.forecast_legacy_race(forecast,coefficients,panel)
