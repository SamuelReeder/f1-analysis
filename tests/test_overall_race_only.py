"""Race-only evidence boundaries and the inherited joint state prior."""
import copy

import numpy as np
import pytest

from analysis.overall_race_only import build_race_only, race_only_model, without_qualifying
from f1rank.overall.design import build
from test_overall import tables


def test_race_evidence_remains_usable_without_qualifying_observations():
    source, clean = tables()
    source["quali_times"] = source["quali_times"].iloc[:0].copy()
    design = build("2022-02", tables=source, clean=clean)
    q = design.qualifying.arrays()
    assert q["n_sessions"] == q["n_car_segments"] == 0
    assert q["y"].size == q["train"].size == 0
    assert len(design.laps["y"]) > 0 and design.outcomes["mask"].any()
    assert set(design.catalog["field"]) == {f"d{i}" for i in range(6)}
    np.testing.assert_array_equal(q["circuit_factor"], np.zeros(q["n_circuits"]))


def test_diagnostics_record_empty_sites_without_changing_existing_decisions():
    from f1rank.overall.diagnostics import check, MIN_CHAINS, MIN_ESS
    draws = np.random.default_rng(0).normal(size=(MIN_CHAINS, MIN_ESS))
    original = check({"parameter": draws}, 0)
    extended = check({"parameter": draws, "mu": np.empty(draws.shape + (0,))}, 0)
    assert extended.pop("empty_sites") == [dict(parameter="mu", shape=list(draws.shape + (0,)))]
    original.pop("empty_sites", None)
    assert extended == original


def test_empty_sites_cannot_supply_convergence_or_hide_constant_coordinates():
    from f1rank.overall.diagnostics import check, MIN_CHAINS, MIN_ESS
    shape = (MIN_CHAINS, MIN_ESS)
    empty = dict(mu=np.empty(shape + (0,)))
    result = check(empty, 0)
    assert not result["converged"] and "No varying posterior coordinates" in result["failures"]
    result = check(dict(empty, level_z=np.zeros(shape + (1,))), 0)
    assert not result["converged"]
    assert "level_z: constant stochastic coordinate" in result["failures"]


def blocks(design):
    return dict(qualifying=design.qualifying.arrays(), laps=design.laps, outcomes=design.outcomes)


def test_race_only_keeps_the_shared_race_history_and_structural_prior_inputs():
    source, clean = tables()
    joint = build("2022-02", tables=source, clean=clean)
    race = build_race_only("2022-02", tables=source, clean=clean)
    assert race.catalog["field"] == joint.catalog["field"]
    for name in ("laps", "outcomes"):
        for key, value in blocks(joint)[name].items():
            np.testing.assert_array_equal(blocks(race)[name][key], value, err_msg=key)
    expected = without_qualifying(joint.qualifying.arrays())
    for key, value in race.qualifying.arrays().items():
        np.testing.assert_array_equal(value, expected[key], err_msg=key)
    assert race.catalog["likelihood_arm"] == "current_race_only"
    assert not race.qualifying.entries.has_time.any()


def test_qualifying_times_positions_and_availability_cannot_change_race_only_inputs():
    from f1rank.overall.fitting import arrays_digest
    source, clean = tables()
    original = build_race_only("2022-02", tables=source, clean=clean)
    source["quali_times"]["time_s"] = np.nan
    source["entries"]["quali_position"] = np.arange(len(source["entries"]))[::-1]
    changed = build_race_only("2022-02", tables=source, clean=clean)
    del source["quali_times"]
    absent = build_race_only("2022-02", tables=source, clean=clean)
    assert arrays_digest(blocks(original)) == arrays_digest(blocks(changed)) == arrays_digest(blocks(absent))
    assert original.catalog == changed.catalog == absent.catalog


def test_default_race_only_loader_never_opens_qualifying_times(monkeypatch):
    from analysis import overall_race_only as module
    source, clean = tables()
    reads = []
    def read(path):
        reads.append(path.stem)
        assert path.stem != "quali_times"
        return source[path.stem]
    monkeypatch.setattr(module.pd, "read_parquet", read)
    design = build_race_only("2022-02", clean=clean)
    assert "race" in reads and design.qualifying.obs.empty


def test_race_only_preserves_cutoff_first_filtering():
    from f1rank.overall.fitting import arrays_digest
    source, clean = tables()
    before = build_race_only("2022-02", tables=source, clean=clean)
    for key in ("race", "entries"):
        source[key].loc[source[key].event_id == "2026-01", "team"] = "future_team"
    clean.loc[clean.event_id == "2026-01", "y"] = 9000.
    after = build_race_only("2022-02", tables=source, clean=clean)
    assert arrays_digest(blocks(before)) == arrays_digest(blocks(after))


@pytest.fixture(scope="module")
def traced():
    import jax
    from numpyro import handlers
    source, clean = tables()
    design = build("2022-02", tables=source, clean=clean)
    q = design.qualifying.arrays()
    # A nonzero learned factor makes its removal observable in this small field.
    q["circuit_factor"] = np.ones(q["n_circuits"])
    args = (q, design.laps, design.outcomes)
    trace = handlers.trace(handlers.seed(race_only_model, jax.random.PRNGKey(19))).get_trace(*args)
    params = {name: site["value"] for name, site in trace.items()
              if site["type"] == "sample" and not site["is_observed"]}
    return args, params, trace


def test_race_only_density_is_the_inherited_model_on_no_qualifying_evidence(traced):
    from numpyro.infer.util import log_density
    from f1rank.overall.model import model
    args, params, _ = traced
    race_density, race_trace = log_density(race_only_model, args, {}, params)
    explicit_density, explicit_trace = log_density(model, (without_qualifying(args[0]), *args[1:]), {}, params)
    assert np.isfinite(race_density)
    np.testing.assert_array_equal(race_density, explicit_density)
    assert race_trace["y"]["value"].size == 0
    np.testing.assert_array_equal(race_trace["car_track"]["value"], 0.)
    for name, site in race_trace.items():
        np.testing.assert_array_equal(site["value"], explicit_trace[name]["value"], err_msg=name)
        if site["type"] == "sample":
            np.testing.assert_array_equal(site["fn"].log_prob(site["value"]),
                explicit_trace[name]["fn"].log_prob(site["value"]), err_msg=name)


def test_model_rejects_indirect_qualifying_evidence_by_construction(traced):
    from numpyro.infer.util import log_density
    args, params, _ = traced
    before, _ = log_density(race_only_model, args, {}, params)
    q = copy.deepcopy(args[0])
    q["y"][:] = np.nan
    q["circuit_factor"][:] = np.nan
    for name in ("obs_entry", "obs_car", "obs_session", "obs_car_segment", "car_segment_session"):
        q[name][:] = -999
    after, trace = log_density(race_only_model, (q, *args[1:]), {}, params)
    np.testing.assert_array_equal(before, after)
    assert np.isfinite(after) and trace["y"]["value"].size == 0


def test_common_priors_and_race_factors_match_the_joint_candidate(traced):
    from numpyro import handlers
    from f1rank.overall.model import model
    args, params, race_trace = traced
    q = dict(args[0], circuit_factor=np.zeros(args[0]["n_circuits"]))
    # Qualifying-only segment coordinates have been marginalized by the empty
    # block. Supply all other draws identically to the full observed model.
    qualifying_only = {"mu", "sigma_u", "car_segment_z", "car_segment", "y"}
    common = {name: value for name, value in params.items() if name not in qualifying_only}
    joint_trace = handlers.trace(handlers.seed(handlers.substitute(model, data=common), 0)).get_trace(
        q, *args[1:])
    for name, site in race_trace.items():
        if name in qualifying_only:
            continue
        np.testing.assert_array_equal(site["value"], joint_trace[name]["value"], err_msg=name)
        if site["type"] == "sample":
            np.testing.assert_array_equal(site["fn"].log_prob(site["value"]),
                joint_trace[name]["fn"].log_prob(site["value"]), err_msg=name)


@pytest.mark.parametrize("target", ["laps", "order", "completion", "grid"])
def test_race_evidence_is_still_used(traced, target):
    from numpyro.infer.util import log_density
    args, params, _ = traced
    changed = copy.deepcopy(args)
    factor = dict(laps="overall_laps", order="overall_finish_order",
                  completion="overall_completion", grid="overall_finish_order")[target]
    if target == "laps":
        changed[1]["y"] = np.asarray(changed[1]["y"]).copy()
        changed[1]["y"][0] += 1.
    elif target == "order":
        for key in ("entry", "log_grid", "mask"):
            changed[2][key][0] = changed[2][key][0, ::-1]
    elif target == "completion": changed[2]["failed"][0] = ~changed[2]["failed"][0]
    elif target == "grid": changed[2]["log_grid"][0, 0] += 1.
    _, before = log_density(race_only_model, args, {}, params)
    _, after = log_density(race_only_model, changed, {}, params)
    assert not np.allclose(before[factor]["fn"].log_factor, after[factor]["fn"].log_factor)


def test_race_only_potential_has_finite_gradient():
    import jax
    from numpyro.infer import init_to_median
    from numpyro.infer.util import initialize_model
    source, clean = tables()
    design = build_race_only("2022-02", tables=source, clean=clean)
    initialized = initialize_model(jax.random.PRNGKey(0), race_only_model,
        model_args=(design.qualifying.arrays(), design.laps, design.outcomes),
        init_strategy=init_to_median(num_samples=3))
    assert np.isfinite(initialized.param_info.potential_energy)
    assert all(np.isfinite(a).all() for a in jax.tree_util.tree_leaves(initialized.param_info.z_grad))


def test_race_only_forecast_uses_matching_states_and_retains_arm_identity():
    from analysis.overall_forecast_catalog import build_catalog
    from analysis.overall_forecast_states import forecast_states
    from test_overall_forecast_states import expanded_parameters, fixture, posterior, trace_model
    source, clean, _, snap = fixture("2022-01", "2022-02")
    design = build_race_only("2022-01", tables=source, clean=clean)
    full = build_race_only("2022-02", tables=source, clean=clean)
    catalog = build_catalog(design, snap)
    p = posterior(design)
    event = forecast_states(design, [snap], p, seed=41,
        training_arrays_key=catalog["training_arrays_key"])["events"][0]
    assert event["catalog"]["likelihood_arm"] == "current_race_only"
    np.testing.assert_array_equal(event["states"]["car_track"], 0.)
    trace = trace_model(full, expanded_parameters(design, full, p, snap, 41))
    entries = full.qualifying.entries.query("event_id == '2022-02'").sort_values("driver_id").entry_idx
    np.testing.assert_allclose(event["states"]["race_skill"][0, 0],
                               np.asarray(trace["overall_race_skill"]["value"])[entries], rtol=3e-6, atol=3e-6)
    with pytest.raises(ValueError, match="training arrays"):
        forecast_states(design, [snap], p, seed=41, training_arrays_key="different-arm-fit")


def test_race_only_conditional_weekend_keeps_complete_draw_batches_and_actual_grid():
    from test_overall_forecast_states import fixture
    from test_overall_observation_kernels import observation_posterior, predict
    source, clean, _, snap = fixture("2022-01", "2022-02")
    design = build_race_only("2022-01", tables=source, clean=clean)
    snap["information_set"] = "actual_grid"
    for i, row in enumerate(snap["entrants"]): row["grid_position"] = i + 1
    p = observation_posterior(design)
    batches = predict(design, [snap], p)
    assert [b["draw_range"] for b in batches] == [[0, 3], [3, 6], [6, 7]]
    for batch in batches:
        event = batch["events"][0]
        assert event["catalog"]["likelihood_arm"] == "current_race_only"
        np.testing.assert_array_equal(event["grid_feature_slot"], np.arange(1, 7))
        assert np.isfinite(event["strength"]).all()
        assert not batch["publication_allowed"] and not batch["convergence_verified"]


@pytest.mark.parametrize("change", ["undeclared_arm", "observations", "factor"])
def test_forecast_cannot_silently_restore_qualifying_inputs(change):
    from analysis.overall_forecast_catalog import build_catalog
    from test_overall_forecast_states import fixture
    source, clean, joint, snap = fixture("2022-01", "2022-02")
    design = build_race_only("2022-01", tables=source, clean=clean)
    if change == "undeclared_arm": del design.catalog["likelihood_arm"]
    elif change == "observations": design.qualifying.obs = joint.qualifying.obs
    elif change == "factor": design.qualifying.circuit_factor[:] = 1.
    with pytest.raises(ValueError):
        build_catalog(design, snap)
