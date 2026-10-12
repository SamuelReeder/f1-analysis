"""Known probability and target-function fixtures; no fitted driver results."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.overall_ranking_recovery import SITES, from_states, specs, validate_protocol
from f1rank.overall.simulation import POINTS, MAX_EXPECTED_POINTS_MCSE
from test_overall_recovery_checks import (generated, posterior_fixture, recovery_archive, archive, study_fixture)


def protocol(*, seasons=65536, field_size=2):
    return dict(scenario_id="synthetic-full-distance", target="finite_simulation_equal_machinery_and_team",
        scenarios=[dict(event_id="synthetic-ref", context=0, laps=4, consecutive_racing_laps=4)],
        advancing=[field_size - 1], seasons=seasons, seed=23, max_points_mcse=MAX_EXPECTED_POINTS_MCSE,
        precision_policy="bounded_points_mcse_fixed_workload_no_retry")


def states():
    # Grid has no effect and no driver retires. Race PL is analytically solvable.
    values = dict(skill=np.zeros(2), overall_race_skill=np.array([-.4, .4]),
        sd_driver_form=np.array(0.), overall_sd_day=np.array(0.), overall_shared_weekend=np.array(1.),
        sigma0=np.array(0.), sigma_tau=np.array(0.), nu=np.array(4.),
        overall_grid=np.zeros(1), overall_pace=np.ones(1), overall_hazard=np.zeros(1))
    catalog = dict(field=["a", "b"], field_entries=[0, 1])
    return values, catalog


def key(kind, *drivers):
    return json.dumps(["finite_simulation", kind, *drivers], separators=(",", ":"))


def test_fixed_function_matches_exact_two_driver_expected_points():
    values, catalog = states()
    result, numeric = from_states(values, catalog, protocol())
    strength = values["overall_race_skill"]
    win = np.exp(strength - strength.max()); win /= win.sum()
    exact = POINTS[0] * win + POINTS[1] * (1 - win)
    for i, driver in enumerate(catalog["field"]):
        error = abs(result[key("points", driver)] - exact[i])
        assert error < 6 * numeric["points_mcse"][key("points", driver)]
    assert numeric["points_precision_passed"]
    assert result[key("rank_low", "b")] == result[key("rank_high", "b")] == 1
    assert result[key("best_share", "b")] == 1
    assert not numeric["exact_expected_points"] and not numeric["establishes_rank_precision"]
    assert not numeric["publication_allowed"]
    difference = key("points_difference", "b", "a")
    assert result[difference] == result[key("points", "b")] - result[key("points", "a")]
    # Here points sum to a constant, so paired error is twice a marginal error.
    np.testing.assert_allclose(numeric["points_mcse"][difference],
                               2 * numeric["points_mcse"][key("points", "a")])


def test_permutation_repetition_and_driver_nuisance_do_not_change_the_function():
    values, catalog = states()
    policy = protocol(seasons=128)
    first, numerical = from_states(values, catalog, policy)
    again, _ = from_states(dict(reversed(list(values.items()))), catalog, policy)
    assert first == again
    # Car and team states are inferred jointly but excluded from this target.
    nuisance = dict(values, car=np.array([99., -99.]), compat=np.array([-99., 99.]),
                    overall_car=np.array([999., -999.]), overall_team=np.array([-999., 999.]))
    neutral, _ = from_states(nuisance, catalog, policy)
    assert neutral == first
    moved, moved_numerical = from_states(values, dict(field=["b", "a"], field_entries=[1, 0]), policy)
    for name in ("points", "rank_low", "rank_high", "best_share"):
        for driver in catalog["field"]:
            assert moved[key(name, driver)] == first[key(name, driver)]
    assert moved[key("points_difference", "a", "b")] == -first[key("points_difference", "b", "a")]
    assert moved_numerical["max_driver_points_mcse"] == numerical["max_driver_points_mcse"]


def test_faster_race_state_increases_the_same_fixed_function():
    values, catalog = states()
    policy = protocol(seasons=1024)
    before, _ = from_states(values, catalog, policy)
    values["overall_race_skill"][0] += 1.
    after, _ = from_states(values, catalog, policy)
    assert after[key("points", "a")] > before[key("points", "a")]


def test_known_state_adapter_matches_joint_posterior_target(monkeypatch):
    from analysis import overall_ranking_recovery as ranking
    from f1rank.overall.simulation import JointDraws
    values, catalog = states()
    # Explicit interface fixture; this Boolean is not numerical fit evidence.
    posterior = {site: np.broadcast_to(value, (4, 3) + value.shape) for site, value in values.items()}
    expected = JointDraws.from_posterior(posterior, catalog["field_entries"], catalog["field"],
                                        checked={"converged": True})
    def inspect(joint, scenarios, **kwargs):
        assert joint.drivers == expected.drivers
        for name in SITES:
            np.testing.assert_array_equal(getattr(joint, name)[0], getattr(expected, name)[0])
        return dict(points=np.zeros((2, 2, 2)), n_races=1)
    monkeypatch.setattr(ranking, "simulate", inspect)
    from_states(values, catalog, protocol(seasons=4))


def test_small_constant_sample_cannot_pass_precision():
    values, catalog = states()
    # A fixed tiny sample can be constant even with stochastic line-order noise.
    values["overall_race_skill"] = np.array([-100., 100.])
    result, numeric = from_states(values, catalog, protocol(seasons=4))
    assert numeric["max_driver_points_mcse"] == 0
    assert numeric["worst_case_driver_points_mcse_bound"] > MAX_EXPECTED_POINTS_MCSE
    assert not numeric["points_precision_passed"]


def test_exact_finite_function_ties_remain_shared(monkeypatch):
    import analysis.overall_ranking_recovery as ranking
    values, catalog = states()
    def balanced(joint, scenarios, **kwargs):
        return dict(points=np.broadcast_to(np.array([[25., 18.], [18., 25.]]), (2, 2, 2)), n_races=1)
    monkeypatch.setattr(ranking, "simulate", balanced)
    result, _ = from_states(values, catalog, protocol(seasons=4))
    for driver in catalog["field"]:
        assert result[key("rank_low", driver)] == 1 and result[key("rank_high", driver)] == 2
        assert result[key("best_share", driver)] == .5
    assert result[key("points_difference", "b", "a")] == 0


@pytest.mark.parametrize("change", ["small", "odd", "seed", "precision", "format", "duplicate", "context", "missing", "target"])
def test_unregistered_or_incomplete_scenario_fails(change):
    policy = protocol()
    if change == "small": policy["seasons"] = 2
    elif change == "odd": policy["seasons"] += 1
    elif change == "seed": policy["seed"] = -1
    elif change == "precision": policy["max_points_mcse"] *= 2
    elif change == "format": policy["advancing"] = [2]
    elif change == "duplicate": policy["scenarios"].append(policy["scenarios"][0])
    elif change == "context": policy["scenarios"][0]["context"] = 1
    elif change == "missing": del policy["scenarios"][0]["consecutive_racing_laps"]
    elif change == "target": policy["target"] = "exact_expected_points"
    with pytest.raises(ValueError):
        validate_protocol(policy, field_size=2, n_contexts=1)


@pytest.mark.parametrize("change", ["parameter", "nonfinite", "field", "duplicate", "hazard"])
def test_missing_state_or_unsupported_outcome_is_preserved_as_failure(change):
    values, catalog = states()
    if change == "parameter": del values["nu"]
    elif change == "nonfinite": values["skill"][0] = np.nan
    elif change == "field": catalog["field_entries"][0] = 99
    elif change == "duplicate": catalog["field_entries"][0] = 1
    elif change == "hazard": values["overall_hazard"][:] = .999999
    with pytest.raises(ValueError):
        from_states(values, catalog, protocol(seasons=4))


def test_registered_quantities_and_low_precision_reach_complete_failure_record(generated):
    from analysis.overall_recovery_checks import evaluate, quantity_specs
    policy = protocol(seasons=4, field_size=len(generated["catalog"]["field"]))
    result = evaluate(generated, posterior_fixture(generated), study_id="fixture", randomization_seed=17,
                      ranking_protocol=policy)
    expected = quantity_specs(generated["blocks"], generated["catalog"], ranking_protocol=policy)
    assert set(result["quantities"]) == {row["id"] for row in expected}
    assert {r["id"] for r in specs(generated["catalog"]["field"])} <= set(result["quantities"])
    assert result["state"] == "failed_evaluation"
    assert result["evaluation_failure"] == "ranking_functional_simulation_precision"
    assert len(result["ranking_numerics"]) == 1 + len(result["selected_indices"])
    assert all(q["ties"] == len(result["selected_indices"]) for q in result["quantities"].values())
    assert not result["fit_verified"] and not result["publication_allowed"]


def register_ranking(recovery_archive, generated):
    from analysis import overall_ranking_recovery as ranking
    from analysis.overall_recovery_checks import quantity_specs
    directory, path, study_id, receipt = recovery_archive
    reg = json.loads(path.read_text())
    policy = protocol(seasons=4, field_size=len(generated["catalog"]["field"]))
    reg["ranking_functionals"] = dict(source_sha256=hashlib.sha256(Path(ranking.__file__).read_bytes()).hexdigest(),
        by_case={case: copy.deepcopy(policy) for case in reg["quantities_by_case"]})
    reg["quantities_by_case"][generated["case"]] = [s["id"] for s in quantity_specs(
        generated["blocks"], generated["catalog"], ranking_protocol=policy)]
    path.write_text(json.dumps(reg))
    receipt["specification"]["registration_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt["run_key"] = hashlib.sha256(json.dumps(receipt["specification"], sort_keys=True).encode()).hexdigest()
    (directory / "receipt.json").write_text(json.dumps(receipt))
    return directory, path, study_id, reg


def test_checked_archive_keeps_successful_fit_separate_from_failed_point_precision(generated, recovery_archive):
    from analysis.overall_recovery_checks import evaluate_checked
    directory, path, study_id, _ = register_ranking(recovery_archive, generated)
    result = evaluate_checked(generated, directory, path, study_id=study_id)
    assert result["fit_verified"] and result["state"] == "failed_evaluation"
    assert result["evaluation_failure"] == "ranking_functional_simulation_precision"
    assert not result["publication_allowed"]


def test_checked_archive_rejects_unregistered_functional_source(generated, recovery_archive, monkeypatch):
    from analysis import overall_recovery_checks as checks
    directory, path, study_id, reg = register_ranking(recovery_archive, generated)
    reg["ranking_functionals"]["source_sha256"] = "b" * 64
    path.write_text(json.dumps(reg))
    monkeypatch.setattr(checks, "evaluate", lambda *a, **kw: pytest.fail("Unregistered functional evaluated"))
    with pytest.raises(ValueError, match="source"):
        checks.evaluate_checked(generated, directory, path, study_id=study_id)


@pytest.mark.parametrize("change", ["missing", "precision", "protocol"])
def test_aggregate_cannot_ignore_ranking_precision_evidence(change):
    from analysis.overall_recovery_checks import summarize
    reg, rows = study_fixture()
    policy = protocol()
    digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    reg["ranking_functionals"] = dict(source_sha256="b" * 64,
        by_case={case: copy.deepcopy(policy) for case in reg["quantities_by_case"]})
    for row in rows:
        info = dict(protocol_sha256=digest, points_precision_passed=True)
        row["ranking_numerics"] = [dict(role="truth", numerics=info.copy())] + [
            dict(role="posterior", chain=c, numerics=info.copy()) for c in range(reg["fit_settings"]["chains"])]
    if change == "precision":
        rows[-1]["ranking_numerics"][0]["numerics"]["points_precision_passed"] = False
        result = summarize(rows, reg, registration_sha256="a" * 64)
        assert not result["evaluable"] and not result["tests"]
        assert result["problems"] == [dict(study=rows[-1]["study_id"], reason="ranking_functional_simulation_precision")]
    else:
        if change == "missing": rows[-1]["ranking_numerics"].pop()
        else: rows[-1]["ranking_numerics"][0]["numerics"]["protocol_sha256"] = "c" * 64
        with pytest.raises(ValueError):
            summarize(rows, reg, registration_sha256="a" * 64)
