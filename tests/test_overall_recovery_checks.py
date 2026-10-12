"""Exact probability fixtures and complete-study accounting, without MCMC."""
import copy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pytest

from analysis.overall_recovery_checks import (OBSERVATIONS, ROLES, evaluate, generate_with_parameters,
    quantity_specs, quantity_values, rank_statistic, recovery_intervals, summarize, validate_inventory)
from test_overall_posterior_archive import archive  # Reuse an explicit cache-format fixture.


def test_tie_randomization_includes_the_generating_truth():
    values = [-1., 0., 0., 1.]
    assert [rank_statistic(values, 0., tie_uniform=u)["rank"] for u in (0., .4, .8)] == [1, 2, 3]
    assert rank_statistic(values, -2., tie_uniform=.5)["rank"] == 0
    assert rank_statistic(values, 2., tie_uniform=.5)["rank"] == 4
    assert [rank_statistic([0.] * 4, 0., tie_uniform=(i + .5) / 5)["rank"] for i in range(5)] == list(range(5))


@pytest.mark.parametrize("draws,truth,u", [([], 0., .5), ([np.nan], 0., .5), ([0.], np.inf, .5),
    ([[0., 1.]], 0., .5), ([0.], 0., 1.), ([0.], 0., -1.)])
def test_bad_rank_inputs_cannot_be_clipped_or_dropped(draws, truth, u):
    with pytest.raises(ValueError):
        rank_statistic(draws, truth, tie_uniform=u)


def exact_binary_ranks(*, ignore_data, likelihood_quantity):
    """Enumerate a binary Bayesian model and every posterior/tie outcome.

    Symmetric prior; observation matches its truth with probability 0.9.
    This checks the SBC null exactly, without a random test that can flake.
    """
    histogram = np.zeros(5)
    for truth, y in itertools.product((0, 1), repeat=2):
        data_mass = .5 * (.9 if y == truth else .1)
        p = .5 if ignore_data else (.9 if y else .1)
        f = (lambda value: .9 if value == y else .1) if likelihood_quantity else (lambda value: value)
        for draws in itertools.product((0, 1), repeat=4):
            mass = data_mass * np.prod([p if d else 1 - p for d in draws])
            values = [f(d) for d in draws]
            ties = sum(v == f(truth) for v in values)
            for tie in range(ties + 1):
                rank = rank_statistic(values, f(truth), tie_uniform=(tie + .5) / (ties + 1))["rank"]
                histogram[rank] += mass / (ties + 1)
    return histogram


def test_exact_posterior_has_uniform_parameter_and_data_dependent_ranks():
    for likelihood in (False, True):
        np.testing.assert_allclose(exact_binary_ranks(ignore_data=False, likelihood_quantity=likelihood),
                                   np.ones(5) / 5, atol=1e-14)


def test_likelihood_quantity_detects_prior_only_inference_that_parameter_ranks_miss():
    np.testing.assert_allclose(exact_binary_ranks(ignore_data=True, likelihood_quantity=False),
                               np.ones(5) / 5, atol=1e-14)
    likelihood = exact_binary_ranks(ignore_data=True, likelihood_quantity=True)
    np.testing.assert_allclose(likelihood.sum(), 1., atol=1e-14)
    assert likelihood[-1] > likelihood[0] and not np.allclose(likelihood, np.ones(5) / 5)


@pytest.fixture(scope="module")
def generated():
    return generate_with_parameters("connected", 23)


def test_outer_truth_capture_leaves_existing_generation_unchanged(generated):
    from f1rank.overall.fitting import arrays_digest
    from f1rank.overall.recovery import generate
    blocks, catalog, truth = generate("connected", 23)
    assert arrays_digest(blocks) == generated["arrays_key"] and catalog == generated["catalog"]
    for site in truth:
        np.testing.assert_array_equal(truth[site], generated["truth"][site])
    assert not set(OBSERVATIONS) & set(generated["parameters"])
    assert not any(n.startswith("generated_") for n in generated["parameters"])


def test_quantities_cover_all_current_contrasts_and_observation_factors(generated):
    import jax.numpy as jnp
    from numpyro.infer.util import log_density
    from f1rank.overall.model import model
    blocks, cat, params = (generated[k] for k in ("blocks", "catalog", "parameters"))
    convert = lambda block: {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in block.items()}
    density, trace = log_density(model, tuple(convert(blocks[n]) for n in ("qualifying", "laps", "outcomes")),
                                {}, {k: jnp.asarray(v) for k, v in params.items()})
    values = quantity_values(blocks, cat, params)
    reference = sum(float(trace[n]["fn"].log_prob(trace[n]["value"]).sum()) for n in OBSERVATIONS)
    np.testing.assert_allclose(values["likelihood:joint"], reference)
    assert not np.isclose(values["likelihood:joint"], density)
    for site in OBSERVATIONS:
        assert "likelihood:" + site in values
    i, j = cat["field_entries"][:2]
    key = json.dumps(["overall_race_skill", "driver_difference", cat["field"][1], cat["field"][0]], separators=(",", ":"))
    np.testing.assert_allclose(values[key], generated["truth"]["overall_race_skill"][j] - generated["truth"]["overall_race_skill"][i])
    specs = quantity_specs(blocks, cat)
    assert set(values) == {s["id"] for s in specs}
    assert len(values) == len(specs)


def test_observed_site_substitution_is_not_a_recovery_quantity(generated):
    params = dict(generated["parameters"], y=generated["blocks"]["qualifying"]["y"])
    with pytest.raises(ValueError, match="responses"):
        quantity_values(generated["blocks"], generated["catalog"], params)


def posterior_fixture(generated):
    return {name: np.broadcast_to(value, (4, 3) + value.shape).copy()
            for name, value in {**generated["parameters"], **generated["truth"]}.items()}


def test_one_joint_draw_per_chain_is_reproducible_and_not_a_fit_certificate(generated):
    posterior = posterior_fixture(generated)
    a = evaluate(generated, posterior, study_id="fixture", randomization_seed=17)
    b = evaluate(generated, dict(reversed(list(posterior.items()))), study_id="fixture", randomization_seed=17)
    assert a == b and not a["fit_verified"] and not a["publication_allowed"]
    assert [i[0] for i in a["selected_indices"]] == list(range(4))
    assert all(0 <= i[1] < 3 for i in a["selected_indices"])
    assert all(q["posterior_draws"] == 4 and q["ties"] == 4 for q in a["quantities"].values())


def test_recovery_intervals_match_existing_pairwise_summary_on_the_same_draws(generated):
    from f1rank.overall.recovery import recovery_metrics
    post = posterior_fixture(generated)
    rng = np.random.default_rng(0)
    for site in ("skill", "overall_race_skill"):
        post[site] += rng.normal(0., .1, size=post[site].shape)
    old = recovery_metrics(post, generated["truth"], generated["catalog"])
    new = {r["quantity"]: r for r in recovery_intervals(generated, post)}
    for row in old:
        key = json.dumps([row["site"], "driver_difference", row["driver_i"], row["driver_j"]], separators=(",", ":"))
        actual = new[key]
        np.testing.assert_allclose(actual["interval"], [row["q05"], row["q95"]])
        assert actual["covered"] == row["covered"] and actual["posterior_draws"] == row["n_draws"]


@pytest.mark.parametrize("change", ["missing", "nonfinite", "axes"])
def test_recovery_intervals_require_complete_joint_state_draws(generated, change):
    post = posterior_fixture(generated)
    if change == "missing": del post["overall_car"]
    elif change == "nonfinite": post["overall_car"][-1, -1, -1] = np.nan
    else: post["overall_car"] = post["overall_car"][:-1]
    with pytest.raises(ValueError):
        recovery_intervals(generated, post)


@pytest.mark.parametrize("change", ["bad_last_draw", "axis", "missing", "truth", "state_truth", "data", "role"])
def test_invalid_truth_or_posterior_cannot_leave_a_selected_success(generated, change):
    g, posterior = copy.deepcopy(generated), posterior_fixture(generated)
    if change == "bad_last_draw": posterior["sd_level"][-1, -1] = np.nan
    elif change == "axis": posterior["sd_level"] = posterior["sd_level"].reshape(3, 4)
    elif change == "missing": del posterior["sd_level"]
    elif change == "truth": g["parameters"]["sd_level"] = np.asarray(999.)
    elif change == "state_truth": g["truth"]["skill"][0] += 1.
    elif change == "data": g["blocks"]["qualifying"]["y"][0] += 1.
    elif change == "role": g["role"] = "stress"
    with pytest.raises(ValueError):
        evaluate(g, posterior, study_id="fixture", randomization_seed=17)


def study_fixture(replicates=25):
    studies, synthetic, records = [], {}, []
    for ci, (case, role) in enumerate(ROLES.items()):
        for index in range(replicates):
            identity = f"{case}-{index}"
            seed = ci * replicates + index
            studies.append(identity)
            synthetic[identity] = dict(case=case, seed=seed, role=role)
            # These are explicit report-format fixtures, never fitted evidence.
            records.append(dict(study_id=identity, case=case, generation_seed=seed, role=role,
                state="evaluated", fit_verified=True, selection_policy="one_draw_per_chain", randomization_seed=17,
                registration_sha256="a" * 64,
                quantities={q: dict(rank=index % 5, posterior_draws=4) for q in ("theta", "likelihood:joint")}))
    registration = dict(purpose="recovery_calibration", studies=studies, synthetic=synthetic,
        replicates_per_case={case: replicates for case in ROLES},
        quantities_by_case={case: ["theta", "likelihood:joint"] for case in ROLES},
        fit_settings=dict(chains=4), selection_policy="one_draw_per_chain", randomization_seed=17,
        family_error_probability=.05, recovery_interval_probability=.9)
    return registration, records


def test_complete_inventory_keeps_fixed_truth_stress_out_of_uniform_rank_tests():
    reg, rows = study_fixture()
    result = summarize(rows, reg, registration_sha256="a" * 64)
    assert result["evaluable"] and result["no_detected_miscalibration"]
    assert all(t["case"] != "dominant_teammate" for t in result["tests"])
    assert len(result["stress_studies"]) == reg["replicates_per_case"]["dominant_teammate"]
    assert not result["establishes_full_recovery"] and not result["establishes_adequate_power"]
    assert not result["publication_allowed"]


def test_dependent_quantities_do_not_multiply_the_number_of_trials():
    reg, rows = study_fixture()
    for row in rows:
        row["quantities"]["likelihood:joint"]["rank"] = 0
    result = summarize(rows, reg, registration_sha256="a" * 64)
    assert not result["no_detected_miscalibration"]
    assert all(t["replicates"] == 25 for t in result["tests"])
    assert all(t["adjusted_p_value"] == min(1., t["p_value"] * result["family_size"]) for t in result["tests"])


@pytest.mark.parametrize("change", ["missing", "failed", "unverified", "running"])
def test_missing_or_failed_replicate_prevents_selected_subset_calibration(change):
    reg, rows = study_fixture()
    if change == "missing": rows.pop()
    elif change == "failed": rows[-1]["state"] = "failed_fit"
    elif change == "running": rows[-1]["state"] = "running"
    else: rows[-1]["fit_verified"] = False
    result = summarize(rows, reg, registration_sha256="a" * 64)
    assert not result["evaluable"] and result["no_detected_miscalibration"] is None
    assert result["problems"] and not result["tests"]
    assert result["all_required_processed"] == (change in ("failed", "unverified"))


@pytest.mark.parametrize("change", ["case", "role", "seed", "count", "likelihood", "policy", "alpha", "purpose"])
def test_incomplete_or_relabelled_inventory_is_rejected(change):
    reg, _ = study_fixture()
    first = reg["studies"][0]
    if change == "case": reg["synthetic"][first]["case"] = "easier"
    elif change == "role": reg["synthetic"][first]["role"] = "stress"
    elif change == "seed": reg["synthetic"][first]["seed"] = reg["synthetic"][reg["studies"][1]]["seed"]
    elif change == "count": reg["replicates_per_case"]["connected"] -= 1
    elif change == "likelihood": reg["quantities_by_case"]["connected"] = ["theta"]
    elif change == "policy": reg["selection_policy"] = "all_correlated_iterations"
    elif change == "alpha": reg["family_error_probability"] = 1.
    elif change == "purpose": reg["purpose"] = "numerical_profile"
    with pytest.raises(ValueError):
        validate_inventory(reg)


@pytest.mark.parametrize("change", ["duplicate", "seed", "quantity", "support", "randomization", "registration"])
def test_result_identity_and_quantity_changes_are_rejected(change):
    reg, rows = study_fixture()
    if change == "duplicate": rows.append(copy.deepcopy(rows[0]))
    elif change == "seed": rows[0]["generation_seed"] += 1
    elif change == "quantity": del rows[0]["quantities"]["theta"]
    elif change == "support": rows[0]["quantities"]["theta"]["rank"] = 99
    elif change == "randomization": rows[0]["randomization_seed"] += 1
    elif change == "registration": rows[0]["registration_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        summarize(rows, reg, registration_sha256="a" * 64)


@pytest.fixture
def recovery_archive(archive, generated, tmp_path, monkeypatch):
    """Real archive/checking path, with only the committed-registration I/O mocked.

    Constant draws and the cache-format receipt are explicit software fixtures;
    no fit or scientific convergence is claimed.
    """
    import analysis.overall_recovery_checks as checks
    directory, temporary, _, save = archive
    reg, _ = study_fixture()
    study_id = "connected-23"
    reg["quantities_by_case"]["connected"] = [s["id"] for s in quantity_specs(
        generated["blocks"], generated["catalog"])]
    reg["evaluator_sha256"] = hashlib.sha256(Path(checks.__file__).read_bytes()).hexdigest()
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(reg))
    registration_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    spec = dict(source_key="fixture-source", arrays_key=generated["arrays_key"],
        catalog=generated["catalog"], registration_sha256=registration_sha,
        study=study_id, synthetic=True, settings=reg["fit_settings"])
    run_key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    receipt = save(posterior_fixture(generated), settings=reg["fit_settings"],
        attempts=[dict(samples=3)], catalog=generated["catalog"], specification=spec, run_key=run_key)
    monkeypatch.setattr(checks, "committed_registration", lambda p: json.loads(Path(p).read_text()))
    return directory, path, study_id, copy.deepcopy(receipt)


def test_checked_recovery_consumes_the_registered_archive(generated, recovery_archive):
    from analysis.overall_recovery_checks import evaluate_checked
    directory, path, study_id, receipt = recovery_archive
    result = evaluate_checked(generated, directory, path, study_id=study_id)
    assert result["fit_verified"] and not result["publication_allowed"]
    assert result["run_key"] == receipt["run_key"]
    assert result["registration_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["recovery_intervals"]


@pytest.mark.parametrize("change", ["study", "synthetic", "registration", "catalog", "source", "run_key",
    "settings", "arrays", "evaluator", "quantity", "failed"])
def test_checked_recovery_rejects_foreign_or_failed_fit_before_evaluation(
        generated, recovery_archive, monkeypatch, change):
    import analysis.overall_recovery_checks as checks
    directory, path, study_id, receipt = recovery_archive
    if change in ("evaluator", "quantity"):
        reg = json.loads(path.read_text())
        if change == "evaluator": reg["evaluator_sha256"] = "b" * 64
        else: reg["quantities_by_case"]["connected"].pop(0)
        path.write_text(json.dumps(reg))
    else:
        if change == "study": receipt["specification"]["study"] = "another-study"
        elif change == "synthetic": receipt["specification"]["synthetic"] = False
        elif change == "registration": receipt["specification"]["registration_sha256"] = "b" * 64
        elif change == "catalog": receipt["catalog"]["generation_seed"] += 1
        elif change == "source": receipt["source"]["key"] = "different-source"
        elif change == "run_key": receipt["run_key"] = "different-run"
        elif change == "settings": receipt["settings"]["target_accept"] = .5
        elif change == "arrays": receipt["specification"]["arrays_key"] = "b" * 64
        elif change == "failed": receipt["state"] = "failed_convergence"
        (directory / "receipt.json").write_text(json.dumps(receipt))
    monkeypatch.setattr(checks, "evaluate", lambda *a, **kw: pytest.fail("Foreign fit reached evaluation"))
    with pytest.raises(ValueError):
        checks.evaluate_checked(generated, directory, path, study_id=study_id)


def test_registration_mutation_during_evaluation_cannot_return_success(
        generated, recovery_archive, monkeypatch):
    import analysis.overall_recovery_checks as checks
    directory, path, study_id, _ = recovery_archive
    def evaluate_and_mutate(*args, **kwargs):
        with path.open("a") as out:
            out.write("\n")
        return {}
    monkeypatch.setattr(checks, "evaluate", evaluate_and_mutate)
    with pytest.raises(ValueError, match="changed during evaluation"):
        checks.evaluate_checked(generated, directory, path, study_id=study_id)
