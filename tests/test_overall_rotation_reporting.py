"""Complete matched-study accounting with synthetic draws, never MCMC fits."""
import copy
import json

import numpy as np
import pytest

from analysis import overall_rotation_profile as runner
from f1rank.overall import fitting


@pytest.fixture
def draws():
    rng = np.random.default_rng(81)
    post = {name: rng.normal(size=(4, 128, 5))
            for name in ("skill", "overall_race_skill")}
    catalog = dict(field=["synthetic_z", "synthetic_a", "synthetic_b"],
                   field_entries=[3, 0, 4])
    return post, catalog


def test_complete_contrasts_match_grouped_arithmetic(draws):
    post, catalog = draws
    before = {name: values.copy() for name, values in post.items()}
    report = runner.contrast_reports(post, catalog, 1)
    expected = [(site, i, j) for site in post for i in range(3) for j in range(i)]
    assert len(report["current_driver_contrasts"]) == len(expected)
    for row, (site, i, j) in zip(report["current_driver_contrasts"], expected):
        a, b = catalog["field_entries"][i], catalog["field_entries"][j]
        contrast = post[site][:, :, a] - post[site][:, :, b]
        assert (row["site"], row["driver_i"], row["driver_j"]) == (
            site, catalog["field"][i], catalog["field"][j])
        assert row["diagnostic"] == runner.check({"contrast": contrast}, 1)
        np.testing.assert_array_equal(row["chain_means"], contrast.mean(axis=1))
        np.testing.assert_array_equal(row["chain_standard_deviations"], contrast.std(axis=1, ddof=1))
    for name in post:
        np.testing.assert_array_equal(post[name], before[name])
    assert not report["changes_attempt_convergence_decision"]
    assert not report["publication_allowed"]


@pytest.mark.parametrize("fault", ["driver", "entry", "bounds", "negative", "float", "axes", "nonfinite"])
def test_incomplete_or_invalid_field_diagnostics_are_rejected(draws, fault):
    post, catalog = draws
    if fault == "driver":
        catalog["field"][1] = catalog["field"][0]
    elif fault == "entry":
        catalog["field_entries"][1] = catalog["field_entries"][0]
    elif fault == "bounds":
        catalog["field_entries"][1] = 9
    elif fault == "negative":
        catalog["field_entries"][1] = -1
    elif fault == "float":
        catalog["field_entries"][1] = .5
    elif fault == "axes":
        post["overall_race_skill"] = post["overall_race_skill"][:, :-1]
    else:
        post["skill"][0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        runner.contrast_reports(post, catalog, 0)


@pytest.fixture
def matched(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    blocks = dict(qualifying={"y": np.array([.2])}, laps={"y": np.array([.4])},
                  outcomes={"mask": np.array([True])})
    truth = dict(skill=np.array([.1, .2]), unused_state=np.array([.3]))
    record = dict(case="connected", seed=2, arrays_key=fitting.arrays_digest(blocks),
                  truth={name: value.tolist() for name, value in truth.items()})
    path = tmp_path / "original.truth.json"
    path.write_text(json.dumps(record))
    registration = dict(synthetic={"toy":dict(case="connected", seed=2)},
        original_arrays_sha256={"toy":record["arrays_key"]},
        original_truth_files={"toy":dict(path=path.name, sha256=runner.digest(path))})
    return registration, blocks, truth, path


def test_every_original_truth_is_compared(matched):
    registration, blocks, truth, path = matched
    assert runner.matched_truth_record(registration, "toy", blocks, truth) == json.loads(path.read_text())
    truth["unused_state"][0] += .1
    with pytest.raises(ValueError, match="Generated truths differ"):
        runner.matched_truth_record(registration, "toy", blocks, truth)


@pytest.mark.parametrize("fault", ["observations", "missing_state", "extra_state", "file"])
def test_changed_truth_inputs_are_rejected(matched, fault):
    registration, blocks, truth, path = matched
    if fault == "observations":
        blocks["laps"]["y"][0] += 1
    elif fault == "missing_state":
        truth.pop("unused_state")
    elif fault == "extra_state":
        truth["unexpected"] = np.array([0.])
    else:
        path.write_text('{}')
    with pytest.raises(ValueError):
        runner.matched_truth_record(registration, "toy", blocks, truth)


@pytest.mark.parametrize("finish_successfully", [False, True])
def test_every_retry_preserves_reports_without_changing_the_gate(
        tmp_path, monkeypatch, draws, finish_successfully):
    post, catalog = draws
    registration = dict(schema_version=1, status="frozen", purpose="numerical_profile",
        fit_settings=fitting.FIT_SETTINGS, retry=[list(x) for x in fitting.RETRY], studies=["toy"])
    path = tmp_path / "registration.json"
    path.write_text(json.dumps(registration))
    monkeypatch.setattr(fitting, "committed_registration", lambda _: registration)
    monkeypatch.setattr(fitting, "snapshot", lambda: dict(key="synthetic-fixture", inputs={}))
    monkeypatch.setattr(fitting, "require_snapshot", lambda _: None)
    monkeypatch.setattr(fitting, "require_clean_inputs", lambda _: None)
    monkeypatch.setattr(runner, "require_prototype_sources", lambda _: None)
    monkeypatch.setattr(runner, "check", lambda *args: dict(converged=False, reason="fixture contrast failure"))
    seen, original_diagnostics = [], []
    def fake_sample(blocks, settings, registered, rotation):
        seen.append(settings.copy())
        diagnostic = dict(converged=finish_successfully and len(seen) == 2,
                          divergences=0, parameters=[])
        original_diagnostics.append(diagnostic)
        # A sampling stub exercises receipts only; no chain is actually fitted.
        expanded = {name: np.resize(value, (settings['chains'], settings['samples'], 5))
                    for name, value in post.items()}
        return expanded, diagnostic
    monkeypatch.setattr(runner, "sample_rotated", fake_sample)
    blocks = dict(qualifying={"y":np.array([0.])}, laps={}, outcomes={})
    directory = tmp_path / "fit"
    def execute():
        return fitting.run(blocks, catalog, directory, path, study="toy", synthetic=True,
            sample=lambda b, s: runner.sample_with_reports(b, s, registration, None, catalog))
    if finish_successfully:
        _, receipt = execute()
        assert receipt["state"] == "converged"
    else:
        with pytest.raises(RuntimeError, match="All registered"):
            execute()
        receipt = json.loads((directory / "receipt.json").read_text())
        assert receipt["state"] == "failed_convergence"
    assert len(seen) == (2 if finish_successfully else len(fitting.RETRY))
    for number, (attempt, settings) in enumerate(zip(receipt["attempts"], seen), 1):
        expected = fitting.RETRY[number - 1]
        assert settings["seed"] == fitting.FIT_SETTINGS["seed"] + number - 1
        assert settings["warmup"] == expected[0] * fitting.FIT_SETTINGS["warmup"]
        assert settings["samples"] == expected[1] * fitting.FIT_SETTINGS["samples"]
        assert settings["target_accept"] == (expected[2] or fitting.FIT_SETTINGS["target_accept"])
        diagnostic = attempt["diagnostics"]
        assert diagnostic["converged"] == original_diagnostics[number - 1]["converged"]
        assert "comparison_reports" not in original_diagnostics[number - 1]
        reports = diagnostic["comparison_reports"]
        assert len(reports["current_driver_contrasts"]) == 6
        assert not reports["all_reported_contrasts_passed"]
        assert not reports["changes_attempt_convergence_decision"]
        assert json.loads((directory / f"attempt_{number}.json").read_text()) == attempt
        if not diagnostic["converged"]:
            assert (directory / f"failed_{number}.npz").exists()


def test_proposal_pins_the_original_truths_without_authorizing_a_run():
    proposal = json.loads((runner.ROOT / "docs/overall_rotation_profile_proposal.json").read_text())
    assert not proposal["user_approved"]
    assert proposal["status"] == "proposal_pending_user_approval"
    runner.require_prototype_sources(proposal)
    assert set(proposal["original_truth_files"]) == set(proposal["studies"])
    for study in proposal["studies"]:
        record = proposal["original_truth_files"][study]
        original = json.loads((runner.ROOT / record["path"]).read_text())
        assert original["arrays_key"] == proposal["original_arrays_sha256"][study]
        assert original["case"] == proposal["synthetic"][study]["case"]
        assert original["seed"] == proposal["synthetic"][study]["seed"]


def test_registration_rejects_an_incomplete_truth_inventory(monkeypatch):
    proposal = json.loads((runner.ROOT / "docs/overall_rotation_profile_proposal.json").read_text())
    registration = copy.deepcopy(proposal)
    registration.update(status="frozen", user_approved=True)
    registration["original_truth_files"].pop(registration["studies"][0])
    monkeypatch.setattr(runner, "committed_registration", lambda _: registration)
    with pytest.raises(ValueError, match="complete original truth inventory"):
        runner.checked_registration("synthetic-fixture.json")
