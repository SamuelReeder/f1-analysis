"""Synthetic archive fixtures only; no fitting or scientific acceptance."""
import copy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from analysis import overall_stable_profile_diagnostics as audit
from f1rank.overall.bounded_diagnostics import check
from f1rank.overall.inference import contrast_reports


@pytest.fixture
def saved(tmp_path):
    rng = np.random.default_rng(913)
    settings = dict(chains=4, samples=128, max_tree_depth=10)
    arrays = dict(
        stochastic=np.ones((4, 128), dtype=np.float32),
        skill=rng.normal(size=(4, 128, 3)).astype(np.float32),
        overall_race_skill=rng.normal(size=(4, 128, 3)).astype(np.float32),
        empty=np.empty((4, 128, 0), dtype=np.float32),
    )
    extras = dict(diverging=np.zeros((4, 128), dtype=bool),
                  num_steps=np.ones((4, 128), dtype=np.int32),
                  accept_prob=np.full((4, 128), .73, dtype=np.float32))
    extras["diverging"][0, 0] = True
    extras["num_steps"][1, 0] = 2**settings["max_tree_depth"] - 1
    inventory = {name: dict(shape=list(value.shape[2:]), dtype=value.dtype.str)
                 for name, value in arrays.items()}
    catalog = dict(field=["A", "B", "C"], field_entries=[2, 0, 1])
    diagnostic = check(arrays, int(extras["diverging"].sum()))
    diagnostic.update(
        comparison_reports=contrast_reports(arrays, catalog, diagnostic["divergences"]),
        tree_depth_limit_fraction=float(np.mean(
            extras["num_steps"] >= 2**settings["max_tree_depth"] - 1)),
        mean_accept_prob=float(extras["accept_prob"].mean()), runtime={"fixture": True},
    )
    assert diagnostic["converged"] is False
    archive, sampler = tmp_path / "saved.npz", tmp_path / "sampler.npz"
    np.savez_compressed(archive, **arrays)
    np.savez_compressed(sampler, **extras)
    recorded = dict(settings, diagnostics=diagnostic, sampler_file=sampler.name,
                    sampler_sha256=audit.review.sha(sampler),
                    posterior_sites={name: dict(shape=list(value.shape), dtype=value.dtype.str)
                                     for name, value in arrays.items()})
    temporary = tmp_path / "maps"
    temporary.mkdir()
    return dict(arrays=arrays, extras=extras, inventory=inventory, recorded=recorded,
                catalog=catalog, archive=archive, sampler=sampler, temporary=temporary)


def reproduce(saved):
    return audit.reproduce(saved["archive"], saved["sampler"], saved["inventory"],
                           saved["recorded"], saved["catalog"], saved["temporary"])


def test_failed_fit_reports_and_sampler_extras_are_reproduced(saved):
    result = reproduce(saved)
    assert result["archive_arrays_verified"] and result["sampler_extras_recounted"]
    assert result["coordinate_report_equal"] and result["comparison_reports_equal"]
    assert result["sampler_report_equal"]
    assert result["coordinate_report"]["converged"] is False
    assert result["sampler_report"]["divergences"] == 1
    assert result["other_recorded_fields_not_recomputed"] == ["runtime"]
    assert not list(saved["temporary"].iterdir())


def test_every_numerical_difference_survives_without_tolerances(saved):
    diagnostic = saved["recorded"]["diagnostics"]
    diagnostic["parameters"][0]["rhat_max"] += .1
    diagnostic["comparison_reports"]["current_driver_contrasts"][-1]["chain_means"][-1] += .1
    diagnostic["divergences"] += 1
    diagnostic["mean_accept_prob"] += .1
    diagnostic["tree_depth_limit_fraction"] += .1
    result = reproduce(saved)
    assert not result["coordinate_report_equal"]
    assert not result["comparison_reports_equal"]
    assert not result["sampler_report_equal"]
    assert len(result["sampler_difference_paths"]) == 3
    assert result["coordinate_report"]["divergences"] == 1


@pytest.mark.parametrize("change", [
    "missing", "extra", "duplicate", "axes", "dtype", "steps", "probability", "nonfinite", "object",
])
def test_invalid_sampler_archive_is_rejected(saved, change):
    extras = saved["extras"]
    if change == "missing":
        del extras["diverging"]
    elif change == "extra":
        extras["unknown"] = extras["num_steps"]
    elif change == "axes":
        extras["num_steps"] = extras["num_steps"][:, :-1]
    elif change == "dtype":
        extras["diverging"] = extras["diverging"].astype(np.int32)
    elif change == "steps":
        extras["num_steps"][0, 0] = 0
    elif change == "probability":
        extras["accept_prob"][0, 0] = 1.1
    elif change == "nonfinite":
        extras["accept_prob"][0, 0] = np.nan
    elif change == "object":
        extras["diverging"] = extras["diverging"].astype(object)
    np.savez_compressed(saved["sampler"], **extras)
    if change == "duplicate":
        with zipfile.ZipFile(saved["sampler"], "a") as archive:
            with pytest.warns(UserWarning, match="Duplicate name"):
                archive.writestr("diverging.npy", archive.read("diverging.npy"))
    with pytest.raises(ValueError):
        reproduce(saved)


def test_incomplete_posterior_cannot_be_rescued_by_valid_extras(saved):
    arrays = dict(saved["arrays"])
    del arrays["empty"]
    np.savez_compressed(saved["archive"], **arrays)
    with pytest.raises(ValueError):
        reproduce(saved)


def studies_fixture(monkeypatch, tmp_path, saved):
    from analysis.overall_stable_profile import PREFLIGHT_OUTPUT
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    original = dict(studies=[], issues=[], records_consistent=True,
                    recorded_all_converged=False, terminal_service={"state": "exited"})
    registration = dict(id="fixture", evidence_sha256={})
    for case in ("first", "middle", "last"):
        cache = tmp_path / "outputs/championship/cache/fixture" / case
        cache.mkdir(parents=True)
        inventory = tmp_path / PREFLIGHT_OUTPUT / case / "result.json"
        inventory.parent.mkdir(parents=True)
        inventory.write_text(json.dumps(dict(sites=saved["inventory"])))
        registration["evidence_sha256"][inventory.relative_to(tmp_path).as_posix()] = audit.review.sha(inventory)
        attempts, reviewed = [], []
        for index in (1, 2):
            archive, sampler = cache / f"failed_{index}.npz", cache / f"sampler_{index}.npz"
            archive.write_bytes(saved["archive"].read_bytes())
            sampler.write_bytes(saved["sampler"].read_bytes())
            recorded = copy.deepcopy(saved["recorded"])
            recorded.update(sampler_file=sampler.name, sampler_sha256=audit.review.sha(sampler))
            attempts.append(recorded)
            reviewed.append(dict(
                attempt=index, recorded=recorded, issues=[], archive_present=True,
                archive=archive.relative_to(tmp_path).as_posix(), archive_sha256=audit.review.sha(archive),
                archive_role="unaccepted_diagnostic_only"))
        original["studies"].append(dict(
            case=case, completion={"state": "failed"}, issues=[], attempts=reviewed,
            receipt=dict(state="failed_convergence", catalog=saved["catalog"], attempts=attempts)))
    return original, registration


@pytest.mark.parametrize("change", ["sampler_bytes", "missing_receipt", "earlier_attempt", "inventory_bytes"])
def test_bad_middle_evidence_keeps_all_other_cases_and_attempts(monkeypatch, tmp_path, saved, change):
    original, registration = studies_fixture(monkeypatch, tmp_path, saved)
    middle = original["studies"][1]
    if change == "sampler_bytes":
        (tmp_path / "outputs/championship/cache/fixture/middle/sampler_1.npz").write_bytes(b"changed")
    elif change == "missing_receipt":
        middle.pop("receipt")
        middle["attempts"] = []
    elif change == "earlier_attempt":
        middle["attempts"][0]["issues"].append("Original attempt is inconsistent")
        middle["issues"].append("attempt 1: Original attempt is inconsistent")
    else:
        name = next(name for name in registration["evidence_sha256"] if "/middle/" in name)
        (tmp_path / name).write_text("{}")
    rows = audit.verify_studies(original, registration, saved["temporary"], lambda _: None)
    assert [row["case"] for row in rows] == ["first", "middle", "last"]
    assert all(attempt["state"] == "verified" for row in (rows[0], rows[-1]) for attempt in row["attempts"])
    assert not audit.common.all_reproduced(rows)
    if change in ("sampler_bytes", "earlier_attempt"):
        assert [attempt["state"] for attempt in rows[1]["attempts"]] == ["failed", "verified"]
    assert all(row["original_completion"]["state"] == "failed" for row in rows)


@pytest.mark.parametrize("changed_inventory", [False, True])
def test_audit_preserves_failure_and_rechecks_complete_evidence(monkeypatch, tmp_path, saved, changed_inventory):
    original, registration = studies_fixture(monkeypatch, tmp_path, saved)
    calls = []
    def evidence(_):
        calls.append(True)
        if len(calls) > 1 and changed_inventory:
            raise ValueError("Committed review no longer matches complete terminal evidence")
        return original, registration, {}
    monkeypatch.setattr(audit, "evidence", evidence)
    monkeypatch.setattr(audit.common, "actual_scope", lambda _: {"fixture": True})
    monkeypatch.setattr(audit.subprocess, "check_output", lambda *a, **kw: "fixture-commit\n")
    result = audit.run(tmp_path / "review.json", tmp_path / "audit", "fixture.scope")
    assert result["all_processed"]
    assert len(calls) == 2
    assert result["all_recorded_reports_reproduced"] is (not changed_inventory)
    assert result["all_sampler_extras_recounted"] is (not changed_inventory)
    assert result["recorded_all_converged"] is False
    for key in ("mcmc_run", "scientific_acceptance_established", "full_recovery_complete",
                "sbc_complete", "publication_allowed", "full_goal_complete"):
        assert result[key] is False
    assert json.loads((tmp_path / "audit/audit.json").read_text()) == result
    with pytest.raises(FileExistsError):
        audit.run(tmp_path / "review.json", tmp_path / "audit", "fixture.scope")


def test_evidence_replays_review_and_rejects_changed_stable_fields(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    monkeypatch.setattr(audit, "SOURCES", ())
    monkeypatch.setattr(audit.review.original, "committed", lambda _: None)
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps({"id": "fixture"}))
    current = dict(registration=registration.name, input_sha256={}, studies=[],
                   recorded_at="now", terminal_service={"state": "absent"})
    saved = dict(current, recorded_at="earlier", terminal_service={"state": "exited"})
    path = tmp_path / "review.json"
    path.write_text(json.dumps(saved))
    monkeypatch.setattr(audit.review, "collect", lambda _: current)
    assert audit.evidence(path)[1] == {"id": "fixture"}
    current["studies"].append({"case": "unexpected"})
    with pytest.raises(ValueError, match="no longer matches"):
        audit.evidence(path)
