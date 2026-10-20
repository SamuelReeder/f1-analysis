"""Candidate execution with invented archives, never driver/scientific evidence."""
import ast
import copy
import functools
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from f1rank.overall import fitting, inference, simulation_controller as controller
from test_overall_simulation_controller import case as old_case
from test_overall_production_inference import definitions

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["posterior_archive", "paired_simulation", "own_points", "summary"])
def test_migrated_calculations_preserve_every_original_definition(name):
    before = definitions(ROOT / "analysis" / f"overall_{name}.py")
    after = definitions(ROOT / "f1rank/overall" / f"{name}.py")
    # Only the summary's artifact role changes; no numerical expression changes.
    after = {k: v.replace("canonical_overall_candidate_summary", "development_canonical_overall_summary")
             for k, v in after.items()}
    assert after == before


def test_all_runtime_dependencies_are_hashed_package_modules():
    from analysis.overall_ranking_recovery import SITES
    assert controller.SITES == SITES
    for name in ("posterior_archive", "paired_simulation", "own_points", "summary", "simulation_moments", "simulation_controller"):
        relative = f"f1rank/overall/{name}.py"
        assert relative in controller.source_fingerprints()
        for node in ast.walk(ast.parse((ROOT / relative).read_text())):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("analysis")


def commit(root, path):
    subprocess.run(["git", "add", path.name], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
        "commit", "-qm", "Freeze synthetic software fixture", "-m",
        "Co-Authored-By: Codex <noreply@openai.com>"], cwd=root, check=True, capture_output=True)


@pytest.fixture
def candidate(old_case, monkeypatch, request):
    case = old_case
    synthetic = getattr(request, "param", True)
    registration = json.loads((ROOT / "docs/overall_profile_protocol.json").read_text())
    study = "cache_format_fixture_only"
    registration.update(schema_version=fitting.COORDINATE_RUN_SCHEMA, studies=[study],
        inference_method=inference.METHOD, registered_model_inputs_key=case.receipt["source"]["key"],
        purpose="numerical_profile" if synthetic else "real_data_acceptance")
    regpath = case.root / "fit-registration.json"
    regpath.write_text(json.dumps(registration))
    commit(case.root, regpath)
    shape = (fitting.FIT_SETTINGS["chains"], fitting.FIT_SETTINGS["samples"])
    # Cache-format arrays only: repeating these invented draws does not make them
    # converged samples. The fixture replaces the sampler and never runs MCMC.
    for name, value in case.posterior.items():
        original = value.reshape((-1,) + value.shape[2:])
        index = np.arange(np.prod(shape)) % len(original)
        case.posterior[name] = original[index].reshape(shape + value.shape[2:])
    source = case.receipt["source"]
    monkeypatch.setattr(fitting, "snapshot", lambda: copy.deepcopy(source))
    monkeypatch.setattr(fitting, "require_clean_inputs", lambda _: None)
    monkeypatch.setattr(fitting, "committed_registration", functools.partial(fitting.committed_registration, root=case.root))
    def sampled(blocks, settings, catalog):
        assert settings == fitting.FIT_SETTINGS and catalog == case.design.catalog
        extras = dict(diverging=np.zeros(shape, bool), num_steps=np.full(shape, 3, dtype=np.int32),
                      accept_prob=np.full(shape, .75, dtype=np.float32))
        diagnostic = dict(converged=True, divergences=0, parameters=[], inference_method=inference.METHOD,
                          mean_accept_prob=.75, tree_depth_limit_fraction=0.)
        return inference.SampleResult(case.posterior, diagnostic, extras)
    monkeypatch.setattr(inference, "sample", sampled)
    case.run = case.run.parent / "production-format-fit"
    blocks = dict(qualifying=case.design.qualifying.arrays(), laps=case.design.laps, outcomes=case.design.outcomes)
    _, case.receipt = fitting.run(blocks, case.design.catalog, case.run, regpath, study=study, synthetic=synthetic)
    case.protocol.update(purpose=controller.PURPOSE, sources=controller.source_fingerprints(),
        synthetic=synthetic, posterior_shape=list(shape),
        fit_registration=dict(path=regpath.name, sha256=controller.digest(regpath)),
        driver_names={d: f"Invented fixture {d}" for d in case.design.catalog["field"]},
        scenario_label="Synthetic fixed scenario")
    case.protocol["workload"].update(draw_batch_size=controller.MAX_BLOCK_DRAWS)
    case.protocol["fit"].update(run_key=case.receipt["run_key"], receipt_sha256=controller.digest(case.run/"receipt.json"),
                                 posterior_sha256=case.receipt["posterior_sha256"])
    def freeze():
        case.path.write_text(json.dumps(case.protocol))
        commit(case.root, case.path)
    case.freeze = freeze
    case.freeze()
    return case


def execute(case):
    case.maps.mkdir(exist_ok=True)
    return controller.execute(case.run, case.design, case.path, case.reference_path, case.output,
                              registration_root=case.root, temporary_parent=case.maps)


def load(case):
    return controller.load_candidate(case.output, case.run, case.design, case.path, case.reference_path,
                                     registration_root=case.root)


def test_complete_registered_candidate_runs_and_failed_precision_stays_unpublished(candidate):
    summary, receipt = execute(candidate)
    assert receipt["state"] == "complete" and receipt["completed_blocks"] == receipt["schedule"]["blocks"]
    assert receipt["conditional_simulation_precision_passed"] is False
    assert receipt["components_sha256"] is None and not (candidate.output / "components.json").exists()
    assert not receipt["publication_allowed"] and not receipt["scientific_evidence_verified"]
    assert (candidate.output/"protocol.json").read_bytes() == candidate.path.read_bytes()
    checked, again, components = load(candidate)
    assert checked == summary and again == receipt and components is None
    assert not list(candidate.maps.iterdir())
    with pytest.raises(FileExistsError): execute(candidate)


@pytest.mark.parametrize("candidate", [False], indirect=True)
def test_registered_real_role_is_consumed_without_granting_acceptance(candidate):
    # The role is exercised using invented arrays and a fake sampler only.
    summary, receipt = execute(candidate)
    assert summary["synthetic"] is False and receipt["synthetic"] is False
    assert summary["scientific_evidence_verified"] is False and receipt["publication_allowed"] is False
    assert load(candidate)[2] is None


@pytest.mark.parametrize("change", ["role", "method", "source", "seed", "missing_attempt", "changed_attempt", "diagnostic"])
def test_registration_and_attempt_mismatches_are_rejected(candidate, change):
    protocol, receipt = copy.deepcopy(candidate.protocol), copy.deepcopy(candidate.receipt)
    if change == "role": protocol["synthetic"] = False
    elif change == "method": receipt["schema_version"] = fitting.RUN_SCHEMA
    elif change == "source": receipt["source"]["key"] = "b" * 64
    elif change == "seed": receipt["specification"]["settings"]["seed"] += 1
    elif change == "missing_attempt": receipt["attempts"] = []
    elif change == "changed_attempt": receipt["attempts"][0]["warmup"] += 1
    else: receipt["attempts"][0]["diagnostics"]["converged"] = False
    with pytest.raises(ValueError):
        controller.require_fit_registration(protocol, receipt, root=candidate.root)


@pytest.mark.parametrize("change", ["summary", "moments", "receipt", "protocol", "reference", "fit", "unexpected_components"])
def test_changed_or_mixed_candidate_files_cannot_be_loaded(candidate, change):
    execute(candidate)
    paths = dict(summary=candidate.output/"summary.json", moments=candidate.output/"diagnostic_moments.npz",
                 receipt=candidate.output/"receipt.json", protocol=candidate.path, reference=candidate.reference_path,
                 fit=candidate.run/"sampler_1.npz", unexpected_components=candidate.output/"components.json")
    path = paths[change]
    if change == "receipt":
        data = json.loads(path.read_text()); data["publication_allowed"] = True
        path.write_text(json.dumps(data))
    elif change == "unexpected_components": path.write_text("{}")
    else:
        with path.open("ab") as handle: handle.write(b"\n")
    with pytest.raises((ValueError, OSError, json.JSONDecodeError)):
        load(candidate)


def test_simulation_exception_retains_progress_and_prevents_partial_components(candidate, monkeypatch):
    generator = controller.paired_simulation_block
    calls = 0
    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2: raise RuntimeError("Synthetic interrupted candidate")
        return generator(*args, **kwargs)
    monkeypatch.setattr(controller, "paired_simulation_block", fail)
    with pytest.raises(RuntimeError, match="interrupted candidate"): execute(candidate)
    receipt = json.loads((candidate.output/"receipt.json").read_text())
    assert receipt["state"] == "failed_exception" and receipt["completed_blocks"] == 1
    assert not (candidate.output/"components.json").exists() and not (candidate.output/"summary.json").exists()
    with pytest.raises(ValueError): load(candidate)


@pytest.mark.parametrize("change", ["moments", "fit_registration", "protocol_copy", "reference_copy"])
def test_mutations_after_their_first_check_are_caught_at_candidate_exit(candidate, monkeypatch, change):
    execute(candidate)
    original = fitting.checked_archive
    paths = dict(moments=candidate.output/"diagnostic_moments.npz",
        fit_registration=candidate.root/candidate.protocol["fit_registration"]["path"],
        protocol_copy=candidate.output/"protocol.json", reference_copy=candidate.output/"reference.json")
    @contextmanager
    def mutate(*args, **kwargs):
        with original(*args, **kwargs) as checked:
            yield checked
        with paths[change].open("ab") as handle: handle.write(b"\n")
    monkeypatch.setattr(fitting, "checked_archive", mutate)
    with pytest.raises(ValueError, match="changed while reading"):
        load(candidate)


@pytest.mark.parametrize("change", ["block_inventory_sha256", "last_completed_block", "reference_source_availability_verified"])
def test_reader_recomputes_inventory_and_retains_unverified_source_status(candidate, change):
    execute(candidate)
    path = candidate.output/"receipt.json"
    receipt = json.loads(path.read_text())
    receipt[change] = {"block_inventory_sha256":"a"*64, "last_completed_block":[0,0,1,0,1],
                       "reference_source_availability_verified": True}[change]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="inconsistent candidate"):
        load(candidate)


def test_component_writer_keeps_successful_paired_statistics_unpublished(tmp_path):
    # Invented export fixture; this is not a simulated or accepted candidate.
    data = json.loads((ROOT/"tests/fixtures/overall_current_contract.json").read_text())
    e = data["evidence"]
    identity = {k: e[k] for k in ("fit_id", "posterior_key", "scenario_key", "simulation_key")}
    summary = dict(model_version=data["model_version"], target=data["target"], identity=identity,
        field=e["field"], posterior_shape=[2, e["posterior_draws"]//2], replicates_per_draw=e["replicates_per_draw"],
        n_races=e["scenario"]["races"], complete_draw_and_replicate_inventory=True,
        expected_points_precision_passed=True, explanation_precision_passed=True, pairwise_points_precision_passed=True,
        standings=data["standings"], contributions=data["contributions"], pairwise=data["pairwise"])
    protocol=dict(driver_names={r["driver_id"]: r["name"] for r in data["standings"]}, scenario_label=e["scenario"]["label"])
    reference=dict(scenario_id=e["scenario"]["id"])
    kwargs=dict(data_as_of=e["data_as_of"], recorded_at=e["recorded_at"])
    fingerprint = controller._export_components(tmp_path, summary, protocol, reference, **kwargs)
    assert fingerprint == controller.digest(tmp_path/"components.json")
    output = json.loads((tmp_path/"components.json").read_text())
    assert output["publication_allowed"] is False and output["scientific_evidence_verified"] is False
    assert "status" not in output and "schema_version" not in output
    assert output["pairwise"] == data["pairwise"]
    with pytest.raises(FileExistsError):
        controller._export_components(tmp_path, summary, protocol, reference, **kwargs)
