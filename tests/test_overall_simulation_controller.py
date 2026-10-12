"""Synthetic cache-format and schedule checks, never real fit acceptance."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import overall_simulation_controller as controller
from analysis.overall_ranking_recovery import SITES
from analysis.overall_summary import OverallSummary
from f1rank.overall import fitting
from f1rank.overall.design import build
from f1rank.overall.simulation import JointDraws
from test_overall import tables, draws
from test_overall_summary import compare_nested


def commit_protocol(root, protocol):
    path = root / "protocol.json"
    path.write_text(json.dumps(protocol, indent=2) + "\n")
    subprocess.run(["git", "add", "protocol.json"], cwd=root, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Test fixture", "-c", "user.email=fixture@localhost",
        "commit", "-qm", "Freeze synthetic controller fixture", "-m",
        "Co-Authored-By: Codex <noreply@openai.com>"], cwd=root, check=True, capture_output=True)
    return path


@pytest.fixture
def case(tmp_path, monkeypatch):
    source, clean = tables()
    design = build("2022-02", tables=source, clean=clean)
    shape = (2, 5)
    q = design.qualifying.arrays()
    joint = draws(n=q["n_entries"], nd=np.prod(shape))
    posterior = {site: np.asarray(getattr(joint, name)).reshape(shape + getattr(joint, name).shape[1:]).copy()
                 for name, site in SITES.items()}
    for site in ("overall_grid", "overall_pace", "overall_hazard"):
        posterior[site] = np.repeat(posterior[site], len(design.outcomes["context_circuit"]), axis=-1)
    posterior["skill"] = np.arange(np.prod(shape) * q["n_entries"]).reshape(shape + (q["n_entries"],)) / 50
    posterior["overall_race_skill"] = -posterior["skill"] / 2
    posterior["unused"] = np.ones(shape + (3,))
    run = tmp_path / "fit"
    run.mkdir()
    repository = tmp_path / "registration"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True, capture_output=True)
    source_key = dict(key="a" * 64, inputs={})
    def require(expected):
        if expected != source_key:
            raise ValueError("Synthetic cache source changed")
    monkeypatch.setattr(fitting, "require_snapshot", require)
    identity = controller.design_identity(design)
    reference = dict(schema_version=1, scenario_id="synthetic-fixed-reference",
        future_driver_development=False, sprint_points=False, forecast_of_actual_future_calendar=False,
        unresolved_before_publication=["synthetic fixture only; completion and calibration unresolved"],
        events=[dict(event_id="2021-22", circuit_id=design.catalog["circuits"][0],
            wet=False, neutralised=False, reference_distance_laps=4, weight=1.)])
    reference_path = tmp_path / "reference.json"
    reference_path.write_text(json.dumps(reference, indent=2) + "\n")
    protocol = dict(schema_version=1, status="frozen", purpose=controller.PURPOSE,
        model_version=controller.CONTRACT["model_version"], target=controller.CONTRACT["target"],
        runtime=controller.runtime_identity(),
        sources=controller.source_fingerprints(), reference_sha256=controller.digest(reference_path),
        fit={}, posterior_shape=list(shape), workload=dict(seed=29, replicates=5,
            draw_batch_size=3, replicate_batch_size=3, stream_policy=controller.STREAM_POLICY),
        advancing=[4, 2], race_control={"2021-22":4})
    def save(*, synthetic=True, state="converged", catalog=None):
        path = run / "posterior.npz"
        np.savez_compressed(path, **posterior)
        specification = dict(source_key=source_key["key"], arrays_key=identity["arrays_key"],
            catalog=copy.deepcopy(design.catalog if catalog is None else catalog), synthetic=synthetic,
            settings=dict(chains=shape[0], samples=shape[1]), study="cache_format_fixture_only")
        key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
        receipt = dict(state=state, diagnostics=dict(converged=state == "converged"), source=source_key,
            specification=specification, catalog=copy.deepcopy(specification["catalog"]), run_key=key,
            settings=dict(chains=shape[0]), attempts=[dict(samples=shape[1])],
            posterior_file=path.name, posterior_sha256=controller.digest(path), publication_allowed=False)
        (run / "receipt.json").write_text(json.dumps(receipt))
        protocol["fit"] = dict(run_key=key, receipt_sha256=controller.digest(run / "receipt.json"),
            posterior_sha256=receipt["posterior_sha256"], arrays_key=identity["arrays_key"], catalog_key=identity["catalog_key"])
        commit_protocol(repository, protocol)
        return receipt
    receipt = save()
    return SimpleNamespace(design=design, posterior=posterior, protocol=protocol, receipt=receipt,
        run=run, root=repository, path=repository / "protocol.json", reference=reference,
        reference_path=reference_path, output=tmp_path / "simulation", maps=tmp_path / "maps", save=save)


def execute(case):
    case.maps.mkdir(exist_ok=True)
    return controller.execute(case.run, case.design, case.path, case.reference_path, case.output,
                              registration_root=case.root, temporary_parent=case.maps)


@pytest.mark.parametrize("total,maximum,expected", [(2, 2, [[0, 2]]), (5, 3, [[0, 3], [3, 5]]),
    (7, 3, [[0, 3], [3, 5], [5, 7]]), (129, 128, [[0, 127], [127, 129]])])
def test_partitions_retain_every_draw_without_singletons(total, maximum, expected):
    actual = controller.block_ranges(total, maximum)
    assert actual == expected
    assert [i for a, b in actual for i in range(a, b)] == list(range(total))
    assert all(2 <= b - a <= maximum for a, b in actual)


@pytest.mark.parametrize("total,maximum", [(3, 2), (1, 3), (5, 1), (True, 3)])
def test_impossible_partition_never_drops_or_duplicates_a_sample(total, maximum):
    with pytest.raises(ValueError):
        controller.block_ranges(total, maximum)


def test_named_contexts_follow_the_design_axis_instead_of_a_borrowed_index():
    triples = [(1, 1, 1), (0, 0, 0), (1, 0, 0), (0, 1, 0),
               (1, 0, 1), (0, 0, 1), (1, 1, 0), (0, 1, 1)]
    design = SimpleNamespace(catalog=dict(circuits=["a", "b"], cutoff="2026-01"),
        outcomes=dict(n_circuits=2, **{name:np.array([t[i] for t in triples]) for i, name in
            enumerate(("context_circuit", "context_wet", "context_sc"))}))
    reference = dict(schema_version=1, scenario_id="synthetic", future_driver_development=False,
        sprint_points=False, forecast_of_actual_future_calendar=False, unresolved_before_publication=[],
        events=[dict(event_id="2025-01", circuit_id="b", wet=True, neutralised=True,
            reference_distance_laps=5, weight=1.)])
    result = controller.reference_scenarios(reference, design, {"2025-01":4})
    assert result[0].context == 0
    assert result[0].consecutive_racing_laps == 4


def test_checked_archive_full_schedule_matches_direct_paired_summary(case, monkeypatch):
    calls = []
    generator = controller.paired_simulation_block
    def inspect(joint, scenarios, **kwargs):
        chain, start = kwargs["chain"], kwargs["draw_start"]
        stop = start + len(joint.qualifying)
        indices = case.design.catalog["field_entries"]
        for destination, site in SITES.items():
            expected = case.posterior[site][chain, start:stop]
            if destination in ("qualifying", "race"):
                expected = expected[:, indices]
            np.testing.assert_array_equal(getattr(joint, destination), expected)
        result, probes = generator(joint, scenarios, **kwargs)
        calls.append((copy.deepcopy(kwargs), copy.deepcopy(result), probes.copy()))
        return result, probes
    monkeypatch.setattr(controller, "paired_simulation_block", inspect)
    result, receipt = execute(case)
    assert receipt["state"] == "complete" and receipt["completed_blocks"] == receipt["schedule"]["blocks"]
    assert receipt["runtime"] == result["simulation_runtime"] == controller.runtime_identity()
    addresses = [(k["seed"], k["chain"], k["draw_start"], k["replicate_start"]) for k, _, _ in calls]
    assert len(set(addresses)) == len(addresses) == 8
    assert not list(case.maps.iterdir())
    assert not receipt["conditional_simulation_precision_passed"]
    assert not result["publication_allowed"] and not result["scientific_evidence_verified"]
    assert result["unresolved_reference_assumptions"] == case.reference["unresolved_before_publication"]
    direct = OverallSummary(drivers=case.design.catalog["field"], posterior_shape=[2, 5],
        replicates=5, n_races=1, identity=result["identity"])
    for kwargs, baseline, probes in calls:
        direct.add(baseline, probes, chain=kwargs["chain"], draw_start=kwargs["draw_start"],
                   replicate_start=kwargs["replicate_start"], identity=result["identity"])
    expected = direct.finalize()
    compare_nested({k:result[k] for k in expected}, expected)
    from analysis.overall_simulation_moments import load_checked
    metadata, moments = load_checked(case.output, expected_simulation_key=receipt["simulation_key"],
        expected_archive_sha256=result["diagnostic_moments"]["sha256"])
    assert metadata == receipt["diagnostic_moments"] == result["diagnostic_moments"]
    for name, value in direct.grouped_moments()[1].items():
        np.testing.assert_allclose(moments[name], value, atol=1e-13)
        assert not moments[name].flags.writeable
    assert controller.digest(case.output / "summary.json") == receipt["summary_sha256"]
    assert json.loads((case.output / "receipt.json").read_text()) == receipt
    assert json.loads((case.output / "summary.json").read_text()) == result
    with pytest.raises(FileExistsError):
        execute(case)


def test_failure_preserves_progress_and_does_not_create_partial_summary(case, monkeypatch):
    generator = controller.paired_simulation_block
    calls = 0
    def fail(joint, scenarios, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Synthetic interrupted block")
        return generator(joint, scenarios, **kwargs)
    monkeypatch.setattr(controller, "paired_simulation_block", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        execute(case)
    receipt = json.loads((case.output / "receipt.json").read_text())
    assert receipt["state"] == "failed_exception" and receipt["completed_blocks"] == 1
    assert "Synthetic interrupted block" in receipt["traceback"]
    assert not (case.output / "summary.json").exists() and not list(case.maps.iterdir())
    assert not (case.output / "diagnostic_moments.npz").exists()
    with pytest.raises(FileExistsError):
        execute(case)
    assert calls == 2


@pytest.mark.parametrize("change", ["protocol", "reference", "catalog", "receipt", "archive"])
def test_mutated_inputs_cannot_produce_a_completed_summary(case, monkeypatch, change):
    generator = controller.paired_simulation_block
    altered = False
    def mutate(joint, scenarios, **kwargs):
        nonlocal altered
        result = generator(joint, scenarios, **kwargs)
        if not altered:
            if change == "protocol": case.path.write_text(case.path.read_text() + "\n")
            elif change == "reference": case.reference_path.write_text(case.reference_path.read_text() + "\n")
            elif change == "catalog": case.design.catalog["mutated"] = True
            else:
                path = case.run / ("receipt.json" if change == "receipt" else "posterior.npz")
                with path.open("ab") as handle: handle.write(b"\n")
            altered = True
        return result
    monkeypatch.setattr(controller, "paired_simulation_block", mutate)
    with pytest.raises(ValueError):
        execute(case)
    assert json.loads((case.output / "receipt.json").read_text())["state"] == "failed_exception"
    assert not (case.output / "summary.json").exists() and not list(case.maps.iterdir())
    assert not (case.output / "diagnostic_moments.npz").exists()


@pytest.mark.parametrize("change", ["not_committed", "source", "target", "field_mapping", "shape", "partition",
    "reference", "future_reference", "unknown_circuit", "weight", "race_control", "qualifying_format", "stream_word", "runtime"])
def test_invalid_frozen_inputs_stop_before_any_simulation(case, monkeypatch, change):
    if change == "not_committed":
        case.path.write_text(case.path.read_text() + "\n")
    elif change == "source": case.protocol["sources"]["analysis/overall_summary.py"] = "0" * 64
    elif change == "target": case.protocol["target"] = "current_team"
    elif change == "runtime": case.protocol["runtime"]["numpy"] = "different-version"
    elif change == "field_mapping": case.design.catalog["field_entries"].reverse()
    elif change == "shape": case.protocol["posterior_shape"] = [1, 5]
    elif change == "partition": case.protocol["workload"]["draw_batch_size"] = 2
    elif change == "stream_word": case.protocol["workload"]["seed"] = controller.MAX_STREAM_WORD + 1
    elif change == "reference": case.protocol["reference_sha256"] = "0" * 64
    elif change in ("future_reference", "unknown_circuit", "weight"):
        row = case.reference["events"][0]
        if change == "future_reference":
            row["event_id"] = "2026-01"
            case.protocol["race_control"] = {"2026-01":4}
        elif change == "unknown_circuit": row["circuit_id"] = "not_seen"
        else: row["weight"] = .5
        case.reference_path.write_text(json.dumps(case.reference))
        case.protocol["reference_sha256"] = controller.digest(case.reference_path)
    elif change == "race_control": case.protocol["race_control"] = {}
    elif change == "qualifying_format": case.protocol["advancing"] = [6, 2]
    if change not in ("not_committed", "field_mapping"):
        commit_protocol(case.root, case.protocol)
    monkeypatch.setattr(controller, "paired_simulation_block", lambda *a, **k: pytest.fail("Unexpected simulation"))
    with pytest.raises(ValueError):
        execute(case)
    assert not case.output.exists()


@pytest.mark.parametrize("change", ["failed_fit", "real_fit", "wrong_catalog", "missing_site", "wrong_site_shape", "unused_nan"])
def test_invalid_or_real_archive_never_enters_development_simulation(case, monkeypatch, change):
    if change == "failed_fit": case.save(state="failed_convergence")
    elif change == "real_fit": case.save(synthetic=False)
    elif change == "wrong_catalog": case.save(catalog={**case.design.catalog, "field":list(reversed(case.design.catalog["field"]))})
    else:
        if change == "missing_site": del case.posterior["skill"]
        elif change == "wrong_site_shape": case.posterior["overall_pace"] = case.posterior["overall_pace"][..., :1]
        else: case.posterior["unused"][0, 0, 0] = np.nan
        case.save()
    monkeypatch.setattr(controller, "paired_simulation_block", lambda *a, **k: pytest.fail("Unexpected simulation"))
    with pytest.raises(ValueError):
        execute(case)
    assert not (case.output / "summary.json").exists()
    assert json.loads((case.output / "receipt.json").read_text())["state"] == "failed_exception"
    assert not list(case.maps.iterdir())
