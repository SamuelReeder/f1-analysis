"""Canonical candidate simulation controller; publication requires scientific acceptance.

The original research implementation and its evidence remain unchanged.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
import traceback

import numpy as np

from .posterior_archive import mapped_checked_posterior
from .simulation import POSTERIOR_SITES as SITES
from .simulation_moments import write_once as write_moments
from .summary import (CONTRACT, MAX_BLOCK_DRAWS, MAX_BLOCK_REPLICATES, MAX_STREAM_WORD,
                                     OverallSummary, paired_simulation_block)
from f1rank.artifacts import atomic_json, digest
from f1rank.overall.fitting import arrays_digest
from f1rank.overall.simulation import JointDraws, Scenario
from . import fitting
from .inference import METHOD
from .publication import prepare_components
from .simulation_moments import load_checked as load_moments

ROOT = Path(__file__).resolve().parents[2]
SOURCE_NAMES = tuple(sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "f1rank").rglob("*.py"))) + (
    "contracts/overall.json", "docs/overall_candidate.md", "docs/overall_paired_simulation.md",
    "docs/overall_own_points.md", "docs/overall_functional_diagnostics.md",
    "docs/overall_summary.md", "docs/overall_production_simulation.md")
LOADED_PYTHON = {name: digest(ROOT / name) for name in SOURCE_NAMES if name.endswith(".py")}
PURPOSE = "canonical_candidate_simulation"
STREAM_POLICY = "seed_chain_draw_start_replicate_start_v1"


def runtime_identity():
    return dict(python=platform.python_version(), implementation=platform.python_implementation(),
        numpy=np.__version__, bit_generator=type(np.random.default_rng(0).bit_generator).__name__,
        platform=sys.platform, machine=platform.machine(), byteorder=sys.byteorder)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _keys(value, required):
    if not isinstance(value, dict) or set(value) != set(required):
        raise ValueError("Missing or unsupported simulation protocol fields")


def _integer(value, minimum=0):
    return type(value) is int and minimum <= value <= np.iinfo(np.int64).max


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def source_fingerprints():
    current = {p.relative_to(ROOT).as_posix() for p in (ROOT / "f1rank").rglob("*.py")}
    if current != set(LOADED_PYTHON):
        raise ValueError("Simulation source inventory changed after import")
    result = {name: digest(ROOT / name) for name in SOURCE_NAMES}
    if any(result[name] != value for name, value in LOADED_PYTHON.items()):
        raise ValueError("Simulation source changed after import")
    return result


def require_committed_sources(expected):
    for name, fingerprint in expected.items():
        content = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        if hashlib.sha256(content).hexdigest() != fingerprint:
            raise ValueError("Commit simulation sources before executing the registered workload")


def block_ranges(total, maximum):
    """A complete bounded partition; rebalance a trailing singleton."""
    if not _integer(total, 2) or not _integer(maximum, 2):
        raise ValueError("Simulation partitions require positive nonsingleton blocks")
    rows, start = [], 0
    while start < total:
        size = min(maximum, total - start)
        if total - start - size == 1:
            size -= 1
        if size < 2:
            raise ValueError("Declared block size cannot partition this complete inventory")
        rows.append([start, start + size])
        start += size
    return rows


def design_identity(design):
    catalog, q = design.catalog, design.qualifying
    field, entries, circuits = (catalog.get(name) for name in ("field", "field_entries", "circuits"))
    if (not isinstance(field, list) or len(field) < 2 or not all(_text(d) for d in field)
            or len(set(field)) != len(field) or not isinstance(entries, list)
            or len(entries) != len(field) or not all(_integer(i) and i < len(q.entries) for i in entries)
            or len(set(entries)) != len(entries)
            or not isinstance(circuits, list) or not circuits or not all(_text(c) for c in circuits)
            or len(set(circuits)) != len(circuits)
            or not isinstance(catalog.get("cutoff"), str)
            or re.fullmatch(r"[0-9]{4}-[0-9]{2}", catalog["cutoff"]) is None):
        raise ValueError("Design lacks a complete current field, entry mapping or circuit catalog")
    current = q.entries.iloc[entries]
    if current.driver_id.tolist() != field or not (current.event_id == catalog["cutoff"]).all():
        raise ValueError("Current driver identities do not match the selected posterior states")
    q_arrays = q.arrays()
    if q_arrays["n_entries"] != len(q.entries):
        raise ValueError("Qualifying state dimension disagrees with its entry catalog")
    blocks = dict(qualifying=q_arrays, laps=design.laps, outcomes=design.outcomes)
    return dict(arrays_key=arrays_digest(blocks), catalog_key=_hash(catalog),
                field=field.copy(), entries=entries.copy(), n_entries=len(q.entries))


def reference_scenarios(reference, design, race_control):
    """Bind named common conditions to this fit's actual context ordering."""
    if (not isinstance(reference, dict) or type(reference.get("schema_version")) is not int
            or reference["schema_version"] != 1 or not _text(reference.get("scenario_id"))
            or reference.get("future_driver_development") is not False
            or reference.get("sprint_points") is not False
            or reference.get("forecast_of_actual_future_calendar") is not False
            or not isinstance(reference.get("unresolved_before_publication"), list)
            or not all(_text(s) for s in reference["unresolved_before_publication"])):
        raise ValueError("Require the explicit fixed development reference calendar")
    rows = reference.get("events")
    if (not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows)
            or not all(_text(row.get("event_id")) for row in rows)
            or len({row["event_id"] for row in rows}) != len(rows)
            or not isinstance(race_control, dict) or set(race_control) != {row["event_id"] for row in rows}):
        raise ValueError("Every unique reference event needs an explicit race-control assumption")
    circuits, outcomes = design.catalog["circuits"], design.outcomes
    if outcomes.get("n_circuits") != len(circuits):
        raise ValueError("Fitted circuit catalog and context dimensions disagree")
    arrays = [np.asarray(outcomes.get(name)) for name in ("context_circuit", "context_wet", "context_sc")]
    shape = (len(circuits) * 4,)
    if any(a.shape != shape or a.dtype.kind not in "iu" for a in arrays):
        raise ValueError("Fitted contexts need a complete circuit/condition inventory")
    triples = list(zip(*[a.tolist() for a in arrays]))
    expected = {(i, wet, neutral) for i in range(len(circuits)) for wet in (0, 1) for neutral in (0, 1)}
    if len(set(triples)) != len(triples) or set(triples) != expected:
        raise ValueError("Fitted contexts repeat or omit a circuit/condition category")
    lookup = {key: i for i, key in enumerate(triples)}
    result = []
    for row in rows:
        event = row["event_id"]
        if (re.fullmatch(r"[0-9]{4}-[0-9]{2}", event) is None or event.endswith("-00")
                or event > design.catalog["cutoff"] or row.get("circuit_id") not in circuits
                or type(row.get("wet")) is not bool or type(row.get("neutralised")) is not bool
                or not _integer(row.get("reference_distance_laps"), 2)
                or type(row.get("weight")) not in (float, int) or row["weight"] != 1 / len(rows)):
            raise ValueError("Reference event is unknown, beyond cutoff or inconsistent with the equal-weight scenario")
        laps, control = row["reference_distance_laps"], race_control[event]
        if not _integer(control) or control > laps:
            raise ValueError("Invalid explicit reference race-control assumption")
        key = (circuits.index(row["circuit_id"]), int(row["wet"]), int(row["neutralised"]))
        result.append(Scenario(event, lookup[key], laps, control))
    return result


def committed_protocol(path, *, root=ROOT):
    path, root = Path(path).resolve(), Path(root).resolve()
    relative = path.relative_to(root).as_posix()
    committed = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root)
    if committed != path.read_bytes():
        raise ValueError("Commit the unchanged simulation protocol before execution")
    return json.loads(committed), hashlib.sha256(committed).hexdigest()


def validate_protocol(protocol, reference, reference_sha256, design):
    _keys(protocol, ("schema_version", "status", "purpose", "model_version", "target", "sources", "runtime",
        "reference_sha256", "fit", "posterior_shape", "workload", "advancing", "race_control",
        "fit_registration", "synthetic", "driver_names", "scenario_label"))
    if (type(protocol["schema_version"]) is not int or protocol["schema_version"] != 1
            or protocol["status"] != "frozen" or protocol["purpose"] != PURPOSE
            or protocol["model_version"] != CONTRACT["model_version"] or protocol["target"] != CONTRACT["target"]
            or protocol["sources"] != source_fingerprints()
            or protocol["runtime"] != runtime_identity()
            or not _sha(reference_sha256) or protocol["reference_sha256"] != reference_sha256):
        raise ValueError("Simulation protocol, reference or registered sources do not match")
    registration = protocol["fit_registration"]
    _keys(registration, ("path", "sha256"))
    path = registration["path"]
    if (type(protocol["synthetic"]) is not bool or not _text(path) or Path(path).is_absolute()
            or ".." in Path(path).parts or not _sha(registration["sha256"])
            or not _text(protocol["scenario_label"])):
        raise ValueError("Bind the fit registration, provenance role and scenario label")
    fit = protocol["fit"]
    _keys(fit, ("run_key", "receipt_sha256", "posterior_sha256", "arrays_key", "catalog_key"))
    if not all(_sha(value) for value in fit.values()):
        raise ValueError("Fit identities must be complete SHA-256 values")
    identity = design_identity(design)
    names = protocol["driver_names"]
    if not isinstance(names, dict) or set(names) != set(identity["field"]) or not all(_text(v) for v in names.values()):
        raise ValueError("Bind names for the complete current field")
    if any(fit[name] != identity[name] for name in ("arrays_key", "catalog_key")):
        raise ValueError("Simulation design differs from the registered fit arrays or catalog")
    shape, workload = protocol["posterior_shape"], protocol["workload"]
    _keys(workload, ("seed", "replicates", "draw_batch_size", "replicate_batch_size", "stream_policy"))
    if (not isinstance(shape, list) or len(shape) != 2
            or not all(_integer(n, 2) and n <= MAX_STREAM_WORD for n in shape)
            or not _integer(workload["seed"]) or workload["seed"] > MAX_STREAM_WORD
            or not _integer(workload["replicates"], 2) or workload["replicates"] > MAX_STREAM_WORD
            or workload["stream_policy"] != STREAM_POLICY
            or not _integer(workload["draw_batch_size"], 2) or workload["draw_batch_size"] > MAX_BLOCK_DRAWS
            or not _integer(workload["replicate_batch_size"], 2) or workload["replicate_batch_size"] > MAX_BLOCK_REPLICATES):
        raise ValueError("Invalid grouped draw axes or fixed simulation workload")
    draws = block_ranges(shape[1], workload["draw_batch_size"])
    replicates = block_ranges(workload["replicates"], workload["replicate_batch_size"])
    advancing = protocol["advancing"]
    if (not isinstance(advancing, list) or not advancing
            or any(not _integer(a, 1) or a >= b for a, b in zip(advancing, [len(identity["field"]), *advancing]))):
        raise ValueError("Require the explicit qualifying format for the complete field")
    scenarios = reference_scenarios(reference, design, protocol["race_control"])
    schedule = dict(chains=list(range(shape[0])), draw_ranges=draws, replicate_ranges=replicates,
                    blocks=shape[0] * len(draws) * len(replicates), stream_policy=STREAM_POLICY)
    return identity, scenarios, schedule


def require_fit_registration(protocol, receipt, *, root=ROOT):
    """Authenticate how the fit was executed; scientific acceptance is separate."""
    spec = protocol["fit_registration"]
    root = Path(root).resolve()
    path = (root / spec["path"]).resolve()
    path.relative_to(root)
    registered = fitting.committed_registration(path, root=root)
    if digest(path) != spec["sha256"]:
        raise ValueError("Fit registration changed")
    s = receipt["specification"]
    study = s.get("study")
    expected_purpose = ("numerical_profile", "recovery_calibration") if protocol["synthetic"] else ("real_data_acceptance",)
    if (registered["schema_version"] != fitting.COORDINATE_RUN_SCHEMA
            or registered["inference_method"] != METHOD
            or registered.get("purpose") not in expected_purpose
            or study not in registered["studies"]
            or receipt.get("schema_version") != fitting.COORDINATE_RUN_SCHEMA
            or s.get("registration_sha256") != spec["sha256"]
            or registered["registered_model_inputs_key"] != receipt["source"]["key"]
            or s.get("synthetic") is not protocol["synthetic"]
            or receipt.get("publication_allowed") is not False
            or s.get("settings") != fitting.study_settings(registered, study)
            or receipt.get("settings") != s["settings"]):
        raise ValueError("Fit does not match its registered implementation, role or source")
    attempts = receipt.get("attempts")
    if not isinstance(attempts, list) or not 1 <= len(attempts) <= len(fitting.RETRY):
        raise ValueError("Require the complete fixed attempt inventory")
    seeds = fitting.sampling_seeds(registered)[study]
    for i, attempt in enumerate(attempts):
        warmup, samples, target = fitting.RETRY[i]
        expected = dict(s["settings"], warmup=warmup * fitting.FIT_SETTINGS["warmup"],
            samples=samples * fitting.FIT_SETTINGS["samples"], seed=seeds[i],
            target_accept=target if target is not None else fitting.FIT_SETTINGS["target_accept"])
        if (any(attempt.get(k) != v for k, v in expected.items())
                or attempt.get("diagnostics", {}).get("converged") is not (i == len(attempts)-1)
                or attempt.get("diagnostics", {}).get("inference_method") != METHOD):
            raise ValueError("Fit attempts disagree with the registered execution")
    if receipt.get("diagnostics") != attempts[-1]["diagnostics"]:
        raise ValueError("Fit final diagnostic record disagrees with its attempts")
    return registered


def _joint_block(posterior, identity, chain, first, last):
    values = {}
    for destination, site in SITES.items():
        value = np.asarray(posterior[site][chain, first:last])
        if destination in ("qualifying", "race"):
            value = value[:, identity["entries"]]
        values[destination] = value
    return JointDraws(tuple(identity["field"]), **values)


def _export_components(output_dir, summary, protocol, reference, *, data_as_of, recorded_at):
    """Write portable candidate statistics only when their fixed precision passes."""
    if not all(summary.get(name) is True for name in (
            "expected_points_precision_passed", "explanation_precision_passed", "pairwise_points_precision_passed")):
        return None
    components = prepare_components(summary, names=protocol["driver_names"],
        scenario=dict(id=reference["scenario_id"], label=protocol["scenario_label"], races=summary["n_races"]),
        data_as_of=data_as_of, recorded_at=recorded_at)
    path = Path(output_dir) / "components.json"
    if path.exists():
        raise FileExistsError("Preserve existing candidate components")
    atomic_json(path, components)
    return digest(path)


def _block_inventory(schedule):
    inventory, last_block = hashlib.sha256(), None
    for chain in schedule["chains"]:
        for first, last in schedule["draw_ranges"]:
            for rep_first, rep_last in schedule["replicate_ranges"]:
                last_block = [chain, first, last, rep_first, rep_last]
                inventory.update(json.dumps(last_block, separators=(",", ":")).encode() + b"\n")
    return inventory.hexdigest(), last_block


def execute(run_dir, design, protocol_path, reference_path, output_dir, *, registration_root=ROOT,
            temporary_parent=None):
    """Build an unpublished candidate from a registered fit; never grant acceptance."""
    protocol_path, reference_path = Path(protocol_path), Path(reference_path)
    output_dir, run_dir = Path(output_dir), Path(run_dir)
    if output_dir.exists():
        raise FileExistsError("Preserve the existing simulation execution, even if incomplete or failed")
    protocol, protocol_key = committed_protocol(protocol_path, root=registration_root)
    protocol_bytes = protocol_path.read_bytes()
    if hashlib.sha256(protocol_bytes).hexdigest() != protocol_key:
        raise ValueError("Simulation protocol changed after its committed read")
    reference_bytes = reference_path.read_bytes()
    reference = json.loads(reference_bytes)
    reference_key = hashlib.sha256(reference_bytes).hexdigest()
    identity, scenarios, schedule = validate_protocol(protocol, reference, reference_key, design)
    require_committed_sources(protocol["sources"])
    scenario_key = _hash(dict(reference_sha256=reference_key, race_control=protocol["race_control"],
                             advancing=protocol["advancing"]))
    simulation_key = _hash(dict(protocol_sha256=protocol_key, scenario_key=scenario_key, schedule=schedule))
    shared = dict(fit_id=protocol["fit"]["run_key"], posterior_key=protocol["fit"]["posterior_sha256"],
                  scenario_key=scenario_key, simulation_key=simulation_key)
    receipt = dict(schema_version=1, purpose=PURPOSE, state="running", simulation_key=simulation_key,
        protocol_sha256=protocol_key, reference_sha256=reference_key, fit=protocol["fit"],
        schedule=schedule, completed_blocks=0, last_completed_block=None,
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(), synthetic=protocol["synthetic"],
        conditional_simulation_precision_passed=False, scientific_evidence_verified=False,
        reference_source_availability_verified=False, publication_allowed=False)
    receipt["runtime"] = protocol["runtime"]
    receipt.update(source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        registration_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=registration_root, text=True).strip())
    output_dir.mkdir(parents=True, exist_ok=False)
    (output_dir / "protocol.json").write_bytes(protocol_bytes)
    (output_dir / "reference.json").write_bytes(reference_bytes)
    atomic_json(output_dir / "receipt.json", receipt)
    started, inventory = time.monotonic(), hashlib.sha256()
    def require_unchanged_inputs():
        if (digest(protocol_path) != protocol_key or reference_path.read_bytes() != reference_bytes
                or protocol["sources"] != source_fingerprints() or design_identity(design) != identity
                or protocol["runtime"] != runtime_identity()
                or digest(Path(registration_root) / protocol["fit_registration"]["path"]) != protocol["fit_registration"]["sha256"]
                or digest(run_dir / "receipt.json") != protocol["fit"]["receipt_sha256"]
                or receipt["completed_blocks"] != schedule["blocks"]):
            raise ValueError("Simulation inputs or complete schedule changed during execution")
    try:
        fit = protocol["fit"]
        if digest(run_dir / "receipt.json") != fit["receipt_sha256"]:
            raise ValueError("Fit receipt differs from the registered artifact")
        with mapped_checked_posterior(run_dir, names=list(SITES.values()), expected_key=fit["run_key"],
                expected_arrays_key=identity["arrays_key"], temporary_parent=temporary_parent) as (posterior, fitted):
            specification = fitted["specification"]
            require_fit_registration(protocol, fitted, root=registration_root)
            if (fitted.get("catalog") != design.catalog or specification.get("catalog") != design.catalog
                    or specification.get("source_key") != fitted["source"]["key"]
                    or fitted.get("posterior_sha256") != fit["posterior_sha256"]
                    or hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest() != fit["run_key"]):
                raise ValueError("Fit catalog, archive or specification does not match its registered identity")
            shape = tuple(protocol["posterior_shape"])
            contexts = len(design.outcomes["context_circuit"])
            for destination, site in SITES.items():
                tail = ((identity["n_entries"],) if destination in ("qualifying", "race") else
                        (contexts,) if destination in ("grid_coefficient", "pace_coefficient", "retirement_hazard") else ())
                if posterior[site].shape != shape + tail or posterior[site].dtype.kind not in "fiu":
                    raise ValueError("Simulation posterior sites have incompatible state or draw axes")
            summary = OverallSummary(drivers=identity["field"], posterior_shape=list(shape),
                replicates=protocol["workload"]["replicates"], n_races=len(scenarios), identity=shared)
            for chain in schedule["chains"]:
                for first, last in schedule["draw_ranges"]:
                    joint = _joint_block(posterior, identity, chain, first, last)
                    for rep_first, rep_last in schedule["replicate_ranges"]:
                        baseline, probes = paired_simulation_block(joint, scenarios,
                            replicates=rep_last - rep_first, seed=protocol["workload"]["seed"], chain=chain,
                            draw_start=first, replicate_start=rep_first, advancing=protocol["advancing"])
                        summary.add(baseline, probes, chain=chain, draw_start=first,
                                    replicate_start=rep_first, identity=shared)
                        block = [chain, first, last, rep_first, rep_last]
                        inventory.update(json.dumps(block, separators=(",", ":")).encode() + b"\n")
                        receipt.update(completed_blocks=receipt["completed_blocks"] + 1,
                            last_completed_block=block, block_inventory_sha256=inventory.hexdigest())
                        atomic_json(output_dir / "receipt.json", receipt)
            result = summary.finalize()
            moment_metadata, moment_arrays = summary.grouped_moments()
        # A result stays private until the archive's exit checks have passed.
        require_unchanged_inputs()
        result.update(controller_sources=protocol["sources"], protocol_sha256=protocol_key,
            simulation_runtime=protocol["runtime"],
            reference_scenario_id=reference["scenario_id"],
            unresolved_reference_assumptions=reference["unresolved_before_publication"],
            synthetic=protocol["synthetic"], reference_source_availability_verified=False)
        result["diagnostic_moments"] = write_moments(output_dir, moment_metadata, moment_arrays)
        receipt["diagnostic_moments"] = result["diagnostic_moments"]
        require_unchanged_inputs()
        atomic_json(output_dir / "summary.json", result)
        receipt.update(state="complete", summary_sha256=digest(output_dir / "summary.json"),
            conditional_simulation_precision_passed=all(result[name] for name in (
                "expected_points_precision_passed", "explanation_precision_passed", "pairwise_points_precision_passed")))
        require_unchanged_inputs()
        receipt["components_sha256"] = _export_components(output_dir, result, protocol, reference,
            data_as_of=design.catalog["cutoff"], recorded_at=receipt["started_at"])
        require_unchanged_inputs()
    except BaseException as exc:
        receipt.update(state="failed_exception", exception=type(exc).__name__, message=str(exc),
                       traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(seconds=time.monotonic() - started,
                       finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(output_dir / "receipt.json", receipt)
    return result, receipt


def load_candidate(output_dir, run_dir, design, protocol_path, reference_path, *, registration_root=ROOT):
    """Recheck candidate artifacts and dependencies; return no publication decision."""
    output_dir, run_dir = Path(output_dir), Path(run_dir)
    receipt_path = output_dir / "receipt.json"
    receipt_bytes = receipt_path.read_bytes()
    receipt = json.loads(receipt_bytes)
    protocol, key = committed_protocol(protocol_path, root=registration_root)
    reference_bytes = Path(reference_path).read_bytes()
    reference = json.loads(reference_bytes)
    reference_key = hashlib.sha256(reference_bytes).hexdigest()
    identity, scenarios, schedule = validate_protocol(protocol, reference, reference_key, design)
    require_committed_sources(protocol["sources"])
    scenario_key = _hash(dict(reference_sha256=reference_key, race_control=protocol["race_control"],
                             advancing=protocol["advancing"]))
    simulation_key = _hash(dict(protocol_sha256=key, scenario_key=scenario_key, schedule=schedule))
    block_key, last_block = _block_inventory(schedule)
    if (receipt.get("state") != "complete" or receipt.get("purpose") != PURPOSE
            or receipt.get("protocol_sha256") != key or receipt.get("reference_sha256") != reference_key
            or receipt.get("simulation_key") != simulation_key or receipt.get("fit") != protocol["fit"]
            or receipt.get("synthetic") is not protocol["synthetic"]
            or receipt.get("schedule") != schedule or receipt.get("completed_blocks") != schedule["blocks"]
            or receipt.get("block_inventory_sha256") != block_key or receipt.get("last_completed_block") != last_block
            or receipt.get("runtime") != protocol["runtime"]
            or receipt.get("scientific_evidence_verified") is not False
            or receipt.get("reference_source_availability_verified") is not False
            or receipt.get("publication_allowed") is not False
            or digest(output_dir / "protocol.json") != key
            or (output_dir / "reference.json").read_bytes() != reference_bytes
            or digest(output_dir / "summary.json") != receipt.get("summary_sha256")):
        raise ValueError("Incomplete, changed or inconsistent candidate artifacts")
    summary = json.loads((output_dir / "summary.json").read_bytes())
    expected_identity = dict(fit_id=protocol["fit"]["run_key"], posterior_key=protocol["fit"]["posterior_sha256"],
                             scenario_key=scenario_key, simulation_key=simulation_key)
    if (summary.get("identity") != expected_identity or summary.get("controller_sources") != protocol["sources"]
            or summary.get("field") != identity["field"] or summary.get("posterior_shape") != protocol["posterior_shape"]
            or summary.get("replicates_per_draw") != protocol["workload"]["replicates"]
            or summary.get("n_races") != len(scenarios) or summary.get("protocol_sha256") != key
            or summary.get("simulation_runtime") != protocol["runtime"]
            or summary.get("reference_source_availability_verified") is not False
            or summary.get("reference_scenario_id") != reference["scenario_id"]
            or summary.get("unresolved_reference_assumptions") != reference["unresolved_before_publication"]):
        raise ValueError("Candidate summary differs from its registered inventory")
    load_moments(output_dir, expected_simulation_key=simulation_key,
                 expected_archive_sha256=receipt["diagnostic_moments"]["sha256"])
    if digest(run_dir / "receipt.json") != protocol["fit"]["receipt_sha256"]:
        raise ValueError("Candidate fit receipt changed")
    with fitting.checked_archive(run_dir, protocol["fit"]["run_key"]) as (_, fitted):
        require_fit_registration(protocol, fitted, root=registration_root)
        if (fitted.get("catalog") != design.catalog or fitted["specification"].get("catalog") != design.catalog
                or fitted["specification"].get("arrays_key") != identity["arrays_key"]
                or fitted["specification"].get("source_key") != fitted["source"]["key"]
                or hashlib.sha256(json.dumps(fitted["specification"], sort_keys=True).encode()).hexdigest() != fitted["run_key"]
                or fitted.get("posterior_sha256") != protocol["fit"]["posterior_sha256"]):
            raise ValueError("Candidate fit identity changed")
    precise = all(summary.get(name) is True for name in (
        "expected_points_precision_passed", "explanation_precision_passed", "pairwise_points_precision_passed"))
    if receipt.get("conditional_simulation_precision_passed") is not precise:
        raise ValueError("Candidate precision decision disagrees with its summary")
    components = None
    if precise:
        path = output_dir / "components.json"
        if digest(path) != receipt.get("components_sha256"):
            raise ValueError("Candidate export components changed")
        components = json.loads(path.read_bytes())
        expected = prepare_components(summary, names=protocol["driver_names"],
            scenario=dict(id=reference["scenario_id"], label=protocol["scenario_label"], races=len(scenarios)),
            data_as_of=design.catalog["cutoff"], recorded_at=receipt["started_at"])
        if components != expected:
            raise ValueError("Candidate export components disagree with the paired summary")
    elif receipt.get("components_sha256") is not None or (output_dir / "components.json").exists():
        raise ValueError("Failed-precision candidate must not export ranking components")
    if (receipt_path.read_bytes() != receipt_bytes or Path(reference_path).read_bytes() != reference_bytes
            or digest(protocol_path) != key or protocol["sources"] != source_fingerprints()
            or digest(output_dir / "protocol.json") != key
            or (output_dir / "reference.json").read_bytes() != reference_bytes
            or digest(Path(registration_root) / protocol["fit_registration"]["path"]) != protocol["fit_registration"]["sha256"]
            or digest(output_dir / "diagnostic_moments.npz") != receipt["diagnostic_moments"]["sha256"]
            or design_identity(design) != identity or protocol["runtime"] != runtime_identity()
            or digest(run_dir / "receipt.json") != protocol["fit"]["receipt_sha256"]
            or digest(output_dir / "summary.json") != receipt["summary_sha256"]
            or (precise and digest(output_dir / "components.json") != receipt["components_sha256"])):
        raise ValueError("Candidate inputs or artifacts changed while reading")
    return summary, receipt, components
