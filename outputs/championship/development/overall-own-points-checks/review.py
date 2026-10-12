"""Review own-driver scoring equivalence and unchanged fixed resource work."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(content):
    return hashlib.sha256(content).hexdigest()


def committed(path):
    path = Path(path)
    content = path.read_bytes()
    saved = subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    if content != saved:
        raise ValueError(f"Commit the unchanged record first: {path}")
    return content


def main():
    review_sha = sha(committed(__file__))
    records, resources, suites = {}, {}, {}
    files = {"original": "overall-paired-simulation-resource-checks/resources.json",
        "shared_randomness": "overall-paired-simulation-resource-checks/optimized/resources.json",
        "batched_interventions": "overall-paired-batch-checks/resource/resources.json",
        "own_driver_scoring": "overall-own-points-checks/resource/resources.json"}
    for name, filename in files.items():
        content = committed(HERE.parent / filename)
        row = json.loads(content)
        if row["status"] != "complete" or not row["every_output_finite"] or any(row[key] for key in (
                "mcmc_run", "publication_allowed", "full_simulation_complete", "simulation_precision_established")):
            raise ValueError("Require complete, unpublished synthetic block measurements")
        for path, expected in row["source_hashes"].items():
            saved = subprocess.check_output(["git", "show", f"{row['source_commit']}:{path}"], cwd=ROOT)
            if sha(saved) != expected:
                raise ValueError("Resource source differs from its recorded commit")
        block = row["measured_block"]
        if "simulator_invocations" in block:
            block["baseline_and_probe_panels"] = block.pop("simulator_invocations")
        records[name] = row
        start, end = [dt.datetime.fromisoformat(row[key]) for key in ("started_at", "finished_at")]
        resources[name] = dict(file=filename, sha256=sha(content), source_commit=row["source_commit"],
            paired_block_seconds=row["paired_block_seconds"], accumulation_seconds=row["accumulation_seconds"],
            total_monotonic_seconds=row["total_monotonic_seconds"], timestamp_interval_seconds=(end-start).total_seconds(),
            process_max_rss_kib=row["process_max_rss_kib"], limits=row["limits"],
            input_sha256=row["synthetic_parameters_sha256"], output_sha256=row["output_arrays_sha256"])
    reference, current = records["original"], records["own_driver_scoring"]
    same = ("measured_block", "runtime", "field_size", "reference_events", "reference_cutoff",
        "hypothetical_complete_schedules", "synthetic_parameters_sha256", "output_arrays_sha256",
        "output_array_bytes", "output_shapes", "nonzero_paired_probe_losses", "accumulator_storage_bytes",
        "incomplete_finalization_rejection")
    if any(row[key] != reference[key] for row in records.values() for key in same):
        raise ValueError("The fixed inputs, complete outputs or workload schedule changed")
    for filename, revision in (("exhaustive.xml", "ac69b9b"), ("connected.xml", "d8ced5d")):
        content = committed(HERE / filename)
        root = ET.fromstring(content)
        entries = root.findall("testsuite") if root.tag == "testsuites" else [root]
        counts = {key:sum(int(row.get(key, "0")) for row in entries) for key in ("tests", "failures", "errors", "skipped")}
        if counts["failures"] or counts["errors"] or counts["skipped"]:
            raise ValueError("Focused verification did not pass")
        commit = subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT, text=True).strip()
        suites[filename] = dict(source_commit=commit, sha256=sha(content), **counts,
            reported_seconds=sum(float(row.get("time", "0")) for row in entries),
            output_file=filename.replace(".xml", ".txt"),
            output_sha256=sha(committed(HERE / filename.replace(".xml", ".txt"))))
    for path, expected in current["source_hashes"].items():
        saved = subprocess.check_output(["git", "show", f"{suites['connected.xml']['source_commit']}:{path}"], cwd=ROOT)
        if sha(saved) != expected:
            raise ValueError("Resource inputs differ from the connected-test source")
    changed = subprocess.check_output(["git", "diff", "--name-only", "102eaf2", "--", "f1rank", "extract",
        "data", "docs/overall_rotation_profile_proposal.json"], cwd=ROOT, text=True)
    if changed:
        raise ValueError("Fitting, data or pending numerical proposal changed")
    result = dict(recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(), review_source_sha256=review_sha,
        reports=suites, resources=resources, exact_inputs_and_complete_outputs_match=True,
        recorded_duration_ratio_to_original=reference["paired_block_seconds"]/current["paired_block_seconds"],
        recorded_duration_ratio_to_batched=records["batched_interventions"]["paired_block_seconds"]/current["paired_block_seconds"],
        fixed_workload=current["measured_block"], hypothetical_complete_schedules=current["hypothetical_complete_schedules"],
        checks=["Exhaustive small-field distance/order cases spanning the classification threshold match the original full-field scorer's diagonal",
            "Generated full-size fields and no-points controls match exactly, with read-only inputs unchanged",
            "Invalid orders, incomplete axes, invalid distances and unsupported all-retired scenarios are rejected",
            "The full original paired-simulator and connected archive/controller checks remain passing",
            "The unchanged maximum-size measured block has matching input and complete output fingerprints across every recorded implementation"],
        focused_suites_overlap=True, full_repository_suite_rerun=False,
        fitting_sources_data_and_pending_proposal_unchanged=True, mcmc_run=False,
        full_posterior_workload_executed=False, simulation_precision_established=False,
        scientific_evidence_verified=False, publication_allowed=False, full_goal_complete=False,
        limitations=["Specialization for complete strict generated orders; general dead-heat adjudication remains with the original scorer",
            "Measured local resource fixture has zero retirement hazard and full racing distance; smaller exact tests cover retirements",
            "Monotonic durations and UTC intervals are retained separately; their earlier discrepancy has no established cause",
            "Recorded duration ratios apply only to these local blocks, not full-run or intended-runner cost",
            "Full workload feasibility and precision, the failed numerical case, recovery and independent validation remain unresolved"])
    destination = HERE / "verification.json"
    if destination.exists():
        raise FileExistsError("Preserve existing verification")
    destination.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
