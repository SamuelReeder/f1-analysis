"""Compare fixed paired-block implementations without evaluating driver ranks."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
EARLIER = HERE.parent / "overall-paired-simulation-resource-checks"


def sha(content):
    return hashlib.sha256(content).hexdigest()


def committed(path):
    path = Path(path)
    content = path.read_bytes()
    saved = subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    if content != saved:
        raise ValueError(f"Commit unchanged review inputs: {path}")
    return content


def main():
    review_sha = sha(committed(__file__))
    reports, resources = {}, {}
    for name, path in (("original", EARLIER / "resources.json"),
                       ("shared_randomness", EARLIER / "optimized/resources.json"),
                       ("batched_interventions", HERE / "resource/resources.json")):
        content = committed(path)
        row = json.loads(content)
        if row["status"] != "complete" or not row["every_output_finite"]:
            raise ValueError("Require complete finite block measurements")
        if any(row[key] for key in ("mcmc_run", "publication_allowed", "full_simulation_complete", "simulation_precision_established")):
            raise ValueError("A resource block cannot supply scientific acceptance")
        for source, expected in row["source_hashes"].items():
            saved = subprocess.check_output(["git", "show", f"{row['source_commit']}:{source}"], cwd=ROOT)
            if sha(saved) != expected:
                raise ValueError("Recorded source fingerprint disagrees with its commit")
        if "simulator_invocations" in row["measured_block"]:
            row["measured_block"]["baseline_and_probe_panels"] = row["measured_block"].pop("simulator_invocations")
        reports[name] = row
        start, end = (dt.datetime.fromisoformat(row[key]) for key in ("started_at", "finished_at"))
        resources[name] = dict(file=str(path.relative_to(HERE.parent)), sha256=sha(content),
            source_commit=row["source_commit"], limits=row["limits"],
            paired_block_seconds=row["paired_block_seconds"], accumulation_seconds=row["accumulation_seconds"],
            total_monotonic_seconds=row["total_monotonic_seconds"], timestamp_interval_seconds=(end-start).total_seconds(),
            process_max_rss_kib=row["process_max_rss_kib"],
            input_sha256=row["synthetic_parameters_sha256"], output_sha256=row["output_arrays_sha256"])
    original, batched = reports["original"], reports["batched_interventions"]
    matching = ("measured_block", "runtime", "field_size", "reference_events", "reference_cutoff",
        "hypothetical_complete_schedules", "synthetic_parameters_sha256", "output_arrays_sha256",
        "output_array_bytes", "output_shapes", "nonzero_paired_probe_losses", "accumulator_storage_bytes",
        "incomplete_finalization_rejection")
    if any(row[key] != original[key] for row in reports.values() for key in matching):
        raise ValueError("The fixed resource inputs, complete outputs or declared schedules changed")
    xml = committed(HERE / "connected.xml")
    root = ET.fromstring(xml)
    suites = root.findall("testsuite") if root.tag == "testsuites" else [root]
    counts = {key:sum(int(suite.get(key, "0")) for suite in suites) for key in ("tests", "failures", "errors", "skipped")}
    if counts["failures"] or counts["errors"] or counts["skipped"]:
        raise ValueError("Focused checks did not all pass")
    source = subprocess.check_output(["git", "rev-parse", "dcb364c"], cwd=ROOT, text=True).strip()
    for name, expected in batched["source_hashes"].items():
        saved = subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=ROOT)
        if sha(saved) != expected:
            raise ValueError("Resource inputs differ from the focused-test source")
    untouched = ["f1rank", "extract", "data", "docs/overall_rotation_profile_proposal.json"]
    changed = subprocess.check_output(["git", "diff", "--name-only", "102eaf2", "--", *untouched], cwd=ROOT, text=True)
    if changed:
        raise ValueError("Fitting, data or pending scientific proposal changed")
    record = dict(recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(), review_source_sha256=review_sha,
        test_source_commit=source, resources=resources, exact_fixed_input_and_complete_output_match=True,
        tests=dict(file="connected.xml", sha256=sha(xml), **counts,
            reported_seconds=sum(float(suite.get("time", "0")) for suite in suites),
            process_output="connected.txt", process_output_sha256=sha(committed(HERE / "connected.txt"))),
        full_repository_suite_rerun=False, fixed_workload=batched["measured_block"],
        hypothetical_complete_schedules=batched["hypothetical_complete_schedules"],
        recorded_duration_ratio_to_original=original["paired_block_seconds"]/batched["paired_block_seconds"],
        recorded_duration_ratio_to_shared_randomness=reports["shared_randomness"]["paired_block_seconds"]/batched["paired_block_seconds"],
        checks=["Replay preserves original qualifying grids, following Gumbels and retirement uniforms in each tested lane",
            "Changed replay method, shape, degrees of freedom and incomplete/excess requests are rejected",
            "Complete baseline and all own-driver probe points match the original separate simulator calls",
            "Existing failure, source, grouped-posterior and summary-controller checks remain passing"],
        fitting_sources_data_and_pending_proposal_unchanged=True, mcmc_run=False,
        full_posterior_workload_executed=False, simulation_precision_established=False,
        scientific_evidence_verified=False, publication_allowed=False, full_goal_complete=False,
        limitations=["A local maximum-size synthetic block with zero retirement hazard and full racing distance, not a fitted driver table",
            "Smaller exact comparisons include retirements, no-points contexts, input permutations, both saved floating precisions and original failures",
            "Record monotonic durations and UTC intervals separately; earlier records disagree between clocks and no cause is established",
            "Duration ratios describe only these measured blocks; no full-run or intended-runner runtime is inferred",
            "Full simulation resources and precision, the failed numerical case, recovery and independent validation remain unresolved"])
    destination = HERE / "verification.json"
    if destination.exists():
        raise FileExistsError("Preserve the existing review")
    destination.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
