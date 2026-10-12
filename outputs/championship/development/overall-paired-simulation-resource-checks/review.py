"""Join immutable paired-resource measurements and focused verification evidence."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent


def sha(value):
    return hashlib.sha256(value).hexdigest()


def committed(path):
    path = Path(path)
    value = path.read_bytes()
    saved = subprocess.check_output(["git", "show", f"HEAD:{path.relative_to(ROOT).as_posix()}"], cwd=ROOT)
    if saved != value:
        raise ValueError(f"Commit the unchanged evidence before review: {path}")
    return value


def main():
    committed(__file__)
    resource = {}
    for name, relative in (("original", "resources.json"), ("shared_randomness", "optimized/resources.json")):
        content = committed(HERE / relative)
        row = json.loads(content)
        if row["status"] != "complete" or row["publication_allowed"] or row["full_simulation_complete"]:
            raise ValueError("Require completed unpublished block measurements")
        for source, expected in row["source_hashes"].items():
            saved = subprocess.check_output(["git", "show", f"{row['source_commit']}:{source}"], cwd=ROOT)
            if sha(saved) != expected:
                raise ValueError(f"Resource source identity disagrees: {source}")
        if row["incomplete_finalization_rejection"] != "Every declared posterior draw must have its complete replicate inventory":
            raise ValueError("The finalization rejection was not for incomplete coverage")
        begin, end = [dt.datetime.fromisoformat(row[key]) for key in ("started_at", "finished_at")]
        resource[name] = dict(file=relative, sha256=sha(content), source_commit=row["source_commit"],
            paired_block_seconds=row["paired_block_seconds"], accumulation_seconds=row["accumulation_seconds"],
            total_monotonic_seconds=row["total_monotonic_seconds"], timestamp_interval_seconds=(end-begin).total_seconds(),
            process_max_rss_kib=row["process_max_rss_kib"], limits=row["limits"],
            input_sha256=row["synthetic_parameters_sha256"], output_sha256=row["output_arrays_sha256"])
        resource[name]["measurement"] = row
    old, new = [resource[name]["measurement"] for name in ("original", "shared_randomness")]
    paired = old["measured_block"].copy()
    paired["baseline_and_probe_panels"] = paired.pop("simulator_invocations")
    if paired != new["measured_block"]:
        raise ValueError("The paired resource workloads differ")
    keys = ("runtime", "field_size", "reference_events", "reference_cutoff", "hypothetical_complete_schedules",
            "synthetic_parameters_sha256", "output_arrays_sha256", "output_array_bytes", "output_shapes",
            "every_output_finite", "nonzero_paired_probe_losses", "accumulator_storage_bytes")
    if any(old[key] != new[key] for key in keys):
        raise ValueError("Inputs, schedules or exact outputs differ between resource versions")
    for name in resource:
        del resource[name]["measurement"]
    suites = {}
    for file, revision in (("equivalence.xml", "52b9ba1"), ("connected.xml", "237fc49")):
        content = committed(HERE / file)
        root = ET.fromstring(content)
        totals = root.findall("testsuite") if root.tag == "testsuites" else [root]
        counts = {key:sum(int(node.get(key, "0")) for node in totals) for key in ("tests", "failures", "errors", "skipped")}
        if counts["failures"] or counts["errors"] or counts["skipped"]:
            raise ValueError("Focused verification did not pass completely")
        suite_commit = subprocess.check_output(["git", "rev-parse", revision], cwd=ROOT, text=True).strip()
        suites[file] = dict(source_commit=suite_commit, sha256=sha(content), **counts,
            reported_seconds=sum(float(node.get("time", "0")) for node in totals),
            output_file=file.replace(".xml", ".txt"),
            output_sha256=sha(committed(HERE / file.replace(".xml", ".txt"))))
    launch = committed(HERE / "launch-failed.txt").decode()
    status = int(committed(HERE / "launch-failed.status").decode())
    untouched = ["f1rank", "extract", "data", "docs/overall_rotation_profile_proposal.json"]
    changed = subprocess.check_output(["git", "diff", "--name-only", "63bcb7b", "HEAD", "--", *untouched], cwd=ROOT, text=True)
    if changed:
        raise ValueError("Fitting/data/proposal inputs changed during this engineering check")
    report = dict(recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        review_source_sha256=sha(committed(__file__)), reports=suites, resources=resource,
        workload=paired, field_size=new["field_size"], reference_events=new["reference_events"],
        reference_cutoff=new["reference_cutoff"], accumulator_storage_bytes=new["accumulator_storage_bytes"],
        hypothetical_complete_schedules=new["hypothetical_complete_schedules"],
        exact_input_and_output_fingerprints_match=True,
        recorded_paired_block_duration_ratio=old["paired_block_seconds"]/new["paired_block_seconds"],
        initial_launch_failure=dict(exit_code=status, output=launch,
            workload_started=False, correction="Moved Nice from the unsupported scope property to the child command; caps and workload unchanged"),
        fitting_sources_data_and_pending_proposal_unchanged=True,
        tests_overlap=True, mcmc_run=False, publication_allowed=False, full_goal_complete=False,
        limitations=["One maximum-size synthetic block, not a complete posterior simulation or accepted driver ranking",
            "Zero-retirement, full-racing-distance resource fixture; retirement behavior is covered only by smaller equivalence tests",
            "UTC timestamp intervals and monotonic durations disagree; both are retained without a claimed cause or exact wall time",
            "Recorded duration ratio applies only to the local measured block, not a forecast of full-run or hosted-runner cost",
            "Hypothetical schedules are counts under existing bounds, not executed simulations or passed precision checks",
            "Original numerical failure and pending separate-study approval remain; recovery and independent validation are incomplete"])
    destination = HERE / "verification.json"
    if destination.exists():
        raise FileExistsError("Preserve the existing review")
    destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
