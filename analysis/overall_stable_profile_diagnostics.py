"""Reproduce every terminal corrected-profile attempt from its saved arrays.

This reads the original registered inventory, posterior and sampler extras.
It never fits, revises a convergence gate or grants scientific acceptance.
"""
import argparse
from importlib.metadata import version
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import zipfile

from analysis import overall_complete_profile_diagnostics as common
from analysis import overall_stable_profile_review as review

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    "analysis/overall_stable_profile_diagnostics.py",
    "analysis/overall_stable_profile_review.py",
    "analysis/overall_complete_profile_review.py",
    "analysis/overall_complete_profile_diagnostics.py",
    "tests/test_overall_stable_profile_diagnostics.py",
    "docs/overall_stable_profile_diagnostics.md",
)
LOADED = {name: review.sha(ROOT / name) for name in SOURCES}


def relative(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def evidence(path):
    """Replay the terminal review and bind the exact audit implementation."""
    path = Path(path).resolve()
    review.original.committed(path)
    original = review.read(path)
    fingerprints = {relative(path): review.sha(path)}
    for name in SOURCES:
        review.original.committed(ROOT / name)
        review.require(review.sha(ROOT / name) == LOADED[name],
                       "Loaded diagnostic verification source changed")
        fingerprints[name] = LOADED[name]
    # collect checks the authoritative service before importing fit dependencies.
    current = review.collect(ROOT / original["registration"])
    stable = lambda value: {k: v for k, v in value.items()
                            if k not in ("recorded_at", "terminal_service")}
    review.require(stable(current) == stable(original),
                   "Committed review no longer matches complete terminal evidence")
    fingerprints.update(current["input_sha256"])
    registration = review.read(ROOT / current["registration"])
    return current, registration, fingerprints


def reproduce(archive, sampler_archive, inventory, recorded, catalog, temporary_parent):
    """Verify the complete arrays and exactly reproduce all numerical reports."""
    import numpy as np
    from f1rank.overall.acceptance import complete_posterior, _canonical_diagnostics
    from f1rank.overall.bounded_diagnostics import check
    from f1rank.overall.inference import EXTRA_FIELDS, contrast_reports, validate_extras

    with zipfile.ZipFile(sampler_archive) as saved:
        names = saved.namelist()
        review.require(len(names) == len(EXTRA_FIELDS)
                       and set(names) == {name + ".npy" for name in EXTRA_FIELDS},
                       "Sampler archive must contain exactly every registered extra")
    with np.load(sampler_archive, allow_pickle=False) as saved:
        extras = {name: saved[name] for name in saved.files}
    validate_extras(extras, recorded)
    sampler = dict(
        divergences=int(extras["diverging"].sum()),
        tree_depth_limit_fraction=float(np.mean(
            extras["num_steps"] >= 2**recorded["max_tree_depth"] - 1)),
        mean_accept_prob=float(extras["accept_prob"].mean()),
    )
    with complete_posterior(archive, {"attempts": [recorded]}, inventory,
                            temporary_parent=temporary_parent) as post:
        coordinate = check(post, sampler["divergences"])
        comparisons = contrast_reports(post, catalog, sampler["divergences"])
    expected = recorded["diagnostics"]
    coordinate_paths = common.differences(
        _canonical_diagnostics({key: expected.get(key) for key in coordinate}),
        _canonical_diagnostics(coordinate))
    comparison_paths = common.differences(expected.get("comparison_reports"), comparisons)
    sampler_paths = common.differences({key: expected.get(key) for key in sampler}, sampler)
    return dict(
        archive_arrays_verified=True, sampler_extras_recounted=True,
        coordinate_report=coordinate, comparison_reports=comparisons, sampler_report=sampler,
        coordinate_report_equal=not coordinate_paths,
        comparison_reports_equal=not comparison_paths, sampler_report_equal=not sampler_paths,
        coordinate_difference_paths=coordinate_paths, comparison_difference_paths=comparison_paths,
        sampler_difference_paths=sampler_paths,
        other_recorded_fields_not_recomputed=sorted(
            set(expected) - set(coordinate) - set(sampler) - {"comparison_reports"}),
    )


def verify_studies(original, registration, temporary_parent, checkpoint):
    """Retain each original failure and continue across missing or bad attempts."""
    from analysis.overall_stable_profile import PREFLIGHT_OUTPUT

    studies = []
    for study in original["studies"]:
        case = study["case"]
        receipt = study.get("receipt")
        row = dict(case=case, original_completion=study["completion"],
                   original_receipt_state=receipt.get("state") if receipt else None,
                   issues=list(study["issues"]), attempts=[])
        studies.append(row)
        attempts = receipt.get("attempts", []) if receipt else []
        if not attempts:
            row["issues"].append("No saved attempt reports are available for reproduction")
        indexed = {attempt["attempt"]: attempt for attempt in study["attempts"]}
        attempt_issues = {f"attempt {attempt['attempt']}: {issue}"
                          for attempt in study["attempts"] for issue in attempt["issues"]}
        case_issues = [issue for issue in study["issues"] if issue not in attempt_issues]
        for index, recorded in enumerate(attempts, 1):
            started = time.monotonic()
            result = dict(attempt=index, state="running", issues=[],
                          original_diagnostics=recorded.get("diagnostics"),
                          archive_arrays_verified=False, sampler_extras_recounted=False,
                          coordinate_report_equal=False, comparison_reports_equal=False,
                          sampler_report_equal=False, publication_allowed=False)
            row["attempts"].append(result)
            checkpoint(studies)
            try:
                reviewed = indexed.get(index)
                review.require(reviewed is not None, "Attempt has no complete metadata review")
                result.update(archive=reviewed.get("archive"),
                              archive_sha256=reviewed.get("archive_sha256"),
                              original_role=reviewed.get("archive_role"))
                review.require(not case_issues and not reviewed["issues"],
                               "Original case or attempt metadata is inconsistent")
                review.require(reviewed["recorded"] == recorded,
                               "Attempt report differs from reviewed receipt")
                if not reviewed.get("archive_present"):
                    result.update(state="unavailable", issues=["No retained posterior archive"])
                else:
                    cache = ROOT / "outputs/championship/cache" / registration["id"] / case
                    archive = ROOT / reviewed["archive"]
                    sampler = cache / recorded["sampler_file"]
                    review.require(archive.resolve().parent == cache.resolve()
                                   and sampler.resolve().parent == cache.resolve(),
                                   "An archive escaped its registered case")
                    inventory_path = ROOT / PREFLIGHT_OUTPUT / case / "result.json"
                    name = relative(inventory_path)
                    review.require(name in registration["evidence_sha256"],
                                   "Complete preflight inventory is not registered")
                    paths = {
                        archive: reviewed["archive_sha256"],
                        sampler: recorded["sampler_sha256"],
                        inventory_path: registration["evidence_sha256"][name],
                    }
                    for path, expected in paths.items():
                        review.require(review.sha(path) == expected, "Saved audit input bytes changed")
                    inventory = review.read(inventory_path)["sites"]
                    result.update(reproduce(archive, sampler, inventory, recorded,
                                            receipt["catalog"], temporary_parent))
                    for path, expected in paths.items():
                        review.require(review.sha(path) == expected,
                                       "Saved audit input changed during reproduction")
                    result["state"] = ("verified" if all(result[key] for key in (
                        "coordinate_report_equal", "comparison_reports_equal", "sampler_report_equal"))
                        else "different")
            except Exception as exc:
                result.update(state="failed", error=type(exc).__name__, message=str(exc))
                result["issues"].append(f"{type(exc).__name__}: {exc}")
            finally:
                result["seconds"] = time.monotonic() - started
                checkpoint(studies)
        checkpoint(studies)
    return studies


def run(review_path, output, scope):
    actual = common.actual_scope(scope)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(
        purpose="corrected_profile_saved_diagnostic_reproduction", state="running",
        started_at=common.now(), actual_scope=actual, studies=[], issues=[], all_processed=False,
        all_recorded_reports_reproduced=False, all_sampler_extras_recounted=False,
        mcmc_run=False, scientific_acceptance_established=False,
        full_recovery_complete=False, sbc_complete=False, publication_allowed=False,
        full_goal_complete=False,
    )
    destination = output / "audit.json"
    common.save(destination, record)

    def checkpoint(studies):
        record["studies"] = studies
        common.save(destination, record)

    try:
        original, registration, fingerprints = evidence(review_path)
        record.update(
            review=relative(review_path), input_sha256=fingerprints,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            diagnostic_runtime=dict(python=sys.version, packages={
                name: version(name) for name in ("numpy", "arviz-stats", "jax", "numpyro")}),
            recorded_all_converged=original["recorded_all_converged"],
            original_records_consistent=original["records_consistent"],
            terminal_service=original["terminal_service"],
        )
        record["issues"].extend(original["issues"])
        record["studies"] = verify_studies(
            original, registration, ROOT / "outputs/championship/cache", checkpoint)
        record["all_processed"] = True
        # Replay the whole collector, including inventories, rather than checking
        # only already-known paths and overlooking newly added cache files.
        _, final_registration, final_fingerprints = evidence(review_path)
        review.require(final_registration == registration and final_fingerprints == fingerprints,
                       "Complete evidence inventory changed during diagnostic verification")
        record["all_recorded_reports_reproduced"] = (
            original["records_consistent"] and not record["issues"]
            and common.all_reproduced(record["studies"]))
        record["all_sampler_extras_recounted"] = (
            bool(record["studies"]) and all(bool(study["attempts"])
                and all(attempt["sampler_extras_recounted"] for attempt in study["attempts"])
                for study in record["studies"]))
        record["state"] = ("complete" if record["all_recorded_reports_reproduced"] else "incomplete")
    except Exception as exc:
        record.update(state="failed", error=type(exc).__name__, message=str(exc),
                      traceback=traceback.format_exc())
    finally:
        record.update(finished_at=common.now(), seconds=time.monotonic() - started,
                      process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        common.save(destination, record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope-name", required=True)
    args = parser.parse_args()
    result = run(args.review, args.output, args.scope_name)
    raise SystemExit(0 if result["all_recorded_reports_reproduced"] else 1)
