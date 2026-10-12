"""Complete saved-fixture sensitivity audit; no refit or new random draws."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.overall_rank_sensitivity import CONTRACT, SOURCES, envelope, sha, source_fingerprints
from analysis.overall_ranking_resources import scope_limits
from f1rank.artifacts import atomic_json

PREVIOUS = "outputs/championship/development/overall-functional-diagnostic-checks"


def run(output_dir):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = dict(purpose="saved_fixture_rank_sensitivity_audit", status="running",
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(), new_random_draws=False,
        mcmc_run=False, canonical_simulator_run=False, publication_allowed=False)
    try:
        report["limits"] = scope_limits()
        inputs = [f"{PREVIOUS}/noise-audit/audit.json", f"{PREVIOUS}/noise-audit/constructed_traces.npz",
                  f"{PREVIOUS}/storage/resources.json"]
        names = [*SOURCES, "analysis/overall_rank_sensitivity_audit.py", "analysis/overall_ranking_resources.py",
                 "f1rank/artifacts.py", *inputs]
        hashes = {name: sha(ROOT / name) for name in names}
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        for name, expected in hashes.items():
            raw = subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=ROOT)
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Commit the fixed diagnostic and inputs before execution")
        report.update(source_commit=source, input_and_source_sha256=hashes)
        original = json.loads((ROOT / inputs[0]).read_text())
        if original["status"] != "complete" or sha(ROOT / inputs[1]) != original["constructed_traces_sha256"]:
            raise ValueError("The original complete fixture and trace bytes are required")
        delta = 1 - CONTRACT["interval_probability"]
        report.update(error_probability=delta, cases={})
        with np.load(ROOT / inputs[1], allow_pickle=False) as saved:
            expected = {name + suffix for name in original["cases"] for suffix in (
                "_probabilities", "_exact_means", "_simulated_means", "_win_counts", "_conditional_points_variance")}
            if set(saved.files) != expected or len(saved.files) != len(expected):
                raise ValueError("Keep the entire original fixture inventory")
            for name in saved.files:
                if saved[name].dtype.kind not in "fiu" or not np.isfinite(saved[name]).all():
                    raise ValueError("Invalid saved fixture member")
            for name, previous in original["cases"].items():
                exact, noisy = saved[name + "_exact_means"], saved[name + "_simulated_means"]
                if exact.shape != noisy.shape or list(exact.shape[:2]) != original["posterior_shape"]:
                    raise ValueError("The complete saved trace is required")
                counts = np.full(exact.shape[:2], original["replicates"], dtype=np.int64)
                result = envelope(noisy, counts, [f"anonymous_{i}" for i in range(exact.shape[-1])], error_probability=delta)
                atomic_json(folder / f"{name}.json", result)
                radius = np.asarray(result["point_radius_by_draw"])
                difference = exact[..., 1] - exact[..., 0]
                truth_fractions = dict(a_greater=float((difference > 0).mean()),
                                      b_greater=float((difference < 0).mean()), equal=float((difference == 0).mean()))
                report["cases"][name] = dict(result_file=f"{name}.json", result_sha256=sha(folder / f"{name}.json"),
                    original_exact_checker_passed=previous["exact_diagnostics"]["converged"],
                    original_noisy_checker_passed=previous["noisy_diagnostics"]["converged"],
                    family_size=result["family_size"], point_radius_min=float(radius.min()), point_radius_max=float(radius.max()),
                    exact_means_inside_declared_intervals_this_fixture=bool((np.abs(exact - noisy) <= radius[..., None]).all()),
                    exact_empirical_pair_fractions=truth_fractions,
                    empirical_pair_fraction_bounds=result["pairs"][0]["empirical_fraction_bounds"],
                    interpretation="Recorded fixture containment is not proof of general coverage or MCMC convergence.")
        dimensions = json.loads((ROOT / inputs[2]).read_text())
        shape, n, reps = dimensions["posterior_shape"], dimensions["field_size"], dimensions["replicates"]
        if n < len(CONTRACT["points"]):
            raise ValueError("The fixed resource workload retains the full points table")
        before = time.monotonic()
        means = np.broadcast_to(np.pad(np.asarray(CONTRACT["points"], float), (0, n - len(CONTRACT["points"]))), tuple(shape) + (n,))
        losses = np.zeros(means.shape + (len(CONTRACT["required_explanations"]),))
        result = envelope(means, np.full(shape, reps, dtype=np.int64), [f"anonymous_{i}" for i in range(n)],
                          error_probability=delta, probe_loss_mean=losses)
        seconds = time.monotonic() - before
        path = folder / "full_size_resource.json"
        atomic_json(path, result)
        report["resource_workload"] = dict(posterior_shape=shape, field_size=n, replicates=reps,
            family_size=result["family_size"], seconds_including_input_construction=seconds,
            output_file=path.name, output_sha256=sha(path), output_bytes=path.stat().st_size,
            full_trace_diagnostic_only=True, fitted_posterior=False, nested_simulation_executed=False)
        source_fingerprints()
        if hashes != {name: sha(ROOT / name) for name in names}:
            raise ValueError("Diagnostic sources or saved fixtures changed")
        report.update(status="complete", all_original_cases_retained=True,
            mcmc_precision_verified=False, rank_precision_established=False, scientific_evidence_verified=False,
            runtime=dict(python=sys.version, numpy=np.__version__),
            limitations=["Saved synthetic examples, not real drivers or new independent evidence",
                         "Concentration assumes independent replicated seasons conditional on each draw",
                         "Intervals cover simulation error for the fixed trace, not posterior or MCMC uncertainty",
                         "Resource measurement covers this diagnostic alone, not the nested simulator or intended runner"])
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            seconds=time.monotonic()-started, process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(folder / "audit.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    result = run(parser.parse_args().output_dir)
    print(json.dumps(dict(status=result["status"], cases=list(result["cases"]))))
