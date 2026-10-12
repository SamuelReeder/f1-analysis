"""Fixed bounded-point trace audit; no MCMC fit, ranking or gate changes."""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.overall_ranking_resources import scope_limits, sha
from analysis.overall_simulation_controller import runtime_identity
from f1rank.artifacts import atomic_json
from f1rank.overall.diagnostics import check
from f1rank.overall.fitting import FIT_SETTINGS
from f1rank.overall.simulation import MAX_EXPECTED_POINTS_MCSE, POINTS

TRACE_SEED = 20261012
SIMULATION_SEED = 20261013
SEPARATED_OFFSET = 1e-5
SEPARATED_JITTER = 1e-6
MIXED_SCALE = 1e-3


def run(output_dir):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(purpose="fixed_constructed_trace_noise_audit", status="running", started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        mcmc_run=False, canonical_simulator_run=False, publication_allowed=False, original_failures_changed=False)
    started = time.monotonic()
    atomic_json(folder / "audit.json", report)
    try:
        report["limits"] = scope_limits()
        names = ["analysis/overall_functional_noise_audit.py", "analysis/overall_ranking_resources.py",
            "analysis/overall_simulation_controller.py", "docs/overall_functional_diagnostics.md",
            "docs/overall_rotation_profile_proposal.json", "f1rank/artifacts.py", "f1rank/overall/fitting.py",
            "f1rank/overall/diagnostics.py", "f1rank/overall/simulation.py", "f1rank/overall/scoring.py"]
        source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        hashes = {name: sha(ROOT / name) for name in names}
        for name, expected in hashes.items():
            saved = subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=ROOT)
            if hashlib.sha256(saved).hexdigest() != expected:
                raise ValueError("Commit unchanged audit inputs before execution")
        runtime = runtime_identity()
        report.update(source_commit=source, source_hashes=hashes, runtime=runtime)
        shape = (FIT_SETTINGS["chains"], FIT_SETTINGS["samples"])
        replicates = 2 * math.ceil((float(np.max(POINTS)) / MAX_EXPECTED_POINTS_MCSE)**2 / 2)
        low, high = float(POINTS[1]), float(POINTS[0])
        amplitude = high-low
        if amplitude <= 0:
            raise ValueError("The fixed strict-order fixture requires different winning and runner-up points")
        traces = np.random.default_rng(TRACE_SEED)
        probabilities = dict(
            separated_chain=.5 + SEPARATED_OFFSET*np.linspace(-1, 1, shape[0])[:, None]
                + SEPARATED_JITTER*traces.normal(size=shape),
            mixed_control=.5 + MIXED_SCALE*traces.normal(size=shape))
        simulation = np.random.default_rng(SIMULATION_SEED)
        report.update(posterior_shape=list(shape), replicates=replicates,
            fixture=dict(trace_seed=TRACE_SEED, simulation_seed=SIMULATION_SEED,
                separated_offset=SEPARATED_OFFSET, separated_jitter=SEPARATED_JITTER,
                mixed_scale=MIXED_SCALE, winning_points=high, runner_up_points=low,
                diagnostic_api_divergences=0, diagnostic_api_divergences_are_not_observed_sampler_data=True),
            cases={})
        arrays = {}
        for name, p in probabilities.items():
            if not np.isfinite(p).all() or ((p <= 0) | (p >= 1)).any():
                raise ValueError("Invalid constructed probabilities; no clipping or redraws")
            exact = np.stack((low+amplitude*p, high-amplitude*p), axis=-1)
            wins = simulation.binomial(replicates, p)
            empirical = wins/replicates
            approximate = np.stack((low+amplitude*empirical, high-amplitude*empirical), axis=-1)
            variance = amplitude**2*p*(1-p)
            point_bound = float(np.max(POINTS))/(2*np.sqrt(replicates))
            difference_bound = float(np.max(POINTS))/np.sqrt(replicates)
            point_error = np.sqrt(variance/replicates)
            difference_error = 2*point_error
            exact_diagnostics = check({"expected_points": exact}, 0)
            noisy_diagnostics = check({"expected_points": approximate}, 0)
            report["cases"][name] = dict(exact_diagnostics=exact_diagnostics, noisy_diagnostics=noisy_diagnostics,
                max_exact_conditional_points_mcse=float(point_error.max()),
                max_exact_conditional_difference_mcse=float(difference_error.max()),
                points_mcse_bound=point_bound, difference_mcse_bound=difference_bound,
                inherited_conditional_precision_passed=bool(max(point_error.max(), difference_error.max(),
                    point_bound, difference_bound) <= MAX_EXPECTED_POINTS_MCSE),
                exact_chain_means=exact.mean(axis=1).tolist(), noisy_chain_means=approximate.mean(axis=1).tolist(),
                exact_chain_sd=exact.std(axis=1, ddof=1).tolist(), noisy_chain_sd=approximate.std(axis=1, ddof=1).tolist(),
                diagnostic_failure_masked=bool(not exact_diagnostics["converged"] and noisy_diagnostics["converged"]))
            arrays.update({f"{name}_probabilities": p, f"{name}_exact_means": exact,
                f"{name}_simulated_means": approximate, f"{name}_win_counts": wins,
                f"{name}_conditional_points_variance": variance})
        np.savez_compressed(folder / "constructed_traces.npz", **arrays)
        report["constructed_traces_sha256"] = sha(folder / "constructed_traces.npz")
        if hashes != {name:sha(ROOT/name) for name in names} or runtime != runtime_identity():
            raise ValueError("Audit source or runtime changed")
        report.update(status="complete", limitations=[
            "Constructed trace stress case and independent control, not sampler output or evidence about any F1 driver",
            "The solvable strict-order law uses the original points values but is not the canonical weekend simulator",
            "Conditional noise bounds and checker thresholds are unchanged; no result establishes convergence of the latent model",
            "Grouped diagnostic moments and a registered functional-precision procedure are still required for acceptance"])
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
    print(json.dumps({name: {key:row[key] for key in ("inherited_conditional_precision_passed", "diagnostic_failure_masked")}
                      for name, row in result["cases"].items()}))
