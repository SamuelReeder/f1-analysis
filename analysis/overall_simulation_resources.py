"""Capped synthetic paired-block resource measurement; no fit or publication."""
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
from analysis.overall_simulation_controller import SOURCE_NAMES, block_ranges, runtime_identity
from analysis.overall_summary import MAX_BLOCK_DRAWS, MAX_BLOCK_REPLICATES, PROBES, OverallSummary, paired_simulation_block
from f1rank.artifacts import RETRY, atomic_json
from f1rank.overall.fitting import FIT_SETTINGS
from f1rank.overall.simulation import JointDraws, MAX_EXPECTED_POINTS_MCSE, POINTS, QUALIFYING_FIELDS, Scenario

SEED = 20261012


def arrays_sha256(arrays):
    result = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        a = np.ascontiguousarray(value)
        result.update(json.dumps([name, a.dtype.str, list(a.shape)]).encode())
        result.update(memoryview(a).cast("B"))
    return result.hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    output = output_dir / "resources.json"
    started = time.monotonic()
    report = dict(purpose="synthetic_paired_simulation_block_resources", status="running",
        started_at=utc(), completed_stages=[], mcmc_run=False, publication_allowed=False,
        full_simulation_complete=False, simulation_precision_established=False,
        intended_ci_runner_verified=False, real_driver_table_saved=False)
    atomic_json(output, report)
    try:
        report["limits"] = scope_limits()
        names = sorted(set(SOURCE_NAMES) | {
            "analysis/overall_simulation_resources.py", "analysis/overall_ranking_resources.py",
            "docs/overall_simulation_resources.md", "docs/overall_reference_calendar.json",
            "docs/overall_rotation_profile_proposal.json", "data/processed/race.parquet",
            "outputs/championship/development/interface_preflight.json"})
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        hashes = {name: sha(ROOT / name) for name in names}
        for name, fingerprint in hashes.items():
            saved = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
            if hashlib.sha256(saved).hexdigest() != fingerprint:
                raise ValueError("Commit the resource workload and inputs before measurement")
        runtime = runtime_identity()
        report.update(source_commit=commit, source_hashes=hashes, runtime=runtime)
        reference = json.loads((ROOT / "docs/overall_reference_calendar.json").read_text())
        preflight = json.loads((ROOT / "outputs/championship/development/interface_preflight.json").read_text())
        if preflight["source"]["inputs"]["data/processed/race.parquet"] != hashes["data/processed/race.parquet"]:
            raise ValueError("Field-size reference does not describe the committed race data")
        n, races = preflight["counts"]["field"], len(reference["events"])
        nd, nr = MAX_BLOCK_DRAWS, MAX_BLOCK_REPLICATES
        drivers = tuple(f"synthetic_{i:03}" for i in range(n))
        scenarios = [Scenario(row["event_id"], i, row["reference_distance_laps"],
                              row["reference_distance_laps"]) for i, row in enumerate(reference["events"])]
        # The unchanged bound for paired differences/losses is B / sqrt(R).
        full_replicates = 2 * math.ceil((float(np.max(POINTS)) / MAX_EXPECTED_POINTS_MCSE)**2 / 2)
        identity = dict(fit_id="synthetic-resource-no-fit", posterior_key="0" * 64,
                        scenario_key=hashes["docs/overall_reference_calendar.json"],
                        simulation_key=hashes["analysis/overall_simulation_resources.py"])
        shapes = {"initial": (FIT_SETTINGS["chains"], FIT_SETTINGS["samples"]),
                  "largest_retry": (FIT_SETTINGS["chains"], FIT_SETTINGS["samples"] * max(row[1] for row in RETRY))}
        report.update(field_size=n, reference_events=races, reference_cutoff=preflight["cutoff"],
            measured_block=dict(draws=nd, replicates=nr, seed=SEED, chain=0, draw_start=0,
                replicate_start=0, explanations=list(PROBES), baseline_and_probe_panels=1+n*len(PROBES),
                baseline_seasons=nd*nr, baseline_and_probe_seasons=nd*nr*(1+n*len(PROBES))),
            hypothetical_complete_schedules={}, accumulator_storage_bytes={})
        for name, shape in shapes.items():
            accumulator = OverallSummary(drivers=drivers, posterior_shape=shape,
                replicates=full_replicates, n_races=races, identity=identity)
            # Materialize zero pages across the complete arrays, not just the
            # tiny part receiving this measured block.
            for value in vars(accumulator).values():
                if isinstance(value, np.ndarray):
                    value.fill(0)
            report["accumulator_storage_bytes"][name] = accumulator.storage_bytes
            blocks = shape[0] * len(block_ranges(shape[1], nd)) * len(block_ranges(full_replicates, nr))
            seasons = math.prod(shape) * full_replicates
            report["hypothetical_complete_schedules"][name] = dict(posterior_shape=list(shape),
                replicates=full_replicates, blocks=blocks, baseline_seasons=seasons,
                baseline_and_probe_seasons=seasons*(1+n*len(PROBES)), executed=False,
                conditional_difference_mcse_bound=float(np.max(POINTS))/math.sqrt(full_replicates))
            if name != "largest_retry":
                del accumulator
        report["completed_stages"].append("full_accumulator_allocation_and_schedule_accounting")
        atomic_json(output, report)
        lane = np.arange(nd, dtype=float)[:, None]
        driver = np.arange(n, dtype=float)[None, :]
        values = dict(qualifying=np.broadcast_to(np.linspace(-.5, .5, n), (nd, n)).copy()
                        + .05*np.sin(lane + driver),
            race=np.broadcast_to(np.linspace(.6, -.6, n), (nd, n)).copy() + .05*np.cos(lane - driver),
            q_weekend_sd=np.full(nd, .2), r_weekend_sd=np.full(nd, .3),
            shared_weekend=np.ones(nd), q_scale=np.full(nd, .5), q_log_scale_sd=np.full(nd, .5),
            q_df=np.full(nd, 4.), grid_coefficient=np.ones((nd, races)),
            pace_coefficient=np.ones((nd, races)), retirement_hazard=np.zeros((nd, races)))
        input_key = arrays_sha256(values)
        draws = JointDraws(drivers, **values)
        before = time.monotonic()
        baseline, probes = paired_simulation_block(draws, scenarios, replicates=nr, seed=SEED,
            chain=0, draw_start=0, replicate_start=0, advancing=QUALIFYING_FIELDS[n])
        report["paired_block_seconds"] = time.monotonic() - before
        report["completed_stages"].append("complete_paired_block")
        atomic_json(output, report)
        before = time.monotonic()
        accumulator.add(baseline, probes, chain=0, draw_start=0, replicate_start=0, identity=identity)
        report["accumulation_seconds"] = time.monotonic() - before
        try:
            accumulator.finalize()
        except ValueError as exc:
            if str(exc) != "Every declared posterior draw must have its complete replicate inventory":
                raise
            report["incomplete_finalization_rejection"] = str(exc)
        else:
            raise AssertionError("An incomplete block cannot supply a completed posterior summary")
        if arrays_sha256(values) != input_key:
            raise ValueError("Simulation mutated its synthetic input parameters")
        arrays = {name: value for name, value in baseline.items() if isinstance(value, np.ndarray)}
        arrays["probe_points"] = probes
        report.update(synthetic_parameters_sha256=input_key, output_arrays_sha256=arrays_sha256(arrays),
            output_array_bytes=sum(a.nbytes for a in arrays.values()),
            output_shapes={k:list(v.shape) for k,v in arrays.items()},
            every_output_finite=all(bool(np.isfinite(a).all()) for a in arrays.values()),
            nonzero_paired_probe_losses=int(np.count_nonzero(baseline["points"][..., None] - probes)))
        if not report["every_output_finite"]:
            raise ValueError("Nonfinite simulation result")
        if any(sha(ROOT / name) != fingerprint for name, fingerprint in hashes.items()) or runtime != runtime_identity():
            raise ValueError("Bound inputs or runtime changed during measurement")
        report["completed_stages"].append("accumulation_and_input_rechecks")
        report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=repr(exc), traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_at=utc(), total_monotonic_seconds=time.monotonic()-started,
            process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    result = run(parser.parse_args().output_dir)
    print(json.dumps({k:result[k] for k in ("status", "paired_block_seconds", "accumulation_seconds", "process_max_rss_kib")}))
