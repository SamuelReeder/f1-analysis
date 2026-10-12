"""Local resource check with synthetic parameters at the committed data shape.

No MCMC, fitted driver values, predictive scores or ranking. Temporary memory
maps contain arbitrary finite workload values, not an accepted posterior. The
fit ladder supplies the largest draw shape; it is not run or modified here.
"""
import argparse
from contextlib import ExitStack
import datetime as dt
import hashlib
import json
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.overall_forecast_catalog import build_catalog
from analysis.overall_forecast_states import DEFAULT_DRAW_BATCH, iter_forecast_states
from analysis.overall_observation_kernels import CONTEXT_NAMES, iter_conditional_weekends
from analysis.overall_posterior_archive import mapped_checked_posterior
from f1rank.artifacts import RETRY
from f1rank.overall.design import build
from f1rank.overall.fitting import FIT_SETTINGS
from f1rank.overall.provenance import snapshot
from f1rank.overall.simulation import QUALIFYING_FIELDS

WORKLOAD_SEED = 20261012


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def scope_limits():
    line = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))
    group = Path("/sys/fs/cgroup") / line.split("::", 1)[1].lstrip("/")
    memory = (group / "memory.max").read_text().strip()
    cpu = (group / "cpu.max").read_text().strip()
    quota, period = cpu.split()
    if memory == "max" or int(memory) > 3 * 1024**3 or quota == "max" or int(quota) > 2 * int(period):
        raise ValueError("Run this resource check in the required capped systemd scope")
    return dict(cgroup=str(group), memory_max=memory, cpu_max=cpu)


def parameter_template(design, *, observations=False):
    q = design.qualifying.arrays()
    scalar = {key:.15 for key in (
        "sd_level", "sd_skill_race", "sd_skill_season", "overall_sd_level", "overall_sd_drift_race",
        "overall_sd_drift_season", "sd_car_race", "sd_car_season", "sd_car_reset", "sd_car_event",
        "sd_track_load", "sd_compat", "overall_sd_team", "overall_sd_car_day", "sd_driver_form", "overall_sd_day")}
    scalar.update(exp_gain=.4, exp_scale=20., age_slope=-.02, overall_qualifying_link=.8,
        overall_team_link=.6, overall_car_link=1.1, overall_shared_weekend=.7, rho=.8, rho_reset=.4)
    vectors = dict(level_z=q["n_drivers"], skill_z=q["n_entries"], overall_level_z=q["n_drivers"],
        overall_walk_z=q["n_entries"], car_step_z=q["n_cars"], car_jump_z=q["n_cars"],
        track_load_z=q["n_team_seasons"], compat_z=q["n_era_stints"], overall_team_z=q["n_era_stints"],
        overall_car_contrast=design.outcomes["car_basis"].shape[1])
    if observations:
        scalar.update(sigma0=.4, sigma_tau=.3, nu=5., sd_car_segment=.2)
        for name in CONTEXT_NAMES:
            for part, value in dict(base=-3. if name == "hazard" else .2, wet=.1, sc=-.2).items():
                scalar[f"overall_{name}_{part}"] = value
            scalar[f"overall_sd_{name}_circuit"] = .3
            vectors[f"overall_{name}_circuit_z"] = design.outcomes["n_circuits"]
    return scalar, vectors


def observation_arrays(event):
    """Only numeric conditional outputs, without recording a driver table."""
    for name in ("grid_feature_slot", "strength", "running_positions", "order_log_probability",
                 "joint_order_completion_log_probability"):
        yield name, event[name]
    for block in ("completion", "context"):
        for name, value in sorted(event[block].items()):
            yield block + ":" + name, value
    yield "qualifying:positions", event["qualifying"]["positions"]
    for i, segment in enumerate(event["qualifying"]["segments"]):
        for name, value in sorted(segment.items()):
            yield f"qualifying:{i}:{name}", value


def run(*, compressed=False, observations=False):
    limits = scope_limits()
    tracked = ["analysis/overall_forecast_states.py", "analysis/overall_forecast_catalog.py",
        "analysis/overall_forecast_resources.py", "f1rank/overall/design.py", "f1rank/overall/fitting.py",
        "f1rank/model.py", "f1rank/overall/model.py", "docs/overall_rotation_profile_proposal.json",
        "analysis/overall_posterior_archive.py", "analysis/overall_observation_kernels.py",
        "f1rank/overall/order.py", "f1rank/overall/simulation.py"]
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for name in tracked:
        saved = subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=ROOT)
        if hashlib.sha256(saved).hexdigest() != sha(ROOT / name):
            raise ValueError("Commit resource-check inputs before measuring")
    preflight_path = ROOT / "outputs/championship/development/interface_preflight.json"
    preflight = json.loads(preflight_path.read_text())
    begun = time.monotonic()
    design = build(preflight["cutoff"])
    build_seconds = time.monotonic() - begun
    q = design.qualifying
    cutoff = design.catalog["cutoff"]
    year, last_round = map(int, cutoff.split("-"))
    event_id = f"{year}-{last_round + 1:02}"
    last_date = dt.date.fromisoformat(str(q.events.date.max())[:10])
    origin = str(last_date + dt.timedelta(days=1)) + "T00:00:00Z"
    field = q.entries[q.entries.event_id == cutoff].sort_values("driver_id")
    snap = dict(schema_version=1, information_set="prequalifying", origin_at=origin,
        source=dict(kind="retrospective_reconstruction", captured_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                    reference="Synthetic resource workload: hypothetical unchanged field and schedule; no real forecast"),
        event=dict(event_id=event_id, circuit_id=str(q.events.iloc[-1].circuit_id), scheduled_laps=60,
            qualifying_start=str(last_date + dt.timedelta(days=13)) + "T14:00:00Z",
            race_start=str(last_date + dt.timedelta(days=14)) + "T14:00:00Z"),
        calendar_after_cutoff=[event_id], entrants=[dict(driver_id=r.driver_id, team=r.team,
            prior_entries_after_cutoff=0) for r in field.itertuples()])
    catalog = build_catalog(design, snap)
    shape = (FIT_SETTINGS["chains"], FIT_SETTINGS["samples"] * max(row[1] for row in RETRY))
    scalars, vectors = parameter_template(design, observations=observations)
    total_bytes = int(np.prod(shape) * np.dtype("float32").itemsize * (len(scalars) + sum(vectors.values())))
    with tempfile.TemporaryDirectory(prefix="f1-overall-forecast-workload-") as directory:
        if shutil.disk_usage(directory).free < (4 if compressed else 2) * total_bytes:
            raise ValueError("Insufficient temporary disk for the synthetic workload")
        rng = np.random.default_rng(WORKLOAD_SEED)
        parameters = {name:np.full(shape, value, dtype=np.float32) for name, value in scalars.items()}
        generated_at = time.monotonic()
        for name, width in vectors.items():
            path = Path(directory) / (name + ".npy")
            array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape + (width,))
            for start in range(0, shape[1], DEFAULT_DRAW_BATCH):
                count = min(DEFAULT_DRAW_BATCH, shape[1] - start)
                array[:, start:start + count] = rng.normal(size=(shape[0], count, width))
            array.flush()
            del array
            parameters[name] = np.load(path, mmap_mode="r")
        generation_seconds = time.monotonic() - generated_at
        names = list(parameters)
        compression_seconds, compressed_bytes = 0., None
        if compressed:
            compression_at = time.monotonic()
            archive = Path(directory) / "posterior.npz"
            # This receipt is ONLY a temporary cache-format fixture. Its Boolean
            # convergence field exercises the read path; it is not a diagnostic,
            # fitted posterior, recovery result or publishable evidence.
            np.savez_compressed(archive, **parameters, unused_full_history=parameters["skill_z"])
            compressed_bytes = archive.stat().st_size
            receipt = dict(schema_version=1, state="converged", diagnostics=dict(converged=True, fixture_only=True),
                source=snapshot(), run_key="synthetic-archive-resource-fixture", posterior_file=archive.name,
                posterior_sha256=sha(archive), settings=dict(chains=shape[0]), attempts=[dict(samples=shape[1])],
                specification=dict(arrays_key=catalog["training_arrays_key"], synthetic=True),
                purpose="cache_format_fixture_only", publication_allowed=False)
            (Path(directory) / "receipt.json").write_text(json.dumps(receipt))
            for value in parameters.values():
                if isinstance(value, np.memmap): value._mmap.close()
            del parameters
            compression_seconds = time.monotonic() - compression_at
        stream, stop, batches, max_batch_bytes = hashlib.sha256(), 0, 0, 0
        parameter_key, observation_key = None, None
        loading_at = time.monotonic()
        with ExitStack() as stack:
            if compressed:
                parameters, _ = stack.enter_context(mapped_checked_posterior(directory, names=names,
                    expected_key="synthetic-archive-resource-fixture",
                    expected_arrays_key=catalog["training_arrays_key"], temporary_parent=directory))
            loading_seconds = time.monotonic() - loading_at
            forecast_at = time.monotonic()
            args = dict(seed=WORKLOAD_SEED, training_arrays_key=catalog["training_arrays_key"])
            if observations:
                scenario = {event_id:dict(wet=False, neutralised=False,
                    grid_policy="qualifying_order_no_penalties",
                    exposure_laps={r.driver_id:snap["event"]["scheduled_laps"] for r in field.itertuples()})}
                iterator = iter_conditional_weekends(design, [snap], parameters, scenario,
                    advancing=QUALIFYING_FIELDS[len(field)], **args)
            else:
                iterator = iter_forecast_states(design, [snap], parameters, **args)
            for batch in iterator:
                if batch["draw_range"][0] != stop:
                    raise ValueError("Incomplete forecast draw coverage")
                stop = batch["draw_range"][1]
                batches += 1
                state_key = batch["state_parameter_key"] if observations else batch["consumed_posterior_key"]
                if parameter_key not in (None, state_key):
                    raise ValueError("Posterior identity changed during the resource check")
                parameter_key = state_key
                if observations:
                    if observation_key not in (None, batch["observation_parameter_key"]):
                        raise ValueError("Observation parameters changed during the resource check")
                    observation_key = batch["observation_parameter_key"]
                arrays = dict(observation_arrays(batch["events"][0])) if observations else batch["events"][0]["states"]
                max_batch_bytes = max(max_batch_bytes, sum(value.nbytes for value in arrays.values()))
                for name, value in sorted(arrays.items()):
                    if not np.isfinite(value).all():
                        raise ValueError("Nonfinite workload output")
                    stream.update(name.encode())
                    stream.update(value.tobytes())
            if stop != shape[1]:
                raise ValueError("The resource check omitted posterior draws")
            forecast_seconds = time.monotonic() - forecast_at
            closing_at = time.monotonic()
        closing_seconds = time.monotonic() - closing_at
    # A conservative allocation estimate from the preceding implementation:
    # these historical entry-state arrays alone used float64 at this input dtype.
    entry_states = ["qualifying_walk", "race_residual", "qualifying_skill", "race_skill"]
    old_entry_bytes = int(len(entry_states) * np.prod(shape) * len(q.entries) * np.dtype("float64").itemsize)
    return dict(purpose="synthetic_forecast_resource_check", source_commit=source,
        source_hashes={name:sha(ROOT / name) for name in tracked}, limits=limits,
        recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(), cutoff=cutoff,
        preflight_reference_sha256=sha(preflight_path), training_arrays_key=catalog["training_arrays_key"],
        counts=dict(entries=len(q.entries), cars=len(q.cars), drivers=len(q.drivers), forecast_field=len(field)),
        workload=dict(posterior_shape=list(shape), dtype="float32", seed=WORKLOAD_SEED,
            parameter_bytes=total_bytes, parameters_sha256=parameter_key, draws_from_fixed_retry_ladder=True,
            archive_io="Temporary compressed cache-format fixture with checked source and checksum" if compressed else
                "Temporary uncompressed memory maps; compressed accepted-posterior loading is not exercised",
            compressed_bytes=compressed_bytes, synthetic_receipt_fixture=compressed,
            unused_archive_site_validated=compressed,
            construction="Arbitrary finite synthetic parameter values; not fitted values or a recovery experiment"),
        forecast=dict(kind="conditional_weekend" if observations else "latent_states",
            batches=batches, draws_per_chain=stop, max_batch_output_bytes=max_batch_bytes,
            observation_parameters_sha256=observation_key,
            batch_stream_sha256=stream.hexdigest(), every_output_finite=True,
            conditions_forecast=False, classification_generated=False),
        baseline_lower_bound=dict(source_revision="deb276e", entry_states=entry_states,
            entry_state_bytes=old_entry_bytes, parameter_plus_entry_state_bytes=total_bytes + old_entry_bytes,
            interpretation="Calculated lower bound, not a measured baseline peak; omits other history arrays, transients and process overhead"),
        seconds=dict(build_design=build_seconds, generate_workload=generation_seconds,
                     compress_workload=compression_seconds, load_and_validate=loading_seconds,
                     forecast=forecast_seconds, close_and_recheck=closing_seconds,
                     total=time.monotonic() - begun),
        process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        mcmc_run=False, fitted_parameters_used=False, real_predictions_scored=False,
        intended_ci_runner_verified=False, publication_allowed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--compressed", action="store_true", help="Exercise a temporary compressed cache-format fixture")
    parser.add_argument("--observations", action="store_true", help="Include conditional qualifying, order and completion observations")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Preserve the existing resource record")
    result = run(compressed=args.compressed, observations=args.observations)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key:result[key] for key in ("counts", "forecast", "seconds", "process_max_rss_kib")}))
