"""Synthetic full-product legacy forecast workload; no fitting or real scores."""
import argparse
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

from analysis.overall_forecast_catalog import build_qualifying_catalog
from analysis.overall_forecast_resources import WORKLOAD_SEED, scope_limits, sha
from analysis.overall_forecast_states import DEFAULT_DRAW_BATCH
from analysis.overall_legacy_prequalifying import (
    GRID_POLICY, forecast_legacy_qualifying, integrate_legacy_prequalifying)
from analysis.overall_legacy_resources import parameter_template
from f1rank.artifacts import RETRY
from f1rank.design import STAGE1_TABLES, build_design, load_tables
from f1rank.overall.fitting import FIT_SETTINGS, arrays_digest
from f1rank.overall.simulation import QUALIFYING_FIELDS


def run():
    limits = scope_limits()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    # Include all local Python modules in the two importing trees, plus the
    # reference shape, data and probability/workload definitions.
    tracked = subprocess.check_output(["git", "ls-files", "analysis/*.py", "f1rank/*.py"],
                                      cwd=ROOT, text=True).splitlines()
    tracked += ["docs/overall_legacy_prequalifying.md",
        "docs/overall_legacy_prequalifying_resources.md",
        "outputs/championship/development/interface_preflight.json",
        *[f"data/processed/{name}.parquet" for name in STAGE1_TABLES]]
    hashes = {name: sha(ROOT / name) for name in tracked}
    for name in tracked:
        saved = subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=ROOT)
        if hashlib.sha256(saved).hexdigest() != hashes[name]:
            raise ValueError("Commit workload inputs before measuring")
    begun = time.monotonic()
    cutoff = json.loads((ROOT / "outputs/championship/development/interface_preflight.json").read_text())["cutoff"]
    tables = {name: (frame.loc[frame.event_id <= cutoff].copy() if "event_id" in frame else frame.copy())
              for name, frame in load_tables().items()}
    design = build_design(2010, end_event=cutoff, tables=tables)
    q = design.arrays()
    field = design.entries[design.entries.event_id == cutoff].sort_values("driver_id")
    last = dt.date.fromisoformat(str(design.events.date.max())[:10])
    year, previous_round = map(int, cutoff.split("-"))
    event_id = f"{year}-{previous_round + 1:02}"
    origin = str(last + dt.timedelta(days=1)) + "T00:00:00Z"
    snap = dict(schema_version=1, information_set="prequalifying", origin_at=origin,
        source=dict(kind="retrospective_reconstruction", captured_at=origin,
            reference="Synthetic resource fixture: hypothetical future field and schedule; no real forecast"),
        event=dict(event_id=event_id, circuit_id=str(design.events.iloc[-1].circuit_id), scheduled_laps=60,
            qualifying_start=str(last + dt.timedelta(days=13)) + "T14:00:00Z",
            race_start=str(last + dt.timedelta(days=14)) + "T14:00:00Z"),
        calendar_after_cutoff=[event_id], entrants=[dict(driver_id=row.driver_id, team=row.team,
            prior_entries_after_cutoff=0) for row in field.itertuples()])
    catalog = build_qualifying_catalog(design, snap, cutoff=cutoff)
    build_seconds = time.monotonic() - begun
    shape = (FIT_SETTINGS["chains"], FIT_SETTINGS["samples"] * max(row[1] for row in RETRY))
    scalars, vectors, _ = parameter_template(q)
    total_bytes = int(np.prod(shape) * np.dtype("float32").itemsize * (len(scalars) + sum(vectors.values())))
    with tempfile.TemporaryDirectory(prefix="f1-legacy-prequalifying-resource-") as folder:
        directory = Path(folder)
        if shutil.disk_usage(directory).free < 2 * total_bytes:
            raise ValueError("Insufficient temporary disk for the synthetic workload")
        generate_at = time.monotonic()
        rng = np.random.default_rng(WORKLOAD_SEED)
        parameters = {name: np.full(shape, value, dtype=np.float32) for name, value in scalars.items()}
        try:
            for name, width in vectors.items():
                path = directory / (name + ".npy")
                value = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=shape + (width,))
                for start in range(0, shape[1], DEFAULT_DRAW_BATCH):
                    count = min(DEFAULT_DRAW_BATCH, shape[1] - start)
                    value[:, start:start + count] = rng.normal(size=(shape[0], count, width))
                value.flush()
                value._mmap.close()
                parameters[name] = np.load(path, mmap_mode="r", allow_pickle=False)
            # Vary all three coordinates, with within-vector dependence. These
            # are arbitrary fixture values, not draws from fitted coefficients.
            z = rng.normal(size=shape + (3,))
            coefficients = dict(c_grid=np.exp(-.2 + .2 * z[..., 0]),
                a_car=.6 + .2 * z[..., 0] + .1 * z[..., 1],
                b_driver=1. + .1 * z[..., 1] + .2 * z[..., 2])
            if len(np.unique(np.stack(list(coefficients.values()), axis=-1).reshape(-1, 3), axis=0)) != np.prod(shape):
                raise ValueError("Synthetic coefficient vectors must all be distinct")
            generate_seconds = time.monotonic() - generate_at
            forecast_at = time.monotonic()
            forecast = forecast_legacy_qualifying(design, [snap], parameters, cutoff=cutoff,
                fit_metadata=dict(start_season=2010, model_kw={}), seed=WORKLOAD_SEED,
                training_arrays_key=catalog["training_arrays_key"],
                advancing={event_id: QUALIFYING_FIELDS[len(field)]}, grid_policy=GRID_POLICY)
            forecast_seconds = time.monotonic() - forecast_at
        finally:
            for value in parameters.values():
                if isinstance(value, np.memmap):
                    value._mmap.close()
        del parameters
    event = forecast["events"][0]
    if (forecast["posterior_shape"] != list(shape) or event["grid_draws"].shape != shape + (len(field),)
            or any(value.shape[:2] != shape or not np.isfinite(value).all() for value in event["states"].values())):
        raise ValueError("Forecast did not retain complete finite chain/draw states and grids")
    panel = {event_id: {driver: i + 1 for i, driver in enumerate(event["drivers"])}}
    unique_grids = len(np.unique(event["grid_draws"].reshape(-1, len(field)), axis=0))
    integrations, integration_seconds = {}, {}
    for arm in ("legacy_grid_qualifying", "grid_only"):
        started = time.monotonic()
        result = integrate_legacy_prequalifying(forecast, coefficients, panel, arm=arm)
        integration_seconds[arm] = time.monotonic() - started
        probability = result["log_probabilities"]
        if (result["qualifying_posterior_shape"] != list(shape)
                or result["coefficient_posterior_shape"] != list(shape)
                or result["unique_grid_counts"] != [unique_grids]
                or result["integration"] != "full_empirical_product"
                or not np.isfinite(probability).all() or (probability > 0).any()):
            raise ValueError("Full empirical integration failed its coverage or finite-mass checks")
        integrations[arm] = dict(log_probabilities=probability.tolist(),
            output_sha256=arrays_digest(dict(result=dict(log_probabilities=probability))),
            coefficient_parameter_key=result["coefficient_parameter_key"],
            full_product_pairs=int(np.prod(shape)) ** 2,
            evaluated_unique_grid_coefficient_pairs=unique_grids * int(np.prod(shape)),
            every_grid_and_coefficient_retained=True,
            acceptance_flags={key: result[key] for key in (
                "source_availability_verified", "convergence_verified", "forecast_precision_verified",
                "official_grid_policy_complete", "independent_evidence_established", "publication_allowed")})
    if hashes != {name: sha(ROOT / name) for name in tracked}:
        raise ValueError("Workload inputs changed during measurement")
    return dict(purpose="synthetic_legacy_prequalifying_full_product_resource_check",
        source_commit=source, source_hashes=hashes, limits=limits,
        recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(), cutoff=cutoff,
        training_arrays_key=catalog["training_arrays_key"],
        counts=dict(entries=len(design.entries), cars=len(design.cars), drivers=len(design.drivers), field=len(field)),
        workload=dict(seed=WORKLOAD_SEED, qualifying_shape=list(shape), coefficient_shape=list(shape),
            parameter_bytes=total_bytes, parameter_dtype="float32", coefficient_dtype="float64",
            draw_batch_size=DEFAULT_DRAW_BATCH, grid_batch_size=DEFAULT_DRAW_BATCH,
            coefficient_batch_size=DEFAULT_DRAW_BATCH, advancing=list(QUALIFYING_FIELDS[len(field)]),
            unique_grids=unique_grids, unique_joint_coefficient_vectors=int(np.prod(shape)),
            construction="Arbitrary varying fixture values, hypothetical future event and full synthetic response panel",
            draw_shape_reference="Largest existing retry shape for a workload; no baseline sampler settings selected"),
        forecast=dict(consumed_posterior_key=forecast["consumed_posterior_key"],
            observation_key=forecast["qualifying_observation_key"], grid_draws_key=forecast["grid_draws_key"],
            future_features_key=forecast["future_features_key"], full_draw_coverage=True),
        integrations=integrations,
        seconds=dict(build_design=build_seconds, generate_workload=generate_seconds,
            qualifying_forecast=forecast_seconds, integrations=integration_seconds, total=time.monotonic() - begun),
        process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        temporary_maps_removed=True, mcmc_run=False, accepted_fit_loaded=False,
        real_predictions_scored=False, forecast_precision_verified=False,
        intended_ci_runner_verified=False, publication_allowed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    output = Path(parser.parse_args().output)
    if output.exists():
        raise FileExistsError("Preserve the existing resource record")
    result = run()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: result[key] for key in ("counts", "workload", "seconds", "process_max_rss_kib")}))
