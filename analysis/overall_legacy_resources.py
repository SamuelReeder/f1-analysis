"""Legacy archive/forecast workload with synthetic draws at current data size.

No sampling, accepted fits or real predictions. The largest existing retry shape
is a resource workload, not a selection of future baseline sampler settings.
"""
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
from analysis.overall_forecast_resources import scope_limits, sha, WORKLOAD_SEED
from analysis.overall_forecast_states import DEFAULT_DRAW_BATCH
from analysis.overall_legacy_archive import mapped_legacy_posterior
from analysis.overall_legacy_forecast import forecast_legacy_features, forecast_legacy_race
from f1rank.artifacts import RETRY
from f1rank.design import build_design, load_tables, STAGE1_TABLES
from f1rank.fit import INIT_HYPER, meta_path, save
from f1rank.overall.fitting import FIT_SETTINGS


def parameter_template(q):
    scalars = {name:INIT_HYPER[name] for name in (
        "sd_level", "sd_skill_race", "sd_skill_season", "exp_gain", "exp_scale", "age_slope",
        "rho", "rho_reset", "sd_car_race", "sd_car_season", "sd_car_reset", "sd_track_load",
        "sd_compat", "sd_driver_form", "sigma0", "sigma_tau", "nu", "sd_car_segment")}
    scalars["sd_car_event"] = .15
    required = dict(level_z=q["n_drivers"], skill_z=q["n_entries"], car_step_z=q["n_cars"],
        car_jump_z=q["n_cars"], track_load_z=q["n_team_seasons"], compat_z=q["n_stints"])
    # Historical blocks are validated but not mapped by the forecast reader.
    # Values are arbitrary: these are shape/format fixtures, not model draws.
    historical = dict(skill=q["n_entries"], compat=q["n_entries"], driver_form=q["n_entries"],
        car=q["n_cars"], car_track=q["n_cars"], car_event=q["n_cars"],
        car_segment=q["n_car_segments"], track_load=q["n_team_seasons"],
        mu=q["n_sessions"], sigma_u=q["n_sessions"])
    return scalars, required, historical


def forecast_digest(design, snap, posterior, metadata, training_key, shape):
    result = forecast_legacy_features(design, [snap], posterior,
        cutoff=str(design.events.event_id.max()), fit_metadata=metadata,
        seed=WORKLOAD_SEED, training_arrays_key=training_key)
    event = result["events"][0]
    # Hypothetical full running panel and coefficient arrays, not real outcomes.
    panel = {event["event_id"]:{name:i + 1 for i, name in enumerate(event["drivers"])}}
    coefficients = {name:np.full(shape, value) for name,value in
                    dict(c_grid=.8, a_car=.6, b_driver=1.).items()}
    race = forecast_legacy_race(result, coefficients, panel)
    digest = hashlib.sha256()
    arrays = {**event["states"], **{f"feature_{n}":v for n,v in event["race_features"].items()},
              "race_log_probabilities":race["log_probabilities"]}
    for name,value in sorted(arrays.items()):
        if not np.isfinite(value).all(): raise ValueError("Nonfinite synthetic forecast output")
        digest.update(f"{name}:{value.shape}:{value.dtype.str}".encode())
        digest.update(value.tobytes())
    if result["posterior_shape"] != list(shape) or any(v.shape[:2] != shape for v in event["states"].values()):
        raise ValueError("Forecast did not retain every chain/draw")
    return dict(output_sha256=digest.hexdigest(), consumed_posterior_key=result["consumed_posterior_key"],
        coefficient_parameter_key=race["coefficient_parameter_key"], every_output_finite=True,
        posterior_shape=list(shape), full_draw_coverage=True,
        output_bytes=sum(v.nbytes for v in arrays.values()), field_size=len(event["drivers"]))


def run():
    limits = scope_limits()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked = ["analysis/overall_legacy_resources.py", "analysis/overall_legacy_archive.py",
        "analysis/overall_legacy_forecast.py", "analysis/overall_posterior_archive.py",
        "analysis/overall_forecast_resources.py", "analysis/overall_forecast_catalog.py",
        "analysis/overall_forecast_states.py", "analysis/overall_baselines.py",
        "f1rank/fit.py", "f1rank/design.py", "f1rank/model.py", "f1rank/lineage.py",
        "f1rank/artifacts.py", "f1rank/overall/fitting.py", "f1rank/overall/order.py",
        "outputs/championship/development/interface_preflight.json",
        *[f"data/processed/{name}.parquet" for name in STAGE1_TABLES]]
    hashes = {name:sha(ROOT/name) for name in tracked}
    for name in tracked:
        saved = subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=ROOT)
        if hashlib.sha256(saved).hexdigest() != hashes[name]:
            raise ValueError("Commit workload inputs before measuring")
    started = time.monotonic()
    preflight = ROOT/"outputs/championship/development/interface_preflight.json"
    cutoff = json.loads(preflight.read_text())["cutoff"]
    tables = {name:(f.loc[f.event_id <= cutoff].copy() if "event_id" in f else f.copy())
              for name,f in load_tables().items()}
    design = build_design(2010, end_event=cutoff, tables=tables)
    build_seconds = time.monotonic() - started
    q = design.arrays()
    field = design.entries[design.entries.event_id == cutoff].sort_values("driver_id")
    last = dt.date.fromisoformat(str(design.events.date.max())[:10])
    year, previous_round = map(int, cutoff.split("-")); event_id = f"{year}-{previous_round+1:02}"
    snap = dict(schema_version=1, information_set="actual_grid",
        origin_at=str(last+dt.timedelta(days=13))+"T18:00:00Z",
        source=dict(kind="retrospective_reconstruction", captured_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            reference="Synthetic resource workload: hypothetical field, grid and schedule; no real forecast"),
        event=dict(event_id=event_id, circuit_id=str(design.events.iloc[-1].circuit_id), scheduled_laps=60,
            qualifying_start=str(last+dt.timedelta(days=13))+"T14:00:00Z",
            race_start=str(last+dt.timedelta(days=14))+"T14:00:00Z"),
        calendar_after_cutoff=[event_id], entrants=[dict(driver_id=r.driver_id,team=r.team,
            prior_entries_after_cutoff=0,grid_position=i+1 if i < len(field)-1 else 0)
            for i,r in enumerate(field.itertuples())])
    catalog = build_qualifying_catalog(design, snap, cutoff=cutoff)
    shape = (FIT_SETTINGS["chains"], FIT_SETTINGS["samples"] * max(row[1] for row in RETRY))
    scalars, required, historical = parameter_template(q)
    vectors = {**required, **historical}
    total_bytes = int(np.prod(shape)*np.dtype("float32").itemsize*(len(scalars)+sum(vectors.values())))
    with tempfile.TemporaryDirectory(prefix="f1-legacy-resource-") as folder:
        directory = Path(folder)
        if shutil.disk_usage(directory).free < 4*total_bytes:
            raise ValueError("Insufficient temporary disk for the synthetic workload")
        generate_at = time.monotonic(); rng = np.random.default_rng(WORKLOAD_SEED)
        parameters = {name:np.full(shape,value,dtype=np.float32) for name,value in scalars.items()}
        for name,width in vectors.items():
            path = directory/(name+".npy")
            value = np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=shape+(width,))
            for start in range(0,shape[1],DEFAULT_DRAW_BATCH):
                count = min(DEFAULT_DRAW_BATCH,shape[1]-start)
                value[:,start:start+count] = rng.normal(size=(shape[0],count,width))
            value.flush(); value._mmap.close()
            parameters[name] = np.load(path,mmap_mode="r",allow_pickle=False)
        generate_seconds = time.monotonic()-generate_at
        first_at = time.monotonic()
        direct = forecast_digest(design,snap,parameters,dict(start_season=2010,model_kw={}),catalog["training_arrays_key"],shape)
        direct_seconds = time.monotonic()-first_at
        archive = directory/"legacy.npz"; compress_at = time.monotonic()
        save(archive,parameters,dict(elapsed_s=0.,divergences=0,mean_steps=0.),design,
            model_kw={},settings=dict(chains=shape[0],samples=shape[1]),
            purpose="synthetic_legacy_archive_workload",fixture_only=True,publication_allowed=False)
        for value in parameters.values():
            if isinstance(value,np.memmap): value._mmap.close()
        del parameters
        compression_seconds = time.monotonic()-compress_at
        archive_bytes = archive.stat().st_size
        identities = dict(expected_archive_sha256=sha(archive),expected_metadata_sha256=sha(meta_path(archive)),
            expected_training_arrays_key=catalog["training_arrays_key"])
        loading_at = time.monotonic()
        with mapped_legacy_posterior(archive,design,names=[*scalars,*required],
                temporary_parent=directory,**identities) as (mapped,info,receipt):
            loading_seconds = time.monotonic()-loading_at
            forecast_at = time.monotonic()
            restored = forecast_digest(design,snap,mapped,receipt["metadata"],catalog["training_arrays_key"],shape)
            forecast_seconds = time.monotonic()-forecast_at
            if direct != restored: raise ValueError("Compressed legacy read changed the synthetic forecast")
            handles = [value._mmap for value in mapped.values()]
            closing_at = time.monotonic()
        closing_seconds = time.monotonic()-closing_at
        if not all(h.closed for h in handles) or list(directory.glob("f1-overall-posterior-*")):
            raise ValueError("Legacy reader did not close and remove its maps")
    if hashes != {name:sha(ROOT/name) for name in tracked}:
        raise ValueError("Workload inputs changed during measurement")
    return dict(purpose="synthetic_legacy_archive_forecast_resource_check",source_commit=source,
        source_hashes=hashes,recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(),limits=limits,
        cutoff=cutoff,training_arrays_key=catalog["training_arrays_key"],
        counts=dict(entries=len(design.entries),cars=len(design.cars),drivers=len(design.drivers)),
        workload=dict(seed=WORKLOAD_SEED,posterior_shape=list(shape),dtype="float32",
            parameter_bytes=total_bytes,compressed_bytes=archive_bytes,draw_batch_size=DEFAULT_DRAW_BATCH,
            historical_members_validated=sorted(historical),posterior_members=len(scalars)+len(vectors),
            construction="Arbitrary synthetic values at legacy archive state dimensions; no fitted posterior",
            draw_shape_reference="Largest existing Overall retry shape; not future baseline sampler settings"),
        forecast=restored,compressed_and_direct_forecasts_identical=True,temporary_maps_removed=True,
        seconds=dict(build_design=build_seconds,generate_workload=generate_seconds,direct_forecast=direct_seconds,
            compress_workload=compression_seconds,load_and_validate=loading_seconds,forecast=forecast_seconds,
            close_and_recheck=closing_seconds,total=time.monotonic()-started),
        process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        mcmc_run=False,accepted_fit_loaded=False,real_predictions_scored=False,
        intended_ci_runner_verified=False,publication_allowed=False)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",required=True)
    args=parser.parse_args(); output=Path(args.output)
    if output.exists(): raise FileExistsError("Preserve the existing resource record")
    result=run(); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({k:result[k] for k in ("counts","forecast","seconds","process_max_rss_kib")}))
