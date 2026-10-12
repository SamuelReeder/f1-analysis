"""Capped synthetic storage-format workload; no fit or canonical simulation."""
import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.overall_ranking_resources import scope_limits, sha
from analysis.overall_simulation_controller import SOURCE_NAMES, runtime_identity
from analysis.overall_simulation_moments import load_checked, write_once
from analysis.overall_simulation_resources import arrays_sha256
from analysis.overall_summary import _source_fingerprints
from f1rank.artifacts import RETRY, atomic_json
from f1rank.overall.fitting import FIT_SETTINGS
from f1rank.overall.simulation import MAX_EXPECTED_POINTS_MCSE, POINTS

SEED = 20261012
CLASS_PROBABILITIES = np.array([.2, .3, .5])


def run(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    report = dict(purpose="synthetic_grouped_moment_storage_resources", status="running",
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(), mcmc_run=False,
        canonical_execution_run=False, full_posterior_workload_executed=False,
        self_supplied_format_fixture_metadata=True, scientific_evidence_verified=False, publication_allowed=False)
    begun = time.monotonic()
    try:
        report["limits"] = scope_limits()
        names = sorted(set(SOURCE_NAMES) | {"analysis/overall_moment_resources.py",
            "analysis/overall_ranking_resources.py", "analysis/overall_simulation_resources.py",
            "docs/overall_moment_resources.md", "docs/overall_rotation_profile_proposal.json",
            "outputs/championship/development/interface_preflight.json", "data/processed/race.parquet"})
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
        hashes = {name:sha(ROOT/name) for name in names}
        for name, value in hashes.items():
            saved = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
            if hashlib.sha256(saved).hexdigest() != value:
                raise ValueError("Commit the resource workload and inputs before measurement")
        runtime = runtime_identity()
        report.update(source_commit=commit, source_hashes=hashes, runtime=runtime)
        preflight = json.loads((ROOT/"outputs/championship/development/interface_preflight.json").read_text())
        if preflight["source"]["inputs"]["data/processed/race.parquet"] != hashes["data/processed/race.parquet"]:
            raise ValueError("Field reference no longer matches the committed race data")
        n = preflight["counts"]["field"]
        shape = (FIT_SETTINGS["chains"], FIT_SETTINGS["samples"]*max(row[1] for row in RETRY))
        reps = 2*math.ceil((float(np.max(POINTS))/MAX_EXPECTED_POINTS_MCSE)**2/2)
        arrays = dict(replicate_counts=np.full(shape,reps,dtype=np.int64), points_mean=np.empty(shape+(n,)),
            points_sample_covariance=np.empty(shape+(n,n)), probe_loss_mean=np.empty(shape+(n,3)),
            probe_loss_sample_variance=np.empty(shape+(n,3)))
        point_table = np.zeros(n)
        point_table[:min(n,len(POINTS))] = POINTS[:n]
        rng = np.random.default_rng(SEED)
        started = time.monotonic()
        for c in range(shape[0]):
            orders = np.argsort(rng.uniform(size=(shape[1],len(CLASS_PROBABILITIES),n)),axis=-1)
            scores = point_table[orders]
            counts = rng.multinomial(reps,CLASS_PROBABILITIES,size=shape[1])
            mean = (scores*counts[...,None]).sum(axis=1)/reps
            centered = scores-mean[:,None,:]
            arrays["points_mean"][c] = mean
            arrays["points_sample_covariance"][c] = np.einsum("dkn,dkm,dk->dnm",centered,centered,counts)/(reps-1)
            for probe in range(3):
                losses = scores-np.roll(scores,probe+1,axis=-1)
                loss_mean = (losses*counts[...,None]).sum(axis=1)/reps
                arrays["probe_loss_mean"][c,...,probe] = loss_mean
                arrays["probe_loss_sample_variance"][c,...,probe] = ((losses-loss_mean[:,None,:])**2*counts[...,None]).sum(axis=1)/(reps-1)
        report["construction_seconds"] = time.monotonic()-started
        identity = dict(fit_id="storage-format-fixture-no-fit", posterior_key="1"*64,
            scenario_key="2"*64, simulation_key="3"*64)
        meta = dict(schema_version=1,purpose="grouped_conditional_simulation_moments",identity=identity,
            field=[f"synthetic_{i:03}" for i in range(n)],posterior_shape=list(shape),replicates_per_draw=reps,
            n_races=1,probe_names=["qualifying_pace","total_race_pace","all"],units="points_per_grand_prix",
            sources=_source_fingerprints(),
            conditional_covariance="unbiased_single_season_gp_average; divide_by_replicates_for_mean_error",
            mcmc_precision_verified=False,rank_precision_established=False,scientific_evidence_verified=False,publication_allowed=False)
        before = arrays_sha256(arrays)
        with tempfile.TemporaryDirectory(prefix="f1-diagnostic-moment-format-") as directory:
            folder = Path(directory)
            started = time.monotonic()
            metadata = write_once(folder,meta,arrays)
            report["write_seconds"] = time.monotonic()-started
            summary = {key:value for key,value in meta.items() if key not in ("schema_version","sources","purpose")}
            summary.update(purpose="development_canonical_overall_summary",synthetic=True,**meta["sources"],
                complete_draw_and_replicate_inventory=True,diagnostic_moments=metadata)
            atomic_json(folder/"summary.json",summary)
            receipt = dict(purpose="synthetic_canonical_simulation",state="complete",synthetic=True,
                simulation_key=identity["simulation_key"],summary_sha256=sha(folder/"summary.json"),
                completed_blocks=1,schedule=dict(blocks=1),diagnostic_moments=metadata,
                fit=dict(run_key=identity["fit_id"],posterior_sha256=identity["posterior_key"]),
                scientific_evidence_verified=False,publication_allowed=False,storage_format_fixture=True)
            atomic_json(folder/"receipt.json",receipt)
            started = time.monotonic()
            loaded_metadata, loaded = load_checked(folder,expected_simulation_key=identity["simulation_key"],
                expected_archive_sha256=metadata["sha256"])
            report["read_seconds"] = time.monotonic()-started
            if loaded_metadata != metadata or any(not np.array_equal(arrays[key],value) or value.flags.writeable for key,value in loaded.items()):
                raise ValueError("Checked round-trip changed diagnostic arrays or write protection")
            report.update(archive_bytes=(folder/metadata["file"]).stat().st_size,archive_sha256=metadata["sha256"])
        if before != arrays_sha256(arrays):
            raise ValueError("Storage validation mutated its supplied moments")
        if hashes != {name:sha(ROOT/name) for name in names} or runtime != runtime_identity():
            raise ValueError("Bound resource input or runtime changed")
        report.update(status="complete",posterior_shape=list(shape),field_size=n,reference_cutoff=preflight["cutoff"],
            replicates=reps,seed=SEED,class_probabilities=CLASS_PROBABILITIES.tolist(),
            array_bytes=sum(a.nbytes for a in arrays.values()),array_values_sha256=before,
            checked_round_trip_exact=True,loaded_arrays_read_only=True,temporary_fixture_removed=not folder.exists(),
            limitations=["Constructed finite-mixture point moments with placeholder fit identities; not a canonical posterior or simulation run",
                "The temporary self-supplied receipt checks consistency only; it is not scientific provenance",
                "Local storage/reading cost only, not full fitting, nested simulation, scientific acceptance or intended-runner resources"])
    except BaseException as exc:
        report.update(status="failed",error=repr(exc),traceback=traceback.format_exc())
        raise
    finally:
        report.update(finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),total_monotonic_seconds=time.monotonic()-begun,
            process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        atomic_json(output_dir/"resources.json",report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir",required=True)
    result = run(parser.parse_args().output_dir)
    print(json.dumps({key:result[key] for key in ("status","posterior_shape","field_size","array_bytes","archive_bytes","process_max_rss_kib")}))
