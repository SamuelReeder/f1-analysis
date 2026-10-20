"""Canonical candidate simulation moments; publication requires scientific acceptance.

The original research implementation and its evidence remain unchanged.
"""
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np

from f1rank.artifacts import digest

FILENAME = "diagnostic_moments.npz"
PURPOSE = "grouped_conditional_simulation_moments"
MEMBERS = {"replicate_counts", "points_mean", "points_sample_covariance",
           "probe_loss_mean", "probe_loss_sample_variance"}


def write_once(folder, metadata, arrays):
    """Create a numeric archive without replacing an existing diagnostic file."""
    folder = Path(folder)
    path = folder / FILENAME
    if set(arrays) != MEMBERS:
        raise ValueError("Incomplete diagnostic moment members")
    fd, temporary = tempfile.mkstemp(dir=folder, prefix=FILENAME+".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            np.savez_compressed(handle, **arrays)
        # Same-directory hard linking publishes a complete file exclusively.
        # A concurrent writer cannot be overwritten by an atomic replacement.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {**metadata, "file":FILENAME, "sha256":digest(path),
            "array_schema":{name:dict(shape=list(a.shape), dtype=a.dtype.str) for name,a in arrays.items()}}


def load_checked(folder, *, expected_simulation_key, expected_archive_sha256):
    """Verify a complete companion against caller-supplied content identities.

These consistency checks do not authenticate a fit or establish any numerical
or scientific acceptance. No new success flag is inferred from saved moments.
"""
    if not all(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
               for value in (expected_simulation_key, expected_archive_sha256)):
        raise ValueError("Expected simulation and archive identities are required")
    folder = Path(folder)
    receipt_path, summary_path, path = folder/"receipt.json", folder/"summary.json", folder/FILENAME
    receipt_key = digest(receipt_path)
    receipt = json.loads(receipt_path.read_text())
    if (not isinstance(receipt,dict) or receipt.get("state") != "complete" or type(receipt.get("synthetic")) is not bool
            or receipt.get("purpose") != "canonical_candidate_simulation"
            or receipt.get("publication_allowed") is not False or receipt.get("scientific_evidence_verified") is not False
            or receipt.get("simulation_key") != expected_simulation_key
            or receipt.get("summary_sha256") != digest(summary_path)
            or receipt.get("completed_blocks") != receipt.get("schedule", {}).get("blocks")):
        raise ValueError("Require a completed matching development simulation")
    summary = json.loads(summary_path.read_text())
    if not isinstance(summary,dict):
        raise ValueError("Invalid diagnostic summary")
    meta = summary.get("diagnostic_moments")
    flags = ("mcmc_precision_verified", "rank_precision_established", "scientific_evidence_verified", "publication_allowed")
    if (not isinstance(meta, dict) or receipt.get("diagnostic_moments") != meta
            or not isinstance(meta.get("identity"),dict) or not isinstance(meta.get("sources"),dict)
            or summary.get("purpose") != "canonical_overall_candidate_summary" or summary.get("synthetic") is not receipt["synthetic"]
            or meta.get("schema_version") != 1 or meta.get("purpose") != PURPOSE
            or meta.get("file") != FILENAME or meta.get("sha256") != expected_archive_sha256
            or digest(path) != expected_archive_sha256 or meta.get("identity") != summary.get("identity")
            or meta.get("identity", {}).get("simulation_key") != expected_simulation_key
            or meta.get("identity", {}).get("fit_id") != receipt.get("fit", {}).get("run_key")
            or meta.get("identity", {}).get("posterior_key") != receipt.get("fit", {}).get("posterior_sha256")
            or any(meta.get(key) is not False or summary.get(key) is not False for key in flags)
            or meta.get("units") != "points_per_grand_prix"
            or meta.get("conditional_covariance") != "unbiased_single_season_gp_average; divide_by_replicates_for_mean_error"
            or meta.get("probe_names") != ["qualifying_pace", "total_race_pace", "all"]
            or meta.get("field") != summary.get("field")
            or meta.get("posterior_shape") != summary.get("posterior_shape")
            or meta.get("replicates_per_draw") != summary.get("replicates_per_draw")
            or meta.get("n_races") != summary.get("n_races")
            or summary.get("complete_draw_and_replicate_inventory") is not True
            or any(meta.get("sources", {}).get(key) != summary.get(key) for key in (
                "definition_sha256", "implementation_sha256", "contract_sha256", "simulator_sources"))):
        raise ValueError("Diagnostic companion metadata, source or content identity disagrees")
    sources = meta.get("sources", {})
    fingerprints = [sources.get(key) for key in ("definition_sha256", "implementation_sha256", "contract_sha256")]
    dependencies = sources.get("simulator_sources")
    if (not isinstance(dependencies, dict) or not dependencies
            or not all(isinstance(key,str) and key for key in dependencies)
            or not all(isinstance(value,str) and re.fullmatch(r"[0-9a-f]{64}", value)
                       for value in [*fingerprints, *dependencies.values()])):
        raise ValueError("Incomplete diagnostic source fingerprints")
    shape, field, reps = meta["posterior_shape"], meta["field"], meta["replicates_per_draw"]
    if (not isinstance(shape, list) or len(shape) != 2 or not all(type(v) is int and v >= 2 for v in shape)
            or not isinstance(field, list) or len(field) < 2 or not all(isinstance(v,str) and v for v in field)
            or len(set(field)) != len(field) or type(reps) is not int or reps < 2
            or type(meta["n_races"]) is not int or meta["n_races"] < 1):
        raise ValueError("Invalid complete grouped diagnostic inventory")
    n = len(field)
    expected = dict(replicate_counts=tuple(shape), points_mean=tuple(shape)+(n,),
        points_sample_covariance=tuple(shape)+(n,n), probe_loss_mean=tuple(shape)+(n,3),
        probe_loss_sample_variance=tuple(shape)+(n,3))
    with np.load(path, allow_pickle=False) as saved:
        if len(saved.files) != len(MEMBERS) or set(saved.files) != MEMBERS:
            raise ValueError("Missing, duplicate or unknown diagnostic array")
        arrays = {name:saved[name] for name in MEMBERS}
    schema = meta.get("array_schema", {})
    if not isinstance(schema,dict) or set(schema) != MEMBERS:
        raise ValueError("Incomplete diagnostic array schema")
    for name, value in arrays.items():
        kind = "iu" if name == "replicate_counts" else "f"
        if (value.shape != expected[name] or value.dtype.kind not in kind or not np.isfinite(value).all()
                or schema[name] != dict(shape=list(value.shape), dtype=value.dtype.str)):
            raise ValueError("Diagnostic array axes, type or values disagree")
    cov = arrays["points_sample_covariance"]
    scale = np.maximum(1., np.max(np.abs(cov), axis=(-2,-1)))
    tolerance = 64*np.finfo(float).eps*n*scale
    if (not np.all(arrays["replicate_counts"] == reps)
            or (arrays["probe_loss_sample_variance"] < 0).any()
            or (np.diagonal(cov, axis1=-2, axis2=-1) < 0).any()
            or (np.max(np.abs(cov-cov.swapaxes(-1,-2)), axis=(-2,-1)) > tolerance).any()
            or (np.linalg.eigvalsh((cov+cov.swapaxes(-1,-2))/2)[...,0] < -tolerance).any()):
        raise ValueError("Diagnostic counts or conditional covariance are inconsistent")
    if digest(receipt_path) != receipt_key or digest(path) != expected_archive_sha256 or digest(summary_path) != receipt["summary_sha256"]:
        raise ValueError("Diagnostic artifacts changed while reading")
    for value in arrays.values():
        value.flags.writeable = False
    return meta, arrays
