"""Independent acceptance evidence. Recorded success flags are not decisions.

The convergence evaluator is implemented here. The remaining scientific checks
must supply their own registered, verified evidence before publication exists.
This module never advances a public pointer or authorizes another fit attempt.
"""
from contextlib import contextmanager
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import time
import traceback
import zipfile

import numpy as np

from ..artifacts import atomic_json, digest
from . import fitting, inference
from .bounded_diagnostics import check
from .posterior_archive import _header, _mapped_numeric_archive
from .simulation_controller import (ROOT, design_identity, require_committed_sources,
                                    require_fit_registration, source_fingerprints)

DEFINITION = "docs/overall_acceptance.md"
PURPOSE = "overall_convergence_evidence"


def retained_inventory(blocks):
    """Derive every retained sample/deterministic site's shape from the model.

    Prior draws supply shapes only. No MCMC, fitted values, observed ranking,
    diagnostic outcome or caller-supplied site list selects this inventory.
    """
    import jax
    import jax.numpy as jnp
    from numpyro import handlers
    from .control_rotation import build_rotation
    from .batched_rotation import prepare
    from .weekend_coordinates import weekend_model
    model = weekend_model(prepare(build_rotation(blocks["laps"])))
    args = tuple({k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in blocks[key].items()}
                 for key in ("qualifying", "laps", "outcomes"))
    try:
        trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(0))).get_trace(*args)
        return {name: dict(shape=list(np.shape(site["value"])), dtype=np.asarray(site["value"]).dtype.str)
                for name, site in trace.items() if site["type"] == "deterministic"
                or (site["type"] == "sample" and not site["is_observed"])}
    finally:
        jax.clear_caches()


def _canonical_diagnostics(report):
    result = copy.deepcopy(report)
    for key in ("parameters", "constants", "empty_sites"):
        if key in result:
            result[key] = sorted(result[key], key=lambda row: row["parameter"])
    if "failures" in result:
        result["failures"] = sorted(result["failures"])
    return result


@contextmanager
def complete_posterior(archive, receipt, inventory, *, temporary_parent=None):
    """Reject omitted/extra fields, even if a receipt repeats the wrong list."""
    attempt = receipt["attempts"][-1]
    axes = (attempt["chains"], attempt["samples"])
    expected = {name: dict(shape=list(axes) + spec["shape"], dtype=spec["dtype"])
                for name, spec in inventory.items()}
    if not expected or attempt.get("posterior_sites") != expected:
        raise ValueError("Recorded posterior inventory differs from the complete model")
    with zipfile.ZipFile(archive) as saved:
        members = saved.infolist()
        names = [m.filename[:-4] for m in members]
        if (any(not m.filename.endswith(".npy") for m in members) or len(names) != len(set(names))
                or set(names) != set(expected)):
            raise ValueError("Archive omits or adds a model posterior field")
        for name, member in zip(names, members):
            with saved.open(member) as stream:
                shape, _, dtype, _ = _header(stream, member, axes)
            if expected[name] != dict(shape=list(shape), dtype=dtype.str):
                raise ValueError("Archive posterior shape or dtype differs from the model")
    with _mapped_numeric_archive(archive, names=list(expected), axes=axes,
                                 temporary_parent=temporary_parent) as (posterior, _):
        yield posterior


def evaluate_convergence(run_dir, design, registration_path, output_dir, *, registration_root=ROOT,
                         temporary_parent=None):
    """Recompute the final fit's complete numerical and driver-contrast reports.

    A consistent failed numerical result is retained with passed=false. Invalid
    identities or incomplete evidence produce a failed-exception record. Neither
    result permits another sampler attempt, scientific acceptance or publication.
    """
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    registration_root = Path(registration_root).resolve()
    registration_path = Path(registration_path).resolve()
    relative = registration_path.relative_to(registration_root).as_posix()
    if output_dir.exists():
        raise FileExistsError("Preserve the existing convergence evidence, including failures")
    sources = dict(source_fingerprints(), **{DEFINITION: digest(ROOT / DEFINITION)})
    require_committed_sources(sources)
    identity = design_identity(design)
    blocks = dict(qualifying=design.qualifying.arrays(), laps=design.laps, outcomes=design.outcomes)
    registration_sha = digest(registration_path)
    receipt_sha = digest(run_dir / "receipt.json")
    original = json.loads((run_dir / "receipt.json").read_bytes())
    protocol = dict(fit_registration=dict(path=relative, sha256=registration_sha),
                    synthetic=original["specification"].get("synthetic"))
    if type(protocol["synthetic"]) is not bool:
        raise ValueError("The fit must retain its explicit provenance role")
    require_fit_registration(protocol, original, root=registration_root)
    spec = original["specification"]
    if (spec.get("arrays_key") != identity["arrays_key"] or spec.get("catalog") != design.catalog
            or original.get("catalog") != design.catalog or spec.get("source_key") != original["source"]["key"]
            or hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest() != original["run_key"]):
        raise ValueError("Convergence design and fit identities disagree")
    record = dict(purpose=PURPOSE, state="running", passed=False, sources=sources,
        fit_id=original["run_key"], posterior_sha256=original["posterior_sha256"], fit_receipt_sha256=receipt_sha,
        registration_sha256=registration_sha, synthetic=protocol["synthetic"],
        arrays_key=identity["arrays_key"], catalog=copy.deepcopy(design.catalog),
        started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        scientific_acceptance_established=False, publication_allowed=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(output_dir / "receipt.json", record)
    started = time.monotonic()
    def unchanged():
        if (digest(run_dir / "receipt.json") != receipt_sha or digest(registration_path) != registration_sha
                or design_identity(design) != identity
                or sources != dict(source_fingerprints(), **{DEFINITION: digest(ROOT / DEFINITION)})):
            raise ValueError("Convergence evidence inputs changed during evaluation")
    try:
        inventory = retained_inventory(blocks)
        unchanged()
        with fitting.checked_archive(run_dir, original["run_key"]) as (archive, receipt):
            if receipt != original:
                raise ValueError("Fit receipt changed before convergence evaluation")
            with complete_posterior(archive, receipt, inventory, temporary_parent=temporary_parent) as post:
                last = receipt["attempts"][-1]
                extra_path = run_dir / last["sampler_file"]
                with np.load(extra_path, allow_pickle=False) as saved:
                    extras = {name: saved[name] for name in saved.files}
                inference.validate_extras(extras, last)
                numerical = check(post, int(extras["diverging"].sum()))
                comparisons = inference.contrast_reports(post, design.catalog, numerical["divergences"])
                recorded = last["diagnostics"]
                numerical_match = _canonical_diagnostics({k: recorded.get(k) for k in numerical}) == _canonical_diagnostics(numerical)
                contrasts_match = recorded.get("comparison_reports") == comparisons
                extras_match = (recorded.get("divergences") == int(extras["diverging"].sum())
                    and recorded.get("tree_depth_limit_fraction") == float(np.mean(extras["num_steps"] >= 2**last["max_tree_depth"]-1))
                    and recorded.get("mean_accept_prob") == float(extras["accept_prob"].mean()))
                result = dict(purpose=PURPOSE, fit_id=receipt["run_key"], posterior_sha256=receipt["posterior_sha256"],
                    model_inventory=inventory, posterior_shape=[last["chains"], last["samples"]],
                    numerical=numerical, comparisons=comparisons,
                    recorded_parameter_diagnostics_match=numerical_match,
                    recorded_comparison_diagnostics_match=contrasts_match,
                    sampler_extras_recounted=True, recorded_sampler_diagnostics_match=extras_match,
                    prior_attempt_parameter_diagnostics_recomputed=False,
                    parameter_convergence_passed=numerical["converged"],
                    driver_contrast_convergence_passed=comparisons["all_reported_contrasts_passed"],
                    passed=bool(numerical["converged"] and comparisons["all_reported_contrasts_passed"]
                                and numerical_match and contrasts_match and extras_match),
                    functional_simulation_precision_established=False,
                    scientific_acceptance_established=False, publication_allowed=False)
        unchanged()
        atomic_json(output_dir / "convergence.json", result)
        unchanged()
        record.update(state="complete", passed=result["passed"], convergence_sha256=digest(output_dir / "convergence.json"))
    except BaseException as exc:
        record.update(state="failed_exception", passed=False, exception=type(exc).__name__, message=str(exc),
                      traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.monotonic()-started, finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(output_dir / "receipt.json", record)
    return result, record
