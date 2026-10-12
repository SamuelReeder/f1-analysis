"""Run only the separately approved, frozen coordinate-rotation profile.

The pending proposal cannot run. This never rewrites the original profile, fits
real drivers, weakens its numerical gates, or changes the fixed retry ladder.
Launch only as a memory-capped f1.slice user service after approval/registration.
"""
import os
os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.overall_control_rotation import build_rotation, rotated_model
from f1rank.artifacts import atomic_json, digest
from f1rank.overall.diagnostics import check
from f1rank.overall.fitting import arrays_digest, committed_registration, run
from f1rank.overall.provenance import snapshot, require_snapshot
from f1rank.overall.recovery import generate, recovery_metrics

METHOD = "data_only_lap_control_rotation_v1"


def require_prototype_sources(registration):
    expected_files = dict(registration["additional_source_sha256"])
    for record in registration["original_truth_files"].values():
        if record["path"] in expected_files and expected_files[record["path"]] != record["sha256"]:
            raise ValueError("Conflicting original truth fingerprints")
        expected_files[record["path"]] = record["sha256"]
    for name, expected in expected_files.items():
        path = ROOT / name
        if digest(path) != expected:
            raise ValueError("Registered rotation implementation changed")
        committed = subprocess.check_output(["git", "show", f"HEAD:{name}"], cwd=ROOT)
        if committed != path.read_bytes():
            raise ValueError("Rotation implementation must be committed")


def checked_registration(path):
    registration = committed_registration(path)
    if registration.get("purpose") != "numerical_profile" or registration.get("method") != METHOD:
        raise ValueError("This runner is only for the coordinate-rotation numerical profile")
    if registration.get("user_approved") is not True:
        raise ValueError("The post-failure experiment requires the requested user approval")
    parent_path = ROOT / registration["parent_registration"]
    parent = json.loads(parent_path.read_text())
    if digest(parent_path) != registration["parent_registration_sha256"]:
        raise ValueError("The original profile registration changed")
    if any(registration[key] != parent[key] for key in ("studies", "synthetic", "fit_settings", "retry", "diagnostics")):
        raise ValueError("Matched study must retain the original inventory, truths, settings and gates")
    truth_files = registration.get("original_truth_files", {})
    if set(truth_files) != set(parent["studies"]):
        raise ValueError("The complete original truth inventory is required")
    for study, record in truth_files.items():
        expected_path = f"outputs/championship/development/{parent['id']}/{study}.truth.json"
        if record.get("path") != expected_path:
            raise ValueError("Matched truths must come from the original profile")
    if snapshot()["key"] != registration["registered_model_inputs_key"]:
        raise ValueError("Model/data inputs differ from the separately proposed experiment")
    require_prototype_sources(registration)
    return registration


def matched_truth_record(registration, study, blocks, truth):
    """Compare every generating truth before sampling, including unused states."""
    spec = registration["synthetic"][study]
    record = dict(case=spec["case"], seed=spec["seed"], arrays_key=arrays_digest(blocks),
                  truth={key: np.asarray(value).tolist() for key, value in truth.items()})
    if record["arrays_key"] != registration["original_arrays_sha256"][study]:
        raise ValueError("Synthetic observations changed; never redraw the failed study")
    original = registration["original_truth_files"][study]
    path = ROOT / original["path"]
    if digest(path) != original["sha256"]:
        raise ValueError("Original generating truth file changed")
    if record != json.loads(path.read_text()):
        raise ValueError("Generated truths differ from the original matched study")
    return record


def contrast_reports(post, catalog, divergences):
    """Report every pair separately without changing the attempt's retry gate."""
    drivers, entries = catalog["field"], np.asarray(catalog["field_entries"])
    if (len(drivers) < 2 or len(set(drivers)) != len(drivers)
            or entries.shape != (len(drivers),) or entries.dtype.kind not in "iu"
            or len(np.unique(entries)) != len(entries) or (entries < 0).any()):
        raise ValueError("Current-field identities and entry indices must be complete and unique")
    rows, shape = [], None
    for site in ("skill", "overall_race_skill"):
        values = np.asarray(post[site])
        if (values.ndim != 3 or min(values.shape[:2]) < 2
                or entries.max() >= values.shape[-1] or not np.isfinite(values).all()):
            raise ValueError("Current driver states require finite complete grouped draws")
        if shape is not None and values.shape[:2] != shape:
            raise ValueError("Qualifying and race states must share chain/draw axes")
        shape = values.shape[:2]
        current = values[:, :, entries]
        for i in range(len(drivers)):
            for j in range(i):
                difference = current[:, :, i] - current[:, :, j]
                rows.append(dict(site=site, driver_i=drivers[i], driver_j=drivers[j],
                    diagnostic=check({"contrast": difference}, divergences),
                    chain_means=difference.mean(axis=1).tolist(),
                    chain_standard_deviations=difference.std(axis=1, ddof=1).tolist()))
    return dict(current_driver_contrasts=rows,
        all_reported_contrasts_passed=all(row["diagnostic"]["converged"] for row in rows),
        changes_attempt_convergence_decision=False, publication_allowed=False)


def sample_with_reports(blocks, settings, registration, rotation, catalog):
    post, diagnostic = sample_rotated(blocks, settings, registration, rotation)
    result = dict(diagnostic)
    result["comparison_reports"] = contrast_reports(post, catalog, diagnostic["divergences"])
    require_prototype_sources(registration)
    return post, result


def sample_rotated(blocks, settings, registration, rotation):
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS, init_to_median

    require_prototype_sources(registration)
    if jax.local_device_count() < settings["chains"]:
        raise RuntimeError("Insufficient chain devices for the registered profile")
    arguments = tuple({k:jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in blocks[key].items()}
                      for key in ("qualifying", "laps", "outcomes"))
    kernel = NUTS(rotated_model(rotation), target_accept_prob=settings["target_accept"],
        max_tree_depth=settings["max_tree_depth"],
        init_strategy=init_to_median(num_samples=settings["init_median_samples"]))
    sampler = MCMC(kernel, num_warmup=settings["warmup"], num_samples=settings["samples"],
        num_chains=settings["chains"], chain_method="parallel", progress_bar=False)
    sampler.run(jax.random.PRNGKey(settings["seed"]), *arguments,
                extra_fields=("diverging", "num_steps", "accept_prob"))
    post = {k:np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
    extras = sampler.get_extra_fields(group_by_chain=True)
    diagnostic = check(post, int(np.asarray(extras["diverging"]).sum()))
    diagnostic["tree_depth_limit_fraction"] = float(np.mean(np.asarray(extras["num_steps"]) >= 2**settings["max_tree_depth"] - 1))
    diagnostic["mean_accept_prob"] = float(np.mean(np.asarray(extras["accept_prob"])))
    diagnostic["runtime"] = dict(jax=jax.__version__, numpyro=numpyro.__version__, backend=jax.default_backend())
    diagnostic["control_rotation_sha256"] = rotation.fingerprint()
    require_prototype_sources(registration)
    jax.clear_caches()
    return post, diagnostic


def execute(registration_path):
    registration_path = Path(registration_path).resolve()
    registration = checked_registration(registration_path)
    inputs = snapshot()
    destination = ROOT / "outputs/championship/development" / registration["id"]
    if destination.exists():
        raise FileExistsError("Preserve the recorded experiment; inspect it instead of rerunning")
    destination.mkdir(parents=True)
    rows = []
    for study in registration["studies"]:
        spec = registration["synthetic"][study]
        cache = ROOT / "outputs/championship/cache" / registration["id"] / study
        try:
            require_snapshot(inputs)
            require_prototype_sources(registration)
            blocks, catalog, truth = generate(spec["case"], spec["seed"])
            truth_record = matched_truth_record(registration, study, blocks, truth)
            rotation = build_rotation(blocks["laps"])
            catalog.update(method=METHOD, control_rotation_sha256=rotation.fingerprint(),
                           additional_source_sha256=registration["additional_source_sha256"])
            truth_file = destination / f"{study}.truth.json"
            atomic_json(truth_file, truth_record)
            post, receipt = run(blocks, catalog, cache, registration_path, study=study, synthetic=True,
                sample=lambda b, s: sample_with_reports(b, s, registration, rotation, catalog))
            result = dict(study=study, state="converged", receipt=receipt,
                truth_sha256=digest(truth_file), recovery=recovery_metrics(post, truth, catalog))
            del post
        except Exception as exc:
            receipt = cache / "receipt.json"
            result = dict(study=study, state="failed", error=type(exc).__name__, message=str(exc),
                          receipt=json.loads(receipt.read_text()) if receipt.exists() else None)
        atomic_json(destination / f"{study}.json", result)
        rows.append(dict(study=study, state=result["state"]))
        print(json.dumps(rows[-1]), flush=True)
    require_snapshot(inputs)
    require_prototype_sources(registration)
    result = dict(registration=str(registration_path.relative_to(ROOT)), studies=rows,
        all_processed=len(rows) == len(registration["studies"]), all_converged=all(r["state"] == "converged" for r in rows),
        method=METHOD, source=inputs, synthetic_recovery_complete=False, sbc_complete=False,
        publication_allowed=False, original_profile_unchanged=True)
    atomic_json(destination / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", required=True)
    args = parser.parse_args()
    result = execute(args.registration)
    raise SystemExit(0 if result["all_converged"] else 1)
