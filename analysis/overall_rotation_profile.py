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
    for name, expected in registration["additional_source_sha256"].items():
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
    if snapshot()["key"] != registration["registered_model_inputs_key"]:
        raise ValueError("Model/data inputs differ from the separately proposed experiment")
    require_prototype_sources(registration)
    return registration


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
            if arrays_digest(blocks) != registration["original_arrays_sha256"][study]:
                raise ValueError("Synthetic observations changed; never redraw the failed study")
            rotation = build_rotation(blocks["laps"])
            catalog.update(method=METHOD, control_rotation_sha256=rotation.fingerprint(),
                           additional_source_sha256=registration["additional_source_sha256"])
            truth_file = destination / f"{study}.truth.json"
            atomic_json(truth_file, dict(case=spec["case"], seed=spec["seed"], arrays_key=arrays_digest(blocks),
                                        truth={k:v.tolist() for k, v in truth.items()}))
            post, receipt = run(blocks, catalog, cache, registration_path, study=study, synthetic=True,
                sample=lambda b, s: sample_rotated(b, s, registration, rotation))
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
