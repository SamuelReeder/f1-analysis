"""Immutable, checked fits for the successor. Failures are evidence, never caches.

Only registered runs may enter this runner. A numerical-profile registration is
restricted to synthetic data and cannot authorize real-data fitting/publication.
"""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

import numpy as np

from ..artifacts import RETRY, atomic_json, digest
from .diagnostics import (check, RHAT_LIMIT, MIN_ESS, MAX_MCSE_SD_RATIO, MAX_DIVERGENCE_RATE)
from .provenance import ROOT, snapshot, require_snapshot

FIT_SETTINGS = {"warmup": 800, "samples": 800, "chains": 4, "target_accept": .9,
                "max_tree_depth": 10, "init_median_samples": 15, "seed": 0}
RUN_SCHEMA = 1


def arrays_digest(blocks):
    h = hashlib.sha256()
    for block_name, block in sorted(blocks.items()):
        for name, value in sorted(block.items()):
            a = np.asarray(value)
            if a.dtype.hasobject or not np.isfinite(a).all():
                raise ValueError(f"Invalid input {block_name}.{name}")
            h.update(f"{block_name}:{name}:{a.dtype.str}:{a.shape}".encode())
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def committed_registration(path, root=ROOT):
    path = Path(path).resolve()
    relative = path.relative_to(root).as_posix()
    saved = subprocess.check_output(["git", "show", f"HEAD:{relative}"], cwd=root)
    if saved != path.read_bytes():
        raise ValueError("Fit registration must be committed without local changes")
    registration = json.loads(saved)
    if registration.get("schema_version") != RUN_SCHEMA or registration.get("status") != "frozen":
        raise ValueError("Fit requires a frozen supported registration")
    if registration.get("fit_settings") != FIT_SETTINGS or registration.get("retry") != [list(x) for x in RETRY]:
        raise ValueError("Registered settings do not match the fixed runner")
    numerical = dict(rank_rhat_strict_upper=RHAT_LIMIT, bulk_ess_min=MIN_ESS,
                     tail_ess_min=MIN_ESS, mean_mcse_sd_upper=MAX_MCSE_SD_RATIO,
                     divergences_per_draw_upper=MAX_DIVERGENCE_RATE)
    if registration.get("diagnostics") != numerical:
        raise ValueError("Registered diagnostics do not match the checker")
    return registration


def require_clean_inputs(inputs, root=ROOT):
    # Hashes alone do not prove that scientific decisions preceded fitting.
    # Every source/data/contract file must exist at this committed revision.
    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0"))
    if set(inputs["inputs"]) - tracked:
        raise ValueError("Uncommitted Overall source/data/contract inputs")
    names = sorted(inputs["inputs"])
    subprocess.run(["git", "diff", "--exit-code", "HEAD", "--", *names], cwd=root, check=True,
                   stdout=subprocess.DEVNULL)


def load_checked(run_dir, expected_key=None):
    run_dir = Path(run_dir)
    receipt = json.loads((run_dir / "receipt.json").read_text())
    if receipt.get("state") != "converged" or receipt["diagnostics"].get("converged") is not True:
        raise ValueError("Failed or incomplete fit is not a posterior cache")
    if expected_key is not None and receipt["run_key"] != expected_key:
        raise ValueError("Fit does not match requested inputs")
    require_snapshot(receipt["source"])
    archive = run_dir / receipt["posterior_file"]
    if digest(archive) != receipt["posterior_sha256"]:
        raise ValueError("Posterior archive changed")
    with np.load(archive, allow_pickle=False) as saved:
        post = {name: saved[name] for name in saved.files}
    expected_shape = (receipt["settings"]["chains"], receipt["attempts"][-1]["samples"])
    if not post or any(a.shape[:2] != expected_shape or not np.isfinite(a).all() for a in post.values()):
        raise ValueError("Invalid posterior draw axes/values")
    return post, receipt


def _sample(blocks, settings):
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS, init_to_median
    from .model import model
    if jax.local_device_count() < settings["chains"]:
        raise RuntimeError("Insufficient chain devices; set XLA host-device count before importing JAX")
    arguments = tuple({name: jnp.asarray(value) if isinstance(value, np.ndarray) else value
                       for name, value in blocks[key].items()} for key in ("qualifying", "laps", "outcomes"))
    kernel = NUTS(model, target_accept_prob=settings["target_accept"],
                  max_tree_depth=settings["max_tree_depth"],
                  init_strategy=init_to_median(num_samples=settings["init_median_samples"]))
    sampler = MCMC(kernel, num_warmup=settings["warmup"], num_samples=settings["samples"],
                   num_chains=settings["chains"], chain_method="parallel", progress_bar=False)
    sampler.run(jax.random.PRNGKey(settings["seed"]), *arguments,
                extra_fields=("diverging", "num_steps", "accept_prob"))
    post = {name: np.asarray(value) for name, value in sampler.get_samples(group_by_chain=True).items()}
    extras = sampler.get_extra_fields(group_by_chain=True)
    result = check(post, int(np.asarray(extras["diverging"]).sum()))
    result["tree_depth_limit_fraction"] = float(np.mean(np.asarray(extras["num_steps"]) >= 2 ** settings["max_tree_depth"] - 1))
    result["mean_accept_prob"] = float(np.mean(np.asarray(extras["accept_prob"])))
    result["runtime"] = dict(jax=jax.__version__, numpyro=numpyro.__version__, backend=jax.default_backend())
    jax.clear_caches()
    return post, result


def run(blocks, catalog, run_dir, registration_path, *, study, synthetic, sample=_sample):
    registration_path, run_dir = Path(registration_path), Path(run_dir)
    registration = committed_registration(registration_path)
    if study not in registration.get("studies", []):
        raise ValueError("Study is not in the registered inventory")
    if not synthetic and registration.get("purpose") != "real_data_acceptance":
        raise ValueError("This registration cannot authorize real-data fits")
    inputs = snapshot()
    require_clean_inputs(inputs)
    specification = dict(source_key=inputs["key"], arrays_key=arrays_digest(blocks), catalog=catalog,
                         registration_sha256=digest(registration_path), study=study,
                         synthetic=synthetic, settings=FIT_SETTINGS)
    run_key = hashlib.sha256(json.dumps(specification, sort_keys=True).encode()).hexdigest()
    if run_dir.exists():
        # A terminal failure is not rerun. An interrupted run is not guessed dead
        # from a file: the operator checks the service before choosing recovery.
        return load_checked(run_dir, run_key)
    run_dir.mkdir(parents=True, exist_ok=False)
    receipt = dict(schema_version=RUN_SCHEMA, run_key=run_key, specification=specification,
                   source=inputs, catalog=catalog, settings=FIT_SETTINGS, attempts=[],
                   state="running", publication_allowed=False,
                   source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   started_at=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid())
    atomic_json(run_dir / "receipt.json", receipt)
    started = time.monotonic()
    try:
        for attempt, (warm, draws, accept) in enumerate(RETRY):
            require_snapshot(inputs)
            settings = dict(FIT_SETTINGS, warmup=warm * FIT_SETTINGS["warmup"],
                samples=draws * FIT_SETTINGS["samples"], seed=FIT_SETTINGS["seed"] + attempt,
                target_accept=accept if accept is not None else FIT_SETTINGS["target_accept"])
            begun = time.monotonic()
            print(json.dumps(dict(study=study, attempt=attempt + 1, state="sampling", settings=settings)), flush=True)
            post, diagnostic = sample(blocks, settings)
            require_snapshot(inputs)
            if digest(registration_path) != specification["registration_sha256"]:
                raise ValueError("Registration changed while fitting")
            record = dict(**settings, diagnostics=diagnostic, seconds=time.monotonic() - begun)
            receipt["attempts"].append(record)
            atomic_json(run_dir / f"attempt_{attempt + 1}.json", record)
            receipt["diagnostics"] = diagnostic
            atomic_json(run_dir / "receipt.json", receipt)
            rows = diagnostic.get("parameters", [])
            print(json.dumps(dict(study=study, attempt=attempt + 1, seconds=record["seconds"],
                converged=diagnostic["converged"], divergences=diagnostic.get("divergences"),
                worst_rhat=max((r["rhat_max"] or float("inf") for r in rows), default=None))), flush=True)
            if diagnostic["converged"]:
                archive = run_dir / "posterior.npz"
                temporary = archive.with_suffix(".tmp.npz")
                np.savez_compressed(temporary, **post)
                require_snapshot(inputs)
                temporary.replace(archive)
                receipt.update(state="converged", posterior_file=archive.name,
                               posterior_sha256=digest(archive))
                break
            # Keep every failed draw set for diagnosing sampler failures. It can
            # never pass load_checked or enter a ranking.
            np.savez_compressed(run_dir / f"failed_{attempt + 1}.npz", **post)
            del post
        else:
            receipt["state"] = "failed_convergence"
    except BaseException as exc:
        receipt.update(state="failed_exception", exception=type(exc).__name__, message=str(exc),
                       traceback=traceback.format_exc())
        raise
    finally:
        receipt.update(seconds=time.monotonic() - started,
                       max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                       finished_at=dt.datetime.now(dt.timezone.utc).isoformat())
        atomic_json(run_dir / "receipt.json", receipt)
    if receipt["state"] != "converged":
        raise RuntimeError("All registered fit attempts failed; preserve the failure and do not publish")
    return post, receipt
