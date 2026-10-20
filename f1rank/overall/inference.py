"""Registered production inference using the verified coordinate implementation.

This selects a numerical implementation, not a new prior, retry or acceptance
rule. No existing profile registration selects it. Every requested sampler extra
survives in the receipt/archive path so audits can recount the diagnostics.
"""
from typing import NamedTuple

import numpy as np

from .bounded_diagnostics import check

METHOD = "conditional_weekend_batched_controls_v1"
EXTRA_FIELDS = ("diverging", "num_steps", "accept_prob")


class SampleResult(NamedTuple):
    posterior: dict
    diagnostics: dict
    extras: dict


def contrast_reports(post, catalog, divergences):
    """Keep the complete existing pair reports separate from the retry gate."""
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


def validate_extras(extras, settings):
    axes = (settings["chains"], settings["samples"])
    if set(extras) != set(EXTRA_FIELDS):
        raise ValueError("Retain every registered sampler extra")
    if any(np.asarray(a).shape != axes or not np.isfinite(a).all() for a in extras.values()):
        raise ValueError("Sampler extras need complete finite chain/draw axes")
    if (extras["diverging"].dtype.kind != "b" or extras["num_steps"].dtype.kind not in "iu"
            or (extras["num_steps"] < 1).any()
            or extras["accept_prob"].dtype.kind != "f"
            or ((extras["accept_prob"] < 0) | (extras["accept_prob"] > 1)).any()):
        raise ValueError("Invalid sampler extras")


def sample(blocks, settings, catalog):
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer import MCMC, NUTS, init_to_median
    from .batched_rotation import prepare
    from .control_rotation import build_rotation
    from .weekend_coordinates import weekend_model

    if jax.local_device_count() < settings["chains"]:
        raise RuntimeError("Insufficient registered chain devices")
    rotation = prepare(build_rotation(blocks["laps"]))
    arguments = tuple({k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in blocks[key].items()}
                      for key in ("qualifying", "laps", "outcomes"))
    kernel = NUTS(weekend_model(rotation), target_accept_prob=settings["target_accept"],
        max_tree_depth=settings["max_tree_depth"],
        init_strategy=init_to_median(num_samples=settings["init_median_samples"]))
    sampler = MCMC(kernel, num_warmup=settings["warmup"], num_samples=settings["samples"],
        num_chains=settings["chains"], chain_method="parallel", progress_bar=False)
    try:
        sampler.run(jax.random.PRNGKey(settings["seed"]), *arguments, extra_fields=EXTRA_FIELDS)
        post = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
        extras = {k: np.asarray(v) for k, v in sampler.get_extra_fields(group_by_chain=True).items()}
        validate_extras(extras, settings)
        if not post or any(a.shape[:2] != (settings["chains"], settings["samples"]) for a in post.values()):
            raise ValueError("Retain complete posterior chain/draw axes")
        diagnostic = check(post, int(extras["diverging"].sum()))
        diagnostic["comparison_reports"] = contrast_reports(post, catalog, diagnostic["divergences"])
        diagnostic["tree_depth_limit_fraction"] = float(np.mean(extras["num_steps"] >= 2**settings["max_tree_depth"]-1))
        diagnostic["mean_accept_prob"] = float(np.mean(extras["accept_prob"]))
        diagnostic["runtime"] = dict(jax=jax.__version__, numpyro=numpyro.__version__, backend=jax.default_backend())
        diagnostic["control_rotation_sha256"] = rotation.fingerprint()
        diagnostic["inference_method"] = METHOD
        return SampleResult(post, diagnostic, extras)
    finally:
        jax.clear_caches()
