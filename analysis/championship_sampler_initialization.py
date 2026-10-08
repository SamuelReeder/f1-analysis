"""Inspect the failed run's initial states; no warm-up, posterior fit, or prediction.

Uses the exact registered seed, model, inputs and initialization rule. This cannot
produce a passing replacement fit or write championship/dashboard outputs.
"""
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
from numpyro.infer import init_to_median

from analysis import championship_archive_conservative as study
from f1rank import artifacts, racemulti

OUT = study.ROOT / "outputs/analysis/championship_sampler_initialization"


def ranges(tree):
    return {name: {"min": float(np.min(x)), "max": float(np.max(x)),
                   "norm": float(np.linalg.norm(x)), "finite": bool(np.isfinite(x).all())}
            for name, value in tree.items() for x in [np.asarray(value)]}


def main():
    protocol = study.check_registration()
    if OUT.exists():
        raise FileExistsError("Initialization diagnostic is already recorded")
    failure = json.loads((study.OUT / "failure.json").read_text())
    if failure["kind"] != "exhausted_registered_fit":
        raise ValueError("Expected the preserved sampler failure")
    season = protocol["first_test_season"]
    clean, _ = study.archive.prepare(study.archive.check_registration())
    with patch.object(study.archive, "CACHE", study.CACHE):
        driver, _, _ = study.archive.qualifying(season, protocol)
    train = clean[clean.season < season].merge(
        driver.rename(columns={"driver": "quali"}), on=["event_id", "driver_id"],
        how="left", validate="many_to_one").dropna(subset=["quali"])
    d, _ = racemulti.arrays(train)
    settings = protocol["race_pace_sampler"]
    keys = study.c.jax.random.split(study.c.jax.random.PRNGKey(settings["seed"]), settings["chains"])
    states = []
    started = time.monotonic()
    for chain, key in enumerate(keys):
        kernel = racemulti.NUTS(racemulti.model, target_accept_prob=settings["target_accept"],
                               init_strategy=init_to_median(num_samples=15))
        # These are the arguments used by MCMC._single_chain_mcmc. NUTS.init sets
        # up adaptation but makes no warm-up or posterior transitions.
        state = kernel.init(key, settings["warmup"], init_params=None, model_args=(d,), model_kwargs={})
        states.append({"chain": chain, "iteration": int(state.i),
                       "potential_energy": float(state.potential_energy),
                       "step_size": float(state.adapt_state.step_size),
                       "unconstrained_parameters": ranges(state.z), "gradients": ranges(state.z_grad)})
        assert int(state.i) == 0
    study.check_registration()
    result = {"purpose": "Numerical initialization diagnostic only; no candidate refit or predictive scores",
              "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "protocol_sha256": artifacts.digest(study.PROTOCOL), "sampler_settings": settings,
              "gate": False, "publication_authorized": False, "states": states,
              "elapsed_seconds": time.monotonic() - started}
    OUT.mkdir(parents=True)
    artifacts.atomic_json(OUT / "summary.json", result)
    inputs = [study.PROTOCOL, study.OUT / "failure.json", Path(__file__),
              *(study.ROOT / k for k in protocol["input_hashes"]),
              *(study.ROOT / k for k in protocol["source_hashes"]),
              *(study.ROOT / k for k in failure["used_cache_hashes"])]
    artifacts.record(OUT, [OUT / "summary.json"], model="sampler-initialization-diagnostic", inputs=inputs)
    print(json.dumps({"states": [{k: v for k, v in row.items() if k not in ("unconstrained_parameters", "gradients")}
                                for row in states], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
