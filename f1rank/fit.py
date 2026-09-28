"""Fit the model with NUTS and save posterior draws of the states and hyperparameters."""

import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
from numpyro.infer import MCMC, NUTS, init_to_value  # noqa: E402

from .design import Design, build_design  # noqa: E402
from .model import model  # noqa: E402

numpyro.enable_x64(False)

ROOT = Path(__file__).resolve().parent.parent
FITS = ROOT / "outputs" / "fits"

HYPER = ["sd_level", "sd_skill_race", "sd_skill_season", "exp_gain", "exp_scale", "age_slope",
         "rho", "rho_reset", "sd_car_race", "sd_car_season", "sd_car_reset",
         "sigma0", "sigma_tau", "nu"]


def to_jax(arrays: dict) -> dict:
    return {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in arrays.items()}


INIT_HYPER = {"sd_level": 0.2, "sd_skill_race": 0.01, "sd_skill_season": 0.07,
              "exp_gain": 0.25, "exp_scale": 20.0, "age_slope": -0.02, "rho": 0.85,
              "rho_reset": 0.7, "sd_car_race": 0.1, "sd_car_season": 0.4, "sd_car_reset": 0.6,
              "sigma0": 0.3, "sigma_tau": 0.4, "nu": 6.0, "sd_track_load": 0.1,
              "slow_scale_ratio": 1.2, "p_compromised": 0.15, "compromised_extra_scale": 2.0,
              "compromised_shift": 1.0, "sd_car_segment": 0.1,
              "sd_driver_form": 0.05, "sd_compat": 0.05, "sd_placebo": 0.05}


def initial_values(design: Design, seed: int) -> dict:
    """Start states near the data: car = team's mean pace at the event, mu = segment mean."""
    rng = np.random.default_rng(seed)
    o = design.obs.loc[design.train] if design.train is not None else design.obs
    seg_mean = o.groupby("session_idx").y.mean()
    mu = np.zeros(len(design.sessions))
    mu[seg_mean.index] = seg_mean.to_numpy()
    resid = o.y - o.session_idx.map(seg_mean)
    car = np.zeros(len(design.cars))
    car_mean = resid.groupby(o.car_idx).mean()
    car[car_mean.index] = car_mean.to_numpy()
    # carry states forward into events with no training data (forecast period)
    cars = design.cars.assign(v=np.where(np.isin(np.arange(len(car)), car_mean.index), car, np.nan))
    car = cars.groupby("team").v.ffill().fillna(0.0).to_numpy()
    small = lambda n: 0.1 * rng.standard_normal(n)
    return {
        **INIT_HYPER,
        "car_state": car, "car_total": car, "mu": mu,
        "level_z": np.zeros(len(design.drivers)), "skill_z": np.zeros(len(design.entries)),
        "sigma_u": np.zeros(len(design.sessions)),
        "car_segment_z": np.zeros(int(design.obs.car_segment_idx.max()) + 1),
        "driver_form_z": np.zeros(len(design.entries)),
        "compat_z": np.zeros(int(design.entries.stint_idx.max()) + 1),
        "placebo_z": np.zeros(int(design.entries.placebo_idx.max()) + 1),
        "track_load_z": small(int(design.cars.team_season_idx.max()) + 1),
    }


def fit(design: Design, warmup=1000, samples=1000, chains=4, seed=0, target_accept=0.85,
        max_tree_depth=10, model_fn=model, progress=True, extra_sites=()):
    data = to_jax(design.arrays())
    init = {k: jnp.asarray(v) for k, v in initial_values(design, seed).items()}
    sites = numpyro.handlers.trace(numpyro.handlers.substitute(
        numpyro.handlers.seed(model_fn, 0), data=init)).get_trace(data)
    init = {k: v for k, v in init.items() if k in sites}
    kernel = NUTS(model_fn, target_accept_prob=target_accept, max_tree_depth=max_tree_depth,
                  init_strategy=init_to_value(values=init))
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method="parallel" if chains > 1 else "sequential",
                progress_bar=progress)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), data, extra_fields=("num_steps", "diverging"))
    jax.block_until_ready(mcmc.get_samples())
    elapsed = time.time() - t0
    keep = ["skill", "car", "car_track", "car_event", "track_load", "sd_track_load",
            "sd_car_event", "slow_scale_ratio", "p_compromised", "compromised_extra_scale",
            "compromised_shift", "sd_car_segment", "sd_driver_form", "sd_compat", "compat", "sd_placebo",
            "mu", "sigma_u",
            *HYPER, *extra_sites]
    post = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items() if k in keep}
    post = {k: v.astype(np.float32) for k, v in post.items()}
    extra = mcmc.get_extra_fields(group_by_chain=True)
    info = {"elapsed_s": elapsed,
            "divergences": int(np.asarray(extra["diverging"]).sum()),
            "mean_steps": float(np.asarray(extra["num_steps"]).mean())}
    return post, info


def save(path: Path, post: dict, info: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **post, **{f"info_{k}": v for k, v in info.items()})


def load(path: Path) -> tuple[dict, dict]:
    z = np.load(path)
    post = {k: z[k] for k in z.files if not k.startswith("info_")}
    info = {k[5:]: z[k].item() for k in z.files if k.startswith("info_")}
    return post, info


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--start", type=int, default=2010)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--name", default="main")
    args = p.parse_args()

    design = build_design(args.start)
    post, info = fit(design, args.warmup, args.samples, args.chains, target_accept=0.9,
                     extra_sites=("car_segment",))
    save(FITS / f"{args.name}.npz", post, info)
    print(info)


if __name__ == "__main__":
    main()
