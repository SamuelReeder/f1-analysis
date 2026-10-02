"""Fit the model with NUTS and save posterior draws of the states and hyperparameters.

Each fit `<name>.npz` has a sidecar `<name>.meta.json` recording the data it was fitted
on: a fingerprint of every model input, the last event in the data, the training cutoff
and stable identifiers for every state. Loading a fit against a design checks the
fingerprint, so draws are never matched to states of a different design (for example
after a new race has shifted the indices). `design_for` rebuilds the exact design a fit
was made on, for evaluating older fits against newer data.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
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
    # standard-normal (non-centred) sites start at their prior mean, with the size they
    # have under this model's options (e.g. the team-effect unit), so they are traced
    # from the prior; the other sites (some improper) are traced at their start values
    fixed = {k: v for k, v in init.items() if not k.endswith("_z")}
    sites = numpyro.handlers.trace(numpyro.handlers.substitute(
        numpyro.handlers.seed(model_fn, 0), data=fixed)).get_trace(data)
    init = {k: (v if v.shape == sites[k]["value"].shape else jnp.zeros_like(sites[k]["value"]))
            for k, v in init.items() if k in sites}
    kernel = NUTS(model_fn, target_accept_prob=target_accept, max_tree_depth=max_tree_depth,
                  init_strategy=init_to_value(values=init))
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method="parallel" if chains > 1 else "sequential",
                progress_bar=progress)
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(seed), data, extra_fields=("num_steps", "diverging"))
    jax.block_until_ready(mcmc.get_samples())
    elapsed = time.time() - t0
    # every fitted term is kept, so residuals and forecasts can use all of them
    keep = ["skill", "car", "car_track", "car_event", "car_segment", "driver_form", "compat",
            "track_load", "sd_track_load", "sd_car_event", "slow_scale_ratio", "p_compromised",
            "compromised_extra_scale", "compromised_shift", "sd_car_segment", "sd_driver_form",
            "sd_compat", "sd_placebo", "mu", "sigma_u",
            *HYPER, *extra_sites]
    post = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items() if k in keep}
    post = {k: v.astype(np.float32) for k, v in post.items()}
    extra = mcmc.get_extra_fields(group_by_chain=True)
    info = {"elapsed_s": elapsed,
            "divergences": int(np.asarray(extra["diverging"]).sum()),
            "mean_steps": float(np.asarray(extra["num_steps"]).mean())}
    return post, info


class IncompatibleFit(RuntimeError):
    """A saved fit does not belong to the design it is being matched to."""


def fingerprint(design: Design) -> str:
    """Hash of every model input (data, indices, cutoff, circuit factor)."""
    h = hashlib.sha256(str(design.start_season).encode())
    for k, v in sorted(design.arrays().items()):
        a = np.asarray(v)
        if a.dtype.kind == "f":
            a = np.round(a.astype(np.float64), 8)
        h.update(f"{k}:{a.dtype}:{a.shape}".encode())
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


def design_ids(design: Design) -> dict[str, list[str]]:
    """Stable identifiers for every state, in the order the draws are stored."""
    ev = design.events.event_id.to_numpy()
    e, c, s = design.entries, design.cars, design.sessions
    stints = e.groupby("stint_idx")[["driver_id", "team"]].first()
    return {
        "events": list(ev),
        "drivers": design.drivers.driver_id.tolist(),
        "entries": (e.event_id + "|" + e.driver_id).tolist(),
        "cars": (c.team + "|" + ev[c.event_idx.to_numpy()]).tolist(),
        "sessions": (s.event_id + "|" + s.segment).tolist(),
        "stints": (stints.driver_id + "|" + stints.team).tolist(),
    }


def last_train_event(design: Design) -> str | None:
    if design.train is None or design.train.all():
        return None
    return design.events.event_id.iloc[int(design.obs.event_idx[design.train].max())]


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_dirty() -> bool | None:
    """Whether the package differs from the recorded commit (uncommitted changes)."""
    try:
        return bool(subprocess.run(["git", "status", "--porcelain", "--", "f1rank"], cwd=ROOT,
                                   capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        return None


def meta_path(path: Path) -> Path:
    return path.with_name(path.stem + ".meta.json")


def describe(design: Design, **extra) -> dict:
    return {
        "fingerprint": fingerprint(design), "start_season": int(design.start_season),
        "data_as_of": design.events.event_id.iloc[-1], "last_train_event": last_train_event(design),
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "git_commit": _git_commit(), "git_dirty": _git_dirty(), **extra, "ids": design_ids(design),
        **({"sprint_quali_until": design.sprint_quali_until} if design.sprint_quali_until else {}),
    }


def save(path: Path, post: dict, info: dict, design: Design, **extra) -> None:
    """Save draws plus a metadata sidecar. `extra` records how the fit was made."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **post, **{f"info_{k}": v for k, v in info.items()})
    meta_path(path).write_text(json.dumps(describe(design, **extra), indent=1))


def load_meta(path: Path) -> dict | None:
    m = meta_path(path)
    return json.loads(m.read_text()) if m.exists() else None


def check(path: Path, design: Design, match: str = "fingerprint") -> dict:
    """Raise IncompatibleFit unless the fit at `path` was made on `design`.

    match="fingerprint": identical model inputs. match="ids": same states in the same
    order (for synthetic fits, whose lap times differ from the real data by design).
    """
    meta = load_meta(path)
    if meta is None:
        raise IncompatibleFit(f"{path.name}: no metadata ({meta_path(path).name}); refit it")
    if match == "fingerprint":
        ok = meta["fingerprint"] == fingerprint(design)
    else:
        ok = meta["ids"] == design_ids(design)
    if not ok:
        raise IncompatibleFit(
            f"{path.name} was fitted on data as of {meta['data_as_of']} (start {meta['start_season']}, "
            f"trained to {meta['last_train_event'] or 'the end'}); the design here has data as of "
            f"{design.events.event_id.iloc[-1]} (trained to {last_train_event(design) or 'the end'}). "
            "Refit, or rebuild the design with fit.design_for().")
    return meta


def load(path: Path, design: Design | None = None, match: str = "fingerprint") -> tuple[dict, dict]:
    """Load draws. Pass the design whenever draws will be indexed by its states."""
    if design is not None:
        check(path, design, match)
    z = np.load(path)
    post = {k: z[k] for k in z.files if not k.startswith("info_")}
    info = {k[5:]: z[k].item() for k in z.files if k.startswith("info_")}
    return post, info


def design_for(path: Path) -> Design:
    """Rebuild the design a fit was made on (same start, data as-of and cutoff)."""
    meta = load_meta(path)
    if meta is None:
        raise IncompatibleFit(f"{path.name}: no metadata ({meta_path(path).name}); refit it")
    design = build_design(meta["start_season"], end_event=meta["data_as_of"],
                          sprint_quali_until=meta.get("sprint_quali_until"))
    if meta["last_train_event"]:
        ev = design.events.set_index("event_id").event_idx
        design = design.with_cutoff(int(ev[meta["last_train_event"]]))
    return design


def common_data_as_of(paths: list[Path]) -> str:
    """The shared data as-of date of a batch of fits; raises if they differ."""
    metas = {p.name: load_meta(p) for p in paths}
    missing = [n for n, m in metas.items() if m is None]
    if missing:
        raise IncompatibleFit(f"fits without metadata: {', '.join(missing)}")
    dates = {n: m["data_as_of"] for n, m in metas.items()}
    if len(set(dates.values())) != 1:
        latest = max(dates.values())
        stale = sorted(n for n, d in dates.items() if d != latest)
        raise IncompatibleFit(f"fits use different data: {len(stale)} are older than {latest} "
                              f"({', '.join(stale[:8])}{'...' if len(stale) > 8 else ''}); rerun them")
    return next(iter(dates.values()))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=2010)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--samples", type=int, default=1000)
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--name", default="main")
    args = p.parse_args()

    design = build_design(args.start)
    settings = dict(warmup=args.warmup, samples=args.samples, chains=args.chains, target_accept=0.9)
    post, info = fit(design, **settings)
    save(FITS / f"{args.name}.npz", post, info, design, job=args.name, model_kw={}, settings=settings)
    print(info)


if __name__ == "__main__":
    main()
