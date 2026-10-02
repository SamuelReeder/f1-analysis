"""Total race pace with weekend covariates known before the race (pre-registered variants).

Identical to race_total_model (v1) except that each car covariate adds beta x x to the
mean of that team's laps in that race and each driver covariate adds beta x x to that
driver's laps in that race, beta ~ Normal(0, 1) per covariate. Decisions are fixed in
docs/race_features.md. v1's likelihood is reused unchanged: adding beta x x to the mean is
the same as subtracting it from the outcome. This is a separate file so the cached v1
fits, which hash race_total_model.py, stay valid.
"""
import hashlib
import json
from pathlib import Path
import time

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, init_to_median

from . import race_total_model as base
from .artifacts import diagnostics, require_convergence

MODEL = "total-dry-race-pace-features-v1"
CACHE = base.ROOT / "outputs" / "race_features" / "cache"
KEEP = base.KEEP | {"beta_car", "beta_drv"}


def matrix(keys: list[str], x: pd.DataFrame | None) -> np.ndarray:
    """Covariate rows for "event|id" keys; 0 where the covariate is missing."""
    if x is None or x.empty:
        return np.zeros((len(keys), 0))
    return x.reindex(keys).fillna(0.).to_numpy(float)


def design(C, x_car: pd.DataFrame | None, x_drv: pd.DataFrame | None):
    """v1's design plus covariate matrices for each car-race and driver-race.

    x_car is indexed by "event_id|team", x_drv by "event_id|driver_id", one column per
    covariate."""
    d, catalog = base.design(C)
    car_races = [str(x) for x in pd.factorize(C.event_id + "|" + C.team, sort=True)[1]]
    driver_races = [str(x) for x in pd.factorize(C.event_id + "|" + C.driver_id, sort=True)[1]]
    d.update(x_car=jnp.asarray(matrix(car_races, x_car)), x_drv=jnp.asarray(matrix(driver_races, x_drv)))
    catalog = dict(catalog, car_covariates=[] if x_car is None else list(x_car.columns),
                   driver_covariates=[] if x_drv is None else list(x_drv.columns))
    return d, catalog


def model(d, driver=True, car=True):
    offset = jnp.zeros_like(d["y"])
    if d["x_car"].shape[1]:
        beta = numpyro.sample("beta_car", dist.Normal(0, 1).expand([d["x_car"].shape[1]]))
        offset = offset + (d["x_car"] @ beta)[d["cr"]]
    if d["x_drv"].shape[1]:
        beta = numpyro.sample("beta_drv", dist.Normal(0, 1).expand([d["x_drv"].shape[1]]))
        offset = offset + (d["x_drv"] @ beta)[d["dr"]]
    base.model({**d, "y": d["y"] - offset}, driver=driver, car=car)


def fit(C, name, x_car=None, x_drv=None, *, warmup=800, samples=800, tries=3):
    """v1's fitting procedure (sampler, retries, checks) for the variant."""
    d, catalog = design(C, x_car, x_drv)
    sources = (Path(__file__), Path(base.__file__))
    source = b"".join(p.read_bytes() for p in sources)
    settings = dict(warmup=warmup, samples=samples, tries=tries)
    h = hashlib.sha256(source + json.dumps([catalog, settings], sort_keys=True).encode())
    for key, value in sorted(d.items()):
        a = np.asarray(value)
        h.update(f"{key}:{a.dtype}:{a.shape}".encode())
        h.update(a.tobytes())
    key = h.hexdigest()
    path = CACHE / f"{name}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            if meta["key"] == key:
                require_convergence(meta["diagnostics"])
                print(f"{name}: reuse checked fit", flush=True)
                return {k: z[k].copy() for k in z.files if k != "meta"}, meta
    attempts = []
    for attempt in range(tries):
        started = time.monotonic()
        kernel = NUTS(model, target_accept_prob=.9, dense_mass=[("skill_raw", "package_contrast")],
                      init_strategy=init_to_median(num_samples=15))
        mcmc = MCMC(kernel, num_warmup=warmup * (attempt + 1), num_samples=samples * 2 ** attempt,
                    num_chains=4, chain_method="vectorized" if jax.default_backend() == "gpu" else "parallel",
                    progress_bar=False)
        print(f"{name}: attempt {attempt + 1}, {len(C):,} laps, {C.event_id.nunique()} races", flush=True)
        mcmc.run(jax.random.PRNGKey(attempt), d, extra_fields=("diverging",))
        grouped = mcmc.get_samples(group_by_chain=True)
        checked = diagnostics(grouped, int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))
        attempts.append(dict(**checked, seconds=round(time.monotonic() - started, 1)))
        print(f"{name}: {attempts[-1]}", flush=True)
        if checked["converged"]:
            break
    require_convergence(checked)
    if b"".join(p.read_bytes() for p in sources) != source:
        raise RuntimeError("Race model source changed during fitting")
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k in KEEP}
    meta = dict(model=MODEL, key=key, settings=settings, diagnostics=checked, attempts=attempts,
                backend=jax.default_backend(), jax_version=jax.__version__, numpyro_version=numpyro.__version__,
                catalog=catalog, n_laps=len(C), n_races=C.event_id.nunique(),
                data_as_of=str(C.event_id.max()), first_event=str(C.event_id.min()))
    CACHE.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, meta=json.dumps(meta), **post)
    temporary.replace(path)
    jax.clear_caches()
    return post, meta


def covariate_effect(post, meta, entries, x_car=None, x_drv=None, *, mean=False) -> np.ndarray:
    """beta x x for each entry (draws, or their mean): the team's car covariates plus the
    driver's own, at the entry's race."""
    cat = meta["catalog"]
    n = meta["diagnostics"]["n_draws"]
    out = np.zeros((n, len(entries)))
    for key, cols, x, ids in (("beta_car", cat["car_covariates"], x_car, entries.team),
                              ("beta_drv", cat["driver_covariates"], x_drv, entries.driver_id)):
        if not cols:
            continue
        X = matrix(list(entries.event_id.astype(str) + "|" + ids.astype(str)), x[cols])
        out += post[key] @ X.T
    return out.mean(0) if mean else out


def predict(post, meta, entries, x_car=None, x_drv=None, *, noise=False, seed=0):
    """v1's per-entry draws (same random numbers for unseen states and race-day terms)
    plus the covariate effect."""
    return base.predict(post, meta, entries, noise=noise, seed=seed) + covariate_effect(
        post, meta, entries, x_car, x_drv)
