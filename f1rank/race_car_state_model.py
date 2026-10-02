"""Total race pace with a within-season car state (pre-registered variant).

Identical to race_total_model except the car: a team-season level plus a random
walk over that team's dry races in the season, centred across teams at every race.
Decisions are fixed in docs/race_car_state.md. This is a separate file so the
cached v1 fits, which hash race_total_model.py, stay valid.
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

MODEL = "total-dry-race-pace-carstate-v1"
CACHE = base.ROOT / "outputs" / "race_car_state" / "cache"
SD_DRIFT = .1


def design(C):
    d, catalog = base.design(C)
    _, car_races = pd.factorize(C.event_id + "|" + C.team, sort=True)
    car_races = [str(x) for x in car_races]
    car_index = {name: i for i, name in enumerate(catalog["cars"])}
    cr_car = np.array([car_index[name[:4] + "|" + name.split("|")[1]] for name in car_races])
    # car_races are sorted by event, so a stable sort by car keeps race order within it.
    order = np.argsort(cr_car, kind="stable")
    first = np.ones(len(order), bool)
    first[1:] = cr_car[order][1:] != cr_car[order][:-1]
    d.update(cr_order=jnp.asarray(order), cr_first=jnp.asarray(first),
             cr_car_sorted=jnp.asarray(cr_car[order]))
    catalog = dict(catalog, car_races=car_races)
    return d, catalog


def model(d, driver=True, car=True):
    def normal(name, size, scale):
        return numpyro.sample(name, dist.Normal(0, scale).expand([size]))
    def centred(value, group, count):
        means = jnp.bincount(group, weights=value, length=count) / jnp.bincount(group, length=count)
        return value - means[group]
    def varying(name, size, scale, group=None, count=None):
        sd = numpyro.sample("sd_" + name, dist.HalfNormal(scale))
        value = sd * normal(name + "_z", size, 1)
        if group is not None:
            value = centred(value, group, count)
        return numpyro.deterministic(name, value)
    intercept = normal("intercept", d["n_race"], 3)
    trend = normal("trend", d["n_race"], 2)
    compound = normal("compound", d["n_rc"], 3)
    slope = normal("slope", d["n_rc"], .3)
    traffic = numpyro.sample("traffic", dist.Normal(0, 1).expand([d["n_race"], 3]))
    day = varying("day", d["n_dr"], .3, d["dr_race"], d["n_race"])
    car_day = varying("car_day", d["n_cr"], .4, d["cr_race"], d["n_race"])
    wear = varying("wear", d["n_dr"], .05, d["dr_race"], d["n_race"])
    mean = (intercept[d["race"]] + trend[d["race"]] * d["lap"]
            + jnp.where(d["nonref"], compound[d["rc"]], 0)
            + (slope[d["rc"]] + wear[d["dr"]]) * d["age"]
            + (traffic[d["race"]] * d["traffic"]).sum(-1)
            + day[d["dr"]] + car_day[d["cr"]])
    if driver:
        raw_skill = normal("skill_raw", d["n_driver"], .5)
        skill = numpyro.deterministic("skill", raw_skill - raw_skill.mean())
        form = varying("form", d["n_ds"], .2, d["ds_year"], d["n_year"])
        driver_pace = numpyro.deterministic("driver_pace", centred(
            skill[d["ds_driver"]] + form, d["ds_year"], d["n_year"]))
        mean = mean + driver_pace[d["ds"]]
    if car:
        package = numpyro.deterministic("package", d["car_basis"] @ normal(
            "package_contrast", d["n_car"] - d["n_year"], 1.5))
        sd_drift = numpyro.sample("sd_drift", dist.HalfNormal(SD_DRIFT))
        step = jnp.where(d["cr_first"], 0., sd_drift * normal("drift_z", d["n_cr"], 1))
        walk = jnp.cumsum(step)
        # Restart the cumulative sum at each team-season's first race.
        start = jax.ops.segment_max(jnp.where(d["cr_first"], walk, -jnp.inf),
                                    d["cr_car_sorted"], num_segments=d["n_car"])
        sorted_drift = walk - start[d["cr_car_sorted"]]
        drift = jnp.zeros(d["n_cr"]).at[d["cr_order"]].set(sorted_drift)
        drift = numpyro.deterministic("drift", centred(drift, d["cr_race"], d["n_race"]))
        mean = mean + package[d["car"]] + drift[d["cr"]]
    rho = numpyro.sample("rho", dist.Uniform(-.5, .95))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    nu = numpyro.sample("nu", dist.Gamma(2, .1))
    residual = d["y"] - mean
    previous = jnp.concatenate([jnp.zeros(1), residual[:-1]])
    location = jnp.where(d["prev"], rho * previous, 0)
    scale = jnp.where(d["prev"], sigma, sigma / jnp.sqrt(1 - rho ** 2))
    numpyro.factor("laps", dist.StudentT(nu, location, scale).log_prob(residual).sum())


KEEP = base.KEEP | {"drift", "sd_drift"}


def fit(C, name, *, warmup=800, samples=800, tries=3):
    """Full model only; the driver-only baseline is the cached v1 ablation."""
    d, catalog = design(C)
    source = Path(__file__).read_bytes() + Path(base.__file__).read_bytes()
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
        n = samples * 2 ** attempt
        mcmc = MCMC(kernel, num_warmup=warmup * (attempt + 1), num_samples=n, num_chains=4,
                    chain_method="vectorized" if jax.default_backend() == "gpu" else "parallel",
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
    if Path(__file__).read_bytes() + Path(base.__file__).read_bytes() != source:
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


def predict(post, meta, entries, steps, *, noise=False, seed=0):
    """Joint per-entry draws in percent; `steps` maps event_id -> k races ahead.

    The car is the team's last fitted state plus a k-step walk shared by teammates.
    """
    rng = np.random.default_rng(seed)
    cat = meta["catalog"]
    n = meta["diagnostics"]["n_draws"]
    drivers = {name: i for i, name in enumerate(cat["drivers"])}
    seasons = {name: i for i, name in enumerate(cat["driver_seasons"])}
    cars = {name: i for i, name in enumerate(cat["cars"])}
    last = {}
    for i, name in enumerate(cat["car_races"]):  # sorted by event: the last one wins
        event, team = name.split("|")
        last[event[:4] + "|" + team] = i
    draws = np.zeros((n, len(entries)))
    memo = {}
    def shared(key, scale):
        if key not in memo:
            memo[key] = rng.normal(size=n) * scale
        return memo[key]
    for j, r in enumerate(entries.itertuples()):
        season = str(r.event_id)[:4]
        ds, cs = season + "|" + r.driver_id, season + "|" + r.team
        if "skill" in post:
            draws[:, j] += (post["skill"][:, drivers[r.driver_id]] if r.driver_id in drivers
                            else shared("driver|" + r.driver_id, .5))
            draws[:, j] += (post["form"][:, seasons[ds]] if ds in seasons
                            else shared("form|" + ds, post["sd_form"]))
        if "package" in post:
            k = steps[r.event_id]
            draws[:, j] += (post["package"][:, cars[cs]] + post["drift"][:, last[cs]] if cs in cars
                            else shared("car|" + cs, 1.5))
            draws[:, j] += shared(f"walk|{r.event_id}|{r.team}", np.sqrt(k) * post["sd_drift"])
        if noise:
            draws[:, j] += shared("day|" + r.event_id + "|" + r.driver_id, post["sd_day"])
            draws[:, j] += shared("car_day|" + r.event_id + "|" + r.team, post["sd_car_day"])
    return draws
