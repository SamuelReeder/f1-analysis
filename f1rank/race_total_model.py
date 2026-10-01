"""Joint lap model for total dry-race pace, independent of qualifying results.

Driver = lasting effect + current-season form. Car = team-season effect.
Race-specific car and driver deviations are integrated out of the headline.
Tyres, race lap, traffic and serial residual correlation enter the lap likelihood.
Only FastF1-era dry races are used: compounds before 2018 are not observed.
"""
import hashlib
import json
import os
from pathlib import Path
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer import MCMC, NUTS, init_to_median

from .artifacts import diagnostics, require_convergence
from .racepace import PROCESSED, REF_AGE, WET, clean_laps

MODEL = "total-dry-race-pace-v1"
ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "outputs" / "race_total" / "cache"


def laps():
    raw = pd.read_parquet(PROCESSED / "race_laps.parquet")
    timeline = pd.read_parquet(PROCESSED / "timeline.parquet")
    weather = pd.read_parquet(PROCESSED / "race_weather.parquet")
    rain = weather.groupby("event_id").rainfall.any()
    entries = pd.read_parquet(PROCESSED / "race.parquet")
    out = []
    for event, L in raw.groupby("event_id"):
        if rain.get(event, False) or L.compound.isin(WET).any():
            continue
        C = clean_laps(L, timeline[timeline.event_id == event])
        if C.driver_id.nunique() >= 8:
            out.append(C.assign(event_id=event))
    C = pd.concat(out, ignore_index=True).drop(columns="team").merge(
        entries[["event_id", "driver_id", "team"]], on=["event_id", "driver_id"],
        how="left", validate="many_to_one")
    if C.team.isna().any():
        raise ValueError("Race entry has no team lineage")
    C["season"] = C.event_id.str[:4].astype(int)
    return C.sort_values(["event_id", "stint_key", "lap_number"], ignore_index=True)


def design(C):
    def codes(values):
        indices, names = pd.factorize(values, sort=True)
        return indices, [str(x) for x in names]
    race, races = codes(C.event_id)
    driver, drivers = codes(C.driver_id)
    ds, driver_seasons = codes(C.season.astype(str) + "|" + C.driver_id)
    car, cars = codes(C.season.astype(str) + "|" + C.team)
    cr, car_races = codes(C.event_id + "|" + C.team)
    dr, driver_races = codes(C.event_id + "|" + C.driver_id)
    rc, compounds = codes(C.event_id + "|" + C.compound)
    years = sorted(C.season.astype(str).unique())
    year_index = {s: i for i, s in enumerate(years)}
    race_index = {s: i for i, s in enumerate(races)}
    driver_index = {s: i for i, s in enumerate(drivers)}
    ref = C.groupby("event_id").compound.agg(lambda s: s.value_counts().index[0])
    prev = ((C.stint_key == C.stint_key.shift()) & (C.event_id == C.event_id.shift())
            & (C.lap_number == C.lap_number.shift() + 1)).to_numpy()
    d = dict(y=C.y.to_numpy(), race=race, driver=driver, ds=ds, car=car, cr=cr, dr=dr, rc=rc,
             age=C.tyre_life.to_numpy(float) - REF_AGE,
             lap=(C.lap_number - C.groupby("event_id").lap_number.transform("mean")).to_numpy() / 10,
             nonref=(C.compound != C.event_id.map(ref)).to_numpy(),
             traffic=C[["close", "near", "unpressured"]].to_numpy(float), prev=prev,
             ds_year=np.array([year_index[s.split("|")[0]] for s in driver_seasons]),
             ds_driver=np.array([driver_index[s.split("|")[1]] for s in driver_seasons]),
             car_year=np.array([year_index[s.split("|")[0]] for s in cars]),
             cr_race=np.array([race_index[s.split("|")[0]] for s in car_races]),
             dr_race=np.array([race_index[s.split("|")[0]] for s in driver_races]),
             n_year=len(years),
             n_race=len(races), n_driver=len(drivers), n_ds=len(driver_seasons), n_car=len(cars),
             n_cr=len(car_races), n_dr=len(driver_races), n_rc=len(compounds))
    catalog = dict(drivers=drivers, driver_seasons=driver_seasons, cars=cars, races=races)
    return {k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in d.items()}, catalog


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
    # These deviations are context in all ablations, not lasting ability.
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
        package = numpyro.deterministic("package", centred(normal("package_raw", d["n_car"], 1.5),
                                                              d["car_year"], d["n_year"]))
        mean = mean + package[d["car"]]
    rho = numpyro.sample("rho", dist.Uniform(-.5, .95))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1))
    nu = numpyro.sample("nu", dist.Gamma(2, .1))
    residual = d["y"] - mean
    previous = jnp.concatenate([jnp.zeros(1), residual[:-1]])
    location = jnp.where(d["prev"], rho * previous, 0)
    scale = jnp.where(d["prev"], sigma, sigma / jnp.sqrt(1 - rho ** 2))
    numpyro.factor("laps", dist.StudentT(nu, location, scale).log_prob(residual).sum())


KEEP = {"skill", "form", "package", "sd_form", "sd_day", "sd_car_day", "rho", "sigma", "nu"}


def fit(C, name, *, driver=True, car=True, warmup=800, samples=800, tries=3):
    d, catalog = design(C)
    source = Path(__file__).read_bytes()
    settings = dict(driver=driver, car=car, warmup=warmup, samples=samples, tries=tries)
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
        dense = [("skill_raw", "package_raw")] if driver and car else False
        kernel = NUTS(model, target_accept_prob=.9, dense_mass=dense,
                      init_strategy=init_to_median(num_samples=15))
        mcmc = MCMC(kernel, num_warmup=warmup * (attempt + 1), num_samples=samples * 2 ** attempt,
                    num_chains=4, chain_method="vectorized" if jax.default_backend() == "gpu" else "parallel",
                    progress_bar=False)
        print(f"{name}: attempt {attempt + 1}, {len(C):,} laps, {C.event_id.nunique()} races", flush=True)
        mcmc.run(jax.random.PRNGKey(attempt), d, driver=driver, car=car, extra_fields=("diverging",))
        grouped = mcmc.get_samples(group_by_chain=True)
        checked = diagnostics(grouped,
                              int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))
        from numpyro.diagnostics import split_gelman_rubin
        mixing = []
        for parameter, values in grouped.items():
            values = np.asarray(values).reshape(4, samples * 2 ** attempt, -1)
            varying = np.ptp(values.reshape(-1, values.shape[-1]), axis=0) > 0
            if varying.any():
                rhat = np.asarray(split_gelman_rubin(values[:, :, varying]))
                mixing.append(dict(parameter=parameter, rhat=float(np.nanmax(rhat))))
        mixing.sort(key=lambda r: r["rhat"], reverse=True)
        attempts.append(dict(**checked, seconds=round(time.monotonic() - started, 1), worst_parameters=mixing[:5]))
        print(f"{name}: {attempts[-1]}", flush=True)
        if checked["converged"]:
            break
        CACHE.mkdir(parents=True, exist_ok=True)
        worst = {x["parameter"] for x in mixing[:5]}
        np.savez_compressed(CACHE / f"{name}.failed_{attempt}.npz",
                            **{k: np.asarray(v) for k, v in grouped.items() if k in worst})
    require_convergence(checked)
    if Path(__file__).read_bytes() != source:
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


def predict(post, meta, entries, *, noise=False, seed=0):
    """Joint per-entry draws in percent, optionally with future race-day variation.

    Unknown driver/season/car states receive prior draws, never a fitted zero.
    Shared car-day draws preserve covariance between teammates.
    """
    rng = np.random.default_rng(seed)
    cat = meta["catalog"]
    n = meta["diagnostics"]["n_draws"]
    p = {k: {name: i for i, name in enumerate(cat[k])} for k in ("drivers", "driver_seasons", "cars")}
    draws = np.zeros((n, len(entries)))
    memo = {}
    def unseen(key, scale):
        if key not in memo:
            memo[key] = rng.normal(size=n) * scale
        return memo[key]
    for j, r in enumerate(entries.itertuples()):
        season = str(r.event_id)[:4]
        ds, cs = season + "|" + r.driver_id, season + "|" + r.team
        if "skill" in post:
            draws[:, j] += (post["skill"][:, p["drivers"][r.driver_id]] if r.driver_id in p["drivers"]
                            else unseen("driver|" + r.driver_id, .5))
            draws[:, j] += (post["form"][:, p["driver_seasons"][ds]] if ds in p["driver_seasons"]
                            else unseen("form|" + ds, post["sd_form"]))
        if "package" in post:
            draws[:, j] += (post["package"][:, p["cars"][cs]] if cs in p["cars"]
                            else unseen("car|" + cs, 1.5))
        if noise:
            draws[:, j] += unseen("day|" + r.event_id + "|" + r.driver_id, post["sd_day"])
            draws[:, j] += unseen("car_day|" + r.event_id + "|" + r.team, post["sd_car_day"])
    return draws


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="full")
    parser.add_argument("--end")
    parser.add_argument("--warmup", type=int, default=800)
    parser.add_argument("--samples", type=int, default=800)
    parser.add_argument("--no-driver", action="store_true")
    parser.add_argument("--no-car", action="store_true")
    args = parser.parse_args()
    C = laps()
    if args.end:
        C = C[C.event_id <= args.end]
    fit(C, args.name, driver=not args.no_driver, car=not args.no_car,
        warmup=args.warmup, samples=args.samples)
