"""Simple results benchmark: a rank-ordered logit on finishing orders (stage 2, step 5).

    python -m f1rank.benchmark

Every later part of the racing model has to improve held-out results over this. For each
race from 2010, the full classification (finishers, then retirements by laps completed, as
Jolpica ranks them) is modelled as a Plackett-Luce ranking with strength

    ratings:            a * car + b * driver
    ratings + results:  a * car + b * driver + u[driver]    (u: driver results effect, shrunk)
    grid:               -c * log(grid slot)                  (pit-lane starts: last slot)
    grid + ratings:     -c * log(grid slot) + a * car + b * driver

car is stage 1's car rating at that circuit (percent of lap time; posterior median) and
driver is stage 1's pace in the current car. Both use qualifying only, including that
weekend's session, which precedes the race. (They come from the main fit, which also
smooths with later qualifying data; every model compared here uses the same inputs.)

Held-out: each season from 2014 is predicted by models fitted on the seasons before it.
Scores per race: log-likelihood of the observed order, Spearman correlation of predicted
and observed order, and teammate head-to-head accuracy. Paired differences per race get a
bootstrap 95% interval over races.

Writes outputs/benchmark/: summary.json, heldout_races.csv.
"""

import json
import os
from functools import partial
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RATINGS = ROOT / "outputs" / "ratings"
OUT = ROOT / "outputs" / "benchmark"
FIRST_SEASON, FIRST_TEST = 2010, 2014
N_BOOT = 2000
VARIANTS = ("ratings", "ratings_results", "grid", "grid_ratings")


def race_orders() -> pd.DataFrame:
    race = pd.read_parquet(PROCESSED / "race.parquet")
    race = race[race.event_id.str[:4].astype(int) >= FIRST_SEASON]
    ds = pd.read_parquet(RATINGS / "driver_series.parquet")[["event_id", "driver_id", "in_team_median"]]
    cs = pd.read_parquet(RATINGS / "car_series.parquet")[["event_id", "team", "at_circuit_median"]]
    R = race.merge(ds, on=["event_id", "driver_id"]).merge(cs, on=["event_id", "team"])
    R = R.rename(columns={"in_team_median": "driver", "at_circuit_median": "car"})
    n = R.groupby("event_id").driver_id.transform("size")
    R["grid_slot"] = np.where(R.grid > 0, R.grid, n)  # pit-lane start: last
    R["season"] = R.event_id.str[:4].astype(int)
    R = R.sort_values(["event_id", "position"]).reset_index(drop=True)
    R["rank"] = R.groupby("event_id").cumcount()  # 0 = winner, among rated starters
    return R


def arrays(R: pd.DataFrame, drivers: list[str]) -> dict:
    races = sorted(R.event_id.unique())
    width = int(R.groupby("event_id").size().max())
    ri = {e: i for i, e in enumerate(races)}
    di = {d: i for i, d in enumerate(drivers)}
    shape = (len(races), width)
    out = {k: np.zeros(shape) for k in ("car", "driver", "log_grid")}
    drv, mask = np.full(shape, -1), np.zeros(shape, bool)
    r, k = R.event_id.map(ri).to_numpy(), R["rank"].to_numpy()
    out["car"][r, k], out["driver"][r, k] = R.car, R.driver
    out["log_grid"][r, k] = np.log(R.grid_slot)
    drv[r, k] = R.driver_id.map(di).fillna(-1).astype(int)
    mask[r, k] = True
    return {**{k: jnp.asarray(v) for k, v in out.items()}, "drv": jnp.asarray(drv), "mask": jnp.asarray(mask),
            "n_drv": len(drivers)}


def plackett_luce(s, mask):
    """Log-likelihood of each race's order (rows sorted winner first; padding masked)."""
    s = jnp.where(mask, s, -1e9)  # padding: no probability (finite, so gradients stay finite)
    tail = jax.lax.cumlogsumexp(s[:, ::-1], axis=1)[:, ::-1]  # log sum over positions k..end
    return jnp.where(mask, s - tail, 0.0).sum(1)


def model(d, variant="ratings"):
    if variant in ("grid", "grid_ratings"):
        c = numpyro.sample("c", dist.Normal(0, 5))
        s = -c * d["log_grid"]
        if variant == "grid_ratings":
            s = s + numpyro.sample("a", dist.Normal(0, 5)) * d["car"] + numpyro.sample("b", dist.Normal(0, 5)) * d["driver"]
    else:
        a = numpyro.sample("a", dist.Normal(0, 5))
        b = numpyro.sample("b", dist.Normal(0, 5))
        s = a * d["car"] + b * d["driver"]
        if variant == "ratings_results":
            sd = numpyro.sample("sd_u", dist.HalfNormal(1.0))
            u = numpyro.deterministic("u", sd * numpyro.sample("u_z", dist.Normal(0, 1).expand([d["n_drv"]])))
            s = s + jnp.where(d["drv"] >= 0, u[jnp.maximum(d["drv"], 0)], 0.0)
    ll = plackett_luce(s, d["mask"])
    numpyro.factor("ll", ll.sum())
    numpyro.deterministic("ll_race", ll)
    numpyro.deterministic("strength", s)


def fit(R, drivers, variant, warmup=400, samples=400) -> dict:
    mcmc = MCMC(NUTS(partial(model, variant=variant)), num_warmup=warmup, num_samples=samples, num_chains=4,
                chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), arrays(R, drivers))
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k not in ("ll_race", "strength")}


def predict(post, R, drivers, variant) -> tuple[np.ndarray, np.ndarray]:
    """Log predictive density per race and posterior-mean strengths (races x slots)."""
    from numpyro.infer import Predictive
    out = Predictive(partial(model, variant=variant), posterior_samples=post,
                     return_sites=["ll_race", "strength"])(jax.random.PRNGKey(1), arrays(R, drivers))
    ll = np.asarray(out["ll_race"])
    m = ll.max(0)
    return m + np.log(np.exp(ll - m).mean(0)), np.asarray(out["strength"]).mean(0)


def race_scores(R: pd.DataFrame, strength: np.ndarray) -> pd.DataFrame:
    rows = []
    for i, (event_id, g) in enumerate(R.groupby("event_id", sort=True)):
        s = strength[i, g["rank"].to_numpy()]
        pred_order = (-s).argsort().argsort()
        rho = spearmanr(pred_order, g["rank"].to_numpy())[0]
        h2h = []
        for _, t in g.assign(s=s).groupby("team"):
            if len(t) == 2:
                h2h.append((t.s.iloc[0] > t.s.iloc[1]) == (t["rank"].iloc[0] < t["rank"].iloc[1]))
        rows.append({"event_id": event_id, "spearman": rho, "h2h_correct": np.sum(h2h), "h2h_n": len(h2h)})
    return pd.DataFrame(rows)


def paired(d: np.ndarray, rng) -> dict:
    boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean_diff_per_race": float(d.mean()), "ci95": [float(lo), float(hi)]}


def main() -> None:
    R = race_orders()
    drivers = sorted(R.driver_id.unique())
    rng = np.random.default_rng(0)
    rows = []
    for S in range(FIRST_TEST, R.season.max() + 1):
        train, test = R[R.season < S], R[R.season == S]
        per = {}
        for v in VARIANTS:
            post = fit(train, drivers, v)
            lp, strength = predict(post, test, drivers, v)
            sc = race_scores(test, strength).assign(variant=v, log_lik=lp, season=S)
            per[v] = sc
        rows += list(per.values())
        print(f"held out {S}", flush=True)
    H = pd.concat(rows, ignore_index=True)
    W = H.pivot(index="event_id", columns="variant", values="log_lik")
    summary = {"races": int(W.shape[0]), "seasons_held_out": [FIRST_TEST, int(R.season.max())],
               "per_variant": {v: {"mean_log_lik_per_race": float(H[H.variant == v].log_lik.mean()),
                                   "mean_spearman": float(H[H.variant == v].spearman.mean()),
                                   "teammate_h2h_accuracy": float(H[H.variant == v].h2h_correct.sum()
                                                                  / H[H.variant == v].h2h_n.sum())}
                               for v in VARIANTS},
               "ratings_results_minus_ratings": paired((W.ratings_results - W.ratings).to_numpy(), rng),
               "ratings_minus_grid": paired((W.ratings - W.grid).to_numpy(), rng),
               "grid_ratings_minus_grid": paired((W.grid_ratings - W.grid).to_numpy(), rng)}
    full = fit(R, drivers, "ratings_results", warmup=800, samples=800)
    summary["full_fit"] = {k: [round(float(x), 3) for x in np.percentile(full[k], [5, 50, 95])]
                           for k in ("a", "b", "sd_u")}
    OUT.mkdir(parents=True, exist_ok=True)
    H.to_csv(OUT / "heldout_races.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
