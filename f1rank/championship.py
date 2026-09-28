"""Equal-car championship: a sequential weekend simulation with equal machinery (stage 2, step 8).

    python -m f1rank.championship

The weekend chain (docs/racing_approach.md), with each stage from the part of the racing
model that has passed its gates:

1. Qualifying: each driver's qualifying pace (stage 1) + weekend form + the car's one-event
   effect + session noise (stage 1's estimates) -> grid order. With equal cars the car term is the same for all.
2. Race given the grid: finishing order among cars still running, a Plackett-Luce ranking
   with strength  c_grid * (-log grid slot) + a_car * car + b_driver * driver, fitted on
   real classified finishers with the real cars (car and driver from stage 1). With equal
   cars the car term is the same for everyone and drops out. First-lap, overtaking
   and strategy effects are not simulated lap by lap: their average effect is folded into
   this grid-to-finish mapping, and no driver-specific part enters, because none has passed
   its gate yet (first-lap and overtaking ratings, race-specific pace).
3. Retirements: per-lap hazards from the reliability model with equal mechanical risk
   (the average team-season) and each driver's own-error effect only if driver error rates
   passed their gate; otherwise every driver has the average rate.
4. Points: the current system (25-18-15-12-10-8-6-4-2-1), no fastest-lap point.

Two versions of the driver term:
- in_team (headline): each driver keeps their team-specific effect ("every car made as
  fast as the average car, each driver keeping their fit with their current team").
- portable (experimental): portable skill only.

The race stage is checked on held-out seasons: grid + ratings must predict finishing orders
better than the grid alone and than the ratings alone (as in benchmark.py). Outputs are
posterior simulations: expected points per race, P(title), rank ranges.

Writes outputs/championship/: summary.json, standings.csv.
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
from numpyro.infer import MCMC, NUTS, Predictive  # noqa: E402

from .benchmark import FIRST_TEST, plackett_luce, race_orders  # noqa: E402
from .fit import FITS, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "championship"
POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1])
N_SEASONS = 2000       # simulated seasons (one posterior draw each)
N_BOOT = 2000


def finishers(R: pd.DataFrame) -> pd.DataFrame:
    """Classified finishers only (retirements are the reliability model's job), re-ranked."""
    from .timeline import FINISHED
    F = R[R.status.str.match(FINISHED.pattern)].copy()
    F["rank"] = F.groupby("event_id").cumcount()
    return F


def arrays(R: pd.DataFrame) -> dict:
    races = sorted(R.event_id.unique())
    width = int(R.groupby("event_id").size().max())
    ri = {e: i for i, e in enumerate(races)}
    shape = (len(races), width)
    car, drv, log_grid, mask = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape, bool)
    r, k = R.event_id.map(ri).to_numpy(), R["rank"].to_numpy()
    car[r, k], drv[r, k] = R.car, R.driver
    log_grid[r, k] = np.log(R.grid_slot)
    mask[r, k] = True
    return {"car": jnp.asarray(car), "driver": jnp.asarray(drv), "log_grid": jnp.asarray(log_grid),
            "mask": jnp.asarray(mask)}


def race_model(d, variant="grid_ratings"):
    s = jnp.zeros_like(d["car"])
    if variant in ("grid_ratings", "grid"):
        s = s - numpyro.sample("c_grid", dist.Normal(0, 5)) * d["log_grid"]
    if variant in ("grid_ratings", "ratings"):
        s = s + numpyro.sample("a_car", dist.Normal(0, 5)) * d["car"]
        s = s + numpyro.sample("b_driver", dist.Normal(0, 5)) * d["driver"]
    ll = plackett_luce(s, d["mask"])
    numpyro.factor("ll", ll.sum())
    numpyro.deterministic("ll_race", ll)


def fit_race(R, variant="grid_ratings", warmup=500, samples=500) -> dict:
    mcmc = MCMC(NUTS(partial(race_model, variant=variant)), num_warmup=warmup, num_samples=samples,
                num_chains=4, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), arrays(R))
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k != "ll_race"}


def log_pred(post, R, variant="grid_ratings") -> np.ndarray:
    ll = np.asarray(Predictive(partial(race_model, variant=variant), posterior_samples=post,
                               return_sites=["ll_race"])(jax.random.PRNGKey(1), arrays(R))["ll_race"])
    m = ll.max(0)
    return m + np.log(np.exp(ll - m).mean(0))


def heldout(F: pd.DataFrame, rng) -> dict:
    """Finishing orders of held-out seasons: grid + ratings vs grid only and vs ratings only."""
    lp = {v: [] for v in ("grid_ratings", "grid", "ratings")}
    for S in range(FIRST_TEST, F.season.max() + 1):
        train, test = F[F.season < S], F[F.season == S]
        for v in lp:
            lp[v].append(log_pred(fit_race(train, v), test, v))
        print(f"held out {S}", flush=True)
    lp = {v: np.concatenate(x) for v, x in lp.items()}
    out = {}
    for other in ("grid", "ratings"):
        d = lp["grid_ratings"] - lp[other]
        boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[f"vs_{other}"] = {"n_races": int(len(d)), "mean_diff_per_race": float(d.mean()),
                              "ci95": [float(lo), float(hi)]}
    return out


def simulate(Q: np.ndarray, form_sd: np.ndarray, noise_sd: np.ndarray, b_driver: np.ndarray, c_grid: np.ndarray,
             p_retire: np.ndarray, n_races: int, rng) -> np.ndarray:
    """Points per simulated season (draws x drivers). Q: driver qualifying pace per draw."""
    S, n = Q.shape
    points = np.zeros((S, n))
    for _ in range(n_races):
        quali = Q + rng.standard_normal((S, n)) * form_sd[:, None] + rng.standard_normal((S, n)) * noise_sd[:, None]
        slot = (-quali).argsort(1).argsort(1) + 1
        strength = b_driver[:, None] * Q - c_grid[:, None] * np.log(slot)  # equal cars: no car term
        g = rng.gumbel(size=(S, n))  # Plackett-Luce by the Gumbel-max trick
        running = rng.random((S, n)) >= p_retire[None, :]
        key = np.where(running, strength + g, -np.inf)
        order = (-key).argsort(1)
        pos = order.argsort(1)
        pts = np.where(pos < len(POINTS), POINTS[np.minimum(pos, len(POINTS) - 1)], 0) * running
        points += pts
    return points


def main() -> None:
    rng = np.random.default_rng(0)
    R = race_orders()
    F = finishers(R)
    test = heldout(F, rng)
    race_post = fit_race(F, warmup=800, samples=800)

    # stage 1 draws at the latest event
    from .design import build_design
    design = build_design(2010)
    post, _ = load(FITS / "main.npz", design)
    last = design.events.event_idx.max()
    rows = np.flatnonzero(design.entries.event_idx.to_numpy() == last)
    ids = design.entries.driver_id.to_numpy()[rows]
    flat = lambda k: post[k].reshape(-1, *post[k].shape[2:])  # noqa: E731
    skill = flat("skill")[:, rows]
    compat = flat("compat")[:, rows]
    versions = {"in_team": skill + compat, "portable": skill}
    pick = rng.choice(skill.shape[0], N_SEASONS, replace=True)
    # weekend-to-weekend variation in qualifying: driver form, the car's one-event effect and
    # session noise (each car is equal on average, not identical every weekend)
    form_sd = flat("sd_driver_form")[pick]
    noise_sd = np.sqrt(flat("sd_car_event")[pick] ** 2 + flat("sigma0")[pick] ** 2)
    race_pick = rng.integers(len(race_post["b_driver"]), size=N_SEASONS)

    # retirements: reliability model, equal cars; driver own-error effects only if gated
    rel = json.loads((ROOT / "outputs" / "reliability" / "summary.json").read_text())
    per100 = {c: v[1] / 100 for c, v in rel["hazard_per_100_laps_later_laps"].items()}
    lap1 = {c: v[1] / 100 for c, v in rel["hazard_lap1_per_100_starts"].items()}
    laps = 57  # median race distance since 2010
    p_common = 1 - np.exp(-(sum(lap1.values()) + (laps - 1) * sum(per100.values())))
    p_retire = np.full(len(ids), p_common)
    if rel["gate_driver_error_ranking"]:
        dr = pd.read_csv(ROOT / "outputs" / "reliability" / "drivers.csv").set_index("driver_id")
        mult = dr.own_error_rate_multiplier_median.reindex(ids).fillna(1.0).to_numpy()
        own = lap1["own_error"] + (laps - 1) * per100["own_error"]
        p_retire = 1 - np.exp(-(sum(lap1.values()) + (laps - 1) * sum(per100.values()) + own * (mult - 1)))
    n_races = int(design.events[design.events.season == design.events.season.max()].shape[0])
    n_races = max(n_races, 24)  # a full season

    names = design.drivers.set_index("driver_id").name
    out_rows, summary = [], {"heldout_race_stage": test, "n_races_simulated": n_races,
                             "simulated_seasons": N_SEASONS, "p_retire_per_race": float(p_common),
                             "driver_error_rates_used": bool(rel["gate_driver_error_ranking"]),
                             "b_driver_q05_q50_q95": [float(x) for x in np.percentile(race_post["b_driver"], [5, 50, 95])],
                             "a_car_q05_q50_q95": [float(x) for x in np.percentile(race_post["a_car"], [5, 50, 95])],
                             "c_grid_q05_q50_q95": [float(x) for x in np.percentile(race_post["c_grid"], [5, 50, 95])]}
    for version, Q in versions.items():
        Qs = Q[pick] - Q[pick].mean(1, keepdims=True)
        pts = simulate(Qs, form_sd, noise_sd, race_post["b_driver"][race_pick], race_post["c_grid"][race_pick],
                       p_retire, n_races, rng)
        rank = (-pts).argsort(1).argsort(1) + 1
        for i, d in enumerate(ids):
            out_rows.append({"version": version, "driver_id": d, "name": names.get(d, d),
                             "points_per_race": float(pts[:, i].mean() / n_races),
                             "p_title": float((rank[:, i] == 1).mean()),
                             "rank_median": float(np.median(rank[:, i])),
                             "rank_lo": int(np.percentile(rank[:, i], 5)), "rank_hi": int(np.percentile(rank[:, i], 95))})
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(out_rows).sort_values(["version", "points_per_race"], ascending=[True, False])
    table.to_csv(OUT / "standings.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    print(table[table.version == "in_team"].head(10).to_string())


if __name__ == "__main__":
    main()
