"""Consistency: how much a driver's clean race laps scatter, beyond the car and context.

    python -m f1rank.consistency

docs/racing_approach.md lists consistency as lap-time spread explained by race, car,
context and driver, with the driver part reported only if it passes the gates. For each
dry race used by stage A (racepace.py; from 2010 when the Jolpica stage A exists, with its
weaker observation model: unknown compounds), the stage-A regression is refitted without the
bootstrap (same clean laps, same terms: driver pace and degradation, lap trend, compounds,
dirty air, unpressured laps) and each driver's residual spread is measured robustly
(1.4826 x median absolute residual). Teammates share the car and most of the context, so
the comparison is the log ratio of teammates' spreads in the same race:

    log spread(a) - log spread(b) ~ Student-t(c[a] - c[b], sqrt(se^2 + extra^2))

with se from the large-sample variance of a log MAD (about 1.36 / laps, for each driver),
and c a driver consistency effect (shrunk towards 0; negative = steadier).

Held-out test (as for race pace): for each season from the third, driver effects fitted on
the earlier seasons predict that season's teammate gaps (pair-season means over at least 4
races, both drivers seen in training) against zero. Paired difference in squared error,
bootstrap 95% interval over pair-seasons; the driver ranking is published only if the
interval is below zero.

Writes outputs/consistency/: summary.json, drivers.csv, pairs.csv, and the driver-effect draws
for championship.py's entry test (heldout_effects.npz: per held-out season, fitted on the
seasons before it; driver_draws.npz: fitted on all seasons).
"""

import json
import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

from .racepace import (OLD_COMPOUNDS, OUT as RACE, PROCESSED, SLICKS, WET, clean_laps, design,  # noqa: E402
                       fastf1_races, huber, jolpica_races)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "consistency"
VAR_LOG_MAD = 1.36
MIN_RACES = 4
N_BOOT = 2000


def races():
    """(event_id, laps, compounds) for every dry race stage A used."""
    used = set(pd.read_csv(RACE / "stage_a_pairs.csv").event_id)
    for event_id, L, wet in fastf1_races():
        if event_id in used and not wet and not L.compound.isin(WET).any():
            yield event_id, L, SLICKS
    old = RACE / "stage_a_pairs_old.csv"
    if old.exists():
        used = set(pd.read_csv(old).event_id)
        for event_id, L, wet in jolpica_races():
            if event_id in used and not wet:
                yield event_id, L, OLD_COMPOUNDS


def pairs() -> pd.DataFrame:
    timeline = pd.read_parquet(PROCESSED / "timeline.parquet")
    rows = []
    for event_id, L, compounds in races():
        C = clean_laps(L, timeline[timeline.event_id == event_id], compounds).reset_index(drop=True)
        X, drivers, _ = design(C)
        r = C.y.to_numpy() - X @ huber(X, C.y.to_numpy())
        C["r"] = r
        spread = C.groupby("driver_id").r.agg(lambda x: 1.4826 * np.median(np.abs(x - np.median(x))))
        n = C.groupby("driver_id").size()
        team = L.groupby("driver_id").team.first()
        for tm, g in team[team.index.isin(spread.index)].groupby(team[team.index.isin(spread.index)]):
            if len(g) != 2:
                continue
            a, b = sorted(g.index)
            rows.append({"event_id": event_id, "team": tm, "a": a, "b": b,
                         "gap": float(np.log(spread[a]) - np.log(spread[b])),
                         "se": float(np.sqrt(VAR_LOG_MAD / n[a] + VAR_LOG_MAD / n[b])),
                         "spread_a": float(spread[a]), "spread_b": float(spread[b])})
    P = pd.DataFrame(rows)
    P["season"] = P.event_id.str[:4].astype(int)
    return P


def model(d):
    sd_c = numpyro.sample("sd_c", dist.HalfNormal(0.3))
    c = numpyro.deterministic("c", sd_c * numpyro.sample("c_z", dist.Normal(0, 1).expand([d["n"]])))
    extra = numpyro.sample("extra", dist.HalfNormal(0.5))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    mean = c[d["a"]] - c[d["b"]]
    numpyro.sample("y", dist.StudentT(nu, mean, jnp.sqrt(d["se"] ** 2 + extra ** 2)), obs=d["y"])


def fit(P: pd.DataFrame, drivers: list[str], warmup=800, samples=800) -> dict:
    from .artifacts import diagnostics, require_convergence
    idx = {x: i for i, x in enumerate(drivers)}
    d = {"n": len(drivers), "a": jnp.asarray(P.a.map(idx).to_numpy()), "b": jnp.asarray(P.b.map(idx).to_numpy()),
         "se": jnp.asarray(P.se.to_numpy()), "y": jnp.asarray(P.gap.to_numpy())}
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warmup, num_samples=samples, num_chains=4,
                chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), d, extra_fields=("diverging",))
    require_convergence(diagnostics(mcmc.get_samples(group_by_chain=True),
                                   int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())))
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    post["_divergences"] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return post


def heldout(P: pd.DataFrame, rng) -> dict:
    rows, effects = [], {}
    for S in sorted(P.season.unique())[2:]:
        train, test = P[P.season < S], P[P.season == S]
        drivers = sorted(set(train.a) | set(train.b))
        draws = fit(train, drivers)["c"]
        effects[f"{S}_drivers"], effects[f"{S}_c"] = np.array(drivers), draws.astype(np.float32)
        c = pd.Series(draws.mean(0), index=drivers)
        rows.append(test.assign(pred=c.reindex(test.a).to_numpy() - c.reindex(test.b).to_numpy(),
                                seen=test.a.isin(drivers) & test.b.isin(drivers)))
        print(f"held out {S}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "heldout_effects.npz", **effects)
    H = pd.concat(rows)
    H = H[H.seen]
    ps = H.groupby(["season", "a", "b"]).agg(n=("gap", "size"), obs=("gap", "mean"), pred=("pred", "mean"))
    ps = ps[ps.n >= MIN_RACES]
    d = ((ps.obs - ps.pred) ** 2 - ps.obs ** 2).to_numpy()
    boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_pair_seasons": int(len(d)), "mse_diff": float(d.mean()), "mse_diff_ci95": [float(lo), float(hi)],
            "units": "log ratio of teammates' lap-time spreads; mse in its square"}


def main() -> None:
    rng = np.random.default_rng(0)
    P = pairs()
    drivers = sorted(set(P.a) | set(P.b))
    post = fit(P, drivers, warmup=1000, samples=1000)
    test = heldout(P, rng)
    q = lambda x: [round(float(v), 4) for v in np.percentile(x, [5, 50, 95])]  # noqa: E731
    # split-half stability over races (reported, not a gate)
    odd = P.event_id.rank(method="dense").astype(int) % 2 == 1
    c1 = pd.Series(fit(P[odd], drivers, 500, 500)["c"].mean(0), index=drivers)
    c2 = pd.Series(fit(P[~odd], drivers, 500, 500)["c"].mean(0), index=drivers)
    summary = {"teammate_races": int(len(P)), "races": int(P.event_id.nunique()), "drivers": len(drivers),
               "median_spread_pct": float(np.median(np.r_[P.spread_a, P.spread_b])),
               "sd_driver_log_q05_q50_q95": q(post["sd_c"]), "extra_sd_q05_q50_q95": q(post["extra"]),
               "nu_q05_q50_q95": q(post["nu"]), "divergences": post["_divergences"],
               "split_half_corr_of_driver_means": float(c1.corr(c2)),
               "heldout_vs_zero": test, "gate_driver_ranking": bool(test["mse_diff_ci95"][1] < 0)}
    OUT.mkdir(parents=True, exist_ok=True)
    races = pd.concat([P.a, P.b]).value_counts()
    pd.DataFrame({"driver_id": drivers, "log_spread_effect_median": np.median(post["c"], 0),
                  "q05": np.percentile(post["c"], 5, 0), "q95": np.percentile(post["c"], 95, 0),
                  "teammate_races": races.reindex(drivers).to_numpy()}
                 ).sort_values("log_spread_effect_median").to_csv(OUT / "drivers.csv", index=False)
    np.savez_compressed(OUT / "driver_draws.npz", drivers=np.array(drivers), c=post["c"].astype(np.float32))
    P.to_csv(OUT / "pairs.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    from .artifacts import input_files, record
    record(OUT, [OUT / f for f in ("summary.json", "drivers.csv", "pairs.csv", "heldout_effects.npz", "driver_draws.npz")],
           model="consistency-v2", inputs=input_files()
           + [p for p in (RACE / "stage_a_pairs.csv", RACE / "stage_a_pairs_old.csv") if p.exists()],
           details={"training_before_seasons": [int(s) for s in sorted(P.season.unique())[2:]]})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
