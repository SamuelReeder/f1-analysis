"""Stage B of race pace: teammate gaps across races (see racepace.py).

    python -m f1rank.racepace stage-b

For teammates a < b in a race, stage A gives the pace gap and degradation gap with their
bootstrap covariance. The model:

    (pace gap, deg gap) ~ bivariate Student-t(nu, mean, stage-A covariance + extra variation)
    mean = (gamma * qualifying gap + u[a] - u[b],  v[a] - v[b])
    (u, v)[driver] ~ bivariate normal(0, sd_u, sd_v, correlation rho)

gamma links race pace to stage 1's pace in the current car at that event (qualifying gap);
u is a driver's race-specific pace (beyond what qualifying predicts), v their degradation
relative to the race's common slope (positive = less degradation), both shrunk towards 0.

Held-out test: for each season S from the third season on, the model is fitted on the
earlier seasons and predicts season S's teammate gaps. Per pair-season (mean over its
races) it compares the qualifying link alone (gamma * qualifying gap) with the link plus
the race-specific part, as a paired difference in squared error with a bootstrap 95%
interval over pair-seasons. The race-specific part gets its own ranking only if the
interval is below zero (docs/racing_approach.md). Degradation effects are tested the same
way against zero.

The qualifying gap uses stage 1's main fit, which uses all qualifying data. That is the
strongest qualifying baseline available (it includes the same weekend's qualifying, which
precedes the race), so it makes the test harder for the race-specific part.

Writes outputs/race/: stage_b_summary.json, heldout_pairs.csv, current_race_pace.csv.
"""

import json
import os

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

from .racepace import OUT, ROOT  # noqa: E402

RATINGS = ROOT / "outputs" / "ratings"
SEC_PER_PCT = 0.9  # as in stage 1: 1% of a 90 s lap
N_BOOT = 2000
MIN_HELDOUT_RACES = 4  # races a pair-season needs in the held-out test


def model(d):
    gamma = numpyro.sample("gamma", dist.Normal(1.0, 1.0))
    sd = jnp.stack([numpyro.sample("sd_u", dist.HalfNormal(0.2)), numpyro.sample("sd_v", dist.HalfNormal(0.05))])
    L = numpyro.sample("corr_chol", dist.LKJCholesky(2, 2.0))
    z = numpyro.sample("uv_z", dist.Normal(0, 1).expand([d["n_drivers"], 2]))
    uv = numpyro.deterministic("uv", z @ (sd[:, None] * L).T)
    extra = jnp.stack([numpyro.sample("extra_pace", dist.HalfNormal(0.2)),
                       numpyro.sample("extra_deg", dist.HalfNormal(0.05))])
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    a, b = d["a"], d["b"]
    mean = jnp.stack([gamma * d["dq"] + uv[a, 0] - uv[b, 0], uv[a, 1] - uv[b, 1]], axis=-1)
    cov = d["obs_cov"] + jnp.diag(extra ** 2)
    numpyro.sample("obs", dist.MultivariateStudentT(nu, mean, scale_tril=jnp.linalg.cholesky(cov)), obs=d["y"])


def load_pairs() -> pd.DataFrame:
    P = pd.read_csv(OUT / "stage_a_pairs.csv")
    series = pd.read_parquet(RATINGS / "driver_series.parquet")
    q = series.set_index(["event_id", "driver_id"]).in_team_median
    P["quali_gap"] = [q.get((e, a), np.nan) - q.get((e, b), np.nan) for e, a, b in zip(P.event_id, P.a, P.b)]
    P["season"] = P.event_id.str[:4].astype(int)
    return P.dropna(subset=["quali_gap"]).reset_index(drop=True)


def fit(P: pd.DataFrame, drivers: list[str], seed: int = 0, warmup: int = 800, samples: int = 800) -> dict:
    idx = {d: i for i, d in enumerate(drivers)}
    cov = np.stack([np.stack([P.pace_gap_se ** 2, P.pace_deg_cov], -1),
                    np.stack([P.pace_deg_cov, P.deg_gap_se ** 2], -1)], -2)
    cov = cov + np.eye(2) * 1e-8
    data = {"n_drivers": len(drivers), "a": jnp.asarray(P.a.map(idx).to_numpy()),
            "b": jnp.asarray(P.b.map(idx).to_numpy()), "dq": jnp.asarray(P.quali_gap.to_numpy()),
            "obs_cov": jnp.asarray(cov), "y": jnp.asarray(P[["pace_gap", "deg_gap"]].to_numpy())}
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warmup, num_samples=samples, num_chains=4,
                chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), data, extra_fields=("diverging",))
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    post["corr"] = post["corr_chol"][:, 1, 0]
    from numpyro.diagnostics import summary
    s = summary(mcmc.get_samples(group_by_chain=True))
    post["rhat_max"] = float(max(np.nanmax(v["r_hat"]) for v in s.values()))
    post["divergences"] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return post


def q(x) -> list[float]:
    return [round(float(v), 4) for v in np.percentile(x, [5, 50, 95])]


def heldout(P: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    seasons = sorted(P.season.unique())
    rows = []
    for S in seasons[2:]:
        train, test = P[P.season < S], P[P.season == S]
        drivers = sorted(set(train.a) | set(train.b))
        post = fit(train, drivers)
        u = pd.Series(post["uv"][:, :, 0].mean(0), index=drivers)
        v = pd.Series(post["uv"][:, :, 1].mean(0), index=drivers)
        g = post["gamma"].mean()
        t = test.assign(pred_quali=g * test.quali_gap,
                        pred_full=g * test.quali_gap + u.reindex(test.a).fillna(0).to_numpy()
                        - u.reindex(test.b).fillna(0).to_numpy(),
                        pred_deg=v.reindex(test.a).fillna(0).to_numpy() - v.reindex(test.b).fillna(0).to_numpy(),
                        both_seen=test.a.isin(drivers) & test.b.isin(drivers))
        rows.append(t)
        print(f"held out {S}: {len(test)} teammate races, gamma {g:.2f}", flush=True)
    H = pd.concat(rows, ignore_index=True)
    ps = H.groupby(["season", "a", "b"]).agg(
        n=("pace_gap", "size"), pace_gap=("pace_gap", "mean"), deg_gap=("deg_gap", "mean"),
        pred_quali=("pred_quali", "mean"), pred_full=("pred_full", "mean"), pred_deg=("pred_deg", "mean"),
        both_seen=("both_seen", "first")).reset_index()
    ps = ps[ps.n >= MIN_HELDOUT_RACES]

    def paired(obs, base, new, subset) -> dict:
        o, b_, n_ = (x[subset].to_numpy() for x in (obs, base, new))
        d = (o - n_) ** 2 - (o - b_) ** 2
        boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        rmse = lambda p: float(np.sqrt(np.mean((o - p) ** 2)))  # noqa: E731
        return {"n_pair_seasons": int(len(d)), "rmse_baseline": rmse(b_), "rmse_model": rmse(n_),
                "mse_diff": float(d.mean()), "mse_diff_ci95": [float(lo), float(hi)]}

    seen = ps.both_seen.to_numpy()
    zero = pd.Series(0.0, index=ps.index)
    summary = {
        "units": "percent of lap time (x 0.9 = seconds per 90 s lap); mse in percent squared",
        "race_specific_pace_vs_quali_link": paired(ps.pace_gap, ps.pred_quali, ps.pred_full, seen),
        "quali_link_vs_zero": paired(ps.pace_gap, zero, ps.pred_quali, np.ones(len(ps), bool)),
        "degradation_vs_zero": paired(ps.deg_gap, zero, ps.pred_deg, seen),
    }
    return H, summary


def stage_b() -> None:
    P = load_pairs()
    rng = np.random.default_rng(0)
    drivers = sorted(set(P.a) | set(P.b))
    post = fit(P, drivers, warmup=1000, samples=1000)
    H, test = heldout(P, rng)
    gate = test["race_specific_pace_vs_quali_link"]["mse_diff_ci95"][1] < 0
    gate_deg = test["degradation_vs_zero"]["mse_diff_ci95"][1] < 0
    summary = {
        "n_teammate_races": int(len(P)), "n_races": int(P.event_id.nunique()), "n_drivers": len(drivers),
        "seasons": sorted(int(s) for s in P.season.unique()),
        "gamma_q05_q50_q95": q(post["gamma"]), "sd_race_specific_q05_q50_q95": q(post["sd_u"]),
        "sd_degradation_q05_q50_q95": q(post["sd_v"]), "corr_pace_deg_q05_q50_q95": q(post["corr"]),
        "extra_pace_sd_q05_q50_q95": q(post["extra_pace"]), "nu_q05_q50_q95": q(post["nu"]),
        "rhat_max": post["rhat_max"], "divergences": post["divergences"],
        "heldout": test,
        "gate_race_specific_pace": bool(gate), "gate_degradation": bool(gate_deg),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    H.to_csv(OUT / "heldout_pairs.csv", index=False)
    # current grid: drivers at the latest event with a stage-1 rating
    cur = pd.read_csv(RATINGS / "current_drivers.csv")
    n_draws = len(post["gamma"])
    rows = []
    for d in cur.driver_id:
        i = drivers.index(d) if d in drivers else None
        u = post["uv"][:, i, 0] if i is not None else np.zeros(n_draws)  # no races yet: prior mean
        v = post["uv"][:, i, 1] if i is not None else np.zeros(n_draws)
        rows.append({"driver_id": d, "race_specific_s": np.median(u) * SEC_PER_PCT,
                     "race_specific_q05_s": np.percentile(u, 5) * SEC_PER_PCT,
                     "race_specific_q95_s": np.percentile(u, 95) * SEC_PER_PCT,
                     "degradation_s_per_10_laps": np.median(v) * 10 * SEC_PER_PCT,
                     "degradation_q05": np.percentile(v, 5) * 10 * SEC_PER_PCT,
                     "degradation_q95": np.percentile(v, 95) * 10 * SEC_PER_PCT,
                     "teammate_races": int(((P.a == d) | (P.b == d)).sum())})
    C = cur[["driver_id", "name", "team", "in_team_median_s"]].merge(pd.DataFrame(rows), on="driver_id")
    C.to_csv(OUT / "current_race_pace.csv", index=False)
    (OUT / "stage_b_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
