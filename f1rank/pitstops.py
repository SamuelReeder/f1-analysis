"""Pit stops as a team operations rating (stage 2; docs/racing_approach.md, "The qualities").

    python -m f1rank.pitstops

Pit-lane time per stop (pit entry to pit exit, so stationary time plus the lane itself),
relative to the race's median stop, which removes the circuit's pit-lane length and speed
limit:
- from 2018, FastF1: the car's pit-out time on the out-lap minus its pit-in time on the
  in-lap;
- 2011-2017, Jolpica's recorded stop durations (when data/processed/jolpica_pitstops.parquet
  exists).

Stops more than 5 s faster than the race median (drive-throughs) or more than 20 s slower
(repairs, stop-go penalties, red flags) are excluded, as are stops whose in-lap had a red
flag; the Student-t likelihood keeps the remaining slow stops (crew errors are part of the
rating) from dominating.

    relative time = race effect (small) + team (lineage, lasting)
                  + team-season deviation + Student-t noise

This rates team operations, separate from car performance and from driver ratings; in the
equal-car championship every team gets average stops, so it does not change the driver
comparison.

Held-out test: for each season S from 2019, fit on every earlier season plus the first half
of S's races and predict the second half, with and without the team terms. Paired
difference in log predictive density per race, bootstrap 95% interval over races; the team
rating is published only if the interval is above zero.

Writes outputs/pitstops/: summary.json, team_seasons.csv.
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

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "pitstops"
FAST_S, SLOW_S = -5.0, 20.0
FIRST_TEST = 2019
N_BOOT = 2000


def stops() -> pd.DataFrame:
    L = pd.read_parquet(PROCESSED / "race_laps.parquet").sort_values(["event_id", "driver_id", "lap_number"])
    L["next_out"] = L.groupby(["event_id", "driver_id"]).pit_out_time.shift(-1)
    P = L[L.pit_in_time.notna() & ~L.track_status.astype(str).str.contains("5")].copy()
    P["lane"] = P.next_out - P.pit_in_time
    P = P[["event_id", "driver_id", "lap_number", "lane"]].assign(source="fastf1")
    old = PROCESSED / "jolpica_pitstops.parquet"
    if old.exists():
        J = pd.read_parquet(old)
        J = J[J.event_id < P.event_id.min()].rename(columns={"lap": "lap_number", "duration": "lane"})
        P = pd.concat([P, J[["event_id", "driver_id", "lap_number", "lane"]].assign(source="jolpica")],
                      ignore_index=True)
    P = P.dropna(subset=["lane"])
    race = pd.read_parquet(PROCESSED / "race.parquet")[["event_id", "driver_id", "team"]]
    P = P.merge(race, on=["event_id", "driver_id"])
    P["rel"] = P.lane - P.groupby("event_id").lane.transform("median")
    P = P[P.rel.between(FAST_S, SLOW_S)].copy()
    P["season"] = P.event_id.str[:4].astype(int)
    P["team_season"] = P.team + "|" + P.season.astype(str)
    return P.reset_index(drop=True)


def model(d, team_terms=True):
    sd_race = numpyro.sample("sd_race", dist.HalfNormal(0.5))
    race = sd_race * numpyro.sample("race_z", dist.Normal(0, 1).expand([d["n_races"]]))
    mean = numpyro.sample("mu", dist.Normal(0, 2)) + race[d["race"]]
    if team_terms:
        sd_team = numpyro.sample("sd_team", dist.HalfNormal(1.0))
        sd_ts = numpyro.sample("sd_ts", dist.HalfNormal(1.0))
        team = sd_team * numpyro.sample("team_z", dist.Normal(0, 1).expand([d["n_teams"]]))
        ts = numpyro.deterministic("ts", sd_ts * numpyro.sample("ts_z", dist.Normal(0, 1).expand([d["n_team_seasons"]])))
        numpyro.deterministic("team", team)
        mean = mean + team[d["team"]] + jnp.where(d["ts"] >= 0, ts[jnp.maximum(d["ts"], 0)], 0.0)
    sigma = numpyro.sample("sigma", dist.HalfNormal(3.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    lp = dist.StudentT(nu, mean, sigma).log_prob(d["y"])
    numpyro.factor("ll", lp.sum())
    numpyro.deterministic("lp", lp)


def codes(P: pd.DataFrame, lev: dict) -> dict:
    get = lambda col, key: jnp.asarray(P[col].map({x: i for i, x in enumerate(lev[key])})  # noqa: E731
                                       .fillna(-1).astype(int).to_numpy())
    return {"y": jnp.asarray(P.rel.to_numpy(float)), "race": get("event_id", "races"), "team": get("team", "teams"),
            "ts": get("team_season", "team_seasons"), **{f"n_{k}": len(v) for k, v in lev.items()}}


def levels(P: pd.DataFrame) -> dict:
    return {"races": sorted(P.event_id.unique()), "teams": sorted(P.team.unique()),
            "team_seasons": sorted(P.team_season.unique())}


def fit(P, lev, warmup=500, samples=500, **kw) -> dict:
    from .artifacts import diagnostics, fit_until_converged

    def run(w, s, accept, seed):
        mcmc = MCMC(NUTS(partial(model, **kw), target_accept_prob=accept or 0.9), num_warmup=w, num_samples=s,
                    num_chains=4, chain_method="parallel", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed), codes(P, lev), extra_fields=("diverging",))
        return mcmc, diagnostics(
            {k: v for k, v in mcmc.get_samples(group_by_chain=True).items() if k != "lp"},
            int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))

    mcmc = fit_until_converged(run, warmup, samples)
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k != "lp"}
    post["_divergences"] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return post


def log_pred(post, P, lev, **kw) -> np.ndarray:
    samples = {k: v for k, v in post.items() if not k.startswith("_")}
    lp = np.asarray(Predictive(partial(model, **kw), posterior_samples=samples, return_sites=["lp"])(
        jax.random.PRNGKey(1), codes(P, lev))["lp"])
    m = lp.max(0)
    return m + np.log(np.exp(lp - m).mean(0))


def heldout(P: pd.DataFrame, rng) -> dict:
    rows = []
    lev = levels(P)
    for S in range(FIRST_TEST, int(P.season.max()) + 1):
        races = sorted(P[P.season == S].event_id.unique())
        first = set(races[:len(races) // 2])
        train = P[(P.season < S) | P.event_id.isin(first)]
        test = P[(P.season == S) & ~P.event_id.isin(first)]
        with_t = log_pred(fit(train, lev), test, lev)
        without = log_pred(fit(train, lev, team_terms=False), test, lev, team_terms=False)
        rows.append(pd.DataFrame({"event_id": test.event_id, "diff": with_t - without}))
        print(f"held out second half of {S}", flush=True)
    d = pd.concat(rows).groupby("event_id")["diff"].sum().to_numpy()
    boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_races": int(len(d)), "mean_diff_per_race": float(d.mean()), "ci95": [float(lo), float(hi)]}


def main() -> None:
    rng = np.random.default_rng(0)
    P = stops()
    lev = levels(P)
    post = fit(P, lev, warmup=1000, samples=1000)
    test = heldout(P, rng)
    q = lambda x: [round(float(v), 3) for v in np.percentile(x, [5, 50, 95])]  # noqa: E731
    total = post["team"][:, [lev["teams"].index(t.split("|")[0]) for t in lev["team_seasons"]]] + post["ts"]
    T = pd.DataFrame({"team_season": lev["team_seasons"], "rel_s_median": np.median(total, 0),
                      "rel_s_q05": np.percentile(total, 5, 0), "rel_s_q95": np.percentile(total, 95, 0)})
    T[["team", "season"]] = T.team_season.str.split("|", expand=True)
    n = P.groupby("team_season").size()
    T["stops"] = T.team_season.map(n).to_numpy()
    T["share_slow_3s"] = T.team_season.map(P.assign(slow=P.rel > 3).groupby("team_season").slow.mean()).to_numpy()
    summary = {"stops": int(len(P)), "races": int(P.event_id.nunique()),
               "sources": P.source.value_counts().to_dict(), "seasons": [int(P.season.min()), int(P.season.max())],
               "excluded_rule": f"relative time outside [{FAST_S}, {SLOW_S}] s, or red flag on the in-lap",
               "sd_team_q05_q50_q95": q(post["sd_team"]), "sd_team_season_q05_q50_q95": q(post["sd_ts"]),
               "sigma_q05_q50_q95": q(post["sigma"]), "nu_q05_q50_q95": q(post["nu"]),
               "divergences": post["_divergences"], "heldout_team_terms": test,
               "gate_team_ops_rating": bool(test["ci95"][0] > 0)}
    OUT.mkdir(parents=True, exist_ok=True)
    T.sort_values(["season", "rel_s_median"]).to_csv(OUT / "team_seasons.csv", index=False)
    from .artifacts import fit_record
    summary["fit_attempts"] = fit_record()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    from .artifacts import record
    record(OUT, [OUT / "summary.json", OUT / "team_seasons.csv"], model="pitstops-v2")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
