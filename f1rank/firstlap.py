"""First-lap performance: positions gained from the grid slot to the end of lap 1 (stage 2, step 6).

    python -m f1rank.firstlap

"First-lap performance" includes the launch, corner fighting and incidents; it is not a
pure start rating (telemetry could later separate the launch). 2018 onward (FastF1 lap-1
positions), every starter from a grid slot:

    gained = slot effect (what that slot gains on average)
           + side effect per circuit (odd vs even slots; the racing line differs by track)
           + start tyre (soft / medium / hard / wet)
           + car (team-season, shrunk) + driver (shrunk) + Student-t noise

Excluded: pit-lane starts and cars that did not complete lap 1 (the incident and
reliability models cover those). Cars in a contact incident on lap 1 stay in (fighting is
part of the quality); a sensitivity variant drops them.

Held-out: each season from 2020 is predicted from the seasons before it, with and without
driver effects; paired difference in log predictive density per race, bootstrap 95%
interval over races. The same test on starts by drivers who changed team since their
previous season is the transfer test. Driver first-lap ratings get a standalone ranking
only if the overall interval is above zero and the transfer interval is not entirely below
zero. Also reported: the coverage of 90% predictive intervals.

Writes outputs/firstlap/: summary.json, drivers.csv.
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
OUT = ROOT / "outputs" / "firstlap"
FIRST_TEST = 2020
N_BOOT = 2000
TYRE = {"HYPERSOFT": "soft", "ULTRASOFT": "soft", "SUPERSOFT": "soft", "SOFT": "soft", "MEDIUM": "medium",
        "HARD": "hard", "SUPERHARD": "hard", "INTERMEDIATE": "wet", "WET": "wet"}


def starts(drop_contact: bool = False) -> pd.DataFrame:
    laps = pd.read_parquet(PROCESSED / "race_laps.parquet")
    race = pd.read_parquet(PROCESSED / "race.parquet")
    events = pd.read_parquet(PROCESSED / "events.parquet")[["event_id", "circuit_id"]]
    l1 = laps[laps.lap_number == 1][["event_id", "driver_id", "position", "compound"]].rename(
        columns={"position": "pos1"})
    S = race.merge(l1, on=["event_id", "driver_id"]).merge(events, on="event_id")
    S = S[(S.grid > 0) & S.pos1.notna()].copy()
    if drop_contact:
        T = pd.read_parquet(PROCESSED / "timeline.parquet")
        c = T[(T.kind == "incident_noted") & T.detail.str.startswith("contact") & (T.lap_start == 1)]
        S = S[~S.set_index(["event_id", "driver_id"]).index.isin(list(zip(c.event_id, c.driver_id)))]
    S["gained"] = S.grid - S.pos1
    S["tyre"] = S.compound.map(TYRE).fillna("medium")
    S["side"] = (S.grid % 2).astype(int)  # 1 = odd slots (pole side)
    S["season"] = S.event_id.str[:4].astype(int)
    S["team_season"] = S.team + "|" + S.season.astype(str)
    return S.reset_index(drop=True)


def codes(S: pd.DataFrame, levels: dict) -> dict:
    get = lambda col, key: jnp.asarray(S[col].map({x: i for i, x in enumerate(levels[key])})  # noqa: E731
                                       .fillna(-1).astype(int).to_numpy())
    return {"y": jnp.asarray(S.gained.to_numpy(float)), "slot": jnp.asarray(np.minimum(S.grid.to_numpy(), 24) - 1),
            "circuit": get("circuit_id", "circuits"), "side": jnp.asarray(S.side.to_numpy()),
            "tyre": get("tyre", "tyres"), "ts": get("team_season", "team_seasons"), "drv": get("driver_id", "drivers"),
            "race": get("event_id", "races"), **{f"n_{k}": len(v) for k, v in levels.items()}}


def model(d, driver_effects=True):
    slot = numpyro.sample("slot", dist.Normal(0, 3).expand([24]))
    sd_side = numpyro.sample("sd_side", dist.HalfNormal(1.0))
    side = sd_side * numpyro.sample("side_z", dist.Normal(0, 1).expand([d["n_circuits"]]))
    tyre = numpyro.sample("tyre", dist.Normal(0, 1).expand([d["n_tyres"]]))
    sd_car = numpyro.sample("sd_car", dist.HalfNormal(1.0))
    car = sd_car * numpyro.sample("car_z", dist.Normal(0, 1).expand([d["n_team_seasons"]]))
    mean = (slot[d["slot"]] + jnp.where(d["circuit"] >= 0, side[jnp.maximum(d["circuit"], 0)], 0.0) * d["side"]
            + tyre[d["tyre"]] + jnp.where(d["ts"] >= 0, car[jnp.maximum(d["ts"], 0)], 0.0))
    if driver_effects:
        sd_d = numpyro.sample("sd_driver", dist.HalfNormal(1.0))
        drv = numpyro.deterministic("driver", sd_d * numpyro.sample("driver_z", dist.Normal(0, 1).expand([d["n_drivers"]])))
        mean = mean + jnp.where(d["drv"] >= 0, drv[jnp.maximum(d["drv"], 0)], 0.0)
    sigma = numpyro.sample("sigma", dist.HalfNormal(3.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    lp = dist.StudentT(nu, mean, sigma).log_prob(d["y"])
    numpyro.factor("ll", lp.sum())
    numpyro.deterministic("lp", lp)
    numpyro.deterministic("mu", mean)


def levels(S: pd.DataFrame) -> dict:
    return {"circuits": sorted(S.circuit_id.unique()), "tyres": sorted(set(TYRE.values())),
            "team_seasons": sorted(S.team_season.unique()), "drivers": sorted(S.driver_id.unique()),
            "races": sorted(S.event_id.unique())}


def fit(S, lev, warmup=500, samples=500, **kw) -> dict:
    mcmc = MCMC(NUTS(partial(model, **kw), target_accept_prob=0.9), num_warmup=warmup, num_samples=samples,
                num_chains=4, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), codes(S, lev), extra_fields=("diverging",))
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k not in ("lp", "mu")}
    post["_divergences"] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return post


def log_pred(post, S, lev, rng=None, **kw):
    """Log predictive density per start; with rng, also whether each observation falls in
    its 90% predictive interval."""
    samples = {k: v for k, v in post.items() if not k.startswith("_")}
    out = Predictive(partial(model, **kw), posterior_samples=samples, return_sites=["lp", "mu"])(
        jax.random.PRNGKey(1), codes(S, lev))
    lp = np.asarray(out["lp"])
    m = lp.max(0)
    dens = m + np.log(np.exp(lp - m).mean(0))
    if rng is None:
        return dens
    mu = np.asarray(out["mu"])
    rep = mu + post["sigma"][:, None] * rng.standard_t(post["nu"][:, None], mu.shape)
    lo, hi = np.percentile(rep, [5, 95], axis=0)
    return dens, (S.gained.to_numpy() >= lo) & (S.gained.to_numpy() <= hi)


def heldout(S: pd.DataFrame, rng) -> dict:
    """Driver effects on held-out seasons: all starts, and (transfer test) starts by drivers
    in a different team from their previous season; plus 90% interval coverage."""
    rows = []
    for season in range(FIRST_TEST, S.season.max() + 1):
        train, test = S[S.season < season], S[S.season == season]
        lev = levels(S)  # all levels, so held-out units get the prior
        lev["team_seasons"] = sorted(train.team_season.unique())  # new team-seasons: prior mean 0
        with_d, covered = log_pred(fit(train, lev), test, lev, rng=rng)
        without = log_pred(fit(train, lev, driver_effects=False), test, lev, driver_effects=False)
        last_team = train.sort_values("event_id").groupby("driver_id").team.last()
        moved = test.driver_id.map(last_team).notna().to_numpy() & (test.team != test.driver_id.map(last_team)).to_numpy()
        rows.append(pd.DataFrame({"event_id": test.event_id, "diff": with_d - without, "moved": moved,
                                  "covered": covered}))
        print(f"held out {season}", flush=True)
    H = pd.concat(rows)

    def paired(sub: pd.DataFrame) -> dict:
        d = sub.groupby("event_id")["diff"].sum().to_numpy()
        boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"n_races": int(len(d)), "n_starts": int(len(sub)), "mean_diff_per_race": float(d.mean()),
                "ci95": [float(lo), float(hi)]}

    return {**paired(H), "transfer": paired(H[H.moved]), "cov90": float(H.covered.mean())}


def main() -> None:
    rng = np.random.default_rng(0)
    S = starts()
    lev = levels(S)
    post = fit(S, lev, warmup=1000, samples=1000)
    q = lambda x: [round(float(v), 3) for v in np.percentile(x, [5, 50, 95])]  # noqa: E731
    test = heldout(S, rng)
    S2 = starts(drop_contact=True)
    post2 = fit(S2, levels(S2), warmup=1000, samples=1000)
    drivers = lev["drivers"]
    dr = pd.DataFrame({"driver_id": drivers, "gained_per_start_median": np.median(post["driver"], 0),
                       "q05": np.percentile(post["driver"], 5, 0), "q95": np.percentile(post["driver"], 95, 0),
                       "starts": S.groupby("driver_id").size().reindex(drivers).to_numpy()})
    no_contact = pd.Series(np.median(post2["driver"], 0), index=levels(S2)["drivers"])
    dr["gained_per_start_no_contact"] = dr.driver_id.map(no_contact)
    summary = {"starts": int(len(S)), "races": int(S.event_id.nunique()),
               "seasons": [int(S.season.min()), int(S.season.max())],
               "sd_driver_q05_q50_q95": q(post["sd_driver"]), "sd_car_q05_q50_q95": q(post["sd_car"]),
               "sd_side_q05_q50_q95": q(post["sd_side"]), "sigma_q05_q50_q95": q(post["sigma"]),
               "tyre_effects_median": dict(zip(lev["tyres"], np.median(post["tyre"], 0).round(3).tolist())),
               "divergences": post["_divergences"], "heldout_driver_effects": test,
               # standalone ranking: held-out improvement overall, and not worse for drivers who changed
               # team (the doc's transfer test; this threshold was written down after the transfer result
               # was seen, but that result, an interval entirely below zero, fails any version of it)
               "gate_driver_ranking": bool(test["ci95"][0] > 0 and test["transfer"]["ci95"][1] >= 0),
               "sensitivity_no_lap1_contact": {
                   "starts": int(len(S2)),
                   "driver_rank_corr": float(pd.Series(dr.gained_per_start_median).corr(
                       dr.gained_per_start_no_contact, method="spearman"))}}
    OUT.mkdir(parents=True, exist_ok=True)
    dr.sort_values("gained_per_start_median", ascending=False).to_csv(OUT / "drivers.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
