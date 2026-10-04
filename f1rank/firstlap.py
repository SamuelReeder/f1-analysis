"""First-lap performance: positions gained from the grid slot to the end of lap 1 (stage 2, step 6).

    python -m f1rank.firstlap

"First-lap performance" includes the launch, corner fighting and incidents; it is not a
pure start rating (telemetry could later separate the launch). Every starter from a grid
slot, in races (lap-1 positions from FastF1 from 2018, and from Jolpica's lap-by-lap
positions before that when data/processed/jolpica_laps.parquet exists) and sprint races
(FastF1, 2021 onward; the team is the one the driver raced for that weekend):

    gained = slot effect (what that slot gains on average)
           + side effect per circuit (odd vs even slots; the racing line differs by track)
           + start tyre (soft / medium / hard / wet; "unknown" before 2018)
           + sprint (a shift for sprint starts)
           + team (lineage, lasting across seasons) + car (team-season deviation, shrunk)
           + driver (shrunk) + Student-t noise

The lasting team term was added after the first build failed the transfer test: without it,
a team's start strength that carries over from season to season (launch systems, start
procedures, starting-tyre choices) can only be credited to drivers who stay, which fits
the pattern seen (driver effects helped overall but hurt drivers who changed team). The
model without it is still run and reported (`no_lasting_team`); the gate rule is unchanged.

Excluded: pit-lane starts and cars that did not complete lap 1 (the incident and
reliability models cover those). Cars in a contact incident on lap 1 stay in (fighting is
part of the quality); a sensitivity variant drops them.

Held-out: each season from FIRST_TEST is predicted from the seasons before it, with and
without driver effects; paired difference in log predictive density per race (a sprint is
its own race), bootstrap 95% interval over races. The same test on starts by drivers who changed team since their
previous season is the transfer test. Driver first-lap ratings get a standalone ranking
only if the overall interval is above zero and the transfer interval is not entirely below
zero. Also reported: the coverage of 90% predictive intervals.

Writes outputs/firstlap/: summary.json, drivers.csv, driver_draws.npz (full-data driver
effect draws, for the equal-car championship), and heldout_effects.npz (per held-out
season, posterior draws of the driver effects fitted on the seasons before it, for the
overall rating's entry test in championship.py).
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
FIRST_TEST = 2012  # first held-out season (docs/racing_approach.md, decisions fixed before the final runs)
N_BOOT = 2000
TYRE = {"HYPERSOFT": "soft", "ULTRASOFT": "soft", "SUPERSOFT": "soft", "SOFT": "soft", "MEDIUM": "medium",
        "HARD": "hard", "SUPERHARD": "hard", "INTERMEDIATE": "wet", "WET": "wet"}


def lap1(laps: pd.DataFrame) -> pd.DataFrame:
    return laps[laps.lap_number == 1][["event_id", "driver_id", "position", "compound"]].rename(
        columns={"position": "pos1"})


def starts(drop_contact: bool = False) -> pd.DataFrame:
    race = pd.read_parquet(PROCESSED / "race.parquet")
    events = pd.read_parquet(PROCESSED / "events.parquet")[["event_id", "circuit_id"]]
    L = lap1(pd.read_parquet(PROCESSED / "race_laps.parquet"))
    old = PROCESSED / "jolpica_laps.parquet"
    if old.exists():  # before FastF1: Jolpica lap-1 positions, start tyre unknown
        J = pd.read_parquet(old)
        J = J[(J.lap_number == 1) & (J.event_id < L.event_id.min())]
        L = pd.concat([L, J[["event_id", "driver_id", "position"]].rename(columns={"position": "pos1"})
                       .assign(compound="UNKNOWN")], ignore_index=True)
    S = race.merge(L, on=["event_id", "driver_id"]).assign(session="R")
    if drop_contact:
        T = pd.read_parquet(PROCESSED / "timeline.parquet")
        c = T[(T.kind == "incident_noted") & T.detail.str.startswith("contact") & (T.lap_start == 1)]
        S = S[~S.set_index(["event_id", "driver_id"]).index.isin(list(zip(c.event_id, c.driver_id)))]
    sprint = PROCESSED / "sprint_classification.parquet"
    if sprint.exists():
        C = pd.read_parquet(sprint)[["event_id", "driver_id", "grid_position"]].rename(columns={"grid_position": "grid"})
        C = C.merge(lap1(pd.read_parquet(PROCESSED / "sprint_laps.parquet")), on=["event_id", "driver_id"])
        C = C.merge(race[["event_id", "driver_id", "team"]], on=["event_id", "driver_id"])
        S = pd.concat([S, C.assign(session="S")], ignore_index=True)
    S = S.merge(events, on="event_id")
    S = S[(S.grid > 0) & S.pos1.notna()].copy()
    S["grid"] = S.grid.astype(int)
    S["start_id"] = S.event_id + "_" + S.session
    S["gained"] = S.grid - S.pos1
    S["tyre"] = S.compound.map(TYRE).fillna("medium").where(S.compound != "UNKNOWN", "unknown")
    S["side"] = (S.grid % 2).astype(int)  # 1 = odd slots (pole side)
    S["season"] = S.event_id.str[:4].astype(int)
    S["team_season"] = S.team + "|" + S.season.astype(str)
    return S.sort_values(["start_id", "grid"]).reset_index(drop=True)


def codes(S: pd.DataFrame, levels: dict) -> dict:
    get = lambda col, key: jnp.asarray(S[col].map({x: i for i, x in enumerate(levels[key])})  # noqa: E731
                                       .fillna(-1).astype(int).to_numpy())
    return {"y": jnp.asarray(S.gained.to_numpy(float)), "slot": jnp.asarray(np.minimum(S.grid.to_numpy(), 24) - 1),
            "circuit": get("circuit_id", "circuits"), "side": jnp.asarray(S.side.to_numpy()),
            "tyre": get("tyre", "tyres"), "ts": get("team_season", "team_seasons"), "drv": get("driver_id", "drivers"),
            "race": get("start_id", "races"), "team": get("team", "teams"),
            "sprint": jnp.asarray((S.session == "S").to_numpy(float)),
            **{f"n_{k}": len(v) for k, v in levels.items()}}


def model(d, driver_effects=True, lasting_team=True):
    slot = numpyro.sample("slot", dist.Normal(0, 3).expand([24]))
    sd_side = numpyro.sample("sd_side", dist.HalfNormal(1.0))
    side = sd_side * numpyro.sample("side_z", dist.Normal(0, 1).expand([d["n_circuits"]]))
    tyre = numpyro.sample("tyre", dist.Normal(0, 1).expand([d["n_tyres"]]))
    sprint = numpyro.sample("sprint", dist.Normal(0, 1))
    sd_car = numpyro.sample("sd_car", dist.HalfNormal(1.0))
    car = sd_car * numpyro.sample("car_z", dist.Normal(0, 1).expand([d["n_team_seasons"]]))
    mean = (slot[d["slot"]] + jnp.where(d["circuit"] >= 0, side[jnp.maximum(d["circuit"], 0)], 0.0) * d["side"]
            + tyre[d["tyre"]] + sprint * d["sprint"] + jnp.where(d["ts"] >= 0, car[jnp.maximum(d["ts"], 0)], 0.0))
    if lasting_team:
        sd_team = numpyro.sample("sd_team", dist.HalfNormal(1.0))
        team = sd_team * numpyro.sample("team_z", dist.Normal(0, 1).expand([d["n_teams"]]))
        mean = mean + jnp.where(d["team"] >= 0, team[jnp.maximum(d["team"], 0)], 0.0)
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
    return {"circuits": sorted(S.circuit_id.unique()), "tyres": sorted(set(TYRE.values()) | {"unknown"}),
            "teams": sorted(S.team.unique()),
            "team_seasons": sorted(S.team_season.unique()), "drivers": sorted(S.driver_id.unique()),
            "races": sorted(S.start_id.unique())}


def fit(S, lev, warmup=500, samples=500, **kw) -> dict:
    from .artifacts import diagnostics, fit_until_converged

    def run(w, s, accept, seed):
        mcmc = MCMC(NUTS(partial(model, **kw), target_accept_prob=accept or 0.9), num_warmup=w, num_samples=s,
                    num_chains=4, chain_method="parallel", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed), codes(S, lev), extra_fields=("diverging",))
        return mcmc, diagnostics(
            {k: v for k, v in mcmc.get_samples(group_by_chain=True).items() if k not in ("lp", "mu")},
            int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))

    mcmc = fit_until_converged(run, warmup, samples)
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


def heldout(S: pd.DataFrame, rng, lasting_team=True, effects: dict | None = None) -> dict:
    """Driver effects on held-out seasons: all starts, and (transfer test) starts by drivers
    in a different team from their previous season; plus 90% interval coverage. Also compares
    the lasting team term against none (both without driver effects). With `effects`, stores
    each season's driver-effect draws in it."""
    rows = []
    for season in range(FIRST_TEST, S.season.max() + 1):
        train, test = S[S.season < season], S[S.season == season]
        lev = levels(S)  # all levels, so held-out units get the prior
        lev["team_seasons"] = sorted(train.team_season.unique())  # new team-seasons: prior mean 0
        post = fit(train, lev, lasting_team=lasting_team)
        with_d, covered = log_pred(post, test, lev, rng=rng, lasting_team=lasting_team)
        without = log_pred(fit(train, lev, driver_effects=False, lasting_team=lasting_team), test, lev,
                           driver_effects=False, lasting_team=lasting_team)
        other = log_pred(fit(train, lev, driver_effects=False, lasting_team=not lasting_team), test, lev,
                         driver_effects=False, lasting_team=not lasting_team)
        if effects is not None:
            effects[str(season)] = post["driver"].astype(np.float32)
        last_team = train.sort_values("event_id").groupby("driver_id").team.last()
        moved = test.driver_id.map(last_team).notna().to_numpy() & (test.team != test.driver_id.map(last_team)).to_numpy()
        rows.append(pd.DataFrame({"start_id": test.start_id, "diff": with_d - without, "moved": moved,
                                  "covered": covered, "team_diff": (without - other) * (1 if lasting_team else -1)}))
        print(f"held out {season}", flush=True)
        jax.clear_caches()  # one compilation per season's shapes; free them
    H = pd.concat(rows)

    def paired(sub: pd.DataFrame) -> dict:
        d = sub.groupby("start_id")["diff"].sum().to_numpy()
        boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return {"n_races": int(len(d)), "n_starts": int(len(sub)), "mean_diff_per_race": float(d.mean()),
                "ci95": [float(lo), float(hi)]}

    team = paired(H.assign(diff=H.team_diff))
    return {**paired(H), "transfer": paired(H[H.moved]), "stayed": paired(H[~H.moved]),
            "cov90": float(H.covered.mean()), "lasting_team_vs_none_without_drivers": team}


def main() -> None:
    rng = np.random.default_rng(0)
    S = starts()
    lev = levels(S)
    post = fit(S, lev, warmup=1000, samples=1000)
    q = lambda x: [round(float(v), 3) for v in np.percentile(x, [5, 50, 95])]  # noqa: E731
    effects = {"drivers": np.array(lev["drivers"])}
    test = heldout(S, rng, effects=effects)
    test_old = heldout(S, rng, lasting_team=False)
    S2 = starts(drop_contact=True)
    post2 = fit(S2, levels(S2), warmup=1000, samples=1000)
    drivers = lev["drivers"]
    dr = pd.DataFrame({"driver_id": drivers, "gained_per_start_median": np.median(post["driver"], 0),
                       "q05": np.percentile(post["driver"], 5, 0), "q95": np.percentile(post["driver"], 95, 0),
                       "starts": S.groupby("driver_id").size().reindex(drivers).to_numpy()})
    no_contact = pd.Series(np.median(post2["driver"], 0), index=levels(S2)["drivers"])
    dr["gained_per_start_no_contact"] = dr.driver_id.map(no_contact)
    summary = {"starts": int(len(S)), "races": int((S.session == "R").groupby(S.start_id).first().sum()),
               "sprints": int((S.session == "S").groupby(S.start_id).first().sum()),
               "sprint_shift_q05_q50_q95": q(post["sprint"]),
               "seasons": [int(S.season.min()), int(S.season.max())],
               "sd_driver_q05_q50_q95": q(post["sd_driver"]), "sd_car_q05_q50_q95": q(post["sd_car"]),
               "sd_team_q05_q50_q95": q(post["sd_team"]),
               "sd_side_q05_q50_q95": q(post["sd_side"]), "sigma_q05_q50_q95": q(post["sigma"]),
               "tyre_effects_median": dict(zip(lev["tyres"], np.median(post["tyre"], 0).round(3).tolist())),
               "divergences": post["_divergences"], "heldout_driver_effects": test,
               "no_lasting_team": {"heldout_driver_effects": test_old,
                                   "gate_driver_ranking": bool(test_old["ci95"][0] > 0
                                                               and test_old["transfer"]["ci95"][1] >= 0)},
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
    np.savez_compressed(OUT / "heldout_effects.npz", **effects)
    np.savez_compressed(OUT / "driver_draws.npz", drivers=np.array(drivers), driver=post["driver"].astype(np.float32))
    from .artifacts import fit_record
    summary["fit_attempts"] = fit_record()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    from .artifacts import record
    record(OUT, [OUT / f for f in ("summary.json", "drivers.csv", "heldout_effects.npz", "driver_draws.npz")],
           model="firstlap-v2", details={"training_before_seasons": sorted(int(k) for k in effects if k != "drivers")})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
