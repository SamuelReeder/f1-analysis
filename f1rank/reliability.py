"""Reliability and driver errors: competing risks per lap (stage 2, step 4 of the racing plan).

    python -m f1rank.reliability

Every starter is at risk of retiring on each lap they start. Each retirement comes from one
of the timeline's causes (mechanical, own error, other driver, external, team operations,
unknown), and its cause is uncertain: it counts as a fractional event for each cause, in
proportion to the timeline's cause probabilities (the expected complete-data likelihood).
A free mixture over causes is not used: with a separate hazard for "unknown", which gets
some probability for every retirement, it can absorb every event. A retirement ends
exposure to all causes (censoring), and the laps completed before it stay in the data.

Hazard per lap, by cause (log scale):
    mechanical     base + lap-1 effect + era + supplier-season effect (power unit, shared by
                   customer teams; shrunk) + team-season effect (shrunk)
    own error      base + lap-1 effect + team-season effect + traffic + driver effect (shrunk)
    other causes   base + lap-1 effect
    every cause    + wet-race effect (one coefficient per cause for FastF1's wet races from
                   2018, another for pre-2018 races flagged wet by the precipitation proxy,
                   which is less accurate; conditions.py)
The first lap has its own rates because incidents cluster there. Eras are the regulation
periods (lineage.REGULATION_RESETS); suppliers come from data/reference/power_units.csv
(powerunits.py). Traffic is the car's share of green laps within 1 s of the car ahead
(conditions.py), centred; driver-races without lap data get the mean.

Data: races from 2010 (Jolpica classification, timeline causes). Not at risk: did not
start. Disqualified and withdrawn cars are treated as not retiring (censored at their laps).

Held-out tests (each season from 2014 predicted from the seasons before it), as paired
differences in log predictive density per race with a bootstrap 95% interval over races:
- driver effects on own-error risk vs the same model without them (both keep the
  team-season terms). Driver error rates get a standalone ranking only if the interval is
  above zero.
- team-season mechanical effects: within each held-out season, the second half of the
  season is predicted from its first half (plus earlier seasons), with vs without them
  (supplier-season and era terms kept in both).
- supplier-season mechanical effects: the same design, with vs without them (team-season
  terms kept in both).
Writes outputs/reliability/: summary.json, drivers.csv, team_seasons.csv, supplier_seasons.csv.
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

from .lineage import REGULATION_RESETS  # noqa: E402
from .timeline import CAUSES, FINISHED, P_COLS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "reliability"
FIRST_SEASON = 2010
FIRST_TEST = 2014
N_BOOT = 2000


def driver_races() -> pd.DataFrame:
    """One row per starter: exposure on lap 1 and later laps, retirement and cause probabilities."""
    race = pd.read_parquet(PROCESSED / "race.parquet")
    race = race[race.event_id.str[:4].astype(int) >= FIRST_SEASON].copy()
    T = pd.read_parquet(PROCESSED / "timeline.parquet")
    dns = set(zip(T.event_id[T.kind == "did_not_start"], T.driver_id[T.kind == "did_not_start"]))
    race = race[[(e, d) not in dns for e, d in zip(race.event_id, race.driver_id)]]
    ret = T[T.kind == "retirement"][["event_id", "driver_id", "car_lap", *P_COLS]]
    race = race.merge(ret, on=["event_id", "driver_id"], how="left", indicator=True)
    race["retired"] = race.pop("_merge") == "both"
    # laps started: completed laps, plus the lap a retiring car stopped in
    race["laps_started"] = race.laps + race.retired.astype(int)
    race = race[race.laps_started > 0].copy()
    race["event_lap1"] = race.retired & (race.car_lap == 1)
    race["exp_lap1"] = 1.0
    race["exp_later"] = (race.laps_started - 1).astype(float)
    race[P_COLS] = race[P_COLS].fillna(0.0)
    race["season"] = race.event_id.str[:4].astype(int)
    race["team_season"] = race.team + "|" + race.season.astype(str)
    race["era"] = np.searchsorted(sorted(REGULATION_RESETS), race.season.to_numpy(), side="right")
    pu = pd.read_csv(ROOT / "data" / "reference" / "power_units.csv")
    race = race.merge(pu[["season", "team", "supplier"]], on=["season", "team"], how="left")
    assert race.supplier.notna().all(), "team-season without a power-unit supplier"
    race["supplier_season"] = race.supplier + "|" + race.season.astype(str)
    cond = pd.read_parquet(PROCESSED / "race_conditions.parquet")
    race = race.merge(cond[["event_id", "wet", "wet_source"]], on="event_id", how="left")
    race["wet_fastf1"] = (race.wet.fillna(False) & (race.wet_source == "fastf1")).astype(float)
    race["wet_proxy"] = (race.wet.fillna(False) & (race.wet_source == "precipitation")).astype(float)
    tr = pd.read_parquet(PROCESSED / "race_traffic.parquet")[["event_id", "driver_id", "traffic_share"]]
    race = race.merge(tr, on=["event_id", "driver_id"], how="left")
    race["has_traffic"] = race.traffic_share.notna()
    race["traffic"] = (race.traffic_share - race.traffic_share.mean()).fillna(0.0)
    finished = race.status.str.match(FINISHED.pattern)
    assert not (finished & race.retired).any()
    return race.reset_index(drop=True)


def model(d, driver_effects=True, team_effects=True, supplier_effects=True):
    k = len(CAUSES)
    base = numpyro.sample("base", dist.Normal(-7.0, 2.0).expand([k]))
    lap1 = numpyro.sample("lap1", dist.Normal(0.0, 3.0).expand([k]))
    unit = jnp.zeros((d["n"], k))              # unit effects (log scale), same on lap 1 and later
    wet_ff = numpyro.sample("wet_fastf1", dist.Normal(0.0, 1.0).expand([k]))
    wet_px = numpyro.sample("wet_proxy", dist.Normal(0.0, 1.0).expand([k]))
    unit = unit + d["wet_ff"][:, None] * wet_ff + d["wet_px"][:, None] * wet_px
    era = numpyro.sample("era", dist.Normal(0.0, 1.0).expand([d["n_era"]]))
    unit = unit.at[:, 0].add(era[d["era"]])
    b_traffic = numpyro.sample("b_traffic", dist.Normal(0.0, 2.0))
    unit = unit.at[:, 1].add(b_traffic * d["traffic"])
    if supplier_effects:
        sd_s = numpyro.sample("sd_supplier", dist.HalfNormal(1.0))
        sup = numpyro.deterministic("supplier", sd_s * numpyro.sample("supplier_z", dist.Normal(0, 1).expand([d["n_ss"]])))
        unit = unit.at[:, 0].add(jnp.where(d["ss"] >= 0, sup[jnp.maximum(d["ss"], 0)], 0.0))
    ts = d["ts"]
    known = ts >= 0
    # own-error risk always has a team-season term (the car and team context), so the driver
    # test removes only the driver terms
    sd_to = numpyro.sample("sd_team_own", dist.HalfNormal(1.0))
    to = sd_to * numpyro.sample("team_own_z", dist.Normal(0, 1).expand([d["n_ts"]]))
    unit = unit.at[:, 1].add(jnp.where(known, to[jnp.maximum(ts, 0)], 0.0))
    if team_effects:
        sd_t = numpyro.sample("sd_team", dist.HalfNormal(1.0))
        t = numpyro.deterministic("team", sd_t * numpyro.sample("team_z", dist.Normal(0, 1).expand([d["n_ts"]])))
        unit = unit.at[:, 0].add(jnp.where(known, t[jnp.maximum(ts, 0)], 0.0))
    if driver_effects:
        sd_d = numpyro.sample("sd_driver", dist.HalfNormal(1.0))
        dr = numpyro.deterministic("driver", sd_d * numpyro.sample("driver_z", dist.Normal(0, 1).expand([d["n_drv"]])))
        unit = unit.at[:, 1].add(jnp.where(d["drv"] >= 0, dr[jnp.maximum(d["drv"], 0)], 0.0))
    log_first = base + lap1 + unit
    log_later = base + unit
    cum = d["exp1"][:, None] * jnp.exp(log_first) + d["exp_later"][:, None] * jnp.exp(log_later)
    log_event = jnp.where(d["ev1"][:, None], log_first, log_later)
    # uncertain causes enter as fractional events: each retirement adds its timeline cause
    # probabilities to the event counts of the causes (the expected complete-data likelihood)
    ll = (d["p"] * log_event).sum(1) - cum.sum(1)
    numpyro.factor("ll", (ll * d["w"]).sum())
    numpyro.deterministic("ll_i", ll)


SUPPLIER_SEASONS: list[str] = []


def arrays(R: pd.DataFrame, drivers, team_seasons, weight=None) -> dict:
    di = {x: i for i, x in enumerate(drivers)}
    ti = {x: i for i, x in enumerate(team_seasons)}
    si = {x: i for i, x in enumerate(SUPPLIER_SEASONS)}
    return {"n": len(R), "n_drv": len(drivers), "n_ts": len(team_seasons),
            "n_ss": len(SUPPLIER_SEASONS), "n_era": len(REGULATION_RESETS) + 1,
            "ss": jnp.asarray(R.supplier_season.map(si).fillna(-1).astype(int).to_numpy()),
            "era": jnp.asarray(R.era.to_numpy()), "traffic": jnp.asarray(R.traffic.to_numpy()),
            "wet_ff": jnp.asarray(R.wet_fastf1.to_numpy()), "wet_px": jnp.asarray(R.wet_proxy.to_numpy()),
            "drv": jnp.asarray(R.driver_id.map(di).fillna(-1).astype(int).to_numpy()),
            "ts": jnp.asarray(R.team_season.map(ti).fillna(-1).astype(int).to_numpy()),
            "exp1": jnp.asarray(R.exp_lap1.to_numpy()), "exp_later": jnp.asarray(R.exp_later.to_numpy()),
            "ev1": jnp.asarray(R.event_lap1.to_numpy()), "retired": jnp.asarray(R.retired.to_numpy()),
            "p": jnp.asarray(R[P_COLS].to_numpy()),
            "w": jnp.asarray(np.ones(len(R)) if weight is None else weight)}


def fit(R: pd.DataFrame, drivers, team_seasons, weight=None, warmup=500, samples=500, **kw) -> dict:
    from functools import partial
    from .artifacts import diagnostics, require_convergence
    mcmc = MCMC(NUTS(partial(model, **kw), target_accept_prob=0.9), num_warmup=warmup, num_samples=samples,
                num_chains=4, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), arrays(R, drivers, team_seasons, weight), extra_fields=("diverging",))
    require_convergence(diagnostics(
        {k: v for k, v in mcmc.get_samples(group_by_chain=True).items() if k != "ll_i"},
        int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())))
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k != "ll_i"}
    post["divergences"] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return post


def log_pred(post: dict, R: pd.DataFrame, drivers, team_seasons, **kw) -> np.ndarray:
    """Log predictive density per row (log mean over draws of the likelihood)."""
    from numpyro.infer import Predictive
    keep = {k: v for k, v in post.items() if k != "divergences"}
    ll = Predictive(lambda d: model(d, **kw), posterior_samples=keep, return_sites=["ll_i"])(
        jax.random.PRNGKey(1), arrays(R, drivers, team_seasons))["ll_i"]
    ll = np.asarray(ll)
    m = ll.max(0)
    return m + np.log(np.exp(ll - m).mean(0))


def paired(diff_by_race: pd.Series, rng) -> dict:
    d = diff_by_race.to_numpy()
    boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_races": int(len(d)), "mean_diff_per_race": float(d.mean()), "ci95": [float(lo), float(hi)]}


def heldout(R: pd.DataFrame, rng) -> dict:
    drv_rows, team_rows, sup_rows = [], [], []
    for S in range(FIRST_TEST, R.season.max() + 1):
        train, test = R[R.season < S], R[R.season == S]
        drivers = sorted(R.driver_id.unique())
        # drivers: unseen drivers get the prior mean (index still valid, no data)
        tss = sorted(train.team_season.unique())
        full = fit(train, drivers, tss, team_effects=False)
        base = fit(train, drivers, tss, team_effects=False, driver_effects=False)
        lp_full = log_pred(full, test, drivers, tss, team_effects=False)
        lp_base = log_pred(base, test, drivers, tss, team_effects=False, driver_effects=False)
        drv_rows.append(pd.DataFrame({"event_id": test.event_id, "diff": lp_full - lp_base}))
        # team-season and supplier-season effects: second half of season S from its first half
        # plus earlier seasons
        rounds = test.event_id.str[5:].astype(int)
        first, second = test[rounds <= rounds.median()], test[rounds > rounds.median()]
        tr = pd.concat([train, first])
        tss = sorted(tr.team_season.unique())
        both = fit(tr, drivers, tss, driver_effects=False)
        no_team = fit(tr, drivers, tss, driver_effects=False, team_effects=False)
        no_sup = fit(tr, drivers, tss, driver_effects=False, supplier_effects=False)
        lp_both = log_pred(both, second, drivers, tss, driver_effects=False)
        team_rows.append(pd.DataFrame({"event_id": second.event_id, "diff": lp_both - log_pred(
            no_team, second, drivers, tss, driver_effects=False, team_effects=False)}))
        sup_rows.append(pd.DataFrame({"event_id": second.event_id, "diff": lp_both - log_pred(
            no_sup, second, drivers, tss, driver_effects=False, supplier_effects=False)}))
        print(f"held out {S}", flush=True)
        jax.clear_caches()  # one compilation per season's shapes; free them
    by_race = lambda rows: pd.concat(rows).groupby("event_id")["diff"].sum()  # noqa: E731
    return {"driver_effects": paired(by_race(drv_rows), rng), "team_season_effects": paired(by_race(team_rows), rng),
            "supplier_season_effects": paired(by_race(sup_rows), rng)}


def main() -> None:
    R = driver_races()
    SUPPLIER_SEASONS[:] = sorted(R.supplier_season.unique())
    rng = np.random.default_rng(0)
    drivers, tss = sorted(R.driver_id.unique()), sorted(R.team_season.unique())
    post = fit(R, drivers, tss, warmup=1000, samples=1000)
    test = heldout(R, rng)
    gate_drivers = test["driver_effects"]["ci95"][0] > 0
    gate_teams = test["team_season_effects"]["ci95"][0] > 0
    gate_suppliers = test["supplier_season_effects"]["ci95"][0] > 0
    q = lambda x: [round(float(v), 4) for v in np.percentile(x, [5, 50, 95])]  # noqa: E731
    laps = float(R.exp_lap1.sum() + R.exp_later.sum())
    summary = {
        "seasons": [int(R.season.min()), int(R.season.max())], "driver_races": int(len(R)),
        "retirements": int(R.retired.sum()), "laps_at_risk": laps,
        "expected_events_by_cause": {c: round(float(R[f"p_{c}"].sum()), 1) for c in CAUSES},
        "hazard_per_100_laps_later_laps": {c: q(100 * np.exp(post["base"][:, i])) for i, c in enumerate(CAUSES)},
        "hazard_lap1_per_100_starts": {c: q(100 * np.exp(post["base"][:, i] + post["lap1"][:, i]))
                                       for i, c in enumerate(CAUSES)},
        "sd_driver_own_error_log_q05_q50_q95": q(post["sd_driver"]),
        "sd_team_season_mechanical_log_q05_q50_q95": q(post["sd_team"]),
        "sd_supplier_season_mechanical_log_q05_q50_q95": q(post["sd_supplier"]),
        "era_mechanical_log_q05_q50_q95": {str(i): q(post["era"][:, i]) for i in range(post["era"].shape[1])},
        "wet_fastf1_log_q05_q50_q95": {c: q(post["wet_fastf1"][:, i]) for i, c in enumerate(CAUSES)},
        "wet_proxy_log_q05_q50_q95": {c: q(post["wet_proxy"][:, i]) for i, c in enumerate(CAUSES)},
        "traffic_own_error_log_q05_q50_q95": q(post["b_traffic"]),
        "driver_races_with_traffic": int(R.has_traffic.sum()),
        "wet_races": {"fastf1": int(R[R.wet_fastf1 > 0].event_id.nunique()),
                      "precipitation_proxy": int(R[R.wet_proxy > 0].event_id.nunique())},
        "divergences": post["divergences"], "heldout": test,
        "gate_driver_error_ranking": bool(gate_drivers), "gate_team_reliability": bool(gate_teams),
        "gate_supplier_reliability": bool(gate_suppliers),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    counts = R.groupby("driver_id").agg(races=("event_id", "size"), laps=("exp_later", "sum"),
                                        own_error_events=("p_own_error", "sum"))
    dr = pd.DataFrame({"driver_id": drivers, "own_error_rate_multiplier_median": np.exp(np.median(post["driver"], 0)),
                       "q05": np.exp(np.percentile(post["driver"], 5, 0)),
                       "q95": np.exp(np.percentile(post["driver"], 95, 0))}).merge(counts, on="driver_id")
    dr.sort_values("own_error_rate_multiplier_median").to_csv(OUT / "drivers.csv", index=False)
    tm = pd.DataFrame({"team_season": tss, "mechanical_rate_multiplier_median": np.exp(np.median(post["team"], 0)),
                       "q05": np.exp(np.percentile(post["team"], 5, 0)),
                       "q95": np.exp(np.percentile(post["team"], 95, 0))})
    tm.to_csv(OUT / "team_seasons.csv", index=False)
    pd.DataFrame({"supplier_season": SUPPLIER_SEASONS,
                  "mechanical_rate_multiplier_median": np.exp(np.median(post["supplier"], 0)),
                  "q05": np.exp(np.percentile(post["supplier"], 5, 0)),
                  "q95": np.exp(np.percentile(post["supplier"], 95, 0))}).to_csv(OUT / "supplier_seasons.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    from .artifacts import record
    record(OUT, [OUT / f for f in ("summary.json", "drivers.csv", "team_seasons.csv", "supplier_seasons.csv")],
           model="reliability-v2")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
