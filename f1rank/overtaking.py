"""Overtaking and defending: pass probability within battle episodes (see battles.py).

Per episode lap (did the attacker pass on the next lap?):

    logit p = base + era (2026 overtake mode vs DRS) + circuit (shrunk)
            + b_pace * pace diff + b_age * tyre-age diff / 10 + b_soft * softer compound
            + b_speed * straight-line speed diff / 10 + b_gap * gap
            + attacking car[team-season of A] - defending car[team-season of D] (shrunk)
            + attacker[A] - defender[D] (shrunk) + episode effect (repeated laps, shrunk)

The car terms stay in every comparison, so the driver test removes only driver terms.

Feasibility (run first, criteria fixed before the results): on the real episode structure
and covariates, outcomes are simulated with the covariate, circuit and episode effects
estimated from the real data and attacker/defender effects of the size the real data
suggest (and half and double that). Feasible if, at the estimated size, attacker and
defender effects are recovered with correlation >= 0.7 and 90% intervals covering 85-97%
of true values (drivers with at least 40 episodes on that side), averaged over 3 truths.

If feasible, a held-out test (seasons from 2020, each from the seasons before it; paired
log predictive density per race, bootstrap 95% interval) decides whether attacker and
defender effects are rated: the interval must be above zero, and the same test on episodes
involving a driver in a new team (transfer) must not be entirely below zero. Otherwise
overtaking is reported only as opportunity counts.

Writes outputs/battles/: feasibility.json, summary.json, drivers.csv.
"""

import json
import os
from functools import partial

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro.infer import MCMC, NUTS, Predictive  # noqa: E402

from .battles import OUT, PROCESSED  # noqa: E402

N_BOOT = 2000
FIRST_TEST = 2020
MIN_EPISODES = 40
X_COLS = ["pace_diff", "tyre_age_diff", "softer", "speed_diff", "gap"]


def load() -> pd.DataFrame:
    E = pd.read_parquet(OUT / "episodes.parquet")
    ev = pd.read_parquet(PROCESSED / "events.parquet")[["event_id", "circuit_id"]]
    E = E.merge(ev, on="event_id")
    team = pd.read_parquet(PROCESSED / "race.parquet").set_index(["event_id", "driver_id"]).team
    for side, col in (("att", "attacker"), ("def", "defender")):
        E[f"{side}_team_season"] = [f"{team.get((e, d), '?')}|{e[:4]}" for e, d in zip(E.event_id, E[col])]
    E["tyre_age_diff"] = E.tyre_age_diff / 10
    E["speed_diff"] = E.speed_diff / 10
    for c in X_COLS:
        E[c] = E[c].fillna(0.0)  # missing pace (driver without a stage-A estimate) or speed: no difference
    return E.reset_index(drop=True)


def data(E: pd.DataFrame, lev: dict, y=None) -> dict:
    idx = lambda col, key: jnp.asarray(E[col].map({x: i for i, x in enumerate(lev[key])})  # noqa: E731
                                       .fillna(-1).astype(int).to_numpy())
    return {"x": jnp.asarray(E[X_COLS].to_numpy(float)), "era": jnp.asarray((E.season >= 2026).to_numpy(float)),
            "circuit": idx("circuit_id", "circuits"), "att": idx("attacker", "drivers"),
            "att_ts": idx("att_team_season", "team_seasons"), "def_ts": idx("def_team_season", "team_seasons"),
            "dfn": idx("defender", "drivers"), "ep": idx("episode", "episodes"),
            "y": jnp.asarray(E.passed.to_numpy() if y is None else y, dtype=float),
            **{f"n_{k}": len(v) for k, v in lev.items()}}


def levels(E: pd.DataFrame) -> dict:
    return {"circuits": sorted(E.circuit_id.unique()), "drivers": sorted(set(E.attacker) | set(E.defender)),
            "episodes": sorted(E.episode.unique()),
            "team_seasons": sorted(set(E.att_team_season) | set(E.def_team_season))}


def take(v, i):
    return jnp.where(i >= 0, v[jnp.maximum(i, 0)], 0.0)


def model(d, driver_effects=True):
    base = numpyro.sample("base", dist.Normal(-1.5, 2.0))
    era = numpyro.sample("era", dist.Normal(0, 1.0))
    beta = numpyro.sample("beta", dist.Normal(0, 2.0).expand([len(X_COLS)]))
    sd_c = numpyro.sample("sd_circuit", dist.HalfNormal(1.0))
    circ = sd_c * numpyro.sample("circuit_z", dist.Normal(0, 1).expand([d["n_circuits"]]))
    sd_e = numpyro.sample("sd_episode", dist.HalfNormal(1.0))
    epi = sd_e * numpyro.sample("episode_z", dist.Normal(0, 1).expand([d["n_episodes"]]))
    sd_ca = numpyro.sample("sd_car_attack", dist.HalfNormal(0.5))
    sd_cd = numpyro.sample("sd_car_defend", dist.HalfNormal(0.5))
    ca = sd_ca * numpyro.sample("car_attack_z", dist.Normal(0, 1).expand([d["n_team_seasons"]]))
    cd = sd_cd * numpyro.sample("car_defend_z", dist.Normal(0, 1).expand([d["n_team_seasons"]]))
    eta = (base + era * d["era"] + d["x"] @ beta + take(circ, d["circuit"]) + take(epi, d["ep"])
           + take(ca, d["att_ts"]) - take(cd, d["def_ts"]))
    if driver_effects:
        sd_a = numpyro.sample("sd_attack", dist.HalfNormal(0.5))
        sd_d = numpyro.sample("sd_defend", dist.HalfNormal(0.5))
        a = numpyro.deterministic("attack", sd_a * numpyro.sample("attack_z", dist.Normal(0, 1).expand([d["n_drivers"]])))
        b = numpyro.deterministic("defend", sd_d * numpyro.sample("defend_z", dist.Normal(0, 1).expand([d["n_drivers"]])))
        eta = eta + take(a, d["att"]) - take(b, d["dfn"])
    lp = dist.Bernoulli(logits=eta).log_prob(d["y"])
    numpyro.factor("ll", lp.sum())
    numpyro.deterministic("lp", lp)
    numpyro.deterministic("eta", eta)


def fit(E, lev, y=None, warmup=500, samples=500, seed=0, **kw) -> dict:
    mcmc = MCMC(NUTS(partial(model, **kw), target_accept_prob=0.85), num_warmup=warmup, num_samples=samples,
                num_chains=4, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(seed), data(E, lev, y), extra_fields=("diverging",))
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k not in ("lp", "eta")}
    post["_divergences"] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return post


def feasibility() -> dict:
    E = load()
    lev = levels(E)
    real = fit(E, lev)
    size = {"attack": float(np.median(real["sd_attack"])), "defend": float(np.median(real["sd_defend"]))}
    # fixed parts of the simulation: the real fit's posterior medians
    base_eta = (np.median(real["base"]) + np.median(real["era"]) * (E.season >= 2026).to_numpy()
                + E[X_COLS].to_numpy() @ np.median(real["beta"], 0))
    circ = np.median(real["circuit_z"], 0) * np.median(real["sd_circuit"])
    base_eta += E.circuit_id.map(dict(zip(lev["circuits"], circ))).to_numpy()
    for side, sign in (("attack", 1), ("defend", -1)):
        car = np.median(real[f"car_{side}_z"], 0) * np.median(real[f"sd_car_{side}"])
        col = "att_team_season" if side == "attack" else "def_team_season"
        base_eta += sign * E[col].map(dict(zip(lev["team_seasons"], car))).to_numpy()
    sd_ep = float(np.median(real["sd_episode"]))
    n_att = E.groupby("attacker").episode.nunique().reindex(lev["drivers"]).fillna(0).to_numpy()
    n_def = E.groupby("defender").episode.nunique().reindex(lev["drivers"]).fillna(0).to_numpy()
    di = {x: i for i, x in enumerate(lev["drivers"])}
    ai, bi = E.attacker.map(di).to_numpy(), E.defender.map(di).to_numpy()
    ei = E.episode.map({x: i for i, x in enumerate(lev["episodes"])}).to_numpy()
    results = []
    for scale in (0.5, 1.0, 2.0):
        for seed in (1, 2, 3):
            rng = np.random.default_rng([seed, int(scale * 10)])
            a = rng.normal(0, scale * size["attack"], len(lev["drivers"]))
            b = rng.normal(0, scale * size["defend"], len(lev["drivers"]))
            ep = rng.normal(0, sd_ep, len(lev["episodes"]))
            eta = base_eta + a[ai] - b[bi] + ep[ei]
            y = rng.random(len(E)) < 1 / (1 + np.exp(-eta))
            post = fit(E, lev, y=y, seed=seed)
            res = {"scale": scale, "seed": seed}
            for name, true, n in (("attack", a, n_att), ("defend", b, n_def)):
                m = n >= MIN_EPISODES
                draws = post[name][:, m]
                lo, hi = np.percentile(draws, [5, 95], axis=0)
                res[name] = {"n_drivers": int(m.sum()), "corr": float(np.corrcoef(draws.mean(0), true[m])[0, 1]),
                             "cov90": float(np.mean((true[m] >= lo) & (true[m] <= hi)))}
            results.append(res)
            print(res, flush=True)
    at = [r for r in results if r["scale"] == 1.0]
    mean = lambda side, k: float(np.mean([r[side][k] for r in at]))  # noqa: E731
    feasible = all(mean(s, "corr") >= 0.7 and 0.85 <= mean(s, "cov90") <= 0.97 for s in ("attack", "defend"))
    out = {"estimated_sd_logit": size, "sd_episode": sd_ep, "min_episodes": MIN_EPISODES,
           "criteria": "at the estimated size, mean over 3 truths: corr >= 0.7 and 90% coverage 85-97%, "
                       "for both attacker and defender effects",
           "at_estimated_size": {s: {k: mean(s, k) for k in ("corr", "cov90")} for s in ("attack", "defend")},
           "runs": results, "feasible": bool(feasible)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "feasibility.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "runs"}, indent=1))
    return out


def log_pred(post, E, lev, **kw) -> np.ndarray:
    samples = {k: v for k, v in post.items() if not k.startswith("_")}
    lp = np.asarray(Predictive(partial(model, **kw), posterior_samples=samples, return_sites=["lp"])(
        jax.random.PRNGKey(1), data(E, lev))["lp"])
    m = lp.max(0)
    return m + np.log(np.exp(lp - m).mean(0))


def fit_and_test() -> dict:
    feas = json.loads((OUT / "feasibility.json").read_text())
    E = load()
    lev = levels(E)
    post = fit(E, lev, warmup=800, samples=800)
    q = lambda x: [round(float(v), 3) for v in np.percentile(x, [5, 50, 95])]  # noqa: E731
    ep = E.groupby("episode").agg(att=("attacker", "first"), dfn=("defender", "first"), passed=("passed", "max"))
    counts = pd.DataFrame({"episodes_attacking": ep.groupby("att").size(), "passes_made": ep.groupby("att").passed.sum(),
                           "episodes_defending": ep.groupby("dfn").size(),
                           "passes_suffered": ep.groupby("dfn").passed.sum()}).fillna(0)
    summary = {"episode_laps": int(len(E)), "episodes": int(E.episode.nunique()), "passes": int(E.passed.sum()),
               "feasible": feas["feasible"],
               "beta_q05_q50_q95": {c: q(post["beta"][:, i]) for i, c in enumerate(X_COLS)},
               "sd_attack_q05_q50_q95": q(post["sd_attack"]), "sd_defend_q05_q50_q95": q(post["sd_defend"]),
               "sd_car_attack_q05_q50_q95": q(post["sd_car_attack"]), "sd_car_defend_q05_q50_q95": q(post["sd_car_defend"]),
               "era_2026_q05_q50_q95": q(post["era"]), "divergences": post["_divergences"]}
    if feas["feasible"]:
        rng = np.random.default_rng(0)
        rows = []
        for season in range(FIRST_TEST, E.season.max() + 1):
            train, test = E[E.season < season], E[E.season == season]
            lv = levels(E)
            lv["team_seasons"] = sorted(set(train.att_team_season) | set(train.def_team_season))
            w = log_pred(fit(train, lv), test, lv)
            wo = log_pred(fit(train, lv, driver_effects=False), test, lv, driver_effects=False)
            # transfer: an attacker or defender in a different team from their previous season
            last = pd.concat([train[["event_id", "attacker", "att_team_season"]].set_axis(["e", "d", "ts"], axis=1),
                              train[["event_id", "defender", "def_team_season"]].set_axis(["e", "d", "ts"], axis=1)])
            last_team = last.sort_values("e").groupby("d").ts.last().str.split("|").str[0]
            moved = np.zeros(len(test), bool)
            for col, ts in (("attacker", "att_team_season"), ("defender", "def_team_season")):
                prev = test[col].map(last_team)
                moved |= (prev.notna() & (prev != test[ts].str.split("|").str[0])).to_numpy()
            rows.append(pd.DataFrame({"event_id": test.event_id, "diff": w - wo, "moved": moved}))
            print(f"held out {season}", flush=True)
        H = pd.concat(rows)

        def paired(sub):
            d = sub.groupby("event_id")["diff"].sum().to_numpy()
            boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
            lo, hi = np.percentile(boot, [2.5, 97.5])
            return {"n_races": int(len(d)), "n_episode_laps": int(len(sub)), "mean_diff_per_race": float(d.mean()),
                    "ci95": [float(lo), float(hi)]}

        summary["heldout_driver_effects"] = {**paired(H), "transfer": paired(H[H.moved])}
        t = summary["heldout_driver_effects"]
        summary["gate_driver_ranking"] = bool(t["ci95"][0] > 0 and t["transfer"]["ci95"][1] >= 0)
    else:
        summary["gate_driver_ranking"] = False
    drivers = lev["drivers"]
    dr = pd.DataFrame({"driver_id": drivers, "attack_median": np.median(post["attack"], 0),
                       "attack_q05": np.percentile(post["attack"], 5, 0), "attack_q95": np.percentile(post["attack"], 95, 0),
                       "defend_median": np.median(post["defend"], 0),
                       "defend_q05": np.percentile(post["defend"], 5, 0), "defend_q95": np.percentile(post["defend"], 95, 0)})
    dr = dr.merge(counts, left_on="driver_id", right_index=True, how="left")
    dr.to_csv(OUT / "drivers.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return summary
