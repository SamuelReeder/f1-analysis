"""Check the two-stage race-pace shortcut against a joint lap-level model on one season.

    python -m f1rank.racejoint --season 2024     # agreement check on one season
    python -m f1rank.racejoint --heldout         # held-out test with the joint model

docs/racing_approach.md accepts the two-stage shortcut (racepace.py stage A per race, then
racemodel.py across races) only if it agrees with a joint model of the laps themselves:
driver posteriors within a quarter of a posterior SD.

The joint model uses the same clean laps and the same terms as stage A, but in one
likelihood, with AR(1) Student-t errors within each stint instead of a bootstrap:

    y = race lap trend + race compound + race compound x (tyre age - 10)
      + race dirty-air and unpressured terms + car (team x race, free)
      + gamma * qualifying pace + u[driver] + w[driver, race]
      + (v[driver] + x[driver, race]) * (tyre age - 10) + AR(1) error

Stage B is refitted on the same season's teammate gaps, and u (race-specific pace) and v
(degradation) are compared driver by driver. Only within-team differences are identified
in either model, so drivers are compared as deviations from their season's teammates'
mean. Writes outputs/race/joint_check_<season>.json.
"""

import argparse
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

from .racemodel import RATINGS, fit as fit_stage_b, load_pairs  # noqa: E402
from .racepace import OUT, PROCESSED, REF_AGE, WET, clean_laps  # noqa: E402

TOLERANCE_SD = 0.25


def season_laps(season: int) -> pd.DataFrame:
    laps = pd.read_parquet(PROCESSED / "race_laps.parquet")
    laps = laps[laps.event_id.str[:4].astype(int) == season]
    timeline = pd.read_parquet(PROCESSED / "timeline.parquet")
    weather = pd.read_parquet(PROCESSED / "race_weather.parquet")
    rain = weather.groupby("event_id").rainfall.any()
    used = set(pd.read_csv(OUT / "stage_a_pairs.csv").event_id)
    out = []
    for event_id, L in laps.groupby("event_id"):
        if rain.get(event_id, False) or L.compound.isin(WET).any() or event_id not in used:
            continue
        out.append(clean_laps(L, timeline[timeline.event_id == event_id]).assign(event_id=event_id))
    C = pd.concat(out, ignore_index=True)
    q = pd.read_parquet(RATINGS / "driver_series.parquet").set_index(["event_id", "driver_id"]).in_team_median
    C["quali"] = [q.get(k, np.nan) for k in zip(C.event_id, C.driver_id)]
    return C.dropna(subset=["quali"]).sort_values(["event_id", "stint_key", "lap_number"], ignore_index=True)


def arrays(C: pd.DataFrame) -> tuple[dict, list[str]]:
    code = lambda s: pd.factorize(s, sort=True)  # noqa: E731
    race, races = code(C.event_id)
    drv, drivers = code(C.driver_id)
    car, _ = code(C.event_id + "|" + C.team)
    dr, _ = code(C.event_id + "|" + C.driver_id)
    rc, _ = code(C.event_id + "|" + C.compound)
    # compound offsets relative to each race's most used compound
    ref = C.groupby("event_id").compound.agg(lambda s: s.value_counts().index[0])
    not_ref = (C.compound != C.event_id.map(ref)).to_numpy()
    prev_same = ((C.stint_key == C.stint_key.shift()) & (C.lap_number == C.lap_number.shift() + 1)).to_numpy()
    lapc = (C.lap_number - C.groupby("event_id").lap_number.transform("mean")).to_numpy() / 10
    d = {"y": C.y.to_numpy(), "race": race, "drv": drv, "car": car, "dr": dr, "rc": rc, "not_ref": not_ref,
         "age": C.tyre_life.to_numpy(float) - REF_AGE, "lapc": lapc, "quali": C.quali.to_numpy(),
         "dirty": C[["close", "near", "unpressured"]].to_numpy(float), "prev": prev_same,
         "n_race": len(races), "n_drv": len(drivers), "n_car": int(car.max()) + 1, "n_dr": int(dr.max()) + 1,
         "n_rc": int(rc.max()) + 1}
    return {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in d.items()}, list(drivers)


def model(d):
    trend = numpyro.sample("trend", dist.Normal(0, 2).expand([d["n_race"]]))
    comp = numpyro.sample("comp", dist.Normal(0, 3).expand([d["n_rc"]]))
    slope = numpyro.sample("slope", dist.Normal(0, 0.3).expand([d["n_rc"]]))
    dirty = numpyro.sample("dirty", dist.Normal(0, 1).expand([d["n_race"], 3]))
    car = numpyro.sample("car", dist.Normal(0, 5).expand([d["n_car"]]))
    gamma = numpyro.sample("gamma", dist.Normal(1.0, 1.0))
    sd_u = numpyro.sample("sd_u", dist.HalfNormal(0.2))
    sd_v = numpyro.sample("sd_v", dist.HalfNormal(0.05))
    sd_w = numpyro.sample("sd_w", dist.HalfNormal(0.3))
    sd_x = numpyro.sample("sd_x", dist.HalfNormal(0.05))
    u = numpyro.deterministic("u", sd_u * numpyro.sample("u_z", dist.Normal(0, 1).expand([d["n_drv"]])))
    v = numpyro.deterministic("v", sd_v * numpyro.sample("v_z", dist.Normal(0, 1).expand([d["n_drv"]])))
    w = sd_w * numpyro.sample("w_z", dist.Normal(0, 1).expand([d["n_dr"]]))
    x = sd_x * numpyro.sample("x_z", dist.Normal(0, 1).expand([d["n_dr"]]))
    mean = (trend[d["race"]] * d["lapc"] + jnp.where(d["not_ref"], comp[d["rc"]], 0.0)
            + slope[d["rc"]] * d["age"] + (dirty[d["race"]] * d["dirty"]).sum(-1) + car[d["car"]]
            + gamma * d["quali"] + u[d["drv"]] + w[d["dr"]] + (v[d["drv"]] + x[d["dr"]]) * d["age"])
    rho = numpyro.sample("rho", dist.Uniform(-0.5, 0.95))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    r = d["y"] - mean
    r_prev = jnp.concatenate([jnp.zeros(1), r[:-1]])
    loc = jnp.where(d["prev"], rho * r_prev, 0.0)
    scale = jnp.where(d["prev"], sigma, sigma / jnp.sqrt(1 - rho ** 2))
    numpyro.factor("laps", dist.StudentT(nu, loc, scale).log_prob(r).sum())


def within_team(draws: np.ndarray, drivers: list[str], teams: dict) -> np.ndarray:
    """Each driver minus the mean of their season's team (draws x drivers)."""
    out = np.full_like(draws, np.nan)
    for team, members in teams.items():
        idx = [drivers.index(m) for m in members if m in drivers]
        if len(idx) >= 2:
            out[:, idx] = draws[:, idx] - draws[:, idx].mean(1, keepdims=True)
    return out


def fit_joint(season: int, warmup: int = 800, samples: int = 800) -> tuple[dict, list[str], pd.DataFrame]:
    C = season_laps(season)
    d, drivers = arrays(C)
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=warmup, num_samples=samples,
                num_chains=4, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), d)
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k in ("u", "v", "gamma")}, drivers, C


def heldout(first: int = 2019, n_boot: int = 2000) -> dict:
    """Race-specific pace estimated from season S-1 only (joint model, and two-stage model),
    used to predict season S's teammate pace gaps, against the qualifying link alone."""
    P = load_pairs()
    rng = np.random.default_rng(0)
    rows = []
    for S in range(first, int(P.season.max()) + 1):
        joint, drivers, _ = fit_joint(S - 1)
        prev = P[P.season == S - 1]
        two = fit_stage_b(prev, sorted(set(prev.a) | set(prev.b)))
        two_drivers = sorted(set(prev.a) | set(prev.b))
        test = P[P.season == S]
        for method, post, names in (("joint", joint, drivers), ("two_stage", {**two, "u": two["uv"][:, :, 0]},
                                                                  two_drivers)):
            u = pd.Series(post["u"].mean(0), index=names)
            g = float(post["gamma"].mean())
            t = test.assign(method=method, pred_quali=g * test.quali_gap,
                            pred_full=g * test.quali_gap + u.reindex(test.a).fillna(0).to_numpy()
                            - u.reindex(test.b).fillna(0).to_numpy(),
                            both_seen=test.a.isin(names) & test.b.isin(names))
            rows.append(t)
        print(f"held out {S}", flush=True)
    H = pd.concat(rows, ignore_index=True)
    out = {"design": "race-specific pace from season S-1 only predicts season S (pair-season means, >= 4 races, "
                     "both drivers seen in S-1)", "per_method": {}}
    for method, g in H.groupby("method"):
        ps = g.groupby(["season", "a", "b"]).agg(n=("pace_gap", "size"), obs=("pace_gap", "mean"),
                                                 base=("pred_quali", "mean"), full=("pred_full", "mean"),
                                                 seen=("both_seen", "first")).reset_index()
        ps = ps[(ps.n >= 4) & ps.seen]
        d = ((ps.obs - ps.full) ** 2 - (ps.obs - ps.base) ** 2).to_numpy()
        boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(n_boot)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out["per_method"][method] = {"n_pair_seasons": int(len(d)), "mse_diff": float(d.mean()),
                                     "mse_diff_ci95": [float(lo), float(hi)],
                                     "improves": bool(hi < 0)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "joint_heldout.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--season", type=int, default=2024)
    p.add_argument("--warmup", type=int, default=800)
    p.add_argument("--samples", type=int, default=800)
    p.add_argument("--heldout", action="store_true",
                   help="held-out test of race-specific pace, one training season at a time, joint vs two-stage")
    args = p.parse_args()
    if args.heldout:
        heldout()
        return
    C = season_laps(args.season)
    d, drivers = arrays(C)
    mcmc = MCMC(NUTS(model, target_accept_prob=0.9), num_warmup=args.warmup, num_samples=args.samples,
                num_chains=4, chain_method="parallel", progress_bar=False)
    mcmc.run(jax.random.PRNGKey(0), d, extra_fields=("diverging",))
    joint = {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k in ("u", "v", "gamma", "rho")}
    from numpyro.diagnostics import summary
    s = summary(mcmc.get_samples(group_by_chain=True))
    rhat = float(max(np.nanmax(v["r_hat"]) for v in s.values()))

    P = load_pairs()
    P = P[P.season == args.season]
    # the same driver order as the joint model (plus any driver only in the pairs)
    two = fit_stage_b(P, drivers + sorted((set(P.a) | set(P.b)) - set(drivers)))
    two["uv"] = two["uv"][:, :len(drivers)]
    # season teams: drivers who were teammates in this season's stage-A pairs
    teams = {}
    for r in P.itertuples():
        teams.setdefault(r.team, set()).update([r.a, r.b])
    teams = {k: sorted(v) for k, v in teams.items()}
    rows, agree = [], {}
    for name, j, t in (("race_specific_pace", joint["u"], two["uv"][:, :, 0]),
                       ("degradation", joint["v"], two["uv"][:, :, 1])):
        jw, tw = within_team(j, drivers, teams), within_team(t, drivers, teams)
        ok = ~np.isnan(jw[0])
        sd = np.maximum(jw[:, ok].std(0), tw[:, ok].std(0))
        z = np.abs(jw[:, ok].mean(0) - tw[:, ok].mean(0)) / sd
        agree[name] = {"n_drivers": int(ok.sum()), "share_within_quarter_sd": float(np.mean(z <= TOLERANCE_SD)),
                       "median_abs_diff_in_sd": float(np.median(z)), "max_abs_diff_in_sd": float(z.max()),
                       "corr_of_means": float(np.corrcoef(jw[:, ok].mean(0), tw[:, ok].mean(0))[0, 1])}
        for i, dname in enumerate(np.array(drivers)[ok]):
            rows.append({"quantity": name, "driver_id": dname, "joint_mean": float(jw[:, ok].mean(0)[i]),
                         "two_stage_mean": float(tw[:, ok].mean(0)[i]), "diff_in_sd": float(z[i])})
    out = {"season": args.season, "n_laps": int(len(C)), "n_races": int(C.event_id.nunique()),
           "tolerance_sd": TOLERANCE_SD, "joint_rhat_max": rhat,
           "joint_divergences": int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()),
           "gamma_joint_q05_q50_q95": [float(x) for x in np.percentile(joint["gamma"], [5, 50, 95])],
           "gamma_two_stage_q05_q50_q95": [float(x) for x in np.percentile(two["gamma"], [5, 50, 95])],
           "ar1_rho_joint": float(np.median(joint["rho"])), "agreement": agree,
           "accepted": all(a["share_within_quarter_sd"] >= 0.9 for a in agree.values()),
           "acceptance_rule": "at least 90% of drivers within a quarter of a posterior SD, for both quantities",
           "drivers": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"joint_check_{args.season}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "drivers"}, indent=1))


if __name__ == "__main__":
    main()
