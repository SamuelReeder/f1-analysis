"""Wet-weather pace: a driver x wet interaction (docs/racing_approach.md, experimental).

    python -m f1rank.wetpace

Races from 2018 (FastF1) with laps on intermediate or wet tyres. Conditions change from lap
to lap in the wet, so each lap is compared with the field on that lap:

    y = -100 log(lap time / median lap time of cars on intermediates or wets on that lap)

(percent, positive = faster). Clean wet laps: intermediate or wet tyres, green flag for the
whole lap, not lap 1, not a pit in or out lap, outside the timeline's exclusions for that
car (as in racepace.clean_laps), and |y| <= 10. Per race, a Huber regression of y on driver,
compound (wet vs intermediate) and dirty air gives each driver's wet pace; teammates' gap
has a moving-block bootstrap standard error. Drivers need MIN_LAPS clean wet laps.

Across races:

    wet gap(a, b) ~ Student-t(gamma * qualifying gap + w[a] - w[b], sqrt(se^2 + extra^2))

with the qualifying gap from stage 1 at that event (mostly dry sessions) and w a driver's
wet-specific pace (shrunk towards 0). Held-out: for each season from the third, w and gamma
fitted on earlier seasons predict that season's wet teammate gaps (each teammate race,
both drivers seen in training) with and without w. Paired difference in squared error,
bootstrap 95% interval over races; a wet ranking is published only if it is below zero.

Writes outputs/wet/: summary.json, drivers.csv, pairs.csv.
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

from .racemodel import RATINGS  # noqa: E402
from .racepace import BLOCK, PROCESSED, WET, exclusions, huber, leader_lap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "wet"
MIN_LAPS = 8
MAX_ABS_Y = 10.0
N_BOOT_A = 200
N_BOOT = 2000


def wet_laps(L: pd.DataFrame, tl: pd.DataFrame) -> pd.DataFrame:
    L = L.sort_values(["lap_number", "time"]).copy()
    L["race_lap"] = leader_lap(L)(L.time.to_numpy() - 1e-3)
    L["gap_ahead"] = L.groupby("lap_number").time.diff()
    ok = (L.compound.isin(WET) & (L.track_status.astype(str) == "1") & (L.lap_number > 1)
          & L.pit_in_time.isna() & L.pit_out_time.isna() & L.lap_time.notna() & L.driver_id.notna())
    per, wide = exclusions(tl)
    drop = L.race_lap.isin(wide).to_numpy().copy()
    for d, windows in per.items():
        m = (L.driver_id == d).to_numpy()
        for a, b in windows:
            drop |= m & (L.race_lap >= a).to_numpy() & (L.race_lap <= b).to_numpy()
    C = L[ok.to_numpy() & ~drop].copy()
    C["y"] = -100 * np.log(C.lap_time / C.groupby("lap_number").lap_time.transform("median"))
    C = C[C.y.abs() <= MAX_ABS_Y]
    C["close"] = (C.gap_ahead < 1.0).astype(float)
    C["near"] = ((C.gap_ahead >= 1.0) & (C.gap_ahead < 2.0)).astype(float)
    n = C.groupby("driver_id").size()
    return C[C.driver_id.isin(n[n >= MIN_LAPS].index)].reset_index(drop=True)


def race_pairs(C: pd.DataFrame, rng) -> list[dict]:
    drivers = sorted(C.driver_id.unique())
    D = (C.driver_id.to_numpy()[:, None] == np.array(drivers)[None, :]).astype(float)
    X = np.hstack([D, (C.compound == "WET").to_numpy(float)[:, None], C[["close", "near"]].to_numpy(float)])
    y = C.y.to_numpy()
    beta = huber(X, y)
    resid = y - X @ beta
    keys = (C.driver_id + "|" + C.stint.astype(str)).to_numpy()
    idx = pd.Series(np.arange(len(C))).groupby(keys).indices
    draws = np.zeros((N_BOOT_A, len(drivers)))
    for b in range(N_BOOT_A):
        e = np.empty_like(resid)
        for rows in idx.values():
            rows = np.sort(rows)
            m = len(rows)
            starts = rng.integers(0, max(1, m - BLOCK + 1), size=int(np.ceil(m / BLOCK)))
            take = np.concatenate([rows[s:s + BLOCK] for s in starts])[:m]
            e[rows] = resid[take] if len(take) == m else resid[rows]
        draws[b] = huber(X, X @ beta + e)[:len(drivers)]
    pos = {d: i for i, d in enumerate(drivers)}
    team = C.groupby("driver_id").team.first()
    out = []
    for tm, g in team.groupby(team):
        if len(g) != 2:
            continue
        a, b = sorted(g.index)
        out.append({"team": tm, "a": a, "b": b, "wet_gap": float(beta[pos[a]] - beta[pos[b]]),
                    "se": float((draws[:, pos[a]] - draws[:, pos[b]]).std()),
                    "laps_a": int((C.driver_id == a).sum()), "laps_b": int((C.driver_id == b).sum())})
    return out


def pairs() -> pd.DataFrame:
    from .qualifying import features
    laps = pd.read_parquet(PROCESSED / "race_laps.parquet")
    timeline = pd.read_parquet(PROCESSED / "timeline.parquet")
    q = features()[0].set_index(["event_id", "driver_id"]).driver
    rng = np.random.default_rng(0)
    rows = []
    for event_id, L in laps.groupby("event_id"):
        if not L.compound.isin(WET).any():
            continue
        C = wet_laps(L, timeline[timeline.event_id == event_id])
        if C.driver_id.nunique() < 6:
            continue
        for r in race_pairs(C, rng):
            rows.append({"event_id": event_id, **r,
                         "quali_gap": q.get((event_id, r["a"]), np.nan) - q.get((event_id, r["b"]), np.nan)})
        print(event_id, len(C), "wet laps", flush=True)
    P = pd.DataFrame(rows).dropna(subset=["quali_gap"])
    P["season"] = P.event_id.str[:4].astype(int)
    return P.reset_index(drop=True)


def model(d, wet_effects=True):
    gamma = numpyro.sample("gamma", dist.Normal(1.0, 1.0))
    mean = gamma * d["dq"]
    if wet_effects:
        sd_w = numpyro.sample("sd_w", dist.HalfNormal(0.5))
        w = numpyro.deterministic("w", sd_w * numpyro.sample("w_z", dist.Normal(0, 1).expand([d["n"]])))
        mean = mean + w[d["a"]] - w[d["b"]]
    extra = numpyro.sample("extra", dist.HalfNormal(1.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    numpyro.sample("y", dist.StudentT(nu, mean, jnp.sqrt(d["se"] ** 2 + extra ** 2)), obs=d["y"])


def fit(P, drivers, warmup=800, samples=800, **kw) -> dict:
    from functools import partial
    from .artifacts import diagnostics, fit_until_converged
    idx = {x: i for i, x in enumerate(drivers)}
    d = {"n": len(drivers), "a": jnp.asarray(P.a.map(idx).to_numpy()), "b": jnp.asarray(P.b.map(idx).to_numpy()),
         "dq": jnp.asarray(P.quali_gap.to_numpy()), "se": jnp.asarray(P.se.to_numpy()),
         "y": jnp.asarray(P.wet_gap.to_numpy())}
    def run(w, s, accept, seed):
        mcmc = MCMC(NUTS(partial(model, **kw), target_accept_prob=accept or 0.9), num_warmup=w, num_samples=s,
                    num_chains=4, chain_method="parallel", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed), d, extra_fields=("diverging",))
        return mcmc, diagnostics(mcmc.get_samples(group_by_chain=True),
                                 int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))

    mcmc = fit_until_converged(run, warmup, samples)
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    post["_divergences"] = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
    return post


def heldout(P: pd.DataFrame, rng) -> dict:
    from .qualifying import features, unconverged_folds
    rows = []
    seasons = sorted(P.season.unique())[2:]
    excluded = [int(S) for S in seasons if S in unconverged_folds()]
    for S in seasons:
        if S in excluded:
            print(f"held out {S}: left out, its qualifying fold did not converge", flush=True)
            continue
        q = features(int(S))[0].set_index(["event_id", "driver_id"]).driver
        fold = P[P.season <= S].copy()
        fold["quali_gap"] = [q.get((e, a), np.nan) - q.get((e, b), np.nan)
                             for e, a, b in zip(fold.event_id, fold.a, fold.b)]
        if fold.quali_gap.isna().any():
            raise ValueError(f"Qualifying fold {S} does not cover the wet races")
        train, test = fold[fold.season < S], fold[fold.season == S]
        drivers = sorted(set(train.a) | set(train.b))
        full = fit(train, drivers)
        base = fit(train, drivers, wet_effects=False)
        w = pd.Series(full["w"].mean(0), index=drivers)
        rows.append(test.assign(pred_full=full["gamma"].mean() * test.quali_gap + w.reindex(test.a).to_numpy()
                                - w.reindex(test.b).to_numpy(),
                                pred_base=base["gamma"].mean() * test.quali_gap,
                                seen=test.a.isin(drivers) & test.b.isin(drivers)))
        print(f"held out {S}", flush=True)
    H = pd.concat(rows)
    H = H[H.seen]
    H["d"] = (H.wet_gap - H.pred_full) ** 2 - (H.wet_gap - H.pred_base) ** 2
    d = H.groupby("event_id").d.sum().to_numpy()
    boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_races": int(len(d)), "n_teammate_races": int(len(H)), "sq_err_diff_per_race": float(d.mean()),
            "ci95": [float(lo), float(hi)], "excluded_unconverged_qualifying_folds": excluded}


def main() -> None:
    rng = np.random.default_rng(0)
    P = pairs()
    drivers = sorted(set(P.a) | set(P.b))
    post = fit(P, drivers, warmup=1000, samples=1000)
    test = heldout(P, rng)
    q = lambda x: [round(float(v), 4) for v in np.percentile(x, [5, 50, 95])]  # noqa: E731
    summary = {"wet_teammate_races": int(len(P)), "wet_races": int(P.event_id.nunique()),
               "seasons": sorted(int(s) for s in P.season.unique()), "drivers": len(drivers),
               "gamma_q05_q50_q95": q(post["gamma"]), "sd_wet_specific_q05_q50_q95": q(post["sd_w"]),
               "extra_sd_q05_q50_q95": q(post["extra"]), "divergences": post["_divergences"],
               "units": "percent of lap time", "heldout_wet_effects_vs_quali_link": test,
               "gate_wet_ranking": bool(test["ci95"][1] < 0)}
    OUT.mkdir(parents=True, exist_ok=True)
    races = pd.concat([P.a, P.b]).value_counts()
    pd.DataFrame({"driver_id": drivers, "wet_specific_pct_median": np.median(post["w"], 0),
                  "q05": np.percentile(post["w"], 5, 0), "q95": np.percentile(post["w"], 95, 0),
                  "wet_teammate_races": races.reindex(drivers).to_numpy()}
                 ).sort_values("wet_specific_pct_median", ascending=False).to_csv(OUT / "drivers.csv", index=False)
    P.to_csv(OUT / "pairs.csv", index=False)
    from .artifacts import fit_record
    summary["fit_attempts"] = fit_record()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    from .artifacts import input_files, record
    from .qualifying import POLICY, dependencies
    record(OUT, [OUT / f for f in ("summary.json", "drivers.csv", "pairs.csv")], model="wetpace-v2",
           inputs=input_files() + dependencies([None, *map(int, sorted(P.season.unique())[2:])]),
           details={"qualifying_policy": POLICY})
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
