"""Race pace and degradation from one joint lap-level model across seasons (stage 2, step 3).

    python -m f1rank.racemulti --heldout     # leave-future-out test of the driver terms
    python -m f1rank.racemulti               # fit on every season; summary and drivers
    python -m f1rank.racemulti --only 2026 2024   # pre-fit these held-out seasons (parallel workers)
    (--fastf1-only: 2018 onward only, without the Jolpica laps; outputs get a _fastf1only suffix)
    --telemetry-races: FastF1 races with aligned telemetry only (suffix _fastf1only_telemetry)
    --coast: the same races, with the coasting covariate (suffix _fastf1only_coast)

Coasting (docs/racing_approach.md, telemetry use 1). With --coast each lap gets
b_coast x (its lift-and-coast seconds minus the race median; extract/telemetry.py), so the
driver terms describe pace at typical coasting, and the held-out targets are adjusted the
same way: each teammate race gap minus b_coast (from the training fit) x the teammates'
difference in mean coasting over their clean laps. It is compared with --telemetry-races
(same races, no covariate). Races whose telemetry alignment misses FastF1's finish-line
speed trap by more than TRAP_MAX_KMH (median) are not used by either.

The one-season joint model (racejoint.py) showed the two-stage shortcut is not accurate
enough for published numbers, and could only test race-specific pace learned from a single
season. This model pools every dry race since 2018 in one likelihood, so a driver's
race-specific pace can build up across seasons:

    y = race lap trend + race compound + race compound x (tyre age - 10)
      + race dirty-air and unpressured terms + car (team x race, free)
      + gamma * qualifying pace
      + u[driver] + e[driver, season] + w[driver, race]                 (race-specific pace)
      + (v[driver] + f[driver, season] + x[driver, race]) * (tyre age - 10)   (degradation)
      + AR(1) Student-t error within each stint

Laps before 2018 come from Jolpica (racepace.jolpica_races, the races its stage A used),
through the weaker observation model the approach specifies: compounds are unknown, so the
compound offset is replaced by a stint offset (per car and stint, shrunk, sd estimated) and
the degradation slope is common per race; these laps have their own noise scale
(sigma x k_old). The stint offsets exist only when such laps are in the fit, and k_old only
when FastF1 laps are too (it is defined relative to them: with Jolpica laps alone, as in
the held-out fits for 2012-2018, sigma and k_old are not separately identified).

y is percent of the race's median clean lap (positive = faster), from racepace.clean_laps;
the qualifying pace is stage 1's pace in the current car at that event (percent). Only
within-team differences identify the driver terms (the car term is free per team and race).

Held-out test: for each season S from 2012 (2020 with --fastf1-only), fit on every season
before S and predict
season S's teammate gaps (stage A's per-race gaps, averaged per pair and season, both
drivers seen in training) with gamma x qualifying gap alone, and with the driver terms
u (pace) and v (degradation) added. Improvement: paired difference in squared error, with a
bootstrap 95% interval over pair-seasons that must lie below zero.

Writes outputs/race/: multi_heldout.json and multi_heldout_effects.npz (per held-out season,
the driver-term draws fitted on the seasons before it, for the overall rating's entry test in
championship.py); multi_summary.json and multi_drivers.csv (driver
terms; a standalone ranking only where the held-out gate passes).
"""

import argparse
import json
import os
import time

os.environ.setdefault("XLA_FLAGS", "--xla_force_host_platform_device_count=4")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
from numpyro.infer import MCMC, NUTS  # noqa: E402

from .racemodel import RATINGS, load_pairs  # noqa: E402
from .racepace import OLD_COMPOUNDS, OUT, PROCESSED, REF_AGE, WET, clean_laps, jolpica_races  # noqa: E402

FIRST_SEASON, FIRST_TEST = 2018, 2020
FIRST_TEST_OLD = 2012
TRAP_MAX_KMH = 5.0
N_BOOT = 2000
# the lap-level fits are large: they run on a GPU when JAX has one (.venv-gpu), with the
# chains vectorised on the one device; on CPU, one chain per core
CHAINS = "vectorized" if jax.default_backend() == "gpu" else "parallel"


def all_laps(old: bool = True, telemetry: bool = False) -> pd.DataFrame:
    """Clean dry laps of every race stage A used, with the qualifying pace at that event."""
    laps = pd.read_parquet(PROCESSED / "race_laps.parquet")
    timeline = pd.read_parquet(PROCESSED / "timeline.parquet")
    weather = pd.read_parquet(PROCESSED / "race_weather.parquet")
    rain = weather.groupby("event_id").rainfall.any()
    used = set(pd.read_csv(OUT / "stage_a_pairs.csv").event_id)
    out = []
    for event_id, L in laps.groupby("event_id"):
        if rain.get(event_id, False) or L.compound.isin(WET).any() or event_id not in used:
            continue
        out.append(clean_laps(L, timeline[timeline.event_id == event_id]).assign(event_id=event_id, source="fastf1"))
    old_pairs = OUT / "stage_a_pairs_old.csv"
    if old and old_pairs.exists():
        used_old = set(pd.read_csv(old_pairs).event_id)
        for event_id, L, wet in jolpica_races():
            if event_id in used_old and not wet:
                out.append(clean_laps(L, timeline[timeline.event_id == event_id], OLD_COMPOUNDS)
                           .assign(event_id=event_id, source="jolpica"))
    C = pd.concat(out, ignore_index=True)
    q = pd.read_parquet(RATINGS / "driver_series.parquet").set_index(["event_id", "driver_id"]).in_team_median
    C["quali"] = [q.get(k, np.nan) for k in zip(C.event_id, C.driver_id)]
    C["season"] = C.event_id.str[:4].astype(int)
    if telemetry:
        T = pd.read_parquet(PROCESSED / "race_telemetry.parquet")
        T = T[T.trap_diff_kmh <= TRAP_MAX_KMH]
        C = C.merge(T[["event_id", "driver_number", "lap_number", "coast_s"]],
                    on=["event_id", "driver_number", "lap_number"], how="inner")
        C["coast"] = C.coast_s - C.groupby("event_id").coast_s.transform("median")
    return C.dropna(subset=["quali"]).sort_values(["event_id", "stint_key", "lap_number"], ignore_index=True)


def arrays(C: pd.DataFrame) -> tuple[dict, list[str]]:
    code = lambda s: pd.factorize(s, sort=True)  # noqa: E731
    race, races = code(C.event_id)
    drv, drivers = code(C.driver_id)
    car, _ = code(C.event_id + "|" + C.team)
    ds, _ = code(C.season.astype(str) + "|" + C.driver_id)
    dr, _ = code(C.event_id + "|" + C.driver_id)
    rc, _ = code(C.event_id + "|" + C.compound)
    ref = C.groupby("event_id").compound.agg(lambda s: s.value_counts().index[0])
    not_ref = (C.compound != C.event_id.map(ref)).to_numpy()
    prev_same = ((C.stint_key == C.stint_key.shift()) & (C.event_id == C.event_id.shift())
                 & (C.lap_number == C.lap_number.shift() + 1)).to_numpy()
    lapc = (C.lap_number - C.groupby("event_id").lap_number.transform("mean")).to_numpy() / 10
    old = (C.source == "jolpica").to_numpy()
    st, _ = code((C.event_id + "|" + C.stint_key).where(old))  # NaN (FastF1 laps) -> -1
    d = {"y": C.y.to_numpy(), "race": race, "drv": drv, "car": car, "ds": ds, "dr": dr, "rc": rc,
         "old": old, "st": st, "n_st": int(st.max()) + 1, "mixed": bool(old.any() and not old.all()),
         "coast": C.coast.to_numpy() if "coast" in C and COAST else np.zeros(0),
         "not_ref": not_ref, "age": C.tyre_life.to_numpy(float) - REF_AGE, "lapc": lapc,
         "quali": C.quali.to_numpy(), "dirty": C[["close", "near", "unpressured"]].to_numpy(float),
         "prev": prev_same, "n_race": len(races), "n_drv": len(drivers), "n_car": int(car.max()) + 1,
         "n_ds": int(ds.max()) + 1, "n_dr": int(dr.max()) + 1, "n_rc": int(rc.max()) + 1}
    return {k: (jnp.asarray(v) if isinstance(v, np.ndarray) else v) for k, v in d.items()}, list(drivers)


def model(d):
    trend = numpyro.sample("trend", dist.Normal(0, 2).expand([d["n_race"]]))
    comp = numpyro.sample("comp", dist.Normal(0, 3).expand([d["n_rc"]]))
    slope = numpyro.sample("slope", dist.Normal(0, 0.3).expand([d["n_rc"]]))
    dirty = numpyro.sample("dirty", dist.Normal(0, 1).expand([d["n_race"], 3]))
    car = numpyro.sample("car", dist.Normal(0, 5).expand([d["n_car"]]))
    gamma = numpyro.sample("gamma", dist.Normal(1.0, 1.0))
    sd = {k: numpyro.sample(f"sd_{k}", dist.HalfNormal(s))
          for k, s in (("u", 0.2), ("e", 0.2), ("w", 0.3), ("v", 0.05), ("f", 0.05), ("x", 0.05))}
    z = lambda name, n: numpyro.sample(f"{name}_z", dist.Normal(0, 1).expand([n]))  # noqa: E731
    u = numpyro.deterministic("u", sd["u"] * z("u", d["n_drv"]))
    v = numpyro.deterministic("v", sd["v"] * z("v", d["n_drv"]))
    e = sd["e"] * z("e", d["n_ds"])
    f = sd["f"] * z("f", d["n_ds"])
    w = sd["w"] * z("w", d["n_dr"])
    x = sd["x"] * z("x", d["n_dr"])
    mean = (trend[d["race"]] * d["lapc"] + jnp.where(d["not_ref"], comp[d["rc"]], 0.0)
            + slope[d["rc"]] * d["age"] + (dirty[d["race"]] * d["dirty"]).sum(-1) + car[d["car"]]
            + gamma * d["quali"] + u[d["drv"]] + e[d["ds"]] + w[d["dr"]]
            + (v[d["drv"]] + f[d["ds"]] + x[d["dr"]]) * d["age"])
    rho = numpyro.sample("rho", dist.Uniform(-0.5, 0.95))
    sigma = numpyro.sample("sigma", dist.HalfNormal(1.0))
    nu = numpyro.sample("nu", dist.Gamma(2.0, 0.1))
    if d["coast"].shape[0]:  # lift-and-coast seconds (--coast)
        mean = mean + numpyro.sample("b_coast", dist.Normal(0.0, 2.0)) * d["coast"]
    if d["n_st"] > 0:  # laps without compounds (before 2018): stint offsets
        sd_st = numpyro.sample("sd_stint", dist.HalfNormal(1.0))
        st = sd_st * numpyro.sample("stint_z", dist.Normal(0, 1).expand([d["n_st"]]))
        mean = mean + jnp.where(d["st"] >= 0, st[jnp.maximum(d["st"], 0)], 0.0)
    if d["mixed"]:  # their own noise scale, relative to FastF1 laps (not identified without them)
        k_old = numpyro.sample("k_old", dist.LogNormal(0.0, 0.5))
        sigma = sigma * jnp.where(d["old"], k_old, 1.0)
    r = d["y"] - mean
    r_prev = jnp.concatenate([jnp.zeros(1), r[:-1]])
    loc = jnp.where(d["prev"], rho * r_prev, 0.0)
    scale = jnp.where(d["prev"], sigma, sigma / jnp.sqrt(1 - rho ** 2))
    numpyro.factor("laps", dist.StudentT(nu, loc, scale).log_prob(r).sum())


KEEP = ("u", "v", "gamma", "rho", "sigma", "nu", "sd_u", "sd_e", "sd_w", "sd_v", "sd_f", "sd_x", "sd_stint", "k_old",
        "b_coast")
COAST = False  # set by --coast


RHAT_MAX = 1.05
MODEL_VERSION = "2"  # change with the model: saved held-out fits of another version are not reused


def fit(C: pd.DataFrame, warmup: int = 500, samples: int = 500, tries: int = 3) -> tuple[dict, list[str]]:
    """NUTS fit, started at the prior medians (random starts can put the scales far in their
    tails, where a chain can stick). If any kept parameter has R-hat above RHAT_MAX, the fit
    is repeated with another seed and a longer warmup; every attempt is recorded."""
    from numpyro.diagnostics import summary
    from numpyro.infer import init_to_median
    d, drivers = arrays(C)
    attempts = []
    for attempt in range(tries):
        t0 = time.time()
        mcmc = MCMC(NUTS(model, target_accept_prob=0.9, init_strategy=init_to_median(num_samples=15)),
                    num_warmup=warmup * (1 + attempt), num_samples=samples,
                    num_chains=4, chain_method=CHAINS, progress_bar=False)
        mcmc.run(jax.random.PRNGKey(attempt), d, extra_fields=("diverging",))
        grouped = mcmc.get_samples(group_by_chain=True)
        s = summary({k: v for k, v in grouped.items() if k in KEEP})
        rhat = float(max(np.nanmax(v["r_hat"]) for v in s.values()))
        div = int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum())
        attempts.append({"seed": attempt, "warmup": warmup * (1 + attempt), "rhat_max": rhat, "divergences": div,
                         "minutes": round((time.time() - t0) / 60, 1)})
        print(f"  fit attempt {attempt}: {attempts[-1]}", flush=True)
        if rhat <= RHAT_MAX:
            break
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k in KEEP}
    post["_rhat_max"], post["_divergences"] = rhat, div
    post["_minutes"] = sum(a["minutes"] for a in attempts)
    post["_backend"] = jax.default_backend()
    post["_attempts"] = attempts
    post["_converged"] = rhat <= RHAT_MAX
    return post, drivers


def checkpointed(C: pd.DataFrame, path) -> tuple[dict, list[str]]:
    """fit(C), saved to path so an interrupted held-out run resumes where it stopped. A saved
    fit is reused only if its training laps (sha256 of the design columns) and MODEL_VERSION
    are identical."""
    import hashlib
    key = hashlib.sha256(MODEL_VERSION.encode() + pd.util.hash_pandas_object(
        C[["event_id", "driver_id", "lap_number", "stint", "y", "quali"] + (["coast"] if COAST else [])], index=False
    ).to_numpy().tobytes()).hexdigest()
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if str(z["key"]) == key:
            meta = json.loads(str(z["meta"]))
            print(f"  reusing {path.name}", flush=True)
            return {**{k: z[k] for k in z.files if k not in ("key", "meta", "drivers")}, **meta}, [str(x) for x in z["drivers"]]
    post, drivers = fit(C)
    if not post["_converged"]:
        return post, drivers
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = {k: post[k] for k in post if k.startswith("_")}
    np.savez_compressed(path, key=key, meta=json.dumps(meta), drivers=np.array(drivers),
                        **{k: v for k, v in post.items() if not k.startswith("_")})
    return post, drivers


def target_pairs() -> pd.DataFrame:
    """Stage-A teammate gaps (FastF1, plus Jolpica before 2018) with the qualifying gap."""
    P = load_pairs().assign(source="fastf1")
    old = OUT / "stage_a_pairs_old.csv"
    if old.exists():
        O = pd.read_csv(old)
        q = pd.read_parquet(RATINGS / "driver_series.parquet").set_index(["event_id", "driver_id"]).in_team_median
        O["quali_gap"] = [q.get((e, a), np.nan) - q.get((e, b), np.nan) for e, a, b in zip(O.event_id, O.a, O.b)]
        O["season"] = O.event_id.str[:4].astype(int)
        P = pd.concat([P, O.dropna(subset=["quali_gap"]).assign(source="jolpica")], ignore_index=True)
    return P


def paired(d: np.ndarray, rng) -> dict:
    boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_pair_seasons": int(len(d)), "mse_diff": float(d.mean()), "mse_diff_ci95": [float(lo), float(hi)],
            "improves": bool(hi < 0)}


def heldout(C: pd.DataFrame, sfx: str = "") -> dict:
    P = target_pairs()
    P = P[P.source.isin(C.source.unique()) & P.event_id.isin(C.event_id.unique())]
    if "coast" in C:
        mc = C.groupby(["event_id", "driver_id"]).coast.mean()
        P["coast_gap"] = [mc.get((e, a), np.nan) - mc.get((e, b), np.nan) for e, a, b in zip(P.event_id, P.a, P.b)]
        P = P.dropna(subset=["coast_gap"])
    rng = np.random.default_rng(0)
    rows, fits, effects = [], {}, {}
    first = FIRST_TEST_OLD if (C.source == "jolpica").any() else FIRST_TEST
    for S in range(first, int(C.season.max()) + 1):
        post, drivers = checkpointed(C[C.season < S], OUT / f"multi_heldout_cache{sfx}" / f"{S}.npz")
        fits[S] = {k: post[k] for k in ("_rhat_max", "_divergences", "_minutes", "_backend", "_attempts", "_converged")}
        if not post["_converged"]:
            raise RuntimeError(f"fit for held-out season {S} did not converge: {post['_attempts']}")
        effects[f"{S}_drivers"] = np.array(drivers)
        effects[f"{S}_u"] = post["u"].astype(np.float32)
        effects[f"{S}_v"] = post["v"].astype(np.float32)
        u = pd.Series(post["u"].mean(0), index=drivers)
        v = pd.Series(post["v"].mean(0), index=drivers)
        g = float(post["gamma"].mean())
        t = P[P.season == S]
        if COAST:  # targets at typical coasting, with the training fit's coefficient
            t = t.assign(pace_gap=t.pace_gap - float(post["b_coast"].mean()) * t.coast_gap)
            fits[S]["b_coast"] = float(post["b_coast"].mean())
        rows.append(t.assign(pred_base=g * t.quali_gap,
                             pred_pace=g * t.quali_gap + u.reindex(t.a).to_numpy() - u.reindex(t.b).to_numpy(),
                             pred_deg=v.reindex(t.a).to_numpy() - v.reindex(t.b).to_numpy(),
                             seen=t.a.isin(drivers) & t.b.isin(drivers)))
        print(f"held out {S}: {fits[S]}", flush=True)
        jax.clear_caches()
    H = pd.concat(rows, ignore_index=True)
    H = H[H.seen]
    ps = H.groupby(["season", "a", "b"]).agg(n=("pace_gap", "size"), pace=("pace_gap", "mean"),
                                             deg=("deg_gap", "mean"), base=("pred_base", "mean"),
                                             full=("pred_pace", "mean"), dv=("pred_deg", "mean")).reset_index()
    ps = ps[ps.n >= 4]
    out = {"design": f"fit on every season before S (from {int(C.season.min())}); predict season S's teammate gaps "
                     "(pair-season means of stage A gaps, >= 4 races, both drivers seen in training)",
           "units": "percent of lap time; mse in percent squared",
           "fits": fits,
           "race_specific_pace_vs_quali_link": paired(((ps.pace - ps.full) ** 2 - (ps.pace - ps.base) ** 2).to_numpy(), rng),
           "quali_link_vs_zero": paired(((ps.pace - ps.base) ** 2 - ps.pace ** 2).to_numpy(), rng),
           "degradation_vs_zero": paired(((ps.deg - ps.dv) ** 2 - ps.deg ** 2).to_numpy(), rng)}
    out["gate_race_specific_pace"] = out["race_specific_pace_vs_quali_link"]["improves"]
    out["gate_degradation"] = out["degradation_vs_zero"]["improves"]
    OUT.mkdir(parents=True, exist_ok=True)
    out["laps_by_source"] = C.source.value_counts().to_dict()
    np.savez_compressed(OUT / f"multi_heldout_effects{sfx}.npz", **effects)
    (OUT / f"multi_heldout{sfx}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return out


def prefit(C: pd.DataFrame, seasons: list[int], sfx: str = "") -> None:
    """Fit and save the held-out training fits for these seasons only (parallel workers; a
    later --heldout run reuses them)."""
    for S in seasons:
        post, _ = checkpointed(C[C.season < S], OUT / f"multi_heldout_cache{sfx}" / f"{S}.npz")
        print(f"prefit {S}: converged {post['_converged']}, {post['_attempts']}", flush=True)
        jax.clear_caches()


def full(C: pd.DataFrame, sfx: str = "") -> None:
    post, drivers = fit(C, warmup=800, samples=800)
    q = lambda x: [round(float(v), 4) for v in np.percentile(x, [5, 50, 95])]  # noqa: E731
    summary = {"n_laps": int(len(C)), "n_races": int(C.event_id.nunique()),
               "seasons": [int(C.season.min()), int(C.season.max())], "n_drivers": len(drivers),
               "rhat_max": post["_rhat_max"], "divergences": post["_divergences"], "minutes": round(post["_minutes"], 1),
               "backend": post["_backend"], "attempts": post["_attempts"],
               **{f"{k}_q05_q50_q95": q(post[k]) for k in KEEP if k not in ("u", "v") and k in post},
               "laps_by_source": C.source.value_counts().to_dict(),
               "gates": f"in multi_heldout{sfx}.json"}
    laps = C.groupby("driver_id").size()
    seasons = C.groupby("driver_id").season.nunique()
    D = pd.DataFrame({"driver_id": drivers,
                      "race_specific_pct_median": np.median(post["u"], 0),
                      "race_specific_pct_q05": np.percentile(post["u"], 5, 0),
                      "race_specific_pct_q95": np.percentile(post["u"], 95, 0),
                      "degradation_pct_per_lap_median": np.median(post["v"], 0),
                      "degradation_pct_per_lap_q05": np.percentile(post["v"], 5, 0),
                      "degradation_pct_per_lap_q95": np.percentile(post["v"], 95, 0),
                      "clean_laps": laps.reindex(drivers).to_numpy(), "seasons": seasons.reindex(drivers).to_numpy()})
    D.sort_values("race_specific_pct_median", ascending=False).to_csv(OUT / f"multi_drivers{sfx}.csv", index=False)
    np.savez_compressed(OUT / f"multi_driver_draws{sfx}.npz", drivers=np.array(drivers), u=post["u"].astype(np.float32),
                        v=post["v"].astype(np.float32))
    (OUT / f"multi_summary{sfx}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--heldout", action="store_true")
    p.add_argument("--timing", type=int, nargs=2, metavar=("FIRST", "LAST"),
                   help="time a short fit on these seasons")
    p.add_argument("--fastf1-only", action="store_true", help="2018 onward only (no Jolpica laps)")
    p.add_argument("--telemetry-races", action="store_true", help="FastF1 races with aligned telemetry only")
    p.add_argument("--coast", action="store_true", help="those races, with the coasting covariate")
    p.add_argument("--only", type=int, nargs="+", metavar="S",
                   help="only fit and save the held-out training fits for these seasons")
    args = p.parse_args()
    global COAST
    COAST = args.coast
    tel = args.coast or args.telemetry_races
    C = all_laps(old=not (args.fastf1_only or tel), telemetry=tel)
    sfx = ("_fastf1only_coast" if args.coast else "_fastf1only_telemetry" if tel
           else "_fastf1only" if args.fastf1_only else "")
    if args.timing:
        post, _ = fit(C[C.season.between(*args.timing)], warmup=150, samples=150)
        print({k: post[k] for k in ("_rhat_max", "_divergences", "_minutes", "_attempts")},
              len(C[C.season.between(*args.timing)]))
        return
    if args.only:
        prefit(C, args.only, sfx)
    elif args.heldout:
        heldout(C, sfx)
    else:
        full(C, sfx)


if __name__ == "__main__":
    main()
