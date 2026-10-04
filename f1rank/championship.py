"""Equal-car championship: a sequential weekend simulation with equal machinery (stage 2, step 8).

    python -m f1rank.championship

The weekend chain (docs/racing_approach.md), with each stage from the part of the racing
model that has passed its gates:

1. Qualifying: each driver's qualifying pace (stage 1) + weekend form + the car's one-event
   effect + session noise (stage 1's estimates) -> grid order. With equal cars the car term is the same for all.
2. Race given the grid: finishing order among cars still running, a Plackett-Luce ranking
   with strength  c_grid * (-log grid slot) + a_car * car + b_driver * driver, fitted on
   real classified finishers with the real cars (car and driver from stage 1). With equal
   cars the car term is the same for everyone and drops out. First-lap, overtaking
   and strategy effects are not simulated lap by lap: their average effect is folded into
   this grid-to-finish mapping, and a driver-specific racing quality adds to the strength
   (b_q * quality) only if it passes the entry test below.
3. Retirements: per-lap hazards from the reliability model with equal mechanical risk
   (the average team-season) and each driver's own-error effect only if driver error rates
   passed their gate; otherwise every driver has the average rate.
4. Points: the current system (25-18-15-12-10-8-6-4-2-1), no fastest-lap point.

Two versions of the driver term:
- in_team (headline): each driver keeps their team-specific effect ("every car made as
  fast as the average car, each driver keeping their fit with their current team").
- portable (experimental): portable skill only.

The race stage is checked on held-out seasons: grid + ratings must predict finishing orders
better than the grid alone and than the ratings alone (as in benchmark.py). Outputs are
posterior simulations: expected points per race, P(title), rank ranges.

Entry test: qualifying inputs are frozen before each held-out season. Candidate qualities
use their own earlier-season fits. Backward conditional ablation removes qualities that
fail to improve the combined model, then retests the survivors. An outer held-out season
never participates in its own selection: selection uses at least three earlier test
seasons. Season-block bootstrap intervals assess the complete selection procedure against
the base race model. If that outer gate fails, no racing quality enters the simulation.
Training uses posterior means; prediction integrates quality draws and preserves covariance
between qualities saved by the same fit. The scenario remains experimental and retains
team-specific effects in its headline version.

Contribution breakdown (in_team version): per driver, the expected points per race lost when
qualifying pace, or one entered quality, is set to the field average with the others kept,
and when all are (one-at-a-time and all-at-once; correlated qualities have no unique
allocation, so both are reported).

Writes outputs/championship/: summary.json, standings.csv, contributions.csv.
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

from .benchmark import FIRST_TEST, plackett_luce, race_orders  # noqa: E402
from .fit import FITS, load  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "championship"
POINTS = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1])
N_SEASONS = 2000       # simulated seasons (one posterior draw each)
N_BOOT = 2000


def finishers(R: pd.DataFrame) -> pd.DataFrame:
    """Classified finishers only (retirements are the reliability model's job), re-ranked."""
    from .timeline import FINISHED
    F = R[R.status.str.match(FINISHED.pattern)].copy()
    F["rank"] = F.groupby("event_id").cumcount()
    return F


ENTRY_FIRST_SEASON = 2010
QUALITIES = ("first_lap", "race_specific_pace", "degradation", "consistency", "overtaking_attack",
             "overtaking_defend")


def slots(R: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    races = sorted(R.event_id.unique())
    ri = {e: i for i, e in enumerate(races)}
    return R.event_id.map(ri).to_numpy(), R["rank"].to_numpy()


def arrays(R: pd.DataFrame) -> dict:
    races = sorted(R.event_id.unique())
    width = int(R.groupby("event_id").size().max())
    shape = (len(races), width)
    car, drv, log_grid, mask = np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape, bool)
    qcols = [c for c in R.columns if c.startswith("quality_")]
    qual = np.zeros(shape + (max(1, len(qcols)),))
    r, k = slots(R)
    car[r, k], drv[r, k] = R.car, R.driver
    log_grid[r, k] = np.log(R.grid_slot)
    for j, c in enumerate(qcols):
        qual[r, k, j] = R[c]
    mask[r, k] = True
    return {"car": jnp.asarray(car), "driver": jnp.asarray(drv), "log_grid": jnp.asarray(log_grid),
            "quality": jnp.asarray(qual), "mask": jnp.asarray(mask)}


def race_model(d, variant="grid_ratings"):
    s = jnp.zeros_like(d["car"])
    if variant in ("grid_ratings", "grid", "grid_ratings_quality"):
        s = s - numpyro.sample("c_grid", dist.Normal(0, 5)) * d["log_grid"]
    if variant in ("grid_ratings", "ratings", "grid_ratings_quality"):
        s = s + numpyro.sample("a_car", dist.Normal(0, 5)) * d["car"]
        s = s + numpyro.sample("b_driver", dist.Normal(0, 5)) * d["driver"]
    if variant == "grid_ratings_quality":
        b_q = numpyro.sample("b_quality", dist.Normal(0, 5).expand([d["quality"].shape[-1]]))
        s = s + (d["quality"] * b_q).sum(-1)
    ll = plackett_luce(s, d["mask"])
    numpyro.factor("ll", ll.sum())
    numpyro.deterministic("ll_race", ll)


def fit_race(R, variant="grid_ratings", warmup=500, samples=500) -> dict:
    from .artifacts import diagnostics, fit_until_converged

    def run(w, s, accept, seed):
        mcmc = MCMC(NUTS(partial(race_model, variant=variant), target_accept_prob=accept or 0.8), num_warmup=w,
                    num_samples=s, num_chains=4, chain_method="parallel", progress_bar=False)
        mcmc.run(jax.random.PRNGKey(seed), arrays(R), extra_fields=("diverging",))
        return mcmc, diagnostics(
            {k: v for k, v in mcmc.get_samples(group_by_chain=True).items() if k != "ll_race"},
            int(np.asarray(mcmc.get_extra_fields()["diverging"]).sum()))

    mcmc = fit_until_converged(run, warmup, samples)
    return {k: np.asarray(v) for k, v in mcmc.get_samples().items() if k != "ll_race"}


def log_pred(post, R, variant="grid_ratings") -> np.ndarray:
    ll = np.asarray(Predictive(partial(race_model, variant=variant), posterior_samples=post,
                               return_sites=["ll_race"])(jax.random.PRNGKey(1), arrays(R))["ll_race"])
    m = ll.max(0)
    return m + np.log(np.exp(ll - m).mean(0))


def heldout(F: pd.DataFrame, rng) -> dict:
    """Finishing orders of held-out seasons: grid + ratings vs grid only and vs ratings only."""
    from .qualifying import unconverged_folds
    lp = {v: [] for v in ("grid_ratings", "grid", "ratings")}
    excluded = [S for S in unconverged_folds() if FIRST_TEST <= S <= F.season.max()]
    for S in range(FIRST_TEST, F.season.max() + 1):
        if S in excluded:
            print(f"held out {S}: left out, its qualifying fold did not converge", flush=True)
            continue
        fold = finishers(race_orders(S))
        train, test = fold[fold.season < S], fold[fold.season == S]
        for v in lp:
            lp[v].append(log_pred(fit_race(train, v), test, v))
        print(f"held out {S}", flush=True)
    lp = {v: np.concatenate(x) for v, x in lp.items()}
    out = {"excluded_unconverged_qualifying_folds": excluded}
    for other in ("grid", "ratings"):
        d = lp["grid_ratings"] - lp[other]
        boot = np.array([d[rng.integers(len(d), size=len(d))].mean() for _ in range(N_BOOT)])
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[f"vs_{other}"] = {"n_races": int(len(d)), "mean_diff_per_race": float(d.mean()),
                              "ci95": [float(lo), float(hi)]}
    return out


# held-out driver-effect draws (per season S: "<S>_drivers", "<S>_<key>") and the full-data file
QUALITY_FILES = {"race_specific_pace": ("race/multi_heldout_effects.npz", "u"),
                 "degradation": ("race/multi_heldout_effects.npz", "v"),
                 "consistency": ("consistency/heldout_effects.npz", "c"),
                 "overtaking_attack": ("battles/heldout_effects.npz", "attack"),
                 "overtaking_defend": ("battles/heldout_effects.npz", "defend")}
FULL_FILES = {"race/multi_heldout_effects.npz": "race/multi_driver_draws.npz",
              "consistency/heldout_effects.npz": "consistency/driver_draws.npz",
              "battles/heldout_effects.npz": "battles/driver_draws.npz"}


def quality_draws(name: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per held-out season S: (driver ids, draws x drivers) of a driver quality fitted on the
    seasons before S, as saved by the quality's own held-out test."""
    from .artifacts import require
    if name == "first_lap":
        require(ROOT / "outputs" / "firstlap", required_outputs=[ROOT / "outputs" / "firstlap" / "heldout_effects.npz"])
        z = np.load(ROOT / "outputs" / "firstlap" / "heldout_effects.npz")
        return {int(k): (z["drivers"], z[k]) for k in z.files if k != "drivers"}
    path, key = QUALITY_FILES[name]
    p = ROOT / "outputs" / path
    # Missing optional battle draws are distinct from stale, present draws.
    if not p.exists():
        raise FileNotFoundError(2, "no held-out draws", str(p))
    require(p.parent, name="multi_heldout.manifest.json" if name in ("race_specific_pace", "degradation")
            else "manifest.json", required_outputs=[p])
    z = np.load(ROOT / "outputs" / path)
    seasons = sorted({int(k.split("_")[0]) for k in z.files})
    return {S: (z[f"{S}_drivers"], z[f"{S}_{key}"]) for S in seasons}


def full_quality_draws(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(driver ids, draws x drivers) of a quality fitted on all its data."""
    from .artifacts import require
    if name == "first_lap":
        require(ROOT / "outputs" / "firstlap", required_outputs=[ROOT / "outputs" / "firstlap" / "driver_draws.npz"])
        z = np.load(ROOT / "outputs" / "firstlap" / "driver_draws.npz")
        return z["drivers"], z["driver"]
    path, key = QUALITY_FILES[name]
    p = ROOT / "outputs" / FULL_FILES[path]
    require(p.parent, name="multi_full.manifest.json" if name in ("race_specific_pace", "degradation")
            else "manifest.json", required_outputs=[p])
    z = np.load(p)
    return z["drivers"], z[key]


MIN_SELECTION_SEASONS = 3


def paired_seasons(diffs: dict[int, np.ndarray]) -> dict:
    """Resample whole seasons so repeated driver/team effects stay in their blocks."""
    years = sorted(diffs)
    if not years:
        return {"n_races": 0, "seasons": [], "mean_diff_per_race": 0.0,
                "ci95": [0.0, 0.0], "enters": False}
    sums = np.array([np.sum(diffs[S]) for S in years])
    counts = np.array([len(diffs[S]) for S in years])
    picks = np.random.default_rng(0).integers(len(years), size=(N_BOOT, len(years)))
    boot = sums[picks].sum(1) / counts[picks].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"n_races": int(counts.sum()), "seasons": [min(years), max(years)],
            "mean_diff_per_race": float(sums.sum() / counts.sum()),
            "ci95": [float(lo), float(hi)],
            "enters": bool(len(years) >= MIN_SELECTION_SEASONS and lo > 0)}


def select_qualities(candidates, seasons, score):
    """Backward removal, using only the supplied inner seasons.

    Every surviving quality must improve the current combined model relative to
    removing that quality. Re-test after each removal. No outer-season score is
    accessed here. The same deterministic rule is used for deployment selection.
    """
    selected, tests = sorted(candidates), {}
    if len(seasons) < MIN_SELECTION_SEASONS:
        return [], tests
    while selected:
        full = {S: score(S, tuple(selected)) for S in seasons}
        checks = {}
        for name in selected:
            without = tuple(n for n in selected if n != name)
            checks[name] = paired_seasons({S: full[S] - score(S, without) for S in seasons})
            checks[name]["conditioned_on"] = list(without)
        tests.update(checks)
        failed = [n for n in selected if not checks[n]["enters"]]
        if not failed:
            break
        drop = min(failed, key=lambda n: (checks[n]["mean_diff_per_race"], n))
        tests[drop]["enters"] = False
        selected.remove(drop)
    return selected, tests


def nested_selection(candidates, seasons, score):
    outer, selections = {}, {}
    for S in seasons:
        inner = [s for s in seasons if s < S]
        if len(inner) < MIN_SELECTION_SEASONS:
            continue
        chosen, _ = select_qualities(candidates, inner, score)
        selections[str(S)] = chosen
        outer[S] = score(S, tuple(chosen)) - score(S, ())
    chosen, entry = select_qualities(candidates, seasons, score)
    result = paired_seasons(outer)
    result.update(selected_by_outer_season=selections, gate=result.pop("enters"),
                  design="season-ahead qualifying; backward conditional ablations on inner seasons only; "
                         "outer test of the selection procedure; season-block bootstrap")
    return chosen, entry, result


def quality_log_pred(post, test, names, effects, season, n_draws=1000):
    """Integrate quality draws; preserve covariance for qualities from one fit."""
    d = arrays(test)
    r, k = slots(test)
    rng = np.random.default_rng(season)
    pick = rng.integers(len(post["b_driver"]), size=n_draws)
    strength = (-post["c_grid"][pick, None, None] * d["log_grid"]
                + post["a_car"][pick, None, None] * d["car"]
                + post["b_driver"][pick, None, None] * d["driver"])
    source_picks = {}
    for j, name in enumerate(names):
        drivers, draws = effects[name][season]
        source = QUALITY_FILES[name][0] if name in QUALITY_FILES else name
        if source not in source_picks:
            source_picks[source] = rng.integers(len(draws), size=n_draws)
        q = np.zeros((n_draws,) + tuple(d["mask"].shape))
        col = pd.Series(np.arange(len(drivers)), index=drivers).reindex(test.driver_id).to_numpy()
        known = ~np.isnan(col)
        q[:, r[known], k[known]] = draws[source_picks[source]][:, col[known].astype(int)]
        strength = strength + post["b_quality"][pick, j, None, None] * q
    ll = np.asarray(jax.vmap(plackett_luce, in_axes=(0, None))(jnp.asarray(strength), d["mask"]))
    m = ll.max(0)
    return m + np.log(np.exp(ll - m).mean(0))


def combined_entry_tests(effects, last_season):
    """Evaluate the entire selection procedure on outer held-out seasons."""
    from functools import lru_cache

    from .qualifying import unconverged_folds
    candidates = sorted(effects)
    if not candidates:
        return [], {}, {"status": "no candidate qualities", "gate": False}
    seasons = sorted(set.intersection(*(set(effects[n]) for n in candidates)))
    excluded = [S for S in unconverged_folds() if S <= last_season]
    seasons = [S for S in seasons if S <= last_season and S not in excluded]
    folds = {}

    @lru_cache(maxsize=None)
    def score(S, names):
        if S not in folds:
            folds[S] = finishers(race_orders(S))
        F = folds[S]
        train = F[(F.season >= ENTRY_FIRST_SEASON) & (F.season < S)].copy()
        test = F[F.season == S]
        if train.empty or test.empty:
            raise ValueError(f"Empty championship fold {S}")
        if not names:
            return log_pred(fit_race(train), test)
        for j, name in enumerate(names):
            drivers, draws = effects[name][S]
            train[f"quality_{j}"] = train.driver_id.map(pd.Series(draws.mean(0), index=drivers)).fillna(0.0)
        post = fit_race(train, "grid_ratings_quality")
        result = quality_log_pred(post, test, names, effects, S)
        print(f"combined test held out {S}: {', '.join(names)}", flush=True)
        jax.clear_caches()
        return result

    chosen, entry, result = nested_selection(candidates, seasons, score)
    result["excluded_unconverged_qualifying_folds"] = excluded
    return chosen, entry, result


BREAKDOWN_SEED = 12345


def breakdown(Qs, extras, form_sd, noise_sd, race_post, race_pick, p_retire, n_races, ids, names) -> list[dict]:
    """Expected points per race lost when one of a driver's qualities is set to the field
    average (0 after centring), the others kept, and when all are; conditional model
    estimates, not a unique allocation (docs/racing_approach.md). Every simulation uses the
    same random numbers, so differences are not swamped by simulation noise."""
    def run(Q, parts):
        extra = sum(parts.values()) if parts else None
        return simulate(Q, form_sd, noise_sd, race_post["b_driver"][race_pick], race_post["c_grid"][race_pick],
                        p_retire, n_races, np.random.default_rng(BREAKDOWN_SEED), extra).mean(0) / n_races
    full = run(Qs, extras)
    rows = []
    for i, d in enumerate(ids):
        row = {"driver_id": d, "name": names.get(d, d), "points_per_race": float(full[i])}
        Q0 = Qs.copy()
        Q0[:, i] = 0.0
        row["loss_qualifying_pace_at_average"] = float(full[i] - run(Q0, extras)[i])
        for n in extras:
            parts = {k: v.copy() for k, v in extras.items()}
            parts[n][:, i] = 0.0
            row[f"loss_{n}_at_average"] = float(full[i] - run(Qs, parts)[i])
        parts = {k: v.copy() for k, v in extras.items()}
        for v in parts.values():
            v[:, i] = 0.0
        row["loss_all_at_average"] = float(full[i] - run(Q0, parts)[i])
        rows.append(row)
    return sorted(rows, key=lambda r: -r["points_per_race"])


def simulate(Q: np.ndarray, form_sd: np.ndarray, noise_sd: np.ndarray, b_driver: np.ndarray, c_grid: np.ndarray,
             p_retire: np.ndarray, n_races: int, rng, extra: np.ndarray | None = None) -> np.ndarray:
    """Points per simulated season (draws x drivers). Q: driver qualifying pace per draw;
    extra: race-stage strength from qualities that passed the entry test (draws x drivers)."""
    S, n = Q.shape
    points = np.zeros((S, n))
    for _ in range(n_races):
        quali = Q + rng.standard_normal((S, n)) * form_sd[:, None] + rng.standard_normal((S, n)) * noise_sd[:, None]
        slot = (-quali).argsort(1).argsort(1) + 1
        strength = b_driver[:, None] * Q - c_grid[:, None] * np.log(slot)  # equal cars: no car term
        if extra is not None:
            strength = strength + extra
        g = rng.gumbel(size=(S, n))  # Plackett-Luce by the Gumbel-max trick
        running = rng.random((S, n)) >= p_retire[None, :]
        key = np.where(running, strength + g, -np.inf)
        order = (-key).argsort(1)
        pos = order.argsort(1)
        pts = np.where(pos < len(POINTS), POINTS[np.minimum(pos, len(POINTS) - 1)], 0) * running
        points += pts
    return points


def main() -> None:
    from .artifacts import input_files, record, require
    from .qualifying import POLICY, dependencies, features, unconverged_folds
    rng = np.random.default_rng(0)
    # Preflight every dependency before starting expensive validation. Legacy files
    # without provenance must be regenerated, never quietly treated as current.
    require(ROOT / "outputs" / "reliability", required_outputs=[
        ROOT / "outputs" / "reliability" / "summary.json", ROOT / "outputs" / "reliability" / "drivers.csv"])
    missing, effects = {}, {}
    for name in QUALITIES:
        try:
            effects[name] = quality_draws(name)
        except FileNotFoundError as e:
            if not name.startswith("overtaking_"):
                raise
            missing[name] = {"not_run": f"no held-out draws ({Path(e.filename).relative_to(ROOT)})"}
        except KeyError:  # e.g. defender effects when only attacker effects were tested
            if not name.startswith("overtaking_"):
                raise
            missing[name] = {"not_run": "no held-out draws for this quality"}
    full = {n: full_quality_draws(n) for n in effects}
    R = race_orders()
    F = finishers(R)
    all_seasons = sorted(set().union(*(set(v) for v in effects.values())))
    for S in all_seasons:
        if S not in unconverged_folds():
            features(S)
    test = heldout(F, rng)
    selected, entry, combined = combined_entry_tests(effects, int(F.season.max()))
    entry.update(missing)
    entered = selected if combined["gate"] else []
    if entered:
        # race stage refitted on the seasons the qualities cover, with every quality that entered
        Fq = F[F.season >= ENTRY_FIRST_SEASON].assign(**{
            f"quality_{j}": F.driver_id.map(pd.Series(full[n][1].mean(0), index=full[n][0])).fillna(0.0)
            for j, n in enumerate(entered)})
        race_post = fit_race(Fq, "grid_ratings_quality", warmup=800, samples=800)
    else:
        race_post = fit_race(F, warmup=800, samples=800)

    # stage 1 draws at the latest event
    from .design import build_design
    design = build_design(2010)
    post, _ = load(FITS / "main.npz", design)
    last = design.events.event_idx.max()
    rows = np.flatnonzero(design.entries.event_idx.to_numpy() == last)
    ids = design.entries.driver_id.to_numpy()[rows]
    flat = lambda k: post[k].reshape(-1, *post[k].shape[2:])  # noqa: E731
    skill = flat("skill")[:, rows]
    compat = flat("compat")[:, rows]
    versions = {"in_team": skill + compat, "portable": skill}
    pick = rng.choice(skill.shape[0], N_SEASONS, replace=True)
    # weekend-to-weekend variation in qualifying: driver form, the car's one-event effect and
    # session noise (each car is equal on average, not identical every weekend)
    form_sd = flat("sd_driver_form")[pick]
    noise_sd = np.sqrt(flat("sd_car_event")[pick] ** 2 + flat("sigma0")[pick] ** 2)
    race_pick = rng.integers(len(race_post["b_driver"]), size=N_SEASONS)

    # retirements: reliability model, equal cars; driver own-error effects only if gated
    rel = json.loads((ROOT / "outputs" / "reliability" / "summary.json").read_text())
    per100 = {c: v[1] / 100 for c, v in rel["hazard_per_100_laps_later_laps"].items()}
    lap1 = {c: v[1] / 100 for c, v in rel["hazard_lap1_per_100_starts"].items()}
    laps = 57  # median race distance since 2010
    p_common = 1 - np.exp(-(sum(lap1.values()) + (laps - 1) * sum(per100.values())))
    p_retire = np.full(len(ids), p_common)
    if rel["gate_driver_error_ranking"]:
        dr = pd.read_csv(ROOT / "outputs" / "reliability" / "drivers.csv").set_index("driver_id")
        mult = dr.own_error_rate_multiplier_median.reindex(ids).fillna(1.0).to_numpy()
        own = lap1["own_error"] + (laps - 1) * per100["own_error"]
        p_retire = 1 - np.exp(-(sum(lap1.values()) + (laps - 1) * sum(per100.values()) + own * (mult - 1)))
    extra, extras = None, {}
    if entered:
        extra = np.zeros((N_SEASONS, len(ids)))
        source_picks = {}
        for j, n in enumerate(entered):
            drivers, draws = full[n]
            col = pd.Series(np.arange(len(drivers)), index=drivers).reindex(ids).to_numpy()
            val = np.zeros((N_SEASONS, len(ids)))
            known = ~np.isnan(col)
            source = QUALITY_FILES[n][0] if n in QUALITY_FILES else n
            if source not in source_picks:
                source_picks[source] = rng.integers(len(draws), size=N_SEASONS)
            val[:, known] = draws[source_picks[source]][:, col[known].astype(int)]
            # centred on the current field in each draw (as qualifying pace is), drivers without
            # data for the quality at the field average
            val[:, ~known] = val[:, known].mean(1, keepdims=True)
            val -= val.mean(1, keepdims=True)
            extras[n] = race_post["b_quality"][race_pick, j][:, None] * val
            extra += extras[n]
    n_races = int(design.events[design.events.season == design.events.season.max()].shape[0])
    n_races = max(n_races, 24)  # a full season

    names = design.drivers.set_index("driver_id").name
    out_rows, summary = [], {"heldout_race_stage": test, "entry_tests": entry, "qualities_entered": entered,
                             "combined_validation": combined, "qualities_selected": selected,
                             "qualifying_validation_policy": POLICY,
                             "status": "experimental equal-car scenario; team-specific effects retained in headline",
                             "n_races_simulated": n_races,
                             "simulated_seasons": N_SEASONS, "p_retire_per_race": float(p_common),
                             "driver_error_rates_used": bool(rel["gate_driver_error_ranking"]),
                             "b_driver_q05_q50_q95": [float(x) for x in np.percentile(race_post["b_driver"], [5, 50, 95])],
                             "a_car_q05_q50_q95": [float(x) for x in np.percentile(race_post["a_car"], [5, 50, 95])],
                             "c_grid_q05_q50_q95": [float(x) for x in np.percentile(race_post["c_grid"], [5, 50, 95])]}
    for version, Q in versions.items():
        Qs = Q[pick] - Q[pick].mean(1, keepdims=True)
        pts = simulate(Qs, form_sd, noise_sd, race_post["b_driver"][race_pick], race_post["c_grid"][race_pick],
                       p_retire, n_races, rng, extra)
        rank = (-pts).argsort(1).argsort(1) + 1
        for i, d in enumerate(ids):
            out_rows.append({"version": version, "driver_id": d, "name": names.get(d, d),
                             "points_per_race": float(pts[:, i].mean() / n_races),
                             "p_title": float((rank[:, i] == 1).mean()),
                             "rank_median": float(np.median(rank[:, i])),
                             "rank_lo": int(np.percentile(rank[:, i], 5)), "rank_hi": int(np.percentile(rank[:, i], 95))})
        if version == "in_team":
            summary["contribution_breakdown"] = breakdown(Qs, extras, form_sd, noise_sd, race_post, race_pick,
                                                          p_retire, n_races, ids, names)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary.pop("contribution_breakdown")).to_csv(OUT / "contributions.csv", index=False)
    table = pd.DataFrame(out_rows).sort_values(["version", "points_per_race"], ascending=[True, False])
    table.to_csv(OUT / "standings.csv", index=False)
    from .artifacts import fit_record
    summary["fit_attempts"] = fit_record()
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    dependency_paths = [ROOT / "outputs" / "reliability" / f for f in ("summary.json", "drivers.csv", "manifest.json")]
    for n in effects:
        if n == "first_lap":
            dependency_paths += [ROOT / "outputs" / "firstlap" / f for f in
                                 ("heldout_effects.npz", "driver_draws.npz", "manifest.json")]
        else:
            path, _ = QUALITY_FILES[n]
            dependency_paths += [ROOT / "outputs" / p for p in (path, FULL_FILES[path])]
    record(OUT, [OUT / f for f in ("summary.json", "standings.csv", "contributions.csv")], model="championship-v2",
           inputs=input_files() + dependencies([None, *all_seasons]) + dependency_paths,
           details={"qualifying_policy": POLICY, "combined_gate": combined["gate"], "status": "experimental"})
    print(json.dumps(summary, indent=1))
    print(table[table.version == "in_team"].head(10).to_string())


if __name__ == "__main__":
    main()
