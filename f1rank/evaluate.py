"""Score validation fits: forecasting (leave-future-out), synthetic recovery, sensitivity.

    python -m f1rank.evaluate lfo | synth | sens | placebo | all

Forecast targets never use data after the cutoff, including the circuit factor.
Each batch of fits is scored on the data it was fitted on: the design is rebuilt as of
the batch's data date, every fit is checked against it (see fit.py), and a batch whose
fits were made on different data is refused.
"""

import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import lsqr
from scipy.stats import spearmanr

from .design import Design, build_design
from .fit import FITS, common_data_as_of, load, load_meta
from .jobs import SENS, SYNTH_SOURCE, lfo_cutoffs, lfo_design
from .ratings import SEC_PER_PCT, driver_leaderboard, flat

ROOT = FITS.parent
REPORTS = ROOT / "validation"
MIN_TRAIN_ENTRIES = 10  # "established" driver: at least this many events before the cutoff


# --------------------------------------------------------------------------- baselines

def static_two_way(design: Design) -> tuple[pd.Series, pd.Series]:
    """AKM-style benchmark on training data: constant driver effect + team-season car
    effect + segment intercepts, lightly ridge-regularised. Returns (driver, team_season)."""
    o = design.obs.loc[design.train]
    e = design.entries.set_index("entry_idx")
    drv = e.driver_idx.loc[o.entry_idx].to_numpy()
    ts = design.cars.set_index("car_idx").team_season_idx.loc[o.car_idx].to_numpy()
    ses = o.session_idx.to_numpy()
    n_d, n_ts, n_s = len(design.drivers), design.cars.team_season_idx.max() + 1, len(design.sessions)
    rows = np.arange(len(o))
    X = sp.hstack([
        sp.csr_matrix((np.ones(len(o)), (rows, ses)), shape=(len(o), n_s)),
        sp.csr_matrix((np.ones(len(o)), (rows, drv)), shape=(len(o), n_d)),
        sp.csr_matrix((np.ones(len(o)), (rows, ts)), shape=(len(o), n_ts)),
    ]).tocsr()
    beta = lsqr(X, o.y.to_numpy(), damp=0.3, atol=1e-10, btol=1e-10, iter_lim=20000)[0]
    d_eff = pd.Series(beta[n_s:n_s + n_d], index=design.drivers.driver_id)
    ts_eff = pd.Series(beta[n_s + n_d:], index=np.arange(n_ts))
    seen = np.bincount(drv, minlength=n_d) > 0
    d_eff[~seen] = np.nan
    return d_eff, ts_eff


def teammate_deltas(design: Design, cutoff: int) -> tuple[pd.Series, pd.Series]:
    """Naive benchmark: each driver's mean raw gap to their teammate(s) in their latest
    training season, and each pairing's mean raw gap in their latest season together."""
    p = session_pairs(design)
    p = p[p.event_idx <= cutoff]
    both = pd.concat([
        p[["driver_a", "season", "gap"]].rename(columns={"driver_a": "driver"}),
        p[["driver_b", "season", "gap"]].rename(columns={"driver_b": "driver"}).assign(gap=-p.gap),
    ])
    last = both.groupby("driver").season.transform("max")
    delta = both[both.season == last].groupby("driver").gap.mean()
    lastp = p.groupby(["driver_a", "driver_b"]).season.transform("max")
    pair = p[p.season == lastp].groupby(["driver_a", "driver_b"]).gap.mean()
    return delta, pair


# --------------------------------------------------------------------------- helpers

def session_pairs(design: Design) -> pd.DataFrame:
    """Teammate pairs in the same segment: gap = y_a - y_b (percent), driver_a < driver_b."""
    e = design.entries.set_index("entry_idx")
    o = design.obs.assign(driver=e.driver_id.loc[design.obs.entry_idx].to_numpy(),
                          season=e.season.loc[design.obs.entry_idx].to_numpy())
    p = o.merge(o, on=["session_idx", "car_idx"], suffixes=("_a", "_b"))
    p = p[p.driver_a < p.driver_b]
    return pd.DataFrame({
        "session_idx": p.session_idx, "event_idx": p.event_idx_a, "season": p.season_a,
        "driver_a": p.driver_a, "driver_b": p.driver_b,
        "entry_a": p.entry_idx_a, "entry_b": p.entry_idx_b, "gap": p.y_a - p.y_b,
    }).reset_index(drop=True)


def _coverage(draws: np.ndarray, obs: np.ndarray, level: float) -> float:
    lo, hi = np.percentile(draws, [50 - level / 2, 50 + level / 2], axis=0)
    return float(np.mean((obs >= lo) & (obs <= hi)))


def _rmse(a, b) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


# --------------------------------------------------------------------------- forecasting

def draw_noise(post: dict, sigma: np.ndarray, rng) -> np.ndarray:
    """One noise draw per posterior draw and column, from the fitted likelihood."""
    if "p_compromised" in post:
        from .model import COMPROMISED_DF
        p_bad = post["p_compromised"].reshape(-1, 1)
        wide = 1 + post["compromised_extra_scale"].reshape(-1, 1)
        shift = post["compromised_shift"].reshape(-1, 1)
        bad = rng.random(sigma.shape) < p_bad
        return np.where(bad, sigma * (wide * rng.standard_t(COMPROMISED_DF, sigma.shape) - shift),
                        sigma * rng.standard_normal(sigma.shape))
    nu = post["nu"].reshape(-1, 1)
    t = rng.standard_t(nu, sigma.shape)
    if "slow_scale_ratio" not in post:
        return sigma * t
    slow = sigma * post["slow_scale_ratio"].reshape(-1, 1)
    slow_side = rng.random(sigma.shape) < slow / (sigma + slow)
    return np.where(slow_side, -slow, sigma) * np.abs(t)


def crps(draws: np.ndarray, obs: np.ndarray, rng, n: int = 400) -> float:
    """Mean CRPS from draws (S, n_obs): E|X - y| - 0.5 E|X - X'|."""
    idx = rng.choice(draws.shape[0], min(n, draws.shape[0]), replace=False)
    x = draws[idx]
    x2 = x[rng.permutation(len(x))]
    return float(np.mean(np.abs(x - obs).mean(0) - 0.5 * np.abs(x - x2).mean(0)))


def evaluate_cutoff(design: Design, name: str, cutoff: int, rng, fit_file=None,
                    end_of_season: bool | None = None) -> dict:
    dtrain = design.with_cutoff(cutoff)
    post, _ = load(fit_file or FITS / f"{name}.npz", dtrain)
    ev = design.events
    season = ev.season[cutoff]
    if end_of_season is None:
        end_of_season = name.startswith("lfo_end")
    test_season = season + 1 if end_of_season else season
    test_events = ev[(ev.event_idx > cutoff) & (ev.season == test_season)].event_idx
    if not len(test_events):
        return {}
    n_train = dtrain.entries[dtrain.entries.event_idx <= cutoff].groupby("driver_id").size()
    established = lambda d: n_train.get(d, 0) >= MIN_TRAIN_ENTRIES  # noqa: E731

    skill = flat(post, "skill")
    if "compat" in post:
        # a forecast uses the driver's effect in that team: portable skill plus the
        # stint's compatibility (prior mean 0 for a stint not yet observed)
        skill = skill + flat(post, "compat")
    n_draw = skill.shape[0]
    sigma = (post["sigma0"].reshape(-1, 1) * np.exp(post["sigma_tau"].reshape(-1, 1)
                                                     * flat(post, "sigma_u")))

    # ---- teammate gaps (driver component only)
    p = session_pairs(design)
    p = p[p.event_idx.isin(test_events)].reset_index(drop=True)
    if "sd_driver_form" in post:
        # future weekends carry a one-event form draw for each driver (shared by that
        # driver's segments at the event), as in the model
        form = rng.standard_normal(skill.shape) * post["sd_driver_form"].reshape(-1, 1)
        skill = skill + form
    diff = skill[:, p.entry_a] - skill[:, p.entry_b]
    sig = sigma[:, p.session_idx]
    noise = draw_noise(post, sig, rng) - draw_noise(post, sig, rng)
    p["pred"] = diff.mean(0)
    p["cov50"] = np.nan
    rep = diff + noise
    p["est"] = [established(a) and established(b) for a, b in zip(p.driver_a, p.driver_b)]

    d_eff, ts_eff = static_two_way(dtrain)
    p["akm"] = (d_eff.reindex(p.driver_a).to_numpy() - d_eff.reindex(p.driver_b).to_numpy())
    delta, pair = teammate_deltas(design, cutoff)
    pair_key = list(zip(p.driver_a, p.driver_b))
    known_pair = np.array([k in pair.index for k in pair_key])
    p["new_pair"] = ~known_pair
    p["naive"] = np.where(known_pair, [pair.get(k, np.nan) for k in pair_key],
                          delta.reindex(p.driver_a).to_numpy() - delta.reindex(p.driver_b).to_numpy())
    for col in ("akm", "naive"):
        p[col] = p[col].fillna(0.0)

    est = p.est.to_numpy()
    tm = {"n_sessions": int(est.sum())}
    for col in ("pred", "akm", "naive"):
        tm[f"rmse_{col}"] = _rmse(p.gap[est], p[col][est])
    tm["rmse_zero"] = _rmse(p.gap[est], 0)
    tm["cov50"] = _coverage(rep[:, est], p.gap[est].to_numpy(), 50)
    tm["cov90"] = _coverage(rep[:, est], p.gap[est].to_numpy(), 90)
    tm["crps"] = crps(rep[:, est], p.gap[est].to_numpy(), rng)

    # pairing-level: mean over the pair's test sessions (much less noise)
    g = p[est].groupby(["driver_a", "driver_b"])
    pl = g[["gap", "pred", "akm", "naive"]].mean().join(g.size().rename("n")).join(
        g.new_pair.first())
    pl = pl[pl.n >= 6]
    pair_rows = []
    for (a, b), row in pl.iterrows():
        m = ((p.driver_a == a) & (p.driver_b == b) & p.est).to_numpy()
        # interval for the pairing mean: skill-diff draws averaged over the pair's sessions
        mean_draws = diff[:, m].mean(1) + noise[:, m].mean(1)
        lo, hi = np.percentile(mean_draws, [5, 95])
        pair_rows.append({"driver_a": a, "driver_b": b, "cutoff": name, **row.to_dict(),
                          "in90": bool(lo <= row.gap <= hi)})

    # ---- full relative pace (car + driver) within each test segment
    car = flat(post, "car") + flat(post, "car_track")
    if "car_event" in post:
        car = car + flat(post, "car_event")
    o = design.obs[design.obs.event_idx.isin(test_events)].copy()
    o["pred"] = (car[:, o.car_idx] + skill[:, o.entry_idx]).mean(0)
    e = design.entries.set_index("entry_idx")
    o["driver"] = e.driver_id.loc[o.entry_idx].to_numpy()
    # baseline: driver's mean relative pace over their last 5 training events (recent form)
    tr = design.obs[design.obs.event_idx <= cutoff].copy()
    tr["r"] = tr.y - tr.groupby("session_idx").y.transform("mean")
    tr["driver"] = e.driver_id.loc[tr.entry_idx].to_numpy()
    per_ev = tr.groupby(["driver", "event_idx"]).r.mean().reset_index()
    recent = per_ev.sort_values("event_idx").groupby("driver").tail(5).groupby("driver").r.mean()
    # drivers without history take their team's recent form
    tr["team"] = design.cars.set_index("car_idx").team.loc[tr.car_idx].to_numpy()
    team_ev = tr.groupby(["team", "event_idx"]).r.mean().reset_index()
    team_recent = team_ev.sort_values("event_idx").groupby("team").tail(5).groupby("team").r.mean()
    o["team"] = design.cars.set_index("car_idx").team.loc[o.car_idx].to_numpy()
    o["form"] = o.driver.map(recent).fillna(o.team.map(team_recent)).fillna(0.0)
    # static two-way baseline: driver effect + latest team-season effect of that team
    c = design.cars.set_index("car_idx")
    ts_now = c.team_season_idx.loc[o.car_idx].to_numpy()
    last_ts = design.cars[design.cars.event_idx <= cutoff].groupby("team").team_season_idx.max()
    ts_use = np.where(np.isin(ts_now, dtrain.cars[dtrain.cars.event_idx <= cutoff].team_season_idx),
                      ts_now, c.team.loc[o.car_idx].map(last_ts).fillna(-1).astype(int))
    o["akm"] = d_eff.reindex(o.driver).fillna(d_eff.median()).to_numpy() + \
        ts_eff.reindex(ts_use).fillna(ts_eff.median()).to_numpy()
    rel = {}
    for col in ("y", "pred", "form", "akm"):
        o[col + "_c"] = o[col] - o.groupby("session_idx")[col].transform("mean")
    for col in ("pred", "form", "akm"):
        rel[f"rmse_{col}"] = _rmse(o.y_c, o[col + "_c"])
        rel[f"spearman_{col}"] = float(np.nanmean([
            spearmanr(g.y_c, g[col + "_c"])[0] for _, g in o.groupby("session_idx") if len(g) >= 8]))
    return {"name": name, "cutoff": int(cutoff), "test_season": int(test_season),
            "n_test_events": int(len(test_events)), "teammate": tm, "pace": rel,
            "pairs": pair_rows}


def evaluate_lfo() -> dict:
    """Every leave-future-out cutoff with a converged fit; each fit is checked against the
    design its job builds from the current data (jobs.lfo_design), so a stale fit fails."""
    names = {f.stem for f in FITS.glob("lfo_*.npz")}
    failed = sorted(f.name.removesuffix(".failed.json") for f in FITS.glob("lfo_*.failed.json"))
    rng = np.random.default_rng(0)
    results, convergence = [], {}
    for name in lfo_cutoffs(build_design(2010)):
        if name not in names:
            continue
        design, cutoff = lfo_design(name)
        r = evaluate_cutoff(design, name, cutoff, rng)
        attempts = load_meta(FITS / f"{name}.npz").get("attempts", [])
        convergence[name] = attempts[-1]["diagnostics"] if attempts else None
        if r:
            results.append(r)
    pairs = pd.DataFrame([row for r in results for row in r["pairs"]])
    tm = pd.DataFrame([{"name": r["name"], **r["teammate"]} for r in results])
    pace = pd.DataFrame([{"name": r["name"], **r["pace"]} for r in results])
    w = tm.n_sessions
    summary = {
        "n_cutoffs": len(results),
        "excluded_unconverged": failed,
        "convergence": convergence,
        "teammate_session": {
            **{k: float(np.sqrt(np.average(tm[k] ** 2, weights=w)) * SEC_PER_PCT)
               for k in ("rmse_pred", "rmse_akm", "rmse_naive", "rmse_zero")},
            "cov50": float(np.average(tm.cov50, weights=w)),
            "cov90": float(np.average(tm.cov90, weights=w)),
            "crps_s": float(np.average(tm.crps, weights=w) * SEC_PER_PCT),
        },
        "pace_session": {k: float(pace[k].mean() * (SEC_PER_PCT if k.startswith("rmse") else 1))
                         for k in pace.columns if k != "name"},
    }
    for label, sub in (("pairing_all", pairs), ("pairing_new", pairs[pairs.new_pair]),
                       ("pairing_continuing", pairs[~pairs.new_pair])):
        if len(sub):
            summary[label] = {
                "n": int(len(sub)),
                **{f"rmse_{c}": _rmse(sub.gap, sub[c]) * SEC_PER_PCT for c in ("pred", "akm", "naive")},
                "rmse_zero": _rmse(sub.gap, 0) * SEC_PER_PCT,
                "in90": float(sub.in90.mean()),
            }
    REPORTS.mkdir(parents=True, exist_ok=True)
    tm.to_csv(REPORTS / "lfo_teammate.csv", index=False)
    pace.to_csv(REPORTS / "lfo_pace.csv", index=False)
    pairs.to_csv(REPORTS / "lfo_pairings.csv", index=False)
    return summary


# --------------------------------------------------------------------------- synthetic recovery

SEED_METRICS = {  # summarised across the clean seeds
    "skill_cov90": ("skill", "cov90"), "car_cov90": ("car", "cov90"), "skill_corr": ("skill", "corr"),
    "car_corr": ("car", "corr"), "skill_rmse_s": ("skill", "rmse_s"),
    "grid_spearman": ("current_grid", "spearman"), "grid_cov90": ("current_grid", "cov90"),
    "grid_in_team_spearman": ("current_grid_in_team", "spearman"),
    "grid_in_team_cov90": ("current_grid_in_team", "cov90"),
    "teammate_diff_cov90": ("teammate_diff", "cov90"),
    "cross_team_cov90": ("cross_team_diff_current", "cov90"),
}


def synth_metrics(design: Design, post: dict, truth: dict) -> dict:
    """Recovery of the synthetic truth: every state, the current grid, teammate and
    between-team differences, and the car share of latent variation."""
    last = design.events.event_idx.max()
    res = {}
    for comp in ("skill", "car"):
        draws, true = flat(post, comp), truth[comp]
        m = draws.mean(0)
        res[comp] = {
            "corr": float(np.corrcoef(m, true)[0, 1]),
            "rmse_s": _rmse(m, true) * SEC_PER_PCT,
            "bias_s": float(np.mean(m - true)) * SEC_PER_PCT,
            "cov50": _coverage(draws, true, 50), "cov90": _coverage(draws, true, 90),
        }
    # current grid: rank recovery and coverage at the latest event
    rows = np.flatnonzero(design.entries.event_idx.to_numpy() == last)
    dr, tr = flat(post, "skill")[:, rows], truth["skill"][rows]
    dr = dr - dr.mean(1, keepdims=True)
    tr = tr - tr.mean()
    res["current_grid"] = {
        "spearman": float(spearmanr(dr.mean(0), tr)[0]),
        "rmse_s": _rmse(dr.mean(0), tr) * SEC_PER_PCT,
        "cov90": _coverage(dr, tr, 90),
    }
    if "compat" in post and "compat" in truth:
        ci = flat(post, "skill")[:, rows] + flat(post, "compat")[:, rows]
        ci = ci - ci.mean(1, keepdims=True)
        ti = truth["skill"][rows] + truth["compat"][rows]
        ti = ti - ti.mean()
        res["current_grid_in_team"] = {"spearman": float(spearmanr(ci.mean(0), ti)[0]),
                                       "rmse_s": _rmse(ci.mean(0), ti) * SEC_PER_PCT,
                                       "cov90": _coverage(ci, ti, 90)}
    # teammate differences (the part the design identifies directly)
    pe = design.entries.merge(design.entries, on=["event_idx", "team"])
    pe = pe[pe.driver_id_x < pe.driver_id_y]
    ia, ib = pe.entry_idx_x.to_numpy(), pe.entry_idx_y.to_numpy()
    dd = flat(post, "skill")[:, ia] - flat(post, "skill")[:, ib]
    td = truth["skill"][ia] - truth["skill"][ib]
    res["teammate_diff"] = {"corr": float(np.corrcoef(dd.mean(0), td)[0, 1]),
                            "cov90": _coverage(dd, td, 90)}
    # between-team driver comparisons (identified only through the network)
    cross = design.entries[design.entries.event_idx == last]
    cx = cross.merge(cross, how="cross")
    cx = cx[(cx.team_x != cx.team_y) & (cx.driver_id_x < cx.driver_id_y)]
    ia, ib = cx.entry_idx_x.to_numpy(), cx.entry_idx_y.to_numpy()
    dd = flat(post, "skill")[:, ia] - flat(post, "skill")[:, ib]
    td = truth["skill"][ia] - truth["skill"][ib]
    res["cross_team_diff_current"] = {"corr": float(np.corrcoef(dd.mean(0), td)[0, 1]),
                                      "cov90": _coverage(dd, td, 90)}
    # car vs driver variance share per season: truth vs estimate
    shares = []
    for season, g in design.events.groupby("season"):
        ev_idx = g.event_idx.to_numpy()
        cm = np.isin(design.cars.event_idx, ev_idx)
        em = np.isin(design.entries.event_idx, ev_idx)
        t_share = truth["car"][cm].var() / (truth["car"][cm].var() + truth["skill"][em].var())
        ec, es = flat(post, "car")[:, cm], flat(post, "skill")[:, em]
        e_share = np.mean(ec.var(1) / (ec.var(1) + es.var(1)))
        shares.append((season, t_share, e_share))
    s = np.array([x[1:] for x in shares])
    res["car_share"] = {"truth_mean": float(s[:, 0].mean()), "est_mean": float(s[:, 1].mean()),
                        "max_abs_err": float(np.abs(s[:, 0] - s[:, 1]).max())}
    return res


def evaluate_synth() -> dict:
    """Scenarios (one shared truth, clean + misspecified) and clean seeds (independent
    truths, noise and hyperparameters), with a summary across the seeds."""
    from .jobs import N_SYNTH_SEEDS
    from .simulate import SCENARIOS
    names = [f"synth_{sc}" for sc in SCENARIOS] + [f"synth_seed{k}" for k in range(1, N_SYNTH_SEEDS + 1)]
    names = [n for n in names if (FITS / f"{n}.npz").exists() and (FITS / f"{n}_truth.npz").exists()]
    design = build_design(2010, end_event=common_data_as_of([FITS / f"{n}.npz" for n in names]))
    source = load_meta(SYNTH_SOURCE) or {}
    out = {"scenarios": {}, "seeds": {}}
    for name in names:
        f = FITS / f"{name}.npz"
        meta = load_meta(f)
        if meta.get("source_fingerprint") != source.get("fingerprint"):
            print(f"note: {name} was made from an earlier synthetic source")
        # synthetic lap times differ from the real ones by design: match the states instead
        post, info = load(f, design, match="ids")
        res = {"divergences": info.get("divergences"),
               **synth_metrics(design, post, dict(np.load(FITS / f"{name}_truth.npz")))}
        if name.startswith("synth_seed"):
            res["draw"] = meta.get("draw")
            res["hyper"] = {k: meta["hyper"][k] for k in ("sd_level", "sd_compat", "sd_driver_form")
                            if k in meta.get("hyper", {})}
            out["seeds"][name[len("synth_"):]] = res
        else:
            out["scenarios"][name[len("synth_"):]] = res
    if out["seeds"]:
        vals = {m: [r[a][b] for r in out["seeds"].values() if a in r] for m, (a, b) in SEED_METRICS.items()}
        out["seeds_summary"] = {"n_seeds": len(out["seeds"]),
                                **{m: {"mean": float(np.mean(v)), "min": float(np.min(v)),
                                       "max": float(np.max(v))} for m, v in vals.items() if v}}
    return out


# --------------------------------------------------------------------------- sensitivity

def _compare_ratings(table: pd.DataFrame) -> dict:
    """Each variant column vs `main`: rank correlation, largest rating and rank shifts."""
    out = {}
    rank_main = table.main.rank(ascending=False)
    for name in table.columns.drop("main"):
        both = table[["main", name]].dropna()
        out[name] = {"spearman": float(spearmanr(both.main, both[name])[0]),
                     "max_abs_shift_s": float((both[name] - both.main).abs().max()),
                     "max_rank_shift": int((rank_main - table[name].rank(ascending=False)).abs().max())}
    return out


def _flag_unstable(table: pd.DataFrame) -> list[str]:
    # flag drivers whose rating moves by more than ~a quarter of a typical 90% interval
    # across variants, or whose rank moves 8+ places (a 0.05 s / 4-place rule, set before
    # the results, flagged every driver - it is smaller than the posterior uncertainty)
    spread = table.max(1) - table.min(1)
    ranks = table.rank(ascending=False)
    rank_range = ranks.max(1) - ranks.min(1)
    table["spread_s"], table["rank_range"] = spread, rank_range
    return table.index[(spread > 0.10) | (rank_range >= 8)].tolist()


def evaluate_sens() -> dict:
    """The current grid under each sensitivity variant, for portable skill (unprefixed
    keys) and pace in the current car (in_team_*; for sens_nocompat, which has no
    team-specific effect, its skill is its pace in the current car)."""
    names = [n for n in SENS if (FITS / f"{n}.npz").exists()]
    as_of = common_data_as_of([FITS / "main.npz", *(FITS / f"{n}.npz" for n in names)])
    main_design = build_design(2010, end_event=as_of)
    base = driver_leaderboard(main_design, load(FITS / "main.npz", main_design)[0]).set_index("driver_id")
    portable = pd.DataFrame({"main": base["median"] * SEC_PER_PCT})
    in_team = pd.DataFrame({"main": base["in_team_median"] * SEC_PER_PCT})
    for name in names:
        d = build_design(SENS[name].get("start", 2010), end_event=as_of)
        lb = driver_leaderboard(d, load(FITS / f"{name}.npz", d)[0]).set_index("driver_id")
        portable[name] = (lb["median"] * SEC_PER_PCT).reindex(portable.index)
        col = "in_team_median" if "in_team_median" in lb else "median"
        in_team[name] = (lb[col] * SEC_PER_PCT).reindex(in_team.index)
    out = _compare_ratings(portable)
    for name, r in _compare_ratings(in_team).items():
        out[name].update({f"in_team_{k}": v for k, v in r.items()})
    out["unstable_drivers"] = _flag_unstable(portable)
    out["unstable_drivers_in_team"] = _flag_unstable(in_team)
    REPORTS.mkdir(parents=True, exist_ok=True)
    portable.to_csv(REPORTS / "sensitivity_current_drivers.csv")
    in_team.to_csv(REPORTS / "sensitivity_current_drivers_in_team.csv")
    return out


# --------------------------------------------------------------------------- placebo

def evaluate_placebo() -> dict:
    """Team-specific effect vs a placebo effect for each half of a multi-season stint in
    the same team (the `placebo` job). A team-associated difference shows in sd_compat;
    generic drift within a team would show equally in sd_placebo. Neither says why the
    team-associated difference exists (car handling, support, role, adaptation, selection)."""
    path = FITS / "placebo.npz"
    post, info = load(path, build_design(2010, end_event=load_meta(path)["data_as_of"]))
    q = lambda k: [round(float(x), 4) for x in np.percentile(post[k], [5, 50, 95])]  # noqa: E731
    return {"description": "Team-specific effect (one per driver x team lineage) vs placebo (each half "
                           "of a multi-season stint in the same team). A team-associated difference "
                           "shows in sd_compat; generic career drift would show equally in sd_placebo.",
            "sd_compat_q05_q50_q95": q("sd_compat"), "sd_placebo_q05_q50_q95": q("sd_placebo"),
            "p_placebo_gt_compat": float((post["sd_placebo"] > post["sd_compat"]).mean()), "fit": info}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("what", choices=["lfo", "synth", "sens", "placebo", "all"])
    args = p.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    fns = {"lfo": evaluate_lfo, "synth": evaluate_synth, "sens": evaluate_sens, "placebo": evaluate_placebo}
    for what in (fns if args.what == "all" else [args.what]):
        res = fns[what]()
        (REPORTS / f"{what}_summary.json").write_text(json.dumps(res, indent=1))
        print(what, json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
