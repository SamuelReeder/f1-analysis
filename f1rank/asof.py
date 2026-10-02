"""Qualifying ratings as they stood after each race, and forecasts of the next qualifying.

    python -m f1rank.asof fit 2026-07           # one event (data through the event before it)
    python -m f1rank.asof fit --season 2026     # every event of a season without a record
    python -m f1rank.asof fit --latest          # the latest event, as the scheduled refresh does
    python -m f1rank.asof summary               # outputs/asof/summary.json and its manifest

The fit for event k uses the design through k with the likelihood cut after the event
before k (Design.with_cutoff), exactly like the leave-future-out validation fits:

- its states at k-1 are the ratings that the data available after k-1 support (the
  revised history in outputs/ratings/ also uses every later race);
- its predictive draws at k forecast k's qualifying, which the fit never saw. Teammate
  gaps are forecast as in evaluate.py (driver effect in the team, one form draw per
  driver and event, fitted session noise), and the predicted order of the field
  (car + circuit + driver) is compared with the real order in each segment.

Each fit writes a write-once record, outputs/asof/<event>_<fit id>.json, after event k:
it needs k's entry list and lap times to score the forecast. The fit id is its creation
time, so records computed long after their event (the 2026 rounds before the scheduled
refresh began) are identifiable; none of them were published before the event. Records
are never rewritten when past data are revised; the summary uses the newest record per
event. Fits that fail the publication convergence checks write no record.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .artifacts import atomic_json, record, write_once

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "asof"
START = 2010
# Cheaper than the main fit (1500/1500): these fits only need forecasts and one event's states.
SETTINGS = dict(warmup=1000, samples=1000, chains=4, target_accept=0.9)
# A fit that fails the convergence checks is repeated once with the main fit's chain
# lengths and a new seed; if that also fails, the event gets no record.
RETRY = dict(SETTINGS, warmup=1500, samples=1500, seed=1)
SEC = 0.9  # seconds per percent of a 90 s lap


def _estimate(draws: np.ndarray) -> list[dict]:
    """Summaries in seconds, relative to the field, with 90% rank ranges."""
    q = np.percentile(draws, [5, 50, 95], axis=0) * SEC
    ranks = (-draws).argsort(1).argsort(1) + 1
    lo, hi = np.percentile(ranks, [5, 95], axis=0)
    return [dict(q05=round(float(q[0, i]), 4), median=round(float(q[1, i]), 4), q95=round(float(q[2, i]), 4),
                 rank_lo=int(np.floor(lo[i])), rank_hi=int(np.ceil(hi[i]))) for i in range(draws.shape[1])]


def ratings_at(design, post, event_idx: int) -> dict:
    from .ratings import flat
    e, c = design.entries, design.cars
    rows = np.flatnonzero(e.event_idx.to_numpy() == event_idx)
    skill = flat(post, "skill")[:, rows]
    skill = skill - skill.mean(1, keepdims=True)
    compat = flat(post, "compat")[:, rows]
    in_team = skill + compat - compat.mean(1, keepdims=True)
    names = design.drivers.set_index("driver_id").name
    drivers = [dict(id=d, name=names[d], team=t, headline=h, portable=p)
               for d, t, h, p in zip(e.driver_id.to_numpy()[rows], e.constructor_name.to_numpy()[rows],
                                     _estimate(in_team), _estimate(skill))]
    crow = np.flatnonzero(c.event_idx.to_numpy() == event_idx)
    car = flat(post, "car")[:, crow]
    cars = [dict(id=t, name=n, pace=p) for t, n, p in
            zip(c.team.to_numpy()[crow], c.constructor_name.to_numpy()[crow],
                _estimate(car - car.mean(1, keepdims=True)))]
    return dict(drivers=sorted(drivers, key=lambda r: -r["headline"]["median"]),
                cars=sorted(cars, key=lambda r: -r["pace"]["median"]))


def forecast(design, post, event_idx: int, cutoff: int, seed: int = 0) -> dict:
    """Out-of-sample forecasts of event `event_idx` from a fit trained through `cutoff`."""
    from .evaluate import MIN_TRAIN_ENTRIES, draw_noise, session_pairs, teammate_deltas
    from .ratings import flat
    rng = np.random.default_rng(seed)
    skill = flat(post, "skill") + flat(post, "compat")
    skill = skill + rng.standard_normal(skill.shape) * post["sd_driver_form"].reshape(-1, 1)
    sigma = post["sigma0"].reshape(-1, 1) * np.exp(post["sigma_tau"].reshape(-1, 1) * flat(post, "sigma_u"))
    segments = design.sessions.set_index("session_idx").segment
    n_train = design.entries[design.entries.event_idx <= cutoff].groupby("driver_id").size()

    p = session_pairs(design)
    p = p[p.event_idx == event_idx].reset_index(drop=True)
    diff = skill[:, p.entry_a] - skill[:, p.entry_b]
    sig = sigma[:, p.session_idx]
    rep = diff + draw_noise(post, sig, rng) - draw_noise(post, sig, rng)
    lo, hi = np.percentile(rep, [5, 95], axis=0)
    delta, pair = teammate_deltas(design, cutoff)
    team = design.entries.set_index("entry_idx").team
    pairs = []
    for i, r in p.iterrows():
        key = (r.driver_a, r.driver_b)
        naive = pair.get(key, delta.get(r.driver_a, np.nan) - delta.get(r.driver_b, np.nan))
        pairs.append(dict(segment=segments[r.session_idx], team=team[r.entry_a], a=r.driver_a, b=r.driver_b,
                          predicted=round(float(diff[:, i].mean()) * SEC, 4),
                          q05=round(float(lo[i]) * SEC, 4), q95=round(float(hi[i]) * SEC, 4),
                          observed=round(float(r.gap) * SEC, 4),
                          naive=None if pd.isna(naive) else round(float(naive) * SEC, 4),
                          established=bool(n_train.get(r.driver_a, 0) >= MIN_TRAIN_ENTRIES
                                           and n_train.get(r.driver_b, 0) >= MIN_TRAIN_ENTRIES)))

    # predicted order of the field: car (with circuit and event terms) plus driver, per segment
    car = flat(post, "car") + flat(post, "car_track") + flat(post, "car_event")
    o = design.obs[design.obs.event_idx == event_idx].copy()
    o["pred"] = (car[:, o.car_idx] + skill[:, o.entry_idx]).mean(0)
    order = []
    for sid, g in o.groupby("session_idx"):
        if len(g) >= 8:
            order.append(dict(segment=segments[sid], n=int(len(g)),
                              spearman=round(float(spearmanr(g.y, g.pred)[0]), 4)))

    scored = [r for r in pairs if r["established"]]
    obs = np.array([r["observed"] for r in scored])
    pred = np.array([r["predicted"] for r in scored])
    naive = np.array([r["naive"] if r["naive"] is not None else 0. for r in scored])
    inside = [r["q05"] <= r["observed"] <= r["q95"] for r in scored]
    rmse = lambda x: round(float(np.sqrt(np.mean((obs - x) ** 2))), 4) if len(scored) else None  # noqa: E731
    summary = dict(n_pairs=len(scored), rmse=rmse(pred), rmse_naive=rmse(naive), rmse_zero=rmse(0.),
                   coverage90=round(float(np.mean(inside)), 4) if scored else None,
                   order_spearman=round(float(np.mean([s["spearman"] for s in order])), 4) if order else None)
    return dict(pairs=pairs, order=order, summary=summary)


def event_design(event_id: str):
    from .design import build_design
    design = build_design(START, end_event=event_id)
    ev = design.events.set_index("event_id").event_idx
    k = int(ev[event_id])
    if k == 0:
        raise ValueError(f"{event_id}: no earlier event to train on")
    return design.with_cutoff(k - 1), k


def fit_event(event_id: str, attempts: tuple[dict, ...] = (SETTINGS, RETRY)) -> Path | None:
    from .artifacts import diagnostics
    import gc
    import jax
    from .fit import _git_commit, _git_dirty, fit, fingerprint
    design, k = event_design(event_id)
    tried = []
    for settings in attempts:
        post, info = fit(design, progress=False, **settings)
        checked = diagnostics(post, info["divergences"])
        tried.append({"settings": settings, **info, **checked})
        print(f"{event_id}: attempt {len(tried)} {checked}", flush=True)
        if checked["converged"]:
            break
        # release the failed draws and compiled programs before the longer retry
        post = None
        gc.collect()
        jax.clear_caches()
    else:
        print(f"{event_id}: no record, no attempt passed the convergence checks", flush=True)
        return None
    events = design.events.set_index("event_idx")
    fit_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    rec = dict(
        event={k_: str(events.loc[k, k_]) for k_ in ("event_id", "race_name", "date")},
        trained_through={k_: str(events.loc[k - 1, k_]) for k_ in ("event_id", "race_name", "date")},
        fit=dict(fit_id=fit_id, fingerprint=fingerprint(design), settings=settings,
                 git_commit=_git_commit(), git_dirty=_git_dirty(), diagnostics={**info, **checked},
                 **({"failed_attempts": tried[:-1]} if len(tried) > 1 else {})),
        ratings=ratings_at(design, post, k - 1),
        forecast=forecast(design, post, k, k - 1))
    path = OUT / f"{event_id}_{fit_id}.json"
    write_once(path, rec)
    print(f"{event_id}: {path.name} {rec['forecast']['summary']}", flush=True)
    return path


def records() -> dict[str, dict]:
    """Newest record per event."""
    out = {}
    for path in sorted(OUT.glob("*_*.json")):
        if path.name in ("summary.json",):
            continue
        rec = json.loads(path.read_text())
        out[rec["event"]["event_id"]] = rec | {"file": path.name}
    return out


def summarise() -> dict:
    recs = records()
    rows = [dict(event=r["event"], trained_through=r["trained_through"]["event_id"],
                 fit_id=r["fit"]["fit_id"], file=r["file"],
                 **r["forecast"]["summary"]) for r in recs.values()]
    scored = [r for r in rows if r["n_pairs"]]
    n = sum(r["n_pairs"] for r in scored)
    pooled = lambda key: round(float(np.sqrt(sum(r[key] ** 2 * r["n_pairs"] for r in scored) / n)), 4)  # noqa: E731
    summary = dict(events=rows, pooled=dict(
        n_events=len(scored), n_pairs=n, rmse=pooled("rmse"), rmse_naive=pooled("rmse_naive"),
        rmse_zero=pooled("rmse_zero"),
        coverage90=round(sum(r["coverage90"] * r["n_pairs"] for r in scored) / n, 4),
        order_spearman=round(float(np.mean([r["order_spearman"] for r in scored if r["order_spearman"] is not None])), 4),
    ) if n else None)
    atomic_json(OUT / "summary.json", summary)
    files = [OUT / r["file"] for r in recs.values()]
    record(OUT, [OUT / "summary.json", *files], model="quali-asof-v1",
           inputs=[ROOT / "f1rank" / "asof.py"])
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["fit", "summary"])
    p.add_argument("event", nargs="?")
    p.add_argument("--season", type=int)
    p.add_argument("--latest", action="store_true")
    args = p.parse_args()
    if args.command == "summary":
        print(json.dumps(summarise()["pooled"], indent=1))
        return
    events = pd.read_parquet(ROOT / "data" / "processed" / "events.parquet")
    if args.latest:
        targets = [events.event_id.max()]
    elif args.season:
        done = {e for e in records()}
        targets = [e for e in events[events.season == args.season].event_id if e not in done]
    elif args.event:
        targets = [args.event]
    else:
        p.error("give an event, --season or --latest")
    for event_id in targets:
        if args.latest and event_id in records():
            print(f"{event_id}: record exists")
            continue
        fit_event(event_id)


if __name__ == "__main__":
    main()
