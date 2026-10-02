"""Forecasts of the next qualifying, published before it happens, and their scores.

    python -m f1rank.forecast next     # after export: forecast the season's next event
    python -m f1rank.forecast score    # score published forecasts whose event has happened

`next` forecasts the next event of the current season (from Jolpica's schedule) with the
published main fit (outputs/fits/main.npz, the fit behind outputs/ratings), and writes
outputs/forecasts/<event>_<fit id>.json. There is one forecast per event: a later run,
even after a refit, leaves the first one as published. It refuses to write within two
days of the race date, so a record exists only if it was made before that weekend's
qualifying; the git history of the scheduled refresh shows when each was committed.

The fit has no states for the next event, so they are the latest event's states plus
the model's one-race-ahead terms, each drawn once per posterior draw:
- driver: in-team pace at the latest event (portable skill + team-specific effect),
  plus one skill-walk step (Normal, sd_skill_race) and a one-weekend form draw
  (Normal, sd_driver_form). The experience and age trend moves by one event; that
  change is left out.
- teammate gap in a segment: the difference of the two drivers' paces plus each
  driver's lap noise from the fitted likelihood, at a new session's noise scale
  (sigma0 * exp(sigma_tau * u), u ~ Normal(0, 1)), as in evaluate.py and asof.py.
- car: the persistent car at the latest event, plus one within-season step
  (Student-t with CAR_STEP_DF, scale sd_car_race), a new event transient (Normal,
  sd_car_event) and the circuit term (team-season loading x the next circuit's factor,
  centred across teams; 0 for a circuit without a factor).
- lineup: the latest event's entrants. Scoring uses only pairs that were teammates at
  the event, and the order of the drivers who set times.
After a season's last race no forecast is made: the next season's lineups and cars are
not known from the data.

`score` compares each forecast with the event's qualifying, in the units and with the
pair rules of asof.py: teammate gaps per segment (both drivers with at least
MIN_TRAIN_ENTRIES earlier entries), their 90% interval coverage, the error of a zero
gap, and the rank correlation of predicted (car + driver) and actual pace in each
segment. It writes outputs/forecasts/scores.json and the directory's manifest.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from .artifacts import atomic_json, record, write_once

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "forecasts"
SEC = 0.9  # seconds per percent of a 90 s lap
MIN_DAYS_BEFORE_RACE = 2
MODEL = "quali-forecast-v1"


def next_event(schedule: list[dict], latest_event_id: str) -> dict | None:
    """The first race of the latest event's season after it, from a Jolpica season schedule."""
    season, rnd = (int(x) for x in latest_event_id.split("-"))
    later = sorted((r for r in schedule if int(r["season"]) == season and int(r["round"]) > rnd),
                   key=lambda r: int(r["round"]))
    if not later:
        return None
    r = later[0]
    return {"event_id": f"{season}-{int(r['round']):02d}", "race_name": r["raceName"], "date": r["date"],
            "circuit_id": r["Circuit"]["circuitId"]}


def _summary(draws: np.ndarray) -> list[dict]:
    """Medians and 90% intervals in seconds, with 90% rank ranges (1 = fastest)."""
    q = np.percentile(draws, [5, 50, 95], axis=0) * SEC
    ranks = (-draws).argsort(1).argsort(1) + 1
    lo, hi = np.percentile(ranks, [5, 95], axis=0)
    return [dict(median=round(float(q[1, i]), 4), q05=round(float(q[0, i]), 4), q95=round(float(q[2, i]), 4),
                 rank_lo=int(np.floor(lo[i])), rank_hi=int(np.ceil(hi[i]))) for i in range(draws.shape[1])]


def make_forecast(design, post: dict, target: dict, seed: int = 0) -> dict:
    """Forecast `target` (an event after the design's last event) from draws at that last event."""
    from .evaluate import draw_noise
    from .model import CAR_STEP_DF
    from .ratings import flat
    rng = np.random.default_rng(seed)
    last = design.events.iloc[-1]
    e, c = design.entries, design.cars
    e_rows = np.flatnonzero(e.event_idx.to_numpy() == last.event_idx)
    c_rows = np.flatnonzero(c.event_idx.to_numpy() == last.event_idx)
    col = lambda name: post[name].reshape(-1, 1)  # noqa: E731
    pace = (flat(post, "skill") + flat(post, "compat"))[:, e_rows]
    n = pace.shape[0]
    pace = (pace + rng.standard_normal(pace.shape) * col("sd_skill_race")
            + rng.standard_normal(pace.shape) * col("sd_driver_form"))
    ids = e.driver_id.to_numpy()[e_rows].astype(str)
    teams = e.team.to_numpy()[e_rows].astype(str)

    pairs = []
    for team in sorted(set(teams)):
        members = sorted(np.flatnonzero(teams == team), key=lambda i: ids[i])
        if len(members) != 2:
            continue
        i, j = members
        sigma = col("sigma0") * np.exp(col("sigma_tau") * rng.standard_normal((n, 1)))
        diff = pace[:, i] - pace[:, j]
        rep = diff + (draw_noise(post, sigma, rng) - draw_noise(post, sigma, rng))[:, 0]
        lo, hi = np.percentile(rep, [5, 95])
        pairs.append(dict(team=team, a=ids[i], b=ids[j], predicted=round(float(diff.mean()) * SEC, 4),
                          q05=round(float(lo) * SEC, 4), q95=round(float(hi) * SEC, 4)))

    car = flat(post, "car")[:, c_rows]
    car = (car + rng.standard_t(CAR_STEP_DF, car.shape) * col("sd_car_race")
           + rng.standard_normal(car.shape) * col("sd_car_event"))
    circuits = design.events.drop_duplicates("circuit_id").set_index("circuit_id").circuit_idx
    if "track_load" in post and target["circuit_id"] in circuits.index:
        factor = float(design.circuit_factor[int(circuits[target["circuit_id"]])])
        track = flat(post, "track_load")[:, c.team_season_idx.to_numpy()[c_rows]] * factor
        car = car + track - track.mean(1, keepdims=True)
    car = car - car.mean(1, keepdims=True)
    car_teams = c.team.to_numpy()[c_rows].astype(str)
    cars = [dict(team=t, name=str(name), **s)
            for t, name, s in zip(car_teams, c.constructor_name.to_numpy()[c_rows], _summary(car))]
    # Expected (car + driver) pace of every entrant, relative to the field, for the order.
    total = car[:, [list(car_teams).index(t) for t in teams]] + pace - pace.mean(1, keepdims=True)
    drivers = [dict(id=d, team=t, **s) for d, t, s in zip(ids, teams, _summary(total))]
    return dict(pairs=pairs, cars=sorted(cars, key=lambda r: -r["median"]),
                drivers=sorted(drivers, key=lambda r: -r["median"]))


def write_next(schedule: list[dict] | None = None, today: dt.date | None = None) -> Path | None:
    from .design import build_design
    design = build_design(2010)
    last = design.events.iloc[-1]
    if schedule is None:
        from .fetch import BASE, _get
        schedule = _get(f"{BASE}/{int(last.season)}.json")["RaceTable"]["Races"]
    target = next_event(schedule, str(last.event_id))
    if target is None:
        print(f"no forecast: {last.event_id} is the last race of {last.season}")
        return None
    published = sorted(OUT.glob(f"{target['event_id']}_*.json"))
    if published:
        # One forecast per event: the first one published stands, even after a refit.
        print(f"forecast for {target['event_id']} already published, left unchanged: {published[0].name}")
        return published[0]
    today = today or dt.datetime.now(dt.timezone.utc).date()
    if (dt.date.fromisoformat(target["date"]) - today).days < MIN_DAYS_BEFORE_RACE:
        print(f"no forecast: {target['event_id']} is on {target['date']}, too close to publish beforehand")
        return None
    from .artifacts import diagnostics, require_convergence
    from .fit import FITS, load, load_meta
    path = FITS / "main.npz"
    post, info = load(path, design)  # refuses a fit made on other data
    require_convergence(diagnostics(post, info["divergences"]))
    meta = load_meta(path)
    out = OUT / f"{target['event_id']}_{meta['created_utc']}.json"
    rec = dict(model=MODEL, event=target,
               trained_through={k: str(last[k]) for k in ("event_id", "race_name", "date")},
               lineup_from=str(last.event_id),
               fit=dict(fit_id=meta["created_utc"], fingerprint=meta["fingerprint"]),
               created_utc=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               **make_forecast(design, post, target))
    write_once(out, rec)
    print(f"forecast written: {out.name}")
    return out


def score_record(design, rec: dict) -> dict | None:
    """Scores of one forecast against its event's qualifying, or None before it happened."""
    from .evaluate import MIN_TRAIN_ENTRIES, session_pairs
    ev = design.events.set_index("event_id")
    event_id = rec["event"]["event_id"]
    if event_id not in ev.index or not (design.obs.event_idx == ev.event_idx[event_id]).any():
        return None
    k = int(ev.event_idx[event_id])
    segments = design.sessions.set_index("session_idx").segment
    n_before = design.entries[design.entries.event_idx < k].groupby("driver_id").size()
    forecast = {(r["a"], r["b"]): r for r in rec["pairs"]}
    p = session_pairs(design)
    rows = []
    for r in p[p.event_idx == k].itertuples():
        f = forecast.get((r.driver_a, r.driver_b))
        if f is None or min(n_before.get(r.driver_a, 0), n_before.get(r.driver_b, 0)) < MIN_TRAIN_ENTRIES:
            continue
        rows.append(dict(segment=segments[r.session_idx], a=r.driver_a, b=r.driver_b,
                         predicted=f["predicted"], q05=f["q05"], q95=f["q95"], observed=round(float(r.gap) * SEC, 4)))
    expected = {d["id"]: d["median"] for d in rec["drivers"]}
    e = design.entries.set_index("entry_idx").driver_id
    order = []
    o = design.obs[design.obs.event_idx == k]
    for sid, g in o.groupby("session_idx"):
        pred = e.loc[g.entry_idx].map(expected)
        known = pred.notna().to_numpy()
        if known.sum() >= 8:
            order.append(dict(segment=segments[sid], n=int(known.sum()),
                              spearman=round(float(spearmanr(g.y.to_numpy()[known], pred.to_numpy()[known])[0]), 4)))
    obs = np.array([r["observed"] for r in rows])
    pred = np.array([r["predicted"] for r in rows])
    rmse = lambda x: round(float(np.sqrt(np.mean((obs - x) ** 2))), 4) if rows else None  # noqa: E731
    return dict(event=rec["event"], trained_through=rec["trained_through"]["event_id"],
                created_utc=rec["created_utc"], n_pairs=len(rows), rmse=rmse(pred), rmse_zero=rmse(0.),
                coverage90=round(float(np.mean([r["q05"] <= r["observed"] <= r["q95"] for r in rows])), 4)
                if rows else None,
                order_spearman=round(float(np.mean([s["spearman"] for s in order])), 4) if order else None,
                pairs=rows, order=order)


def score() -> dict:
    from .design import build_design
    design = build_design(2010)
    files = sorted(p for p in OUT.glob("*_*.json"))
    events = []
    for path in files:
        s = score_record(design, json.loads(path.read_text()))
        if s is not None:
            events.append(dict(file=path.name, **s))
    scored = [s for s in events if s["n_pairs"]]
    n = sum(s["n_pairs"] for s in scored)
    pooled = dict(n_events=len(scored), n_pairs=n,
                  rmse=round(float(np.sqrt(sum(s["rmse"] ** 2 * s["n_pairs"] for s in scored) / n)), 4),
                  rmse_zero=round(float(np.sqrt(sum(s["rmse_zero"] ** 2 * s["n_pairs"] for s in scored) / n)), 4),
                  coverage90=round(sum(s["coverage90"] * s["n_pairs"] for s in scored) / n, 4)) if n else None
    summary = dict(model=MODEL, events=events, pooled=pooled)
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "scores.json", summary)
    record(OUT, [OUT / "scores.json", *files], model=MODEL,
           inputs=[ROOT / "f1rank" / "forecast.py", ROOT / "data" / "processed" / "quali_times.parquet",
                   ROOT / "data" / "processed" / "entries.parquet"])
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["next", "score"])
    args = p.parse_args()
    if args.command == "next":
        write_next()
    else:
        print(json.dumps(score()["pooled"], indent=1))


if __name__ == "__main__":
    main()
