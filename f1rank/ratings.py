"""Turn posterior draws into rating tables.

All ratings are relative to the field at the event (the average driver and the
average car entered), positive = faster. Model units are percent of lap time;
tables also give seconds per 90-second lap.
"""

import numpy as np
import pandas as pd

from .design import Design

SEC_PER_PCT = 0.9  # 1% of a 90 s lap
QUANTILES = {"q05": 5, "q25": 25, "median": 50, "q75": 75, "q95": 95}


def flat(post: dict, key: str) -> np.ndarray:
    x = post[key]
    return x.reshape(-1, *x.shape[2:])


def summarise(draws: np.ndarray, prefix: str = "") -> pd.DataFrame:
    q = np.percentile(draws, list(QUANTILES.values()), axis=0)
    return pd.DataFrame({prefix + k: q[i] for i, k in enumerate(QUANTILES)})


def driver_series(design: Design, post: dict) -> pd.DataFrame:
    """One row per driver per event: portable skill relative to the field at that
    event, and (if modelled) the effect in that team including car compatibility."""
    e = design.entries
    out = e[["driver_id", "event_id", "event_idx", "season", "team", "constructor_name"]]
    out = out.reset_index(drop=True).join(summarise(flat(post, "skill")))
    if "compat" in post:
        out = out.join(summarise(flat(post, "skill") + flat(post, "compat"), "in_team_"))
    return out.merge(design.drivers[["driver_id", "name", "code"]], on="driver_id")


def car_series(design: Design, post: dict) -> pd.DataFrame:
    """One row per team per event: track-neutral car pace and pace at that circuit."""
    c = design.cars
    base = flat(post, "car")
    at_track = base + flat(post, "car_track") if "car_track" in post else base
    out = c[["team", "constructor_name", "event_idx", "season"]].reset_index(drop=True)
    out = out.join(summarise(base)).join(summarise(at_track, "at_circuit_"))
    return out.merge(design.events[["event_idx", "event_id", "race_name"]], on="event_idx")


def _evidence(design: Design, driver_ids, upto_event: int) -> pd.DataFrame:
    e = design.entries[design.entries.event_idx <= upto_event]
    mates = e.merge(e, on=["event_idx", "team"])
    mates = mates[mates.driver_id_x != mates.driver_id_y]
    return pd.DataFrame({
        "events": e.groupby("driver_id").size(),
        "seasons": e.groupby("driver_id").season.nunique(),
        "teammates": mates.groupby("driver_id_x").driver_id_y.nunique(),
    }).reindex(driver_ids).fillna(0).astype(int)


def rank_summary(draws: np.ndarray) -> pd.DataFrame:
    """Rank distribution (1 = fastest) from joint draws of shape (S, n)."""
    ranks = (-draws).argsort(axis=1).argsort(axis=1) + 1
    lo, med, hi = np.percentile(ranks, [5, 50, 95], axis=0)
    return pd.DataFrame({
        "rank_median": med, "rank_lo": lo.astype(int), "rank_hi": hi.astype(int),
        "p_fastest": (ranks == 1).mean(0), "p_top3": (ranks <= 3).mean(0),
    })


def driver_leaderboard(design: Design, post: dict, event_idx: int | None = None) -> pd.DataFrame:
    """Portable skill (what the driver would bring to any car) at an event. With the
    compatibility term, compat_* gives the driver's fit with their current team."""
    event_idx = design.events.event_idx.max() if event_idx is None else event_idx
    e = design.entries
    rows = np.flatnonzero(e.event_idx.to_numpy() == event_idx)
    draws = flat(post, "skill")[:, rows]
    draws = draws - draws.mean(axis=1, keepdims=True)  # relative to this field
    ids = e.driver_id.to_numpy()[rows]
    out = pd.DataFrame({"driver_id": ids, "team": e.constructor_name.to_numpy()[rows]})
    out = out.join(summarise(draws)).join(rank_summary(draws))
    if "compat" in post:
        c = flat(post, "compat")[:, rows]
        in_team = draws + c - c.mean(axis=1, keepdims=True)
        out = out.join(summarise(c, "compat_")).join(summarise(in_team, "in_team_"))
        out["in_team_rank"] = pd.Series(np.median(in_team, 0)).rank(ascending=False,
                                                                    method="first").astype(int)
    out = out.merge(design.drivers[["driver_id", "name"]], on="driver_id")
    out = out.merge(_evidence(design, ids, event_idx), left_on="driver_id", right_index=True)
    return out.sort_values("median", ascending=False, ignore_index=True)


def car_leaderboard(design: Design, post: dict, event_idx: int | None = None) -> pd.DataFrame:
    event_idx = design.events.event_idx.max() if event_idx is None else event_idx
    c = design.cars
    rows = np.flatnonzero(c.event_idx.to_numpy() == event_idx)
    draws = flat(post, "car")[:, rows]
    draws = draws - draws.mean(axis=1, keepdims=True)
    gap = draws - draws.max(axis=1, keepdims=True)  # to the fastest car in each draw
    out = pd.DataFrame({"team": c.team.to_numpy()[rows],
                        "constructor": c.constructor_name.to_numpy()[rows]})
    out = out.join(summarise(draws)).join(summarise(gap, "gap_")).join(rank_summary(draws))
    if "track_load" in post:
        ts = c.team_season_idx.to_numpy()[rows]
        out = out.join(summarise(flat(post, "track_load")[:, ts], "track_"))
    return out.sort_values("median", ascending=False, ignore_index=True)


def fmt_seconds(pct: float) -> str:
    return f"{pct * SEC_PER_PCT:+.3f}s"
