"""Build model inputs from the processed tables.

Pace is measured per qualifying segment (Q1/Q2/Q3 of one event) as
    y = -100 * log(lap_time / segment_median)
so y is in percent of lap time and positive means faster. A segment-level
intercept in the model absorbs track evolution and the changing composition of
Q2/Q3, so only differences between cars in the same segment carry information.
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .lineage import REGULATION_RESETS

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"

MAX_GAP_PCT = 5.0       # laps this far off the segment median are not representative
MIN_SEGMENT_TIMES = 4   # a segment needs this many valid laps to compare cars
AGE_DECLINE_FROM = 32.0  # age after which a linear decline term applies
RACES_PER_SEASON_PRE_DATA = 17  # to estimate experience for debuts before the data
SIGN_ANCHOR = ("spa", "monaco")  # circuit factor oriented so the first is above the second
MIN_FACTOR_CIRCUITS = 3  # circuits with enough history needed to estimate the circuit factor


@dataclass
class Design:
    start_season: int
    events: pd.DataFrame     # event_idx order; includes season, circuit
    sessions: pd.DataFrame   # one row per (event, segment)
    entries: pd.DataFrame    # driver states, sorted by driver then event
    cars: pd.DataFrame       # car states, sorted by team then event
    obs: pd.DataFrame        # one row per lap time used
    drivers: pd.DataFrame    # driver_idx order
    teams: list[str]
    train: np.ndarray = field(default=None)  # bool per obs: included in the likelihood
    circuit_factor: np.ndarray = field(default=None)  # per circuit, from training data

    def arrays(self) -> dict[str, np.ndarray]:
        e, c, o, s = self.entries, self.cars, self.obs, self.sessions
        return {
            "n_drivers": len(self.drivers), "n_entries": len(e), "n_cars": len(c),
            "n_sessions": len(s), "n_events": len(self.events),
            "n_circuits": int(self.events.circuit_idx.max()) + 1,
            "n_team_seasons": int(c.team_season_idx.max()) + 1,
            "entry_event": e.event_idx.to_numpy(),
            "car_event": c.event_idx.to_numpy(),
            "car_team_season": c.team_season_idx.to_numpy(),
            "car_circuit": self.events.circuit_idx.to_numpy()[c.event_idx.to_numpy()],
            "circuit_factor": self.circuit_factor,
            "entry_driver": e.driver_idx.to_numpy(),
            "entry_stint": e.stint_idx.to_numpy(),
            "entry_placebo": e.placebo_idx.to_numpy(),
            "n_placebo": int(e.placebo_idx.max()) + 1,
            "n_stints": int(e.stint_idx.max()) + 1,
            "entry_first": e.first.to_numpy(),
            "entry_same_season": e.same_season.to_numpy(),
            "entry_event_gap": e.event_gap.to_numpy().astype(float),
            "entry_season_gap": e.season_gap.to_numpy().astype(float),
            "entry_experience": e.experience.to_numpy().astype(float),
            "entry_age_over": np.maximum(0.0, e.age.to_numpy() - AGE_DECLINE_FROM),
            "car_first": c.first.to_numpy(),
            "car_season_start": c.season_start.to_numpy(),
            "car_reset": c.reset.to_numpy(),
            "obs_session": o.session_idx.to_numpy(),
            "obs_entry": o.entry_idx.to_numpy(),
            "obs_car": o.car_idx.to_numpy(),
            "obs_car_segment": o.car_segment_idx.to_numpy(),
            "n_car_segments": int(o.car_segment_idx.max()) + 1,
            "car_segment_session": o.groupby("car_segment_idx").session_idx.first().to_numpy(),
            "y": o.y.to_numpy(),
            "train": self.train if self.train is not None else np.ones(len(o), bool),
        }

    def with_cutoff(self, last_train_event_idx: int) -> "Design":
        """Copy whose likelihood only uses events up to and including the cutoff."""
        out = Design(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        out.train = (self.obs.event_idx <= last_train_event_idx).to_numpy()
        out.circuit_factor = circuit_factors(out)
        return out


def circuit_factors(design: "Design", ridge: float = 5.0, iters: int = 200,
                    min_events: int = 3, spread_quantile: float = 0.85,
                    clip: float = 1.0) -> np.ndarray:
    """1-D circuit characteristic from a rank-1 fit of car pace deviations.

    Each team's pace at an event, relative to the segment mean, is detrended
    within its team-season (mean + linear trend). What remains is modelled as
    loading[team-season] * factor[circuit]. Only training observations are
    used, so forecasts never see the circuit factor of the future. High-spread
    segments (mostly wet) are excluded and deviations are clipped so a few
    chaotic sessions cannot define the axis. Circuits with fewer than
    `min_events` training events get 0 (no track-specific adjustment); with fewer
    than MIN_FACTOR_CIRCUITS such circuits (a short history), every circuit gets 0.
    """
    o = design.obs if design.train is None else design.obs.loc[design.train]
    o = o.assign(r=o.y - o.groupby("session_idx").y.transform("median"))
    spread = o.groupby("session_idx").r.transform(lambda r: np.median(np.abs(r)))
    o = o[spread <= spread.quantile(spread_quantile)]
    c = design.cars.set_index("car_idx")
    pace = o.groupby("car_idx").r.mean().rename("r").to_frame().join(c)
    pace["circuit"] = design.events.circuit_idx.to_numpy()[pace.event_idx]
    # detrend within team-season
    def detrend(g):
        x = g.event_idx - g.event_idx.mean()
        slope = (x * g.r).sum() / max((x * x).sum(), 1e-9)
        return g.r - g.r.mean() - slope * x
    pace["dev"] = pace.groupby("team_season_idx", group_keys=False).apply(detrend)

    n_ts, n_c = int(design.cars.team_season_idx.max()) + 1, int(design.events.circuit_idx.max()) + 1
    events_per_circuit = pace.groupby("circuit").event_idx.nunique().reindex(range(n_c), fill_value=0)
    seen = events_per_circuit.to_numpy() >= min_events
    if seen.sum() < MIN_FACTOR_CIRCUITS:
        return np.zeros(n_c)  # too little history to define an axis: no track adjustment
    pace = pace[seen[pace.circuit]]
    ts, circ = pace.team_season_idx.to_numpy(), pace.circuit.to_numpy()
    dev = pace.dev.clip(-clip, clip).to_numpy()
    rng = np.random.default_rng(0)
    f = rng.standard_normal(n_c)
    for _ in range(iters):
        lam = np.bincount(ts, dev * f[circ], n_ts) / (np.bincount(ts, f[circ] ** 2, n_ts) + ridge)
        f = np.bincount(circ, dev * lam[ts], n_c) / (np.bincount(circ, lam[ts] ** 2, n_c) + ridge)
    f[~seen] = 0.0
    scale = f[seen].std()
    if not np.isfinite(scale) or scale < 1e-9:
        return np.zeros(n_c)
    f /= scale
    idx = design.events.groupby("circuit_id").circuit_idx.first()
    hi, lo = (idx.get(c) for c in SIGN_ANCHOR)
    if hi is not None and lo is not None and f[hi] < f[lo]:
        f = -f
    return f


def load_tables() -> dict[str, pd.DataFrame]:
    return {p.stem: pd.read_parquet(p) for p in PROCESSED.glob("*.parquet")}


def build_design(start_season: int = 2010, end_event: str | None = None,
                 tables: dict[str, pd.DataFrame] | None = None) -> Design:
    t = tables or load_tables()
    events = t["events"].query("season >= @start_season").copy()
    if end_event:
        events = events[events.event_id <= end_event]
    events = events.sort_values("event_id", ignore_index=True)
    events["event_idx"] = np.arange(len(events))
    circuits = sorted(events.circuit_id.unique())
    events["circuit_idx"] = events.circuit_id.map({c: i for i, c in enumerate(circuits)})
    ev = events.set_index("event_id")

    drivers = t["drivers"].set_index("driver_id")

    # ---- experience counts use every entry since 2006, before any filtering
    all_entries = t["entries"].merge(t["events"][["event_id", "season", "date"]], on="event_id")
    all_entries = all_entries.sort_values(["driver_id", "event_id"])
    all_entries["n_prior"] = all_entries.groupby("driver_id").cumcount()
    first_data_season = t["events"].season.min()
    pre = drivers.debut_season.clip(upper=first_data_season)
    pre_data = (first_data_season - pre) * RACES_PER_SEASON_PRE_DATA
    all_entries["experience"] = all_entries.n_prior + all_entries.driver_id.map(pre_data)

    # ---- lap times -> pace, with outlier and thin-segment filtering
    q = t["quali_times"][t["quali_times"].event_id.isin(ev.index)].copy()
    q["median"] = q.groupby(["event_id", "segment"]).time_s.transform("median")
    q["y"] = -100 * np.log(q.time_s / q["median"])
    q = q[q.y >= -MAX_GAP_PCT]
    q = q[q.groupby(["event_id", "segment"]).y.transform("size") >= MIN_SEGMENT_TIMES]

    # ---- entries (driver states): only driver-events with at least one valid lap
    ent = all_entries.merge(q[["event_id", "driver_id"]].drop_duplicates(),
                            on=["event_id", "driver_id"])
    ent["event_idx"] = ent.event_id.map(ev.event_idx)
    ent = ent.sort_values(["driver_id", "event_idx"], ignore_index=True)
    driver_ids = sorted(ent.driver_id.unique())
    ent["driver_idx"] = ent.driver_id.map({d: i for i, d in enumerate(driver_ids)})
    prev = ent.groupby("driver_id")[["event_idx", "season"]].shift()
    ent["first"] = prev.event_idx.isna()
    ent["same_season"] = ~ent["first"] & (prev.season == ent.season)
    ent["event_gap"] = (ent.event_idx - prev.event_idx).fillna(0).astype(int)
    ent["season_gap"] = (ent.season - prev.season).fillna(0).astype(int)
    dob = pd.to_datetime(ent.driver_id.map(drivers.dob))
    ent["age"] = (pd.to_datetime(ent.date) - dob).dt.days / 365.25
    ent["entry_idx"] = np.arange(len(ent))
    ent["stint_idx"] = ent.groupby(["driver_id", "team"], sort=True).ngroup()
    # placebo split (diagnostic): stints spanning 2+ seasons are cut in two at the
    # middle season boundary; each half can get its own effect
    seasons = ent.groupby("stint_idx").season.agg(["min", "max"])
    cut = ((seasons["min"] + seasons["max"] + 1) // 2).where(seasons["max"] > seasons["min"])
    ent["placebo_half"] = (ent.season >= ent.stint_idx.map(cut)).astype(int)
    split = ent.stint_idx.map(cut).notna()
    ent["placebo_idx"] = np.where(split, ent.groupby(["stint_idx", "placebo_half"]).ngroup(), -1)

    # ---- car states: one per team per event with any valid lap
    cars = ent[["team", "event_idx", "season"]].drop_duplicates()
    cars = cars.sort_values(["team", "event_idx"], ignore_index=True)
    teams = sorted(cars.team.unique())
    prev_c = cars.groupby("team")[["season"]].shift()
    cars["first"] = prev_c.season.isna()
    cars["season_start"] = ~cars["first"] & (prev_c.season != cars.season)
    cars["reset"] = cars.season_start & cars.season.isin(REGULATION_RESETS)
    cars["car_idx"] = np.arange(len(cars))
    cars["team_season_idx"] = cars.groupby(["team", "season"], sort=True).ngroup()
    names = t["entries"].merge(events[["event_id", "event_idx"]], on="event_id")
    names = names.groupby(["team", "event_idx"]).constructor_name.first()
    cars["constructor_name"] = [names[(tm, i)] for tm, i in zip(cars.team, cars.event_idx)]

    # ---- sessions and observations
    q["event_idx"] = q.event_id.map(ev.event_idx)
    sessions = q[["event_id", "event_idx", "segment"]].drop_duplicates()
    sessions = sessions.sort_values(["event_idx", "segment"], ignore_index=True)
    sessions["session_idx"] = np.arange(len(sessions))
    obs = q.merge(sessions[["event_id", "segment", "session_idx"]], on=["event_id", "segment"])
    obs = obs.merge(ent[["event_id", "driver_id", "entry_idx", "team"]], on=["event_id", "driver_id"])
    obs = obs.merge(cars[["team", "event_idx", "car_idx"]], on=["team", "event_idx"])
    obs = obs.sort_values(["session_idx", "entry_idx"], ignore_index=True)
    obs["car_segment_idx"] = obs.groupby(["session_idx", "car_idx"], sort=True).ngroup()

    drv = drivers.loc[driver_ids].reset_index()
    drv["driver_idx"] = np.arange(len(drv))

    design = Design(start_season=start_season, events=events, sessions=sessions,
                    entries=ent, cars=cars, obs=obs, drivers=drv, teams=teams)
    design.circuit_factor = circuit_factors(design)
    return design


if __name__ == "__main__":
    d = build_design()
    print(f"events={len(d.events)} sessions={len(d.sessions)} obs={len(d.obs)} "
          f"entries={len(d.entries)} cars={len(d.cars)} drivers={len(d.drivers)} teams={len(d.teams)}")
    print(d.entries[["first", "same_season"]].value_counts())
    print(d.cars[["first", "season_start", "reset"]].value_counts())
    print(d.entries.query("driver_id in ['alonso','piastri','arvid_lindblad']")
          .groupby("driver_id")[["experience", "age"]].agg(["min", "max"]))
