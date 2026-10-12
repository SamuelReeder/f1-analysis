"""Cutoff-safe shared catalog for the successor qualifying/lap/outcome model."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..design import STAGE1_TABLES, PROCESSED, build_design
from ..racepace import WET, clean_laps

FIRST_QUALIFYING_SEASON = 2010
FIRST_LAP_SEASON = 2018
MIN_CLEAN_DRIVERS = 8
TABLES = (*STAGE1_TABLES, "race_laps", "race_weather", "timeline")
NON_START_OR_EXCLUDED = r"^(Did not start|Did not qualify|Did not prequalify|Withdrew|Excluded|Disqualified)$"
FINISHED = r"^(Finished|\+[0-9]+ Laps?)$"


@dataclass
class OverallDesign:
    qualifying: object
    laps: dict
    outcomes: dict
    catalog: dict


def bounded_tables(tables, cutoff):
    """Filter before computing any median, circuit factor, catalog or context."""
    if cutoff not in set(tables["events"].event_id):
        raise ValueError("Cutoff must identify an observed event")
    out = {name: (frame.loc[frame.event_id <= cutoff].copy()
                  if "event_id" in frame else frame.copy()) for name, frame in tables.items()}
    q, r = out["entries"], out["race"]
    keys = ["event_id", "driver_id"]
    for name, frame in (("entries", q), ("race", r)):
        if frame.duplicated(keys).any():
            raise ValueError(f"Duplicate driver-event in {name}")
    paired = q[keys + ["team"]].merge(r[keys + ["team"]], on=keys, suffixes=("_q", "_r"))
    if (paired.team_q != paired.team_r).any():
        raise ValueError("Qualifying/race team identity conflict")
    # A substitute or a driver with no qualifying entry still needs a latent state.
    missing = r.loc[~pd.MultiIndex.from_frame(r[keys]).isin(pd.MultiIndex.from_frame(q[keys]))].copy()
    missing["constructor_name"] = missing.team
    missing["quali_position"] = np.nan
    out["entries"] = pd.concat([q, missing.reindex(columns=q.columns)], ignore_index=True)
    observed = set(out["entries"].driver_id)
    out["drivers"] = out["drivers"].loc[out["drivers"].driver_id.isin(observed)].copy()
    if observed - set(out["drivers"].driver_id):
        raise ValueError("Entrant missing driver metadata")
    return out


def known_rainfall(weather):
    """An event with no observed rainfall readings has unknown context."""
    observed = weather.dropna(subset=["rainfall"])
    return observed.groupby("event_id").rainfall.any()


def clean_training_laps(tables):
    raw, weather, timeline = (tables[k] for k in ("race_laps", "race_weather", "timeline"))
    rain = known_rainfall(weather)
    chunks = []
    for event, frame in raw.groupby("event_id", sort=True):
        if (int(event[:4]) < FIRST_LAP_SEASON or event not in rain.index
                or rain[event] or frame.compound.isin(WET).any()):
            continue
        clean = clean_laps(frame, timeline[timeline.event_id == event])
        if clean.driver_id.nunique() >= MIN_CLEAN_DRIVERS:
            chunks.append(clean.assign(event_id=event))
    if not chunks:
        raise ValueError("No eligible dry laps in the training window")
    result = pd.concat(chunks, ignore_index=True).drop(columns="team").merge(
        tables["race"][["event_id", "driver_id", "team"]], on=["event_id", "driver_id"],
        how="left", validate="many_to_one")
    if result.team.isna().any():
        raise ValueError("Lap has no canonical race entrant")
    result["season"] = result.event_id.str[:4].astype(int)
    return result.sort_values(["event_id", "stint_key", "lap_number"], ignore_index=True)


def build(cutoff, *, tables=None, clean=None):
    """`clean` is an injected cleaned-lap fixture for synthetic checks only.

    Real runs must use the default cleaning path, which truncates raw data first.
    Every lap and outcome maps to the SAME driver-event and car-event states.
    """
    from ..race_total_model import design as lap_design
    if tables is None:
        tables = {name: pd.read_parquet(PROCESSED / f"{name}.parquet") for name in TABLES}
    t = bounded_tables(tables, cutoff)
    q = build_design(FIRST_QUALIFYING_SEASON, end_event=cutoff, tables=t)
    C = clean_training_laps(t) if clean is None else clean.loc[clean.event_id <= cutoff].copy()
    if C.empty:
        raise ValueError("No training laps")
    ld, _ = lap_design(C)
    entry_key = pd.MultiIndex.from_frame(q.entries[["event_id", "driver_id"]])
    car_key = pd.MultiIndex.from_frame(q.cars[["event_idx", "team"]])
    events = q.events.set_index("event_id")

    def entry_index(frame):
        result = entry_key.get_indexer(pd.MultiIndex.from_frame(frame[["event_id", "driver_id"]]))
        if (result < 0).any():
            raise ValueError("Observation missing from shared entry catalog")
        return result

    entry_car = car_key.get_indexer(pd.MultiIndex.from_frame(q.entries[["event_idx", "team"]]))
    ld["entry"] = entry_index(C)
    # Context is circuit x wet/dry x neutralised/green. Missing weather is excluded,
    # never silently relabelled dry. Old lap archives do not supply this likelihood.
    races = t["race"].loc[t["race"].event_id >= f"{FIRST_LAP_SEASON}-01"].copy()
    weather = known_rainfall(t["race_weather"])
    covered = set(t["race_laps"].event_id) & set(weather.index)
    wet_tyres = t["race_laps"].groupby("event_id").compound.agg(lambda s: s.isin(WET).any())
    excluded_events = sorted(set(races.event_id) - covered)
    races = races[races.event_id.isin(covered)]
    # A pit-lane start uses the full entered field, as it does in a forecast.
    # Later DNS/DSQ result exclusions cannot change this pre-race feature.
    races["grid_slot"] = np.where(races.grid > 0, races.grid, races.groupby("event_id").driver_id.transform("size"))
    excluded_status = races[races.status.str.match(NON_START_OR_EXCLUDED)].copy()
    races = races[~races.status.str.match(NON_START_OR_EXCLUDED)].copy()
    if races.empty:
        raise ValueError("No eligible race outcomes")
    neutral = set(t["timeline"].loc[t["timeline"].kind.isin(["safety_car", "vsc", "red_flag"]), "event_id"])
    circ = races.event_id.map(events.circuit_idx).to_numpy(int)
    wet = (races.event_id.map(weather) | races.event_id.map(wet_tyres).fillna(False)).to_numpy(int)
    sc = races.event_id.isin(neutral).to_numpy(int)
    races["context"] = circ * 4 + wet * 2 + sc
    races["entry_idx"] = entry_index(races)
    races["running"] = races.status.str.match(FINISHED)
    if races.laps.isna().any() or (races.laps < 0).any() or (races.grid_slot < 1).any():
        raise ValueError("Invalid completion or grid data")
    F = races[races.running].sort_values(["event_id", "position"])
    evs = sorted(F.event_id.unique())
    if not evs:
        raise ValueError("No running finishers")
    width = int(F.groupby("event_id").size().max())
    entry, grid, mask = np.zeros((len(evs), width), int), np.zeros((len(evs), width)), np.zeros((len(evs), width), bool)
    row = F.event_id.map({e: i for i, e in enumerate(evs)}).to_numpy()
    col = F.groupby("event_id").cumcount().to_numpy()
    entry[row, col], grid[row, col], mask[row, col] = F.entry_idx, np.log(F.grid_slot), True
    context = F.groupby("event_id").context.first().reindex(evs).to_numpy(int)
    n_circ = int(q.events.circuit_idx.max()) + 1
    car_year = q.cars.groupby("team_season_idx").season.first().to_numpy()
    from ..race_total_model import contrast_basis
    od = dict(entry=entry, log_grid=grid, mask=mask, context=context,
              risk_entry=races.entry_idx.to_numpy(int), risk_context=races.context.to_numpy(int),
              risk_distance=races.groupby("event_id").laps.transform("max").to_numpy(float),
              completed=races.laps.to_numpy(float), failed=(~races.running).to_numpy(),
              entry_car=entry_car, n_circuits=n_circ,
              context_circuit=np.repeat(np.arange(n_circ), 4),
              context_wet=np.tile([0, 0, 1, 1], n_circ), context_sc=np.tile([0, 1, 0, 1], n_circ),
              car_basis=contrast_basis(car_year))
    field = t["race"].loc[t["race"].event_id == cutoff].sort_values("driver_id")
    if field.empty:
        raise ValueError("Cutoff has no race entrants")
    return OverallDesign(q, ld, od, dict(cutoff=cutoff, field=field.driver_id.tolist(),
        field_entries=entry_index(field).tolist(), circuits=q.events.groupby("circuit_idx").circuit_id.first().tolist(),
        excluded_context_events=excluded_events,
        excluded_status_rows=excluded_status[["event_id", "driver_id", "status"]].to_dict("records"),
        outcome_events=evs, first_lap_event=str(C.event_id.min()), last_lap_event=str(C.event_id.max())))
