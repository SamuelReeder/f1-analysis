"""Freeze a common scenario from committed reference-season context, not rankings.

Every completed reference-season GP is required. A missing distance, weather or
lap archive stops construction rather than selecting a more favorable subset.
The result is a virtual full-distance calendar, not a forecast of future weather.
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SEASON = 2025
INPUTS = ("events", "race", "race_weather", "race_laps", "timeline")
WET_COMPOUNDS = {"INTERMEDIATE", "WET"}
NEUTRALISATIONS = {"safety_car", "vsc", "red_flag"}


def construct(tables, season=REFERENCE_SEASON):
    events = tables["events"].loc[tables["events"].season == season].sort_values("event_id")
    if events.empty or events.event_id.duplicated().any():
        raise ValueError("Missing or duplicate reference events")
    ids = set(events.event_id)
    for name in ("race", "race_weather", "race_laps"):
        missing = ids - set(tables[name].event_id)
        if missing:
            raise ValueError(f"Missing {name} reference events: {sorted(missing)}")
    weather = tables["race_weather"].groupby("event_id").rainfall.agg(
        lambda s: None if s.isna().all() else bool(s.dropna().any()))
    wet_tyres = tables["race_laps"].groupby("event_id").compound.agg(lambda s: s.isin(WET_COMPOUNDS).any())
    neutral = set(tables["timeline"].loc[tables["timeline"].kind.isin(NEUTRALISATIONS), "event_id"])
    rows = []
    for event in events.itertuples():
        winner = tables["race"].loc[(tables["race"].event_id == event.event_id) & (tables["race"].position == 1)]
        if len(winner) != 1:
            raise ValueError(f"No unique winner distance at {event.event_id}")
        laps = float(winner.laps.iloc[0])
        if not (laps >= 2 and laps.is_integer()) or weather[event.event_id] is None:
            raise ValueError(f"Incomplete reference conditions at {event.event_id}")
        rows.append(dict(event_id=event.event_id, circuit_id=event.circuit_id,
            reference_distance_laps=int(laps), wet=bool(weather[event.event_id] or wet_tyres[event.event_id]),
            neutralised=event.event_id in neutral, weight=1 / len(events)))
    return dict(schema_version=1, scenario_id=f"overall-common-{season}-v1", reference_season=season,
        target="Current driver performance under equal machinery and team support",
        calendar_policy="Every completed GP of the reference season, equally weighted",
        distance_policy="Virtual full-distance GPs, each scheduled for its reference winner's completed laps",
        conditions_policy="All drivers share each reference event's wet/dry and neutralisation context",
        field_policy="Latest completed race entrants, including entrants without qualifying times",
        future_driver_development=False, sprint_points=False, forecast_of_actual_future_calendar=False,
        unresolved_before_publication=["completion/incident model", "lapping", "shortened-race behavior", "predictive calibration"],
        events=rows)


def main(output):
    paths = {name: ROOT / "data" / "processed" / f"{name}.parquet" for name in INPUTS}
    tables = {name: pd.read_parquet(path) for name, path in paths.items()}
    result = construct(tables)
    result["inputs"] = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                        for path in paths.values()}
    own = Path(__file__).resolve()
    result["inputs"][str(own.relative_to(ROOT))] = hashlib.sha256(own.read_bytes()).hexdigest()
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    output = Path(output)
    if output.exists():
        if output.read_text() != encoded:
            raise ValueError("Reference artifact already exists and differs; do not overwrite it")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded)
    print(f"Reference artifact verified: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=str(ROOT / "docs" / "overall_reference_calendar.json"))
    main(parser.parse_args().output)
