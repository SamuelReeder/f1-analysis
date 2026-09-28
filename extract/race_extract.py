"""Extract race timing from FastF1 (2018 onward) into per-race Parquet tables.

This is the stage-2 data layer's FastF1 step (docs/racing_approach.md). It runs in the
FastF1 environment (FastF1 requires pandas < 3):

    .venv-fastf1/bin/python extract/race_extract.py [--first 2018] [--last 2026] [--refresh]

Reads the event list from data/processed/events.parquet. For each race it writes
data/raw/fastf1_tables/<event_id>_R/:

    laps.parquet          one row per driver per lap (times in seconds of session time)
    messages.parquet      race-control messages (dated, with the race lap they refer to)
    track_status.parquet  track status changes (green, yellow, SC, VSC, red)
    weather.parquet       weather samples (about one a minute)
    results.parquet       classification with status, grid and FastF1 driver ids
    manifest.json         FastF1 version, retrieval time, row counts and a
                          sha256 per table, so a build can record what it used

Races already extracted are skipped unless --refresh. Races FastF1 has no timing for are
listed in data/raw/fastf1_tables/failures.json. `python -m f1rank.racedata` (main
environment) combines the tables into data/processed/race_*.parquet.
"""

import argparse
import datetime as dt
import hashlib
import json
import time

import fastf1
import pandas as pd

from ff1 import ROOT, enable_cache, patient

PROCESSED = ROOT / "data" / "processed"
# FastF1 also queries Jolpica (about 2 requests per race, even when cached): pausing between
# races keeps that under Jolpica's limit of 500 requests an hour
PAUSE_S = 15
TABLES = ROOT / "data" / "raw" / "fastf1_tables"

LAP_COLS = ["Driver", "DriverNumber", "Team", "LapNumber", "Position", "LapTime", "Time", "LapStartTime",
            "Stint", "Compound", "TyreLife", "FreshTyre", "PitInTime", "PitOutTime",
            "Sector1Time", "Sector2Time", "Sector3Time", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
            "TrackStatus", "IsAccurate", "Deleted", "DeletedReason", "FastF1Generated"]


def seconds(df: pd.DataFrame) -> pd.DataFrame:
    """Timedelta columns to float seconds."""
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_timedelta64_dtype(df[c]):
            df[c] = df[c].dt.total_seconds()
    return df


def extract(season: int, rnd: int, out) -> dict:
    s = fastf1.get_session(season, rnd, "R")
    patient(s.load, laps=True, telemetry=False, weather=True, messages=True)
    if s.laps.empty:
        raise ValueError("no lap timing")
    results = s.results.copy()
    ids = results.set_index("Abbreviation").DriverId
    laps = seconds(s.laps[LAP_COLS].copy())
    laps.insert(0, "driver_id", laps.Driver.map(ids))
    laps["Deleted"] = laps.Deleted.astype("boolean")
    laps["FreshTyre"] = laps.FreshTyre.astype("boolean")
    tables = {
        "laps": laps,
        "messages": pd.DataFrame(s.race_control_messages),
        "track_status": seconds(pd.DataFrame(s.track_status)),
        "weather": seconds(pd.DataFrame(s.weather_data)),
        "results": seconds(results.reset_index(drop=True)),
    }
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"season": season, "round": rnd, "session": "R", "fastf1_version": fastf1.__version__,
                "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_name": s.event.EventName, "tables": {}}
    for name, df in tables.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest["tables"][name] = {"rows": int(len(df)),
                                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--first", type=int, default=2018)
    p.add_argument("--last", type=int, default=2100)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    enable_cache()
    events = pd.read_parquet(PROCESSED / "events.parquet")
    events = events[events.season.between(args.first, args.last)]
    fail_path = TABLES / "failures.json"
    failures = json.loads(fail_path.read_text()) if fail_path.exists() else {}
    for ev in events.itertuples():
        out = TABLES / f"{ev.event_id}_R"
        if (out / "manifest.json").exists() and not args.refresh:
            continue
        time.sleep(PAUSE_S)
        try:
            m = extract(ev.season, ev.round, out)
            failures.pop(ev.event_id, None)
            print(f"{ev.event_id}: {m['tables']['laps']['rows']} laps, "
                  f"{m['tables']['messages']['rows']} messages", flush=True)
        except Exception as e:  # noqa: BLE001 - record and continue with the next race
            failures[ev.event_id] = f"{type(e).__name__}: {e}"
            print(f"{ev.event_id}: FAILED {failures[ev.event_id]}", flush=True)
        TABLES.mkdir(parents=True, exist_ok=True)
        fail_path.write_text(json.dumps(failures, indent=1))


if __name__ == "__main__":
    main()
