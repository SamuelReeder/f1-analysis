"""Combine the per-race FastF1 tables into processed race tables (stage 2 data layer).

    python -m f1rank.racedata

Reads data/raw/fastf1_tables/<event_id>_R/ (written by extract/race_extract.py in the FastF1
environment) and writes to data/processed/:

    race_laps.parquet           one row per driver per lap: times in seconds of session time,
                                position, compound, tyre age, stint, pit in/out, track status
    race_messages.parquet       race-control messages, with the race lap they refer to
    race_track_status.parquet   track status changes (1 green, 2 yellow, 4 SC, 5 red,
                                6 VSC deployed, 7 VSC ending)
    race_weather.parquet        weather samples
    race_classification.parquet FastF1 classification: car number, driver id, grid, status
    race_sources.json           per race: FastF1 version, retrieval time and table hashes

Sprint races (<event_id>_S/, 2021 onward) go to the same tables with a sprint_ prefix
(sprint_laps.parquet, ...), listed under "sprints" in race_sources.json.

Per-lap telemetry summaries (extract/telemetry.py: coasting and full-throttle seconds per
lap) go to race_telemetry.parquet, with each race's alignment check (shift, speed-trap
agreement) in the columns and under "telemetry" in race_sources.json.

Column names are snake_case versions of FastF1's.
"""

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TABLES = ROOT / "data" / "raw" / "fastf1_tables"
PROCESSED = ROOT / "data" / "processed"
KINDS = {"laps": "race_laps", "messages": "race_messages", "track_status": "race_track_status",
         "weather": "race_weather", "results": "race_classification"}


def snake(name: str) -> str:
    name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name)
    return re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def build(session: str = "R") -> dict[str, pd.DataFrame]:
    frames = {k: [] for k in KINDS}
    sources = {}
    for d in sorted(TABLES.glob(f"*_{session}")):
        if not (d / "manifest.json").exists():
            continue
        event_id = d.name[:-2]
        manifest = json.loads((d / "manifest.json").read_text())
        sources[event_id] = {k: manifest[k] for k in ("fastf1_version", "retrieved_utc", "event_name")}
        sources[event_id]["sha256"] = {k: v["sha256"] for k, v in manifest["tables"].items()}
        for kind in KINDS:
            df = pd.read_parquet(d / f"{kind}.parquet")
            df.insert(0, "event_id", event_id)
            frames[kind].append(df)
    out = {}
    prefix = {"R": "race", "S": "sprint"}[session]
    for kind, name in KINDS.items():
        name = name.replace("race", prefix, 1)
        df = pd.concat(frames[kind], ignore_index=True)
        df.columns = [snake(c) if c != "event_id" else c for c in df.columns]
        out[name] = df
    c = out[f"{prefix}_classification"]
    c["driver_number"] = c.driver_number.astype(str)
    out[f"{prefix}_laps"]["driver_number"] = out[f"{prefix}_laps"].driver_number.astype(str)
    return out, sources


def telemetry() -> tuple[pd.DataFrame, dict]:
    frames, sources = [], {}
    for d in sorted(TABLES.glob("*_R")):
        if not (d / "telemetry.json").exists():
            continue
        event_id = d.name[:-2]
        info = json.loads((d / "telemetry.json").read_text())
        sources[event_id] = info
        frames.append(pd.read_parquet(d / "telemetry_laps.parquet").assign(
            event_id=event_id, shift_s=info["shift_s"], trap_diff_kmh=info["speed_trap_median_abs_diff_kmh"]))
    if not frames:
        return pd.DataFrame(), sources
    T = pd.concat(frames, ignore_index=True)
    T.columns = [snake(c) for c in T.columns]
    T["driver_number"] = T.driver_number.astype(str)
    return T, sources


def main() -> None:
    tables, sources = build()
    sprint_tables, sprint_sources = build("S")
    tel, tel_sources = telemetry()
    if len(tel):
        tables["race_telemetry"] = tel
    for name, df in {**tables, **sprint_tables}.items():
        df.to_parquet(PROCESSED / f"{name}.parquet", index=False)
        print(f"{name:28s} {len(df):8d} rows")
    failures = TABLES / "failures.json"
    (PROCESSED / "race_sources.json").write_text(json.dumps({
        "races": sources,
        "sprints": sprint_sources,
        "telemetry": tel_sources,
        "not_available": json.loads(failures.read_text()) if failures.exists() else {},
    }, indent=1))


if __name__ == "__main__":
    main()
