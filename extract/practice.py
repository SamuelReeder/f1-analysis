"""Practice lap timing (2022 onward) from FastF1, for the pre-registered race feature tests
(docs/race_features.md).

One practice session per race weekend, the one in which teams do their race-fuel long
runs: the second practice on a conventional weekend, the only practice on a sprint
weekend. Runs in the FastF1 environment (FastF1 requires pandas < 3):

    .venv-fastf1/bin/python extract/practice.py [--first 2022] [--last 2026] [--refresh]

Reads the event list from data/processed/events.parquet (rounds that have race timing in
data/raw/fastf1_tables/<event_id>_R/). For each weekend it writes
data/raw/fastf1_tables/<event_id>_FP/:

    laps.parquet      one row per driver per lap (times in seconds), with FastF1's
                      abbreviation and team; driver ids are matched to the weekend's race
                      results by f1rank.race_features
    manifest.json     the session used, FastF1 version, retrieval time, row count, sha256

Weekends already extracted are skipped unless --refresh. Failures are listed in
data/raw/fastf1_tables/practice_failures.json.
"""

import argparse
import datetime as dt
import hashlib
import json
import time

import fastf1
import pandas as pd

from ff1 import ROOT, enable_cache, patient
from race_extract import LAP_COLS, PAUSE_S, seconds

PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "data" / "raw" / "fastf1_tables"
SPRINT_SESSIONS = {"Sprint", "Sprint Shootout", "Sprint Qualifying"}


def long_run_session(season: int, rnd: int) -> str:
    """'Practice 2' on a conventional weekend, 'Practice 1' on a sprint weekend."""
    event = fastf1.get_event(season, rnd)
    names = {event[f"Session{i}"] for i in range(1, 6)}
    return "Practice 1" if names & SPRINT_SESSIONS else "Practice 2"


def extract(season: int, rnd: int, out) -> dict:
    name = long_run_session(season, rnd)
    s = fastf1.get_session(season, rnd, name)
    patient(s.load, laps=True, telemetry=False, weather=False, messages=False)
    if s.laps.empty:
        raise ValueError(f"no lap timing in {name}")
    laps = seconds(s.laps[LAP_COLS].copy())
    laps["Deleted"] = laps.Deleted.astype("boolean")
    laps["FreshTyre"] = laps.FreshTyre.astype("boolean")
    out.mkdir(parents=True, exist_ok=True)
    path = out / "laps.parquet"
    laps.to_parquet(path, index=False)
    manifest = {"season": season, "round": rnd, "session": name, "fastf1_version": fastf1.__version__,
                "retrieved_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "event_name": s.event.EventName,
                "tables": {"laps": {"rows": int(len(laps)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--first", type=int, default=2022)
    p.add_argument("--last", type=int, default=2100)
    p.add_argument("--refresh", action="store_true")
    args = p.parse_args()
    enable_cache()
    events = pd.read_parquet(PROCESSED / "events.parquet")
    events = events[events.season.between(args.first, args.last)
                    & [(TABLES / f"{e}_R" / "manifest.json").exists() for e in events.event_id]]
    fail_path = TABLES / "practice_failures.json"
    failures = json.loads(fail_path.read_text()) if fail_path.exists() else {}
    for ev in events.itertuples():
        out = TABLES / f"{ev.event_id}_FP"
        if (out / "manifest.json").exists() and not args.refresh:
            continue
        time.sleep(PAUSE_S)
        try:
            m = extract(ev.season, ev.round, out)
            failures.pop(ev.event_id, None)
            print(f"{ev.event_id}: {m['session']}, {m['tables']['laps']['rows']} laps", flush=True)
        except Exception as e:  # noqa: BLE001 - record and continue with the next weekend
            failures[ev.event_id] = f"{type(e).__name__}: {e}"
            print(f"{ev.event_id}: FAILED {failures[ev.event_id]}", flush=True)
        fail_path.write_text(json.dumps(failures, indent=1))


if __name__ == "__main__":
    main()
