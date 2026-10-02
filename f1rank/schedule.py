"""Decide whether a scheduled refresh has new data to fit.

python -m f1rank.schedule check [--force]

A refresh is due when Jolpica lists a race result newer than the committed
data/processed/race.parquet, or when the latest race (2018 onward) still has no
extracted FastF1 timing: FastF1 publishes timing separately and can lag, so a run
shortly after a race retries until the timing appears or RETRY_DAYS have passed.

The decision is printed as JSON and, under GitHub Actions, written to $GITHUB_OUTPUT
(due, event, race_name, reason) for the workflow's later steps.
"""
import argparse
import datetime as dt
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "data" / "raw" / "fastf1_tables"
FIRST_TIMING_SEASON = 2018
RETRY_DAYS = 10


def event_key(event_id: str) -> tuple[int, int]:
    season, rnd = event_id.split("-")
    return int(season), int(rnd)


def latest_jolpica_race() -> dict | None:
    """The most recent race with published results, or None before a season's first race."""
    from .fetch import BASE, _get
    races = _get(f"{BASE}/current/last/results.json?limit=1")["RaceTable"]["Races"]
    if not races:
        return None
    r = races[0]
    return {"event_id": f"{int(r['season'])}-{int(r['round']):02d}", "race_name": r["raceName"],
            "date": r["date"]}


def decide(latest: dict | None, known: pd.DataFrame, events: pd.DataFrame,
           has_timing, today: dt.date) -> dict:
    """Pure decision rule; `has_timing(event_id)` reports extracted FastF1 timing."""
    newest = max(known.event_id, key=event_key)
    if latest is not None and event_key(latest["event_id"]) > event_key(newest):
        return dict(due=True, event=latest["event_id"], race_name=latest["race_name"],
                    reason=f"Jolpica has results for {latest['event_id']}; the data end at {newest}")
    row = events[events.event_id == newest]
    name = str(row.race_name.iloc[0]) if len(row) else newest
    date = dt.date.fromisoformat(str(row.date.iloc[0])) if len(row) else today
    if (event_key(newest)[0] >= FIRST_TIMING_SEASON and not has_timing(newest)
            and (today - date).days <= RETRY_DAYS):
        return dict(due=True, event=newest, race_name=name,
                    reason=f"race timing for {newest} has not been extracted yet")
    return dict(due=False, event=newest, race_name=name, reason=f"no race after {newest}")


def check(force: bool = False) -> dict:
    known = pd.read_parquet(PROCESSED / "race.parquet", columns=["event_id"]).drop_duplicates()
    events = pd.read_parquet(PROCESSED / "events.parquet")
    result = decide(latest_jolpica_race(), known, events,
                    lambda e: (TABLES / f"{e}_R" / "manifest.json").exists(), dt.date.today())
    if force and not result["due"]:
        result = dict(result, due=True, reason="forced: " + result["reason"])
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["check"])
    p.add_argument("--force", action="store_true", help="refresh even without new data")
    args = p.parse_args()
    result = check(args.force)
    print(json.dumps(result, indent=1))
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as out:
            for key, value in result.items():
                out.write(f"{key}={str(value).lower() if isinstance(value, bool) else value}\n")


if __name__ == "__main__":
    main()
