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
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
TABLES = ROOT / "data" / "raw" / "fastf1_tables"
FIRST_TIMING_SEASON = 2018
RETRY_DAYS = 10


def unpublished_input_change(root: Path) -> str | None:
    """Compare with the published receipt, not an intermediate failed refresh.

    Fetch/build may have succeeded before a later fit failed. Comparing remote
    data only with that updated cache would otherwise incorrectly skip the retry.
    """
    from dashboard.verify_publication import digest, sources
    directory = root / "dashboard/public/data"
    path = directory / "publication.json"
    if not path.exists():
        return "the published source receipt is missing"
    receipt = json.loads(path.read_text())
    if (receipt.get("schema_version") != 1 or not isinstance(receipt.get("sources"), dict)
            or receipt.get("pointer_sha256") != digest(directory / "latest.json")):
        raise ValueError("Invalid published source receipt; inspect publication before refreshing")
    # Editorial reports invalidate a portable receipt but do not need a refit.
    # These are the input namespaces used by the fitting/export pipeline.
    def input_name(name):
        return name.startswith(("data/", "f1rank/", "extract/", "contracts/")) or name == "docs/overall_candidate.md"
    expected = {name: value for name, value in receipt["sources"].items() if input_name(name)}
    actual = {name for name in sources(root) if input_name(name)}
    if set(expected) != actual:
        return "the published source inventory changed"
    for name, value in sorted(expected.items()):
        if digest(root / name) != value:
            return f"a published input changed: {name}"
    return None


def _completed_rows(races: list[dict], season: int, last_round: int, endpoint: str) -> dict:
    """Ignore future qualifying, and ignore JSON or driver-row ordering only."""
    key = {"qualifying": "QualifyingResults", "results": "Results"}[endpoint]
    result = {}
    for race in races:
        rnd = int(race["round"])
        if int(race["season"]) != season or rnd < 1 or rnd in result:
            raise ValueError("Invalid or duplicate race in correction check")
        if rnd > last_round:
            continue
        rows = race[key]
        drivers = [r["Driver"]["driverId"] for r in rows]
        if len(set(drivers)) != len(drivers):
            raise ValueError("Duplicate driver rows in correction check")
        result[rnd] = {**race, key: sorted(rows, key=lambda r: r["Driver"]["driverId"])}
    return result


def result_revisions(event: str, processed: Path) -> list[str]:
    """Read-only comparison of the latest season's completed qualifying/results.

    Raw hashes must still agree with the processed inputs. A previous failed run
    or a locally corrected cache must not hide a correction from the scheduler.
    Older seasons and revised FastF1 timing need an explicit source refresh.
    """
    from .fetch import RAW, download_table
    season, rnd = event_key(event)
    path = processed / "sources.json"
    used = json.loads(path.read_text()).get("jolpica", {}) if path.exists() else {}
    changed = []
    for endpoint in ("qualifying", "results"):
        name = f"{season}_{endpoint}.json"
        path = RAW / name
        if not path.exists():
            changed.append(name)
            continue
        saved = path.read_bytes()
        if used.get(name, {}).get("sha256") != hashlib.sha256(saved).hexdigest():
            changed.append(name)
            continue
        before = _completed_rows(json.loads(saved), season, rnd, endpoint)
        after = _completed_rows(download_table(season, endpoint), season, rnd, endpoint)
        if before != after:
            changed.append(name)
    return changed


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
    if not result["due"]:
        change = unpublished_input_change(ROOT)
        if change:
            result = dict(result, due=True, reason=change)
        else:
            revised = result_revisions(result["event"], PROCESSED)
            if revised:
                result = dict(result, due=True, reason="revised completed-race sources: " + ", ".join(revised))
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
