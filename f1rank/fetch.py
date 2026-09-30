"""Download qualifying and race results from the Jolpica (Ergast-compatible) API.

Raw JSON pages are cached under data/raw/jolpica/ so reruns only fetch what is
missing. The current season is always refetched because new rounds appear, and
`--refresh-all` refetches every season (sources revise past data, e.g. retirement
statuses). When a refetch changes a file, the previous version is kept in
data/raw/jolpica/archive/ under its retrieval time. manifest.json records each file's
retrieval time and sha256; `build` copies the entries it used to data/processed/sources.json.

Stage 2 adds sprint results (per season) and, with `--laps FIRST LAST`, lap-by-lap times and
positions plus pit stops per race (data/raw/jolpica/laps/). Lap data are about 11 requests per
race, so `--laps` paces itself under Jolpica's hourly limit (about 3-4 hours for 2010-2017).
"""

import argparse
import datetime as dt
import hashlib
import json
import time
from pathlib import Path

import requests

BASE = "https://api.jolpi.ca/ergast/f1"
RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "jolpica"
PAGE = 100
MIN_INTERVAL = 0.35  # seconds between requests; Jolpica allows ~4 req/s burst
LAPS_INTERVAL = 8.0  # Jolpica allows 500 requests an hour; bulk lap downloads stay under it

MANIFEST = RAW / "manifest.json"

_last_request = 0.0


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def save_versioned(out: Path, text: str) -> None:
    """Write a raw response; archive the previous version if the content changed."""
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {}
    digest = hashlib.sha256(text.encode()).hexdigest()
    old = manifest.get(out.name)
    if out.exists() and old and old["sha256"] != digest:
        archive = RAW / "archive" / f"{out.stem}.{old['retrieved_utc'].replace(':', '')}.json"
        archive.parent.mkdir(parents=True, exist_ok=True)
        out.replace(archive)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    if not old or old["sha256"] != digest:
        manifest[out.name] = {"retrieved_utc": _now(), "sha256": digest}
    else:
        manifest[out.name]["checked_utc"] = _now()  # refetched, unchanged
    MANIFEST.write_text(json.dumps(manifest, indent=1, sort_keys=True))


def _get(url: str, interval: float = MIN_INTERVAL) -> dict:
    global _last_request
    for attempt in range(8):
        wait = interval - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        resp = requests.get(url, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(min(600, 60 * 2 ** attempt) if resp.status_code == 429 else min(60, 2 ** attempt))
            continue
        resp.raise_for_status()
        return resp.json()["MRData"]
    raise RuntimeError(f"giving up on {url}")


def fetch_table(season: int, endpoint: str, refresh: bool = False) -> list[dict]:
    """Return every race in a season for an endpoint, merging paginated rows by round."""
    out = RAW / f"{season}_{endpoint}.json"
    if out.exists() and not refresh:
        return json.loads(out.read_text())

    list_key = {"qualifying": "QualifyingResults", "results": "Results", "sprint": "SprintResults"}[endpoint]
    races: dict[str, dict] = {}
    offset, total = 0, None
    while total is None or offset < total:
        data = _get(f"{BASE}/{season}/{endpoint}.json?limit={PAGE}&offset={offset}")
        total = int(data["total"])
        for race in data["RaceTable"]["Races"]:
            rows = race.pop(list_key)
            races.setdefault(race["round"], {**race, list_key: []})[list_key].extend(rows)
        offset += PAGE

    merged = sorted(races.values(), key=lambda r: int(r["round"]))
    save_versioned(out, json.dumps(merged))
    return merged


def fetch_race(season: int, rnd: int, endpoint: str) -> dict | None:
    """Lap-by-lap timings ("laps") or pit stops ("pitstops") for one race, cached.
    Returns None when Jolpica has no rows for the race."""
    out = RAW / "laps" / f"{season}_{rnd:02d}_{endpoint}.json"
    if out.exists():
        return json.loads(out.read_text())
    list_key = {"laps": "Laps", "pitstops": "PitStops"}[endpoint]
    race, rows, offset, total = None, [], 0, None
    while total is None or offset < total:
        data = _get(f"{BASE}/{season}/{rnd}/{endpoint}.json?limit={PAGE}&offset={offset}", LAPS_INTERVAL)
        total = int(data["total"])
        for r in data["RaceTable"]["Races"]:
            race = race or {k: v for k, v in r.items() if k != list_key}
            rows.extend(r[list_key])
        offset += PAGE
    if race is None:
        return None
    if endpoint == "laps":  # a lap can be split over two pages
        merged: dict[str, list] = {}
        for lap in rows:
            merged.setdefault(lap["number"], []).extend(lap["Timings"])
        rows = [{"number": n, "Timings": t} for n, t in sorted(merged.items(), key=lambda kv: int(kv[0]))]
    doc = {**race, list_key: rows}
    save_versioned(out, json.dumps(doc))
    return doc


def fetch_debuts(driver_ids: set[str]) -> dict[str, dict]:
    """First championship race (season, round, date) for each driver, cached."""
    out = RAW / "debuts.json"
    debuts = json.loads(out.read_text()) if out.exists() else {}
    for driver_id in sorted(driver_ids - debuts.keys()):
        race = _get(f"{BASE}/drivers/{driver_id}/results.json?limit=1")["RaceTable"]["Races"][0]
        debuts[driver_id] = {"season": int(race["season"]), "round": int(race["round"]),
                             "date": race["date"]}
        out.write_text(json.dumps(debuts, indent=1, sort_keys=True))
    return debuts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=int, default=2010)
    parser.add_argument("--last", type=int, default=dt.date.today().year)
    parser.add_argument("--refresh-all", action="store_true", help="refetch every season")
    parser.add_argument("--laps", type=int, nargs=2, metavar=("FIRST", "LAST"),
                        help="only fetch lap times and pit stops for these seasons (slow)")
    args = parser.parse_args()

    if args.laps:
        for season in range(args.laps[0], args.laps[1] + 1):
            for race in fetch_table(season, "results"):
                rnd = int(race["round"])
                laps = fetch_race(season, rnd, "laps")
                stops = fetch_race(season, rnd, "pitstops") if season >= 2012 else None
                n = sum(len(lap["Timings"]) for lap in laps["Laps"]) if laps else 0
                print(f"{season}-{rnd:02d} lap timings={n} pit stops={len(stops['PitStops']) if stops else 0}",
                      flush=True)
        return

    driver_ids = set()
    for season in range(args.first, args.last + 1):
        refresh = args.refresh_all or season == dt.date.today().year
        for endpoint in ("qualifying", "results") + (("sprint",) if season >= 2021 else ()):
            races = fetch_table(season, endpoint, refresh=refresh)
            if endpoint == "sprint":
                print(f"{season} sprint     races={len(races):2d}")
                continue
            n_rows = sum(len(r.get("QualifyingResults", r.get("Results", []))) for r in races)
            print(f"{season} {endpoint:10s} races={len(races):2d} rows={n_rows}")
            for r in races:
                for row in r.get("QualifyingResults", r.get("Results", [])):
                    driver_ids.add(row["Driver"]["driverId"])

    debuts = fetch_debuts(driver_ids)
    print(f"debuts for {len(debuts)} drivers")


if __name__ == "__main__":
    main()
