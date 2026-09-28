"""Download qualifying and race results from the Jolpica (Ergast-compatible) API.

Raw JSON pages are cached under data/raw/jolpica/ so reruns only fetch what is
missing. The current season is always refetched because new rounds appear, and
`--refresh-all` refetches every season (sources revise past data, e.g. retirement
statuses). When a refetch changes a file, the previous version is kept in
data/raw/jolpica/archive/ under its retrieval time. manifest.json records each file's
retrieval time and sha256; `build` copies the entries it used to data/processed/sources.json.
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


def _get(url: str) -> dict:
    global _last_request
    for attempt in range(8):
        wait = MIN_INTERVAL - (time.monotonic() - _last_request)
        if wait > 0:
            time.sleep(wait)
        _last_request = time.monotonic()
        resp = requests.get(url, timeout=30)
        if resp.status_code == 429 or resp.status_code >= 500:
            time.sleep(min(60, 2 ** attempt))
            continue
        resp.raise_for_status()
        return resp.json()["MRData"]
    raise RuntimeError(f"giving up on {url}")


def fetch_table(season: int, endpoint: str, refresh: bool = False) -> list[dict]:
    """Return every race in a season for an endpoint, merging paginated rows by round."""
    out = RAW / f"{season}_{endpoint}.json"
    if out.exists() and not refresh:
        return json.loads(out.read_text())

    list_key = {"qualifying": "QualifyingResults", "results": "Results"}[endpoint]
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
    args = parser.parse_args()

    driver_ids = set()
    for season in range(args.first, args.last + 1):
        refresh = args.refresh_all or season == dt.date.today().year
        for endpoint in ("qualifying", "results"):
            races = fetch_table(season, endpoint, refresh=refresh)
            n_rows = sum(len(r.get("QualifyingResults", r.get("Results", []))) for r in races)
            print(f"{season} {endpoint:10s} races={len(races):2d} rows={n_rows}")
            for r in races:
                for row in r.get("QualifyingResults", r.get("Results", [])):
                    driver_ids.add(row["Driver"]["driverId"])

    debuts = fetch_debuts(driver_ids)
    print(f"debuts for {len(debuts)} drivers")


if __name__ == "__main__":
    main()
