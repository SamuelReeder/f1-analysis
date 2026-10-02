"""Turn cached Jolpica JSON into tidy Parquet tables under data/processed/.

Tables
------
events      one row per Grand Prix: season, round, date, circuit
drivers     one row per driver: name, code, date of birth
entries     one row per driver per event: constructor, team lineage, quali position (empty
            for a driver who set no qualifying time and is known only from the race results)
quali_times one row per driver per qualifying segment with a lap time (Q1/Q2/Q3), with its
            source: "jolpica", or "fastf1" for events Jolpica has no times for (filled by
            extract/quali_fill.py from FastF1 lap timing and checked against Jolpica)
race        one row per driver per race: grid, finish position, status (for later stages)
sprint_quali_times  one row per driver per sprint qualifying part (SQ1/SQ2/SQ3, 2023 onward),
            from extract/sprint_quali.py; read only by the pre-registered sprint qualifying
            test (docs/sprint_qualifying.md), not by the published model
"""

import datetime as dt
import hashlib
import json
from pathlib import Path

import pandas as pd

from .lineage import lineage_of

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "jolpica"
OUT = ROOT / "data" / "processed"
SUPPLEMENT = ROOT / "data" / "supplements" / "quali_times_fastf1.json"
SPRINT_QUALI = ROOT / "data" / "supplements" / "sprint_quali_times.json"

# 2006-2009 Q3 was run with race fuel loads, so those times are not pace.
RACE_FUEL_Q3_SEASONS = range(2006, 2010)


def parse_laptime(value: str | None) -> float | None:
    if not value:
        return None
    minutes, _, seconds = value.rpartition(":")
    try:
        return (int(minutes) * 60 if minutes else 0) + float(seconds)
    except ValueError:
        return None


def build() -> dict[str, pd.DataFrame]:
    events, drivers, entries, times, race = [], {}, [], [], []

    for path in sorted(RAW.glob("*_qualifying.json")):
        season = int(path.name[:4])
        for r in json.loads(path.read_text()):
            event_id = f"{season}-{int(r['round']):02d}"
            events.append({
                "event_id": event_id, "season": season, "round": int(r["round"]),
                "date": r["date"], "race_name": r["raceName"],
                "circuit_id": r["Circuit"]["circuitId"],
            })
            for q in r["QualifyingResults"]:
                d, c = q["Driver"], q["Constructor"]
                drivers[d["driverId"]] = {
                    "driver_id": d["driverId"], "code": d.get("code"),
                    "name": f"{d['givenName']} {d['familyName']}",
                    "dob": d.get("dateOfBirth"), "nationality": d.get("nationality"),
                }
                entries.append({
                    "event_id": event_id, "driver_id": d["driverId"],
                    "constructor_id": c["constructorId"], "constructor_name": c["name"],
                    "team": lineage_of(c["constructorId"]),
                    "quali_position": int(q["position"]),
                })
                for seg in ("Q1", "Q2", "Q3"):
                    if seg == "Q3" and season in RACE_FUEL_Q3_SEASONS:
                        continue
                    t = parse_laptime(q.get(seg))
                    if t is not None:
                        times.append({"event_id": event_id, "driver_id": d["driverId"],
                                      "segment": seg, "time_s": t})

    entered = {(e["event_id"], e["driver_id"]) for e in entries}
    event_ids = {e["event_id"] for e in events}
    for path in sorted(RAW.glob("*_results.json")):
        season = int(path.name[:4])
        for r in json.loads(path.read_text()):
            event_id = f"{season}-{int(r['round']):02d}"
            for res in r["Results"]:
                d, c = res["Driver"], res["Constructor"]
                # Jolpica omits a driver who set no qualifying time from the qualifying
                # results; a race entry shows they were entered for the event.
                if event_id in event_ids and (event_id, d["driverId"]) not in entered:
                    drivers.setdefault(d["driverId"], {
                        "driver_id": d["driverId"], "code": d.get("code"),
                        "name": f"{d['givenName']} {d['familyName']}",
                        "dob": d.get("dateOfBirth"), "nationality": d.get("nationality"),
                    })
                    entries.append({
                        "event_id": event_id, "driver_id": d["driverId"],
                        "constructor_id": c["constructorId"], "constructor_name": c["name"],
                        "team": lineage_of(c["constructorId"]), "quali_position": None,
                    })
                race.append({
                    "event_id": event_id, "driver_id": res["Driver"]["driverId"],
                    "constructor_id": res["Constructor"]["constructorId"],
                    "team": lineage_of(res["Constructor"]["constructorId"]),
                    "grid": int(res["grid"]), "position": int(res["position"]),
                    "position_text": res["positionText"], "status": res["status"],
                    "laps": int(res["laps"]),
                })

    times = pd.DataFrame(times).assign(source="jolpica")
    if SUPPLEMENT.exists():
        fill = pd.DataFrame(json.loads(SUPPLEMENT.read_text())["times"]).assign(source="fastf1")
        covered = fill.event_id.isin(times.event_id)
        if covered.any():  # Jolpica has since added times: prefer them
            print(f"supplement ignored for {sorted(fill.event_id[covered].unique())}: Jolpica has times")
        times = pd.concat([times, fill[~covered]], ignore_index=True)

    debuts = json.loads((RAW / "debuts.json").read_text())
    for driver_id, d in drivers.items():
        d["debut_season"] = debuts[driver_id]["season"]
        d["debut_date"] = debuts[driver_id]["date"]

    tables = {
        "events": pd.DataFrame(events).sort_values("event_id", ignore_index=True),
        "drivers": pd.DataFrame(drivers.values()).sort_values("driver_id", ignore_index=True),
        "entries": pd.DataFrame(entries).astype({"quali_position": "Int64"}),
        "quali_times": times,
        "race": pd.DataFrame(race),
        "sprint_quali_times": (pd.DataFrame(json.loads(SPRINT_QUALI.read_text())["times"]).assign(source="fastf1")
                               if SPRINT_QUALI.exists() else
                               pd.DataFrame(columns=["event_id", "driver_id", "segment", "time_s", "source"])),
    }

    # A driver entered twice for one event would silently double-count.
    dup = tables["entries"].duplicated(["event_id", "driver_id"])
    if dup.any():
        raise ValueError(f"duplicate entries:\n{tables['entries'][dup]}")
    return tables


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, df in build().items():
        df.to_parquet(OUT / f"{name}.parquet", index=False)
        print(f"{name:12s} {len(df):6d} rows")
    # which versions of the raw responses this build used (see fetch.py)
    manifest = RAW / "manifest.json"
    known = json.loads(manifest.read_text()) if manifest.exists() else {}
    used = sorted(RAW.glob("*_qualifying.json")) + sorted(RAW.glob("*_results.json"))
    sources = {}
    for path in used:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        entry = known.get(path.name)
        if entry and entry["sha256"] == digest:
            sources[path.name] = {"retrieved_utc": entry["retrieved_utc"], "sha256": digest}
        else:  # fetched before versioning: the file time is when it was downloaded
            mtime = dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
            sources[path.name] = {"retrieved_utc": mtime.strftime("%Y-%m-%dT%H:%M:%SZ"), "sha256": digest,
                                  "retrieved_from_file_time": True}
    (OUT / "sources.json").write_text(json.dumps({
        "jolpica": sources,
        "supplement": str(SUPPLEMENT.relative_to(ROOT)) if SUPPLEMENT.exists() else None,
        "sprint_quali_supplement": str(SPRINT_QUALI.relative_to(ROOT)) if SPRINT_QUALI.exists() else None,
    }, indent=1))


if __name__ == "__main__":
    main()
