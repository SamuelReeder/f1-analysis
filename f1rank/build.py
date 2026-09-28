"""Turn cached Jolpica JSON into tidy Parquet tables under data/processed/.

Tables
------
events      one row per Grand Prix: season, round, date, circuit
drivers     one row per driver: name, code, date of birth
entries     one row per driver per event: constructor, team lineage, quali position
quali_times one row per driver per qualifying segment with a lap time (Q1/Q2/Q3)
race        one row per driver per race: grid, finish position, status (for later stages)
"""

import json
from pathlib import Path

import pandas as pd

from .lineage import lineage_of

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "jolpica"
OUT = ROOT / "data" / "processed"

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

    for path in sorted(RAW.glob("*_results.json")):
        season = int(path.name[:4])
        for r in json.loads(path.read_text()):
            event_id = f"{season}-{int(r['round']):02d}"
            for res in r["Results"]:
                race.append({
                    "event_id": event_id, "driver_id": res["Driver"]["driverId"],
                    "constructor_id": res["Constructor"]["constructorId"],
                    "team": lineage_of(res["Constructor"]["constructorId"]),
                    "grid": int(res["grid"]), "position": int(res["position"]),
                    "position_text": res["positionText"], "status": res["status"],
                    "laps": int(res["laps"]),
                })

    debuts = json.loads((RAW / "debuts.json").read_text())
    for driver_id, d in drivers.items():
        d["debut_season"] = debuts[driver_id]["season"]
        d["debut_date"] = debuts[driver_id]["date"]

    tables = {
        "events": pd.DataFrame(events).sort_values("event_id", ignore_index=True),
        "drivers": pd.DataFrame(drivers.values()).sort_values("driver_id", ignore_index=True),
        "entries": pd.DataFrame(entries),
        "quali_times": pd.DataFrame(times),
        "race": pd.DataFrame(race),
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


if __name__ == "__main__":
    main()
