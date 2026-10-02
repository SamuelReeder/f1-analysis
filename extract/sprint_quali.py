"""Sprint qualifying times (2023 onward) from FastF1 lap timing, for the pre-registered
sprint qualifying test (docs/sprint_qualifying.md).

Sprint qualifying (the 2023 "Sprint Shootout", "Sprint Qualifying" since 2024) is a
one-lap session in three parts, SQ1/SQ2/SQ3, like Q1/Q2/Q3. Jolpica has no times for it
and FastF1's results carry none, so each driver's best lap per part is rebuilt from lap
timing with the method of extract/quali_fill.py (laps FastF1 flags as deleted by race
control are excluded); on the 2025 reference events that method matched 1,014 of 1,015
Jolpica qualifying times (outputs/analysis/quali_fill/check.json). In addition, a lap
named in a "LAP DELETED" message is excluded unless the session has a "REINSTATED"
message for that car (the first extraction kept one such lap, 2025-02 HUL lap 2, as a
best). Every "LAP DELETED" message is listed with whether its lap was excluded or kept.

Drivers are identified by their abbreviation in that weekend's sprint results
(data/raw/fastf1_tables/<event>_S/results.parquet, which carry Jolpica driver ids).

Runs in the FastF1 environment:

    .venv-fastf1/bin/python extract/sprint_quali.py

Writes data/supplements/sprint_quali_times.json (read by f1rank.build).
"""

import datetime as dt
import json
import re

import fastf1
import pandas as pd

from ff1 import ROOT, enable_cache, patient
from quali_fill import LAP_DELETED

PROCESSED = ROOT / "data" / "processed"
CAR = re.compile(r"CAR (\d+) ")
TABLES = ROOT / "data" / "raw" / "fastf1_tables"
SUPPLEMENT = ROOT / "data" / "supplements" / "sprint_quali_times.json"
FIRST_SEASON = 2023  # 2021-2022 sprint weekends had no separate qualifying session
NAMES = ("Sprint Shootout", "Sprint Qualifying")


def session_name(season: int, rnd: int) -> str:
    event = fastf1.get_event(season, rnd)
    names = [event[f"Session{i}"] for i in range(1, 6) if event[f"Session{i}"] in NAMES]
    if len(names) != 1:
        raise ValueError(f"{season} round {rnd}: no single sprint qualifying session ({names})")
    return names[0]


def segment_bests(event_id: str, season: int, rnd: int, entrants: set[str]) -> tuple[pd.DataFrame, dict]:
    name = session_name(season, rnd)
    s = fastf1.get_session(season, rnd, name)
    patient(s.load, laps=True, telemetry=False, weather=False, messages=True)
    ids = pd.read_parquet(TABLES / f"{event_id}_S" / "results.parquet").set_index("Abbreviation").DriverId
    messages = s.race_control_messages.Message
    reinstated = {m.group(1) for text in messages if "REINSTATED" in text and (m := CAR.search(text))}
    # laps named in a deletion message, unless that car has a reinstatement message
    named = {(m.group(2), int(m.group(3))) for text in messages
             if (m := LAP_DELETED.search(text)) and m.group(1) not in reinstated}
    rows, kept, by_message = [], set(), []
    for seg, laps in zip(("SQ1", "SQ2", "SQ3"), s.laps.split_qualifying_sessions()):
        if laps is None:
            continue
        laps = laps[~laps.Deleted.fillna(False).astype(bool)].dropna(subset=["LapTime"])
        drop = pd.Series([(d, int(n)) in named for d, n in zip(laps.Driver, laps.LapNumber)], index=laps.index)
        by_message += [f"{seg} {d} lap {int(n)}" for d, n in zip(laps.Driver[drop], laps.LapNumber[drop])]
        laps = laps[~drop]
        best = laps.loc[laps.groupby("Driver").LapTime.idxmin()]
        kept |= {(r.Driver, int(r.LapNumber)) for r in best.itertuples()}
        for r in best.itertuples():
            if r.Driver not in ids.index or not ids[r.Driver]:
                raise ValueError(f"{event_id}: {r.Driver} is not in the sprint results")
            rows.append({"event_id": event_id, "driver_id": ids[r.Driver], "segment": seg,
                         "time_s": round(r.LapTime.total_seconds(), 3)})
    deletions = [{"message": text, "lap_is_a_kept_best": (m.group(2), int(m.group(3))) in kept}
                 for text in messages if (m := LAP_DELETED.search(text))]
    out = pd.DataFrame(rows, columns=["event_id", "driver_id", "segment", "time_s"])
    unknown = set(out.driver_id) - entrants
    if unknown:
        raise ValueError(f"{event_id}: FastF1 drivers not entered in Jolpica: {unknown}")
    return out, {"session": name, "messages": deletions, "excluded_only_by_message": by_message}


def main() -> None:
    enable_cache()
    events = pd.read_parquet(PROCESSED / "events.parquet")
    entries = pd.read_parquet(PROCESSED / "entries.parquet")
    sprints = {p.name[:-2] for p in TABLES.glob("*_S") if (p / "results.parquet").exists()}
    events = events[(events.season >= FIRST_SEASON) & events.event_id.isin(sprints)]
    times, deletions = [], {}
    for ev in events.itertuples():
        t, deletions[ev.event_id] = segment_bests(
            ev.event_id, ev.season, ev.round, set(entries[entries.event_id == ev.event_id].driver_id))
        times += t.to_dict("records")
        print(f"{ev.event_id}: {len(t)} times, {t.segment.nunique()} parts", flush=True)
    SUPPLEMENT.parent.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT.write_text(json.dumps({
        "source": "FastF1 lap timing (F1 live timing): best non-deleted lap per driver and sprint "
                  "qualifying part (method of extract/quali_fill.py)",
        "fastf1_version": fastf1.__version__,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "events": events.event_id.tolist(),
        "lap_deletions": deletions,
        "times": times,
    }, indent=1))
    flat = [m for v in deletions.values() for m in v["messages"]]
    extra = [f"{e} {x}" for e, v in deletions.items() for x in v["excluded_only_by_message"]]
    print(f"{len(times)} times from {len(events)} events; {len(flat)} deletion messages, "
          f"{sum(m['lap_is_a_kept_best'] for m in flat)} named laps kept as a best; "
          f"excluded only by message: {extra}")


if __name__ == "__main__":
    main()
