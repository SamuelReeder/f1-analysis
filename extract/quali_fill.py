"""Fill qualifying times missing from Jolpica with times rebuilt from FastF1 lap timing.

Jolpica has no Q1/Q2/Q3 times for some events (currently 2025 Miami: every time is
blank). For such events (2018 onward, where FastF1 timing exists), this rebuilds each
driver's best lap per segment from FastF1's lap timing: the session is split into
Q1/Q2/Q3 by FastF1, laps FastF1 flags as deleted by race control are excluded, and the
fastest remaining lap per driver and segment is kept. FastF1 does not attach every
race-control deletion to a lap (parsing the messages directly did worse in the check:
deletions can be reinstated), and times removed after the session are not visible in lap
timing, so the check lists every time where the method and Jolpica disagree, and every
"LAP DELETED" message of a filled event with whether the lap it names was kept.

The same method is checked against Jolpica on every other event of the same seasons, and
the agreement is written out, so the fill can be judged before it is used.

Runs in the FastF1 environment (FastF1 requires pandas < 3):

    .venv-fastf1/bin/python extract/quali_fill.py

Reads data/processed/ (run f1rank.build first). Writes:

    data/supplements/quali_times_fastf1.json  filled times with provenance (read by f1rank.build)
    outputs/analysis/quali_fill/check.json    agreement with Jolpica on the reference events
"""

import datetime as dt
import json
import re

import fastf1
import pandas as pd

from ff1 import ROOT, enable_cache, patient

PROCESSED = ROOT / "data" / "processed"
SUPPLEMENT = ROOT / "data" / "supplements" / "quali_times_fastf1.json"
CHECK = ROOT / "outputs" / "analysis" / "quali_fill" / "check.json"
FIRST_TIMING_SEASON = 2018
EXACT = 0.0005  # seconds: times are given to the millisecond


LAP_DELETED = re.compile(r"CAR (\d+) \((\w+)\) LAP DELETED.* LAP (\d+)")


def segment_bests(season: int, rnd: int, entrants: set[str]) -> tuple[pd.DataFrame, list[dict]]:
    """Best non-deleted lap per driver and segment, from FastF1 lap timing; and every
    'LAP DELETED' race-control message, with whether the lap it names is a kept best."""
    s = fastf1.get_session(season, rnd, "Q")
    patient(s.load, laps=True, telemetry=False, weather=False, messages=True)
    ids = s.results.set_index("Abbreviation").DriverId
    rows, kept = [], set()
    for seg, laps in zip(("Q1", "Q2", "Q3"), s.laps.split_qualifying_sessions()):
        if laps is None:
            continue
        laps = laps[~laps.Deleted.fillna(False).astype(bool)].dropna(subset=["LapTime"])
        best = laps.loc[laps.groupby("Driver").LapTime.idxmin()]
        kept |= {(r.Driver, int(r.LapNumber)) for r in best.itertuples()}
        for r in best.itertuples():
            rows.append({"driver_id": ids[r.Driver], "segment": seg, "time_s": round(r.LapTime.total_seconds(), 3)})
    deletions = [{"message": text, "lap_is_a_kept_best": (m.group(2), int(m.group(3))) in kept}
                 for text in s.race_control_messages.Message if (m := LAP_DELETED.search(text))]
    out = pd.DataFrame(rows, columns=["driver_id", "segment", "time_s"])
    unknown = set(out.driver_id) - entrants
    if unknown:
        raise ValueError(f"{season} round {rnd}: FastF1 drivers not entered in Jolpica: {unknown}")
    return out, deletions


def main() -> None:
    enable_cache()
    events = pd.read_parquet(PROCESSED / "events.parquet")
    entries = pd.read_parquet(PROCESSED / "entries.parquet")
    times = pd.read_parquet(PROCESSED / "quali_times.parquet")
    if "source" in times:
        times = times[times.source == "jolpica"]
    events = events[events.season >= FIRST_TIMING_SEASON]
    has_times = set(times.event_id)
    gaps = events[~events.event_id.isin(has_times)]
    print(f"events without Jolpica times: {gaps.event_id.tolist()}")
    reference = events[events.season.isin(gaps.season) & events.event_id.isin(has_times)]

    compared, mismatches, only_fastf1, only_jolpica = 0, [], [], []
    for ev in reference.itertuples():
        mine, _ = segment_bests(ev.season, ev.round, set(entries[entries.event_id == ev.event_id].driver_id))
        theirs = times[times.event_id == ev.event_id]
        m = mine.merge(theirs, on=["driver_id", "segment"], how="outer", suffixes=("_fastf1", "_jolpica"),
                       indicator=True)
        both = m[m._merge == "both"]
        compared += len(both)
        bad = both[(both.time_s_fastf1 - both.time_s_jolpica).abs() > EXACT]
        mismatches += [{"event_id": ev.event_id, **r} for r in
                       bad[["driver_id", "segment", "time_s_fastf1", "time_s_jolpica"]].to_dict("records")]
        only_fastf1 += [f"{ev.event_id}|{r.driver_id}|{r.segment}" for r in m[m._merge == "left_only"].itertuples()]
        only_jolpica += [f"{ev.event_id}|{r.driver_id}|{r.segment}" for r in m[m._merge == "right_only"].itertuples()]
        print(f"{ev.event_id}: {len(both)} compared, {len(bad)} differ", flush=True)

    check = {
        "reference_events": reference.event_id.tolist(),
        "n_compared": compared,
        "n_exact": compared - len(mismatches),
        "share_exact": (compared - len(mismatches)) / compared if compared else None,
        "mismatches": mismatches,
        "only_in_fastf1": only_fastf1,
        "only_in_jolpica": only_jolpica,
    }
    CHECK.parent.mkdir(parents=True, exist_ok=True)
    CHECK.write_text(json.dumps(check, indent=1))

    filled, deletions = [], {}
    for ev in gaps.itertuples():
        t, deletions[ev.event_id] = segment_bests(ev.season, ev.round,
                                                  set(entries[entries.event_id == ev.event_id].driver_id))
        filled += [{"event_id": ev.event_id, **r} for r in t.to_dict("records")]
        print(f"{ev.event_id}: filled {len(t)} times", flush=True)
    # a deleted lap FastF1 did not flag would show up here as a kept best
    check["filled_event_lap_deletions"] = deletions
    CHECK.write_text(json.dumps(check, indent=1))
    SUPPLEMENT.parent.mkdir(parents=True, exist_ok=True)
    SUPPLEMENT.write_text(json.dumps({
        "source": "FastF1 lap timing (F1 live timing): best non-deleted lap per driver and segment",
        "fastf1_version": fastf1.__version__,
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "check": str(CHECK.relative_to(ROOT)),
        "events": gaps.event_id.tolist(),
        "times": filled,
    }, indent=1))
    print(json.dumps({k: v for k, v in check.items() if k not in ("reference_events",)}, indent=1))


if __name__ == "__main__":
    main()
