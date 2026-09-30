"""Timeline rules on small hand-made inputs (no downloaded data needed)."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from f1rank import timeline as tl


def msgs(*rows):
    return pd.DataFrame([{"time": pd.Timestamp("2024-01-01") + pd.Timedelta(minutes=i), "message": m, "lap": lap}
                         for i, (lap, m) in enumerate(rows)])


NUMBERS = {"1": "max_verstappen", "4": "norris", "44": "hamilton"}


def test_incident_messages_are_grouped_and_classified():
    rows = tl.message_events("2024-01", msgs(
        (5, "TURN 1 INCIDENT INVOLVING CARS 1 (VER) AND 4 (NOR) NOTED - CAUSING A COLLISION"),
        (7, "FIA STEWARDS: TURN 1 INCIDENT INVOLVING CARS 1 (VER) AND 4 (NOR) UNDER INVESTIGATION - CAUSING A COLLISION"),
        (9, "FIA STEWARDS: 5 SECOND TIME PENALTY FOR CAR 1 (VER) - CAUSING A COLLISION"),
        (10, "INCIDENT INVOLVING CAR 44 (HAM) NOTED - TRACK LIMITS"),
        (12, "CAR 44 (HAM) STOPPED AT TURN 4"),
        (13, "CAR 4 (NOR) SPUN AND CONTINUED AT TURN 2"),
    ), NUMBERS)
    by = {(r["kind"], r["driver_id"]): r for r in rows}
    inc = by["incident_noted", "max_verstappen"]
    assert inc["lap_start"] == 5 and inc["involved"] == "norris" and inc["detail"].startswith("contact")
    assert len(json.loads(inc["evidence"])) == 2 and "under investigation" in inc["detail"]
    assert by["penalty", "max_verstappen"]["reason"] == "CAUSING A COLLISION"
    assert by["incident_noted", "hamilton"]["detail"].startswith("other")  # track limits is not contact
    assert by["stoppage", "hamilton"]["lap_start"] == 12
    assert by["off_track", "norris"]["detail"] == "spun and continued"
    assert sum(r["kind"] == "incident_noted" for r in rows) == 3


def test_neutralisation_periods():
    ends = np.arange(1, 11) * 90.0
    laps = pd.DataFrame({"lap_number": np.arange(1, 11), "time": ends})
    ts = pd.DataFrame({"time": [100.0, 400.0, 500.0], "status": ["4", "1", "5"]})
    rows = tl.neutralisations("2024-01", ts, tl.leader_lap(laps))
    sc, red = rows
    assert (sc["kind"], sc["lap_start"], sc["lap_end"]) == ("safety_car", 2, 5)
    assert (red["kind"], red["lap_start"], red["lap_end"]) == ("red_flag", 6, None)


def test_retirement_lap_adds_laps_missing_from_fastf1():
    laps = pd.DataFrame({"driver_id": "x", "lap_number": [1, 2, 3], "time": [90.0, 180.0, 270.0]})
    leader = pd.DataFrame({"lap_number": np.arange(1, 11), "time": np.arange(1, 11) * 85.0})
    lap_at = tl.leader_lap(leader)
    assert tl.race_lap_of_retirement(SimpleNamespace(driver_id="x", laps=3), laps, lap_at) == lap_at(270.0)
    assert tl.race_lap_of_retirement(SimpleNamespace(driver_id="x", laps=5), laps, lap_at) == lap_at(270.0) + 2
    assert tl.race_lap_of_retirement(SimpleNamespace(driver_id="x", laps=0), laps, lap_at) == 1


def test_cause_probabilities_sum_to_one():
    rng = np.random.default_rng(0)
    R = pd.DataFrame({f: rng.random(12) < 0.4 for f in tl.FEATURES})
    R["status"] = ["Engine", "Collision", "Retired", "Puncture", "Retired", "Accident"] * 2
    R["cls"] = R.status.map(tl.status_class)
    R["penalised"] = rng.random(12) < 0.3
    R["other_penalised"] = rng.random(12) < 0.3
    R["event_id"], R["driver_id"], R["fastf1"] = "2024-01", [f"d{i}" for i in range(12)], True
    W = tl.fit_cause_model(R[tl.FEATURES].to_numpy(float), rng.integers(0, 3, 12))
    C = tl.cause_probabilities(R, W, np.full(3, 1 / 3))
    np.testing.assert_allclose(C[tl.P_COLS].sum(axis=1), 1.0)
    assert C.loc[0, "p_mechanical"] == pytest.approx(tl.CODED_CONFIDENCE)


def test_overrides_must_match_a_row(tmp_path, monkeypatch):
    T = pd.DataFrame([tl._row("2024-01", "x", 5, None, "retirement", "Retired", [], p_unknown=1.0)])
    T["reviewed"], T["cause_basis"] = False, "inferred from evidence"
    path = tmp_path / "ov.csv"
    cols = "event_id,driver_id,kind,lap_start," + ",".join(tl.P_COLS) + ",note,reviewer,reviewed_on\n"
    path.write_text(cols + "2024-01,x,retirement,5,0,1,0,0,0,0,onboard video,me,2026-09-28\n")
    monkeypatch.setattr(tl, "OVERRIDES", path)
    out = tl.apply_overrides(T.copy())
    assert out.loc[0, "p_own_error"] == 1 and out.loc[0, "reviewed"] and out.loc[0, "cause_basis"] == "reviewed"
    path.write_text(cols + "2024-01,x,retirement,6,0,1,0,0,0,0,wrong lap,me,2026-09-28\n")
    with pytest.raises(ValueError):
        tl.apply_overrides(T.copy())


def test_withdrawn_cars_by_laps_completed():
    row = lambda status, text, laps: SimpleNamespace(status=status, position_text=text, laps=laps)  # noqa: E731
    assert tl.admin_kind(row("Engine", "W", 0)) == "did_not_start"     # failed before the start
    assert tl.admin_kind(row("Withdrew", "W", 0)) == "withdrew"
    assert tl.admin_kind(row("Withdrew", "W", 47)) is None             # raced, then retired
    assert tl.status_class("Withdrew") == "other" and "Withdrew" in tl.UNCODED
    assert tl.admin_kind(row("Engine", "R", 12)) is None
    assert tl.admin_kind(row("Disqualified", "D", 57)) == "disqualified"
