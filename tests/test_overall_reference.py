import importlib.util
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location("overall_reference", Path(__file__).parents[1] / "analysis/overall_reference_calendar.py")
reference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference)


def tables():
    return dict(events=pd.DataFrame(dict(event_id=["2025-01", "2025-02", "2026-01"],
                season=[2025, 2025, 2026], circuit_id=["a", "b", "c"])),
        race=pd.DataFrame(dict(event_id=["2025-01", "2025-01", "2025-02", "2026-01"],
                position=[1, 2, 1, 1], laps=[60, 59, 50, 70])),
        race_weather=pd.DataFrame(dict(event_id=["2025-01", "2025-02", "2026-01"], rainfall=[False, True, False])),
        race_laps=pd.DataFrame(dict(event_id=["2025-01", "2025-02", "2026-01"], compound=["HARD", "INTERMEDIATE", "HARD"])),
        timeline=pd.DataFrame(dict(event_id=["2025-02"], kind=["safety_car"])))


def test_calendar_uses_every_reference_event_equally_and_no_driver_ranking():
    result = reference.construct(tables())
    assert [e["event_id"] for e in result["events"]] == ["2025-01", "2025-02"]
    assert sum(e["weight"] for e in result["events"]) == 1
    assert result["events"][1]["wet"] and result["events"][1]["neutralised"]
    assert [e["reference_distance_laps"] for e in result["events"]] == [60, 50]
    assert all("driver_id" not in e for e in result["events"])


def test_missing_reference_data_fails_without_dropping_a_race():
    t = tables()
    t["race_weather"] = t["race_weather"].iloc[:1]
    with pytest.raises(ValueError, match="Missing race_weather"):
        reference.construct(t)


def test_later_outcomes_cannot_change_frozen_reference_mix():
    t = tables()
    first = reference.construct(t)
    t["race"].loc[t["race"].event_id == "2026-01", "laps"] = 500
    t["race_weather"].loc[t["race_weather"].event_id == "2026-01", "rainfall"] = True
    assert reference.construct(t) == first
