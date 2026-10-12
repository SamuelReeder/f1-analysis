"""Unknown rainfall is not evidence of a dry session or a known race context."""
import pandas as pd
import pytest

from test_overall import tables
from f1rank.overall import design


def test_all_null_weather_does_not_enter_outcome_context():
    t, clean = tables()
    t["race_weather"]["rainfall"] = t["race_weather"].rainfall.astype("boolean")
    t["race_weather"].loc[t["race_weather"].event_id == "2022-02", "rainfall"] = pd.NA
    result = design.build("2022-02", tables=t, clean=clean)
    assert "2022-02" in result.catalog["excluded_context_events"]
    assert "2022-02" not in result.catalog["outcome_events"]


def test_all_null_weather_does_not_attempt_dry_lap_cleaning(monkeypatch):
    t, _ = tables()
    t["race_weather"]["rainfall"] = pd.Series(pd.NA, index=t["race_weather"].index, dtype="boolean")

    def unexpected_cleaning(*args):
        pytest.fail("An unknown-weather event reached dry-lap cleaning")

    monkeypatch.setattr(design, "clean_laps", unexpected_cleaning)
    with pytest.raises(ValueError, match="No eligible dry laps"):
        design.clean_training_laps(t)
