"""The archive adapter preserves the existing race-stage inputs and admission rules."""
import pandas as pd
import pytest

from analysis import championship_archive as archive
from f1rank import benchmark, qualifying


def test_lap_admission_requires_a_pair_and_sufficient_clean_drivers():
    rows = pd.DataFrame({"driver_id": list("abcdefghij"),
                         "team": ["A", "A", "B", "B", "C", "C", "D", "D", "E", "E"]})
    assert archive.admit_race(rows)
    assert not archive.admit_race(rows.iloc[:-1])
    assert not archive.admit_race(rows.assign(team=rows.driver_id))


def test_archive_race_adapter_matches_canonical_mapping_with_the_registered_start(tmp_path, monkeypatch):
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    race = pd.DataFrame({
        "event_id": ["2005-01", "2006-01", "2007-01", "2007-01", "2007-01", "2008-01"],
        "driver_id": ["a", "a", "a", "b", "c", "a"], "team": ["A", "A", "A", "B", "C", "A"],
        "grid": [1, 1, 2, 1, 0, 1], "position": [1, 1, 2, 1, 3, 1], "status": ["Finished"] * 6})
    race.to_parquet(processed / "race.parquet", index=False)
    drivers = race[["event_id", "driver_id"]].assign(driver=0.1)
    cars = race[["event_id", "team"]].assign(car=0.2)
    monkeypatch.setattr(archive, "ROOT", tmp_path)
    monkeypatch.setattr(benchmark, "PROCESSED", processed)
    monkeypatch.setattr(benchmark, "FIRST_SEASON", 2006)
    monkeypatch.setattr(qualifying, "features", lambda season: (drivers, cars))
    actual = archive.race_frame(2007, 2006, drivers, cars)
    expected = benchmark.race_orders(2007)
    pd.testing.assert_frame_equal(actual, expected)
    assert set(actual.event_id) == {"2006-01", "2007-01"}
    assert actual.loc[actual.driver_id == "c", "grid_slot"].tolist() == [3]
    with pytest.raises(ValueError, match="Incomplete"):
        archive.race_frame(2007, 2006, drivers[drivers.event_id != "2007-01"], cars)
