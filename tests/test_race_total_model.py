"""Covariance and race-entry regressions for total race pace."""
import numpy as np
import pandas as pd

from f1rank import race_total_model as model


def test_race_entries_include_drivers_absent_from_qualifying(monkeypatch):
    drivers = [f"driver{i}" for i in range(8)]
    raw = pd.DataFrame(dict(event_id=["2026-01"] * 8, driver_id=drivers, team=["Display name"] * 8,
                            compound=["HARD"] * 8, stint_key=drivers, lap_number=[2] * 8))
    race = pd.DataFrame(dict(event_id=["2026-01"] * 8, driver_id=drivers, team=["lineage"] * 8))
    frames = {"race_laps.parquet": raw, "timeline.parquet": pd.DataFrame({"event_id": []}),
              "race_weather.parquet": pd.DataFrame({"event_id": ["2026-01"], "rainfall": [False]}),
              "race.parquet": race}
    monkeypatch.setattr(model.pd, "read_parquet", lambda path: frames[path.name])
    monkeypatch.setattr(model, "clean_laps", lambda laps, timeline: laps)
    result = model.laps()
    assert set(result.driver_id) == set(drivers)
    assert set(result.team) == {"lineage"}


def test_predictive_car_uncertainty_is_shared_by_teammates():
    n = 3200
    post = dict(package=np.zeros((n, 1)), sd_day=np.zeros(n), sd_car_day=np.full(n, .4))
    meta = dict(diagnostics={"n_draws": n}, catalog={"drivers": [], "driver_seasons": [], "cars": ["2025|old"]})
    # Both the unknown car and its race-day deviation are shared within a team.
    entries = pd.DataFrame(dict(event_id=["2026-01", "2026-01", "2026-02"],
                                driver_id=["a", "b", "a"], team=["new"] * 3))
    draws = model.predict(post, meta, entries, noise=True)
    np.testing.assert_array_equal(draws[:, 0], draws[:, 1])
    assert np.std(draws[:, 0]) > 1
    # The package persists across races; the car-day departure does not.
    assert .5 < np.std(draws[:, 0] - draws[:, 2]) < .65


def test_new_season_form_carries_uncertainty_not_a_zero_rating():
    n = 3200
    post = dict(skill=np.full((n, 1), .2), form=np.full((n, 1), .1), sd_form=np.full(n, .3))
    meta = dict(diagnostics={"n_draws": n},
                catalog={"drivers": ["a"], "driver_seasons": ["2025|a"], "cars": []})
    entries = pd.DataFrame(dict(event_id=["2025-01", "2026-01", "2026-02"],
                                driver_id=["a"] * 3, team=["x"] * 3))
    draws = model.predict(post, meta, entries)
    np.testing.assert_allclose(draws[:, 0], .3)
    assert .27 < draws[:, 1].std() < .33
    np.testing.assert_array_equal(draws[:, 1], draws[:, 2])
