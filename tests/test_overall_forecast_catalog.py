"""Forecast-field boundaries, without test-weekend outcomes or fitted ranks."""
import copy

import pytest

from analysis.overall_forecast_catalog import build_catalog
from f1rank.overall.design import build
from test_overall import tables


@pytest.fixture(scope="module")
def design():
    source, clean = tables()
    return build("2022-02", tables=source, clean=clean)


@pytest.fixture
def request_snapshot(design):
    field = design.qualifying.entries.query("event_id == '2022-02'")
    return dict(schema_version=1, information_set="prequalifying", origin_at="2022-03-04T12:00:00Z",
        source=dict(kind="archived_asof", captured_at="2022-03-03T12:00:00Z", reference="synthetic announcement"),
        event=dict(event_id="2022-03", circuit_id="track", qualifying_start="2022-03-05T14:00:00Z",
                   race_start="2022-03-06T14:00:00Z", scheduled_laps=60),
        calendar_after_cutoff=["2022-03"],
        entrants=[dict(driver_id=row.driver_id, team=row.team, prior_entries_after_cutoff=0)
                  for row in field.itertuples()])


def test_complete_field_is_retained_even_without_a_qualifying_measurement(design, request_snapshot):
    result = build_catalog(design, request_snapshot)
    assert result["field"] == [f"d{i}" for i in range(6)]
    missing_qualifier = next(r for r in result["entries"] if r["driver_id"] == "d5")
    assert missing_qualifier["latest_entry_index"] is not None
    assert not missing_qualifier["new_driver"]
    assert missing_qualifier["event_gap"] == 1
    assert result["future_weather"] == result["future_neutralisation"] == "unknown"
    assert not result["prediction_available"] and not result["publication_allowed"]
    assert not result["source_availability_verified"]


def test_driver_order_and_equivalent_timezones_do_not_change_the_catalogue(design, request_snapshot):
    original = build_catalog(design, request_snapshot)
    request_snapshot["entrants"].reverse()
    request_snapshot["origin_at"] = "2022-03-04T07:00:00-05:00"
    assert build_catalog(design, request_snapshot) == original


def test_unseen_driver_car_and_circuit_keep_explicit_missing_state_references(design, request_snapshot):
    request_snapshot["entrants"].append(dict(driver_id="rookie", team="new_team", dob="2002-01-01",
        debut_season=2022, experience_before_cutoff=0, prior_entries_after_cutoff=0))
    request_snapshot["event"]["circuit_id"] = "new_circuit"
    result = build_catalog(design, request_snapshot)
    rookie = next(row for row in result["entries"] if row["driver_id"] == "rookie")
    assert rookie["new_driver"] and rookie["new_car"] and rookie["new_team_era"]
    assert rookie["driver_index"] is rookie["latest_entry_index"] is rookie["latest_car_index"] is None
    assert rookie["era_stint_index"] is None
    assert result["new_circuit"] and result["circuit_index"] is None
    assert "rookie" in result["field"]


def test_unknown_history_does_not_imply_rookie_experience(design, request_snapshot):
    request_snapshot["entrants"].append(dict(driver_id="returning", team="t0", dob="1980-01-01",
        debut_season=2000, experience_before_cutoff=123, prior_entries_after_cutoff=0))
    result = build_catalog(design, request_snapshot)
    driver = next(r for r in result["entries"] if r["driver_id"] == "returning")
    assert driver["new_driver"] and driver["experience"] == 123
    assert driver["latest_entry_index"] is None and driver["latest_car_index"] is not None


def test_team_transfer_retains_portable_history_and_identifies_the_new_stint(design, request_snapshot):
    row = next(r for r in request_snapshot["entrants"] if r["driver_id"] == "d0")
    row["team"] = "t2"
    result = build_catalog(design, request_snapshot)
    driver = next(r for r in result["entries"] if r["driver_id"] == "d0")
    assert driver["latest_entry_index"] is not None and driver["new_team_era"]
    assert driver["era_stint_index"] is None and not driver["new_driver"]
    assert driver["latest_car_index"] is not None


def test_teammates_share_new_car_identity(design, request_snapshot):
    request_snapshot["entrants"][0]["team"] = "new_team"
    request_snapshot["entrants"][1]["team"] = "new_team"
    result = build_catalog(design, request_snapshot)
    same = [r for r in result["entries"] if r["team"] == "new_team"]
    assert len(same) == 2 and same[0]["car_key"] == same[1]["car_key"]
    assert all(r["new_car"] and r["latest_car_index"] is None for r in same)


def test_calendar_gap_and_announced_participation_are_separate(design, request_snapshot):
    request_snapshot["calendar_after_cutoff"] = ["2022-03", "2022-04", "2022-05"]
    request_snapshot["event"]["event_id"] = "2022-05"
    request_snapshot["entrants"][0]["prior_entries_after_cutoff"] = 2
    result = build_catalog(design, request_snapshot)
    assert all(row["event_gap"] == 3 for row in result["entries"])
    for row in result["entries"]:
        history = design.qualifying.entries[design.qualifying.entries.driver_id == row["driver_id"]]
        before = int(history.sort_values("event_idx").iloc[-1].experience) + 1
        expected = before + (2 if row["driver_id"] == request_snapshot["entrants"][0]["driver_id"] else 0)
        assert row["experience"] == expected


def test_regulation_change_does_not_reuse_an_old_team_era_effect(design, request_snapshot):
    request_snapshot["calendar_after_cutoff"] = ["2023-01", "2024-01", "2025-01", "2026-01"]
    request_snapshot["event"].update(event_id="2026-01", qualifying_start="2026-03-05T14:00:00Z",
                                     race_start="2026-03-06T14:00:00Z")
    request_snapshot["origin_at"] = "2026-03-04T12:00:00Z"
    result = build_catalog(design, request_snapshot)
    assert all(row["new_team_era"] and row["car_reset"] and row["car_season_start"] for row in result["entries"])
    assert all(row["season_gap"] == 4 for row in result["entries"])


def test_actual_grid_is_explicit_and_multiple_pit_lane_starts_are_retained(design, request_snapshot):
    request_snapshot["information_set"] = "actual_grid"
    request_snapshot["origin_at"] = "2022-03-06T13:00:00Z"
    for i, row in enumerate(request_snapshot["entrants"]):
        row["grid_position"] = i + 1 if i < 4 else 0
    result = build_catalog(design, request_snapshot)
    assert len(result["entries"]) == 6
    assert [r["grid_feature_slot"] for r in result["entries"]] == [1, 2, 3, 4, 6, 6]


@pytest.mark.parametrize("location,key,value", [
    ("entrant", "position", 1), ("entrant", "laps", 60), ("entrant", "status", "Finished"),
    ("entrant", "quali_time", 80.), ("entrant", "grid_position", 1),
    ("event", "wet", False), ("event", "neutralised", True), ("event", "winner_laps", 60),
])
def test_realized_outcomes_and_disallowed_origin_features_are_rejected(design, request_snapshot, location, key, value):
    row = request_snapshot["entrants"][0] if location == "entrant" else request_snapshot["event"]
    row[key] = value
    with pytest.raises(ValueError, match="unsupported forecast fields"):
        build_catalog(design, request_snapshot)


@pytest.mark.parametrize("change", ["duplicate_driver", "missing_metadata", "metadata_conflict",
    "origin_after_qualifying", "origin_after_race", "naive_time", "late_snapshot", "calendar_gap",
    "too_many_entries", "future_birth", "future_debut", "boolean_experience", "known_event"])
def test_invalid_or_future_information_cannot_be_silently_repaired(design, request_snapshot, change):
    first = request_snapshot["entrants"][0]
    if change == "duplicate_driver": request_snapshot["entrants"].append(dict(first))
    elif change == "missing_metadata": first["driver_id"] = "unknown"
    elif change == "metadata_conflict": first["dob"] = "2001-01-01"
    elif change == "origin_after_qualifying": request_snapshot["origin_at"] = "2022-03-05T15:00:00Z"
    elif change == "origin_after_race": request_snapshot["origin_at"] = "2022-03-06T15:00:00Z"
    elif change == "naive_time": request_snapshot["origin_at"] = "2022-03-04T12:00:00"
    elif change == "late_snapshot": request_snapshot["source"]["captured_at"] = "2022-03-07T12:00:00Z"
    elif change == "calendar_gap":
        request_snapshot["event"]["event_id"] = "2022-04"
        request_snapshot["calendar_after_cutoff"] = ["2022-04"]
    elif change == "too_many_entries": first["prior_entries_after_cutoff"] = 1
    elif change == "boolean_experience": first["prior_entries_after_cutoff"] = True
    elif change == "known_event": request_snapshot["event"]["event_id"] = "2022-02"
    else:
        first.update(driver_id="unknown", dob="2025-01-01" if change == "future_birth" else "2000-01-01",
            debut_season=2023 if change == "future_debut" else 2022, experience_before_cutoff=0)
    with pytest.raises(ValueError):
        build_catalog(design, request_snapshot)


def test_reconstructed_archives_cannot_be_promoted_to_independent_asof_evidence(design, request_snapshot):
    request_snapshot["source"].update(kind="retrospective_reconstruction", captured_at="2026-10-12T00:00:00Z")
    result = build_catalog(design, request_snapshot)
    assert result["source_claim"]["kind"] == "retrospective_reconstruction"
    assert not result["independent_evidence_established"] and not result["source_availability_verified"]


def test_held_out_table_changes_do_not_change_training_or_forecast_catalogue(request_snapshot):
    source, clean = tables()
    baseline = build("2022-02", tables=source, clean=clean)
    source["quali_times"].loc[source["quali_times"].event_id > "2022-02", "time_s"] = 1e6
    source["race"].loc[source["race"].event_id > "2022-02", "position"] = 99
    source["race_weather"].loc[source["race_weather"].event_id > "2022-02", "rainfall"] = True
    clean.loc[clean.event_id > "2022-02", "y"] = -1e6
    changed = build("2022-02", tables=source, clean=clean)
    assert build_catalog(baseline, request_snapshot) == build_catalog(changed, request_snapshot)


def test_future_training_state_is_rejected(design, request_snapshot):
    invalid = copy.deepcopy(design)
    invalid.qualifying.entries.loc[0, "event_id"] = "2022-03"
    with pytest.raises(ValueError, match="beyond"):
        build_catalog(invalid, request_snapshot)
