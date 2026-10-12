"""Known-outcome sporting fixtures, independent of any fitted driver ranking."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from f1rank.overall.scoring import (
    CLASSIFICATION_DENOMINATOR, CLASSIFICATION_NUMERATOR, GP_POINT_TABLES,
    MIN_CONSECUTIVE_RACING_LAPS, RULESET_ID, classify, gp_points_table, score_grand_prix,
)
from f1rank.overall.simulation import Scenario, championship_countback, simulate
from test_overall import draws


@pytest.mark.parametrize("distance,expected", [
    (2, [6, 4, 3, 2, 1, 0, 0, 0, 0, 0]),
    (24, [6, 4, 3, 2, 1, 0, 0, 0, 0, 0]),
    (25, [13, 10, 8, 6, 5, 4, 3, 2, 1, 0]),
    (49, [13, 10, 8, 6, 5, 4, 3, 2, 1, 0]),
    (50, [19, 14, 12, 10, 8, 6, 4, 3, 2, 1]),
    (74, [19, 14, 12, 10, 8, 6, 4, 3, 2, 1]),
    (75, [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]),
    (100, [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]),
    (101, [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]),
])
def test_distance_thresholds_and_every_points_place(distance, expected):
    # Distance governs even when an event ends for a reason other than suspension.
    position, points = score_grand_prix([distance] * 12, np.arange(1, 13),
        scheduled_laps=100, leader_laps=distance, consecutive_racing_laps=2)
    np.testing.assert_array_equal(position, np.arange(1, 13))
    np.testing.assert_array_equal(points, expected + [0, 0])


@pytest.mark.parametrize("leader,racing", [(0, 0), (1, 1), (2, 0), (100, 1)])
def test_minimum_consecutive_racing_applies_at_every_distance(leader, racing):
    table = gp_points_table(scheduled_laps=100, leader_laps=leader,
                            consecutive_racing_laps=racing)
    assert not table.any()


def test_shortened_classification_and_end_signal_distance_are_distinct():
    # A final classification countback must not change which distance column
    # applies. Neither distance is inferred from the other in the scorer.
    positions, points = score_grand_prix([24, 23, 21, 20], [3, 1, 2, 4],
        scheduled_laps=100, leader_laps=25, consecutive_racing_laps=2)
    np.testing.assert_array_equal(positions, [1, 2, 3, 0])
    np.testing.assert_array_equal(points, [13, 10, 8, 0])


def test_classification_rounds_down_and_does_not_require_a_running_finisher():
    positions, points = score_grand_prix([53, 47, 46, 52], [4, 2, 1, 3],
        scheduled_laps=70, leader_laps=53, consecutive_racing_laps=2)
    np.testing.assert_array_equal(positions, [1, 3, 0, 2])
    np.testing.assert_array_equal(points, [25, 15, 0, 18])


def test_dead_heat_shares_all_occupied_places_and_preserves_points():
    positions, points = score_grand_prix([60] * 5, [1, 1, 3, 3, 3],
        scheduled_laps=60, leader_laps=60, consecutive_racing_laps=2)
    np.testing.assert_array_equal(positions, [1, 1, 3, 3, 3])
    np.testing.assert_allclose(points, [21.5, 21.5, 37 / 3, 37 / 3, 37 / 3])
    assert points.sum() == pytest.approx(80)


def test_dead_heat_straddling_last_scoring_place_shares_zero_slot_too():
    line = np.array([*range(1, 10), 10, 10, 12])
    positions, points = score_grand_prix([60] * 12, line,
        scheduled_laps=60, leader_laps=60, consecutive_racing_laps=2)
    np.testing.assert_array_equal(positions, line)
    np.testing.assert_array_equal(points[-3:], [.5, .5, 0])


def test_equal_crossing_with_different_lap_counts_is_not_a_dead_heat():
    positions = classify([60, 59, 60], [1, 1, 3])
    np.testing.assert_array_equal(positions, [1, 3, 2])


def test_batched_context_and_permutation_keep_each_event_and_driver_aligned():
    laps = np.array([[60, 59, 58, 53], [20, 20, 20, 19]])
    line = np.array([[1, 2, 3, 4], [1, 1, 3, 4]])
    context = dict(scheduled_laps=80, leader_laps=np.array([60, 20]),
                   consecutive_racing_laps=np.array([2, 1]))
    a = score_grand_prix(laps, line, **context)
    np.testing.assert_array_equal(a[1], [[25, 18, 15, 0], [0, 0, 0, 0]])
    permutation = [2, 0, 3, 1]
    b = score_grand_prix(laps[:, permutation], line[:, permutation], **context)
    for left, right in zip(a, b):
        np.testing.assert_array_equal(left[:, permutation], right)


@pytest.mark.parametrize("context", [
    {"scheduled_laps": 0}, {"leader_laps": 59}, {"consecutive_racing_laps": 61},
    {"consecutive_racing_laps": None}, {"consecutive_racing_laps": np.nan},
    {"consecutive_racing_laps": True}, {"consecutive_racing_laps": 1.5},
    {"scheduled_laps": -1}, {"scheduled_laps": np.inf},
    {"scheduled_laps": "60"}, {"scheduled_laps": 2**63},
    {"consecutive_racing_laps": [2, 2]},
])
def test_unknown_impossible_or_misaligned_event_context_is_rejected(context):
    kwargs = dict(scheduled_laps=60, leader_laps=60, consecutive_racing_laps=2)
    kwargs.update(context)
    with pytest.raises(ValueError):
        score_grand_prix([60, 59], [1, 2], **kwargs)


@pytest.mark.parametrize("line", [[1, 1, 2], [2, 3, 4], [1, 2.5, 3], [1, 2, np.nan]])
def test_crossing_ranks_cannot_have_inconsistent_occupied_slots(line):
    with pytest.raises(ValueError):
        classify([60, 60, 60], line)


def test_distance_and_race_control_metadata_cannot_be_omitted():
    with pytest.raises(TypeError):
        score_grand_prix([60, 59], [1, 2])
    with pytest.raises(TypeError):
        Scenario("ref", 0, 60)


def test_championship_countback_reaches_lower_race_and_qualifying_places():
    # Equal wins and seconds: race thirds precede qualifying wins.
    result = championship_countback(np.array([25., 25., 0., 0.]),
        np.array([[3, 4, 1, 2]]), np.array([[2, 1, 3, 4]]))
    np.testing.assert_array_equal(result["rank_low"], [1, 2, 3, 4])
    # Equal race counts and qualifying wins: compare qualifying seconds next.
    result = championship_countback(np.array([1., 1., 0.]),
        np.array([[0, 0, 1], [0, 0, 1]]), np.array([[1, 3, 2], [2, 1, 3]]))
    np.testing.assert_array_equal(result["rank_low"], [1, 2, 3])


def test_residual_sporting_ties_keep_title_mass_and_rank_interval():
    result = championship_countback(np.zeros((2, 3)),
        np.zeros((2, 4, 3)), np.zeros((2, 4, 3)))
    np.testing.assert_array_equal(result["rank_low"], np.ones((2, 3)))
    np.testing.assert_array_equal(result["rank_high"], np.full((2, 3), 3))
    np.testing.assert_allclose(result["title_share"], 1 / 3)
    np.testing.assert_allclose(result["title_share"].sum(axis=-1), 1)


def test_simulator_uses_explicit_race_control_context_for_points():
    d = draws(nd=2)
    result = simulate(d, [Scenario("ref", 0, 60, 1)], replicates=2, advancing=(4, 2))
    assert not result["points"].any()
    np.testing.assert_allclose(result["title_share"].sum(axis=-1), 1)
    with pytest.raises(ValueError, match="Consecutive"):
        simulate(d, [Scenario("ref", 0, 60, 61)], replicates=2, advancing=(4, 2))


def test_new_scorer_does_not_invent_a_shortened_event_generator():
    d = draws(nd=2)
    d = replace(d, retirement_hazard=np.full_like(d.retirement_hazard, .999))
    with pytest.raises(ValueError, match="shortened-race model required"):
        simulate(d, [Scenario("ref", 0, 60, 60)], replicates=2, advancing=(4, 2))


def test_frozen_event_rules_match_executable_constants():
    path = Path(__file__).resolve().parents[1] / "docs/overall_event_rules.json"
    contract = json.loads(path.read_text())
    assert contract["id"] == RULESET_ID
    assert contract["minimum_consecutive_racing_laps"] == MIN_CONSECUTIVE_RACING_LAPS
    assert contract["classification_fraction"] == [CLASSIFICATION_NUMERATOR, CLASSIFICATION_DENOMINATOR]
    np.testing.assert_array_equal(contract["points_tables"], GP_POINT_TABLES)
