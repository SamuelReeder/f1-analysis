import numpy as np
import pytest

from analysis.championship_model_audit import paired_points, standings


def test_audit_distinguishes_season_outcomes_from_mean_points_order_and_ties():
    points = np.array([[25., 0.], [0., 18.], [0., 18.], [10., 10.]])
    rows = standings(points, ["a", "b"], "example", 1)
    assert rows[0]["expected_points_order"] == 2
    pair = paired_points(points, ["a", "b"], "a", 1)[1]
    assert pair["p_more_season_points"] == .25
    assert pair["p_equal_season_points"] == .25
    assert pair["mean_points_per_race_difference"] == -2.75
    assert pair["monte_carlo_se"] == pytest.approx(np.std([25, -18, -18, 0], ddof=1) / 2)


def test_audit_pairwise_results_follow_driver_identifiers_after_permutation():
    points = np.array([[25., 0.], [0., 18.], [10., 10.]])
    original = paired_points(points, ["a", "b"], "a", 24)
    permuted = paired_points(points[:, ::-1], ["b", "a"], "a", 24)
    assert original == permuted[::-1]
