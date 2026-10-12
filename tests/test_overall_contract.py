"""The display must describe the actual simulator's units and precision."""
import json
from pathlib import Path

import numpy as np

from f1rank.overall.simulation import MAX_EXPECTED_POINTS_MCSE, POINTS, uncertainty


def test_display_contract_matches_simulation():
    contract = json.loads((Path(__file__).resolve().parents[1] / "contracts/overall.json").read_text())
    np.testing.assert_array_equal(contract["points"], POINTS)
    assert contract["max_conditional_points_mcse"] == MAX_EXPECTED_POINTS_MCSE
    # No within-draw noise: the displayed band must match the registered mass,
    # without confounding posterior uncertainty with a season-outcome interval.
    means = np.linspace(5, 20, 100).reshape(100, 1, 1)
    points = np.repeat(means, 2, axis=1)
    actual = uncertainty({"points": points, "n_races": 1})
    tail = (1 - contract["interval_probability"]) / 2
    expected = np.quantile(means[:, 0, :], [tail, 1 - tail], axis=0).T
    np.testing.assert_allclose(actual["expected_points_interval"], expected)
