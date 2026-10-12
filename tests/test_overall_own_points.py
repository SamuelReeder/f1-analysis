"""Own-driver specialization against the authoritative complete-field scorer."""
import itertools

import numpy as np
import pytest

from analysis.overall_own_points import own_driver_points
from f1rank.overall.scoring import score_grand_prix
from f1rank.overall.simulation import Scenario


def full_field_diagonal(laps, orders, scenario):
    line = np.argsort(orders, axis=-1) + 1
    _, points = score_grand_prix(np.broadcast_to(laps[:, None, :], orders.shape), line,
        scheduled_laps=scenario.laps, leader_laps=scenario.laps,
        consecutive_racing_laps=scenario.consecutive_racing_laps)
    field = np.arange(laps.shape[1])
    return points[:, field, field]


@pytest.mark.parametrize("control", [0, 1, 2, 10])
def test_exhaustive_small_field_distances_and_independent_crossing_orders(control):
    distances = np.array([row for row in itertools.product([0, 8, 9, 10], repeat=3) if max(row) == 10])
    permutations = list(itertools.permutations(range(3)))
    interventions = np.array(list(itertools.product(permutations, repeat=3)))
    laps = np.repeat(distances, len(interventions), axis=0)
    orders = np.tile(interventions, (len(distances), 1, 1))
    scenario = Scenario("full", 0, 10, control)
    np.testing.assert_array_equal(own_driver_points(laps, orders, scenario), full_field_diagonal(laps, orders, scenario))


@pytest.mark.parametrize("n", [20, 22, 24])
@pytest.mark.parametrize("control", [0, 1, 2, 60])
def test_complete_generated_fields_match_and_do_not_mutate_inputs(n, control):
    rng = np.random.default_rng(71)
    laps = rng.integers(0, 61, (128, n))
    laps[:, 0] = 60
    orders = np.argsort(rng.normal(size=(128, n, n)), axis=-1)
    before_laps, before_orders = laps.copy(), orders.copy()
    laps.flags.writeable = False
    orders.flags.writeable = False
    scenario = Scenario("full", 0, 60, control)
    np.testing.assert_array_equal(own_driver_points(laps, orders, scenario), full_field_diagonal(laps, orders, scenario))
    np.testing.assert_array_equal(laps, before_laps)
    np.testing.assert_array_equal(orders, before_orders)


@pytest.mark.parametrize("case", ["duplicate", "negative", "outside", "floating", "missing_target",
    "wrong_draws", "negative_laps", "fractional_laps", "unbounded_laps", "past_leader", "all_retired", "array_control"])
def test_invalid_or_unsupported_generated_fields_are_rejected(case):
    laps = np.array([[60, 59, 53], [60, 30, 55]])
    orders = np.broadcast_to(np.arange(3), (2, 3, 3)).copy()
    scenario = Scenario("full", 0, 60, 60)
    if case == "duplicate": orders[0, 1] = [0, 0, 2]
    elif case == "negative": orders[0, 1, 1] = -1
    elif case == "outside": orders[0, 1, 1] = 3
    elif case == "floating": orders = orders.astype(float)
    elif case == "missing_target": orders = orders[:, :-1]
    elif case == "wrong_draws": orders = orders[:1]
    elif case == "negative_laps": laps[0, 1] = -1
    elif case == "fractional_laps": laps = laps.astype(float) + .5
    elif case == "unbounded_laps": laps[0, 1] = np.iinfo(np.int64).max
    elif case == "past_leader": laps[0, 1] = 61
    elif case == "all_retired": laps[:] = 59
    else: scenario = Scenario("full", 0, 60, np.array([60, 60]))
    with pytest.raises(ValueError):
        own_driver_points(laps, orders, scenario)
