"""Exact own-driver points for complete, strict generated crossing orders.

This is an internal simulator specialization, not general race adjudication.
The baseline still uses the original full-field scorer and championship rules.
"""
import numpy as np

from f1rank.overall.scoring import (CLASSIFICATION_DENOMINATOR, CLASSIFICATION_NUMERATOR,
                                  _counts, gp_points_table)


def own_driver_points(completed_laps, orders, scenario):
    """Return (draw, intervened driver) points with a complete field in each order.

Distances are shared (draw, competitor); orders have axes (draw, intervened
driver, competitor). Every target is present in identity order. Native crossing
orders are strict permutations, so classification cannot have a dead heat.
"""
    laps = _counts("completed_laps", completed_laps)
    order = np.asarray(orders)
    if (laps.ndim != 2 or laps.shape[0] < 1 or laps.shape[1] < 2
            or order.shape != (laps.shape[0], laps.shape[1], laps.shape[1])
            or order.dtype.kind not in "iu"):
        raise ValueError("Own-driver scoring needs complete draw, target and competitor axes")
    n = laps.shape[1]
    if (order < 0).any() or (order >= n).any():
        raise ValueError("Generated crossing order contains an invalid competitor")
    order = order.astype(np.intp, copy=False)
    present = np.zeros(order.shape, dtype=bool)
    np.put_along_axis(present, order, True, axis=-1)
    if not present.all():
        raise ValueError("Generated crossing orders must contain every competitor exactly once")
    if not np.isscalar(scenario.laps) or not np.isscalar(scenario.consecutive_racing_laps):
        raise ValueError("Reference distance and race control must be scalar")
    table = gp_points_table(scheduled_laps=scenario.laps, leader_laps=scenario.laps,
                           consecutive_racing_laps=scenario.consecutive_racing_laps)
    winner = laps.max(axis=-1)
    if (winner > scenario.laps).any():
        raise ValueError("Classification distance exceeds the leader's end-signal distance")
    if (winner < scenario.laps).any():
        raise ValueError("No full-distance finisher: shortened-race model required")
    drivers = np.arange(n)
    crossing_position = np.argmax(order == drivers[None, :, None], axis=-1)
    ordered_laps = np.take_along_axis(np.broadcast_to(laps[:, None, :], order.shape), order, axis=-1)
    more_distance = (laps[:, None, :] > laps[:, :, None]).sum(axis=-1)
    same_distance_before = ((ordered_laps == laps[:, :, None])
                            & (drivers < crossing_position[:, :, None])).sum(axis=-1)
    rank = 1 + more_distance + same_distance_before
    threshold = CLASSIFICATION_NUMERATOR * winner // CLASSIFICATION_DENOMINATOR
    eligible = (winner[:, None] > 0) & (laps >= threshold[:, None])
    return np.where(eligible & (rank <= len(table)), table[np.minimum(rank - 1, len(table) - 1)], 0.)
