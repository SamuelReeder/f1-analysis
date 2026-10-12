"""Deterministic GP scoring under the frozen rules in overall_event_rules.json.

These functions score supplied outcomes; they do not generate race stoppages,
lapping, incidents or race-control decisions. All distances exclude formation
laps. The leader's distance at the end signal can differ from the final
classification distance after a countback.
"""
import numpy as np

RULESET_ID = "fia-f1-2026-a03-b09-gp"
CLASSIFICATION_NUMERATOR = 9
CLASSIFICATION_DENOMINATOR = 10
MIN_CONSECUTIVE_RACING_LAPS = 2
GP_POINT_TABLES = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [6, 4, 3, 2, 1, 0, 0, 0, 0, 0],
    [13, 10, 8, 6, 5, 4, 3, 2, 1, 0],
    [19, 14, 12, 10, 8, 6, 4, 3, 2, 1],
    [25, 18, 15, 12, 10, 8, 6, 4, 2, 1],
], dtype=float)
GP_POINT_TABLES.flags.writeable = False
POINTS = GP_POINT_TABLES[-1]


def _counts(name, value):
    a = np.asarray(value)
    # Reserve headroom for exact integer threshold arithmetic below.
    if (a.dtype.kind not in "iuf" or not np.isfinite(a).all()
            or (a < 0).any() or (a != np.floor(a)).any()
            or (a > np.iinfo(np.int64).max // 100).any()):
        raise ValueError(f"{name} must contain finite non-negative integer counts")
    return a.astype(np.int64)


def _event_counts(name, value, shape):
    a = _counts(name, value)
    if a.shape not in ((), shape):
        raise ValueError(f"{name} must be scalar or match the event axes")
    return np.broadcast_to(a, shape)


def gp_points_table(*, scheduled_laps, leader_laps, consecutive_racing_laps):
    """Return the points table for each event (FIA A2.2.1).

    consecutive_racing_laps is the leader's longest uninterrupted run of complete
    laps without SC/VSC, not the total number of green laps. It is required even
    for a full-distance race. Missing race-control information must not be filled
    with a presumed points-eligible value. End reason is irrelevant to this rule.
    """
    scheduled, leader, racing = np.broadcast_arrays(
        _counts("scheduled_laps", scheduled_laps),
        _counts("leader_laps", leader_laps),
        _counts("consecutive_racing_laps", consecutive_racing_laps))
    if (scheduled == 0).any():
        raise ValueError("Scheduled distance must be positive")
    if (racing > leader).any():
        raise ValueError("Consecutive racing laps cannot exceed leader distance")
    # Integer comparisons make equality at each boundary unambiguous.
    column = (1 + (4 * leader >= scheduled).astype(int)
              + (2 * leader >= scheduled).astype(int)
              + (4 * leader >= 3 * scheduled).astype(int))
    column = np.where(racing >= MIN_CONSECUTIVE_RACING_LAPS, column, 0)
    return GP_POINT_TABLES[column]


def classify(completed_laps, line_order):
    """Classify an adjudicated field by distance, then last control-line crossing.

    The last axis is the field. line_order uses competition ranks: a dead heat
    occupies consecutive slots, e.g. [1, 1, 3]. Equal crossing ranks at different
    distances are not a classification tie. Returns competition ranks, with zero
    for unclassified cars. Inputs exclude DNS/DSQ and already reflect penalties
    and any race-control countback; this function does not adjudicate them.
    """
    laps = _counts("completed_laps", completed_laps)
    line = _counts("line_order", line_order)
    if laps.shape != line.shape or laps.ndim < 1 or laps.shape[-1] < 2:
        raise ValueError("Completion and line-order arrays must agree on a field")
    expected = 1 + (line[..., None, :] < line[..., :, None]).sum(axis=-1)
    if not np.array_equal(line, expected):
        raise ValueError("Line order must use consecutive occupied competition ranks")
    more_laps = laps[..., None, :] > laps[..., :, None]
    same_laps = laps[..., None, :] == laps[..., :, None]
    earlier = line[..., None, :] < line[..., :, None]
    rank = 1 + (more_laps | (same_laps & earlier)).sum(axis=-1)
    winner_laps = laps.max(axis=-1, keepdims=True)
    threshold = CLASSIFICATION_NUMERATOR * winner_laps // CLASSIFICATION_DENOMINATOR
    eligible = (winner_laps > 0) & (laps >= threshold)
    return np.where(eligible, rank, 0)


def score_grand_prix(completed_laps, line_order, *, scheduled_laps, leader_laps,
                    consecutive_racing_laps):
    """Return classification and points, sharing occupied slots for dead heats.

    Distance and race-control counts are explicit scalar or per-event inputs.
    leader_laps is measured at the end signal; completed_laps is the adjudicated
    classification distance. Neither is inferred from a grid or finish order.
    """
    positions = classify(completed_laps, line_order)
    shape = positions.shape[:-1]
    scheduled = _event_counts("scheduled_laps", scheduled_laps, shape)
    leader = _event_counts("leader_laps", leader_laps, shape)
    racing = _event_counts("consecutive_racing_laps", consecutive_racing_laps, shape)
    if (np.asarray(completed_laps).max(axis=-1) > leader).any():
        raise ValueError("Classification distance exceeds the leader's end-signal distance")
    table = gp_points_table(scheduled_laps=scheduled, leader_laps=leader,
                            consecutive_racing_laps=racing)
    cumulative = np.concatenate((np.zeros(shape + (1,)), table.cumsum(axis=-1)), axis=-1)
    tied = (positions[..., :, None] == positions[..., None, :]).sum(axis=-1)
    start = np.clip(positions - 1, 0, table.shape[-1])
    end = np.clip(positions + tied - 1, 0, table.shape[-1])
    total = (np.take_along_axis(cumulative, end, axis=-1)
             - np.take_along_axis(cumulative, start, axis=-1))
    return positions, np.where(positions > 0, total / tied, 0.)
