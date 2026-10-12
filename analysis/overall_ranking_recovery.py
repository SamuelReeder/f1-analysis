"""Fixed finite-simulation ranking quantities for known-truth recovery.

These are development test functions, not published rankings or exact expected
points. The same simulator seed and workload define a deterministic function of
each parameter value. Conditional simulation error is retained separately.
"""
import hashlib
import json

import numpy as np

from f1rank.overall.simulation import JointDraws, MAX_EXPECTED_POINTS_MCSE, POINTS, Scenario, simulate


SITES = dict(qualifying="skill", race="overall_race_skill", q_weekend_sd="sd_driver_form",
    r_weekend_sd="overall_sd_day", shared_weekend="overall_shared_weekend",
    q_scale="sigma0", q_log_scale_sd="sigma_tau", q_df="nu", grid_coefficient="overall_grid",
    pace_coefficient="overall_pace", retirement_hazard="overall_hazard")


def validate_protocol(protocol, *, field_size, n_contexts):
    required = {"scenario_id", "scenarios", "advancing", "seasons", "seed", "target",
                "max_points_mcse", "precision_policy"}
    if not isinstance(protocol, dict) or set(protocol) != required:
        raise ValueError("Ranking functionals require their complete fixed protocol")
    if (not isinstance(protocol["scenario_id"], str) or not protocol["scenario_id"]
            or protocol["target"] != "finite_simulation_equal_machinery_and_team"
            or protocol["precision_policy"] != "bounded_points_mcse_fixed_workload_no_retry"
            or protocol["max_points_mcse"] != MAX_EXPECTED_POINTS_MCSE):
        raise ValueError("Unregistered ranking target or precision policy")
    if (type(protocol["seasons"]) is not int or protocol["seasons"] < 4
            or protocol["seasons"] % 2 or type(protocol["seed"]) is not int or protocol["seed"] < 0):
        raise ValueError("Need a fixed even simulation workload and nonnegative seed")
    advancing = protocol["advancing"]
    if (not isinstance(advancing, list) or not advancing
            or any(type(a) is not int or not 0 < a < b for a, b in zip(advancing, [field_size, *advancing]))):
        raise ValueError("Qualifying format must be explicit for the entire field")
    rows = protocol["scenarios"]
    if not isinstance(rows, list) or not rows:
        raise ValueError("An explicit nonempty scenario is required")
    identities = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(Scenario.__dataclass_fields__):
            raise ValueError("Incomplete reference event")
        if (not isinstance(row["event_id"], str) or not row["event_id"] or row["event_id"] in identities
                or type(row["context"]) is not int or not 0 <= row["context"] < n_contexts
                or type(row["laps"]) is not int or row["laps"] < 2
                or type(row["consecutive_racing_laps"]) is not int
                or not 0 <= row["consecutive_racing_laps"] <= row["laps"]):
            raise ValueError("Invalid or repeated reference event")
        identities.add(row["event_id"])


def specs(drivers):
    """Every current driver and pair; no selection based on estimated ranks."""
    if (len(drivers) < 2 or len(set(drivers)) != len(drivers)
            or any(not isinstance(d, str) or not d for d in drivers)):
        raise ValueError("Complete unique driver identities are required")
    rows = []
    for i, driver in enumerate(drivers):
        for kind in ("points", "rank_low", "rank_high", "best_share"):
            rows.append(dict(id=json.dumps(["finite_simulation", kind, driver], separators=(",", ":")),
                             kind=kind, indices=[i]))
        for j in range(i):
            rows.append(dict(id=json.dumps(["finite_simulation", "points_difference", driver, drivers[j]],
                                           separators=(",", ":")), kind="points_difference", indices=[i, j]))
    return rows


def from_states(states, catalog, protocol):
    """Evaluate a known parameter value, without fabricating fit diagnostics.

    Repeated equal states batch independent seasons on the simulator's draw axis;
    those axes are ONLY simulation lanes, not MCMC samples. Truth and posterior
    parameters use the same lane layout and seed, independent of their values or
    position in the posterior. This makes the finite functional reproducible.
    """
    drivers, entries = tuple(catalog["field"]), np.asarray(catalog["field_entries"])
    required = set(SITES.values())
    if not required <= set(states):
        raise ValueError("Ranking recovery is missing joint state or observation sites")
    values = {}
    for dest, site in SITES.items():
        value = np.asarray(states[site])
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise ValueError(f"Invalid ranking parameter: {site}")
        values[dest] = value
    specs(drivers)
    if (entries.ndim != 1 or len(entries) != len(drivers) or entries.dtype.kind not in "iu"
            or len(np.unique(entries)) != len(entries) or (entries < 0).any()):
        raise ValueError("Field must reference unique current states")
    for name in ("qualifying", "race"):
        if values[name].ndim != 1 or (entries >= len(values[name])).any():
            raise ValueError("A current field state is missing")
        values[name] = values[name][entries]
    if values["grid_coefficient"].ndim != 1:
        raise ValueError("Expected one context axis at a parameter value")
    validate_protocol(protocol, field_size=len(drivers), n_contexts=len(values["grid_coefficient"]))
    lanes = protocol["seasons"] // 2
    joint = JointDraws(drivers, **{name: np.broadcast_to(value, (lanes,) + value.shape)
                                  for name, value in values.items()})
    result = simulate(joint, [Scenario(**row) for row in protocol["scenarios"]],
        replicates=2, seed=protocol["seed"], advancing=tuple(protocol["advancing"]))
    # Season totals become GP averages before any comparison or error estimate.
    points = result["points"].reshape(protocol["seasons"], len(drivers)) / result["n_races"]
    if not np.isfinite(points).all() or ((points < 0) | (points > np.max(POINTS))).any():
        raise ValueError("Simulated points are outside the frozen scoring bounds")
    means = points.mean(axis=0)
    variance = points.var(axis=0, ddof=1)
    se = np.sqrt(variance / len(points))
    # Independent GP-average seasons are bounded by the frozen points table.
    # This variance bound does not mistake a small constant sample for precision.
    worst_case_se = float(np.max(POINTS) / (2 * np.sqrt(len(points))))
    if not all(np.isfinite(x).all() for x in (means, variance, se)):
        raise ValueError("Nonfinite simulated ranking quantity")
    low = 1 + (means[None, :] > means[:, None]).sum(axis=1)
    equal = (means[None, :] == means[:, None]).sum(axis=1)
    high = low + equal - 1
    best = np.where(low == 1, 1 / equal, 0.)
    quantities, mcse = {}, {}
    for spec in specs(drivers):
        i, *other = spec["indices"]
        if spec["kind"] == "points_difference":
            difference = points[:, i] - points[:, other[0]]
            quantities[spec["id"]] = float(means[i] - means[other[0]])
            # Pairwise error retains the covariance from the SAME season.
            mcse[spec["id"]] = float(difference.std(ddof=1) / np.sqrt(len(points)))
        else:
            value = dict(points=means, rank_low=low, rank_high=high, best_share=best)[spec["kind"]][i]
            quantities[spec["id"]] = float(value)
            if spec["kind"] == "points":
                mcse[spec["id"]] = float(se[i])
    numerics = dict(scenario_id=protocol["scenario_id"], seasons=protocol["seasons"],
        seed=protocol["seed"], n_races=result["n_races"], points_mcse=mcse,
        max_driver_points_mcse=float(se.max()), worst_case_driver_points_mcse_bound=worst_case_se,
        points_precision_passed=bool((se <= MAX_EXPECTED_POINTS_MCSE).all()
                                    and worst_case_se <= MAX_EXPECTED_POINTS_MCSE),
        protocol_sha256=hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
        exact_expected_points=False, establishes_rank_precision=False,
        publication_allowed=False)
    return quantities, numerics
