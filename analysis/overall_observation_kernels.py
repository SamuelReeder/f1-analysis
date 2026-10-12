"""Conditional observation kernels of the existing joint Overall model.

Development predictions only. Explicit scenarios are not forecasts of unknown
weather, neutralisation, exposure or grid penalties. No fitting or publication.
Every batch retains the chain/draw pairing of latent states and coefficients.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.overall_forecast_states import DEFAULT_DRAW_BATCH, iter_forecast_states


Q_PARAMETERS = ("sigma0", "sigma_tau", "nu", "sd_car_segment")
CONTEXT_NAMES = ("grid", "pace", "hazard")


def _finite(name, value):
    if np.asarray(value).dtype.kind not in "fiu" or not np.isfinite(value).all():
        raise ValueError(f"Nonfinite or nonnumeric {name}; no clipping or replacement")


def _observation_parameters(design, posterior, batch_size):
    shapes = {name: () for name in Q_PARAMETERS}
    for name in CONTEXT_NAMES:
        shapes.update({f"overall_{name}_{part}": () for part in ("base", "wet", "sc")})
        shapes[f"overall_sd_{name}_circuit"] = ()
        shapes[f"overall_{name}_circuit_z"] = (design.outcomes["n_circuits"],)
    if any(name not in posterior for name in shapes) or "sd_level" not in posterior:
        raise ValueError("Missing joint observation parameters")
    shape = np.asarray(posterior["sd_level"]).shape
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("Grouped chain and draw axes are required")
    result, digest = {}, hashlib.sha256()
    # Validate ALL draws before yielding; a bad last draw cannot be skipped.
    for name, tail in sorted(shapes.items()):
        a = np.asarray(posterior[name])
        if a.shape != shape + tail or a.dtype.kind not in "fiu":
            raise ValueError(f"Invalid joint axes for {name}")
        digest.update(f"parameters:{name}:{a.dtype.str}:{a.shape}".encode())
        for chain in range(shape[0]):
            for start in range(0, shape[1], batch_size):
                part = a[chain, start:start + batch_size]
                _finite(name, part)
                if name in ("sigma0", "nu") and (part <= 0).any():
                    raise ValueError(f"Nonpositive {name}")
                if (name in ("sigma_tau", "sd_car_segment") or name.startswith("overall_sd_")) and (part < 0).any():
                    raise ValueError(f"Negative scale {name}")
                digest.update(np.ascontiguousarray(part).tobytes())
        result[name] = a
    return result, shape, digest.hexdigest()


class ObservationNoise:
    """Replay identity-keyed full draw streams, then slice a contiguous batch."""
    def __init__(self, seed, shape, draw_range):
        if type(seed) is not int or seed < 0:
            raise ValueError("An explicit nonnegative integer seed is required")
        if (len(shape) != 2 or any(type(n) is not int or n < 2 for n in shape)
                or len(draw_range) != 2 or any(type(n) is not int for n in draw_range)
                or not 0 <= draw_range[0] < draw_range[1] <= shape[1]):
            raise ValueError("Invalid chain/draw shape or batch range")
        self.seed, self.shape, self.draw_range = seed, tuple(shape), tuple(draw_range)

    def draw(self, site, identity, *, distribution="normal", df=None):
        key = json.dumps(["overall_observation", self.seed, site, identity],
                         separators=(",", ":"), allow_nan=False).encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest(), "big"))
        if distribution == "normal":
            value = rng.normal(size=self.shape)
        elif distribution == "student":
            if df is None or np.shape(df) != self.shape or not np.isfinite(df).all() or (df <= 0).any():
                raise ValueError("Student-t noise requires the full joint degrees-of-freedom array")
            value = rng.standard_t(df)
        elif distribution == "uniform":
            value = rng.random(size=self.shape)
        elif distribution == "gumbel":
            value = rng.gumbel(size=self.shape)
        else:
            raise ValueError("Unknown observation-noise distribution")
        start, stop = self.draw_range
        return value[:, start:stop].copy()


def _rank(scores, tie_noise, active):
    order = np.lexsort((tie_noise, -np.where(active, scores, -np.inf)), axis=-1)
    return np.where(active, np.argsort(order, axis=-1) + 1, 0)


def qualifying_segments(event, parameters, noise, *, advancing, full_df):
    """Positions conditional on all entrants setting times; not the official grid.

    Higher transformed pace is faster. Segment intercepts cancel. Car-segment
    effects are shared by teammates and centred over the active car units, as in
    the likelihood. Session scale is common to every entrant in that segment.
    """
    total = np.asarray(event["states"]["qualifying_total"])
    _finite("qualifying total", total)
    drivers, cars, car_index = event["drivers"], event["cars"], np.asarray(event["entry_car"])
    n = len(drivers)
    if (total.ndim != 3 or total.shape[-1] != n or len(set(drivers)) != n
            or len(set(cars)) != len(cars) or car_index.shape != (n,)
            or car_index.dtype.kind not in "iu" or (car_index < 0).any() or (car_index >= len(cars)).any()):
        raise ValueError("Invalid qualifying field or state axes")
    if not isinstance(advancing, (tuple, list)) or any(
            type(a) is not int or not 0 < a < b for a, b in zip(advancing, (n, *advancing))):
        raise ValueError("Qualifying fields must strictly shrink")
    shape = total.shape[:2]
    if any(np.shape(parameters[name]) != shape for name in Q_PARAMETERS):
        raise ValueError("Qualifying coefficients must share state chain/draw axes")
    active, positions, segments = np.ones(total.shape, bool), np.zeros(total.shape, int), []
    for segment, keep in enumerate((*advancing, 0)):
        identity = [event["event_id"], segment]
        u = noise.draw("qualifying_scale", identity)
        with np.errstate(over="ignore", under="ignore"):
            scale = parameters["sigma0"] * np.exp(parameters["sigma_tau"] * u)
        _finite("qualifying session scale", scale)
        if (scale <= 0).any():
            raise ValueError("Qualifying session scale underflow; do not replace it")
        car_z = np.stack([noise.draw("qualifying_car_segment", [*identity, car]) for car in cars], axis=-1)
        car_effect = parameters["sd_car_segment"][..., None] * car_z
        active_cars = np.stack([active[..., car_index == i].any(axis=-1) for i in range(len(cars))], axis=-1)
        car_effect -= (car_effect * active_cars).sum(axis=-1, keepdims=True) / active_cars.sum(axis=-1, keepdims=True)
        location = total + car_effect[..., car_index]
        residual = np.stack([noise.draw("qualifying_residual", [*identity, driver],
            distribution="student", df=full_df) for driver in drivers], axis=-1)
        pace = location + scale[..., None] * residual
        _finite("qualifying observation", pace)
        tie = np.stack([noise.draw("qualifying_tie", [*identity, driver], distribution="uniform")
                        for driver in drivers], axis=-1)
        rank = _rank(pace, tie, active)
        eliminated = active & (rank > keep)
        positions[eliminated] = rank[eliminated]
        segments.append(dict(active=active.copy(), location=location, scale=scale,
            residual=residual, pace=pace, car_effect=car_effect, car_active=active_cars))
        active &= rank <= keep
    return dict(positions=positions, segments=segments,
                interpretation="qualifying_positions_conditional_on_all_entrants_setting_times")


def context_coefficients(parameters, circuit, *, wet, neutralised):
    """Known-circuit coefficients in the existing training centring convention."""
    if type(wet) is not bool or type(neutralised) is not bool:
        raise ValueError("Explicit conditional wet/neutralised booleans are required")
    result = {}
    for name in CONTEXT_NAMES:
        z = parameters[f"overall_{name}_circuit_z"]
        if type(circuit) is not int or not 0 <= circuit < z.shape[-1]:
            raise ValueError("Unseen circuit coefficients require a registered extension policy")
        effect = parameters[f"overall_sd_{name}_circuit"][..., None] * z
        eta = (parameters[f"overall_{name}_base"] + effect[..., circuit] - effect.mean(axis=-1)
               + parameters[f"overall_{name}_wet"] * wet + parameters[f"overall_{name}_sc"] * neutralised)
        _finite(name + " context", eta)
        if name == "hazard":
            # Log probabilities avoid rounding a finite logit to zero or one.
            result.update(hazard_logit=eta, log_hazard=-np.logaddexp(0., -eta),
                          log_survival=-np.logaddexp(0., eta))
        else:
            with np.errstate(over="ignore"):
                result[name] = np.exp(eta)
            _finite(name + " coefficient", result[name])
            if (result[name] == 0).any():
                raise ValueError(f"Underflow in {name} coefficient; do not replace it")
    return result


def race_strength(total, grid_slot, coefficients):
    total, grid = np.asarray(total), np.asarray(grid_slot)
    _finite("race total", total)
    _finite("grid slots", grid)
    if (total.ndim != 3 or grid.shape not in (total.shape, (total.shape[-1],))
            or (grid < 1).any() or (grid != np.floor(grid)).any()
            or any(np.shape(coefficients[name]) != total.shape[:2] for name in ("pace", "grid"))):
        raise ValueError("Race/grid/coefficient axes or slots are invalid")
    result = coefficients["pace"][..., None] * total - coefficients["grid"][..., None] * np.log(grid)
    _finite("race strength", result)
    return result


def order_log_probability(strength, positions):
    """PL log probability per joint draw on the supplied running panel.

    Positions are per driver: zero outside the panel, contiguous positive ranks
    inside it. This is p(order | panel, parameters), before posterior integration.
    Empty and singleton panels have log probability zero.
    """
    s, rank = np.asarray(strength), np.asarray(positions)
    _finite("order strengths", s)
    _finite("running positions", rank)
    if s.ndim < 1 or s.shape[-1] < 1 or rank.shape not in (s.shape, (s.shape[-1],)):
        raise ValueError("Order and strength axes do not match")
    rank = np.broadcast_to(rank, s.shape)
    if (rank < 0).any() or (rank != np.floor(rank)).any():
        raise ValueError("Running positions must be nonnegative integers")
    mask = rank > 0
    order = np.argsort(np.where(mask, rank, np.inf), axis=-1)
    ordered_rank = np.take_along_axis(rank, order, axis=-1)
    expected = np.arange(1, s.shape[-1] + 1)
    if ((ordered_rank > 0) & (ordered_rank != expected)).any():
        raise ValueError("Running positions must be unique and contiguous")
    ordered = np.take_along_axis(s, order, axis=-1)
    included = ordered_rank > 0
    # Relative suffix normalizers preserve tied groups even at large offsets.
    # Keep the NumPy implementation independent of the differentiable fit helper.
    maximum = np.zeros(s.shape[:-1])
    norm, result = np.zeros_like(maximum), np.zeros_like(maximum)
    present = np.zeros(maximum.shape, bool)
    for i in range(s.shape[-1] - 1, -1, -1):
        active = included[..., i]
        value = np.where(active, ordered[..., i], 0.)
        previous = np.where(present, maximum, value)
        new_maximum = np.maximum(value, previous)
        relative = value - new_maximum
        new_norm = np.exp(relative) + norm * np.exp(previous - new_maximum)
        result += np.where(active, relative - np.log(new_norm), 0.)
        maximum, norm = np.where(active, new_maximum, maximum), np.where(active, new_norm, norm)
        present |= active
    return result


def completion_log_probability(coefficients, completed, failed):
    """Exact geometric failure / right-censored exposure factor per entrant."""
    laps, failure = np.asarray(completed), np.asarray(failed)
    _finite("completed laps", laps)
    if (failure.shape != laps.shape or failure.dtype.kind != "b" or (laps < 0).any()
            or (laps != np.floor(laps)).any() or laps.shape[:-1] != np.shape(coefficients["log_hazard"])):
        raise ValueError("Invalid completion observation or joint axes")
    return (laps * coefficients["log_survival"][..., None]
            + np.where(failure, coefficients["log_hazard"][..., None], 0.))


def draw_completion(event, coefficients, exposure, noise):
    """Censored geometric observations at explicitly supplied fixed exposures."""
    distance = np.asarray(exposure)
    n = len(event["drivers"])
    if distance.shape != (n,) or distance.dtype.kind not in "iu" or (distance <= 0).any():
        raise ValueError("Explicit positive integer exposure for each entrant is required")
    if (distance >= np.iinfo(np.int64).max).any():
        raise ValueError("Exposure cannot be represented safely")
    u = np.stack([noise.draw("completion", [event["event_id"], driver], distribution="uniform")
                  for driver in event["drivers"]], axis=-1)
    waiting = -np.log1p(-u)
    rate = -coefficients["log_survival"][..., None]
    failed = waiting < rate * distance
    # Divide only failures. Tiny hazards may round rate to zero, in which case
    # every finite exposure is censored; do not cast an infinite lifetime to int.
    failure_lap = np.floor(np.divide(waiting, rate, out=np.zeros_like(waiting), where=failed)).astype(np.int64)
    completed = np.where(failed, failure_lap, distance)
    return dict(completed=completed, failed=failed, exposure_laps=distance,
                log_probability=completion_log_probability(coefficients, completed, failed))


def draw_running_order(event, strength, running, noise):
    if np.shape(running) != np.shape(strength) or np.asarray(running).dtype.kind != "b":
        raise ValueError("Running panel must match the strength axes")
    perturb = np.stack([noise.draw("race_order", [event["event_id"], driver], distribution="gumbel")
                        for driver in event["drivers"]], axis=-1)
    tie = np.stack([noise.draw("race_order_tie", [event["event_id"], driver], distribution="uniform")
                   for driver in event["drivers"]], axis=-1)
    score = strength + perturb
    _finite("perturbed race strengths", score)
    return _rank(score, tie, running)


def _scenarios(design, snapshots, scenarios, advancing):
    if not isinstance(scenarios, dict) or set(scenarios) != {s["event"]["event_id"] for s in snapshots}:
        raise ValueError("An explicit conditional scenario for every forecast event is required")
    circuits = set(design.catalog["circuits"])
    for snap in snapshots:
        event, entrants = snap["event"], snap["entrants"]
        scenario = scenarios[event["event_id"]]
        if not isinstance(scenario, dict) or set(scenario) != {"wet", "neutralised", "exposure_laps", "grid_policy"}:
            raise ValueError("Scenario requires wet, neutralised, exposure_laps and grid_policy")
        if type(scenario["wet"]) is not bool or type(scenario["neutralised"]) is not bool:
            raise ValueError("Unknown conditions cannot be silently labelled dry or green")
        if event["circuit_id"] not in circuits:
            raise ValueError("Unseen circuit coefficients require a registered extension policy")
        exposure = scenario["exposure_laps"]
        if (not isinstance(exposure, dict) or set(exposure) != {r["driver_id"] for r in entrants}
                or any(type(v) is not int or not 0 < v <= event["scheduled_laps"] for v in exposure.values())):
            raise ValueError("Each entrant needs an explicit exposure within the scheduled distance")
        policy = "actual_grid" if snap["information_set"] == "actual_grid" else "qualifying_order_no_penalties"
        if scenario["grid_policy"] != policy:
            raise ValueError("Grid scenario does not match the forecast information set")
        if not isinstance(advancing, (tuple, list)) or any(
                type(a) is not int or not 0 < a < b for a, b in zip(advancing, (len(entrants), *advancing))):
            raise ValueError("Qualifying fields must strictly shrink")


def iter_conditional_weekends(design, snapshots, posterior, scenarios, *, advancing, seed,
                              training_arrays_key, draw_batch_size=DEFAULT_DRAW_BATCH):
    """One conditional future trajectory per joint posterior draw, in batches.

    The SAME posterior mapping supplies states and observation coefficients. The
    caller must load it through checked archive verification and finish consuming
    every batch before recording a complete output. This utility certifies neither
    convergence nor scenario/source validity. Repeated trajectories need a frozen
    replication/precision policy; this is not a new ranking producer.
    """
    if type(draw_batch_size) is not int or draw_batch_size < 1:
        raise ValueError("Draw batch size must be a positive integer")
    _scenarios(design, snapshots, scenarios, advancing)
    p, shape, parameter_key = _observation_parameters(design, posterior, draw_batch_size)
    ObservationNoise(seed, shape, (0, shape[1]))
    scenario_key = hashlib.sha256(json.dumps(scenarios, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for batch in iter_forecast_states(design, snapshots, posterior, seed=seed,
            training_arrays_key=training_arrays_key, draw_batch_size=draw_batch_size):
        start, stop = batch["draw_range"]
        part = {name:value[:, start:stop] for name, value in p.items()}
        noise, events = ObservationNoise(seed, shape, (start, stop)), []
        for event in batch["events"]:
            scenario = scenarios[event["event_id"]]
            qualifying = qualifying_segments(event, part, noise, advancing=advancing, full_df=p["nu"])
            coefficients = context_coefficients(part, event["catalog"]["circuit_index"],
                wet=scenario["wet"], neutralised=scenario["neutralised"])
            if scenario["grid_policy"] == "actual_grid":
                slots = {r["driver_id"]: r["grid_feature_slot"] for r in event["catalog"]["entries"]}
                grid = np.array([slots[driver] for driver in event["drivers"]])
            else:
                grid = qualifying["positions"]
            strength = race_strength(event["states"]["race_total"], grid, coefficients)
            exposure = np.array([scenario["exposure_laps"][driver] for driver in event["drivers"]])
            completion = draw_completion(event, coefficients, exposure, noise)
            order = draw_running_order(event, strength, ~completion["failed"], noise)
            order_factor = order_log_probability(strength, order)
            events.append(dict(event_id=event["event_id"], drivers=event["drivers"],
                catalog=event["catalog"], qualifying=qualifying, grid_feature_slot=grid,
                context=coefficients, strength=strength, completion=completion,
                running_positions=order, order_log_probability=order_factor,
                joint_order_completion_log_probability=order_factor + completion["log_probability"].sum(axis=-1)))
        yield dict(purpose="development_conditional_weekend", axes=["chain", "draw", "driver"],
            events=events, posterior_shape=list(shape), draw_range=[start, stop], seed=seed,
            training_arrays_key=training_arrays_key, state_parameter_key=batch["consumed_posterior_key"],
            observation_parameter_key=parameter_key, scenario_sha256=scenario_key,
            state_implementation_sha256=batch["implementation_sha256"], implementation_sha256=source_hash,
            qualifying_advancing=list(advancing), conditional_outcomes_available=True,
            conditions_forecast=False, official_grid_generated=False, classification_generated=False,
            convergence_verified=False, source_availability_verified=False,
            independent_evidence_established=False, publication_allowed=False)
