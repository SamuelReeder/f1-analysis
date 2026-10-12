"""Propagate joint latent states without observing a forecast weekend.

Development utility, not a posterior loader, outcome predictor or publication
gate. A production caller must load a source-matched, converged joint posterior
through the fitter. This module preserves chain/draw axes and the existing model
transitions; it does not supply future conditions or fit new parameters.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

from analysis.overall_forecast_catalog import build_catalog, _time
from f1rank.model import CAR_FIRST_SD, CAR_STEP_DF
from f1rank.lineage import REGULATION_RESETS

# The fixed scale of overall_car_contrast in f1rank.overall.model.model.
# Its trace prior is checked against this value in the equivalence fixture.
RACE_CAR_SD = 1.5
DEFAULT_DRAW_BATCH = 128
TRACK_POLICY_PATH = Path(__file__).resolve().parents[1] / "docs/overall_track_factor_policy.json"
TRACK_POLICY_BYTES = TRACK_POLICY_PATH.read_bytes()
TRACK_POLICY = json.loads(TRACK_POLICY_BYTES)
UNSEEN_TRACK_POLICY = TRACK_POLICY["id"]


def _parameters(design, posterior, *, validate_values=True):
    q = design.qualifying.arrays()
    scalar = ("sd_level", "sd_skill_race", "sd_skill_season", "exp_gain", "exp_scale", "age_slope",
        "overall_sd_level", "overall_sd_drift_race", "overall_sd_drift_season", "overall_qualifying_link",
        "rho", "rho_reset", "sd_car_race", "sd_car_season", "sd_car_reset", "sd_car_event",
        "sd_track_load", "sd_compat", "overall_team_link", "overall_sd_team", "overall_car_link",
        "overall_sd_car_day", "sd_driver_form", "overall_shared_weekend", "overall_sd_day")
    vectors = dict(level_z=q["n_drivers"], skill_z=q["n_entries"],
        overall_level_z=q["n_drivers"], overall_walk_z=q["n_entries"],
        car_step_z=q["n_cars"], car_jump_z=q["n_cars"], track_load_z=q["n_team_seasons"],
        compat_z=q["n_era_stints"], overall_team_z=q["n_era_stints"],
        overall_car_contrast=design.outcomes["car_basis"].shape[1])
    names = [*scalar, *vectors]
    if any(name not in posterior for name in names):
        raise ValueError("Joint posterior lacks required latent states; centred ratings cannot substitute")
    # Preserve archive precision instead of copying every stored latent array.
    p = {name: np.asarray(posterior[name]) for name in names}
    shape = p["sd_level"].shape
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("Grouped chain and draw axes are required")
    for name, value in p.items():
        expected = shape + ((vectors[name],) if name in vectors else ())
        if value.dtype.kind not in "fiu" or value.shape != expected:
            raise ValueError(f"Invalid joint axes or values for {name}")
        if validate_values:
            _valid_values(name, value)
    return p, q


def _valid_values(name, value):
    if not np.isfinite(value).all():
        raise ValueError(f"Nonfinite values for {name}")
    if (name.startswith("sd_") or name.startswith("overall_sd_")) and (value < 0).any():
        raise ValueError(f"Negative scale for {name}")
    if name == "exp_scale" and (value <= 0).any():
        raise ValueError("Invalid experience scale")
    if name in ("rho", "rho_reset") and ((value < 0) | (value > 1)).any():
        raise ValueError("Invalid car carryover")


def _checked_parameter_digest(parameters, batch_size):
    """Same identity as arrays_digest, without a full array's bytes/boolean copy.

    Validate every draw before yielding anything, so a bad final batch cannot
    silently leave a usable partial forecast. C-order bytes preserve the existing
    identity even for noncontiguous arrays and memory-mapped archive members.
    """
    digest = hashlib.sha256()
    for name, value in sorted(parameters.items()):
        digest.update(f"parameters:{name}:{value.dtype.str}:{value.shape}".encode())
        for chain in range(value.shape[0]):
            for start in range(0, value.shape[1], batch_size):
                part = value[chain, start:start + batch_size]
                _valid_values(name, part)
                digest.update(np.ascontiguousarray(part).tobytes())
    return digest.hexdigest()


def _walk(first, increments, carry=None):
    result = np.empty_like(increments)
    for i, start in enumerate(first):
        result[..., i] = increments[..., i]
        if not start:
            result[..., i] += result[..., i - 1] * (1 if carry is None else carry[..., i])
    return result


def _history(design, p, q):
    e, c = design.qualifying.entries, design.qualifying.cars
    if not np.array_equal(e.entry_idx, np.arange(len(e))) or not np.array_equal(c.car_idx, np.arange(len(c))):
        raise ValueError("Training states are not in their declared index order")
    states = {}
    for dest, level, zlevel, race, season, zstep in (
        ("qualifying_walk", "sd_level", "level_z", "sd_skill_race", "sd_skill_season", "skill_z"),
        ("race_residual", "overall_sd_level", "overall_level_z", "overall_sd_drift_race", "overall_sd_drift_season", "overall_walk_z")):
        step = np.where(q["entry_same_season"], p[race][..., None] * np.sqrt(q["entry_event_gap"]),
            p[season][..., None] * np.sqrt(np.maximum(q["entry_season_gap"], 1)))
        increments = np.where(q["entry_first"], p[level][..., None] * p[zlevel][..., q["entry_driver"]], step * p[zstep])
        states[dest] = _walk(q["entry_first"], increments)
    first, start, reset = q["car_first"], q["car_season_start"], q["car_reset"]
    carry = np.where(start, np.where(reset, p["rho_reset"][..., None], p["rho"][..., None]), 1.)
    jump = np.where(first, CAR_FIRST_SD,
        np.where(reset, p["sd_car_reset"][..., None], p["sd_car_season"][..., None]))
    increments = np.where(first | start, jump * p["car_jump_z"], p["sd_car_race"][..., None] * p["car_step_z"])
    states["car_persistent"] = _walk(first, increments, carry)
    states["qualifying_skill"] = states["qualifying_walk"] + _trend(p, q["entry_experience"], q["entry_age_over"])
    states["race_skill"] = p["overall_qualifying_link"][..., None] * states["qualifying_skill"] + states["race_residual"]
    states["qualifying_team"] = p["sd_compat"][..., None] * p["compat_z"][..., q["entry_era_stint"]]
    states["race_team"] = (p["overall_team_link"][..., None] * states["qualifying_team"]
        + p["overall_sd_team"][..., None] * p["overall_team_z"][..., q["entry_era_stint"]])
    states["car_season_contrast"] = p["overall_car_contrast"] @ np.asarray(design.outcomes["car_basis"]).T
    return states


def _trend(p, experience, age_over):
    return (p["exp_gain"][..., None] * (1 - np.exp(-np.asarray(experience) / p["exp_scale"][..., None]))
        + p["age_slope"][..., None] * np.asarray(age_over))


def reconstruct_history(design, posterior):
    """Reconstruct uncentred states; these values are not comparable driver ranks."""
    p, q = _parameters(design, posterior)
    return _history(design, p, q)


class _Noise:
    """Streams attach to state identities, independently of field row ordering."""
    def __init__(self, seed, shape, draw_slice=None):
        if type(seed) is not int or seed < 0:
            raise ValueError("An explicit nonnegative integer forecast seed is required")
        self.seed, self.shape, self.draw_slice = seed, shape, draw_slice

    def draw(self, site, identity, *, student=False):
        key = json.dumps([self.seed, site, identity], separators=(",", ":"), allow_nan=False).encode()
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest(), "big"))
        value = rng.standard_t(CAR_STEP_DF, size=self.shape) if student else rng.normal(size=self.shape)
        # Replay the original short chain/draw stream before taking the batch.
        # Skipping Normal/Student-t outputs by raw RNG steps would be incorrect:
        # their algorithms can consume a variable number of underlying values.
        return value if self.draw_slice is None else value[:, self.draw_slice].copy()


def _centre(value):
    return value - value.mean(axis=-1, keepdims=True)


def _catalogs(design, snapshots, training_arrays_key):
    if not isinstance(snapshots, (list, tuple)) or not snapshots:
        raise ValueError("A complete future field path is required")
    catalogs = [build_catalog(design, s) for s in snapshots]
    return validate_catalog_path(snapshots, catalogs, training_arrays_key)


def validate_catalog_path(snapshots, catalogs, training_arrays_key):
    """Shared announced-path checks for separately identified model catalogues."""
    if not catalogs or len(catalogs) != len(snapshots):
        raise ValueError("A complete matching future catalogue path is required")
    if any(c["training_arrays_key"] != training_arrays_key for c in catalogs):
        raise ValueError("Posterior and forecast training arrays do not match")
    origins = {c["origin_at"] for c in catalogs}
    if len(origins) != 1:
        raise ValueError("A forecast path must use one information origin")
    seen, metadata = {}, {}
    for i, (s, catalog) in enumerate(zip(snapshots, catalogs)):
        if catalog["calendar_after_cutoff"] != [c["event_id"] for c in catalogs[:i + 1]]:
            raise ValueError("The path omits an intervening field or changes its calendar")
        if i and _time(s["event"]["race_start"]) <= _time(snapshots[i - 1]["event"]["race_start"]):
            raise ValueError("Future event dates must be chronological")
        for row in s["entrants"]:
            driver = row["driver_id"]
            if row["prior_entries_after_cutoff"] != seen.get(driver, 0):
                raise ValueError("Announced participation does not match the future path")
            seen[driver] = seen.get(driver, 0) + 1
            declared = {k:row[k] for k in ("dob", "debut_season", "experience_before_cutoff") if k in row}
            if driver in metadata and declared != metadata[driver]:
                raise ValueError("Driver metadata changed along a single-origin forecast path")
            metadata[driver] = declared
    return catalogs


def _terminal_history(design, p, q):
    """Only latest walks are needed; never allocate draw x historical-entry states."""
    e, c = design.qualifying.entries, design.qualifying.cars
    if not np.array_equal(e.entry_idx, np.arange(len(e))) or not np.array_equal(c.car_idx, np.arange(len(c))):
        raise ValueError("Training states are not in their declared index order")
    drivers, cars = {}, {}
    for row in e.itertuples():
        i = row.entry_idx
        if q["entry_first"][i]:
            # Match the array recurrence's promotion from the floating gap array.
            qw = (p["sd_level"] * p["level_z"][..., row.driver_idx]).astype(float)
            rr = (p["overall_sd_level"] * p["overall_level_z"][..., row.driver_idx]).astype(float)
        else:
            _, _, qw, rr = drivers[row.driver_id]
            same = q["entry_same_season"][i]
            gap = np.sqrt(q["entry_event_gap"][i] if same else max(q["entry_season_gap"][i], 1))
            qw = qw + p["sd_skill_race" if same else "sd_skill_season"] * gap * p["skill_z"][..., i]
            rr = rr + p["overall_sd_drift_race" if same else "overall_sd_drift_season"] * gap * p["overall_walk_z"][..., i]
        drivers[row.driver_id] = (row.event_idx, row.season, qw, rr)
    for row in c.itertuples():
        i = row.car_idx
        first, start, reset = q["car_first"][i], q["car_season_start"][i], q["car_reset"][i]
        if first or start:
            jump = np.where(first, CAR_FIRST_SD,
                np.where(reset, p["sd_car_reset"], p["sd_car_season"]))
            car = jump * p["car_jump_z"][..., i]
        else:
            car = p["sd_car_race"] * p["car_step_z"][..., i]
        if not first:
            carry = p["rho_reset" if reset else "rho"] if start else 1.
            car = car + carry * cars[row.team][1]
        cars[row.team] = (row.season, car)
    return drivers, cars


def _forecast_batch(design, catalogs, p, q, noise):
    driver_state, car_state = _terminal_history(design, p, q)
    stints, loads, race_cars = {}, {}, {}
    season_contrast = p["overall_car_contrast"] @ np.asarray(design.outcomes["car_basis"]).T
    for row in design.qualifying.entries.drop_duplicates("era_stint_idx").itertuples():
        key = (row.driver_id, row.team, row.era)
        stints[key] = (p["sd_compat"] * p["compat_z"][..., row.era_stint_idx],
            p["overall_sd_team"] * p["overall_team_z"][..., row.era_stint_idx])
    season_cars = design.qualifying.cars.groupby("team_season_idx").first()
    counts = season_cars.groupby("season").size().to_dict()
    for index, row in season_cars.iterrows():
        key = (row.team, int(row.season))
        loads[key] = p["sd_track_load"] * p["track_load_z"][..., index]
        # The fitted contrasts integrate out the iid Normal common season mean.
        # Restore that unobserved mean before drawing a new team in that season.
        # Simply combining zero-sum old contrasts with an independent new team
        # would understate the newcomer's relative uncertainty.
        mean = RACE_CAR_SD / np.sqrt(counts[row.season]) * noise.draw("race_car_mean", int(row.season))
        race_cars[key] = season_contrast[..., index] + mean

    events = []
    for catalog in catalogs:
        event, season = catalog["event_id"], int(catalog["event_id"][:4])
        entries = catalog["entries"]
        drivers, teams = catalog["field"], sorted({r["team"] for r in entries})
        entry_car = np.array([teams.index(r["team"]) for r in entries])
        qwalk, residual, compat, rextra, qday, rday = [], [], [], [], [], []
        for row in entries:
            driver, team = row["driver_id"], row["team"]
            if driver not in driver_state:
                qw = p["sd_level"] * noise.draw("driver_level", driver)
                rr = p["overall_sd_level"] * noise.draw("race_level", driver)
            else:
                last_event, last_season, qw, rr = driver_state[driver]
                same = last_season == season
                gap = np.sqrt(catalog["event_index"] - last_event if same else max(season - last_season, 1))
                qw = qw + gap * p["sd_skill_race" if same else "sd_skill_season"] * noise.draw("driver_step", [driver, event])
                rr = rr + gap * p["overall_sd_drift_race" if same else "overall_sd_drift_season"] * noise.draw("race_step", [driver, event])
            driver_state[driver] = (catalog["event_index"], season, qw, rr)
            stint = (driver, team, catalog["era"])
            if stint not in stints:
                stints[stint] = (p["sd_compat"] * noise.draw("compatibility", stint),
                    p["overall_sd_team"] * noise.draw("race_team", stint))
            cq, cr = stints[stint]
            qform = p["sd_driver_form"] * noise.draw("qualifying_day", [driver, event])
            rform = p["overall_shared_weekend"] * qform + p["overall_sd_day"] * noise.draw("race_day", [driver, event])
            qwalk.append(qw); residual.append(rr); compat.append(cq); rextra.append(cr)
            qday.append(qform); rday.append(rform)
        stack = lambda x: np.stack(x, axis=-1)
        uncentred_q = stack(qwalk) + _trend(p, [r["experience"] for r in entries], [r["age_over"] for r in entries])
        skill = _centre(uncentred_q)
        race_skill = _centre(p["overall_qualifying_link"][..., None] * uncentred_q + stack(residual))
        qteam = _centre(stack(compat))
        rteam = _centre(p["overall_team_link"][..., None] * stack(compat) + stack(rextra))
        qform, rform = _centre(stack(qday)), _centre(stack(rday))
        persistent, track, qcars, rcars = [], [], [], []
        if catalog["likelihood_arm"] == "current_race_only":
            factor, factor_policy = 0., "race_only_axis_removed"
        elif catalog["new_circuit"]:
            factor = (TRACK_POLICY["distribution"]["mean"]
                      + TRACK_POLICY["distribution"]["standard_deviation"]
                      * noise.draw("qualifying_circuit_factor", catalog["circuit_id"]))
            factor_policy = UNSEEN_TRACK_POLICY
        else:
            factor = float(q["circuit_factor"][catalog["circuit_index"]])
            factor_policy = "fixed_training_factor"
        for team in teams:
            if team not in car_state:
                car = CAR_FIRST_SD * noise.draw("car_first", [team, event])
            else:
                last_season, car = car_state[team]
                if last_season == season:
                    car = car + p["sd_car_race"] * noise.draw("car_step", [team, event], student=True)
                else:
                    reset = season in REGULATION_RESETS
                    car = car * p["rho_reset" if reset else "rho"] + p["sd_car_reset" if reset else "sd_car_season"] * noise.draw("car_jump", [team, event])
            car_state[team] = (season, car)
            key = (team, season)
            if key not in loads:
                loads[key] = p["sd_track_load"] * noise.draw("track_load", key)
                race_cars[key] = RACE_CAR_SD * noise.draw("race_car", key)
            fit_track = factor * loads[key]
            persistent.append(car); track.append(fit_track)
            qcars.append(car + fit_track + p["sd_car_event"] * noise.draw("car_qualifying_day", [team, event]))
            rcars.append(p["overall_car_link"] * (car + fit_track) + race_cars[key]
                + p["overall_sd_car_day"] * noise.draw("car_race_day", [team, event]))
        qcar, rcar = _centre(stack(qcars)), _centre(stack(rcars))
        states = dict(qualifying_skill=skill, race_skill=race_skill, qualifying_team=qteam, race_team=rteam,
            qualifying_day=qform, race_day=rform, qualifying_car=qcar, race_car=rcar,
            car_persistent=_centre(stack(persistent)), car_track=_centre(stack(track)),
            qualifying_total=skill + qteam + qform + qcar[..., entry_car],
            race_total=race_skill + rteam + rform + rcar[..., entry_car])
        if any(not np.isfinite(value).all() for value in states.values()):
            raise ValueError("Nonfinite forecast state; no clipping or silent replacement is allowed")
        events.append(dict(event_id=event, drivers=drivers, cars=teams, entry_car=entry_car,
            states=states, catalog=catalog, circuit_factor=factor,
            circuit_factor_policy=factor_policy))
    return events


def iter_forecast_states(design, snapshots, posterior, *, seed, training_arrays_key,
                         draw_batch_size=DEFAULT_DRAW_BATCH, unseen_track_policy=None):
    """Yield a complete announced path for each contiguous batch of joint draws.

    All chains stay together. The caller must account for every draw_range;
    stopping the iterator early does not constitute a complete prediction.
    Input posterior arrays may be read-only memory maps. This is not a loader or
    fit checker, and callers must still verify archive/source/convergence.
    """
    if type(draw_batch_size) is not int or draw_batch_size < 1:
        raise ValueError("Draw batch size must be a positive integer")
    p, q = _parameters(design, posterior, validate_values=False)
    shape = p["sd_level"].shape
    _Noise(seed, shape)  # fail before scanning arrays for an invalid seed
    catalogs = _catalogs(design, snapshots, training_arrays_key)
    if unseen_track_policy not in (None, UNSEEN_TRACK_POLICY):
        raise ValueError("Unsupported unseen qualifying track-factor policy")
    if unseen_track_policy is None and any(c["new_circuit"] and c["likelihood_arm"] != "current_race_only" for c in catalogs):
        raise ValueError("Unseen qualifying track factor requires an explicit population policy")
    key = _checked_parameter_digest(p, draw_batch_size)
    implementation = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for start in range(0, shape[1], draw_batch_size):
        stop = min(start + draw_batch_size, shape[1])
        part = {name:value[:, start:stop] for name, value in p.items()}
        events = _forecast_batch(design, catalogs, part, q, _Noise(seed, shape, slice(start, stop)))
        yield dict(purpose="development_state_forecast", axes=["chain", "draw", "driver_or_car"],
            events=events, training_arrays_key=training_arrays_key, seed=seed,
            implementation_sha256=implementation, consumed_posterior_key=key,
            posterior_shape=list(shape), draw_range=[start, stop],
            unseen_track_policy=unseen_track_policy,
            track_policy_sha256=hashlib.sha256(TRACK_POLICY_BYTES).hexdigest(),
            latent_states_propagated=True, convergence_verified=False, source_availability_verified=False,
            outcome_prediction_available=False, independent_evidence_established=False, publication_allowed=False)


def forecast_states(design, snapshots, posterior, *, seed, training_arrays_key,
                    draw_batch_size=DEFAULT_DRAW_BATCH, unseen_track_policy=None):
    """Collect the complete forecast; use the iterator for bounded output memory.

    Supply a snapshot for every intervening event at one information origin.
    This convenience wrapper retains every forecast state but reconstructs only
    a batch of terminal history at once. Batching does not change the draw set.
    """
    result = None
    for batch in iter_forecast_states(design, snapshots, posterior, seed=seed,
            training_arrays_key=training_arrays_key, draw_batch_size=draw_batch_size,
            unseen_track_policy=unseen_track_policy):
        start, stop = batch["draw_range"]
        if result is None:
            result = {k:v for k, v in batch.items() if k not in ("events", "draw_range")}
            shape = tuple(batch["posterior_shape"])
            result["events"] = [dict(event, states={name:np.empty(shape + value.shape[2:], dtype=value.dtype)
                for name, value in event["states"].items()}) for event in batch["events"]]
            for event in result["events"]:
                if isinstance(event["circuit_factor"], np.ndarray):
                    event["circuit_factor"] = np.empty(shape)
        for output, event in zip(result["events"], batch["events"]):
            if isinstance(event["circuit_factor"], np.ndarray):
                output["circuit_factor"][:, start:stop] = event["circuit_factor"]
            for name, value in event["states"].items():
                output["states"][name][:, start:stop] = value
    return result
