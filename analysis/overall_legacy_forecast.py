"""Legacy qualifying-only future features, preserving lineage effects and medians.

Development forecast adapter, not a fit loader or verified source archive. The
standard qualifying model supplies every state; no race-updated parameters enter.
"""
import hashlib
from pathlib import Path

import numpy as np

from analysis.overall_baselines import (COEFFICIENTS, legacy_feature_medians,
                                        log_probabilities, ordered_arrays)
from analysis.overall_forecast_catalog import build_qualifying_catalog
from analysis.overall_forecast_states import (DEFAULT_DRAW_BATCH, _Noise, _centre,
    _checked_parameter_digest, _trend, validate_catalog_path)
from f1rank.lineage import REGULATION_RESETS
from f1rank.model import CAR_FIRST_SD
from f1rank.overall.fitting import arrays_digest


POLICY = "legacy_qualifying_forecast_v1"
# Pass this to f1rank.fit.fit(extra_sites=...) when a separately registered
# baseline fit is run. Its default export drops these sampled coordinates;
# centred historical ratings cannot replace the uncentred terminal states.
FORECAST_EXTRA_SITES = ("level_z", "skill_z", "car_step_z", "car_jump_z",
                        "track_load_z", "compat_z")


class LegacyNoise(_Noise):
    def draw(self, site, identity, *, student=False):
        return super().draw("legacy_qualifying:" + site, identity, student=student)


def _parameters(design, posterior, fit_metadata):
    if (not isinstance(fit_metadata, dict) or fit_metadata.get("model_kw") != {} or fit_metadata.get("start_season") != 2010
            or design.start_season != 2010 or design.sprint_quali_until is not None):
        raise ValueError("Legacy forecast requires the standard qualifying-only model")
    if any(name.startswith("overall_") for name in posterior):
        raise ValueError("Race-updated successor draws cannot substitute for a qualifying-only fit")
    q = design.arrays()
    scalar = ("sd_level", "sd_skill_race", "sd_skill_season", "exp_gain", "exp_scale", "age_slope",
              "rho", "rho_reset", "sd_car_race", "sd_car_season", "sd_car_reset",
              "sd_track_load", "sd_compat", "sd_car_event", "sd_driver_form")
    vectors = dict(level_z=q["n_drivers"], skill_z=q["n_entries"], car_step_z=q["n_cars"],
                   car_jump_z=q["n_cars"], track_load_z=q["n_team_seasons"], compat_z=q["n_stints"])
    missing = sorted(set((*scalar, *vectors)) - set(posterior))
    if missing:
        raise ValueError("Missing legacy qualifying state draws: " + ", ".join(missing)
                         + "; the default legacy archive omits forecast coordinates. "
                         "Registered forecast fits must retain FORECAST_EXTRA_SITES.")
    p = {name: np.asarray(posterior[name]) for name in (*scalar, *vectors)}
    shape = p["sd_level"].shape
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("Legacy posterior requires paired chain/draw axes")
    for name, value in p.items():
        expected = shape + ((vectors[name],) if name in vectors else ())
        if value.shape != expected or value.dtype.kind not in "fiu":
            raise ValueError(f"Invalid legacy axes for {name}")
    return p, q


def _terminal(design, p, q):
    e, c = design.entries, design.cars
    if not np.array_equal(e.entry_idx, np.arange(len(e))) or not np.array_equal(c.car_idx, np.arange(len(c))):
        raise ValueError("Legacy training states are not in declared index order")
    drivers, cars = {}, {}
    for row in e.itertuples():
        i = row.entry_idx
        if q["entry_first"][i]:
            walk = (p["sd_level"] * p["level_z"][..., row.driver_idx]).astype(float)
        else:
            _, _, walk = drivers[row.driver_id]
            same = q["entry_same_season"][i]
            gap = np.sqrt(q["entry_event_gap"][i] if same else max(q["entry_season_gap"][i], 1))
            walk = walk + p["sd_skill_race" if same else "sd_skill_season"] * gap * p["skill_z"][..., i]
        drivers[row.driver_id] = (row.event_idx, row.season, walk)
    for row in c.itertuples():
        i = row.car_idx
        first, start, reset = q["car_first"][i], q["car_season_start"][i], q["car_reset"][i]
        if first or start:
            jump = np.where(first, CAR_FIRST_SD, np.where(reset, p["sd_car_reset"], p["sd_car_season"]))
            car = jump * p["car_jump_z"][..., i]
        else:
            car = p["sd_car_race"] * p["car_step_z"][..., i]
        if not first:
            car = car + (p["rho_reset" if reset else "rho"] if start else 1.) * cars[row.team][1]
        cars[row.team] = (row.season, car)
    return drivers, cars


def _batch(design, catalogs, p, q, noise):
    drivers, cars = _terminal(design, p, q)
    compat = {(r.driver_id, r.team):p["sd_compat"] * p["compat_z"][..., r.stint_idx]
              for r in design.entries.drop_duplicates("stint_idx").itertuples()}
    loads = {(r.team, int(r.season)):p["sd_track_load"] * p["track_load_z"][..., index]
             for index, r in design.cars.groupby("team_season_idx").first().iterrows()}
    events = []
    for catalog in catalogs:
        event, year = catalog["event_id"], int(catalog["event_id"][:4])
        entries, field = catalog["entries"], catalog["field"]
        teams = sorted({r["team"] for r in entries})
        entry_car = np.array([teams.index(r["team"]) for r in entries])
        walk, effect, form = [], [], []
        for row in entries:
            driver, team = row["driver_id"], row["team"]
            if driver not in drivers:
                state = p["sd_level"] * noise.draw("driver_level", driver)
            else:
                previous_event, previous_year, state = drivers[driver]
                same = previous_year == year
                gap = np.sqrt(catalog["event_index"] - previous_event if same else max(year - previous_year, 1))
                state = state + gap * p["sd_skill_race" if same else "sd_skill_season"] * noise.draw("driver_step", [driver, event])
            drivers[driver] = (catalog["event_index"], year, state)
            pair = (driver, team)
            if pair not in compat:
                compat[pair] = p["sd_compat"] * noise.draw("compatibility", pair)
            walk.append(state)
            effect.append(compat[pair])
            form.append(p["sd_driver_form"] * noise.draw("qualifying_day", [driver, event]))
        stack = lambda values: np.stack(values, axis=-1)
        skill = _centre(stack(walk) + _trend(p, [r["experience"] for r in entries], [r["age_over"] for r in entries]))
        compatibility, day = _centre(stack(effect)), _centre(stack(form))
        persistent, track, car_day = [], [], []
        factor = 0. if catalog["new_circuit"] else float(q["circuit_factor"][catalog["circuit_index"]])
        for team in teams:
            if team not in cars:
                car = CAR_FIRST_SD * noise.draw("car_first", [team, event])
            else:
                last_year, car = cars[team]
                if year == last_year:
                    car = car + p["sd_car_race"] * noise.draw("car_step", [team, event], student=True)
                else:
                    reset = year in REGULATION_RESETS
                    car = p["rho_reset" if reset else "rho"] * car + p["sd_car_reset" if reset else "sd_car_season"] * noise.draw("car_jump", [team, event])
            cars[team] = (year, car)
            pair = (team, year)
            if pair not in loads:
                loads[pair] = p["sd_track_load"] * noise.draw("track_load", pair)
            persistent.append(car)
            track.append(loads[pair] * factor)
            car_day.append(p["sd_car_event"] * noise.draw("car_qualifying_day", [team, event]))
        c, t, transient = [_centre(stack(v)) for v in (persistent, track, car_day)]
        states = dict(skill=skill, compat=compatibility, car=c, car_track=t,
            driver_form=day, car_event=transient,
            qualifying_total=skill + compatibility + day + (c + t + transient)[..., entry_car])
        if any(not np.isfinite(value).all() for value in states.values()):
            raise ValueError("Nonfinite legacy forecast state; no silent replacement")
        events.append(dict(event_id=event, drivers=field, cars=teams, entry_car=entry_car,
            states=states, catalog=catalog, circuit_factor=factor,
            circuit_factor_policy="legacy_unseen_no_adjustment" if catalog["new_circuit"] else "fixed_training_factor"))
    return events


def iter_legacy_qualifying_states(design, snapshots, posterior, *, cutoff, fit_metadata,
                                  seed, training_arrays_key, draw_batch_size=DEFAULT_DRAW_BATCH):
    """Propagate the standard legacy model; certify neither fit nor source claims."""
    if type(draw_batch_size) is not int or draw_batch_size < 1:
        raise ValueError("Draw batch size must be a positive integer")
    if not isinstance(snapshots, (list, tuple)) or not snapshots:
        raise ValueError("Complete announced fields are required")
    p, q = _parameters(design, posterior, fit_metadata)
    shape = p["sd_level"].shape
    LegacyNoise(seed, shape)
    catalogs = [build_qualifying_catalog(design, s, cutoff=cutoff) for s in snapshots]
    validate_catalog_path(snapshots, catalogs, training_arrays_key)
    key = _checked_parameter_digest(p, draw_batch_size)
    source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for start in range(0, shape[1], draw_batch_size):
        stop = min(start + draw_batch_size, shape[1])
        part = {name:value[:, start:stop] for name, value in p.items()}
        events = _batch(design, catalogs, part, q, LegacyNoise(seed, shape, slice(start, stop)))
        yield dict(purpose="development_legacy_qualifying_states", policy=POLICY,
            events=events, posterior_shape=list(shape), draw_range=[start, stop], seed=seed,
            training_arrays_key=training_arrays_key, consumed_posterior_key=key,
            implementation_sha256=source, source_availability_verified=False,
            convergence_verified=False, independent_evidence_established=False, publication_allowed=False)


def forecast_legacy_features(design, snapshots, posterior, **kwargs):
    """Collect all small future-state arrays, then take medians of paired sums."""
    result, stop = None, 0
    for batch in iter_legacy_qualifying_states(design, snapshots, posterior, **kwargs):
        start, stop = batch["draw_range"]
        shape = tuple(batch["posterior_shape"])
        if result is None:
            result = {k:v for k,v in batch.items() if k not in ("events", "draw_range")}
            result["events"] = [dict(event, states={name:np.empty(shape + value.shape[2:])
                for name, value in event["states"].items()}) for event in batch["events"]]
        for output, event in zip(result["events"], batch["events"]):
            for name, value in event["states"].items():
                output["states"][name][:, start:stop] = value
    if result is None or stop != result["posterior_shape"][1]:
        raise ValueError("Incomplete legacy qualifying forecast")
    for event in result["events"]:
        features = legacy_feature_medians(event["states"])
        event["race_features"] = dict(driver=features["driver"], car=features["car"][event["entry_car"]])
    result["feature_policy"] = "median_of_paired_persistent_states_without_future_transients"
    return result


def forecast_legacy_race(qualifying_forecast, coefficient_posterior, running_positions):
    """Actual-grid conditional scores using forecast legacy plug-in features.

    Running positions are later response data, keyed by every declared driver.
    No grid or qualifying features are read from those outcomes. This adapter
    does not choose finish-status exclusions or certify source/convergence.
    Coefficient draws need not share the qualifying fit's draw count: its
    feature medians are fixed under the legacy baseline definition.
    """
    if (qualifying_forecast.get("purpose") != "development_legacy_qualifying_states"
            or qualifying_forecast.get("policy") != POLICY
            or qualifying_forecast.get("feature_policy") != "median_of_paired_persistent_states_without_future_transients"):
        raise ValueError("Legacy race prediction requires the complete qualifying feature forecast")
    events = qualifying_forecast["events"]
    if not isinstance(running_positions, dict) or set(running_positions) != {e["event_id"] for e in events}:
        raise ValueError("Every forecast event needs its complete declared response panel")
    rows = []
    for event in events:
        catalog, field = event["catalog"], event["drivers"]
        positions = running_positions[event["event_id"]]
        if catalog["information_set"] != "actual_grid":
            raise ValueError("Legacy race-stage prediction requires an actual-grid forecast origin")
        if not isinstance(positions, dict) or set(positions) != set(field):
            raise ValueError("Running positions must retain the full declared forecast field")
        grid = {r["driver_id"]:r["grid_feature_slot"] for r in catalog["entries"]}
        rows.append(dict(event_id=event["event_id"], drivers=field,
            grid_slot=[grid[d] for d in field], running_position=[positions[d] for d in field],
            **event["race_features"]))
    arm = "legacy_grid_qualifying"
    panel = ordered_arrays(rows, arm=arm)
    probabilities = log_probabilities(coefficient_posterior, panel)
    coefficients = {name:np.asarray(coefficient_posterior[name]) for name in COEFFICIENTS[arm]}
    return dict(purpose="development_legacy_race_forecast", arm=arm,
        event_ids=panel["event_ids"], fields=panel["fields"],
        log_probabilities=probabilities, axes=["chain", "draw", "event"],
        coefficient_posterior_shape=list(probabilities.shape[:2]),
        coefficient_parameter_key=arrays_digest(dict(coefficients=coefficients)),
        qualifying_parameter_key=qualifying_forecast["consumed_posterior_key"],
        qualifying_training_arrays_key=qualifying_forecast["training_arrays_key"],
        qualifying_posterior_shape=qualifying_forecast["posterior_shape"],
        origins=[dict(event_id=e["event_id"], origin_at=e["catalog"]["origin_at"],
                      snapshot_sha256=e["catalog"]["snapshot_sha256"]) for e in events],
        source_availability_verified=False, convergence_verified=False,
        independent_evidence_established=False, publication_allowed=False)
