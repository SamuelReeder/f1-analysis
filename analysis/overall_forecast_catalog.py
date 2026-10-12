"""Outcome-free forecast catalogue for the successor's development evaluation.

An explicit field/calendar snapshot is separate from training and later targets.
This maps identities and state history; it does not fit, propagate latent states,
infer unknown race conditions, score a prediction or certify source availability.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from f1rank.design import AGE_DECLINE_FROM
from f1rank.lineage import REGULATION_RESETS
from f1rank.overall.fitting import arrays_digest

SCHEMA = 1


def _keys(value, required, optional=()):
    if not isinstance(value, dict) or set(value) - set(required) - set(optional) or set(required) - set(value):
        raise ValueError("Missing or unsupported forecast fields; outcome data are not features")


def _identity(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_]+", value):
        raise ValueError("Invalid driver, team or circuit identity")
    return value


def _event(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}", value) or value.endswith("-00"):
        raise ValueError("Invalid forecast event identity")
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer at least {minimum}")
    return value


def _time(value):
    if not isinstance(value, str):
        raise ValueError("Forecast times must be explicit timezone-aware strings")
    try:
        time = dt.datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Invalid forecast timestamp") from exc
    if time.tzinfo is None or time.utcoffset() is None:
        raise ValueError("Forecast times must include a timezone")
    return time.astimezone(dt.timezone.utc)


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_snapshot(snapshot, cutoff):
    """Validate declared information only; do not read a held-out result table."""
    _keys(snapshot, ("schema_version", "information_set", "origin_at", "source", "event",
                     "calendar_after_cutoff", "entrants"))
    if type(snapshot["schema_version"]) is not int or snapshot["schema_version"] != SCHEMA:
        raise ValueError("Unsupported forecast schema")
    information = snapshot["information_set"]
    if information not in ("prequalifying", "actual_grid"):
        raise ValueError("Unknown forecast information set")
    origin = _time(snapshot["origin_at"])
    source = snapshot["source"]
    _keys(source, ("kind", "captured_at", "reference"))
    if source["kind"] not in ("archived_asof", "retrospective_reconstruction"):
        raise ValueError("Unknown source-vintage category")
    captured = _time(source["captured_at"])
    if not isinstance(source["reference"], str) or not source["reference"].strip():
        raise ValueError("An explicit snapshot source reference is required")
    if source["kind"] == "archived_asof" and captured > origin:
        raise ValueError("An as-of snapshot cannot be captured after the forecast origin")
    event = snapshot["event"]
    _keys(event, ("event_id", "circuit_id", "qualifying_start", "race_start", "scheduled_laps"))
    event_id = _event(event["event_id"])
    if event_id <= _event(cutoff):
        raise ValueError("Training must end before the forecast event")
    _identity(event["circuit_id"])
    qualifying, race = _time(event["qualifying_start"]), _time(event["race_start"])
    if qualifying >= race or origin >= race or (information == "prequalifying" and origin > qualifying):
        raise ValueError("Forecast origin contradicts the declared session schedule")
    if race.year != int(event_id[:4]):
        raise ValueError("Event identity and calendar year disagree")
    _integer(event["scheduled_laps"], "Scheduled laps", 1)
    calendar = snapshot["calendar_after_cutoff"]
    if not isinstance(calendar, list) or not calendar or any(_event(e) <= cutoff for e in calendar):
        raise ValueError("A future event calendar is required")
    if calendar != sorted(set(calendar)) or calendar[-1] != event_id:
        raise ValueError("Future calendar must be unique, ordered and end at the target")
    # Across season boundaries, the source must establish the preceding season's
    # final round. Within one season an omitted round is detectable here.
    for before, after in zip([cutoff, *calendar[:-1]], calendar):
        if before[:4] == after[:4] and int(after[-2:]) != int(before[-2:]) + 1:
            raise ValueError("Future calendar omits an intervening round")
        if before[:4] != after[:4] and after[-2:] != "01":
            raise ValueError("A new forecast season must start at its first round")
    entrants = snapshot["entrants"]
    if not isinstance(entrants, list) or len(entrants) < 2:
        raise ValueError("Forecast needs the complete declared field")
    required = ("driver_id", "team", "prior_entries_after_cutoff")
    metadata = ("dob", "debut_season", "experience_before_cutoff")
    for row in entrants:
        _keys(row, (*required, "grid_position") if information == "actual_grid" else required, metadata)
        _identity(row["driver_id"])
        _identity(row["team"])
        count = _integer(row["prior_entries_after_cutoff"], "Intervening entries")
        if count >= len(calendar):
            raise ValueError("Intervening entries exceed the supplied calendar")
        if information == "actual_grid":
            position = _integer(row["grid_position"], "Grid position")
            if position > len(entrants):
                raise ValueError("Grid position exceeds the declared field")
    if len({r["driver_id"] for r in entrants}) != len(entrants):
        raise ValueError("Duplicate forecast driver")
    if information == "actual_grid":
        grid = [r["grid_position"] for r in entrants if r["grid_position"] > 0]
        if len(set(grid)) != len(grid):
            raise ValueError("Duplicate assigned grid position; pit-lane zeroes are allowed")
    return dict(snapshot, origin_at=origin.isoformat(), source=dict(source, captured_at=captured.isoformat()),
                event=dict(event, race_start=race.isoformat(), qualifying_start=qualifying.isoformat()),
                entrants=sorted(entrants, key=lambda x:x["driver_id"]))


def build_catalog(design, snapshot):
    """Map the full declared field to history, keeping missing states explicit.

    Last-entry indices are history references, not forecast ratings: the future
    predictor must reconstruct uncentred states, propagate uncertainty, then
    centre on the forecast field. It must never compare separately centred past
    event ratings directly or replace missing entries with zero ability.
    """
    cutoff = _event(design.catalog["cutoff"])
    if design.catalog["last_lap_event"] > cutoff or any(e > cutoff for e in design.catalog["outcome_events"]):
        raise ValueError("Lap/outcome catalogue reaches beyond training cutoff")
    return _build_catalog(design.qualifying, snapshot, cutoff,
        dict(qualifying=design.qualifying.arrays(), laps=design.laps, outcomes=design.outcomes),
        "current_race_only" if design.catalog.get("likelihood_arm") == "current_race_only" else "joint_current")


def build_qualifying_catalog(design, snapshot, *, cutoff):
    """Legacy qualifying-only history has its own arrays and explicit arm identity."""
    return _build_catalog(design, snapshot, _event(cutoff),
                          dict(qualifying=design.arrays()), "legacy_qualifying")


def _build_catalog(q, snapshot, cutoff, blocks, likelihood_arm):
    for frame in (q.events, q.entries):
        if frame.empty or frame.event_id.isna().any() or (frame.event_id > cutoff).any():
            raise ValueError("Training catalogue reaches beyond its declared cutoff")
    race_only = likelihood_arm == "current_race_only"
    if q.obs.empty:
        if not race_only:
            raise ValueError("Absent qualifying observations require the explicit race-only arm")
    elif race_only or q.obs.event_id.isna().any() or (q.obs.event_id > cutoff).any():
        raise ValueError("Qualifying observations contradict the arm or declared cutoff")
    if race_only and np.any(q.circuit_factor):
        raise ValueError("Race-only forecast cannot reuse a qualifying-derived circuit factor")
    if q.entries.duplicated(["event_id", "driver_id"]).any() or q.cars.duplicated(["event_idx", "team"]).any():
        raise ValueError("Training state identities are not unique")
    request = validate_snapshot(snapshot, cutoff)
    event = request["event"]
    race_time, origin = _time(event["race_start"]), _time(request["origin_at"])
    if pd.to_datetime(q.events.date, utc=True).max() >= origin:
        raise ValueError("Training event is not strictly before the forecast origin")
    season = int(event["event_id"][:4])
    era = int(np.searchsorted(sorted(REGULATION_RESETS), season, side="right"))
    future_index = len(q.events) + len(request["calendar_after_cutoff"]) - 1
    known_drivers = q.drivers.set_index("driver_id")
    rows = []
    for row in request["entrants"]:
        driver, team = row["driver_id"], row["team"]
        history = q.entries[q.entries.driver_id == driver].sort_values("event_idx")
        latest = history.iloc[-1] if not history.empty else None
        if latest is None:
            if any(k not in row for k in ("dob", "debut_season", "experience_before_cutoff")):
                raise ValueError("Unseen drivers require announced metadata and experience; no zero fills")
            dob, debut = row["dob"], _integer(row["debut_season"], "Debut season", 1)
            experience = _integer(row["experience_before_cutoff"], "Prior experience")
        else:
            metadata = known_drivers.loc[driver]
            dob, debut = str(metadata.dob), int(metadata.debut_season)
            experience = int(latest.experience) + 1
            expected = dict(dob=dob, debut_season=debut, experience_before_cutoff=experience)
            if any(key in row and row[key] != value for key, value in expected.items()):
                raise ValueError("Announced driver metadata conflicts with the training catalogue")
        try:
            birthday = dt.date.fromisoformat(dob)
        except (TypeError, ValueError) as exc:
            raise ValueError("Driver date of birth is missing or invalid") from exc
        if birthday >= race_time.date() or debut > season:
            raise ValueError("Driver metadata is inconsistent with the forecast event")
        age = (race_time.date() - birthday).days / 365.25
        stint = history[(history.team == team) & (history.era == era)]
        cars = q.cars[q.cars.team == team].sort_values("event_idx")
        last_car = cars.iloc[-1] if not cars.empty else None
        item = dict(driver_id=driver, team=team,
            driver_index=None if latest is None else int(latest.driver_idx),
            latest_entry_index=None if latest is None else int(latest.entry_idx),
            latest_entry_event=None if latest is None else str(latest.event_id),
            event_gap=None if latest is None else future_index - int(latest.event_idx),
            season_gap=None if latest is None else season - int(latest.season),
            same_season=False if latest is None else season == int(latest.season),
            era_stint_index=None if stint.empty else int(stint.iloc[-1].era_stint_idx),
            latest_car_index=None if last_car is None else int(last_car.car_idx),
            latest_car_season=None if last_car is None else int(last_car.season),
            car_season_start=True if last_car is None else season != int(last_car.season),
            car_reset=bool(last_car is not None and season != int(last_car.season) and season in REGULATION_RESETS),
            car_key=team, team_era_key=[driver, team, era],
            experience=experience + row["prior_entries_after_cutoff"],
            age_over=max(0., age - AGE_DECLINE_FROM),
            new_driver=latest is None, new_team_era=stint.empty, new_car=last_car is None)
        if request["information_set"] == "actual_grid":
            # Match the existing observed-grid feature convention without ever
            # interpreting an unknown grid as a pit-lane start.
            item.update(grid_position=row["grid_position"],
                        grid_feature_slot=row["grid_position"] or len(request["entrants"]))
        rows.append(item)
    circuits = q.events.loc[q.events.circuit_id == event["circuit_id"], "circuit_idx"].unique()
    result = dict(schema_version=SCHEMA, purpose="forecast_catalogue_only",
        training_cutoff=cutoff, training_arrays_key=arrays_digest(blocks),
        information_set=request["information_set"], origin_at=request["origin_at"],
        event_id=event["event_id"], event_index=future_index, era=era,
        likelihood_arm=likelihood_arm,
        calendar_after_cutoff=request["calendar_after_cutoff"],
        circuit_id=event["circuit_id"], circuit_index=int(circuits[0]) if len(circuits) else None,
        new_circuit=not len(circuits), scheduled_laps=event["scheduled_laps"],
        field=[r["driver_id"] for r in rows], entries=rows,
        snapshot_sha256=_hash(request), source_claim=request["source"],
        source_availability_verified=False, independent_evidence_established=False,
        future_weather="unknown", future_neutralisation="unknown",
        latent_states_propagated=False, prediction_available=False, publication_allowed=False,
        builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    result["catalogue_sha256"] = _hash(result)
    return result
