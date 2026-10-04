"""Publish an atomic, versioned dashboard dataset, or serve the built dashboard.

publish reads verified exports only; refresh fetches, fits and publishes together.
refresh --races also extracts race timing and rebuilds the race timeline.
A failure preserves the last good release and records the failure in status.json.
The frontend polls this small status file and pointer, never mixed CSV generations.
"""
import argparse
import contextlib
import datetime as dt
import functools
import hashlib
import http.server
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .artifacts import StaleArtifact, atomic_json, digest, require
from .race_publication import load as load_race_pace

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "dashboard" / "public" / "data"
RATINGS = ROOT / "outputs" / "ratings"
SCHEMA_VERSION = 1
QUANTILES = ("q05", "q25", "median", "q75", "q95")


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def read_json(path):
    return json.loads(path.read_text())


def estimate(row, prefix=""):
    result = {q: round(float(row[prefix + q + "_s"]), 6) for q in QUANTILES}
    for key in ("rank_lo", "rank_hi", "rank_median", "p_fastest", "p_top3"):
        if prefix + key in row:
            result[key] = round(float(row[prefix + key]), 6)
    return result


def comparisons(ids, draws, probabilities=None):
    """Joint differences preserve posterior covariance; never subtract marginal intervals."""
    if draws.ndim != 2 or draws.shape[1] != len(ids) or not np.isfinite(draws).all():
        raise ValueError("Invalid joint comparison draws")
    result = {}
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            diff = draws[:, i] - draws[:, j]
            lo, med, hi = np.quantile(diff, [.05, .5, .95])
            p = float(probabilities.loc[a, b]) if probabilities is not None else float(
                (diff > 0).mean() + .5 * (diff == 0).mean())
            result[f"{a}|{b}"] = {"q05": round(float(lo), 6), "median": round(float(med), 6),
                                    "q95": round(float(hi), 6), "p_ahead": round(p, 6)}
    return result


def racing_health(overall=None):
    rows = []
    specs = [
        ("Qualifying-adjusted race pace & tyre management", "race/multi_heldout.json", "multi_heldout.manifest.json",
         ["gate_race_specific_pace", "gate_degradation"]),
        ("Starts", "firstlap/summary.json", "manifest.json", ["gate_driver_ranking"]),
        ("Consistency", "consistency/summary.json", "manifest.json", ["gate_driver_ranking"]),
        ("Reliability & errors", "reliability/summary.json", "manifest.json",
         ["gate_driver_error_ranking", "gate_team_reliability"]),
        ("Pit stop operations", "pitstops/summary.json", "manifest.json", ["gate_team_ops_rating"]),
        ("Wet pace", "wet/summary.json", "manifest.json", ["gate_wet_ranking"]),
        ("Overtaking & defending", "battles/summary.json", "manifest.json", ["gate_driver_ranking"]),
        ("Equal-car championship", "championship/summary.json", "manifest.json", []),
    ]
    for label, file, manifest, gates in specs:
        if file == "championship/summary.json":
            result = overall if overall is not None else overall_result()
            rows.append({"name": label, "source": f"outputs/{file}",
                         "status": {"established": "passed", "not established": "experimental"}.get(
                             result["status"], result["status"]), "reason": result["reason"]})
            continue
        path = ROOT / "outputs" / file
        row = {"name": label, "source": f"outputs/{file}", "status": "unavailable",
               "reason": "No published result yet."}
        try:
            require(path.parent, name=manifest, required_outputs=[path])
            summary = read_json(path)
            passed = all(summary.get(g) is True for g in gates) if gates else bool(
                summary.get("combined_validation", {}).get("gate"))
            row.update(status="passed" if passed else "experimental",
                       reason="Recorded validation passed. Ranking integration is pending." if passed
                       else "Validation does not yet support a standalone ranking.")
        except (StaleArtifact, ValueError, OSError) as exc:
            if path.exists():
                row.update(status="stale", reason="Data or code changed after these results were recorded; "
                                                  "they need regeneration before they count as current.")
            row["detail"] = str(exc).replace(str(ROOT) + "/", "")
        rows.append(row)
    return rows


def overall_result():
    """Publish only a current, validated equal-car scenario; never fall back to legacy results.

    No fitting or simulation happens here. The headline retains driver-team effects;
    portable standings are deliberately not mixed with its contribution breakdown.
    """
    directory = ROOT / "outputs" / "championship"
    paths = [directory / name for name in ("summary.json", "standings.csv", "contributions.csv")]
    empty = {"standings": [], "contributions": [], "evidence": None}
    try:
        manifest = require(directory, required_outputs=paths)
        marker = digest(directory / "manifest.json")
        summary = read_json(paths[0])
        combined = summary["combined_validation"]
        entered, selected = summary["qualities_entered"], summary["qualities_selected"]
        entry = summary["entry_tests"]
        race_stage = summary["heldout_race_stage"]
        if (not isinstance(combined, dict) or not isinstance(race_stage, dict)
                or not isinstance(entry, dict) or any(not isinstance(v, dict) for v in entry.values())
                or any(not isinstance(names, list) or any(not isinstance(n, str) or not n for n in names)
                       or len(set(names)) != len(names) for names in (entered, selected))):
            raise ValueError("Invalid championship validation summary")
        passed = combined.get("gate") is True
        if set(entered) != (set(selected) if passed else set()):
            raise ValueError("Entered qualities disagree with the combined validation gate")
        if not set(selected) <= entry.keys():
            raise ValueError("Selected qualities have no entry tests")
        excluded = {key: value.get("excluded_unconverged_qualifying_folds", [])
                    for key, value in (("combined_validation", combined), ("heldout_race_stage", race_stage))}
        if any(not isinstance(years, list) or any(type(y) is not int for y in years)
               for years in excluded.values()):
            raise ValueError("Invalid excluded test seasons")
        decisions = []
        for name, test in entry.items():
            if name in entered:
                reason = "Passed the conditional entry test and the combined held-out validation."
            elif test.get("not_run"):
                reason = f"Not tested: {test['not_run']}"
            elif name in selected:
                reason = "Selected by the conditional entry test, but the combined held-out validation did not pass."
            else:
                reason = "Did not establish an improvement when tested alongside the other candidate qualities."
            decisions.append({"quality": name, "entered": name in entered,
                              "tested": not bool(test.get("not_run")), "reason": reason})
        evidence = {"recorded_at": manifest["generated_at"], "fit_id": manifest["fit_id"],
                    "data_as_of": manifest.get("data_as_of"),
                    "qualities_entered": entered, "qualities_selected": selected,
                    "qualities_not_entered": [d["quality"] for d in decisions if d["tested"] and not d["entered"]],
                    "quality_decisions": decisions, "entry_tests": entry,
                    "excluded_test_seasons": excluded, "combined_validation": combined,
                    "heldout_race_stage": race_stage,
                    "simulated_seasons": summary["simulated_seasons"],
                    "n_races_simulated": summary["n_races_simulated"],
                    "driver_error_rates_used": summary["driver_error_rates_used"]}
        if (any(type(evidence[k]) is not int or evidence[k] <= 0
                for k in ("simulated_seasons", "n_races_simulated"))
                or type(evidence["driver_error_rates_used"]) is not bool):
            raise ValueError("Invalid championship simulation settings")
        standings, contributions = [], []
        if passed:
            table = pd.read_csv(paths[1], dtype={"driver_id": str, "name": str})
            columns = ["driver_id", "name", "points_per_race", "p_title", "rank_median", "rank_lo", "rank_hi"]
            table = table.loc[table.version == "in_team", columns].copy()
            numeric = columns[2:]
            table[numeric] = table[numeric].apply(pd.to_numeric, errors="raise")
            if (table.empty or table.driver_id.duplicated().any() or table[columns[:2]].isna().any().any()
                    or not np.isfinite(table[numeric].to_numpy()).all()
                    or not table.p_title.between(0, 1).all() or (table.points_per_race < 0).any()
                    or (table.rank_lo < 1).any() or (table.rank_hi > len(table)).any()
                    or (table.rank_lo > table.rank_median).any() or (table.rank_median > table.rank_hi).any()
                    or ((table[["rank_lo", "rank_hi"]] % 1) != 0).any().any()):
                raise ValueError("Invalid headline championship standings")
            table = table.sort_values("points_per_race", ascending=False)
            components = ["qualifying_pace", *entered, "all"]
            losses = [f"loss_{name}_at_average" for name in components]
            parts = pd.read_csv(paths[2], dtype={"driver_id": str, "name": str})[
                ["driver_id", "name", "points_per_race", *losses]].copy()
            parts[["points_per_race", *losses]] = parts[["points_per_race", *losses]].apply(pd.to_numeric, errors="raise")
            if (parts.driver_id.duplicated().any() or parts[["driver_id", "name"]].isna().any().any()
                    or set(parts.driver_id) != set(table.driver_id)
                    or not np.isfinite(parts[["points_per_race", *losses]].to_numpy()).all()):
                raise ValueError("Invalid championship contribution breakdown")
            standings = table.to_dict("records")
            parts = parts.set_index("driver_id").loc[table.driver_id].reset_index()
            contributions = [{"driver_id": r["driver_id"], "name": r["name"],
                              "points_per_race": r["points_per_race"],
                              "losses": {name: r[column] for name, column in zip(components, losses)}}
                             for r in parts.to_dict("records")]
        # Also recheck failed-gate evidence. A regeneration can replace any file while
        # it is being read, and its manifest is written last.
        checked = require(directory, required_outputs=paths)
        if checked != manifest or digest(directory / "manifest.json") != marker:
            raise StaleArtifact("Championship outputs changed during publication; retry")
        result = {"status": "established" if passed else "not established",
                  "reason": "Combined held-out validation passed; this equal-car scenario remains experimental."
                  if passed else "The combined held-out validation has not established an overall ranking.",
                  "standings": standings, "contributions": contributions, "evidence": evidence}
        json.dumps(result, allow_nan=False)
        return result
    except StaleArtifact as exc:
        present = any(path.exists() for path in paths)
        return {**empty, "status": "stale" if present else "unavailable",
                "reason": "Championship provenance is missing or stale. Regeneration is required before a ranking can be shown."
                if present else "No verified championship results are available yet.",
                "detail": str(exc).replace(str(ROOT) + "/", "")}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {**empty, "status": "unavailable",
                "reason": "The championship output is incomplete or invalid; no ranking can be shown.",
                "detail": str(exc).replace(str(ROOT) + "/", "")}


def breakdown(ids, teams, in_team, car, lineage, medians=None):
    """Each driver's expected qualifying pace at an average circuit as car + driver.

    Both parts are centred on the event's field, so the total is relative to an
    average driver in an average car. The interval uses the joint draws. `medians`
    ({"drivers": {id: s}, "cars": {team: s}}) supplies the parts' published full-
    posterior medians, so they match the ranking tables rather than the thinned draws.
    """
    column = {t: j for j, t in enumerate(teams)}
    medians = medians or {"drivers": {}, "cars": {}}
    rows = []
    for i, driver in enumerate(ids):
        team = lineage[driver]
        total = in_team[:, i] + car[:, column[team]]
        lo, med, hi = np.quantile(total, [.05, .5, .95])
        part_car = medians["cars"].get(team, np.median(car[:, column[team]]))
        part_driver = medians["drivers"].get(driver, np.median(in_team[:, i]))
        rows.append({"id": driver, "team": team, "car": round(float(part_car), 6),
                     "driver": round(float(part_driver), 6),
                     "total": {"q05": round(float(lo), 6), "median": round(float(med), 6),
                               "q95": round(float(hi), 6)}})
    return sorted(rows, key=lambda r: -r["total"]["median"])


def car_state_result():
    """The pre-registered within-season car test (docs/race_car_state.md), as recorded.

    A one-off research record: its outputs must match their manifest, but later data
    do not rerun it, so it is shown with its own date rather than as current evidence.
    """
    directory = ROOT / "outputs" / "race_car_state"
    if not (directory / "validation.json").exists():
        return None
    manifest = read_json(directory / "validation.manifest.json")
    for file, expected in manifest["outputs"].items():
        if digest(ROOT / file) != expected:
            raise ValueError(f"Changed recorded output {file}")
    summary = read_json(directory / "validation.json")
    return {"model": summary["model"], "recorded_at": manifest["generated_at"], "fit_id": manifest["fit_id"],
            "folds": summary["folds"], "cars": summary["cars"],
            "document": "docs/race_car_state.md"}


def race_feature_results():
    """The pre-registered weekend-information tests (docs/race_features.md), as recorded.

    One-off research records like the car-state test: verified against their manifests
    and shown with their own dates. A feature without a recorded result is left out.
    """
    out = []
    for directory in sorted((ROOT / "outputs" / "race_features").glob("*/")):
        if not (directory / "validation.json").exists():
            continue
        manifest = read_json(directory / "validation.manifest.json")
        for file, expected in manifest["outputs"].items():
            if digest(ROOT / file) != expected:
                raise ValueError(f"Changed recorded output {file}")
        summary = read_json(directory / "validation.json")
        out.append({"feature": summary["feature"], "recorded_at": manifest["generated_at"],
                    "metrics": summary["metrics"], "document": "docs/race_features.md"})
    return out


def asof_payload():
    """Ratings after each race and scored next-race forecasts (outputs/asof), if verified."""
    directory = ROOT / "outputs" / "asof"
    if not (directory / "summary.json").exists():
        return None
    manifest = require(directory, required_outputs=[directory / "summary.json"])
    summary = read_json(directory / "summary.json")
    series = {"drivers": {}, "cars": {}}
    latest = None
    for row in summary["events"]:
        if f"outputs/asof/{row['file']}" not in manifest["outputs"]:
            raise ValueError(f"Unrecorded as-of record {row['file']}")
        rec = read_json(directory / row["file"])
        as_of = rec["trained_through"]["event_id"]
        for d in rec["ratings"]["drivers"]:
            series["drivers"].setdefault(d["id"], []).append(
                {"event": as_of, "team": d["team"], "headline": d["headline"], "portable": d["portable"]})
        for c in rec["ratings"]["cars"]:
            series["cars"].setdefault(c["id"], []).append({"event": as_of, "team": c["name"], "pace": c["pace"]})
        if latest is None or rec["event"]["event_id"] > latest["event"]["event_id"]:
            latest = rec
    return {"pooled": summary["pooled"], "events": summary["events"], "series": series,
            "latest": None if latest is None else {"event": latest["event"],
                                                  "trained_through": latest["trained_through"],
                                                  "pairs": latest["forecast"]["pairs"],
                                                  "order": latest["forecast"]["order"]}}


def forecast_payload():
    """The forecast for the next event, published before it, and the scores of earlier
    ones (outputs/forecasts), if verified."""
    directory = ROOT / "outputs" / "forecasts"
    if not (directory / "scores.json").exists():
        return None
    manifest = require(directory, required_outputs=[directory / "scores.json"])
    files = sorted(p for p in directory.glob("*_*.json"))
    for path in files:
        if f"outputs/forecasts/{path.name}" not in manifest["outputs"]:
            raise ValueError(f"Unrecorded forecast {path.name}")
    scores = read_json(directory / "scores.json")
    scored = {s["file"] for s in scores["events"]}
    upcoming = [read_json(p) | {"file": p.name} for p in files if p.name not in scored]
    return {"next": max(upcoming, key=lambda r: (r["event"]["event_id"], r["created_utc"]), default=None),
            "scores": scores}


def build_payload():
    manifest = require(RATINGS, required_outputs=[RATINGS / name for name in (
        "meta.json", "current_drivers.csv", "current_cars.csv", "driver_series.parquet",
        "car_series.parquet", "current_draws.npz", "pairwise_drivers_in_team.csv",
        "pairwise_drivers_portable.csv", "fit_metadata.json")])
    marker = digest(RATINGS / "manifest.json")
    meta = read_json(RATINGS / "meta.json")
    fit_meta = read_json(RATINGS / "fit_metadata.json")
    if (fit_meta.get("created_utc") != meta["fit_id"] or
            fit_meta.get("fingerprint") != meta["fit_fingerprint"]):
        raise ValueError("Export and portable fit metadata do not match")
    if not meta.get("diagnostics", {}).get("converged") or meta.get("pairwise_default") != "in_team":
        raise ValueError("Export is not approved for headline publication")
    d = pd.read_csv(RATINGS / "current_drivers.csv")
    c = pd.read_csv(RATINGS / "current_cars.csv")
    ds = pd.read_parquet(RATINGS / "driver_series.parquet")
    cs = pd.read_parquet(RATINGS / "car_series.parquet")
    events = pd.read_parquet(ROOT / "data" / "processed" / "events.parquet")
    events = events[events.event_id.isin(ds.event_id)].sort_values(["season", "round"])
    if events.empty or events.iloc[-1].event_id != meta["data_as_of"]["event_id"]:
        raise ValueError("Event coverage does not match the export")
    latest = ds[ds.event_id == meta["data_as_of"]["event_id"]].set_index("driver_id")
    drivers = [{"id": r.driver_id, "name": r["name"], "code": str(latest.loc[r.driver_id, "code"]),
                "team": r.team, "lineage": str(latest.loc[r.driver_id, "team"]),
                "has_time": bool(r.has_time), "headline": estimate(r, "in_team_"),
                "portable": estimate(r), "team_effect": estimate(r, "team_effect_"),
                "evidence": {k: int(r[k]) for k in ("events", "seasons", "teammates", "teams")}}
               for _, r in d.iterrows()]
    cars = [{"id": r.team, "name": r.constructor, "pace": estimate(r),
             "gap_to_best": estimate(r, "gap_")} for _, r in c.iterrows()]
    with np.load(RATINGS / "current_draws.npz", allow_pickle=False) as z:
        ids, teams = z["driver_ids"].tolist(), z["teams"].tolist()
        if set(ids) != {r["id"] for r in drivers} or set(teams) != {r["id"] for r in cars}:
            raise ValueError("Draw identifiers do not match the leaderboard")
        pairs = {m: comparisons(ids, z[field], pd.read_csv(RATINGS / filename, index_col=0))
                 for m, field, filename in (
                     ("headline", "in_team_s", "pairwise_drivers_in_team.csv"),
                     ("portable", "portable_skill_s", "pairwise_drivers_portable.csv"))}
        pairs["cars"] = comparisons(teams, z["car_s"])
        comparison_draws = len(z["car_s"])
        split = breakdown(ids, teams, z["in_team_s"], z["car_s"],
                          {i: str(latest.loc[i, "team"]) for i in ids},
                          {"drivers": {r["id"]: r["headline"]["median"] for r in drivers},
                           "cars": {r["id"]: r["pace"]["median"] for r in cars}})
    history = {"drivers": {}, "cars": {}}
    for id_, rows in ds.groupby("driver_id"):
        history["drivers"][id_] = [{"event": r.event_id, "team": r.constructor_name,
                                     "headline": estimate(r, "in_team_"), "portable": estimate(r)}
                                    for _, r in rows.sort_values("event_idx").iterrows()]
    for id_, rows in cs.groupby("team"):
        history["cars"][id_] = [{"event": r.event_id, "team": r.constructor_name, "pace": estimate(r),
                                  "at_circuit": estimate(r, "at_circuit_")}
                                 for _, r in rows.sort_values("event_idx").iterrows()]
    validations = {}
    for name in ("gates", "lfo_summary"):
        path = ROOT / "outputs" / "validation" / f"{name}.json"
        if path.exists():
            # Git checkout mtimes are not research-run timestamps.
            validations[name] = {"data": read_json(path), "recorded_at": None,
                                 "sha256": digest(path)}
    snapshots = []
    for path in sorted((ROOT / "outputs" / "snapshots").glob("*.json")):
        snap = read_json(path)
        sm = snap.get("meta", {})
        if sm.get("export_version") != "2":
            continue  # older schemas used a different comparison metric
        snapshots.append({"file": path.name, "fit_id": sm.get("fit_id"),
                          "generated_at": sm.get("generated_at"), "event": sm.get("data_as_of"),
                          "drivers": snap["drivers"], "cars": snap["cars"]})
    race_snapshots = [read_json(path) for path in sorted((ROOT / "outputs" / "snapshots" / "race").glob("*.json"))]
    require(RATINGS)  # reject a concurrent export or source edit while assembling
    if digest(RATINGS / "manifest.json") != marker:
        raise ValueError("Export changed during publication; retry")
    overall = overall_result()
    return {"schema_version": SCHEMA_VERSION, "meta": meta, "drivers": drivers, "cars": cars,
            "catalog": {"drivers": ds[["driver_id", "name"]].drop_duplicates("driver_id").rename(
                columns={"driver_id": "id"}).sort_values("name").to_dict("records"),
                        "cars": cs.sort_values("event_idx").drop_duplicates("team", keep="last")[[
                            "team", "constructor_name"]].rename(columns={"team": "id", "constructor_name": "name"})
                            .sort_values("name").to_dict("records")},
            "events": events.to_dict("records"), "history": history, "comparisons": pairs,
            "comparison_draws": comparison_draws, "racing": racing_health(overall), "overall": overall,
            "race_pace": load_race_pace(ROOT), "breakdown": split, "asof": asof_payload(),
            "forecast": forecast_payload(),
            "car_state": car_state_result(), "race_features": race_feature_results(),
            "refresh": read_json(ROOT / "outputs/refresh/latest.json")
            if (ROOT / "outputs/refresh/latest.json").exists() else None,
            "validation": validations, "snapshots": snapshots, "race_snapshots": race_snapshots,
            "provenance": {"export_manifest": marker, "inputs": manifest["inputs"],
                           "outputs": manifest["outputs"]}}


def publish(data_dir=DATA):
    """Write immutable content first; switch the current pointer only after success."""
    payload = build_payload()
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    release = hashlib.sha256(encoded).hexdigest()[:20]
    target = data_dir / "releases" / f"{release}.json"
    if target.exists() and read_json(target) != payload:
        raise ValueError("Existing immutable release is corrupt; publication stopped")
    if not target.exists():
        atomic_json(target, payload)
    pointer = {"schema_version": SCHEMA_VERSION, "release": release,
               "url": f"releases/{release}.json", "published_at": now(),
               "data_as_of": payload["meta"]["data_as_of"]}
    atomic_json(data_dir / "latest.json", pointer)
    return pointer


@contextlib.contextmanager
def publication_lock(data_dir):
    """Process-owned lock; a crash releases it without leaving a stale lockout."""
    import fcntl
    data_dir.mkdir(parents=True, exist_ok=True)
    with (data_dir / ".refresh.lock").open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another dashboard refresh is running") from exc
        yield


# Sampler settings of the published main fit (outputs/ratings/fit_metadata.json). A
# refresh refits with the same settings, so an update never quietly publishes fewer draws.
FIT_SETTINGS = {"warmup": 1500, "samples": 1500, "chains": 4}


def refresh_steps(with_races=False, race_python=None, extract_python=None):
    """Keep both exports current when their shared entry/event inputs change."""
    def module(name, *args, python=sys.executable):
        return name, [python, "-m", f"f1rank.{name}", *args]
    steps = [module("fetch"), module("build")]
    if with_races:
        extract = [extract_python or sys.executable, str(ROOT / "extract/race_extract.py")]
        steps += [("race timing", extract), ("sprint timing", extract + ["--sprint"]),
                  module("racedata"), module("timeline")]
    fit_args = [f"--{k}={v}" for k, v in FIT_SETTINGS.items()]
    steps += [module("fit", *fit_args), module("export"),
              module("forecast", "next"), module("forecast", "score"),
              module("asof", "fit", "--latest"), module("asof", "summary")]
    if with_races or (ROOT / "outputs/race_total/pace.json").exists():
        steps += [module("race_total", "--validate", "--export", python=race_python or sys.executable)]
    return steps


def record_refresh(started_at, stages, with_races):
    """Committed record of the last completed fitting run, shown on the dashboard."""
    path = ROOT / "data" / "processed" / "events.parquet"
    latest = pd.read_parquet(path).sort_values("event_id").iloc[-1] if path.exists() else None
    run_url = None
    if os.environ.get("GITHUB_RUN_ID"):
        run_url = (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/"
                   f"{os.environ['GITHUB_REPOSITORY']}/actions/runs/{os.environ['GITHUB_RUN_ID']}")
    atomic_json(ROOT / "outputs" / "refresh" / "latest.json", {
        "started_at": started_at, "finished_at": now(), "races": with_races,
        "event": None if latest is None else {k: str(latest[k]) for k in ("event_id", "race_name", "date")},
        "trigger": os.environ.get("GITHUB_EVENT_NAME", "manual"), "run_url": run_url, "stages": stages})


def run(refresh=False, data_dir=DATA, *, with_races=False, race_python=None, extract_python=None):
    if with_races and not refresh:
        raise ValueError("Race extraction requires refresh")
    with publication_lock(data_dir):
        start = time.monotonic()
        status = {"state": "running", "started_at": now(), "stage": "checking exports"}
        atomic_json(data_dir / "status.json", status)
        log_dir = ROOT / "outputs" / "dashboard"
        log_dir.mkdir(parents=True, exist_ok=True)
        try:
            if refresh:
                stages = []
                with (log_dir / "refresh.log").open("a") as log:
                    for stage, command in refresh_steps(with_races, race_python, extract_python):
                        status.update(stage=stage, updated_at=now())
                        atomic_json(data_dir / "status.json", status)
                        log.write(f"\n{now()} {stage}\n"); log.flush()
                        began = time.monotonic()
                        subprocess.run(command, cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT, check=True)
                        stages.append({"stage": stage, "seconds": round(time.monotonic() - began, 1)})
                record_refresh(status["started_at"], stages, with_races)
            status.update(stage="publishing", updated_at=now())
            atomic_json(data_dir / "status.json", status)
            pointer = publish(data_dir)
            status.update(state="ok", stage="published", release=pointer["release"])
        except BaseException as exc:
            status.update(state="failed", error=str(exc).replace(str(ROOT) + "/", ""))
            raise
        finally:
            status.update(finished_at=now(), duration_s=round(time.monotonic() - start, 2))
            atomic_json(data_dir / "status.json", status)
            with (log_dir / "runs.jsonl").open("a") as log:
                log.write(json.dumps(status) + "\n")
    return pointer


def current_status(data_dir=DATA):
    """Report a terminated refresh accurately even if it could not write its final status."""
    status = read_json(data_dir / "status.json")
    if status.get("state") == "running":
        try:
            with publication_lock(data_dir):
                status = {**status, "state": "failed", "error": "The refresh process stopped before publishing."}
        except RuntimeError:
            pass  # the process still owns its lock
    return status


def serve(port, data_dir=DATA):
    dist = ROOT / "dashboard" / "dist"
    if not (dist / "index.html").exists():
        raise SystemExit("Build the dashboard first: cd dashboard && npm run build")
    class Handler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.split("?", 1)[0] == "/data/status.json":
                try:
                    body = json.dumps(current_status(data_dir)).encode()
                except (OSError, ValueError):
                    self.send_error(404, "No refresh status available")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def translate_path(self, path):
            # Serve only the dashboard and published datasets, never the repository.
            clean = path.split("?", 1)[0]
            if clean.startswith("/data/"):
                candidate = (data_dir / clean.removeprefix("/data/")).resolve()
                if not candidate.is_relative_to(data_dir.resolve()) or candidate.name.startswith("."):
                    return str(dist / "__not_found__")
                return str(candidate)
            return super().translate_path(path)
        def end_headers(self):
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            super().end_headers()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(Handler, directory=dist))
    print(f"Dashboard: http://localhost:{port}", flush=True)
    server.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["publish", "refresh", "serve"], nargs="?", default="publish")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--races", action="store_true", help="Refresh race timing, timeline and rankings too")
    parser.add_argument("--race-python", help="Python environment for race fitting (for example .venv-gpu/bin/python)")
    parser.add_argument("--extract-python", help="Python environment with FastF1 installed")
    args = parser.parse_args()
    if args.command != "refresh" and (args.races or args.race_python or args.extract_python):
        parser.error("Race refresh options require the refresh command")
    if args.command == "serve":
        serve(args.port)
    else:
        print(json.dumps(run(refresh=args.command == "refresh", with_races=args.races,
                             race_python=args.race_python, extract_python=args.extract_python), indent=2))


if __name__ == "__main__":
    main()
