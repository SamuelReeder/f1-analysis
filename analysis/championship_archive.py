"""Separately registered validation of fixed race pace on the earlier racing archive.

python -m analysis.championship_archive register
python -m analysis.championship_archive run  # memory-capped systemd service only
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis import championship_followup as prospective
from f1rank import artifacts, championship as c, conditions, design, fit, jobs, oldlaps, racepace, racemulti, timeline

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/championship_archive"
CACHE = ROOT / "outputs/fits/championship_archive"
PROTOCOL = ROOT / "docs/championship_archive_protocol.json"
AUDIT = ROOT / "outputs/analysis/championship_archive_audit.json"
PLAN = ROOT / "docs/championship_archive.md"


class ExhaustedFit(RuntimeError):
    """A statistical attempt exhausted the frozen convergence ladder."""


def hashes(paths):
    return {str(p.relative_to(ROOT)): artifacts.digest(p) for p in paths}


def sources():
    return {**prospective.source_hashes(), **hashes([Path(__file__), PLAN])}


def register():
    audit = json.loads(AUDIT.read_text())
    if (audit["missing_lap_events"] or audit["missing_qualifying_events"]
            or not audit["all_recorded_benchmarks_exclude_archive"]
            or any(not r["uses_benchmark_race_orders"] or r["direct_race_parquet_read"]
                   for r in audit["championship_history"])):
        raise ValueError("The archive has not passed the registered availability/prior-use audit")
    years = sorted({int(e[:4]) for e in audit["archive_event_ids"]})
    if len(years) - 1 < c.MIN_SELECTION_SEASONS:
        raise ValueError("Insufficient earlier seasons for a training prefix and the unchanged gate")
    inputs = [*artifacts.input_files(), ROOT / audit["dump"], AUDIT,
              *(ROOT / f"data/raw/jolpica/{s}_results.json" for s in years)]
    protocol = {
        "registered_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "registration_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate": list(prospective.CANDIDATE), "first_training_season": years[0],
        "first_test_season": years[1], "last_test_season": years[-1],
        "min_converged_test_seasons": c.MIN_SELECTION_SEASONS,
        "decision_after_date": f"{years[-1]}-12-31",
        "decision_timing": "Historical archive: one analysis after every registered fold, not a prospective claim",
        "test_event_ids": [e for e in audit["archive_event_ids"] if int(e[:4]) > years[0]],
        "primary_baseline": "grid_ratings", "secondary_baselines": ["grid", "ratings"],
        "selection": "Fixed race-specific pace; no subset selection or model tuning",
        "gate": "Unchanged paired_seasons lower 95% bound above zero; all registered archive folds required",
        "qualifying_attempts": list(jobs.ATTEMPTS),
        "race_pace_sampling": "Unchanged racemulti.fit defaults, including its original retry ladder",
        "race_stage_sampling": "Unchanged championship.fit_race defaults and artifacts.RETRY",
        "failure_policy": "An exhausted fit fails this study; preserve diagnostics and do not retune or remove its season",
        "publication": "Separate archive evidence; never overwrite the completed championship or prospective registration",
        "source_hashes": sources(), "input_hashes": hashes(inputs),
    }
    with PROTOCOL.open("x") as f:
        json.dump(protocol, f, indent=2)
        f.write("\n")
    print("Archive study registered; commit its protocol before running", flush=True)


def check_registration():
    relative = str(PROTOCOL.relative_to(ROOT))
    revisions = subprocess.check_output(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", relative], cwd=ROOT, text=True).splitlines()
    if not revisions or subprocess.check_output(
            ["git", "show", f"{revisions[-1]}:{relative}"], cwd=ROOT) != PROTOCOL.read_bytes():
        raise ValueError("Archive registration must be committed and immutable")
    p = json.loads(PROTOCOL.read_text())
    if p["source_hashes"] != sources() or any(artifacts.digest(ROOT / k) != v for k, v in p["input_hashes"].items()):
        raise ValueError("Registered archive sources or inputs changed")
    return p


def admit_race(C):
    """Exactly the structural admission used by stage_a's nonempty pair table.

    Stage-A estimates/bootstrap values are not inputs to racemulti's lap likelihood.
    Its pair table admits a race iff enough drivers survived cleaning and at least
    one team has exactly two such drivers. Computing unused estimates is unnecessary.
    """
    return C.driver_id.nunique() >= 10 and (C.groupby("team").driver_id.nunique() == 2).any()


def prepare(protocol):
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / "clean_laps.parquet"
    receipt = CACHE / "prepared.json"
    if cache.exists() and receipt.exists():
        meta = json.loads(receipt.read_text())
        expected = artifacts.digest(PROTOCOL)
        if meta["protocol_sha256"] == expected and meta["clean_laps_sha256"] == artifacts.digest(cache):
            if all(artifacts.digest(ROOT / p) == h for p, h in meta["weather_hashes"].items()):
                return pd.read_parquet(cache), meta
        raise ValueError("Prepared archive data changed; refuse an unregistered replacement")
    first, last = protocol["first_training_season"], protocol["last_test_season"]
    # The verified dump parser is read-only. Restrict to the registered archive
    # immediately; do not use any later race's observations or fitted parameters.
    archive = ROOT / json.loads(AUDIT.read_text())["dump"]
    frozen_dump = CACHE / "dump"
    frozen_dump.mkdir(parents=True, exist_ok=True)
    link = frozen_dump / archive.name
    if not link.exists():
        link.symlink_to(archive)
    if link.resolve() != archive.resolve() or list(frozen_dump.glob("*.zip")) != [link]:
        raise ValueError("Archive cache does not identify exactly the registered dump")
    with patch.object(oldlaps, "DUMPS", frozen_dump):
        laps, stops = oldlaps.dump_tables()
    laps = laps[laps.event_id.str[:4].astype(int).between(first, last)].copy()
    stops = stops[stops.event_id.isin(laps.event_id)].copy()
    laps = laps.sort_values(["event_id", "driver_id", "lap_number"], ignore_index=True)
    laps["time"] = laps.groupby(["event_id", "driver_id"]).lap_time.cumsum()
    races = pd.read_parquet(ROOT / "data/processed/race.parquet")
    races = races[races.event_id.isin(laps.event_id)].copy()
    # The original pre-FastF1 timeline has no message/lap evidence. Its retirement
    # windows use completed laps and coded administrative outcomes, not the learned
    # cause probabilities (which clean_laps never reads).
    _, retirement_rows = timeline.retirements(races, {}, {})
    tl = retirement_rows.rename(columns={"lap": "lap_start"}).assign(kind="retirement")
    weather_rows = conditions.race_info()
    weather_rows = weather_rows[weather_rows.event_id.isin(laps.event_id)]
    weather = {}
    with patch.object(conditions, "RAW", CACHE / "weather"):
        for row in weather_rows.itertuples():
            value = conditions.precipitation(row)
            cached = json.loads((CACHE / "weather" / f"{row.event_id}.json").read_text())["hourly"]
            precipitation = np.asarray(cached["precipitation"], dtype=float)
            start = pd.Timestamp(f"{row.date}T{row.time.rstrip('Z')}", tz="UTC").floor("h")
            times = pd.to_datetime(cached["time"], utc=True)
            window = (times >= start) & (times <= start + pd.Timedelta(hours=conditions.WINDOW_H))
            if (window.sum() != conditions.WINDOW_H + 1 or not np.isfinite(precipitation[window]).all()
                    or not np.isfinite(value)):
                raise ValueError(f"Invalid precipitation for {row.event_id}")
            weather[row.event_id] = value
    if set(weather) != set(laps.event_id):
        raise ValueError("Weather metadata do not cover the archive")
    frames, admission = [], []
    for event_id, L in laps.groupby("event_id"):
        wet = weather[event_id] >= conditions.WET_MM
        row = {"event_id": event_id, "wet": bool(wet), "precip_mm": weather[event_id]}
        if wet:
            admission.append({**row, "included": False, "reason": "registered wet proxy"})
            continue
        team = races[races.event_id == event_id].set_index("driver_id").team
        framed = oldlaps.race_frame(L, stops[stops.event_id == event_id], team)
        clean = racepace.clean_laps(framed, tl[tl.event_id == event_id], racepace.OLD_COMPOUNDS)
        admitted = bool(admit_race(clean))
        admission.append({**row, "included": admitted, "clean_laps": len(clean),
                          "clean_drivers": int(clean.driver_id.nunique())})
        if admitted:
            frames.append(clean.assign(event_id=event_id, source="jolpica", season=int(event_id[:4])))
    if not frames:
        raise ValueError("No races survived the frozen archive input rules")
    clean = pd.concat(frames, ignore_index=True).sort_values(
        ["event_id", "stint_key", "lap_number"], ignore_index=True)
    clean.to_parquet(cache, index=False)
    meta = {"protocol_sha256": artifacts.digest(PROTOCOL), "clean_laps_sha256": artifacts.digest(cache),
            "admission": admission, "weather_hashes": hashes(sorted((CACHE / "weather").glob("*.json")))}
    artifacts.atomic_json(receipt, meta)
    return clean, meta


def qualifying(season, protocol):
    events = pd.read_parquet(ROOT / "data/processed/events.parquet", columns=["event_id", "season"])
    end = str(events.loc[events.season == season, "event_id"].max())
    d = design.build_design(protocol["first_training_season"], end_event=end)
    cut = int(d.events.loc[d.events.season < season, "event_idx"].max())
    d = d.with_cutoff(cut)
    from f1rank.qualifying import check_cutoff
    check_cutoff(d, season)
    path = CACHE / f"quali_{season}.npz"
    failed = path.with_suffix(".failed.json")
    if failed.exists():
        raise ExhaustedFit(f"Registered qualifying attempts already exhausted for {season}")
    if path.exists():
        fit.check(path, d)
        post, info = fit.load(path, d)
        attempts = fit.load_meta(path)["attempts"]
    else:
        try:
            with patch.object(jobs, "FITS", CACHE):
                post, info, settings, attempts = jobs.fit_with_retries(path.stem, d)
        except RuntimeError as exc:
            if failed.exists():
                raise ExhaustedFit(str(exc)) from exc
            raise
        fit.save(path, post, info, d, model_kw={}, settings=settings, attempts=attempts)
    artifacts.require_convergence(artifacts.diagnostics(post, info["divergences"]))
    flat = lambda key: post[key].reshape(-1, *post[key].shape[2:])
    driver = d.entries[["event_id", "driver_id"]].copy()
    driver["driver"] = np.median(flat("skill") + flat("compat"), axis=0)
    car = d.cars[["team", "event_idx"]].copy()
    car["event_id"] = d.events.event_id.to_numpy()[car.event_idx]
    car["car"] = np.median(flat("car") + flat("car_track"), axis=0)
    return driver, car.drop(columns="event_idx"), attempts


def race_frame(season, first, drivers, cars):
    race = pd.read_parquet(ROOT / "data/processed/race.parquet")
    race = race[race.event_id.str[:4].astype(int).between(first, season)]
    if not set(race.event_id) <= set(drivers.event_id):
        raise ValueError(f"Incomplete qualifying event coverage for archive fold {season}")
    frame = race.merge(drivers, on=["event_id", "driver_id"]).merge(cars, on=["event_id", "team"])
    count = frame.groupby("event_id").driver_id.transform("size")
    frame["grid_slot"] = np.where(frame.grid > 0, frame.grid, count)
    frame["season"] = frame.event_id.str[:4].astype(int)
    frame = frame.sort_values(["event_id", "position"]).reset_index(drop=True)
    frame["rank"] = frame.groupby("event_id").cumcount()
    return frame


def run():
    protocol = check_registration()
    if (OUT / "result.json").exists() or (OUT / "failure.json").exists():
        raise FileExistsError("The archive study has already reached a recorded outcome")
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    records, qualified, pace_fits, used_caches = [], {}, {}, {}
    try:
        clean, preparation = prepare(protocol)
        artifacts.atomic_json(OUT / "preparation.json", preparation)
        for season in range(protocol["first_test_season"], protocol["last_test_season"] + 1):
            print(f"== archive {season} start", flush=True)
            driver, car, attempts = qualifying(season, protocol)
            qualified[str(season)] = attempts
            used_caches.update(hashes([CACHE / f"quali_{season}.npz", CACHE / f"quali_{season}.meta.json"]))
            c.jax.clear_caches()
            train = clean[clean.season < season].merge(
                driver.rename(columns={"driver": "quali"}), on=["event_id", "driver_id"],
                how="left", validate="many_to_one").dropna(subset=["quali"])
            if train.empty or int(train.season.max()) >= season:
                raise ValueError(f"Invalid race-pace training cutoff for {season}")
            post, ids = racemulti.checkpointed(train, CACHE / f"pace_{season}.npz")
            pace_fits[str(season)] = {k: v for k, v in post.items() if k.startswith("_")}
            if not post["_converged"]:
                raise ExhaustedFit(f"Archive race pace failed convergence for {season}")
            used_caches.update(hashes([CACHE / f"pace_{season}.npz"]))
            c.jax.clear_caches()
            frame = race_frame(season, protocol["first_training_season"], driver, car)
            effects = {"race_specific_pace": {season: (np.asarray(ids), post["u"].astype(np.float32))}}
            try:
                with patch.object(c, "ENTRY_FIRST_SEASON", protocol["first_training_season"]), \
                        patch.object(c, "race_orders", lambda s: frame):
                    record = prospective.score_fold(season, protocol["test_event_ids"], effects)
            except RuntimeError as exc:
                if artifacts._FITS and not artifacts._FITS[-1][-1]["converged"]:
                    raise ExhaustedFit(str(exc)) from exc
                raise
            records.append(record)
            artifacts.atomic_json(OUT / f"fold_{season}.json", record)
            print(f"== archive {season} done", flush=True)
            c.jax.clear_caches()
        check_registration()
        if preparation["clean_laps_sha256"] != artifacts.digest(CACHE / "clean_laps.parquet"):
            raise ValueError("Prepared training laps changed during archive study")
        if any(artifacts.digest(ROOT / p) != h for p, h in {
                **used_caches, **preparation["weather_hashes"]}.items()):
            raise ValueError("An archive fit or weather cache changed during evaluation")
        decision = prospective.decide(records, protocol, dt.datetime.now(dt.timezone.utc).date())
        result = {**decision, "evidence": "separate historical archive validation; earlier-era transport test",
                  "protocol_sha256": artifacts.digest(PROTOCOL), "folds": records,
                  "qualifying_attempts": qualified, "race_pace_fits": pace_fits,
                  "used_cache_hashes": used_caches,
                  "race_stage_fit_attempts": artifacts.fit_record(),
                  "elapsed_seconds": time.monotonic() - started}
        artifacts.atomic_json(OUT / "result.json", result)
        outputs = [OUT / "result.json", OUT / "preparation.json", *sorted(OUT.glob("fold_*.json"))]
        inputs = [*(ROOT / p for p in protocol["input_hashes"]),
                  *(ROOT / p for p in protocol["source_hashes"]), PROTOCOL, PLAN, Path(__file__),
                  CACHE / "clean_laps.parquet", CACHE / "prepared.json",
                  *sorted(CACHE.glob("*.npz")), *sorted(CACHE.glob("*.meta.json")),
                  *sorted((CACHE / "weather").glob("*.json"))]
        artifacts.record(OUT, outputs, model="fixed-pace-archive-validation", inputs=inputs,
                         details={"protocol_sha256": artifacts.digest(PROTOCOL), "gate": result["gate"]})
        print("Archive study complete", flush=True)
    except Exception as exc:
        failure = "failure.json" if isinstance(exc, ExhaustedFit) else (
            "operational_failure_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        artifacts.atomic_json(OUT / failure, {
            "gate": False, "publication_authorized": False, "exception": str(exc),
            "kind": "exhausted_registered_fit" if isinstance(exc, ExhaustedFit) else "operational",
            "qualifying_attempts": qualified, "race_pace_fits": pace_fits,
            "qualifying_failures": {p.stem: json.loads(p.read_text()) for p in CACHE.glob("quali_*.failed.json")},
            "race_stage_fit_attempts": artifacts.fit_record(),
            "elapsed_seconds": time.monotonic() - started,
            "protocol_sha256": artifacts.digest(PROTOCOL)})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    action = parser.parse_args().action
    register() if action == "register" else run()


if __name__ == "__main__":
    main()
