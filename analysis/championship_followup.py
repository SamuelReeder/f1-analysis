"""Fixed race-pace follow-up, kept separate from the failed championship experiment.

Register before new outcomes: python -m analysis.championship_followup register
Evaluate the fixed horizon: python -m analysis.championship_followup evaluate
Run evaluation as a memory-capped systemd user service when fits are due.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from f1rank import artifacts, championship as c
from f1rank.qualifying import dependencies, unconverged_folds

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/championship_followup_protocol.json"
OUT = ROOT / "outputs/analysis/championship_followup"
CANDIDATE = ("race_specific_pace",)
BASELINES = ("grid_ratings", "grid", "ratings")


def source_hashes():
    paths = [*sorted((ROOT / "f1rank").glob("*.py")),
             *sorted((ROOT / "extract").glob("*.py")), Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def register():
    now = dt.datetime.now(dt.timezone.utc)
    first = now.year
    last = first + c.MIN_SELECTION_SEASONS - 1
    reference = json.loads((c.OUT / "manifest.json").read_text())
    protocol = {
        "schema_version": 1,
        "registered_at": now.isoformat(),
        "registration_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "candidate": list(CANDIDATE), "primary_baseline": "grid_ratings",
        "secondary_baselines": ["grid", "ratings"],
        "first_test_season": first, "last_test_season": last,
        "outcomes_after_date": now.date().isoformat(),
        "decision_after_date": dt.date(last, 12, 31).isoformat(),
        "development_data_as_of": reference["data_as_of"],
        "min_converged_test_seasons": c.MIN_SELECTION_SEASONS,
        "selection": "fixed race-specific pace; no search or backward selection",
        "primary_rule": "unchanged paired_seasons: lower 95% season-block bound strictly above zero",
        "secondary_rule": "reported descriptively; cannot rescue a failed primary comparison",
        "sampling": "unchanged championship.fit_race defaults and artifacts.RETRY; no seed search",
        "training": "qualifying and quality inputs and race coefficients trained before each test season",
        "failed_folds": "exclude exhausted qualifying folds; report failure if insufficient valid seasons",
        "stopping": "one analysis after the fixed calendar horizon; no interim significance checks or extensions",
        "publication": "research evidence only; never write canonical championship outputs or dashboard data",
        "source_hashes": source_hashes(),
    }
    # An existing registration is immutable, including after an inconclusive result.
    with PROTOCOL.open("x") as f:
        json.dump(protocol, f, indent=2)
        f.write("\n")
    print(f"Registered {PROTOCOL.relative_to(ROOT)}", flush=True)


def read_protocol():
    relative = str(PROTOCOL.relative_to(ROOT))
    registered = subprocess.check_output(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", relative], cwd=ROOT, text=True).splitlines()
    if not registered:
        raise ValueError("Commit the registration before evaluating")
    original = subprocess.check_output(["git", "show", f"{registered[-1]}:{relative}"], cwd=ROOT)
    if original != PROTOCOL.read_bytes():
        raise ValueError("The committed registration was changed; a new study needs a new registration")
    p = json.loads(PROTOCOL.read_text())
    if p["source_hashes"] != source_hashes():
        raise ValueError("Registered model or evaluator code changed; use its frozen revision")
    if (p["candidate"] != list(CANDIDATE) or p["primary_baseline"] != "grid_ratings"
            or p["min_converged_test_seasons"] != c.MIN_SELECTION_SEASONS):
        raise ValueError("Protocol disagrees with the fixed experiment")
    return p


def eligible_events(events, protocol):
    date = pd.to_datetime(events.date, utc=True, errors="raise").dt.date
    years = events.event_id.str[:4].astype(int)
    after = dt.date.fromisoformat(protocol["outcomes_after_date"])
    return events.loc[(date > after) & years.between(
        protocol["first_test_season"], protocol["last_test_season"]), "event_id"].tolist()


def check_event_coverage(local_events, race_registry, protocol):
    """A calendar year being over does not prove that locally downloaded data are complete."""
    expected = set(eligible_events(pd.DataFrame(race_registry, columns=["event_id", "date"]), protocol))
    local = set(eligible_events(local_events, protocol))
    return {"complete": bool(expected) and local == expected,
            "expected_events": sorted(expected), "missing_events": sorted(expected - local),
            "unexpected_events": sorted(local - expected)}


def score_fold(season, eligible, effects):
    frame = c.finishers(c.race_orders(season))
    train = frame[(frame.season >= c.ENTRY_FIRST_SEASON) & (frame.season < season)].copy()
    test = frame[(frame.season == season) & frame.event_id.isin(eligible)].copy()
    expected = sorted(e for e in eligible if int(e[:4]) == season)
    events = sorted(test.event_id.unique().tolist())
    if train.empty or not events or events != expected:
        raise ValueError(f"Empty training data or incomplete test coverage for {season}")
    scores = {name: c.log_pred(c.fit_race(train, name), test, name) for name in BASELINES}
    drivers, draws = effects["race_specific_pace"][season]
    train["quality_0"] = train.driver_id.map(pd.Series(draws.mean(0), index=drivers)).fillna(0.0)
    post = c.fit_race(train, "grid_ratings_quality")
    scores["candidate"] = c.quality_log_pred(post, test, CANDIDATE, effects, season)
    if any(v.shape != (len(events),) or not np.isfinite(v).all() for v in scores.values()):
        raise ValueError(f"Invalid score vector for {season}")
    c.jax.clear_caches()
    return {"season": season, "events": events,
            "training_last_event": str(train.event_id.max()),
            "log_predictive_density": {k: v.astype(float).tolist() for k, v in scores.items()}}


def decide(folds, protocol, today, excluded=()):
    """No significance decision before the fixed horizon or with missing seasons."""
    base = {"gate": False, "status": "awaiting_future_evidence", "publication_authorized": False}
    if today <= dt.date.fromisoformat(protocol["decision_after_date"]):
        return base
    expected = set(range(protocol["first_test_season"], protocol["last_test_season"] + 1))
    years = [f["season"] for f in folds]
    if (len(set(years)) != len(years) or set(years) & set(excluded)
            or set(years) | set(excluded) != expected):
        return {**base, "status": "incomplete_test_seasons"}
    if len(years) < protocol["min_converged_test_seasons"]:
        return {**base, "status": "insufficient_converged_test_seasons"}
    intervals = {}
    seen = set()
    for fold in folds:
        if (not fold["events"] or len(set(fold["events"])) != len(fold["events"])
                or seen.intersection(fold["events"])
                or any(int(e[:4]) != fold["season"] for e in fold["events"])
                or int(fold["training_last_event"][:4]) >= fold["season"]):
            raise ValueError("Invalid fold identity or training cutoff")
        seen.update(fold["events"])
        for name in ("candidate", *BASELINES):
            v = np.asarray(fold["log_predictive_density"][name])
            if v.shape != (len(fold["events"]),) or not np.isfinite(v).all():
                raise ValueError("Invalid fold scores")
    for name in BASELINES:
        intervals[name] = c.paired_seasons({f["season"]: (
            np.asarray(f["log_predictive_density"]["candidate"], dtype=np.float32)
            - np.asarray(f["log_predictive_density"][name], dtype=np.float32)) for f in folds})
    return {**base, "status": "completed", "gate": intervals["grid_ratings"]["enters"],
            "comparisons": intervals}


def evaluate():
    protocol = read_protocol()
    today = dt.datetime.now(dt.timezone.utc).date()
    protocol_hash = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    base = {"protocol_sha256": protocol_hash, "assessed_on": today.isoformat(),
            "decision_after_date": protocol["decision_after_date"]}
    if today <= dt.date.fromisoformat(protocol["decision_after_date"]):
        # Do not fit, score, peek at partial outcomes, or claim that current history passed.
        artifacts.atomic_json(OUT / "status.json", {**base, **decide([], protocol, today)})
        print("Awaiting the registered independent evidence horizon; no gate passed", flush=True)
        return
    if (OUT / "result.json").exists():
        raise FileExistsError("The completed follow-up is immutable")
    events = pd.read_parquet(ROOT / "data/processed/events.parquet")
    years = list(range(protocol["first_test_season"], protocol["last_test_season"] + 1))
    # Retrieve the completed seasons afresh, before fitting, to catch incomplete local
    # seasons. This uses the existing paginated fetcher and its versioned raw archive.
    from f1rank.fetch import RAW, fetch_table
    registry = [{"event_id": f"{s}-{int(r['round']):02d}", "date": r["date"]}
                for s in years for r in fetch_table(s, "results", refresh=True)]
    coverage = check_event_coverage(events, registry, protocol)
    eligible = coverage["expected_events"]
    excluded = [s for s in unconverged_folds() if s in years]
    # Incomplete data do not become a failed or passing statistical result.
    missing = [s for s in years if not any(int(e[:4]) == s for e in eligible)]
    if missing or not coverage["complete"]:
        artifacts.atomic_json(OUT / "status.json", {
            **base, "status": "incomplete_test_seasons", "missing_seasons": missing,
            "coverage": coverage,
            "gate": False, "publication_authorized": False})
        return
    effects = {"race_specific_pace": c.quality_draws("race_specific_pace")}
    inputs = [*artifacts.input_files(), *dependencies(years), PROTOCOL, Path(__file__),
              ROOT / "outputs/race/multi_heldout_effects.npz",
              ROOT / "outputs/race/multi_heldout.manifest.json",
              *(RAW / f"{s}_results.json" for s in years)]
    fingerprints = {p: artifacts.digest(p) for p in inputs}
    folds = [score_fold(s, eligible, effects) for s in years if s not in excluded]
    decision = decide(folds, protocol, today, excluded)
    if fingerprints != {p: artifacts.digest(p) for p in inputs}:
        raise ValueError("Follow-up inputs changed during evaluation")
    artifacts.atomic_json(OUT / "result.json", {
        **base, **decision, "folds": folds, "coverage": coverage,
        "excluded_unconverged_qualifying_folds": excluded,
        "fit_attempts": artifacts.fit_record()})
    artifacts.record(OUT, [OUT / "result.json"], model="fixed-race-specific-pace-followup",
                     inputs=inputs,
                     details={"protocol_sha256": protocol_hash})
    artifacts.atomic_json(OUT / "status.json", {**base, **decision})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "evaluate"))
    action = parser.parse_args().action
    register() if action == "register" else evaluate()


if __name__ == "__main__":
    main()
