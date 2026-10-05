"""An exploratory historical success cannot masquerade as prospective validation."""
import datetime as dt
import json
import subprocess

import numpy as np
import pandas as pd
import pytest

from analysis import championship_followup as f


@pytest.fixture
def protocol():
    return {"first_test_season": 2030, "last_test_season": 2032,
            "outcomes_after_date": "2030-06-01", "decision_after_date": "2032-12-31",
            "min_converged_test_seasons": 3}


def folds():
    return [{"season": y, "events": [f"{y}-15"], "training_last_event": f"{y-1}-20",
             "log_predictive_density": {"candidate": [-1.0], **{b: [-2.0] for b in f.BASELINES}}}
            for y in (2030, 2031, 2032)]


def test_no_early_decision_even_with_positive_scores(protocol):
    result = f.decide(folds(), protocol, dt.date(2032, 12, 31))
    assert result["gate"] is False and result["status"] == "awaiting_future_evidence"
    assert "comparisons" not in result


def test_only_complete_independent_horizon_can_pass(protocol):
    today = dt.date(2033, 1, 1)
    assert f.decide(folds(), protocol, today)["gate"] is True
    assert f.decide(folds()[:-1], protocol, today)["gate"] is False
    assert f.decide(folds()[:-1], protocol, today, excluded=[2032])["gate"] is False
    failed = folds()
    for row in failed:
        row["log_predictive_density"]["candidate"] = [-3.0]
    result = f.decide(failed, protocol, today)
    assert result["status"] == "completed" and result["gate"] is False
    assert result["publication_authorized"] is False


def test_training_or_duplicate_event_leakage_is_rejected(protocol):
    rows = folds()
    rows[0]["training_last_event"] = "2030-01"
    with pytest.raises(ValueError, match="cutoff"):
        f.decide(rows, protocol, dt.date(2033, 1, 1))
    rows = folds()
    rows[0]["events"] *= 2
    with pytest.raises(ValueError, match="identity"):
        f.decide(rows, protocol, dt.date(2033, 1, 1))


def test_outcomes_seen_before_registration_are_not_eligible(protocol):
    events = pd.DataFrame({"event_id": ["2029-20", "2030-01", "2030-02", "2030-03", "2033-01"],
                           "date": ["2029-12-01", "2030-05-31", "2030-06-01", "2030-06-02", "2033-01-01"]})
    assert f.eligible_events(events, protocol) == ["2030-03"]


def test_completed_calendar_does_not_hide_missing_races(protocol):
    registry = [{"event_id": "2030-15", "date": "2030-08-01"},
                {"event_id": "2030-16", "date": "2030-09-01"}]
    coverage = f.check_event_coverage(pd.DataFrame(registry[:1]), registry, protocol)
    assert coverage["complete"] is False and coverage["missing_events"] == ["2030-16"]
    assert f.check_event_coverage(pd.DataFrame(registry), registry, protocol)["complete"] is True
    assert f.check_event_coverage(pd.DataFrame(registry), [], protocol)["complete"] is False


def test_scoring_uses_only_earlier_seasons_for_training(monkeypatch):
    frame = pd.DataFrame({"event_id": ["2029-20", "2030-01", "2030-15", "2031-01"],
                          "season": [2029, 2030, 2030, 2031], "driver_id": ["a"] * 4})
    monkeypatch.setattr(f.c, "race_orders", lambda s: frame)
    monkeypatch.setattr(f.c, "finishers", lambda x: x)
    trained = []

    def fit(train, variant):
        trained.append((train.copy(), variant))
        return {}

    def predict(post, test, *args):
        assert test.event_id.tolist() == ["2030-15"]
        return np.array([-2.0], dtype=np.float32)

    monkeypatch.setattr(f.c, "fit_race", fit)
    monkeypatch.setattr(f.c, "log_pred", predict)
    monkeypatch.setattr(f.c, "quality_log_pred", predict)
    result = f.score_fold(2030, ["2030-15"], {
        "race_specific_pace": {2030: (np.array(["a"]), np.array([[0.1], [0.2]]))}})
    assert all(t.event_id.tolist() == ["2029-20"] for t, _ in trained)
    assert len(trained) == 4 and result["training_last_event"] == "2029-20"


def test_registration_must_be_committed_and_cannot_be_rewritten(tmp_path, monkeypatch):
    def git(*args):
        subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", *args],
                       cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("commit", "-q", "--allow-empty", "-m", "Initial state")
    p = tmp_path / "protocol.json"
    original = {"source_hashes": {}, "candidate": list(f.CANDIDATE),
                "primary_baseline": "grid_ratings", "min_converged_test_seasons": 3}
    p.write_text(json.dumps(original))
    monkeypatch.setattr(f, "ROOT", tmp_path)
    monkeypatch.setattr(f, "PROTOCOL", p)
    monkeypatch.setattr(f, "source_hashes", lambda: {})
    with pytest.raises(ValueError, match="Commit"):
        f.read_protocol()
    git("add", "protocol.json")
    git("commit", "-q", "-m", "Register experiment")
    assert f.read_protocol() == original
    p.write_text(json.dumps({**original, "changed_after_registration": True}))
    git("commit", "-q", "-am", "Attempt to change the registration")
    with pytest.raises(ValueError, match="registration was changed"):
        f.read_protocol()
