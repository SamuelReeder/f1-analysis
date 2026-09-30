"""Dashboard release safety and the statistical meaning of displayed data."""
import json

import numpy as np
import pytest

from f1rank import dashboard


def test_joint_gap_keeps_shared_uncertainty():
    # Strong shared uncertainty cancels in the contrast. Marginal CI subtraction would not.
    shared = np.linspace(-10, 10, 400)
    draws = np.column_stack([shared + .2, shared])
    pairs = dashboard.comparisons(["a", "b"], draws)
    assert pairs["a|b"] == {"q05": .2, "median": .2, "q95": .2, "p_ahead": 1.0}
    assert pairs["b|a"]["median"] == -.2
    assert pairs["b|a"]["p_ahead"] == 0
    assert pairs["a|a"]["p_ahead"] == .5


def test_comparison_rejects_nonfinite_or_misaligned_samples():
    for draws in (np.array([[np.nan, 0]]), np.ones((3, 4)), np.ones(2)):
        with pytest.raises(ValueError, match="comparison draws"):
            dashboard.comparisons(["a", "b"], draws)


@pytest.fixture
def payload(monkeypatch):
    value = {"schema_version": 1, "meta": {"data_as_of": {"event_id": "2026-15"}},
             "drivers": [{"id": "a", "pace": .2}]}
    monkeypatch.setattr(dashboard, "build_payload", lambda: value)
    return value


def test_release_is_immutable_and_pointer_changes_last(tmp_path, payload, monkeypatch):
    first = dashboard.publish(tmp_path)
    contents = (tmp_path / first["url"]).read_bytes()
    first_mtime = (tmp_path / first["url"]).stat().st_mtime_ns
    assert dashboard.publish(tmp_path)["release"] == first["release"]
    assert (tmp_path / first["url"]).stat().st_mtime_ns == first_mtime
    payload["drivers"][0]["pace"] = .3
    write = dashboard.atomic_json
    def verify_pointer(path, value):
        if path.name == "latest.json":
            assert json.loads((tmp_path / value["url"]).read_text()) == payload
        write(path, value)
    monkeypatch.setattr(dashboard, "atomic_json", verify_pointer)
    second = dashboard.publish(tmp_path)
    assert second["release"] != first["release"]
    assert (tmp_path / first["url"]).read_bytes() == contents


def test_failed_publication_keeps_previous_release_and_records_error(tmp_path, payload, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path / "repo")
    dashboard.run(data_dir=tmp_path)
    before = (tmp_path / "latest.json").read_bytes()
    def fail():
        raise ValueError("Fingerprint mismatch")
    monkeypatch.setattr(dashboard, "build_payload", fail)
    with pytest.raises(ValueError, match="Fingerprint"):
        dashboard.run(data_dir=tmp_path)
    assert (tmp_path / "latest.json").read_bytes() == before
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "failed"
    assert status["error"] == "Fingerprint mismatch"
    assert len((dashboard.ROOT / "outputs/dashboard/runs.jsonl").read_text().splitlines()) == 2


def test_concurrent_refresh_cannot_replace_status(tmp_path, payload, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path / "repo")
    with dashboard.publication_lock(tmp_path):
        with pytest.raises(RuntimeError, match="Another dashboard refresh"):
            dashboard.run(data_dir=tmp_path)
    assert not (tmp_path / "status.json").exists()


def test_interrupted_refresh_is_reported_using_process_owned_lock(tmp_path):
    (tmp_path / "status.json").write_text('{"state":"running","stage":"fit"}')
    with dashboard.publication_lock(tmp_path):
        assert dashboard.current_status(tmp_path)["state"] == "running"
    assert dashboard.current_status(tmp_path)["state"] == "failed"


def test_corrupt_immutable_release_cannot_be_republished(tmp_path, payload):
    first = dashboard.publish(tmp_path)
    (tmp_path / first["url"]).write_text('{}')
    before = (tmp_path / "latest.json").read_bytes()
    with pytest.raises(ValueError, match="corrupt"):
        dashboard.publish(tmp_path)
    assert (tmp_path / "latest.json").read_bytes() == before


def test_missing_provenance_cannot_publish(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "RATINGS", tmp_path)
    with pytest.raises(dashboard.StaleArtifact):
        dashboard.build_payload()
    assert not (tmp_path / "latest.json").exists()


def test_legacy_racing_gate_is_never_advertised(tmp_path, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    path = tmp_path / "outputs/pitstops/summary.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"gate_team_ops_rating": true}')
    row = next(r for r in dashboard.racing_health() if r["name"] == "Pit stop operations")
    assert row["status"] == "stale"
    assert "regeneration" in row["reason"]


def test_refresh_runs_the_entire_pipeline_before_publishing(tmp_path, payload, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path / "repo")
    calls = []
    monkeypatch.setattr(dashboard.subprocess, "run", lambda command, **kwargs: calls.append(command[-1]))
    dashboard.run(refresh=True, data_dir=tmp_path)
    assert calls == ["f1rank.fetch", "f1rank.build", "f1rank.fit", "f1rank.export"]
    assert (tmp_path / "latest.json").exists()


def test_checked_current_data_contract():
    if not (dashboard.RATINGS / "manifest.json").exists():
        pytest.skip("Integration check requires a local verified qualifying export")
    result = dashboard.build_payload()
    assert not any(path.startswith("outputs/fits/") for path in result["provenance"]["inputs"])
    json.dumps(result, allow_nan=False)
    event = result["meta"]["data_as_of"]["event_id"]
    for driver in result["drivers"]:
        last = next(p for p in result["history"]["drivers"][driver["id"]] if p["event"] == event)
        for metric in ("headline", "portable"):
            assert last[metric]["median"] == pytest.approx(driver[metric]["median"], abs=1e-5)
            assert driver[metric]["q05"] <= driver[metric]["median"] <= driver[metric]["q95"]
    for pairs in result["comparisons"].values():
        for key, pair in pairs.items():
            a, b = key.split("|")
            assert pair["p_ahead"] + pairs[f"{b}|{a}"]["p_ahead"] == pytest.approx(1)
            assert pair["q05"] <= pair["median"] <= pair["q95"]


def test_export_rejects_mismatched_portable_fit_identity(monkeypatch):
    if not (dashboard.RATINGS / "fit_metadata.json").exists():
        pytest.skip("Requires a local verified export")
    original = dashboard.read_json
    def read(path):
        value = original(path)
        if path.name == "fit_metadata.json":
            value = {**value, "created_utc": "different-fit"}
        return value
    monkeypatch.setattr(dashboard, "read_json", read)
    with pytest.raises(ValueError, match="portable fit metadata"):
        dashboard.build_payload()
