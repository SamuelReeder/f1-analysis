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
    monkeypatch.setattr(dashboard.subprocess, "run", lambda command, **kwargs: calls.append(command[2]))
    dashboard.run(refresh=True, data_dir=tmp_path)
    assert calls == ["f1rank.fetch", "f1rank.build", "f1rank.fit", "f1rank.export", "f1rank.asof", "f1rank.asof"]
    assert (tmp_path / "latest.json").exists()


def test_refresh_refits_with_the_published_sampler_settings():
    fit = next(command for stage, command in dashboard.refresh_steps() if stage == "fit")
    assert fit[3:] == ["--warmup=1500", "--samples=1500", "--chains=4"]
    published = dashboard.RATINGS / "fit_metadata.json"
    if not published.exists():
        pytest.skip("Requires the committed qualifying export")
    settings = json.loads(published.read_text())["settings"]
    assert {k: settings[k] for k in dashboard.FIT_SETTINGS} == dashboard.FIT_SETTINGS


def test_race_refresh_orders_extraction_and_both_fits_before_publication(tmp_path, payload, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path / "repo")
    calls = []
    def execute(command, **kwargs):
        assert not (tmp_path / "latest.json").exists()
        calls.append(command)
    monkeypatch.setattr(dashboard.subprocess, "run", execute)
    dashboard.run(refresh=True, with_races=True, race_python="gpu-python", extract_python="fastf1-python",
                  data_dir=tmp_path)
    assert [c[2] if c[1] == "-m" else "extract" for c in calls] == [
        "f1rank.fetch", "f1rank.build", "extract", "extract", "f1rank.racedata", "f1rank.timeline",
        "f1rank.fit", "f1rank.export", "f1rank.asof", "f1rank.asof", "f1rank.race_total"]
    assert calls[2][0] == calls[3][0] == "fastf1-python"
    assert calls[3][-1] == "--sprint"
    assert calls[-1] == ["gpu-python", "-m", "f1rank.race_total", "--validate", "--export"]
    assert (tmp_path / "latest.json").exists()


def test_qualifying_refresh_keeps_existing_race_export_current_and_failure_preserves_release(tmp_path, payload, monkeypatch):
    monkeypatch.setattr(dashboard, "ROOT", tmp_path / "repo")
    race = dashboard.ROOT / "outputs/race_total/pace.json"
    race.parent.mkdir(parents=True)
    race.write_text('{}')
    dashboard.run(data_dir=tmp_path)
    before = (tmp_path / "latest.json").read_bytes()
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if "f1rank.race_total" in command:
            raise RuntimeError("Race fitting failed")
    monkeypatch.setattr(dashboard.subprocess, "run", execute)
    with pytest.raises(RuntimeError, match="Race fitting failed"):
        dashboard.run(refresh=True, data_dir=tmp_path)
    assert [c[2] for c in calls] == ["f1rank.fetch", "f1rank.build", "f1rank.fit", "f1rank.export",
                                     "f1rank.asof", "f1rank.asof", "f1rank.race_total"]
    assert (tmp_path / "latest.json").read_bytes() == before
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["state"] == "failed" and status["stage"] == "race_total"


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


def test_breakdown_adds_car_and_driver_draw_by_draw():
    # Opposite shared draws cancel in the total, so its interval is tight.
    shared = np.linspace(-1, 1, 401)
    in_team = np.column_stack([.1 + shared, -.1 + shared])
    car = np.column_stack([.5 - shared, -.5 - shared])
    rows = dashboard.breakdown(["a", "b"], ["x", "y"], in_team, car, {"a": "y", "b": "x"})
    assert [r["id"] for r in rows] == ["b", "a"]
    b = rows[0]
    assert b["car"] == pytest.approx(.5) and b["driver"] == pytest.approx(-.1)
    assert b["total"]["q05"] == pytest.approx(.4) and b["total"]["q95"] == pytest.approx(.4)
    # Published full-posterior medians replace the thinned draws' medians for the parts.
    rows = dashboard.breakdown(["a", "b"], ["x", "y"], in_team, car, {"a": "y", "b": "x"},
                               {"drivers": {"b": -.12}, "cars": {"x": .52}})
    assert rows[0]["driver"] == -.12 and rows[0]["car"] == .52
    assert rows[1]["driver"] == pytest.approx(.1)


def test_race_ranks_preserve_covariance_and_use_seconds():
    from f1rank.race_total import estimates
    shared = np.linspace(-100, 100, 1600)
    # A common shift cancels when the field is centred. A 1% gap is 0.9s / 90s.
    rows = estimates(np.column_stack([shared + 1, shared]))
    assert rows[0]["median"] == pytest.approx(.45)
    assert rows[1]["median"] == pytest.approx(-.45)
    assert rows[0]["q05"] == pytest.approx(rows[0]["q95"])
    assert rows[0]["rank_lo"] == rows[0]["rank_hi"] == 1
    assert rows[0]["p_fastest"] == 1


@pytest.fixture
def race_release():
    from copy import deepcopy
    d = dict(converged=True, rhat_max=1.01, n_draws=3200, divergences=0)
    folds = ["2024-10", "2025-10", "2026-07"]
    gate = dict(passed=True, n_races=24, mse_difference_ci95=[-.2, -.01],
                coverage90=.9, mean_interval_width=.5, baseline_interval_width=1.)
    estimate = dict(q05=-.1, median=0., q95=.1, rank_lo=1, rank_hi=1)
    return dict(schema_version=1, model="total-dry-race-pace-v1", diagnostics=d,
                validation=dict(folds=folds, metrics={k: deepcopy(gate) for k in ("drivers", "cars")},
                                fits={f: {k: dict(data_as_of=f, diagnostics=deepcopy(d))
                                      for k in ("full", "no_driver", "no_car")} for f in folds}),
                drivers=[dict(id="a", pace=deepcopy(estimate))], cars=[dict(id="x", pace=deepcopy(estimate))])


def test_race_ranking_requires_predictive_and_uncertainty_evidence(race_release):
    from f1rank.race_publication import check
    check(race_release)
    race_release["validation"]["metrics"]["cars"]["mse_difference_ci95"][1] = .01
    with pytest.raises(ValueError, match="gate disagrees"):
        check(race_release)
    race_release["validation"]["metrics"]["cars"]["passed"] = False
    with pytest.raises(ValueError, match="Unsupported race pace ranking"):
        check(race_release)
    race_release["cars"] = []
    check(race_release)  # The supported driver table can be published independently.


def test_race_season_history_follows_the_same_gates(race_release):
    from copy import deepcopy
    from f1rank.race_publication import check
    season = dict(season=2025, drivers=deepcopy(race_release["drivers"]), cars=deepcopy(race_release["cars"]))
    race_release["seasons"] = [season]
    check(race_release)
    season["cars"][0]["pace"]["rank_hi"] = 2  # one car cannot rank second of one
    with pytest.raises(ValueError, match="intervals"):
        check(race_release)
    season["cars"][0]["pace"]["rank_hi"] = 1
    gate = race_release["validation"]["metrics"]["cars"]
    gate.update(passed=False, coverage90=.8)
    race_release["cars"] = []
    with pytest.raises(ValueError, match="Unsupported race pace ranking"):
        check(race_release)  # a withheld table stays withheld in every season
    season["cars"] = []
    check(race_release)


def test_race_validation_cannot_use_future_data_or_failed_fit(race_release):
    from f1rank.race_publication import check
    fit = race_release["validation"]["fits"]["2024-10"]["no_driver"]
    fit["data_as_of"] = "2024-11"
    with pytest.raises(ValueError, match="validation fit"):
        check(race_release)
    fit["data_as_of"] = "2024-10"
    fit["diagnostics"]["rhat_max"] = 1.2
    with pytest.raises(ValueError, match="validation fit"):
        check(race_release)


def test_race_publication_rejects_changed_source(tmp_path, race_release):
    from f1rank.race_publication import load
    from f1rank.artifacts import record, StaleArtifact
    directory = tmp_path / "outputs/race_total"
    directory.mkdir(parents=True)
    data = directory / "pace.json"
    data.write_text(json.dumps(race_release))
    source = tmp_path / "model.py"
    source.write_text("version 1")
    record(directory, [data], model="test", inputs=[source])
    assert load(tmp_path)["drivers"]
    source.write_text("version 2")
    with pytest.raises(StaleArtifact):
        load(tmp_path)


def test_race_point_forecasts_exclude_future_random_noise():
    import pandas as pd
    from f1rank.race_total import point_prediction
    entries = pd.DataFrame(dict(event_id=["2026-11"] * 2, driver_id=["a", "rookie"], team=["x"] * 2))
    meta = {"catalog": {"drivers": ["a"], "driver_seasons": ["2026|a"], "cars": ["2026|x"]}}
    post = {"skill": np.array([[.1], [.3]]), "form": np.array([[.2], [.2]]), "package": np.array([[1.], [1.]])}
    np.testing.assert_allclose(point_prediction(post, meta, entries), [1.4, 1.])
    # A refitted no-driver model predicts identical teammate means, exactly.
    np.testing.assert_array_equal(point_prediction({"package": post["package"]}, meta, entries), [1., 1.])
