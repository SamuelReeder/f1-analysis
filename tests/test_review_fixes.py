"""Regression checks for historical validation and safe publication."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from f1rank import artifacts, benchmark, championship, export, qualifying, racemulti


def race_laps():
    return pd.DataFrame({"event_id": ["2018-01"] * 4, "driver_id": ["a"] * 4,
                         "lap_number": [2, 3, 4, 5], "stint": [1] * 4, "stint_key": ["a_1"] * 4,
                         "y": [0.1, 0.2, 0.3, 0.4], "quali": [0.0] * 4, "season": [2018] * 4,
                         "team": ["A"] * 4, "compound": ["SOFT"] * 4, "source": ["fastf1"] * 4,
                         "tyre_life": [2, 3, 4, 5], "close": [False] * 4, "near": [False] * 4,
                         "unpressured": [False] * 4, "coast": [0.1, 0.2, 0.3, 0.4]})


@pytest.mark.parametrize("column,value", [("tyre_life", 20), ("compound", "HARD"), ("close", True),
                                         ("near", True), ("unpressured", True), ("team", "B"),
                                         ("stint_key", "a_2"), ("source", "jolpica"), ("coast", 4.0)])
def test_cache_invalidated_by_each_likelihood_input(tmp_path, monkeypatch, column, value):
    calls = []
    def fake_fit(C):
        calls.append(1)
        return {"u": np.zeros((2, 1)), "_converged": True}, ["a"]
    monkeypatch.setattr(racemulti, "fit", fake_fit)
    monkeypatch.setattr(racemulti, "COAST", True)
    C = race_laps()
    path = tmp_path / "fold.npz"
    racemulti.checkpointed(C, path)
    racemulti.checkpointed(C.copy(), path)
    assert len(calls) == 1
    C.loc[1, column] = value
    racemulti.checkpointed(C, path)
    assert len(calls) == 2


def test_failed_full_race_fit_keeps_published_files(tmp_path, monkeypatch):
    prior = tmp_path / "multi_drivers.csv"
    prior.write_text("previous good result")
    monkeypatch.setattr(racemulti, "OUT", tmp_path)
    monkeypatch.setattr(racemulti, "with_qualifying", lambda C: C)
    monkeypatch.setattr(racemulti, "fit", lambda *a, **kw: ({"_converged": False, "_attempts": []}, []))
    with pytest.raises(RuntimeError, match="existing outputs retained"):
        racemulti.full(race_laps())
    assert prior.read_text() == "previous good result"
    assert list(tmp_path.iterdir()) == [prior]


def test_diagnostics_reject_bad_chains_nans_and_divergences():
    rng = np.random.default_rng(5)
    good = {"x": rng.normal(size=(4, 600, 2))}
    assert artifacts.diagnostics(good, 0)["converged"]
    bad = good["x"].copy()
    bad[0] += 10
    assert not artifacts.diagnostics({"x": bad}, 0)["converged"]
    bad[0, 0, 0] = np.nan
    assert not artifacts.diagnostics({"x": bad}, 0)["converged"]
    assert not artifacts.diagnostics(good, 50)["converged"]
    assert not artifacts.diagnostics({"x": good["x"][:1]}, 0)["converged"]


def test_missing_legacy_and_changed_artifacts_are_rejected(tmp_path):
    source, output = tmp_path / "input.csv", tmp_path / "draws.npz"
    source.write_text("original data")
    output.write_text("draws")
    with pytest.raises(artifacts.StaleArtifact):
        artifacts.require(tmp_path)
    artifacts.record(tmp_path, [output], model="test", inputs=[source])
    artifacts.require(tmp_path, required_outputs=[output])
    source.write_text("corrected data")
    with pytest.raises(artifacts.StaleArtifact, match="changed"):
        artifacts.require(tmp_path)
    source.write_text("original data")
    output.write_text("partial new output")
    with pytest.raises(artifacts.StaleArtifact, match="changed"):
        artifacts.require(tmp_path)


def test_report_does_not_advertise_legacy_championship_gates(tmp_path, monkeypatch):
    from f1rank import racereport
    monkeypatch.setattr(racereport, "OUT", tmp_path)
    out = tmp_path / "championship"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"qualities_entered": ["first_lap"]}))
    assert racereport._load("championship/summary.json") is None
    assert "championship/summary.json" in racereport.STALE


def test_historical_cutoff_rejects_main_fit():
    d = SimpleNamespace(obs=pd.DataFrame({"event_id": ["2020-17", "2021-01"]}), train=None,
                        events=pd.DataFrame({"season": [2020, 2021]}))
    with pytest.raises(ValueError, match="test season"):
        qualifying.check_cutoff(d, 2021)
    d.train = np.array([True, False])
    qualifying.check_cutoff(d, 2021)


def test_missing_qualifying_fold_does_not_fall_back_to_main(tmp_path, monkeypatch):
    monkeypatch.setattr(qualifying, "FITS", tmp_path)
    (tmp_path / "main.npz").write_text("must not be loaded")
    qualifying.features.cache_clear()
    with pytest.raises(FileNotFoundError, match="--seasons 2021"):
        qualifying.features(2021)


def test_race_orders_request_the_test_seasons_qualifying_fold(tmp_path, monkeypatch):
    calls = []
    def features(S=None):
        calls.append(S)
        return (pd.DataFrame({"event_id": ["2021-01"], "driver_id": ["a"], "driver": [S or 9999]}),
                pd.DataFrame({"event_id": ["2021-01"], "team": ["A"], "car": [0.0]}))
    monkeypatch.setattr(qualifying, "features", features)
    monkeypatch.setattr(benchmark, "PROCESSED", tmp_path)
    pd.DataFrame({"event_id": ["2021-01"], "driver_id": ["a"], "team": ["A"], "grid": [1],
                  "position": [1]}).to_parquet(tmp_path / "race.parquet")
    assert benchmark.race_orders(2021).driver.tolist() == [2021]
    assert calls == [2021]


def test_race_laps_replace_full_history_quali_with_fold_features(monkeypatch):
    calls = []
    def features(S=None):
        calls.append(S)
        return pd.DataFrame({"event_id": ["2018-01"], "driver_id": ["a"], "driver": [0.75]}), None
    monkeypatch.setattr(qualifying, "features", features)
    C = racemulti.with_qualifying(race_laps().assign(quali=9999), 2019)
    np.testing.assert_allclose(C.quali, 0.75)
    assert calls == [2019]


def test_redundant_quality_fails_conditional_entry():
    def score(S, names):
        # Both beat baseline alone, but B adds nothing when A is present.
        return np.full(3, 2.0 if "a" in names else 1.0 if "b" in names else 0.0)
    selected, tests = championship.select_qualities(["a", "b"], [2012, 2013, 2014], score)
    assert selected == ["a"]
    assert tests["a"]["enters"] and not tests["b"]["enters"]
    assert tests["b"]["conditioned_on"] == ["a"]


def test_outer_year_cannot_choose_its_own_qualities():
    years = list(range(2012, 2019))
    def run(outer_gain):
        def score(S, names):
            gain = outer_gain if S >= 2015 else -1.0
            return np.full(3, gain if "a" in names else 0.0)
        return championship.nested_selection(["a"], years, score)[2]
    positive, negative = run(100.0), run(-100.0)
    assert positive["selected_by_outer_season"]["2015"] == []
    assert negative["selected_by_outer_season"]["2015"] == []
    # Later selection is allowed to react to outcomes from earlier outer seasons.
    assert positive["selected_by_outer_season"]["2018"] != negative["selected_by_outer_season"]["2018"]


def test_quality_prediction_matches_direct_combined_strength():
    R = pd.DataFrame({"event_id": ["2020-01"] * 2, "driver_id": ["a", "b"], "rank": [0, 1],
                      "grid_slot": [1, 2], "car": [0.0, 0.0], "driver": [0.0, 0.0]})
    post = {"c_grid": np.zeros(4), "a_car": np.zeros(4), "b_driver": np.zeros(4),
            "b_quality": np.tile([2.0, 3.0], (4, 1))}
    effects = {n: {2020: (np.array(["a", "b"]), np.tile([1.0, 0.0], (4, 1)))}
               for n in ("race_specific_pace", "degradation")}
    got = championship.quality_log_pred(post, R, tuple(effects), effects, 2020, n_draws=4)
    np.testing.assert_allclose(got, [-np.log1p(np.exp(-5.0))], atol=1e-6)


def test_export_refuses_failed_fit_before_writing(tmp_path, monkeypatch):
    monkeypatch.setattr(export, "build_design", lambda *a: None)
    monkeypatch.setattr(export, "load", lambda *a: ({"skill": np.zeros((1, 2, 2))}, {"divergences": 0}))
    monkeypatch.setattr(export, "load_meta", lambda *a: {})
    monkeypatch.setattr(export, "OUT", tmp_path)
    old = tmp_path / "meta.json"
    old.write_text("previous good publication")
    with pytest.raises(RuntimeError, match="diagnostics"):
        export.export()
    assert old.read_text() == "previous good publication"


def test_export_headline_and_portable_draws_match_their_own_comparisons(tmp_path, monkeypatch):
    from f1rank.design import build_design
    d = build_design(2026)
    rng = np.random.default_rng(10)
    skill = rng.normal(0, .001, (4, 200, len(d.entries))) + d.entries.driver_idx.to_numpy() * .1
    post = {"skill": skill, "compat": -2 * skill,
            "car": rng.normal(size=(4, 200, len(d.cars))), "car_track": np.zeros((4, 200, len(d.cars)))}
    monkeypatch.setattr(export, "build_design", lambda *a: d)
    monkeypatch.setattr(export, "load", lambda *a: (post, {"divergences": 0}))
    monkeypatch.setattr(export, "load_meta", lambda *a: {"created_utc": "test", "fingerprint": "test"})
    monkeypatch.setattr(export, "OUT", tmp_path / "ratings")
    monkeypatch.setattr(export, "SNAPSHOTS", tmp_path / "snapshots")
    export.export()
    z = np.load(export.OUT / "current_draws.npz")
    assert z["driver_ids"].dtype.kind == "U"  # no pickle required
    head = pd.read_csv(export.OUT / "pairwise_drivers.csv", index_col=0)
    portable = pd.read_csv(export.OUT / "pairwise_drivers_portable.csv", index_col=0)
    np.testing.assert_allclose(head, export.pairwise(z["in_team_s"]))
    np.testing.assert_allclose(portable, export.pairwise(z["portable_skill_s"]))
    np.testing.assert_allclose(z["skill_s"], z["portable_skill_s"])
    assert not np.allclose(head, portable)
    meta = json.loads((export.OUT / "meta.json").read_text())
    assert meta["pairwise_default"] == "in_team"


def test_racing_folds_do_not_share_files_with_the_qualifying_validation():
    from f1rank.jobs import _base_design, lfo_cutoffs
    validation = {f"{name}.npz" for name in lfo_cutoffs(_base_design(2010))}
    folds = {qualifying.fit_path(S).name for S in range(2012, 2027)}
    assert not folds & validation


def test_sprint_qualifying_variant_trains_on_it_only_up_to_the_cutoff():
    """docs/sprint_qualifying.md: same forecast targets as the published model."""
    from f1rank.evaluate import session_pairs
    from f1rank.jobs import job_design
    main, var = job_design("lfo_mid2024"), job_design("lfosprint_mid2024")
    cut = main.obs.event_idx[main.train].max()
    sq = var.sessions[var.sessions.segment.str.startswith("SQ")]
    assert len(sq) and sq.event_idx.max() <= cut and not main.sessions.segment.str.startswith("SQ").any()
    def after(d):
        p = session_pairs(d)
        return p[p.event_idx > cut][["event_idx", "driver_a", "driver_b", "gap"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(after(main), after(var))
    assert var.obs.y[var.train].size > main.obs.y[main.train].size


def test_cutoff_designs_end_with_their_test_season():
    from f1rank.fit import fingerprint
    from f1rank.jobs import lfo_design
    design, cut = lfo_design("lfo_end2015")
    assert design.events.event_id.iloc[cut].startswith("2015") and design.events.season.max() == 2016
    mid, cut = lfo_design("lfo_mid2024")
    assert mid.events.season.max() == 2024 and mid.events.event_id.iloc[cut] == "2024-08"
    # the racing fold for 2016 is the same design: trained through 2015, ending with 2016
    events = design.events
    from f1rank.design import build_design
    fold = build_design(2010, end_event=str(events[events.season == 2016].event_id.max()))
    fold = fold.with_cutoff(int(fold.events.loc[fold.events.season < 2016, "event_idx"].max()))
    assert fingerprint(fold) == fingerprint(design)
