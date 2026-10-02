"""Scheduled refresh: when it runs, and the inputs it carries between runs."""
import datetime as dt

import pandas as pd

from f1rank.schedule import decide

KNOWN = pd.DataFrame({"event_id": ["2026-09", "2026-10", "2026-15"]})
EVENTS = pd.DataFrame({"event_id": ["2026-15"], "race_name": ["Azerbaijan Grand Prix"],
                       "date": ["2026-09-26"]})
TODAY = dt.date(2026, 9, 28)


def timing(*extracted):
    return lambda event: event in extracted


def test_new_jolpica_result_is_due():
    latest = {"event_id": "2026-16", "race_name": "Singapore Grand Prix", "date": "2026-10-11"}
    result = decide(latest, KNOWN, EVENTS, timing("2026-15"), TODAY)
    assert result["due"] and result["event"] == "2026-16"


def test_rounds_compare_numerically_across_seasons():
    for latest in ("2026-15", "2026-02", "2025-24"):
        result = decide({"event_id": latest, "race_name": "", "date": ""}, KNOWN, EVENTS,
                        timing("2026-15"), TODAY)
        assert not result["due"]
    result = decide({"event_id": "2027-01", "race_name": "", "date": ""}, KNOWN, EVENTS, timing(), TODAY)
    assert result["due"]


def test_missing_race_timing_is_retried_only_shortly_after_the_race():
    assert decide(None, KNOWN, EVENTS, timing(), TODAY)["due"]
    assert not decide(None, KNOWN, EVENTS, timing("2026-15"), TODAY)["due"]
    assert not decide(None, KNOWN, EVENTS, timing(), dt.date(2026, 10, 20))["due"]


def test_cache_archive_round_trips_and_skips_unneeded_files(tmp_path):
    from f1rank import cachestore
    src, dst = tmp_path / "src", tmp_path / "dst"
    for name in ("data/raw/jolpica/2026_results.json", "data/raw/jolpica/dumps/big.csv",
                 "outputs/race_total/cache/2026-07_full.npz", "outputs/race_total/cache/full.failed_0.npz",
                 "outputs/race_total/cache/recovery_1.npz", "outputs/fits/main.npz"):
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        (src / name).write_text(name)
    archive = tmp_path / "cache.tar.gz"
    assert cachestore.pack(archive, src) == 2
    assert cachestore.unpack(archive, dst) == 2
    assert (dst / "outputs/race_total/cache/2026-07_full.npz").read_text() == "outputs/race_total/cache/2026-07_full.npz"
    assert not (dst / "data/raw/jolpica/dumps").exists()


def test_cache_archive_cannot_write_outside_its_paths(tmp_path):
    import tarfile
    import pytest
    from f1rank import cachestore
    archive = tmp_path / "bad.tar.gz"
    (tmp_path / "x").write_text("x")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "x", arcname="outputs/ratings/meta.json")
    with pytest.raises(ValueError, match="Unexpected paths"):
        cachestore.unpack(archive, tmp_path / "dst")


def test_unconverged_asof_fit_is_retried_then_skipped(tmp_path, monkeypatch):
    """A refresh must not fail (and lose the main refit) when one as-of fit does not converge."""
    from f1rank import asof, fit
    monkeypatch.setattr(asof, "OUT", tmp_path)
    monkeypatch.setattr(asof, "event_design", lambda event_id: (object(), 1))
    calls = []
    def fake_fit(design, progress, **settings):
        calls.append(settings)
        return {}, {"elapsed_s": 1.0, "divergences": 0, "mean_steps": 511.0}
    monkeypatch.setattr(fit, "fit", fake_fit)
    # the keys artifacts.diagnostics returns (divergences is also in fit's info)
    monkeypatch.setattr("f1rank.artifacts.diagnostics", lambda post, div: {
        "rhat_max": 1.07, "divergences": div, "n_draws": 4000, "finite": True, "converged": False})
    assert asof.fit_event("2026-15") is None
    assert calls == [asof.SETTINGS, asof.RETRY]
    assert asof.RETRY["samples"] > asof.SETTINGS["samples"] and asof.RETRY["seed"] != 0
    assert not list(tmp_path.iterdir())


def test_asof_record_keeps_the_failed_attempt(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    from f1rank import asof, fit
    events = pd.DataFrame({"event_idx": [0, 1], "event_id": ["2026-14", "2026-15"],
                           "race_name": ["a", "b"], "date": ["2026-09-12", "2026-09-26"]})
    monkeypatch.setattr(asof, "OUT", tmp_path)
    monkeypatch.setattr(asof, "event_design", lambda event_id: (SimpleNamespace(events=events), 1))
    monkeypatch.setattr(asof, "ratings_at", lambda *a: {})
    monkeypatch.setattr(asof, "forecast", lambda *a: {"summary": {}})
    monkeypatch.setattr(fit, "fingerprint", lambda design: "f")
    monkeypatch.setattr(fit, "fit", lambda design, progress, **s: (
        {}, {"elapsed_s": 1.0, "divergences": 0, "mean_steps": 511.0}))
    outcomes = iter([False, True])
    monkeypatch.setattr("f1rank.artifacts.diagnostics", lambda post, div: {
        "rhat_max": 1.03, "divergences": div, "n_draws": 4000, "finite": True, "converged": next(outcomes)})
    rec = json.loads(asof.fit_event("2026-15").read_text())
    assert rec["fit"]["settings"] == asof.RETRY
    assert [a["settings"] for a in rec["fit"]["failed_attempts"]] == [asof.SETTINGS]
    assert rec["trained_through"]["event_id"] == "2026-14"


def test_next_forecast_is_the_seasons_next_race_and_none_after_the_finale():
    from f1rank.forecast import next_event
    race = lambda season, rnd, date: {"season": str(season), "round": str(rnd), "raceName": f"R{rnd}",  # noqa: E731
                                      "date": date, "Circuit": {"circuitId": f"c{rnd}"}}
    schedule = [race(2026, 17, "2026-10-11"), race(2026, 16, "2026-10-04"), race(2026, 15, "2026-09-26")]
    assert next_event(schedule, "2026-15")["event_id"] == "2026-16"
    assert next_event(schedule, "2026-17") is None


def test_forecast_scores_use_established_teammates_and_the_actual_order():
    import numpy as np
    from f1rank.design import build_design
    from f1rank.evaluate import session_pairs
    from f1rank.forecast import SEC, score_record
    design = build_design(2010)
    k = int(design.events.event_idx.max())
    p = session_pairs(design)
    p = p[p.event_idx == k]
    first = p.groupby(["driver_a", "driver_b"]).gap.first() * SEC
    pairs = [dict(a=a, b=b, predicted=round(float(g), 4), q05=round(float(g) - .1, 4), q95=round(float(g) + .1, 4))
             for (a, b), g in first.items()]
    e = design.entries.set_index("entry_idx").driver_id
    o = design.obs[design.obs.event_idx == k]
    q1 = o[o.session_idx == o.session_idx.min()]  # sessions are ordered Q1, Q2, Q3 within an event
    expected = q1.set_index(q1.entry_idx.map(e)).y
    rec = dict(event={"event_id": design.events.event_id.iloc[k]}, trained_through={"event_id": "x"},
               created_utc="t", pairs=pairs, drivers=[dict(id=d, median=float(v)) for d, v in expected.items()])
    s = score_record(design, rec)
    assert s["n_pairs"] >= len(pairs) and s["rmse_zero"] > 0 and 0 < s["coverage90"] <= 1
    assert all(r["segment"] in ("Q1", "Q2", "Q3") for r in s["pairs"])
    assert s["order"][0] == dict(segment="Q1", n=len(q1), spearman=1.0)
    future = dict(rec, event={"event_id": "2099-01"})
    assert score_record(design, future) is None and np.isfinite(s["rmse"])
