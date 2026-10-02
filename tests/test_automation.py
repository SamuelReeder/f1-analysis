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
