"""Source revisions must reach a refit without bypassing the last-good release."""
import copy
import hashlib
import json
import subprocess
import sys

import pandas as pd
import pytest

from dashboard import verify_publication as publication
from f1rank import build, dashboard, fetch, schedule


def event(endpoint, rnd=15):
    key = {"qualifying": "QualifyingResults", "results": "Results"}[endpoint]
    rows = []
    for i, driver in enumerate(("alpha", "bravo"), 1):
        row = {"Driver": {"driverId": driver, "code": driver[:3], "givenName": driver,
                          "familyName": "Fixture", "dateOfBirth": "2000-01-01", "nationality": "Test"},
               "Constructor": {"constructorId": "mclaren", "name": "McLaren"}, "position": str(i)}
        if endpoint == "qualifying":
            row.update(Q1=f"1:3{i}.000", Q2=f"1:3{i}.000", Q3=f"1:3{i}.000")
        else:
            row.update(grid=str(i), positionText=str(i), status="Finished", laps="50")
        rows.append(row)
    return {"season": "2026", "round": str(rnd), "raceName": "Synthetic Grand Prix",
            "date": "2026-09-26", "Circuit": {"circuitId": "monza"}, key: rows}


@pytest.fixture
def refreshed(tmp_path, monkeypatch):
    root = tmp_path
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".gitignore").write_text("data/raw/\ndashboard/public/data/\noutputs/dashboard/\n")
    raw = root / "data/raw/jolpica"
    raw.mkdir(parents=True)
    processed = root / "data/processed"
    tables = root / "data/raw/fastf1_tables"
    timing = tables / "2026-15_R/manifest.json"
    timing.parent.mkdir(parents=True)
    timing.write_text("{}")
    for module in (dashboard, build, schedule):
        monkeypatch.setattr(module, "ROOT", root)
    for module in (build, fetch):
        monkeypatch.setattr(module, "RAW", raw)
    monkeypatch.setattr(fetch, "MANIFEST", raw / "manifest.json")
    monkeypatch.setattr(build, "OUT", processed)
    monkeypatch.setattr(build, "SUPPLEMENT", root / "data/supplements/quali_times_fastf1.json")
    monkeypatch.setattr(build, "SPRINT_QUALI", root / "data/supplements/sprint_quali_times.json")
    monkeypatch.setattr(schedule, "PROCESSED", processed)
    monkeypatch.setattr(schedule, "TABLES", tables)
    responses = {endpoint: [event(endpoint)] for endpoint in ("qualifying", "results")}
    monkeypatch.setattr(fetch, "download_table", lambda season, endpoint: copy.deepcopy(responses[endpoint]))
    for endpoint in responses:
        fetch.fetch_table(2026, endpoint, refresh=True)
    (raw / "debuts.json").write_text(json.dumps({d: {"season": 2020, "date": "2020-01-01"}
                                                for d in ("alpha", "bravo")}))
    build.main()
    monkeypatch.setattr(schedule, "latest_jolpica_race", lambda: {
        "event_id": "2026-15", "race_name": "Synthetic Grand Prix", "date": "2026-09-26"})
    payload = {"schema_version": 1, "meta": {"data_as_of": {"event_id": "2026-15"}},
               "overall": {"status": "not established", "standings": []}}
    monkeypatch.setattr(dashboard, "build_payload", lambda: payload)
    directory = root / "dashboard/public/data"
    pointer = dashboard.run(data_dir=directory)
    receipt = {"schema_version": 1, "pointer_sha256": publication.digest(directory / "latest.json"),
               "release_sha256": publication.digest(directory / pointer["url"]),
               "sources": {p: publication.digest(root / p) for p in publication.sources(root)}}
    (directory / "publication.json").write_text(json.dumps(receipt))
    return root, raw, processed, directory, responses


def test_unchanged_completed_season_does_not_run_or_mutate_cache(refreshed):
    _, raw, processed, _, _ = refreshed
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in raw.iterdir()}
    assert schedule.result_revisions("2026-15", processed) == []
    assert schedule.check()["due"] is False
    assert {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in raw.iterdir()} == before


@pytest.mark.parametrize("endpoint,key,value", [
    ("results", "status", "Disqualified"), ("qualifying", "Q1", "1:25.000"),
])
def test_corrected_completed_result_triggers_refresh_without_replacing_cache(refreshed, endpoint, key, value):
    _, raw, _, _, responses = refreshed
    path = raw / f"2026_{endpoint}.json"
    original = path.read_bytes()
    row_key = "Results" if endpoint == "results" else "QualifyingResults"
    responses[endpoint][0][row_key][0][key] = value
    decision = schedule.check()
    assert decision["due"] is True and decision["event"] == "2026-15"
    assert decision["reason"] == f"revised completed-race sources: 2026_{endpoint}.json"
    assert path.read_bytes() == original


def test_row_order_and_future_qualifying_are_not_completed_race_corrections(refreshed):
    *_, responses = refreshed
    responses["results"][0]["Results"].reverse()
    responses["qualifying"].append(event("qualifying", rnd=16))
    assert schedule.check()["due"] is False


def test_new_race_avoids_revision_downloads(refreshed, monkeypatch):
    monkeypatch.setattr(schedule, "latest_jolpica_race", lambda: {
        "event_id": "2026-16", "race_name": "New race", "date": "2026-10-04"})
    monkeypatch.setattr(fetch, "download_table", lambda *a: pytest.fail("Unnecessary revision download"))
    assert schedule.check()["due"] is True


def test_forced_refresh_avoids_revision_downloads(refreshed, monkeypatch):
    monkeypatch.setattr(fetch, "download_table", lambda *a: pytest.fail("Unnecessary revision download"))
    assert schedule.check(force=True)["due"] is True


def test_changed_local_raw_data_cannot_hide_behind_matching_remote_data(refreshed):
    _, raw, processed, _, responses = refreshed
    responses["results"][0]["Results"][0]["status"] = "Disqualified"
    fetch.fetch_table(2026, "results", refresh=True)
    assert schedule.result_revisions("2026-15", processed) == ["2026_results.json"]
    assert schedule.check()["due"] is True


def test_missing_cached_source_requires_regeneration(refreshed):
    _, raw, processed, _, _ = refreshed
    (raw / "2026_results.json").unlink()
    assert schedule.result_revisions("2026-15", processed) == ["2026_results.json"]
    assert schedule.check()["due"] is True


def test_failed_fit_after_corrected_fetch_and_build_preserves_release_and_still_retries(refreshed, monkeypatch):
    _, raw, processed, directory, responses = refreshed
    before = (directory / "latest.json").read_bytes()
    old_status = pd.read_parquet(processed / "race.parquet").status.tolist()
    responses["results"][0]["Results"][0]["status"] = "Disqualified"
    assert schedule.check()["due"] is True
    calls = []
    def execute(command, **kwargs):
        calls.append(command)
        if command[2] == "f1rank.fetch":
            assert "--refresh-latest" in command
            for endpoint in responses:
                fetch.fetch_table(2026, endpoint, refresh=True)
        elif command[2] == "f1rank.build":
            build.main()
        else:
            assert command[2] == "f1rank.fit"
            raise RuntimeError("Synthetic fit failure after corrected data were built")
    # Patch only the runner used by dashboard; subprocess is shared by the git
    # inventory reader, so restore it before checking the next schedule decision.
    with monkeypatch.context() as scoped:
        scoped.setattr(dashboard.subprocess, "run", execute)
        with pytest.raises(RuntimeError, match="Synthetic fit failure"):
            dashboard.run(refresh=True, data_dir=directory)
    assert [c[2] for c in calls] == ["f1rank.fetch", "f1rank.build", "f1rank.fit"]
    assert pd.read_parquet(processed / "race.parquet").status.tolist() != old_status
    assert (directory / "latest.json").read_bytes() == before
    assert json.loads((directory / "status.json").read_text())["state"] == "failed"
    # Cache and processed provenance now match remote. Only the published
    # receipt retains the earlier generation and keeps this refresh due.
    assert schedule.result_revisions("2026-15", processed) == []
    decision = schedule.check()
    assert decision["due"] is True and "published input changed" in decision["reason"]


def test_revision_download_failure_leaves_publication_untouched(refreshed, monkeypatch):
    _, raw, _, directory, _ = refreshed
    original_pointer = (directory / "latest.json").read_bytes()
    original_raw = (raw / "2026_results.json").read_bytes()
    def fail(*args):
        raise RuntimeError("Synthetic API outage")
    monkeypatch.setattr(fetch, "download_table", fail)
    with pytest.raises(RuntimeError, match="API outage"):
        schedule.check()
    assert (directory / "latest.json").read_bytes() == original_pointer
    assert (raw / "2026_results.json").read_bytes() == original_raw


def test_new_input_and_model_contract_changes_trigger_refresh_but_editorial_notes_do_not(refreshed):
    root, _, _, _, _ = refreshed
    docs = root / "docs"
    docs.mkdir()
    (docs / "editorial_report.md").write_text("An editorial correction.\n")
    assert schedule.check()["due"] is False
    (docs / "overall_candidate.md").write_text("Changed model contract.\n")
    assert schedule.check()["due"] is True


def test_missing_or_invalid_published_receipt_is_not_treated_as_unchanged(refreshed):
    _, _, _, directory, _ = refreshed
    path = directory / "publication.json"
    receipt = json.loads(path.read_text())
    receipt["pointer_sha256"] = "wrong"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Invalid published source receipt"):
        schedule.check()
    path.unlink()
    assert schedule.check()["due"] is True


def test_refresh_latest_refetches_processed_season_across_new_year(refreshed, monkeypatch):
    class NewYear(fetch.dt.date):
        @classmethod
        def today(cls):
            return cls(2027, 1, 5)
    monkeypatch.setattr(fetch.dt, "date", NewYear)
    calls = []
    monkeypatch.setattr(fetch, "fetch_table", lambda season, endpoint, refresh: calls.append((season, endpoint, refresh)) or [])
    monkeypatch.setattr(fetch, "fetch_debuts", lambda drivers: {})
    monkeypatch.setattr(sys, "argv", ["fetch", "--first", "2026", "--last", "2027", "--refresh-latest"])
    fetch.main()
    assert {year for year, _, _ in calls} == {2026, 2027}
    assert all(refresh for _, _, refresh in calls)


def test_complete_paginated_download_is_read_only_and_does_not_mutate_responses(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    monkeypatch.setattr(fetch, "PAGE", 1)
    whole = event("results")
    pages = [{"total": "2", "RaceTable": {"Races": [{**whole, "Results": [row]}]}} for row in whole["Results"]]
    original = copy.deepcopy(pages)
    seen = []
    def get(url):
        seen.append(url)
        return pages[len(seen) - 1]
    monkeypatch.setattr(fetch, "_get", get)
    assert fetch.download_table(2026, "results") == [whole]
    assert seen[0].endswith("limit=1&offset=0") and seen[1].endswith("limit=1&offset=1")
    assert pages == original and list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("failure", ["truncated", "changed_count", "duplicate", "changed_metadata"])
def test_incomplete_or_inconsistent_pagination_cannot_replace_raw_cache(tmp_path, monkeypatch, failure):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    monkeypatch.setattr(fetch, "PAGE", 1)
    path = tmp_path / "2026_results.json"
    path.write_text("original complete cache")
    whole = event("results")
    pages = [{"total": "2", "RaceTable": {"Races": [{**whole, "Results": [row]}]}} for row in whole["Results"]]
    if failure == "truncated":
        pages[1]["RaceTable"]["Races"][0]["Results"] = []
    elif failure == "changed_count":
        pages[1]["total"] = "3"
    elif failure == "duplicate":
        pages[1]["RaceTable"]["Races"][0]["Results"] = [whole["Results"][0]]
    else:
        pages[1]["RaceTable"]["Races"][0]["raceName"] = "Changed name"
    page = iter(pages)
    monkeypatch.setattr(fetch, "_get", lambda url: next(page))
    with pytest.raises(ValueError):
        fetch.fetch_table(2026, "results", refresh=True)
    assert path.read_text() == "original complete cache"


def test_raw_revisions_preserve_every_distinct_version_even_at_the_same_time(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    monkeypatch.setattr(fetch, "MANIFEST", tmp_path / "manifest.json")
    monkeypatch.setattr(fetch, "_now", lambda: "2026-01-01T00:00:00Z")
    path = tmp_path / "2026_results.json"
    versions = ['[{"version":"original"}]', '[{"version":"corrected"}]', '[{"version":"revised again"}]']
    for value in versions:
        fetch.save_versioned(path, value)
    archives = sorted((tmp_path / "archive").glob("*.json"))
    assert {p.read_text() for p in archives} == set(versions[:-1])
    assert path.read_text() == versions[-1]
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in archives}
    # A reversion and correction can encounter an already-preserved byte version.
    fetch.save_versioned(path, versions[0])
    fetch.save_versioned(path, versions[1])
    assert all((tmp_path / "archive" / p).read_bytes() == saved[0] and
               (tmp_path / "archive" / p).stat().st_mtime_ns == saved[1] for p, saved in before.items())
    assert len(list((tmp_path / "archive").glob("*.json"))) == len(versions)
    assert not list(tmp_path.rglob("*.tmp"))


@pytest.mark.parametrize("metadata", [None, {"sha256": "unrelated", "retrieved_utc": "old"},
                                      {"retrieved_utc": "unverified"}])
def test_pre_versioning_or_mismatched_cache_is_archived_without_inventing_a_retrieval_date(tmp_path, monkeypatch, metadata):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    monkeypatch.setattr(fetch, "MANIFEST", tmp_path / "manifest.json")
    path = tmp_path / "2026_results.json"
    original = '[{"source":"old"}]'
    path.write_text(original)
    if metadata is not None:
        fetch.MANIFEST.write_text(json.dumps({path.name: metadata}))
    fetch.save_versioned(path, '[{"source":"new"}]')
    sha = hashlib.sha256(original.encode()).hexdigest()
    archive = tmp_path / "archive" / f"2026_results.{sha}.json"
    entry = json.loads(fetch.MANIFEST.read_text())[archive.relative_to(tmp_path).as_posix()]
    assert archive.read_text() == original
    assert entry["sha256"] == sha and entry["retrieved_utc"] is None
    assert entry["retrieval_time_unknown"] is True


def test_failed_raw_replacement_keeps_complete_previous_response_and_preservation_record(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    monkeypatch.setattr(fetch, "MANIFEST", tmp_path / "manifest.json")
    path = tmp_path / "2026_results.json"
    fetch.save_versioned(path, '[{"value":"old"}]')
    original = path.read_bytes()
    replace = fetch.os.replace
    def fail_at_response(source, destination):
        if destination == path:
            raise OSError("Synthetic interrupted replacement")
        return replace(source, destination)
    monkeypatch.setattr(fetch.os, "replace", fail_at_response)
    with pytest.raises(OSError, match="interrupted"):
        fetch.save_versioned(path, '[{"value":"new"}]')
    assert path.read_bytes() == original
    archives = list((tmp_path / "archive").glob("*.json"))
    assert len(archives) == 1 and archives[0].read_bytes() == original
    manifest = json.loads(fetch.MANIFEST.read_text())
    assert manifest[path.name]["sha256"] == hashlib.sha256(original).hexdigest()
    assert archives[0].relative_to(tmp_path).as_posix() in manifest
    assert not list(tmp_path.rglob("*.tmp"))


def test_corrupt_immutable_raw_archive_is_never_overwritten(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch, "RAW", tmp_path)
    monkeypatch.setattr(fetch, "MANIFEST", tmp_path / "manifest.json")
    path = tmp_path / "2026_results.json"
    original = '[{"value":"old"}]'
    fetch.save_versioned(path, original)
    archive = tmp_path / "archive" / f"2026_results.{hashlib.sha256(original.encode()).hexdigest()}.json"
    archive.parent.mkdir()
    archive.write_text("corrupt archive")
    with pytest.raises(ValueError, match="immutable raw archive is corrupt"):
        fetch.save_versioned(path, '[{"value":"new"}]')
    assert path.read_text() == original and archive.read_text() == "corrupt archive"


def test_cache_restore_retains_revision_history_and_excludes_interrupted_temporary_files(refreshed, tmp_path):
    from f1rank import cachestore
    root, raw, _, _, _ = refreshed
    original = (raw / "2026_results.json").read_bytes()
    fetch.save_versioned(raw / "2026_results.json", '[{"corrected":"fixture"}]')
    abandoned = raw / "2026_results.json.interrupted.tmp"
    abandoned.write_text("incomplete response")
    archive = tmp_path / "refresh-cache.tar.gz"
    cachestore.pack(archive, root)
    destination = tmp_path / "restored"
    cachestore.unpack(archive, destination)
    restored = destination / "data/raw/jolpica"
    sha = hashlib.sha256(original).hexdigest()
    saved = restored / "archive" / f"2026_results.{sha}.json"
    assert saved.read_bytes() == original
    assert (restored / "2026_results.json").read_text() == '[{"corrected":"fixture"}]'
    assert (restored / "manifest.json").read_bytes() == (raw / "manifest.json").read_bytes()
    assert not (restored / abandoned.name).exists()
