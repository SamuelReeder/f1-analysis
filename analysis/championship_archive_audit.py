"""Audit earlier archive availability and code history without reading race outcomes.

Only identifiers and table membership are read from the lap dump. No lap times,
positions, classification outcomes, or model scores are inspected here.
"""
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/championship_archive_audit.json"


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def benchmark_history():
    path = "f1rank/benchmark.py"
    revisions = git("log", "--all", "--format=%H", "--", path).decode().splitlines()
    records = []
    for revision in revisions:
        source = git("show", f"{revision}:{path}").decode()
        values = {}
        for node in ast.parse(source).body:
            if not isinstance(node, ast.Assign) or len(node.targets) != 1:
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            target = node.targets[0]
            if isinstance(target, ast.Name):
                values[target.id] = value
            elif isinstance(target, ast.Tuple) and isinstance(value, tuple):
                values.update({k.id: v for k, v in zip(target.elts, value) if isinstance(k, ast.Name)})
        records.append({"revision": revision, "first_season": values.get("FIRST_SEASON"),
                        "filters_before_first_season": ">= FIRST_SEASON" in source,
                        "source_sha256": hashlib.sha256(source.encode()).hexdigest()})
    return records


def main():
    history = benchmark_history()
    championship_history = []
    for revision in git("log", "--all", "--format=%H", "--", "f1rank/championship.py").decode().splitlines():
        source = git("show", f"{revision}:f1rank/championship.py").decode()
        tree = ast.parse(source)
        imported = any(isinstance(n, ast.ImportFrom) and n.module == "benchmark"
                       and any(a.name == "race_orders" for a in n.names) for n in tree.body)
        direct_race_read = any(isinstance(n, ast.Constant) and n.value == "race.parquet"
                               for n in ast.walk(tree))
        championship_history.append({"revision": revision, "uses_benchmark_race_orders": imported,
                                     "direct_race_parquet_read": direct_race_read,
                                     "source_sha256": hashlib.sha256(source.encode()).hexdigest()})
    first_racing = min(r["first_season"] for r in history)
    events = pd.read_parquet(ROOT / "data/processed/events.parquet", columns=["event_id", "season", "round"])
    archive_events = events[events.season < first_racing]
    dump_dir = ROOT / "data/raw/jolpica/dumps"
    path = sorted(dump_dir.glob("*.zip"))[-1]
    manifest = json.loads((dump_dir / "manifest.json").read_text())[path.name]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != manifest["sha256"] or digest != manifest["published_sha256"]:
        raise ValueError("Archive dump no longer matches its recorded published hash")
    with zipfile.ZipFile(path) as z:
        def read(name, columns):
            return pd.read_csv(z.open(f"formula_one_{name}.csv"), usecols=columns)

        season = read("season", ["id", "year"]).rename(columns={"id": "season_id"})
        rounds = read("round", ["id", "number", "season_id"]).rename(columns={"id": "round_id"})
        session = read("session", ["id", "round_id", "type"]).rename(columns={"id": "session_id"})
        races = session[session.type == "R"].merge(rounds, on="round_id").merge(season, on="season_id")
        races = races.merge(archive_events, left_on=["year", "number"], right_on=["season", "round"])
        entries = read("sessionentry", ["id", "session_id"]).merge(races[["session_id", "event_id"]], on="session_id")
        mapping = entries.set_index("id").event_id
        counts = Counter()
        with z.open("formula_one_lap.csv") as f:
            for chunk in pd.read_csv(f, usecols=["session_entry_id"], chunksize=100000):
                counts.update(chunk.session_entry_id.map(mapping).dropna().value_counts().to_dict())
    qualifying = pd.read_parquet(ROOT / "data/processed/quali_times.parquet", columns=["event_id"])
    q_events = set(qualifying.event_id)
    result = {
        "source_revision": git("rev-parse", "HEAD").decode().strip(),
        "audit_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dump": str(path.relative_to(ROOT)), "dump_sha256": digest,
        "dump_hash_verified": True,
        "archive_event_ids": sorted(archive_events.event_id.tolist()),
        "lap_rows_by_event": dict(sorted(counts.items())),
        "missing_lap_events": sorted(set(archive_events.event_id) - counts.keys()),
        "missing_qualifying_events": sorted(set(archive_events.event_id) - q_events),
        "benchmark_history": history,
        "championship_history": championship_history,
        "all_recorded_benchmarks_exclude_archive": all(
            r["first_season"] >= first_racing and r["filters_before_first_season"] for r in history),
        "scope": "Availability and recorded-code audit only; no historical target values or scores inspected",
        "prior_use_limit": (
            "Qualifying observations from this era were used by the earlier data-window comparison. "
            "Recorded benchmark code excludes these race outcomes. This does not establish what "
            "humans may have inspected outside the repository."),
        "next_action": (
            "Assess an explicitly separate historical validation of the fixed candidate, with all "
            "choices committed before outcomes are scored. Keep the prospective registration intact."),
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print("Earlier archive metadata audited; outcome values were not inspected", flush=True)


if __name__ == "__main__":
    main()
