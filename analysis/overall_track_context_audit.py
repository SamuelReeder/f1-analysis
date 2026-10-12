"""Rebuild the timeline normally and audit the track-context correction.

No MCMC, driver ranking or revised gate. The normal timeline builder retains
its existing retirement-cause procedure; affected downstream fits remain stale.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from f1rank import timeline
from f1rank.overall.design import FIRST_LAP_SEASON, known_neutralisation, known_rainfall
from analysis.overall_forecast_resources import scope_limits


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def run(baseline):
    limits, begun = scope_limits(), time.monotonic()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    names = ["analysis/overall_track_context_audit.py", "f1rank/timeline.py",
             "f1rank/overall/design.py", "f1rank/overall/recovery.py", "docs/overall_candidate.md"]
    for name in names:
        if subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=ROOT) != (ROOT/name).read_bytes():
            raise ValueError("Commit corrected sources before rebuilding")
    baseline = subprocess.check_output(["git", "rev-parse", baseline], cwd=ROOT, text=True).strip()
    original_bytes = subprocess.check_output(
        ["git", "show", f"{baseline}:data/processed/timeline.parquet"], cwd=ROOT)
    path = ROOT / "data/processed/timeline.parquet"
    if path.read_bytes() != original_bytes:
        raise ValueError("Timeline differs from the specified baseline; preserve and inspect it")
    original = pd.read_parquet(io.BytesIO(original_bytes))
    inputs = [ROOT / "data/processed" / f"{name}.parquet" for name in
              ("race_laps", "race_messages", "race_track_status", "race_weather", "race_classification", "race")]
    inputs += [ROOT / "data/processed/race_sources.json", timeline.OVERRIDES]
    input_hashes = {str(p.relative_to(ROOT)): sha(p) for p in inputs}
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        timeline.main()
    rebuilt = pd.read_parquet(path)
    if input_hashes != {str(p.relative_to(ROOT)): sha(p) for p in inputs}:
        raise ValueError("Timeline inputs changed during regeneration")
    columns = ["lap_start", "lap_end", "detail", "evidence"]
    vsc = []
    events = sorted(set(original.loc[original.kind == "vsc", "event_id"])
                    | set(rebuilt.loc[rebuilt.kind == "vsc", "event_id"]))
    for event in events:
        old = original[(original.event_id == event) & (original.kind == "vsc")][columns]
        new = rebuilt[(rebuilt.event_id == event) & (rebuilt.kind == "vsc")][columns]
        old_records = json.loads(old.to_json(orient="records"))
        new_records = json.loads(new.to_json(orient="records"))
        if old_records != new_records:
            vsc.append(dict(event_id=event, original=old_records, regenerated=new_records))
    weather = known_rainfall(pd.read_parquet(ROOT / "data/processed/race_weather.parquet"))
    raw_status = pd.read_parquet(ROOT / "data/processed/race_track_status.parquet")
    neutral = known_neutralisation(raw_status)
    laps = pd.read_parquet(ROOT / "data/processed/race_laps.parquet", columns=["event_id"])
    eligible = sorted(e for e in set(laps.event_id) & set(weather.index)
                      if e >= f"{FIRST_LAP_SEASON}-01")
    old_neutral = set(original.loc[original.kind.isin(["safety_car", "vsc", "red_flag"]), "event_id"])
    context_changes = [dict(event_id=e, original_neutralised=e in old_neutral,
                           corrected_neutralised=bool(neutral[e]) if e in neutral.index else None)
                       for e in eligible if e not in neutral.index or bool(neutral[e]) != (e in old_neutral)]
    outputs = [path, *(ROOT / "outputs/timeline").glob("*")]
    return dict(source_commit=source, baseline_commit=baseline, limits=limits,
        source_hashes={name: sha(ROOT/name) for name in names}, input_hashes=input_hashes,
        output_hashes={str(p.relative_to(ROOT)): sha(p) for p in outputs if p.is_file()},
        original_timeline_sha256=hashlib.sha256(original_bytes).hexdigest(),
        changed_vsc_events=vsc, changed_vsc_event_count=len(vsc),
        original_timeline_rows=len(original), regenerated_timeline_rows=len(rebuilt),
        outcome_context_changes=context_changes,
        eligible_event_count=len(eligible),
        eligible_events_without_known_track_context=sorted(set(eligible) - set(neutral.index)),
        normal_builder_stdout=log.getvalue(),
        elapsed_seconds=time.monotonic() - begun,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        status_definition_source="https://raw.githubusercontent.com/theOehrly/Fast-F1/v3.8.3/fastf1/_api.py",
        downstream_regeneration_required=True, mcmc_run=False,
        predictive_improvement_established=False, publication_allowed=False)


def compare_laps(baseline):
    """Read-only old/new cleaning comparison on the same cutoff and raw tables."""
    import ast
    from f1rank.overall import design

    limits, begun = scope_limits(), time.monotonic()
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    baseline = subprocess.check_output(["git", "rev-parse", baseline], cwd=ROOT, text=True).strip()
    historical = lambda name: subprocess.check_output(["git", "show", f"{baseline}:{name}"], cwd=ROOT)
    if historical("f1rank/racepace.py") != (ROOT / "f1rank/racepace.py").read_bytes():
        raise ValueError("Other lap-cleaning dependencies changed")
    def leader_ast(text):
        return next(ast.dump(node, include_attributes=False) for node in ast.parse(text).body
                    if isinstance(node, ast.FunctionDef) and node.name == "leader_lap")
    if leader_ast(historical("f1rank/timeline.py")) != leader_ast((ROOT / "f1rank/timeline.py").read_text()):
        raise ValueError("Leader-lap lookup changed")
    names = ["analysis/overall_track_context_audit.py", "f1rank/overall/design.py",
             "f1rank/racepace.py", "f1rank/timeline.py"]
    for name in names:
        if subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=ROOT) != (ROOT/name).read_bytes():
            raise ValueError("Commit impact-audit sources before running")
    module = types.ModuleType("f1rank.overall._original_track_design")
    module.__file__, module.__package__ = str(ROOT / "f1rank/overall/design.py"), "f1rank.overall"
    sys.modules[module.__name__] = module
    try:
        exec(compile(historical("f1rank/overall/design.py"), module.__file__, "exec"), module.__dict__)
        tables = {name: pd.read_parquet(ROOT / "data/processed" / f"{name}.parquet") for name in design.TABLES}
        cutoff = str(tables["events"].event_id.max())
        old_tables = {**tables, "timeline": pd.read_parquet(io.BytesIO(historical("data/processed/timeline.parquet")))}
        before = module.clean_training_laps(module.bounded_tables(old_tables, cutoff))
        after = design.clean_training_laps(design.bounded_tables(tables, cutoff))
    finally:
        del sys.modules[module.__name__]
    keys = ["event_id", "driver_id", "lap_number"]
    if before.duplicated(keys).any() or after.duplicated(keys).any():
        raise ValueError("Duplicate clean-lap identity")
    before, after = before.set_index(keys), after.set_index(keys)
    removed, added = before.index.difference(after.index), after.index.difference(before.index)
    shared = before.index.intersection(after.index)
    change = (before.loc[shared, "y"] - after.loc[shared, "y"]).abs()
    records = lambda index: json.loads(index.to_frame(index=False).to_json(orient="records"))
    return dict(source_commit=source, baseline_commit=baseline, cutoff=cutoff, limits=limits,
        source_hashes={name: sha(ROOT/name) for name in names},
        table_hashes={name: sha(ROOT / "data/processed" / f"{name}.parquet") for name in design.TABLES},
        original_design_sha256=hashlib.sha256(historical("f1rank/overall/design.py")).hexdigest(),
        other_cleaning_dependencies_unchanged=True,
        original_clean_laps=len(before), corrected_clean_laps=len(after),
        original_clean_events=int(before.index.get_level_values("event_id").nunique()),
        corrected_clean_events=int(after.index.get_level_values("event_id").nunique()),
        removed=records(removed), added=records(added),
        shared_laps_with_changed_transformed_pace=int((change > 0).sum()),
        max_absolute_transformed_pace_difference=float(change.max()) if len(change) else None,
        elapsed_seconds=time.monotonic()-begun,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        real_predictions_scored=False, downstream_regeneration_required=True, publication_allowed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--compare-laps", action="store_true", help="Audit cleaning impact without rebuilding outputs")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve the existing audit")
    result = compare_laps(args.baseline) if args.compare_laps else run(args.baseline)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: result[name] for name in
        ("original_clean_laps", "corrected_clean_laps", "elapsed_seconds", "peak_rss_kib")
        if name in result}))
