"""Audit the grid-feature correction and reproduce the original synthetic data.

No model fit, held-out predictive score or driver rating is produced. The real
data audit describes which existing training inputs would change; it cannot
establish an effect on predictions or explain the original profile failure.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit():
    from f1rank.overall.design import FIRST_LAP_SEASON, NON_START_OR_EXCLUDED, FINISHED, known_rainfall
    from f1rank.overall.recovery import REFERENCE_CUTOFF, TRUTH_SITES, generate
    from f1rank.overall.fitting import arrays_digest
    from f1rank.overall.provenance import snapshot

    profile = ROOT / "outputs/championship/development/overall-computation-profile-v1"
    parent_path = ROOT / "docs/overall_profile_protocol.json"
    parent = json.loads(parent_path.read_text())
    recorded = json.loads((profile / "summary.json").read_text())
    sources = ["f1rank/overall/model.py", "f1rank/overall/qualifying.py", "f1rank/overall/recovery.py",
               "f1rank/model.py", "f1rank/race_total_model.py"]
    if any(digest(ROOT / name) != recorded["source"]["inputs"][name] for name in sources):
        raise ValueError("Original likelihood or generator changed")
    studies = []
    for name in parent["studies"]:
        spec = parent["synthetic"][name]
        blocks, _, truth = generate(spec["case"], spec["seed"])
        old = json.loads((profile / f"{name}.json").read_text())
        old_truth = json.loads((profile / f"{name}.truth.json").read_text())
        key = arrays_digest(blocks)
        if key != old["receipt"]["specification"]["arrays_key"]:
            raise ValueError("The correction changed the original numerical-profile observations")
        if any(not np.array_equal(truth[k], old_truth["truth"][k]) for k in TRUTH_SITES):
            raise ValueError("Original synthetic truths changed")
        studies.append(dict(study=name, arrays_key=key, arrays_and_truths_identical=True))
    paths = {name:ROOT / "data/processed" / f"{name}.parquet" for name in ("race", "race_weather", "race_laps")}
    race = pd.read_parquet(paths["race"], columns=["event_id", "driver_id", "grid", "status"])
    weather = pd.read_parquet(paths["race_weather"], columns=["event_id", "rainfall"])
    covered_laps = pd.read_parquet(paths["race_laps"], columns=["event_id"])
    race = race[(race.event_id >= f"{FIRST_LAP_SEASON}-01") & (race.event_id <= REFERENCE_CUTOFF)].copy()
    weather = weather[weather.event_id <= REFERENCE_CUTOFF]
    covered = set(covered_laps.event_id) & set(known_rainfall(weather).index)
    race = race[race.event_id.isin(covered)].copy()
    race["entered_field_size"] = race.groupby("event_id").driver_id.transform("size")
    eligible = race[~race.status.str.match(NON_START_OR_EXCLUDED)].copy()
    eligible["result_filtered_field_size"] = eligible.groupby("event_id").driver_id.transform("size")
    changed = eligible[(eligible.grid == 0) & (eligible.entered_field_size != eligible.result_filtered_field_size)].copy()
    changed["used_in_order_likelihood"] = changed.status.str.match(FINISHED)
    records = changed[["event_id", "driver_id", "entered_field_size", "result_filtered_field_size", "used_in_order_likelihood"]].to_dict("records")
    actual_order = changed[changed.used_in_order_likelihood]
    tracked = ["f1rank/overall/design.py", "docs/overall_candidate.md", "analysis/overall_grid_boundary_audit.py",
               "analysis/overall_forecast_catalog.py", "tests/test_overall_grid_boundary.py"]
    return dict(purpose="grid_feature_boundary_audit", cutoff=REFERENCE_CUTOFF,
        source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        source_hashes={name:digest(ROOT / name) for name in tracked}, model_inputs_key=snapshot()["key"],
        original_profile_registration_sha256=digest(parent_path), original_probability_sources_unchanged=True,
        original_synthetic_studies=studies,
        input_sha256={str(p.relative_to(ROOT)):digest(p) for p in paths.values()},
        changed_grid_feature_rows=len(changed), changed_order_likelihood_rows=len(actual_order),
        changed_order_likelihood_events=int(actual_order.event_id.nunique()), affected_rows=records,
        model_fitted=False, predictive_effect_measured=False, publication_allowed=False,
        interpretation="The correction prevents result exclusions from changing pit-lane grid features. Affected historical inputs are listed without refitting or evaluating outcomes. The original registered synthetic arrays and truths are unchanged; their failed fit remains failed. No cause or cure of that failure is claimed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve the previous audit")
    result = audit()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Saved input audit and matched synthetic-array checks; no fit or prediction")
