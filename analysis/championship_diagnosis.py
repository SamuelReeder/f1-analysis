"""Replay the unchanged combined test, recording exploratory diagnostic evidence.

Run as a memory-capped systemd user service: python -m analysis.championship_diagnosis
See docs/championship_followup.md. Never writes canonical racing or dashboard files.
"""
import datetime as dt
import json
from pathlib import Path
import subprocess
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from f1rank import artifacts, championship as c

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/championship_diagnosis"
RTOL = 1e-6
ATOL = 1e-6


def assert_same(actual, expected, path="result"):
    """Require identical decisions and numerically matching replayed evidence."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"{path}: different fields")
        for key in expected:
            assert_same(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"{path}: different list")
        for i, (a, e) in enumerate(zip(actual, expected)):
            assert_same(a, e, f"{path}[{i}]")
    elif isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if type(actual) is not type(expected) or actual != expected:
            raise AssertionError(f"{path}: {actual!r} != {expected!r}")
    else:
        np.testing.assert_allclose(actual, expected, rtol=RTOL, atol=ATOL, err_msg=path)


def score_key(season, names):
    return f"{season}:" + ",".join(names)


def coefficient_summary(post, names):
    result = {}
    for key, values in post.items():
        values = np.asarray(values)
        columns = names if key == "b_quality" else [key]
        for name, v in zip(columns, values.reshape(len(values), -1).T):
            label = f"b_quality:{name}" if key == "b_quality" else name
            result[label] = {
                "mean": float(v.mean()), "sd": float(v.std()),
                "q05_q50_q95": np.quantile(v, [.05, .5, .95]).tolist(),
                "p_positive": float((v > 0).mean()),
            }
    return result


def comparisons(scores, events, selections):
    differences = {}
    rows = []
    for label in ("original_selection", "fixed_race_specific_pace_exploratory"):
        diffs = {}
        for year, chosen in selections.items():
            season = int(year)
            names = tuple(chosen) if label == "original_selection" else ("race_specific_pace",)
            d = (np.asarray(scores[score_key(season, names)], dtype=np.float32)
                 - np.asarray(scores[score_key(season, ())], dtype=np.float32))
            diffs[season] = d
            rows.append({"comparison": label, "season": season,
                         "qualities": list(names), "n_races": len(d),
                         "sum_diff": float(d.sum()), "mean_diff_per_race": float(d.mean()),
                         "races_improved": int((d > 0).sum()),
                         "race_differences": dict(zip(events[season], d.astype(float).tolist()))})
        differences[label] = {
            "descriptive_interval": c.paired_seasons(diffs),
            "leave_one_season_out_descriptive_only": {
                str(s): c.paired_seasons({y: d for y, d in diffs.items() if y != s}) for s in diffs},
        }
    return {"comparisons": differences, "by_season": rows}


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic evidence: {OUT}")
    started = time.monotonic()
    meta = artifacts.require(c.OUT)
    reference = json.loads((c.OUT / "summary.json").read_text())
    effects = {}
    for name in c.QUALITIES:
        try:
            effects[name] = c.quality_draws(name)
        except FileNotFoundError:
            if "not_run" not in reference["entry_tests"].get(name, {}):
                raise
    manifest_path = c.OUT / "manifest.json"
    before = artifacts.digest(manifest_path)
    provenance = {"source_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "championship_manifest_sha256": before,
        "analysis_sha256": artifacts.digest(Path(__file__)),
        "plan_sha256": artifacts.digest(ROOT / "docs/championship_followup.md"),
        "input_hashes": meta["inputs"], "rtol": RTOL, "atol": ATOL,
        "evidence": "exploratory diagnosis of an already completed experiment"}
    OUT.mkdir(parents=True)
    artifacts.atomic_json(OUT / "provenance.json", provenance)
    scores, fits, events, coverage = {}, {}, {}, {}
    context = []
    original_nested, original_fit, original_orders = c.nested_selection, c.fit_race, c.race_orders

    def record_orders(season):
        frame = original_orders(season)
        finishers = c.finishers(frame)
        train = finishers[(finishers.season >= c.ENTRY_FIRST_SEASON) & (finishers.season < season)]
        test = finishers[finishers.season == season]
        events[season] = sorted(test.event_id.unique().tolist())
        row = {"train_races": int(train.event_id.nunique()), "test_races": len(events[season]),
               "train_finishers": len(train), "test_finishers": len(test),
               "qualities": {}}
        for name in effects:
            drivers, draws = effects[name][season]
            means, sd = pd.Series(draws.mean(0), index=drivers), pd.Series(draws.std(0), index=drivers)
            row["qualities"][name] = {}
            for label, data in (("train", train), ("test", test)):
                mapped = means.reindex(data.driver_id)
                row["qualities"][name][label] = {
                    "drivers": int(data.driver_id.nunique()), "known_rows": int(mapped.notna().sum()),
                    "total_rows": len(data), "nonzero_mean_rows": int((mapped.fillna(0) != 0).sum()),
                    "posterior_mean_spread_across_rows": float(mapped.fillna(0).std()),
                    "average_posterior_sd": float(sd.reindex(data.driver_id).mean()),
                }
        coverage[str(season)] = row
        return frame

    def record_fit(frame, *args, **kwargs):
        t = time.monotonic()
        post = original_fit(frame, *args, **kwargs)
        season, names = context[-1]
        fits[score_key(season, names)] = {
            "elapsed_seconds": time.monotonic() - t,
            "coefficients": coefficient_summary(post, names),
            "attempts": artifacts._FITS[-1],
        }
        return post

    def record_nested(candidates, seasons, score):
        def recorded_score(season, names):
            context.append((season, names))
            try:
                values = score(season, names)
            finally:
                context.pop()
            key = score_key(season, names)
            if key not in scores:
                if values.shape != (len(events[season]),) or not np.isfinite(values).all():
                    raise ValueError(f"Invalid or unaligned scores for {key}")
                scores[key] = values.astype(float).tolist()
                with (OUT / "evaluations.jsonl").open("a") as f:
                    f.write(json.dumps({"season": season, "qualities": list(names),
                                        "event_ids": events[season], "log_predictive_density": scores[key],
                                        "fit": fits[key]}, allow_nan=False) + "\n")
            return values

        result = original_nested(candidates, seasons, recorded_score)
        # Diagnostic only: score the predeclared fixed candidate on the same outer years.
        for year in result[2]["selected_by_outer_season"]:
            recorded_score(int(year), ("race_specific_pace",))
        return result

    with patch.object(c, "nested_selection", record_nested), patch.object(c, "fit_race", record_fit), \
            patch.object(c, "race_orders", record_orders):
        selected, entry, combined = c.combined_entry_tests(effects, int(meta["data_as_of"][:4]))
    assert_same(selected, reference["qualities_selected"], "qualities_selected")
    assert_same(entry, {n: reference["entry_tests"][n] for n in effects}, "entry_tests")
    assert_same(combined, reference["combined_validation"], "combined_validation")
    artifacts.require(c.OUT)
    if artifacts.digest(manifest_path) != before:
        raise AssertionError("Canonical championship manifest changed during replay")
    output = {"provenance": provenance, "replay_matches_reference": True,
              "elapsed_seconds": time.monotonic() - started,
              "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
              "qualities_selected": selected, "entry_tests": entry, "combined_validation": combined,
              "fit_attempts": artifacts.fit_record(), "coverage": coverage,
              **comparisons(scores, events, combined["selected_by_outer_season"])}
    artifacts.atomic_json(OUT / "summary.json", output)
    print("Diagnostic replay complete; reference matched; results are exploratory", flush=True)


if __name__ == "__main__":
    main()
