"""Post hoc numerical diagnosis of every saved profile attempt, never rankings.

Read failed archives directly for diagnosis; they cannot pass the production
posterior loader. Verify the original checker and reproduce the recorded state
diagnostics before examining all current driver contrasts. No refit is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from f1rank.overall.diagnostics import check


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def diagnose(profile_id):
    directory = ROOT / "outputs/championship/development" / profile_id
    summary = json.loads((directory / "summary.json").read_text())
    checker = "f1rank/overall/diagnostics.py"
    if digest(ROOT / checker) != summary["source"]["inputs"][checker]:
        raise ValueError("The registered checker changed; do not silently reclassify old draws")
    results = []
    for spec in summary["studies"]:
        study = spec["study"]
        path = directory / f"{study}.json"
        record = json.loads(path.read_text())
        receipt = record["receipt"]
        if receipt["source"] != summary["source"]:
            raise ValueError("Profile sources disagree")
        if receipt["state"] not in ("converged", "failed_convergence"):
            raise ValueError("This diagnosis requires a terminal sampling result")
        indices = receipt["catalog"]["field_entries"]
        drivers = receipt["catalog"]["field"]
        for number, attempt in enumerate(receipt["attempts"], 1):
            passed = attempt["diagnostics"]["converged"]
            archive = ROOT / "outputs/championship/cache" / profile_id / study / (
                receipt["posterior_file"] if passed else f"failed_{number}.npz")
            archive_digest = digest(archive)
            if passed and archive_digest != receipt["posterior_sha256"]:
                raise ValueError("Saved converged archive changed")
            expected = {p["parameter"]: p for p in attempt["diagnostics"]["parameters"]}
            shapes = (attempt["chains"], attempt["samples"])
            rows = []
            with np.load(archive, allow_pickle=False) as saved:
                for site in ("skill", "overall_race_skill"):
                    values = saved[site]
                    if values.shape[:2] != shapes:
                        raise ValueError("Saved draw axes disagree with the receipt")
                    reproduced = check({site: values}, attempt["diagnostics"]["divergences"])["parameters"][0]
                    for metric in ("rhat_max", "bulk_ess_min", "tail_ess_min", "mcse_sd_max"):
                        if not np.isclose(reproduced[metric], expected[site][metric], rtol=1e-12, atol=1e-12):
                            raise ValueError(f"Saved {site} draws do not reproduce the recorded {metric}")
                    current = values[:, :, indices]
                    for i in range(len(drivers)):
                        for j in range(i):
                            difference = current[:, :, i] - current[:, :, j]
                            diagnostic = check({"contrast": difference}, attempt["diagnostics"]["divergences"])
                            rows.append(dict(site=site, driver_i=drivers[i], driver_j=drivers[j],
                                diagnostic=diagnostic,
                                chain_means=difference.mean(axis=1).tolist(),
                                chain_standard_deviations=difference.std(axis=1, ddof=1).tolist()))
            results.append(dict(study=study, attempt=number,
                original_attempt_converged=passed, archive=str(archive.relative_to(ROOT)),
                archive_sha256=archive_digest, study_record_sha256=digest(path),
                saved_state_diagnostics_reproduced=True, current_driver_contrasts=rows))
    return dict(purpose="posthoc_numerical_diagnosis", profile_id=profile_id,
        profile_summary_sha256=digest(directory / "summary.json"), checker_sha256=digest(ROOT / checker),
        analysis_sha256=digest(__file__), attempts=results, publication_allowed=False,
        refit_performed=False, original_failures_changed=False,
        interpretation="All saved attempts and all current qualifying/race driver contrasts are included. Diagnostics from failed draws are not valid driver estimates, recovery success or ranking evidence. Cases used different registered prior draws, so this comparison cannot isolate the causal effect of teammate-network structure.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="overall-computation-profile-v1")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError("Preserve the earlier diagnosis; use a new named analysis")
    result = diagnose(args.profile)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print("Saved complete post hoc numerical diagnosis; no refit or publication permission")
