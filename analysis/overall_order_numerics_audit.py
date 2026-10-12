"""Verify corrected source preserves the registered synthetic inputs.

No fit, saved-draw analysis, real-driver prediction or acceptance decision.
The old failure is preserved; this audit does not establish its cause.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def audit():
    import numpy as np
    from f1rank.overall.fitting import arrays_digest
    from f1rank.overall.provenance import snapshot
    from f1rank.overall.recovery import generate, TRUTH_SITES

    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    tracked = ["f1rank/design.py", "f1rank/overall/diagnostics.py",
        "f1rank/overall/model.py", "f1rank/overall/order.py", "f1rank/overall/recovery.py",
        "f1rank/overall/simulation.py",
        "analysis/overall_observation_kernels.py", "analysis/overall_order_numerics_audit.py",
        "docs/overall_candidate.md", "docs/overall_rotation_profile_proposal.json"]
    for name in tracked:
        if subprocess.check_output(["git", "show", f"{source}:{name}"], cwd=ROOT) != (ROOT / name).read_bytes():
            raise ValueError("Commit correction and audit inputs before running")
    proposal = json.loads((ROOT / "docs/overall_rotation_profile_proposal.json").read_text())
    parent_path = ROOT / proposal["parent_registration"]
    parent = json.loads(parent_path.read_text())
    if sha(parent_path) != proposal["parent_registration_sha256"]:
        raise ValueError("The original registration changed")
    for name in ("studies", "synthetic", "fit_settings", "retry", "diagnostics"):
        if parent[name] != proposal[name]:
            raise ValueError("The original numerical specification changed")
    inputs = snapshot()
    if inputs["key"] != proposal["registered_model_inputs_key"]:
        raise ValueError("Pending proposal does not identify the corrected source")
    profile = ROOT / "outputs/championship/development/overall-computation-profile-v1"
    summary = json.loads((profile / "summary.json").read_text())
    studies = []
    for study in parent["studies"]:
        spec = parent["synthetic"][study]
        blocks, _, truth = generate(spec["case"], spec["seed"])
        original_path = profile / f"{study}.truth.json"
        original = json.loads(original_path.read_text())
        key = arrays_digest(blocks)
        if key != original["arrays_key"] or key != proposal["original_arrays_sha256"][study]:
            raise ValueError("Arithmetic correction changed the original synthetic inputs")
        if any(not np.array_equal(truth[name], original["truth"][name]) for name in TRUTH_SITES):
            raise ValueError("Arithmetic correction changed original generating truths")
        studies.append(dict(study=study, seed=spec["seed"], arrays_key=key,
            original_truth_sha256=sha(original_path), arrays_and_truths_identical=True))
    return dict(purpose="numerical_correction_input_audit", source_commit=source,
        recorded_at=dt.datetime.now(dt.timezone.utc).isoformat(), model_inputs_key=inputs["key"],
        source_hashes={name:sha(ROOT / name) for name in tracked},
        parent_registration_sha256=sha(parent_path), studies=studies,
        original_summary_sha256=sha(profile / "summary.json"),
        original_study_states=summary["studies"], original_all_converged=summary["all_converged"],
        pending_proposal_status=proposal["status"], pending_proposal_user_approved=proposal["user_approved"],
        causal_explanation_established=False, mcmc_run=False, publication_allowed=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Preserve the recorded audit")
    result = audit()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key:result[key] for key in ("source_commit", "studies", "original_all_converged")}))
