"""Record a fully checked publication locally; verify its portable receipt in CI.

The receipt binds the immutable payload to its committed code, data and summaries.
CI needs no ignored posterior caches and never reconstructs their evidence.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = "outputs/analysis/championship_preview"
PREVIEW_FILES = ("summary.json", "standings.csv", "manifest.json")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources(root):
    files = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard",
         "--", "data", "f1rank", "extract", "outputs", "dashboard/verify_publication.py"], cwd=root,
    ).decode().split("\0")
    return sorted({p for p in files if p and not p.startswith("outputs/analysis/")
                   and p != "outputs/REPORT.md" and not p.endswith(".log")})


def publication(root):
    directory = root / "dashboard/public/data"
    pointer = json.loads((directory / "latest.json").read_text())
    release = pointer.get("release", "")
    if pointer.get("schema_version") != 1 or not re.fullmatch(r"[0-9a-f]{20}", release):
        raise ValueError("Invalid publication pointer")
    if pointer.get("url") != f"releases/{release}.json":
        raise ValueError("Invalid publication URL")
    path = directory / pointer["url"]
    payload = json.loads(path.read_text())
    encoded = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(encoded).hexdigest()[:20] != release:
        raise ValueError("Immutable publication content changed")
    if payload.get("schema_version") != 1 or pointer.get("data_as_of") != payload["meta"]["data_as_of"]:
        raise ValueError("Publication metadata disagrees with its pointer")
    return directory, path, payload


def preview_payload(root):
    """The exploratory equal-car preview (docs/championship_preview.md) as the dashboard shows it.

    Built only from committed preview outputs whose hashes match their manifest, so CI can
    rebuild it without posterior files. It is never presented as a validated ranking.
    """
    import pandas as pd
    directory = root / PREVIEW
    manifest = json.loads((directory / "manifest.json").read_text())
    for name in PREVIEW_FILES[:2]:
        if manifest["outputs"].get(f"{PREVIEW}/{name}") != digest(directory / name):
            raise ValueError(f"Preview output {name} disagrees with its manifest")
    summary = json.loads((directory / "summary.json").read_text())
    if summary.get("gate") is not False or summary.get("production_publication_authorized") is not False:
        raise ValueError("The preview must remain marked as unvalidated")
    table = pd.read_csv(directory / "standings.csv", dtype={"driver_id": str, "name": str})
    table = table.loc[table.version == "in_team"].sort_values("points_per_race", ascending=False)
    if (table.empty or table.driver_id.duplicated().any() or table[["driver_id", "name"]].isna().any().any()
            or not table.p_title.between(0, 1).all() or (table.points_per_race < 0).any()
            or (table.rank_lo < 1).any() or (table.rank_hi > len(table)).any()
            or (table.rank_lo > table.rank_median).any() or (table.rank_median > table.rank_hi).any()):
        raise ValueError("Invalid preview standings")
    events = pd.read_parquet(root / "data/processed/events.parquet").set_index("event_id")
    event = events.loc[summary["data_as_of"]]
    combined = summary["original_combined_validation"]
    return {
        "schema_version": 1, "model": manifest["model"], "validated": False, "status": summary["status"],
        "fit_id": manifest["fit_id"], "recorded_at": manifest["generated_at"],
        "data_as_of": {"event_id": summary["data_as_of"], "race_name": str(event.race_name),
                       "date": str(pd.Timestamp(event.date).date())},
        "scope": summary["scope"], "headline": summary["headline"],
        "components": ["qualifying_pace", *summary["fixed_candidate"]],
        "simulated_seasons": summary["simulated_seasons"], "n_races_simulated": summary["n_races_simulated"],
        "driver_error_rates_used": summary["driver_error_rates_used"],
        "p_retire_common": summary["p_retire_common"],
        "combined_validation": {k: combined[k] for k in ("n_races", "seasons", "mean_diff_per_race", "ci95", "gate")},
        "archive_failures": [{"study": k, "exception": v["exception"]} for k, v in summary["archive_failures"].items()],
        "race_pace_diagnostics": summary["full_pace_diagnostics"],
        "race_stage_retries": len(summary["full_race_stage_fit_attempts"]["retried"]),
        "sources": {f"{PREVIEW}/{name}": digest(directory / name) for name in PREVIEW_FILES},
        "standings": [{"driver_id": r.driver_id, "name": r.name, "points_per_race": round(float(r.points_per_race), 6),
                       "p_title": float(r.p_title), "rank_median": float(r.rank_median),
                       "rank_lo": int(r.rank_lo), "rank_hi": int(r.rank_hi)} for r in table.itertuples()],
    }


def record_preview(root):
    """Rewrite the preview where its full provenance is current; otherwise keep the committed
    copy, which still shows its own data cutoff, as long as it matches its committed outputs."""
    sys.path.insert(0, str(root))
    from f1rank.artifacts import StaleArtifact, require
    target = root / "dashboard/public/data/overall_preview.json"
    directory = root / PREVIEW
    try:
        require(directory, required_outputs=[directory / n for n in PREVIEW_FILES[:2]])
    except StaleArtifact as exc:
        if not target.exists():
            print(f"No overall preview published: {exc}")
            return
        print("Keeping the recorded overall preview; its inputs have changed since it was made")
    else:
        target.write_text(json.dumps(preview_payload(root), indent=1, allow_nan=False) + "\n")


def verify_preview(root, receipt):
    target = root / "dashboard/public/data/overall_preview.json"
    if not target.exists():
        if receipt.get("overall_preview_sha256") is not None:
            raise ValueError("The recorded overall preview is missing")
        return
    if receipt.get("overall_preview_sha256") != digest(target):
        raise ValueError("Overall preview changed after its receipt was recorded")
    if json.loads(target.read_text()) != preview_payload(root):
        raise ValueError("Overall preview disagrees with its committed outputs")


def record(root=ROOT):
    directory, path, payload = publication(root)
    # This step runs where full provenance, including posterior files, is available.
    sys.path.insert(0, str(root))
    from f1rank.dashboard import build_payload
    if payload != build_payload():
        raise ValueError("Publish the fully checked current outputs before recording a receipt")
    record_preview(root)
    preview = directory / "overall_preview.json"
    receipt = {"schema_version": 1, "pointer_sha256": digest(directory / "latest.json"),
               "release_sha256": digest(path), "sources": {p: digest(root / p) for p in sources(root)},
               "overall_preview_sha256": digest(preview) if preview.exists() else None}
    (directory / "publication.json").write_text(json.dumps(receipt, indent=2) + "\n")
    verify(root)


def verify(root=ROOT):
    directory, path, _ = publication(root)
    receipt = json.loads((directory / "publication.json").read_text())
    if receipt.get("schema_version") != 1:
        raise ValueError("Unsupported publication receipt")
    if receipt.get("pointer_sha256") != digest(directory / "latest.json") or receipt.get("release_sha256") != digest(path):
        raise ValueError("Publication changed after its receipt was recorded")
    expected = receipt.get("sources")
    if not isinstance(expected, dict) or sorted(expected) != sources(root):
        raise ValueError("Publication source inventory changed; republish and record a new receipt")
    for source, fingerprint in expected.items():
        if digest(root / source) != fingerprint:
            raise ValueError(f"Publication source changed: {source}; republish and record a new receipt")
    verify_preview(root, receipt)


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        record()
    elif not sys.argv[1:]:
        verify()
    else:
        raise SystemExit("Usage: verify_publication.py [record]")
    print("Checked portable dashboard publication")
