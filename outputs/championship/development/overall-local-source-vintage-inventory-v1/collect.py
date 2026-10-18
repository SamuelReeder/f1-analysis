"""Record local provenance-file availability without reading racing observations."""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


def digest(value):
    return hashlib.sha256(value).hexdigest()


def git_bytes(root, revision, path):
    return subprocess.check_output(
        ["git", "show", f"{revision}:{path}"], cwd=root
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--raw-root", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.revision}^{{commit}}"],
        cwd=root, text=True,
    ).strip()
    own_path = Path(__file__).resolve().relative_to(root).as_posix()
    own_bytes = Path(__file__).read_bytes()
    if git_bytes(root, revision, own_path) != own_bytes:
        raise ValueError("collector must match the committed revision")
    source_path = "data/processed/sources.json"
    source_bytes = git_bytes(root, revision, source_path)
    if (root / source_path).read_bytes() != source_bytes:
        raise ValueError("working source metadata differs from the revision")
    sources = json.loads(source_bytes)["jolpica"]
    if not isinstance(sources, dict):
        raise ValueError("expected a filename-keyed source mapping")
    if any(Path(name).name != name for name in sources):
        raise ValueError("source filename must not contain a directory")
    if args.output.exists():
        raise ValueError("output must be a fresh directory")
    snapshots = {}
    observations = []
    observed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    for index, supplied_root in enumerate(args.raw_root):
        raw_root = supplied_root.absolute()
        manifest_path = raw_root / "manifest.json"
        archive_path = raw_root / "archive"
        manifest = None
        manifest_evidence = {"path": str(manifest_path), "exists": manifest_path.exists()}
        if manifest_path.exists():
            if not manifest_path.is_file():
                raise ValueError("manifest path is not a regular file")
            # Read a bounded metadata file only, never the raw response files.
            with manifest_path.open("rb") as handle:
                raw_bytes = handle.read(65537)
            if len(raw_bytes) > 65536:
                raise ValueError("manifest exceeds this metadata-only audit's bound")
            manifest = json.loads(raw_bytes)
            if not isinstance(manifest, dict) or any(
                not isinstance(entry, dict) for entry in manifest.values()
            ):
                raise ValueError("expected a filename-keyed raw manifest")
            snapshot_name = f"manifest-{index}.json"
            snapshots[snapshot_name] = raw_bytes
            manifest_evidence.update(
                sha256=digest(raw_bytes), snapshot=snapshot_name,
                records=len(manifest),
                field_sets=sorted({tuple(sorted(entry)) for entry in manifest.values()}),
            )
        files = []
        for name, source in sorted(sources.items()):
            entry = None if manifest is None else manifest.get(name)
            files.append({
                "name": name,
                "raw_file_present": (raw_root / name).is_file(),
                "processed_sha256": source["sha256"],
                "processed_retrieved_utc": source["retrieved_utc"],
                "processed_retrieved_from_file_time": source.get("retrieved_from_file_time"),
                "raw_manifest_entry": entry,
                "manifest_hash_agrees": None if entry is None else entry.get("sha256") == source["sha256"],
                "raw_file_content_hash_verified": False,
            })
        observations.append({
            "raw_root": str(raw_root), "resolved_raw_root": str(raw_root.resolve()),
            "raw_root_exists": raw_root.exists(), "raw_root_is_directory": raw_root.is_dir(),
            "manifest": manifest_evidence,
            "archive": {"path": str(archive_path), "exists": archive_path.exists(),
                        "is_directory": archive_path.is_dir(), "contents_audited": False},
            "processed_sources": files,
            "counts": {"processed_sources": len(files),
                       "raw_files_present": sum(item["raw_file_present"] for item in files),
                       "raw_manifest_entries_present": sum(item["raw_manifest_entry"] is not None for item in files),
                       "raw_manifest_hashes_agree": sum(item["manifest_hash_agrees"] is True for item in files)},
        })
    record = {
        "kind": "local_source_vintage_metadata_inventory", "observed_at": observed_at,
        "source_revision": revision, "collector": own_path, "collector_sha256": digest(own_bytes),
        "processed_source_manifest": source_path, "processed_source_manifest_sha256": digest(source_bytes),
        "observations": observations,
        "scope": "Explicit local raw roots and their provenance metadata; no raw racing observations or archive contents were read.",
        "historical_source_availability_verified": False, "publication_allowed": False,
        "limitations": [
            "Absence is a local observation at the recorded time, not proof that no copy exists elsewhere.",
            "A raw file's presence or a manifest hash match does not establish availability before a forecast origin.",
            "Copied manifest timestamps and hashes are recorded claims, not independently authenticated retrieval evidence.",
            "This inspection does not audit FastF1 caches, supplementary inputs, provider histories, or official-grid provenance.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    for name, raw_bytes in snapshots.items():
        (args.output / name).write_bytes(raw_bytes)
    (args.output / "inventory.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "observations": len(observations)}))


if __name__ == "__main__":
    main()
