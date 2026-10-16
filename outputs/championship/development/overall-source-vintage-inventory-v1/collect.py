"""Inventory committed provenance metadata without loading racing observations.

This is a report collector, not an availability verifier or acceptance gate.
Run from the repository root with a committed revision and a fresh output path.
"""

import argparse
import datetime as dt
import hashlib
import json
import subprocess
from pathlib import Path


MANIFEST_GROUPS = {
    "data/processed/sources.json": ("jolpica",),
    "data/processed/race_sources.json": ("races", "sprints", "telemetry"),
    "data/processed/practice_sources.json": ("sessions",),
}
GENERATORS = (
    "f1rank/fetch.py", "f1rank/build.py", "f1rank/racedata.py",
    "f1rank/race_features.py", "extract/race_extract.py",
    "extract/practice.py", "extract/telemetry.py",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", args.revision + "^{commit}"], text=True
    ).strip()

    def committed(path):
        return subprocess.check_output(["git", "show", revision + ":" + path])

    collector = Path(__file__).resolve().relative_to(Path.cwd().resolve()).as_posix()
    assert committed(collector) == Path(__file__).read_bytes(), "uncommitted collector"
    inputs = {p: committed(p) for p in (*MANIFEST_GROUPS, *GENERATORS, collector)}
    manifests = []
    for path, groups in MANIFEST_GROUPS.items():
        manifest = json.loads(inputs[path])
        result = {
            "path": path,
            "top_level_keys": sorted(manifest),
            "groups": {},
            "other_top_level_metadata": {
                k: v for k, v in manifest.items() if k not in groups
            },
        }
        for group in groups:
            records = manifest[group]
            timestamps = {}
            flags = {"true": [], "false": [], "absent": []}
            missing = []
            for name, record in sorted(records.items()):
                value = record.get("retrieved_utc")
                if value is None:
                    missing.append(name)
                else:
                    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
                    assert parsed.utcoffset() is not None, (path, group, name)
                    normalized = parsed.astimezone(dt.timezone.utc).isoformat()
                    timestamps[name] = normalized
                flag = record.get("retrieved_from_file_time")
                assert flag is None or type(flag) is bool, (path, group, name)
                assert flag is not None or "retrieved_from_file_time" not in record
                flags["absent" if flag is None else str(flag).lower()].append(name)
            result["groups"][group] = {
                "record_count": len(records),
                "record_fields": sorted({k for r in records.values() for k in r}),
                "retrieved_utc_by_record": timestamps,
                "earliest_recorded_retrieval": min(timestamps.values(), default=None),
                "latest_recorded_retrieval": max(timestamps.values(), default=None),
                "missing_retrieval_records": missing,
                "retrieved_from_file_time_records": flags,
                "file_time_flag_counts": {k: len(v) for k, v in flags.items()},
            }
        manifests.append(result)

    report = {
        "source_revision": revision,
        "input_sha256": {p: hashlib.sha256(b).hexdigest() for p, b in inputs.items()},
        "scope": "Committed processed provenance manifests and their generator source only.",
        "manifests": manifests,
        "raw_content_hashes_reproduced": False,
        "processed_observations_read": False,
        "forecast_origins_evaluated": [],
        "historical_forecast_availability_verified": False,
        "publication_allowed": False,
        "limits": [
            "Recorded retrieval is not evidence of availability before that retrieval.",
            "File modification times are explicitly distinguished from other recorded times.",
            "An absent file-time flag is not independent proof of a retrieval or publication time.",
            "Raw and archived source bytes, provider publication histories and event-specific origins were not audited.",
            "This inventory neither establishes leakage nor verifies an as-of historical replay.",
        ],
    }
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
