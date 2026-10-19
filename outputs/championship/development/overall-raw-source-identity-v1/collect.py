"""Check bounded local raw-file byte identities against committed provenance."""

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess


MAX_FILES = 128
MAX_FILE_BYTES = 1024 * 1024
MAX_TOTAL_BYTES = 16 * 1024 * 1024
CHUNK_BYTES = 64 * 1024


def git_bytes(root, revision, path):
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=root)


def identity(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path.cwd().resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "--verify", f"{args.revision}^{{commit}}"],
        cwd=root, text=True,
    ).strip()
    collector = Path(__file__).resolve().relative_to(root).as_posix()
    collector_bytes = Path(__file__).read_bytes()
    if git_bytes(root, revision, collector) != collector_bytes:
        raise ValueError("collector must match the committed revision")
    source_path = "data/processed/sources.json"
    source_bytes = git_bytes(root, revision, source_path)
    if (root / source_path).read_bytes() != source_bytes:
        raise ValueError("working source metadata differs from the revision")
    sources = json.loads(source_bytes)["jolpica"]
    if not isinstance(sources, dict) or not 0 < len(sources) <= MAX_FILES:
        raise ValueError("expected a bounded nonempty filename-keyed source mapping")
    for name, source in sources.items():
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("source filename must not contain a directory")
        expected = source["sha256"]
        if not isinstance(expected, str) or len(expected) != 64 or any(
            ch not in "0123456789abcdef" for ch in expected
        ):
            raise ValueError("source must contain a SHA-256 hex digest")
    if args.output.exists():
        raise ValueError("output must be a fresh directory")
    raw_root = args.raw_root.resolve(strict=True)
    initial = {name: (raw_root / name).lstat() for name in sources}
    if any(not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES
           for info in initial.values()):
        raise ValueError("all inputs must be bounded regular files, without symlinks")
    if sum(info.st_size for info in initial.values()) > MAX_TOTAL_BYTES:
        raise ValueError("raw inputs exceed the total byte bound")

    started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    files = []
    total_bytes = 0
    for name, source in sorted(sources.items()):
        path = raw_root / name
        digest = hashlib.sha256()
        size = 0
        with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as handle:
            before = os.fstat(handle.fileno())
            if identity(before) != identity(initial[name]):
                raise ValueError(f"input changed before reading: {name}")
            while chunk := handle.read(CHUNK_BYTES):
                size += len(chunk)
                total_bytes += len(chunk)
                if size > MAX_FILE_BYTES or total_bytes > MAX_TOTAL_BYTES:
                    raise ValueError("raw input grew beyond a byte bound")
                digest.update(chunk)
            if identity(os.fstat(handle.fileno())) != identity(before):
                raise ValueError(f"input changed while reading: {name}")
        if identity(path.lstat()) != identity(before) or size != before.st_size:
            raise ValueError(f"input changed during collection: {name}")
        observed = digest.hexdigest()
        files.append({
            "name": name, "bytes_read": size,
            "expected_sha256": source["sha256"], "observed_sha256": observed,
            "matches_processed_source": observed == source["sha256"],
        })
    if any(identity((raw_root / name).lstat()) != identity(info)
           for name, info in initial.items()):
        raise ValueError("input metadata changed before collection finished")
    if (root / source_path).read_bytes() != source_bytes:
        raise ValueError("processed source metadata changed during collection")
    matched = sum(item["matches_processed_source"] for item in files)
    record = {
        "kind": "local_raw_source_byte_identity", "source_revision": revision,
        "started_at": started_at,
        "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "collector": collector,
        "collector_sha256": hashlib.sha256(collector_bytes).hexdigest(),
        "processed_source_manifest": source_path,
        "processed_source_manifest_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "supplied_raw_root": str(args.raw_root), "resolved_raw_root": str(raw_root),
        "files": files,
        "counts": {"files": len(files), "matching": matched,
                   "mismatching": len(files) - matched, "total_bytes_read": total_bytes},
        "limits": {"files": MAX_FILES, "file_bytes": MAX_FILE_BYTES,
                   "total_bytes": MAX_TOTAL_BYTES, "chunk_bytes": CHUNK_BYTES},
        "all_raw_source_hashes_match": matched == len(files),
        "historical_source_availability_verified": False,
        "publication_allowed": False,
        "limitations": [
            "Byte identity verifies the current local files against the selected processed manifest; it does not authenticate retrieval time or availability at a forecast origin.",
            "File metadata was checked for changes; this is not an atomic filesystem snapshot.",
            "Raw responses were hashed without parsing observations. Archive history, supplementary inputs, FastF1 caches and official-grid provenance were not inspected.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "identity.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), **record["counts"]}))
    return 0 if record["all_raw_source_hashes_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
