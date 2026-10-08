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


def record(root=ROOT):
    directory, path, payload = publication(root)
    # This step runs where full provenance, including posterior files, is available.
    sys.path.insert(0, str(root))
    from f1rank.dashboard import build_payload
    if payload != build_payload():
        raise ValueError("Publish the fully checked current outputs before recording a receipt")
    receipt = {"schema_version": 1, "pointer_sha256": digest(directory / "latest.json"),
               "release_sha256": digest(path), "sources": {p: digest(root / p) for p in sources(root)}}
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


if __name__ == "__main__":
    if sys.argv[1:] == ["record"]:
        record()
    elif not sys.argv[1:]:
        verify()
    else:
        raise SystemExit("Usage: verify_publication.py [record]")
    print("Checked portable dashboard publication")
