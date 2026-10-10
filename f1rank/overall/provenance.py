"""The successor hashes nested source too, independently of old manifests."""
import hashlib
import json
from pathlib import Path

from ..artifacts import digest

ROOT = Path(__file__).resolve().parents[2]


def snapshot(root=ROOT):
    paths = [*(root / "f1rank").rglob("*.py"), *(root / "extract").rglob("*.py"),
             *(root / "data" / "processed").glob("*.*"),
             *(root / "data" / "overrides").glob("*.*"),
             *(root / "data" / "reference").glob("*.*"),
             root / "docs" / "overall_candidate.md"]
    inputs = {str(p.relative_to(root)): digest(p) for p in sorted(paths)}
    key = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return dict(key=key, inputs=inputs)


def require_snapshot(expected, root=ROOT):
    if expected != snapshot(root):
        raise ValueError("Overall inputs or source changed; regenerate the candidate")
