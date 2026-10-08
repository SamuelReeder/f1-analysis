import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import pytest

spec = importlib.util.spec_from_file_location(
    "verify_publication", Path(__file__).resolve().parents[1] / "dashboard/verify_publication.py")
publication = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publication)


@pytest.fixture
def exported(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "outputs/championship/summary.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"combined_validation":{"gate":false}}')
    directory = tmp_path / "dashboard/public/data"
    (directory / "releases").mkdir(parents=True)
    payload = {"schema_version": 1, "meta": {"data_as_of": {"event_id": "test"}},
               "overall": {"status": "not established", "standings": []}}
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    release = hashlib.sha256(encoded).hexdigest()[:20]
    path = directory / f"releases/{release}.json"
    path.write_bytes(encoded)
    pointer = directory / "latest.json"
    pointer.write_text(json.dumps({"schema_version": 1, "release": release,
                                  "url": f"releases/{release}.json", "data_as_of": payload["meta"]["data_as_of"]}))
    receipt = {"schema_version": 1, "pointer_sha256": publication.digest(pointer),
               "release_sha256": publication.digest(path),
               "sources": {"outputs/championship/summary.json": publication.digest(source)}}
    (directory / "publication.json").write_text(json.dumps(receipt))
    return tmp_path, source, path


def test_verifies_export_without_posterior_caches(exported):
    publication.verify(exported[0])


def test_changed_gate_evidence_rejected(exported):
    root, source, _ = exported
    source.write_text('{"combined_validation":{"gate":true}}')
    with pytest.raises(ValueError, match="Publication source changed"):
        publication.verify(root)


def test_tampered_standings_rejected(exported):
    root, _, path = exported
    data = json.loads(path.read_text())
    data["overall"]["standings"] = [{"name": "Unverified driver"}]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Immutable publication content changed"):
        publication.verify(root)


def test_new_data_requires_new_publication(exported):
    root, _, _ = exported
    (root / "data").mkdir()
    (root / "data/new_input.json").write_text("{}")
    with pytest.raises(ValueError, match="source inventory changed"):
        publication.verify(root)
