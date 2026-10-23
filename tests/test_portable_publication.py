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


@pytest.fixture
def previewed(exported):
    """The committed exploratory preview outputs, published into the synthetic root."""
    import shutil
    root = exported[0]
    real = Path(__file__).resolve().parents[1]
    for name in publication.PREVIEW_FILES:
        target = root / publication.PREVIEW / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(real / publication.PREVIEW / name, target)
    events = root / "data/processed/events.parquet"
    events.parent.mkdir(parents=True)
    shutil.copy(real / "data/processed/events.parquet", events)
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    preview = root / "dashboard/public/data/overall_preview.json"
    preview.write_text(json.dumps(publication.preview_payload(root)))
    receipt_path = root / "dashboard/public/data/publication.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["sources"] = {p: publication.digest(root / p) for p in publication.sources(root)}
    receipt["overall_preview_sha256"] = publication.digest(preview)
    receipt_path.write_text(json.dumps(receipt))
    return root, preview, receipt_path


def test_preview_verifies_from_committed_outputs(previewed):
    root, preview, _ = previewed
    publication.verify(root)
    data = json.loads(preview.read_text())
    assert data["validated"] is False and data["combined_validation"]["gate"] is False
    assert data["standings"] == sorted(data["standings"], key=lambda r: -r["points_per_race"])


def test_tampered_preview_rejected(previewed):
    root, preview, receipt_path = previewed
    data = json.loads(preview.read_text())
    data["standings"][0]["points_per_race"] += 1
    preview.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Overall preview changed"):
        publication.verify(root)
    # Even with a matching receipt, the file must equal its committed outputs.
    receipt = json.loads(receipt_path.read_text())
    receipt["overall_preview_sha256"] = publication.digest(preview)
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="disagrees with its committed outputs"):
        publication.verify(root)


def test_preview_outputs_must_match_their_manifest(previewed):
    root, _, _ = previewed
    standings = root / publication.PREVIEW / "standings.csv"
    standings.write_text(standings.read_text().replace("Max Verstappen", "Someone Else"))
    with pytest.raises(ValueError, match="disagrees with its manifest"):
        publication.verify(root)


def test_preview_marked_as_passing_rejected(previewed):
    root, _, _ = previewed
    summary = root / publication.PREVIEW / "summary.json"
    data = json.loads(summary.read_text())
    data["gate"] = True
    summary.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        publication.verify(root)


def test_recorded_preview_cannot_silently_disappear(previewed):
    root, preview, _ = previewed
    preview.unlink()
    with pytest.raises(ValueError, match="recorded overall preview is missing"):
        publication.verify(root)
