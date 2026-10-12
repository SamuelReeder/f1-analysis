"""Cache-format fixtures verify bounded I/O, never real fit convergence."""
import hashlib
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from analysis.overall_posterior_archive import READ_BYTES, mapped_checked_posterior
from f1rank.overall import fitting


@pytest.fixture
def archive(tmp_path, monkeypatch):
    source = dict(key="fixture-source", inputs={})
    def check(expected):
        if expected != source:
            raise ValueError("Overall inputs or source changed")
    monkeypatch.setattr(fitting, "require_snapshot", check)
    directory, temporary = tmp_path / "run", tmp_path / "maps"
    directory.mkdir(); temporary.mkdir()
    values = dict(keep=np.arange(30, dtype=np.float32).reshape(2, 5, 3),
        fortran=np.asfortranarray(np.arange(40, dtype='>f8').reshape(2, 5, 4)),
        empty=np.empty((2, 5, 0)), unused=np.ones((2, 5)))
    def save(data=values, **changes):
        path = directory / "posterior.npz"
        np.savez_compressed(path, **data)
        receipt = dict(state="converged", diagnostics=dict(converged=True),
            source=source.copy(), run_key="fixture-run", posterior_file=path.name,
            posterior_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            settings=dict(chains=2), attempts=[dict(samples=5)],
            specification=dict(arrays_key="fixture-arrays", synthetic=True),
            publication_allowed=False, purpose="cache_format_fixture_only")
        receipt.update(changes)
        (directory / "receipt.json").write_text(json.dumps(receipt))
        return receipt
    save()
    return directory, temporary, values, save


def rehash(directory):
    path = directory / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["posterior_sha256"] = hashlib.sha256((directory / "posterior.npz").read_bytes()).hexdigest()
    path.write_text(json.dumps(receipt))


def test_selected_maps_match_normal_loader_and_close_on_exit(archive):
    directory, temporary, values, _ = archive
    full, original = fitting.load_checked(directory, "fixture-run")
    with mapped_checked_posterior(directory, names=["keep", "fortran", "empty"], expected_key="fixture-run",
            expected_arrays_key="fixture-arrays", temporary_parent=temporary) as (mapped, receipt):
        assert set(mapped) == {"keep", "fortran", "empty"} and receipt == original
        assert len(list(temporary.iterdir())) == 1
        for name, value in mapped.items():
            assert isinstance(value, np.memmap) and not value.flags.writeable
            np.testing.assert_array_equal(value, full[name])
            assert value.dtype == values[name].dtype
        assert mapped["fortran"].flags.f_contiguous
        handles = [value._mmap for value in mapped.values()]
        assert len(list(next(temporary.iterdir()).iterdir())) == len(mapped)
    assert not list(temporary.iterdir()) and all(handle.closed for handle in handles)


def test_exception_in_consumer_still_cleans_up(archive):
    directory, temporary, _, _ = archive
    with pytest.raises(RuntimeError, match="consumer failed"):
        with mapped_checked_posterior(directory, names=["keep"], temporary_parent=temporary) as (mapped, _):
            handle = mapped["keep"]._mmap
            raise RuntimeError("consumer failed")
    assert handle.closed and not list(temporary.iterdir())


@pytest.mark.parametrize("change", ["nan", "wrong_axes", "objects"])
def test_invalid_unselected_site_rejects_whole_posterior(archive, change):
    directory, temporary, values, save = archive
    broken = dict(values)
    broken["unused"] = (np.full((2, 5), np.nan) if change == "nan" else
        np.zeros((2, 4)) if change == "wrong_axes" else np.full((2, 5), object(), dtype=object))
    save(broken)
    with pytest.raises(ValueError):
        with mapped_checked_posterior(directory, names=["keep"], temporary_parent=temporary):
            pytest.fail("An invalid unselected site may not leave a usable subset")
    assert not list(temporary.iterdir())


@pytest.mark.parametrize("change", ["failed", "diagnostics", "source", "run_key", "arrays_key", "checksum", "path"])
def test_shared_fit_and_identity_guards_run_before_decompression(archive, monkeypatch, change):
    directory, temporary, values, save = archive
    args = dict(names=["keep"], expected_key="fixture-run", expected_arrays_key="fixture-arrays", temporary_parent=temporary)
    if change == "failed": save(state="failed_convergence")
    elif change == "diagnostics": save(diagnostics=dict(converged=False))
    elif change == "source": save(source=dict(key="stale", inputs={}))
    elif change == "run_key": args["expected_key"] = "different"
    elif change == "arrays_key": args["expected_arrays_key"] = "different"
    elif change == "checksum": (directory / "posterior.npz").write_bytes(b"changed")
    elif change == "path": save(posterior_file="../elsewhere.npz")
    monkeypatch.setattr(zipfile, "ZipFile", lambda *args, **kwargs: pytest.fail("Unexpected decompression"))
    with pytest.raises(ValueError):
        with mapped_checked_posterior(directory, **args):
            pytest.fail("Invalid cache accepted")
    assert not list(temporary.iterdir())


@pytest.mark.parametrize("change", ["receipt", "archive", "replacement", "source", "returned_metadata"])
def test_mutation_during_consumption_fails_context_exit(archive, monkeypatch, change):
    directory, temporary, _, _ = archive
    with pytest.raises(ValueError):
        with mapped_checked_posterior(directory, names=["keep"], temporary_parent=temporary) as (mapped, receipt):
            handle = mapped["keep"]._mmap
            if change == "receipt":
                with (directory / "receipt.json").open("ab") as out: out.write(b"\n")
            elif change in ("archive", "returned_metadata"):
                with (directory / "posterior.npz").open("ab") as out: out.write(b"changed")
                if change == "returned_metadata":
                    receipt["posterior_sha256"] = hashlib.sha256((directory / "posterior.npz").read_bytes()).hexdigest()
            elif change == "replacement":
                copy = directory / "copy.npz"
                copy.write_bytes((directory / "posterior.npz").read_bytes())
                copy.replace(directory / "posterior.npz")
            else:
                def changed(expected): raise ValueError("Source changed during consumption")
                monkeypatch.setattr(fitting, "require_snapshot", changed)
    assert handle.closed and not list(temporary.iterdir())


@pytest.mark.parametrize("names", [[], ["keep", "keep"], ["missing"], ["../keep"], "keep"])
def test_missing_or_invalid_requested_names_fail(archive, names):
    directory, temporary, _, _ = archive
    with pytest.raises(ValueError):
        with mapped_checked_posterior(directory, names=names, temporary_parent=temporary):
            pytest.fail("Invalid requested sites accepted")
    assert not list(temporary.iterdir())


@pytest.mark.parametrize("change", ["duplicate", "traversal", "truncated", "trailing", "empty"])
def test_invalid_zip_members_fail_without_extraction(archive, change):
    directory, temporary, values, _ = archive
    buffer = io.BytesIO()
    np.save(buffer, values["keep"])
    raw = buffer.getvalue()
    with zipfile.ZipFile(directory / "posterior.npz", "w", compression=zipfile.ZIP_DEFLATED) as out:
        if change != "empty":
            name = "../keep.npy" if change == "traversal" else "keep.npy"
            out.writestr(name, raw[:-1] if change == "truncated" else raw + b"x" if change == "trailing" else raw)
        if change == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                out.writestr("keep.npy", raw)
    rehash(directory)
    with pytest.raises(ValueError):
        with mapped_checked_posterior(directory, names=["keep"], temporary_parent=temporary):
            pytest.fail("Invalid archive accepted")
    assert not list(temporary.iterdir()) and not (temporary / "keep.npy").exists()


def test_decompression_reads_are_bounded(archive, monkeypatch):
    directory, temporary, _, save = archive
    save(dict(keep=np.zeros((2, 5, READ_BYTES // 8), dtype=np.float64)))
    original, counts = zipfile.ZipExtFile.read, []
    def read(self, n=-1):
        counts.append(n)
        assert 0 <= n <= READ_BYTES
        return original(self, n)
    monkeypatch.setattr(zipfile.ZipExtFile, "read", read)
    with mapped_checked_posterior(directory, names=["keep"], temporary_parent=temporary) as (mapped, _):
        assert mapped["keep"].shape == (2, 5, READ_BYTES // 8)
    assert max(counts) == READ_BYTES


def test_mapped_joint_parameters_feed_the_same_forecast(archive):
    from analysis.overall_forecast_catalog import build_catalog
    from analysis.overall_forecast_states import forecast_states
    from test_overall_forecast_states import fixture, posterior
    directory, temporary, _, save = archive
    _, _, design, snap = fixture()
    parameters = posterior(design)
    key = build_catalog(design, snap)["training_arrays_key"]
    save(parameters, attempts=[dict(samples=3)], specification=dict(arrays_key=key, synthetic=True))
    original = forecast_states(design, [snap], parameters, seed=41, training_arrays_key=key)
    with mapped_checked_posterior(directory, names=list(parameters), expected_arrays_key=key,
            temporary_parent=temporary) as (mapped, receipt):
        actual = forecast_states(design, [snap], mapped, seed=41,
            training_arrays_key=receipt["specification"]["arrays_key"])
        assert actual["consumed_posterior_key"] == original["consumed_posterior_key"]
        for name, value in actual["events"][0]["states"].items():
            np.testing.assert_array_equal(value, original["events"][0]["states"][name])
    assert not list(temporary.iterdir())
