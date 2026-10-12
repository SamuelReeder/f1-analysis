"""Legacy-format synthetic I/O checks, without accepted fits or MCMC."""
import hashlib
import io
import zipfile

import numpy as np
import pytest

from analysis.overall_legacy_archive import mapped_legacy_posterior
from f1rank import fit
from f1rank.overall.fitting import arrays_digest
from test_overall_legacy_forecast import predict, setup


@pytest.fixture
def archive(tmp_path):
    _, design, snap, values = setup()
    values.update(unused=np.ones((2, 7, 3)),
        fortran=np.asfortranarray(np.arange(42, dtype='>f8').reshape(2, 7, 3)))
    path = tmp_path / "legacy.npz"
    temporary = tmp_path / "maps"
    temporary.mkdir()
    info = dict(elapsed_s=1., divergences=0, mean_steps=1.)
    names = [n for n in values if n != "unused"]

    def save(data=None, **meta):
        options = dict(model_kw={}, settings=dict(chains=2, samples=7),
            purpose="legacy_archive_format_fixture", publication_allowed=False)
        options.update(meta)
        fit.save(path, values if data is None else data, info, design, **options)
        return arguments()

    def arguments():
        return dict(names=names, temporary_parent=temporary,
            expected_archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            expected_metadata_sha256=hashlib.sha256(fit.meta_path(path).read_bytes()).hexdigest(),
            expected_training_arrays_key=arrays_digest(dict(qualifying=design.arrays())))

    save()
    return path, design, snap, values, info, temporary, save, arguments


def test_matches_original_loader_and_preserves_forecast(archive):
    path, design, snap, values, info, temporary, _, arguments = archive
    original, original_info = fit.load(path, design)
    with mapped_legacy_posterior(path, design, **arguments()) as (mapped, loaded_info, receipt):
        assert loaded_info == original_info == info
        for name, value in mapped.items():
            assert isinstance(value, np.memmap) and not value.flags.writeable
            np.testing.assert_array_equal(value, original[name])
        assert mapped["fortran"].flags.f_contiguous
        first = predict(design, [snap], original)
        second = predict(design, [snap], mapped, metadata=receipt["metadata"])
        assert first["consumed_posterior_key"] == second["consumed_posterior_key"]
        for name, expected in first["events"][0]["states"].items():
            np.testing.assert_array_equal(second["events"][0]["states"][name], expected)
        handles = [v._mmap for v in mapped.values()]
        assert not any(receipt[k] for k in ("convergence_verified", "fit_source_verified",
                                            "source_availability_verified", "publication_allowed"))
    assert all(h.closed for h in handles) and not list(temporary.iterdir())


def test_consumer_exception_cleans_maps(archive):
    path, design, _, _, _, temporary, _, arguments = archive
    with pytest.raises(RuntimeError, match="consumer"):
        with mapped_legacy_posterior(path, design, **arguments()) as (mapped, _, _):
            handles = [v._mmap for v in mapped.values()]
            raise RuntimeError("consumer")
    assert all(h.closed for h in handles) and not list(temporary.iterdir())


@pytest.mark.parametrize("fault", ["archive", "metadata", "arrays", "model", "settings", "missing_settings", "identity_format"])
def test_rejects_identity_or_definition_mismatch(archive, fault):
    path, design, _, _, _, temporary, save, arguments = archive
    args = arguments()
    if fault in ("archive", "metadata"):
        args[f"expected_{fault}_sha256"] = "0" * 64
    elif fault == "arrays": args["expected_training_arrays_key"] = "0" * 64
    elif fault == "model": args = save(model_kw=dict(compatibility_unit="era"))
    elif fault == "settings": args = save(settings=dict(chains=True, samples=7))
    elif fault == "missing_settings": args = save(settings={})
    else: args["expected_archive_sha256"] = None
    with pytest.raises(ValueError):
        with mapped_legacy_posterior(path, design, **args):
            pytest.fail("Invalid archive reached consumer")
    assert not list(temporary.iterdir())


@pytest.mark.parametrize("fault", ["nonfinite", "objects", "axes", "successor", "scalar_shape", "scalar_nonfinite"])
def test_whole_archive_is_validated_including_unselected_members(archive, fault):
    path, design, _, values, info, temporary, _, arguments = archive
    broken = {**values, **{f"info_{k}":v for k,v in info.items()}}
    if fault == "nonfinite": broken["unused"][..., -1] = np.inf
    elif fault == "objects": broken["unused"] = np.full((2, 7), object(), dtype=object)
    elif fault == "axes": broken["unused"] = np.zeros((2, 6))
    elif fault == "successor": broken["overall_level_z"] = np.zeros((2, 7, 3))
    elif fault == "scalar_shape": broken["info_elapsed_s"] = np.zeros(2)
    else: broken["info_mean_steps"] = np.nan
    np.savez_compressed(path, **broken)
    with pytest.raises(ValueError):
        with mapped_legacy_posterior(path, design, **arguments()):
            pytest.fail("Invalid unselected member was ignored")
    assert not list(temporary.iterdir())


@pytest.mark.parametrize("fault", ["archive", "metadata", "replacement", "design", "returned_receipt"])
def test_mutation_during_consumption_prevents_successful_exit(archive, fault):
    path, design, _, _, _, temporary, _, arguments = archive
    with pytest.raises((ValueError, fit.IncompatibleFit)):
        with mapped_legacy_posterior(path, design, **arguments()) as (mapped, _, receipt):
            handles = [v._mmap for v in mapped.values()]
            if fault in ("archive", "returned_receipt"):
                with path.open("ab") as handle: handle.write(b"changed")
                if fault == "returned_receipt":
                    receipt["archive_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            elif fault == "metadata":
                with fit.meta_path(path).open("ab") as handle: handle.write(b"\n")
            elif fault == "replacement":
                copy = path.with_suffix(".replacement")
                copy.write_bytes(path.read_bytes()); copy.replace(path)
            else: design.circuit_factor[:] += .125
    assert all(h.closed for h in handles) and not list(temporary.iterdir())


@pytest.mark.parametrize("fault", ["duplicate", "truncated", "trailing", "path"])
def test_malformed_zip_member_is_not_extracted(archive, fault):
    path, design, _, values, _, temporary, _, arguments = archive
    buffer = io.BytesIO(); np.save(buffer, values["sd_level"])
    raw = buffer.getvalue()
    name = "../sd_level.npy" if fault == "path" else "sd_level.npy"
    with zipfile.ZipFile(path, "w") as saved:
        saved.writestr(name, raw[:-1] if fault == "truncated" else raw + b"x" if fault == "trailing" else raw)
        if fault == "duplicate":
            with pytest.warns(UserWarning): saved.writestr(name, raw)
    args = dict(arguments(), names=["sd_level"])
    with pytest.raises(ValueError):
        with mapped_legacy_posterior(path, design, **args):
            pytest.fail("Malformed archive accepted")
    assert not list(temporary.iterdir())
