"""Original exporter/sidecar integration with synthetic sampler output, no fit."""
import numpy as np
import pytest

from analysis import overall_legacy_forecast as legacy
from f1rank import fit
from test_overall_legacy_forecast import predict, setup


@pytest.mark.parametrize("retain_forecast_sites", [False, True])
def test_original_export_save_load_boundary(tmp_path, monkeypatch, retain_forecast_sites):
    _, design, snap, posterior = setup()
    axes = posterior["sd_level"].shape
    # Synthetic values stand in for sampler output. The original fit function
    # still executes its actual export filter; no sampling/diagnostic claim.
    class SyntheticSampler:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, *args, **kwargs):
            pass

        def get_samples(self, group_by_chain=False):
            return posterior

        def get_extra_fields(self, group_by_chain=False):
            return dict(num_steps=np.ones(axes), diverging=np.zeros(axes, dtype=bool))

    monkeypatch.setattr(fit, "MCMC", SyntheticSampler)
    extra_sites = legacy.FORECAST_EXTRA_SITES if retain_forecast_sites else ()
    exported, info = fit.fit(design, chains=axes[0], samples=axes[1],
                             extra_sites=extra_sites, progress=False)
    path = tmp_path / "synthetic-export.npz"
    fit.save(path, exported, info, design, model_kw={},
             purpose="synthetic_export_fixture", publication_allowed=False)
    loaded, loaded_info = fit.load(path, design)
    metadata = fit.load_meta(path)
    assert metadata["purpose"] == "synthetic_export_fixture"
    assert metadata["publication_allowed"] is False
    assert loaded_info == info

    if not retain_forecast_sites:
        assert not set(legacy.FORECAST_EXTRA_SITES) & set(loaded)
        with pytest.raises(ValueError, match="FORECAST_EXTRA_SITES"):
            predict(design, [snap], loaded, metadata=metadata)
        return

    assert set(posterior) <= set(loaded)
    for name, expected in posterior.items():
        np.testing.assert_array_equal(loaded[name], expected.astype(np.float32))
    direct = predict(design, [snap], exported, metadata=metadata)
    restored = predict(design, [snap], loaded, metadata=metadata)
    assert restored["consumed_posterior_key"] == direct["consumed_posterior_key"]
    for name, expected in direct["events"][0]["states"].items():
        np.testing.assert_array_equal(restored["events"][0]["states"][name], expected)
    assert restored["convergence_verified"] is False
    assert restored["publication_allowed"] is False


def test_original_sidecar_check_rejects_different_design(tmp_path):
    _, design, _, posterior = setup()
    path = tmp_path / "synthetic-export.npz"
    fit.save(path, posterior, {}, design, model_kw={},
             purpose="synthetic_export_fixture", publication_allowed=False)
    _, changed, _, _ = setup(cutoff="2022-01", target="2022-02")
    with pytest.raises(fit.IncompatibleFit):
        fit.load(path, changed)
