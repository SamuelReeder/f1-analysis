"""Generative equivalence, fixed retries and immutable evidence for Overall."""
import json

import numpy as np
import pytest

from f1rank.overall import fitting


def test_proper_qualifying_prior_preserves_legacy_density():
    import jax
    from numpyro import handlers
    from numpyro.infer.util import log_density
    from f1rank.model import model as legacy
    from f1rank.overall.qualifying import qualifying
    from f1rank.overall.recovery import synthetic_design
    q = synthetic_design("connected").qualifying.arrays()
    trace = handlers.trace(handlers.seed(qualifying, jax.random.PRNGKey(8))).get_trace(q, noncentred=False)
    params = {name: site["value"] for name, site in trace.items()
              if site["type"] == "sample" and not site["is_observed"]}
    old, _ = log_density(legacy, (q,), {"compat_unit": "era"}, params)
    new, _ = log_density(qualifying, (q,), {"noncentred": False}, params)
    np.testing.assert_allclose(old, new, rtol=1e-6)
    assert np.isfinite(new)


def test_synthetic_generation_repeats_exactly_and_never_uses_real_ratings():
    from f1rank.overall.recovery import generate, arrays_digest
    first, catalog, truth = generate("connected", 23)
    repeated, again, truth_again = generate("connected", 23)
    assert arrays_digest(first) == arrays_digest(repeated)
    assert catalog == again
    for name in truth:
        np.testing.assert_array_equal(truth[name], truth_again[name])
    assert all(name.startswith("synthetic_") for name in catalog["field"])
    assert np.std(first["qualifying"]["y"]) > 0
    assert np.std(first["laps"]["y"]) > 0
    assert (first["outcomes"]["completed"] <= first["outcomes"]["risk_distance"]).all()
    assert (first["outcomes"]["completed"] >= 0).all()
    assert first["outcomes"]["mask"].all()


def test_recovery_inventory_keeps_confounded_rookie_and_era_cases():
    from f1rank.overall.recovery import synthetic_design
    fixed = synthetic_design("confounded").qualifying.entries
    connected = synthetic_design("connected").qualifying.entries
    assert fixed.groupby("driver_id").team.nunique().eq(1).all()
    assert connected.groupby("driver_id").team.nunique().gt(1).all()
    rookie = synthetic_design("sparse_rookie")
    counts = rookie.qualifying.entries.groupby("driver_id").size()
    assert counts["synthetic_5"] < counts["synthetic_0"]
    assert "synthetic_5" in rookie.catalog["field"]
    assert not rookie.qualifying.entries.query("driver_id == 'synthetic_5'").has_time.all()
    era = synthetic_design("era_boundary").qualifying.cars
    assert set(era.loc[era.reset, "season"]) == {2022, 2026}


@pytest.fixture
def runner(tmp_path, monkeypatch):
    registration = tmp_path / "registration.json"
    registration.write_text(json.dumps(dict(schema_version=1, status="frozen", purpose="numerical_profile",
        fit_settings=fitting.FIT_SETTINGS, retry=[list(x) for x in fitting.RETRY], studies=["toy"])))
    monkeypatch.setattr(fitting, "committed_registration", lambda path: json.loads(path.read_text()))
    monkeypatch.setattr(fitting, "snapshot", lambda: dict(key="snapshot", inputs={}))
    monkeypatch.setattr(fitting, "require_snapshot", lambda expected: None)
    monkeypatch.setattr(fitting, "require_clean_inputs", lambda expected: None)
    blocks = dict(qualifying={"y": np.array([0.])}, laps={"y": np.array([0.])}, outcomes={"mask": np.array([True])})
    return blocks, registration, tmp_path / "fit"


def sampled(settings, success):
    return {"latent": np.zeros((settings["chains"], settings["samples"], 1))}, {
        "converged": success, "divergences": 0, "parameters": []}


def test_registered_retry_ladder_and_failed_evidence_survive(runner):
    blocks, registration, directory = runner
    settings_seen = []
    def sample(blocks, settings):
        settings_seen.append(settings)
        return sampled(settings, len(settings_seen) == len(fitting.RETRY))
    post, receipt = fitting.run(blocks, {}, directory, registration, study="toy", synthetic=True, sample=sample)
    assert [(s["warmup"], s["samples"], s["target_accept"], s["seed"]) for s in settings_seen] == [
        (800, 800, .9, 0), (1600, 1600, .9, 1), (2400, 3200, .9, 2), (2400, 3200, .98, 3)]
    assert receipt["state"] == "converged" and not receipt["publication_allowed"]
    assert (directory / "failed_1.npz").exists() and (directory / "failed_3.npz").exists()
    cached, again = fitting.run(blocks, {}, directory, registration, study="toy", synthetic=True,
                                sample=lambda *args: pytest.fail("A checked fit should be reused"))
    np.testing.assert_array_equal(cached["latent"], post["latent"])
    assert again == receipt


def test_exhausted_fit_is_terminal_and_cannot_be_reused(runner):
    blocks, registration, directory = runner
    with pytest.raises(RuntimeError, match="All registered"):
        fitting.run(blocks, {}, directory, registration, study="toy", synthetic=True,
                    sample=lambda blocks, settings: sampled(settings, False))
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["state"] == "failed_convergence" and len(receipt["attempts"]) == len(fitting.RETRY)
    with pytest.raises(ValueError, match="Failed or incomplete"):
        fitting.run(blocks, {}, directory, registration, study="toy", synthetic=True,
                    sample=lambda *args: pytest.fail("Do not rerun failed fits"))


def test_source_change_during_fit_records_failure_and_does_not_cache(runner, monkeypatch):
    blocks, registration, directory = runner
    changed = False
    def verify(expected):
        if changed:
            raise ValueError("Source changed")
    def sample(blocks, settings):
        nonlocal changed
        changed = True
        return sampled(settings, True)
    monkeypatch.setattr(fitting, "require_snapshot", verify)
    with pytest.raises(ValueError, match="Source changed"):
        fitting.run(blocks, {}, directory, registration, study="toy", synthetic=True, sample=sample)
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["state"] == "failed_exception"
    assert not (directory / "posterior.npz").exists()


def test_profile_registration_cannot_run_real_data(runner):
    blocks, registration, directory = runner
    with pytest.raises(ValueError, match="real-data"):
        fitting.run(blocks, {}, directory, registration, study="toy", synthetic=False)
    assert not directory.exists()


def test_cache_rejects_changed_arrays_and_posterior_bytes(runner):
    blocks, registration, directory = runner
    fitting.run(blocks, {}, directory, registration, study="toy", synthetic=True,
                sample=lambda blocks, settings: sampled(settings, True))
    blocks["qualifying"]["y"][0] = 1
    with pytest.raises(ValueError, match="requested inputs"):
        fitting.run(blocks, {}, directory, registration, study="toy", synthetic=True)
    (directory / "posterior.npz").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="archive changed"):
        fitting.load_checked(directory)
