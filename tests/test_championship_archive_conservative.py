"""Keep an authorized computation change separate from changing the statistical test."""
import numpy as np
import pytest

from analysis import championship_archive_conservative as f


def test_no_scoring_if_any_training_fold_fails():
    called = []

    def train(year):
        called.append(("train", year))
        if year == 2009:
            raise f.archive.ExhaustedFit("test failure")
        return year

    with pytest.raises(f.archive.ExhaustedFit):
        f.run_folds({"first_test_season": 2007, "last_test_season": 2009},
                    train, lambda year, value: called.append(("score", year)))
    assert called == [("train", 2007), ("train", 2008), ("train", 2009)]


def test_all_training_precedes_scoring_and_no_fold_is_removed():
    called = []
    def train(year):
        called.append(("train", year))
        return year
    def score(year, value):
        assert value == year
        called.append(("score", year))
        return year
    result = f.run_folds({"first_test_season": 2007, "last_test_season": 2009}, train, score)
    assert result == [2007, 2008, 2009]
    assert called == [(phase, year) for phase in ("train", "score") for year in result]


def test_wrapper_keeps_original_model_and_fixed_sampler(monkeypatch):
    captured = {}
    data = object()
    monkeypatch.setattr(f.artifacts, "digest", lambda path: "registered")
    monkeypatch.setattr(f.racemulti, "arrays", lambda frame: (data, ["a"]))

    def nuts(model, **settings):
        assert model is f.racemulti.model
        captured["nuts"] = settings
        return "kernel"

    class Sampler:
        def __init__(self, kernel, **settings):
            assert kernel == "kernel"
            captured["mcmc"] = settings
        def run(self, key, d, **settings):
            assert d is data
            captured["key"] = np.asarray(key)
            captured["run"] = settings
        def get_samples(self, group_by_chain=False):
            values = np.random.default_rng(0).normal(size=(4, 2000, 1))
            return {"u": values if group_by_chain else values.reshape(-1, 1)}
        def get_extra_fields(self):
            return {"diverging": np.zeros(8000, dtype=bool)}

    monkeypatch.setattr(f.racemulti, "NUTS", nuts)
    monkeypatch.setattr(f.racemulti, "MCMC", Sampler)
    post, ids = f.fit_pace(None, **f.SAMPLER, protocol_sha256="registered")
    assert ids == ["a"] and post["_converged"]
    assert len(post["_attempts"]) == 1 and post["_n_draws"] == 8000
    assert captured["nuts"]["target_accept_prob"] == 0.98
    assert captured["mcmc"]["num_warmup"] == 1500
    assert captured["mcmc"]["num_samples"] == 2000
    assert captured["mcmc"]["num_chains"] == 4
    np.testing.assert_array_equal(captured["key"], np.asarray(f.c.jax.random.PRNGKey(3)))
    assert captured["run"] == {"extra_fields": ("diverging",)}
    with pytest.raises(ValueError, match="Unregistered"):
        f.fit_pace(None, **{**f.SAMPLER, "seed": 4}, protocol_sha256="registered")
