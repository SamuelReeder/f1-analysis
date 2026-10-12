"""Predeclared independent SBC sampling streams; original profiles stay fixed."""
import json
import math

import numpy as np
import pytest
from scipy.special import beta

from f1rank.overall import fitting
from test_overall_fitting import runner, sampled


def recovery_registration(path):
    reg = json.loads(path.read_text())
    reg.update(purpose="recovery_calibration", studies=["first", "second"],
        synthetic={"first":dict(case="connected", seed=1), "second":dict(case="connected", seed=2)},
        sampling_seeds={"first":[10, 11, 12, 13], "second":[20, 21, 22, 23]})
    path.write_text(json.dumps(reg))
    return reg


def test_recovery_fits_use_every_predeclared_sampling_seed(runner):
    blocks, path, directory = runner
    reg = recovery_registration(path)
    seen, receipts = {}, {}
    for study in reg["studies"]:
        seen[study] = []
        def sample(blocks, settings):
            seen[study].append(settings.copy())
            return sampled(settings, len(seen[study]) == len(fitting.RETRY))
        _, receipts[study] = fitting.run(blocks, {}, directory / study, path,
            study=study, synthetic=True, sample=sample)
    assert {study:[s["seed"] for s in attempts] for study, attempts in seen.items()} == reg["sampling_seeds"]
    for study, receipt in receipts.items():
        assert receipt["settings"]["seed"] == reg["sampling_seeds"][study][0]
        assert [attempt["seed"] for attempt in receipt["attempts"]] == reg["sampling_seeds"][study]
        expected = [(fitting.FIT_SETTINGS["warmup"] * warm, fitting.FIT_SETTINGS["samples"] * draws,
                     accept if accept is not None else fitting.FIT_SETTINGS["target_accept"])
                    for warm, draws, accept in fitting.RETRY]
        assert [(s["warmup"], s["samples"], s["target_accept"]) for s in seen[study]] == expected


def test_reused_posterior_draws_do_not_have_the_independent_sbc_rank_law():
    # A Uniform truth and exact Uniform posterior with no informative data.
    # Conditional on a single reused set of posterior draws, rank probabilities
    # equal its interval spacings. Independent repeats must average over draws.
    fixed = np.array([.1, .2, .8, .9])
    repeated_seed_probabilities = np.diff(np.r_[0., fixed, 1.])
    independent_probabilities = np.array([math.comb(len(fixed), k) * beta(k + 1, len(fixed) - k + 1)
                                          for k in range(len(fixed) + 1)])
    np.testing.assert_allclose(independent_probabilities, np.full(len(fixed) + 1, 1 / (len(fixed) + 1)))
    assert not np.allclose(repeated_seed_probabilities, independent_probabilities)


@pytest.mark.parametrize("change", ["missing", "extra", "short", "same_study", "cross_study", "generation",
    "generation_repeated", "sampling_range", "generation_range", "boolean", "profile"])
def test_bad_sampling_plan_fails_before_any_work(runner, change):
    blocks, path, directory = runner
    reg = recovery_registration(path)
    if change == "missing": del reg["sampling_seeds"]
    elif change == "extra": reg["sampling_seeds"]["foreign"] = [30, 31, 32, 33]
    elif change == "short": reg["sampling_seeds"]["first"].pop()
    elif change == "same_study": reg["sampling_seeds"]["first"][1] = 10
    elif change == "cross_study": reg["sampling_seeds"]["first"][1] = 20
    elif change == "generation": reg["sampling_seeds"]["first"][1] = 1
    elif change == "generation_repeated": reg["synthetic"]["second"]["seed"] = 1
    elif change == "sampling_range": reg["sampling_seeds"]["first"][1] = 2**32
    elif change == "generation_range": reg["synthetic"]["first"]["seed"] = 2**32
    elif change == "boolean": reg["sampling_seeds"]["first"][1] = True
    elif change == "profile": reg["purpose"] = "numerical_profile"
    path.write_text(json.dumps(reg))
    with pytest.raises(ValueError):
        fitting.run(blocks, {}, directory, path, study="first", synthetic=True,
                    sample=lambda *args: pytest.fail("Invalid sampling stream reached the fitter"))
    assert not directory.exists()


def test_calibration_sampling_plan_cannot_be_used_for_real_data(runner):
    blocks, path, directory = runner
    recovery_registration(path)
    with pytest.raises(ValueError, match="real-data"):
        fitting.run(blocks, {}, directory, path, study="first", synthetic=False,
                    sample=lambda *args: pytest.fail("Calibration fit received real data"))
    assert not directory.exists()


def test_changing_even_an_unused_seed_invalidates_a_saved_run(runner):
    blocks, path, directory = runner
    reg = recovery_registration(path)
    _, original = fitting.run(blocks, {}, directory, path, study="first", synthetic=True,
        sample=lambda blocks, settings: sampled(settings, True))
    reg["sampling_seeds"]["first"][-1] += 100
    path.write_text(json.dumps(reg))
    with pytest.raises(ValueError, match="requested inputs"):
        fitting.run(blocks, {}, directory, path, study="first", synthetic=True,
                    sample=lambda *args: pytest.fail("Changed registration cannot rerun a cache"))
    assert json.loads((directory / "receipt.json").read_text()) == original


def test_seed_change_during_fit_is_recorded_as_failure(runner):
    blocks, path, directory = runner
    reg = recovery_registration(path)
    def sample(blocks, settings):
        reg["sampling_seeds"]["first"][-1] += 100
        path.write_text(json.dumps(reg))
        return sampled(settings, True)
    with pytest.raises(ValueError, match="Registration changed"):
        fitting.run(blocks, {}, directory, path, study="first", synthetic=True, sample=sample)
    receipt = json.loads((directory / "receipt.json").read_text())
    assert receipt["state"] == "failed_exception" and not (directory / "posterior.npz").exists()


def test_existing_profile_keeps_its_original_seed_schedule(runner):
    _, path, _ = runner
    reg = json.loads(path.read_text())
    assert fitting.sampling_seeds(reg)["toy"] == [fitting.FIT_SETTINGS["seed"] + i for i in range(len(fitting.RETRY))]
    assert fitting.study_settings(reg, "toy") == fitting.FIT_SETTINGS
