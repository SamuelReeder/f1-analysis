"""Exact finite-probability planning checks without model fits or real ranks."""
import itertools

import numpy as np
import pytest
from scipy.stats import binom, binomtest

from analysis.overall_calibration_power import alternatives, binary_likelihood_ranks, power_bounds, rejection_regions


def test_rejection_regions_match_the_actual_adjusted_test():
    n, chains, family, alpha = 11, 4, 20, .05
    for cutoff, region in enumerate(rejection_regions(n, chains, family, alpha)):
        expected = [k for k in range(n + 1)
                    if min(1., family * binomtest(k, n, (cutoff + 1) / (chains + 1)).pvalue) < alpha]
        assert list(region) == expected


def test_power_bounds_enclose_exact_dependent_cutoff_enumeration():
    probabilities = np.array([.1, .2, .7])
    n, family, alpha = 4, 2, .2
    regions = rejection_regions(n, len(probabilities) - 1, family, alpha)
    exact = 0.
    for ranks in itertools.product(range(len(probabilities)), repeat=n):
        if any(sum(r <= c for r in ranks) in region for c, region in enumerate(regions)):
            exact += float(np.prod(probabilities[list(ranks)]))
    result = power_bounds(probabilities, replicates=n, family_size=family, alpha=alpha)
    assert result["affected_quantity_power_lower_bound"] <= exact + 1e-14
    assert exact <= result["affected_quantity_power_upper_bound"] + 1e-14
    for row in result["cutoffs"]:
        expected = sum(binom.pmf(k, n, row["alternative_probability"]) for k in row["rejection_counts"])
        np.testing.assert_allclose(row["rejection_probability"], expected)


def test_uniform_reference_respects_the_declared_error_budget():
    result = power_bounds(np.full(5, .2), replicates=25, family_size=40, alpha=.05)
    assert all(row["null_rejection_probability"] <= .05 / 40 for row in result["cutoffs"])
    assert result["affected_quantity_power_upper_bound"] <= .05


def test_saturated_power_remains_a_probability():
    # Reproduced in the fixed planning grid: summing binomial PMFs can round
    # above unity. Keep the mathematical probability bounds without retuning.
    result = power_bounds(alternatives(4)["binary_prior_only_inference"],
                          replicates=800, family_size=9496, alpha=.05)
    assert all(0 <= row["rejection_probability"] <= 1 for row in result["cutoffs"])
    assert 0 <= result["affected_quantity_power_lower_bound"] <= result["affected_quantity_power_upper_bound"] <= 1


def test_binary_example_matches_uniform_exact_inference_and_detects_prior_only_sampler():
    exact = binary_likelihood_ranks(4, posterior_match_probability=.9)
    ignored = binary_likelihood_ranks(4, posterior_match_probability=.5)
    np.testing.assert_allclose(exact, np.full(5, .2), atol=1e-14)
    np.testing.assert_allclose(ignored.sum(), 1.)
    assert not np.allclose(ignored, exact)


@pytest.mark.parametrize("probabilities", [[.5, .5], [.5, -.1, .6], [np.nan, .5, .5], [.1, .2, .6]])
def test_invalid_rank_distribution_fails(probabilities):
    with pytest.raises(ValueError):
        power_bounds(probabilities, replicates=25, family_size=4, alpha=.05)


@pytest.mark.parametrize("n,chains,family,alpha", [(0, 4, 4, .05), (25, 1, 4, .05),
    (25, 4, 1, .05), (25, 4, 4, 0.), (25, 4, 4, True)])
def test_invalid_planning_design_fails(n, chains, family, alpha):
    with pytest.raises(ValueError):
        rejection_regions(n, chains, family, alpha)
