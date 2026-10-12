"""The fitted PL probability must not depend on arbitrary strength offsets."""
import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Exercise the helper actually used by the successor's fitted likelihood.
from f1rank.overall.model import plackett_luce


@pytest.mark.parametrize("offset", [0., -2e9, 2e9])
def test_equal_strength_order_has_known_probability_with_padding(offset):
    strength = jnp.full((1, 5), offset)
    mask = jnp.array([[True, True, True, False, False]])
    actual = plackett_luce(strength, mask)
    np.testing.assert_allclose(actual, [-math.lgamma(4)], rtol=1e-6, atol=1e-6)


def test_widely_separated_groups_retain_uncertainty_within_a_tied_suffix():
    strength = jnp.array([[1e9, 0., -1e9, -1e9]])
    actual = plackett_luce(strength, jnp.ones_like(strength, bool))
    np.testing.assert_allclose(actual, [-np.log(2.)], rtol=1e-6, atol=1e-6)


def test_padding_and_empty_panels_have_no_probability_mass_or_gradient():
    strength = jnp.array([[2., -1e10, 1e10, 4.], [3., 6., -9., 12.]])
    mask = jnp.array([[True, False, False, False], [False, False, False, False]])
    np.testing.assert_array_equal(plackett_luce(strength, mask), np.zeros(2))
    gradient = jax.grad(lambda s:plackett_luce(s, mask).sum())(strength)
    np.testing.assert_array_equal(gradient, np.zeros_like(strength))


def test_shifted_equal_strength_gradients_match_the_analytical_score():
    mask = jnp.array([[True, True, True, False, False]])
    expected = np.array([[1 - 1/3, 1 - 1/3 - 1/2, 1 - 1/3 - 1/2 - 1, 0., 0.]])
    for offset in (0., -2e9, 2e9):
        s = jnp.full(mask.shape, offset)
        gradient = jax.grad(lambda v:plackett_luce(v, mask).sum())(s)
        np.testing.assert_allclose(gradient, expected, atol=2e-6, rtol=2e-6)


def test_noncontiguous_mask_agrees_with_deleting_unobserved_entries():
    strength = jnp.array([[.4, -100., 1.2, 300., -.1]])
    mask = jnp.array([[True, False, True, False, True]])
    compact = strength[:, [0, 2, 4]]
    np.testing.assert_allclose(plackett_luce(strength, mask),
        plackett_luce(compact, jnp.ones_like(compact, bool)), rtol=1e-6)


def test_all_small_field_permutations_normalise():
    strengths = np.log([1., 2., 4.])
    ordered = jnp.asarray([strengths[list(order)] for order in itertools.permutations(range(3))])
    np.testing.assert_allclose(jnp.exp(plackett_luce(ordered, jnp.ones_like(ordered, bool))).sum(), 1., atol=1e-6)


@pytest.mark.parametrize("offset", [-2e20, 2e20])
def test_forecast_probability_also_retains_equal_driver_uncertainty(offset):
    from analysis.overall_observation_kernels import order_log_probability
    strength = np.full(5, offset)
    np.testing.assert_allclose(order_log_probability(strength, [1, 2, 3, 0, 0]),
        -math.lgamma(4), rtol=1e-12, atol=1e-12)
