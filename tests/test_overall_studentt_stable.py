import numpy as np
import pytest
import jax
import jax.numpy as jnp

from analysis.overall_studentt_numerics import load_reference
from analysis.overall_studentt_stable import (
    GAMMA_RATIO_MIN_B, NEXT_STIRLING_COEFFICIENT, STIRLING_COEFFICIENTS,
    TAIL_SERIES_LAST_POWER, TAIL_SERIES_MAX_Q, stable_log_prob,
    stable_studentt_calculation, tail_log_prob,
)
from analysis.overall_studentt_stable_audit import input_rows


def test_stirling_coefficients_are_exact_bernoulli_coefficients():
    mp = load_reference()
    with mp.workdps(80):
        for k, (n, d) in enumerate((*STIRLING_COEFFICIENTS, NEXT_STIRLING_COEFFICIENT), 1):
            assert mp.almosteq(mp.mpf(n)/d, mp.bernoulli(2*k)/(2*k*(2*k-1)))


def test_analytic_remainder_bounds_are_below_double_precision_budget():
    mp = load_reference()
    with mp.workdps(80):
        n, d = NEXT_STIRLING_COEFFICIENT
        power = 2*len(STIRLING_COEFFICIENTS)+1
        # Bound each positive-argument log-gamma remainder separately.
        value_bound = abs(mp.mpf(n)/d)*(mp.mpf(GAMMA_RATIO_MIN_B)**(-power)
                                      +(mp.mpf(GAMMA_RATIO_MIN_B)+mp.mpf('.5'))**(-power))
        q = mp.mpf(TAIL_SERIES_MAX_Q)
        relative_tail_bound = 2*q**(TAIL_SERIES_LAST_POWER-1)/((TAIL_SERIES_LAST_POWER+1)*(1-q))
        assert value_bound < np.finfo('float64').eps
        assert relative_tail_bound < 32*np.finfo('float64').eps


def test_cauchy_value_derivative_and_density_transform():
    value, derivative = jax.value_and_grad(lambda z:stable_log_prob(jnp.exp(z), 0., 1., 0.))(jnp.array(0.))
    np.testing.assert_allclose([value, derivative], [-np.log(np.pi), np.log(2)-.5], rtol=2e-6, atol=2e-6)
    x = jnp.array([-3., 0., 1.])
    np.testing.assert_allclose(stable_log_prob(4., 0., 1., x),
        stable_log_prob(4., 2., 3., 2+3*x)+np.log(3), rtol=2e-6, atol=2e-6)


def test_tail_jvp_is_linear_and_supports_reverse_and_higher_derivatives():
    f = lambda p:tail_log_prob(jnp.exp(p[0]), p[1])
    p = jnp.array([0., 1.]); a = jnp.array([2., -1.]); b = jnp.array([-1., 3.])
    jvp = lambda t:jax.jvp(f, (p,), (t,))[1]
    np.testing.assert_allclose(jvp(a+b), jvp(a)+jvp(b), rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(jvp(a), jnp.dot(jax.grad(f)(p), a), rtol=2e-6, atol=2e-6)
    hessian = jax.hessian(f)(p)
    assert np.isfinite(hessian).all()
    np.testing.assert_allclose(hessian, hessian.T, rtol=2e-6, atol=2e-6)


def test_signed_boundary_neighbors_extend_without_replacing_original_rows():
    original = [dict(kind='original', log_df=0., residual=v) for v in (-1., 0., 1.)]
    rows = input_rows(original)
    assert rows[:len(original)] == original
    added = rows[len(original):]
    assert len(added) == 6
    assert len({r['residual'] for r in added}) == 6
    assert {r['residual'] for r in added} == {-r['residual'] for r in added}


def test_substitution_preserves_sampling_and_restores_on_exception():
    from numpyro.distributions import StudentT
    original, sample = StudentT.log_prob, StudentT.sample
    with pytest.raises(RuntimeError):
        with stable_studentt_calculation():
            assert StudentT.log_prob is not original and StudentT.sample is sample
            raise RuntimeError('interrupted audit')
    assert StudentT.log_prob is original and StudentT.sample is sample
