"""Verified equivalent Student-t arithmetic for the Overall likelihood.

The independently checked research implementation is preserved in analysis.
This module changes arithmetic only; sampling and distribution support are inherited.
"""
import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

GAMMA_RATIO_MIN_B = 8.
STIRLING_COEFFICIENTS = (
    (1, 12), (-1, 360), (1, 1260), (-1, 1680), (1, 1188),
    (-691, 360360), (1, 156), (-3617, 122400),
    (43867, 244188), (-174611, 125400),
)
NEXT_STIRLING_COEFFICIENT = (854513, 63756)
TAIL_SERIES_MAX_Q = 1 / 8
TAIL_SERIES_LAST_POWER = 16


def standard_normalizer(df):
    """log(sqrt(df) * beta(1/2, df/2)), including a stable gamma ratio."""
    df = jnp.asarray(df)
    b = df / 2
    half = jnp.asarray(.5, dtype=df.dtype)
    # Keep the inactive branch finite; only the selected branch defines the value.
    small_b = jnp.where(b < GAMMA_RATIO_MIN_B, b, GAMMA_RATIO_MIN_B)
    small = half * jnp.log(2 * small_b) + gammaln(half) + (
        gammaln(small_b) - gammaln(small_b + half))
    large_b = jnp.where(b >= GAMMA_RATIO_MIN_B, b, GAMMA_RATIO_MIN_B)
    h = half / large_b
    x = 1 / (1 + h)
    one_minus_x = h / (1 + h)
    # delta(b) - delta(b+1/2) = (1-x)/b * sum c_k S_(2k-1)/b^(2k-2).
    # S_m = 1 + x + ... + x^(m-1). No subtraction of almost equal powers.
    sums = [jnp.ones_like(large_b)]
    for _ in STIRLING_COEFFICIENTS[1:]:
        sums.append(1 + x + x*x*sums[-1])
    inverse_square = (1 / large_b)**2
    polynomial = jnp.zeros_like(large_b)
    for (numerator, denominator), geometric in reversed(list(zip(STIRLING_COEFFICIENTS, sums))):
        polynomial = polynomial*inverse_square + (numerator/denominator)*geometric
    delta_difference = one_minus_x/large_b * polynomial
    large = half*jnp.log(2*jnp.pi) + half - large_b*jnp.log1p(h) + delta_difference
    return jnp.where(b < GAMMA_RATIO_MIN_B, small, large)


def log1p_minus_fraction(u):
    """log1p(u)-u/(1+u), avoiding cancellation near zero for u >= 0."""
    q = u / (1 + u)
    series = jnp.zeros_like(q)
    for power in range(TAIL_SERIES_LAST_POWER, 1, -1):
        series = series*q + 1/power
    series = q*q*series
    return jnp.where(q <= TAIL_SERIES_MAX_Q, series, jnp.log1p(u)-q)


@jax.custom_jvp
def tail_log_prob(df, residual):
    return -.5*(df+1)*jnp.log1p(residual*residual/df)


@tail_log_prob.defjvp
def tail_log_prob_jvp(primals, tangents):
    df, residual = primals
    df_tangent, residual_tangent = tangents
    u = residual*residual/df
    q = u/(1+u)
    d_df = .5*(q/df - log1p_minus_fraction(u))
    d_residual = -(df+1)*residual/(df+residual*residual)
    return tail_log_prob(df, residual), d_df*df_tangent + d_residual*residual_tangent


def _arguments(df, loc, scale, value):
    dtype = jnp.result_type(df, loc, scale, value, 1.)
    return tuple(jnp.asarray(v, dtype=dtype) for v in (df, loc, scale, value))




def stable_log_prob(df, loc, scale, value):
    df, loc, scale, value = _arguments(df, loc, scale, value)
    return tail_log_prob(df, (value-loc)/scale) - jnp.log(scale) - standard_normalizer(df)



from numpyro.distributions import StudentT
from numpyro.distributions.util import validate_sample

ARITHMETIC_METHOD = "stable_studentt_v1"


class StableStudentT(StudentT):
    """Same Student-t distribution with the verified probability calculation."""

    @validate_sample
    def log_prob(self, value):
        return stable_log_prob(self.df, self.loc, self.scale, value)
