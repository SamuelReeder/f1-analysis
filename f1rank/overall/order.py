"""Numerically stable Plackett–Luce factors for the successor model."""
import jax
import jax.numpy as jnp
import numpy as np


def _sum_parts(a, b, barrier=lambda value: value):
    """TwoSum: keep the roundoff as a second, nonoverlapping sorting key.

    Requires finite operands and no arithmetic overflow. JAX callers supply an
    optimization barrier so a compiled generator preserves the rounding steps.
    This is only used for discrete generation, not the differentiable likelihood.
    """
    high = barrier(a + b)
    b_virtual = barrier(high - a)
    a_virtual = barrier(high - b_virtual)
    b_roundoff = barrier(b - b_virtual)
    a_roundoff = barrier(a - a_virtual)
    return high, barrier(a_roundoff + b_roundoff)


def perturbed_order(strength, perturb, mask):
    """JAX order indices for supplied Gumbels, preserving rounded-away noise.

    Sort the compensated sum, with excluded entrants last. Exact mathematical
    ties retain stable input order, as in the original generator. Artificial ties
    caused by discarding the low part are removed. Callers use native leading
    batch axes; no derivative through this discrete generator is defined.
    Nonfinite arithmetic marks the whole row with invalid indices. Generative
    consumers must reject that sentinel instead of using it for array indexing.
    """
    strength, perturb, mask = map(jnp.asarray, (strength, perturb, mask))
    if strength.ndim < 1 or strength.shape != perturb.shape or strength.shape != mask.shape or mask.dtype != jnp.bool_:
        raise ValueError("Strengths, perturbations and Boolean masks must share axes")
    if not jnp.issubdtype(jnp.result_type(strength, perturb), jnp.floating):
        raise ValueError("Perturbed strengths must be floating point")
    dtype = jnp.result_type(strength, perturb)
    a = jnp.where(mask, strength, 0.).astype(dtype)
    b = jnp.where(mask, perturb, 0.).astype(dtype)
    high, low = _sum_parts(a, b, jax.lax.optimization_barrier)
    valid = jnp.all(jnp.isfinite(a) & jnp.isfinite(b) & jnp.isfinite(high) & jnp.isfinite(low), axis=-1)
    order = jnp.lexsort((-low, -high, ~mask), axis=-1)
    return jnp.where(valid[..., None], order, -1)


def numpy_perturbed_order(strength, perturb, mask=None, *, tie_breaker=None):
    """NumPy counterpart; optional caller-supplied keys handle exact sum ties."""
    strength, perturb = np.asarray(strength), np.asarray(perturb)
    if strength.ndim < 1 or strength.shape != perturb.shape:
        raise ValueError("Strengths and perturbations must share axes")
    mask = np.ones(strength.shape, bool) if mask is None else np.asarray(mask)
    if mask.shape != strength.shape or mask.dtype.kind != "b":
        raise ValueError("Boolean mask must share strength axes")
    dtype = np.result_type(strength, perturb)
    if dtype.kind != "f" or not np.isfinite(strength[mask]).all() or not np.isfinite(perturb[mask]).all():
        raise ValueError("Included strengths and perturbations must be finite floating point")
    a = np.where(mask, strength, 0.).astype(dtype)
    b = np.where(mask, perturb, 0.).astype(dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        high, low = _sum_parts(a, b)
    if not np.isfinite(high).all() or not np.isfinite(low).all():
        raise ValueError("Perturbed strength overflow; do not invent an order")
    keys = (-low, -high, ~mask)
    if tie_breaker is not None:
        tie = np.asarray(tie_breaker)
        if tie.shape != strength.shape or not np.isfinite(tie).all():
            raise ValueError("Exact-tie keys must be finite and share strength axes")
        keys = (tie, *keys)
    return np.lexsort(keys, axis=-1)


def plackett_luce(strength, mask):
    """Log probability of each observed order, with no padding probability.

    Rows are orders, best first. Each reverse step carries the suffix maximum
    and sum of exponentials relative to it. Form the log probability in those
    relative coordinates: subtracting two large absolute log normalizers can
    erase tie uncertainty even when each normalizer is itself computed stably.

    Empty suffixes have an explicit Boolean state. No finite sentinel stands in
    for an excluded entrant, and no all-negative-infinity reduction enters the
    gradient. Noncontiguous masks are equivalent to deleting their entries.
    """
    strength, mask = jnp.asarray(strength), jnp.asarray(mask)
    if strength.ndim != 2 or mask.shape != strength.shape or mask.dtype != jnp.bool_:
        raise ValueError("PL strengths and Boolean masks must have matching race/order axes")
    if not jnp.issubdtype(strength.dtype, jnp.floating):
        raise ValueError("PL strengths must be floating point")
    zeros = jnp.zeros(strength.shape[0], dtype=strength.dtype)

    def step(carry, row):
        maximum, norm, present = carry
        raw, included = row
        value = jnp.where(included, raw, 0.)
        previous = jnp.where(present, maximum, value)
        new_maximum = jnp.maximum(value, previous)
        relative = value - new_maximum
        new_norm = jnp.exp(relative) + norm * jnp.exp(previous - new_maximum)
        log_probability = jnp.where(included, relative - jnp.log(new_norm), 0.)
        return (jnp.where(included, new_maximum, maximum),
                jnp.where(included, new_norm, norm), present | included), log_probability

    _, terms = jax.lax.scan(step, (zeros, zeros, jnp.zeros_like(zeros, dtype=bool)),
                           (strength.T, mask.T), reverse=True)
    return terms.sum(axis=0)
