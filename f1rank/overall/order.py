"""Numerically stable Plackett–Luce factors for the successor model."""
import jax
import jax.numpy as jnp


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
