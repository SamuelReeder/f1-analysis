"""Proposed conditional orthogonal weekend coordinates; no fitting entry point.

The complete prior and likelihood remain those of the original joint model.
See docs/overall_weekend_coordinates.md. Every original entry direction remains.
"""
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer.reparam import Reparam

from analysis.overall_control_rotation import rotated_model
from f1rank.overall.model import model

HYPER_SITES = {
    "sd_driver_form": "overall_weekend_qualifying_scale",
    "overall_sd_day": "overall_weekend_race_scale",
    "overall_shared_weekend": "overall_weekend_loading",
}
TOTAL = "overall_weekend_total_z"
COMPLEMENT = "overall_weekend_complement_z"


def hyper_priors():
    # Bound by the unchanged model snapshot; independently checked against its
    # actual trace in tests. Never estimate these priors from the failed draws.
    return {"sd_driver_form": dist.HalfNormal(.2),
            "overall_sd_day": dist.HalfNormal(.3),
            "overall_shared_weekend": dist.Normal(1., .5)}


def transform(sf, sr, shared, first, second):
    """Self-inverse on each pair, conditional on positive original scales.

    The same map takes total/complement to original qualifying/race innovations
    or original innovations to total/complement. No clipping or learned matrix.
    """
    sf, sr, shared = (jnp.asarray(v) for v in (sf, sr, shared))
    first, second = jnp.asarray(first), jnp.asarray(second)
    if sf.shape or sr.shape or shared.shape or first.ndim != 1 or first.shape != second.shape:
        raise ValueError("Weekend coordinates need scalar hyperparameters and aligned entry vectors")
    marginal = jnp.hypot(shared * sf, sr)
    a, c = shared * sf / marginal, sr / marginal
    return a * first + c * second, c * first - a * second


class SuppliedWeekend(Reparam):
    def __init__(self, value, family):
        self.value, self.family = value, family

    def __call__(self, name, fn, obs):
        base, _, _ = self._unwrap(fn)
        if obs is not None or type(base) is not self.family or fn.shape() != self.value.shape:
            raise ValueError(f"Unexpected original weekend prior at {name}")
        return None, self.value


def weekend_model(control_rotation=None):
    """Reconstruct original sites before evaluating the unchanged likelihood.

    Hyperparameters retain their original priors under new coordinate names,
    avoiding duplicate sample names. The original names are deterministic outputs.
    """
    original = model if control_rotation is None else rotated_model(control_rotation)

    def wrapped(q, laps, outcomes, *, generate=False):
        priors = hyper_priors()
        hypers = {name: numpyro.sample(HYPER_SITES[name], prior) for name, prior in priors.items()}
        standard = dist.Normal(0., 1.).expand([q["n_entries"]])
        u, v = numpyro.sample(TOTAL, standard), numpyro.sample(COMPLEMENT, standard)
        zq, zr = transform(hypers["sd_driver_form"], hypers["overall_sd_day"],
                           hypers["overall_shared_weekend"], u, v)
        replacements = {name: SuppliedWeekend(value, type(priors[name])) for name, value in hypers.items()}
        replacements.update(driver_form_z=SuppliedWeekend(zq, dist.Normal),
                            overall_day_z=SuppliedWeekend(zr, dist.Normal))
        with handlers.reparam(config=replacements):
            return original(q, laps, outcomes, generate=generate)

    return wrapped
