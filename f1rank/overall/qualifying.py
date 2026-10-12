"""Equivalent proper, generative form of the established qualifying likelihood.

The legacy model expresses the car-event Normal prior as an ImproperUniform
site plus a factor. Express that same density as a Normal site, permitting prior
simulation and a noncentred parameterisation without changing the distribution.
"""
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer.reparam import LocScaleReparam
from numpyro.primitives import Messenger

from ..model import CAR_FIRST_SD, linear_recurrence, model as legacy_model


class ProperCarPrior(Messenger):
    def __init__(self, design, *, generate=False):
        super().__init__()
        self.d, self.values, self.generate = design, {}, generate

    def process_message(self, msg):
        if msg["type"] != "sample":
            return
        if msg["name"] == "car_total":
            d, v = self.d, self.values
            first, start, reset = d["car_first"], d["car_season_start"], d["car_reset"]
            carry = jnp.where(start, jnp.where(reset, v["rho_reset"], v["rho"]), 1.)
            jump = jnp.where(first, CAR_FIRST_SD, jnp.where(reset, v["sd_car_reset"], v["sd_car_season"]))
            b = jnp.where(first | start, jump * v["car_jump_z"], v["sd_car_race"] * v["car_step_z"])
            persistent = linear_recurrence(jnp.where(first, 0., carry), b)
            msg["fn"] = dist.Normal(persistent, v["sd_car_event"]).to_event(1)
        elif msg["name"] == "car_event_prior":
            msg["fn"] = dist.Unit(jnp.asarray(0.))  # already counted by the proper Normal
        elif msg["name"] == "y" and self.generate:
            msg["value"], msg["is_observed"] = None, False

    def postprocess_message(self, msg):
        if msg["type"] in ("sample", "deterministic"):
            self.values[msg["name"]] = msg["value"]


def qualifying(q, *, generate=False, noncentred=True):
    def run():
        with ProperCarPrior(q, generate=generate):
            legacy_model(q, compat_unit="era")
    if noncentred:
        with handlers.reparam(config={"car_total": LocScaleReparam(centered=0.)}):
            run()
    else:
        run()
