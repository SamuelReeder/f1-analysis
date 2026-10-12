"""Generated orders must retain the actual perturbations at large offsets."""
import copy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from analysis.overall_observation_kernels import draw_running_order
from f1rank.overall.model import model
from f1rank.overall.simulation import Scenario, simulate
from test_overall import draws
from test_overall_forecast_states import fixture, posterior


def test_actual_joint_model_generator_preserves_supplied_gumbel_order():
    from numpyro import handlers
    _, _, design, _ = fixture()
    p = {name:jnp.asarray(value[0, 0]) for name,value in posterior(design).items()}
    outcomes = copy.deepcopy(design.outcomes)
    outcomes["log_grid"][:] = 1.
    # A common large grid term exposes loss of the order perturbations; the tiny
    # pace coefficient leaves no distinguishable differences in the input state.
    p.update(overall_grid_base=jnp.log(1e9), overall_pace_base=jnp.asarray(-80.),
        overall_sd_grid_circuit=jnp.asarray(0.), overall_grid_wet=jnp.asarray(0.),
        overall_grid_sc=jnp.asarray(0.))
    perturb = np.broadcast_to([-1., 1., .25, -.5, 2., 0.], outcomes["entry"].shape)
    p["generated_order_noise"] = jnp.asarray(perturb)
    arrays = lambda block:{k:jnp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in block.items()}
    run = handlers.seed(handlers.substitute(model, data=p), jax.random.PRNGKey(19))
    trace = handlers.trace(run).get_trace(arrays(design.qualifying.arrays()),
        arrays(design.laps), arrays(outcomes), generate=True)
    expected = np.argsort(-perturb, axis=-1)
    np.testing.assert_array_equal(trace["generated_order"]["value"], expected)


class ControlledNoise:
    def draw(self, site, identity, *, distribution):
        values = {"race_order":dict(a=-1., b=2., c=.5),
                  "race_order_tie":dict(a=0., b=.9, c=.5)}
        return np.full((2, 3), values[site][identity[-1]])


def test_forecast_uses_actual_gumbels_instead_of_unrelated_tie_noise():
    event = dict(event_id="synthetic", drivers=["a", "b", "c"])
    running = np.ones((2, 3, 3), bool)
    for offset in (0., -2e20, 2e20):
        result = draw_running_order(event, np.full(running.shape, offset), running, ControlledNoise())
        np.testing.assert_array_equal(result, np.broadcast_to([3, 1, 2], running.shape))


def test_counterfactual_race_points_ignore_a_common_race_strength_offset():
    d = draws()
    d = replace(d, grid_coefficient=np.zeros_like(d.grid_coefficient),
                q_weekend_sd=np.zeros_like(d.q_weekend_sd), r_weekend_sd=np.zeros_like(d.r_weekend_sd))
    args = dict(scenarios=[Scenario("synthetic", 0, 60, 60)], replicates=3, seed=41, advancing=(4, 2))
    original = simulate(d, **args)
    shifted = simulate(replace(d, race=d.race + 2e20), **args)
    np.testing.assert_array_equal(shifted["points"], original["points"])
