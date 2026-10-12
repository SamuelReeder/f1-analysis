"""Generated orders must retain the actual perturbations at large offsets."""
import copy
from dataclasses import replace
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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


def exact_order(a, b, mask):
    return np.array(sorted(range(len(a)), key=lambda i:(not mask[i],
        -(Fraction(float(a[i])) + Fraction(float(b[i]))) if mask[i] else Fraction(0), i)))


@pytest.mark.parametrize("compiled", [False, True])
def test_jax_generator_matches_exact_rational_sum_order_and_masks(compiled):
    from f1rank.overall.order import perturbed_order
    # Separated groups test the lower suffix as well as a common offset. Merely
    # subtracting a global maximum would still erase noise in that suffix.
    a = np.array([[1e20, 1e20, -1e20, -1e20, 0., 0.],
                  [1., 1., 1., 1., 1., 1.]], np.float32)
    b = np.array([[-1., 2., .5, 1., -2., 3.], [-.5, 1., 0., .1, .9, -.7]], np.float32)
    mask = np.array([[True, True, True, True, True, True], [False, True, False, True, True, False]])
    fn = jax.jit(perturbed_order) if compiled else perturbed_order
    np.testing.assert_array_equal(fn(a,b,mask), np.array([exact_order(x,y,m) for x,y,m in zip(a,b,mask)]))
    ix = [3, 0, 5, 1, 4, 2]
    result = np.asarray(fn(a[:,ix], b[:,ix], mask[:,ix]))
    # Included entrants have distinct exact sums. Excluded suffix order is unused.
    for row in range(len(a)):
        expected = exact_order(a[row], b[row], mask[row])[:mask[row].sum()]
        np.testing.assert_array_equal(np.array(ix)[result[row, :len(expected)]], expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_generator_preserves_exact_sum_order_across_scales(dtype):
    from f1rank.overall.order import numpy_perturbed_order
    rng = np.random.default_rng(91)
    a = np.ldexp(rng.integers(-8, 9, size=(40, 8)).astype(dtype), rng.choice([-20, 0, 20, 80], size=(40, 8)))
    b = rng.normal(size=a.shape).astype(dtype)
    a[:, 1] = a[:, 0]  # Exact ties in strength, broken by retained perturbations.
    mask = rng.random(size=a.shape) > .2
    result = numpy_perturbed_order(a,b,mask)
    np.testing.assert_array_equal(result, np.array([exact_order(x,y,m) for x,y,m in zip(a,b,mask)]))


def test_exact_ties_use_only_the_declared_tie_convention():
    from f1rank.overall.order import numpy_perturbed_order, perturbed_order
    a, b = np.array([1., 2., 3.]), np.array([2., 1., 0.])
    np.testing.assert_array_equal(numpy_perturbed_order(a,b), [0, 1, 2])
    np.testing.assert_array_equal(perturbed_order(a,b,np.ones(3,bool)), [0, 1, 2])
    np.testing.assert_array_equal(numpy_perturbed_order(a,b,tie_breaker=np.array([.8, .1, .4])), [1, 2, 0])


def test_generated_first_place_frequencies_agree_with_pl_probabilities():
    from f1rank.overall.order import perturbed_order
    rng = np.random.default_rng(73)
    n = 24000
    s = np.broadcast_to(np.log([1., 2., 4.]), (n, 3)).astype(np.float32)
    noise = rng.gumbel(size=s.shape).astype(np.float32)
    winners = np.asarray(jax.jit(perturbed_order)(s, noise, np.ones(s.shape,bool)))[:, 0]
    for i,p in enumerate([1/7, 2/7, 4/7]):
        assert abs((winners == i).mean() - p) < 6 * np.sqrt(p * (1-p) / n)


def test_invalid_perturbed_arithmetic_cannot_become_an_order():
    from f1rank.overall.order import numpy_perturbed_order, perturbed_order
    largest = np.finfo(np.float32).max
    a = np.array([[largest, 0.], [np.nan, 0.]], np.float32)
    b = np.array([[largest, 0.], [0., 0.]], np.float32)
    mask = np.ones(a.shape,bool)
    assert (np.asarray(jax.jit(perturbed_order)(a,b,mask)) == -1).all()
    with pytest.raises(ValueError):
        numpy_perturbed_order(a,b,mask)
    # Invalid values outside the declared panel do not enter the sort.
    mask[:, 0] = False
    np.testing.assert_array_equal(perturbed_order(a,b,mask), [[1,0],[1,0]])
    np.testing.assert_array_equal(numpy_perturbed_order(a,b,mask), [[1,0],[1,0]])


def test_recovery_rejects_invalid_generated_order_before_indexing_observations(monkeypatch):
    from numpyro import handlers
    from f1rank.overall import recovery
    # A finite but deliberately overflowing context coefficient exercises the
    # actual generator and its consumer; no model is fitted in this fixture.
    invalid = handlers.substitute(recovery.model, data={"overall_grid_base":jnp.asarray(1000.)})
    monkeypatch.setattr(recovery, "model", invalid)
    with pytest.raises(ValueError, match="Invalid generated order"):
        recovery.generate("connected", 41)
