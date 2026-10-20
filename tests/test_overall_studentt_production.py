"""Production migration, sampler identity and complete frozen primitive inventory."""
import ast
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpyro.distributions import StudentT

from analysis import overall_studentt_stable as prototype
from f1rank.overall import studentt

ROOT = Path(__file__).resolve().parents[1]


def test_production_keeps_every_verified_arithmetic_definition_and_constant():
    def definitions(path):
        tree = ast.parse(path.read_text())
        return {node.name:ast.dump(node) for node in tree.body if isinstance(node, ast.FunctionDef)}
    old = definitions(ROOT/'analysis/overall_studentt_stable.py')
    new = definitions(ROOT/'f1rank/overall/studentt.py')
    assert set(new) == set(old)-{'normalizer_only_log_prob', 'stable_studentt_calculation'}
    for name, definition in new.items():
        assert definition == old[name]
    for name in ('GAMMA_RATIO_MIN_B', 'STIRLING_COEFFICIENTS', 'NEXT_STIRLING_COEFFICIENT',
                 'TAIL_SERIES_MAX_Q', 'TAIL_SERIES_LAST_POWER'):
        assert getattr(studentt, name) == getattr(prototype, name)
    from f1rank.overall.inference import METHOD
    from f1rank.overall.provenance import snapshot
    assert METHOD.endswith(studentt.ARITHMETIC_METHOD)
    assert 'f1rank/overall/studentt.py' in snapshot()['inputs']


def test_complete_primitive_values_and_gradients_match_the_frozen_research_result():
    precision = '64' if jax.config.x64_enabled else '32'
    result = json.loads((ROOT/f'outputs/championship/development/overall-studentt-stable-v1/primitive{precision}/result.json').read_bytes())
    coords = jnp.asarray([[r['log_df'], 0., 0.] for r in result['rows']])
    observations = jnp.asarray([r['residual'] for r in result['rows']])
    def f(p, value):
        return studentt.StableStudentT(jnp.exp(p[0]), p[1], jnp.exp(p[2])).log_prob(value)
    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(f), in_axes=(0, 0)))(coords, observations)
    actual = np.column_stack([np.asarray(values), np.asarray(gradients)])
    np.testing.assert_array_equal(actual, result['methods']['stable']['values'])


def test_distribution_sampling_broadcast_support_and_pytree_are_preserved():
    df = jnp.array([1., 4., 32.]); loc = jnp.array([-2., 0., 3.]); scale = jnp.array([.5, 1., 2.])
    original, stable = StudentT(df, loc, scale), studentt.StableStudentT(df, loc, scale)
    assert studentt.StableStudentT.sample is StudentT.sample
    assert stable.support is original.support and stable.arg_constraints == original.arg_constraints
    leaves, treedef = jax.tree_util.tree_flatten(stable)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert type(restored) is studentt.StableStudentT
    key = jax.random.PRNGKey(23)
    np.testing.assert_array_equal(original.sample(key, (7,)), stable.sample(key, (7,)))
    np.testing.assert_array_equal(stable.log_prob(loc), jax.jit(lambda d:d.log_prob(loc))(restored))
    assert stable.expand((2, 3)).to_event(1).log_prob(jnp.zeros((2, 3))).shape == (2,)
    with pytest.raises(ValueError):
        studentt.StableStudentT(-1., validate_args=True)


@pytest.mark.parametrize('likelihood', ['t', 'mixture', 'split_t'])
def test_legacy_default_stays_original_and_explicit_factory_changes_only_arithmetic(likelihood):
    from numpyro import handlers
    from f1rank.model import model
    from f1rank.overall.recovery import synthetic_design
    q = synthetic_design('connected').qualifying.arrays()
    def trace(**kwargs):
        return handlers.trace(handlers.seed(model, jax.random.PRNGKey(37))).get_trace(q, likelihood=likelihood, **kwargs)
    original, changed = trace(), trace(student_t=studentt.StableStudentT)
    assert original.keys() == changed.keys()
    for name in original:
        if original[name]['type'] in ('sample', 'deterministic'):
            np.testing.assert_array_equal(original[name]['value'], changed[name]['value'], err_msg=name)
    assert type(original['car_step_z']['fn'].base_dist) is StudentT
    assert type(changed['car_step_z']['fn'].base_dist) is studentt.StableStudentT


def test_overall_trace_uses_corrected_distribution_for_prior_qualifying_and_laps():
    from numpyro import handlers
    from f1rank.overall.model import model
    from f1rank.overall.recovery import synthetic_design
    d = synthetic_design('connected')
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(37))).get_trace(
        d.qualifying.arrays(), d.laps, d.outcomes, generate=True)
    for name in ('car_step_z', 'y', 'generated_lap_innovation'):
        distribution = trace[name]['fn']
        while hasattr(distribution, 'base_dist'):
            distribution = distribution.base_dist
        assert type(distribution) is studentt.StableStudentT, name
