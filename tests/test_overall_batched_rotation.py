"""Every original coordinate and derivative must survive batching unchanged."""
import copy

import numpy as np
import pytest

from analysis.overall_batched_rotation import prepare
from analysis.overall_control_rotation import ControlRotation


def original_map():
    shapes = dict(overall_intercept=(2,), overall_traffic=(2, 3), empty=(0,), scalar=())
    assignments = [
        [('overall_traffic', (1, 2)), ('overall_intercept', (0,))],
        [('scalar', ())],
        [('overall_traffic', (0, 2)), ('overall_traffic', (1, 0)), ('overall_traffic', (0, 0))],
        [('overall_intercept', (1,)), ('overall_traffic', (1, 1))],
        [('overall_traffic', (0, 1))],
    ]
    blocks, start = [], 0
    for i, coordinates in enumerate(assignments):
        k = len(coordinates)
        if k == 2:
            basis = np.sqrt(.5) * np.array([[1., -1.], [1., 1.]])
        elif k == 3:
            basis = np.eye(k)[[2, 0, 1]]
        else:
            basis = np.array([[-1.]], dtype='f4' if i == 1 else 'f8')
        scales = (np.arange(k) + 1.5).astype(basis.dtype)
        blocks.append(dict(coordinates=coordinates, basis=basis, scales=scales, start=start, stop=start+k))
        start += k
    return ControlRotation(tuple(blocks), shapes, start)


def flattened(rotation, z):
    import jax.numpy as jnp
    return jnp.concatenate([v.reshape(-1) for v in rotation.original(z).values()])


def test_complete_mixed_layout_reconstruction_derivative_and_inverse():
    import jax
    import jax.numpy as jnp
    original = original_map()
    batched = prepare(original)
    z = jnp.linspace(-2., 2., original.size)
    expected, actual = original.original(z), batched.original(z)
    assert list(actual) == list(expected)
    assert batched.fingerprint() == original.fingerprint()
    assert batched.log_abs_det == original.log_abs_det
    for name in expected:
        assert actual[name].shape == expected[name].shape
        assert actual[name].dtype == expected[name].dtype
        np.testing.assert_allclose(actual[name], expected[name], atol=3e-6, rtol=3e-6)
    np.testing.assert_array_equal(batched.rotated(expected), original.rotated(expected))
    np.testing.assert_allclose(batched.rotated(actual), z, atol=3e-6, rtol=3e-6)
    for transform in (jax.jit, jax.jacrev):
        a = transform(lambda v: flattened(original, v))(z)
        b = transform(lambda v: flattened(batched, v))(z)
        np.testing.assert_allclose(b, a, atol=3e-6, rtol=3e-6)


def test_prepared_map_owns_its_layout_and_preserves_recorded_identity():
    import jax.numpy as jnp
    original = original_map()
    batched = prepare(original)
    z = jnp.linspace(-1., 1., original.size)
    expected = np.asarray(flattened(batched, z))
    fingerprint = batched.fingerprint()
    original.blocks[0]['basis'][0, 0] += 1
    original.blocks[1]['scales'][0] += 1
    assert original.fingerprint() != fingerprint
    assert batched.fingerprint() == fingerprint
    np.testing.assert_array_equal(flattened(batched, z), expected)
    with pytest.raises(TypeError): batched.shapes['scalar'] = (1,)
    with pytest.raises(ValueError): batched.groups[0].basis[0, 0, 0] = 10
    with pytest.raises(ValueError): batched.gathers['scalar'][0] = 10


@pytest.mark.parametrize('change', ['latent_overlap', 'omitted_block', 'repeated_coordinate', 'unknown_site',
    'outside_coordinate', 'coordinate_dimension', 'basis_shape', 'nonorthogonal', 'nonfinite_basis',
    'scales_shape', 'nonpositive_scale', 'nonfinite_scale', 'site_shape', 'vector_size'])
def test_invalid_or_incomplete_maps_are_rejected(change):
    original = copy.deepcopy(original_map())
    if change == 'latent_overlap': original.blocks[1]['start'] -= 1
    elif change == 'omitted_block': original = ControlRotation(original.blocks[:-1], original.shapes, original.size)
    elif change == 'repeated_coordinate': original.blocks[1]['coordinates'][0] = original.blocks[0]['coordinates'][0]
    elif change == 'unknown_site': original.blocks[1]['coordinates'][0] = ('missing', ())
    elif change == 'outside_coordinate': original.blocks[0]['coordinates'][0] = ('overall_traffic', (4, 2))
    elif change == 'coordinate_dimension': original.blocks[0]['coordinates'][0] = ('overall_traffic', (1,))
    elif change == 'basis_shape': original.blocks[0]['basis'] = np.zeros((2, 1))
    elif change == 'nonorthogonal': original.blocks[0]['basis'] *= 2
    elif change == 'nonfinite_basis': original.blocks[0]['basis'][0, 0] = np.nan
    elif change == 'scales_shape': original.blocks[0]['scales'] = np.ones(1)
    elif change == 'nonpositive_scale': original.blocks[0]['scales'][0] = 0
    elif change == 'nonfinite_scale': original.blocks[0]['scales'][0] = np.inf
    elif change == 'site_shape': original.shapes['overall_intercept'] = (3,)
    else: original = ControlRotation(original.blocks, original.shapes, original.size+1)
    with pytest.raises((ValueError, AssertionError)):
        prepare(original)


def test_incomplete_vectors_or_original_sites_are_rejected():
    import jax.numpy as jnp
    batched = prepare(original_map())
    with pytest.raises(ValueError): batched.original(jnp.zeros(batched.size-1))
    with pytest.raises(ValueError): batched.rotated({})
    values = {name: np.zeros(shape) for name, shape in batched.shapes.items()}
    values['scalar'] = np.zeros(1)
    with pytest.raises(ValueError): batched.rotated(values)
