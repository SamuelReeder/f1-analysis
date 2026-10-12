"""Known arithmetic and failed-archive boundaries; no inference experiment."""
import hashlib

import numpy as np
import pytest

from analysis.overall_weekend_diagnosis import (
    SCALARS, VECTORS, centre, correlation_matrix, derived_quantities,
    diagnostic_archive, weekend_moments)


def test_weekend_covariance_follows_the_original_linear_map():
    transform = np.array([[2., 0.], [-1., 3.]])
    covariance = transform @ transform.T
    result = weekend_moments(np.array(2.), np.array(3.), np.array(-.5))
    assert result['race_form_marginal_sd'] == pytest.approx(np.sqrt(covariance[1, 1]))
    assert result['form_cross_covariance'] == covariance[0, 1]
    assert result['form_correlation'] == pytest.approx(covariance[0, 1] / np.sqrt(covariance[0, 0]*covariance[1, 1]))
    assert result['independent_race_variance_fraction'] == pytest.approx(.9)
    # A common event-centering projection multiplies both variances and the covariance.
    projection = np.eye(3) - np.ones((3, 3))/3
    joint = np.kron(covariance, projection)
    assert joint[0, 3]/np.sqrt(joint[0, 0]*joint[3, 3]) == pytest.approx(result['form_correlation'])


@pytest.mark.parametrize('sf,sr,b', [(0, 1, 1), (1, 0, 1), (-1, 1, 1), (1, 1, np.nan)])
def test_undefined_weekend_moments_are_not_filled(sf, sr, b):
    with pytest.raises(ValueError):
        weekend_moments(np.array(sf), np.array(sr), np.array(b))


def test_centering_retains_event_membership_and_does_not_mutate():
    values = np.array([[[2., 4., 8., 10., 12.], [3., 3., 9., 9., 12.]]])
    original = values.copy()
    groups = np.array([0, 1, 0, 1, 1])
    expected = np.array([[[-3., -14/3, 3., 4/3, 10/3], [-3., -5., 3., 1., 4.]]])
    np.testing.assert_allclose(centre(values, groups), expected)
    np.testing.assert_array_equal(values, original)


def test_correlations_preserve_constants_as_undefined():
    values = np.array([[1., 3., 7.], [2., 2., 7.], [3., 1., 7.]])
    result = correlation_matrix(values)
    np.testing.assert_allclose(np.array(result['values'], object)[:2, :2].astype(float), [[1, -1], [-1, 1]])
    assert result['constant_columns'] == [2]
    assert result['values'][2] == [None, None, None]
    assert correlation_matrix(values[:, :1])['values'] == [[1.]]


def test_reconstruction_agrees_with_the_actual_joint_model():
    import jax
    from numpyro import handlers
    from f1rank.overall.model import model
    from f1rank.overall.recovery import synthetic_design
    d = synthetic_design('connected')
    q = d.qualifying.arrays()
    trace = handlers.trace(handlers.seed(model, jax.random.PRNGKey(5))).get_trace(q, d.laps, d.outcomes, generate=True)
    saved = {name: np.broadcast_to(np.asarray(trace[name]['value']), (2, 3) + np.shape(trace[name]['value']))
             for name in SCALARS + VECTORS}
    scalar, fields, error = derived_quantities(saved, q, d.outcomes)
    assert all(row['max_absolute_error'] <= row['allowed_roundoff'] for row in error.values())
    np.testing.assert_array_equal(fields['race_form'], saved['overall_day'])
    np.testing.assert_allclose(fields['race_total_location'], saved['overall_race_skill']
        + saved['overall_team'] + saved['overall_day'] + saved['overall_car'][..., d.outcomes['entry_car']])
    expected_car = centre(saved['car'] + saved['car_event'] + saved['car_track'], q['car_event'])
    np.testing.assert_allclose(fields['qualifying_car'], expected_car[..., d.outcomes['entry_car']], atol=2e-6)
    assert (scalar['race_form_marginal_sd'] > 0).all()
    changed = dict(saved, overall_day=saved['overall_day'] + 1)
    with pytest.raises(ValueError, match='does not reproduce'):
        derived_quantities(changed, q, d.outcomes)


def test_diagnostic_archive_validates_unselected_members_and_preserves_failures(tmp_path):
    path = tmp_path / 'failed_1.npz'
    values = np.arange(24.).reshape(2, 3, 4)
    np.savez_compressed(path, selected=values, unused=np.ones((2, 3)))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with diagnostic_archive(path, digest, (2, 3), ('selected',)) as saved:
        assert not saved['selected'].flags.writeable
        np.testing.assert_array_equal(saved['selected'], values)
    with pytest.raises(ValueError, match='changed'):
        with diagnostic_archive(path, '0'*64, (2, 3), ('selected',)):
            pytest.fail('Changed archive must not be read')
    with pytest.raises(ValueError, match='during analysis'):
        with diagnostic_archive(path, digest, (2, 3), ('selected',)):
            path.write_bytes(b'changed after reading')
    np.savez_compressed(path, selected=values, unused=np.full((2, 3), np.nan))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Nonfinite'):
        with diagnostic_archive(path, digest, (2, 3), ('selected',)):
            pytest.fail('Invalid unused values must not be ignored')
