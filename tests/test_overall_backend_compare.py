"""Full-inventory and corruption checks for the prospective backend screen."""
import json
from pathlib import Path

import numpy as np
import pytest

from analysis import overall_backend_compare as backend


def test_point_inventory_includes_all_chains_and_failed_attempt_endpoints():
    study = {'attempts': [dict(attempt=1, chains=2, samples=3), dict(attempt=2, chains=2, samples=5)]}
    assert backend.point_labels(study) == [
        {'kind': 'generating'},
        *[dict(kind='saved_endpoint', attempt=a, chain=c, draw=d)
          for a, last in ((1, 2), (2, 4)) for c in (0, 1) for d in (0, last)]]


def tolerances():
    return dict(state=dict(atol=1e-4, rtol=1e-4), value_and_gradient=dict(atol=1e-3, rtol=1e-3))


def test_reports_every_failing_field_and_coordinate_without_averaging():
    left = {'potential': np.array([0., 3.]), 'gradient/x': np.zeros((2, 3)),
            'state/skill': np.zeros((2, 2)), 'state/index': np.array([1, 2])}
    right = {n:v.copy() for n,v in left.items()}
    right['gradient/x'][0, 2] = 1.; right['gradient/x'][1, 0] = 1.
    right['state/skill'][1, 1] = .01; right['state/index'][0] = 2
    reports = backend.compare_fields(left, right, tolerances())
    assert set(reports) == set(left)
    assert reports['potential']['passed']
    assert reports['gradient/x']['failing_indices'] == [[0, 2], [1, 0]]
    assert reports['state/skill']['failing_indices'] == [[1, 1]]
    assert reports['state/index']['failing_indices'] == [[0]]


@pytest.mark.parametrize('change', ['missing', 'shape', 'dtype', 'nonfinite'])
def test_incomplete_or_incomparable_arrays_are_rejected(change):
    left = {'gradient/x': np.ones((2, 3), np.float32)}
    right = {n:v.copy() for n,v in left.items()}
    if change == 'missing': right = {}
    elif change == 'shape': right['gradient/x'] = right['gradient/x'][:, :2]
    elif change == 'dtype': right['gradient/x'] = right['gradient/x'].astype(np.float64)
    else: right['gradient/x'][1, 2] = np.nan
    with pytest.raises(ValueError): backend.compare_fields(left, right, tolerances())


def test_state_budget_is_separate_and_integer_state_never_uses_tolerance():
    left = {'potential': np.array([0.]), 'gradient/x': np.array([0.]),
            'state/skill': np.array([0.]), 'state/index': np.array([1000000])}
    right = {'potential': np.array([.0005]), 'gradient/x': np.array([.0005]),
             'state/skill': np.array([.0005]), 'state/index': np.array([1000001])}
    reports = backend.compare_fields(left, right, tolerances())
    assert reports['potential']['passed'] and reports['gradient/x']['passed']
    assert not reports['state/skill']['passed'] and not reports['state/index']['passed']


def prepared_fixture(tmp_path, monkeypatch):
    from analysis.overall_network_precision import pack_blocks
    from f1rank.overall.fitting import arrays_digest
    monkeypatch.setattr(backend, 'ROOT', tmp_path)
    monkeypatch.setattr(backend, 'committed', lambda root, name: backend.sha(root/name))
    cache = tmp_path/'outputs/championship/cache/inputs.npz'; cache.parent.mkdir(parents=True)
    study = dict(case='fixture', attempts=[], preflight='original.json')
    blocks = {'qualifying': {'count': 2, 'x': np.array([1., 2.], np.float32)}}
    arrays, scalars = pack_blocks(blocks); z = {'x': np.array([[.2]], np.float32)}
    arrays['z/x'] = z['x']; np.savez_compressed(cache, **arrays)
    original = dict(arrays_key=arrays_digest(blocks), sites={'skill': {'shape': [1]}}, control_rotation_sha256='fixed')
    (tmp_path/'original.json').write_text(json.dumps(original))
    r = dict(state='complete', operation='prepare', case='fixture', protocol_sha256='protocol',
        point_labels=backend.point_labels(study), collector_inventory=original['sites'],
        arrays_key=original['arrays_key'], control_rotation_sha256='fixed',
        cache=str(cache.relative_to(tmp_path)), cache_sha256=backend.sha(cache),
        block_scalars=scalars, parameter_shapes={'x': [1]}, coordinates_key=arrays_digest({'coordinates': z}))
    path = tmp_path/'prepared.json'; path.write_text(json.dumps(r))
    return {'reference_root': str(tmp_path)}, study, path, cache, r


@pytest.mark.parametrize('change', ['none', 'missing_point', 'new_observations', 'missing_site', 'changed_bytes', 'extra_field'])
def test_fixed_preparation_rejects_changed_data_or_incomplete_inventory(tmp_path, monkeypatch, change):
    p, study, path, cache, r = prepared_fixture(tmp_path, monkeypatch)
    if change == 'missing_point': r['point_labels'] = []
    elif change == 'new_observations': r['arrays_key'] = 'other'
    elif change == 'missing_site': r['collector_inventory'] = {}
    elif change == 'changed_bytes':
        with cache.open('ab') as f: f.write(b'changed')
    elif change == 'extra_field':
        with np.load(cache) as a: values = {n:a[n] for n in a.files}
        values['unexpected'] = np.array([1.]); np.savez_compressed(cache, **values)
        r['cache_sha256'] = backend.sha(cache)
    path.write_text(json.dumps(r))
    if change == 'none':
        _, blocks, z = backend.load_prepared(p, study, path, 'protocol')
        np.testing.assert_array_equal(blocks['qualifying']['x'], [1., 2.])
        assert z['x'].shape == (1, 1)
    else:
        with pytest.raises(ValueError): backend.load_prepared(p, study, path, 'protocol')


def test_original_attempt_or_case_cannot_be_silently_dropped():
    parent = dict(studies=['a', 'b'], synthetic={'a': {'seed': 1}, 'b': {'seed': 2}})
    reviewed = {'studies': [{'case': 'a', 'attempts': []}]}
    with pytest.raises(ValueError, match='every original case'): backend.original_studies(parent, reviewed)
