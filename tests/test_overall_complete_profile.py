"""The complete resource profile preserves inputs, draw coverage and failures."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import overall_complete_profile as profile


def registration_inputs():
    cases = ['connected', 'confounded', 'sparse_rookie', 'dominant_teammate', 'era_boundary', 'real_network']
    preflight = dict(studies=[dict(case=c, seed=i) for i, c in enumerate(cases)], model_source=dict(key='model'))
    parent = dict(fit_settings=dict(seed=0), retry=[[1, 1, None]], diagnostics=dict(limit=.01), method=profile.METHOD)
    p = dict(copy.deepcopy(parent), studies=cases,
        synthetic={r['case']: r.copy() for r in preflight['studies']}, registered_model_inputs_key='model',
        source_sha256={'analysis/overall_bounded_diagnostics.py': 'checker'},
        preflight_results={c: c for c in cases}, storage_results={c: c for c in cases})
    audit = dict(state='complete', all_saved_reports_identical=True,
        attempts=[dict(complete_original_report_equal=True)],
        resource=dict(temporary_arrays_removed=True, diagnostic={'converged': True}),
        model_inputs_key='model', sources={'analysis/overall_bounded_diagnostics.py': 'checker'})
    storage = dict(all_processed=True, all_complete=True, studies=[dict(case=c) for c in cases])
    return p, preflight, parent, audit, storage


def test_complete_prerequisites_are_accepted_without_sampling():
    profile.validate_inventory(*registration_inputs())


@pytest.mark.parametrize('change', [
    'missing_case', 'reordered_case', 'changed_truth_seed', 'changed_settings', 'changed_retry',
    'changed_gate', 'changed_coordinates', 'unfinished_audit', 'changed_report', 'hidden_changed_report',
    'missing_attempts', 'unfinished_fixture', 'missing_fixture_diagnostic', 'changed_model',
    'changed_preflight_model', 'changed_checker', 'unfinished_storage', 'missing_storage_case',
    'missing_preflight_result', 'missing_storage_result'])
def test_incomplete_or_changed_prerequisites_are_rejected(change):
    p, preflight, parent, audit, storage = registration_inputs()
    if change == 'missing_case': p['studies'] = p['studies'][:-1]
    elif change == 'reordered_case': p['studies'] = p['studies'][::-1]
    elif change == 'changed_truth_seed': p['synthetic']['sparse_rookie']['seed'] += 1
    elif change == 'changed_settings': p['fit_settings']['seed'] += 1
    elif change == 'changed_retry': p['retry'].append([2, 2, None])
    elif change == 'changed_gate': p['diagnostics']['limit'] += .1
    elif change == 'changed_coordinates': p['method'] = 'changed'
    elif change == 'unfinished_audit': audit['state'] = 'running'
    elif change == 'changed_report': audit['all_saved_reports_identical'] = False
    elif change == 'hidden_changed_report': audit['attempts'][0]['complete_original_report_equal'] = False
    elif change == 'missing_attempts': audit['attempts'] = []
    elif change == 'unfinished_fixture': audit['resource']['temporary_arrays_removed'] = False
    elif change == 'missing_fixture_diagnostic': audit['resource']['diagnostic'] = {}
    elif change == 'changed_model': audit['model_inputs_key'] = 'other'
    elif change == 'changed_preflight_model': preflight['model_source']['key'] = 'other'
    elif change == 'changed_checker': audit['sources']['analysis/overall_bounded_diagnostics.py'] = 'other'
    elif change == 'unfinished_storage': storage['all_complete'] = False
    elif change == 'missing_storage_case': storage['studies'].pop()
    elif change == 'missing_preflight_result': del p['preflight_results']['confounded']
    else: del p['storage_results']['real_network']
    with pytest.raises(ValueError):
        profile.validate_inventory(p, preflight, parent, audit, storage)


@pytest.mark.parametrize('change', [None, 'field', 'extra_field', 'chain', 'draw', 'coordinate', 'dtype'])
def test_all_collector_fields_axes_and_dtypes_are_required(change):
    post = dict(scalar=np.zeros((4, 800), dtype='f4'), vector=np.zeros((4, 800, 3, 2), dtype='f4'))
    sites = dict(scalar=dict(shape=[], dtype='<f4'), vector=dict(shape=[3, 2], dtype='<f4'))
    if change == 'field': del post['scalar']
    elif change == 'extra_field': post['extra'] = post['scalar']
    elif change == 'chain': post['vector'] = post['vector'][:3]
    elif change == 'draw': post['vector'] = post['vector'][:, :-1]
    elif change == 'coordinate': post['vector'] = post['vector'][..., 0]
    elif change == 'dtype': post['vector'] = post['vector'].astype('f8')
    if change:
        with pytest.raises(ValueError): profile.validate_posterior(post, sites, dict(chains=4, samples=800))
    else:
        profile.validate_posterior(post, sites, dict(chains=4, samples=800))


@pytest.mark.parametrize('middle_exit', [0, 1, -9])
def test_complete_controller_continues_after_failed_or_killed_child(tmp_path, monkeypatch, middle_exit):
    parent = tmp_path / 'outputs/championship/development'
    parent.mkdir(parents=True)
    path = tmp_path / 'protocol.json'
    path.write_text('{}')
    p = dict(id='fixed-profile', studies=['first', 'middle', 'last'])
    monkeypatch.setattr(profile, 'ROOT', tmp_path)
    monkeypatch.setattr(profile, 'checked_registration', lambda _: p)
    monkeypatch.setattr(profile, 'actual_service', lambda _: {})
    calls = []
    def run(command, **kwargs):
        case = command[command.index('--case')+1]
        output = parent / p['id'] / case
        output.mkdir()
        code = middle_exit if case == 'middle' else 0
        calls.append(case)
        if code >= 0:
            (output/'result.json').write_text(json.dumps(dict(state='converged' if code == 0 else 'failed')))
        return SimpleNamespace(returncode=code)
    monkeypatch.setattr(profile.subprocess, 'run', run)
    result = profile.execute(path, 'fixture.service')
    assert calls == p['studies']
    assert result['all_processed']
    assert result['all_converged'] == (middle_exit == 0)
    assert result['studies'][1]['exit_code'] == middle_exit
    assert result['studies'][1]['result_present'] == (middle_exit != -9)
    assert result['publication_allowed'] is False
    assert result['full_recovery_complete'] is False
    original = (parent/p['id']/'summary.json').read_bytes()
    with pytest.raises(FileExistsError): profile.execute(path, 'fixture.service')
    assert (parent/p['id']/'summary.json').read_bytes() == original
