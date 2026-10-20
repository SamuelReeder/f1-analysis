import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis import overall_stable_profile as profile


@pytest.mark.parametrize('change', [None, 'case', 'seed', 'gradient', 'collector', 'source', 'exit'])
def test_only_complete_matching_preflight_can_authorize_new_profile(change):
    p = dict(studies=['case'], synthetic={'case':dict(seed=10)}, model_snapshot=dict(key='model'))
    row = dict(case='case', exit_code=0, state='complete')
    summary = dict(all_processed=True, all_complete=True, model_inputs_key='model', studies=[row])
    record = dict(state='complete', model_inputs_key='model', generation_seed=10,
        all_gradient_coordinates_finite=True, original_collector_inventory_identical=True,
        complete_collector_matches_postprocessing=True, sites={'site':{}})
    records = {'case':record}
    if change == 'case':records['unexpected'] = record
    elif change == 'seed':record['generation_seed'] = 11
    elif change == 'gradient':record['all_gradient_coordinates_finite'] = False
    elif change == 'collector':record['complete_collector_matches_postprocessing'] = False
    elif change == 'source':record['model_inputs_key'] = 'changed'
    elif change == 'exit':row['exit_code'] = -9
    if change:
        with pytest.raises(ValueError):profile.validate_prerequisites(p, summary, records)
    else:
        profile.validate_prerequisites(p, summary, records)


@pytest.mark.parametrize('change', [None, 'missing', 'extra', 'axis', 'dtype', 'earlier', 'empty'])
def test_every_attempt_retains_complete_registered_fields_and_axes(change):
    sites = {'a':dict(shape=[], dtype='<f4'), 'b':dict(shape=[3], dtype='<f4')}
    attempt = dict(chains=4, samples=800,
        posterior_sites={'a':dict(shape=[4, 800], dtype='<f4'), 'b':dict(shape=[4, 800, 3], dtype='<f4')})
    receipt = dict(attempts=[copy.deepcopy(attempt), copy.deepcopy(attempt)])
    fields = receipt['attempts'][-1]['posterior_sites']
    if change == 'missing':del fields['a']
    elif change == 'extra':fields['c'] = fields['a']
    elif change == 'axis':fields['b']['shape'][1] -= 1
    elif change == 'dtype':fields['a']['dtype'] = '<f8'
    elif change == 'earlier':del receipt['attempts'][0]['posterior_sites']['b']
    elif change == 'empty':receipt['attempts'] = []
    if change:
        with pytest.raises(ValueError):profile.validate_attempt_inventories(receipt, sites)
    else:
        profile.validate_attempt_inventories(receipt, sites)


@pytest.mark.parametrize('failure', [None, 'killed', 'inventory'])
def test_controller_keeps_all_cases_and_rejects_missing_inventory(tmp_path, monkeypatch, failure):
    p = dict(id='profile-test', studies=['first', 'middle', 'last'])
    path = tmp_path/'registration.json'; path.write_text('{}')
    (tmp_path/'outputs/championship/development').mkdir(parents=True)
    monkeypatch.setattr(profile, 'ROOT', tmp_path)
    monkeypatch.setattr(profile, 'checked', lambda _:p)
    monkeypatch.setattr(profile, 'actual_service', lambda _: {})
    calls = []
    def run(command, **kwargs):
        case = command[command.index('--case')+1]; calls.append(case)
        output = Path(command[command.index('--output')+1]); output.mkdir()
        failed = case == 'middle' and failure is not None
        if not failed or failure != 'killed':
            (output/'result.json').write_text(json.dumps(dict(state='converged', posterior_inventory_verified=not failed)))
        Path(command[command.index('-o')+1]).write_text('fixture resources')
        return SimpleNamespace(returncode=-9 if failed and failure == 'killed' else 0)
    monkeypatch.setattr(profile.subprocess, 'run', run)
    result = profile.execute(path, 'fixture.service')
    assert calls == p['studies'] and result['all_processed']
    assert result['all_converged'] == (failure is None)
    assert not result['full_recovery_complete'] and not result['sbc_complete'] and not result['publication_allowed']
    with pytest.raises(FileExistsError):profile.execute(path, 'fixture.service')
