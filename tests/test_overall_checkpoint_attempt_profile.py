"""The model binding cannot change the registered numerical workload or inventory."""
import copy
import json
from pathlib import Path

import pytest

from analysis.checkpoint_attempt import profile
from analysis.checkpoint_attempt.controller import settings_inventory
from f1rank.overall.fitting import committed_registration

ROOT=Path(__file__).resolve().parents[1]


def context():
    parent=json.loads((ROOT/profile.PARENT).read_bytes())
    previous=committed_registration(ROOT/profile.base.REGISTRATION)
    p=profile.definition(parent,previous)
    p['catalogs']={'connected':dict(field=['a','b'],field_entries=[0,1])}
    return parent,previous,p


def test_complete_profile_preserves_original_designs_full_settings_and_gates():
    parent,previous,p=context()
    for name in ('studies','synthetic','fit_settings','retry','diagnostics','inference_method',
                 'original_model_inputs_key'):
        if name=='original_model_inputs_key':assert p[name]==previous['registered_model_inputs_key']
        else:assert p[name]==previous[name]
    assert p['registered_model_inputs_key']==profile.snapshot()['key']
    assert p['chunk_draws']==parent['chunk_draws'] and p['chain_method']==parent['chain_method']
    assert p['environments']['sample_and_archive']==parent['environments']['gpu64_a']
    assert p['environments']['diagnostics']==parent['environments']['cpu64_a']
    assert not p['publication_allowed'] and not p['end_to_end_refresh_benchmark']


def test_specification_keeps_every_original_collector_field_at_full_draw_axes():
    _,_,p=context()
    settings=settings_inventory(p,'connected')[0]
    spec=profile.specification(ROOT/profile.PARENT,p,'connected',settings)
    prepared=json.loads((ROOT/spec['prepared']).read_bytes())
    assert set(spec['inventory']['posterior'])==set(prepared['collector_inventory'])
    for name,info in spec['inventory']['posterior'].items():
        assert info['shape']==[settings['chains'],settings['samples'],*prepared['collector_inventory'][name]['shape']]
        assert info['dtype']=='<f8'
    assert set(spec['inventory']['extras'])=={'diverging','num_steps','accept_prob'}
    assert spec['catalog']==p['catalogs']['connected']


def test_worker_request_uses_explicit_isolated_environments_and_exact_retry_index(tmp_path):
    _,_,p=context()
    for index,settings in enumerate(settings_inventory(p,'connected'),1):
        spec=profile.specification(ROOT/profile.PARENT,p,'connected',settings)
        request=profile.request(tmp_path,spec,ROOT/profile.PARENT,p)
        for stage in request['stages']:
            assert stage['command'][0]==p['interpreters'][stage['name']]
            assert stage['command'][stage['command'].index('--attempt')+1]==str(index)
            assert stage['environment']==dict(p['environments'][stage['name']],PATH=p['path'],PYTHONUNBUFFERED='1')
            assert not set(stage['environment'])&set(p['unset_environment'])
    changed=copy.deepcopy(spec);changed['settings']['seed']+=20
    with pytest.raises(ValueError):profile.request(tmp_path,changed,ROOT/profile.PARENT,p)


@pytest.mark.parametrize('active,pid,stopped',[('inactive','0',True),('failed','0',True),('active','17',False)])
def test_recovery_requires_authoritative_previous_service_state(monkeypatch,active,pid,stopped):
    from analysis.checkpoint_attempt import runner
    seen=[]
    monkeypatch.setattr(runner,'require_stopped',lambda owner:seen.append(owner))
    monkeypatch.setattr(runner.subprocess,'check_output',lambda command,**kwargs:f'ActiveState={active}\nMainPID={pid}\n')
    invocation=dict(unit='f1-original.service',owner=dict(pid=17))
    if stopped:runner._prior_stopped(invocation)
    else:
        with pytest.raises(RuntimeError,match='authoritatively terminal'):runner._prior_stopped(invocation)
    assert seen==[invocation['owner']]


def test_model_worker_rejects_a_different_case_attempt_directory_before_device_initialization(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from analysis.checkpoint_attempt import worker
    _,_,p=context()
    monkeypatch.setattr(profile,'checked',lambda path:(p,{}))
    with pytest.raises(ValueError,match='unique registered'):
        worker.worker(SimpleNamespace(protocol=ROOT/profile.PARENT,case='connected',attempt=1,
                                      stage='sample_and_archive',directory=tmp_path))
