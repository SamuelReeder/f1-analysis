"""Whole-inventory comparisons and accounting, without running a fit."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from analysis import overall_batched_rotation_audit as audit


def fixture():
    arrays = {'state/empty':np.zeros((0,), dtype='f4'),
        'coordinate/vector':np.arange(6, dtype='f4').reshape(2, 3),
        'gradient/vector':np.ones((2, 3), dtype='f4'),
        'factor/observation':np.array([-.3, -.7], dtype='f4'),
        'potential_energy':np.array(2., dtype='f4')}
    kinds = ['state', 'coordinate', 'gradient', 'density', 'density']
    inventory = {k:dict(kind=t, shape=list(v.shape), dtype=v.dtype.str) for (k,v),t in zip(arrays.items(),kinds)}
    record = dict(case='fixture', protocol_sha256='fixed-workload', arrays_key='a', parameters_key='p',
        truth_key='t', control_rotation_sha256='r', control_log_abs_det=0., runtime={'fixture':True},
        parameter_coordinates=6, payload_inventory=inventory)
    p = dict(density_tolerance=dict(atol=.005, rtol=.000002),
        state_tolerance=dict(atol=.000003, rtol=.000003), gradient_tolerance=dict(atol=.000003, rtol=.000003))
    return dict(record, method='original'), dict(record, method='batched'), arrays, copy.deepcopy(arrays), p


def test_complete_arrays_and_derivatives_including_empty_sites_are_checked():
    a,b,left,right,p = fixture()
    result = audit.compare_records(a,b,left,right,p)
    assert {r['site'] for r in result} == set(left)
    assert sum(r['coordinates'] for r in result if r['kind']=='gradient') == a['parameter_coordinates']
    assert all(r['max_absolute_discrepancy']==0. for r in result)
    assert all(r['passed'] for r in result)


def test_a_failed_case_preserves_every_numeric_discrepancy_and_later_checks():
    a,b,left,right,p = fixture()
    right['coordinate/vector'][0,0] = np.finfo('f4').tiny
    right['gradient/vector'][1,2] += .01
    right['factor/observation'][1] += .1
    with pytest.raises(audit.ComparisonFailure) as caught:
        audit.compare_records(a,b,left,right,p)
    rows = caught.value.checks
    assert [r['site'] for r in rows] == list(left)
    assert {r['site'] for r in rows if not r['passed']} == {
        'coordinate/vector', 'gradient/vector', 'factor/observation'}
    assert next(r for r in rows if r['site']=='potential_energy')['passed']
    assert next(r for r in rows if r['site']=='state/empty')['passed']
    json.dumps(rows, allow_nan=False)


def test_nonfinite_and_wrong_shape_arrays_do_not_hide_other_comparison_failures():
    a,b,left,right,p = fixture()
    right['coordinate/vector'] = np.zeros(6, dtype='f4')
    right['gradient/vector'][1,2] = np.nan
    right['factor/observation'][1] += .1
    with pytest.raises(audit.ComparisonFailure) as caught:
        audit.compare_records(a,b,left,right,p)
    assert len(caught.value.checks) == len(left)
    assert sum(not r['passed'] for r in caught.value.checks) == 3
    json.dumps(caught.value.checks, allow_nan=False)


@pytest.mark.parametrize('change', ['identity', 'runtime', 'case', 'method', 'protocol', 'missing',
    'extra', 'shape', 'dtype', 'nan', 'gradient', 'factor', 'coordinate', 'inventory'])
def test_incomplete_or_changed_probability_and_derivative_payloads_fail(change):
    a,b,left,right,p = fixture()
    if change=='identity': b['parameters_key']='changed'
    elif change=='runtime': b['runtime']={}
    elif change=='case': b['case']='other'
    elif change=='method': b['method']='original'
    elif change=='protocol': b['protocol_sha256']='other'
    elif change=='missing': right.pop('gradient/vector')
    elif change=='extra': right['extra']=np.ones(1)
    elif change=='shape': right['gradient/vector']=np.ones(6, dtype='f4')
    elif change=='dtype': right['gradient/vector']=right['gradient/vector'].astype('f8')
    elif change=='nan': right['gradient/vector'][1,2]=np.nan
    elif change=='gradient': right['gradient/vector'][1,2]+=.01
    elif change=='factor': right['factor/observation'][1]+=.1
    elif change=='coordinate': right['coordinate/vector'][0,0]=np.finfo('f4').tiny
    else:
        b['payload_inventory']=copy.deepcopy(b['payload_inventory'])
        b['payload_inventory'].pop('gradient/vector')
    with pytest.raises((ValueError, AssertionError)):
        audit.compare_records(a,b,left,right,p)


@pytest.mark.parametrize('failed_return', [1, 137, 0])
def test_every_full_design_and_both_arms_still_accounted_for_after_failure(monkeypatch, tmp_path, failed_return):
    studies=[dict(case=c, seed=i) for i,c in enumerate(
        ('connected','confounded','sparse_rookie','dominant_teammate','era_boundary','real_network'))]
    p=dict(id='audit-fixture', studies=studies)
    path=tmp_path/'protocol.json'; path.write_text(json.dumps(p))
    (tmp_path/'outputs/championship/development').mkdir(parents=True)
    (tmp_path/'outputs/championship/cache').mkdir()
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    monkeypatch.setattr(audit,'checked_protocol',lambda path:p)
    calls=[]

    def run(command, **kwargs):
        calls.append(command)
        case=command[command.index('--case')+1]
        method=command[command.index('--method')+1]
        out=Path(command[command.index('--output')+1]); out.mkdir()
        Path(command[command.index('-o')+1]).write_text('resource fixture')
        kwargs['stdout'].write('orchestration fixture; no numerical evaluation')
        assert all(kwargs['env'][k]==v for k,v in audit.ENVIRONMENT.items())
        failed=case=='confounded' and method=='original'
        if not failed: (out/'result.json').write_text(json.dumps(dict(state='complete')))
        return SimpleNamespace(returncode=failed_return if failed else 0)

    monkeypatch.setattr(audit.subprocess,'run',run)
    result=audit.execute(path)
    assert result['all_processed'] and not result['all_complete']
    assert [(r['case'],r['method']) for r in result['rows']]==[
        (s['case'],m) for s in studies for m in (*audit.METHODS,'compare')]
    assert [(r['case'],r['method']) for r in result['rows'] if r['state']=='failed']==[('confounded','original')]
    for command in calls:
        assert command[:3]==['systemd-run','--user','--scope']
        assert '--slice=f1.slice' in command and 'MemoryMax=3G' in command and 'CPUQuota=200%' in command
        i=command.index('/usr/bin/nice'); assert command[i:i+3]==['/usr/bin/nice','-n','19']
    assert not result['mcmc_run'] and not result['publication_allowed']
    with pytest.raises(FileExistsError): audit.execute(path)


def frozen_fixture(monkeypatch, tmp_path):
    from analysis import overall_recovery_preflight
    previous=dict(studies=[dict(case='connected',seed=42)], source_sha256={}, model_source=dict(inputs={}))
    monkeypatch.setattr(overall_recovery_preflight,'checked_protocol',lambda path:copy.deepcopy(previous))
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    sources={}
    for name in audit.AUDIT_SOURCES:
        target=tmp_path/name; target.parent.mkdir(parents=True,exist_ok=True)
        content=Path(audit.__file__).read_bytes() if name=='analysis/overall_batched_rotation_audit.py' else b'source fixture'
        target.write_bytes(content); sources[name]=audit.sha(target)
    inputs={}
    for name in (audit.PREFLIGHT,f'{audit.PREFLIGHT_OUTPUT}/connected/result.json'):
        target=tmp_path/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text('{}'); inputs[name]=audit.sha(target)
    p=dict(id='fixed-audit-fixture', status='frozen', purpose='unchanged_complete_model_batched_control_evaluation',
        mcmc_allowed=False, publication_allowed=False, methods=list(audit.METHODS),
        resources=copy.deepcopy(audit.RESOURCES), environment=copy.deepcopy(audit.ENVIRONMENT),
        repeat_evaluations=audit.REPEATS, density_tolerance=dict(atol=.005,rtol=.000002),
        state_tolerance=dict(atol=.000003,rtol=.000003), gradient_tolerance=dict(atol=.000003,rtol=.000003),
        preflight_protocol=audit.PREFLIGHT, preflight_output=audit.PREFLIGHT_OUTPUT,
        studies=copy.deepcopy(previous['studies']), source_sha256=sources, input_sha256=inputs)
    monkeypatch.setattr(audit.subprocess,'check_output',lambda command,**kwargs:
        (tmp_path/command[-1].removeprefix('HEAD:')).read_bytes())
    return p,tmp_path/'workload.json'


def test_unchanged_committed_workload_is_required_before_computation(monkeypatch,tmp_path):
    p,path=frozen_fixture(monkeypatch,tmp_path)
    path.write_text(json.dumps(p))
    assert audit.checked_protocol(path)==p
    monkeypatch.setattr(audit.subprocess,'check_output',lambda *a,**kw:b'older committed workload')
    with pytest.raises(ValueError,match='Commit'):
        audit.checked_protocol(path)


@pytest.mark.parametrize('change',['status','mcmc','publication','method','repeats','density','state','gradient',
    'resource','environment','generation','study_inventory','source_inventory','input_inventory',
    'source_bytes','input_bytes','loaded_source'])
def test_changed_settings_inputs_and_required_cases_cannot_silently_enter_the_audit(monkeypatch,tmp_path,change):
    p,path=frozen_fixture(monkeypatch,tmp_path)
    if change=='status': p['status']='proposal'
    elif change=='mcmc': p['mcmc_allowed']=True
    elif change=='publication': p['publication_allowed']=True
    elif change=='method': p['methods']=['batched']
    elif change=='repeats': p['repeat_evaluations']+=1
    elif change in ('density','state','gradient'): p[change+'_tolerance']['atol']*=2
    elif change=='resource': p['resources']['MemoryMax']='6G'
    elif change=='environment': p['environment']['JAX_ENABLE_X64']='true'
    elif change=='generation': p['studies'][0]['seed']+=1
    elif change=='study_inventory': p['studies']=[]
    elif change=='source_inventory': p['source_sha256'].pop('analysis/overall_batched_rotation.py')
    elif change=='input_inventory': p['input_sha256'].pop(f'{audit.PREFLIGHT_OUTPUT}/connected/result.json')
    elif change=='source_bytes': (tmp_path/'analysis/overall_batched_rotation.py').write_text('changed source')
    elif change=='input_bytes': (tmp_path/f'{audit.PREFLIGHT_OUTPUT}/connected/result.json').write_text('changed input')
    else: p['source_sha256']['analysis/overall_batched_rotation_audit.py']='a'*64
    path.write_text(json.dumps(p))
    with pytest.raises(ValueError): audit.checked_protocol(path)

