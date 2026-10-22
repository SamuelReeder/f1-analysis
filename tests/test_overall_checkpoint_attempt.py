"""Complete numeric stage binding, interrupted sealing, and fixed retry policy."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from analysis.checkpoint_archive import finalize, inventory
from analysis.checkpoint_attempt import (
    _artifact, _compare_checkpoint_archive, _validate_report, attempt_key,
    execute_attempt, validate_stage,
)
from analysis.checkpoint_attempt.controller import run_ladder, settings_inventory
from analysis.checkpoint_execution import STAGES, TerminalExecutionFailure
from f1rank.artifacts import RETRY
from f1rank.overall.checkpoints import Checkpoints, canonical, digest
from f1rank.overall.fitting import FIT_SETTINGS
from f1rank.overall.bounded_diagnostics import check
from test_overall_checkpoints import identity, samples, state


def fixture(path):
    ident = identity()
    spec = dict(settings=ident['settings'], chunk_draws=2, chain_method='parallel',
                catalog=dict(field=['a','b'], field_entries=[0,1]), inventory={})
    # The reusable checkpoint fixture needs its own full settings, matching a
    # four-chain diagnostic workload for the end-to-end worker test below.
    ident.update(chunk_draws=spec['chunk_draws'], chain_method=spec['chain_method'])
    with Checkpoints(path / 'preparation', ident) as saved:
        saved.append(state())
        posterior, extras = samples(6)
        saved.append(state(6), posterior=posterior, extras=extras)
        spec['inventory'] = inventory(saved)
    ident['run_key'] = attempt_key(spec)
    with Checkpoints(path / 'checkpoints', ident) as saved:
        saved.append(state())
        saved.append(state(6), posterior=posterior, extras=extras)
    finalize(path / 'checkpoints', path / 'archive', identity=ident,
             expected_inventory=spec['inventory'], verify_inputs=lambda:None)
    return spec


def test_sealed_recovery_compares_complete_checkpoint_bytes(tmp_path):
    spec = fixture(tmp_path)
    _artifact(tmp_path,spec)
    _compare_checkpoint_archive(tmp_path,spec)
    # Replace valid numeric bytes AND their manifest hash: byte validation alone
    # no longer suffices, but reconciliation must still reject changed draws.
    path = tmp_path / 'archive/posterior.npz'
    with np.load(path,allow_pickle=False) as saved:
        post={n:saved[n] for n in saved.files}
    post[next(iter(post))].flat[-1] += 1
    np.savez_compressed(path,**post)
    manifest_path = tmp_path / 'archive/manifest.json'
    manifest = json.loads(manifest_path.read_bytes())
    manifest['files']['posterior.npz'] = digest(path)
    manifest_path.write_bytes(canonical(manifest))
    _artifact(tmp_path,spec)
    with pytest.raises(ValueError,match='differs from its checkpoints'):
        _compare_checkpoint_archive(tmp_path,spec)


@pytest.mark.parametrize('change',['run_key','settings','inventory','tip'])
def test_changed_attempt_or_checkpoint_binding_rejected(tmp_path,change):
    spec = fixture(tmp_path)
    if change=='run_key': spec['catalog']['field'][0]='different'
    elif change=='settings': spec['settings']=dict(spec['settings'],samples=7)
    elif change=='inventory': spec['inventory']['posterior'].pop('beta')
    else:
        path=tmp_path/'archive/manifest.json';record=json.loads(path.read_bytes())
        record['checkpoint_tip_sha256']='0'*64;path.write_bytes(canonical(record))
    with pytest.raises(ValueError): _artifact(tmp_path,spec)


@pytest.mark.parametrize('failed',[False,True])
def test_gate_validator_keeps_original_decisions_and_rejects_edited_metrics(failed):
    rng=np.random.default_rng(91)
    post=dict(skill=rng.normal(size=(4,128,2)),empty=np.empty((4,128,0)),
              car=np.ones((4,128,1)))
    if failed:post['skill'][0]+=8
    report=check(post,0)
    fields={n:dict(shape=list(a.shape),dtype=a.dtype.str) for n,a in post.items()}
    settings=dict(chains=4,samples=128)
    _validate_report(report,fields,settings,0)
    changed=copy.deepcopy(report);changed['converged']=not report['converged']
    with pytest.raises(ValueError):_validate_report(changed,fields,settings,0)
    changed=copy.deepcopy(report);changed['parameters']=[]
    with pytest.raises(ValueError):_validate_report(changed,fields,settings,0)
    changed=copy.deepcopy(report);changed['parameters'][0]['passed']=not changed['parameters'][0]['passed']
    with pytest.raises(ValueError):_validate_report(changed,fields,settings,0)


def test_fixed_retry_inventory_uses_exact_original_settings_and_seeds():
    registration=dict(fit_settings=FIT_SETTINGS,retry=[list(row) for row in RETRY],
                      studies=['fixed'],purpose='numerical_profile')
    result=settings_inventory(registration,'fixed')
    assert result==[dict(FIT_SETTINGS,warmup=800,samples=800,seed=0),
        dict(FIT_SETTINGS,warmup=1600,samples=1600,seed=1),
        dict(FIT_SETTINGS,warmup=2400,samples=3200,seed=2),
        dict(FIT_SETTINGS,warmup=2400,samples=3200,seed=3,target_accept=.98)]
    with pytest.raises(ValueError):settings_inventory(dict(registration,retry=[]),'fixed')


@pytest.mark.parametrize('decisions',[[True],[False,True],[False,False,False,False]])
def test_ladder_advances_only_after_complete_failed_diagnostics(tmp_path,decisions):
    registration=dict(fit_settings=FIT_SETTINGS,retry=[list(row) for row in RETRY],
                      studies=['fixed'],purpose='numerical_profile')
    calls=[]
    def one(path,spec,request,**kwargs):
        calls.append(spec['settings'])
        decision=decisions[len(calls)-1]
        return dict(stages=[dict(verified_summary=dict(converged=decision,diagnostics_sha256='a'*64))])
    result=run_ladder(tmp_path,registration,'fixed',specification_for=lambda s:dict(settings=s),
        request_for=lambda path,spec:{},verify_inputs=lambda:None,execute_one=one)
    assert len(calls)==len(decisions)
    assert result['state']==('numerically_converged' if decisions[-1] else 'failed_convergence')
    assert not result['publication_allowed']
    calls.clear()
    assert run_ladder(tmp_path,registration,'fixed',specification_for=lambda s:dict(settings=s),
        request_for=lambda path,spec:{},verify_inputs=lambda:None,execute_one=one)==result


def test_execution_crash_cannot_advance_or_rerun_the_ladder(tmp_path):
    registration=dict(fit_settings=FIT_SETTINGS,retry=[list(row) for row in RETRY],
                      studies=['fixed'],purpose='numerical_profile')
    calls=[]
    def fail(*args,**kwargs):calls.append(True);raise TerminalExecutionFailure('fixture crash')
    args=dict(specification_for=lambda s:dict(settings=s), request_for=lambda path,spec:{},
              verify_inputs=lambda:None,execute_one=fail)
    with pytest.raises(TerminalExecutionFailure):run_ladder(tmp_path,registration,'fixed',**args)
    assert len(calls)==1
    with pytest.raises(TerminalExecutionFailure):run_ladder(tmp_path,registration,'fixed',**args)
    assert len(calls)==1


def test_separate_real_sampler_archive_and_diagnostic_workers(tmp_path):
    settings=dict(FIT_SETTINGS,warmup=16,samples=12,chains=4,seed=37,max_tree_depth=4)
    axes=[4,12]
    spec=dict(settings=settings,chunk_draws=7,chain_method='vectorized',
        catalog=dict(field=['a','b'],field_entries=[0,1]),inventory=dict(
            posterior={n:dict(shape=[*axes,2],dtype='<f8') for n in ('skill','overall_race_skill')},
            extras=dict(diverging=dict(shape=axes,dtype='|b1'),num_steps=dict(shape=axes,dtype='<i8'),
                        accept_prob=dict(shape=axes,dtype='<f8'))))
    directory=tmp_path/'attempt';definition=tmp_path/'spec.json';definition.write_bytes(canonical(spec))
    worker=tmp_path/'worker.py'
    worker.write_text('''import json,sys\nfrom pathlib import Path\nfrom analysis.checkpoint_attempt import sample_stage,diagnostic_stage\ndirectory=Path(sys.argv[1]);spec=json.loads(Path(sys.argv[2]).read_bytes())\nif sys.argv[3]=='sample_and_archive':\n import numpyro\n import numpyro.distributions as dist\n def model():\n  numpyro.sample('skill',dist.Normal(0,1).expand([2]).to_event(1))\n  numpyro.sample('overall_race_skill',dist.Normal(0,1).expand([2]).to_event(1))\n sample_stage(directory,spec,model=model,args=(),kwargs={},verify_inputs=lambda:None)\nelse:\n diagnostic_stage(directory,spec,verify_inputs=lambda:None)\n''')
    env=dict(PATH='/usr/bin:/bin',PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        JAX_PLATFORMS='cpu',JAX_ENABLE_X64='true',JAX_ENABLE_COMPILATION_CACHE='false',
        XLA_FLAGS='--xla_force_host_platform_device_count=1',
        OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    request=dict(attempt_key=attempt_key(spec),working_directory=str(Path(__file__).resolve().parents[1]),
        stages=[dict(name=name,command=[sys.executable,str(worker),str(directory),str(definition),name],
                     environment=env,result=name+'/result.json') for name in STAGES])
    receipt=execute_attempt(directory,spec,request,verify_inputs=lambda:None)
    assert receipt['state']=='complete'
    assert not receipt['stages'][-1]['verified_summary']['converged']
    children=receipt['invocations'][0]['children']
    assert len(children)==2 and children[0]['process']['pid']!=children[1]['process']['pid']
    assert execute_attempt(directory,spec,request,verify_inputs=lambda:None)==receipt
    result=json.loads((directory/'diagnostics/result.json').read_bytes())
    result['report']['diagnostics']['converged']=True
    with pytest.raises(ValueError):validate_stage(directory,spec,'diagnostics',result)
