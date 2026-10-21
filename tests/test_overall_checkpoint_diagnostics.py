"""Archive-based diagnostics must preserve complete original reports and failures."""
import hashlib

import numpy as np
import pytest

from analysis.checkpoint_diagnostics import diagnose
from f1rank.overall.checkpoints import canonical,digest
from f1rank.overall.bounded_diagnostics import check
from f1rank.overall.inference import contrast_reports


def fixture(path, *, failed=False):
    path.mkdir()
    rng=np.random.default_rng(91);axes=(4,128)
    post=dict(skill=rng.normal(size=(*axes,3)),overall_race_skill=rng.normal(size=(*axes,3)),
              nuisance=rng.normal(size=(*axes,2)),empty=np.empty((*axes,0)))
    if failed:post['nuisance'][0]+=8
    extras=dict(diverging=np.zeros(axes,bool),num_steps=np.full(axes,7,np.int64),
                accept_prob=np.full(axes,.9,np.float64))
    if failed:extras['diverging'][0,0]=True
    settings=dict(chains=4,warmup=32,samples=128,max_tree_depth=10)
    identity=dict(settings=settings,run_key='fixture',runtime=dict(backend='fixture'))
    inventory={group:{n:dict(shape=list(v.shape),dtype=v.dtype.str) for n,v in values.items()}
               for group,values in (('posterior',post),('extras',extras))}
    for group,values in (('posterior',post),('extras',extras)):np.savez_compressed(path/(group+'.npz'),**values)
    record=dict(schema='checkpoint_numeric_archive_v1',state='draws_complete',identity=identity,
        identity_key=hashlib.sha256(canonical(identity)).hexdigest(),inventory=inventory,
        files={name:digest(path/name) for name in ('posterior.npz','extras.npz')},
        convergence_evaluated=False,publication_allowed=False)
    (path/'manifest.json').write_bytes(canonical(record))
    catalog=dict(field=['a','b','c'],field_entries=[0,1,2])
    args=dict(expected_manifest_sha256=digest(path/'manifest.json'),expected_identity_key=record['identity_key'],
        expected_inventory=inventory,catalog=catalog,verify_inputs=lambda:None,temporary_parent=path.parent)
    return post,extras,args


@pytest.mark.parametrize('failed',[False,True])
def test_complete_archive_reports_match_original_in_memory_diagnostics(tmp_path,failed):
    path=tmp_path/'artifact';post,extras,args=fixture(path,failed=failed)
    observed=diagnose(path,**args)
    div=int(extras['diverging'].sum())
    expected=check(post,div)
    expected.update(divergences=div,comparison_reports=contrast_reports(post,args['catalog'],div),
                    tree_depth_limit_fraction=0.,mean_accept_prob=float(extras['accept_prob'].mean()))
    assert observed['diagnostics']==expected
    assert not observed['publication_allowed'] and not observed['attempt_selection_performed']
    assert observed['sampling_runtime']==dict(backend='fixture')
    assert not list(tmp_path.glob('f1-overall-posterior-*'))
    if failed:assert not observed['diagnostics']['converged']


@pytest.mark.parametrize('change',['posterior','extras','manifest','identity','inventory'])
def test_wrong_identity_inventory_or_bytes_rejected(tmp_path,change):
    path=tmp_path/'artifact';_,_,args=fixture(path)
    if change in ('posterior','extras'):
        with (path/(change+'.npz')).open('ab') as stream:stream.write(b'changed')
    elif change=='manifest':(path/'manifest.json').write_text('{}')
    elif change=='identity':args['expected_identity_key']='changed'
    else:args['expected_inventory']={}
    with pytest.raises(ValueError):diagnose(path,**args)


def test_changed_source_during_diagnostics_prevents_result(tmp_path):
    path=tmp_path/'artifact';_,_,args=fixture(path);calls=[]
    def changed():
        calls.append(True)
        if len(calls)>1:raise ValueError('source changed')
    args['verify_inputs']=changed
    with pytest.raises(ValueError,match='source changed'):diagnose(path,**args)
    assert not list(tmp_path.glob('f1-overall-posterior-*'))
