import numpy as np
import pytest
from analysis.checkpoint_model_replay import archive_difference,validate_predecessor


def test_replay_checks_all_fields_and_detects_signed_zero_and_integer_changes(tmp_path):
    left=dict(x=np.array([[0.,-0.]]),count=np.array([[1,2]],dtype=np.int32),empty=np.empty((1,2,0)))
    right={n:v.copy() for n,v in left.items()};right['x'][0,0]=-0.;right['count'][0,1]=3
    for name,values in [('a',left),('b',right)]:np.savez(tmp_path/(name+'.npz'),**values)
    inventory={n:dict(shape=list(v.shape),dtype=v.dtype.str) for n,v in left.items()}
    r=archive_difference(tmp_path/'a.npz',tmp_path/'b.npz',inventory)
    assert set(r['fields'])==set(left) and not r['bitwise_equal']
    assert r['fields']['x']['exact_values'] and not r['fields']['x']['bitwise_equal']
    assert r['fields']['count']['max_absolute']==1 and r['fields']['empty']['bitwise_equal']


@pytest.mark.parametrize('change',['missing','shape','dtype','nonfinite'])
def test_replay_rejects_incomplete_or_invalid_inventory(tmp_path,change):
    a=dict(x=np.ones((2,3)));b=dict(x=a['x'].copy())
    if change=='missing':b={}
    elif change=='shape':b['x']=b['x'][:1]
    elif change=='dtype':b['x']=b['x'].astype(np.float32)
    else:b['x'][0,0]=np.nan
    np.savez(tmp_path/'a.npz',**a);np.savez(tmp_path/'b.npz',**b)
    with pytest.raises(ValueError):archive_difference(tmp_path/'a.npz',tmp_path/'b.npz',{'x':dict(shape=[2,3],dtype='<f8')})


@pytest.mark.parametrize('phase,prior_phase,warmup,draws',[
    ('sampling_stop','warmup_stop',7,0),('resume','sampling_stop',32,7)])
def test_restart_requires_its_exact_planned_predecessor(phase,prior_phase,warmup,draws):
    prior=dict(phase=prior_phase,state='intentional_stop',progress=[dict(warmup_completed=warmup,draws=draws)])
    validate_predecessor(phase,prior)
    for bad in (None,dict(prior,state='failed'),dict(prior,phase='uninterrupted'),
                dict(prior,progress=[dict(warmup_completed=warmup,draws=draws+1)])):
        with pytest.raises(ValueError):validate_predecessor(phase,bad)
