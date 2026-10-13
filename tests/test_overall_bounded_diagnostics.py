"""Whole-report equality with the existing numerical gate; no thresholds changed."""
import warnings

import numpy as np
import pytest

from analysis import overall_bounded_diagnostics as bounded
from f1rank.overall.diagnostics import check as original


def fixture():
    random=np.random.default_rng(719)
    post={'scalar':random.normal(size=(4,400)), 'wide':random.normal(size=(4,400,2,129)).astype('float32'),
          'car':random.normal(size=(4,400,5)), 'stochastic':random.normal(size=(4,400,4)),
          'empty':np.empty((4,400,0,3)), 'integers':random.integers(-3,4,size=(4,400,2))}
    post['car'][...,0]=1
    post['stochastic'][...,1]=2
    post['wide'][-1,:,-1,-1]+=2
    return post


@pytest.mark.parametrize('block',[1,7,128,1000])
def test_every_coordinate_and_failure_matches_original_report(block):
    post=fixture()
    assert bounded.check(post,3,coordinate_block=block)==original(post,3)


@pytest.mark.parametrize('layout',['fortran','reverse','transpose','memory_map'])
def test_noncontiguous_and_mapped_input_remains_exact(tmp_path,layout):
    a=np.random.default_rng(17).normal(size=(4,256,3,5)).astype('float32')
    if layout=='fortran':a=np.asfortranarray(a)
    elif layout=='reverse':a=a[:,::-1,::-1,::-1]
    elif layout=='transpose':a=a.transpose(0,1,3,2)
    else:
        path=tmp_path/'mapped.npy';np.save(path,a);a=np.load(path,mmap_mode='r')
    try:
        assert bounded.check({'test':a},0,coordinate_block=4)==original({'test':a},0)
    finally:
        if layout=='memory_map':a._mmap.close()


@pytest.mark.parametrize('case',['empty','nonfinite','wrong_axes','deterministic_constant','stochastic_constant',
                               'only_empty','short_chain','few_chains','scalar_input','negative_divergence',
                               'excess_divergence','underflow'])
def test_invalid_constant_and_empty_cases_preserve_decisions(case):
    random=np.random.default_rng(13)
    post={'state':random.normal(size=(4,128,2))};divergences=0
    if case=='empty':post={}
    elif case=='nonfinite':post['state'][-1,-1,-1]=np.nan
    elif case=='wrong_axes':post['later']=np.zeros((4,127))
    elif case=='deterministic_constant':post={'car':np.ones((4,128,3))}
    elif case=='stochastic_constant':post['constant']=np.ones((4,128,3))
    elif case=='only_empty':post={'empty':np.empty((4,128,0))}
    elif case=='short_chain':post={'state':post['state'][:,:30]}
    elif case=='few_chains':post={'state':post['state'][:2]}
    elif case=='scalar_input':post={'state':np.asarray(1.)}
    elif case=='negative_divergence':divergences=-1
    elif case=='excess_divergence':divergences=100
    elif case=='underflow':post={'state':post['state']*1e-200}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        if case=='empty':
            with pytest.raises(ValueError,match='Empty'):bounded.check(post,divergences)
            with pytest.raises(ValueError,match='Empty'):original(post,divergences)
        else:assert bounded.check(post,divergences,coordinate_block=1)==original(post,divergences)


def test_a_nonfinite_metric_invalidates_the_whole_site(monkeypatch):
    from arviz_stats.base import array_stats
    rhat=array_stats.rhat
    def nonfinite(data,**kwargs):
        return np.where(data[0,0,:]==1234.,np.nan,rhat(data,**kwargs))
    monkeypatch.setattr(array_stats,'rhat',nonfinite)
    a=np.random.default_rng(21).normal(size=(4,256,11));a[0,0,9]=1234.
    observed=bounded.check({'site':a},0,coordinate_block=3)
    assert observed==original({'site':a},0)
    assert not observed['converged'] and observed['parameters'][0]['rhat_max'] is None
    assert observed['parameters'][0]['bulk_ess_min'] is None


@pytest.mark.parametrize('block',[0,-1,True,1.5])
def test_invalid_computational_block_size_is_rejected(block):
    with pytest.raises(ValueError,match='positive'):
        bounded.check({'x':np.ones((4,128))},0,coordinate_block=block)
