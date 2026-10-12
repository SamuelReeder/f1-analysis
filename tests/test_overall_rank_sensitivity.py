"""Coverage arithmetic and complete companion integration; no fitted rankings."""
import itertools
import json
import math

import numpy as np
import pytest

from analysis import overall_rank_sensitivity as sensitivity
from test_overall_simulation_controller import case, execute


@pytest.mark.parametrize('replicates', [4, 16, 64])
@pytest.mark.parametrize('delta', [.01, .1, .5])
def test_exact_binomial_tail_inventory_satisfies_declared_family_bound(replicates,delta):
    # Enumerate every possible count; this is not a repeated Monte Carlo test.
    counts=np.full((2,2),replicates,dtype=int)
    for p in (0.,.02,.5,.98,1.):
        failure=0.
        for k in range(replicates+1):
            means=np.full((2,2,2),sensitivity.MAX_POINTS*k/replicates)
            r=sensitivity.envelope(means,counts,['a','b'],error_probability=delta)
            radius=r['point_radius_by_draw'][0][0]
            lower=max(0.,means[0,0,0]-radius);upper=min(sensitivity.MAX_POINTS,means[0,0,0]+radius)
            if not lower <= sensitivity.MAX_POINTS*p <= upper:
                failure+=math.comb(replicates,k)*p**k*(1-p)**(replicates-k)
        assert failure*r['family_size'] <= delta+1e-14


def test_every_vertex_of_a_grouped_interval_box_respects_all_comparisons_and_ranks():
    lower=np.array([[[0.,1.,3.],[2.,1.,0.]],[[4.,1.,0.],[1.,1.,1.]]])
    upper=lower+np.array([[[2.,1.,1.],[1.,2.,2.]],[[0.,1.,2.],[0.,0.,0.]]])
    field=['a','b','c'];r=sensitivity.ordering_bounds(lower,upper,field)
    bits=np.asarray(list(itertools.product([0,1],repeat=lower.size))).reshape((-1,)+lower.shape)
    points=lower+bits*(upper-lower)
    exact_low=1+(points[...,None,:]>points[...,:,None]).sum(axis=-1)
    exact_high=3-(points[...,None,:]<points[...,:,None]).sum(axis=-1)
    for i,row in enumerate(r['drivers']):
        lo,hi=row['trace_mean_interval']
        average=points[...,i].mean(axis=(1,2))
        assert (lo <= average).all() and (average <= hi).all()
        for k in range(1,4):
            # Both extremes of each exact tie group must be covered.
            for exact in (exact_low,exact_high):
                pooled=(exact[...,i]<=k).mean(axis=(1,2))
                assert (pooled>=row['rank_cdf_bounds']['lower'][k-1]).all()
                assert (pooled<=row['rank_cdf_bounds']['upper'][k-1]).all()
                chains=(exact[...,i]<=k).mean(axis=2)
                assert (chains>=np.array(row['rank_cdf_bounds']['lower_by_chain'])[:,k-1]).all()
                assert (chains<=np.array(row['rank_cdf_bounds']['upper_by_chain'])[:,k-1]).all()
    for row in r['pairs']:
        i,j=field.index(row['driver_a']),field.index(row['driver_b'])
        difference=points[...,i]-points[...,j]
        for key,predicate in [('a_greater',difference>0),('b_greater',difference<0),('equal',difference==0)]:
            bound=row['empirical_fraction_bounds'][key];fraction=predicate.mean(axis=(1,2))
            assert (fraction>=bound['lower']).all() and (fraction<=bound['upper']).all()
            by_chain=predicate.mean(axis=2)
            assert (by_chain>=bound['lower_by_chain']).all() and (by_chain<=bound['upper_by_chain']).all()


def test_exact_ties_retain_rank_groups_and_are_not_broken_by_names():
    values=np.broadcast_to([2.,2.,1.],(2,2,3))
    result=sensitivity.ordering_bounds(values,values,['z','a','b'])
    for row in result['drivers'][:2]:
        assert row['rank_cdf_bounds']['lower']==[0.,1.,1.]
        assert row['rank_cdf_bounds']['upper']==[1.,1.,1.]
    tied=result['pairs'][0]['empirical_fraction_bounds']
    assert tied['equal']['lower']==tied['equal']['upper']==1.
    assert tied['a_greater']['upper']==tied['b_greater']['upper']==0.


def test_field_permutations_keep_driver_bounds_and_pair_orientation():
    lower=np.array([[[0.,1.,2.],[1.,2.,0.]],[[2.,0.,1.],[3.,2.,1.]]]);upper=lower+.5
    field=['z','a','b'];permutation=[2,0,1]
    before=sensitivity.ordering_bounds(lower,upper,field)
    after=sensitivity.ordering_bounds(lower[...,permutation],upper[...,permutation],[field[i] for i in permutation])
    assert {r['driver_id']:r for r in before['drivers']}=={r['driver_id']:r for r in after['drivers']}
    for row in before['pairs']:
        new=next(p for p in after['pairs'] if {p['driver_a'],p['driver_b']}=={row['driver_a'],row['driver_b']})
        reverse=row['driver_a']!=new['driver_a']
        for key,other in [('a_greater','b_greater'),('b_greater','a_greater'),('equal','equal')]:
            assert row['empirical_fraction_bounds'][key]==new['empirical_fraction_bounds'][other if reverse else key]


def test_full_mean_family_and_variable_replication_are_kept():
    means=np.full((2,2,2),12.);losses=np.zeros((2,2,2,3));counts=np.array([[100,400],[1600,6400]])
    result=sensitivity.envelope(means,counts,['a','b'],error_probability=.1,probe_loss_mean=losses)
    assert result['family_size']==32 and result['replicate_counts']==counts.tolist()
    radius=np.array(result['point_radius_by_draw']).ravel()
    np.testing.assert_allclose(radius[:-1]/radius[1:],2.)
    point_only=sensitivity.envelope(means,counts,['a','b'],error_probability=.1)
    assert result['point_radius_by_draw'][0][0]>point_only['point_radius_by_draw'][0][0]
    assert len(result['explanations'])==2
    assert set(result['explanations'][0]['losses'])==set(sensitivity.PROBES)
    assert all(result[key] is False for key in ('mcmc_precision_verified','rank_precision_established','scientific_evidence_verified','publication_allowed'))


@pytest.mark.parametrize('change',['counts','nan','range','axis','field','delta','losses','complex'])
def test_incomplete_or_invalid_mean_inventory_is_rejected(change):
    means=np.ones((2,2,2));counts=np.full((2,2),10);field=['a','b'];delta=.1;losses=np.zeros((2,2,2,3))
    if change=='counts':counts[0,0]=0
    elif change=='nan':means[0,0,0]=np.nan
    elif change=='range':means[0,0,0]=sensitivity.MAX_POINTS+1
    elif change=='axis':means=means[0]
    elif change=='field':field=['a','a']
    elif change=='delta':delta=True
    elif change=='losses':losses=losses[...,:2]
    else:means=means.astype(complex)
    with pytest.raises(ValueError):sensitivity.envelope(means,counts,field,error_probability=delta,probe_loss_mean=losses)


def test_complete_controller_companion_has_a_bound_diagnostic_without_acceptance(case):
    original,receipt=execute(case)
    destination=case.output.parent/'sensitivity'
    args=dict(expected_simulation_key=receipt['simulation_key'],
        expected_archive_sha256=original['diagnostic_moments']['sha256'],error_probability=.1,output_dir=destination)
    result=sensitivity.analyse_checked(case.output,**args)
    assert result['input_identity']==original['identity']
    assert result['family_size']==np.prod(original['posterior_shape'])*len(original['field'])*(1+len(sensitivity.PROBES))
    assert result['posterior_shape']==original['posterior_shape']
    assert result['complete_draw_and_field_inventory'] is True
    assert result['publication_allowed'] is False and result['mcmc_precision_verified'] is False
    assert result==json.loads((destination/'sensitivity.json').read_text())
    before=sensitivity.sha(destination/'sensitivity.json')
    with pytest.raises(FileExistsError):sensitivity.analyse_checked(case.output,**args)
    assert sensitivity.sha(destination/'sensitivity.json')==before


@pytest.mark.parametrize('change',['failed','archive','during_analysis'])
def test_invalid_or_changed_companion_cannot_supply_a_completed_diagnostic(case,monkeypatch,change):
    original,receipt=execute(case)
    if change=='failed':
        receipt['state']='failed_exception';(case.output/'receipt.json').write_text(json.dumps(receipt))
    elif change=='archive':
        with (case.output/sensitivity.FILENAME).open('ab') as f:f.write(b'changed')
    else:
        kernel=sensitivity.envelope
        def changed(*args,**kwargs):
            result=kernel(*args,**kwargs)
            with (case.output/'summary.json').open('a') as f:f.write('\n')
            return result
        monkeypatch.setattr(sensitivity,'envelope',changed)
    destination=case.output.parent/'sensitivity'
    with pytest.raises(ValueError):
        sensitivity.analyse_checked(case.output,expected_simulation_key=receipt['simulation_key'],
            expected_archive_sha256=original['diagnostic_moments']['sha256'],error_probability=.1,output_dir=destination)
    assert not destination.exists()
