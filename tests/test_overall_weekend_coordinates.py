"""Check distribution equivalence; do not fit or claim computational success."""
import numpy as np
import pytest

from analysis.overall_weekend_coordinates import (
    COMPLEMENT, HYPER_SITES, TOTAL, SuppliedWeekend, hyper_priors, transform, weekend_model)


@pytest.mark.parametrize('sf,sr,b', [(.2,.3,1.), (.2,.3,0.), (.2,.3,-2.),
                                    (1e-5,3.,2.), (3.,1e-5,2.)])
def test_conditional_orthogonality_covariance_inverse_and_total(sf,sr,b):
    first,second=np.array([1.,0.]),np.array([0.,1.])
    matrix=np.asarray(transform(sf,sr,b,first,second))
    np.testing.assert_allclose(matrix @ matrix.T,np.eye(2),atol=3e-7)
    np.testing.assert_allclose(matrix @ matrix,np.eye(2),atol=3e-7)
    np.testing.assert_allclose(abs(np.linalg.det(matrix)),1.,atol=3e-7)
    scales=np.array([[sf,0.],[b*sf,sr]])
    np.testing.assert_allclose((scales @ matrix) @ (scales @ matrix).T, scales @ scales.T,atol=3e-6,rtol=3e-7)
    zq,zr=transform(sf,sr,b,np.arange(7.)-2,np.arange(7.)[::-1]-3)
    u,v=transform(sf,sr,b,zq,zr)
    np.testing.assert_allclose(u,np.arange(7.)-2,atol=1e-6)
    np.testing.assert_allclose(v,np.arange(7.)[::-1]-3,atol=1e-6)
    np.testing.assert_allclose(b*sf*zq+sr*zr,np.hypot(b*sf,sr)*u,atol=2e-6)


def test_full_hyperparameter_dependent_jacobian_has_unit_absolute_determinant():
    import jax
    import jax.numpy as jnp
    def joint(v):
        first,second=transform(v[0],v[1],v[2],v[3:5],v[5:])
        return jnp.concatenate([v[:3],first,second])
    point=jnp.array([.2,.3,-1.2,2.,3.,-.5,1.])
    jacobian=jax.jacfwd(joint)(point)
    assert np.isfinite(jacobian).all()
    np.testing.assert_allclose(abs(np.linalg.det(jacobian)),1.,atol=5e-7)
    np.testing.assert_allclose(joint(joint(point)),point,atol=1e-6)


@pytest.mark.parametrize('case,seed', [('connected',8),('confounded',9),('sparse_rookie',10)])
@pytest.mark.parametrize('with_controls', [False,True])
def test_full_model_and_all_observation_factors_remain_equivalent(case,seed,with_controls):
    import jax
    from numpyro import handlers
    from numpyro.infer.util import log_density
    from analysis.overall_control_rotation import ROTATED_SITE,build_rotation
    from f1rank.overall.model import model
    from f1rank.overall.recovery import synthetic_design
    d=synthetic_design(case);args=(d.qualifying.arrays(),d.laps,d.outcomes)
    trace=handlers.trace(handlers.seed(model,jax.random.PRNGKey(seed))).get_trace(*args)
    params={name:s['value'] for name,s in trace.items() if s['type']=='sample' and not s['is_observed']}
    updated=dict(params)
    # Compare actual original prior families and values, independently of the
    # wrapper's copied specification; a source change must not be silently used.
    for name,prior in hyper_priors().items():
        actual=trace[name]['fn']
        assert type(actual) is type(prior)
        for attr in prior.arg_constraints:
            np.testing.assert_array_equal(getattr(actual,attr),getattr(prior,attr))
        updated[HYPER_SITES[name]]=updated.pop(name)
    updated[TOTAL],updated[COMPLEMENT]=transform(params['sd_driver_form'],params['overall_sd_day'],
        params['overall_shared_weekend'],updated.pop('driver_form_z'),updated.pop('overall_day_z'))
    rotation=build_rotation(d.laps) if with_controls else None
    if rotation is not None:
        updated[ROTATED_SITE]=rotation.rotated({name:updated.pop(name) for name in rotation.shapes})
    before,original_trace=log_density(model,args,{},params)
    after,new_trace=log_density(weekend_model(rotation),args,{},updated)
    np.testing.assert_allclose(after,before+(rotation.log_abs_det if rotation else 0.),atol=5e-3,rtol=2e-6)
    for name,old in original_trace.items():
        if old['type'] not in ('sample','deterministic'):
            continue
        new=new_trace[name]
        if old['type']=='sample' and old['is_observed']:
            np.testing.assert_allclose(new['fn'].log_prob(new['value']),old['fn'].log_prob(old['value']),atol=5e-3,rtol=2e-6,err_msg=name)
        else:
            np.testing.assert_allclose(new['value'],old['value'],atol=3e-6,rtol=3e-6,err_msg=name)
    for name in [*HYPER_SITES,'driver_form_z','overall_day_z']:
        assert new_trace[name]['type']=='deterministic'
    sf,sr,b=[params[name] for name in HYPER_SITES]
    from f1rank.model import centre
    expected=np.hypot(b*sf,sr)*centre(updated[TOTAL],args[0]['entry_event'],args[0]['n_events'])
    np.testing.assert_allclose(new_trace['overall_day']['value'],expected,atol=2e-6,rtol=2e-6)


def test_composed_model_has_finite_unconstrained_gradient():
    import jax
    from numpyro.infer.util import initialize_model
    from analysis.overall_control_rotation import build_rotation
    from f1rank.overall.recovery import synthetic_design
    d=synthetic_design('confounded')
    fit=initialize_model(jax.random.PRNGKey(11),weekend_model(build_rotation(d.laps)),
        model_args=(d.qualifying.arrays(),d.laps,d.outcomes))
    assert all(np.isfinite(v).all() for v in jax.tree_util.tree_leaves(fit.param_info.z_grad))


def test_incompatible_dimensions_and_original_sites_are_rejected():
    import jax.numpy as jnp
    import numpyro.distributions as dist
    with pytest.raises(ValueError,match='aligned'):
        transform(.2,.3,1.,np.zeros(3),np.zeros(4))
    with pytest.raises(ValueError,match='scalar'):
        transform(np.array([.2]),.3,1.,np.zeros(3),np.zeros(3))
    supplied=SuppliedWeekend(jnp.zeros(3),dist.Normal)
    for fn,obs in [(dist.HalfNormal(1).expand([3]),None),(dist.Normal(0,1),None),
                   (dist.Normal(0,1).expand([3]),jnp.zeros(3))]:
        with pytest.raises(ValueError,match='Unexpected'):
            supplied('driver_form_z',fn,obs)
