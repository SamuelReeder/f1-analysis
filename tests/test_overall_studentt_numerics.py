import numpy as np
import pytest

from analysis.overall_studentt_numerics import (assessment,beta_log_prob,beta_studentt_calculation,
                                              inputs,load_reference,reference_row,numerical_reference_row)


def test_cauchy_density_and_log_degree_derivative_have_known_values():
    import jax
    import jax.numpy as jnp
    value,derivative=jax.value_and_grad(lambda z:beta_log_prob(jnp.exp(z),0.,1.,0.))(jnp.array(0.))
    np.testing.assert_allclose(value,-np.log(np.pi),rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(derivative,np.log(2)-.5,rtol=2e-6,atol=2e-6)


def test_location_scale_identity_and_symmetry_preserve_the_density():
    import jax.numpy as jnp
    x=jnp.asarray([-3.,0.,1.])
    base=beta_log_prob(4.,0.,1.,x)
    np.testing.assert_allclose(base,beta_log_prob(4.,0.,1.,-x),rtol=2e-6,atol=2e-6)
    np.testing.assert_allclose(base,beta_log_prob(4.,2.,3.,2+3*x)+np.log(3),rtol=2e-6,atol=2e-6)


def test_temporary_density_substitution_restores_original_even_on_exception():
    from numpyro.distributions import StudentT
    original,sample=StudentT.log_prob,StudentT.sample
    with pytest.raises(RuntimeError):
        with beta_studentt_calculation():
            assert StudentT.log_prob is not original and StudentT.sample is sample
            raise RuntimeError('synthetic interruption')
    assert StudentT.log_prob is original and StudentT.sample is sample


def test_reference_is_independent_and_keeps_the_exact_input_coordinates():
    mp=load_reference()
    row=reference_row(mp,0.,0.,80)
    np.testing.assert_allclose(row,[-np.log(np.pi),np.log(2)-.5,0.,-1.],rtol=1e-14,atol=1e-14)
    assert row==reference_row(mp,0.,0.,120)


def test_all_saved_parameter_points_remain_in_the_declared_cartesian_inventory():
    labels=[{'id':'a'},{'id':'b'}]
    rows=inputs({'point_labels':labels},{'z/nu':np.array([0.,1.],dtype='f4'),
                                       'z/overall_lap_nu':np.array([2.,3.],dtype='f4')})
    saved=[r for r in rows if r['kind']=='saved_coordinate']
    assert len(saved)==36
    assert {tuple((r['point']['id'],r['parameter'])) for r in saved}=={
        ('a','nu'),('a','overall_lap_nu'),('b','nu'),('b','overall_lap_nu')}
    assert all(np.float32(r['log_df'])==r['log_df'] for r in rows)


@pytest.mark.parametrize('log_df',[-6.907755374908447,0.,2.084352731704712,18.42068099975586])
@pytest.mark.parametrize('residual',[-1.,1.])
def test_exact_zero_scale_derivative_is_stable_without_tolerance(log_df,residual):
    mp=load_reference()
    low=reference_row(mp,log_df,residual,80)
    high=reference_row(mp,log_df,residual,120)
    assert low==high
    assert high[3]==0.


@pytest.mark.parametrize('residual',[-100.,-3.,0.,3.,100.])
def test_analytic_reference_agrees_with_independent_numerical_derivatives(residual):
    mp=load_reference()
    assert reference_row(mp,2.084352731704712,residual,120)==numerical_reference_row(mp,2.084352731704712,residual,120)


def test_error_budget_reports_failures_without_dropping_rows():
    ref=np.zeros((3,4))
    actual=np.zeros((3,4),dtype='f4');actual[1,2]=1
    report=assessment(actual,ref,'float32')
    assert report['failing_rows']==[1] and not report['all_within_fixed_error_budget']
    assert report['max_absolute_error'][2]==1
    actual[2,0]=np.inf
    report=assessment(actual,ref,'float32')
    assert report['failing_rows']==[1,2] and report['nonfinite_coordinates']==[[2,0]]
    assert report['max_absolute_error'][0] is None
