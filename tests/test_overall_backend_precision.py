"""Precision lifting, full inventory and repeatability reporting safeguards."""
import numpy as np
import pytest
from analysis.backend_precision import lift,differences,ARMS,PAIRS


def test_lifting_preserves_stored_values_and_discrete_arrays():
    x=np.array([.1,-2.,3.],np.float32)
    y=lift(x)
    assert y.dtype==np.dtype('float64')
    np.testing.assert_array_equal(y,x.astype(np.float64))
    i=np.array([1,2],np.int32);b=np.array([True,False])
    assert lift(i) is i and lift(b) is b
    np.testing.assert_array_equal(i,[1,2])


def test_bitwise_reproduction_is_distinct_from_rounded_numeric_equality():
    left={'potential':np.array([0.,-0.],np.float32)}
    right={'potential':np.array([-0.,0.],np.float32)}
    r=differences(left,right)['potential']
    assert r['exact_values'] and not r['bitwise_equal']
    assert r['max_absolute']==0.


def test_reports_every_field_and_largest_coordinate_across_points():
    left={'potential':np.array([1.,2.]),'gradient/x':np.zeros((2,3)),'state/empty':np.empty((2,0))}
    right={n:v.copy() for n,v in left.items()};right['gradient/x'][1,2]=4.
    r=differences(left,right)
    assert set(r)==set(left)
    assert r['gradient/x']['largest_index']==[1,2]
    assert r['gradient/x']['left_at_largest']==0.
    assert r['gradient/x']['right_at_largest']==4.
    assert r['state/empty']['coordinates']==0 and r['state/empty']['largest_index'] is None


@pytest.mark.parametrize('change',['missing','shape','nonfinite'])
def test_incomplete_comparison_is_rejected(change):
    left={'gradient/x':np.ones((2,3))};right={n:v.copy() for n,v in left.items()}
    if change=='missing':right={}
    elif change=='shape':right['gradient/x']=right['gradient/x'][:1]
    else:right['gradient/x'][0,0]=np.nan
    with pytest.raises(ValueError):differences(left,right)


def test_cross_precision_differences_preserve_dtype_information():
    left={'gradient/x':np.array([.1],np.float32)}
    right={'gradient/x':left['gradient/x'].astype(np.float64)}
    r=differences(left,right)['gradient/x']
    assert r['exact_values'] and not r['same_dtype'] and not r['bitwise_equal']


def test_registered_pairs_include_originals_and_fresh_process_repetitions():
    assert set(ARMS)=={'cpu32','gpu32','cpu64_a','cpu64_b','gpu64_a','gpu64_b'}
    assert ('original_cpu','cpu32') in PAIRS and ('original_gpu','gpu32') in PAIRS
    assert ('cpu64_a','cpu64_b') in PAIRS and ('gpu64_a','gpu64_b') in PAIRS
    assert ('cpu32','cpu64_a') in PAIRS and ('gpu32','gpu64_a') in PAIRS
