import io

import numpy as np
import pytest

from analysis.overall_network_diagnosis import endpoint_rows, gradient_report, read_attempt, scalar_movement, sha


@pytest.mark.parametrize('dtype',['<f4','>f8'])
@pytest.mark.parametrize('shape',[(4,8),(4,8,3),(2,5,0),(2,7,2,3)])
def test_endpoint_reader_preserves_every_selected_coordinate_and_dtype(dtype,shape):
    x=np.arange(np.prod(shape),dtype=dtype).reshape(shape)
    rows=endpoint_rows(io.BytesIO(x.tobytes()),shape,x.dtype)
    for i,row in enumerate(rows):
        np.testing.assert_array_equal(row,x[i//2,0 if i%2==0 else -1])
        assert row.dtype==x.dtype


@pytest.mark.parametrize('raw',[b'',b'\0'*10,b'\0'*33])
def test_endpoint_reader_rejects_missing_or_extra_bytes(raw):
    with pytest.raises(ValueError): endpoint_rows(io.BytesIO(raw),(2,4),np.dtype('f4'))


def test_scalar_movement_retains_constant_chains_and_separation():
    x=np.array([[0,0,0,0],[10,11,12,13]],dtype=np.float32)
    report=scalar_movement(x)
    first,last=report['chains']
    assert first['unique_values']==1 and first['adjacent_unchanged']==3
    assert first['adjacent_rms_over_sd'] is None
    assert last['unique_values']==4 and last['adjacent_unchanged']==0
    assert last['adjacent_rms']==1 and last['quarter_means']==[10,11,12,13]
    assert last['first']==10 and last['last']==13


def test_gradient_report_counts_every_block_without_ranking_selection():
    report=gradient_report(np.float32(4),{'one':np.array([3.,4.]),'two':np.array(12.),'empty':np.empty(0)})
    assert report['gradient_l2']==13
    assert report['gradient_fields']['one']['l2']==5
    assert report['gradient_fields']['two']['coordinates']==1
    assert report['gradient_fields']['empty']['max_absolute']==0
    assert report['potential_ulp']==np.spacing(np.float32(4))


def test_nonfinite_values_are_rejected():
    with pytest.raises(ValueError): scalar_movement(np.full((2,4),np.nan))
    with pytest.raises(ValueError): gradient_report(np.float32(0),{'x':np.array(np.inf)})


def test_attempt_reader_keeps_all_scalar_draws_and_complete_parameter_endpoints(tmp_path):
    from analysis.overall_posterior_archive import _header
    values={'a':np.arange(8,dtype='>f8').reshape(2,4),
            'b':np.arange(24,dtype='f4').reshape(2,4,3),
            'deterministic':np.ones((2,4,2),dtype='f4')}
    path=tmp_path/'diagnostic.npz'
    np.savez_compressed(path,**values)
    inventory={name:dict(shape=list(value.shape[2:]),dtype=value.dtype.str) for name,value in values.items()}
    scalar,points=read_attempt(path,sha(path),(2,4),inventory,{'a':[],'b':[3]},_header)
    assert set(scalar)=={'a'} and scalar['a']['chains'][1]['quarter_means']==[4,5,6,7]
    for i,point in enumerate(points):
        assert set(point)=={'a','b'}
        for name,value in point.items():
            np.testing.assert_array_equal(value,values[name][i//2,0 if i%2==0 else -1])
            assert value.dtype==values[name].dtype
    with pytest.raises(ValueError,match='archive changed'):
        read_attempt(path,'0'*64,(2,4),inventory,{'a':[],'b':[3]},_header)
