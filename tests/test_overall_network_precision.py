import json

import numpy as np
import pytest

from analysis.overall_network_precision import compare,difference,lift,pack_blocks,sha,unpack_blocks


def test_design_roundtrip_keeps_scalar_types_array_shapes_and_exact_bytes():
    blocks={'q':{'count':2,'flag':False,'sd':.3,'f':np.array([1,2],dtype='f4'),
                 'i':np.arange(3,dtype='i4'),'b':np.array([True,False]),'zero':np.array(.2,dtype='f4')}}
    arrays,scalars=pack_blocks(blocks)
    back=unpack_blocks(arrays,scalars)
    for name,value in blocks['q'].items():
        actual=back['q'][name]
        assert type(actual)==type(value)
        if isinstance(value,np.ndarray):
            assert actual.dtype==value.dtype and actual.shape==value.shape and actual.tobytes()==value.tobytes()
        else: assert actual==value


@pytest.mark.parametrize('dtype',['float32','int32','bool'])
def test_lifting_changes_only_float_precision_preserving_exact_values(dtype):
    x=np.array([0,1],dtype=dtype)
    y=lift(x,'float64')
    np.testing.assert_array_equal(x,y)
    assert y.dtype==('float64' if dtype=='float32' else dtype)


def test_differences_include_all_coordinates_and_zero_denominators():
    d=difference(np.array([3.,4.]),np.array([0.,0.]))
    assert d['l2_difference']==5 and d['relative_l2'] is None
    assert d['max_absolute']==4 and d['largest_flat_index']==1
    assert d['left_at_largest']==4 and d['right_at_largest']==0
    empty=difference(np.empty((0,3)),np.empty((0,3)))
    assert empty['shape']==[0,3] and empty['coordinates']==0 and empty['largest_flat_index'] is None


def test_bad_inputs_do_not_silently_drop_a_field():
    with pytest.raises(ValueError): difference(np.ones(2),np.ones(3))
    with pytest.raises(ValueError): difference([np.nan],[0])
    with pytest.raises(ValueError): pack_blocks({'q':{'x':object()}})


@pytest.mark.parametrize('change',[None,'protocol','point','missing','dtype'])
def test_comparison_requires_the_same_complete_points_and_both_actual_precisions(tmp_path,change):
    paths=[]
    for precision in ('float32','float64'):
        payload={'potential':np.array([2.,3.],dtype=precision),
                 'gradient/x':np.array([4.,5.],dtype=precision)}
        record=dict(state='complete',precision=precision,protocol_sha256='protocol',prepared_sha256='inputs',
            point_labels=[{'id':'first'},{'id':'last'}],point_coordinates_sha256='coordinates',arrays_key='arrays',
            control_rotation_sha256='rotation',parameter_shapes={'x':[]},
            points=[{'scalar_constraints':{'x':1.}},{'scalar_constraints':{'x':2.}}],
            all_original_float32_reports_identical=True)
        if precision=='float64':
            if change=='protocol':record['protocol_sha256']='other'
            if change=='point':record['point_labels'][1]={'id':'different'}
            if change=='missing':payload.pop('gradient/x')
            if change=='dtype':payload['gradient/x']=payload['gradient/x'].astype('float32')
        cache=tmp_path/(precision+'.npz');np.savez_compressed(cache,**payload)
        record.update(cache=str(cache),cache_sha256=sha(cache))
        path=tmp_path/(precision+'.json');path.write_text(json.dumps(record));paths.append(path)
    output={'protocol_sha256':'protocol'}
    if change:
        with pytest.raises(ValueError):compare(*paths,output)
    else:
        compare(*paths,output)
        assert len(output['points'])==2 and output['all_original_float32_reports_identical']
        assert all(point['fields']['gradient/x']['max_absolute']==0 for point in output['points'])
