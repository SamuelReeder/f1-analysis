"""Complete archive inventory, chain order and interrupted sealing checks."""
import copy
from pathlib import Path

import numpy as np
import pytest

from analysis.checkpoint_archive import finalize, inventory
from f1rank.overall.checkpoints import Checkpoints
from test_overall_checkpoints import identity, samples, state


def fixture(path):
    ident = identity()
    rows = []
    with Checkpoints(path, ident) as saved:
        saved.append(state())
        for start, count in ((0,4), (4,2)):
            posterior, extras = samples(count)
            posterior['beta'] = np.asfortranarray(posterior['beta'] + start*100)
            posterior['empty'] = np.empty((2,count,0), np.float64)
            posterior['signed'] = np.tile(np.array([0.,-0.],np.float64)[:,None], (1,count))
            rows.append((posterior, extras))
            saved.append(state(start+count), posterior=posterior, extras=extras)
        expected = inventory(saved)
    return ident, expected, rows


def test_finalizer_keeps_complete_site_inventory_chain_order_and_bits(tmp_path):
    ident, expected, chunks = fixture(tmp_path/'source')
    record = finalize(tmp_path/'source', tmp_path/'final', identity=ident,
                      expected_inventory=expected, verify_inputs=lambda:None)
    assert record['convergence_evaluated'] is False and record['publication_allowed'] is False
    for index, group in enumerate(('posterior','extras')):
        with np.load(tmp_path/'final'/(group+'.npz'), allow_pickle=False) as archive:
            assert set(archive.files) == set(expected[group])
            for name in archive.files:
                wanted = np.concatenate([row[index][name] for row in chunks], axis=1)
                assert archive[name].shape == wanted.shape
                assert archive[name].dtype == wanted.dtype
                assert archive[name].tobytes() == wanted.tobytes()
    with pytest.raises(FileExistsError):
        finalize(tmp_path/'source', tmp_path/'final', identity=ident,
                 expected_inventory=expected, verify_inputs=lambda:None)


@pytest.mark.parametrize('change', ['missing_site','extra_site','wrong_dtype','wrong_axes','missing_extras'])
def test_complete_external_inventory_is_mandatory(tmp_path, change):
    ident, expected, _ = fixture(tmp_path/'source'); expected=copy.deepcopy(expected)
    if change=='missing_site': del expected['posterior']['beta']
    if change=='extra_site': expected['posterior']['invented']=expected['posterior']['beta']
    if change=='wrong_dtype': expected['posterior']['beta']['dtype']='<f8'
    if change=='wrong_axes': expected['posterior']['beta']['shape'][1]=5
    if change=='missing_extras': del expected['extras']
    with pytest.raises(ValueError, match='every expected'):
        finalize(tmp_path/'source', tmp_path/'final', identity=ident,
                 expected_inventory=expected, verify_inputs=lambda:None)
    assert not (tmp_path/'final').exists()


def test_incomplete_attempt_cannot_be_finalized(tmp_path):
    ident=identity()
    with Checkpoints(tmp_path/'source', ident) as saved:
        saved.append(state())
        posterior,extras=samples(4)
        saved.append(state(4),posterior=posterior,extras=extras)
    with pytest.raises(ValueError,match='complete retained'):
        finalize(tmp_path/'source',tmp_path/'final',identity=ident,
                 expected_inventory={},verify_inputs=lambda:None)
    assert not (tmp_path/'final').exists()


def test_interrupted_seal_leaves_no_final_artifact_and_can_be_retried(tmp_path, monkeypatch):
    ident,expected,_=fixture(tmp_path/'source')
    rename=Path.rename
    def interrupt(self,target):
        if self.name.startswith('.partial-'): raise InterruptedError('before atomic seal')
        return rename(self,target)
    with monkeypatch.context() as context:
        context.setattr(Path,'rename',interrupt)
        with pytest.raises(InterruptedError):
            finalize(tmp_path/'source',tmp_path/'final',identity=ident,
                     expected_inventory=expected,verify_inputs=lambda:None)
    assert not (tmp_path/'final').exists() and list(tmp_path.glob('.partial-*'))
    finalize(tmp_path/'source',tmp_path/'final',identity=ident,
             expected_inventory=expected,verify_inputs=lambda:None)


def test_input_change_before_seal_prevents_final_artifact(tmp_path):
    ident,expected,_=fixture(tmp_path/'source')
    def changed():
        if list(tmp_path.glob('.partial-*/extras.npz')): raise ValueError('inputs changed')
    with pytest.raises(ValueError,match='inputs changed'):
        finalize(tmp_path/'source',tmp_path/'final',identity=ident,
                 expected_inventory=expected,verify_inputs=changed)
    assert not (tmp_path/'final').exists()


def test_corrupted_checkpoint_is_rejected_before_seal(tmp_path):
    ident,expected,_=fixture(tmp_path/'source')
    with (tmp_path/'source/checkpoint-000001/posterior.npz').open('ab') as file:
        file.write(b'corruption')
    with pytest.raises(ValueError,match='bytes changed'):
        finalize(tmp_path/'source',tmp_path/'final',identity=ident,
                 expected_inventory=expected,verify_inputs=lambda:None)
    assert not (tmp_path/'final').exists()
