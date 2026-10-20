import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis import overall_stable_preflight as preflight


@pytest.mark.parametrize('change', [None, 'arrays_key', 'parameters_key', 'truth_key', 'rotation', 'sites'])
def test_generation_and_complete_storage_must_match_original(change):
    generated = dict(arrays_key='data', parameters_key='parameters', truth_key='truth')
    expected = dict(generated, control_rotation_sha256='rotation')
    sites = dict(a=dict(shape=[2], dtype='<f4'))
    old = copy.deepcopy(sites); rotation = 'rotation'
    if change in generated:
        generated[change] = 'changed'
    elif change == 'rotation':
        rotation = 'changed'
    elif change == 'sites':
        sites['unexpected'] = sites['a']
    if change:
        with pytest.raises(ValueError):
            preflight.validate_generated(generated, expected, sites, old, rotation)
    else:
        preflight.validate_generated(generated, expected, sites, old, rotation)


@pytest.mark.parametrize('failure', ['exception', 'killed', 'missing_result', None])
def test_sequential_preflight_accounts_for_every_case_and_preserves_failures(tmp_path, monkeypatch, failure):
    p = dict(id='test-preflight', studies=['first', 'middle', 'last'], model_snapshot=dict(key='model'))
    protocol = tmp_path/'protocol.json'; protocol.write_text('{}')
    (tmp_path/'outputs/championship/development').mkdir(parents=True)
    monkeypatch.setattr(preflight, 'ROOT', tmp_path)
    monkeypatch.setattr(preflight, 'checked', lambda _:p)
    calls = []
    def run(command, **kwargs):
        case = command[command.index('--case')+1]; calls.append(case)
        output = Path(command[command.index('--output')+1]); output.mkdir()
        failed = case == 'middle' and failure is not None
        if not failed or failure == 'exception':
            (output/'result.json').write_text(json.dumps(dict(state='failed' if failed else 'complete')))
        Path(command[command.index('-o')+1]).write_text('fixture resource record')
        return SimpleNamespace(returncode=(-9 if failure == 'killed' else 1 if failure == 'exception' else 0) if failed else 0)
    monkeypatch.setattr(preflight.subprocess, 'run', run)
    result = preflight.execute(protocol)
    assert calls == p['studies'] and result['all_processed']
    assert result['all_complete'] == (failure is None)
    assert not result['mcmc_executed'] and not result['publication_allowed']
    with pytest.raises(FileExistsError):
        preflight.execute(protocol)
