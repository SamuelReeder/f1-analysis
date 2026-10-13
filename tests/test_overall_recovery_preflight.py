"""Complete preflight orchestration and immutable failure accounting, without fitting."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from analysis import overall_recovery_preflight as runner


def test_all_generated_array_metadata_can_be_saved_without_response_values(tmp_path):
    import numpy as np
    from analysis.overall_recovery_checks import generate_with_parameters
    # Reproduce the already recorded failure's data; no new seed or fit.
    generated = generate_with_parameters('connected', 20261020)
    metadata = runner.block_metadata(generated['blocks'])
    target = tmp_path / 'metadata.json'
    runner.save(target, metadata)
    assert json.loads(target.read_text()) == metadata
    for name, block in generated['blocks'].items():
        for key, value in block.items():
            actual = metadata[name][key]
            if hasattr(value, 'shape') and hasattr(value, 'dtype'):
                assert actual == dict(shape=list(value.shape), dtype=str(value.dtype), bytes=np.asarray(value).nbytes)
            else:
                assert actual == value


def setup(monkeypatch, tmp_path):
    protocol = dict(id='preflight-fixture', studies=[dict(case=c, seed=i) for i, c in enumerate(
        ('connected', 'confounded', 'sparse_rookie', 'dominant_teammate', 'era_boundary', 'real_network'))])
    path = tmp_path / 'protocol.json'
    path.write_text(json.dumps(protocol))
    (tmp_path / 'outputs/championship/development').mkdir(parents=True)
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner, 'checked_protocol', lambda p: protocol)
    return protocol, path


@pytest.mark.parametrize('failure', ['child_exception', 'killed', 'success_exit_without_result', 'failed_result'])
def test_every_case_remains_accounted_for_after_a_child_failure(monkeypatch, tmp_path, failure):
    protocol, path = setup(monkeypatch, tmp_path)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        case = command[command.index('--case') + 1]
        output = Path(command[command.index('--output') + 1])
        output.mkdir()
        (output / 'progress.json').write_text(json.dumps(dict(stage='compiled_potential_and_gradient')))
        Path(command[command.index('-o') + 1]).write_text('resource fixture')
        kwargs['stdout'].write('explicit orchestration fixture, no numerical computation')
        failed = case == 'confounded'
        if not failed or failure in ('child_exception', 'failed_result'):
            (output / 'result.json').write_text(json.dumps(dict(state='failed' if failed else 'complete')))
        return SimpleNamespace(returncode=(1 if failure == 'child_exception' else 137 if failure == 'killed' else 0)
                               if failed else 0)

    monkeypatch.setattr(runner.subprocess, 'run', run)
    result = runner.execute(path)
    assert len(calls) == len(protocol['studies'])
    assert result['all_processed'] and not result['all_complete']
    assert [r['case'] for r in result['studies']] == [r['case'] for r in protocol['studies']]
    assert [r['case'] for r in result['studies'] if r['state'] == 'failed'] == ['confounded']
    assert not result['mcmc_run'] and not result['sbc_complete'] and not result['publication_allowed']
    for command in calls:
        assert command[:3] == ['systemd-run', '--user', '--scope']
        assert '--slice=f1.slice' in command
        assert 'MemoryMax=3G' in command and 'CPUQuota=200%' in command
        assert command[command.index('/usr/bin/nice'):command.index('/usr/bin/nice')+3] == ['/usr/bin/nice', '-n', '19']
    saved = tmp_path / 'outputs/championship/development/preflight-fixture'
    assert json.loads((saved / 'summary.json').read_text()) == result
    for row in result['studies']:
        assert json.loads((saved / (row['case'] + '.completion.json')).read_text()) == row
        assert row['progress_sha256'] and row['output_sha256'] and row['resources_sha256']


def test_existing_preflight_cannot_be_replaced_or_resumed(monkeypatch, tmp_path):
    protocol, path = setup(monkeypatch, tmp_path)
    folder = tmp_path / 'outputs/championship/development' / protocol['id']
    folder.mkdir()
    marker = folder / 'summary.json'
    marker.write_text('original failed output')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **k: pytest.fail('Existing study was relaunched'))
    with pytest.raises(FileExistsError):
        runner.execute(path)
    assert marker.read_text() == 'original failed output'


def protocol_fixture(monkeypatch, tmp_path):
    p = dict(purpose='complete_design_density_gradient_preflight', status='frozen', mcmc_allowed=False,
        publication_allowed=False, id='source-fixture', resources=dict(slice='f1.slice', MemoryMax='3G', CPUQuota='200%', Nice=19),
        density_tolerance=dict(atol=.005, rtol=.000002), state_tolerance=dict(atol=.000003, rtol=.000003),
        studies=[dict(case='connected', seed=42)], source_sha256={'analysis/overall_recovery_preflight.py':runner.SELF_SHA},
        model_source=dict(inputs={}, key=runner.hashlib.sha256(b'{}').hexdigest()))
    (tmp_path / 'analysis').mkdir()
    (tmp_path / 'analysis/overall_recovery_preflight.py').write_bytes(Path(runner.__file__).read_bytes())
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda cmd, **kw: (tmp_path / cmd[-1].removeprefix('HEAD:')).read_bytes())
    return p


@pytest.mark.parametrize('change', ['pending','mcmc','source','loaded_source','cap','tolerance','duplicate','seed','id'])
def test_protocol_rejects_changed_or_incomplete_boundaries(monkeypatch, tmp_path, change):
    p = protocol_fixture(monkeypatch, tmp_path)
    if change == 'pending': p['status']='proposal'
    elif change == 'mcmc': p['mcmc_allowed']=True
    elif change == 'source': (tmp_path / 'analysis/overall_recovery_preflight.py').write_text('changed')
    elif change == 'loaded_source': p['source_sha256']['analysis/overall_recovery_preflight.py']='a'*64
    elif change == 'cap': p['resources']['MemoryMax']='6G'
    elif change == 'tolerance': p['density_tolerance']['atol']=1.
    elif change == 'duplicate': p['studies'].append(p['studies'][0].copy())
    elif change == 'seed': p['studies'][0]['seed']=-1
    elif change == 'id': p['id']='../escape'
    path=tmp_path / 'protocol.json';path.write_text(json.dumps(p))
    with pytest.raises(ValueError): runner.checked_protocol(path)


def test_unchanged_committed_loaded_preflight_sources_are_required(monkeypatch, tmp_path):
    p=protocol_fixture(monkeypatch, tmp_path)
    path=tmp_path / 'protocol.json';path.write_text(json.dumps(p))
    assert runner.checked_protocol(path)==p
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a,**k: b'previous committed bytes')
    with pytest.raises(ValueError,match='Commit'):
        runner.checked_protocol(path)
