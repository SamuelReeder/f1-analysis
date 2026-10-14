"""Fixed retry control and immutable history, using tiny non-scientific arrays."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from analysis import overall_profile_continuation_run as runner
from analysis.overall_profile_continuation import fixed_settings


def fixture(tmp_path):
    parent = dict(fit_settings=dict(warmup=800, samples=800, chains=4, seed=0, target_accept=.9),
                  retry=[[1, 1, None], [2, 2, None], [3, 4, None], [3, 4, .98]])
    plan = dict(case='fixture', preserved_attempts=2,
        remaining_attempts=[dict(attempt=i+1, settings=fixed_settings(parent, i)) for i in (2, 3)])
    original = dict(state='running', source={'key': 'unchanged-scientific-source'},
        attempts=[dict(fixed_settings(parent, i), diagnostics={'converged': False}, seconds=100. + i) for i in (0, 1)])
    cache = tmp_path / 'new-cache'
    cache.mkdir()
    return plan, original, cache


@pytest.mark.parametrize('success', [3, 4, None])
def test_only_unfinished_and_remaining_original_attempts_execute(tmp_path, success):
    plan, original, cache = fixture(tmp_path)
    before = copy.deepcopy(original)
    calls, validations = [], []
    def sample(settings):
        calls.append(copy.deepcopy(settings))
        return {'fixture_draw': np.arange(8).reshape(2, 4)}, dict(converged=len(calls)+2 == success, divergences=0)
    post, receipt = runner.run_remaining(plan, original, cache, sample,
        lambda: validations.append(True), {'started_at': 'new-segment', 'duration_scope': 'new execution only'})
    expected = plan['remaining_attempts'][:1] if success == 3 else plan['remaining_attempts']
    assert calls == [row['settings'] for row in expected]
    assert len(validations) == 3 * len(calls)
    assert receipt['attempts'][:2] == original['attempts'] and original == before
    assert receipt['state'] == ('converged' if success else 'failed_convergence')
    assert len(receipt['attempts']) == 2 + len(calls)
    assert receipt['execution_continuation']['seconds'] >= 0
    assert 'seconds' not in receipt  # No invented uninterrupted whole-run duration.
    assert runner.records.read(cache / 'receipt.json') == receipt
    assert not list(cache.glob('*.tmp.npz'))
    if success:
        assert post is not None and receipt['posterior_sha256'] == runner.records.sha(cache / 'posterior.npz')
    else:
        assert post is None and not (cache / 'posterior.npz').exists()
    for row in expected:
        number = row['attempt']
        archive = cache / ('posterior.npz' if number == success else f'failed_{number}.npz')
        with np.load(archive) as saved:
            np.testing.assert_array_equal(saved['fixture_draw'], np.arange(8).reshape(2, 4))


@pytest.mark.parametrize('failure', ['sample', 'source_before', 'source_after', 'source_before_archive_publish'])
def test_exception_stops_without_trying_another_seed_and_preserves_history(tmp_path, failure):
    plan, original, cache = fixture(tmp_path)
    calls, checks = [], []
    def sample(settings):
        calls.append(settings)
        if failure == 'sample': raise RuntimeError('sampling exception')
        return {'fixture': np.zeros((2, 4))}, dict(converged=False, divergences=0)
    def validate():
        checks.append(True)
        limit = {'source_before': 1, 'source_after': 2, 'source_before_archive_publish': 3}.get(failure)
        if len(checks) == limit: raise RuntimeError('source changed')
    with pytest.raises(RuntimeError):
        runner.run_remaining(plan, original, cache, sample, validate, {})
    receipt = runner.records.read(cache / 'receipt.json')
    assert receipt['state'] == 'failed_exception' and receipt['attempts'][:2] == original['attempts']
    assert len(calls) <= 1 and not (cache / 'attempt_4.json').exists()
    if failure == 'source_before_archive_publish':
        assert len(receipt['attempts']) == 3
        assert (cache / 'failed_3.tmp.npz').exists() and not (cache / 'failed_3.npz').exists()


@pytest.mark.parametrize('existing', ['receipt.json', 'attempt_3.json', 'failed_3.npz', 'posterior.npz', 'failed_3.tmp.npz'])
def test_existing_continuation_bytes_are_never_overwritten(tmp_path, existing):
    plan, original, cache = fixture(tmp_path)
    path = cache / existing
    path.write_bytes(b'preserve original bytes')
    def sample(_):
        return {'fixture': np.zeros((2, 4))}, dict(converged=False, divergences=0)
    with pytest.raises(ValueError):
        runner.run_remaining(plan, original, cache, sample, lambda: None, {})
    assert path.read_bytes() == b'preserve original bytes'
    assert not (cache / 'attempt_4.json').exists()


def test_original_archive_and_attempt_bytes_are_preserved_under_new_names(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'ROOT', tmp_path)
    source, target = tmp_path / 'old', tmp_path / 'new'
    source.mkdir(); target.mkdir()
    archive = source / 'failed_1.npz'
    np.savez_compressed(archive, fixture=np.zeros((2, 3)))
    original = dict(diagnostics={'converged': False}, seconds=999.)
    attempt = source / 'attempt_1.json'
    attempt.write_text(json.dumps(original))
    before = archive.read_bytes(), attempt.read_bytes()
    study = dict(attempts=[dict(attempt=1, archive='old/failed_1.npz', archive_sha256=runner.records.sha(archive), recorded=original)])
    runner.preserve_attempts(study, target)
    assert (target / 'failed_1.npz').read_bytes() == before[0]
    assert (target / 'attempt_1.json').read_bytes() == before[1]
    assert (archive.read_bytes(), attempt.read_bytes()) == before
    assert archive.stat().st_ino == (target / 'failed_1.npz').stat().st_ino
    with pytest.raises(FileExistsError): runner.preserve_attempts(study, target)
    assert (archive.read_bytes(), attempt.read_bytes()) == before


@pytest.mark.parametrize('change', ['plan', 'inputs', 'absent', 'resources', 'id', 'permission'])
def test_frozen_continuation_cannot_change_inputs_or_remaining_work(monkeypatch, tmp_path, change):
    current = dict(original_registration='original.json', review='review.json', diagnostic_audit='audit.json',
        input_sha256={'original': 'hash'}, absent_inputs=['missing'], plan={'remaining': [3, 4]},
        resources=runner.RESOURCES, original_model_inputs_key='model')
    frozen = dict(copy.deepcopy(current), schema_version=1, id=runner.IDENTIFIER, status='frozen',
        purpose='interrupted_profile_execution_continuation', definition=runner.DEFINITION, publication_allowed=False)
    if change == 'plan': frozen['plan']['remaining'] = [4]
    elif change == 'inputs': frozen['input_sha256']['original'] = 'changed'
    elif change == 'absent': frozen['absent_inputs'] = []
    elif change == 'resources': frozen['resources']['shared_CPUQuotaPerSecUSec'] = '8s'
    elif change == 'id': frozen['id'] = 'replacement'
    else: frozen['publication_allowed'] = True
    path = tmp_path / 'registration.json'
    path.write_text(json.dumps(frozen))
    monkeypatch.setattr(runner.records, 'committed', lambda _: None)
    monkeypatch.setattr(runner, 'bound_inputs', lambda *a: (current, {}, {}))
    with pytest.raises(ValueError): runner.checked_registration(path)


@pytest.mark.parametrize('change', ['none', 'shared_cpu', 'memory', 'membership', 'other_work', 'oom_group', 'devices'])
def test_actual_service_enforces_current_resources_and_exclusive_budget(monkeypatch, tmp_path, change):
    unit = f'f1-{runner.IDENTIFIER}.service'
    group = '/user/f1.slice/' + unit
    properties = dict(Slice='f1.slice', MemoryMax=str(runner.RESOURCES['MemoryMax']), Nice='19', Restart='no',
        OOMPolicy='continue', ControlGroup=group, ActiveState='active', SubState='running')
    shared = dict(CPUQuotaPerSecUSec='4s', CPUWeight='10')
    if change == 'shared_cpu': shared['CPUQuotaPerSecUSec'] = '8s'
    if change == 'memory': properties['MemoryMax'] = str(3 * 1024**3)
    monkeypatch.setattr(runner, 'properties', lambda name, fields: shared if name == 'f1.slice' else properties)
    monkeypatch.setattr(runner.os, 'getpriority', lambda *a: 19)
    monkeypatch.setenv('XLA_FLAGS', '--xla_force_host_platform_device_count=' + ('8' if change == 'devices' else '4'))
    monkeypatch.delenv('JAX_PLATFORMS', raising=False)
    monkeypatch.delenv('JAX_ENABLE_X64', raising=False)
    current = tmp_path / unit
    current.mkdir()
    sibling = tmp_path / 'other.scope'
    sibling.mkdir()
    (sibling / 'cgroup.events').write_text('populated ' + ('1' if change == 'other_work' else '0'))
    real_read, real_iter = Path.read_text, Path.iterdir
    def read(path, *a, **kw):
        if str(path) == '/proc/self/cgroup': return '0::' + (group + '/wrong' if change == 'membership' else group)
        if str(path).endswith('/memory.oom.group'): return '1' if change == 'oom_group' else '0'
        return real_read(path, *a, **kw)
    monkeypatch.setattr(Path, 'read_text', read)
    monkeypatch.setattr(Path, 'iterdir', lambda path:
        iter([sibling]) if str(path) == '/sys/fs/cgroup/user/f1.slice' else real_iter(path))
    if change == 'none': assert runner.actual_service(unit)['shared_slice'] == shared
    else:
        with pytest.raises(ValueError): runner.actual_service(unit)
