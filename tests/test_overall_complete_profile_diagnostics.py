"""Saved-byte/report fixtures; these are not scientific acceptance evidence."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from analysis import overall_complete_profile_diagnostics as audit
from analysis.overall_bounded_diagnostics import check
from analysis.overall_weekend_profile import contrast_reports


@pytest.fixture
def posterior(tmp_path):
    rng = np.random.default_rng(913)
    settings = dict(chains=4, samples=128)
    # Nonalphabetical saved order exercises the report's original site order.
    arrays = dict(stochastic=np.ones((4, 128), dtype=np.float32),
        skill=rng.normal(size=(4, 128, 3)).astype(np.float32),
        overall_race_skill=rng.normal(size=(4, 128, 3)).astype(np.float32),
        empty=np.empty((4, 128, 0), dtype=np.float32))
    sites = {k: dict(shape=list(v.shape[2:]), dtype=v.dtype.str) for k, v in arrays.items()}
    catalog = dict(field=['A', 'B', 'C'], field_entries=[2, 0, 1])
    diagnostic = check(arrays, 0)
    diagnostic.update(comparison_reports=contrast_reports(arrays, catalog, 0),
        mean_accept_prob=.9, tree_depth_limit_fraction=.1, runtime={'fixture': True})
    assert diagnostic['converged'] is False  # Deliberately failed stochastic coordinate.
    archive = tmp_path / 'saved.npz'
    np.savez_compressed(archive, **arrays)
    temporary = tmp_path / 'maps'
    temporary.mkdir()
    return arrays, sites, settings, catalog, diagnostic, archive, temporary


def test_complete_arrays_and_failed_reports_reproduce_without_rescuing_fit(posterior):
    _, sites, settings, catalog, diagnostic, archive, temporary = posterior
    observed = audit.reproduce(archive, sites, settings, catalog, diagnostic, temporary)
    assert observed['archive_arrays_verified']
    assert observed['coordinate_report_equal'] and observed['comparison_reports_equal']
    assert observed['coordinate_report']['converged'] is False
    assert observed['comparison_reports'] == diagnostic['comparison_reports']
    assert observed['divergence_flags_recomputed'] is False
    assert set(observed['other_recorded_fields_not_recomputed']) == {
        'mean_accept_prob', 'tree_depth_limit_fraction', 'runtime'}
    assert not list(temporary.iterdir())


@pytest.mark.parametrize('change', ['extra', 'missing', 'dtype', 'axes', 'shape', 'nonfinite', 'object'])
def test_whole_archive_rejects_missing_extra_or_invalid_arrays(posterior, change):
    arrays, sites, settings, catalog, diagnostic, archive, temporary = posterior
    arrays = dict(arrays)
    if change == 'extra':
        arrays['unexpected'] = np.zeros((4, 128), dtype=np.float32)
    elif change == 'missing':
        del arrays['empty']
    elif change == 'dtype':
        arrays['skill'] = arrays['skill'].astype(np.float64)
    elif change == 'axes':
        arrays['skill'] = arrays['skill'][:, :-1]
    elif change == 'shape':
        arrays['skill'] = arrays['skill'][..., :-1]
    elif change == 'nonfinite':
        arrays['stochastic'] = np.full((4, 128), np.nan, dtype=np.float32)
    else:
        arrays['stochastic'] = np.full((4, 128), object(), dtype=object)
    np.savez_compressed(archive, **arrays)
    with pytest.raises(ValueError):
        audit.reproduce(archive, sites, settings, catalog, diagnostic, temporary)
    assert not list(temporary.iterdir())


def test_every_numeric_report_difference_is_retained(posterior):
    _, sites, settings, catalog, diagnostic, archive, temporary = posterior
    expected = copy.deepcopy(diagnostic)
    expected['parameters'][0]['rhat_max'] += .1
    expected['parameters'][-1]['bulk_ess_min'] += 1
    expected['comparison_reports']['current_driver_contrasts'][0]['chain_means'][0] += 1
    expected['comparison_reports']['current_driver_contrasts'][-1]['chain_means'][-1] -= 1
    result = audit.reproduce(archive, sites, settings, catalog, expected, temporary)
    assert len(result['coordinate_difference_paths']) == 2
    assert len(result['comparison_difference_paths']) == 2
    assert not result['coordinate_report_equal'] and not result['comparison_reports_equal']
    assert result['coordinate_report'] == {k: diagnostic[k] for k in audit.REPORT_KEYS}
    assert result['comparison_reports'] == diagnostic['comparison_reports']


def study_fixture(monkeypatch, tmp_path, posterior):
    _, sites, settings, catalog, diagnostic, archive, temporary = posterior
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    original = dict(studies=[], input_sha256={})
    registration = dict(id='fixture', storage_results={})
    for case in ('first', 'middle', 'last'):
        cache = tmp_path / 'outputs/championship/cache/fixture' / case
        cache.mkdir(parents=True)
        target = cache / 'failed_1.npz'
        target.write_bytes(archive.read_bytes())
        storage = tmp_path / f'{case}.storage.json'
        storage.write_text(json.dumps(dict(case=case, state='complete', sites=sites)))
        registration['storage_results'][case] = storage.name
        original['input_sha256'][storage.name] = audit.review.sha(storage)
        recorded = dict(settings, diagnostics=diagnostic)
        attempt = dict(attempt=1, recorded=recorded, issues=[], expected_settings=settings,
            archive=str(target.relative_to(tmp_path)), archive_sha256=audit.review.sha(target),
            archive_present=True, archive_role='unaccepted_diagnostic_only')
        original['studies'].append(dict(case=case, completion=dict(state='failed'), issues=[],
            receipt=dict(state='failed_convergence', catalog=catalog, attempts=[recorded]),
            attempts=[attempt]))
    return original, registration, temporary


@pytest.mark.parametrize('failure', ['changed_bytes', 'absent_archive', 'missing_receipt', 'bad_metadata', 'different_report'])
def test_bad_middle_case_keeps_earlier_and_later_cases(monkeypatch, tmp_path, posterior, failure):
    original, registration, temporary = study_fixture(monkeypatch, tmp_path, posterior)
    middle = original['studies'][1]
    archive = tmp_path / middle['attempts'][0]['archive']
    if failure == 'changed_bytes':
        archive.write_bytes(b'corrupt')
    elif failure == 'absent_archive':
        archive.unlink()
        middle['attempts'][0]['archive_present'] = False
    elif failure == 'missing_receipt':
        middle.pop('receipt')
        middle['attempts'] = []
    elif failure == 'bad_metadata':
        middle['issues'].append('Original metadata failed verification')
    else:
        # A matching receipt/review can still contain a wrong numerical report.
        middle['receipt'] = copy.deepcopy(middle['receipt'])
        middle['receipt']['attempts'][0]['diagnostics']['converged'] = True
        middle['attempts'][0]['recorded'] = middle['receipt']['attempts'][0]
    checkpoints = []
    studies = audit.verify_studies(original, registration, temporary,
        lambda rows: checkpoints.append(copy.deepcopy(rows)))
    assert [s['case'] for s in studies] == ['first', 'middle', 'last']
    assert studies[0]['attempts'][0]['state'] == studies[2]['attempts'][0]['state'] == 'verified'
    assert all(s['original_completion']['state'] == 'failed' for s in studies)
    assert not audit.all_reproduced(studies)
    assert checkpoints[-1] == studies
    if failure == 'missing_receipt':
        assert studies[1]['issues'] and not studies[1]['attempts']
    else:
        expected = 'unavailable' if failure == 'absent_archive' else 'different' if failure == 'different_report' else 'failed'
        assert studies[1]['attempts'][0]['state'] == expected


def test_failed_attempt_does_not_hide_later_attempt(monkeypatch, tmp_path, posterior):
    original, registration, temporary = study_fixture(monkeypatch, tmp_path, posterior)
    middle = original['studies'][1]
    later = copy.deepcopy(middle['attempts'][0])
    later['attempt'] = 2
    middle['attempts'].append(later)
    middle['receipt']['attempts'].append(later['recorded'])
    middle['attempts'][0]['issues'].append('First attempt metadata is inconsistent')
    middle['issues'].append('attempt 1: First attempt metadata is inconsistent')
    result = audit.verify_studies(original, registration, temporary, lambda _: None)
    assert [a['state'] for a in result[1]['attempts']] == ['failed', 'verified']
    assert result[-1]['attempts'][0]['state'] == 'verified'
    assert not audit.all_reproduced(result)


def test_empty_inventory_cannot_pass():
    assert not audit.all_reproduced([])
    assert not audit.all_reproduced([dict(issues=[], attempts=[])])


@pytest.mark.parametrize('input_changed', [False, True])
@pytest.mark.parametrize('interrupted', [False, True])
def test_final_audit_preserves_failed_inference_and_rejects_changed_inputs(monkeypatch, tmp_path, posterior, input_changed, interrupted):
    original, registration, _ = study_fixture(monkeypatch, tmp_path, posterior)
    original.update(issues=[], records_consistent=True, recorded_all_converged=False,
        terminal_service={'state': 'exited'}, registered_resources={'historical': True},
        resource_changes=[{'user_requested_cpu_reduction': True}], runtime_spans_changed_resources=True,
        original_run_complete=not interrupted, interruption_record='interruption.json' if interrupted else None,
        last_recorded_sampling_start={'state': 'sampling'} if interrupted else None)
    monkeypatch.setattr(audit, 'actual_scope', lambda _: {'test_scope': True})
    monkeypatch.setattr(audit, 'evidence', lambda _: (original, registration, {}, []))
    monkeypatch.setattr(audit, 'changed_inputs', lambda *a:
        ['Input changed during verification'] if input_changed else [])
    monkeypatch.setattr(audit.subprocess, 'check_output', lambda *a, **kw: 'fixture-commit\n')
    output = tmp_path / 'audit'
    result = audit.run(tmp_path / 'review.json', output, 'fixture.scope')
    assert result['all_processed']
    assert result['all_recorded_reports_reproduced'] is (not input_changed)
    assert result['state'] == ('incomplete' if input_changed else 'complete')
    assert result['recorded_all_converged'] is False
    assert result['original_run_complete'] is (not interrupted)
    assert result['interruption_record'] == original['interruption_record']
    assert all(s['original_receipt_state'] == 'failed_convergence' for s in result['studies'])
    assert result['original_runtime_spans_changed_resources'] is True
    assert result['original_resource_changes'] == original['resource_changes']
    assert json.loads((output / 'audit.json').read_text()) == result
    for name in ('divergence_flags_recomputed', 'mcmc_run', 'full_recovery_complete',
                 'sbc_complete', 'publication_allowed', 'full_goal_complete'):
        assert result[name] is False
    with pytest.raises(FileExistsError):
        audit.run(tmp_path / 'review.json', output, 'fixture.scope')


def test_input_recheck_covers_removed_changed_and_previously_absent_files(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    tracked = tmp_path / 'original'
    tracked.write_text('fixed')
    fingerprint = {'original': audit.review.sha(tracked)}
    assert not audit.changed_inputs(fingerprint, ['absent'])
    tracked.write_text('changed')
    (tmp_path / 'absent').write_text('appeared')
    assert len(audit.changed_inputs(fingerprint, ['absent'])) == 2
    tracked.unlink()
    assert len(audit.changed_inputs(fingerprint, [])) == 1


@pytest.mark.parametrize('interrupted', [False, True])
def test_evidence_replays_collector_and_rejects_a_changed_review(monkeypatch, tmp_path, interrupted):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    monkeypatch.setattr(audit, 'SOURCES', ())
    monkeypatch.setattr(audit.review, 'committed', lambda _: None)
    registration = tmp_path / 'registration.json'
    registration.write_text(json.dumps(dict(id='fixture')))
    current = dict(registration=registration.name, input_sha256={}, studies=[],
        records_consistent=True, recorded_at='later', terminal_service={'state': 'exited'},
        interruption_record='interruption.json' if interrupted else None,
        absent_terminal_records=['never_written.json'] if interrupted else [])
    saved = dict(current, recorded_at='earlier', terminal_service={'state': 'inactive'})
    path = tmp_path / 'review.json'
    path.write_text(json.dumps(saved))
    calls = []
    def collect(path, **kwargs):
        calls.append((path, kwargs))
        return current
    monkeypatch.setattr(audit.review, 'collect', collect)
    _, _, _, absent = audit.evidence(path)
    assert calls == [(registration, {'interruption': tmp_path / 'interruption.json'} if interrupted else {})]
    assert absent == current['absent_terminal_records']
    current['studies'].append(dict(case='new'))
    with pytest.raises(ValueError, match='no longer matches'):
        audit.evidence(path)


def test_changed_loaded_source_is_rejected_before_collecting(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    source = tmp_path / 'source.py'
    source.write_text('original source')
    monkeypatch.setattr(audit, 'SOURCES', ('source.py',))
    monkeypatch.setattr(audit, 'LOADED', {'source.py': audit.review.sha(source)})
    source.write_text('changed source')
    path = tmp_path / 'review.json'
    path.write_text('{}')
    monkeypatch.setattr(audit.review, 'committed', lambda _: None)
    monkeypatch.setattr(audit.review, 'collect', lambda _: pytest.fail('Source validation must precede collection'))
    with pytest.raises(ValueError, match='Loaded verification source changed'):
        audit.evidence(path)


@pytest.mark.parametrize('change', ['none', 'cpu', 'memory', 'group', 'priority'])
def test_scope_must_be_the_actual_process_scope(monkeypatch, change):
    actual = dict(Slice='f1.slice', MemoryMax=str(3 * 1024**3), CPUQuotaPerSecUSec='2s',
        ControlGroup='/user/f1.slice/check.scope', ActiveState='active')
    if change == 'cpu': actual['CPUQuotaPerSecUSec'] = '4s'
    if change == 'memory': actual['MemoryMax'] = str(13 * 1024**3)
    if change == 'group': actual['ControlGroup'] = '/user/f1.slice/other.scope'
    monkeypatch.setattr(audit.subprocess, 'check_output', lambda *a, **kw:
        '\n'.join(f'{k}={v}' for k, v in actual.items()))
    real_read = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **kw:
        '0::/user/f1.slice/check.scope\n' if str(path) == '/proc/self/cgroup' else real_read(path, *a, **kw))
    monkeypatch.setattr(audit.os, 'getpriority', lambda *a: 0 if change == 'priority' else 19)
    if change == 'none':
        assert audit.actual_scope('check.scope')['Nice'] == 19
    else:
        with pytest.raises(ValueError, match='inside the separate capped'):
            audit.actual_scope('check.scope')
