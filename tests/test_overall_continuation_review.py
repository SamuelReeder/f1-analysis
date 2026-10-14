"""Continuation aggregation must preserve failure, uncertainty and provenance."""
import copy
from pathlib import Path

import pytest

from analysis import overall_continuation_review as review
from analysis.overall_profile_continuation import fixed_settings


@pytest.fixture
def evidence():
    parent = dict(studies=['connected', 'real_network'],
        fit_settings=dict(chains=4, warmup=800, samples=800, seed=0, target_accept=.9),
        retry=[[1, 1, None], [2, 2, None], [3, 4, None], [3, 4, .98]])

    def attempt(index, converged):
        return dict(fixed_settings(parent, index), seconds=10. + index,
                    diagnostics=dict(converged=converged, divergences=0))

    old = dict(state='running', attempts=[attempt(0, False), attempt(1, False)],
        schema_version=1, run_key='original key', source={'key': 'scientific source'},
        catalog={'field': ['A', 'B']}, publication_allowed=False,
        source_commit='original source', started_at='original start', pid=101)
    old['diagnostics'] = old['attempts'][-1]['diagnostics']

    def reviewed(attempts, directory):
        return [dict(attempt=i+1, recorded=copy.deepcopy(row), issues=[],
            expected_settings=fixed_settings(parent, i), archive_present=True,
            archive=f'{directory}/attempt_{i+1}.npz', archive_sha256=f'archive-{i+1}',
            attempt_file_sha256=f'metadata-{i+1}') for i, row in enumerate(attempts)]

    completed = dict(case='connected', completion={'state': 'converged'}, issues=[],
                     receipt={'state': 'converged'}, attempts=[])
    pending = dict(case='real_network', completion=None, completion_record_present=False,
                   issues=[], receipt=old, receipt_sha256='original receipt',
                   attempts=reviewed(old['attempts'], 'old'))
    original = dict(original_run_complete=False, records_consistent=True, issues=[],
                    studies=[completed, pending])
    frozen = dict(plan=dict(case='real_network', preserved_attempts=2,
        completed_case_records=[dict(case='connected', completion={'state': 'converged'})]))
    receipt = copy.deepcopy(old)
    receipt.update(state='converged', pid=202,
        execution_continuation=dict(historical_attempts=2, original_receipt_sha256='original receipt',
            original_cache='old', registration='continuation.json', registration_sha256='registration hash',
            duration_scope='new execution segment only'))
    receipt['attempts'].append(attempt(2, True))
    receipt['diagnostics'] = receipt['attempts'][-1]['diagnostics']
    result = dict(case='real_network', state='converged', receipt=receipt,
        original_profile_complete=False, original_records_replaced=False,
        full_recovery_complete=False, sbc_complete=False, publication_allowed=False,
        registration='continuation.json', registration_sha256='registration hash',
        original_completed_case_records=copy.deepcopy(frozen['plan']['completed_case_records']),
        combined_recorded_convergence=True)
    return [frozen, parent, original, result, receipt, reviewed(receipt['attempts'], 'new'), 0]


def test_continuation_preserves_interrupted_history_and_every_failed_attempt(evidence):
    before = copy.deepcopy(evidence)
    combined = review.combine(*evidence)
    assert evidence == before
    assert combined['records_consistent'] and combined['combined_recorded_convergence']
    assert all(combined[key] is False for key in review.LIMITS)
    case = combined['studies'][-1]
    assert case['completion'] is None and case['completion_record_present'] is False
    assert case['receipt']['state'] == 'running'
    assert [a['diagnostics']['converged'] for a in case['receipt']['attempts']] == [False, False]
    assert [a['recorded']['diagnostics']['converged'] for a in case['continuation']['attempts']] == [False, False, True]


@pytest.mark.parametrize('change', ['source', 'historical_time', 'historical_archive', 'historical_metadata',
    'seed', 'extra_attempt', 'omitted_case', 'original_completion', 'publication', 'exit', 'summary',
    'prior_success', 'missing_archive', 'missing_reviewed_attempt', 'reordered_attempts',
    'receipt_binding', 'cache_binding', 'duration_scope', 'earlier_case_failure'])
def test_changed_evidence_cannot_become_success(evidence, change):
    frozen, parent, original, result, receipt, attempts, _ = evidence
    if change == 'source': receipt['source']['key'] = 'changed'
    elif change == 'historical_time': receipt['attempts'][0]['seconds'] += 1
    elif change == 'historical_archive': attempts[0]['archive_sha256'] = 'changed'
    elif change == 'historical_metadata': attempts[0]['attempt_file_sha256'] = 'changed'
    elif change == 'seed': receipt['attempts'][-1]['seed'] += 1
    elif change == 'extra_attempt':
        receipt['attempts'].extend(copy.deepcopy(receipt['attempts'][-1:]) * 2)
    elif change == 'omitted_case': original['studies'].pop(0)
    elif change == 'original_completion': original['original_run_complete'] = True
    elif change == 'publication': result['publication_allowed'] = True
    elif change == 'exit': evidence[-1] = 1
    elif change == 'summary': result['combined_recorded_convergence'] = False
    elif change == 'prior_success': receipt['attempts'][0]['diagnostics']['converged'] = True
    elif change == 'missing_archive': attempts[-1]['archive_present'] = False
    elif change == 'missing_reviewed_attempt': attempts.pop()
    elif change == 'reordered_attempts': attempts.reverse()
    elif change == 'receipt_binding': receipt['execution_continuation']['original_receipt_sha256'] = 'changed'
    elif change == 'cache_binding': receipt['execution_continuation']['original_cache'] = 'different'
    elif change == 'duration_scope': receipt['execution_continuation']['duration_scope'] = 'whole original run'
    else: original['studies'][0]['completion']['state'] = 'failed_convergence'
    combined = review.combine(*evidence)
    assert not combined['records_consistent'] and not combined['combined_recorded_convergence']
    assert combined['issues'] and combined['publication_allowed'] is False


def test_completed_audit_of_exhausted_failure_never_turns_it_into_convergence(evidence):
    frozen, parent, original, result, receipt, attempts, _ = evidence
    receipt['attempts'][-1]['diagnostics']['converged'] = False
    last = dict(fixed_settings(parent, 3), diagnostics={'converged': False, 'divergences': 0}, seconds=22.)
    receipt['attempts'].append(last)
    receipt.update(state='failed_convergence', diagnostics=last['diagnostics'])
    attempts[-1]['recorded'] = copy.deepcopy(receipt['attempts'][-2])
    attempts.append(dict(attempt=4, recorded=last, expected_settings=fixed_settings(parent, 3),
                         issues=[], archive_present=True))
    result.update(state='failed_convergence', combined_recorded_convergence=False)
    evidence[-1] = 1
    combined = review.combine(*evidence)
    assert combined['records_consistent']
    assert not combined['combined_recorded_convergence']
    assert len(combined['studies'][-1]['continuation']['attempts']) == 4


def test_post_sampling_exception_with_converged_receipt_stays_failed(evidence):
    evidence[3].update(state='failed_exception', error='ValueError', message='recovery step failed')
    evidence[3].pop('combined_recorded_convergence')
    evidence[3].pop('original_completed_case_records')
    evidence[-1] = 1
    combined = review.combine(*evidence)
    assert combined['records_consistent']
    assert not combined['combined_recorded_convergence']
    assert combined['studies'][-1]['continuation']['result']['error'] == 'ValueError'


@pytest.mark.parametrize('failure', ['none', 'bad_metadata', 'missing_archive', 'changed_archive', 'different_report'])
def test_new_audit_preserves_failures_and_continues_later_attempts(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(review, 'ROOT', tmp_path)
    attempts = []
    for i in range(1, 5):
        archive = tmp_path / f'{i}.npz'
        archive.write_bytes(str(i).encode())
        attempts.append(dict(attempt=i, archive=archive.name, archive_present=True, issues=[],
            archive_sha256=review.records.sha(archive), expected_settings={'index': i},
            recorded={'diagnostics': {'converged': False, 'divergences': 0}}))
    if failure == 'bad_metadata': attempts[2]['issues'].append('bad original metadata')
    elif failure == 'missing_archive': attempts[2]['archive_present'] = False
    elif failure == 'changed_archive': (tmp_path / '3.npz').write_bytes(b'changed')
    calls = []

    def reproduce(archive, sites, settings, catalog, diagnostic, temporary):
        calls.append(settings['index'])
        return dict(archive_arrays_verified=True, coordinate_report_equal=True,
            comparison_reports_equal=not (failure == 'different_report' and settings['index'] == 3),
            coordinate_report={'converged': False})

    monkeypatch.setattr(review.diagnostics, 'reproduce', reproduce)
    checkpoints = []
    checked = review.audit_new_attempts(attempts, 2, {'all': 'sites'}, {}, tmp_path,
                                       lambda rows: checkpoints.append(copy.deepcopy(rows)))
    assert [row['attempt'] for row in checked] == [3, 4]
    assert checked[-1]['state'] == 'verified' and 4 in calls
    assert 1 not in calls and 2 not in calls  # Original audits are reused by identity, not rerun.
    assert checked[0]['state'] == ('verified' if failure == 'none' else
                                   'different' if failure == 'different_report' else 'failed')
    assert checkpoints[-1] == checked


def test_memory_guard_runs_before_reading_results_or_creating_output(monkeypatch, tmp_path):
    def occupied(_):
        raise ValueError('other f1 work still active')
    monkeypatch.setattr(review, 'exclusive_scope', occupied)
    with pytest.raises(ValueError, match='active'):
        review.run(tmp_path / 'registration', tmp_path / 'exit', tmp_path / 'output', 'scope')
    assert not (tmp_path / 'output').exists()


def test_partial_extra_and_missing_archives_are_all_retained(tmp_path):
    for name in ('receipt.json', 'attempt_3.json', 'failed_3.tmp.npz', 'failed_4.npz'):
        (tmp_path / name).write_bytes(b'preserve')
    attempts = [dict(attempt=3, archive='cache/failed_3.npz')]
    inventory = review.cache_inventory(tmp_path, attempts)
    assert inventory['missing'] == ['failed_3.npz']
    assert inventory['unexpected'] == ['failed_3.tmp.npz', 'failed_4.npz']
    assert set(inventory['present']) == {path.name for path in tmp_path.iterdir()}
    assert all(path.read_bytes() == b'preserve' for path in tmp_path.iterdir())
