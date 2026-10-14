"""Continuation planning cannot silently buy new seeds or omit prior failures."""
import copy

import pytest

from analysis.overall_profile_continuation import continuation_plan, fixed_settings, REPORT_KEYS


def fixture():
    registration = dict(status='frozen', purpose='numerical_profile', profile='complete_design_inference',
        publication_allowed=False, studies=['completed', 'interrupted'],
        fit_settings=dict(warmup=800, samples=800, chains=4, seed=0, target_accept=.9),
        retry=[[1, 1, None], [2, 2, None], [3, 4, None], [3, 4, .98]])
    event = dict(study='interrupted', attempt=3, state='sampling', settings=fixed_settings(registration, 2))
    metadata = dict(records_consistent=True, all_required_cases_accounted=True, original_run_complete=False,
        interruption_record='original-interruption.json', last_recorded_sampling_start=event,
        publication_allowed=False, studies=[])
    audit = dict(state='complete', all_processed=True, all_recorded_reports_reproduced=True,
        original_run_complete=False, original_records_consistent=True, interruption_record='original-interruption.json',
        last_recorded_sampling_start=event, publication_allowed=False, issues=[], studies=[])
    for case in registration['studies']:
        # Keep a previously failed terminal case; it must never be selectively rerun.
        complete = dict(state='failed', exit_code=1) if case == 'completed' else None
        state = 'failed_convergence' if complete else 'running'
        reviewed = dict(case=case, completion=complete, issues=[], attempts=[], receipt=dict(state=state, attempts=[]))
        checked = dict(case=case, original_completion=complete, original_receipt_state=state, issues=[], attempts=[])
        for index in range(4 if complete else 2):
            settings = fixed_settings(registration, index)
            diagnostic = dict(converged=False, failures=['fixture failure'], parameters=[], constants=[], empty_sites=[],
                n_draws=settings['chains'] * settings['samples'], n_chains=settings['chains'], divergences=0,
                comparison_reports={'fixture': True})
            recorded = dict(settings, diagnostics=diagnostic)
            saved = dict(attempt=index+1, recorded=recorded, expected_settings=settings, archive_present=True,
                archive=f'{case}/failed_{index+1}.npz', archive_sha256=f'{case}-{index}',
                archive_role='unaccepted_diagnostic_only', issues=[])
            reproduced = dict(attempt=index+1, state='verified', archive_arrays_verified=True,
                coordinate_report_equal=True, comparison_reports_equal=True,
                archive=saved['archive'], archive_sha256=saved['archive_sha256'], original_role=saved['archive_role'],
                original_diagnostics=diagnostic, coordinate_report={k: diagnostic[k] for k in REPORT_KEYS},
                comparison_reports=diagnostic['comparison_reports'])
            reviewed['receipt']['attempts'].append(recorded)
            reviewed['attempts'].append(saved)
            checked['attempts'].append(reproduced)
        metadata['studies'].append(reviewed)
        audit['studies'].append(checked)
    return registration, metadata, audit


def test_only_remaining_original_attempts_and_seeds_are_planned():
    registration, metadata, audit = fixture()
    before = copy.deepcopy((registration, metadata, audit))
    plan = continuation_plan(registration, metadata, audit)
    assert plan['case'] == 'interrupted' and plan['preserved_attempts'] == 2
    assert [(a['attempt'], a['settings']['seed'], a['settings']['target_accept'])
            for a in plan['remaining_attempts']] == [(3, 2, .9), (4, 3, .98)]
    assert all(a['settings']['warmup'] == 2400 and a['settings']['samples'] == 3200 for a in plan['remaining_attempts'])
    assert plan['completed_case_records'] == [{'case': 'completed', 'completion': {'state': 'failed', 'exit_code': 1}}]
    assert plan['stop_on_first_convergence'] and not plan['retry_exception'] and not plan['publication_allowed']
    assert (registration, metadata, audit) == before


@pytest.mark.parametrize('change', ['review_failed', 'audit_failed', 'complete_run', 'missing_case', 'reordered_case',
    'missing_attempt', 'unverified_arrays', 'wrong_bytes', 'wrong_report', 'wrong_role', 'changed_seed',
    'event_seed', 'event_attempt', 'new_result', 'terminal_receipt', 'past_success', 'extra_attempt'])
def test_no_changed_or_unverified_evidence_can_authorize_continuation(change):
    registration, metadata, audit = fixture()
    last, checked = metadata['studies'][-1], audit['studies'][-1]
    if change == 'review_failed': metadata['records_consistent'] = False
    elif change == 'audit_failed': audit['all_recorded_reports_reproduced'] = False
    elif change == 'complete_run': metadata['original_run_complete'] = True
    elif change == 'missing_case': metadata['studies'].pop(0)
    elif change == 'reordered_case': audit['studies'].reverse()
    elif change == 'missing_attempt': checked['attempts'].pop()
    elif change == 'unverified_arrays': checked['attempts'][0]['archive_arrays_verified'] = False
    elif change == 'wrong_bytes': checked['attempts'][0]['archive_sha256'] = 'other'
    elif change == 'wrong_report': checked['attempts'][0]['coordinate_report']['converged'] = True
    elif change == 'wrong_role': checked['attempts'][0]['original_role'] = 'accepted'
    elif change == 'changed_seed': last['receipt']['attempts'][0]['seed'] = 99
    elif change == 'event_seed': metadata['last_recorded_sampling_start']['settings']['seed'] = 99
    elif change == 'event_attempt': metadata['last_recorded_sampling_start']['attempt'] = 4
    elif change == 'new_result': last['result'] = {'state': 'failed'}
    elif change == 'terminal_receipt': last['receipt']['state'] = checked['original_receipt_state'] = 'failed_exception'
    elif change == 'past_success':
        last['receipt']['attempts'][0]['diagnostics']['converged'] = True
        checked['attempts'][0]['coordinate_report']['converged'] = True
    else:
        # A forged successful verification of an already exhausted interrupted case
        # still cannot purchase another attempt.
        completed = copy.deepcopy(metadata['studies'][0])
        verified = copy.deepcopy(audit['studies'][0])
        completed.update(case='interrupted', completion=None)
        completed['receipt']['state'] = 'running'
        verified.update(case='interrupted', original_completion=None, original_receipt_state='running')
        metadata['studies'][-1], audit['studies'][-1] = completed, verified
    with pytest.raises(ValueError):
        continuation_plan(registration, metadata, audit)
