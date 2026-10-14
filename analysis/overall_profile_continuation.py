"""Fixed-ladder planning for a verified interrupted profile.

This module does not launch a fit. A separately verified execution controller
and committed continuation registration are required before sampling.
"""

REPORT_KEYS = ('converged', 'failures', 'parameters', 'constants', 'empty_sites',
               'n_draws', 'n_chains', 'divergences')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fixed_settings(registration, index):
    """Zero-based original attempt index, never a new continuation seed schedule."""
    require(type(index) is int and 0 <= index < len(registration['retry']), 'No extra retry is permitted')
    warm, draws, acceptance = registration['retry'][index]
    base = registration['fit_settings']
    return dict(base, warmup=warm * base['warmup'], samples=draws * base['samples'],
        seed=base['seed'] + index,
        target_accept=base['target_accept'] if acceptance is None else acceptance)


def continuation_plan(registration, metadata, audit):
    """Require full saved-report evidence; immutable-file checks belong to the caller."""
    require(registration['status'] == 'frozen' and registration['profile'] == 'complete_design_inference'
            and registration['purpose'] == 'numerical_profile' and registration['publication_allowed'] is False,
            'Only the original frozen numerical profile can be continued')
    require(metadata['records_consistent'] is True and metadata['all_required_cases_accounted'] is True
            and metadata['original_run_complete'] is False and bool(metadata['interruption_record'])
            and metadata['publication_allowed'] is False, 'Need a consistent explicit interruption review')
    require(audit['state'] == 'complete' and audit['all_processed'] is True
            and audit['all_recorded_reports_reproduced'] is True
            and audit['original_run_complete'] is False and audit['original_records_consistent'] is True
            and audit['interruption_record'] == metadata['interruption_record']
            and audit['last_recorded_sampling_start'] == metadata['last_recorded_sampling_start']
            and audit['publication_allowed'] is False and not audit['issues'],
            'Every saved report must reproduce before an execution continuation')
    cases = registration['studies']
    require(bool(cases) and len(cases) == len(set(cases))
            and [r['case'] for r in metadata['studies']] == cases
            and [r['case'] for r in audit['studies']] == cases,
            'Do not omit, reorder or replace any original case')
    pending = []
    for reviewed, checked in zip(metadata['studies'], audit['studies']):
        receipt = reviewed.get('receipt')
        require(not reviewed['issues'] and not checked['issues'] and bool(receipt),
                'Every original case needs consistent receipt evidence')
        require(checked['original_completion'] == reviewed['completion']
                and checked['original_receipt_state'] == receipt['state'], 'Case status differs from the original review')
        recorded = receipt['attempts']
        require(len(recorded) == len(reviewed['attempts']) == len(checked['attempts'])
                and bool(recorded), 'Every recorded attempt must be verified')
        for index, (original, saved, reproduced) in enumerate(zip(recorded, reviewed['attempts'], checked['attempts'])):
            settings = fixed_settings(registration, index)
            require({k: original[k] for k in settings} == settings and saved['expected_settings'] == settings,
                    'The original attempt settings or seed changed')
            require(saved['attempt'] == reproduced['attempt'] == index + 1
                    and saved['recorded'] == original and not saved['issues']
                    and saved['archive_present'] is True, 'An original completed attempt is missing or changed')
            diagnostic = original['diagnostics']
            require(reproduced['state'] == 'verified' and reproduced['archive_arrays_verified'] is True
                    and reproduced['coordinate_report_equal'] is True and reproduced['comparison_reports_equal'] is True
                    and reproduced['archive'] == saved['archive']
                    and reproduced['archive_sha256'] == saved['archive_sha256']
                    and reproduced['original_role'] == saved['archive_role']
                    and reproduced['original_diagnostics'] == diagnostic
                    and reproduced['coordinate_report'] == {k: diagnostic[k] for k in REPORT_KEYS}
                    and reproduced['comparison_reports'] == diagnostic['comparison_reports'],
                    'Saved arrays or numerical reports differ from the original attempt')
        if reviewed['completion'] is None:
            require(receipt['state'] == 'running' and 'result' not in reviewed,
                    'A terminal fit cannot be restarted as an interruption')
            pending.append(reviewed)
        else:
            require(receipt['state'] in ('converged', 'failed_convergence', 'failed_exception'),
                    'A completed case must retain its terminal receipt')
    require(len(pending) == 1, 'This continuation requires exactly one interrupted case with verified saved attempts')
    interrupted = pending[0]
    attempts = interrupted['receipt']['attempts']
    require(len(attempts) < len(registration['retry'])
            and all(a['diagnostics']['converged'] is False for a in attempts),
            'Do not bypass a success or restart an exhausted ladder')
    event = metadata['last_recorded_sampling_start']
    expected = dict(study=interrupted['case'], attempt=len(attempts) + 1, state='sampling',
                    settings=fixed_settings(registration, len(attempts)))
    require(event == expected, 'Restart only the exact unfinished registered attempt')
    return dict(case=interrupted['case'], preserved_attempts=len(attempts),
        interrupted_sampling_start=event,
        remaining_attempts=[dict(attempt=index + 1, settings=fixed_settings(registration, index))
                            for index in range(len(attempts), len(registration['retry']))],
        completed_case_records=[dict(case=row['case'], completion=row['completion'])
                               for row in metadata['studies'] if row['completion'] is not None],
        stop_on_first_convergence=True, retry_exception=False, publication_allowed=False,
        full_goal_complete=False)
