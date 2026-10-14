"""Review a terminal continuation without rewriting the interrupted profile.

This combines recorded computational evidence, then reproduces only the newly
saved reports. Earlier arrays must retain the hashes bound by their completed
audit. This neither samples nor grants scientific acceptance or publication.
"""
import argparse
import copy
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import overall_complete_profile_diagnostics as diagnostics
from analysis import overall_complete_profile_review as records
from analysis.overall_profile_continuation import fixed_settings, require

SOURCES = ('analysis/overall_continuation_review.py',
           'tests/test_overall_continuation_review.py',
           'docs/overall_continuation_review.md')
LOADED = {name: records.sha(ROOT / name) for name in SOURCES}
LIMITS = dict(original_run_complete=False, original_records_replaced=False,
              full_recovery_complete=False, sbc_complete=False,
              publication_allowed=False, full_goal_complete=False)


def exclusive_scope(scope):
    actual = diagnostics.actual_scope(scope)
    group = Path('/sys/fs/cgroup') / actual['ControlGroup'].lstrip('/')
    for sibling in group.parent.iterdir():
        events = sibling / 'cgroup.events'
        if sibling.is_dir() and sibling != group and events.is_file():
            state = dict(line.split() for line in events.read_text().splitlines())
            require(state.get('populated') == '0', 'Wait for all other f1 work to stop')
    return actual


def combine(frozen, parent, original, result, receipt, attempts, exit_status):
    """Keep every case and old failure; a successful segment is not a whole run."""
    studies = copy.deepcopy(original['studies'])
    issues = []

    def check(condition, message):
        if not condition:
            issues.append(message)

    case, preserved = frozen['plan']['case'], frozen['plan']['preserved_attempts']
    check([row['case'] for row in studies] == parent['studies'], 'Original case inventory differs')
    candidates = [row for row in studies if row['case'] == case]
    require(len(candidates) == 1, 'The continued case must occur exactly once')
    study = candidates[0]
    before = study['receipt']
    check(original.get('records_consistent') is True and not original.get('issues')
          and all(not s['issues'] for s in studies), 'Original evidence is inconsistent')
    check(original['original_run_complete'] is False, 'Original profile was not an interruption')
    check(result.get('case') == case, 'Continuation case differs')
    for key in ('original_profile_complete', 'original_records_replaced',
                'full_recovery_complete', 'sbc_complete', 'publication_allowed'):
        check(result.get(key) is False, f'Invalid continuation evidence boundary: {key}')
    check(result.get('receipt') == receipt, 'Result and saved receipt differ')
    check(result.get('state') in ('converged', 'failed_convergence', 'failed_exception'),
          'Continuation has no terminal result')
    check(type(exit_status) is int and 0 <= exit_status <= 255, 'Invalid wrapper exit status')
    check((exit_status == 0) == (result.get('state') == 'converged'),
          'Wrapper exit status differs from the result')
    if result.get('state') != 'failed_exception':
        check(result.get('state') == receipt.get('state'), 'Result and receipt terminal states differ')
        check(result.get('original_completed_case_records') == frozen['plan']['completed_case_records'],
              'Earlier completed cases were changed or omitted')
    # A failure after sampling (for example in descriptive recovery) can coexist
    # with a converged receipt; it must still withhold combined completion.
    if result.get('state') == 'failed_exception':
        check(exit_status != 0, 'An execution exception cannot have a successful exit')
    mutable = {'state', 'pid', 'diagnostics', 'attempts'}
    check({k: receipt.get(k) for k in before if k not in mutable}
          == {k: v for k, v in before.items() if k not in mutable},
          'Original scientific receipt fields changed')
    saved = receipt.get('attempts', [])
    check(len(before['attempts']) == preserved and saved[:preserved] == before['attempts'],
          'Historical attempts were changed or lost')
    check(preserved <= len(saved) <= len(parent['retry']), 'Attempts lie outside the original ladder')
    check(len(attempts) == len(saved), 'Reviewed attempt inventory differs')
    check([row['attempt'] for row in attempts] == list(range(1, len(saved) + 1)),
          'Attempt numbers must be contiguous and unique')
    if receipt.get('state') in ('converged', 'failed_convergence'):
        check(len(saved) > preserved, 'A terminal continuation has no new attempt')
        check(bool(saved) and receipt.get('diagnostics') == saved[-1].get('diagnostics'),
              'Final receipt diagnostics differ')
        check(bool(saved) and (receipt['state'] == 'converged')
              == saved[-1].get('diagnostics', {}).get('converged'), 'Receipt decision differs from its final attempt')
    if receipt.get('state') == 'failed_convergence':
        check(len(saved) == len(parent['retry']), 'Failed continuation did not exhaust the original ladder')
    for index, attempt in enumerate(saved):
        if index >= len(parent['retry']):
            continue
        settings = fixed_settings(parent, index)
        check({k: attempt.get(k) for k in settings} == settings,
              f'Attempt {index + 1} changed its registered settings')
        check(type(attempt.get('diagnostics', {}).get('converged')) is bool,
              f'Attempt {index + 1} has no boolean convergence decision')
        if index < len(saved) - 1:
            check(attempt.get('diagnostics', {}).get('converged') is False,
                  f'Attempt {index + 1} success was bypassed')
    for index, row in enumerate(attempts):
        check(index < len(saved) and row.get('recorded') == saved[index],
              f'Attempt {index + 1} metadata differs from the receipt')
        if receipt.get('state') in ('converged', 'failed_convergence'):
            check(row.get('archive_present') is True, f'Attempt {index + 1} archive is missing')
        if index < preserved and index < len(study['attempts']):
            prior = study['attempts'][index]
            check(row.get('archive_present') is True
                  and row.get('archive_sha256') == prior['archive_sha256']
                  and row.get('attempt_file_sha256') == prior['attempt_file_sha256'],
                  f'Historical attempt {index + 1} bytes changed in the continuation')
        issues.extend(f"attempt {row['attempt']}: {issue}" for issue in row['issues'])
    extension = receipt.get('execution_continuation', {})
    check(extension.get('historical_attempts') == preserved
          and extension.get('original_receipt_sha256') == study['receipt_sha256'],
          'Continuation does not bind the original receipt and attempt count')
    check(extension.get('original_cache') == str(Path(study['attempts'][0]['archive']).parent),
          'Continuation points to a different original cache')
    check(extension.get('duration_scope') == 'new execution segment only',
          'Continuation time must not be labelled as uninterrupted whole-run time')
    check(extension.get('registration') == result.get('registration')
          and extension.get('registration_sha256') == result.get('registration_sha256'),
          'Result and execution registration identities differ')
    study['continuation'] = dict(result=result, receipt=receipt, attempts=attempts,
                                 exit_status=exit_status)
    combined = (result.get('state') == receipt.get('state') == 'converged'
        and len(saved) > preserved and bool(saved)
        and saved[-1]['diagnostics'].get('converged') is True
        and all(s['completion'] is not None and s['completion']['state'] == 'converged'
                for s in studies if s['case'] != case))
    if result.get('state') != 'failed_exception':
        check(result.get('combined_recorded_convergence') is combined,
              'Recorded combined convergence differs from complete case evidence')
    return dict(LIMITS, studies=studies, issues=issues,
        records_consistent=not issues, combined_recorded_convergence=combined and not issues)


def audit_new_attempts(attempts, preserved, sites, catalog, temporary_parent, checkpoint):
    checked = []
    for attempt in attempts[preserved:]:
        begun = time.monotonic()
        row = dict(attempt=attempt['attempt'], state='running', issues=list(attempt['issues']),
                   archive_arrays_verified=False, coordinate_report_equal=False,
                   comparison_reports_equal=False)
        checked.append(row)
        checkpoint(checked)
        try:
            require(not row['issues'], 'Attempt metadata is inconsistent')
            require(attempt.get('archive_present') is True, 'Recorded attempt has no complete saved archive')
            archive = ROOT / attempt['archive']
            row.update(archive=attempt['archive'], archive_sha256=attempt['archive_sha256'])
            require(records.sha(archive) == attempt['archive_sha256'], 'Saved archive changed')
            row.update(diagnostics.reproduce(archive, sites, attempt['expected_settings'],
                catalog, attempt['recorded']['diagnostics'], temporary_parent))
            require(records.sha(archive) == attempt['archive_sha256'], 'Archive changed during reproduction')
            row['state'] = ('verified' if row['coordinate_report_equal']
                            and row['comparison_reports_equal'] else 'different')
        except Exception as exc:
            row.update(state='failed', error=type(exc).__name__, message=str(exc))
            row['issues'].append(f'{type(exc).__name__}: {exc}')
        finally:
            row['seconds'] = time.monotonic() - begun
            checkpoint(checked)
    return checked


def cache_inventory(cache, attempts):
    expected = {'receipt.json'} | {f"attempt_{a['attempt']}.json" for a in attempts}
    expected.update(Path(a['archive']).name for a in attempts if a.get('archive'))
    present = {path.name for path in cache.iterdir()}
    return dict(present=sorted(present), expected=sorted(expected),
                missing=sorted(expected - present), unexpected=sorted(present - expected))


def run(registration, exit_record, output, scope):
    actual = exclusive_scope(scope)
    # Do not even import the fitting controller until the memory budget is free.
    from analysis import overall_profile_continuation_run as controller
    terminal = records.terminal_service(controller.IDENTIFIER)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    destination = output / 'review.json'
    started = time.monotonic()
    report = dict(LIMITS, state='running', started_at=diagnostics.now(),
        actual_scope=actual, terminal_service=terminal, issues=[], new_attempt_audits=[],
        all_new_reports_reproduced=False, combined_recorded_convergence=False,
        combined_computational_evidence_verified=False)
    fingerprints, absent = {}, []

    def checkpoint(rows):
        report['new_attempt_audits'] = rows
        diagnostics.save(destination, report)

    def bind(path, committed=True):
        path = Path(path).resolve()
        if committed:
            records.committed(path)
        fingerprints[diagnostics.relative(path)] = records.sha(path)

    checkpoint([])
    try:
        for name in SOURCES:
            bind(ROOT / name)
            require(fingerprints[name] == LOADED[name], 'Loaded reviewer source changed')
        frozen, parent, original = controller.checked_registration(registration)
        fingerprints.update(frozen['input_sha256'])
        absent.extend(frozen['absent_inputs'])
        report.update(studies=copy.deepcopy(original['studies']),
                      prior_diagnostic_audit=frozen['diagnostic_audit'])
        bind(registration)
        case = frozen['plan']['case']
        raw = ROOT / 'outputs/championship/development' / frozen['id']
        cache = ROOT / 'outputs/championship/cache' / frozen['id'] / case
        bind(raw / 'result.json')
        result = records.read(raw / 'result.json')
        report['continuation_result'] = result
        require(result['registration'] == diagnostics.relative(registration)
                and result['registration_sha256'] == records.sha(registration), 'Result registration differs')
        bind(exit_record)
        live_exit = ROOT / 'outputs/logs' / frozen['id'] / 'exit.status'
        require(Path(exit_record).read_bytes() == live_exit.read_bytes(), 'Saved wrapper exit differs')
        bind(live_exit, committed=False)
        resources = raw.with_name(raw.name + '-launch') / 'resources.txt'
        bind(resources)
        receipt_path = cache / 'receipt.json'
        bind(receipt_path, committed=False)
        receipt = records.read(receipt_path)
        report['continuation_receipt'] = receipt
        attempts = records.attempt_review(receipt, parent, cache)
        inventory = cache_inventory(cache, attempts)
        report['cache_inventory'] = inventory
        for attempt in attempts:
            bind(cache / f"attempt_{attempt['attempt']}.json", committed=False)
            if attempt.get('archive_present'):
                fingerprints[attempt['archive']] = attempt['archive_sha256']
            elif attempt.get('archive'):
                absent.append(attempt['archive'])
        report.update(combine(frozen, parent, original, result, receipt, attempts,
                              int(Path(exit_record).read_text().strip())))
        for name in inventory['unexpected']:
            report['issues'].append(f'Unreferenced continuation cache entry: {name}')
            path = cache / name
            if path.is_file():
                bind(path, committed=False)
        report['issues'].extend(f'Missing continuation cache entry: {name}' for name in inventory['missing'])
        report.update(input_sha256=fingerprints, absent_inputs=absent,
            registration=diagnostics.relative(registration),
            prior_diagnostic_audit=frozen['diagnostic_audit'],
            source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
        # Truth was written before sampling, but can be absent on an early crash.
        if (raw / 'truth.json').is_file():
            bind(raw / 'truth.json')
            prior_truth = ROOT / 'outputs/championship/development' / parent['id'] / case / 'truth.json'
            require(records.read(raw / 'truth.json') == records.read(prior_truth), 'Generated truth changed')
        else:
            report['issues'].append('Continuation has no saved truth record')
            absent.append(diagnostics.relative(raw / 'truth.json'))
        storage = records.read(ROOT / parent['storage_results'][case])
        require(storage['case'] == case and storage['state'] == 'complete', 'Invalid original storage inventory')
        # Retain every new failure even when metadata elsewhere is inconsistent.
        report['new_attempt_audits'] = audit_new_attempts(attempts, frozen['plan']['preserved_attempts'],
            storage['sites'], receipt['catalog'], ROOT / 'outputs/championship/cache', checkpoint)
        report['issues'].extend(diagnostics.changed_inputs(fingerprints, absent))
        report['records_consistent'] = report['records_consistent'] and not report['issues']
        report['all_new_reports_reproduced'] = (bool(report['new_attempt_audits']) and
            all(a['state'] == 'verified' for a in report['new_attempt_audits']))
        report['combined_computational_evidence_verified'] = (report['records_consistent']
            and report['combined_recorded_convergence'] and report['all_new_reports_reproduced'])
        report['state'] = ('complete' if report['records_consistent']
            and report['all_new_reports_reproduced'] else 'incomplete')
    except Exception as exc:
        report.update(state='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc(),
                      records_consistent=False,
                      combined_computational_evidence_verified=False)
    finally:
        report.update(input_sha256=fingerprints, absent_inputs=absent, finished_at=diagnostics.now(),
            seconds=time.monotonic()-started, process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        diagnostics.save(destination, report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--exit-record', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope-name', required=True)
    args = parser.parse_args()
    observed = run(args.registration, args.exit_record, args.output, args.scope_name)
    # Audit execution success is deliberately independent of fit convergence.
    raise SystemExit(0 if observed['state'] == 'complete' else 1)
