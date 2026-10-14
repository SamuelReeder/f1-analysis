"""Reproduce saved complete-profile diagnostics after the original service stops.

Every recorded case and attempt remains in the report, including failed fits.
This never samples, revises an acceptance decision or authorizes publication.
Sampler extra fields were not archived; their recorded values are not verified.
"""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis import overall_complete_profile_review as review

REPORT_KEYS = ('converged', 'failures', 'parameters', 'constants', 'empty_sites',
               'n_draws', 'n_chains', 'divergences')
SOURCES = ('analysis/overall_complete_profile_diagnostics.py',
           'analysis/overall_complete_profile_review.py',
           'analysis/overall_posterior_archive.py',
           'analysis/overall_bounded_diagnostics.py',
           'analysis/overall_weekend_profile.py',
           'analysis/overall_complete_profile.py',
           'tests/test_overall_complete_profile_diagnostics.py',
           'docs/overall_complete_profile_diagnostics.md')
LOADED = {name: review.sha(ROOT / name) for name in SOURCES}


def now():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    import json
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def actual_scope(scope):
    fields = ('Slice', 'MemoryMax', 'CPUQuotaPerSecUSec', 'ControlGroup', 'ActiveState')
    command = ['systemctl', '--user', 'show', scope]
    for field in fields:
        command += ['-p', field]
    actual = dict(line.split('=', 1) for line in subprocess.check_output(command, text=True).splitlines())
    membership = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                      if line.startswith('0::'))
    group = actual.get('ControlGroup', '')
    review.require(actual.get('Slice') == 'f1.slice'
        and actual.get('MemoryMax') == str(3 * 1024**3)
        and actual.get('CPUQuotaPerSecUSec') == '2s'
        and actual.get('ActiveState') == 'active' and bool(group)
        and (membership == group or membership.startswith(group + '/'))
        and os.getpriority(os.PRIO_PROCESS, 0) == 19,
        'Run this process inside the separate capped diagnostic scope')
    return dict(actual, Nice=19)


def evidence(review_path):
    """Replay the terminal metadata checks, without trusting a success flag alone."""
    review_path = Path(review_path).resolve()
    review.committed(review_path)
    original = review.read(review_path)
    fingerprints = {relative(review_path): review.sha(review_path)}
    for name in SOURCES:
        review.committed(ROOT / name)
        review.require(review.sha(ROOT / name) == LOADED[name], 'Loaded verification source changed')
        fingerprints[name] = LOADED[name]
    interruption = original.get('interruption_record')
    current = (review.collect(ROOT / original['registration'], interruption=ROOT / interruption)
               if interruption else review.collect(ROOT / original['registration']))
    # Timestamps and live terminal-unit properties can change between reviews.
    # All evidence, attempts, failures, byte identities and resource changes must match.
    stable = lambda value: {k: v for k, v in value.items()
                           if k not in ('recorded_at', 'terminal_service')}
    review.require(stable(original) == stable(current), 'The committed review no longer matches original evidence')
    fingerprints.update(original['input_sha256'])
    registration = review.read(ROOT / original['registration'])
    absent = list(original.get('absent_terminal_records', []))
    for study in original['studies']:
        cache = ROOT / 'outputs/championship/cache' / registration['id'] / study['case']
        receipt = cache / 'receipt.json'
        if study.get('receipt_present'):
            fingerprints[relative(receipt)] = study['receipt_sha256']
        else:
            absent.append(relative(receipt))
        for attempt in study['attempts']:
            if 'attempt_file_sha256' in attempt:
                path = cache / f"attempt_{attempt['attempt']}.json"
                fingerprints[relative(path)] = attempt['attempt_file_sha256']
            if 'archive' in attempt:
                path = ROOT / attempt['archive']
                if attempt.get('archive_present'):
                    fingerprints[relative(path)] = attempt['archive_sha256']
                else:
                    absent.append(relative(path))
    return current, registration, fingerprints, absent


def changed_inputs(fingerprints, absent):
    issues = []
    for name, expected in fingerprints.items():
        path = ROOT / name
        if not path.is_file() or review.sha(path) != expected:
            issues.append(f'Input changed during verification: {name}')
    for name in absent:
        if (ROOT / name).exists():
            issues.append(f'Previously absent input appeared during verification: {name}')
    return issues


def differences(expected, observed, path='$'):
    """All unequal JSON paths, with no floating tolerance or early exit."""
    if type(expected) is not type(observed):
        return [path]
    if isinstance(expected, dict):
        result = [f'{path}.{key}' for key in sorted(set(expected) ^ set(observed))]
        for key in sorted(set(expected) & set(observed)):
            result.extend(differences(expected[key], observed[key], f'{path}.{key}'))
        return result
    if isinstance(expected, list):
        result = [path + '.length'] if len(expected) != len(observed) else []
        for index, (left, right) in enumerate(zip(expected, observed)):
            result.extend(differences(left, right, f'{path}[{index}]'))
        return result
    return [] if expected == observed else [path]


def reproduce(archive, sites, settings, catalog, diagnostic, temporary_parent):
    """Read complete finite arrays, retaining saved member order for exact reports."""
    from analysis.overall_bounded_diagnostics import check
    from analysis.overall_complete_profile import validate_posterior
    from analysis.overall_posterior_archive import _mapped_numeric_archive
    from analysis.overall_weekend_profile import contrast_reports

    review.require(bool(sites), 'A complete nonempty storage inventory is required')
    with zipfile.ZipFile(archive) as saved:
        names = saved.namelist()
        review.require(len(names) == len(sites) and set(names) == {name + '.npy' for name in sites},
                       'Archive must contain exactly the complete retained-site inventory')
    axes = (settings['chains'], settings['samples'])
    with _mapped_numeric_archive(archive, names=list(sites), axes=axes,
                                 temporary_parent=temporary_parent) as (post, _):
        validate_posterior(post, sites, settings)
        # Divergence flags were not saved. Reuse their recorded count explicitly;
        # this cannot independently reproduce sampler extras or their truth.
        coordinate = check(post, diagnostic['divergences'])
        comparisons = contrast_reports(post, catalog, diagnostic['divergences'])
    expected_coordinate = {key: diagnostic[key] for key in REPORT_KEYS}
    coordinate_differences = differences(expected_coordinate, coordinate)
    comparison_differences = differences(diagnostic['comparison_reports'], comparisons)
    return dict(archive_arrays_verified=True, coordinate_report=coordinate,
        comparison_reports=comparisons, coordinate_report_equal=not coordinate_differences,
        comparison_reports_equal=not comparison_differences,
        coordinate_difference_paths=coordinate_differences,
        comparison_difference_paths=comparison_differences,
        divergence_count_source='original recorded diagnostic; sampler extras unavailable',
        divergence_flags_recomputed=False,
        other_recorded_fields_not_recomputed=sorted(set(diagnostic) - set(REPORT_KEYS) - {'comparison_reports'}))


def verify_studies(original, registration, temporary_parent, checkpoint):
    """A failed attempt never hides later attempts or other required cases."""
    studies = []
    for study in original['studies']:
        case = study['case']
        row = dict(case=case, original_completion=study['completion'],
            original_receipt_state=study.get('receipt', {}).get('state'),
            issues=list(study['issues']), attempts=[])
        studies.append(row)
        receipt = study.get('receipt')
        attempts = receipt.get('attempts', []) if receipt else []
        if not attempts:
            row['issues'].append('No saved attempt reports are available for reproduction')
        indexed = {a['attempt']: a for a in study['attempts']}
        attempt_issues = {f"attempt {a['attempt']}: {issue}"
                          for a in study['attempts'] for issue in a['issues']}
        case_issues = [issue for issue in study['issues'] if issue not in attempt_issues]
        for index, recorded in enumerate(attempts, 1):
            before = time.monotonic()
            result = dict(attempt=index, state='running', issues=[],
                original_diagnostics=recorded.get('diagnostics'),
                archive_arrays_verified=False, coordinate_report_equal=False,
                comparison_reports_equal=False, publication_allowed=False)
            row['attempts'].append(result)
            checkpoint(studies)
            try:
                reviewed = indexed.get(index)
                review.require(reviewed is not None, 'Attempt has no complete metadata review')
                result.update(original_role=reviewed.get('archive_role'),
                    archive=reviewed.get('archive'), archive_sha256=reviewed.get('archive_sha256'))
                review.require(not case_issues and not reviewed['issues'],
                               'Original case or attempt metadata is inconsistent; retain its issues')
                review.require(reviewed['recorded'] == recorded, 'Attempt report differs from reviewed receipt')
                if not reviewed.get('archive_present'):
                    result.update(state='unavailable', issues=['No retained archive for this recorded attempt'])
                else:
                    archive = ROOT / reviewed['archive']
                    cache = ROOT / 'outputs/championship/cache' / registration['id'] / case
                    review.require(archive.resolve().parent == cache.resolve(), 'Archive escaped its original case')
                    review.require(review.sha(archive) == reviewed['archive_sha256'], 'Archive bytes changed')
                    storage_path = registration['storage_results'][case]
                    review.require(storage_path in original['input_sha256'], 'Storage inventory is not bound')
                    storage = review.read(ROOT / storage_path)
                    review.require(storage['case'] == case and storage['state'] == 'complete',
                                   'A complete original storage inventory is required')
                    result.update(reproduce(archive, storage['sites'], reviewed['expected_settings'],
                        receipt['catalog'], recorded['diagnostics'], temporary_parent))
                    review.require(review.sha(archive) == reviewed['archive_sha256'], 'Archive changed during reproduction')
                    result['state'] = ('verified' if result['coordinate_report_equal']
                        and result['comparison_reports_equal'] else 'different')
            except Exception as exc:
                result.update(state='failed', error=type(exc).__name__, message=str(exc))
                result['issues'].append(f'{type(exc).__name__}: {exc}')
            finally:
                result['seconds'] = time.monotonic() - before
                checkpoint(studies)
        checkpoint(studies)
    return studies


def all_reproduced(studies):
    # Missing receipts/attempts cannot make an empty all(...) look successful.
    return bool(studies) and all(not s['issues'] and bool(s['attempts'])
        and all(a['state'] == 'verified' for a in s['attempts']) for s in studies)


def run(review_path, output, scope):
    actual = actual_scope(scope)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    record = dict(purpose='complete_profile_saved_diagnostic_reproduction', state='running',
        started_at=now(), actual_scope=actual, studies=[], issues=[], all_processed=False,
        all_recorded_reports_reproduced=False, divergence_flags_recomputed=False,
        mcmc_run=False, full_recovery_complete=False, sbc_complete=False,
        publication_allowed=False, full_goal_complete=False)
    destination = output / 'audit.json'
    save(destination, record)
    def checkpoint(studies):
        record['studies'] = studies
        save(destination, record)
    try:
        original, registration, fingerprints, absent = evidence(review_path)
        record.update(review=relative(review_path), input_sha256=fingerprints,
            absent_inputs=absent, source_commit=subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            diagnostic_runtime=dict(python=sys.version,
                packages={name: version(name) for name in ('numpy', 'arviz-stats', 'jax', 'numpyro')}),
            recorded_all_converged=original['recorded_all_converged'],
            original_run_complete=original.get('original_run_complete', False),
            interruption_record=original.get('interruption_record'),
            last_recorded_sampling_start=original.get('last_recorded_sampling_start'),
            terminal_service=original['terminal_service'],
            original_records_consistent=original['records_consistent'],
            original_registered_resources=original['registered_resources'],
            original_resource_changes=original['resource_changes'],
            original_runtime_spans_changed_resources=original['runtime_spans_changed_resources'])
        record['issues'].extend(original['issues'])
        # This temporary directory is separate from every original cache; each
        # archive reader cleans its own maps on success and ordinary exceptions.
        cache = ROOT / 'outputs/championship/cache'
        record['studies'] = verify_studies(original, registration, cache, checkpoint)
        record['all_processed'] = True
        record['issues'].extend(changed_inputs(fingerprints, absent))
        record['all_recorded_reports_reproduced'] = (original['records_consistent']
            and not record['issues'] and all_reproduced(record['studies']))
        record['state'] = 'complete' if record['all_recorded_reports_reproduced'] else 'incomplete'
    except Exception as exc:
        record.update(state='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
    finally:
        record.update(finished_at=now(), seconds=time.monotonic() - started,
            process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(destination, record)
    return record


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scope-name', required=True)
    args = parser.parse_args()
    result = run(args.review, args.output, args.scope_name)
    raise SystemExit(0 if result['all_recorded_reports_reproduced'] else 1)
