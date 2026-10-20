"""Review terminal corrected-profile records without changing their decisions."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from analysis import overall_complete_profile_review as original

ROOT = Path(__file__).resolve().parents[1]
DEFINITION = 'docs/overall_stable_profile_review.md'
require, read, sha = original.require, original.read, original.sha
FALSE_FLAGS = ('full_recovery_complete', 'sbc_complete', 'publication_allowed')


def terminal_service(identifier):
    """Authoritative absence is terminal; log timestamps alone never are."""
    observed = original.service_state(identifier)
    state = observed['properties']
    absent = (state.get('LoadState') == 'not-found' and state.get('MainPID') == '0'
              and state.get('ActiveState') == 'inactive' and state.get('SubState') == 'dead')
    stopped = (state.get('LoadState') == 'loaded' and state.get('MainPID') == '0'
               and (state.get('ActiveState') in ('inactive', 'failed') or state.get('SubState') == 'exited'))
    require(absent or stopped, 'Review requires an authoritatively terminal or absent service')
    return observed


def object_key(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def generating_identity(reg, case, previous):
    return dict(**{key:previous[key] for key in ('arrays_key', 'parameters_key', 'truth_key', 'role')},
                seed=reg['synthetic'][case]['seed'])


def receipt_identity(receipt, result, reg, previous, source, registration_sha):
    """Match the complete recorded fit identity, not just its success flag."""
    case = result['case']
    catalog = receipt['catalog']
    require(receipt.get('schema_version') == reg['schema_version']
            and receipt.get('publication_allowed') is False and receipt['source'] == source
            and receipt['settings'] == reg['fit_settings'], 'Receipt source, schema or settings changed')
    spec = receipt['specification']
    require(spec == dict(source_key=reg['registered_model_inputs_key'], arrays_key=previous['arrays_key'],
            catalog=catalog, registration_sha256=registration_sha, study=case, synthetic=True,
            settings=reg['fit_settings']) and object_key(spec) == receipt['run_key'],
            'Receipt specification does not match the registered case')
    require(catalog.get('inference_method') == reg['inference_method']
            and catalog.get('control_rotation_sha256') == previous['control_rotation_sha256']
            and catalog.get('preflight_sha256') == previous['_file_sha256'],
            'Receipt does not bind the registered method and preflight')
    require(result.get('receipt') == receipt, 'Saved and embedded receipt differ')
    require(result.get('model_inputs_key') == reg['registered_model_inputs_key']
            and result.get('registration_sha256') == registration_sha,
            'Child result belongs to another source or registration')
    require(result.get('original_inputs') == generating_identity(reg, case, previous),
            'Original generating inputs changed')


def retained_attempts(receipt, reg, previous, cache):
    """Extend the old ladder review with the complete new retained inventories."""
    rows = original.attempt_review(receipt, reg, cache)
    for index, (row, attempt) in enumerate(zip(rows, receipt['attempts']), 1):
        try:
            expected = {name: dict(shape=[attempt['chains'], attempt['samples'], *site['shape']],
                                  dtype=site['dtype']) for name, site in previous['sites'].items()}
            require(bool(expected) and attempt.get('posterior_sites') == expected,
                    'Every attempt must retain the full preflight field/axis/dtype inventory')
            name = f'sampler_{index}.npz'
            archive = cache / name
            require(attempt.get('sampler_file') == name and archive.resolve().parent == cache.resolve(),
                    'Sampler-extra archive identity changed')
            require(sha(archive) == attempt['sampler_sha256'], 'Sampler-extra archive changed')
            diagnostic = attempt['diagnostics']
            require(diagnostic.get('inference_method') == reg['inference_method']
                    and diagnostic.get('control_rotation_sha256') == previous['control_rotation_sha256'],
                    'Attempt diagnostic method or control map differs')
            comparisons = diagnostic['comparison_reports']
            require(comparisons.get('all_reported_contrasts_passed') is all(
                r['diagnostic']['converged'] is True for r in comparisons['current_driver_contrasts']),
                'Driver-contrast aggregate disagrees with its complete reports')
            row.update(sampler_file=name, sampler_sha256=attempt['sampler_sha256'],
                recorded_driver_contrasts_passed=comparisons['all_reported_contrasts_passed'],
                sampler_extras_recounted=False, posterior_inventory_metadata_verified=True)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            row['issues'].append(f'{type(exc).__name__}: {exc}')
    return rows


def completion_agrees(completion, result):
    require(type(completion.get('exit_code')) is int, 'Missing child exit status')
    passed = (completion['exit_code'] == 0 and result is not None
              and result.get('state') == 'converged'
              and result.get('posterior_inventory_verified') is True)
    require(completion.get('state') == ('converged' if passed else 'failed'),
            'Child completion decision differs from its exit and inventory result')
    return passed


def collect(registration):
    # Check the authoritative process before loading even the registration's
    # numerical dependencies or streaming any potentially large cache archive.
    registration = Path(registration).resolve()
    initial = read(registration)
    identifier = initial['id']
    require(isinstance(identifier, str) and identifier and Path(identifier).name == identifier,
            'Invalid registered profile identifier')
    service = terminal_service(identifier)
    from analysis import overall_stable_profile as profile
    reg = profile.checked(registration)
    p = read(ROOT / profile.PREFLIGHT_PROTOCOL)
    source = p['model_snapshot']
    previous = profile.load_preflight(p)
    output = ROOT / 'outputs/championship/development' / identifier
    cache_root = ROOT / 'outputs/championship/cache' / identifier
    fingerprints, absent = {}, set()

    def bind(path, *, tracked=True, expected=None):
        path = Path(path)
        require(path.resolve().is_relative_to(ROOT), 'Review input escaped the repository')
        if tracked:
            original.committed(path)
        digest = sha(path)
        require(expected is None or digest == expected, f'Bound record changed: {path.name}')
        key = path.relative_to(ROOT).as_posix()
        require(key not in fingerprints or fingerprints[key] == digest, 'Input changed during collection')
        fingerprints[key] = digest
        return digest

    registration_sha = bind(registration)
    for name in (__file__, original.__file__, DEFINITION, 'tests/test_overall_stable_profile_review.py'):
        bind(ROOT / name)
    raw_files = {path.relative_to(output).as_posix() for path in output.rglob('*') if path.is_file()}
    for name in sorted(raw_files):
        bind(output / name)
    summary = read(output / 'summary.json')
    require(summary.get('all_processed') is True and summary.get('registration_sha256') == registration_sha
            and summary.get('registration') == registration.relative_to(ROOT).as_posix()
            and all(summary.get(k) is False for k in FALSE_FLAGS), 'Whole-run summary is incomplete or changed')
    require([row['case'] for row in summary['studies']] == reg['studies'], 'Required case inventory differs')
    exit_path = output.with_name(identifier + '-launch') / 'exit.status'
    bind(exit_path)
    require(exit_path.read_text().strip() == ('0' if summary['all_converged'] else '1'),
            'Final launcher exit status differs from the whole-run result')
    studies, issues, cache_inventories = [], [], {}
    for completion in summary['studies']:
        case = completion['case']
        row = dict(case=case, completion=completion, issues=[], attempts=[])
        studies.append(row)
        cache = cache_root / case
        cached_files = {path.relative_to(cache).as_posix() for path in cache.rglob('*') if path.is_file()}
        cache_inventories[case] = cached_files
        try:
            for name in sorted(cached_files):
                bind(cache / name, tracked=False)
            require(read(output / f'{case}.completion.json') == completion, 'Child completion record changed')
            for suffix, field in (('output.txt', 'log_sha256'), ('resources.txt', 'resources_sha256')):
                path = output / f'{case}.{suffix}'
                if completion[field] is None:
                    require(not path.exists(), 'Unbound child resource record appeared')
                    absent.add(path)
                else:
                    bind(path, expected=completion[field])
            result_path = output / case / 'result.json'
            require(result_path.is_file() == completion['result_present'], 'Child result presence changed')
            result = None
            if result_path.exists():
                bind(result_path, expected=completion['result_sha256'])
                result = read(result_path)
                require(result.get('case') == case and all(result.get(k) is False for k in FALSE_FLAGS),
                        'Child identity or scientific-evidence boundary changed')
                require(result.get('registration_sha256') == registration_sha
                        and result.get('model_inputs_key') == reg['registered_model_inputs_key'],
                        'Child result belongs to another registration or source')
                row['result'] = result
            else:
                require(completion['result_sha256'] is None, 'Missing result cannot have a recorded hash')
                absent.add(result_path)
            passed = completion_agrees(completion, result)
            receipt_path = cache / 'receipt.json'
            if not receipt_path.exists():
                require(not passed and (result is None or 'receipt' not in result), 'Missing fit receipt')
                absent.add(receipt_path)
                row['receipt_present'] = False
                row['unreferenced_cache_files'] = sorted(cached_files)
                continue
            bind(receipt_path, tracked=False)
            receipt = read(receipt_path)
            row.update(receipt_present=True, receipt=receipt)
            expected = dict(previous[case], _file_sha256=sha(ROOT/profile.PREFLIGHT_OUTPUT/case/'result.json'))
            if result is not None:
                receipt_identity(receipt, result, reg, expected, source, registration_sha)
            else:
                # A killed child still has to preserve an identifiable receipt;
                # a missing final result is never manufactured as convergence.
                partial = dict(case=case, receipt=receipt, model_inputs_key=reg['registered_model_inputs_key'],
                    registration_sha256=registration_sha, original_inputs=generating_identity(reg, case, expected))
                receipt_identity(receipt, partial, reg, expected, source, registration_sha)
            truth = read(output / case / 'truth.json')
            require({k:truth[k] for k in ('arrays_key', 'parameters_key', 'truth_key', 'seed', 'role')}
                    == generating_identity(reg, case, expected), 'Saved generating truth identity changed')
            row['attempts'] = retained_attempts(receipt, reg, expected, cache)
            referenced = {'receipt.json'}
            for index, attempt in enumerate(row['attempts'], 1):
                referenced.update((f'attempt_{index}.json', f'sampler_{index}.npz'))
                if 'archive' in attempt:
                    referenced.add(Path(attempt['archive']).name)
                row['issues'].extend(f'attempt {index}: {issue}' for issue in attempt['issues'])
            row['unreferenced_cache_files'] = sorted(cached_files - referenced)
            require(not passed or (receipt['state'] == 'converged' and not row['unreferenced_cache_files']),
                    'Successful child lacks a complete converged cache')
        except (ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as exc:
            row['issues'].append(f'{type(exc).__name__}: {exc}')
    recorded_convergence = all(row['state'] == 'converged' for row in summary['studies'])
    if summary.get('all_converged') is not recorded_convergence:
        issues.append('Whole-run convergence flag differs from required case decisions')
    profile.checked(registration)
    for name, expected in fingerprints.items():
        require(sha(ROOT / name) == expected, f'Input changed during review: {name}')
    require(not any(path.exists() for path in absent), 'Missing terminal record appeared during review')
    require(raw_files == {path.relative_to(output).as_posix() for path in output.rglob('*') if path.is_file()},
            'Raw output inventory changed during review')
    for case, names in cache_inventories.items():
        cache = cache_root / case
        require(names == {path.relative_to(cache).as_posix() for path in cache.rglob('*') if path.is_file()},
                'Cached inventory changed during review')
    require(terminal_service(identifier) == service, 'Terminal service changed during review')
    return dict(purpose='corrected_profile_evidence_review', recorded_at=datetime.now(timezone.utc).isoformat(),
        registration=registration.relative_to(ROOT).as_posix(), input_sha256=fingerprints,
        terminal_service=service, studies=studies, issues=issues, all_required_cases_accounted=True,
        records_consistent=not issues and not any(row['issues'] for row in studies),
        recorded_all_converged=recorded_convergence, archive_arrays_verified=False,
        diagnostics_recomputed=False, sampler_extras_recounted=False,
        scientific_acceptance_established=False, full_goal_complete=False,
        **{key:False for key in FALSE_FLAGS})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the existing review')
    report = collect(args.registration)
    with args.output.open('x') as handle:
        handle.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    raise SystemExit(0 if report['records_consistent'] else 1)
