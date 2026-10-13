"""Collect the complete profile after it stops, preserving every recorded failure.

This reads committed terminal outputs and streams archive fingerprints. It does
not sample, recompute diagnostics, validate archive arrays or grant publication.
Run it in a capped scope after committing the complete raw output inventory.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def committed(path):
    path = Path(path).resolve()
    name = path.relative_to(ROOT).as_posix()
    if subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != path.read_bytes():
        raise ValueError(f'Commit the complete original evidence first: {name}')


def terminal_service(identifier):
    unit = f'f1-{identifier}.service'
    fields = ('LoadState', 'ActiveState', 'SubState', 'MainPID', 'Result', 'MemoryMax', 'Nice')
    command = ['systemctl', '--user', 'show', unit]
    for field in fields:
        command += ['-p', field]
    state = dict(line.split('=', 1) for line in subprocess.check_output(command, text=True).splitlines())
    if (state.get('LoadState') != 'loaded' or state.get('MainPID') != '0'
            or not (state.get('ActiveState') in ('inactive', 'failed') or state.get('SubState') == 'exited')):
        raise ValueError('Review only after the original service is authoritatively terminal')
    return dict(unit=unit, properties=state)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def attempt_review(receipt, reg, cache):
    attempts = receipt['attempts']
    require(len(attempts) <= len(reg['retry']), 'Extra attempts exceed the fixed ladder')
    rows = []
    for index, attempt in enumerate(attempts):
        row = dict(attempt=index + 1, recorded=attempt, issues=[])
        rows.append(row)
        try:
            path = cache / f'attempt_{index + 1}.json'
            require(read(path) == attempt, 'Saved attempt differs from receipt')
            row['attempt_file_sha256'] = sha(path)
            warm, draws, acceptance = reg['retry'][index]
            base = reg['fit_settings']
            settings = dict(base, warmup=warm * base['warmup'], samples=draws * base['samples'],
                seed=base['seed'] + index,
                target_accept=base['target_accept'] if acceptance is None else acceptance)
            row['expected_settings'] = settings
            require({k: attempt[k] for k in settings} == settings, 'Fixed retry settings changed')
            diagnostic = attempt['diagnostics']
            require(diagnostic['n_draws'] == settings['chains'] * settings['samples']
                    and diagnostic['n_chains'] == settings['chains'], 'Draw inventory differs')
            if diagnostic['converged']:
                require(index == len(attempts) - 1, 'An earlier successful attempt was bypassed')
            if index == len(attempts) - 1:
                require(diagnostic == receipt['diagnostics'], 'Final diagnostics differ')
            field = receipt['catalog']['field']
            require(len(field) == len(set(field)) and len(field) >= 2, 'Invalid current field')
            expected = {(site, field[i], field[j]) for site in ('skill', 'overall_race_skill')
                        for i in range(len(field)) for j in range(i)}
            comparisons = diagnostic['comparison_reports']
            pairs = [(r['site'], r['driver_i'], r['driver_j']) for r in comparisons['current_driver_contrasts']]
            require(len(pairs) == len(expected) and set(pairs) == expected
                    and comparisons['changes_attempt_convergence_decision'] is False
                    and comparisons['publication_allowed'] is False, 'Incomplete or altered comparison inventory')
            name = receipt.get('posterior_file', 'posterior.npz') if diagnostic['converged'] else f'failed_{index + 1}.npz'
            require(isinstance(name, str) and Path(name).name == name and name.endswith('.npz'),
                    'Invalid saved archive path')
            archive = cache / name
            require(archive.resolve().parent == cache.resolve(), 'Archive escaped its run directory')
            row.update(archive=str(archive.relative_to(ROOT)), archive_present=archive.is_file(),
                archive_arrays_verified=False, diagnostics_recomputed=False,
                archive_role='converged_profile_only' if receipt['state'] == 'converged' and diagnostic['converged']
                else 'unaccepted_diagnostic_only')
            if archive.is_file():
                row.update(archive_sha256=sha(archive), archive_bytes=archive.stat().st_size)
                if receipt['state'] == 'converged' and diagnostic['converged']:
                    require(row['archive_sha256'] == receipt['posterior_sha256'], 'Converged archive bytes changed')
            else:
                require(receipt['state'] in ('running', 'failed_exception'), 'A required completed-attempt archive is missing')
        except (ValueError, KeyError, OSError, TypeError) as exc:
            row['issues'].append(f'{type(exc).__name__}: {exc}')
    require(receipt['state'] in ('converged', 'failed_convergence', 'failed_exception', 'running'),
            'Unknown receipt state')
    if receipt['state'] in ('converged', 'failed_convergence'):
        require(bool(attempts), 'A completed fit has no attempts')
        require((receipt['state'] == 'converged') == attempts[-1]['diagnostics']['converged'],
                'Final fit decision differs')
    if receipt['state'] == 'failed_convergence':
        require(len(attempts) == len(reg['retry']), 'Failed fit did not exhaust its fixed ladder')
    return rows


def collect(registration):
    registration = Path(registration).resolve()
    committed(Path(__file__))
    committed(registration)
    reg = read(registration)
    require(reg['status'] == 'frozen' and reg['profile'] == 'complete_design_inference'
            and reg['purpose'] == 'numerical_profile' and reg['publication_allowed'] is False,
            'Only the frozen complete computational profile can be reviewed')
    identifier = reg['id']
    require(isinstance(identifier, str) and identifier and Path(identifier).name == identifier,
            'Invalid profile identifier')
    service = terminal_service(identifier)
    output = ROOT / 'outputs/championship/development' / identifier
    summary_path = output / 'summary.json'
    committed(summary_path)
    summary = read(summary_path)
    require(summary['all_processed'] is True and summary['registration'] == str(registration.relative_to(ROOT))
            and summary['registration_sha256'] == sha(registration)
            and summary['publication_allowed'] is False, 'The whole original profile is not complete')
    require([r['case'] for r in summary['studies']] == reg['studies']
            and len(set(reg['studies'])) == len(reg['studies']), 'The required study inventory differs')
    fingerprints = {str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__),
                    str(registration.relative_to(ROOT)): sha(registration),
                    str(summary_path.relative_to(ROOT)): sha(summary_path)}
    issues, studies = [], []
    for group in ('source_sha256', 'evidence_sha256'):
        for name, expected in reg[group].items():
            path = ROOT / name
            committed(path)
            require(sha(path) == expected, f'Bound original source or input changed: {name}')
            fingerprints[name] = expected
    preflight = read(ROOT / reg['preflight_registration'])
    source = preflight['model_source']
    key = hashlib.sha256(json.dumps(source['inputs'], sort_keys=True).encode()).hexdigest()
    require(key == source['key'] == reg['registered_model_inputs_key'], 'Original model identity differs')
    for name, expected in source['inputs'].items():
        path = ROOT / name
        committed(path)
        require(sha(path) == expected, f'Original model input changed: {name}')
        fingerprints[name] = expected
    changes = []
    for path in sorted(output.with_name(identifier + '-resource-changes').glob('*.json')):
        committed(path)
        fingerprints[str(path.relative_to(ROOT))] = sha(path)
        changes.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), record=read(path)))

    for completion in summary['studies']:
        case = completion['case']
        row = dict(case=case, completion=completion, issues=[], attempts=[])
        studies.append(row)
        cache = ROOT / 'outputs/championship/cache' / identifier / case
        result_path = output / case / 'result.json'
        try:
            path = output / f'{case}.completion.json'
            committed(path)
            require(read(path) == completion, 'Completion record differs from the complete summary')
            fingerprints[str(path.relative_to(ROOT))] = sha(path)
            for suffix, field in (('output.txt', 'log_sha256'), ('resources.txt', 'resources_sha256')):
                path = output / f'{case}.{suffix}'
                if completion[field] is None:
                    require(not path.exists(), 'Unbound resource record appeared')
                    continue
                committed(path)
                require(sha(path) == completion[field], f'Original {suffix} changed')
                fingerprints[str(path.relative_to(ROOT))] = sha(path)
            require(result_path.is_file() == completion['result_present'], 'Result presence differs')
            result = None
            if result_path.is_file():
                committed(result_path)
                require(sha(result_path) == completion['result_sha256'], 'Original result changed')
                fingerprints[str(result_path.relative_to(ROOT))] = sha(result_path)
                result = read(result_path)
                row['result'] = result
                require(result['case'] == case and result['registration_sha256'] == sha(registration),
                        'Result identity differs')
            passed = completion['exit_code'] == 0 and result is not None and result['state'] == 'converged'
            require(completion['state'] == ('converged' if passed else 'failed'), 'Child completion decision differs')
            receipt_path = cache / 'receipt.json'
            if receipt_path.exists():
                receipt = read(receipt_path)
                row.update(receipt_present=True, receipt=receipt, receipt_sha256=sha(receipt_path))
                require(result is None or 'receipt' not in result or result['receipt'] == receipt,
                        'Result receipt differs from saved receipt')
                require(receipt['publication_allowed'] is False and receipt['source'] == source,
                        'Saved receipt source or publication permission differs')
                spec = receipt['specification']
                previous = read(ROOT / reg['preflight_results'][case])
                require(spec['registration_sha256'] == sha(registration) and spec['source_key'] == key
                        and spec['study'] == case and spec['synthetic'] is True
                        and spec['settings'] == reg['fit_settings'] and spec['catalog'] == receipt['catalog']
                        and spec['arrays_key'] == previous['arrays_key']
                        and hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest() == receipt['run_key'],
                        'Original fit specification differs')
                truth_path = output / case / 'truth.json'
                committed(truth_path)
                truth = read(truth_path)
                fingerprints[str(truth_path.relative_to(ROOT))] = sha(truth_path)
                require(all(truth[k] == previous[k] for k in ('arrays_key', 'parameters_key', 'truth_key'))
                        and truth['seed'] == reg['synthetic'][case]['seed'], 'Original generating identities differ')
                row['attempts'] = attempt_review(receipt, reg, cache)
                row['issues'].extend(f"attempt {a['attempt']}: {issue}" for a in row['attempts'] for issue in a['issues'])
                require(not passed or receipt['state'] == 'converged', 'Successful child lacks a converged receipt')
            else:
                row['receipt_present'] = False
                require(not passed, 'Successful child has no fit receipt')
        except (ValueError, KeyError, OSError, TypeError, subprocess.CalledProcessError) as exc:
            row['issues'].append(f'{type(exc).__name__}: {exc}')
    recorded_convergence = all(s['state'] == 'converged' for s in summary['studies'])
    if summary['all_converged'] != recorded_convergence:
        issues.append('Whole-profile convergence decision differs from its recorded cases')
    for name, expected in fingerprints.items():
        if sha(ROOT / name) != expected:
            issues.append(f'Original source or evidence changed during review: {name}')
    return dict(purpose='complete_design_profile_evidence_review',
        recorded_at=datetime.now(timezone.utc).isoformat(),
        reviewer_sha256=sha(__file__), terminal_service=service, input_sha256=fingerprints,
        registration=str(registration.relative_to(ROOT)), registered_resources=reg['resources'],
        resource_changes=changes, runtime_spans_changed_resources=bool(changes),
        studies=studies, issues=issues, all_required_cases_accounted=True,
        recorded_all_converged=summary['all_converged'],
        records_consistent=not issues and not any(row['issues'] for row in studies),
        archive_arrays_verified=False, diagnostics_recomputed=False, full_recovery_complete=False,
        sbc_complete=False, publication_allowed=False, full_goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the existing review')
    result = collect(args.registration)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    raise SystemExit(0 if result['records_consistent'] else 1)
