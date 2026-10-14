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


def service_state(identifier):
    unit = f'f1-{identifier}.service'
    fields = ('LoadState', 'ActiveState', 'SubState', 'MainPID', 'Result', 'MemoryMax', 'Nice')
    command = ['systemctl', '--user', 'show', unit]
    for field in fields:
        command += ['-p', field]
    state = dict(line.split('=', 1) for line in subprocess.check_output(command, text=True).splitlines())
    return dict(unit=unit, properties=state)


def terminal_service(identifier):
    observed = service_state(identifier)
    state = observed['properties']
    if (state.get('LoadState') != 'loaded' or state.get('MainPID') != '0'
            or not (state.get('ActiveState') in ('inactive', 'failed') or state.get('SubState') == 'exited')):
        raise ValueError('Review only after the original service is authoritatively terminal')
    return observed


def require(condition, message):
    if not condition:
        raise ValueError(message)


def interrupted_service(identifier, record):
    observed = service_state(identifier)
    old = record['terminal_unit_observation']
    expected_command = ['systemctl', '--user', 'show', observed['unit']]
    require(old['exit_code'] == 0 and old['command'][:4] == expected_command,
            'Interruption must bind the original service observation')
    prior = dict(line.split('=', 1) for line in old['stdout'].splitlines())
    for state in (prior, observed['properties']):
        require(state.get('LoadState') == 'not-found' and state.get('MainPID') == '0'
                and state.get('ActiveState') == 'inactive' and state.get('SubState') == 'dead',
                'The original service must still be absent; inspect any new execution separately')
    require(Path('/proc/sys/kernel/random/boot_id').read_text().strip() == record['boot_id'],
            'The environment changed again; capture a fresh interruption observation')
    return observed


def interrupted_inventory(path, registration, reg, output):
    """Read a committed interruption without fabricating child exit records."""
    path = Path(path).resolve()
    committed(path)
    record = read(path)
    require(record['state'] == 'interrupted_without_final_record'
            and record['registration'] == str(registration.relative_to(ROOT))
            and record['registration_sha256'] == sha(registration)
            and record['publication_allowed'] is False and record['recovery_started'] is False,
            'Need the original committed interruption observation')
    service = interrupted_service(reg['id'], record)
    require(not (output / 'summary.json').exists(), 'Do not substitute an interruption for a completed summary')
    fingerprints = {str(path.relative_to(ROOT)): sha(path)}
    raw = record['raw_output_inventory']
    require({str(p.relative_to(ROOT)) for p in output.rglob('*') if p.is_file()} == set(raw),
            'Original interrupted output inventory changed')
    for name, expected in raw.items():
        source = ROOT / name
        require(source.resolve().is_relative_to(output.resolve()), 'Raw record escaped its profile')
        committed(source)
        require(sha(source) == expected['sha256'] and source.stat().st_size == expected['bytes'],
                f'Original interrupted output changed: {name}')
        fingerprints[name] = expected['sha256']
    cache = ROOT / 'outputs/championship/cache' / reg['id']
    metadata, archives = record['copied_cache_metadata'], record['cached_archive_inventory']
    require({str(p.relative_to(ROOT)) for p in cache.rglob('*') if p.is_file()}
            == set(metadata) | set(archives), 'Original interrupted cache inventory changed')
    for name, expected in metadata.items():
        source, saved = ROOT / name, ROOT / expected['snapshot']
        require(source.resolve().is_relative_to(cache.resolve())
                and saved.resolve().is_relative_to(path.parent / 'cache-metadata'),
                'Cached metadata escaped its original snapshot')
        committed(saved)
        require(sha(source) == sha(saved) == expected['sha256'], f'Original cached metadata changed: {name}')
        fingerprints[name] = fingerprints[expected['snapshot']] = expected['sha256']
    for name, expected in archives.items():
        source = ROOT / name
        require(source.resolve().is_relative_to(cache.resolve()), 'Archive escaped its original profile')
        require(sha(source) == expected['sha256'] and source.stat().st_size == expected['bytes'],
                f'Original retained archive changed: {name}')
        fingerprints[name] = expected['sha256']
    require([r['case'] for r in record['cases']] == reg['studies'], 'Interrupted case inventory differs')
    rows, absent = [], [str((output / 'summary.json').relative_to(ROOT)),
                       f"outputs/logs/{reg['id']}/exit.status"]
    for case_record in record['cases']:
        case = case_record['case']
        completion_path, result_path = output / f'{case}.completion.json', output / case / 'result.json'
        receipt_path = cache / case / 'receipt.json'
        for key, source in (('completion_present', completion_path), ('result_present', result_path),
                            ('receipt_present', receipt_path)):
            require(source.is_file() == case_record[key], 'Interrupted file presence changed')
        completion = read(completion_path) if case_record['completion_present'] else None
        require(completion == case_record.get('original_completion'), 'Original completion differs from interruption')
        for source in (completion_path, result_path):
            if not source.exists():
                absent.append(str(source.relative_to(ROOT)))
        if receipt_path.exists():
            receipt = read(receipt_path)
            require(receipt['state'] == case_record['receipt_state']
                    and len(receipt['attempts']) == case_record['recorded_attempt_count'],
                    'Interrupted receipt inventory differs')
        hashes = {suffix: raw.get(str((output / f'{case}.{suffix}').relative_to(ROOT)), {}).get('sha256')
                  for suffix in ('output.txt', 'resources.txt')}
        rows.append(dict(case=case, completion=completion, file_hashes=hashes,
            result_present=case_record['result_present'],
            result_sha256=raw.get(str(result_path.relative_to(ROOT)), {}).get('sha256')))
    require(set(absent) == set(record['missing_terminal_records'])
            and all(not (ROOT / name).exists() for name in absent), 'Missing terminal record inventory changed')
    event = record['last_recorded_sampling_start']
    if event is not None:
        require(event['study'] in reg['studies'] and event['state'] == 'sampling'
                and type(event['attempt']) is int and 1 <= event['attempt'] <= len(reg['retry']),
                'Last sampling event differs from the registered inventory')
        events = []
        for line in (output / f"{event['study']}.output.txt").read_text().splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get('state') == 'sampling':
                events.append(item)
        require(bool(events) and events[-1] == event, 'Sampling start differs from the original log')
        warm, draws, accept = reg['retry'][event['attempt'] - 1]
        base = reg['fit_settings']
        expected = dict(base, warmup=warm * base['warmup'], samples=draws * base['samples'],
            seed=base['seed'] + event['attempt'] - 1,
            target_accept=base['target_accept'] if accept is None else accept)
        require(event['settings'] == expected, 'Interrupted sampling settings changed')
    return rows, fingerprints, service, record, absent


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


def collect(registration, *, interruption=None):
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
    output = ROOT / 'outputs/championship/development' / identifier
    require(bool(reg['studies']) and len(set(reg['studies'])) == len(reg['studies']),
            'The required study inventory differs')
    fingerprints = {str(Path(__file__).resolve().relative_to(ROOT)): sha(__file__),
                    str(registration.relative_to(ROOT)): sha(registration)}
    interruption_record, absent = None, []
    if interruption is not None:
        inventory, original_files, service, interruption_record, absent = interrupted_inventory(
            interruption, registration, reg, output)
        fingerprints.update(original_files)
        summary = None
    else:
        service = terminal_service(identifier)
        summary_path = output / 'summary.json'
        committed(summary_path)
        summary = read(summary_path)
        require(summary['all_processed'] is True and summary['registration'] == str(registration.relative_to(ROOT))
                and summary['registration_sha256'] == sha(registration)
                and summary['publication_allowed'] is False, 'The whole original profile is not complete')
        require([r['case'] for r in summary['studies']] == reg['studies'], 'The required study inventory differs')
        fingerprints[str(summary_path.relative_to(ROOT))] = sha(summary_path)
        inventory = [dict(case=c['case'], completion=c,
            file_hashes={'output.txt': c['log_sha256'], 'resources.txt': c['resources_sha256']},
            result_present=c['result_present'], result_sha256=c['result_sha256']) for c in summary['studies']]
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

    for item in inventory:
        case, completion = item['case'], item['completion']
        row = dict(case=case, completion=completion, issues=[], attempts=[],
            completion_record_present=completion is not None)
        studies.append(row)
        cache = ROOT / 'outputs/championship/cache' / identifier / case
        result_path = output / case / 'result.json'
        try:
            path = output / f'{case}.completion.json'
            if completion is not None:
                committed(path)
                require(read(path) == completion, 'Completion record differs from original inventory')
                fingerprints[str(path.relative_to(ROOT))] = sha(path)
            else:
                require(not path.exists(), 'A missing completion record appeared')
            for suffix, expected in item['file_hashes'].items():
                path = output / f'{case}.{suffix}'
                if expected is None:
                    require(not path.exists(), 'Unbound resource record appeared')
                    continue
                committed(path)
                require(sha(path) == expected, f'Original {suffix} changed')
                fingerprints[str(path.relative_to(ROOT))] = sha(path)
            require(result_path.is_file() == item['result_present'], 'Result presence differs')
            result = None
            if result_path.is_file():
                committed(result_path)
                require(sha(result_path) == item['result_sha256'], 'Original result changed')
                fingerprints[str(result_path.relative_to(ROOT))] = sha(result_path)
                result = read(result_path)
                row['result'] = result
                require(result['case'] == case and result['registration_sha256'] == sha(registration),
                        'Result identity differs')
            passed = (completion is not None and completion['exit_code'] == 0
                      and result is not None and result['state'] == 'converged')
            if completion is not None:
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
    recorded_convergence = all(s['completion'] is not None and s['completion']['state'] == 'converged' for s in inventory)
    if summary is not None and summary['all_converged'] != recorded_convergence:
        issues.append('Whole-profile convergence decision differs from its recorded cases')
    for name, expected in fingerprints.items():
        if sha(ROOT / name) != expected:
            issues.append(f'Original source or evidence changed during review: {name}')
    if any((ROOT / name).exists() for name in absent):
        issues.append('A missing terminal record appeared during review')
    return dict(purpose='complete_design_profile_evidence_review',
        recorded_at=datetime.now(timezone.utc).isoformat(),
        reviewer_sha256=sha(__file__), terminal_service=service, input_sha256=fingerprints,
        registration=str(registration.relative_to(ROOT)), registered_resources=reg['resources'],
        resource_changes=changes, runtime_spans_changed_resources=bool(changes),
        studies=studies, issues=issues, all_required_cases_accounted=True,
        recorded_all_converged=recorded_convergence,
        original_run_complete=summary is not None,
        interruption_record=str(Path(interruption).resolve().relative_to(ROOT)) if interruption is not None else None,
        last_recorded_sampling_start=interruption_record['last_recorded_sampling_start'] if interruption_record else None,
        absent_terminal_records=absent,
        records_consistent=not issues and not any(row['issues'] for row in studies),
        archive_arrays_verified=False, diagnostics_recomputed=False, full_recovery_complete=False,
        sbc_complete=False, publication_allowed=False, full_goal_complete=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--interruption', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve the existing review')
    result = collect(args.registration, interruption=args.interruption)
    with args.output.open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    raise SystemExit(0 if result['records_consistent'] else 1)
