"""Collect immutable matched-profile evidence after the whole service stops.

This reports the existing decisions. It neither fits nor changes acceptance.
Posterior archives remain local; their byte identities are included in the review.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text())


def committed(path):
    name = path.relative_to(ROOT).as_posix()
    if subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != path.read_bytes():
        raise ValueError(f'Uncommitted evidence or registration: {name}')


def collect(registration_path):
    registration_path = registration_path.resolve()
    committed(registration_path)
    reg = read(registration_path)
    if reg['status'] != 'frozen' or reg['user_approved'] is not True:
        raise ValueError('Only the approved frozen experiment can be reviewed')
    destination = ROOT / 'outputs/championship/development' / reg['id']
    summary_path = destination / 'summary.json'
    committed(summary_path)
    summary = read(summary_path)
    expected_summary = str(registration_path.relative_to(ROOT))
    if (summary['registration'] != expected_summary or not summary['all_processed']
            or [r['study'] for r in summary['studies']] != reg['studies']
            or summary['method'] != reg['method'] or summary['publication_allowed']):
        raise ValueError('Incomplete or different experiment summary')
    source = summary['source']
    source_key = hashlib.sha256(json.dumps(source['inputs'], sort_keys=True).encode()).hexdigest()
    if source_key != source['key'] or source_key != reg['registered_model_inputs_key']:
        raise ValueError('Model source identity differs')
    for name, fingerprint in source['inputs'].items():
        path = ROOT / name
        committed(path)
        if digest(path) != fingerprint:
            raise ValueError(f'Model input changed: {name}')
    for name, fingerprint in reg['additional_source_sha256'].items():
        path = ROOT / name
        committed(path)
        if digest(path) != fingerprint:
            raise ValueError(f'Registered experiment source changed: {name}')
    studies = []
    for expected in summary['studies']:
        study = expected['study']
        path = destination / f'{study}.json'
        committed(path)
        result = read(path)
        receipt = result['receipt']
        cache = ROOT / 'outputs/championship/cache' / reg['id'] / study
        if (result['study'] != study or result['state'] != expected['state']
                or read(cache / 'receipt.json') != receipt or receipt['source'] != source
                or receipt['publication_allowed'] or receipt['state'] not in ('converged', 'failed_convergence')):
            raise ValueError(f'Incomplete, changed or exceptional study: {study}')
        if (result['state'] == 'converged') != (receipt['state'] == 'converged'):
            raise ValueError('Study and receipt decisions differ')
        spec = receipt['specification']
        if (spec['registration_sha256'] != digest(registration_path)
                or spec['source_key'] != source_key or spec['study'] != study
                or spec['arrays_key'] != reg['original_arrays_sha256'][study]
                or spec['synthetic'] is not True or spec['settings'] != reg['fit_settings']
                or hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest() != receipt['run_key']):
            raise ValueError('Recorded fit specification differs')
        truth_path = destination / f'{study}.truth.json'
        committed(truth_path)
        original = reg['original_truth_files'][study]
        original_path = ROOT / original['path']
        committed(original_path)
        if digest(original_path) != original['sha256'] or truth_path.read_bytes() != original_path.read_bytes():
            raise ValueError('Original generating truths differ')
        if read(truth_path)['arrays_key'] != spec['arrays_key']:
            raise ValueError('Original synthetic observations differ')
        attempts = []
        raw_attempts = receipt['attempts']
        if not raw_attempts or len(raw_attempts) > len(reg['retry']):
            raise ValueError('Invalid attempt inventory')
        for index, attempt in enumerate(raw_attempts):
            if read(cache / f'attempt_{index + 1}.json') != attempt:
                raise ValueError('Saved attempt and receipt differ')
            warm, draws, accept = reg['retry'][index]
            settings = dict(reg['fit_settings'], warmup=warm * reg['fit_settings']['warmup'],
                samples=draws * reg['fit_settings']['samples'], seed=reg['fit_settings']['seed'] + index,
                target_accept=accept if accept is not None else reg['fit_settings']['target_accept'])
            if {k: attempt[k] for k in settings} != settings:
                raise ValueError('Fixed retry settings changed')
            d = attempt['diagnostics']
            last = index == len(raw_attempts) - 1
            if d['converged'] and (not last or receipt['state'] != 'converged'):
                raise ValueError('An earlier successful attempt was bypassed')
            if last and d != receipt['diagnostics']:
                raise ValueError('Final diagnostics differ')
            if d['n_draws'] != settings['chains'] * settings['samples']:
                raise ValueError('Draw inventory differs')
            comparisons = d['comparison_reports']
            field = receipt['catalog']['field']
            expected_pairs = {(site, field[i], field[j]) for site in ('skill', 'overall_race_skill')
                              for i in range(len(field)) for j in range(i)}
            reported = comparisons['current_driver_contrasts']
            actual_pairs = [(r['site'], r['driver_i'], r['driver_j']) for r in reported]
            if (len(actual_pairs) != len(expected_pairs) or set(actual_pairs) != expected_pairs
                    or comparisons['changes_attempt_convergence_decision'] or comparisons['publication_allowed']):
                raise ValueError('Incomplete or altered driver comparison inventory')
            archive = cache / (receipt['posterior_file'] if d['converged'] else f'failed_{index + 1}.npz')
            fingerprint = digest(archive)
            if d['converged'] and fingerprint != receipt['posterior_sha256']:
                raise ValueError('Converged archive bytes changed')
            parameters = d['parameters']
            def extreme(key, operation):
                values = [p[key] for p in parameters]
                return operation(values) if values and all(v is not None for v in values) else None
            attempts.append(dict(attempt=index + 1, settings=settings, seconds=attempt['seconds'],
                converged=d['converged'], draws=d['n_draws'], divergences=d['divergences'],
                rhat_max=extreme('rhat_max', max), bulk_ess_min=extreme('bulk_ess_min', min),
                tail_ess_min=extreme('tail_ess_min', min), mcse_sd_max=extreme('mcse_sd_max', max),
                failed_parameters=[p for p in parameters if not p['passed']],
                failures=d['failures'], tree_depth_limit_fraction=d['tree_depth_limit_fraction'],
                mean_accept_prob=d['mean_accept_prob'], control_rotation_sha256=d['control_rotation_sha256'],
                current_driver_contrasts_reported=len(reported),
                failed_driver_contrasts=[r for r in reported if not r['diagnostic']['converged']],
                archive=str(archive.relative_to(ROOT)), archive_bytes=archive.stat().st_size,
                archive_sha256=fingerprint, archive_role='converged_profile_only' if d['converged'] else 'failed_diagnostic_only',
                runtime=d['runtime']))
        if receipt['state'] == 'failed_convergence' and len(attempts) != len(reg['retry']):
            raise ValueError('Failed study did not exhaust its fixed ladder')
        if (receipt['state'] == 'converged') != attempts[-1]['converged']:
            raise ValueError('Final decision differs')
        interval = (datetime.fromisoformat(receipt['finished_at']) - datetime.fromisoformat(receipt['started_at'])).total_seconds()
        studies.append(dict(study=study, state=result['state'], receipt_state=receipt['state'],
            study_file_sha256=digest(path), source_commit=receipt['source_commit'], run_key=receipt['run_key'],
            arrays_and_truths_match_original=True, truth_sha256=digest(truth_path),
            receipt_monotonic_seconds=receipt['seconds'], process_cumulative_max_rss_kib=receipt['max_rss_kib'],
            started_at=receipt['started_at'], finished_at=receipt['finished_at'], timestamp_interval_seconds=interval,
            attempts=attempts))
    if summary['all_converged'] != all(s['state'] == 'converged' for s in studies):
        raise ValueError('Whole-experiment decision differs')
    return dict(purpose='complete_matched_profile_evidence_review', recorded_at=datetime.now(timezone.utc).isoformat(),
        evidence_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        reviewer_sha256=digest(Path(__file__)), registration=expected_summary,
        registration_sha256=digest(registration_path), summary_sha256=digest(summary_path),
        model_inputs_key=source_key, all_processed=True, all_converged=summary['all_converged'], studies=studies,
        total_attempts=sum(len(s['attempts']) for s in studies),
        total_receipt_monotonic_seconds=sum(s['receipt_monotonic_seconds'] for s in studies),
        full_recovery_complete=False, sbc_complete=False, real_driver_ranking_produced=False,
        publication_allowed=False, full_goal_complete=False, original_failures_changed=False)


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
    print(json.dumps({k: result[k] for k in ('all_processed', 'all_converged', 'total_attempts')}))
