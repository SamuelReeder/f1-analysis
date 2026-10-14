"""Registered execution recovery, preserving the original numerical profile."""
import argparse
import copy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import overall_complete_profile_review as records
from analysis.overall_profile_continuation import continuation_plan, require

IDENTIFIER = 'overall-complete-design-continuation-v1'
DEFINITION = 'docs/overall_profile_continuation.md'
SOURCES = ('analysis/overall_profile_continuation.py', 'analysis/overall_profile_continuation_run.py',
           'tests/test_overall_profile_continuation.py', 'tests/test_overall_profile_continuation_run.py', DEFINITION)
LOADED = {name: records.sha(ROOT / name) for name in SOURCES}
RESOURCES = dict(MemoryMax=13 * 1024**3, Slice='f1.slice', Nice=19,
                 shared_CPUQuotaPerSecUSec='4s', shared_CPUWeight='10', Restart='no')


def now():
    return datetime.now(timezone.utc).isoformat()


def relative(path):
    return str(Path(path).resolve().relative_to(ROOT))


def save(path, record):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def bound_inputs(review_path, audit_path):
    from analysis.overall_complete_profile_diagnostics import evidence
    from analysis.overall_complete_profile import checked_registration
    records.committed(audit_path)
    audit = records.read(audit_path)
    metadata, parent, fingerprints, absent = evidence(review_path)
    require(audit['review'] == relative(review_path) and audit['input_sha256'] == fingerprints
            and audit['absent_inputs'] == absent, 'Audit must bind the same complete original evidence')
    # Also retain the original runtime-implementation checks, not merely package labels.
    require(checked_registration(ROOT / metadata['registration']) == parent,
            'Original numerical registration changed')
    plan = continuation_plan(parent, metadata, audit)
    fingerprints[relative(audit_path)] = records.sha(audit_path)
    for name in SOURCES:
        records.committed(ROOT / name)
        require(records.sha(ROOT / name) == LOADED[name], 'Loaded continuation source changed')
        fingerprints[name] = LOADED[name]
    return dict(original_registration=metadata['registration'], review=relative(review_path),
        diagnostic_audit=relative(audit_path), input_sha256=fingerprints, absent_inputs=absent,
        plan=plan, resources=RESOURCES, original_model_inputs_key=parent['registered_model_inputs_key']), parent, metadata


def freeze(review_path, audit_path, output, scope):
    from analysis.overall_complete_profile_diagnostics import actual_scope
    actual_scope(scope)
    require(not Path(output).exists(), 'Preserve the existing continuation registration')
    payload, _, _ = bound_inputs(review_path, audit_path)
    record = dict(payload, schema_version=1, id=IDENTIFIER, status='frozen',
        purpose='interrupted_profile_execution_continuation', definition=DEFINITION,
        registered_at=now(), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        publication_allowed=False)
    with Path(output).open('x') as handle:
        handle.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    return record


def checked_registration(path):
    path = Path(path).resolve()
    records.committed(path)
    frozen = records.read(path)
    require(frozen['schema_version'] == 1 and frozen['id'] == IDENTIFIER and frozen['status'] == 'frozen'
            and frozen['purpose'] == 'interrupted_profile_execution_continuation'
            and frozen['definition'] == DEFINITION and frozen['publication_allowed'] is False,
            'Use the separately frozen execution continuation')
    current, parent, metadata = bound_inputs(ROOT / frozen['review'], ROOT / frozen['diagnostic_audit'])
    require({k: frozen[k] for k in current} == current, 'Frozen continuation inputs or remaining attempts changed')
    return frozen, parent, metadata


def properties(unit, fields):
    command = ['systemctl', '--user', 'show', unit]
    for field in fields:
        command += ['-p', field]
    return dict(line.split('=', 1) for line in subprocess.check_output(command, text=True).splitlines())


def actual_service(unit):
    require(unit == f'f1-{IDENTIFIER}.service', 'Use the registered continuation service')
    require(os.environ.get('XLA_FLAGS') == '--xla_force_host_platform_device_count=4'
            and os.environ.get('JAX_PLATFORMS') in (None, 'cpu')
            and os.environ.get('JAX_ENABLE_X64', 'false').lower() in ('false', '0'),
            'Keep the original CPU chain-device and precision configuration')
    actual = properties(unit, ('Slice', 'MemoryMax', 'Nice', 'Restart', 'OOMPolicy',
                               'ControlGroup', 'ActiveState', 'SubState'))
    require(actual['Slice'] == 'f1.slice' and actual['MemoryMax'] == str(RESOURCES['MemoryMax'])
            and actual['Nice'] == '19' and actual['Restart'] == 'no' and actual['OOMPolicy'] == 'continue'
            and actual['ActiveState'] == 'active' and actual['SubState'] == 'running',
            'The continuation needs its original memory budget and fixed operating limits')
    group = actual['ControlGroup']
    membership = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    require(bool(group) and membership == group and os.getpriority(os.PRIO_PROCESS, 0) == 19,
            'Worker must belong to the capped continuation service')
    shared = properties('f1.slice', ('CPUQuotaPerSecUSec', 'CPUWeight'))
    require(shared == dict(CPUQuotaPerSecUSec='4s', CPUWeight='10'), 'Keep the user-requested lower shared CPU limit')
    cgroup = Path('/sys/fs/cgroup') / group.lstrip('/')
    require((cgroup / 'memory.oom.group').read_text().strip() == '0', 'Preserve the recorded OOM-group setting')
    for sibling in cgroup.parent.iterdir():
        if sibling.is_dir() and sibling != cgroup and (sibling / 'cgroup.events').exists():
            state = dict(line.split() for line in (sibling / 'cgroup.events').read_text().splitlines())
            require(state.get('populated') == '0', 'No other f1 fit or heavy scope may overlap this service')
    return dict(service=actual, shared_slice=shared, kernel_memory_oom_group='0')


def recheck(frozen, registration_path):
    from analysis.overall_complete_profile import checked_registration as original_registration
    require(records.sha(registration_path) == frozen['continuation_registration_sha256'], 'Continuation registration changed')
    for name, expected in frozen['input_sha256'].items():
        require(records.sha(ROOT / name) == expected, f'Bound input changed: {name}')
    require(all(not (ROOT / name).exists() for name in frozen['absent_inputs']), 'A previously absent original record appeared')
    original_registration(ROOT / frozen['original_registration'])


def preserve_attempts(study, cache):
    """Create new names for immutable old bytes; never rewrite the source files."""
    for row in study['attempts']:
        source = ROOT / row['archive']
        require(records.sha(source) == row['archive_sha256'], 'An original saved archive changed')
        target = cache / f"failed_{row['attempt']}.npz"
        os.link(source, target)
        original = source.parent / f"attempt_{row['attempt']}.json"
        require(records.read(original) == row['recorded'], 'An original attempt record changed')
        with (cache / original.name).open('xb') as handle:
            handle.write(original.read_bytes())


def run_remaining(plan, original_receipt, cache, sampler, validate, execution):
    """The sampler/checker are the frozen original; retry only its remaining settings."""
    import numpy as np
    receipt = copy.deepcopy(original_receipt)
    require(receipt['state'] == 'running' and len(receipt['attempts']) == plan['preserved_attempts'],
            'Original interrupted receipt differs from the verified plan')
    require(not (cache / 'receipt.json').exists(), 'Never overwrite an earlier continuation receipt')
    require(bool(plan['remaining_attempts']) and all(row['attempt'] == len(receipt['attempts']) + i + 1
            for i, row in enumerate(plan['remaining_attempts'])), 'Only a contiguous remaining ladder may execute')
    receipt.update(state='running', execution_continuation=execution, pid=os.getpid())
    # Historical attempt seconds stay unchanged. This clock measures only new work.
    started = time.monotonic()
    post = None
    save(cache / 'receipt.json', receipt)
    try:
        for planned in plan['remaining_attempts']:
            number, settings = planned['attempt'], planned['settings']
            require(number == len(receipt['attempts']) + 1, 'Do not skip or repeat an original attempt')
            require(not (cache / f'attempt_{number}.json').exists()
                    and not (cache / f'failed_{number}.npz').exists()
                    and not (cache / 'posterior.npz').exists(), 'Never overwrite an attempt or accepted archive')
            validate()
            print(json.dumps(dict(study=plan['case'], attempt=number, state='sampling', settings=settings,
                                  execution_segment='continuation')), flush=True)
            begun = time.monotonic()
            post, diagnostic = sampler(settings)
            validate()
            require(type(diagnostic['converged']) is bool, 'The original checker must supply the attempt decision')
            attempt = dict(settings, diagnostics=diagnostic, seconds=time.monotonic() - begun)
            receipt['attempts'].append(attempt)
            receipt['diagnostics'] = diagnostic
            save(cache / f'attempt_{number}.json', attempt)
            save(cache / 'receipt.json', receipt)
            archive = cache / ('posterior.npz' if diagnostic['converged'] else f'failed_{number}.npz')
            temporary = archive.with_suffix('.tmp.npz')
            require(not temporary.exists(), 'Preserve unexpected partial archive bytes')
            np.savez_compressed(temporary, **post)
            validate()
            temporary.replace(archive)
            print(json.dumps(dict(study=plan['case'], attempt=number, state='recorded',
                converged=diagnostic['converged'], divergences=diagnostic['divergences'])), flush=True)
            if diagnostic['converged']:
                receipt.update(state='converged', posterior_file=archive.name, posterior_sha256=records.sha(archive))
                break
            del post
            post = None
        else:
            receipt['state'] = 'failed_convergence'
    except BaseException as exc:
        receipt.update(state='failed_exception', exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        receipt['execution_continuation'].update(finished_at=now(), seconds=time.monotonic()-started,
            process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(cache / 'receipt.json', receipt)
    return post, receipt


def execute(path, unit):
    actual = actual_service(unit)
    frozen, parent, metadata = checked_registration(path)
    frozen['continuation_registration_sha256'] = records.sha(path)
    output = ROOT / 'outputs/championship/development' / IDENTIFIER
    cache = ROOT / 'outputs/championship/cache' / IDENTIFIER / frozen['plan']['case']
    require(not output.exists() and not cache.parent.exists(), 'Preserve every earlier continuation execution')
    output.mkdir()
    begun = time.monotonic()
    result = dict(state='running', case=frozen['plan']['case'], actual_resources=actual,
        registration=relative(path), registration_sha256=records.sha(path), started_at=now(),
        original_profile_complete=False, original_records_replaced=False,
        full_recovery_complete=False, sbc_complete=False, publication_allowed=False)
    save(output / 'progress.json', result)
    try:
        import jax
        from analysis.overall_complete_profile import sample, METHOD
        from analysis.overall_control_rotation import build_rotation
        from analysis.overall_recovery_checks import generate_with_parameters
        from f1rank.overall.recovery import recovery_metrics
        require(jax.default_backend() == 'cpu' and not jax.config.x64_enabled
                and jax.local_device_count() == parent['fit_settings']['chains'],
                'Keep the original CPU precision and chain-device runtime')
        case = frozen['plan']['case']
        generated = generate_with_parameters(case, parent['synthetic'][case]['seed'])
        expected = records.read(ROOT / parent['preflight_results'][case])
        require(all(generated[k] == expected[k] for k in ('arrays_key', 'parameters_key', 'truth_key')),
                'Never redraw or replace the original synthetic design')
        blocks, catalog, truth = (generated[k] for k in ('blocks', 'catalog', 'truth'))
        rotation = build_rotation(blocks['laps'])
        require(rotation.fingerprint() == expected['control_rotation_sha256'], 'Original coordinate map changed')
        catalog.update(method=METHOD, control_rotation_sha256=rotation.fingerprint(), additional_source_sha256=parent['source_sha256'])
        study = next(s for s in metadata['studies'] if s['case'] == case)
        require(catalog == study['receipt']['catalog'], 'Original catalogue changed')
        sizes = records.read(ROOT / parent['storage_results'][case])['sites']
        save(output / 'truth.json', dict(seed=generated['seed'], role=generated['role'],
            arrays_key=generated['arrays_key'], parameters_key=generated['parameters_key'], truth_key=generated['truth_key'],
            truth={k: v.tolist() for k, v in truth.items()}))
        cache.mkdir(parents=True)
        preserve_attempts(study, cache)
        execution = dict(registration=relative(path), registration_sha256=records.sha(path),
            original_receipt_sha256=study['receipt_sha256'], original_cache=str((ROOT / study['attempts'][0]['archive']).parent.relative_to(ROOT)),
            started_at=now(), source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            historical_attempts=frozen['plan']['preserved_attempts'], duration_scope='new execution segment only')
        post, receipt = run_remaining(frozen['plan'], study['receipt'], cache,
            lambda settings: sample(blocks, settings, rotation, catalog, sizes),
            lambda: recheck(frozen, path), execution)
        result.update(state=receipt['state'], receipt=receipt)
        if receipt['state'] == 'converged':
            result['descriptive_recovery'] = recovery_metrics(post, truth, catalog)
        recheck(frozen, path)
        result['combined_recorded_convergence'] = (receipt['state'] == 'converged' and all(
            r['completion']['state'] == 'converged' for r in frozen['plan']['completed_case_records']))
        result['original_completed_case_records'] = frozen['plan']['completed_case_records']
    except BaseException as exc:
        result.update(state='failed_exception', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if (cache / 'receipt.json').exists():
            result['receipt'] = records.read(cache / 'receipt.json')
        raise
    finally:
        result.update(finished_at=now(), execution_seconds=time.monotonic()-begun,
            process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(output / 'result.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--review', type=Path)
    parser.add_argument('--audit', type=Path)
    parser.add_argument('--scope-name')
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--unit')
    args = parser.parse_args()
    if args.freeze:
        require(args.review is not None and args.audit is not None and args.scope_name is not None,
                'Freeze only from the committed review and completed audit in a capped scope')
        freeze(args.review, args.audit, args.freeze, args.scope_name)
    else:
        require(args.registration is not None and args.unit is not None, 'Execution needs its frozen registration and service')
        result = execute(args.registration, args.unit)
        raise SystemExit(0 if result['state'] == 'converged' else 1)
