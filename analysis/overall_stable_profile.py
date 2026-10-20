"""New source-bound computational profile; preserve the exhausted original run."""
import argparse
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
from analysis import overall_stable_preflight as preflight

sha, save, committed = preflight.sha, preflight.save, preflight.committed
PREFLIGHT_PROTOCOL = 'docs/overall_stable_preflight_protocol.json'
PREFLIGHT_OUTPUT = 'outputs/championship/development/overall-stable-preflight-v1'
ARITHMETIC_RUNTIME = 'docs/overall_studentt_numerics_v2_protocol.json'
SOURCES = ('analysis/overall_stable_profile.py', 'tests/test_overall_stable_profile.py',
           'docs/overall_stable_profile.md')
RESOURCES = dict(slice='f1.slice', MemoryMax='13G', slice_CPUQuota='400%', CPUWeight=10, Nice=19)
ENVIRONMENT = dict(JAX_PLATFORMS='cpu', JAX_ENABLE_X64='false', JAX_ENABLE_COMPILATION_CACHE='false',
    XLA_FLAGS='--xla_force_host_platform_device_count=4', OPENBLAS_NUM_THREADS='2',
    OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')


def definition(p):
    from f1rank.overall.fitting import COORDINATE_RUN_SCHEMA
    return dict(id='overall-stable-profile-v1', schema_version=COORDINATE_RUN_SCHEMA,
        purpose='numerical_profile', profile='corrected_studentt_complete_design',
        inference_method=p['inference_method'], registered_model_inputs_key=p['model_snapshot']['key'],
        studies=p['studies'], synthetic=p['synthetic'], fit_settings=p['fit_settings'],
        retry=p['retry'], diagnostics=p['diagnostics'], resources=RESOURCES, environment=ENVIRONMENT,
        full_recovery_complete=False, sbc_complete=False, publication_allowed=False)


def preflight_evidence(p):
    names = [PREFLIGHT_PROTOCOL, ARITHMETIC_RUNTIME, PREFLIGHT_OUTPUT+'/summary.json']
    names.extend(f'{PREFLIGHT_OUTPUT}/{case}/result.json' for case in p['studies'])
    return names


def validate_prerequisites(p, summary, records):
    if (summary.get('all_processed') is not True or summary.get('all_complete') is not True
            or summary.get('model_inputs_key') != p['model_snapshot']['key']
            or [row['case'] for row in summary['studies']] != p['studies']
            or set(records) != set(p['studies'])):
        raise ValueError('Require the complete matching preflight inventory')
    for row in summary['studies']:
        record = records[row['case']]
        if (row['exit_code'] != 0 or row['state'] != 'complete' or record['state'] != 'complete'
                or record['model_inputs_key'] != p['model_snapshot']['key']
                or record['generation_seed'] != p['synthetic'][row['case']]['seed']
                or record['all_gradient_coordinates_finite'] is not True
                or record['original_collector_inventory_identical'] is not True
                or record['complete_collector_matches_postprocessing'] is not True
                or not record['sites']):
            raise ValueError('Preflight failures cannot enter a new computational fit')


def load_preflight(p):
    summary = json.loads((ROOT/PREFLIGHT_OUTPUT/'summary.json').read_bytes())
    records = {case:json.loads((ROOT/PREFLIGHT_OUTPUT/case/'result.json').read_bytes()) for case in p['studies']}
    if summary['protocol_sha256'] != sha(ROOT/PREFLIGHT_PROTOCOL):
        raise ValueError('Preflight belongs to another registration')
    for row in summary['studies']:
        if sha(ROOT/PREFLIGHT_OUTPUT/row['case']/'result.json') != row['result_sha256']:
            raise ValueError('Preflight result differs from its complete summary')
    validate_prerequisites(p, summary, records)
    return records


def freeze(path):
    from analysis.overall_studentt_production_audit import checked as checked_migration
    p = preflight.checked(ROOT/PREFLIGHT_PROTOCOL)
    checked_migration(ROOT/preflight.MIGRATION_PROTOCOL)
    load_preflight(p)
    runtime = dict(p['runtime_source_sha256'], **json.loads((ROOT/ARITHMETIC_RUNTIME).read_bytes())['library_sha256'])
    for name, value in runtime.items():
        if sha(name) != value:
            raise ValueError('Verified numerical runtime changed')
    result = dict(definition(p), status='frozen', registered_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={name:committed(name) for name in SOURCES},
        evidence_sha256={name:committed(name) for name in preflight_evidence(p)}, runtime_source_sha256=runtime)
    with Path(path).open('x') as f:
        json.dump(result, f, indent=2); f.write('\n')


def checked(path):
    from f1rank.overall.fitting import committed_registration
    p = preflight.checked(ROOT/PREFLIGHT_PROTOCOL)
    result = committed_registration(path)
    if any(result.get(k) != v for k, v in definition(p).items()):
        raise ValueError('Changed computational profile or numerical procedure')
    if set(result['source_sha256']) != set(SOURCES) or set(result['evidence_sha256']) != set(preflight_evidence(p)):
        raise ValueError('Incomplete fit source or evidence inventory')
    for name, value in {**result['source_sha256'], **result['evidence_sha256']}.items():
        if committed(name) != value:
            raise ValueError('Registered profile source or evidence changed')
    runtime = dict(p['runtime_source_sha256'], **json.loads((ROOT/ARITHMETIC_RUNTIME).read_bytes())['library_sha256'])
    if result['runtime_source_sha256'] != runtime or any(sha(name) != value for name, value in runtime.items()):
        raise ValueError('Verified runtime changed')
    load_preflight(p)
    return result


def actual_service(unit):
    def properties(name, keys):
        command = ['systemctl', '--user', 'show', name]
        for key in keys:
            command += ['-p', key]
        return dict(line.split('=', 1) for line in subprocess.check_output(command, text=True).splitlines())
    actual = properties(unit, ('Slice', 'MemoryMax', 'Nice', 'CPUWeight', 'ControlGroup', 'ActiveState'))
    expected = dict(Slice='f1.slice', MemoryMax=str(13*1024**3), Nice='19', CPUWeight='10', ActiveState='active')
    if any(actual.get(k) != v for k, v in expected.items()) or os.getpriority(os.PRIO_PROCESS, 0) != 19:
        raise ValueError('Use the declared capped low-priority fit service')
    membership = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    if membership != actual['ControlGroup']:
        raise ValueError('Worker is outside the verified service')
    shared = properties('f1.slice', ('CPUQuotaPerSecUSec', 'CPUWeight'))
    if shared != dict(CPUQuotaPerSecUSec='4s', CPUWeight='10'):
        raise ValueError('Preserve the user-reduced shared CPU quota and weight')
    own = Path('/sys/fs/cgroup')/membership.lstrip('/')
    for sibling in own.parent.iterdir():
        if sibling.is_dir() and sibling != own:
            events = dict(line.split() for line in (sibling/'cgroup.events').read_text().splitlines())
            if events.get('populated') != '0':
                raise ValueError('Another fit or heavy analysis is active')
    if (own.parent/'cgroup.procs').read_text().strip() or any(os.environ.get(k) != v for k, v in ENVIRONMENT.items()):
        raise ValueError('Uncapped shared work or changed fit environment')
    return dict(service=actual, shared_slice=shared, environment=ENVIRONMENT)


def validate_attempt_inventories(receipt, sites):
    if not receipt.get('attempts'):
        raise ValueError('No completed fit attempt inventory')
    for attempt in receipt['attempts']:
        expected = {name:dict(shape=[attempt['chains'], attempt['samples'], *site['shape']], dtype=site['dtype'])
                    for name, site in sites.items()}
        if attempt.get('posterior_sites') != expected:
            raise ValueError('Every retained posterior field and chain/draw axis is required')


def child(path, case, output, unit):
    p = checked(path)
    if case not in p['studies']:
        raise ValueError('Unknown registered case')
    output = Path(output); output.mkdir(exist_ok=False)
    cache = ROOT/'outputs/championship/cache'/p['id']/case
    if cache.exists():
        raise FileExistsError('Never reuse an existing or exhausted computational case')
    begun = time.monotonic()
    record = dict(case=case, state='running', started_at=datetime.now(timezone.utc).isoformat(),
        registration_sha256=sha(path), model_inputs_key=p['registered_model_inputs_key'],
        full_recovery_complete=False, sbc_complete=False, publication_allowed=False)
    try:
        record['actual_resources'] = actual_service(unit)
        record['stage'] = 'regenerate_original_problem'; save(output/'progress.json', record)
        from analysis.overall_recovery_checks import generate_with_parameters
        from f1rank.overall.control_rotation import build_rotation
        from f1rank.overall.fitting import run
        from f1rank.overall.recovery import recovery_metrics
        generated = generate_with_parameters(case, p['synthetic'][case]['seed'])
        expected = json.loads((ROOT/PREFLIGHT_OUTPUT/case/'result.json').read_bytes())
        if any(generated[k] != expected[k] for k in preflight.IDENTITIES):
            raise ValueError('Original generated observations or truth changed')
        blocks, catalog, truth = (generated[k] for k in ('blocks', 'catalog', 'truth'))
        rotation = build_rotation(blocks['laps'])
        if rotation.fingerprint() != expected['control_rotation_sha256']:
            raise ValueError('Original control map changed')
        record['original_inputs'] = {k:generated[k] for k in (*preflight.IDENTITIES, 'seed', 'role')}
        catalog.update(inference_method=p['inference_method'], control_rotation_sha256=rotation.fingerprint(),
                       preflight_sha256=sha(ROOT/PREFLIGHT_OUTPUT/case/'result.json'))
        save(output/'truth.json', dict(record['original_inputs'], truth={k:v.tolist() for k, v in truth.items()}))
        checked(path)
        record['stage'] = 'fitting'; save(output/'progress.json', record)
        post, receipt = run(blocks, catalog, cache, path, study=case, synthetic=True)
        validate_attempt_inventories(receipt, expected['sites'])
        record.update(state='converged', receipt=receipt, posterior_inventory_verified=True,
                      recovery=recovery_metrics(post, truth, catalog))
        checked(path)
    except BaseException as exc:
        record.update(state='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if (cache/'receipt.json').exists():
            record['receipt'] = json.loads((cache/'receipt.json').read_bytes())
        raise
    finally:
        record.update(seconds=time.monotonic()-begun, process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_at=datetime.now(timezone.utc).isoformat())
        save(output/'result.json', record)


def execute(path, unit):
    path = Path(path).resolve(); p = checked(path); actual_service(unit)
    output = ROOT/'outputs/championship/development'/p['id']; output.mkdir(exist_ok=False)
    rows = []
    for case in p['studies']:
        checked(path)
        log = output/f'{case}.output.txt'; resources = output/f'{case}.resources.txt'
        command = ['/usr/bin/time', '-v', '-o', str(resources), sys.executable, '-u', str(Path(__file__).resolve()),
            'child', '--registration', str(path), '--unit', unit, '--case', case, '--output', str(output/case)]
        with log.open('x') as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        result_path = output/case/'result.json'
        result = json.loads(result_path.read_bytes()) if result_path.exists() else None
        success = (process.returncode == 0 and result and result.get('state') == 'converged'
                   and result.get('posterior_inventory_verified') is True)
        row = dict(case=case, exit_code=process.returncode, state='converged' if success else 'failed',
            result_present=result is not None, result_sha256=sha(result_path) if result_path.exists() else None,
            log_sha256=sha(log), resources_sha256=sha(resources) if resources.exists() else None, command=command)
        save(output/f'{case}.completion.json', row); rows.append(row)
        save(output/'progress.json', dict(studies=rows, all_processed=False, publication_allowed=False))
        print(json.dumps(row), flush=True)
    checked(path)
    result = dict(registration=str(path.relative_to(ROOT)), registration_sha256=sha(path), studies=rows,
        all_processed=True, all_converged=all(r['state'] == 'converged' for r in rows),
        full_recovery_complete=False, sbc_complete=False, publication_allowed=False)
    save(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run', 'child'])
    parser.add_argument('--registration', required=True)
    for name in ('unit', 'case', 'output'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.registration)
    elif args.action == 'child':
        child(args.registration, args.case, args.output, args.unit)
    else:
        result = execute(args.registration, args.unit)
        raise SystemExit(0 if result['all_converged'] else 1)
