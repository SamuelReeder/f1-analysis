"""Matched full-model computation audit; no sampler, fit or publication action."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.overall_recovery_preflight import sha, save

SELF_SHA = sha(__file__)
METHODS = ('original', 'batched')
REPEATS = 8
RESOURCES = dict(slice='f1.slice', MemoryMax='3G', CPUQuota='200%', Nice=19)
ENVIRONMENT = dict(JAX_ENABLE_X64='false', JAX_PLATFORMS='cpu', JAX_ENABLE_COMPILATION_CACHE='false',
    OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
    XLA_FLAGS='--xla_force_host_platform_device_count=1')
PREFLIGHT = 'docs/overall_recovery_preflight_v2_protocol.json'
PREFLIGHT_OUTPUT = 'outputs/championship/development/overall-recovery-preflight-v2'
AUDIT_SOURCES = ('analysis/overall_batched_rotation.py',
    'analysis/overall_batched_rotation_audit.py', 'tests/test_overall_batched_rotation.py',
    'tests/test_overall_batched_rotation_audit.py', 'docs/overall_batched_rotation.md')


def freeze(path):
    """Write the fixed workload; execution additionally requires its commit."""
    from analysis.overall_recovery_preflight import checked_protocol as checked_preflight
    previous = checked_preflight(ROOT / PREFLIGHT)
    sources = dict(previous['source_sha256'], **previous['model_source']['inputs'])
    for name in AUDIT_SOURCES:
        if subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != (ROOT/name).read_bytes():
            raise ValueError(f'Commit the audit source first: {name}')
        sources[name] = sha(ROOT/name)
    inputs = {PREFLIGHT: sha(ROOT/PREFLIGHT)}
    for spec in previous['studies']:
        name = f"{PREFLIGHT_OUTPUT}/{spec['case']}/result.json"
        inputs[name] = sha(ROOT/name)
    protocol = dict(id='overall-batched-rotation-audit-v1', status='frozen',
        purpose='unchanged_complete_model_batched_control_evaluation',
        registered_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        preflight_protocol=PREFLIGHT, preflight_output=PREFLIGHT_OUTPUT,
        studies=previous['studies'], methods=list(METHODS),
        resources=RESOURCES, environment=ENVIRONMENT, repeat_evaluations=REPEATS,
        density_tolerance=previous['density_tolerance'], state_tolerance=previous['state_tolerance'],
        gradient_tolerance=previous['state_tolerance'], source_sha256=sources, input_sha256=inputs,
        mcmc_allowed=False, publication_allowed=False,
        interpretation='Full potential and gradient at the original generating point, not fitting throughput, convergence or recovery.',
        stopping='Every original design and both implementations once, then complete comparisons; no retry or tolerance change.')
    with Path(path).open('x') as handle:
        handle.write(json.dumps(protocol, indent=2, allow_nan=False)+'\n')
    return protocol


def checked_protocol(path):
    path = Path(path).resolve()
    content = path.read_bytes()
    if subprocess.check_output(['git', 'show', f'HEAD:{path.relative_to(ROOT)}'], cwd=ROOT) != content:
        raise ValueError('Commit the computation workload before execution')
    p = json.loads(content)
    if (p['purpose'] != 'unchanged_complete_model_batched_control_evaluation' or p['status'] != 'frozen'
            or p['mcmc_allowed'] is not False or p['publication_allowed'] is not False
            or not re.fullmatch('[a-z][a-z0-9-]*', p['id'])
            or p['methods'] != list(METHODS) or p['resources'] != RESOURCES
            or p['environment'] != ENVIRONMENT or p['repeat_evaluations'] != REPEATS
            or p['density_tolerance'] != dict(atol=.005, rtol=.000002)
            or p['state_tolerance'] != dict(atol=.000003, rtol=.000003)
            or p['gradient_tolerance'] != p['state_tolerance']
            or p['preflight_protocol'] != PREFLIGHT or p['preflight_output'] != PREFLIGHT_OUTPUT
            or p['source_sha256'].get('analysis/overall_batched_rotation_audit.py') != SELF_SHA):
        raise ValueError('Wrong or changed complete computation workload')
    from analysis.overall_recovery_preflight import checked_protocol as checked_preflight
    previous = checked_preflight(ROOT/PREFLIGHT)
    if set(p['source_sha256']) != set(previous['source_sha256']) | set(previous['model_source']['inputs']) | set(AUDIT_SOURCES):
        raise ValueError('Bind the complete original and batched source inventory')
    if p['studies'] != previous['studies']:
        raise ValueError('Keep every original design and generation seed')
    required = {PREFLIGHT, *(f"{PREFLIGHT_OUTPUT}/{s['case']}/result.json" for s in p['studies'])}
    if set(p['input_sha256']) != required:
        raise ValueError('Bind every original preflight result')
    for group in ('source_sha256', 'input_sha256'):
        for name, expected in p[group].items():
            target = ROOT/name
            if (sha(target) != expected or subprocess.check_output(
                    ['git', 'show', f'HEAD:{name}'], cwd=ROOT) != target.read_bytes()):
                raise ValueError(f'Bound computation source or input changed: {name}')
    return p


def check_scope(scope):
    properties = subprocess.check_output(['systemctl', '--user', 'show', scope,
        '-p', 'Slice', '-p', 'MemoryMax', '-p', 'CPUQuotaPerSecUSec'], text=True)
    actual = dict(line.split('=', 1) for line in properties.splitlines())
    if (actual != dict(Slice='f1.slice', MemoryMax=str(3*1024**3), CPUQuotaPerSecUSec='2s')
            or os.getpriority(os.PRIO_PROCESS, 0) != 19):
        raise ValueError('Use the fixed capped scope and priority')
    if any(os.environ.get(k) != v for k, v in ENVIRONMENT.items()):
        raise ValueError('Use the same explicit numerical environment for every child')
    return dict(actual, Nice=19)


def collect(case, method, protocol, record, payload_path, phase):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro
    from numpyro.infer.util import log_density, potential_energy, unconstrain_fn
    from analysis.overall_recovery_checks import generate_with_parameters
    from analysis.overall_control_rotation import ROTATED_SITE, build_rotation
    from analysis.overall_batched_rotation import prepare
    from analysis.overall_weekend_coordinates import HYPER_SITES, TOTAL, COMPLEMENT, transform, weekend_model
    spec = next(s for s in protocol['studies'] if s['case'] == case)
    previous = json.loads((ROOT/protocol['preflight_output']/case/'result.json').read_text())
    record['runtime'] = dict(python=sys.version, numpy=np.__version__, jax=jax.__version__,
        numpyro=numpyro.__version__, backend=jax.default_backend(), x64=jax.config.x64_enabled,
        devices=[str(d) for d in jax.devices()], environment=ENVIRONMENT)
    phase('matched_generation')
    generated = generate_with_parameters(case, spec['seed'])
    for key in ('arrays_key', 'parameters_key', 'truth_key'):
        if generated[key] != previous[key]: raise ValueError('Original generating input changed')
        record[key] = generated[key]
    blocks, params = generated['blocks'], generated['parameters']
    original = build_rotation(blocks['laps'])
    if original.fingerprint() != previous['control_rotation_sha256']:
        raise ValueError('Original coordinate map changed')
    # Use the original inverse even in the batched arm: identical supplied coordinates.
    updated = dict(params)
    for old, new in HYPER_SITES.items(): updated[new] = updated.pop(old)
    updated[TOTAL], updated[COMPLEMENT] = transform(params['sd_driver_form'], params['overall_sd_day'],
        params['overall_shared_weekend'], updated.pop('driver_form_z'), updated.pop('overall_day_z'))
    updated[ROTATED_SITE] = original.rotated({name: updated.pop(name) for name in original.shapes})
    updated = {k: jnp.asarray(v) for k, v in updated.items()}
    args = tuple({k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in blocks[name].items()}
                 for name in ('qualifying', 'laps', 'outcomes'))
    phase('coordinate_preparation')
    start = time.monotonic()
    rotation = original if method == 'original' else prepare(original)
    record['preparation_seconds'] = time.monotonic()-start
    record.update(control_rotation_sha256=rotation.fingerprint(), control_log_abs_det=rotation.log_abs_det)
    wrapped = weekend_model(rotation)
    phase('complete_trace_and_parameters')
    density, trace = log_density(wrapped, args, {}, updated)
    payload, kinds = {}, {}

    def retain(name, value, kind):
        a = np.asarray(value)
        if a.dtype.kind not in 'biufc' or not np.isfinite(a).all():
            raise ValueError(f'Nonfinite or unsupported complete array: {name}')
        payload[name], kinds[name] = a, kind

    retain('joint_density', density, 'density')
    for name, site in trace.items():
        if site['type'] not in ('sample', 'deterministic'): continue
        retain('state/'+name, site['value'], 'state')
        if site['type'] == 'sample':
            retain('factor/'+name, site['fn'].log_prob(site['value']), 'density')
    del trace
    z = unconstrain_fn(wrapped, args, {}, updated)
    for name, value in z.items(): retain('coordinate/'+name, value, 'coordinate')
    phase('compile_and_first_complete_evaluation')
    start = time.monotonic()
    compute = jax.jit(jax.value_and_grad(lambda v: potential_energy(wrapped, args, {}, v)))
    energy, gradient = jax.block_until_ready(compute(z))
    record['compile_and_first_evaluation_seconds'] = time.monotonic()-start
    if set(gradient) != set(z): raise ValueError('Incomplete derivative inventory')
    retain('potential_energy', energy, 'density')
    for name, value in gradient.items():
        if value.shape != z[name].shape: raise ValueError('Incomplete derivative shape')
        retain('gradient/'+name, value, 'gradient')
    phase('repeated_complete_evaluations')
    durations = []
    for _ in range(protocol['repeat_evaluations']):
        start = time.monotonic()
        last_energy, last_gradient = jax.block_until_ready(compute(z))
        durations.append(time.monotonic()-start)
    np.testing.assert_array_equal(last_energy, energy)
    for name in gradient: np.testing.assert_array_equal(last_gradient[name], gradient[name])
    record.update(repeated_evaluation_seconds=durations,
        parameter_coordinates=sum(v.size for v in z.values()),
        payload_inventory={k:dict(kind=kinds[k], shape=list(v.shape), dtype=v.dtype.str) for k, v in payload.items()})
    phase('save_complete_comparison_arrays')
    with Path(payload_path).open('xb') as handle: np.savez(handle, **payload)
    record['payload_sha256'] = sha(payload_path)


def compare_records(original, batched, left, right, protocol):
    """Compare every saved state, probability factor and derivative coordinate."""
    import numpy as np
    if (original['case'] != batched['case'] or original['method'] != METHODS[0]
            or batched['method'] != METHODS[1] or original['protocol_sha256'] != batched['protocol_sha256']):
        raise ValueError('Compare the same case and protocol with distinct specified implementations')
    for key in ('arrays_key', 'parameters_key', 'truth_key', 'control_rotation_sha256',
                'control_log_abs_det', 'runtime', 'parameter_coordinates', 'payload_inventory'):
        if original[key] != batched[key]: raise ValueError(f'Matched identity or inventory differs: {key}')
    inventory = original['payload_inventory']
    if not inventory or set(left) != set(right) or set(left) != set(inventory):
        raise ValueError('Both complete comparison payloads are required')
    checks = []
    for key, spec in inventory.items():
        a, b = np.asarray(left[key]), np.asarray(right[key])
        if (a.shape != tuple(spec['shape']) or b.shape != a.shape
                or a.dtype.str != spec['dtype'] or b.dtype != a.dtype
                or not np.isfinite(a).all() or not np.isfinite(b).all()):
            raise ValueError(f'Incomplete or nonfinite comparison array: {key}')
        if spec['kind'] == 'coordinate': np.testing.assert_array_equal(a, b, err_msg=key)
        else:
            tolerance = protocol[spec['kind']+'_tolerance']
            np.testing.assert_allclose(b, a, err_msg=key, **tolerance)
        delta = np.abs(b.astype(float)-a.astype(float))
        checks.append(dict(site=key, kind=spec['kind'], coordinates=int(a.size),
            max_absolute_discrepancy=float(delta.max()) if delta.size else 0.))
    return checks


def child(path, case, method, output, cache, scope):
    p = checked_protocol(path)
    output, cache = Path(output), Path(cache)
    output.mkdir(exist_ok=False)
    start = time.monotonic()
    record = dict(case=case, method=method, state='running', stage='scope',
        started_at=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(path),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        mcmc_run=False, publication_allowed=False)
    def phase(stage):
        record['stage'] = stage
        save(output/'progress.json', record)
    try:
        record['actual_scope'] = check_scope(scope)
        if method in METHODS:
            collect(case, method, p, record, cache/f'{case}-{method}.npz', phase)
        else:
            import numpy as np
            records = [json.loads((output.parent/m/'result.json').read_text()) for m in METHODS]
            paths = [cache/f'{case}-{m}.npz' for m in METHODS]
            for r, file in zip(records, paths):
                if r['state'] != 'complete' or sha(file) != r['payload_sha256']:
                    raise ValueError('A complete verified arm is missing')
            phase('all_full_model_coordinates')
            with np.load(paths[0], allow_pickle=False) as a, np.load(paths[1], allow_pickle=False) as b:
                record['comparisons'] = compare_records(*records, a, b, p)
        phase('source_recheck')
        if checked_protocol(path) != p: raise ValueError('Computation workload changed')
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-start,
            process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(output/'result.json', record)


def execute(path):
    p = checked_protocol(path)
    output = ROOT/'outputs/championship/development'/p['id']
    cache = ROOT/'outputs/championship/cache'/p['id']
    output.mkdir(exist_ok=False)
    cache.mkdir(exist_ok=False)
    rows = []
    for study in p['studies']:
        case = study['case']
        (output/case).mkdir()
        for method in (*METHODS, 'compare'):
            checked_protocol(path)
            scope = f"f1-batch-audit-{case.replace('_', '-')}-{method}.scope"
            log, resources = output/case/f'{method}.output.txt', output/case/f'{method}.resources.txt'
            command = ['systemd-run', '--user', '--scope', '--unit', scope, '--slice=f1.slice', '-q',
                '-p', 'MemoryMax=3G', '-p', 'CPUQuota=200%', '--', '/usr/bin/time', '-v', '-o', str(resources),
                '/usr/bin/nice', '-n', '19', sys.executable, str(Path(__file__).resolve()),
                '--protocol', str(Path(path).resolve()), '--case', case, '--method', method,
                '--output', str(output/case/method), '--cache', str(cache), '--scope-name', scope]
            with log.open('x') as handle:
                process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                    env=dict(os.environ, **ENVIRONMENT))
            result_path = output/case/method/'result.json'
            result = json.loads(result_path.read_text()) if result_path.exists() else None
            row = dict(case=case, method=method, process_exit_code=process.returncode, command=command,
                state='complete' if process.returncode == 0 and result and result['state']=='complete' else 'failed',
                result_sha256=sha(result_path) if result_path.exists() else None,
                resources_sha256=sha(resources) if resources.exists() else None, output_sha256=sha(log))
            save(output/case/f'{method}.completion.json', row)
            rows.append(row)
            print(json.dumps(dict(case=case, method=method, state=row['state'])), flush=True)
    summary = dict(protocol_sha256=sha(path), rows=rows, all_processed=True,
        all_complete=all(r['state']=='complete' for r in rows), mcmc_run=False, publication_allowed=False)
    save(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--freeze', type=Path)
    parser.add_argument('--protocol', type=Path)
    parser.add_argument('--case')
    parser.add_argument('--method', choices=(*METHODS, 'compare'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--scope-name')
    args = parser.parse_args()
    if args.freeze:
        freeze(args.freeze)
    elif args.protocol and args.case:
        if not all((args.method, args.output, args.cache, args.scope_name)): parser.error('Incomplete child arguments')
        child(args.protocol, args.case, args.method, args.output, args.cache, args.scope_name)
    elif args.protocol:
        raise SystemExit(0 if execute(args.protocol)['all_complete'] else 1)
    else:
        parser.error('Provide a protocol or a new frozen workload destination')
