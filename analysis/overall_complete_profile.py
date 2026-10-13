"""Complete-design numerical profile; unchanged model, settings and retry gate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback

os.environ.setdefault('XLA_FLAGS', '--xla_force_host_platform_device_count=4')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METHOD = 'conditional_weekend_with_control_rotation_v1'
RESOURCES = dict(slice='f1.slice', MemoryMax='13G', slice_CPUQuota='800%', CPUWeight=10, Nice=19)
SOURCES = ('analysis/overall_complete_profile.py', 'analysis/overall_bounded_diagnostics.py',
           'analysis/overall_weekend_profile.py', 'analysis/overall_weekend_coordinates.py',
           'analysis/overall_control_rotation.py', 'analysis/overall_profile_geometry.py',
           'analysis/overall_recovery_checks.py', 'docs/overall_complete_profile.md')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


LOADED = {name: sha(ROOT / name) for name in SOURCES}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_files(mapping, *, committed):
    for name, expected in mapping.items():
        path = ROOT / name
        if sha(path) != expected:
            raise ValueError(f'Registered input changed: {name}')
        if committed and subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != path.read_bytes():
            raise ValueError(f'Commit the registered input: {name}')


def validate_inventory(p, preflight, parent, audit, storage):
    expected = {s['case']: dict(case=s['case'], seed=s['seed']) for s in preflight['studies']}
    if p['studies'] != list(expected) or p['synthetic'] != expected:
        raise ValueError('Preserve every original preflight case and generating seed')
    if (any(p[k] != parent[k] for k in ('fit_settings', 'retry', 'diagnostics'))
            or p['method'] != parent['method']):
        raise ValueError('Preserve the successful coordinate method and fixed numerical procedure')
    if (audit.get('state') != 'complete' or audit.get('all_saved_reports_identical') is not True
            or not audit.get('attempts') or not all(a['complete_original_report_equal'] for a in audit['attempts'])
            or audit.get('resource', {}).get('temporary_arrays_removed') is not True
            or not audit['resource'].get('diagnostic')):
        raise ValueError('Complete saved-attempt equivalence and full diagnostic resource evidence required')
    if (audit['model_inputs_key'] != p['registered_model_inputs_key']
            or preflight['model_source']['key'] != p['registered_model_inputs_key']
            or audit['sources']['analysis/overall_bounded_diagnostics.py'] != p['source_sha256']['analysis/overall_bounded_diagnostics.py']):
        raise ValueError('Use the measured bounded checker and unchanged model inputs')
    if (storage['all_processed'] is not True or storage['all_complete'] is not True
            or [s['case'] for s in storage['studies']] != p['studies']
            or set(p['preflight_results']) != set(expected) or set(p['storage_results']) != set(expected)):
        raise ValueError('Complete preflight and storage evidence is required')


def checked_registration(path):
    from f1rank.overall.fitting import committed_registration
    from f1rank.overall.provenance import snapshot
    p = committed_registration(path)
    if (p.get('purpose') != 'numerical_profile' or p.get('profile') != 'complete_design_inference'
            or p.get('resources') != RESOURCES or p.get('publication_allowed') is not False
            or not re.fullmatch('[a-z][a-z0-9-]*', p.get('id', ''))):
        raise ValueError('Use the frozen complete-design computational registration')
    if any(p['source_sha256'].get(k) != v for k, v in LOADED.items()):
        raise ValueError('Loaded profile implementation differs from its registration')
    verify_files(p['source_sha256'], committed=True)
    verify_files(p['evidence_sha256'], committed=True)
    verify_files(p['runtime_source_sha256'], committed=False)
    def evidence(name):
        if name not in p['evidence_sha256']:
            raise ValueError('Every dependency must be bound before sampling')
        return json.loads((ROOT / name).read_text())
    preflight, parent, audit, storage = [evidence(p[k]) for k in
        ('preflight_registration', 'parent_registration', 'bounded_diagnostic_audit', 'storage_summary')]
    validate_inventory(p, preflight, parent, audit, storage)
    if snapshot()['key'] != p['registered_model_inputs_key']:
        raise ValueError('Original model/data inputs changed')
    for study in p['studies']:
        prior, sizes = evidence(p['preflight_results'][study]), evidence(p['storage_results'][study])
        if (prior['case'] != study or prior['generation_seed'] != p['synthetic'][study]['seed']
                or prior['state'] != 'complete' or not prior['all_gradient_coordinates_finite']
                or sizes['case'] != study or sizes['state'] != 'complete'
                or any(sizes[k] != prior[k] for k in ('arrays_key', 'parameters_key', 'truth_key'))):
            raise ValueError('Original preflight and storage identities must match every study')
    real = evidence(p['storage_results']['real_network'])
    largest = max(real['accounting']['attempts'], key=lambda a: a['posterior_payload_bytes'])
    if (audit['resource']['complete_site_inventory'] != real['sites']
            or audit['resource']['logical_payload_bytes'] != largest['posterior_payload_bytes']
            or audit['resource']['axes'] != [largest['chains'], largest['samples_per_chain']]):
        raise ValueError('The measured diagnostic workload must cover the complete largest retry')
    return p


def actual_service(unit):
    def properties(name, keys):
        command = ['systemctl', '--user', 'show', name]
        for key in keys:
            command += ['-p', key]
        return dict(line.split('=', 1) for line in subprocess.check_output(command, text=True).splitlines())
    actual = properties(unit, ('Slice', 'MemoryMax', 'Nice', 'CPUWeight', 'ControlGroup'))
    if (any(actual[k] != v for k, v in dict(Slice='f1.slice', MemoryMax=str(13*1024**3), Nice='19', CPUWeight='10').items())
            or os.getpriority(os.PRIO_PROCESS, 0) != 19):
        raise ValueError('Use the registered capped fit service')
    groups = [line.split(':', 2)[-1] for line in Path('/proc/self/cgroup').read_text().splitlines()]
    if actual['ControlGroup'] not in groups:
        raise ValueError('The worker must belong to the verified service')
    shared = properties('f1.slice', ('CPUQuotaPerSecUSec', 'CPUWeight'))
    if shared != dict(CPUQuotaPerSecUSec='8s', CPUWeight='10'):
        raise ValueError('The shared CPU cap and priority must remain unchanged')
    return dict(service=actual, shared_slice=shared)


def validate_posterior(post, sites, settings):
    import numpy as np
    axes = (settings['chains'], settings['samples'])
    if set(post) != set(sites):
        raise ValueError('Retain every recorded posterior field')
    for name, value in post.items():
        spec = sites[name]
        if value.shape != axes + tuple(spec['shape']) or value.dtype != np.dtype(spec['dtype']):
            raise ValueError('Retain the complete posterior axes and original dtype')


def sample(blocks, settings, rotation, catalog, sites):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro
    from numpyro.infer import MCMC, NUTS, init_to_median
    from analysis.overall_bounded_diagnostics import check
    from analysis.overall_weekend_coordinates import weekend_model
    from analysis.overall_weekend_profile import contrast_reports
    if jax.local_device_count() < settings['chains']:
        raise RuntimeError('Insufficient registered chain devices')
    arguments = tuple({k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in blocks[key].items()}
                      for key in ('qualifying', 'laps', 'outcomes'))
    kernel = NUTS(weekend_model(rotation), target_accept_prob=settings['target_accept'],
        max_tree_depth=settings['max_tree_depth'], init_strategy=init_to_median(num_samples=settings['init_median_samples']))
    sampler = MCMC(kernel, num_warmup=settings['warmup'], num_samples=settings['samples'],
        num_chains=settings['chains'], chain_method='parallel', progress_bar=False)
    sampler.run(jax.random.PRNGKey(settings['seed']), *arguments,
                extra_fields=('diverging', 'num_steps', 'accept_prob'))
    post = {k: np.asarray(v) for k, v in sampler.get_samples(group_by_chain=True).items()}
    validate_posterior(post, sites, settings)
    extras = sampler.get_extra_fields(group_by_chain=True)
    diagnostic = check(post, int(np.asarray(extras['diverging']).sum()))
    diagnostic['comparison_reports'] = contrast_reports(post, catalog, diagnostic['divergences'])
    diagnostic['tree_depth_limit_fraction'] = float(np.mean(np.asarray(extras['num_steps']) >= 2**settings['max_tree_depth']-1))
    diagnostic['mean_accept_prob'] = float(np.mean(np.asarray(extras['accept_prob'])))
    diagnostic['runtime'] = dict(jax=jax.__version__, numpyro=numpyro.__version__, backend=jax.default_backend())
    diagnostic['control_rotation_sha256'] = rotation.fingerprint()
    diagnostic['weekend_coordinate_method'] = METHOD
    diagnostic['bounded_checker_sha256'] = LOADED['analysis/overall_bounded_diagnostics.py']
    jax.clear_caches()
    return post, diagnostic


def child(path, case, output, unit):
    p = checked_registration(path)
    if case not in p['studies']:
        raise ValueError('Study is not in the complete fixed inventory')
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    record = dict(case=case, state='running', stage='service', started_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        registration_sha256=sha(path), full_recovery_complete=False, sbc_complete=False, publication_allowed=False)
    cache = ROOT / 'outputs/championship/cache' / p['id'] / case
    try:
        record['actual_resources'] = actual_service(unit)
        record['stage'] = 'generation'
        save(output / 'progress.json', record)
        from analysis.overall_recovery_checks import generate_with_parameters
        from analysis.overall_control_rotation import build_rotation
        from f1rank.overall.fitting import run
        from f1rank.overall.recovery import recovery_metrics
        generated = generate_with_parameters(case, p['synthetic'][case]['seed'])
        expected = json.loads((ROOT / p['preflight_results'][case]).read_text())
        if any(generated[k] != expected[k] for k in ('arrays_key', 'parameters_key', 'truth_key')):
            raise ValueError('Never redraw or change the original preflight observations and truth')
        record['original_inputs'] = {k: generated[k] for k in ('seed', 'role', 'arrays_key', 'parameters_key', 'truth_key')}
        blocks, catalog, truth = (generated[k] for k in ('blocks', 'catalog', 'truth'))
        rotation = build_rotation(blocks['laps'])
        if rotation.fingerprint() != expected['control_rotation_sha256']:
            raise ValueError('The original coordinate map changed')
        sizes = json.loads((ROOT / p['storage_results'][case]).read_text())['sites']
        catalog.update(method=METHOD, control_rotation_sha256=rotation.fingerprint(), additional_source_sha256=p['source_sha256'])
        save(output / 'truth.json', dict(record['original_inputs'], truth={k: v.tolist() for k, v in truth.items()}))
        record['stage'] = 'fitting'
        save(output / 'progress.json', record)
        def checked_sample(b, s):
            checked_registration(path)
            result = sample(b, s, rotation, catalog, sizes)
            checked_registration(path)
            return result
        post, receipt = run(blocks, catalog, cache, path, study=case, synthetic=True, sample=checked_sample)
        record.update(state='converged', receipt=receipt, recovery=recovery_metrics(post, truth, catalog))
        checked_registration(path)
    except BaseException as exc:
        record.update(state='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        if (cache / 'receipt.json').exists():
            record['receipt'] = json.loads((cache / 'receipt.json').read_text())
        raise
    finally:
        import resource
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started,
                      process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(output / 'result.json', record)
    return record


def execute(path, unit):
    path = Path(path).resolve()
    p = checked_registration(path)
    actual_service(unit)
    output = ROOT / 'outputs/championship/development' / p['id']
    output.mkdir(exist_ok=False)
    rows = []
    for case in p['studies']:
        checked_registration(path)
        log, resources = output / f'{case}.output.txt', output / f'{case}.resources.txt'
        command = ['/usr/bin/time', '-v', '-o', str(resources), sys.executable, '-u', str(Path(__file__).resolve()),
                   '--registration', str(path), '--unit', unit, '--case', case, '--output', str(output / case)]
        with log.open('x') as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        result_path = output / case / 'result.json'
        result = json.loads(result_path.read_text()) if result_path.exists() else None
        row = dict(case=case, exit_code=process.returncode,
            state='converged' if process.returncode == 0 and result and result['state'] == 'converged' else 'failed',
            result_present=result is not None, result_sha256=sha(result_path) if result_path.exists() else None,
            log_sha256=sha(log), resources_sha256=sha(resources) if resources.exists() else None, command=command)
        save(output / f'{case}.completion.json', row)
        rows.append(row)
        save(output / 'progress.json', dict(studies=rows, all_processed=False, publication_allowed=False))
        print(json.dumps(row), flush=True)
    checked_registration(path)
    result = dict(registration=str(path.relative_to(ROOT)), registration_sha256=sha(path), studies=rows,
        all_processed=True, all_converged=all(r['state'] == 'converged' for r in rows),
        full_recovery_complete=False, sbc_complete=False, publication_allowed=False)
    save(output / 'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path, required=True)
    parser.add_argument('--unit', required=True)
    parser.add_argument('--case')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.case:
        child(args.registration, args.case, args.output, args.unit)
    else:
        result = execute(args.registration, args.unit)
        raise SystemExit(0 if result['all_converged'] else 1)
