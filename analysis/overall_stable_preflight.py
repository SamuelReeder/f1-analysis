"""Verify unchanged generation, coordinates and storage before corrected fitting."""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PARENT = 'docs/overall_complete_profile_protocol.json'
ARITHMETIC = 'outputs/championship/development/overall-studentt-stable-v1-execution/verification.json'
MIGRATION = 'outputs/championship/development/overall-studentt-production-audit-v1-execution/verification.json'
MIGRATION_PROTOCOL = 'docs/overall_studentt_production_protocol.json'
SOURCES = ('analysis/overall_stable_preflight.py', 'analysis/overall_recovery_checks.py',
           'analysis/overall_fit_storage.py', 'analysis/overall_network_precision.py',
           'tests/test_overall_stable_preflight.py', 'docs/overall_stable_profile.md')
IDENTITIES = ('arrays_key', 'parameters_key', 'truth_key')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    temporary = Path(path).with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def committed(name):
    path = ROOT/name
    if subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != path.read_bytes():
        raise ValueError(f'Commit the registered input: {name}')
    return sha(path)


def definition(parent):
    return dict(id='overall-stable-preflight-v1', purpose='corrected_arithmetic_generation_and_storage_preflight',
        studies=parent['studies'], synthetic=parent['synthetic'], fit_settings=parent['fit_settings'],
        retry=parent['retry'], diagnostics=parent['diagnostics'],
        preflight_results=parent['preflight_results'], storage_results=parent['storage_results'],
        resources=dict(slice='f1.slice', MemoryMax='3G', CPUQuota='200%', Nice=19),
        sampling_allowed=False, publication_allowed=False)


def dependencies(parent):
    return (PARENT, ARITHMETIC, MIGRATION, MIGRATION_PROTOCOL, parent['preflight_registration'],
            *parent['preflight_results'].values(), *parent['storage_results'].values())


def freeze(path):
    from analysis.overall_studentt_production_audit import checked as checked_migration
    from f1rank.overall.fitting import FIT_SETTINGS, RETRY, require_clean_inputs
    from f1rank.overall.inference import METHOD
    from f1rank.overall.provenance import snapshot
    from f1rank.overall.recovery import SYNTHETIC_CASES
    parent = json.loads((ROOT/PARENT).read_bytes())
    if (parent['studies'] != list(SYNTHETIC_CASES) or parent['fit_settings'] != FIT_SETTINGS
            or parent['retry'] != [list(row) for row in RETRY]):
        raise ValueError('Preserve the entire original numerical profile')
    checked_migration(ROOT/MIGRATION_PROTOCOL)
    arithmetic = json.loads((ROOT/ARITHMETIC).read_bytes())
    migration = json.loads((ROOT/MIGRATION).read_bytes())
    inputs = snapshot(); require_clean_inputs(inputs)
    if (arithmetic['primitive_passed_both_precisions'] is not True
            or set(migration['precisions']) != {'float32', 'float64'}
            or any(not r['all_points_identical'] or r['model_inputs_key'] != inputs['key']
                   for r in migration['precisions'].values())):
        raise ValueError('Require complete passing arithmetic and production migration evidence')
    tolerances = json.loads((ROOT/parent['preflight_registration']).read_bytes())
    p = dict(definition(parent), status='frozen', registered_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        model_snapshot=inputs, inference_method=METHOD,
        source_sha256={name:committed(name) for name in SOURCES},
        input_sha256={name:committed(name) for name in dependencies(parent)},
        runtime_source_sha256=parent['runtime_source_sha256'],
        density_tolerance=tolerances['density_tolerance'], state_tolerance=tolerances['state_tolerance'])
    for name, expected in p['runtime_source_sha256'].items():
        if sha(name) != expected:
            raise ValueError('Original numerical runtime changed')
    with Path(path).open('x') as f:
        json.dump(p, f, indent=2); f.write('\n')


def checked(path):
    from f1rank.overall.inference import METHOD
    from f1rank.overall.provenance import require_snapshot
    path = Path(path).resolve(); committed(path.relative_to(ROOT).as_posix())
    p = json.loads(path.read_bytes()); parent = json.loads((ROOT/PARENT).read_bytes())
    if any(p.get(k) != v for k, v in definition(parent).items()) or p.get('status') != 'frozen':
        raise ValueError('Changed preflight inventory, settings or purpose')
    if set(p['source_sha256']) != set(SOURCES) or set(p['input_sha256']) != set(dependencies(parent)):
        raise ValueError('Incomplete source or evidence inventory')
    for name, value in {**p['source_sha256'], **p['input_sha256']}.items():
        if committed(name) != value:
            raise ValueError('Changed committed preflight source or input')
    tolerances = json.loads((ROOT/parent['preflight_registration']).read_bytes())
    if any(p[k] != tolerances[k] for k in ('density_tolerance', 'state_tolerance')):
        raise ValueError('Original equivalence criteria changed')
    if p['runtime_source_sha256'] != parent['runtime_source_sha256']:
        raise ValueError('Changed runtime inventory')
    for name, value in p['runtime_source_sha256'].items():
        if sha(name) != value:
            raise ValueError('Numerical runtime changed')
    require_snapshot(p['model_snapshot'])
    if p['inference_method'] != METHOD:
        raise ValueError('Production inference method changed')
    return p


def validate_generated(generated, expected, sites, old_sites, rotation):
    if any(generated[k] != expected[k] for k in IDENTITIES):
        raise ValueError('Preserve the exact original generated observations, parameters and truth')
    if rotation != expected['control_rotation_sha256']:
        raise ValueError('Original coordinate map changed')
    if sites != old_sites:
        raise ValueError('Complete collector inventory changed')


def child(path, case, output, scope):
    p = checked(path)
    if case not in p['studies']:
        raise ValueError('Unknown preflight case')
    output = Path(output); output.mkdir(exist_ok=False)
    started = time.monotonic()
    record = dict(case=case, state='running', protocol_sha256=sha(path),
        model_inputs_key=p['model_snapshot']['key'], generation_seed=p['synthetic'][case]['seed'],
        started_at=datetime.now(timezone.utc).isoformat(), mcmc_executed=False, publication_allowed=False)
    def phase(name):
        record['stage'] = name; save(output/'progress.json', record)
    try:
        from analysis.overall_network_precision import scope_properties
        record['actual_scope'] = scope_properties(scope, 'float32')
        import jax
        import jax.numpy as jnp
        import numpy as np
        from numpyro.infer.mcmc import _collect_and_postprocess
        from numpyro.infer.util import constrain_fn, log_density, potential_energy, unconstrain_fn
        from analysis.overall_fit_storage import accounting, inventory
        from analysis.overall_recovery_checks import generate_with_parameters
        from f1rank.overall.control_rotation import ROTATED_SITE, build_rotation
        from f1rank.overall.batched_rotation import prepare
        from f1rank.overall.weekend_coordinates import HYPER_SITES, TOTAL, COMPLEMENT, transform, weekend_model
        from f1rank.overall.model import model
        if jax.config.x64_enabled:
            raise ValueError('Preserve original float32 generation and fitting precision')
        phase('generation')
        generated = generate_with_parameters(case, p['synthetic'][case]['seed'])
        expected = json.loads((ROOT/p['preflight_results'][case]).read_bytes())
        if any(generated[k] != expected[k] for k in IDENTITIES):
            raise ValueError('Never redraw or change the original generated problem')
        record.update({k:generated[k] for k in (*IDENTITIES, 'role')})
        blocks, parameters = generated['blocks'], generated['parameters']
        rotation = prepare(build_rotation(blocks['laps']))
        updated = dict(parameters)
        for original, new in HYPER_SITES.items():
            updated[new] = updated.pop(original)
        updated[TOTAL], updated[COMPLEMENT] = transform(parameters['sd_driver_form'], parameters['overall_sd_day'],
            parameters['overall_shared_weekend'], updated.pop('driver_form_z'), updated.pop('overall_day_z'))
        updated[ROTATED_SITE] = rotation.rotated({name:updated.pop(name) for name in rotation.shapes})
        updated = {name:jnp.asarray(value) for name, value in updated.items()}
        args = tuple({k:jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in blocks[name].items()}
                     for name in ('qualifying', 'laps', 'outcomes'))
        wrapped = weekend_model(rotation)
        phase('complete_coordinate_equivalence')
        before, a = log_density(model, args, {}, {k:jnp.asarray(v) for k, v in parameters.items()})
        after, b = log_density(wrapped, args, {}, updated)
        np.testing.assert_allclose(after, before+rotation.log_abs_det, **p['density_tolerance'])
        reports = []
        for name, old in a.items():
            if old['type'] not in ('sample', 'deterministic'):
                continue
            observed = old['type'] == 'sample' and old['is_observed']
            left = old['fn'].log_prob(old['value']) if observed else old['value']
            right = b[name]['fn'].log_prob(b[name]['value']) if observed else b[name]['value']
            np.testing.assert_allclose(right, left, err_msg=name,
                                      **p['density_tolerance' if observed else 'state_tolerance'])
            delta = np.abs(np.asarray(right)-np.asarray(left))
            reports.append(dict(site=name, observation_factor=observed,
                maximum_absolute_discrepancy=float(delta.max()) if delta.size else 0.))
        record.update(joint_density=dict(original=float(before), transformed=float(after),
            control_log_abs_det=rotation.log_abs_det), state_and_factor_checks=reports)
        del a, b
        phase('actual_collector_and_complete_gradient')
        z = unconstrain_fn(wrapped, args, {}, updated)
        postprocess = partial(constrain_fn, wrapped, args, {}, return_deterministic=True)
        direct = postprocess(z)
        collected = _collect_and_postprocess(postprocess, ('z',), ())((SimpleNamespace(z=z),))
        if set(direct) != set(collected):
            raise ValueError('Collector discarded a postprocessed field')
        for name in direct:
            np.testing.assert_array_equal(direct[name], collected[name])
        sites = inventory(collected)
        old_storage = json.loads((ROOT/p['storage_results'][case]).read_bytes())
        validate_generated(generated, expected, sites, old_storage['sites'], rotation.fingerprint())
        retained = accounting(sites, p['fit_settings'], p['retry'])
        if retained != old_storage['accounting']:
            raise ValueError('Original retained-posterior storage accounting changed')
        energy, gradient = jax.jit(jax.value_and_grad(lambda values:potential_energy(wrapped, args, {}, values)))(z)
        jax.block_until_ready((energy, gradient))
        if (not np.isfinite(energy) or set(gradient) != set(z)
                or any(g.shape != z[k].shape or not np.isfinite(g).all() for k, g in gradient.items())):
            raise ValueError('Incomplete or nonfinite production gradient')
        record.update(sites=sites, accounting=retained, control_rotation_sha256=rotation.fingerprint(),
            complete_collector_matches_postprocessing=True, original_collector_inventory_identical=True,
            potential_energy=float(energy), all_gradient_coordinates_finite=True,
            gradient_coordinate_count=sum(g.size for g in gradient.values()))
        checked(path)
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.monotonic()-started, process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_at=datetime.now(timezone.utc).isoformat())
        save(output/'result.json', record)


def execute(path):
    path = Path(path).resolve(); p = checked(path)
    output = ROOT/'outputs/championship/development'/p['id']; output.mkdir(exist_ok=False)
    rows = []
    for case in p['studies']:
        checked(path)
        scope = f"f1-stable-preflight-{case.replace('_', '-')}"
        log = output/f'{case}.output.txt'; resources = output/f'{case}.resources.txt'
        command = ['systemd-run', '--user', '--scope', '--slice=f1.slice', '--unit', scope, '-q',
            '-p', 'MemoryMax=3G', '-p', 'CPUQuota=200%', '--', '/usr/bin/nice', '-n', '19',
            '/usr/bin/time', '-v', '-o', str(resources), 'env', 'JAX_PLATFORMS=cpu', 'JAX_ENABLE_X64=false',
            'JAX_ENABLE_COMPILATION_CACHE=false', 'XLA_FLAGS=--xla_force_host_platform_device_count=1',
            'OPENBLAS_NUM_THREADS=2', 'OMP_NUM_THREADS=2', 'MKL_NUM_THREADS=2',
            sys.executable, '-u', str(Path(__file__).resolve()), 'child', '--protocol', str(path),
            '--case', case, '--output', str(output/case), '--scope', scope+'.scope']
        with log.open('x') as handle:
            child_process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        result_path = output/case/'result.json'
        result = json.loads(result_path.read_bytes()) if result_path.exists() else None
        success = child_process.returncode == 0 and result and result.get('state') == 'complete'
        row = dict(case=case, exit_code=child_process.returncode, state='complete' if success else 'failed',
            result_sha256=sha(result_path) if result_path.exists() else None, log_sha256=sha(log),
            resources_sha256=sha(resources) if resources.exists() else None, command=command)
        save(output/f'{case}.completion.json', row); rows.append(row)
        save(output/'progress.json', dict(studies=rows, all_processed=False, publication_allowed=False))
        print(json.dumps(row), flush=True)
    checked(path)
    result = dict(protocol=str(path.relative_to(ROOT)), protocol_sha256=sha(path),
        model_inputs_key=p['model_snapshot']['key'], studies=rows, all_processed=True,
        all_complete=all(r['state'] == 'complete' for r in rows),
        mcmc_executed=False, full_fit_memory_verified=False, publication_allowed=False)
    save(output/'summary.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['freeze', 'run', 'child'])
    parser.add_argument('--protocol', required=True)
    for name in ('case', 'output', 'scope'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.protocol)
    elif args.action == 'child':
        child(args.protocol, args.case, args.output, args.scope)
    else:
        result = execute(args.protocol)
        raise SystemExit(0 if result['all_complete'] else 1)
