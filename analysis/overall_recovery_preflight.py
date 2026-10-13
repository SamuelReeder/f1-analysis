"""Complete fixed-design density/gradient preflight, with no MCMC or retry."""
import argparse
from datetime import datetime, timezone
import hashlib
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


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


SELF_SHA = sha(__file__)


def block_metadata(blocks):
    """Describe NumPy and JAX arrays alike; never serialize response values."""
    import numpy as np
    def describe(value):
        if hasattr(value, 'shape') and hasattr(value, 'dtype'):
            array = np.asarray(value)
            return dict(shape=list(array.shape), dtype=str(array.dtype), bytes=int(array.nbytes))
        if value is None or type(value) in (int, float, bool, str):
            return value
        raise TypeError(f'Unsupported input metadata type: {type(value).__name__}')
    return {name: {key: describe(value) for key, value in block.items()} for name, block in blocks.items()}


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def checked_protocol(path):
    path = Path(path).resolve()
    name = path.relative_to(ROOT).as_posix()
    content = path.read_bytes()
    if subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != content:
        raise ValueError('Commit the fixed preflight protocol before execution')
    protocol = json.loads(content)
    if (protocol.get('purpose') != 'complete_design_density_gradient_preflight'
            or protocol.get('status') != 'frozen' or protocol.get('mcmc_allowed') is not False
            or protocol.get('publication_allowed') is not False
            or not re.fullmatch('[a-z][a-z0-9-]*', protocol.get('id', ''))):
        raise ValueError('Not the fixed computation-only preflight')
    if (protocol['resources'] != dict(slice='f1.slice', MemoryMax='3G', CPUQuota='200%', Nice=19)
            or protocol['density_tolerance'] != dict(atol=.005, rtol=.000002)
            or protocol['state_tolerance'] != dict(atol=.000003, rtol=.000003)):
        raise ValueError('Retain the defined resources and existing tolerances')
    cases, seeds = [], []
    for row in protocol['studies']:
        if (set(row) != {'case', 'seed'} or not re.fullmatch('[a-z_]+', row['case'])
                or type(row['seed']) is not int or not 0 <= row['seed'] < 2**32):
            raise ValueError('Invalid fixed generation inventory')
        cases.append(row['case']); seeds.append(row['seed'])
    if not cases or len(set(cases)) != len(cases) or len(set(seeds)) != len(seeds):
        raise ValueError('Repeated or empty generation inventory')
    if protocol['source_sha256'].get('analysis/overall_recovery_preflight.py') != SELF_SHA:
        raise ValueError('Loaded preflight implementation differs from registration')
    sources = dict(protocol['model_source']['inputs'])
    source_key = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    if source_key != protocol['model_source']['key']:
        raise ValueError('Invalid model source identity')
    sources.update(protocol['source_sha256'])
    for filename, fingerprint in sources.items():
        target = ROOT / filename
        if sha(target) != fingerprint:
            raise ValueError(f'Registered source changed: {filename}')
        if subprocess.check_output(['git', 'show', f'HEAD:{filename}'], cwd=ROOT) != target.read_bytes():
            raise ValueError(f'Registered source must be committed: {filename}')
    return protocol


def child(protocol_path, case, output, scope):
    protocol = checked_protocol(protocol_path)
    spec = next(r for r in protocol['studies'] if r['case'] == case)
    output = Path(output)
    output.mkdir(exist_ok=False)
    started = time.monotonic()
    record = dict(case=case, generation_seed=spec['seed'], state='running',
        started_at=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(protocol_path),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        stage='scope', phases=[], mcmc_run=False, full_recovery_complete=False,
        sbc_complete=False, publication_allowed=False)
    phase_start = started

    def phase(name):
        nonlocal phase_start
        now = time.monotonic()
        record['phases'].append(dict(stage=record['stage'], seconds=now-phase_start))
        phase_start = now
        record['stage'] = name
        save(output / 'progress.json', record)

    try:
        properties = subprocess.check_output(['systemctl', '--user', 'show', scope,
            '-p', 'Slice', '-p', 'MemoryMax', '-p', 'CPUQuotaPerSecUSec'], text=True)
        actual = dict(line.split('=', 1) for line in properties.splitlines())
        if actual != dict(Slice='f1.slice', MemoryMax=str(3 * 1024**3), CPUQuotaPerSecUSec='2s'):
            raise ValueError('The numerical child is not in its required capped scope')
        if os.getpriority(os.PRIO_PROCESS, 0) != 19:
            raise ValueError('The numerical child lacks the registered priority')
        record['actual_scope'] = dict(actual, Nice=19)
        phase('imports')
        import jax
        import jax.numpy as jnp
        import numpy as np
        import numpyro
        from numpyro.infer.util import log_density, potential_energy, unconstrain_fn
        from analysis.overall_recovery_checks import generate_with_parameters
        from analysis.overall_control_rotation import ROTATED_SITE, build_rotation
        from analysis.overall_weekend_coordinates import HYPER_SITES, TOTAL, COMPLEMENT, transform, weekend_model
        from f1rank.overall.model import model
        from f1rank.overall.recovery import SYNTHETIC_CASES
        if [r['case'] for r in protocol['studies']] != list(SYNTHETIC_CASES):
            raise ValueError('Preflight must preserve the complete original design inventory')
        record['runtime'] = dict(python=sys.version, numpy=np.__version__, jax=jax.__version__,
            numpyro=numpyro.__version__, backend=jax.default_backend(), x64=jax.config.x64_enabled)
        phase('generation')
        generated = generate_with_parameters(case, spec['seed'])
        blocks, catalog, parameters = (generated[k] for k in ('blocks', 'catalog', 'parameters'))
        record.update(arrays_key=generated['arrays_key'], parameters_key=generated['parameters_key'],
            truth_key=generated['truth_key'], role=generated['role'], field_size=len(catalog['field']),
            blocks=block_metadata(blocks),
            original_parameter_shapes={k: list(v.shape) for k, v in parameters.items()},
            original_parameter_coordinates=sum(v.size for v in parameters.values()))
        phase('coordinate_map')
        rotation = build_rotation(blocks['laps'])
        updated = dict(parameters)
        for original, new in HYPER_SITES.items():
            updated[new] = updated.pop(original)
        updated[TOTAL], updated[COMPLEMENT] = transform(parameters['sd_driver_form'], parameters['overall_sd_day'],
            parameters['overall_shared_weekend'], updated.pop('driver_form_z'), updated.pop('overall_day_z'))
        updated[ROTATED_SITE] = rotation.rotated({name: updated.pop(name) for name in rotation.shapes})
        updated = {k: jnp.asarray(v) for k, v in updated.items()}
        model_args = tuple({k: jnp.asarray(v) if isinstance(v, np.ndarray) else v for k, v in blocks[name].items()}
                           for name in ('qualifying', 'laps', 'outcomes'))
        wrapped = weekend_model(rotation)
        record.update(control_rotation_sha256=rotation.fingerprint(),
            transformed_parameter_shapes={k: list(v.shape) for k, v in updated.items()},
            transformed_parameter_coordinates=sum(v.size for v in updated.values()))
        if record['original_parameter_coordinates'] != record['transformed_parameter_coordinates']:
            raise ValueError('The coordinate transformation discarded a direction')
        phase('density_equivalence')
        before, original_trace = log_density(model, model_args, {}, {k: jnp.asarray(v) for k, v in parameters.items()})
        after, new_trace = log_density(wrapped, model_args, {}, updated)
        record['joint_density'] = dict(original=float(before), transformed=float(after),
            control_log_abs_det=rotation.log_abs_det,
            absolute_discrepancy=abs(float(after)-float(before)-rotation.log_abs_det))
        np.testing.assert_allclose(after, before + rotation.log_abs_det, **protocol['density_tolerance'])
        checked = []
        for name, old in original_trace.items():
            if old['type'] not in ('sample', 'deterministic'):
                continue
            new = new_trace[name]
            observed = old['type'] == 'sample' and old['is_observed']
            old_value = old['fn'].log_prob(old['value']) if observed else old['value']
            new_value = new['fn'].log_prob(new['value']) if observed else new['value']
            tolerance = protocol['density_tolerance'] if observed else protocol['state_tolerance']
            np.testing.assert_allclose(new_value, old_value, err_msg=name, **tolerance)
            delta = np.abs(np.asarray(new_value) - np.asarray(old_value))
            checked.append(dict(site=name, observation_factor=observed, shape=list(delta.shape),
                                max_absolute_discrepancy=float(delta.max()) if delta.size else 0.))
        record['all_original_state_and_factor_checks'] = checked
        del original_trace, new_trace
        phase('unconstrained_coordinates')
        unconstrained = unconstrain_fn(wrapped, model_args, {}, updated)
        if set(unconstrained) != set(updated) or not all(np.isfinite(v).all() for v in unconstrained.values()):
            raise ValueError('Incomplete or nonfinite unconstrained parameters')
        record['unconstrained_coordinates'] = sum(v.size for v in unconstrained.values())
        phase('compiled_potential_and_gradient')
        compute = jax.jit(jax.value_and_grad(lambda z: potential_energy(wrapped, model_args, {}, z)))
        energy, gradient = compute(unconstrained)
        jax.block_until_ready((energy, gradient))
        if (not np.isfinite(energy) or set(gradient) != set(unconstrained)
                or any(g.shape != unconstrained[k].shape or not np.isfinite(g).all() for k, g in gradient.items())):
            raise ValueError('Incomplete or nonfinite full gradient')
        record.update(potential_energy=float(energy), all_gradient_coordinates_finite=True,
            gradient_coordinate_count=sum(g.size for g in gradient.values()))
        phase('source_recheck')
        if checked_protocol(protocol_path) != protocol:
            raise ValueError('Preflight protocol changed during calculation')
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record['phases'].append(dict(stage=record['stage'], seconds=time.monotonic()-phase_start))
        record.update(finished_at=datetime.now(timezone.utc).isoformat(), seconds=time.monotonic()-started,
                      process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(output / 'result.json', record)


def execute(protocol_path):
    protocol_path = Path(protocol_path).resolve()
    protocol = checked_protocol(protocol_path)
    output = ROOT / 'outputs/championship/development' / protocol['id']
    output.mkdir(exist_ok=False)
    rows = []
    for spec in protocol['studies']:
        checked_protocol(protocol_path)
        case = spec['case']
        scope = f"f1-{protocol['id']}-{case.replace('_', '-')}.scope"
        log = output / f'{case}.output.txt'
        resource_log = output / f'{case}.resources.txt'
        command = ['systemd-run', '--user', '--scope', '--unit', scope, '--slice=f1.slice', '-q',
            '-p', 'MemoryMax=3G', '-p', 'CPUQuota=200%', '--', '/usr/bin/time', '-v', '-o', str(resource_log),
            '/usr/bin/nice', '-n', '19', sys.executable, str(Path(__file__).resolve()),
            '--protocol', str(protocol_path), '--case', case, '--output', str(output / case), '--scope-name', scope]
        with log.open('x') as handle:
            process = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
        result_path = output / case / 'result.json'
        child_result = json.loads(result_path.read_text()) if result_path.exists() else None
        row = dict(case=case, generation_seed=spec['seed'], process_exit_code=process.returncode,
            state='complete' if process.returncode == 0 and child_result and child_result['state'] == 'complete' else 'failed',
            child_result_present=child_result is not None, command=command,
            output_sha256=sha(log), resources_sha256=sha(resource_log) if resource_log.exists() else None,
            result_sha256=sha(result_path) if result_path.exists() else None,
            progress_sha256=sha(output / case / 'progress.json') if (output / case / 'progress.json').exists() else None)
        save(output / f'{case}.completion.json', row)
        rows.append(row)
        print(json.dumps(dict(case=case, state=row['state'], exit_code=process.returncode)), flush=True)
    checked_protocol(protocol_path)
    summary = dict(protocol=str(protocol_path.relative_to(ROOT)), protocol_sha256=sha(protocol_path),
        studies=rows, all_processed=True, all_complete=all(r['state'] == 'complete' for r in rows),
        mcmc_run=False, full_recovery_complete=False, sbc_complete=False, publication_allowed=False)
    save(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--case')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--scope-name')
    args = parser.parse_args()
    if args.case:
        if args.output is None or args.scope_name is None:
            parser.error('Child requires output and its scope identity')
        child(args.protocol, args.case, args.output, args.scope_name)
    else:
        raise SystemExit(0 if execute(args.protocol)['all_complete'] else 1)
