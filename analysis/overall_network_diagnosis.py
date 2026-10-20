"""Post hoc movement and gradient diagnosis of every failed network attempt.

This reads unaccepted diagnostic archives. It neither fits nor accepts a model.
The frozen research checkout supplies the original numerical implementation.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFINITION = 'docs/overall_network_diagnosis.md'
REVIEW = 'outputs/championship/development/overall-continuation-review-v1/review.json'
PREFLIGHT = 'outputs/championship/development/overall-recovery-preflight-v2/real_network/result.json'
CHUNK_BYTES = 1024 * 1024


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def scalar_movement(values):
    """Describe each chain separately; these are not posterior estimates."""
    values = np.asarray(values)
    if (values.ndim != 2 or min(values.shape) < 2 or values.dtype.kind != 'f'
            or not np.isfinite(values).all()):
        raise ValueError('Need complete finite floating-point chains')
    rows = []
    for chain in values:
        x = chain.astype(np.float64)
        delta = np.diff(x)
        sd = float(x.std(ddof=1))
        jump = float(np.sqrt(np.mean(delta**2)))
        rows.append(dict(first=float(x[0]), last=float(x[-1]), minimum=float(x.min()),
            maximum=float(x.max()), mean=float(x.mean()), standard_deviation=sd,
            unique_values=int(len(np.unique(chain))), adjacent_unchanged=int(np.sum(delta == 0)),
            adjacent_rms=jump, adjacent_rms_over_sd=jump/sd if sd else None,
            quarter_means=[float(part.mean()) for part in np.array_split(x, 4) if len(part)]))
    return dict(shape=list(values.shape), dtype=values.dtype.str, chains=rows)


def endpoint_rows(stream, shape, dtype):
    """Read every byte, retaining the first/last draw of each chain only."""
    chains, draws = shape[:2]
    if chains < 2 or draws < 2:
        raise ValueError('Keep the original chain and draw axes')
    row_bytes = math.prod(shape[2:]) * dtype.itemsize
    wanted = [c*draws+d for c in range(chains) for d in (0, draws-1)]
    result, offset = [], 0
    def discard(count):
        while count:
            size = min(count, CHUNK_BYTES)
            if len(stream.read(size)) != size:
                raise ValueError('Truncated parameter array')
            count -= size
    for index in wanted:
        discard(index*row_bytes-offset)
        raw = stream.read(row_bytes)
        if len(raw) != row_bytes:
            raise ValueError('Truncated endpoint')
        point = np.frombuffer(raw, dtype=dtype).reshape(shape[2:]).copy()
        if not np.isfinite(point).all():
            raise ValueError('Nonfinite saved endpoint')
        result.append(point)
        offset = (index+1)*row_bytes
    discard(chains*draws*row_bytes-offset)
    if stream.read(1):
        raise ValueError('Unexpected trailing parameter bytes')
    return result


def read_attempt(path, expected_sha, axes, inventory, parameter_shapes, header):
    if sha(path) != expected_sha:
        raise ValueError('Original diagnostic archive changed')
    endpoints = [{} for _ in range(axes[0]*2)]
    scalars = {}
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        names = [m.filename[:-4] for m in members]
        if (any(not m.filename.endswith('.npy') for m in members)
                or len(set(names)) != len(names) or set(names) != set(inventory)):
            raise ValueError('Original retained inventory differs')
        for name, member in zip(names, members):
            with archive.open(member) as stream:
                shape, fortran, dtype, size = header(stream, member, axes)
                if (list(shape[2:]) != inventory[name]['shape']
                        or dtype.str != inventory[name]['dtype']):
                    raise ValueError('Original field shape or dtype differs')
                if name not in parameter_shapes:
                    continue
                if list(shape[2:]) != parameter_shapes[name] or fortran:
                    raise ValueError('Unsupported or changed parameter layout')
                if len(shape) == 2:
                    raw = stream.read(size)
                    if len(raw) != size or stream.read(1):
                        raise ValueError('Truncated scalar parameter')
                    values = np.frombuffer(raw, dtype=dtype).reshape(shape)
                    scalars[name] = scalar_movement(values)
                    points = [values[c:c+1,d:d+1].reshape(()).copy()
                              for c in range(axes[0]) for d in (0,axes[1]-1)]
                else:
                    points = endpoint_rows(stream, shape, dtype)
                for point, value in zip(endpoints, points):
                    point[name] = value
    if any(set(point) != set(parameter_shapes) for point in endpoints):
        raise ValueError('Incomplete endpoint parameter inventory')
    if sha(path) != expected_sha:
        raise ValueError('Diagnostic archive changed during reading')
    return scalars, endpoints


def gradient_report(potential, gradient):
    energy = np.asarray(potential)
    if energy.shape or not np.isfinite(energy):
        raise ValueError('Nonfinite endpoint potential')
    fields = {}
    for name, value in sorted(gradient.items()):
        x = np.asarray(value, dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError('Nonfinite endpoint gradient')
        fields[name] = dict(shape=list(x.shape), coordinates=int(x.size),
            l2=float(np.linalg.norm(x.ravel())), max_absolute=float(np.abs(x).max()) if x.size else 0.)
    return dict(potential=float(energy), potential_dtype=energy.dtype.str,
        potential_ulp=float(abs(np.spacing(energy))),
        gradient_l2=math.sqrt(sum(row['l2']**2 for row in fields.values())), gradient_fields=fields)


def run(reference_root, output, scope):
    reference_root, output = Path(reference_root).resolve(), Path(output).resolve()
    sources = ['analysis/overall_network_diagnosis.py', 'tests/test_overall_network_diagnosis.py', DEFINITION]
    source_hashes = {}
    for name in sources:
        raw = (ROOT/name).read_bytes()
        if subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=ROOT) != raw:
            raise ValueError('Commit the diagnostic definition and implementation first')
        source_hashes[name] = sha(ROOT/name)
    output.mkdir(parents=True, exist_ok=False)
    begun = time.monotonic()
    record = dict(state='running', purpose='post_hoc_failed_network_movement_and_gradients',
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,text=True).strip(),
        source_sha256=source_hashes, reference_root=str(reference_root),
        started_at=datetime.now(timezone.utc).isoformat(), attempts=[],
        mcmc_executed=False, scientific_acceptance_established=False, publication_allowed=False)
    def checkpoint(): save(output/'progress.json', record)
    checkpoint()
    try:
        # Import the checked, frozen numerical model, not the new production copy.
        sys.path.insert(0, str(reference_root))
        from analysis.overall_batched_rotation_audit import checked_protocol, check_scope
        from analysis.overall_posterior_archive import _header
        from analysis.overall_recovery_checks import generate_with_parameters
        from analysis.overall_control_rotation import ROTATED_SITE, build_rotation
        from analysis.overall_batched_rotation import prepare
        from analysis.overall_weekend_coordinates import HYPER_SITES, TOTAL, COMPLEMENT, transform, weekend_model
        import jax
        import jax.numpy as jnp
        from numpyro.infer.util import potential_energy, unconstrain_fn
        record['actual_scope'] = check_scope(scope)
        registered = checked_protocol(reference_root/'docs/overall_batched_rotation_audit_protocol.json')
        record['reference_commit'] = subprocess.check_output(['git','rev-parse','HEAD'],cwd=reference_root,text=True).strip()
        inputs = [REVIEW, PREFLIGHT, 'docs/overall_complete_profile_protocol.json']
        for name in inputs:
            if subprocess.check_output(['git','show',f'HEAD:{name}'],cwd=reference_root) != (reference_root/name).read_bytes():
                raise ValueError('Original evidence must remain committed')
        record['input_sha256'] = {name:sha(reference_root/name) for name in inputs}
        review, prior, protocol = [json.loads((reference_root/name).read_bytes()) for name in inputs]
        if (review['state'] != 'complete' or not review['records_consistent']
                or not review['all_new_reports_reproduced'] or review['combined_recorded_convergence']):
            raise ValueError('Require the complete independently reproduced failed profile')
        study = next(s for s in review['studies'] if s['case'] == 'real_network')
        attempts = study['continuation']['attempts']
        if len(attempts) != len(protocol['retry']) or any(a['recorded']['diagnostics']['converged'] for a in attempts):
            raise ValueError('Retain the entire exhausted attempt inventory')
        storage_name = protocol['storage_results']['real_network']
        record['input_sha256'][storage_name] = sha(reference_root/storage_name)
        inventory = json.loads((reference_root/storage_name).read_bytes())['sites']
        generated = generate_with_parameters('real_network', prior['generation_seed'])
        identity = ('arrays_key','parameters_key','truth_key')
        if any(generated[k] != prior[k] for k in identity):
            raise ValueError('Original generating observations and truth differ')
        record['original_input_identities'] = {k:generated[k] for k in identity}
        blocks, params = generated['blocks'], generated['parameters']
        original = build_rotation(blocks['laps'])
        if original.fingerprint() != prior['control_rotation_sha256']:
            raise ValueError('Original control coordinates changed')
        updated = dict(params)
        for old,new in HYPER_SITES.items(): updated[new] = updated.pop(old)
        updated[TOTAL],updated[COMPLEMENT] = transform(params['sd_driver_form'],params['overall_sd_day'],
            params['overall_shared_weekend'],updated.pop('driver_form_z'),updated.pop('overall_day_z'))
        updated[ROTATED_SITE] = original.rotated({name:updated.pop(name) for name in original.shapes})
        model = weekend_model(prepare(original))
        args = tuple({k:jnp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in blocks[name].items()}
                     for name in ('qualifying','laps','outcomes'))
        shapes = prior['transformed_parameter_shapes']
        if {name:list(np.shape(value)) for name,value in updated.items()} != shapes:
            raise ValueError('Original transformed parameter inventory differs')
        energy_gradient = jax.jit(jax.value_and_grad(lambda z: potential_energy(model,args,{},z)))
        def evaluate(point):
            values = {name:jnp.asarray(value) for name,value in point.items()}
            z = unconstrain_fn(model,args,{},values)
            nonfinite = [name for name,value in z.items() if not np.isfinite(np.asarray(value)).all()]
            if nonfinite:
                return dict(state='nonfinite_unconstrained_endpoint', parameters=nonfinite)
            potential,gradient = energy_gradient(z)
            return dict(state='complete', **gradient_report(potential,gradient))
        record['parameter_shapes'] = shapes
        record['runtime'] = dict(jax=jax.__version__, x64=jax.config.x64_enabled, numpy=np.__version__)
        record['generating_point'] = evaluate(updated)
        checkpoint()
        for a in attempts:
            settings = a['expected_settings']
            archive = reference_root/a['archive']
            row = dict(attempt=a['attempt'], state='reading', archive=a['archive'], archive_sha256=a['archive_sha256'],
                recorded_converged=False, recorded_tree_depth_limit_fraction=a['recorded']['diagnostics']['tree_depth_limit_fraction'],
                recorded_mean_accept_prob=a['recorded']['diagnostics']['mean_accept_prob'],
                sampler_extras_independently_verified=False, endpoints=[])
            record['attempts'].append(row); checkpoint()
            scalar, points = read_attempt(archive,a['archive_sha256'],(settings['chains'],settings['samples']),inventory,shapes,_header)
            row['scalar_movement'] = scalar
            row['state'] = 'evaluating'; checkpoint()
            for index,point in enumerate(points):
                row['endpoints'].append(dict(chain=index//2, draw=0 if index%2==0 else settings['samples']-1, **evaluate(point)))
                checkpoint()
            row['state'] = 'complete'; checkpoint()
        if any(sha(reference_root/name) != value for name,value in record['input_sha256'].items()):
            raise ValueError('Original evidence changed during diagnosis')
        checked_protocol(reference_root/'docs/overall_batched_rotation_audit_protocol.json')
        if any(sha(ROOT/name) != value for name,value in source_hashes.items()):
            raise ValueError('Diagnostic source changed during execution')
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed_exception',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.monotonic()-begun,finished_at=datetime.now(timezone.utc).isoformat(),
                      process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(output/'result.json',record)
    return record


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--reference-root',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--scope',required=True)
    args=parser.parse_args()
    result=run(args.reference_root,args.output,args.scope)
    print(json.dumps({k:result[k] for k in ('state','seconds','process_max_rss_kib')}))
