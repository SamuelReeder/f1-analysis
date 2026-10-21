"""Frozen full-model CPU/GPU value, gradient and state comparison; no fitting."""
import argparse
from datetime import datetime, timezone
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REGISTRATION = 'docs/overall_stable_profile_protocol.json'
PREFLIGHT = 'docs/overall_stable_preflight_protocol.json'
REVIEW = 'outputs/championship/development/overall-stable-profile-review-v1/review.json'
AUDIT = 'outputs/championship/development/overall-stable-profile-diagnostics-v1/audit.json'
DEFINITION = 'docs/overall_backend_comparison.md'
TEST = 'tests/test_overall_backend_compare.py'
ENV = dict(JAX_ENABLE_X64='false', JAX_ENABLE_COMPILATION_CACHE='false',
           XLA_FLAGS='--xla_force_host_platform_device_count=1',
           OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
ENVIRONMENTS = dict(cpu=dict(ENV, JAX_PLATFORMS='cpu'),
    gpu=dict(ENV, JAX_PLATFORMS='cuda', XLA_PYTHON_CLIENT_ALLOCATOR='bfc',
             XLA_CLIENT_MEM_FRACTION='0.125', XLA_PYTHON_CLIENT_PREALLOCATE='false'))
POINT_POLICY = 'generating_point_then_first_and_last_draw_of_every_chain_of_every_recorded_attempt'
COORDINATE_POLICY = 'prepare_once_on_cpu_in_float32_then_reuse_exact_arrays_on_both_backends'
RESOURCES = dict(slice='f1.slice', MemoryMax='3G', CPUQuota='200%', Nice=19)
GPU_ADMISSION = dict(pool_fraction=.125, max_share_of_observed_free_memory=.5)


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def committed(root, name):
    path = root/name
    if subprocess.check_output(['git', 'show', f'HEAD:{name}'], cwd=root) != path.read_bytes():
        raise ValueError(f'Uncommitted input: {path}')
    return sha(path)


def sources():
    paths = [*(ROOT/'analysis').glob('*.py'), *(ROOT/'f1rank').rglob('*.py'),
             *(ROOT/'extract').rglob('*.py'), ROOT/TEST, ROOT/DEFINITION,
             ROOT/'docs/overall_candidate.md']
    for folder in ('processed', 'overrides', 'reference'):
        paths.extend((ROOT/'data'/folder).glob('*.*'))
    return sorted(str(p.relative_to(ROOT)) for p in paths)


def original_studies(parent, reviewed):
    if [s['case'] for s in reviewed['studies']] != parent['studies']:
        raise ValueError('Preserve every original case')
    studies = []
    for study in reviewed['studies']:
        case = study['case']
        if not study['attempts']:
            raise ValueError('Original case has no retained attempt inventory')
        attempts = []
        for a in study['attempts']:
            if a['issues'] or not a['archive_present']:
                raise ValueError('Original attempt inventory is incomplete')
            settings = a['expected_settings']
            attempts.append(dict(attempt=a['attempt'], archive=a['archive'],
                archive_sha256=a['archive_sha256'], chains=settings['chains'], samples=settings['samples']))
        studies.append(dict(case=case, seed=parent['synthetic'][case]['seed'],
            preflight=f'outputs/championship/development/overall-stable-preflight-v1/{case}/result.json',
            attempts=attempts))
    return studies


def freeze(destination, reference):
    reference = Path(reference).resolve()
    parent, preflight, reviewed, audit = [read(reference/p) for p in (REGISTRATION, PREFLIGHT, REVIEW, AUDIT)]
    if not (reviewed['records_consistent'] and audit['all_processed']
            and audit['all_recorded_reports_reproduced'] and audit['all_sampler_extras_recounted']
            and not audit['issues']):
        raise ValueError('Require complete verified original computational evidence')
    original = preflight['model_snapshot']['inputs']
    for name, expected in original.items():
        if committed(reference, name) != expected or committed(ROOT, name) != expected:
            raise ValueError('The original model or data changed')
    evidence = [REGISTRATION, PREFLIGHT, REVIEW, AUDIT]
    studies = original_studies(parent, reviewed)
    evidence.extend(s['preflight'] for s in studies)
    runtime = {}
    for backend, directory in [('cpu', '.venv'), ('gpu', '.venv-gpu')]:
        runtime[backend] = {}
        for name in parent['runtime_source_sha256']:
            path = Path(name.replace('/.venv/', f'/{directory}/'))
            runtime[backend][str(path)] = sha(path)
        if list(runtime[backend].values()) != list(parent['runtime_source_sha256'].values()):
            raise ValueError('Registered numerical Python sources differ')
    p = dict(id='overall-backend-comparison-v1', purpose='fixed_full_model_backend_comparison',
        registered_at=datetime.now(timezone.utc).isoformat(), reference_root=str(reference),
        source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_sha256={n: committed(ROOT, n) for n in sources()},
        reference_sha256={n: committed(reference, n) for n in evidence},
        original_model_inputs=original, runtime_source_sha256=runtime, studies=studies,
        point_policy=POINT_POLICY, coordinate_policy=COORDINATE_POLICY,
        environments=ENVIRONMENTS, resources=RESOURCES, gpu_admission=GPU_ADMISSION,
        value_and_gradient_tolerance=preflight['density_tolerance'], state_tolerance=preflight['state_tolerance'],
        sampling_allowed=False, scientific_acceptance_allowed=False, publication_allowed=False)
    with Path(destination).open('x') as f:
        json.dump(p, f, indent=2); f.write('\n')


def checked(path):
    path = Path(path).resolve()
    committed(ROOT, str(path.relative_to(ROOT)))
    p = read(path)
    if (p['purpose'] != 'fixed_full_model_backend_comparison' or p['environments'] != ENVIRONMENTS
            or p['sampling_allowed'] or p['publication_allowed'] or p['scientific_acceptance_allowed']
            or p['point_policy'] != POINT_POLICY or p['coordinate_policy'] != COORDINATE_POLICY
            or p['resources'] != RESOURCES or p['gpu_admission'] != GPU_ADMISSION
            or set(p['source_sha256']) != set(sources())):
        raise ValueError('Changed comparison definition or source inventory')
    for name, expected in p['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise ValueError(f'Changed comparison input: {name}')
    reference = Path(p['reference_root'])
    parent, preflight, reviewed = [read(reference/n) for n in (REGISTRATION, PREFLIGHT, REVIEW)]
    if (p['studies'] != original_studies(parent, reviewed)
            or p['original_model_inputs'] != preflight['model_snapshot']['inputs']
            or p['value_and_gradient_tolerance'] != preflight['density_tolerance']
            or p['state_tolerance'] != preflight['state_tolerance']
            or set(p['reference_sha256']) != {REGISTRATION, PREFLIGHT, REVIEW, AUDIT,
                                             *(s['preflight'] for s in p['studies'])}):
        raise ValueError('Original inventory or comparison rules changed')
    for name, expected in p['reference_sha256'].items():
        if sha(reference/name) != expected:
            raise ValueError(f'Changed original evidence: {name}')
    for name, expected in p['original_model_inputs'].items():
        if sha(reference/name) != expected:
            raise ValueError(f'Changed original source/data: {name}')
    for files in p['runtime_source_sha256'].values():
        if any(sha(name) != expected for name, expected in files.items()):
            raise ValueError('Registered numerical runtime source changed')
    return p


def point_labels(study):
    return [dict(kind='generating')] + [dict(kind='saved_endpoint', attempt=a['attempt'], chain=c, draw=d)
        for a in study['attempts'] for c in range(a['chains']) for d in (0, a['samples']-1)]


def check_environment(p, backend, scope):
    from analysis.overall_complete_profile_diagnostics import actual_scope
    actual = actual_scope(scope)
    for name, value in p['environments'][backend].items():
        if os.environ.get(name) != value:
            raise ValueError(f'Wrong explicit backend environment: {name}')
    if backend == 'gpu' and os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION'):
        raise ValueError('Conflicting deprecated device allocation setting')
    own = Path('/sys/fs/cgroup')/actual['ControlGroup'].lstrip('/')
    for sibling in own.parent.iterdir():
        if sibling.is_dir() and sibling != own and (sibling/'cgroup.events').exists():
            if dict(line.split() for line in (sibling/'cgroup.events').read_text().splitlines()).get('populated') != '0':
                raise ValueError('Other heavy work is active')
    if backend == 'gpu':
        import csv
        query = ['nvidia-smi', '--query-gpu=name,memory.total,memory.free,driver_version', '--format=csv,noheader,nounits']
        rows = list(csv.reader(subprocess.check_output(query, text=True).splitlines()))
        if len(rows) != 1:
            raise ValueError('This registration requires the observed single-device inventory')
        total, free = float(rows[0][1]), float(rows[0][2])
        admission = p['gpu_admission']
        if total*admission['pool_fraction'] > free*admission['max_share_of_observed_free_memory']:
            raise ValueError('Insufficient currently free device memory for the registered pool')
        actual['device_admission_observation'] = rows
    import jax
    if (jax.default_backend() not in (('gpu', 'cuda') if backend == 'gpu' else ('cpu',))
            or jax.config.x64_enabled or len(jax.local_devices()) != 1):
        raise ValueError('Wrong backend, precision or device count')
    actual['backend'] = jax.default_backend()
    actual['device_kinds'] = [d.device_kind for d in jax.local_devices()]
    return actual


def cache_path(path):
    path = Path(path).resolve()
    path.relative_to(ROOT/'outputs/championship/cache')
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError('Never replace a prior numeric artifact')
    return path


def prepare(p, study, cache, record):
    import jax.numpy as jnp
    import numpy as np
    from numpyro.infer.util import unconstrain_fn
    from analysis.overall_network_precision import pack_blocks, unpack_blocks
    from analysis.overall_network_diagnosis import read_attempt
    from analysis.overall_posterior_archive import _header
    from analysis.overall_recovery_checks import generate_with_parameters
    from f1rank.overall.batched_rotation import prepare as batch_rotation
    from f1rank.overall.control_rotation import build_rotation, ROTATED_SITE
    from f1rank.overall.fitting import arrays_digest
    from f1rank.overall.weekend_coordinates import HYPER_SITES, TOTAL, COMPLEMENT, transform, weekend_model
    reference = Path(p['reference_root'])
    expected = read(reference/study['preflight'])
    generated = generate_with_parameters(study['case'], study['seed'])
    if any(generated[n] != expected[n] for n in ('arrays_key', 'parameters_key', 'truth_key')):
        raise ValueError('Original synthetic problem changed')
    blocks, parameters = generated['blocks'], generated['parameters']
    rotation = batch_rotation(build_rotation(blocks['laps']))
    if rotation.fingerprint() != expected['control_rotation_sha256']:
        raise ValueError('Original rotation changed')
    model = weekend_model(rotation)
    args = tuple({k:jnp.asarray(v) if isinstance(v, np.ndarray) else v for k,v in blocks[name].items()}
                 for name in ('qualifying', 'laps', 'outcomes'))
    updated = dict(parameters)
    for old, new in HYPER_SITES.items(): updated[new] = updated.pop(old)
    updated[TOTAL], updated[COMPLEMENT] = transform(parameters['sd_driver_form'], parameters['overall_sd_day'],
        parameters['overall_shared_weekend'], updated.pop('driver_form_z'), updated.pop('overall_day_z'))
    updated[ROTATED_SITE] = rotation.rotated({n:updated.pop(n) for n in rotation.shapes})
    points = []
    def retain(point):
        z = {n:np.asarray(v) for n,v in unconstrain_fn(model, args, {}, {n:jnp.asarray(v) for n,v in point.items()}).items()}
        if any(v.dtype != np.dtype('float32') or not np.isfinite(v).all() for v in z.values()):
            raise ValueError('Complete original finite float32 coordinates required')
        if points and {n:v.shape for n,v in z.items()} != {n:v.shape for n,v in points[0].items()}:
            raise ValueError('Coordinate inventory changed between points')
        points.append(z)
    retain(updated)
    shapes = {n:list(v.shape) for n,v in points[0].items()}
    if sum(v.size for v in points[0].values()) != expected['gradient_coordinate_count']:
        raise ValueError('Incomplete registered gradient coordinates')
    for attempt in study['attempts']:
        _, endpoints = read_attempt(reference/attempt['archive'], attempt['archive_sha256'],
            (attempt['chains'], attempt['samples']), expected['sites'], shapes, _header)
        for point in endpoints: retain(point)
    labels = point_labels(study)
    if len(labels) != len(points): raise ValueError('Missing or additional declared points')
    arrays, scalars = pack_blocks(blocks)
    if arrays_digest(unpack_blocks(arrays, scalars)) != generated['arrays_key']:
        raise ValueError('Serialized observations changed')
    coordinates = {n:np.stack([z[n] for z in points]) for n in shapes}
    arrays.update({f'z/{n}':v for n,v in coordinates.items()})
    with cache.open('xb') as f: np.savez_compressed(f, **arrays)
    record.update(cache=str(cache.relative_to(ROOT)), cache_sha256=sha(cache), block_scalars=scalars,
        point_labels=labels, parameter_shapes=shapes, collector_inventory=expected['sites'],
        arrays_key=generated['arrays_key'], coordinates_key=arrays_digest({'coordinates':coordinates}),
        control_rotation_sha256=rotation.fingerprint())


def load_prepared(p, study, path, protocol_sha):
    import numpy as np
    from analysis.overall_network_precision import unpack_blocks
    from f1rank.overall.fitting import arrays_digest
    path = Path(path).resolve()
    committed(ROOT, str(path.relative_to(ROOT)))
    r = read(path)
    original = read(Path(p['reference_root'])/study['preflight'])
    if (r['state'] != 'complete' or r['operation'] != 'prepare' or r['case'] != study['case']
            or r['protocol_sha256'] != protocol_sha or r['point_labels'] != point_labels(study)
            or r['arrays_key'] != original['arrays_key']
            or r['collector_inventory'] != original['sites']
            or r['control_rotation_sha256'] != original['control_rotation_sha256']):
        raise ValueError('Incomplete or mismatched fixed inputs')
    archive = ROOT/r['cache']
    archive.resolve().relative_to(ROOT/'outputs/championship/cache')
    if sha(archive) != r['cache_sha256']: raise ValueError('Prepared arrays changed')
    with np.load(archive, allow_pickle=False) as saved: arrays = {n:saved[n] for n in saved.files}
    blocks = unpack_blocks(arrays, r['block_scalars'])
    coordinates = {n:arrays[f'z/{n}'] for n in r['parameter_shapes']}
    if (arrays_digest(blocks) != r['arrays_key']
            or arrays_digest({'coordinates':coordinates}) != r['coordinates_key']
            or any(v.shape != (len(r['point_labels']), *r['parameter_shapes'][n])
                   or v.dtype != np.dtype('float32') or not np.isfinite(v).all() for n,v in coordinates.items())
            or set(arrays) != {f'block/{b}/{n}' for b,values in blocks.items() for n,v in values.items()
                              if hasattr(v, 'dtype')} | {f'z/{n}' for n in coordinates}):
        raise ValueError('Incomplete or changed prepared inventory')
    return r, blocks, coordinates


def evaluate(p, study, prepared_path, cache, record):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from numpyro.infer.util import constrain_fn, potential_energy
    from f1rank.overall.batched_rotation import prepare as batch_rotation
    from f1rank.overall.control_rotation import build_rotation
    from f1rank.overall.weekend_coordinates import weekend_model
    prepared, blocks, coordinates = load_prepared(p, study, prepared_path, record['protocol_sha256'])
    rotation = batch_rotation(build_rotation(blocks['laps']))
    if rotation.fingerprint() != prepared['control_rotation_sha256']:
        raise ValueError('Coordinate map differs across backends')
    model = weekend_model(rotation)
    args = tuple({k:jnp.asarray(v) if isinstance(v, np.ndarray) else v for k,v in blocks[n].items()}
                 for n in ('qualifying', 'laps', 'outcomes'))
    postprocess = partial(constrain_fn, model, args, {}, return_deterministic=True)
    value_gradient = jax.value_and_grad(lambda z: potential_energy(model, args, {}, z))
    def kernel(z):
        value, gradient = value_gradient(z)
        return value, gradient, postprocess(z)
    point = lambda i: {n:jnp.asarray(v[i]) for n,v in coordinates.items()}
    start = time.monotonic(); lowered = jax.jit(kernel).lower(point(0)); lower_seconds = time.monotonic()-start
    start = time.monotonic(); compiled = lowered.compile(); compile_seconds = time.monotonic()-start
    records, times, transfers = [], [], []
    for i in range(len(prepared['point_labels'])):
        z = point(i); jax.block_until_ready(z)
        start = time.monotonic(); value, gradient, state = jax.block_until_ready(compiled(z)); times.append(time.monotonic()-start)
        start = time.monotonic()
        row = dict(potential=np.asarray(value), **{f'gradient/{n}':np.asarray(v) for n,v in gradient.items()},
                   **{f'state/{n}':np.asarray(v) for n,v in state.items()})
        transfers.append(time.monotonic()-start)
        if set(gradient) != set(coordinates) or set(state) != set(prepared['collector_inventory']):
            raise ValueError('Incomplete gradient or collector output')
        if any(v.dtype.kind not in 'biuf' or not np.isfinite(v).all() for v in row.values()):
            raise ValueError('Nonfinite or unsupported full-model result')
        for n, v in state.items():
            expected = prepared['collector_inventory'][n]
            if list(v.shape) != expected['shape'] or np.asarray(v).dtype.str != expected['dtype']:
                raise ValueError('Complete retained site shape or dtype changed')
        records.append(row)
    arrays = {n:np.stack([row[n] for row in records]) for n in records[0]}
    start = time.monotonic()
    with cache.open('xb') as f: np.savez_compressed(f, **arrays)
    record.update(prepared=str(Path(prepared_path).resolve().relative_to(ROOT)), prepared_sha256=sha(prepared_path),
        cache=str(cache.relative_to(ROOT)), cache_sha256=sha(cache), point_labels=prepared['point_labels'],
        field_shapes={n:list(v.shape) for n,v in arrays.items()}, field_dtypes={n:v.dtype.str for n,v in arrays.items()},
        lowering_seconds=lower_seconds, compile_seconds=compile_seconds, evaluation_seconds=times,
        transfer_seconds=transfers, archive_seconds=time.monotonic()-start)


def compare_fields(left, right, tolerances):
    import numpy as np
    if not left or set(left) != set(right): raise ValueError('Complete matching output inventories required')
    reports = {}
    for n in left:
        a, b = np.asarray(left[n]), np.asarray(right[n])
        if a.shape != b.shape or a.dtype != b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Compare only complete finite fields with matching shapes and dtypes')
        tolerance = tolerances['state'] if n.startswith('state/') else tolerances['value_and_gradient']
        delta = np.abs(b.astype(np.float64)-a.astype(np.float64))
        limit = tolerance['atol'] + tolerance['rtol']*np.abs(a.astype(np.float64))
        passed = delta <= limit if a.dtype.kind == 'f' else a == b
        reports[n] = dict(shape=list(a.shape), coordinates=int(a.size), passed=bool(passed.all()),
            maximum_absolute_difference=float(delta.max()) if a.size else 0.,
            maximum_budget_fraction=float((delta/limit).max()) if a.size else 0.,
            failing_indices=np.argwhere(~passed).tolist())
    return reports


def compare(p, study, paths, record):
    import numpy as np
    arrays, saved = [], []
    for backend, path in zip(('cpu', 'gpu'), paths, strict=True):
        path = Path(path).resolve(); committed(ROOT, str(path.relative_to(ROOT)))
        r = read(path)
        if (r['state'] != 'complete' or r['case'] != study['case'] or r['operation'] != 'evaluate'
                or r['backend'] != backend or r['protocol_sha256'] != record['protocol_sha256']
                or r['point_labels'] != point_labels(study)):
            raise ValueError('Missing or mismatched complete backend evaluation')
        archive = ROOT/r['cache']; archive.resolve().relative_to(ROOT/'outputs/championship/cache')
        if sha(archive) != r['cache_sha256']: raise ValueError('Backend output bytes changed')
        with np.load(archive, allow_pickle=False) as a: arrays.append({n:a[n] for n in a.files})
        if ({n:list(v.shape) for n,v in arrays[-1].items()} != r['field_shapes']
                or {n:v.dtype.str for n,v in arrays[-1].items()} != r['field_dtypes']):
            raise ValueError('Backend output inventory differs')
        saved.append(r)
    if saved[0]['prepared_sha256'] != saved[1]['prepared_sha256']:
        raise ValueError('Backend arms used different fixed inputs')
    reports = compare_fields(*arrays, dict(state=p['state_tolerance'], value_and_gradient=p['value_and_gradient_tolerance']))
    record.update(point_labels=point_labels(study), fields=reports,
        within_registered_budgets=all(r['passed'] for r in reports.values()),
        evaluations=[dict(path=str(Path(path).resolve().relative_to(ROOT)), sha256=sha(path)) for path in paths])


def run(args):
    p = checked(args.protocol)
    study = next(s for s in p['studies'] if s['case'] == args.case)
    output = Path(args.output); output.mkdir(parents=True, exist_ok=False)
    record = dict(purpose=p['purpose'], operation=args.operation, case=args.case, backend=args.backend,
        state='running', protocol_sha256=sha(args.protocol), started_at=datetime.now(timezone.utc).isoformat(),
        sampling_executed=False, publication_allowed=False, scientific_acceptance_established=False)
    start = time.monotonic(); save(output/'result.json', record)
    try:
        if args.operation in ('prepare', 'compare') and args.backend != 'cpu':
            raise ValueError('Prepare and compare under the registered CPU environment')
        record['actual_scope'] = check_environment(p, args.backend, args.scope)
        if args.operation == 'prepare': prepare(p, study, cache_path(args.cache), record)
        elif args.operation == 'evaluate': evaluate(p, study, args.prepared, cache_path(args.cache), record)
        else: compare(p, study, (args.cpu, args.gpu), record)
        checked(args.protocol)
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed', error=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.monotonic()-start, process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_at=datetime.now(timezone.utc).isoformat())
        save(output/'result.json', record)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('operation', choices=('freeze', 'prepare', 'evaluate', 'compare'))
    parser.add_argument('--protocol', required=True); parser.add_argument('--reference')
    parser.add_argument('--case'); parser.add_argument('--backend', choices=('cpu', 'gpu'), default='cpu')
    parser.add_argument('--scope'); parser.add_argument('--output'); parser.add_argument('--cache')
    parser.add_argument('--prepared'); parser.add_argument('--cpu'); parser.add_argument('--gpu')
    args = parser.parse_args()
    if args.operation == 'freeze': freeze(args.protocol, args.reference)
    else: run(args)


if __name__ == '__main__': main()
