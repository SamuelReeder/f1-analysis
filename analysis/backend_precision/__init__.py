"""Post hoc backend/precision diagnosis with complete fixed inputs; no MCMC."""
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import resource
import subprocess
import time
import traceback

from analysis import overall_backend_compare as base

ROOT = base.ROOT
PARENT = 'docs/overall_backend_comparison_protocol.json'
RESULTS = 'outputs/championship/development/overall-backend-comparison-v1'
SOURCES = ('analysis/backend_precision/__init__.py', 'analysis/backend_precision/__main__.py',
           'tests/test_overall_backend_precision.py', 'docs/overall_backend_precision.md')
ARMS = {
    'cpu32': ('cpu', 'float32'), 'gpu32': ('gpu', 'float32'),
    'cpu64_a': ('cpu', 'float64'), 'cpu64_b': ('cpu', 'float64'),
    'gpu64_a': ('gpu', 'float64'), 'gpu64_b': ('gpu', 'float64'),
}
PAIRS = (
    ('original_cpu', 'cpu32'), ('original_gpu', 'gpu32'),
    ('cpu64_a', 'cpu64_b'), ('gpu64_a', 'gpu64_b'), ('cpu64_a', 'gpu64_a'),
    ('cpu32', 'cpu64_a'), ('gpu32', 'gpu64_a'),
    ('original_cpu', 'cpu64_a'), ('original_gpu', 'gpu64_a'),
)


def environments(parent):
    return {arm:dict(parent['environments'][backend], JAX_ENABLE_X64=str(precision == 'float64').lower())
            for arm,(backend,precision) in ARMS.items()}


def evidence(parent):
    return [PARENT, *[f'{RESULTS}/{operation}/{backend}/{s["case"]}/result.json'
        for s in parent['studies'] for operation,backend in
        (('prepare','cpu'), ('evaluate','cpu'), ('evaluate','gpu'), ('compare','cpu'))]]


def freeze(destination):
    parent = base.checked(ROOT/PARENT)
    for study in parent['studies']:
        for operation,backend in (('prepare','cpu'), ('evaluate','cpu'), ('evaluate','gpu'), ('compare','cpu')):
            r = base.read(ROOT/f'{RESULTS}/{operation}/{backend}/{study["case"]}/result.json')
            if r['state'] != 'complete' or r['protocol_sha256'] != base.sha(ROOT/PARENT):
                raise ValueError('Require complete original evaluations, including the failed comparison')
    p = dict(id='overall-backend-precision-v1', purpose='posthoc_full_inventory_precision_and_repeatability',
        registered_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={n:base.committed(ROOT,n) for n in SOURCES},
        evidence_sha256={n:base.committed(ROOT,n) for n in evidence(parent)},
        studies=[s['case'] for s in parent['studies']], arms={n:list(v) for n,v in ARMS.items()},
        pairs=[list(v) for v in PAIRS], environments=environments(parent),
        matmul_precision=None, unset_environment=['JAX_DEFAULT_MATMUL_PRECISION', 'XLA_PYTHON_CLIENT_MEM_FRACTION'],
        resources=parent['resources'], gpu_admission=parent['gpu_admission'],
        lifting='same_stored_float32_observations_and_coordinates_cast_to_float64_without_regeneration',
        rotation='same_original_numpy_basis_then_conversion_to_execution_precision',
        original_float32_reproduction='complete_dtype_shape_and_bitwise_comparison',
        higher_precision_is_exact_truth=False, sampling_allowed=False,
        changes_original_screen=False, scientific_acceptance_allowed=False, publication_allowed=False)
    with Path(destination).open('x') as f: json.dump(p,f,indent=2); f.write('\n')


def checked(path):
    path=Path(path).resolve();base.committed(ROOT,str(path.relative_to(ROOT)))
    p=base.read(path);parent=base.checked(ROOT/PARENT)
    if (p['arms']!={n:list(v) for n,v in ARMS.items()} or p['pairs']!=[list(v) for v in PAIRS]
            or p['environments']!=environments(parent) or p['studies']!=[s['case'] for s in parent['studies']]
            or p['matmul_precision'] is not None or p['resources']!=parent['resources']
            or p['gpu_admission']!=parent['gpu_admission']
            or any(p[k] for k in ('sampling_allowed','changes_original_screen','scientific_acceptance_allowed','publication_allowed'))
            or set(p['source_sha256'])!=set(SOURCES) or set(p['evidence_sha256'])!=set(evidence(parent))):
        raise ValueError('Changed diagnostic inventory or definition')
    for name,expected in {**p['source_sha256'],**p['evidence_sha256']}.items():
        if base.sha(ROOT/name)!=expected:raise ValueError(f'Changed diagnostic input: {name}')
    return p,parent


def scope(p,parent,arm,unit):
    from analysis.overall_complete_profile_diagnostics import actual_scope
    actual=actual_scope(unit)
    own=Path('/sys/fs/cgroup')/actual['ControlGroup'].lstrip('/')
    for sibling in own.parent.iterdir():
        if sibling.is_dir() and sibling!=own and (sibling/'cgroup.events').exists():
            if dict(line.split() for line in (sibling/'cgroup.events').read_text().splitlines()).get('populated')!='0':
                raise ValueError('Other heavy work is active')
    if any(os.environ.get(n)!=v for n,v in p['environments'][arm].items()):
        raise ValueError('Use the registered backend and precision environment')
    if any(os.environ.get(n) for n in p['unset_environment']):
        raise ValueError('Unset the registered optional environment overrides')
    backend,precision=ARMS[arm]
    if backend=='gpu':
        import csv
        rows=list(csv.reader(subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.free,driver_version','--format=csv,noheader,nounits'],text=True).splitlines()))
        if len(rows)!=1:raise ValueError('Wrong device inventory')
        admission=p['gpu_admission']
        if float(rows[0][1])*admission['pool_fraction']>float(rows[0][2])*admission['max_share_of_observed_free_memory']:
            raise ValueError('Insufficient free device memory for registered pool')
        actual['device_admission_observation']=rows
    import jax
    from importlib.metadata import version
    if (jax.default_backend() not in (('gpu','cuda') if backend=='gpu' else ('cpu',))
            or bool(jax.config.x64_enabled)!=(precision=='float64') or len(jax.local_devices())!=1
            or jax.config.jax_default_matmul_precision is not None):
        raise ValueError('Wrong actual backend, precision, devices or matmul configuration')
    actual.update(backend=jax.default_backend(),x64=bool(jax.config.x64_enabled),
        matmul_precision=jax.config.jax_default_matmul_precision,
        device_kinds=[d.device_kind for d in jax.local_devices()],
        packages={n:version(n) for n in ('numpy','scipy','jax','jaxlib','numpyro')})
    return actual


def lift(value):
    import numpy as np
    a=np.asarray(value)
    return a.astype(np.float64) if a.dtype.kind=='f' else value


def evaluate64(parent,study,prepared_path,cache,record):
    import jax
    import jax.numpy as jnp
    import numpy as np
    from numpyro.infer.util import constrain_fn,potential_energy
    from f1rank.overall.batched_rotation import prepare as batch_rotation
    from f1rank.overall.control_rotation import build_rotation
    from f1rank.overall.weekend_coordinates import weekend_model
    prepared,blocks,coordinates=base.load_prepared(parent,study,prepared_path,base.sha(ROOT/PARENT))
    rotation=batch_rotation(build_rotation(blocks['laps']))
    if rotation.fingerprint()!=prepared['control_rotation_sha256']:raise ValueError('Original rotation changed')
    model=weekend_model(rotation)
    args=tuple({k:jnp.asarray(lift(v)) if hasattr(v,'dtype') or type(v) is float else v for k,v in blocks[n].items()}
               for n in ('qualifying','laps','outcomes'))
    postprocess=partial(constrain_fn,model,args,{},return_deterministic=True)
    value_gradient=jax.value_and_grad(lambda z:potential_energy(model,args,{},z))
    def kernel(z):
        value,gradient=value_gradient(z)
        return value,gradient,postprocess(z)
    point=lambda i:{n:jnp.asarray(lift(v[i])) for n,v in coordinates.items()}
    started=time.monotonic();lowered=jax.jit(kernel).lower(point(0));lower_seconds=time.monotonic()-started
    started=time.monotonic();compiled=lowered.compile();compile_seconds=time.monotonic()-started
    records,times,transfers=[],[],[]
    for i in range(len(prepared['point_labels'])):
        z=point(i);jax.block_until_ready(z)
        started=time.monotonic();value,gradient,state=jax.block_until_ready(compiled(z));times.append(time.monotonic()-started)
        started=time.monotonic()
        row=dict(potential=np.asarray(value),**{f'gradient/{n}':np.asarray(v) for n,v in gradient.items()},
                 **{f'state/{n}':np.asarray(v) for n,v in state.items()})
        transfers.append(time.monotonic()-started)
        if set(gradient)!=set(coordinates) or set(state)!=set(prepared['collector_inventory']):
            raise ValueError('Incomplete gradient or retained state')
        for n,v in row.items():
            if not np.isfinite(v).all() or v.dtype.kind not in 'biuf':raise ValueError('Nonfinite or unsupported output')
            if (n=='potential' or n.startswith('gradient/')) and v.dtype!=np.dtype('float64'):
                raise ValueError('Potential and complete gradient must use the selected precision')
        for n,v in state.items():
            expected=prepared['collector_inventory'][n];dtype=np.dtype(expected['dtype'])
            if dtype.kind=='f':dtype=np.dtype('float64')
            if list(v.shape)!=expected['shape'] or np.asarray(v).dtype!=dtype:
                raise ValueError('Changed retained inventory or precision')
        records.append(row)
    arrays={n:np.stack([r[n] for r in records]) for n in records[0]}
    started=time.monotonic()
    with cache.open('xb') as f:np.savez_compressed(f,**arrays)
    record.update(prepared=str(prepared_path.relative_to(ROOT)),prepared_sha256=base.sha(prepared_path),
        cache=str(cache.relative_to(ROOT)),cache_sha256=base.sha(cache),point_labels=prepared['point_labels'],
        field_shapes={n:list(v.shape) for n,v in arrays.items()},field_dtypes={n:v.dtype.str for n,v in arrays.items()},
        lowering_seconds=lower_seconds,compile_seconds=compile_seconds,evaluation_seconds=times,
        transfer_seconds=transfers,archive_seconds=time.monotonic()-started)


def load_evaluation(path,case,labels,protocol_sha,arm=None):
    import numpy as np
    path=Path(path).resolve();base.committed(ROOT,str(path.relative_to(ROOT)));r=base.read(path)
    if (r['state']!='complete' or r['case']!=case or r['point_labels']!=labels or r['protocol_sha256']!=protocol_sha
            or (arm is not None and r.get('arm')!=arm)):
        raise ValueError('Incomplete or mismatched evaluation')
    archive=ROOT/r['cache'];archive.resolve().relative_to(ROOT/'outputs/championship/cache')
    if base.sha(archive)!=r['cache_sha256']:raise ValueError('Evaluation array bytes changed')
    with np.load(archive,allow_pickle=False) as a:arrays={n:a[n] for n in a.files}
    if ({n:list(v.shape) for n,v in arrays.items()}!=r['field_shapes']
            or {n:v.dtype.str for n,v in arrays.items()}!=r['field_dtypes']):
        raise ValueError('Evaluation inventory changed')
    return r,arrays


def differences(left,right):
    import numpy as np
    from analysis.overall_network_precision import difference
    if not left or set(left)!=set(right):raise ValueError('Keep every original output field')
    report={}
    for name in left:
        a,b=np.asarray(left[name]),np.asarray(right[name])
        if a.dtype.kind not in 'biuf' or b.dtype.kind not in 'biuf':raise ValueError('Numeric fields required')
        row=difference(a,b)
        index=np.unravel_index(row['largest_flat_index'],a.shape) if a.size else None
        row.update(largest_index=list(map(int,index)) if index is not None else None,
            same_dtype=a.dtype==b.dtype,exact_values=bool(np.array_equal(a,b)),
            bitwise_equal=a.dtype==b.dtype and a.tobytes()==b.tobytes())
        report[name]=row
    return report


def compare(p,parent,study,evaluations,record):
    labels=base.point_labels(study);arrays={};receipts={};identities={}
    for backend in ('cpu','gpu'):
        name='original_'+backend;path=ROOT/f'{RESULTS}/evaluate/{backend}/{study["case"]}/result.json'
        receipts[name],arrays[name]=load_evaluation(path,study['case'],labels,base.sha(ROOT/PARENT))
        identities[name]=dict(path=str(path.relative_to(ROOT)),sha256=base.sha(path))
    for arm in ARMS:
        path=Path(evaluations)/arm/study['case']/'result.json'
        receipts[arm],arrays[arm]=load_evaluation(path,study['case'],labels,record['protocol_sha256'],arm)
        identities[arm]=dict(path=str(path.resolve().relative_to(ROOT)),sha256=base.sha(path))
    if len({r['prepared_sha256'] for r in receipts.values()})!=1:raise ValueError('Different fixed inputs across arms')
    comparisons={f'{a}__{b}':differences(arrays[a],arrays[b]) for a,b in PAIRS}
    reproduction={backend:all(v['bitwise_equal'] for v in comparisons[f'original_{backend}__{backend}32'].values())
                  for backend in ('cpu','gpu')}
    budgets=base.compare_fields(arrays['cpu64_a'],arrays['gpu64_a'],
        dict(state=parent['state_tolerance'],value_and_gradient=parent['value_and_gradient_tolerance']))
    record.update(point_labels=labels,evaluations=identities,comparisons=comparisons,
        original_float32_reproduced=reproduction,
        higher_precision_cross_backend_screen=budgets,
        higher_precision_within_original_budgets=all(v['passed'] for v in budgets.values()),
        original_decision_changed=False,higher_precision_is_exact_truth=False)


def run(args):
    p,parent=checked(args.protocol);study=next(s for s in parent['studies'] if s['case']==args.case)
    output=Path(args.output);output.mkdir(parents=True,exist_ok=False)
    record=dict(purpose=p['purpose'],operation=args.operation,case=args.case,arm=args.arm,state='running',
        protocol_sha256=base.sha(args.protocol),base_protocol_sha256=base.sha(ROOT/PARENT),
        started_at=datetime.now(timezone.utc).isoformat(),sampling_executed=False,
        publication_allowed=False,scientific_acceptance_established=False,changes_original_screen=False)
    start=time.monotonic();base.save(output/'result.json',record)
    try:
        record['actual_scope']=scope(p,parent,args.arm if args.operation=='evaluate' else 'cpu32',args.scope)
        if args.operation=='evaluate':
            cache=base.cache_path(args.cache);prepared=ROOT/f'{RESULTS}/prepare/cpu/{args.case}/result.json'
            if ARMS[args.arm][1]=='float32':
                numeric={'protocol_sha256':base.sha(ROOT/PARENT)}
                base.evaluate(parent,study,prepared,cache,numeric)
                record.update({k:v for k,v in numeric.items() if k!='protocol_sha256'})
            else:evaluate64(parent,study,prepared,cache,record)
        else:compare(p,parent,study,args.evaluations,record)
        checked(args.protocol);record['state']='complete'
    except BaseException as exc:
        record.update(state='failed',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc());raise
    finally:
        record.update(seconds=time.monotonic()-start,process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_at=datetime.now(timezone.utc).isoformat());base.save(output/'result.json',record)
