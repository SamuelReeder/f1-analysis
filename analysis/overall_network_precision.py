"""Matched precision investigation on fixed failed-network coordinates; no MCMC."""
import argparse
from datetime import datetime, timezone
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('network_diagnostic_reader',ROOT/'analysis/overall_network_diagnosis.py')
diagnosis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnosis)
sha,save=diagnosis.sha,diagnosis.save
SOURCES=('analysis/overall_network_precision.py','analysis/overall_network_diagnosis.py',
         'tests/test_overall_network_precision.py','docs/overall_network_precision.md')
DIAGNOSTIC='outputs/championship/development/overall-network-diagnosis-v1/result.json'
ORIGINAL_PROTOCOL='docs/overall_batched_rotation_audit_protocol.json'
ENVIRONMENT=dict(JAX_PLATFORMS='cpu',JAX_ENABLE_COMPILATION_CACHE='false',
    XLA_FLAGS='--xla_force_host_platform_device_count=1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
PRECISIONS=('float32','float64')


def committed(root,name):
    raw=(root/name).read_bytes()
    if subprocess.check_output(['git','show',f'HEAD:{name}'],cwd=root)!=raw:
        raise ValueError(f'Uncommitted input: {name}')
    return sha(root/name)


def freeze(path,reference):
    reference=Path(reference).resolve()
    original=json.loads((reference/'docs/overall_complete_profile_protocol.json').read_bytes())
    inputs=[diagnosis.REVIEW,diagnosis.PREFLIGHT,ORIGINAL_PROTOCOL,
            'docs/overall_complete_profile_protocol.json',original['storage_results']['real_network']]
    p=dict(id='overall-network-precision-v1',purpose='fixed_network_coordinate_precision_comparison',
        status='frozen',registered_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={name:committed(ROOT,name) for name in SOURCES},
        diagnostic_sha256=committed(ROOT,DIAGNOSTIC),reference_root=str(reference),
        reference_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=reference,text=True).strip(),
        reference_input_sha256={name:committed(reference,name) for name in inputs},
        precisions=list(PRECISIONS),environment=ENVIRONMENT,
        point_policy='original_generating_point_then_first_and_last_draw_of_every_chain_of_every_exhausted_attempt',
        coordinate_policy='reconstruct_once_in_float32_then_lift_the_same_unconstrained_values',
        data_policy='generate_once_in_original_float32_verify_original_hashes_then_reuse_exact_arrays',
        rotation_policy='build_from_original_numpy_arrays_before_any_lifting_verify_original_fingerprint',
        sampling_allowed=False,acceptance_gate=False,publication_allowed=False)
    with Path(path).open('x') as f: f.write(json.dumps(p,indent=2)+'\n')
    return p


def checked(path):
    path=Path(path).resolve()
    committed(ROOT,path.relative_to(ROOT).as_posix())
    p=json.loads(path.read_bytes())
    if (p['precisions']!=list(PRECISIONS) or p['environment']!=ENVIRONMENT
            or p['purpose']!='fixed_network_coordinate_precision_comparison'
            or p['sampling_allowed'] is not False or p['acceptance_gate'] is not False
            or p['publication_allowed'] is not False):
        raise ValueError('Changed or unsupported numerical investigation')
    for name,value in p['source_sha256'].items():
        if name not in SOURCES or committed(ROOT,name)!=value:
            raise ValueError('Numerical investigation source changed')
    if set(p['source_sha256'])!=set(SOURCES) or committed(ROOT,DIAGNOSTIC)!=p['diagnostic_sha256']:
        raise ValueError('Incomplete investigation source or changed prior diagnosis')
    reference=Path(p['reference_root'])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=reference,text=True).strip()!=p['reference_commit']:
        raise ValueError('Preserve the frozen reference checkout')
    for name,value in p['reference_input_sha256'].items():
        if committed(reference,name)!=value: raise ValueError('Original evidence changed')
    return p


def scope_properties(scope,precision):
    fields=('Slice','ControlGroup','MemoryMax','CPUQuotaPerSecUSec','ActiveState')
    command=['systemctl','--user','show',scope]
    for key in fields: command+=['-p',key]
    actual=dict(line.split('=',1) for line in subprocess.check_output(command,text=True).splitlines())
    if (actual['Slice']!='f1.slice' or actual['MemoryMax']!=str(3*1024**3)
            or actual['CPUQuotaPerSecUSec']!='2s' or actual['ActiveState']!='active'
            or os.getpriority(os.PRIO_PROCESS,0)!=19):
        raise ValueError('Use the standing capped analysis scope')
    membership=next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
    if actual['ControlGroup']!=membership: raise ValueError('Wrong active process scope')
    own=Path('/sys/fs/cgroup')/membership.lstrip('/')
    for sibling in own.parent.iterdir():
        if sibling.is_dir() and sibling!=own:
            events=dict(line.split() for line in (sibling/'cgroup.events').read_text().splitlines())
            if events.get('populated')!='0': raise ValueError('Other heavy work is active')
    if (own.parent/'cgroup.procs').read_text().strip(): raise ValueError('Uncapped work in shared slice')
    expected=dict(ENVIRONMENT,JAX_ENABLE_X64='true' if precision=='float64' else 'false')
    if any(os.environ.get(k)!=v for k,v in expected.items()): raise ValueError('Use the explicit precision environment')
    return dict(actual,Nice=19,environment=expected)


def pack_blocks(blocks):
    arrays,scalars={},{}
    for block,values in blocks.items():
        scalars[block]={}
        for name,value in values.items():
            if type(value) in (int,float,bool): scalars[block][name]=value
            else:
                a=np.asarray(value)
                if a.dtype.kind not in 'biuf' or not np.isfinite(a).all(): raise ValueError('Unsupported design value')
                arrays[f'block/{block}/{name}']=a
    return arrays,scalars


def unpack_blocks(arrays,scalars):
    blocks={name:dict(values) for name,values in scalars.items()}
    for name,value in arrays.items():
        if name.startswith('block/'):
            _,block,key=name.split('/')
            blocks[block][key]=value
    return blocks


def lift(value,precision):
    a=np.asarray(value)
    if a.dtype.kind=='f': return a.astype(precision)
    return a


def difference(left,right):
    left,right=np.asarray(left,dtype=np.float64),np.asarray(right,dtype=np.float64)
    if left.shape!=right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Compare complete matching finite fields')
    delta=left-right
    ref=float(np.linalg.norm(right.ravel())); norm=float(np.linalg.norm(delta.ravel()))
    index=int(np.argmax(np.abs(delta))) if delta.size else None
    return dict(shape=list(left.shape),coordinates=int(left.size),max_absolute=float(np.max(np.abs(delta))) if delta.size else 0.,
        l2_difference=norm,reference_l2=ref,relative_l2=norm/ref if ref else None,
        largest_flat_index=index,left_at_largest=float(left.ravel()[index]) if index is not None else None,
        right_at_largest=float(right.ravel()[index]) if index is not None else None)


def original_modules(p):
    sys.path.insert(0,p['reference_root'])
    from analysis.overall_batched_rotation_audit import checked_protocol
    checked_protocol(Path(p['reference_root'])/ORIGINAL_PROTOCOL)


def prepare_inputs(p,cache,record,checkpoint):
    original_modules(p)
    import jax.numpy as jnp
    from numpyro.infer.util import unconstrain_fn
    from analysis.overall_recovery_checks import generate_with_parameters
    from analysis.overall_control_rotation import ROTATED_SITE,build_rotation
    from analysis.overall_batched_rotation import prepare
    from analysis.overall_weekend_coordinates import HYPER_SITES,TOTAL,COMPLEMENT,transform,weekend_model
    from analysis.overall_posterior_archive import _header
    from f1rank.overall.fitting import arrays_digest
    reference=Path(p['reference_root'])
    prior=json.loads((reference/diagnosis.PREFLIGHT).read_bytes())
    review=json.loads((reference/diagnosis.REVIEW).read_bytes())
    diagnostic=json.loads((ROOT/DIAGNOSTIC).read_bytes())
    protocol=json.loads((reference/'docs/overall_complete_profile_protocol.json').read_bytes())
    inventory=json.loads((reference/protocol['storage_results']['real_network']).read_bytes())['sites']
    generated=generate_with_parameters('real_network',prior['generation_seed'])
    if any(generated[k]!=prior[k] for k in ('arrays_key','parameters_key','truth_key')):
        raise ValueError('Original generated problem changed')
    blocks,params=generated['blocks'],generated['parameters']
    original=build_rotation(blocks['laps'])
    if original.fingerprint()!=prior['control_rotation_sha256']: raise ValueError('Original rotation changed')
    model=weekend_model(prepare(original))
    args=tuple({k:jnp.asarray(v) if isinstance(v,np.ndarray) else v for k,v in blocks[name].items()}
               for name in ('qualifying','laps','outcomes'))
    updated=dict(params)
    for old,new in HYPER_SITES.items(): updated[new]=updated.pop(old)
    updated[TOTAL],updated[COMPLEMENT]=transform(params['sd_driver_form'],params['overall_sd_day'],params['overall_shared_weekend'],
        updated.pop('driver_form_z'),updated.pop('overall_day_z'))
    updated[ROTATED_SITE]=original.rotated({name:updated.pop(name) for name in original.shapes})
    shapes=prior['transformed_parameter_shapes']
    points=[]; labels=[]; expected=[]
    def retain(point,label,report):
        z=unconstrain_fn(model,args,{}, {name:jnp.asarray(value) for name,value in point.items()})
        z={name:np.asarray(value) for name,value in z.items()}
        if ({name:list(value.shape) for name,value in z.items()}!=shapes
                or any(value.dtype!=np.dtype('float32') or not np.isfinite(value).all() for value in z.values())):
            raise ValueError('Require the complete original finite float32 coordinates')
        points.append(z); labels.append(label); expected.append(report)
    retain(updated,dict(kind='generating'),diagnostic['generating_point'])
    study=next(s for s in review['studies'] if s['case']=='real_network')
    attempts=study['continuation']['attempts']
    if len(attempts)!=len(protocol['retry']): raise ValueError('Require every exhausted attempt')
    for a,previous in zip(attempts,diagnostic['attempts'],strict=True):
        settings=a['expected_settings']
        if a['attempt']!=previous['attempt'] or a['archive_sha256']!=previous['archive_sha256']:
            raise ValueError('Diagnostic and archived attempt differ')
        _,ends=diagnosis.read_attempt(reference/a['archive'],a['archive_sha256'],
            (settings['chains'],settings['samples']),inventory,shapes,_header)
        for i,(point,report) in enumerate(zip(ends,previous['endpoints'],strict=True)):
            retain(point,dict(kind='saved_endpoint',attempt=a['attempt'],chain=i//2,
                             draw=0 if i%2==0 else settings['samples']-1),report)
        record['prepared_attempts']=a['attempt']; checkpoint()
    arrays,scalars=pack_blocks(blocks)
    if arrays_digest(unpack_blocks(arrays,scalars))!=generated['arrays_key']:
        raise ValueError('Design serialization changed the original inputs')
    arrays.update({f'z/{name}':np.stack([point[name] for point in points]) for name in shapes})
    with cache.open('xb') as f: np.savez_compressed(f,**arrays)
    record.update(cache=str(cache),cache_sha256=sha(cache),block_scalars=scalars,point_labels=labels,
        expected_float32_reports=expected,parameter_shapes=shapes,arrays_key=generated['arrays_key'],
        control_rotation_sha256=original.fingerprint(),original_input_identities={k:generated[k] for k in ('arrays_key','parameters_key','truth_key')},
        point_coordinates_sha256=arrays_digest({'coordinates':{name:arrays[f'z/{name}'] for name in shapes}}))


def evaluate(p,prepared_path,precision,cache,record,checkpoint):
    original_modules(p)
    import jax
    import jax.numpy as jnp
    import numpyro
    from numpyro.infer.util import potential_energy,constrain_fn
    from analysis.overall_control_rotation import build_rotation
    from analysis.overall_batched_rotation import prepare
    from analysis.overall_weekend_coordinates import weekend_model
    from f1rank.overall.fitting import arrays_digest
    prepared=json.loads(Path(prepared_path).read_bytes())
    record['prepared_sha256']=sha(prepared_path)
    if (prepared['state']!='complete' or prepared['protocol_sha256']!=record['protocol_sha256']
            or sha(prepared['cache'])!=prepared['cache_sha256']): raise ValueError('Changed or incomplete fixed inputs')
    with np.load(prepared['cache'],allow_pickle=False) as saved: arrays={name:saved[name] for name in saved.files}
    blocks=unpack_blocks(arrays,prepared['block_scalars'])
    if arrays_digest(blocks)!=prepared['arrays_key']: raise ValueError('Loaded original design differs')
    rotation=build_rotation(blocks['laps'])
    if rotation.fingerprint()!=prepared['control_rotation_sha256']: raise ValueError('Original numeric rotation changed')
    model=weekend_model(prepare(rotation))
    args=tuple({k:jnp.asarray(lift(v,precision)) if hasattr(v,'dtype') else v for k,v in blocks[name].items()}
               for name in ('qualifying','laps','outcomes'))
    shapes=prepared['parameter_shapes']
    coordinates={name:arrays[f'z/{name}'] for name in shapes}
    if arrays_digest({'coordinates':coordinates})!=prepared['point_coordinates_sha256']:
        raise ValueError('The common unconstrained coordinates changed')
    if jax.config.x64_enabled!=(precision=='float64'): raise ValueError('Wrong JAX precision mode')
    value_gradient=jax.jit(jax.value_and_grad(lambda z:potential_energy(model,args,{},z)))
    scalars=[name for name,shape in shapes.items() if not shape]
    def constrained_scalars(z):
        all_values=constrain_fn(model,args,{},z)
        return {name:all_values[name] for name in scalars}
    constrain=jax.jit(constrained_scalars)
    record.update(precision=precision,point_labels=prepared['point_labels'],parameter_shapes=shapes,
        point_coordinates_sha256=prepared['point_coordinates_sha256'],arrays_key=prepared['arrays_key'],
        control_rotation_sha256=rotation.fingerprint(),points=[],
        runtime=dict(jax=jax.__version__,numpyro=numpyro.__version__,numpy=np.__version__,x64=jax.config.x64_enabled))
    payload={f'gradient/{name}':[] for name in shapes}; payload['potential']=[]
    for i,label in enumerate(prepared['point_labels']):
        z={name:jnp.asarray(lift(value[i],precision)) for name,value in coordinates.items()}
        energy,gradient=value_gradient(z)
        report=diagnosis.gradient_report(energy,gradient)
        scalar_values=constrain(z)
        if any(np.asarray(v).dtype!=np.dtype(precision) for v in [energy,*gradient.values(),*scalar_values.values()]):
            raise ValueError('Mixed output precision')
        prior=prepared['expected_float32_reports'][i]
        exact=(report=={k:prior[k] for k in report}) if precision=='float32' else None
        record['points'].append(dict(label=label,**report,original_float32_report_identical=exact,
                                    scalar_constraints={k:float(v) for k,v in scalar_values.items()}))
        for name,value in gradient.items(): payload[f'gradient/{name}'].append(np.asarray(value))
        payload['potential'].append(np.asarray(energy))
        checkpoint()
    if (sha(prepared_path)!=record['prepared_sha256'] or sha(prepared['cache'])!=prepared['cache_sha256']):
        raise ValueError('Inputs changed during fixed-point evaluation')
    with cache.open('xb') as f: np.savez_compressed(f,**{name:np.stack(values) for name,values in payload.items()})
    record.update(cache=str(cache),cache_sha256=sha(cache),
        all_original_float32_reports_identical=all(row['original_float32_report_identical'] for row in record['points']) if precision=='float32' else None)


def compare(left_path,right_path,record):
    records=[json.loads(Path(p).read_bytes()) for p in (left_path,right_path)]
    left,right=records
    identities=('protocol_sha256','prepared_sha256','point_labels','point_coordinates_sha256','arrays_key','control_rotation_sha256','parameter_shapes')
    if (left['precision']!='float32' or right['precision']!='float64'
            or any(r['state']!='complete' or sha(r['cache'])!=r['cache_sha256'] for r in records)
            or any(left[k]!=right[k] for k in identities)
            or left['protocol_sha256']!=record['protocol_sha256']):
        raise ValueError('Require both complete matched precision calculations')
    record['evaluation_sha256']={str(path):sha(path) for path in (left_path,right_path)}
    record.update(point_labels=left['point_labels'],points=[],
                  all_original_float32_reports_identical=left['all_original_float32_reports_identical'])
    with np.load(left['cache'],allow_pickle=False) as a,np.load(right['cache'],allow_pickle=False) as b:
        names={f'gradient/{name}' for name in left['parameter_shapes']}|{'potential'}
        if set(a.files)!=names or set(b.files)!=names: raise ValueError('Incomplete derivative inventory')
        for name in names:
            shape=(len(left['point_labels']),)+tuple(left['parameter_shapes'][name[9:]] if name.startswith('gradient/') else ())
            if (a[name].shape!=shape or b[name].shape!=shape or a[name].dtype!=np.dtype('float32')
                    or b[name].dtype!=np.dtype('float64')): raise ValueError('Changed precision or coordinate inventory')
        for i,label in enumerate(left['point_labels']):
            row=dict(label=label,fields={},scalar_constraints={})
            for name in sorted(names):
                row['fields'][name]=difference(a[name][i],b[name][i])
            for name,value in left['points'][i]['scalar_constraints'].items():
                row['scalar_constraints'][name]=difference(value,right['points'][i]['scalar_constraints'][name])
            record['points'].append(row)
    if any(sha(path)!=value for path,value in record['evaluation_sha256'].items()): raise ValueError('Changed evaluation records')
    if any(sha(r['cache'])!=r['cache_sha256'] for r in records): raise ValueError('Changed derivative cache')


def execute(args):
    p=checked(args.protocol)
    precision=args.precision if args.action=='evaluate' else 'float32'
    actual=scope_properties(args.scope,precision)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    if args.cache:
        cache=Path(args.cache).resolve();cache.parent.mkdir(parents=True,exist_ok=True)
        if cache.exists(): raise FileExistsError('Preserve existing numerical cache')
    else: cache=None
    begun=time.monotonic()
    record=dict(state='running',action=args.action,started_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        protocol_sha256=sha(args.protocol),actual_scope=actual,
        mcmc_executed=False,scientific_acceptance_established=False,publication_allowed=False)
    def checkpoint(): save(output/'progress.json',record)
    checkpoint()
    try:
        if args.action=='prepare': prepare_inputs(p,cache,record,checkpoint)
        elif args.action=='evaluate': evaluate(p,args.prepared,args.precision,cache,record,checkpoint)
        else: compare(args.left,args.right,record)
        checked(args.protocol)
        record['state']='complete'
    except BaseException as exc:
        record.update(state='failed_exception',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.monotonic()-begun,finished_at=datetime.now(timezone.utc).isoformat(),
                      process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        save(output/'result.json',record)
    return record


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['freeze','prepare','evaluate','compare'])
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--reference-root')
    parser.add_argument('--scope');parser.add_argument('--output');parser.add_argument('--cache')
    parser.add_argument('--precision',choices=PRECISIONS);parser.add_argument('--prepared')
    parser.add_argument('--left');parser.add_argument('--right')
    args=parser.parse_args()
    if args.action=='freeze': freeze(args.protocol,args.reference_root)
    else:
        result=execute(args)
        print(json.dumps({k:result[k] for k in ('state','action','seconds','process_max_rss_kib')}))
