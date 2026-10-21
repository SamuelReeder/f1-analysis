"""Frozen full-design checkpoint engineering workload; never a converged fit."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import traceback

from analysis import backend_determinism as deterministic
from analysis import backend_precision as precision

base = precision.base
ROOT = base.ROOT
PARENT = 'docs/overall_backend_determinism_protocol.json'
RESULTS = 'outputs/championship/development/overall-backend-determinism-v1'
CHECKS = ('outputs/championship/development/overall-checkpoint-engineering-checks-v1',
          'outputs/championship/development/overall-checkpoint-engineering-gpu-checks-v1')
SOURCES = ('analysis/checkpoint_model_replay/__init__.py', 'analysis/checkpoint_model_replay/__main__.py',
           'analysis/checkpoint_sampling/__init__.py', 'analysis/checkpoint_archive/__init__.py',
           'tests/test_overall_checkpoint_model_replay.py', 'docs/overall_checkpoint_model_replay.md')
PHASES = ('uninterrupted', 'warmup_stop', 'sampling_stop', 'resume')
EXPECTED_EXIT = dict(uninterrupted=0, warmup_stop=23, sampling_stop=23, resume=0)


def definition(parent):
    # An engineering workload, not a shortened production fit or convergence gate.
    return dict(id='overall-checkpoint-model-replay-v1', purpose='complete_design_checkpoint_engineering',
        studies=parent['studies'], phases=list(PHASES), expected_exit=EXPECTED_EXIT,
        settings=dict(chains=4, warmup=32, samples=25, seed=17, target_accept=.9,
                      max_tree_depth=10, init_median_samples=15), chunk_draws=7,
        chain_method='vectorized', max_phase_seconds=1800, terminate_grace_seconds=10,
        environments=parent['environments'], unset_environment=parent['unset_environment'],
        resources=parent['resources'], gpu_admission=parent['gpu_admission'], matmul_precision=None,
        interpreter='/home/sreeder/f1-analysis/.venv-gpu/bin/python',
        repetition_requirement='every_posterior_extra_and_final_sampler_state_bitwise_equal',
        full_fit_executed=False, convergence_evaluated=False, publication_allowed=False,
        scientific_acceptance_allowed=False, changes_original_screen=False)


def evidence(parent):
    return [PARENT,
        *[f'{RESULTS}/compare/cpu32/{case}/result.json' for case in parent['studies']],
        *[f'{folder}/{name}' for folder in CHECKS for name in ('run.json','stdout.txt')]]


def freeze(path):
    parent, original = deterministic.checked(ROOT/PARENT)
    for case in parent['studies']:
        r = base.read(ROOT/f'{RESULTS}/compare/cpu32/{case}/result.json')
        if (r['state'] != 'complete' or r['protocol_sha256'] != base.sha(ROOT/PARENT)
                or not r['numerical_candidate_passed']):
            raise ValueError('Require complete deterministic numerical agreement')
    for folder in CHECKS:
        r = base.read(ROOT/folder/'run.json')
        if r['exit_code'] != 0 or not r['sources_unchanged']:
            raise ValueError('Require the completed checkpoint engineering checks')
        for name, expected in r['source_sha256'].items():
            if base.committed(ROOT,name) != expected:
                raise ValueError('Checked checkpoint implementation changed')
    runtime = {}
    for backend in ('cpu','gpu'):
        first = next(iter(original['runtime_source_sha256'][backend]))
        packages = Path(first).parents[2]
        for name in ('numpyro/util.py','numpyro/infer/hmc_util.py','jax/_src/lax/lax.py'):
            target = packages/name
            runtime[str(target)] = base.sha(target)
    p = dict(definition(parent), registered_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={n:base.committed(ROOT,n) for n in SOURCES},
        evidence_sha256={n:base.committed(ROOT,n) for n in evidence(parent)}, runtime_sha256=runtime)
    with Path(path).open('x') as stream:
        json.dump(p,stream,indent=2);stream.write('\n')


def checked(path):
    path=Path(path).resolve();base.committed(ROOT,str(path.relative_to(ROOT)))
    p=base.read(path);parent,original=deterministic.checked(ROOT/PARENT)
    if (any(p.get(k)!=v for k,v in definition(parent).items())
            or set(p['source_sha256'])!=set(SOURCES)
            or set(p['evidence_sha256'])!=set(evidence(parent))):
        raise ValueError('Changed complete-design engineering procedure')
    for n,expected in {**p['source_sha256'],**p['evidence_sha256']}.items():
        if base.sha(ROOT/n)!=expected: raise ValueError(f'Changed engineering input: {n}')
    if any(base.sha(n)!=h for n,h in p['runtime_sha256'].items()):
        raise ValueError('Changed collection runtime')
    return p,original


def paths(p,case,phase,output):
    arm='uninterrupted' if phase=='uninterrupted' else 'resumed'
    cache=ROOT/'outputs/championship/cache'/p['id']/case/arm
    return cache,Path(output)/'sample'/case/phase


def validate_predecessor(phase,prior):
    expected = {'sampling_stop':('warmup_stop',7,0), 'resume':('sampling_stop',32,7)}
    if phase not in expected:
        if prior is not None: raise ValueError('Initial execution must have no predecessor')
        return
    previous,warmup,draws=expected[phase]
    if (not prior or prior['phase']!=previous or prior['state']!='intentional_stop'
            or prior['progress'][-1]['warmup_completed']!=warmup or prior['progress'][-1]['draws']!=draws):
        raise ValueError('Require the exact planned committed checkpoint boundary')


class PlannedStop(Exception):
    pass


def worker(args):
    p,original=checked(args.protocol)
    if args.phase not in PHASES: raise ValueError('Unknown execution phase')
    study=next(s for s in original['studies'] if s['case']==args.case)
    cache,output=paths(p,args.case,args.phase,args.output)
    output.mkdir(parents=True,exist_ok=False)
    record=dict(case=args.case,phase=args.phase,state='running',protocol_sha256=base.sha(args.protocol),
        started_at=datetime.now(timezone.utc).isoformat(),progress=[],full_fit_executed=False,
        convergence_evaluated=False,publication_allowed=False)
    started=time.monotonic();base.save(output/'result.json',record)
    try:
        record['actual_scope']=precision.scope(p,original,'gpu64_a',args.scope)
        previous={'sampling_stop':'warmup_stop','resume':'sampling_stop'}.get(args.phase)
        prior=None
        if previous:
            prior_path=Path(args.output)/'sample'/args.case/previous/'result.json'
            prior=base.read(prior_path)
            if prior['case']!=args.case or prior['protocol_sha256']!=record['protocol_sha256']:
                raise ValueError('Predecessor belongs to another execution')
            record['predecessor_sha256']=base.sha(prior_path)
        elif cache.exists():
            raise ValueError('Never reuse an existing initial execution')
        validate_predecessor(args.phase,prior)
        from f1rank.overall.checkpoints import Checkpoints,canonical
        if prior is not None:
            ident=base.read(cache/'checkpoints/identity.json')
            with Checkpoints(cache/'checkpoints',ident) as saved:
                tip=saved.root/f'checkpoint-{len(saved.records)-1:06d}'/'manifest.json'
                if (base.sha(tip)!=prior['checkpoint_tip_sha256']
                        or saved.draws!=prior['progress'][-1]['draws']
                        or saved.warmup_completed!=prior['progress'][-1]['warmup_completed']):
                    raise ValueError('Predecessor checkpoint changed')
        prepared_path=ROOT/f'{precision.RESULTS}/prepare/cpu/{args.case}/result.json'
        prepared,blocks,_=base.load_prepared(original,study,prepared_path,base.sha(ROOT/precision.PARENT))
        import jax.numpy as jnp
        import numpy as np
        from f1rank.overall.batched_rotation import prepare as batch_rotation
        from f1rank.overall.control_rotation import build_rotation
        from f1rank.overall.weekend_coordinates import weekend_model
        from analysis.checkpoint_sampling import run_chunks
        from analysis.checkpoint_archive import finalize,inventory
        rotation=batch_rotation(build_rotation(blocks['laps']))
        if rotation.fingerprint()!=prepared['control_rotation_sha256']:
            raise ValueError('Original control map changed')
        arguments=tuple({k:jnp.asarray(precision.lift(v)) if hasattr(v,'dtype') or type(v) is float else v
                         for k,v in blocks[b].items()} for b in ('qualifying','laps','outcomes'))
        specification=dict(protocol_sha256=record['protocol_sha256'],case=args.case,
            prepared_sha256=base.sha(prepared_path),settings=p['settings'])
        key=hashlib.sha256(canonical(specification)).hexdigest()
        record.update(specification=specification,run_key=key,cache=str(cache.relative_to(ROOT)))
        def verify():
            checked(args.protocol)
            if base.sha(prepared_path)!=specification['prepared_sha256']:
                raise ValueError('Prepared input changed')
        def progress(row):
            record['progress'].append(row);base.save(output/'result.json',record)
            if ((args.phase=='warmup_stop' and row['warmup_completed']==p['chunk_draws'] and row['draws']==0)
                    or (args.phase=='sampling_stop' and row['draws']==p['chunk_draws'])):
                raise PlannedStop()
        try:
            record['sampling']=run_chunks(weekend_model(rotation),arguments,kwargs={},settings=p['settings'],
                run_key=key,directory=cache/'checkpoints',chunk_draws=p['chunk_draws'],
                verify_inputs=verify,chain_method=p['chain_method'],on_checkpoint=progress)
        except PlannedStop:
            if args.phase not in ('warmup_stop','sampling_stop'):raise
            record['state']='intentional_stop'
        else:
            if args.phase in ('warmup_stop','sampling_stop'):
                raise ValueError('Planned interruption was not reached')
            ident=base.read(cache/'checkpoints/identity.json')
            with Checkpoints(cache/'checkpoints',ident) as saved:
                observed=inventory(saved)
            axes=[p['settings']['chains'],p['settings']['samples']]
            expected_post={n:dict(shape=[*axes,*v['shape']],
                dtype=np.dtype('float64').str if np.dtype(v['dtype']).kind=='f' else v['dtype'])
                for n,v in prepared['collector_inventory'].items()}
            if observed['posterior']!=expected_post:
                raise ValueError('Require the complete registered model collector')
            # NumPyro extras have fixed names/axes and checked numeric kinds; its
            # counter width is determined by the bound runtime, not a model site.
            from f1rank.overall.inference import validate_extras
            with Checkpoints(cache/'checkpoints',ident) as saved:
                for row,_,extras in saved.chunks():
                    validate_extras(extras,dict(p['settings'],samples=row['draws']-row['start_draw']))
            record['archive']=finalize(cache/'checkpoints',cache/'archive',identity=ident,
                expected_inventory=dict(posterior=expected_post,extras=observed['extras']),verify_inputs=verify)
            record['archive_manifest_sha256']=base.sha(cache/'archive/manifest.json')
            record['state']='complete'
        ident=base.read(cache/'checkpoints/identity.json')
        with Checkpoints(cache/'checkpoints',ident) as saved:
            record['checkpoint_tip_sha256']=base.sha(saved.root/f'checkpoint-{len(saved.records)-1:06d}'/'manifest.json')
        verify()
    except BaseException as exc:
        record.update(state='failed',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.monotonic()-started,process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_at=datetime.now(timezone.utc).isoformat());base.save(output/'result.json',record)
    return EXPECTED_EXIT[args.phase]


def launch(args):
    p,_=checked(args.protocol);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=[];record=dict(protocol_sha256=base.sha(args.protocol),source_commit=subprocess.check_output(
        ['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),started_at=datetime.now(timezone.utc).isoformat(),
        studies=rows,full_fit_executed=False,publication_allowed=False)
    begun=time.monotonic();base.save(out/'execution.json',record)
    for case in p['studies']:
        predecessor_ok=True
        for phase in PHASES:
            row=dict(case=case,phase=phase);rows.append(row)
            if phase in ('sampling_stop','resume') and not predecessor_ok:
                row.update(state='blocked',reason='planned predecessor failed',exit_code=None)
                base.save(out/'execution.json',record);continue
            env=dict(os.environ,**p['environments']['gpu64_a'])
            for name in p['unset_environment']:env.pop(name,None)
            cmd=['/usr/bin/time','-v','-o',str(out/f'{case}-{phase}.resources.txt'),p['interpreter'],
                 '-u','-m','analysis.checkpoint_model_replay','worker','--protocol',str(Path(args.protocol).resolve()),
                 '--case',case,'--phase',phase,'--scope',args.scope,'--output',str(out)]
            row.update(state='running',command=cmd);base.save(out/'execution.json',record)
            with (out/f'{case}-{phase}.stdout.txt').open('w') as stream:
                process=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                row['pid']=process.pid;base.save(out/'execution.json',record)
                try:
                    status=process.wait(timeout=p['max_phase_seconds'])
                except subprocess.TimeoutExpired:
                    row['timeout']=True;os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=p['terminate_grace_seconds'])
                    except subprocess.TimeoutExpired:pass
                    # The timing wrapper can exit before a descendant. Terminate
                    # the owned process group even if its original leader exited.
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    process.wait()
                    status=process.returncode
            row.update(exit_code=status,state='complete' if status==EXPECTED_EXIT[phase] else 'failed')
            path=out/'sample'/case/phase/'result.json'
            if path.exists():
                row.update(report=str(path.relative_to(ROOT)),report_sha256=base.sha(path),report_state=base.read(path)['state'])
            expected_state='intentional_stop' if EXPECTED_EXIT[phase] else 'complete'
            predecessor_ok=status==EXPECTED_EXIT[phase] and row.get('report_state')==expected_state
            row['expected_outcome']=predecessor_ok
            base.save(out/'execution.json',record)
            print(json.dumps({k:v for k,v in row.items() if k!='command'}),flush=True)
    record.update(all_processed=True,all_expected_outcomes=all(r.get('expected_outcome',False) for r in rows),
        finished_at=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-begun)
    base.save(out/'execution.json',record)
    return 0 if record['all_expected_outcomes'] else 1


def archive_difference(left,right,expected):
    """Compare complete finite fields one at a time, retaining all mismatches."""
    import zipfile
    import numpy as np
    names=set(expected)
    for path in (left,right):
        with zipfile.ZipFile(path) as saved:
            if len(saved.namelist())!=len(names) or set(saved.namelist())!={n+'.npy' for n in names}:
                raise ValueError('Incomplete or duplicate archive fields')
    fields={}
    with np.load(left,allow_pickle=False) as a,np.load(right,allow_pickle=False) as b:
        for name,info in expected.items():
            first,second=a[name],b[name]
            if any(list(v.shape)!=info['shape'] or v.dtype.str!=info['dtype'] for v in (first,second)):
                raise ValueError('Changed complete archive shape or dtype')
            fields[name]=precision.differences({name:first},{name:second})[name]
    return dict(fields=fields,bitwise_equal=all(v['bitwise_equal'] for v in fields.values()))


def compare(args):
    p,original=checked(args.protocol)
    precision.scope(p,original,'cpu32',args.scope)
    output=Path(args.output);execution=base.read(output/'execution.json')
    base.committed(ROOT,str((output/'execution.json').relative_to(ROOT)))
    result=dict(protocol_sha256=base.sha(args.protocol),studies=[],publication_allowed=False,
        full_fit_executed=False,convergence_evaluated=False)
    if (execution['protocol_sha256']!=result['protocol_sha256'] or not execution['all_processed']
            or [(r['case'],r['phase']) for r in execution['studies']]!=[(c,s) for c in p['studies'] for s in PHASES]):
        raise ValueError('Require a terminal complete execution inventory')
    for case in p['studies']:
        row=dict(case=case,state='unavailable');result['studies'].append(row)
        records={r['phase']:r for r in execution['studies'] if r['case']==case}
        if any(not r.get('expected_outcome') for r in records.values()):
            row['reason']='Declared execution failed or was blocked';continue
        reports={}
        for phase,r in records.items():
            path=ROOT/r['report'];base.committed(ROOT,r['report'])
            if base.sha(path)!=r['report_sha256']:raise ValueError('Execution report changed')
            reports[phase]=base.read(path)
        left,right=[ROOT/reports[phase]['cache'] for phase in ('uninterrupted','resume')]
        manifests=[]
        for directory,phase in ((left,'uninterrupted'),(right,'resume')):
            manifest=directory/'archive/manifest.json'
            if base.sha(manifest)!=reports[phase]['archive_manifest_sha256']:
                raise ValueError('Final archive manifest changed')
            m=base.read(manifest);manifests.append(m)
            for name,expected in m['files'].items():
                if base.sha(directory/'archive'/name)!=expected:raise ValueError('Final archive bytes changed')
        if (manifests[0]['identity']!=manifests[1]['identity'] or manifests[0]['inventory']!=manifests[1]['inventory']
                or reports['uninterrupted']['run_key']!=reports['resume']['run_key']):
            raise ValueError('Different target, runtime or collector between replay arms')
        row['archives']={group:archive_difference(left/'archive'/(group+'.npz'),right/'archive'/(group+'.npz'),fields)
                         for group,fields in manifests[0]['inventory'].items()}
        from f1rank.overall.checkpoints import Checkpoints,_array_shapes
        tips=[]
        for directory,phase in ((left,'uninterrupted'),(right,'resume')):
            with Checkpoints(directory/'checkpoints',manifests[0]['identity']) as saved:
                tip=saved.root/f'checkpoint-{len(saved.records)-1:06d}'
                if base.sha(tip/'manifest.json')!=reports[phase]['checkpoint_tip_sha256']:
                    raise ValueError('Final sampler checkpoint changed')
                tips.append((tip,base.read(tip/'manifest.json')['tree']))
        row['sampler_tree_equal']=tips[0][1]==tips[1][1]
        row['sampler_state']=archive_difference(tips[0][0]/'state.npz',tips[1][0]/'state.npz',_array_shapes(tips[0][0]/'state.npz'))
        row.update(state='complete',replay_passed=row['sampler_tree_equal'] and row['sampler_state']['bitwise_equal']
                   and all(v['bitwise_equal'] for v in row['archives'].values()))
    checked(args.protocol)
    result.update(all_processed=True,all_replays_passed=all(r.get('replay_passed',False) for r in result['studies']))
    with (output/'comparison.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False);stream.write('\n')
    return 0 if result['all_replays_passed'] else 1
