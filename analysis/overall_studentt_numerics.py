"""Independent Student-t arithmetic investigation, without fitting or publication."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import traceback

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fixed_network_precision',ROOT/'analysis/overall_network_precision.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
sha,save=previous.sha,previous.save
DEPENDENCY='outputs/championship/development/overall-studentt-reference-dependency/dependency.json'
PREPARED='outputs/championship/development/overall-network-precision-v1/prepare/result.json'
ORIGINAL_PROTOCOL='docs/overall_network_precision_protocol.json'
SOURCES=('analysis/overall_studentt_numerics.py','analysis/overall_network_precision.py',
         'analysis/overall_network_diagnosis.py','tests/test_overall_studentt_numerics.py','docs/overall_studentt_numerics.md')
REFERENCE_DIGITS=(80,120)
RESIDUALS=(-10000.,-100.,-3.,-1.,0.,1.,3.,100.,10000.)
ERROR_EPS_MULTIPLIERS={'float32':32,'float64':256}
DF_QUARTER_EXPONENTS=tuple(range(-12,33))
METHODS=('original','log_beta')


def beta_log_prob(df,loc,scale,value):
    """Same Student-t density; change only the normalizer's evaluation."""
    import jax.numpy as jnp
    from jax.scipy.special import betaln
    y=(value-loc)/scale
    normalizer=jnp.log(scale)+.5*jnp.log(df)+betaln(.5,.5*df)
    return -.5*(df+1.)*jnp.log1p(y**2/df)-normalizer


@contextmanager
def beta_studentt_calculation():
    """Temporarily substitute arithmetic in a fresh deterministic audit process.

    No installed source file, sampler, distribution support or prior changes.
    The original method is always restored, including after an exception.
    """
    from numpyro.distributions import StudentT
    from numpyro.distributions.util import validate_sample
    original=StudentT.log_prob
    @validate_sample
    def replacement(self,value):
        return beta_log_prob(self.df,self.loc,self.scale,value)
    StudentT.log_prob=replacement
    try: yield
    finally: StudentT.log_prob=original


def load_reference():
    dependency=json.loads((ROOT/DEPENDENCY).read_bytes())
    wheel=ROOT/dependency['wheel']
    if sha(wheel)!=dependency['sha256']: raise ValueError('Changed reference wheel')
    sys.path.insert(0,str(wheel))
    import mpmath
    if mpmath.__version__!=dependency['version'] or not mpmath.__file__.startswith(str(wheel)):
        raise ValueError('Use the isolated pinned reference implementation')
    return mpmath


def reference_row(mp,log_df,residual,digits):
    """Differentiate the defining log-gamma density at arbitrary precision."""
    with mp.workdps(digits):
        z,x=mp.mpf(float(log_df)),mp.mpf(float(residual))
        def f(z,loc,log_scale):
            df=mp.exp(z);scale=mp.exp(log_scale);y=(x-loc)/scale
            return (mp.loggamma((df+1)/2)-mp.loggamma(df/2)-mp.log(df*mp.pi)/2
                    -log_scale-(df+1)*mp.log1p(y*y/df)/2)
        zero=mp.mpf(0)
        return [float(f(z,zero,zero)),float(mp.diff(lambda v:f(v,zero,zero),z)),
                float(mp.diff(lambda v:f(z,v,zero),zero)),float(mp.diff(lambda v:f(z,zero,v),zero))]


def inputs(prepared,arrays):
    rows=[]
    for index,label in enumerate(prepared['point_labels']):
        for name in ('nu','overall_lap_nu'):
            rows.append(dict(kind='saved_coordinate',point=label,parameter=name,
                             log_df=float(arrays[f'z/{name}'][index])))
    for power in DF_QUARTER_EXPONENTS:
        rows.append(dict(kind='declared_grid',quarter_exponent=power,log_df=float(np.float32(np.log(10.**(power/4))))))
    for df in (1.,2.,4.,16.):
        rows.append(dict(kind='declared_special',nominal_df=df,log_df=float(np.float32(np.log(df)))))
    boundary=np.float32(np.log(16.))
    for direction in (-np.inf,np.inf):
        rows.append(dict(kind='log16_neighbor',direction='below' if direction<0 else 'above',
                         log_df=float(np.nextafter(boundary,np.float32(direction)))))
    return [dict(row,residual=residual) for row in rows for residual in RESIDUALS]


def assessment(actual,reference,precision):
    actual,reference=np.asarray(actual),np.asarray(reference,dtype=np.float64)
    if actual.shape!=reference.shape or not np.isfinite(reference).all():
        raise ValueError('Incomplete value/derivative inventory or invalid reference')
    error=np.abs(actual.astype(np.float64)-reference)
    limit=ERROR_EPS_MULTIPLIERS[precision]*np.finfo(precision).eps*(1+np.abs(reference))
    finite=np.isfinite(actual)
    passed=finite & (error<=limit)
    return dict(all_within_fixed_error_budget=bool(passed.all()),nonfinite_coordinates=np.argwhere(~finite).tolist(),
        failing_rows=np.flatnonzero(~passed.all(axis=1)).tolist(),
        max_absolute_error=[float(error[:,i].max()) if finite[:,i].all() else None for i in range(actual.shape[1])],
        error_over_budget_max=[float((error/limit)[:,i].max()) if finite[:,i].all() else None for i in range(actual.shape[1])])


def library_sources():
    numpyro_root=Path(importlib.util.find_spec('numpyro').origin).parent
    jax_root=Path(importlib.util.find_spec('jax').origin).parent
    files=[numpyro_root/'distributions/continuous.py',numpyro_root/'distributions/transforms.py',
           jax_root/'_src/scipy/special.py',jax_root/'_src/third_party/scipy/betaln.py']
    return {str(path):sha(path) for path in files}


def freeze(path):
    original=previous.checked(ROOT/ORIGINAL_PROTOCOL)
    prepared=json.loads((ROOT/PREPARED).read_bytes())
    dependency=json.loads((ROOT/DEPENDENCY).read_bytes())
    if sha(prepared['cache'])!=prepared['cache_sha256'] or sha(ROOT/dependency['wheel'])!=dependency['sha256']:
        raise ValueError('Original numerical cache or reference wheel differs')
    p=dict(id='overall-studentt-numerics-v1',purpose='independent_studentt_primitive_and_matched_model_arithmetic',
        status='frozen',registered_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={name:previous.committed(ROOT,name) for name in SOURCES},
        input_sha256={name:previous.committed(ROOT,name) for name in (DEPENDENCY,PREPARED,ORIGINAL_PROTOCOL)},
        library_sha256=library_sources(),reference_digits=list(REFERENCE_DIGITS),residuals=list(RESIDUALS),
        df_quarter_exponents=list(DF_QUARTER_EXPONENTS),error_eps_multipliers=ERROR_EPS_MULTIPLIERS,
        methods=list(METHODS),precisions=list(previous.PRECISIONS),
        network_coordinate_source_sha256=prepared['cache_sha256'],reference_wheel_sha256=dependency['sha256'],
        original_reference_commit=original['reference_commit'],sampling_allowed=False,publication_allowed=False)
    with Path(path).open('x') as f:f.write(json.dumps(p,indent=2)+'\n')


def checked(path):
    path=Path(path).resolve();previous.committed(ROOT,path.relative_to(ROOT).as_posix())
    p=json.loads(path.read_bytes())
    expected=dict(reference_digits=list(REFERENCE_DIGITS),residuals=list(RESIDUALS),
        df_quarter_exponents=list(DF_QUARTER_EXPONENTS),error_eps_multipliers=ERROR_EPS_MULTIPLIERS,
        methods=list(METHODS),precisions=list(previous.PRECISIONS),sampling_allowed=False,publication_allowed=False)
    if any(p.get(k)!=v for k,v in expected.items()):raise ValueError('Changed numerical workload or criterion')
    if set(p['source_sha256'])!=set(SOURCES):raise ValueError('Incomplete numerical source inventory')
    for name,value in dict(p['source_sha256'],**p['input_sha256']).items():
        if previous.committed(ROOT,name)!=value:raise ValueError('Changed numerical source or evidence')
    if p['library_sha256']!=library_sources():raise ValueError('Installed numerical implementation changed')
    previous.checked(ROOT/ORIGINAL_PROTOCOL)
    dependency=json.loads((ROOT/DEPENDENCY).read_bytes())
    if sha(ROOT/dependency['wheel'])!=p['reference_wheel_sha256']:raise ValueError('Changed reference wheel')
    return p


def primitive(precision,record,checkpoint):
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    prepared=json.loads((ROOT/PREPARED).read_bytes())
    if sha(prepared['cache'])!=prepared['cache_sha256']:raise ValueError('Changed fixed coordinates')
    with np.load(prepared['cache'],allow_pickle=False) as saved:
        arrays={name:saved[name] for name in ('z/nu','z/overall_lap_nu')}
    rows=inputs(prepared,arrays)
    mp=load_reference()
    refs=[];stable=[]
    for row in rows:
        low=reference_row(mp,row['log_df'],row['residual'],REFERENCE_DIGITS[0])
        high=reference_row(mp,row['log_df'],row['residual'],REFERENCE_DIGITS[1])
        stable.append(low==high);refs.append(high)
    reference=np.asarray(refs,dtype=np.float64)
    record.update(reference_version=mp.__version__,reference_float64_rounding_stable=all(stable),
                  columns=['log_density','d_log_df','d_location','d_log_scale'],rows=rows,reference=refs,methods={})
    checkpoint()
    coords=jnp.asarray([[row['log_df'],0.,0.] for row in rows],dtype=precision)
    observations=jnp.asarray([row['residual'] for row in rows],dtype=precision)
    for method in METHODS:
        def f(p,x):
            df,loc,scale=jnp.exp(p[0]),p[1],jnp.exp(p[2])
            return dist.StudentT(df,loc,scale).log_prob(x) if method=='original' else beta_log_prob(df,loc,scale,x)
        values,grads=jax.jit(jax.vmap(jax.value_and_grad(f),in_axes=(0,0)))(coords,observations)
        actual=np.column_stack([np.asarray(values),np.asarray(grads)])
        if actual.dtype!=np.dtype(precision):raise ValueError('Wrong primitive output precision')
        encoded=[[float(v) if np.isfinite(v) else str(v) for v in row] for row in actual]
        record['methods'][method]=dict(values=encoded,**assessment(actual,reference,precision))
        checkpoint()
    if not all(stable):raise ValueError('Independent reference did not stabilize at the declared precisions')
    if sha(prepared['cache'])!=prepared['cache_sha256']:raise ValueError('Inputs changed during primitive check')


def execute(args):
    protocol=checked(args.protocol)
    actual=previous.scope_properties(args.scope,args.precision)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    begun=time.monotonic()
    record=dict(state='running',action=args.action,precision=args.precision,
        started_at=datetime.now(timezone.utc).isoformat(),studentt_protocol_sha256=sha(args.protocol),
        protocol_sha256=sha(ROOT/ORIGINAL_PROTOCOL),actual_scope=actual,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        mcmc_executed=False,scientific_acceptance_established=False,publication_allowed=False)
    def checkpoint():save(output/'progress.json',record)
    checkpoint()
    try:
        if args.action=='primitive':primitive(args.precision,record,checkpoint)
        elif args.action=='network':
            cache=Path(args.cache).resolve();cache.parent.mkdir(parents=True,exist_ok=True)
            if cache.exists():raise FileExistsError('Preserve the existing diagnostic cache')
            record['calculation']='log_beta_studentt_normalizer'
            with beta_studentt_calculation():
                previous.evaluate(previous.checked(ROOT/ORIGINAL_PROTOCOL),ROOT/PREPARED,args.precision,cache,record,checkpoint)
        else:
            previous.compare(args.left,args.right,record)
            record['calculation']='log_beta_studentt_normalizer_precision_comparison'
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
    parser.add_argument('action',choices=['freeze','primitive','network','compare'])
    parser.add_argument('--protocol',required=True);parser.add_argument('--precision',choices=previous.PRECISIONS,default='float32')
    parser.add_argument('--scope');parser.add_argument('--output');parser.add_argument('--cache')
    parser.add_argument('--left');parser.add_argument('--right')
    args=parser.parse_args()
    if args.action=='freeze':freeze(args.protocol)
    else:
        result=execute(args)
        print(json.dumps({k:result[k] for k in ('state','action','seconds','process_max_rss_kib')}))
