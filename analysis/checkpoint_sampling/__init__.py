"""Unselected instrumented checkpoint sampler with explicit chunk compilation.

Uses the installed NumPyro collection implementation. Its private API source is
bound to checkpoint identity, and numerical replay must be verified before use.
No production registration selects this path and it supplies no acceptance gate.
"""
from functools import partial
import hashlib
import inspect
import time

from f1rank.overall.checkpoints import SCHEMA, Checkpoints, canonical, digest
from f1rank.overall.resumable_sampling import runtime_identity


def _signature(state):
    import jax
    leaves, tree = jax.tree.flatten(state)
    return (str(tree), tuple((tuple(x.shape), str(x.dtype), bool(getattr(x,'weak_type',False)))
                            if hasattr(x,'shape') else (type(x).__name__, x) for x in leaves))


def run_chunks(model, args, *, kwargs, settings, run_key, directory, chunk_draws,
               verify_inputs, chain_method='parallel', on_checkpoint=None):
    import jax
    import jax.numpy as jnp
    import numpy as np
    import numpyro.util
    from numpyro.infer import MCMC, NUTS, init_to_median
    from f1rank.overall.fitting import FIT_SETTINGS
    from f1rank.overall.inference import EXTRA_FIELDS, validate_extras

    if (set(settings) != set(FIT_SETTINGS) or type(chunk_draws) is not int or chunk_draws < 1
            or any(type(settings[k]) is not int or settings[k] < 1
                   for k in ('chains','samples','warmup','max_tree_depth','init_median_samples'))
            or type(settings['seed']) is not int or not 0 <= settings['seed'] < 2**32
            or not 0 < settings['target_accept'] < 1):
        raise ValueError('Explicit complete sampler and chunk settings required')
    if not isinstance(run_key,str) or len(run_key)!=64 or any(c not in '0123456789abcdef' for c in run_key):
        raise ValueError('A source/data/registration-bound attempt key is required')
    if chain_method not in ('parallel','vectorized'):
        raise ValueError('Select parallel or vectorized chains explicitly')
    if chain_method=='parallel' and jax.local_device_count()<settings['chains']:
        raise ValueError('Insufficient chain devices; do not silently change execution')
    verify_inputs()
    identity = dict(schema=SCHEMA, run_key=run_key, settings=settings, chunk_draws=chunk_draws,
        chain_method=chain_method, runtime=runtime_identity(), implementation={
            'staged_sampling':digest(__file__), 'checkpoints':digest(inspect.getfile(Checkpoints)),
            'collection_runtime':digest(inspect.getfile(numpyro.util))})
    timings=[]
    with Checkpoints(directory,identity) as saved:
        state=saved.state();verify_inputs()
        resumed_draws,resumed_warmup=saved.draws,saved.warmup_completed
        kernel=NUTS(model,target_accept_prob=settings['target_accept'],max_tree_depth=settings['max_tree_depth'],
                    init_strategy=init_to_median(num_samples=settings['init_median_samples']))
        executables={}

        def notify(row):
            timings.append(row)
            if on_checkpoint is not None:
                on_checkpoint(dict(row,draws=saved.draws,warmup_completed=saved.warmup_completed,
                                   checkpoint_generation=len(saved.records)-1))

        if state is None:
            begun=time.monotonic();key=jax.random.PRNGKey(settings['seed'])
            def initialize(rng):
                return kernel.init(rng,settings['warmup'],init_params=None,model_args=args,model_kwargs=kwargs)
            if settings['chains']==1:
                state=initialize(key)
            elif chain_method=='vectorized':
                state=initialize(jax.random.split(key,settings['chains']))
            else:
                state=jax.pmap(initialize)(jax.random.split(key,settings['chains']))
            state=jax.block_until_ready(state)
            init_seconds=time.monotonic()-begun
            verify_inputs();begun=time.monotonic();saved.append(state)
            notify(dict(stage='initialization', initialization_including_compilation_seconds=init_seconds,
                        checkpoint_seconds=time.monotonic()-begun))

        while saved.warmup_completed<settings['warmup'] or saved.draws<settings['samples']:
            verify_inputs()
            warmup=saved.warmup_completed<settings['warmup']
            remaining=(settings['warmup']-saved.warmup_completed) if warmup else (settings['samples']-saved.draws)
            count=min(chunk_draws,remaining)
            key=(count,warmup,_signature(state))
            lower_seconds=compile_seconds=0.
            reused=key in executables
            if not reused:
                current=MCMC(kernel,num_warmup=settings['warmup'],num_samples=count,num_chains=settings['chains'],
                    chain_method=chain_method,progress_bar=False,
                    postprocess_fn=(lambda values:{}) if warmup else None)
                current._args,current._kwargs=args,kwargs
                current._set_collection_params(0,count,count,'sample')
                fields=tuple(dict.fromkeys((current._sample_field,*current._default_fields,*EXTRA_FIELDS)))
                mapping=partial(current._single_chain_mcmc,args=args,kwargs=kwargs,
                                collect_fields=fields,remove_sites=())
                def collect(st, mapping=mapping):
                    values=(st.rng_key,st,None)
                    if settings['chains']==1:
                        draws,last=mapping(values)
                        return jax.tree.map(lambda x:x[jnp.newaxis,...],draws),last
                    if chain_method=='parallel':
                        return jax.pmap(mapping)(values)
                    draws,last=mapping(values)
                    return jax.tree.map(lambda x:jnp.swapaxes(x,0,1),draws),last
                begun=time.monotonic();lowered=jax.jit(collect).lower(state)
                lower_seconds=time.monotonic()-begun
                begun=time.monotonic();executables[key]=lowered.compile()
                compile_seconds=time.monotonic()-begun
            begun=time.monotonic();draws,state=jax.block_until_ready(executables[key](state))
            execution_seconds=time.monotonic()-begun
            begun=time.monotonic()
            if warmup:
                posterior=extras=None
            else:
                posterior={n:np.asarray(v) for n,v in draws['z'].items()}
                extras={n:np.asarray(draws[n]) for n in EXTRA_FIELDS}
                validate_extras(extras,dict(settings,samples=count))
            transfer_seconds=time.monotonic()-begun
            verify_inputs();begun=time.monotonic()
            saved.append(state,posterior=posterior,extras=extras)
            notify(dict(stage='warmup' if warmup else 'sampling',steps=count,
                lowering_seconds=lower_seconds,compilation_seconds=compile_seconds,
                synchronized_execution_seconds=execution_seconds,host_transfer_seconds=transfer_seconds,
                checkpoint_seconds=time.monotonic()-begun,reused_executable=reused))
        verify_inputs()
        return dict(state='draws_complete',publication_allowed=False,convergence_evaluated=False,
            identity_key=hashlib.sha256(canonical(identity)).hexdigest(),resumed_draws=resumed_draws,
            resumed_warmup=resumed_warmup,total_draws=saved.draws,total_warmup=saved.warmup_completed,
            checkpoints=len(saved.records),timings=timings)
