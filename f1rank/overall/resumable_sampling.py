"""Unselected development path for warmup and same-target continuation.

No existing registration or production runner invokes this module. Completion
means all declared draws were stored, not convergence or scientific acceptance.
Warmup retains the original total adaptation schedule across checkpoint chunks.
"""
import hashlib
import inspect
import os
import platform
import time

from .checkpoints import SCHEMA, Checkpoints, canonical, digest


def runtime_identity():
    import jax
    import jaxlib
    import numpy as np
    import numpyro
    from jax._src.lax import lax
    from numpyro.infer import hmc, hmc_util, mcmc, util
    return dict(
        python=platform.python_version(), jax=jax.__version__, jaxlib=jaxlib.__version__,
        numpy=np.__version__, numpyro=numpyro.__version__,
        backend=jax.default_backend(), x64=bool(jax.config.jax_enable_x64),
        matmul_precision=jax.config.jax_default_matmul_precision,
        devices=[str(d.device_kind) for d in jax.local_devices()],
        sampler_sources={module.__name__: digest(inspect.getfile(module))
                         for module in (hmc, hmc_util, mcmc, util, lax)},
        environment={name: os.environ.get(name) for name in (
            "JAX_PLATFORMS", "XLA_FLAGS", "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "JAX_ENABLE_COMPILATION_CACHE",
            "CUDA_VISIBLE_DEVICES", "JAX_CUDA_VISIBLE_DEVICES",
            "XLA_PYTHON_CLIENT_ALLOCATOR", "XLA_CLIENT_MEM_FRACTION",
            "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_PYTHON_CLIENT_PREALLOCATE")})


def run_chunks(model, args, *, kwargs, settings, run_key, directory, chunk_draws,
               verify_inputs, chain_method="parallel"):
    """Continue only the exact attempt; callers own registration and acceptance.

    ``verify_inputs`` must validate the frozen registration and complete current
    source/data/array identity against ``run_key``. It is called before loading,
    after each sampling operation and before returning. No checkpoint can turn a
    failed attempt into another retry. The caller must also enforce terminal
    attempt state before invoking this unselected development implementation.
    """
    import jax
    import numpy as np
    from numpyro.infer import MCMC, NUTS, init_to_median
    from .inference import EXTRA_FIELDS, validate_extras
    from .fitting import FIT_SETTINGS

    if (set(settings) != set(FIT_SETTINGS) or type(chunk_draws) is not int or chunk_draws < 1
            or any(type(settings[k]) is not int or settings[k] < 1
                   for k in ("chains", "samples", "warmup", "max_tree_depth", "init_median_samples"))
            or type(settings["seed"]) is not int or not 0 <= settings["seed"] < 2**32
            or not 0 < settings["target_accept"] < 1):
        raise ValueError("Explicit complete sampler and chunk settings required")
    if not isinstance(run_key, str) or len(run_key) != 64 or any(c not in "0123456789abcdef" for c in run_key):
        raise ValueError("A source/data/registration-bound attempt key is required")
    if chain_method not in ("parallel", "vectorized"):
        raise ValueError("Select parallel or vectorized chains explicitly")
    if chain_method == "parallel" and jax.local_device_count() < settings["chains"]:
        raise ValueError("Insufficient chain devices; do not silently change execution")
    verify_inputs()
    identity = dict(schema=SCHEMA, run_key=run_key, settings=settings, chunk_draws=chunk_draws,
                    chain_method=chain_method,
                    runtime=runtime_identity(), implementation={
                        "resumable_sampling": digest(__file__),
                        "checkpoints": digest(inspect.getfile(Checkpoints))})
    timings = []
    with Checkpoints(directory, identity) as saved:
        state = saved.state()
        verify_inputs()
        resumed_draws = saved.draws
        resumed_warmup = saved.warmup_completed
        if saved.draws > settings["samples"]:
            raise ValueError("Checkpoint has more than the registered draw count")
        kernel = NUTS(model, target_accept_prob=settings["target_accept"],
                      max_tree_depth=settings["max_tree_depth"],
                      init_strategy=init_to_median(num_samples=settings["init_median_samples"]))

        # Reuse each collection shape's compiled sampler within this process.
        samplers = {}
        def sampler(count, *, warmup=False):
            key = (count, warmup)
            if key not in samplers:
                samplers[key] = MCMC(kernel, num_warmup=settings["warmup"],
                    num_samples=count, num_chains=settings["chains"],
                    chain_method=chain_method, progress_bar=False,
                    postprocess_fn=(lambda values: {}) if warmup else None)
            return samplers[key]

        if state is None:
            verify_inputs()
            started = time.monotonic()
            initial_key = jax.random.PRNGKey(settings["seed"])
            def initialize(key):
                return kernel.init(key, settings["warmup"], init_params=None,
                                   model_args=args, model_kwargs=kwargs)
            if settings["chains"] == 1:
                state = initialize(initial_key)
            elif chain_method == "vectorized":
                # NumPyro initializes the full batched state and vectorizes its
                # transition kernel when given a batch of chain keys.
                state = initialize(jax.random.split(initial_key, settings["chains"]))
            else:
                state = jax.pmap(initialize)(jax.random.split(initial_key, settings["chains"]))
            state = jax.block_until_ready(state)
            verify_inputs()
            saved.append(state)
            timings.append(dict(stage="initialization_and_checkpoint", seconds=time.monotonic()-started))
        while saved.warmup_completed < settings["warmup"]:
            verify_inputs()
            count = min(chunk_draws, settings["warmup"]-saved.warmup_completed)
            started = time.monotonic()
            current = sampler(count, warmup=True)
            # NumPyro's collection label does not control HMC adaptation. The
            # kernel retains the original total warmup and its saved iteration,
            # adaptation windows and random keys. These draws are discarded.
            current.post_warmup_state = state
            current.run(state.rng_key, *args, extra_fields=EXTRA_FIELDS, **kwargs)
            state = jax.block_until_ready(current.last_state)
            verify_inputs()
            saved.append(state)
            timings.append(dict(stage="warmup_and_checkpoint", steps=count,
                                seconds=time.monotonic()-started))
        while saved.draws < settings["samples"]:
            verify_inputs()
            count = min(chunk_draws, settings["samples"]-saved.draws)
            started = time.monotonic()
            current = sampler(count)
            current.post_warmup_state = state
            current.run(state.rng_key, *args, extra_fields=EXTRA_FIELDS, **kwargs)
            state = jax.block_until_ready(current.last_state)
            posterior = {k: np.asarray(v) for k, v in current.get_samples(group_by_chain=True).items()}
            extras = {k: np.asarray(v) for k, v in current.get_extra_fields(group_by_chain=True).items()}
            validate_extras(extras, dict(settings, samples=count))
            verify_inputs()
            saved.append(state, posterior=posterior, extras=extras)
            timings.append(dict(stage="sampling_and_checkpoint", draws=count,
                                seconds=time.monotonic()-started))
        verify_inputs()
        return dict(state="draws_complete", publication_allowed=False, convergence_evaluated=False,
                    identity_key=hashlib.sha256(canonical(identity)).hexdigest(),
                    resumed_draws=resumed_draws, resumed_warmup=resumed_warmup,
                    total_draws=saved.draws, total_warmup=saved.warmup_completed,
                    checkpoints=len(saved.records), timings=timings)
