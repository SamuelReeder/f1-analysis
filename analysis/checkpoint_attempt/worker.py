"""Registered full-model worker binding; no model-generation or sampler override."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import resource
import time
import traceback

from analysis.checkpoint_execution import _write
from . import diagnostic_stage, sample_stage
from .controller import settings_inventory
from . import profile


def worker(args):
    p, original = profile.checked(args.protocol)
    choices = settings_inventory(p,args.case)
    if not 1 <= args.attempt <= len(choices):
        raise ValueError('Attempt is not in the fixed registered retry inventory')
    settings = choices[args.attempt-1]
    spec = profile.specification(args.protocol,p,args.case,settings)
    directory = Path(args.directory).resolve()
    if directory != profile.ROOT / 'outputs/championship/cache' / p['id'] / args.case / f'attempt-{args.attempt:02d}':
        raise ValueError('Use the unique registered case/attempt directory')
    actual = profile.actual_resource(p,args.stage)
    record = dict(case=args.case,attempt=args.attempt,stage=args.stage,actual_scope=actual,
        started_at=datetime.now(timezone.utc).isoformat(),publication_allowed=False)
    begun = time.monotonic()
    def verify():
        current,_ = profile.checked(args.protocol)
        if current != p:
            raise ValueError('Registered profile changed during a worker stage')
    try:
        if args.stage == 'sample_and_archive':
            # Admission and actual device/precision checks precede any model work.
            parent,_ = profile.replay.checked(profile.ROOT/profile.PARENT)
            record['device_admission'] = profile.replay.precision.scope(
                parent,original,'gpu64_a',actual['unit'])
            study = next(s for s in original['studies'] if s['case']==args.case)
            prepared,blocks,coordinates = profile.base.load_prepared(original,study,
                profile.ROOT/spec['prepared'],profile.base.sha(profile.ROOT/profile.replay.precision.PARENT))
            del coordinates
            import jax.numpy as jnp
            from f1rank.overall.batched_rotation import prepare
            from f1rank.overall.control_rotation import build_rotation
            from f1rank.overall.weekend_coordinates import weekend_model
            rotation = prepare(build_rotation(blocks['laps']))
            if rotation.fingerprint() != prepared['control_rotation_sha256']:
                raise ValueError('Frozen control coordinate map changed')
            arguments = tuple({k:jnp.asarray(profile.replay.precision.lift(v))
                               if hasattr(v,'dtype') or type(v) is float else v
                               for k,v in blocks[name].items()}
                              for name in ('qualifying','laps','outcomes'))
            sample_stage(directory,spec,model=weekend_model(rotation),args=arguments,kwargs={},
                verify_inputs=verify,observe=lambda:profile.observe_resource(actual,gpu=True))
        else:
            diagnostic_stage(directory,spec,verify_inputs=verify,
                             observe=lambda:profile.observe_resource(actual))
        record['state'] = 'complete'
    except BaseException as exc:
        record.update(state='failed',error=type(exc).__name__,message=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        record.update(seconds=time.monotonic()-begun,process_max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      finished_at=datetime.now(timezone.utc).isoformat(),final_resource=profile.observe_resource(actual))
        stage_dir = directory/args.stage;stage_dir.mkdir(parents=True,exist_ok=True)
        _write(stage_dir/f'worker-{os.getpid()}-{time.time_ns()}.json',record)
