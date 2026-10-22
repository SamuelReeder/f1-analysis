from datetime import datetime,timezone
import json
from pathlib import Path
import time

from analysis.checkpoint_attempt import profile
from f1rank.overall.checkpoints import canonical

protocol=profile.ROOT/'docs/overall_checkpoint_attempt_profile_v2_protocol.json'
out=profile.ROOT/'outputs/championship/development/overall-checkpoint-fit-preflight-v2'
p,original=profile.checked(protocol)
actual=profile.actual_resource(p,'sample_and_archive')
parent,_=profile.replay.checked(profile.ROOT/profile.PARENT)
started=time.monotonic()
observed=profile.replay.precision.scope(parent,original,'gpu64_a',actual['unit'])
from f1rank.overall.batched_rotation import prepare
from f1rank.overall.control_rotation import build_rotation
import jax
import jax.numpy as jnp
rows=[]
for case in p['studies']:
    study=next(row for row in original['studies'] if row['case']==case)
    path=profile.ROOT/f'{profile.replay.precision.RESULTS}/prepare/cpu/{case}/result.json'
    prepared,blocks,coordinates=profile.base.load_prepared(original,study,path,
        profile.base.sha(profile.ROOT/profile.replay.precision.PARENT))
    del coordinates
    rotation=prepare(build_rotation(blocks['laps']))
    if rotation.fingerprint()!=prepared['control_rotation_sha256']:
        raise ValueError('Frozen control basis changed')
    arguments=tuple({k:jnp.asarray(profile.replay.precision.lift(v))
                     if hasattr(v,'dtype') or type(v) is float else v
                     for k,v in blocks[name].items()} for name in ('qualifying','laps','outcomes'))
    jax.block_until_ready(arguments)
    rows.append(dict(case=case,state='complete',arrays_key=prepared['arrays_key'],
                     prepared_sha256=profile.base.sha(path),control_rotation_sha256=rotation.fingerprint()))
    del arguments,blocks,rotation
profile.checked(protocol)
result=dict(protocol_sha256=profile.base.sha(protocol),state='complete',studies=rows,
            actual_scope=observed,resource=profile.observe_resource(actual,gpu=True),
            seconds=time.monotonic()-started,recorded_at=datetime.now(timezone.utc).isoformat(),
            mcmc_executed=False,scientific_acceptance_established=False,publication_allowed=False)
(out/'result.json').write_bytes(canonical(result))
print(json.dumps(result))
