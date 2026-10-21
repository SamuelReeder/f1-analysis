"""Instrumented collection must preserve the actual NumPyro draw sequence."""
import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from analysis.checkpoint_sampling import run_chunks
from test_overall_checkpoints import small_model, numerical_settings, collected


@pytest.mark.parametrize('chains,method',[(1,'parallel'),(2,'parallel'),(2,'vectorized')])
def test_staged_collection_matches_uninterrupted_numpyro(tmp_path,chains,method):
    import jax
    import jax.numpy as jnp
    from numpyro.infer import MCMC,NUTS,init_to_median
    from f1rank.overall.inference import EXTRA_FIELDS
    settings=dict(numerical_settings(),chains=chains)
    data=jnp.array([-.3,.1,.6]);progress=[]
    result=run_chunks(small_model,(data,),kwargs={},settings=settings,run_key=hashlib.sha256(b'staged-fixture').hexdigest(),
        directory=tmp_path,chunk_draws=7,verify_inputs=lambda:None,chain_method=method,on_checkpoint=progress.append)
    sampler=MCMC(NUTS(small_model,target_accept_prob=settings['target_accept'],max_tree_depth=settings['max_tree_depth'],
        init_strategy=init_to_median(num_samples=settings['init_median_samples'])),num_warmup=settings['warmup'],
        num_samples=settings['samples'],num_chains=chains,chain_method=method,progress_bar=False)
    sampler.run(jax.random.PRNGKey(settings['seed']),data,extra_fields=EXTRA_FIELDS)
    post,extras=collected(tmp_path)
    for actual,expected in ((post,sampler.get_samples(group_by_chain=True)),(extras,sampler.get_extra_fields(group_by_chain=True))):
        assert set(actual)==set(expected)
        for n in actual: np.testing.assert_array_equal(actual[n],expected[n])
    assert result['state']=='draws_complete' and result['publication_allowed'] is False
    assert progress[-1]['draws']==settings['samples']
    rows=[r for r in result['timings'] if r['stage']!='initialization']
    assert any(r['reused_executable'] for r in rows)
    assert all(all(r[k]>=0 for k in ('lowering_seconds','compilation_seconds','synchronized_execution_seconds',
                                   'host_transfer_seconds','checkpoint_seconds')) for r in rows)
    jax.clear_caches()


@pytest.mark.parametrize('phase',['warmup','sampling'])
def test_fresh_process_restart_preserves_full_sequence(tmp_path,phase):
    script=tmp_path/'child.py'
    script.write_text('''
import hashlib,sys
from pathlib import Path
import jax.numpy as jnp
from analysis.checkpoint_sampling import run_chunks
from test_overall_checkpoints import small_model,numerical_settings
def stop(row):
 if ((sys.argv[2]=='warmup' and row['warmup_completed']==7 and row['draws']==0)
     or (sys.argv[2]=='sampling' and row['draws']==7)): raise SystemExit(23)
run_chunks(small_model,(jnp.array([-.3,.1,.6]),),kwargs={},settings=dict(numerical_settings(),chains=2),
 run_key=hashlib.sha256(b'staged-fixture').hexdigest(),directory=Path(sys.argv[1]),chunk_draws=7,
 verify_inputs=lambda:None,chain_method='vectorized',on_checkpoint=stop)
''')
    root=Path(__file__).resolve().parents[1]
    env=dict(os.environ,PYTHONPATH=os.pathsep.join((str(root),str(root/'tests'))))
    env['XLA_FLAGS']=env.get('XLA_FLAGS','').replace('--xla_force_host_platform_device_count=2',
                                                  '--xla_force_host_platform_device_count=1')
    resumed,uninterrupted=tmp_path/'resumed',tmp_path/'uninterrupted'
    first=subprocess.run([sys.executable,str(script),str(resumed),phase],env=env,capture_output=True,text=True,timeout=240)
    assert first.returncode==23,first.stderr
    for directory in (resumed,uninterrupted):
        r=subprocess.run([sys.executable,str(script),str(directory),'complete'],env=env,capture_output=True,text=True,timeout=240)
        assert r.returncode==0,r.stderr
    for actual,expected in zip(collected(resumed),collected(uninterrupted)):
        assert actual.keys()==expected.keys()
        for name in actual: np.testing.assert_array_equal(actual[name],expected[name])
