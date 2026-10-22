"""A newly registered computational profile using the tested model/attempt path.

All studies are synthetic, including the real-network design. No profile result
can authorize a driver ranking, calibration claim, or production deployment.
"""
from datetime import datetime, timezone
from importlib.metadata import distribution
import json
import os
from pathlib import Path
import subprocess

from analysis import checkpoint_model_replay as replay
from analysis import overall_backend_compare as base
from f1rank.overall.fitting import committed_registration
from f1rank.overall.provenance import snapshot
from . import attempt_key

ROOT = base.ROOT
PARENT = 'docs/overall_checkpoint_model_replay_protocol.json'
REPLAY = 'outputs/championship/development/overall-checkpoint-model-replay-execution-v2'
CHECKS = 'outputs/championship/development/overall-checkpoint-attempt-checks-v1'
SOURCES = ('analysis/checkpoint_attempt/__init__.py', 'analysis/checkpoint_attempt/controller.py',
    'analysis/checkpoint_attempt/profile.py', 'analysis/checkpoint_attempt/worker.py',
    'analysis/checkpoint_attempt/__main__.py', 'analysis/checkpoint_attempt/runner.py',
    'analysis/checkpoint_execution/__init__.py',
    'analysis/checkpoint_execution/__main__.py', 'analysis/checkpoint_diagnostics/__init__.py',
    'tests/test_overall_checkpoint_attempt.py', 'tests/test_overall_checkpoint_attempt_profile.py',
    'docs/overall_checkpoint_attempt.md', 'docs/overall_checkpoint_attempt_profile.md')


def definition(parent, original):
    return dict(id='overall-checkpoint-fit-profile-v1', schema_version=2,
        status='frozen', purpose='numerical_profile', profile='deterministic_gpu_checkpoint_complete_design',
        inference_method=original['inference_method'],
        registered_model_inputs_key=original['registered_model_inputs_key'],
        studies=parent['studies'], synthetic=original['synthetic'],
        fit_settings=original['fit_settings'], retry=original['retry'], diagnostics=original['diagnostics'],
        chunk_draws=parent['chunk_draws'], chain_method=parent['chain_method'],
        environments=dict(sample_and_archive=parent['environments']['gpu64_a'],
                          diagnostics=parent['environments']['cpu64_a']),
        unset_environment=parent['unset_environment'], resources=parent['resources'],
        shared_slice=dict(CPUQuotaPerSecUSec='4s', CPUWeight='10'), gpu_admission=parent['gpu_admission'],
        interpreters=dict(sample_and_archive=parent['interpreter'],
                         diagnostics='/home/sreeder/f1-analysis/.venv/bin/python'),
        path='/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib',
        full_recovery_complete=False, sbc_complete=False, publication_allowed=False,
        end_to_end_refresh_benchmark=False, changes_original_results=False)


def evidence(parent):
    return [PARENT, base.REGISTRATION, f'{REPLAY}/comparison.json', f'{REPLAY}/terminal.json',
        f'{CHECKS}/run.json', f'{CHECKS}/stdout.txt',
        *[f'{REPLAY}/sample/{case}/resume/result.json' for case in parent['studies']],
        *[f'{replay.precision.RESULTS}/prepare/cpu/{case}/result.json' for case in parent['studies']]]


def catalogs(parent, original):
    reviewed = base.read(Path(original['reference_root']) / base.REVIEW)
    return {row['case']:row['receipt']['catalog'] for row in reviewed['studies']
            if row['case'] in parent['studies']}


def freeze(destination):
    parent, original = replay.checked(ROOT / PARENT)
    previous = committed_registration(ROOT / base.REGISTRATION)
    comparison = base.read(ROOT / REPLAY / 'comparison.json')
    terminal = base.read(ROOT / REPLAY / 'terminal.json')
    checks = base.read(ROOT / CHECKS / 'run.json')
    if (not comparison['all_replays_passed'] or not comparison['all_processed']
            or [r['case'] for r in comparison['studies']] != parent['studies']
            or not terminal['all_phases_processed_and_expected']
            or checks['exit_code'] != 0 or not checks['sources_unchanged']):
        raise ValueError('Require complete replay and focused attempt verification')
    for name,expected in checks['source_sha256'].items():
        if base.committed(ROOT,name) != expected:
            raise ValueError('The tested attempt implementation changed')
    diagnostic = distribution('arviz-stats')
    runtime = {str(diagnostic.locate_file(name)):base.sha(diagnostic.locate_file(name))
        for name in diagnostic.files if str(name).startswith('arviz_stats/') and str(name).endswith('.py')}
    if not runtime:
        raise ValueError('Bind the complete installed diagnostic Python sources')
    model = snapshot()
    if model['key'] != previous['registered_model_inputs_key']:
        raise ValueError('The frozen complete model or data changed')
    value = dict(definition(parent, previous), registered_at=datetime.now(timezone.utc).isoformat(),
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        source_sha256={n:base.committed(ROOT,n) for n in SOURCES},
        evidence_sha256={n:base.committed(ROOT,n) for n in evidence(parent)},
        runtime_sha256=runtime, catalogs=catalogs(parent,original))
    with Path(destination).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def checked(path):
    path = Path(path).resolve()
    registration = committed_registration(path)
    parent, original = replay.checked(ROOT / PARENT)
    previous = committed_registration(ROOT / base.REGISTRATION)
    if (any(registration.get(k) != v for k,v in definition(parent,previous).items())
            or set(registration['source_sha256']) != set(SOURCES)
            or set(registration['evidence_sha256']) != set(evidence(parent))
            or registration['catalogs'] != catalogs(parent,original)
            or snapshot()['key'] != registration['registered_model_inputs_key']):
        raise ValueError('Changed complete-model computational procedure')
    for name,expected in {**registration['source_sha256'],**registration['evidence_sha256']}.items():
        if base.sha(ROOT/name) != expected:
            raise ValueError(f'Changed profile input: {name}')
    if any(base.sha(name) != expected for name,expected in registration['runtime_sha256'].items()):
        raise ValueError('Changed CPU diagnostic runtime')
    return registration, original


def specification(protocol, registration, case, settings):
    if case not in registration['studies']:
        raise ValueError('Study is not in the complete registered inventory')
    prepared_path = ROOT / f'{replay.precision.RESULTS}/prepare/cpu/{case}/result.json'
    prepared = base.read(prepared_path)
    replayed = base.read(ROOT / REPLAY / f'sample/{case}/resume/result.json')
    axes = [settings['chains'],settings['samples']]
    import numpy as np
    fields = dict(posterior={name:dict(shape=[*axes,*info['shape']],
        dtype=np.dtype('float64').str if np.dtype(info['dtype']).kind=='f' else info['dtype'])
        for name,info in prepared['collector_inventory'].items()},
        extras={name:dict(shape=axes,dtype=info['dtype'])
                for name,info in replayed['archive']['inventory']['extras'].items()})
    return dict(protocol_sha256=base.sha(protocol), case=case, settings=settings,
        prepared=str(prepared_path.relative_to(ROOT)), prepared_sha256=base.sha(prepared_path),
        catalog=registration['catalogs'][case], inventory=fields,
        chunk_draws=registration['chunk_draws'], chain_method=registration['chain_method'])


def request(directory, spec, protocol, registration):
    directory = Path(directory).resolve()
    from .controller import settings_inventory
    index = settings_inventory(registration,spec['case']).index(spec['settings']) + 1
    return dict(attempt_key=attempt_key(spec),working_directory=str(ROOT),stages=[dict(
        name=name, command=[registration['interpreters'][name],'-u','-m','analysis.checkpoint_attempt',
            'worker','--protocol',str(Path(protocol).resolve()),'--directory',str(directory),
            '--case',spec['case'],'--attempt',str(index),'--stage',name],
        environment=dict(registration['environments'][name],PATH=registration['path'],PYTHONUNBUFFERED='1'),
        result=name+'/result.json') for name in ('sample_and_archive','diagnostics')])


def actual_resource(registration, stage):
    from analysis.overall_complete_profile_diagnostics import actual_scope
    membership = next(line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines()
                      if line.startswith('0::'))
    unit = Path(membership).name
    actual = actual_scope(unit)
    current = dict(line.split('=',1) for line in subprocess.check_output(
        ['systemctl','--user','show','f1.slice','-p','CPUQuotaPerSecUSec','-p','CPUWeight'],text=True).splitlines())
    if current != registration['shared_slice']:
        raise ValueError('The shared CPU quota or priority changed')
    own = Path('/sys/fs/cgroup') / actual['ControlGroup'].lstrip('/')
    for sibling in own.parent.iterdir():
        if sibling.is_dir() and sibling != own and (sibling/'cgroup.events').exists():
            if dict(line.split() for line in (sibling/'cgroup.events').read_text().splitlines()).get('populated') != '0':
                raise ValueError('Other heavy work is active')
    if (any(os.environ.get(n) != v for n,v in registration['environments'][stage].items())
            or any(os.environ.get(n) for n in registration['unset_environment'])):
        raise ValueError('Use the exact registered worker environment')
    return dict(actual,unit=unit,shared_slice=current)


def observe_resource(actual, *, gpu=False):
    group = Path('/sys/fs/cgroup') / actual['ControlGroup'].lstrip('/')
    row = dict(recorded_at=datetime.now(timezone.utc).isoformat())
    for name in ('memory.current','memory.peak','memory.max'):
        row[name] = (group/name).read_text().strip()
    for name in ('memory.events','cpu.stat'):
        row[name] = dict(line.split() for line in (group/name).read_text().splitlines())
    if gpu:
        import jax
        # Supported allocator stats include a peak since this interpreter's
        # context began. Preserve all keys and absence rather than invent a peak.
        row['device_allocator_stats'] = [d.memory_stats() for d in jax.local_devices()]
    return row
