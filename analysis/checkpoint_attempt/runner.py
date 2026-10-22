"""Sequential service controller for the newly frozen synthetic computational run."""
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time

from analysis.checkpoint_execution import _write, process_identity, require_stopped
from .controller import run_ladder
from . import profile


def _prior_stopped(invocation):
    require_stopped(invocation['owner'])
    fields = dict(line.split('=',1) for line in subprocess.check_output([
        'systemctl','--user','show',invocation['unit'],'-p','ActiveState','-p','MainPID'],text=True).splitlines())
    if fields.get('ActiveState') not in ('inactive','failed') or fields.get('MainPID') != '0':
        raise RuntimeError('The previous capped service must be authoritatively terminal before recovery')


def run(args):
    p,_ = profile.checked(args.protocol)
    actual = profile.actual_resource(p,'diagnostics')
    output = profile.ROOT / 'outputs/championship/development' / p['id']
    cache = profile.ROOT / 'outputs/championship/cache' / p['id']
    output.mkdir(parents=True,exist_ok=True)
    with (output/'.controller.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        path = output/'execution.json'
        if path.exists():
            record = json.loads(path.read_bytes())
            if record['protocol_sha256'] != profile.base.sha(args.protocol):
                raise ValueError('Cannot change a profile execution')
            if record.get('all_processed'):
                raise FileExistsError('This profile execution is terminal; preserve it')
            for invocation in record['invocations']:
                _prior_stopped(invocation)
        else:
            record = dict(protocol_sha256=profile.base.sha(args.protocol),studies=[],invocations=[],
                          publication_allowed=False,full_recovery_complete=False,sbc_complete=False,
                          end_to_end_refresh_benchmark=False,all_processed=False)
        identity = subprocess.check_output(['systemctl','--user','show',actual['unit'],
            '-p','InvocationID','--value'],text=True).strip()
        invocation = dict(owner=process_identity(__import__('os').getpid()),unit=actual['unit'],
            invocation_id=identity,actual_scope=actual,started_at=datetime.now(timezone.utc).isoformat())
        record['invocations'].append(invocation); _write(path,record)
        begun = time.monotonic()
        def verify():
            current,_ = profile.checked(args.protocol)
            if current != p:
                raise ValueError('Frozen profile changed')
        for case in p['studies']:
            row = dict(case=case,state='running')
            record['studies'] = [r for r in record['studies'] if r['case'] != case] + [row]
            _write(path,record)
            try:
                result = run_ladder(cache/case,p,case,
                    specification_for=lambda settings:profile.specification(args.protocol,p,case,settings),
                    request_for=lambda directory,spec:profile.request(directory,spec,args.protocol,p),
                    verify_inputs=verify)
                result_path = cache/case/'result.json'
                row.update(state=result['state'],attempts=result['attempts'],
                    result=str(result_path.relative_to(profile.ROOT)),result_sha256=profile.base.sha(result_path))
            except Exception as exc:
                row.update(state='failed_execution',error=type(exc).__name__,message=str(exc))
            _write(path,record)
            print(json.dumps(row),flush=True)
        verify()
        invocation.update(finished_at=datetime.now(timezone.utc).isoformat(),seconds=time.monotonic()-begun,
                          final_resource=profile.observe_resource(actual))
        record.update(all_processed=True,all_numerical_gates_passed=all(
            r['state']=='numerically_converged' for r in record['studies']))
        _write(path,record)
        return 0 if record['all_numerical_gates_passed'] else 1
