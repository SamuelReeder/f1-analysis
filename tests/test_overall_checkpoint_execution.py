"""Exercise real process boundaries without fitting or selecting a retry."""
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

from analysis.checkpoint_execution import (
    STAGES, TerminalExecutionFailure, execute, process_identity, require_stopped,
)
from f1rank.overall.checkpoints import canonical, digest

ROOT = Path(__file__).resolve().parents[1]


def validate(request, name, path, value):
    assert value == dict(attempt_key=request['attempt_key'], stage=name, converged=False)
    return dict(attempt_key=value['attempt_key'], converged=value['converged'])


def make_request(directory, *, fail=False, missing=False, hold=False):
    stages = []
    for name in STAGES:
        output = directory / name / 'result.json'
        code = ('import json,os,time;from pathlib import Path;'
                f'p=Path({str(output)!r});'
                f'r={dict(attempt_key='a'*64, stage=name, converged=False)!r};'
                'p.write_text(json.dumps(r));')
        if name == STAGES[0] and missing:
            code = 'pass'
        if name == STAGES[0] and fail:
            code += 'raise SystemExit(9)'
        if name == STAGES[0] and hold:
            code += 'time.sleep(120)'
        stages.append(dict(name=name, command=[sys.executable, '-c', code],
                           environment=dict(PATH=os.defpath), result=f'{name}/result.json'))
    return dict(attempt_key='a'*64, working_directory=str(ROOT), stages=stages)


def run(directory, request, verify=lambda: None):
    return execute(directory, request, verify_inputs=verify, validate_result=validate)


def test_sequential_workers_preserve_failed_diagnostic_and_cache(tmp_path):
    directory = tmp_path / 'attempt'; request = make_request(directory)
    receipt = run(directory, request)
    assert receipt['state'] == 'complete'
    assert not receipt['publication_allowed'] and not receipt['attempt_selection_performed']
    assert [s['name'] for s in receipt['stages']] == list(STAGES)
    assert all(s['verified_summary']['converged'] is False for s in receipt['stages'])
    children = receipt['invocations'][0]['children']
    assert len(children) == len(STAGES) and all(c['exit_code'] == 0 for c in children)
    assert children[0]['process']['pid'] != children[1]['process']['pid']
    before = (directory / 'receipt.json').read_bytes()
    assert run(directory, request) == receipt
    assert (directory / 'receipt.json').read_bytes() == before


@pytest.mark.parametrize('mode', ['fail', 'missing'])
def test_observed_execution_failure_is_never_rerun(tmp_path, mode):
    directory = tmp_path / 'attempt'; request = make_request(directory, **{mode: True})
    with pytest.raises((TerminalExecutionFailure, FileNotFoundError)):
        run(directory, request)
    before = (directory / 'receipt.json').read_bytes()
    receipt = json.loads(before)
    assert receipt['state'] == 'failed_execution'
    assert not (directory / 'diagnostics').exists()
    with pytest.raises(TerminalExecutionFailure):
        run(directory, request)
    assert (directory / 'receipt.json').read_bytes() == before


@pytest.mark.parametrize('change', ['request', 'result'])
def test_completed_request_and_result_are_immutable(tmp_path, change):
    directory = tmp_path / 'attempt'; request = make_request(directory)
    run(directory, request)
    if change == 'request':
        request = copy.deepcopy(request); request['attempt_key'] = 'b'*64
    else:
        path = directory / request['stages'][0]['result']
        path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError):
        run(directory, request)


def test_stale_status_and_free_lock_do_not_prove_owner_dead(tmp_path):
    directory = tmp_path / 'attempt'; request = make_request(directory)
    run(directory, request)
    path = directory / 'receipt.json'; receipt = json.loads(path.read_text())
    receipt['state'] = 'running'; path.write_bytes(canonical(receipt))
    with pytest.raises(RuntimeError, match='still alive'):
        run(directory, request)


def test_process_identity_rejects_live_pid_and_accepts_previous_incarnation():
    own = process_identity(os.getpid())
    with pytest.raises(RuntimeError, match='still alive'):
        require_stopped(own)
    require_stopped(dict(own, start_ticks=str(int(own['start_ticks']) + 1)))
    require_stopped(dict(own, boot_id='previous-boot'))


def _wait_for(predicate, timeout=10):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError('Fixture did not reach its process boundary')


def _sealed_fixture(path):
    try:
        return json.loads(path.read_bytes()).get('converged') is False
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def test_parent_death_requires_child_death_then_reconciles_sealed_result(tmp_path):
    directory = tmp_path / 'attempt'; request = make_request(directory, hold=True)
    definition = tmp_path / 'input.json'; definition.write_bytes(canonical(request))
    code = ('import json,sys;from pathlib import Path;'
            'from analysis.checkpoint_execution import execute;'
            'defn=json.loads(Path(sys.argv[2]).read_bytes());'
            'execute(sys.argv[1],defn,verify_inputs=lambda:None,'
            'validate_result=lambda req,name,path,value:value)')
    parent = subprocess.Popen([sys.executable, '-c', code, str(directory), str(definition)],
                              cwd=ROOT, start_new_session=True)
    child = None
    try:
        _wait_for(lambda: _sealed_fixture(directory / STAGES[0] / 'result.json'))
        receipt = json.loads((directory / 'receipt.json').read_text())
        child = receipt['invocations'][0]['children'][0]['process']
        parent.kill(); parent.wait()
        with pytest.raises((BlockingIOError, RuntimeError)):
            run(directory, request)
        os.killpg(child['pid'], signal.SIGKILL)
        _wait_for(lambda: _is_stopped(child))
        result = run(directory, request)
        assert result['state'] == 'complete'
        assert result['stages'][0]['recovered_sealed_result']
        assert len(result['invocations']) == 2
        assert len(result['invocations'][1]['children']) == 1
        assert result['invocations'][1]['children'][0]['stage'] == 'diagnostics'
    finally:
        if parent.poll() is None:
            parent.kill(); parent.wait()
        if child is not None:
            try:
                os.killpg(child['pid'], signal.SIGKILL)
            except ProcessLookupError:
                pass


def _is_stopped(identity):
    try:
        require_stopped(identity)
    except RuntimeError:
        return False
    return True


def test_start_barrier_prevents_unrecorded_worker(tmp_path):
    directory = tmp_path / 'attempt'; directory.mkdir()
    request = make_request(directory)
    definition = directory / 'request.json'; definition.write_bytes(canonical(request))
    with (directory / 'lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        child = subprocess.Popen([sys.executable, '-m', 'analysis.checkpoint_execution', str(definition),
                                  STAGES[0], str(lock.fileno()), digest(definition)], cwd=ROOT,
                                 pass_fds=(lock.fileno(),), stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = child.communicate(b'', timeout=10)
    assert child.returncode != 0 and b'before authorizing' in stderr
    assert not (directory / STAGES[0] / 'result.json').exists()


def test_changed_sources_after_sampling_prevent_diagnostics(tmp_path):
    directory = tmp_path / 'attempt'; request = make_request(directory)
    def verify():
        if (directory / STAGES[0] / 'result.json').exists():
            raise ValueError('bound source changed')
    with pytest.raises(ValueError, match='bound source changed'):
        run(directory, request, verify)
    assert not (directory / 'diagnostics').exists()
    assert json.loads((directory / 'receipt.json').read_text())['state'] == 'failed_execution'
