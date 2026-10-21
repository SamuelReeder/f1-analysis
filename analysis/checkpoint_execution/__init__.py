"""Durable, sequential execution of one registered checkpoint attempt.

Sampling/archive construction and CPU diagnostics run in separate interpreters.
This component does not choose retries or grant scientific acceptance. Its caller
must verify the frozen request and every result's complete identity and contents.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

from f1rank.overall.checkpoints import canonical, digest, _sync_directory

STAGES = ('sample_and_archive', 'diagnostics')


class TerminalExecutionFailure(RuntimeError):
    """An observed worker failure is evidence, not an automatic restart request."""


def process_identity(pid):
    """Linux process identity includes boot and start ticks, protecting PID reuse."""
    raw = Path(f'/proc/{pid}/stat').read_text()
    fields = raw[raw.rfind(')') + 2:].split()
    return dict(pid=pid, boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start_ticks=fields[19], state=fields[0])


def require_stopped(previous):
    if previous['boot_id'] != Path('/proc/sys/kernel/random/boot_id').read_text().strip():
        return
    try:
        observed = process_identity(previous['pid'])
    except FileNotFoundError:
        return
    # A zombie cannot execute or mutate artifacts. Unreadable process state is
    # deliberately not treated as evidence of death.
    if (observed['start_ticks'] == previous['start_ticks']
            and observed['state'] not in ('Z', 'X')):
        raise RuntimeError('A previous attempt process is still alive')


def _write(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix='.partial-', dir=path.parent)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _validate_request(request, directory):
    if (set(request) != {'attempt_key', 'working_directory', 'stages'}
            or not isinstance(request['attempt_key'], str)
            or len(request['attempt_key']) != 64
            or any(c not in '0123456789abcdef' for c in request['attempt_key'])
            or not Path(request['working_directory']).is_absolute()
            or [s.get('name') for s in request['stages']] != list(STAGES)):
        raise ValueError('Require one complete registered attempt and ordered stages')
    for stage in request['stages']:
        if (set(stage) != {'name', 'command', 'environment', 'result'}
                or not isinstance(stage['command'], list) or not stage['command']
                or not all(isinstance(x, str) and x for x in stage['command'])
                or not Path(stage['command'][0]).is_absolute()
                or not isinstance(stage['environment'], dict)
                or not all(isinstance(k, str) and isinstance(v, str) for k, v in stage['environment'].items())
                or stage['result'] != stage['name'] + '/result.json'):
            raise ValueError('Explicit interpreter, environment and stage result required')
        if (directory / stage['result']).resolve().parent != directory / stage['name']:
            raise ValueError('Stage result escaped the attempt directory')


def _result(directory, request, stage, validate_result):
    path = directory / stage['result']
    before = digest(path)
    value = json.loads(path.read_bytes())
    summary = validate_result(request, stage['name'], path, value)
    # The validator must check archive bytes, inventory and scientific identity;
    # a worker's status string alone is not enough to complete a stage.
    canonical(summary)
    if digest(path) != before:
        raise ValueError('Stage result changed during verification')
    return dict(result_sha256=before, verified_summary=summary)


def execute(directory, request, *, verify_inputs, validate_result):
    """Complete or resume the same attempt; never change its request or settings.

    The caller supplies source/runtime/registration verification and an archive
    validator. Resource admission belongs to that registered caller. Every child
    runs in the caller's capped service; only one child is live at a time.
    """
    directory = Path(directory).resolve()
    request = json.loads(canonical(request))
    _validate_request(request, directory)
    verify_inputs()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.execution.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        definition = directory / 'request.json'
        receipt_path = directory / 'receipt.json'
        if definition.exists():
            if definition.read_bytes() != canonical(request):
                raise ValueError('Cannot change an existing attempt request')
        else:
            if any(p.name != '.execution.lock' and not p.name.startswith('.partial-')
                   for p in directory.iterdir()):
                raise ValueError('Existing outputs without a registered request')
            _write(definition, request)
        key = hashlib.sha256(canonical(request)).hexdigest()
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_bytes())
            if receipt['request_sha256'] != key:
                raise ValueError('Execution belongs to another request')
            if receipt['state'] == 'failed_execution':
                raise TerminalExecutionFailure('Preserve the failed execution; do not rerun it')
            predecessor_sha256 = digest(receipt_path)
        else:
            receipt = dict(request_sha256=key, attempt_key=request['attempt_key'], state='running',
                           stages=[], invocations=[], publication_allowed=False,
                           attempt_selection_performed=False)
            predecessor_sha256 = None
        if receipt['state'] == 'complete':
            for stage, recorded in zip(request['stages'], receipt['stages'], strict=True):
                if recorded['name'] != stage['name'] or recorded['state'] != 'complete':
                    raise ValueError('Incomplete stage inventory')
                current = _result(directory, request, stage, validate_result)
                if any(recorded[k] != v for k, v in current.items()):
                    raise ValueError('Completed stage result changed')
            verify_inputs()
            return receipt
        for previous in receipt['invocations']:
            require_stopped(previous['owner'])
            for child in previous['children']:
                require_stopped(child['process'])
        if (receipt['state'] != 'running' or len(receipt['stages']) > len(STAGES)
                or any(row['name'] != STAGES[i] or row['state'] not in ('pending', 'running', 'complete')
                       for i, row in enumerate(receipt['stages']))):
            raise ValueError('Invalid interrupted stage inventory')
        invocation = dict(owner=process_identity(os.getpid()), started_at=_now(), children=[],
                          predecessor_receipt_sha256=predecessor_sha256)
        receipt['invocations'].append(invocation)
        _write(receipt_path, receipt)
        for index, stage in enumerate(request['stages']):
            verify_inputs()
            if digest(definition) != key:
                raise ValueError('Frozen execution request changed')
            if index < len(receipt['stages']):
                recorded = receipt['stages'][index]
                if recorded['name'] != stage['name']:
                    raise ValueError('Stage ordering changed')
            else:
                recorded = dict(name=stage['name'], state='pending')
                receipt['stages'].append(recorded)
                _write(receipt_path, receipt)
            result_path = directory / stage['result']
            if recorded['state'] == 'complete':
                observed = _result(directory, request, stage, validate_result)
                if any(recorded[k] != v for k, v in observed.items()):
                    raise ValueError('Completed stage result changed')
                continue
            if result_path.exists():
                # A worker may have sealed its result before its parent died.
                # Reconcile only after OS death checks and full validation.
                observed = _result(directory, request, stage, validate_result)
                recorded.update(state='complete', recovered_sealed_result=True, **observed)
                _write(receipt_path, receipt)
                continue
            stage_dir = directory / stage['name']
            stage_dir.mkdir(exist_ok=True)
            log = stage_dir / f'invocation-{len(receipt["invocations"]):06d}.log'
            recorded['state'] = 'running'; _write(receipt_path, receipt)
            child = None
            try:
                # The wrapper waits for stdin. The child identity is durably
                # recorded before allowing it to execute the numerical command.
                # It inherits the lock, including across exec, so an orphaned
                # worker prevents a concurrent replacement controller.
                with log.open('xb') as stream:
                    child = subprocess.Popen(
                        [sys.executable, '-m', 'analysis.checkpoint_execution', str(definition),
                         stage['name'], str(lock.fileno()), key], cwd=request['working_directory'],
                        env=stage['environment'], stdin=subprocess.PIPE, stdout=stream,
                        stderr=subprocess.STDOUT, pass_fds=(lock.fileno(),), start_new_session=True)
                    owned = dict(stage=stage['name'], process=process_identity(child.pid), log=str(log.relative_to(directory)))
                    invocation['children'].append(owned); _write(receipt_path, receipt)
                    child.stdin.write(b'1'); child.stdin.flush(); child.stdin.close()
                    status = child.wait()
                owned.update(exit_code=status, finished_at=_now()); _write(receipt_path, receipt)
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    pass
                else:
                    os.killpg(child.pid, signal.SIGKILL)
                    raise TerminalExecutionFailure('Stage exited with descendants still running')
                if status != 0:
                    raise TerminalExecutionFailure('Registered stage process failed')
                observed = _result(directory, request, stage, validate_result)
                verify_inputs()
                if digest(definition) != key:
                    raise ValueError('Frozen execution request changed')
                recorded.update(state='complete', **observed)
                _write(receipt_path, receipt)
            except BaseException as exc:
                if child is not None:
                    # This is the newly owned process group, never a name match.
                    try:
                        os.killpg(child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    child.wait()
                recorded.update(state='failed_execution', error=type(exc).__name__, message=str(exc))
                receipt.update(state='failed_execution', finished_at=_now())
                _write(receipt_path, receipt)
                raise
        verify_inputs()
        receipt.update(state='complete', finished_at=_now())
        _write(receipt_path, receipt)
        return receipt


def child_main():
    definition, name, lock_fd, expected_sha256 = sys.argv[1:]
    if digest(definition) != expected_sha256:
        raise SystemExit('Frozen execution request changed before child start')
    request = json.loads(Path(definition).read_bytes())
    stage = next(s for s in request['stages'] if s['name'] == name)
    if sys.stdin.buffer.read(1) != b'1':
        raise SystemExit('Controller exited before authorizing its recorded child')
    os.set_inheritable(int(lock_fd), True)
    os.execvpe(stage['command'][0], stage['command'], stage['environment'])
