"""Fixed retry selection around complete, immutable checkpoint attempts."""
import fcntl
import json
from pathlib import Path

from analysis.checkpoint_execution import TerminalExecutionFailure, _write
from f1rank.artifacts import RETRY
from f1rank.overall.fitting import FIT_SETTINGS, sampling_seeds
from . import attempt_key, execute_attempt


def settings_inventory(registration, study):
    if (registration['fit_settings'] != FIT_SETTINGS
            or registration['retry'] != [list(row) for row in RETRY]):
        raise ValueError('Preserve the complete existing fit settings and retry ladder')
    seeds = sampling_seeds(registration)[study]
    return [dict(FIT_SETTINGS, warmup=warm*FIT_SETTINGS['warmup'],
                 samples=draws*FIT_SETTINGS['samples'], seed=seeds[index],
                 target_accept=FIT_SETTINGS['target_accept'] if accept is None else accept)
            for index, (warm,draws,accept) in enumerate(RETRY)]


def _run_ladder(directory, registration, study, *, specification_for, request_for,
                verify_inputs, execute_one):
    """Resume the same attempt, or advance ONLY after complete failed diagnostics.

    Terminal execution failures never select a numerical retry. All failed draw
    sets remain in their original attempt directories. Contrasts remain reports;
    their pass/fail status does not change the original parameter-based gate.
    """
    verify_inputs()
    directory = Path(directory).resolve()
    settings = settings_inventory(registration, study)
    specifications = [specification_for(row) for row in settings]
    if any(spec['settings'] != row for spec,row in zip(specifications, settings, strict=True)):
        raise ValueError('An attempt specification changed the fixed numerical procedure')
    definition = dict(study=study, attempt_keys=[attempt_key(s) for s in specifications])
    directory.mkdir(parents=True, exist_ok=True)
    request_path = directory / 'ladder.json'
    if request_path.exists():
        if json.loads(request_path.read_bytes()) != definition:
            raise ValueError('Cannot change an existing study or its complete retry inventory')
    else:
        _write(request_path, definition)
    terminal_path = directory / 'result.json'
    previous = json.loads(terminal_path.read_bytes()) if terminal_path.exists() else None
    if previous is not None and previous['state'] == 'failed_execution':
        raise TerminalExecutionFailure('Preserve the failed study execution; never select a retry')
    result = dict(study=study, definition=definition, attempts=[],
                  state='running', publication_allowed=False,
                  full_recovery_complete=False, sbc_complete=False)
    try:
        for index, spec in enumerate(specifications, 1):
            verify_inputs()
            path = directory / f'attempt-{index:02d}'
            execution = execute_one(path, spec, request_for(path,spec), verify_inputs=verify_inputs)
            decision = execution['stages'][-1]['verified_summary']
            if type(decision['converged']) is not bool:
                raise ValueError('Require a complete Boolean diagnostic decision')
            result['attempts'].append(dict(attempt=index, settings=spec['settings'],
                attempt_key=attempt_key(spec), converged=decision['converged'],
                diagnostics_sha256=decision['diagnostics_sha256']))
            if decision['converged']:
                result['state'] = 'numerically_converged'
                break
        else:
            result['state'] = 'failed_convergence'
        verify_inputs()
    except BlockingIOError:
        # An orphaned stage may still hold the inherited attempt lock. Waiting
        # for authoritative process death is not a scientific/execution failure.
        raise
    except BaseException as exc:
        if previous is None:
            result.update(state='failed_execution', error=type(exc).__name__, message=str(exc))
            _write(terminal_path, result)
        raise
    if previous is not None:
        if previous != result:
            raise ValueError('The recorded terminal study changed during revalidation')
    else:
        _write(terminal_path, result)
    return result


def run_ladder(directory, registration, study, *, specification_for, request_for,
               verify_inputs, execute_one=execute_attempt):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / '.ladder.lock').open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return _run_ladder(directory, registration, study, specification_for=specification_for,
                           request_for=request_for, verify_inputs=verify_inputs, execute_one=execute_one)
