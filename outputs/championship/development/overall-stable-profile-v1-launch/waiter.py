"""One quiet waiter, checking at the user's requested interval; never launches fits."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/sreeder/.codex/worktrees/overall-production/f1-analysis')
UNIT = 'f1-overall-stable-profile-v1.service'
INTERVAL_SECONDS = 20 * 60
output = ROOT/'outputs/championship/development/overall-stable-profile-v1'
state_file = ROOT/'outputs/logs/overall-stable-profile-v1-waiter.json'
state_file.parent.mkdir(parents=True, exist_ok=True)
lock = state_file.with_suffix('.lock').open('a+')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
lock.seek(0); lock.truncate(); lock.write(str(os.getpid())+'\n'); lock.flush()
seen = set(json.loads(state_file.read_text()).get('reported_failed_cases', [])) if state_file.exists() else set()
while True:
    time.sleep(INTERVAL_SECONDS)
    properties = subprocess.run(['systemctl', '--user', 'show', UNIT, '-p', 'ActiveState', '-p', 'SubState',
        '-p', 'MainPID', '-p', 'Result', '-p', 'ExecMainStatus'], capture_output=True, text=True)
    if properties.returncode:
        # A transient observation error does not justify restarting the fit.
        observation = dict(observed_at=datetime.now(timezone.utc).isoformat(), observation_error=properties.stderr,
                           reported_failed_cases=sorted(seen))
    else:
        service = dict(line.split('=', 1) for line in properties.stdout.splitlines())
        completions = [json.loads(p.read_text()) for p in sorted(output.glob('*.completion.json'))]
        failed = {row['case'] for row in completions if row['state'] == 'failed'}
        new_failures = sorted(failed-seen); seen |= failed
        terminal = service.get('MainPID') == '0' and service.get('SubState') not in ('start', 'start-pre', 'start-post')
        observation = dict(observed_at=datetime.now(timezone.utc).isoformat(), service=service,
            completed_cases=[dict(case=row['case'], state=row['state'], exit_code=row['exit_code']) for row in completions],
            new_failures=new_failures, reported_failed_cases=sorted(seen), terminal=terminal)
    temporary = state_file.with_suffix('.tmp')
    temporary.write_text(json.dumps(observation, indent=2)+'\n'); temporary.replace(state_file)
    if observation.get('terminal') or observation.get('new_failures'):
        print(json.dumps(observation), flush=True)
        break
