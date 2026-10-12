import datetime as dt
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

root = Path.cwd()
folder = root / 'outputs/logs/overall-prequalifying-resources'
record = folder / 'resource-run.json'
if record.exists():
    raise FileExistsError('Preserve previous run evidence')
line = next(line for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
group = Path('/sys/fs/cgroup') / line.split('::', 1)[1].lstrip('/')
command = [sys.executable, '-u', 'analysis/overall_legacy_prequalifying_resources.py', '--output', str(folder / 'resource.json')]
begun = time.monotonic()
started = dt.datetime.now(dt.timezone.utc).isoformat()
with (folder / 'resource.txt').open('w') as log:
    result = subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT)
record.write_text(json.dumps(dict(source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), command=command,
    started_at=started, finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    limits=dict(cgroup=str(group), memory_max=(group / 'memory.max').read_text().strip(), cpu_max=(group / 'cpu.max').read_text().strip()),
    exit_code=result.returncode, elapsed_seconds=time.monotonic()-begun,
    peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss), indent=2)+'\n')
print(record.read_text())
raise SystemExit(result.returncode)
