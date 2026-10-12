import datetime as dt
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

folder=Path('outputs/logs/overall-summary-checks')
label=sys.argv[1]
command=[sys.executable,'-m','pytest','-q','tests/test_overall_summary.py','tests/test_overall_contract.py','--junitxml='+str(folder/(label+'.xml'))]
record=folder/(label+'-run.json')
if record.exists(): raise FileExistsError(record)
line=next(line for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::'))
group=Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')
begun=time.monotonic()
started=dt.datetime.now(dt.timezone.utc).isoformat()
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
with (folder/(label+'.txt')).open('w') as log:
    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
value=dict(source_commit=source,command=command,started_at=started,
    finished_at=dt.datetime.now(dt.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-begun,
    limits=dict(cgroup=str(group),memory_max=(group/'memory.max').read_text().strip(),cpu_max=(group/'cpu.max').read_text().strip()),
    exit_code=result.returncode,peak_child_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
record.write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2))
raise SystemExit(result.returncode)
