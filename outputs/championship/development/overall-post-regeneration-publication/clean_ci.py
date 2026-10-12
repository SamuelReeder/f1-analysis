import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time

root=Path.cwd(); out=root/'outputs/logs/overall-post-regeneration-publication'
report=out/'clean-ci-run.json'
if report.exists(): raise FileExistsError(report)
if subprocess.check_output(['ss','-H','-ltn','sport = :4180'],text=True).strip():
    raise RuntimeError('Browser test port is already in use')
source=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
line=next(x for x in Path('/proc/self/cgroup').read_text().splitlines() if x.startswith('0::'))
group=Path('/sys/fs/cgroup')/line.split('::',1)[1].lstrip('/')
env=dict(os.environ)
env['PATH']='/home/sreeder/.nvm/versions/node/v22.20.0/bin:'+env['PATH']
env['DASHBOARD_TEST_URL']='http://127.0.0.1:4180/f1-analysis/'
env['PLAYWRIGHT_JSON_OUTPUT_FILE']=str(out/'clean-ci-browser.json')
record=dict(source_commit=source,started_at=dt.datetime.now(dt.timezone.utc).isoformat(),
    limits=dict(cgroup=str(group),memory_max=(group/'memory.max').read_text().strip(),cpu_max=(group/'cpu.max').read_text().strip()),
    stages=[],fresh_checkout=True,fresh_python_environment=True,fresh_npm_install=True,
    hosted_github_runner=False,actual_fit_or_refresh_run=False,
    qualification='Local application commands from the Dashboard workflow; existing OS browser libraries are reused rather than running the hosted runner OS-package installation',
    driver_ranking_source='Current committed portable payload; no ignored posterior fits copied')
started=time.monotonic(); directory=Path(tempfile.mkdtemp(prefix='f1-overall-clean-ci-')); checkout=directory/'repo'
record['checkout']=str(checkout)

def save():
    record['elapsed_seconds']=time.monotonic()-started
    record['peak_child_rss_kib']=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    record['cgroup_memory_peak_bytes']=(group/'memory.peak').read_text().strip()
    report.write_text(json.dumps(record,indent=2)+'\n')

def run(name,command,cwd=root):
    print(name,flush=True); step_started=time.monotonic()
    with (out/(name+'.txt')).open('w') as log:
        result=subprocess.run(command,cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
    record['stages'].append(dict(name=name,command=command,exit_code=result.returncode,seconds=time.monotonic()-step_started,log=name+'.txt'))
    save()
    if result.returncode:
        print((out/(name+'.txt')).read_text()[-4000:],flush=True)
        raise RuntimeError(name+' failed')

try:
    run('clean-clone',['git','clone','--no-local','--depth','1','--branch','overall-model-implementation','--quiet',str(root),str(checkout)])
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=checkout,text=True).strip()!=source: raise RuntimeError('Clone source differs')
    if (checkout/'outputs/fits').exists(): raise RuntimeError('Ignored posterior directory copied')
    run('clean-venv',['/usr/bin/python3.12','-m','venv',str(checkout/'.venv-ci')],checkout)
    python=str(checkout/'.venv-ci/bin/python')
    run('clean-python-install',[python,'-m','pip','install','--disable-pip-version-check','-r','dashboard/requirements.txt'],checkout)
    run('clean-portable-receipt',[python,'dashboard/verify_publication.py'],checkout)
    run('clean-python-tests',[python,'-m','pytest','tests/test_dashboard.py','tests/test_portable_publication.py','-q','--junitxml='+str(out/'clean-python-tests.xml')],checkout)
    run('clean-npm-install',['npm','ci','--prefix','dashboard'],checkout)
    run('clean-build',['npm','--prefix','dashboard','run','build','--','--base=/f1-analysis/'],checkout)
    run('clean-browser-install',['npx','playwright','install','chromium'],checkout/'dashboard')
    run('clean-browser-tests',['npm','test','--','--workers=2','--reporter=line,json'],checkout/'dashboard')
    record['tracked_files_unchanged']=not bool(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=checkout,text=True).strip())
    if not record['tracked_files_unchanged']: raise RuntimeError('CI commands changed tracked files')
    record['exit_code']=0
except Exception as error:
    record['exit_code']=1; record['failure']=str(error)
finally:
    save()
print(json.dumps({k:record[k] for k in ('source_commit','exit_code','elapsed_seconds','peak_child_rss_kib','checkout')}),flush=True)
sys.exit(record['exit_code'])
