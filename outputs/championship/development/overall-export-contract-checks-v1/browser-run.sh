#!/usr/bin/env bash
set -euo pipefail
cd /home/sreeder/.codex/worktrees/overall-production/f1-analysis/dashboard
export DASHBOARD_TEST_URL=http://127.0.0.1:4180/f1-analysis/
node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 4180 --strictPort --base /f1-analysis/ > ../outputs/championship/development/overall-export-contract-checks-v1/server.txt 2>&1 &
preview_pid=$!
trap 'kill "$preview_pid" 2>/dev/null || true; wait "$preview_pid" 2>/dev/null || true' EXIT
python3 - "$preview_pid" <<'PY'
import os,sys,time,urllib.request
for _ in range(50):
    os.kill(int(sys.argv[1]),0)
    try:
        with urllib.request.urlopen('http://127.0.0.1:4180/f1-analysis/',timeout=.2) as response:
            if response.status == 200:
                break
    except OSError:
        time.sleep(.1)
else:
    raise SystemExit('The local preview did not become ready')
PY
npx playwright test --workers=2
