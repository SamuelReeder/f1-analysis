#!/bin/bash
set -eu
cd /home/sreeder/.codex/worktrees/overall-production/f1-analysis
test ! -e outputs/championship/development/overall-stable-profile-v1
test ! -e outputs/championship/cache/overall-stable-profile-v1
active_f1_units=$(systemctl --user list-units --state=running,activating --no-legend 'f1-*.service' 'f1-*.scope')
if [ -n "$active_f1_units" ]; then
    printf '%s\n' 'Refusing to overlap an active fit or heavy analysis.' >&2
    exit 1
fi
systemd-run --user --unit=f1-overall-stable-profile-v1 --service-type=exec --slice=f1.slice \
  -p MemoryMax=13G -p CPUWeight=10 -p Nice=19 -p OOMPolicy=continue -p Restart=no \
  -p WorkingDirectory=/home/sreeder/.codex/worktrees/overall-production/f1-analysis \
  /bin/bash /home/sreeder/.codex/worktrees/overall-production/f1-analysis/outputs/championship/development/overall-stable-profile-v1-launch/service.sh
