#!/bin/bash
set -eu
cd /home/sreeder/.codex/worktrees/overall-production/f1-analysis
profile_launch=outputs/championship/development/overall-stable-profile-v1-launch
trap 'printf "%s\n" "$?" > "$profile_launch/exit.status"' EXIT
/usr/bin/time -v -o "$profile_launch/resources.txt" \
  env JAX_PLATFORMS=cpu JAX_ENABLE_X64=false JAX_ENABLE_COMPILATION_CACHE=false \
  XLA_FLAGS=--xla_force_host_platform_device_count=4 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/sreeder/f1-analysis/.venv/bin/python -u analysis/overall_stable_profile.py run \
  --registration docs/overall_stable_profile_protocol.json --unit f1-overall-stable-profile-v1.service \
  > "$profile_launch/stdout.txt" 2>&1
