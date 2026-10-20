#!/bin/bash
set -eu
cd /home/sreeder/.codex/worktrees/overall-production/f1-analysis
student_output=outputs/championship/development/overall-studentt-production-audit-v1
student_logs=outputs/championship/development/overall-studentt-production-audit-v1-execution
student_cache=outputs/championship/cache/overall-studentt-production-audit-v1
student_protocol=docs/overall_studentt_production_protocol.json
trap 'printf "%s\n" "$?" > "$student_logs/exit.status"' EXIT
run_phase() {
    student_phase="$1"
    student_flag="$2"
    shift 2
    systemd-run --user --scope --slice=f1.slice --unit="f1-studentt-production-$student_phase" \
      -q -p MemoryMax=3G -p CPUQuota=200% -- /usr/bin/nice -n 19 \
      /usr/bin/time -v -o "$student_logs/$student_phase.resources.txt" \
      env JAX_PLATFORMS=cpu JAX_ENABLE_X64="$student_flag" JAX_ENABLE_COMPILATION_CACHE=false \
      XLA_FLAGS=--xla_force_host_platform_device_count=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      /home/sreeder/f1-analysis/.venv/bin/python -u "$@" \
      > "$student_logs/$student_phase.stdout.txt" 2>&1
}
run_phase tests32 false -m pytest -q tests/test_overall_studentt_production.py
run_phase tests64 true -m pytest -q tests/test_overall_studentt_production.py
run_phase network32 false analysis/overall_studentt_production_audit.py evaluate --precision float32 \
  --protocol "$student_protocol" --scope f1-studentt-production-network32.scope \
  --output "$student_output/float32" --cache "$student_cache/float32.npz"
run_phase network64 true analysis/overall_studentt_production_audit.py evaluate --precision float64 \
  --protocol "$student_protocol" --scope f1-studentt-production-network64.scope \
  --output "$student_output/float64" --cache "$student_cache/float64.npz"
