#!/bin/bash
set -eu
cd /home/sreeder/.codex/worktrees/overall-production/f1-analysis
student_output=outputs/championship/development/overall-studentt-stable-v1
student_logs=outputs/championship/development/overall-studentt-stable-v1-execution
student_cache=outputs/championship/cache/overall-studentt-stable-v1
student_protocol=docs/overall_studentt_stable_protocol.json
trap 'printf "%s\n" "$?" > "$student_logs/exit.status"' EXIT

run_phase() {
    student_phase="$1"
    student_flag="$2"
    shift 2
    systemd-run --user --scope --slice=f1.slice --unit="f1-studentt-stable-$student_phase" \
      -q -p MemoryMax=3G -p CPUQuota=200% -- /usr/bin/nice -n 19 \
      /usr/bin/time -v -o "$student_logs/$student_phase.resources.txt" \
      env JAX_PLATFORMS=cpu JAX_ENABLE_X64="$student_flag" JAX_ENABLE_COMPILATION_CACHE=false \
      XLA_FLAGS=--xla_force_host_platform_device_count=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      /home/sreeder/f1-analysis/.venv/bin/python -u "$@" \
      > "$student_logs/$student_phase.stdout.txt" 2>&1
}

run_audit() {
    audit_phase="$1"
    audit_flag="$2"
    shift 2
    run_phase "$audit_phase" "$audit_flag" analysis/overall_studentt_stable_audit.py "$@" \
      --protocol "$student_protocol" --scope "f1-studentt-stable-$audit_phase.scope" \
      --output "$student_output/$audit_phase"
}

run_phase tests32 false -m pytest -q tests/test_overall_studentt_stable.py tests/test_overall_studentt_numerics.py
run_phase tests64 true -m pytest -q tests/test_overall_studentt_stable.py tests/test_overall_studentt_numerics.py
run_audit primitive32 false primitive --precision float32
run_audit primitive64 true primitive --precision float64
run_audit network32 false network --precision float32 --cache "$student_cache/network32.npz"
run_audit network64 true network --precision float64 --cache "$student_cache/network64.npz"
run_audit compare false compare --left "$student_output/network32/result.json" --right "$student_output/network64/result.json"
