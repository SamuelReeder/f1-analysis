#!/bin/bash
set -eu
cd /home/sreeder/.codex/worktrees/overall-production/f1-analysis
precision_output=outputs/championship/development/overall-network-precision-v1
precision_logs=outputs/championship/development/overall-network-precision-v1-execution
precision_cache=outputs/championship/cache/overall-network-precision-v1
precision_protocol=docs/overall_network_precision_protocol.json
trap 'printf "%s\n" "$?" > "$precision_logs/exit.status"' EXIT

run_precision_phase() {
    precision_phase="$1"
    precision_flag="$2"
    shift 2
    systemd-run --user --scope --slice=f1.slice --unit="f1-network-precision-$precision_phase" \
      -q -p MemoryMax=3G -p CPUQuota=200% -- /usr/bin/nice -n 19 \
      /usr/bin/time -v -o "$precision_logs/$precision_phase.resources.txt" \
      env JAX_PLATFORMS=cpu JAX_ENABLE_X64="$precision_flag" JAX_ENABLE_COMPILATION_CACHE=false \
      XLA_FLAGS=--xla_force_host_platform_device_count=1 OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
      /home/sreeder/f1-analysis/.venv/bin/python -u analysis/overall_network_precision.py "$@" \
      --protocol "$precision_protocol" --scope "f1-network-precision-$precision_phase.scope" \
      --output "$precision_output/$precision_phase" > "$precision_logs/$precision_phase.stdout.txt" 2>&1
}

run_precision_phase prepare false prepare --cache "$precision_cache/inputs.npz"
run_precision_phase float32 false evaluate --precision float32 --prepared "$precision_output/prepare/result.json" --cache "$precision_cache/float32.npz"
run_precision_phase float64 true evaluate --precision float64 --prepared "$precision_output/prepare/result.json" --cache "$precision_cache/float64.npz"
run_precision_phase compare false compare --left "$precision_output/float32/result.json" --right "$precision_output/float64/result.json"
