#!/bin/bash
# Opt-in NVIDIA MPS for ONE Slurm allocation: source this file, call `mps_start` before the
# packer and `mps_stop` after (or rely on the EXIT trap). Everything launched from this shell
# afterwards (the packer's jobs) attaches to the job-private MPS server, so several small jobs
# on the same GPU run with overlapping kernels instead of being time-sliced by the driver.
#
# Facts verified on this cluster (2026-11-07, skypilot-pod-2, a one-GPU a40-burst allocation):
#   * nvidia-cuda-mps-control / -server exist on the nodes and start as an ordinary user with a
#     job-private pipe/log directory under the node-local /tmp (GPUs stay in compute mode Default,
#     so no admin action is needed and other users' processes on the node are unaffected).
#   * two concurrent 4096^2 fp32 matmul loops: 8622 matmuls / 40 s under MPS vs 7892 without
#     (one process alone: 7317), i.e. ~+9 % aggregate throughput on a compute-bound pair.
#   * `echo quit | nvidia-cuda-mps-control` stops daemon and server cleanly.
#
# Usage in an sbatch body:
#     source scripts/ops/slurm_mps.sh && mps_start
#     uv run python -m scripts.ops.slurm_pack --jobs-file ... --claim-dir ...
#     mps_stop
#
# Notes: MPS is per node and per user; the pipe dir MUST be node-local (never the shared FS) and
# unique per allocation, which is why it carries $SLURM_JOB_ID. Jobs inherit CUDA_MPS_* from the
# environment -- the packer passes its environment through. Never start MPS for a running
# campaign's existing processes; they keep running outside it.

mps_start() {
    local base="${TMPDIR:-/tmp}/mps-${USER}-${SLURM_JOB_ID:-$$}"
    export CUDA_MPS_PIPE_DIRECTORY="$base/pipe" CUDA_MPS_LOG_DIRECTORY="$base/log"
    mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    if ! command -v nvidia-cuda-mps-control >/dev/null; then
        echo "mps_start: nvidia-cuda-mps-control not found; running WITHOUT MPS" >&2
        unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
        return 1
    fi
    if nvidia-cuda-mps-control -d; then
        echo "mps_start: daemon up, pipe=$CUDA_MPS_PIPE_DIRECTORY"
        trap mps_stop EXIT
        return 0
    fi
    echo "mps_start: daemon failed to start; running WITHOUT MPS" >&2
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    return 1
}

mps_stop() {
    [ -n "${CUDA_MPS_PIPE_DIRECTORY:-}" ] || return 0
    echo quit | nvidia-cuda-mps-control 2>/dev/null
    sleep 1
    echo "mps_stop: daemon down ($(ps -eo comm | grep -c '^nvidia-cuda-mps') mps processes left)"
    rm -rf "$(dirname "$CUDA_MPS_PIPE_DIRECTORY")"
    unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY
    trap - EXIT
}
