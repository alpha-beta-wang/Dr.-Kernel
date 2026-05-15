#!/bin/bash

#SBATCH -J drk-rl
#SBATCH -o ../logs/drk-rl-%j.out
#SBATCH -e ../logs/drk-rl-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH -t 3-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:16
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos=gpu-long
#SBATCH --cpus-per-task=16
#
# --- SLURM tuning for constrained environments ---
# If you have fewer GPU quota or shorter time limit, adjust these:
#   --gres=gpu:8              # reduce to 8 GPUs if 16 is unavailable
#   -t 6:00:00                # shorter walltime (gpu-short QoS max 6h)
#   --qos=gpu-short           # use if gpu-long is not available
#   --constraint="A100|L40|L40S"  # exclude 24GB GPUs (A30) to prevent OOM
#   --exclude=<hostname>      # exclude specific problematic nodes
#   --cpus-per-task=32        # increase CPU if running Ray + KernelGYM together
#
# Log output paths also need to change if you lack write permission to /workspace/I/

set -euo pipefail

log_section() {
    echo
    echo "===== $1 ====="
}

print_error_lines() {
    local err_log
    err_log="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}/../logs/drk-rl-${SLURM_JOB_ID:-unknown}.err"

    echo "===== CRITICAL ERROR LOG ====="
    echo "Error log: ${err_log}"
    if [[ ! -f "${err_log}" ]]; then
        echo "error log not found"
        return
    fi

    local pattern='error|exception|traceback|fatal|failed|runtimeerror'
    if grep -Eani "${pattern}" "${err_log}" >/dev/null; then
        echo "Matched error lines (latest 80):"
        grep -Eani "${pattern}" "${err_log}" | tail -n 80
    else
        echo "No keyword matched, show last 120 lines:"
        tail -n 120 "${err_log}"
    fi
}

on_exit() {
    local exit_code=$?
    if declare -F stop_kernelgym_services >/dev/null 2>&1; then
        stop_kernelgym_services
    fi
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    fi
    log_section "JOB SUMMARY"
    echo "Exit code: ${exit_code}"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    if [[ "${exit_code}" -eq 0 ]]; then
        echo "[SUCCESS] RL job finished successfully"
    else
        echo "[FAILED] RL job failed"
        print_error_lines
    fi
}

trap on_exit EXIT

echo "Job start at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"

ulimit -l unlimited
ulimit -v unlimited
ulimit -n 65535
ulimit -u 4125556
# If ulimit -u 4125556 fails on restricted accounts, use: ulimit -u 65535 2>/dev/null || true

source /tools/module_env.sh
module purge
module load cluster-tools/v1.0
module load slurm-tools/v1.0
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES

log_section "SYSTEM INFO"
echo "which python: $(which python || true)"
echo "which python3: $(which python3 || true)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-unset}"
echo "HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-unset}"
module list || true
cluster-quota || true
nvidia-smi --format=csv --query-gpu=index,name,driver_version,memory.total || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# ===== 运行时变量 =====
export SLURM_GPUS_ON_NODE=16
# For 8-GPU setup, set SLURM_GPUS_ON_NODE=8 and see batch size / TP notes below.

export DRKERNEL_DATA_ROOT="${DRKERNEL_DATA_ROOT:-/nfs_global/I/${USER}/WangYongsheng/drkernel}"
export RL_MODEL_PATH="${DRKERNEL_DATA_ROOT}/models/hkust-nlp/drkernel-14b"
export RUN_NAME="drkernel-14b-rl-4gpu"
export N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
export NNODES="${SLURM_NNODES:-1}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-60}"

# ===== Slurm 专属覆盖（覆盖 YAML 默认值） =====
# 以下参数与 YAML 不同（16-GPU vs 8-GPU tuned），需要显式覆盖
export TRAIN_BATCH_SIZE=2
export PPO_MINI_BATCH_SIZE=2
export ROLLOUT_N=1
export ENABLE_MULTI_TURN=False
export MAX_TURN=1
export ROLLOUT_GPU_MEMORY_UTIL=0.74
export ROLLOUT_MODE=sync

# 其他RL训练参数见 config/defaults.yaml 或 python3 config/load_config.py rl --help
# Slurm中可通过 export 覆盖YAML默认值（优先级: CLI > env var > YAML）

export WANDB_MODE=offline
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${PROJECT_DIR}/.cache/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${PROJECT_DIR}/.cache/huggingface/hub"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton"
export TMPDIR="/tmp/drkernel-rl-${SLURM_JOB_ID:-local}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export RAY_TMPDIR="${TMPDIR}/ray"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRITON_CACHE_DIR}" "${TMPDIR}"

log_section "RUN CONFIG"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DRKERNEL_DATA_ROOT=${DRKERNEL_DATA_ROOT}"
echo "RL_MODEL_PATH=${RL_MODEL_PATH}"
echo "RUN_NAME=${RUN_NAME}"
echo "N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
echo "NNODES=${NNODES}"
echo "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}"
echo "HF_HOME=${HF_HOME}"
echo "TMPDIR=${TMPDIR}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/local_ops/start_kernelgym_services.sh"
start_kernelgym_services

gpu_monitor() {
    while true; do
        echo
        echo "===== GPU MEMORY SNAPSHOT $(date '+%Y-%m-%d %H:%M:%S') ====="
        nvidia-smi --format=csv --query-gpu=index,uuid,name,utilization.gpu,memory.used,memory.total || true
        echo "--- compute apps ---"
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader || true
        sleep "${GPU_MONITOR_INTERVAL}"
    done
}

gpu_monitor &
GPU_MONITOR_PID=$!

bash "${PROJECT_DIR}/local_ops/run_rl.sh" "$@"
