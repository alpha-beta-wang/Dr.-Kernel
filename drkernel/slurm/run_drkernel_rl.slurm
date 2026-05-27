#!/bin/bash

#SBATCH -J drk-rl
#SBATCH -o ../logs/drk-rl-%j.out
#SBATCH -e ../logs/drk-rl-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH -t 3-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos=gpu-long
#SBATCH --cpus-per-task=32
#SBATCH --constraint="A100|L40|L40S"

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

ulimit -l unlimited
ulimit -v unlimited
ulimit -n 65535
ulimit -u 65536 2>/dev/null || true

source /tools/module_env.sh
module purge
module load cluster-tools/v1.0
module load slurm-tools/v1.0

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

# ===== 路径和环境（不同于YAML默认值时通过sbatch --export覆盖） =====
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"
export NNODES="${SLURM_NNODES:-1}"
export DRKERNEL_DATA_ROOT="${DRKERNEL_DATA_ROOT:-/nfs_global/I/${USER}/WangYongsheng/drkernel}"
export RL_MODEL_PATH="${RL_MODEL_PATH:-${DRKERNEL_DATA_ROOT}/models/hkust-nlp/drkernel-14b}"
export RUN_NAME="drkernel-14b-rl-8gpu"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-60}"
export WANDB_MODE=offline

# 所有RL超参数见 config/defaults.yaml 或 bash local_ops/run_rl.sh --help
# 优先级: CLI > env var > YAML。sbatch时通过 --export=ALL,KEY=VALUE 覆盖

# ===== 缓存和临时目录 =====
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton"
export TMPDIR="/tmp/${USER}/drkernel-rl-${SLURM_JOB_ID:-local}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export RAY_TMPDIR="${TMPDIR}/ray"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRITON_CACHE_DIR}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

export RAY_LOG_TO_STDERR=1
export RAY_memory=8000000000

log_section "RUN CONFIG"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DRKERNEL_DATA_ROOT=${DRKERNEL_DATA_ROOT}"
echo "RL_MODEL_PATH=${RL_MODEL_PATH}"
echo "N_GPUS_PER_NODE=${N_GPUS_PER_NODE}"
echo "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}"
echo "TMPDIR=${TMPDIR}"

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
