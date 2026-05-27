#!/bin/bash

#SBATCH -J drk-sft
#SBATCH -o ../logs/drk-sft-%j.out
#SBATCH -e ../logs/drk-sft-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH -t 1-6:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos=gpu-normal
#SBATCH --cpus-per-task=16
#SBATCH --exclude=r8l40s-a05

set -euo pipefail

log_section() {
    echo
    echo "===== $1 ====="
}

print_error_lines() {
    local err_log
    err_log="${PROJECT_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}/../logs/drk-sft-${SLURM_JOB_ID:-unknown}.err"
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
    if [[ -n "${GPU_MONITOR_PID:-}" ]]; then
        kill "${GPU_MONITOR_PID}" >/dev/null 2>&1 || true
    fi
    log_section "JOB SUMMARY"
    echo "Exit code: ${exit_code}"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    if [[ "${exit_code}" -eq 0 ]]; then
        echo "[SUCCESS] cold-start job finished successfully"
    else
        echo "[FAILED] cold-start job failed"
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
ulimit -u 65536 || true

source /tools/module_env.sh
module purge
module load cluster-tools/v1.0
module load slurm-tools/v1.0

PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "${PROJECT_DIR}"

# ===== 路径和环境（不同于YAML默认值时通过sbatch --export覆盖） =====
export GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-4}}"
export NNODES="${SLURM_NNODES:-1}"
export DRKERNEL_DATA_ROOT="${DRKERNEL_DATA_ROOT:-/nfs_global/I/${USER}/WangYongsheng/drkernel}"
export BASE_MODEL_DIR="${BASE_MODEL_DIR:-${DRKERNEL_DATA_ROOT}/models/Qwen/Qwen3-14B-Base}"
export RUN_NAME="drkernel-14b-coldstart-4gpu"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-60}"
export WANDB_MODE=offline

# 所有SFT参数见 config/defaults.yaml 或 bash local_ops/run_coldstart.sh --help
# 优先级: CLI > env var > YAML。sbatch时通过 --export=ALL,KEY=VALUE 覆盖

# ===== 缓存和临时目录 =====
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/.cache/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton"
export TMPDIR="/tmp/${USER}/drc-${SLURM_JOB_ID:-local}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRITON_CACHE_DIR}"
mkdir -p "${TMPDIR}"

log_section "RUN CONFIG"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DRKERNEL_DATA_ROOT=${DRKERNEL_DATA_ROOT}"
echo "BASE_MODEL_DIR=${BASE_MODEL_DIR}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}"
echo "TMPDIR=${TMPDIR}"

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

bash "${PROJECT_DIR}/local_ops/run_coldstart.sh" "$@"
