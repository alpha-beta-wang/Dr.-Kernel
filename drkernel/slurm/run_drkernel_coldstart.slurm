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
    local script_dir err_log
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    err_log="${script_dir}/../logs/drk-sft-${SLURM_JOB_ID:-unknown}.err"

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
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "SLURM_NODELIST=${SLURM_NODELIST:-unset}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE:-unset}"

ulimit -l unlimited
ulimit -v unlimited
ulimit -n 65535
ulimit -u 4125556

source /tools/module_env.sh
module purge
module load cluster-tools/v1.0
module load slurm-tools/v1.0

log_section "SYSTEM INFO"
echo "which python: $(which python || true)"
echo "which python3: $(which python3 || true)"
echo "which gcc: $(which gcc || true)"
module list || true
cluster-quota || true
nvidia-smi --format=csv --query-gpu=index,name,driver_version,memory.total || true
nvidia-smi -q -d ECC || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

export DRKERNEL_DATA_ROOT="${DRKERNEL_DATA_ROOT:-/nfs_global/I/${USER}/WangYongsheng/drkernel}"
export BASE_MODEL_DIR="${DRKERNEL_DATA_ROOT}/models/Qwen/Qwen3-14B-Base"
export RUN_NAME="drkernel-14b-coldstart-4gpu"
export GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
export NNODES="${SLURM_NNODES:-1}"
export TRAIN_BATCH_SIZE=32
export MICRO_BATCH_SIZE_PER_GPU=1
export SP_SIZE=2
export WANDB_MODE=offline
export MAX_LENGTH="${MAX_LENGTH:-12288}"
export SAVE_FREQ="${SAVE_FREQ:-100}"
export CPU_OFFLOAD="${CPU_OFFLOAD:-True}"
export OFFLOAD_PARAMS="${OFFLOAD_PARAMS:-True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-60}"

export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${PROJECT_DIR}/.cache/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${PROJECT_DIR}/.cache/huggingface/hub"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton"
export TMPDIR="/tmp/drkernel-coldstart-${SLURM_JOB_ID:-local}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRITON_CACHE_DIR}" "${TMPDIR}"

log_section "RUN CONFIG"
echo "PROJECT_DIR=${PROJECT_DIR}"
echo "DRKERNEL_DATA_ROOT=${DRKERNEL_DATA_ROOT}"
echo "BASE_MODEL_DIR=${BASE_MODEL_DIR}"
echo "RUN_NAME=${RUN_NAME}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"
echo "NNODES=${NNODES}"
echo "TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE}"
echo "MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU}"
echo "SP_SIZE=${SP_SIZE}"
echo "MAX_LENGTH=${MAX_LENGTH}"
echo "SAVE_FREQ=${SAVE_FREQ}"
echo "CPU_OFFLOAD=${CPU_OFFLOAD}"
echo "OFFLOAD_PARAMS=${OFFLOAD_PARAMS}"
echo "PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF}"
echo "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}"
echo "HF_HOME=${HF_HOME}"
echo "TMPDIR=${TMPDIR}"

log_section "PREFLIGHT CHECKS"
source "${PROJECT_DIR}/.venv-drkernel/bin/activate"
python --version
python - <<'PY'
import os
from pathlib import Path

base_model_dir = Path(os.environ["BASE_MODEL_DIR"])
data_root = Path(os.environ["DRKERNEL_DATA_ROOT"])
coldstart_dataset = data_root / "datasets" / "hkust-nlp" / "drkernel-coldstart-8k" / "drkernel-coldstart-8k.parquet"

print(f"base_model_exists={base_model_dir.exists()}")
print(f"coldstart_dataset={coldstart_dataset}")
print(f"coldstart_dataset_exists={coldstart_dataset.exists()}")

if not base_model_dir.exists():
    raise SystemExit(f"Missing base model dir: {base_model_dir}")
if not coldstart_dataset.exists():
    raise SystemExit(f"Missing cold-start dataset: {coldstart_dataset}")
PY

python - <<'PY'
import traceback

mods = ["torch", "transformers", "datasets", "flash_attn"]
for name in mods:
    try:
        mod = __import__(name)
        print(f"{name}: OK {getattr(mod, '__version__', 'no_version')}")
    except Exception as exc:
        print(f"{name}: FAIL {exc!r}")
        traceback.print_exc()
        raise
PY

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

log_section "START TRAINING"
# bash "${PROJECT_DIR}/local_ops/run_coldstart.sh" "$@"
bash "${PROJECT_DIR}/local_ops/run_coldstart.sh" "$@" | awk '{
    printf "=== [TRAINING INFO] %s %s ===\n", strftime("%Y-%m-%d %H:%M:%S"), $0
}'
