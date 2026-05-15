#!/bin/bash

#SBATCH -J drk-eval
#SBATCH -o ../logs/drk-eval-%j.out
#SBATCH -e ../logs/drk-eval-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH -t 0-6:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --constraint="L40|L40S"  # 排除A30 24GB节点防止OOM；80GB GPU(A100)可删除此行
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos=gpu-short  # gpu-short队列更快(6h上限)；如果eval超6h改回gpu-normal或gpu-long
#SBATCH --cpus-per-task=32  # Ray + KernelGYM + worker需要较多CPU；如果节点CPU不足可降回16

set -euo pipefail

log_section() {
    echo
    echo "===== $1 ====="
}

echo "Job start at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Host: $(hostname)"

ulimit -l unlimited
ulimit -v unlimited
ulimit -n 65535
ulimit -u 65536  # 合理进程数上限；如遇"too many processes"可提高，如遇SIGKILL可降低

source /tools/module_env.sh
module purge
module load cluster-tools/v1.0
module load slurm-tools/v1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

# ===== 路径覆盖（不同于YAML默认值时在此export） =====
export DRKERNEL_DATA_ROOT="${DRKERNEL_DATA_ROOT:-/nfs_global/I/${USER}/WangYongsheng/drkernel}"
export MODEL_PATH="${DRKERNEL_DATA_ROOT}/models/hkust-nlp/drkernel-14b"

# ===== 运行时变量（每个job不同） =====
export RUN_NAME="drkernel-14b-eval-4gpu"
export N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"
export NNODES="${SLURM_NNODES:-1}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-60}"
export N_SAMPLES=4
export BATCH_SIZE=32
export WANDB_MODE=offline

# 其他eval调优参数见 config/defaults.yaml 或 python3 config/load_config.py eval --help
# Slurm中可通过 export 覆盖YAML默认值（优先级: CLI > env var > YAML）

# ===== 临时目录（短路径避免socket AF_UNIX路径过长） =====
export TMPDIR="/tmp/${USER}/drkernel_${SLURM_JOB_ID}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export RAY_TMPDIR="${TMPDIR}/ray"

# ===== HuggingFace 缓存 =====
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRITON_CACHE_DIR}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

# Ray 调试选项
export RAY_LOG_TO_STDERR=1
export RAY_memory=8000000000  # 8GB for ray

log_section "RUN CONFIG"
echo "GPU_MONITOR_INTERVAL=${GPU_MONITOR_INTERVAL}"
echo "TMPDIR=${TMPDIR}"
echo "RAY_TMPDIR=${RAY_TMPDIR}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/local_ops/start_kernelgym_services.sh"
trap stop_kernelgym_services EXIT
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

bash "${PROJECT_DIR}/local_ops/run_eval.sh" "$@"
