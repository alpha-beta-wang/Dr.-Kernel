#!/bin/bash

#SBATCH -J drk-eval
#SBATCH -o ../logs/drk-eval-%j.out
#SBATCH -e ../logs/drk-eval-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH -t 0-6:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos=gpu-normal
#SBATCH --cpus-per-task=16

set -euo pipefail

log_section() {
    echo
    echo "===== $1 ====="
}

echo "Job start at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Host: $(hostname)"

# 移除有问题的 ulimit 或降低到合理值
ulimit -l unlimited
ulimit -v unlimited
ulimit -n 65535
# ulimit -u 4125556  # 注释掉这行

source /tools/module_env.sh
module purge
module load cluster-tools/v1.0
module load slurm-tools/v1.0

PROJECT_DIR="/workspace/I/qimeng6/WangYongsheng/KernelGYM/drkernel"
cd "${PROJECT_DIR}"

export DRKERNEL_DATA_ROOT="/nfs_global/I/qimeng6/WangYongsheng/drkernel"
export MODEL_PATH="${DRKERNEL_DATA_ROOT}/models/hkust-nlp/drkernel-14b"
export RUN_NAME="drkernel-14b-eval-4gpu"
export N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-8}"  # 改为8
export NNODES="${SLURM_NNODES:-1}"
export GPU_MONITOR_INTERVAL="${GPU_MONITOR_INTERVAL:-60}"
export N_SAMPLES=4
export BATCH_SIZE=32
export WANDB_MODE=offline

# ===== 关键修改：使用短路径 =====
export TMPDIR="/tmp/${USER}/drkernel_${SLURM_JOB_ID}"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export RAY_TMPDIR="${TMPDIR}/ray"

# HuggingFace 缓存可以用长路径（不受socket限制影响）
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton"

# 创建所有需要的目录
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
