#!/bin/bash

#SBATCH -J drk-rl
#SBATCH -o ../logs/drk-rl-%j.out
#SBATCH -e ../logs/drk-rl-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH -t 1-12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos=gpu-normal
#SBATCH --cpus-per-task=16

set -euo pipefail

echo "Job start at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Host: $(hostname)"

ulimit -l unlimited
ulimit -v unlimited
ulimit -n 65535
ulimit -u 4125556

source /tools/module_env.sh
module purge
module load cluster-tools/v1.0
module load slurm-tools/v1.0

PROJECT_DIR="/workspace/I/qimeng6/WangYongsheng/KernelGYM/drkernel"
cd "${PROJECT_DIR}"

export DRKERNEL_DATA_ROOT="/nfs_global/I/qimeng6/WangYongsheng/drkernel"
export RL_MODEL_PATH="${DRKERNEL_DATA_ROOT}/models/hkust-nlp/drkernel-14b"
export RUN_NAME="drkernel-14b-rl-4gpu"
export N_GPUS_PER_NODE="${SLURM_GPUS_ON_NODE:-4}"
export NNODES="${SLURM_NNODES:-1}"
export TRAIN_BATCH_SIZE=8
export PPO_MINI_BATCH_SIZE=8
export ROLLOUT_N=8
export WANDB_MODE=offline

export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${PROJECT_DIR}/.cache/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${PROJECT_DIR}/.cache/huggingface/hub"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton"
export TMPDIR="${PROJECT_DIR}/.tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRITON_CACHE_DIR}" "${TMPDIR}"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/local_ops/start_kernelgym_services.sh"
trap stop_kernelgym_services EXIT
start_kernelgym_services

bash "${PROJECT_DIR}/local_ops/run_rl.sh" "$@"
