#!/bin/bash
#SBATCH -J ppo-smoketest
#SBATCH -p r8nv-gpu-hw
#SBATCH --qos=gpu-longlong
#SBATCH -t 0-00:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32

set -e

source /share/personal/S/huanglongsheng/KernelGYM-hls/drkernel/local_ops/common_env.sh
activate_drkernel_venv

DRKERNEL_ROOT=/share/personal/S/huanglongsheng/KernelGYM-hls/drkernel
DRKERNEL_DATA_ROOT=/share/personal/S/huanglongsheng/drkernel_data
RL_MODEL_PATH=/share/personal/S/huanglongsheng/drkernel_data/checkpoints/coldstart/global_step_300-sft-761600/global_step_600

export PPO_MICRO_TOKEN=9344
export SP_SIZE=2
export TRAIN_BATCH_SIZE=16
export ROLLOUT_N=8
export PPO_MINI_BATCH_SIZE=16
export MAX_PROMPT_LENGTH=1152
export MAX_RESPONSE_LENGTH=8192
export RL_MODEL_PATH
export DRKERNEL_DATA_ROOT

cd "${DRKERNEL_ROOT}"

echo "Starting PPO smoke test on $(hostname) at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

torchrun --nproc_per_node=8 --master_port=29501 local_ops/ppo_smoketest.py

echo "PPO smoke test completed at $(date)"
