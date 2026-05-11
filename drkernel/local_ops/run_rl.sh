#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

activate_drkernel_venv
ensure_common_dirs

if [[ -z "${KERNELGYM_SERVER_URL:-}" ]]; then
    echo "[ERROR] KERNELGYM_SERVER_URL is required for RL training" >&2
    exit 1
fi

RL_TRAIN_DATASET="${RL_TRAIN_DATASET:-$(resolve_dataset_file "hkust-nlp/drkernel-rl-data")}"
RL_VALID_DATASET="${RL_VALID_DATASET:-$(resolve_dataset_file "hkust-nlp/drkernel-validation-data")}"
RL_MODEL_PATH="${RL_MODEL_PATH:-${HF_MODEL_ROOT}/${DRKERNEL_EVAL_MODEL_REPO}}"
RUN_NAME="${RUN_NAME:-drkernel-14b-rl-local}"
PROJECT_NAME="${PROJECT_NAME:-drkernel}"
HDFS_CHECKPOINT_PATH="${HDFS_CHECKPOINT_PATH:-${DRKERNEL_CHECKPOINT_ROOT}/rl}"
MODEL_NAME="${MODEL_NAME:-$(basename "${RL_MODEL_PATH}")}"
MODEL_PATH="${MODEL_PATH:-${RL_MODEL_PATH}}"

if [[ ! -f "${RL_TRAIN_DATASET}" ]]; then
    echo "[ERROR] RL train dataset not found: ${RL_TRAIN_DATASET}" >&2
    exit 1
fi
if [[ ! -f "${RL_VALID_DATASET}" ]]; then
    echo "[ERROR] RL validation dataset not found: ${RL_VALID_DATASET}" >&2
    exit 1
fi

mkdir -p "${HDFS_CHECKPOINT_PATH}"

TRAIN_DATASET=("${RL_TRAIN_DATASET}")
VALID_DATASET=("${RL_VALID_DATASET}")
REWARD_MANAGER="${REWARD_MANAGER:-kernel_async}"
REWARD_FUNC_NAME="${REWARD_FUNC_NAME:-calculate_reward_speedup}"
ALGORITHM="${ALGORITHM:-trloo}"
ENABLE_MULTI_TURN="${ENABLE_MULTI_TURN:-True}"
MAX_TURN="${MAX_TURN:-3}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-16}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
ROLLOUT_N="${ROLLOUT_N:-16}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
IS_GET_LAST_TURN="${IS_GET_LAST_TURN:-True}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-100}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export REWARD_SERVER_URL="${REWARD_SERVER_URL:-${KERNELGYM_SERVER_URL}}"
export SERVER_WITH_TRAINING="${SERVER_WITH_TRAINING:-False}"
export SERVER_WITH_TRAINING_NODES="${SERVER_WITH_TRAINING_NODES:-0}"
unset ROCR_VISIBLE_DEVICES
unset HIP_VISIBLE_DEVICES
export TMPDIR="${TMPDIR:-/tmp/drkernel-rl-${SLURM_JOB_ID:-local}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
export RAY_TMPDIR="${RAY_TMPDIR:-${TMPDIR}/ray}"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"

# shellcheck disable=SC1091
source "${DRKERNEL_ROOT}/kernel/scripts/rl/train_rl_common.sh"
main "$@"
