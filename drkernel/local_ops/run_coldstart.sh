#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

activate_drkernel_venv
ensure_common_dirs

BASE_MODEL_DIR="${BASE_MODEL_DIR:-${HF_MODEL_ROOT}/${DRKERNEL_BASE_MODEL_REPO}}"
COLDSTART_DATA_FILE="${COLDSTART_DATA_FILE:-$(resolve_dataset_file "hkust-nlp/drkernel-coldstart-8k")}"
RUN_NAME="${RUN_NAME:-drkernel-14b-coldstart-local}"
PROJECT_NAME="${PROJECT_NAME:-kernel-sft}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${DRKERNEL_CHECKPOINT_ROOT}/coldstart}"
LOG_DIR="${LOG_DIR:-${DRKERNEL_LOG_ROOT}/coldstart}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-2}"
MAX_LENGTH="${MAX_LENGTH:-12288}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-4}"
SAVE_FREQ="${SAVE_FREQ:-50}"
LEARNING_RATE="${LEARNING_RATE:-2e-5}"
SP_SIZE="${SP_SIZE:-4}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-${ARNOLD_WORKER_GPU:-8}}}"
NNODES="${NNODES:-${SLURM_NNODES:-${ARNOLD_WORKER_NUM:-1}}}"
NODE_RANK="${NODE_RANK:-${SLURM_NODEID:-${ARNOLD_ID:-0}}}"
MASTER_ADDR="${MASTER_ADDR:-${SLURM_LAUNCH_NODE_IPADDR:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}}"
MASTER_PORT="${MASTER_PORT:-29500}"

mkdir -p "${CHECKPOINT_DIR}" "${LOG_DIR}"

if [[ ! -d "${BASE_MODEL_DIR}" ]]; then
    echo "[ERROR] Base model directory not found: ${BASE_MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -f "${COLDSTART_DATA_FILE}" ]]; then
    echo "[ERROR] Cold-start dataset file not found: ${COLDSTART_DATA_FILE}" >&2
    exit 1
fi

export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export TMPDIR="${TMPDIR:-/tmp/drkernel-coldstart-${SLURM_JOB_ID:-local}}"
export TEMP="${TEMP:-${TMPDIR}}"
export TMP="${TMP:-${TMPDIR}}"
mkdir -p "${TMPDIR}"

torchrun \
    --nproc-per-node "${GPUS_PER_NODE}" \
    --master-addr "${MASTER_ADDR}" \
    --node-rank "${NODE_RANK}" \
    --master-port "${MASTER_PORT}" \
    --nnodes "${NNODES}" \
    -m kernel.fsdp_sft_trainer \
    data.multiturn.enable=True \
    data.train_files="${COLDSTART_DATA_FILE}" \
    data.val_files="${COLDSTART_DATA_FILE}" \
    data.train_batch_size="${TRAIN_BATCH_SIZE}" \
    data.micro_batch_size_per_gpu="${MICRO_BATCH_SIZE_PER_GPU}" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.max_length="${MAX_LENGTH}" \
    data.truncation=right \
    model.partial_pretrain="${BASE_MODEL_DIR}" \
    model.enable_gradient_checkpointing=True \
    model.fsdp_config.model_dtype=bf16 \
    model.fsdp_config.cpu_offload=True \
    model.fsdp_config.offload_params=True \
    ulysses_sequence_parallel_size="${SP_SIZE}" \
    use_remove_padding=True \
    model.strategy=fsdp \
    optim.lr="${LEARNING_RATE}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.default_local_dir="${CHECKPOINT_DIR}/${RUN_NAME}" \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.logger='["console","wandb"]' \
    "$@"
