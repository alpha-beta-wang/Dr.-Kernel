#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

activate_drkernel_venv
ensure_common_dirs

if [[ -z "${KERNELGYM_SERVER_URL:-}" && -z "${REWARD_SERVER_URL:-}" ]]; then
    echo "[ERROR] KERNELGYM_SERVER_URL or REWARD_SERVER_URL is required for evaluation" >&2
    exit 1
fi

EVAL_DATASET="${EVAL_DATASET:-$(resolve_dataset_file "hkust-nlp/drkernel-validation-data")}"
MODEL_PATH="${MODEL_PATH:-${HF_MODEL_ROOT}/${DRKERNEL_EVAL_MODEL_REPO}}"
MODEL_NAME="${MODEL_NAME:-$(basename "${MODEL_PATH}")}"
RUN_NAME="${RUN_NAME:-drkernel-14b-eval-local}"
PROJECT_NAME="${PROJECT_NAME:-kernel-grading}"
OUTPUT_DIR="${OUTPUT_DIR:-${DRKERNEL_RESULT_ROOT}/${RUN_NAME}}"
OUTPUT_PATH="${OUTPUT_PATH:-${OUTPUT_DIR}/graded_results.parquet}"
METRICS_OUTPUT_PATH="${METRICS_OUTPUT_PATH:-${OUTPUT_DIR}/metrics.json}"
RAW_RESPONSE_PATH="${RAW_RESPONSE_PATH:-${OUTPUT_DIR}/raw_responses.jsonl}"

if [[ ! -f "${EVAL_DATASET}" ]]; then
    echo "[ERROR] Evaluation dataset not found: ${EVAL_DATASET}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

REWARD_SERVER_URL="${REWARD_SERVER_URL:-${KERNELGYM_SERVER_URL}}"
REWARD_MANAGER="${REWARD_MANAGER:-kernel_async}"
REWARD_FUNC_NAME="${REWARD_FUNC_NAME:-calculate_reward_speedup}"
REWARD_WEIGHTS="${REWARD_WEIGHTS:-0.3_0.4_0.3}"
N_SAMPLES="${N_SAMPLES:-8}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MULTI_TURN="${MULTI_TURN:-True}"
MAX_USER_TURNS="${MAX_USER_TURNS:-3}"
ROLLOUT_MODE="${ROLLOUT_MODE:-async_vllm}"
ROLLOUT_GPU_MEMORY_UTIL="${ROLLOUT_GPU_MEMORY_UTIL:-0.5}"
ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE="${ROLLOUT_TENSOR_MODEL_PARALLEL_SIZE:-1}"
NUM_PERF_TRIALS="${NUM_PERF_TRIALS:-10}"
NUM_CORRECT_TRIALS="${NUM_CORRECT_TRIALS:-5}"
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"

# shellcheck disable=SC1091
source "${DRKERNEL_ROOT}/kernel/scripts/eval/grading_common.sh"
main "$@"
