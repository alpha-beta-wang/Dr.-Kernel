#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

# --help / -h
if [[ "${1:-}" == "--help" ]] || [[ "${1:-}" == "-h" ]]; then
    python3 "${DRKERNEL_ROOT}/config/load_config.py" eval --help
    exit 0
fi

activate_drkernel_venv
ensure_common_dirs

if [[ -z "${KERNELGYM_SERVER_URL:-}" && -z "${REWARD_SERVER_URL:-}" ]]; then
    echo "[ERROR] KERNELGYM_SERVER_URL or REWARD_SERVER_URL is required for evaluation" >&2
    exit 1
fi

# Load YAML config (skip keys already set in environment)
eval "$(python3 "${DRKERNEL_ROOT}/config/load_config.py" eval "$@")"

# Runtime-derived values (use YAML-loaded or env-provided values)
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
NNODES="${NNODES:-${SLURM_NNODES:-1}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"

# shellcheck disable=SC1091
source "${DRKERNEL_ROOT}/kernel/scripts/eval/grading_common.sh"
main "$@"
