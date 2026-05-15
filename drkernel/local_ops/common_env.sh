#!/usr/bin/env bash

set -euo pipefail

DRKERNEL_LOCAL_OPS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRKERNEL_ROOT="$(cd "${DRKERNEL_LOCAL_OPS_DIR}/.." && pwd)"
KERNELGYM_ROOT="$(cd "${DRKERNEL_ROOT}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${KERNELGYM_ROOT}/.." && pwd)"

export DRKERNEL_ROOT
export KERNELGYM_ROOT
export WORKSPACE_ROOT
export PYTHONPATH="${DRKERNEL_ROOT}:${KERNELGYM_ROOT}:${PYTHONPATH:-}"

export SOFTWARE_ROOT="${SOFTWARE_ROOT:-${WORKSPACE_ROOT}/software}"
export REDIS_INSTALL_ROOT="${REDIS_INSTALL_ROOT:-${SOFTWARE_ROOT}/redis-7.2.5}"
export FLASH_ATTN_CACHE_DIR="${FLASH_ATTN_CACHE_DIR:-${SOFTWARE_ROOT}/flash-attn}"

export DRKERNEL_VENV_DIR="${DRKERNEL_VENV_DIR:-${DRKERNEL_ROOT}/.venv-drkernel}"
export KERNELGYM_VENV_DIR="${KERNELGYM_VENV_DIR:-${KERNELGYM_ROOT}/.venv-kgym}"

export DRKERNEL_DATA_ROOT="${DRKERNEL_DATA_ROOT:-/nfs_global/I/${USER}/WangYongsheng/drkernel}"
export HF_DATA_ROOT="${HF_DATA_ROOT:-${DRKERNEL_DATA_ROOT}/datasets}"
export HF_MODEL_ROOT="${HF_MODEL_ROOT:-${DRKERNEL_DATA_ROOT}/models}"
export DRKERNEL_CHECKPOINT_ROOT="${DRKERNEL_CHECKPOINT_ROOT:-${DRKERNEL_DATA_ROOT}/checkpoints}"
export DRKERNEL_RESULT_ROOT="${DRKERNEL_RESULT_ROOT:-${DRKERNEL_DATA_ROOT}/results}"
export DRKERNEL_LOG_ROOT="${DRKERNEL_LOG_ROOT:-${DRKERNEL_DATA_ROOT}/logs}"

export UV_BIN="${UV_BIN:-$(command -v uv || true)}"

export DRKERNEL_BASE_MODEL_REPO="${DRKERNEL_BASE_MODEL_REPO:-Qwen/Qwen3-14B-Base}"
export DRKERNEL_EVAL_MODEL_REPO="${DRKERNEL_EVAL_MODEL_REPO:-hkust-nlp/drkernel-14b}"

ensure_dir() {
    mkdir -p "$1"
}

ensure_common_dirs() {
    ensure_dir "${SOFTWARE_ROOT}"
    ensure_dir "${FLASH_ATTN_CACHE_DIR}"
    ensure_dir "${HF_DATA_ROOT}"
    ensure_dir "${HF_MODEL_ROOT}"
    ensure_dir "${DRKERNEL_CHECKPOINT_ROOT}"
    ensure_dir "${DRKERNEL_RESULT_ROOT}"
    ensure_dir "${DRKERNEL_LOG_ROOT}"
}

activate_drkernel_venv() {
    if [[ ! -f "${DRKERNEL_VENV_DIR}/bin/activate" ]]; then
        echo "[ERROR] Dr.Kernel uv env not found: ${DRKERNEL_VENV_DIR}" >&2
        return 1
    fi
    # shellcheck disable=SC1090
    source "${DRKERNEL_VENV_DIR}/bin/activate"
}

resolve_dataset_file() {
    local repo_id="$1"
    local preferred_name="${2:-}"
    local base_dir="${HF_DATA_ROOT}/${repo_id}"

    if [[ -n "${preferred_name}" && -f "${base_dir}/${preferred_name}" ]]; then
        printf '%s\n' "${base_dir}/${preferred_name}"
        return 0
    fi

    find "${base_dir}" -maxdepth 2 -type f \( -name '*.parquet' -o -name '*.jsonl' \) | sort | head -n 1
}

latest_checkpoint_dir() {
    local root_dir="$1"
    if [[ ! -d "${root_dir}" ]]; then
        return 1
    fi
    find "${root_dir}" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}
