#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

if [[ -z "${UV_BIN}" ]]; then
    echo "[ERROR] uv not found in PATH" >&2
    exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
ABI_FLAG="${ABI_FLAG:-FALSE}"
FLASH_ATTN_URL="${FLASH_ATTN_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abi${ABI_FLAG}-cp310-cp310-linux_x86_64.whl}"
FLASH_ATTN_WHEEL="${FLASH_ATTN_CACHE_DIR}/$(basename "${FLASH_ATTN_URL}")"

ensure_common_dirs

cd "${DRKERNEL_ROOT}"
git submodule update --init

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

"${UV_BIN}" venv --python "${PYTHON_VERSION}" "${DRKERNEL_VENV_DIR}"
# shellcheck disable=SC1090
source "${DRKERNEL_VENV_DIR}/bin/activate"

uv pip install -U pip setuptools wheel
uv pip install -e "${DRKERNEL_ROOT}/verl" --no-build-isolation --no-deps

uv pip install --no-cache-dir "ray==2.47.1"
uv pip install --no-cache-dir \
    "vllm==0.10.2" "torch==2.8.0" "torchvision==0.23.0" "torchaudio==2.8.0" \
    tensordict torchdata "transformers[hf_xet]==4.56.0" accelerate datasets peft hf-transfer \
    "numpy<2.0.0" "pyarrow==15.0.2" pandas codetiming hydra-core pylatexenc qwen-vl-utils \
    dill pybind11 liger-kernel mathruler decord torchcodec pytest yapf py-spy pre-commit ruff \
    pipx sandbox-fusion logfire gradio huggingface_hub "protobuf==3.20" "wandb==0.16.6"

if [[ ! -f "${FLASH_ATTN_WHEEL}" ]]; then
    curl -L --fail --retry 3 -o "${FLASH_ATTN_WHEEL}" "${FLASH_ATTN_URL}"
fi
uv pip install --no-cache-dir "${FLASH_ATTN_WHEEL}"

python - <<'PY'
import importlib
mods = ["torch", "ray", "vllm", "datasets", "huggingface_hub"]
for name in mods:
    mod = importlib.import_module(name)
    print(name, getattr(mod, "__version__", "ok"))
PY
