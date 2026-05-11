#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

activate_drkernel_venv
ensure_common_dirs

python "${SCRIPT_DIR}/download_hf_assets.py" \
    --root "${HF_MODEL_ROOT}" \
    --preset models \
    "$@"
