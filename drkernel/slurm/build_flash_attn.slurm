#!/bin/bash

#SBATCH -J drk-fa-build
#SBATCH -o ../logs/drk-fa-build-%j.out
#SBATCH -e ../logs/drk-fa-build-%j.err
#SBATCH -p r8nv-gpu-hw
#SBATCH -t 0-04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos=gpu-normal
#SBATCH --cpus-per-task=16

set -euo pipefail

log_section() {
    echo
    echo "===== $1 ====="
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv-drkernel"
SOURCE_DIR="${PROJECT_DIR}/../../software/flash-attn-src"
WHEEL_DIR="${PROJECT_DIR}/../../software/flash-attn-dist"
BUILD_ROOT="/tmp/flash-attn-build-${SLURM_JOB_ID}"
BUILD_SRC="${BUILD_ROOT}/src"

cleanup() {
    local exit_code=$?
    log_section "SUMMARY"
    echo "Exit code: ${exit_code}"
    echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
    if [[ -d "${WHEEL_DIR}" ]]; then
        echo "Wheel output:"
        ls -lh "${WHEEL_DIR}" || true
    fi
    if [[ "${exit_code}" -eq 0 ]]; then
        echo "[SUCCESS] flash-attn build and install completed"
    else
        echo "[FAILED] flash-attn build/install failed"
    fi
}

trap cleanup EXIT

echo "Job start at $(date '+%Y-%m-%d %H:%M:%S')"
echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"

ulimit -l unlimited
ulimit -v unlimited
ulimit -n 65535
ulimit -u 4125556

source /tools/module_env.sh
module purge
module load cluster-tools/v1.0
module load slurm-tools/v1.0
module load gcc/9.3.0
module load cuda-cudnn/12.4-9.1.1

log_section "SYSTEM INFO"
module list || true
echo "which gcc: $(which gcc || true)"
echo "which g++: $(which g++ || true)"
echo "which nvcc: $(which nvcc || true)"
gcc --version | head -n 1 || true
g++ --version | head -n 1 || true
nvcc -V | tail -n 1 || true
nvidia-smi --format=csv --query-gpu=index,name,driver_version,memory.total || true

if [[ ! -f "${VENV_DIR}/bin/activate" ]]; then
    echo "Virtualenv not found at ${VENV_DIR}" >&2
    exit 1
fi

if [[ ! -d "${SOURCE_DIR}" ]]; then
    echo "Source tree not found at ${SOURCE_DIR}" >&2
    exit 1
fi

mkdir -p "${WHEEL_DIR}"
command rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_SRC}"
rsync -a --delete --exclude build --exclude dist "${SOURCE_DIR}/" "${BUILD_SRC}/"

export CUDA_HOME="$(dirname "$(dirname "$(which nvcc)")")"
export PATH="/tools/cluster-software/gcc/gcc-9.3.0/bin:${CUDA_HOME}/bin:/usr/bin:/bin:${PATH}"
export CC="/tools/cluster-software/gcc/gcc-9.3.0/bin/gcc"
export CXX="/tools/cluster-software/gcc/gcc-9.3.0/bin/g++"
export CUDAHOSTCXX="/tools/cluster-software/gcc/gcc-9.3.0/bin/g++"
export MAX_JOBS=4
export NVCC_THREADS=4
export FLASH_ATTENTION_FORCE_BUILD=TRUE
export FLASH_ATTN_CUDA_ARCHS=80
export UV_LINK_MODE=copy

source "${VENV_DIR}/bin/activate"

log_section "BUILD ENV"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "CUDAHOSTCXX=${CUDAHOSTCXX}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "FLASH_ATTN_CUDA_ARCHS=${FLASH_ATTN_CUDA_ARCHS}"
echo "MAX_JOBS=${MAX_JOBS}"

cd "${BUILD_SRC}"

log_section "BUILD WHEEL"
pip wheel . -w "${WHEEL_DIR}" --no-build-isolation --no-deps -v

LATEST_WHEEL="$(ls -1t "${WHEEL_DIR}"/flash_attn-*.whl | head -n 1)"
if [[ -z "${LATEST_WHEEL}" ]]; then
    echo "No flash-attn wheel produced" >&2
    exit 1
fi

log_section "INSTALL WHEEL"
pip install --no-deps --force-reinstall "${LATEST_WHEEL}"

log_section "VERIFY IMPORT"
python - <<'PY'
import flash_attn
from flash_attn.bert_padding import pad_input, unpad_input

print("flash_attn_version", getattr(flash_attn, "__version__", "unknown"))
print("flash_attn_file", flash_attn.__file__)
print("bert_padding_ok", callable(pad_input), callable(unpad_input))
PY

EXT_PATH="$(python - <<'PY'
import pathlib
import flash_attn

root = pathlib.Path(flash_attn.__file__).resolve().parent
matches = sorted(root.glob("flash_attn_2_cuda*.so"))
print(matches[0] if matches else "")
PY
)"

if [[ -n "${EXT_PATH}" ]]; then
    log_section "EXTENSION LINKAGE"
    ldd "${EXT_PATH}" || true
fi
