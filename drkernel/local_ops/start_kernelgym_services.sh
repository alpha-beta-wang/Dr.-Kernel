#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common_env.sh"

KGYM_PYTHON="${KERNELGYM_VENV_DIR}/bin/python"
REDIS_BIN="${REDIS_INSTALL_ROOT}/src/redis-server"
REDIS_CLI="${REDIS_INSTALL_ROOT}/src/redis-cli"

start_kernelgym_services() {
    if [[ ! -x "${KGYM_PYTHON}" ]]; then
        echo "[ERROR] KernelGYM python not found: ${KGYM_PYTHON}" >&2
        return 1
    fi
    if [[ ! -x "${REDIS_BIN}" ]]; then
        echo "[ERROR] redis-server not found: ${REDIS_BIN}" >&2
        return 1
    fi

    local project_dir="${KERNELGYM_ROOT}"
    local log_dir="${project_dir}/logs"
    local tmp_dir="${project_dir}/.tmp"
    mkdir -p "${log_dir}" "${tmp_dir}" "${HOME}/redis_data"

    export PYTHONUNBUFFERED=1
    export TOKENIZERS_PARALLELISM=false
    export HF_HOME="${project_dir}/.cache/huggingface"
    export TRANSFORMERS_CACHE="${project_dir}/.cache/huggingface/transformers"
    export HUGGINGFACE_HUB_CACHE="${project_dir}/.cache/huggingface/hub"
    export TRITON_CACHE_DIR="${project_dir}/.cache/triton"
    export TMPDIR="${TMPDIR:-${tmp_dir}}"
    export TEMP="${TEMP:-${TMPDIR}}"
    export TMP="${TMP:-${TMPDIR}}"
    mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRITON_CACHE_DIR}"

    export REDIS_HOST="${REDIS_HOST:-127.0.0.1}"
    export REDIS_PORT="${REDIS_PORT:-6380}"
    export REDIS_PASSWORD="${REDIS_PASSWORD:-}"
    export API_HOST="${API_HOST:-0.0.0.0}"
    export API_PORT="${API_PORT:-10907}"
    export NODE_ID="${NODE_ID:-slurm-${SLURM_JOB_ID:-manual}}"

    local gpu_count
    gpu_count="$("${KGYM_PYTHON}" - <<'PY'
import os
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if not cvd.strip():
    print(1)
else:
    print(len([x for x in cvd.split(",") if x.strip()]))
PY
)"
    export GPU_DEVICES="$("${KGYM_PYTHON}" - <<PY
n = int("${gpu_count}")
print("[" + ",".join(str(i) for i in range(n)) + "]")
PY
)"
    export KERNELGYM_SERVER_URL="http://127.0.0.1:${API_PORT}"

    cat > "${project_dir}/.env" <<EOF
REDIS_HOST=${REDIS_HOST}
REDIS_PORT=${REDIS_PORT}
REDIS_PASSWORD=${REDIS_PASSWORD}
API_HOST=${API_HOST}
API_PORT=${API_PORT}
GPU_DEVICES=${GPU_DEVICES}
NODE_ID=${NODE_ID}
EOF

    cat > "${project_dir}/redis.conf" <<EOF
bind 127.0.0.1
port ${REDIS_PORT}
protected-mode yes
daemonize no
dir ${HOME}/redis_data
logfile ${log_dir}/redis-${SLURM_JOB_ID:-manual}.log
save ""
appendonly no
EOF

    "${REDIS_BIN}" "${project_dir}/redis.conf" &
    export REDIS_PID=$!
    sleep 3
    "${REDIS_CLI}" -h 127.0.0.1 -p "${REDIS_PORT}" ping

    (cd "${project_dir}" && "${KGYM_PYTHON}" -m kernelgym.server.api.server > "${log_dir}/api_server.log" 2>&1 &) 
    export API_PID
    API_PID="$(pgrep -f "kernelgym.server.api.server" | tail -n 1)"

    local ready=0
    for _ in $(seq 1 180); do
        if curl -sS "http://127.0.0.1:${API_PORT}/docs" >/dev/null 2>&1; then
            ready=1
            break
        fi
        sleep 2
    done
    if [[ "${ready}" -ne 1 ]]; then
        echo "[ERROR] KernelGYM API did not become ready" >&2
        return 1
    fi

    export WORKER_PIDS=""
    local gpu_list
    gpu_list="$("${KGYM_PYTHON}" - <<'PY'
import json, os
raw = os.environ.get("GPU_DEVICES", "[]")
parsed = json.loads(raw)
print(" ".join(str(x) for x in parsed))
PY
)"
    for gpu in ${gpu_list}; do
        local worker_id="worker_gpu_${gpu}"
        (cd "${project_dir}" && "${KGYM_PYTHON}" -m kernelgym.worker.single_worker \
            --worker-id "${worker_id}" \
            --device "cuda:${gpu}" \
            --persistent > "${log_dir}/worker_gpu_${gpu}.log" 2>&1 &) 
        local pid
        pid="$(pgrep -f "single_worker --worker-id ${worker_id}" | tail -n 1)"
        WORKER_PIDS="${WORKER_PIDS} ${pid}"
        sleep 1
    done

    ready=0
    for _ in $(seq 1 180); do
        if curl -sS "http://127.0.0.1:${API_PORT}/workers/status" | grep -q 'worker_gpu_'; then
            ready=1
            break
        fi
        sleep 2
    done
    if [[ "${ready}" -ne 1 ]]; then
        echo "[ERROR] KernelGYM workers did not become ready" >&2
        return 1
    fi

    (cd "${project_dir}" && "${KGYM_PYTHON}" -m kernelgym.worker.worker_monitor > "${log_dir}/worker_monitor.log" 2>&1 &) 
    export WORKER_MONITOR_PID
    WORKER_MONITOR_PID="$(pgrep -f "kernelgym.worker.worker_monitor" | tail -n 1)"
    echo "[INFO] KernelGYM ready at ${KERNELGYM_SERVER_URL}"
}

stop_kernelgym_services() {
    if [[ -n "${WORKER_MONITOR_PID:-}" ]] && kill -0 "${WORKER_MONITOR_PID}" 2>/dev/null; then
        kill "${WORKER_MONITOR_PID}" || true
    fi
    if [[ -n "${WORKER_PIDS:-}" ]]; then
        for pid in ${WORKER_PIDS}; do
            if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
                kill "${pid}" || true
            fi
        done
    fi
    if [[ -n "${API_PID:-}" ]] && kill -0 "${API_PID}" 2>/dev/null; then
        kill "${API_PID}" || true
    fi
    if [[ -n "${REDIS_PID:-}" ]] && kill -0 "${REDIS_PID}" 2>/dev/null; then
        kill "${REDIS_PID}" || true
    fi
}
