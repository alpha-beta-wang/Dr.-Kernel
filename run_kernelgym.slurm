#!/bin/bash

#SBATCH -J kernelgym
#SBATCH -o logs/kergym-%j.out
#SBATCH -e logs/kergym-%j.err

#SBATCH -p r8nv-gpu-hw
#SBATCH -t 0-12:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --gres-flags=enforce-binding
#SBATCH --qos=gpu-normal
#SBATCH --cpus-per-task=16

set -euo pipefail

echo "Job start at $(date "+%Y-%m-%d %H:%M:%S")"
echo "Submit dir: ${SLURM_SUBMIT_DIR}"
echo "Job run on host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ---------------------------
# ulimit
# ---------------------------
ulimit -l unlimited
ulimit -v unlimited
ulimit -n 65535
ulimit -u 4125556

# ---------------------------
# modules
# ---------------------------
source /tools/module_env.sh
module purge

module load cluster-tools/v1.0
module load slurm-tools/v1.0

# 不再使用 python3/3.8.16 作为运行环境
# 只保留你集群里现成的 CUDA 模块
# module load cuda-cudnn/11.6-8.4.1

echo "===== MODULES ====="
module list || true
echo "which gcc: $(which gcc || true)"
echo "which python: $(which python || true)"
echo "which python3: $(which python3 || true)"

cluster-quota || true
nvidia-smi --format=csv --query-gpu=name,driver_version,power.limit || true

echo "Using GPU(s) ${CUDA_VISIBLE_DEVICES:-unset}"
echo "This job is assigned the following resources by SLURM:"
scontrol show jobid "$SLURM_JOB_ID" -dd | awk '/IDX/ {print $2, $4}' || true

# ---------------------------
# monitor
# ---------------------------
(sleep 3h && slurm-gpu-atop-log-stats "$SLURM_JOB_ID" "${CUDA_VISIBLE_DEVICES:-0}") &
echo "Main program continues to run. Monitoring information will be exported after three hours."

# ---------------------------
# paths
# ---------------------------
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
VENV_DIR="${PROJECT_DIR}/.venv-kgym"

SOFTWARE_DIR="$(cd "${PROJECT_DIR}/../software" && pwd)"
REDIS_BIN="${SOFTWARE_DIR}/redis-7.2.5/src/redis-server"
REDIS_CLI="${SOFTWARE_DIR}/redis-7.2.5/src/redis-cli"
export PATH="${SOFTWARE_DIR}/redis-7.2.5/src:${PATH}"

cd "${PROJECT_DIR}"

mkdir -p logs
mkdir -p "${HOME}/redis_data"

# ---------------------------
# activate uv venv
# ---------------------------
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "[ERROR] uv environment not found: ${VENV_DIR}"
    echo "Please create it first on the login node:"
    echo "  uv python install 3.10"
    echo "  uv venv --python 3.10 .venv-kgym"
    echo "  source .venv-kgym/bin/activate"
    echo "  uv pip install -U pip setuptools wheel"
    echo "  uv pip install -r requirements.txt"
    echo "  uv pip install pydantic-settings"
    exit 1
fi

source "${VENV_DIR}/bin/activate"

echo "===== PYTHON PACKAGE CHECK ====="
python - <<'PY'
mods = ["fastapi", "uvicorn", "redis", "torch"]
ok = True
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", "no_version"))
    except Exception as e:
        ok = False
        print(m, "FAIL", repr(e))
if not ok:
    raise SystemExit("Missing required Python packages")
PY

echo "===== PYTHON ENV ====="
echo "python: $(which python)"
python --version
pip --version

echo "===== SERVICE URL ====="
IP_ADDR=$(hostname -I | tr ' ' '\n' | grep '^10\.' | head -n 1)  # GET ThE IP ADDRESS OF THIS MACHINE
echo "The IP address of this machine is $IP_ADDR" 
PORT=$(python3 -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")  # find available port for KernelGYM serve
echo "The port for KernelGYM serve is $PORT"

# ---------------------------
# export env vars
# ---------------------------
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${PROJECT_DIR}/.cache/huggingface"
export TRANSFORMERS_CACHE="${PROJECT_DIR}/.cache/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${PROJECT_DIR}/.cache/huggingface/hub"
export TRITON_CACHE_DIR="${PROJECT_DIR}/.cache/triton"
export TMPDIR="${PROJECT_DIR}/.tmp"
export TEMP="${PROJECT_DIR}/.tmp"
export TMP="${PROJECT_DIR}/.tmp"

mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TRITON_CACHE_DIR}" "${TMPDIR}"

# 端口：同一作业内自用，固定即可；如果你经常多开作业，可以改成和 JOB_ID 相关
export REDIS_HOST=127.0.0.1
export REDIS_PORT=6380
export REDIS_PASSWORD=
export API_HOST=0.0.0.0
export API_PORT=10907
export NODE_ID="slurm-${SLURM_JOB_ID}"

# 根据申请到的 GPU 数量，自动生成 GPU_DEVICES=[0,1,2,3]
GPU_COUNT=$(python - <<'PY'
import os
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
if not cvd.strip():
    print(1)
else:
    print(len([x for x in cvd.split(",") if x.strip()]))
PY
)

export GPU_DEVICES=$(python - <<PY
n = int("${GPU_COUNT}")
print("[" + ",".join(str(i) for i in range(n)) + "]")
PY
)

# 训练/评测脚本可直接用
export KERNELGYM_SERVER_URL="http://127.0.0.1:${API_PORT}"

echo "===== EXPORTED VARS ====="
echo "REDIS_HOST=${REDIS_HOST}"
echo "REDIS_PORT=${REDIS_PORT}"
echo "API_HOST=${API_HOST}"
echo "API_PORT=${API_PORT}"
echo "NODE_ID=${NODE_ID}"
echo "GPU_DEVICES=${GPU_DEVICES}"
echo "KERNELGYM_SERVER_URL=${KERNELGYM_SERVER_URL}"

# ---------------------------
# write .env for KernelGYM
# ---------------------------
cat > .env <<EOF
REDIS_HOST=${REDIS_HOST}
REDIS_PORT=${REDIS_PORT}
REDIS_PASSWORD=${REDIS_PASSWORD}
API_HOST=${API_HOST}
API_PORT=${API_PORT}
GPU_DEVICES=${GPU_DEVICES}
NODE_ID=${NODE_ID}
EOF

echo "===== .env ====="
cat .env

# ---------------------------
# sanity check: torch in compute node
# ---------------------------
echo "===== TORCH CHECK ON COMPUTE NODE ====="
python - <<'PY'
import torch, os
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("cuda available:", torch.cuda.is_available())
print("cuda device count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"gpu{i}:", torch.cuda.get_device_name(i))
PY

# ---------------------------
# cleanup
# ---------------------------
cleanup() {
    echo "===== CLEANUP ====="
    if [ -n "${WORKER_MONITOR_PID:-}" ] && kill -0 "${WORKER_MONITOR_PID}" 2>/dev/null; then
        kill "${WORKER_MONITOR_PID}" || true
    fi
    if [ -n "${API_PID:-}" ] && kill -0 "${API_PID}" 2>/dev/null; then
        kill "${API_PID}" || true
    fi
    if [ -n "${WORKER_PIDS:-}" ]; then
        for pid in ${WORKER_PIDS}; do
            if kill -0 "${pid}" 2>/dev/null; then
                kill "${pid}" || true
            fi
        done
    fi
    if [ -n "${KERNELGYM_PID:-}" ] && kill -0 "${KERNELGYM_PID}" 2>/dev/null; then
        kill "${KERNELGYM_PID}" || true
    fi
    if [ -n "${REDIS_PID:-}" ] && kill -0 "${REDIS_PID}" 2>/dev/null; then
        kill "${REDIS_PID}" || true
    fi
}
trap cleanup EXIT

# ---------------------------
# start redis
# ---------------------------
echo "===== START REDIS ====="
cat > redis.conf <<EOF
bind 127.0.0.1
port ${REDIS_PORT}
protected-mode yes
daemonize no
dir ${HOME}/redis_data
logfile ${PROJECT_DIR}/logs/redis-${SLURM_JOB_ID}.log
save ""
appendonly no
EOF

"${REDIS_BIN}" redis.conf &
REDIS_PID=$!

sleep 3
"${REDIS_CLI}" -h 127.0.0.1 -p "${REDIS_PORT}" ping

# ---------------------------
# start KernelGYM
# ---------------------------
echo "===== START KERNELGYM ====="
LOG_DIR="logs"
KERNELGYM_LOG="${LOG_DIR}/kernelgym-${SLURM_JOB_ID}.log"
REDIS_KEY_PREFIX="kernelgym"

GPU_LIST=$(python - <<'PY'
import json
import os

raw = os.environ.get("GPU_DEVICES", "")
if not raw:
    print("0")
    raise SystemExit(0)
try:
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        print(" ".join(str(x) for x in parsed))
    else:
        print(str(parsed))
except Exception:
    print(" ".join([x.strip() for x in raw.split(",") if x.strip()]))
PY
)

echo "Checking Redis..." | tee "${KERNELGYM_LOG}"
REDIS_AUTH_ARGS=()
if [ -n "${REDIS_PASSWORD}" ]; then
    REDIS_AUTH_ARGS=(-a "${REDIS_PASSWORD}")
fi

echo "Starting API server..." | tee -a "${KERNELGYM_LOG}"
python -m kernelgym.server.api.server > "${LOG_DIR}/api_server.log" 2>&1 &
API_PID=$!
echo "API server PID: ${API_PID}" | tee -a "${KERNELGYM_LOG}"

echo "===== WAIT FOR API READY ====="
API_READY=0
for i in $(seq 1 180); do
    if curl -sS "http://127.0.0.1:${API_PORT}/docs" >/dev/null 2>&1; then
        API_READY=1
        echo "API server is up."
        break
    fi
    sleep 2
done

if [ "${API_READY}" -ne 1 ]; then
    echo "[ERROR] API server did not come up in time."
    ps -fp "${API_PID}" || true
    ss -ltnp | grep "${API_PORT}" || true
    tail -n 100 "${LOG_DIR}/api_server.log" || true
    exit 1
fi

if command -v redis-cli >/dev/null 2>&1; then
    redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
        DEL "${REDIS_KEY_PREFIX}:expected_workers" > /dev/null 2>&1 || true
fi

echo "Starting GPU workers..." | tee -a "${KERNELGYM_LOG}"
WORKER_PIDS=""
EXPECTED_WORKERS=""
for gpu in ${GPU_LIST}; do
    WORKER_ID="worker_gpu_${gpu}"
    echo "Launching ${WORKER_ID} on cuda:${gpu}" | tee -a "${KERNELGYM_LOG}"
    python -m kernelgym.worker.single_worker \
        --worker-id "${WORKER_ID}" \
        --device "cuda:${gpu}" \
        --persistent \
        > "${LOG_DIR}/worker_gpu_${gpu}.log" 2>&1 &
    WORKER_PID=$!
    WORKER_PIDS="${WORKER_PIDS} ${WORKER_PID}"
    EXPECTED_WORKERS="${EXPECTED_WORKERS} ${WORKER_ID}"

    if command -v redis-cli >/dev/null 2>&1; then
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
            SADD "${REDIS_KEY_PREFIX}:expected_workers" "${WORKER_ID}" > /dev/null 2>&1 || true
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
            HSET "${REDIS_KEY_PREFIX}:expected_worker:${WORKER_ID}" \
            device "cuda:${gpu}" hostname "$(hostname)" node_id "${NODE_ID:-}" > /dev/null 2>&1 || true
        redis-cli -h "${REDIS_HOST}" -p "${REDIS_PORT}" "${REDIS_AUTH_ARGS[@]}" --no-auth-warning \
            HSET "${REDIS_KEY_PREFIX}:worker_process:${WORKER_ID}" \
            pid "${WORKER_PID}" start_time "$(date -Iseconds)" device "cuda:${gpu}" > /dev/null 2>&1 || true
    fi

    echo "Worker PID: ${WORKER_PID}" | tee -a "${KERNELGYM_LOG}"
    sleep 0.3
done

echo "KernelGym core services started (monitor deferred)." | tee -a "${KERNELGYM_LOG}"

# ---------------------------
# check api port and process before health check
# ---------------------------
echo "===== PORT CHECK BEFORE HEALTH ====="
ss -ltnp | grep "${API_PORT}" || true
ps -fp "${API_PID}" || true
pgrep -af "kernelgym|uvicorn|fastapi" || true

# ---------------------------
# wait for workers ready
# ---------------------------
echo "===== WAIT FOR WORKERS READY ====="
READY=0
READY_STABLE_COUNT=0
for i in $(seq 1 180); do
    STATUS_JSON=$(curl -sS "http://127.0.0.1:${API_PORT}/workers/status" || true)
    ALL_WORKERS_PRESENT=1
    for worker_id in ${EXPECTED_WORKERS}; do
        if ! echo "${STATUS_JSON}" | grep -q "\"${worker_id}\""; then
            ALL_WORKERS_PRESENT=0
            READY_STABLE_COUNT=0
            break
        fi
    done

    if [ "${ALL_WORKERS_PRESENT}" -eq 1 ]; then
        READY_STABLE_COUNT=$((READY_STABLE_COUNT + 1))
        echo "Workers registered check ${READY_STABLE_COUNT}/3"
        if [ "${READY_STABLE_COUNT}" -ge 3 ]; then
            READY=1
            echo "KernelGYM workers are stably registered."
            break
        fi
    fi
    sleep 2
done

if [ "${READY}" -ne 1 ]; then
    echo "[ERROR] KernelGYM workers did not become ready in time."

    echo "===== PROCESS CHECK ====="
    ps -fp "${API_PID}" || true
    pgrep -af "kernelgym|uvicorn|fastapi" || true

    echo "===== PORT CHECK ====="
    ss -ltnp | grep "${API_PORT}" || true

    echo "===== redis log ====="
    tail -n 12 "logs/redis-${SLURM_JOB_ID}.log" || true

    echo "===== kernelgym launcher log ====="
    tail -n 2 "logs/kernelgym-${SLURM_JOB_ID}.log" || true

    echo "===== api/server logs ====="
    tail -n 50 logs/api_server.log || true
    tail -n 50 logs/api.log || true
    tail -n 50 logs/server.log || true

    echo "===== worker logs ====="
    tail -n 50 logs/worker_gpu_0.log || true
    tail -n 30 logs/worker_gpu_1.log || true
    tail -n 30 logs/worker_gpu_2.log || true
    tail -n 30 logs/worker_gpu_3.log || true

    exit 1
fi

echo "===== START WORKER MONITOR ====="
python -m kernelgym.worker.worker_monitor --persistent > "${LOG_DIR}/worker_monitor.log" 2>&1 &
WORKER_MONITOR_PID=$!
echo "Worker monitor PID: ${WORKER_MONITOR_PID}" | tee -a "${KERNELGYM_LOG}"

sleep 3
ps -fp "${WORKER_MONITOR_PID}" || true
tail -n 20 "${LOG_DIR}/worker_monitor.log" || true

echo "===== FINAL STATUS CHECK ====="
curl -sS "http://127.0.0.1:${API_PORT}/workers/status"
echo
curl -sS "http://127.0.0.1:${API_PORT}/docs" >/dev/null && echo "API docs reachable"

# ---------------------------
# TODO: put your training/eval command here
# echo "===== RUN SMOKE TEST ====="
# python3 smoke_test.py > "logs/smoke_test-${SLURM_JOB_ID}.log" 2>&1 || {
#     echo "[ERROR] smoke test failed"
#     tail -n 20 "logs/smoke_test-${SLURM_JOB_ID}.log" || true
#     exit 1
# }

# echo "===== SMOKE TEST LOG ====="
# tail -n 20 "logs/smoke_test-${SLURM_JOB_ID}.log" || true
echo "===== KERNEL BENCH EVAL ====="
python3 kernel_bench_eval.py > "logs/kernel_bench_eval-${SLURM_JOB_ID}.log" 2>&1 || {
    echo "[ERROR] kernel bench eval failed"
    tail -n 20 "logs/kernel_bench_eval-${SLURM_JOB_ID}.log" || true
    exit 1
}

echo "===== KERNEL BENCH EVAL ====="
tail -n 20 "logs/kernel_bench_eval-${SLURM_JOB_ID}.log" || true
# ---------------------------
# 例子：
# cd drkernel
# bash kernel/scripts/sft/8b-coldstart.sh
#
# 如果你暂时只想把服务跑起来做验证，先不加训练命令



echo "===== SERVICE IS READY ====="
echo "KERNELGYM_SERVER_URL=${KERNELGYM_SERVER_URL}"
echo "Logs:"
echo "  logs/redis-${SLURM_JOB_ID}.log"
echo "  logs/kernelgym-${SLURM_JOB_ID}.log"


# 没有训练命令时，保持作业存活，方便你观察日志
tail -f "logs/kernelgym-${SLURM_JOB_ID}.log"
