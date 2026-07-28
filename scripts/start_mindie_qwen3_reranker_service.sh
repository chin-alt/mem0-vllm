#!/usr/bin/env bash
set -euo pipefail

# Run this script inside the official MindIE 2.1.RC1 300I-Duo container.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${MODEL_PATH:-/models/qwen3-reranker-4b}"
MINDIE_MODEL_NAME="${MINDIE_MODEL_NAME:-qwen3-reranker-4b}"
MINDIE_CONFIG="${MINDIE_CONFIG:-/usr/local/Ascend/mindie/latest/mindie-service/conf/config.json}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
MINDIE_PORT="${MINDIE_PORT:-1025}"
MINDIE_MANAGEMENT_PORT="${MINDIE_MANAGEMENT_PORT:-1026}"
MINDIE_METRICS_PORT="${MINDIE_METRICS_PORT:-1027}"
MINDIE_LISTEN_ADDRESS="${MINDIE_LISTEN_ADDRESS:-127.0.0.1}"
MINDIE_LOG="${MINDIE_LOG:-/tmp/mindie-reranker/mindie-service.log}"
MINDIE_PID_FILE="${MINDIE_PID_FILE:-/tmp/mindie-reranker/mindie-service.pid}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-900}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export MINDIE_LOG_TO_STDOUT="${MINDIE_LOG_TO_STDOUT:-1}"
export MINDIE_LOG_LEVEL="${MINDIE_LOG_LEVEL:-INFO}"

IFS=',' read -r -a visible_devices <<< "${ASCEND_RT_VISIBLE_DEVICES}"
logical_devices=""
for device_index in "${!visible_devices[@]}"; do
  [[ -n "${logical_devices}" ]] && logical_devices+=","
  logical_devices+="${device_index}"
done

source_if_present() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    # shellcheck disable=SC1090
    source "${path}"
  fi
}

source_if_present /usr/local/Ascend/ascend-toolkit/set_env.sh
source_if_present /usr/local/Ascend/cann/set_env.sh
source_if_present /usr/local/Ascend/nnal/atb/set_env.sh
source_if_present /usr/local/Ascend/atb-models/set_env.sh
source_if_present /usr/local/Ascend/atb_llm/set_env.sh
source_if_present /usr/local/Ascend/mindie/latest/mindie-llm/set_env.sh
source_if_present /usr/local/Ascend/mindie/latest/mindie-service/set_env.sh

if [[ ! -f "${MINDIE_CONFIG}" ]]; then
  alternate_config=/usr/local/Ascend/mindie/latest/mindieservice/conf/config.json
  if [[ -f "${alternate_config}" ]]; then
    MINDIE_CONFIG="${alternate_config}"
  else
    echo "[error] MindIE config.json not found: ${MINDIE_CONFIG}" >&2
    exit 2
  fi
fi

daemon_path="$(dirname "${MINDIE_CONFIG}")/../bin/mindieservice_daemon"
if [[ ! -x "${daemon_path}" ]]; then
  echo "[error] mindieservice_daemon not found beside ${MINDIE_CONFIG}" >&2
  exit 2
fi

mkdir -p "$(dirname "${MINDIE_LOG}")"
"${PYTHON_BIN}" "${REPO_ROOT}/scripts/configure_mindie_qwen3_reranker.py" \
  --config "${MINDIE_CONFIG}" \
  --model_path "${MODEL_PATH}" \
  --model_name "${MINDIE_MODEL_NAME}" \
  --npu_devices "${logical_devices}" \
  --max_length "${MAX_LENGTH}" \
  --max_batch_size "${MAX_BATCH_SIZE}" \
  --max_prefill_tokens "${MAX_PREFILL_TOKENS}" \
  --port "${MINDIE_PORT}" \
  --management_port "${MINDIE_MANAGEMENT_PORT}" \
  --metrics_port "${MINDIE_METRICS_PORT}" \
  --listen_address "${MINDIE_LISTEN_ADDRESS}" \
  --patch_model_dtype

if [[ -f "${MINDIE_PID_FILE}" ]]; then
  old_pid="$(cat "${MINDIE_PID_FILE}" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
    echo "[mindie] service already running, pid=${old_pid}"
    exit 0
  fi
fi

run_dir="$(dirname "${MINDIE_LOG}")/run"
mkdir -p "${run_dir}"
cd "${run_dir}"
nohup "${daemon_path}" >"${MINDIE_LOG}" 2>&1 &
daemon_pid=$!
printf '%s\n' "${daemon_pid}" >"${MINDIE_PID_FILE}"
echo "[mindie] starting pid=${daemon_pid}; log=${MINDIE_LOG}"

deadline=$((SECONDS + STARTUP_TIMEOUT))
while (( SECONDS < deadline )); do
  if ! kill -0 "${daemon_pid}" 2>/dev/null; then
    echo "[error] MindIE exited during startup" >&2
    tail -n 200 "${MINDIE_LOG}" >&2 || true
    exit 3
  fi
  if grep -q "Daemon start success" "${MINDIE_LOG}"; then
    echo "[mindie] ready: http://${MINDIE_LISTEN_ADDRESS}:${MINDIE_PORT}/v1/completions"
    exit 0
  fi
  sleep 5
done

echo "[error] MindIE did not become ready within ${STARTUP_TIMEOUT}s" >&2
tail -n 200 "${MINDIE_LOG}" >&2 || true
exit 4
