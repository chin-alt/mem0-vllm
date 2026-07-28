#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:v0.10.2rc1-310p}"
CONTAINER_NAME="${CONTAINER_NAME:-memranker-gte-vllm-310p}"
PULL_IMAGE="${PULL_IMAGE:-1}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
HOST_REPO_PATH="${HOST_REPO_PATH:-${REPO_ROOT}}"
HOST_DATA_PATH="${HOST_DATA_PATH:-/home/reranker_experiment/data/latency_delay}"
HOST_MODEL_PATH="${HOST_MODEL_PATH:-/home/reranker_experiment/model/GTE/GTE}"
HOST_OUTPUT_PATH="${HOST_OUTPUT_PATH:-/home/reranker_experiment/output4b/business_matrix_GTE_vllm}"
DATASET="${DATASET:-all}"
DTYPE="${DTYPE:-float16}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_LENGTH}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${BATCH_SIZE}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
INSTALL_EVAL_DEPS="${INSTALL_EVAL_DEPS:-1}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirrors.huaweicloud.com/repository/pypi/simple}"

for path in "${HOST_REPO_PATH}" "${HOST_DATA_PATH}" "${HOST_MODEL_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[missing] ${path}" >&2
    exit 3
  fi
done
mkdir -p "${HOST_OUTPUT_PATH}"

if ! command -v npu-smi >/dev/null 2>&1; then
  echo "[missing] npu-smi is not available on the host" >&2
  exit 3
fi
npu-smi info >/dev/null

if [[ "${PULL_IMAGE}" == "1" ]]; then
  docker pull "${IMAGE}"
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

docker_args=(
  run --rm
  --name "${CONTAINER_NAME}"
  --shm-size=16g
  --network=host
  --privileged=true
  --user root
  -v "${HOST_REPO_PATH}:/workspace/memranker"
  -v "${HOST_DATA_PATH}:/workspace/data:ro"
  -v "${HOST_MODEL_PATH}:/models/gte:ro"
  -v "${HOST_OUTPUT_PATH}:/workspace/output"
  -w /workspace/memranker
  -e "ASCEND_RT_VISIBLE_DEVICES=${DEVICE_INDEX}"
  -e ASCEND_RUNTIME_OPTIONS=NODRV
  -e BACKEND=vllm
  -e DATA_ROOT=/workspace/data
  -e MODEL_PATH=/models/gte
  -e OUTPUT_ROOT=/workspace/output
  -e "DATASET=${DATASET}"
  -e "DTYPE=${DTYPE}"
  -e "MAX_LENGTH=${MAX_LENGTH}"
  -e "BATCH_SIZE=${BATCH_SIZE}"
  -e "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
  -e "MAX_NUM_SEQS=${MAX_NUM_SEQS}"
  -e "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}"
  -e "INSTALL_EVAL_DEPS=${INSTALL_EVAL_DEPS}"
  -e "PIP_INDEX_URL=${PIP_INDEX_URL}"
  -e ENFORCE_EAGER=1
  -e HF_HUB_OFFLINE=1
  -e TRANSFORMERS_OFFLINE=1
)

for mount in /usr/local/Ascend/driver /usr/local/Ascend/firmware /usr/local/dcmi; do
  [[ -e "${mount}" ]] && docker_args+=(-v "${mount}:${mount}:ro")
done
[[ -e /etc/ascend_install.info ]] && docker_args+=(
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro
)

if [[ -x /usr/local/bin/npu-smi ]]; then
  docker_args+=(-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro)
elif [[ -x /usr/local/sbin/npu-smi ]]; then
  docker_args+=(-v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi:ro)
fi

docker "${docker_args[@]}" "${IMAGE}" bash -lc '
  set -euo pipefail
  python scripts/check_gte_vllm_310p_env.py --model_path "${MODEL_PATH}"
  if ! python -c "import openpyxl" >/dev/null 2>&1; then
    if [[ "${INSTALL_EVAL_DEPS}" != "1" ]]; then
      echo "[missing] openpyxl; rerun with INSTALL_EVAL_DEPS=1" >&2
      exit 4
    fi
    python -m pip install openpyxl==3.1.5 -i "${PIP_INDEX_URL}"
  fi
  bash scripts/eval_business_gte_310p.sh
'
