#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.1.RC1-300I-Duo-py311-openeuler24.03-lts}"
CONTAINER_NAME="${CONTAINER_NAME:-memranker-mindie-310p}"
HOST_MODEL_PATH="${HOST_MODEL_PATH:?Set HOST_MODEL_PATH to the Qwen3-Reranker model directory on the host}"
HOST_REPO_PATH="${HOST_REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
MINDIE_MODEL_NAME="${MINDIE_MODEL_NAME:-qwen3-reranker-4b}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-32}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-32768}"
PULL_IMAGE="${PULL_IMAGE:-1}"

if [[ ! -f "${HOST_MODEL_PATH}/config.json" ]]; then
  echo "[error] HOST_MODEL_PATH has no config.json: ${HOST_MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${HOST_REPO_PATH}/business_eval_mindie.py" ]]; then
  echo "[error] HOST_REPO_PATH is not this repository: ${HOST_REPO_PATH}" >&2
  exit 2
fi
if docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
  echo "[error] container already exists: ${CONTAINER_NAME}" >&2
  echo "[error] stop/remove it explicitly, or choose another CONTAINER_NAME." >&2
  exit 2
fi

if [[ -f /usr/local/Ascend/driver/version.info ]]; then
  echo "[host] $(grep -m1 '^Version=' /usr/local/Ascend/driver/version.info || true)"
fi
npu-smi info >/dev/null
NPU_SMI_PATH="$(command -v npu-smi)"
MOUNT_ARGS=(
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v "${NPU_SMI_PATH}:${NPU_SMI_PATH}:ro"
)
[[ -d /usr/local/Ascend/firmware ]] && MOUNT_ARGS+=(
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro
)
[[ -d /usr/local/dcmi ]] && MOUNT_ARGS+=(
  -v /usr/local/dcmi:/usr/local/dcmi:ro
)

if [[ "${PULL_IMAGE}" == "1" ]]; then
  docker pull "${IMAGE}"
fi

docker run -dit \
  --name "${CONTAINER_NAME}" \
  --net=host \
  --shm-size=64g \
  --privileged=true \
  --user root \
  --entrypoint /bin/bash \
  -e ASCEND_RUNTIME_OPTIONS=NODRV \
  -e ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES}" \
  -e MODEL_PATH=/models/qwen3-reranker \
  -e MINDIE_MODEL_NAME="${MINDIE_MODEL_NAME}" \
  -e MAX_LENGTH="${MAX_LENGTH}" \
  -e MAX_BATCH_SIZE="${MAX_BATCH_SIZE}" \
  -e MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS}" \
  "${MOUNT_ARGS[@]}" \
  -v "${HOST_MODEL_PATH}:/models/qwen3-reranker:rw" \
  -v "${HOST_REPO_PATH}:/workspace/memranker:ro" \
  "${IMAGE}" -lc "sleep infinity"

docker exec "${CONTAINER_NAME}" \
  bash /workspace/memranker/scripts/start_mindie_qwen3_reranker_service.sh

echo "[done] MindIE container=${CONTAINER_NAME}"
echo "[done] endpoint=http://127.0.0.1:1025/v1/completions"
echo "[log]  docker exec ${CONTAINER_NAME} tail -f /tmp/mindie-reranker/mindie-service.log"
