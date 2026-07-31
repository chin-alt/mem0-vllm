#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run a one-token pure ATB smoke/latency test for a W8A8SC Qwen3-Reranker.

This bypasses vLLM and MindIE Service. It calls ATB Models examples.run_pa
directly inside the same 300I-Duo image used for W8A8SC compression.

Overrides:
  HOST_MODEL_PATH          W8A8SC model directory.
  INPUT_TEXT               Complete Qwen3-Reranker prompt.
  IMAGE                    Default: MindIE 2.1.RC1 300I-Duo image.
  PULL_IMAGE               Default: 0
  ASCEND_RT_VISIBLE_DEVICES Default: 0
  TP_SIZE                  Must match compression. Default: 1
  MAX_LENGTH               Default: 1024
  MASTER_PORT              Default: 20038
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "[invalid] this script accepts environment variables, not positional arguments" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_REPO_PATH="${HOST_REPO_PATH:-${REPO_ROOT}}"
HOST_MODEL_PATH="${HOST_MODEL_PATH:-/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8SC}"
IMAGE="${IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.1.RC1-300I-Duo-py311-openeuler24.03-lts}"
PULL_IMAGE="${PULL_IMAGE:-0}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MASTER_PORT="${MASTER_PORT:-20038}"
SHM_SIZE="${SHM_SIZE:-16g}"
INPUT_TEXT="${INPUT_TEXT:-<|im_start|>system
 Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>
<|im_start|>user
<Instruct>: 判断文档是否满足查询

<Query>: 冰饮

<Document>: 这是一杯冰咖啡。<|im_end|>
<|im_start|>assistant
<think>

</think>

}"

if [[ "${PULL_IMAGE}" != "0" && "${PULL_IMAGE}" != "1" ]]; then
  echo "[invalid] PULL_IMAGE must be 0 or 1; got ${PULL_IMAGE}" >&2
  exit 2
fi
for number in TP_SIZE MAX_LENGTH MASTER_PORT; do
  value="${!number}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "[invalid] ${number} must be a positive integer; got ${value}" >&2
    exit 2
  fi
done
if ! command -v docker >/dev/null 2>&1; then
  echo "[missing] docker is not installed" >&2
  exit 3
fi
if ! command -v npu-smi >/dev/null 2>&1 || ! npu-smi info >/dev/null; then
  echo "[missing] npu-smi cannot access the host NPU/driver" >&2
  exit 3
fi
if [[ ! -f "${HOST_MODEL_PATH}/config.json" ]]; then
  echo "[missing] W8A8SC model config: ${HOST_MODEL_PATH}/config.json" >&2
  exit 3
fi
checker="${HOST_REPO_PATH}/scripts/check_qwen3_reranker_w8a8sc_310p.py"
if [[ ! -f "${checker}" ]]; then
  echo "[missing] ${checker}" >&2
  exit 3
fi
if command -v python3 >/dev/null 2>&1; then
  host_python=python3
elif command -v python >/dev/null 2>&1; then
  host_python=python
else
  echo "[missing] python3 or python is required for model validation" >&2
  exit 3
fi
"${host_python}" "${checker}" \
  --model-path "${HOST_MODEL_PATH}" \
  --expected w8a8sc

if [[ "${PULL_IMAGE}" == "1" ]]; then
  docker pull "${IMAGE}"
elif ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[missing] local image ${IMAGE}; rerun with PULL_IMAGE=1" >&2
  exit 3
fi

NPU_SMI_PATH="$(command -v npu-smi)"
mount_args=(
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v "${NPU_SMI_PATH}:${NPU_SMI_PATH}:ro"
)
[[ -d /usr/local/Ascend/firmware ]] && mount_args+=(
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro
)
[[ -d /usr/local/dcmi ]] && mount_args+=(
  -v /usr/local/dcmi:/usr/local/dcmi:ro
)

docker run --rm \
  --network host \
  --shm-size="${SHM_SIZE}" \
  --privileged=true \
  --user root \
  --entrypoint /bin/bash \
  -e ASCEND_RUNTIME_OPTIONS=NODRV \
  -e "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}" \
  -e "TP_SIZE=${TP_SIZE}" \
  -e "MAX_LENGTH=${MAX_LENGTH}" \
  -e "MASTER_PORT=${MASTER_PORT}" \
  -e "INPUT_TEXT=${INPUT_TEXT}" \
  "${mount_args[@]}" \
  -v "${HOST_MODEL_PATH}:/models/w8a8sc:ro" \
  "${IMAGE}" -lc '
    set -euo pipefail
    source_if_present() {
      local path="$1"
      if [[ -f "${path}" ]]; then
        set +u
        source "${path}"
        set -u
      fi
    }
    source_if_present /usr/local/Ascend/ascend-toolkit/set_env.sh
    source_if_present /usr/local/Ascend/cann/set_env.sh
    source_if_present /usr/local/Ascend/nnal/atb/set_env.sh
    source_if_present /usr/local/Ascend/atb-models/set_env.sh
    source_if_present /usr/local/Ascend/mindie/latest/mindie-llm/set_env.sh

    atb_root="${ATB_SPEED_HOME_PATH:-}"
    if [[ -z "${atb_root}" ]]; then
      for candidate in \
        /usr/local/Ascend/atb-models \
        /usr/local/Ascend/mindie/latest/mindie-llm/atb-models \
        /usr/local/Ascend/mindie/latest/atb-models; do
        if [[ -f "${candidate}/examples/run_pa.py" ]]; then
          atb_root="${candidate}"
          break
        fi
      done
    fi
    if [[ -z "${atb_root}" || ! -f "${atb_root}/examples/run_pa.py" ]]; then
      echo "[missing] examples.run_pa in the image" >&2
      exit 4
    fi
    export MINDIE_LOG_TO_STDOUT=1
    cd "${atb_root}"
    python3 -m torch.distributed.run \
      --nproc_per_node "${TP_SIZE}" \
      --master_port "${MASTER_PORT}" \
      -m examples.run_pa \
      --model_path /models/w8a8sc \
      --input_texts "${INPUT_TEXT}" \
      --max_input_length "${MAX_LENGTH}" \
      --max_prefill_tokens "${MAX_LENGTH}" \
      --max_output_length 1 \
      --max_batch_size 1
  '
