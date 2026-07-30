#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the complete Qwen3-Reranker-0.6B static-W8A8 CPU export inside a
vLLM-Ascend image that contains the legacy CANN ModelSlim overlay.

The host needs Docker, but does not need a CANN toolkit installation. This
script does not expose an NPU to the container and never changes the host
driver.

Important environment overrides:
  IMAGE                    Quantization image. Default:
                           quay.nju.edu.cn/ascend/vllm-ascend:v0.9.0rc2
  PULL_IMAGE               Pull IMAGE before running. Default: 1
  HOST_REPO_PATH           This repository. Default: detected automatically
  TRAIN_JSONL              Host training JSONL.
                           Default: /home/reranker_experiment/data/split/train.jsonl
  FLOAT_MODEL_PATH         Host merged float model directory.
                           Default: /home/reranker_experiment/model/qwen3_reranker_06b_lora_merged
  QUANT_MODEL_PATH         Host static-W8A8 output directory.
                           Default: /home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8-static-safe
  CALIB_JSONL              Host calibration output JSONL.
  HOST_MODELSLIM_CACHE     Persistent host checkout cache.
                           Default: /home/reranker_experiment/deps
  HOST_VENV_CACHE          Persistent, image-specific host venv cache.
                           Default: /home/reranker_experiment/venvs-container-v090
  MAX_LENGTH               Default: 1024
  CALIB_SAMPLES            Default: 64
  CALIB_BACKEND            pooling or generate. Default: pooling
  PIP_INDEX_URL            Default: https://pypi.tuna.tsinghua.edu.cn/simple
  REINSTALL_MODELSLIM      Force reinstall inside the container venv. Default: 0
  ANTI_METHOD              Anti-outlier method: m1 or none. Default: m1
  QUANTIZE_DOWN_PROJ       Quantize down_proj too. Default: 0
  PRODUCTION_INSTRUCTION   Optional instruction override.
  SHM_SIZE                 Docker shared-memory size. Default: 8g

The quantized output directory must be absent or empty. Existing results are
never deleted or overwritten.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:v0.9.0rc2}"
PULL_IMAGE="${PULL_IMAGE:-1}"
HOST_REPO_PATH="${HOST_REPO_PATH:-${REPO_ROOT}}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/reranker_experiment/data/split/train.jsonl}"
FLOAT_MODEL_PATH="${FLOAT_MODEL_PATH:-/home/reranker_experiment/model/qwen3_reranker_06b_lora_merged}"
QUANT_MODEL_PATH="${QUANT_MODEL_PATH:-/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8-static-safe}"
CALIB_BACKEND="${CALIB_BACKEND:-pooling}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
CALIB_SAMPLES="${CALIB_SAMPLES:-64}"
CALIB_LENGTH_BINS="${CALIB_LENGTH_BINS:-4}"
CALIB_SEED="${CALIB_SEED:-20260730}"
CALIB_JSONL="${CALIB_JSONL:-/home/reranker_experiment/data/calibration/qwen3_reranker_w8a8_static_${CALIB_BACKEND}_len${MAX_LENGTH}_n${CALIB_SAMPLES}.jsonl}"
HOST_MODELSLIM_CACHE="${HOST_MODELSLIM_CACHE:-/home/reranker_experiment/deps}"
HOST_VENV_CACHE="${HOST_VENV_CACHE:-/home/reranker_experiment/venvs-container-v090}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
REINSTALL_MODELSLIM="${REINSTALL_MODELSLIM:-0}"
ANTI_METHOD="${ANTI_METHOD:-m1}"
QUANTIZE_DOWN_PROJ="${QUANTIZE_DOWN_PROJ:-0}"
ALLOW_MULTIPLE_INSTRUCTIONS="${ALLOW_MULTIPLE_INSTRUCTIONS:-0}"
PRODUCTION_INSTRUCTION="${PRODUCTION_INSTRUCTION:-}"
SHM_SIZE="${SHM_SIZE:-8g}"

for flag in PULL_IMAGE REINSTALL_MODELSLIM QUANTIZE_DOWN_PROJ ALLOW_MULTIPLE_INSTRUCTIONS; do
  value="${!flag}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "[invalid] ${flag} must be 0 or 1" >&2
    exit 2
  fi
done
if [[ "${CALIB_BACKEND}" != "pooling" && "${CALIB_BACKEND}" != "generate" ]]; then
  echo "[invalid] CALIB_BACKEND must be pooling or generate" >&2
  exit 2
fi
if [[ "${ANTI_METHOD}" != "m1" && "${ANTI_METHOD}" != "none" ]]; then
  echo "[invalid] ANTI_METHOD must be m1 or none" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[missing] docker is not available on the host" >&2
  exit 3
fi
if [[ ! -f "${HOST_REPO_PATH}/scripts/quantize_qwen3_reranker_w8a8_static_310p.sh" ]]; then
  echo "[missing] repository workflow under HOST_REPO_PATH=${HOST_REPO_PATH}" >&2
  exit 3
fi
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[missing] training JSONL: ${TRAIN_JSONL}" >&2
  exit 3
fi
if [[ ! -f "${FLOAT_MODEL_PATH}/config.json" ]]; then
  echo "[missing] float model config: ${FLOAT_MODEL_PATH}/config.json" >&2
  exit 3
fi

existing_quant_file=""
if [[ -d "${QUANT_MODEL_PATH}" ]]; then
  existing_quant_file="$(find "${QUANT_MODEL_PATH}" -mindepth 1 -print -quit)"
fi
if [[ -n "${existing_quant_file}" ]]; then
  echo "[exists] quantized output is not empty: ${QUANT_MODEL_PATH}" >&2
  echo "[exists] choose a new QUANT_MODEL_PATH; this script never deletes model output" >&2
  exit 3
fi

mkdir -p \
  "${QUANT_MODEL_PATH}" \
  "$(dirname "${CALIB_JSONL}")" \
  "${HOST_MODELSLIM_CACHE}" \
  "${HOST_VENV_CACHE}"

if [[ "${PULL_IMAGE}" == "1" ]]; then
  echo "[image] pulling ${IMAGE}"
  docker pull "${IMAGE}"
elif ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[missing] Docker image is not local: ${IMAGE}" >&2
  echo "[missing] rerun with PULL_IMAGE=1" >&2
  exit 3
fi

calib_name="$(basename "${CALIB_JSONL}")"
docker_args=(
  run --rm
  --network host
  --shm-size="${SHM_SIZE}"
  --user root
  --entrypoint /bin/bash
  -v "${HOST_REPO_PATH}:/workspace/mem0-vllm:ro"
  -v "${TRAIN_JSONL}:/inputs/train.jsonl:ro"
  -v "$(dirname "${CALIB_JSONL}"):/calibration:rw"
  -v "${FLOAT_MODEL_PATH}:/models/float:ro"
  -v "${QUANT_MODEL_PATH}:/models/quant:rw"
  -v "${HOST_MODELSLIM_CACHE}:/cache/deps:rw"
  -v "${HOST_VENV_CACHE}:/cache/venvs:rw"
  -w /workspace/mem0-vllm
  -e TRAIN_JSONL=/inputs/train.jsonl
  -e FLOAT_MODEL_PATH=/models/float
  -e QUANT_MODEL_PATH=/models/quant
  -e "CALIB_JSONL=/calibration/${calib_name}"
  -e MODELSLIM_DIR=/cache/deps/msit-modelslim-vllm-8.1
  -e MODELSLIM_VENV=/cache/venvs/modelslim-vllm-8.1
  -e "MAX_LENGTH=${MAX_LENGTH}"
  -e "CALIB_SAMPLES=${CALIB_SAMPLES}"
  -e "CALIB_LENGTH_BINS=${CALIB_LENGTH_BINS}"
  -e "CALIB_SEED=${CALIB_SEED}"
  -e "CALIB_BACKEND=${CALIB_BACKEND}"
  -e "PIP_INDEX_URL=${PIP_INDEX_URL}"
  -e "REINSTALL_MODELSLIM=${REINSTALL_MODELSLIM}"
  -e "ANTI_METHOD=${ANTI_METHOD}"
  -e "QUANTIZE_DOWN_PROJ=${QUANTIZE_DOWN_PROJ}"
  -e "ALLOW_MULTIPLE_INSTRUCTIONS=${ALLOW_MULTIPLE_INSTRUCTIONS}"
  -e INSTALL_MODELSLIM=1
  -e RUN_BENCHMARK=0
  -e PULL_IMAGE=0
  -e TORCH_DEVICE_BACKEND_AUTOLOAD=0
)
if [[ -n "${PRODUCTION_INSTRUCTION}" ]]; then
  docker_args+=(-e "PRODUCTION_INSTRUCTION=${PRODUCTION_INSTRUCTION}")
fi

echo "[image] validating legacy ModelSlim overlay"
echo "[quantization] output=${QUANT_MODEL_PATH}"
docker "${docker_args[@]}" "${IMAGE}" -lc '
  set -euo pipefail

  cann_set_env=""
  for candidate in \
    /usr/local/Ascend/ascend-toolkit/set_env.sh \
    /usr/local/Ascend/ascend-toolkit/latest/set_env.sh \
    /usr/local/Ascend/latest/set_env.sh; do
    if [[ -f "${candidate}" ]]; then
      cann_set_env="${candidate}"
      break
    fi
  done
  if [[ -z "${cann_set_env}" ]]; then
    echo "[missing] IMAGE has no CANN set_env.sh: ${IMAGE:-container image}" >&2
    exit 4
  fi

  set +u
  source "${cann_set_env}"
  set -u
  export CANN_SET_ENV="${cann_set_env}"
  anti_dir="${ASCEND_HOME_PATH:-}/python/site-packages/msmodelslim/pytorch/llm_ptq/anti_outlier"
  python_abi="$(python3 -c "import sys; print(\"cpython-%d%d\" % sys.version_info[:2])")"
  anti_utils="$(find "${anti_dir}" -maxdepth 1 -type f \
    -name "anti_utils.${python_abi}-*.so" -print -quit 2>/dev/null || true)"
  if [[ -z "${anti_utils}" ]]; then
    echo "[missing] IMAGE has no ${python_abi}-compatible anti_utils*.so under: ${anti_dir}" >&2
    echo "[missing] use an image with the full CANN 8.0/8.1 ModelSlim Python package" >&2
    echo "[diagnostic] compiled anti_utils candidates in IMAGE:" >&2
    find "${anti_dir}" -maxdepth 1 -type f -name "anti_utils*.so" -print 2>/dev/null || true
    exit 4
  fi

  echo "[cann] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
  echo "[cann] anti_utils=${anti_utils}"
  python3 --version
  exec bash scripts/quantize_qwen3_reranker_w8a8_static_310p.sh
'

echo "[done] host quantized model: ${QUANT_MODEL_PATH}"
echo "[done] host calibration data: ${CALIB_JSONL}"
