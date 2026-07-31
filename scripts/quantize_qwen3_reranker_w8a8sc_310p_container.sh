#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
One-command W8A8S -> W8A8SC export and ATB smoke test on Ascend 310P.

Defaults match the reranker experiment layout:
  TRAIN_JSONL=/home/reranker_experiment/data/split/train.jsonl
  FLOAT_MODEL_PATH=/home/reranker_experiment/model/qwen3_reranker_06b_lora_merged
  W8A8S_MODEL_PATH=/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8S
  W8A8SC_MODEL_PATH=/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8SC

Important overrides:
  IMAGE                    MindIE 300I-Duo image. Default is 2.1.RC1.
  PULL_IMAGE               Pull IMAGE first. Default: 1
  ASCEND_RT_VISIBLE_DEVICES  Default: 0
  MAX_LENGTH               Default: 1024
  CALIB_SAMPLES            Default: 64
  CALIB_BACKEND            Default: generate
  TP_SIZE                  Default: 1
  MULTIPROCESS_NUM         Default: 4
  RUN_ATB_SMOKE            Default: 1
  REINSTALL_MODELSLIM      Default: 0
  PIP_INDEX_URL            Default: https://mirror.nju.edu.cn/pypi/web/simple/

The host driver is mounted read-only and never installed or changed. Both
output directories must be absent or empty.
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
IMAGE="${IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.1.RC1-300I-Duo-py311-openeuler24.03-lts}"
PULL_IMAGE="${PULL_IMAGE:-1}"
HOST_REPO_PATH="${HOST_REPO_PATH:-${REPO_ROOT}}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/reranker_experiment/data/split/train.jsonl}"
FLOAT_MODEL_PATH="${FLOAT_MODEL_PATH:-/home/reranker_experiment/model/qwen3_reranker_06b_lora_merged}"
W8A8S_MODEL_PATH="${W8A8S_MODEL_PATH:-/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8S}"
W8A8SC_MODEL_PATH="${W8A8SC_MODEL_PATH:-/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8SC}"
CALIB_BACKEND="${CALIB_BACKEND:-generate}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
CALIB_SAMPLES="${CALIB_SAMPLES:-64}"
CALIB_LENGTH_BINS="${CALIB_LENGTH_BINS:-4}"
CALIB_SEED="${CALIB_SEED:-20260731}"
CALIB_JSONL="${CALIB_JSONL:-/home/reranker_experiment/data/calibration/qwen3_reranker_w8a8s_${CALIB_BACKEND}_len${MAX_LENGTH}_n${CALIB_SAMPLES}.jsonl}"
HOST_MODELSLIM_CACHE="${HOST_MODELSLIM_CACHE:-/home/reranker_experiment/deps}"
HOST_VENV_CACHE="${HOST_VENV_CACHE:-/home/reranker_experiment/venvs-container-mindie21}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"
MULTIPROCESS_NUM="${MULTIPROCESS_NUM:-4}"
MASTER_PORT="${MASTER_PORT:-20037}"
RUN_ATB_SMOKE="${RUN_ATB_SMOKE:-1}"
REINSTALL_MODELSLIM="${REINSTALL_MODELSLIM:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirror.nju.edu.cn/pypi/web/simple/}"
PRODUCTION_INSTRUCTION="${PRODUCTION_INSTRUCTION:-}"
ALLOW_MULTIPLE_INSTRUCTIONS="${ALLOW_MULTIPLE_INSTRUCTIONS:-0}"
FRACTION="${FRACTION:-0.011}"
SIGMA_FACTOR="${SIGMA_FACTOR:-4.0}"
SHM_SIZE="${SHM_SIZE:-16g}"

for flag in PULL_IMAGE RUN_ATB_SMOKE REINSTALL_MODELSLIM ALLOW_MULTIPLE_INSTRUCTIONS; do
  value="${!flag}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "[invalid] ${flag} must be 0 or 1; got ${value}" >&2
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
for file in \
  "${TRAIN_JSONL}" \
  "${FLOAT_MODEL_PATH}/config.json" \
  "${HOST_REPO_PATH}/scripts/quantize_qwen3_reranker_w8a8sc_310p.sh"; do
  if [[ ! -f "${file}" ]]; then
    echo "[missing] ${file}" >&2
    exit 3
  fi
done
for output_dir in "${W8A8S_MODEL_PATH}" "${W8A8SC_MODEL_PATH}"; do
  existing=""
  if [[ -d "${output_dir}" ]]; then
    existing="$(find "${output_dir}" -mindepth 1 -print -quit)"
  fi
  if [[ -n "${existing}" ]]; then
    echo "[exists] output is not empty: ${output_dir}" >&2
    exit 3
  fi
done

mkdir -p \
  "${W8A8S_MODEL_PATH}" \
  "${W8A8SC_MODEL_PATH}" \
  "$(dirname "${CALIB_JSONL}")" \
  "${HOST_MODELSLIM_CACHE}" \
  "${HOST_VENV_CACHE}"

if [[ "${PULL_IMAGE}" == "1" ]]; then
  echo "[image] pulling ${IMAGE}"
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

calib_name="$(basename "${CALIB_JSONL}")"
docker_args=(
  run --rm
  --network host
  --shm-size="${SHM_SIZE}"
  --privileged=true
  --user root
  --entrypoint /bin/bash
  -e ASCEND_RUNTIME_OPTIONS=NODRV
  -e "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
  -e PYTORCH_NPU_ALLOC_CONF=expandable_segments:False
  -e TRAIN_JSONL=/inputs/train.jsonl
  -e FLOAT_MODEL_PATH=/models/float
  -e W8A8S_MODEL_PATH=/models/w8a8s
  -e W8A8SC_MODEL_PATH=/models/w8a8sc
  -e "CALIB_JSONL=/calibration/${calib_name}"
  -e MODELSLIM_DIR=/cache/deps/msit-modelslim-vllm-8.1
  -e MODELSLIM_VENV=/cache/venvs/modelslim-w8a8sc-mindie21
  -e "MAX_LENGTH=${MAX_LENGTH}"
  -e "MAX_PREFILL_TOKENS=${MAX_LENGTH}"
  -e "CALIB_SAMPLES=${CALIB_SAMPLES}"
  -e "CALIB_LENGTH_BINS=${CALIB_LENGTH_BINS}"
  -e "CALIB_SEED=${CALIB_SEED}"
  -e "CALIB_BACKEND=${CALIB_BACKEND}"
  -e "TP_SIZE=${TP_SIZE}"
  -e "MULTIPROCESS_NUM=${MULTIPROCESS_NUM}"
  -e "MASTER_PORT=${MASTER_PORT}"
  -e "RUN_ATB_SMOKE=${RUN_ATB_SMOKE}"
  -e "REINSTALL_MODELSLIM=${REINSTALL_MODELSLIM}"
  -e "PIP_INDEX_URL=${PIP_INDEX_URL}"
  -e "ALLOW_MULTIPLE_INSTRUCTIONS=${ALLOW_MULTIPLE_INSTRUCTIONS}"
  -e "FRACTION=${FRACTION}"
  -e "SIGMA_FACTOR=${SIGMA_FACTOR}"
  "${mount_args[@]}"
  -v "${HOST_REPO_PATH}:/workspace/memranker:ro"
  -v "${TRAIN_JSONL}:/inputs/train.jsonl:ro"
  -v "${FLOAT_MODEL_PATH}:/models/float:ro"
  -v "${W8A8S_MODEL_PATH}:/models/w8a8s:rw"
  -v "${W8A8SC_MODEL_PATH}:/models/w8a8sc:rw"
  -v "$(dirname "${CALIB_JSONL}"):/calibration:rw"
  -v "${HOST_MODELSLIM_CACHE}:/cache/deps:rw"
  -v "${HOST_VENV_CACHE}:/cache/venvs:rw"
  -w /workspace/memranker
)
if [[ -n "${PRODUCTION_INSTRUCTION}" ]]; then
  docker_args+=(-e "PRODUCTION_INSTRUCTION=${PRODUCTION_INSTRUCTION}")
fi

echo "[config] image=${IMAGE}"
echo "[config] float=${FLOAT_MODEL_PATH}"
echo "[config] w8a8s=${W8A8S_MODEL_PATH}"
echo "[config] w8a8sc=${W8A8SC_MODEL_PATH}"
echo "[config] device=${ASCEND_RT_VISIBLE_DEVICES} tp=${TP_SIZE}"
echo "[config] pip=${PIP_INDEX_URL}"

docker "${docker_args[@]}" "${IMAGE}" -lc \
  'exec bash scripts/quantize_qwen3_reranker_w8a8sc_310p.sh'

echo "[done] host W8A8S model: ${W8A8S_MODEL_PATH}"
echo "[done] host W8A8SC model: ${W8A8SC_MODEL_PATH}"
echo "[next] rerun pure ATB inference with scripts/run_qwen3_reranker_w8a8sc_atb_310p_container.sh"
