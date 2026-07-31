#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the already-exported Qwen3-Reranker-0.6B static-W8A8 model with
vLLM-Ascend on one Ascend 310P.

This is the host-side entry point to use after a machine restart. It:
  * starts Docker with systemd when requested and necessary;
  * verifies npu-smi, the local image, data, and static-W8A8 model;
  * reads the exact reranker instruction from the calibration/training JSONL;
  * applies the pinned vLLM-Ascend 0.10.0rc1 310P patches in the container;
  * runs a pooling evaluation and writes latency/quality metrics.

Environment overrides:
  HOST_MODEL_PATH          Static-W8A8 model directory.
                           Default: /home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8-static-noanti
  HOST_DATA_PATH           Business evaluation data root.
                           Default: /home/reranker_experiment/data/latency_delay
  TRAIN_JSONL              JSONL used to recover the production instruction.
                           Default: /home/reranker_experiment/data/split/train.jsonl
  HOST_OUTPUT_PATH         Output directory. Default includes a timestamp.
  INSTRUCTION              Explicit instruction; skips reading TRAIN_JSONL.
  DATASET                  all, 0428caption, 0428keyword, or 0625caption.
                           Default: 0428caption
  BATCH_SIZE               Default: 1
  MAX_LENGTH               Default: 1024
  MAX_NUM_SEQS             Default: BATCH_SIZE
  MAX_NUM_BATCHED_TOKENS   Default: MAX_LENGTH * BATCH_SIZE
  WARMUP_PAIRS             Default: BATCH_SIZE
  DEVICE_INDEX             Default: 0
  GPU_MEMORY_UTILIZATION   KV-cache memory budget. Default: 0.20
  ENABLE_PREFIX_CACHING    Default: 0
  IMAGE                    Default: quay.nju.edu.cn/ascend/vllm-ascend:v0.10.0rc1-310p
  PULL_IMAGE               Pull IMAGE before running. Default: 0
  START_DOCKER             Try systemctl start docker if needed. Default: 1
  INSTALL_EVAL_DEPS        Install openpyxl in the temporary container. Default: 1
  PIP_INDEX_URL            Default: https://mirror.nju.edu.cn/pypi/web/simple/
  DRY_RUN                  Validate and print configuration without Docker/NPU execution. Default: 0

Examples:
  bash scripts/run_qwen3_reranker_w8a8_310p_inference.sh

  BATCH_SIZE=8 MAX_NUM_BATCHED_TOKENS=8192 \
    bash scripts/run_qwen3_reranker_w8a8_310p_inference.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "[invalid] this script accepts environment variables, not positional arguments" >&2
  usage >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

HOST_REPO_PATH="${HOST_REPO_PATH:-${REPO_ROOT}}"
HOST_MODEL_PATH="${HOST_MODEL_PATH:-/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8-static-noanti}"
HOST_DATA_PATH="${HOST_DATA_PATH:-/home/reranker_experiment/data/latency_delay}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/reranker_experiment/data/split/train.jsonl}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_OUTPUT_PATH="${HOST_OUTPUT_PATH:-/home/reranker_experiment/output/qwen3_w8a8_pooling_${RUN_TAG}}"
DATASET="${DATASET:-0428caption}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${BATCH_SIZE}}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((MAX_LENGTH * BATCH_SIZE))}"
WARMUP_PAIRS="${WARMUP_PAIRS:-${BATCH_SIZE}}"
DEVICE_INDEX="${DEVICE_INDEX:-0}"
# Keep ample first-run headroom even though the pinned stable plugin performs
# incremental K/V cache format conversion.
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.20}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
IMAGE="${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:v0.10.0rc1-310p}"
PULL_IMAGE="${PULL_IMAGE:-0}"
START_DOCKER="${START_DOCKER:-1}"
INSTALL_EVAL_DEPS="${INSTALL_EVAL_DEPS:-1}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirror.nju.edu.cn/pypi/web/simple/}"
DRY_RUN="${DRY_RUN:-0}"

for flag in PULL_IMAGE START_DOCKER INSTALL_EVAL_DEPS ENABLE_PREFIX_CACHING DRY_RUN; do
  value="${!flag}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "[invalid] ${flag} must be 0 or 1; got: ${value}" >&2
    exit 2
  fi
done
for number in BATCH_SIZE MAX_LENGTH MAX_NUM_SEQS MAX_NUM_BATCHED_TOKENS WARMUP_PAIRS; do
  value="${!number}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
    echo "[invalid] ${number} must be a non-negative integer; got: ${value}" >&2
    exit 2
  fi
done
if ((BATCH_SIZE < 1 || MAX_LENGTH < 1 || MAX_NUM_SEQS < 1 || MAX_NUM_BATCHED_TOKENS < 1)); then
  echo "[invalid] batch, length, sequence, and token limits must be positive" >&2
  exit 2
fi
if ((MAX_NUM_BATCHED_TOKENS < MAX_LENGTH)); then
  echo "[invalid] MAX_NUM_BATCHED_TOKENS must be at least MAX_LENGTH" >&2
  exit 2
fi

for path in "${HOST_REPO_PATH}" "${HOST_DATA_PATH}" "${HOST_MODEL_PATH}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[missing] ${path}" >&2
    exit 3
  fi
done
for file in \
  "${HOST_MODEL_PATH}/config.json" \
  "${HOST_MODEL_PATH}/quant_model_description.json"; do
  if [[ ! -f "${file}" ]]; then
    echo "[missing] static-W8A8 model file: ${file}" >&2
    exit 3
  fi
done
if ! compgen -G "${HOST_MODEL_PATH}/*.safetensors" >/dev/null; then
  echo "[missing] no safetensors weights under ${HOST_MODEL_PATH}" >&2
  exit 3
fi

if [[ -z "${INSTRUCTION:-}" ]]; then
  if [[ ! -f "${TRAIN_JSONL}" ]]; then
    echo "[missing] TRAIN_JSONL=${TRAIN_JSONL}; set INSTRUCTION explicitly to bypass it" >&2
    exit 3
  fi
  if command -v python3 >/dev/null 2>&1; then
    HOST_PYTHON=python3
  elif command -v python >/dev/null 2>&1; then
    HOST_PYTHON=python
  else
    echo "[missing] python3/python is required to read TRAIN_JSONL" >&2
    exit 3
  fi
  INSTRUCTION="$("${HOST_PYTHON}" - "${TRAIN_JSONL}" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        row = json.loads(line)
        instruction = row.get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise SystemExit(
                f"[invalid] {path}:{line_number} has no non-empty instruction"
            )
        print(instruction)
        break
    else:
        raise SystemExit(f"[invalid] {path} has no JSONL records")
PY
)"
fi
if [[ -z "${INSTRUCTION}" ]]; then
  echo "[invalid] reranker instruction is empty" >&2
  exit 3
fi

echo "[config] repo=${HOST_REPO_PATH}"
echo "[config] model=${HOST_MODEL_PATH}"
echo "[config] data=${HOST_DATA_PATH}/${DATASET}"
echo "[config] output=${HOST_OUTPUT_PATH}"
echo "[config] image=${IMAGE}"
echo "[config] device=${DEVICE_INDEX} batch=${BATCH_SIZE} max_length=${MAX_LENGTH}"
echo "[config] max_num_seqs=${MAX_NUM_SEQS} max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS}"
echo "[config] gpu_memory_utilization=${GPU_MEMORY_UTILIZATION} (310P KV-format-safe default: 0.20)"
echo "[config] instruction_chars=${#INSTRUCTION}"

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] validation complete; Docker/NPU execution skipped"
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[missing] docker is not installed or not on PATH" >&2
  exit 3
fi
if ! docker info >/dev/null 2>&1; then
  if [[ "${START_DOCKER}" == "1" ]] && command -v systemctl >/dev/null 2>&1; then
    echo "[docker] starting Docker service"
    systemctl start docker
  fi
fi
if ! docker info >/dev/null 2>&1; then
  echo "[missing] Docker daemon is unavailable; start it or set START_DOCKER=1" >&2
  exit 3
fi
if ! command -v npu-smi >/dev/null 2>&1; then
  echo "[missing] npu-smi is not installed or not on PATH" >&2
  exit 3
fi
if ! npu-smi info >/dev/null; then
  echo "[error] npu-smi cannot access the host NPU/driver" >&2
  exit 3
fi
if [[ "${PULL_IMAGE}" == "0" ]] && ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[missing] local image ${IMAGE}" >&2
  echo "[missing] rerun with PULL_IMAGE=1 to pull it from the NJU registry" >&2
  exit 3
fi

mkdir -p "${HOST_OUTPUT_PATH}"

HOST_REPO_PATH="${HOST_REPO_PATH}" \
HOST_DATA_PATH="${HOST_DATA_PATH}" \
HOST_MODEL_PATH="${HOST_MODEL_PATH}" \
HOST_OUTPUT_PATH="${HOST_OUTPUT_PATH}" \
DATASET="${DATASET}" \
SCORING_BACKEND=pooling \
INSTRUCTION="${INSTRUCTION}" \
VLLM_QUANTIZATION=ascend \
DTYPE=float16 \
MAX_LENGTH="${MAX_LENGTH}" \
BATCH_SIZE="${BATCH_SIZE}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS}" \
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS}" \
WARMUP_PAIRS="${WARMUP_PAIRS}" \
DEVICE_INDEX="${DEVICE_INDEX}" \
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING}" \
IMAGE="${IMAGE}" \
PULL_IMAGE="${PULL_IMAGE}" \
INSTALL_EVAL_DEPS="${INSTALL_EVAL_DEPS}" \
PIP_INDEX_URL="${PIP_INDEX_URL}" \
bash "${SCRIPT_DIR}/run_qwen3_reranker_vllm_310p_container.sh"

echo "[done] inference output=${HOST_OUTPUT_PATH}"
find "${HOST_OUTPUT_PATH}" -name metrics.json -type f -print
