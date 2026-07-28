#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/home/reranker_experiment/model/gte-multilingual-reranker-base}"
DATA_ROOT="${DATA_ROOT:-/home/reranker_experiment/data/latency_delay}"
DATASET="${DATASET:-0428caption}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/business_gte_310p_$(date +%Y%m%d_%H%M%S)}"
DEVICE="${DEVICE:-npu:0}"
DTYPE="${DTYPE:-fp16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

if [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
  set +u
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  set -u
fi

case "${DATASET}" in
  0428caption)
    DATASET_DIR="${DATA_ROOT}/0428caption"
    RECALL_FILE="${RECALL_FILE:-${DATASET_DIR}/retrieve_id_caption_0416.json}"
    GT_DOC_ID_COL="${GT_DOC_ID_COL:-PageId_new}"
    ;;
  0428keyword)
    DATASET_DIR="${DATA_ROOT}/0428keyword"
    RECALL_FILE="${RECALL_FILE:-${DATASET_DIR}/id_keywords_pair_new.json}"
    GT_DOC_ID_COL="${GT_DOC_ID_COL:-PageId_new}"
    ;;
  0625caption)
    DATASET_DIR="${DATA_ROOT}/0625caption"
    RECALL_FILE="${RECALL_FILE:-${DATASET_DIR}/0625_raw_recall_result.json}"
    GT_DOC_ID_COL="${GT_DOC_ID_COL:-PageId}"
    ;;
  *)
    echo "Unsupported DATASET=${DATASET}; use 0428caption, 0428keyword, or 0625caption." >&2
    exit 2
    ;;
esac

if [[ -z "${GT_FILE:-}" ]]; then
  if [[ "${DATASET}" == "0625caption" && -f "${DATASET_DIR}/gtfile-20260617.xlsx" ]]; then
    GT_FILE="${DATASET_DIR}/gtfile-20260617.xlsx"
  else
    GT_FILE="$(find "${DATASET_DIR}" -maxdepth 1 -type f -name '*.xlsx' | sort | head -1)"
  fi
fi

for required in "${MODEL_PATH}" "${GT_FILE}" "${RECALL_FILE}"; do
  if [[ ! -e "${required}" ]]; then
    echo "[missing] ${required}" >&2
    exit 3
  fi
done

RUN_DIR="${OUTPUT_ROOT}/${DATASET}__gte_multilingual_reranker_base"
echo "[gte] model=${MODEL_PATH}"
echo "[gte] dataset=${DATASET} device=${DEVICE} dtype=${DTYPE}"
echo "[gte] max_length=${MAX_LENGTH} batch_size=${BATCH_SIZE}"
echo "[gte] output=${RUN_DIR}"

"${PYTHON_BIN}" src/evaluate_business_gte_npu.py \
  --gt_file "${GT_FILE}" \
  --recall_file "${RECALL_FILE}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${RUN_DIR}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --gt_doc_id_col "${GT_DOC_ID_COL}" \
  --score_activation sigmoid \
  --local_files_only

echo "[done] ${RUN_DIR}/metrics.json"
