#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Evaluate one model with a local MindIE service on Ascend 310P/Atlas 300I Duo.

The endpoint defaults to 127.0.0.1. No external model API is used; MindIE runs
the model on the local NPU and returns one-token yes/no log probabilities.

Required/important environment variables:
  MINDIE_MODEL_NAME        modelName in MindIE config.json. Default: qwen3-reranker-4b
  MODEL_PATH               Optional model path recorded in metrics.
  MINDIE_ENDPOINT          Default: http://127.0.0.1:1025/v1/completions
  DATA_ROOT                Default: data/latency_delay
  DATASET_NAME             Optional: 0428caption, 0428keyword, or 0625caption
  OUTPUT_ROOT              Default: outputs/business_matrix_ascend_mindie_<timestamp>
  BATCH_SIZE               Concurrent loopback requests. Default: 32
  MAX_REQUEST_CHARS        Per-prompt character limit. Default: 32000
  MAX_LENGTH               MindIE max input token setting, recorded in metrics. Default: 8192
  TOP_LOGPROBS             Default: 5 (MindIE 2.1 compatible maximum)
  REQUEST_MODE             concurrent (default) or list
  REQUEST_TIMEOUT          Seconds per request. Default: 600
  REQUEST_RETRIES          Default: 2
  EXTRA_REQUEST_JSON       Optional JSON merged into every request.
  PYTHON_BIN               Host Python 3.9+ executable. Default: python
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/business_matrix_ascend_mindie_${RUN_TAG}}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/latency_delay}"
MINDIE_MODEL_NAME="${MINDIE_MODEL_NAME:-qwen3-reranker-4b}"
MODEL_PATH="${MODEL_PATH:-}"
MINDIE_ENDPOINT="${MINDIE_ENDPOINT:-http://127.0.0.1:1025/v1/completions}"
MAX_LENGTH="${MAX_LENGTH:-8192}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_REQUEST_CHARS="${MAX_REQUEST_CHARS:-32000}"
TOP_LOGPROBS="${TOP_LOGPROBS:-5}"
REQUEST_MODE="${REQUEST_MODE:-concurrent}"
MISSING_LOGPROB_FLOOR="${MISSING_LOGPROB_FLOOR:--20}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-600}"
REQUEST_RETRIES="${REQUEST_RETRIES:-2}"
EXPECTED_FBETA_BETA="${EXPECTED_FBETA_BETA:-0.3}"
INSTRUCTION="${INSTRUCTION:-Given a user query, retrieve relevant documents that answer the query.}"
GT_QUERY_COL="${GT_QUERY_COL:-query}"
TOP_K_LIST="${TOP_K_LIST:-1 3 5 10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
SKIP_MISSING="${SKIP_MISSING:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if sys.version_info[:2] >= (3, 9) else 1)'; then
  echo "[error] business_eval_mindie.py requires Python 3.9 or newer." >&2
  exit 2
fi

DATASET_NAMES=("0428caption" "0428keyword" "0625caption")
DATASET_DIRS=(
  "${DATA_ROOT}/0428caption"
  "${DATA_ROOT}/0428keyword"
  "${DATA_ROOT}/0625caption"
)
RECALL_FILES=(
  "${DATA_ROOT}/0428caption/retrieve_id_caption_0416.json"
  "${DATA_ROOT}/0428keyword/id_keywords_pair_new.json"
  "${DATA_ROOT}/0625caption/0625_raw_recall_result.json"
)
GT_DOC_ID_COLS=("PageId_new" "PageId_new" "PageId")
GT_FILE_HINTS=("" "" "gtfile-20260617.xlsx")
read -r -a TOP_K_ARGS <<< "${TOP_K_LIST}"
mkdir -p "${OUTPUT_ROOT}"

find_gt_file() {
  local dataset_dir="$1"
  local hint="$2"
  if [[ -n "${hint}" && -f "${dataset_dir}/${hint}" ]]; then
    printf '%s\n' "${dataset_dir}/${hint}"
    return 0
  fi
  find "${dataset_dir}" -maxdepth 1 -type f -name "*.xlsx" | sort | sed -n '1p'
}

copy_named_outputs() {
  local run_dir="$1"
  local run_name="$2"
  [[ -f "${run_dir}/metrics.json" ]] && cp "${run_dir}/metrics.json" "${OUTPUT_ROOT}/${run_name}_metrics.json"
  [[ -f "${run_dir}/business_eval.csv" ]] && cp "${run_dir}/business_eval.csv" "${OUTPUT_ROOT}/${run_name}_business_eval.csv"
  [[ -f "${run_dir}/business_eval.xlsx" ]] && cp "${run_dir}/business_eval.xlsx" "${OUTPUT_ROOT}/${run_name}_business_eval.xlsx"
}

run_one() {
  local dataset_name="$1"
  local dataset_dir="$2"
  local recall_file="$3"
  local gt_doc_id_col="$4"
  local gt_file_hint="$5"
  local run_name="${dataset_name}__${MINDIE_MODEL_NAME}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local gt_file=""

  if [[ "${SKIP_EXISTING}" == "1" && -f "${run_dir}/metrics.json" ]]; then
    echo "[skip] ${run_name}: metrics.json already exists"
    copy_named_outputs "${run_dir}" "${run_name}"
    return 0
  fi
  if [[ ! -d "${dataset_dir}" || ! -f "${recall_file}" ]]; then
    echo "[missing] dataset or recall file: ${dataset_dir} / ${recall_file}" >&2
    [[ "${SKIP_MISSING}" == "1" ]] && return 0
    return 3
  fi
  gt_file="$(find_gt_file "${dataset_dir}" "${gt_file_hint}")"
  if [[ -z "${gt_file}" || ! -f "${gt_file}" ]]; then
    echo "[missing] no ground-truth xlsx under ${dataset_dir}" >&2
    [[ "${SKIP_MISSING}" == "1" ]] && return 0
    return 3
  fi

  echo "[run:mindie] dataset=${dataset_name} model=${MINDIE_MODEL_NAME} output=${run_dir}"
  local optional_args=()
  [[ -n "${MODEL_PATH}" ]] && optional_args+=(--model_path "${MODEL_PATH}")
  [[ -n "${EXTRA_REQUEST_JSON:-}" ]] && optional_args+=(--extra_request_json "${EXTRA_REQUEST_JSON}")

  "${PYTHON_BIN}" business_eval_mindie.py \
    --gt_file "${gt_file}" \
    --recall_file "${recall_file}" \
    --model_name "${MINDIE_MODEL_NAME}" \
    --output_dir "${run_dir}" \
    --endpoint "${MINDIE_ENDPOINT}" \
    --instruction "${INSTRUCTION}" \
    --gt_query_col "${GT_QUERY_COL}" \
    --gt_doc_id_col "${gt_doc_id_col}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --max_request_chars "${MAX_REQUEST_CHARS}" \
    --top_logprobs "${TOP_LOGPROBS}" \
    --request_mode "${REQUEST_MODE}" \
    --missing_logprob_floor "${MISSING_LOGPROB_FLOOR}" \
    --request_timeout "${REQUEST_TIMEOUT}" \
    --request_retries "${REQUEST_RETRIES}" \
    --expected_fbeta_beta "${EXPECTED_FBETA_BETA}" \
    --top_k_list "${TOP_K_ARGS[@]}" \
    "${optional_args[@]}"
  copy_named_outputs "${run_dir}" "${run_name}"
}

for dataset_idx in "${!DATASET_NAMES[@]}"; do
  if [[ -n "${DATASET_NAME:-}" && "${DATASET_NAMES[$dataset_idx]}" != "${DATASET_NAME}" ]]; then
    continue
  fi
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    run_one \
      "${DATASET_NAMES[$dataset_idx]}" "${DATASET_DIRS[$dataset_idx]}" \
      "${RECALL_FILES[$dataset_idx]}" "${GT_DOC_ID_COLS[$dataset_idx]}" \
      "${GT_FILE_HINTS[$dataset_idx]}" || true
  else
    run_one \
      "${DATASET_NAMES[$dataset_idx]}" "${DATASET_DIRS[$dataset_idx]}" \
      "${RECALL_FILES[$dataset_idx]}" "${GT_DOC_ID_COLS[$dataset_idx]}" \
      "${GT_FILE_HINTS[$dataset_idx]}"
  fi
done

metrics_count="$(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -name metrics.json | wc -l | tr -d '[:space:]')"
if [[ "${metrics_count}" == "0" ]]; then
  echo "[error] no metrics.json files were produced under ${OUTPUT_ROOT}" >&2
  exit 3
fi
"${PYTHON_BIN}" src/summarize_business_matrix.py \
  --output_root "${OUTPUT_ROOT}" \
  --summary_csv "${OUTPUT_ROOT}/summary_metrics.csv" \
  --summary_json "${OUTPUT_ROOT}/summary_metrics.json" \
  --summary_xlsx "${OUTPUT_ROOT}/summary_metrics.xlsx"
echo "[done] MindIE matrix outputs: ${OUTPUT_ROOT}"
