#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run Sentence-Transformers CrossEncoder business evaluation for mbert/modernbert
models on the 0428caption, 0428keyword, and 0625caption datasets.

Recommended usage:
  MODERNBERT_MODEL_PATH=outputs/modernbert_pointwise/best \
  MBERT_MODEL_PATH=outputs/mbert_pointwise/best \
  CUDA_VISIBLE_DEVICES=0 \
  bash scripts/eval_business_matrix_crossencoder.sh

Environment overrides:
  OUTPUT_ROOT              Output root. Default: outputs/business_matrix_crossencoder_<timestamp>
  CUDA_VISIBLE_DEVICES     GPU ids for evaluation. Default: 0
  MODERNBERT_MODEL_PATH    ModernBERT CrossEncoder checkpoint. Default: outputs/modernbert_pointwise/best
  MBERT_MODEL_PATH         mBERT CrossEncoder checkpoint. Default: outputs/mbert_pointwise/best
  MODEL_NAMES              Optional whitespace-separated model names.
  MODEL_PATHS              Optional |-separated model paths matching MODEL_NAMES.
  MAX_LENGTH               Max sequence length. Default: 2048
  BATCH_SIZE               CrossEncoder predict batch size. Default: 32
  EXPECTED_FBETA_BETA      Beta for dynamic Expected-Fbeta cutoff. Default: 0.3
  PRECISION                fp16, bf16, or fp32. Default: bf16
  ATTN_IMPLEMENTATION      transformers attention backend. Default: sdpa
  SCORE_ACTIVATION         sigmoid, identity, or default. Default: sigmoid
  LOCAL_FILES_ONLY         Force offline local model loading. Default: 1
  SKIP_EXISTING            Skip a run if metrics.json already exists. Default: 1
  CONTINUE_ON_ERROR        Continue remaining runs after one failure. Default: 1
  SKIP_MISSING             Skip missing model/data paths. Default: 1
  POST_RUN_SLEEP           Seconds to wait after each run. Default: 2
  PYTHON_BIN               Python executable. Default: python

Outputs:
  <OUTPUT_ROOT>/<dataset>__<model>/metrics.json
  <OUTPUT_ROOT>/<dataset>__<model>/business_eval.xlsx
  <OUTPUT_ROOT>/<dataset>__<model>/business_eval.csv
  <OUTPUT_ROOT>/<dataset>__<model>_metrics.json
  <OUTPUT_ROOT>/<dataset>__<model>_business_eval.xlsx
  <OUTPUT_ROOT>/<dataset>__<model>_business_eval.csv
  <OUTPUT_ROOT>/summary_metrics.xlsx
  <OUTPUT_ROOT>/summary_metrics.csv
  <OUTPUT_ROOT>/summary_metrics.json

Notes:
  The script finds the 0428 Excel files by globbing *.xlsx inside each dataset
  directory, avoiding shell encoding issues with Chinese filenames.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/business_matrix_crossencoder_${RUN_TAG}}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-32}"
EXPECTED_FBETA_BETA="${EXPECTED_FBETA_BETA:-0.3}"
PRECISION="${PRECISION:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SCORE_ACTIVATION="${SCORE_ACTIVATION:-sigmoid}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
INSTRUCTION="${INSTRUCTION:-Given a user query, retrieve relevant documents that answer the query.}"
GT_QUERY_COL="${GT_QUERY_COL:-query}"
TOP_K_LIST="${TOP_K_LIST:-1 3 5 10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SKIP_MISSING="${SKIP_MISSING:-1}"
POST_RUN_SLEEP="${POST_RUN_SLEEP:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

DATA_ROOT="${DATA_ROOT:-/home/c50061497/MemOS/src/memos/reranker/memranker/data/latency_delay}"
MODERNBERT_MODEL_PATH="${MODERNBERT_MODEL_PATH:-/home/c50061497/MemOS/src/memos/reranker/memranker/outputs/modernbert_pointwise/best}"
MBERT_MODEL_PATH="${MBERT_MODEL_PATH:-/home/c50061497/MemOS/src/memos/reranker/memranker/outputs/mbert_pointwise/best}"

if [[ -n "${MODEL_NAMES:-}" || -n "${MODEL_PATHS:-}" ]]; then
  if [[ -z "${MODEL_NAMES:-}" || -z "${MODEL_PATHS:-}" ]]; then
    echo "MODEL_NAMES and MODEL_PATHS must be set together." >&2
    exit 2
  fi
  read -r -a MODEL_NAME_ARRAY <<< "${MODEL_NAMES}"
  IFS='|' read -r -a MODEL_PATH_ARRAY <<< "${MODEL_PATHS}"
else
  MODEL_NAME_ARRAY=("modernbert" "mbert")
  MODEL_PATH_ARRAY=("${MODERNBERT_MODEL_PATH}" "${MBERT_MODEL_PATH}")
fi

if [[ "${#MODEL_NAME_ARRAY[@]}" -ne "${#MODEL_PATH_ARRAY[@]}" ]]; then
  echo "MODEL_NAMES count (${#MODEL_NAME_ARRAY[@]}) does not match MODEL_PATHS count (${#MODEL_PATH_ARRAY[@]})." >&2
  exit 2
fi

DATASET_NAMES=(
  "0428caption"
  "0428keyword"
  "0625caption"
)
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
GT_DOC_ID_COLS=(
  "PageId_new"
  "PageId_new"
  "PageId"
)
GT_FILE_HINTS=(
  ""
  ""
  "gtfile-20260617.xlsx"
)

read -r -a TOP_K_ARGS <<< "${TOP_K_LIST}"
mkdir -p "${OUTPUT_ROOT}"

precision_args=()
if [[ "${PRECISION}" == "fp16" ]]; then
  precision_args+=(--fp16)
elif [[ "${PRECISION}" == "bf16" ]]; then
  precision_args+=(--bf16)
elif [[ "${PRECISION}" != "fp32" ]]; then
  echo "Unsupported PRECISION=${PRECISION}; use fp16, bf16, or fp32." >&2
  exit 2
fi

attn_args=()
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  attn_args+=(--attn_implementation "${ATTN_IMPLEMENTATION}")
fi

local_model_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  local_model_args+=(--local_files_only)
fi

find_gt_file() {
  local dataset_dir="$1"
  local hint="$2"
  if [[ -n "${hint}" && -f "${dataset_dir}/${hint}" ]]; then
    printf '%s\n' "${dataset_dir}/${hint}"
    return 0
  fi
  local found=()
  while IFS= read -r file; do
    found+=("${file}")
  done < <(find "${dataset_dir}" -maxdepth 1 -type f -name "*.xlsx" | sort)
  if [[ "${#found[@]}" -eq 0 ]]; then
    return 1
  fi
  printf '%s\n' "${found[0]}"
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
  local model_name="$6"
  local model_path="$7"
  local run_name="${dataset_name}__${model_name}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local gt_file=""

  if [[ "${SKIP_EXISTING}" == "1" && -f "${run_dir}/metrics.json" ]]; then
    echo "[skip] ${run_name}: ${run_dir}/metrics.json already exists"
    copy_named_outputs "${run_dir}" "${run_name}"
    return 0
  fi

  if [[ ! -e "${model_path}" ]]; then
    echo "[missing] model path does not exist: ${model_path}" >&2
    [[ "${SKIP_MISSING}" == "1" ]] && return 0
    return 3
  fi
  if [[ ! -d "${dataset_dir}" ]]; then
    echo "[missing] dataset dir does not exist: ${dataset_dir}" >&2
    [[ "${SKIP_MISSING}" == "1" ]] && return 0
    return 3
  fi
  if ! gt_file="$(find_gt_file "${dataset_dir}" "${gt_file_hint}")"; then
    echo "[missing] no .xlsx ground-truth file found under: ${dataset_dir}" >&2
    [[ "${SKIP_MISSING}" == "1" ]] && return 0
    return 3
  fi
  if [[ ! -f "${recall_file}" ]]; then
    echo "[missing] recall file does not exist: ${recall_file}" >&2
    [[ "${SKIP_MISSING}" == "1" ]] && return 0
    return 3
  fi

  echo "======================================================================"
  echo "[run:crossencoder] dataset=${dataset_name} model=${model_name}"
  echo "[run:crossencoder] gt=${gt_file}"
  echo "[run:crossencoder] recall=${recall_file}"
  echo "[run:crossencoder] output=${run_dir}"
  echo "======================================================================"

  "${PYTHON_BIN}" src/evaluate_business_crossencoder.py \
    --gt_file "${gt_file}" \
    --recall_file "${recall_file}" \
    --model_path "${model_path}" \
    --output_dir "${run_dir}" \
    --instruction "${INSTRUCTION}" \
    --gt_query_col "${GT_QUERY_COL}" \
    --gt_doc_id_col "${gt_doc_id_col}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --expected_fbeta_beta "${EXPECTED_FBETA_BETA}" \
    --score_activation "${SCORE_ACTIVATION}" \
    --top_k_list "${TOP_K_ARGS[@]}" \
    "${precision_args[@]}" \
    "${attn_args[@]}" \
    "${local_model_args[@]}"

  copy_named_outputs "${run_dir}" "${run_name}"
  if [[ "${POST_RUN_SLEEP}" != "0" ]]; then
    sleep "${POST_RUN_SLEEP}"
  fi
}

for dataset_idx in "${!DATASET_NAMES[@]}"; do
  for model_idx in "${!MODEL_NAME_ARRAY[@]}"; do
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      if ! run_one \
        "${DATASET_NAMES[$dataset_idx]}" \
        "${DATASET_DIRS[$dataset_idx]}" \
        "${RECALL_FILES[$dataset_idx]}" \
        "${GT_DOC_ID_COLS[$dataset_idx]}" \
        "${GT_FILE_HINTS[$dataset_idx]}" \
        "${MODEL_NAME_ARRAY[$model_idx]}" \
        "${MODEL_PATH_ARRAY[$model_idx]}"; then
        echo "[failed] ${DATASET_NAMES[$dataset_idx]}__${MODEL_NAME_ARRAY[$model_idx]}" >&2
      fi
    else
      run_one \
        "${DATASET_NAMES[$dataset_idx]}" \
        "${DATASET_DIRS[$dataset_idx]}" \
        "${RECALL_FILES[$dataset_idx]}" \
        "${GT_DOC_ID_COLS[$dataset_idx]}" \
        "${GT_FILE_HINTS[$dataset_idx]}" \
        "${MODEL_NAME_ARRAY[$model_idx]}" \
        "${MODEL_PATH_ARRAY[$model_idx]}"
    fi
  done
done

"${PYTHON_BIN}" src/summarize_business_matrix.py \
  --output_root "${OUTPUT_ROOT}" \
  --summary_csv "${OUTPUT_ROOT}/summary_metrics.csv" \
  --summary_json "${OUTPUT_ROOT}/summary_metrics.json" \
  --summary_xlsx "${OUTPUT_ROOT}/summary_metrics.xlsx"

echo "[done] CrossEncoder matrix outputs: ${OUTPUT_ROOT}"
