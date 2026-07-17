#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run LoCoMo candidate JSONL evaluation for modernbert/mbert CrossEncoder models,
then run per-category and bad-case analysis for each model.

Recommended:
  TEST_FILE=data/locomo/locomo_qwen3_embedding_candidates.jsonl \
  MODERNBERT_MODEL_PATH=outputs/modernbert_pointwise/best \
  MBERT_MODEL_PATH=outputs/mbert_pointwise/best \
  bash scripts/eval_locomo_crossencoder.sh

Environment is the same style as eval_cmteb_r_crossencoder.sh. Important knobs:
  TEST_FILE            Default: data/locomo/locomo_qwen3_embedding_candidates.jsonl
  OUTPUT_ROOT          Default: outputs/locomo_crossencoder_<timestamp>
  TOP_K                LoCoMo analysis top-k. Default: 10
  BAD_NDCG_THRESHOLD   Default: 0.5
  BAD_RANK_THRESHOLD   Default: 10
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
TEST_FILE="${TEST_FILE:-data/locomo/locomo_qwen3_embedding_candidates.jsonl}"
DATASET_NAME="${DATASET_NAME:-locomo}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/locomo_crossencoder_${RUN_TAG}}"
MODERNBERT_MODEL_PATH="${MODERNBERT_MODEL_PATH:-outputs/modernbert_pointwise/best}"
MBERT_MODEL_PATH="${MBERT_MODEL_PATH:-outputs/mbert_pointwise/best}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PRECISION="${PRECISION:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SCORE_ACTIVATION="${SCORE_ACTIVATION:-sigmoid}"
EXPECTED_FBETA_BETAS="${EXPECTED_FBETA_BETAS:-0.2 0.3 0.5 0.7 1.0}"
RELEVANCE_THRESHOLD="${RELEVANCE_THRESHOLD:-0.7}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SKIP_MISSING="${SKIP_MISSING:-1}"
POST_RUN_SLEEP="${POST_RUN_SLEEP:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TOP_K="${TOP_K:-10}"
BAD_NDCG_K="${BAD_NDCG_K:-10}"
BAD_NDCG_THRESHOLD="${BAD_NDCG_THRESHOLD:-0.5}"
BAD_RANK_THRESHOLD="${BAD_RANK_THRESHOLD:-10}"
DOC_SNIPPET_CHARS="${DOC_SNIPPET_CHARS:-320}"
CATEGORY_MAP_FILE="${CATEGORY_MAP_FILE:-}"

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
  echo "MODEL_NAMES count does not match MODEL_PATHS count." >&2
  exit 2
fi

mkdir -p "${OUTPUT_ROOT}"
read -r -a BETA_ARGS <<< "${EXPECTED_FBETA_BETAS}"

precision_args=()
if [[ "${PRECISION}" == "fp16" ]]; then
  precision_args+=(--fp16)
elif [[ "${PRECISION}" == "bf16" ]]; then
  precision_args+=(--bf16)
elif [[ "${PRECISION}" != "fp32" ]]; then
  echo "Unsupported PRECISION=${PRECISION}; use fp16, bf16, or fp32." >&2
  exit 2
fi

extra_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  extra_args+=(--local_files_only)
fi
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  extra_args+=(--attn_implementation "${ATTN_IMPLEMENTATION}")
fi

analysis_extra_args=()
if [[ -n "${CATEGORY_MAP_FILE}" ]]; then
  analysis_extra_args+=(--category_map_file "${CATEGORY_MAP_FILE}")
fi

run_one() {
  local model_name="$1"
  local model_path="$2"
  local run_name="${DATASET_NAME}__${model_name}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${run_dir}/overall_metrics.json" ]]; then
    echo "[skip] ${run_name}: ${run_dir}/overall_metrics.json already exists"
  else
    if [[ ! -f "${TEST_FILE}" ]]; then
      echo "[missing] test file does not exist: ${TEST_FILE}" >&2
      [[ "${SKIP_MISSING}" == "1" ]] && return 0
      return 3
    fi
    if [[ ! -e "${model_path}" ]]; then
      echo "[missing] model path does not exist: ${model_path}" >&2
      [[ "${SKIP_MISSING}" == "1" ]] && return 0
      return 3
    fi

    echo "======================================================================"
    echo "[run:locomo-crossencoder] model=${model_name}"
    echo "[run:locomo-crossencoder] test=${TEST_FILE}"
    echo "[run:locomo-crossencoder] output=${run_dir}"
    echo "======================================================================"

    "${PYTHON_BIN}" src/evaluate_jsonl_crossencoder.py \
      --test_file "${TEST_FILE}" \
      --model_path "${model_path}" \
      --output_dir "${run_dir}" \
      --max_length "${MAX_LENGTH}" \
      --batch_size "${BATCH_SIZE}" \
      --relevance_threshold "${RELEVANCE_THRESHOLD}" \
      --score_activation "${SCORE_ACTIVATION}" \
      --expected_fbeta_betas "${BETA_ARGS[@]}" \
      "${precision_args[@]}" \
      "${extra_args[@]}"
  fi

  if [[ -f "${run_dir}/predictions.jsonl" ]]; then
    "${PYTHON_BIN}" src/analyze_locomo_results.py \
      --eval_dir "${run_dir}" \
      --test_file "${TEST_FILE}" \
      --output_dir "${run_dir}/locomo_analysis" \
      --relevance_threshold "${RELEVANCE_THRESHOLD}" \
      --top_k "${TOP_K}" \
      --bad_ndcg_k "${BAD_NDCG_K}" \
      --bad_ndcg_threshold "${BAD_NDCG_THRESHOLD}" \
      --bad_rank_threshold "${BAD_RANK_THRESHOLD}" \
      --doc_snippet_chars "${DOC_SNIPPET_CHARS}" \
      "${analysis_extra_args[@]}"

    "${PYTHON_BIN}" src/diagnose_locomo_eval.py \
      --eval_dir "${run_dir}" \
      --test_file "${TEST_FILE}" \
      --output_file "${run_dir}/locomo_diagnostics.json" \
      --relevance_threshold "${RELEVANCE_THRESHOLD}" \
      --ndcg_k "${BAD_NDCG_K}"
  fi

  [[ "${POST_RUN_SLEEP}" != "0" ]] && sleep "${POST_RUN_SLEEP}"
}

for model_idx in "${!MODEL_NAME_ARRAY[@]}"; do
  if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
    if ! run_one "${MODEL_NAME_ARRAY[$model_idx]}" "${MODEL_PATH_ARRAY[$model_idx]}"; then
      echo "[failed] ${MODEL_NAME_ARRAY[$model_idx]}" >&2
    fi
  else
    run_one "${MODEL_NAME_ARRAY[$model_idx]}" "${MODEL_PATH_ARRAY[$model_idx]}"
  fi
done

"${PYTHON_BIN}" src/summarize_jsonl_matrix.py \
  --output_root "${OUTPUT_ROOT}" \
  --summary_csv "${OUTPUT_ROOT}/summary_metrics.csv" \
  --summary_json "${OUTPUT_ROOT}/summary_metrics.json" \
  --summary_xlsx "${OUTPUT_ROOT}/summary_metrics.xlsx"

echo "[done] LoCoMo CrossEncoder outputs: ${OUTPUT_ROOT}"
