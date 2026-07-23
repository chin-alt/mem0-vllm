#!/usr/bin/env bash
set -euo pipefail

BETTER_EVAL_DIR="${BETTER_EVAL_DIR:-outputs/locomo_mem_4blora_vllm_dp}"
WORSE_EVAL_DIR="${WORSE_EVAL_DIR:-outputs/locomo_qwen_4blora_vlllm_dp}"
BETTER_NAME="${BETTER_NAME:-mem_reranker}"
WORSE_NAME="${WORSE_NAME:-qwen_soft_label}"
TEST_FILE="${TEST_FILE:-data/locomo/locomo_qwen3_embedding_candidates.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${BETTER_EVAL_DIR}/compare_${WORSE_NAME}}"
RELEVANCE_THRESHOLD="${RELEVANCE_THRESHOLD:-0.7}"
NDCG_K="${NDCG_K:-10}"
TOP_K_DOCS="${TOP_K_DOCS:-5}"
MIN_DELTA="${MIN_DELTA:-0.2}"
WORSE_MAX_NDCG="${WORSE_MAX_NDCG:-0.5}"
REQUIRE_BETTER_POSITIVE_RANK="${REQUIRE_BETTER_POSITIVE_RANK:-0}"
REPORT_TOP_N="${REPORT_TOP_N:-50}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXTRA_ARGS=()
if [[ "${REQUIRE_BETTER_POSITIVE_RANK}" == "1" ]]; then
  EXTRA_ARGS+=(--require_better_positive_rank)
fi
if [[ -n "${BETTER_PREDICTIONS_FILE:-}" ]]; then
  EXTRA_ARGS+=(--better_predictions_file "${BETTER_PREDICTIONS_FILE}")
fi
if [[ -n "${WORSE_PREDICTIONS_FILE:-}" ]]; then
  EXTRA_ARGS+=(--worse_predictions_file "${WORSE_PREDICTIONS_FILE}")
fi

"${PYTHON_BIN}" src/compare_locomo_runs.py \
  --better_eval_dir "${BETTER_EVAL_DIR}" \
  --worse_eval_dir "${WORSE_EVAL_DIR}" \
  --better_name "${BETTER_NAME}" \
  --worse_name "${WORSE_NAME}" \
  --test_file "${TEST_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --relevance_threshold "${RELEVANCE_THRESHOLD}" \
  --ndcg_k "${NDCG_K}" \
  --top_k_docs "${TOP_K_DOCS}" \
  --min_delta "${MIN_DELTA}" \
  --worse_max_ndcg "${WORSE_MAX_NDCG}" \
  --report_top_n "${REPORT_TOP_N}" \
  "${EXTRA_ARGS[@]}"
