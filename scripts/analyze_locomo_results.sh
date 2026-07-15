#!/usr/bin/env bash
set -euo pipefail

EVAL_DIR="${EVAL_DIR:-outputs/locomo_memreranker_4b_vllm_dp}"
PREDICTIONS_FILE="${PREDICTIONS_FILE:-}"
TEST_FILE="${TEST_FILE:-data/locomo/locomo_qwen3_embedding_candidates.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RELEVANCE_THRESHOLD="${RELEVANCE_THRESHOLD:-0.7}"
TOP_K="${TOP_K:-10}"
BAD_NDCG_K="${BAD_NDCG_K:-10}"
BAD_NDCG_THRESHOLD="${BAD_NDCG_THRESHOLD:-0.5}"
BAD_RANK_THRESHOLD="${BAD_RANK_THRESHOLD:-10}"
DOC_SNIPPET_CHARS="${DOC_SNIPPET_CHARS:-320}"
CATEGORY_MAP_FILE="${CATEGORY_MAP_FILE:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXTRA_ARGS=()
if [[ -n "${PREDICTIONS_FILE}" ]]; then
  EXTRA_ARGS+=(--predictions_file "${PREDICTIONS_FILE}")
else
  EXTRA_ARGS+=(--eval_dir "${EVAL_DIR}")
fi
if [[ -n "${TEST_FILE}" ]]; then
  EXTRA_ARGS+=(--test_file "${TEST_FILE}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  EXTRA_ARGS+=(--output_dir "${OUTPUT_DIR}")
fi
if [[ -n "${CATEGORY_MAP_FILE}" ]]; then
  EXTRA_ARGS+=(--category_map_file "${CATEGORY_MAP_FILE}")
fi

"${PYTHON_BIN}" src/analyze_locomo_results.py \
  "${EXTRA_ARGS[@]}" \
  --relevance_threshold "${RELEVANCE_THRESHOLD}" \
  --top_k "${TOP_K}" \
  --bad_ndcg_k "${BAD_NDCG_K}" \
  --bad_ndcg_threshold "${BAD_NDCG_THRESHOLD}" \
  --bad_rank_threshold "${BAD_RANK_THRESHOLD}" \
  --doc_snippet_chars "${DOC_SNIPPET_CHARS}"
