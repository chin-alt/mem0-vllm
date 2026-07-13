#!/usr/bin/env bash
set -euo pipefail

INPUT_FILE="${INPUT_FILE:-data/cmteb_r/cmteb_r_qwen3_embedding_candidates.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-data/cmteb_r/cmteb_r_qwen3_embedding_candidates_10pct_maxlen2048.jsonl}"
SAMPLE_RATIO="${SAMPLE_RATIO:-0.10}"
SEED="${SEED:-42}"
MAX_PAIR_CHARS="${MAX_PAIR_CHARS:-${MAX_LENGTH:-2048}}"
MAX_DOC_CHARS="${MAX_DOC_CHARS:-0}"
MAX_QUERY_CHARS="${MAX_QUERY_CHARS:-0}"
DROP_IF_PAIR_CHARS_GT="${DROP_IF_PAIR_CHARS_GT:-0}"
MAX_DOCS_PER_QUERY="${MAX_DOCS_PER_QUERY:-0}"
KEEP_RELEVANT_WHEN_CAPPING="${KEEP_RELEVANT_WHEN_CAPPING:-0}"
ALLOW_NO_RELEVANT_GROUPS="${ALLOW_NO_RELEVANT_GROUPS:-0}"
RELEVANCE_THRESHOLD="${RELEVANCE_THRESHOLD:-7.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXTRA_ARGS=()
if [[ "${KEEP_RELEVANT_WHEN_CAPPING}" == "1" ]]; then
  EXTRA_ARGS+=(--keep_relevant_when_capping)
fi
if [[ "${ALLOW_NO_RELEVANT_GROUPS}" == "1" ]]; then
  EXTRA_ARGS+=(--allow_no_relevant_groups)
fi

"${PYTHON_BIN}" src/prepare_eval_subset.py \
  --input_file "${INPUT_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --sample_ratio "${SAMPLE_RATIO}" \
  --seed "${SEED}" \
  --max_pair_chars "${MAX_PAIR_CHARS}" \
  --max_doc_chars "${MAX_DOC_CHARS}" \
  --max_query_chars "${MAX_QUERY_CHARS}" \
  --drop_if_pair_chars_gt "${DROP_IF_PAIR_CHARS_GT}" \
  --max_docs_per_query "${MAX_DOCS_PER_QUERY}" \
  --relevance_threshold "${RELEVANCE_THRESHOLD}" \
  "${EXTRA_ARGS[@]}"
