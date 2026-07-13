#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data/cmteb_r}"
QRELS_INPUT_DIR="${QRELS_INPUT_DIR:-${INPUT_DIR}}"
OUTPUT_FILE="${OUTPUT_FILE:-data/cmteb_r/cmteb_r_bm25_candidates.jsonl}"
DATASETS="${DATASETS:-T2Retrieval}"
CANDIDATE_TOP_K="${CANDIDATE_TOP_K:-100}"
MAX_QUERIES_PER_DATASET="${MAX_QUERIES_PER_DATASET:-1000}"
DOC_MAX_CHARS="${DOC_MAX_CHARS:-0}"
INDEX_DOC_MAX_CHARS="${INDEX_DOC_MAX_CHARS:-2048}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ENSURE_POSITIVES="${ENSURE_POSITIVES:-0}"
INSTRUCTION="${INSTRUCTION:-Given a Chinese search query, retrieve relevant passages that answer the query.}"

read -r -a DATASET_ARGS <<< "${DATASETS}"
EXTRA_ARGS=()
if [[ "${ENSURE_POSITIVES}" == "1" ]]; then
  EXTRA_ARGS+=(--ensure_positives)
fi

"${PYTHON_BIN}" src/build_cmteb_r_candidates.py \
  --input_dir "${INPUT_DIR}" \
  --qrels_input_dir "${QRELS_INPUT_DIR}" \
  --output_file "${OUTPUT_FILE}" \
  --datasets "${DATASET_ARGS[@]}" \
  --candidate_top_k "${CANDIDATE_TOP_K}" \
  --max_queries_per_dataset "${MAX_QUERIES_PER_DATASET}" \
  --doc_max_chars "${DOC_MAX_CHARS}" \
  --index_doc_max_chars "${INDEX_DOC_MAX_CHARS}" \
  --instruction "${INSTRUCTION}" \
  "${EXTRA_ARGS[@]}"
