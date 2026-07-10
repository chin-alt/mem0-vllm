#!/usr/bin/env bash
set -euo pipefail

INPUT_DIR="${INPUT_DIR:-data/cmteb_r/raw}"
OUTPUT_FILE="${OUTPUT_FILE:-data/cmteb_r/cmteb_r_eval.jsonl}"
DATASETS="${DATASETS:-T2Retrieval MMarcoRetrieval DuRetrieval CovidRetrieval CmedqaRetrieval EcomRetrieval MedicalRetrieval}"
NEGATIVES_PER_QUERY="${NEGATIVES_PER_QUERY:-15}"
MAX_QUERIES_PER_DATASET="${MAX_QUERIES_PER_DATASET:-1000}"
MAX_DOCS_PER_QUERY="${MAX_DOCS_PER_QUERY:-32}"
DOC_MAX_CHARS="${DOC_MAX_CHARS:-0}"
SEED="${SEED:-42}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INSTRUCTION="${INSTRUCTION:-Given a Chinese search query, retrieve relevant passages that answer the query.}"
SKIP_MISSING_QRELS="${SKIP_MISSING_QRELS:-0}"
SUPERVISION_STRATEGY="${SUPERVISION_STRATEGY:-auto}"
MIN_ID_MATCH_RATIO="${MIN_ID_MATCH_RATIO:-0.8}"
QRELS_INPUT_DIR="${QRELS_INPUT_DIR:-}"
QRELS_DATASET_SUFFIX="${QRELS_DATASET_SUFFIX:--qrels}"

read -r -a DATASET_ARGS <<< "${DATASETS}"
EXTRA_ARGS=()
if [[ "${SKIP_MISSING_QRELS}" == "1" ]]; then
  EXTRA_ARGS+=(--skip_missing_qrels)
fi

"${PYTHON_BIN}" src/prepare_cmteb_r.py \
  --input_dir "${INPUT_DIR}" \
  --output_file "${OUTPUT_FILE}" \
  --datasets "${DATASET_ARGS[@]}" \
  --instruction "${INSTRUCTION}" \
  --negatives_per_query "${NEGATIVES_PER_QUERY}" \
  --max_queries_per_dataset "${MAX_QUERIES_PER_DATASET}" \
  --max_docs_per_query "${MAX_DOCS_PER_QUERY}" \
  --doc_max_chars "${DOC_MAX_CHARS}" \
  --seed "${SEED}" \
  --qrels_input_dir "${QRELS_INPUT_DIR}" \
  --qrels_dataset_suffix="${QRELS_DATASET_SUFFIX}" \
  --supervision_strategy "${SUPERVISION_STRATEGY}" \
  --min_id_match_ratio "${MIN_ID_MATCH_RATIO}" \
  "${EXTRA_ARGS[@]}"
