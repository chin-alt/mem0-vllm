#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-outputs/modernbert_pointwise/best}"
DOCS_FILE="${DOCS_FILE:-data/docs.jsonl}"
OUTPUT_FILE="${OUTPUT_FILE:-predictions_modernbert_ranked.json}"
QUERY="${QUERY:-Which pocket camera ships faster?}"
INSTRUCTION="${INSTRUCTION:-Judge whether the document answers the query.}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXTRA_ARGS=()
if [[ "${BF16:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--bf16)
fi
if [[ "${FP16:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--fp16)
fi
if [[ "${LOCAL_FILES_ONLY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--local_files_only)
fi
if [[ -n "${ATTN_IMPLEMENTATION:-}" ]]; then
  EXTRA_ARGS+=(--attn_implementation "${ATTN_IMPLEMENTATION}")
fi

"${PYTHON_BIN}" src/predict_modernbert.py \
  --model_path "${MODEL_PATH}" \
  --instruction "${INSTRUCTION}" \
  --query "${QUERY}" \
  --docs_file "${DOCS_FILE}" \
  --output_file "${OUTPUT_FILE}" \
  --top_k "${TOP_K:-10}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
