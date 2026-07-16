#!/usr/bin/env bash
set -euo pipefail

TEST_FILE="${TEST_FILE:-data/split_seed42/test.jsonl}"
MODEL_PATH="${MODEL_PATH:-outputs/modernbert_pointwise/best}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/modernbert_eval}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RELEVANCE_THRESHOLD="${RELEVANCE_THRESHOLD:-0.7}"
PYTHON_BIN="${PYTHON_BIN:-python}"

EXTRA_ARGS=()
if [[ -n "${DEFAULT_INSTRUCTION:-}" ]]; then
  EXTRA_ARGS+=(--default_instruction "${DEFAULT_INSTRUCTION}")
fi
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

"${PYTHON_BIN}" src/evaluate_modernbert.py \
  --test_file "${TEST_FILE}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --relevance_threshold "${RELEVANCE_THRESHOLD}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
