#!/usr/bin/env bash
set -euo pipefail

TEST_FILE="${TEST_FILE:-data/cmteb_r/cmteb_r_qwen3_embedding_candidates.jsonl}"
MODEL_PATH="${MODEL_PATH:-outputs/modernbert_pointwise/best}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cmteb_r_crossencoder_dp_eval}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PRECISION="${PRECISION:-bf16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
SCORE_ACTIVATION="${SCORE_ACTIVATION:-sigmoid}"
RELEVANCE_THRESHOLD="${RELEVANCE_THRESHOLD:-0.7}"
EXPECTED_FBETA_BETAS="${EXPECTED_FBETA_BETAS:-0.2 0.3 0.5 0.7 1.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
DEVICES="${DEVICES:-auto}"
NUM_SHARDS="${NUM_SHARDS:-auto}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
PROGRESS_POLL_INTERVAL="${PROGRESS_POLL_INTERVAL:-1.0}"

EXTRA_ARGS=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--local_files_only)
fi
if [[ "${SHOW_PROGRESS}" == "1" ]]; then
  EXTRA_ARGS+=(--show_progress)
else
  EXTRA_ARGS+=(--no_show_progress)
fi
if [[ -n "${ATTN_IMPLEMENTATION}" ]]; then
  EXTRA_ARGS+=(--attn_implementation "${ATTN_IMPLEMENTATION}")
fi
if [[ "${PRECISION}" == "fp16" ]]; then
  EXTRA_ARGS+=(--fp16)
elif [[ "${PRECISION}" == "bf16" ]]; then
  EXTRA_ARGS+=(--bf16)
elif [[ "${PRECISION}" != "fp32" ]]; then
  echo "Unsupported PRECISION=${PRECISION}; use fp16, bf16, or fp32." >&2
  exit 2
fi
read -r -a BETA_ARGS <<< "${EXPECTED_FBETA_BETAS}"

"${PYTHON_BIN}" src/evaluate_jsonl_crossencoder_dp.py \
  --test_file "${TEST_FILE}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --relevance_threshold "${RELEVANCE_THRESHOLD}" \
  --score_activation "${SCORE_ACTIVATION}" \
  --devices "${DEVICES}" \
  --num_shards "${NUM_SHARDS}" \
  --progress_poll_interval "${PROGRESS_POLL_INTERVAL}" \
  --expected_fbeta_betas "${BETA_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"
