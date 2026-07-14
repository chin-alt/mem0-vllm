#!/usr/bin/env bash
set -euo pipefail

TEST_FILE="${TEST_FILE:-data/cmteb_r/cmteb_r_qwen3_embedding_candidates.jsonl}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-Reranker-4B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cmteb_r_vllm_dp_eval}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SCORING_BACKEND="${SCORING_BACKEND:-pooling}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
SORT_DESCENDING="${SORT_DESCENDING:-0}"
EXPECTED_FBETA_BETAS="${EXPECTED_FBETA_BETAS:-0.2 0.3 0.5 0.7 1.0}"
DEVICES="${DEVICES:-auto}"
NUM_SHARDS="${NUM_SHARDS:-auto}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"
PROGRESS_POLL_INTERVAL="${PROGRESS_POLL_INTERVAL:-1.0}"

EXTRA_ARGS=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--local_files_only)
fi
if [[ "${SORT_DESCENDING}" == "1" ]]; then
  EXTRA_ARGS+=(--sort_descending)
fi
if [[ "${SHOW_PROGRESS}" == "1" ]]; then
  EXTRA_ARGS+=(--show_progress)
else
  EXTRA_ARGS+=(--no_show_progress)
fi
read -r -a BETA_ARGS <<< "${EXPECTED_FBETA_BETAS}"

"${PYTHON_BIN}" src/evaluate_jsonl_vllm_dp.py \
  --test_file "${TEST_FILE}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --scoring_backend "${SCORING_BACKEND}" \
  --dtype "${DTYPE}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max_num_seqs "${MAX_NUM_SEQS}" \
  --devices "${DEVICES}" \
  --num_shards "${NUM_SHARDS}" \
  --progress_poll_interval "${PROGRESS_POLL_INTERVAL}" \
  --expected_fbeta_betas "${BETA_ARGS[@]}" \
  --sort_by_length \
  "${EXTRA_ARGS[@]}"
