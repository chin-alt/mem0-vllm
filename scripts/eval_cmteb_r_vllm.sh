#!/usr/bin/env bash
set -euo pipefail

TEST_FILE="${TEST_FILE:-data/cmteb_r/cmteb_r_eval.jsonl}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-Reranker-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cmteb_r_vllm_eval}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-64}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
SORT_DESCENDING="${SORT_DESCENDING:-0}"

EXTRA_ARGS=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--local_files_only)
fi
if [[ "${SORT_DESCENDING}" == "1" ]]; then
  EXTRA_ARGS+=(--sort_descending)
fi

"${PYTHON_BIN}" src/evaluate_jsonl_vllm.py \
  --test_file "${TEST_FILE}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --dtype "${DTYPE}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
  --max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max_num_seqs "${MAX_NUM_SEQS}" \
  --sort_by_length \
  "${EXTRA_ARGS[@]}"
