#!/usr/bin/env bash
set -euo pipefail

TEST_FILE="${TEST_FILE:-data/cmteb_r/cmteb_r_eval.jsonl}"
MODEL_PATH="${MODEL_PATH:-models/Qwen3-Reranker-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/cmteb_r_vllm_eval}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SCORING_BACKEND="${SCORING_BACKEND:-pooling}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-auto}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
SORT_DESCENDING="${SORT_DESCENDING:-0}"
EXPECTED_FBETA_BETAS="${EXPECTED_FBETA_BETAS:-0.2 0.3 0.5 0.7 1.0}"

detect_visible_gpu_count() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" != "-1" ]]; then
    local count=0
    local device
    IFS=',' read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    for device in "${visible_devices[@]}"; do
      device="${device//[[:space:]]/}"
      if [[ -n "${device}" ]]; then
        count=$((count + 1))
      fi
    done
    if [[ "${count}" -gt 0 ]]; then
      echo "${count}"
      return
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L | wc -l
    return
  fi
  echo 1
}

if [[ "${TENSOR_PARALLEL_SIZE}" == "auto" ]]; then
  TENSOR_PARALLEL_SIZE="$(detect_visible_gpu_count)"
fi
if [[ "${TENSOR_PARALLEL_SIZE}" -lt 1 ]]; then
  TENSOR_PARALLEL_SIZE=1
fi

EXTRA_ARGS=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  EXTRA_ARGS+=(--local_files_only)
fi
if [[ "${SORT_DESCENDING}" == "1" ]]; then
  EXTRA_ARGS+=(--sort_descending)
fi
read -r -a BETA_ARGS <<< "${EXPECTED_FBETA_BETAS}"

"${PYTHON_BIN}" src/evaluate_jsonl_vllm.py \
  --test_file "${TEST_FILE}" \
  --model_path "${MODEL_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --max_length "${MAX_LENGTH}" \
  --batch_size "${BATCH_SIZE}" \
  --scoring_backend "${SCORING_BACKEND}" \
  --dtype "${DTYPE}" \
  --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
  --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
  --max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --max_num_seqs "${MAX_NUM_SEQS}" \
  --expected_fbeta_betas "${BETA_ARGS[@]}" \
  --sort_by_length \
  "${EXTRA_ARGS[@]}"
