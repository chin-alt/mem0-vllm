#!/usr/bin/env bash
set -euo pipefail

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

TRAIN_FILE="${TRAIN_FILE:-data/split_seed42/train.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/modernbert_pointwise}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-answerdotai/ModernBERT-base}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-8}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
WARMUP_RATIO="${WARMUP_RATIO:-0.03}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
EVAL_RATIO="${EVAL_RATIO:-0.1}"
TEST_RATIO="${TEST_RATIO:-0.1}"
SEED="${SEED:-42}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-16}"
RELEVANCE_THRESHOLD="${RELEVANCE_THRESHOLD:-0.7}"
PYTHON_BIN="${PYTHON_BIN:-python}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"

EXTRA_ARGS=()
if [[ -n "${DEV_FILE:-}" ]]; then
  EXTRA_ARGS+=(--dev_file "${DEV_FILE}")
fi
if [[ -n "${TEST_FILE:-}" ]]; then
  EXTRA_ARGS+=(--test_file "${TEST_FILE}")
fi
if [[ -n "${SPLIT_FILE:-}" ]]; then
  EXTRA_ARGS+=(--split_file "${SPLIT_FILE}")
fi
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
if [[ "${GRADIENT_CHECKPOINTING:-1}" == "0" ]]; then
  EXTRA_ARGS+=(--no-gradient_checkpointing)
fi
if [[ "${DISABLE_TQDM:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_tqdm)
fi

LAUNCHER=("${PYTHON_BIN}")
if [[ "${NUM_PROCESSES}" != "1" ]]; then
  MIXED_PRECISION="${MIXED_PRECISION:-no}"
  if [[ "${BF16:-0}" == "1" ]]; then
    MIXED_PRECISION="bf16"
  elif [[ "${FP16:-0}" == "1" ]]; then
    MIXED_PRECISION="fp16"
  fi
  LAUNCHER=(accelerate launch --num_processes "${NUM_PROCESSES}" --mixed_precision "${MIXED_PRECISION}")
fi

"${LAUNCHER[@]}" src/train_modernbert_pointwise.py \
  --train_file "${TRAIN_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --max_length "${MAX_LENGTH}" \
  --epochs "${EPOCHS}" \
  --lr "${LR}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --eval_ratio "${EVAL_RATIO}" \
  --test_ratio "${TEST_RATIO}" \
  --seed "${SEED}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --relevance_threshold "${RELEVANCE_THRESHOLD}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
