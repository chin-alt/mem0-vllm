#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

TRAIN_FILE="${TRAIN_FILE:-data/split_seed42/train.jsonl}"
DEV_FILE="${DEV_FILE:-data/split_seed42/dev.jsonl}"
TEST_FILE="${TEST_FILE:-data/split_seed42/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3_reranker_listwise_lora}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-Reranker-0.6B}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
NUM_PROCESSES="${NUM_PROCESSES:-1}"
MIXED_PRECISION="${MIXED_PRECISION:-fp16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"

PRECISION_ARGS=()
if [[ "${MIXED_PRECISION}" == "fp16" ]]; then
  PRECISION_ARGS+=(--fp16)
elif [[ "${MIXED_PRECISION}" == "bf16" ]]; then
  PRECISION_ARGS+=(--bf16)
fi

accelerate launch \
  --num_processes "${NUM_PROCESSES}" \
  --mixed_precision "${MIXED_PRECISION}" \
  src/train_listwise.py \
  --train_file "${TRAIN_FILE}" \
  --dev_file "${DEV_FILE}" \
  --test_file "${TEST_FILE}" \
  --output_dir "${OUTPUT_DIR}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --max_length "${MAX_LENGTH}" \
  --epochs "${EPOCHS:-3}" \
  --lr "${LR:-2e-5}" \
  --per_device_train_group_batch_size "${GROUP_BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-8}" \
  --warmup_ratio "${WARMUP_RATIO:-0.03}" \
  --weight_decay "${WEIGHT_DECAY:-0.01}" \
  --attn_implementation "${ATTN_IMPLEMENTATION}" \
  --teacher_score_scale "${TEACHER_SCORE_SCALE:-normalized}" \
  --teacher_temperature "${TEACHER_TEMPERATURE:-1.0}" \
  --model_temperature "${MODEL_TEMPERATURE:-1.0}" \
  --loss_type "${LOSS_TYPE:-kl}" \
  --min_group_size "${MIN_GROUP_SIZE:-2}" \
  --max_group_size "${MAX_GROUP_SIZE:-16}" \
  --group_truncation "${GROUP_TRUNCATION:-input_order}" \
  --use_lora \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --gradient_checkpointing \
  "${PRECISION_ARGS[@]}" \
  "$@"
