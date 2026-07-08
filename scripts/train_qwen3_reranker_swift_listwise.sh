#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if ! command -v swift >/dev/null 2>&1; then
  echo "ERROR: ms-swift CLI not found. Install it with: pip install -U -r requirements-swift.txt" >&2
  exit 127
fi

SWIFT_SFT_HELP="$(swift sft --help 2>&1 || true)"
if ! grep -q -- "--task_type" <<<"${SWIFT_SFT_HELP}" \
  || ! grep -q -- "--loss_type" <<<"${SWIFT_SFT_HELP}"; then
  cat >&2 <<'EOF'
ERROR: The installed ms-swift CLI does not support native Qwen3 reranker listwise training.

This script requires the newer ms-swift reranker API:
  swift sft --model ... --task_type generative_reranker --loss_type listwise_reranker

Your ms-swift 2.x CLI uses older arguments such as --model_id_or_path,
--sft_type, --dtype, and --use_flash_attn. It does not expose --task_type or
--loss_type, so it cannot run the native listwise_reranker loss used here.

Fix on the training machine:
  pip uninstall -y ms-swift swift
  pip install -U -r requirements-swift.txt
  swift sft --help | grep -E "task_type|loss_type|--model "

If the cluster must stay on ms-swift 2.6.1, use the repository's custom
listwise soft-label trainer instead:
  bash scripts/train_qwen3_reranker_listwise.sh
EOF
  exit 2
fi

has_swift_arg() {
  grep -Eq "(^|[[:space:]])$1([[:space:],]|$)" <<<"${SWIFT_SFT_HELP}"
}

TRAIN_FILE="${TRAIN_FILE:-data/split_seed42/train.jsonl}"
DEV_FILE="${DEV_FILE:-data/split_seed42/dev.jsonl}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-Reranker-0.6B}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen3_reranker_swift_listwise_lora}"
SWIFT_DATA_DIR="${SWIFT_DATA_DIR:-data/swift_listwise_seed42}"

SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-5e-5}"
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
SAVE_STEPS="${SAVE_STEPS:-100}"
EVAL_STEPS="${EVAL_STEPS:-100}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-4}"
DATASET_NUM_PROC="${DATASET_NUM_PROC:-4}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
ATTN_IMPL="${ATTN_IMPL:-flash_attn}"
PADDING_FREE="${PADDING_FREE:-true}"
TUNER_TYPE="${TUNER_TYPE:-lora}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"
LORA_R="${LORA_R:-16}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
DEEPSPEED="${DEEPSPEED:-}"
SPLIT_DATASET_RATIO="${SPLIT_DATASET_RATIO:-0.05}"
LOAD_FROM_CACHE_FILE="${LOAD_FROM_CACHE_FILE:-true}"
DATALOADER_DROP_LAST="${DATALOADER_DROP_LAST:-true}"
USE_HF="${USE_HF:-false}"

POSITIVE_THRESHOLD="${POSITIVE_THRESHOLD:-0.7}"
NEGATIVE_THRESHOLD="${NEGATIVE_THRESHOLD:-}"
POSITIVE_STRATEGY="${POSITIVE_STRATEGY:-threshold_or_top1}"
MIN_GROUP_SIZE="${MIN_GROUP_SIZE:-2}"
MAX_POSITIVE_MESSAGES_EXPORT="${MAX_POSITIVE_MESSAGES_EXPORT:-0}"
MAX_NEGATIVE_MESSAGES_EXPORT="${MAX_NEGATIVE_MESSAGES_EXPORT:-0}"

export MAX_POSITIVE_SAMPLES="${MAX_POSITIVE_SAMPLES:-1}"
export MAX_NEGATIVE_SAMPLES="${MAX_NEGATIVE_SAMPLES:-7}"
export LISTWISE_RERANKER_TEMPERATURE="${LISTWISE_RERANKER_TEMPERATURE:-1.0}"
export LISTWISE_RERANKER_MIN_GROUP_SIZE="${LISTWISE_RERANKER_MIN_GROUP_SIZE:-2}"
export GENERATIVE_RERANKER_POSITIVE_TOKEN="${GENERATIVE_RERANKER_POSITIVE_TOKEN:-yes}"
export GENERATIVE_RERANKER_NEGATIVE_TOKEN="${GENERATIVE_RERANKER_NEGATIVE_TOKEN:-no}"

mkdir -p "${SWIFT_DATA_DIR}"
SWIFT_TRAIN_FILE="${SWIFT_DATA_DIR}/train.swift.jsonl"
SWIFT_DEV_FILE="${SWIFT_DATA_DIR}/dev.swift.jsonl"

EXPORT_COMMON_ARGS=(
  --positive_threshold "${POSITIVE_THRESHOLD}"
  --positive_strategy "${POSITIVE_STRATEGY}"
  --min_group_size "${MIN_GROUP_SIZE}"
  --max_positive_messages "${MAX_POSITIVE_MESSAGES_EXPORT}"
  --max_negative_messages "${MAX_NEGATIVE_MESSAGES_EXPORT}"
  --sort_by_label
)
if [[ -n "${NEGATIVE_THRESHOLD}" ]]; then
  EXPORT_COMMON_ARGS+=(--negative_threshold "${NEGATIVE_THRESHOLD}")
fi

python src/export_swift_reranker_data.py \
  --input_file "${TRAIN_FILE}" \
  --output_file "${SWIFT_TRAIN_FILE}" \
  "${EXPORT_COMMON_ARGS[@]}"

VAL_ARGS=()
if [[ -n "${DEV_FILE}" && -f "${DEV_FILE}" ]]; then
  python src/export_swift_reranker_data.py \
    --input_file "${DEV_FILE}" \
    --output_file "${SWIFT_DEV_FILE}" \
    "${EXPORT_COMMON_ARGS[@]}"
  VAL_ARGS+=(--val_dataset "${SWIFT_DEV_FILE}")
else
  VAL_ARGS+=(--split_dataset_ratio "${SPLIT_DATASET_RATIO}")
fi

LORA_ARGS=()
if [[ "${TUNER_TYPE}" == "lora" || "${TUNER_TYPE}" == "longlora" ]]; then
  LORA_ARGS+=(
    --lora_rank "${LORA_R}"
    --lora_alpha "${LORA_ALPHA}"
    --lora_dropout "${LORA_DROPOUT}"
  )
  if has_swift_arg "--target_modules"; then
    LORA_ARGS+=(--target_modules "${TARGET_MODULES}")
  elif has_swift_arg "--lora_target_modules"; then
    LORA_ARGS+=(--lora_target_modules "${TARGET_MODULES}")
  fi
fi

DEEPSPEED_ARGS=()
if [[ -n "${DEEPSPEED}" ]]; then
  DEEPSPEED_ARGS+=(--deepspeed "${DEEPSPEED}")
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-${NUM_PROCESSES:-1}}"
export NPROC_PER_NODE

MODEL_ARGS=()
if has_swift_arg "--model"; then
  MODEL_ARGS+=(--model "${MODEL_NAME_OR_PATH}")
elif has_swift_arg "--model_id_or_path"; then
  MODEL_ARGS+=(--model_id_or_path "${MODEL_NAME_OR_PATH}")
else
  echo "ERROR: Could not find --model or --model_id_or_path in swift sft --help." >&2
  exit 2
fi

TRAIN_TYPE_ARGS=()
if [[ "${TUNER_TYPE}" == "full" ]]; then
  if has_swift_arg "--tuner_type"; then
    TRAIN_TYPE_ARGS+=(--tuner_type full)
  elif has_swift_arg "--train_type"; then
    TRAIN_TYPE_ARGS+=(--train_type full)
  elif has_swift_arg "--sft_type"; then
    TRAIN_TYPE_ARGS+=(--sft_type full)
  fi
elif has_swift_arg "--tuner_type"; then
  TRAIN_TYPE_ARGS+=(--tuner_type "${TUNER_TYPE}")
elif has_swift_arg "--train_type"; then
  TRAIN_TYPE_ARGS+=(--train_type "${TUNER_TYPE}")
elif has_swift_arg "--sft_type"; then
  TRAIN_TYPE_ARGS+=(--sft_type "${TUNER_TYPE}")
else
  echo "ERROR: Could not find --tuner_type, --train_type, or --sft_type in swift sft --help." >&2
  exit 2
fi

PRECISION_ARGS=()
if has_swift_arg "--torch_dtype"; then
  PRECISION_ARGS+=(--torch_dtype "${TORCH_DTYPE}")
elif has_swift_arg "--dtype"; then
  case "${TORCH_DTYPE}" in
    float16) PRECISION_ARGS+=(--dtype fp16) ;;
    bfloat16) PRECISION_ARGS+=(--dtype bf16) ;;
    float32) PRECISION_ARGS+=(--dtype fp32) ;;
    *) PRECISION_ARGS+=(--dtype "${TORCH_DTYPE}") ;;
  esac
fi

ATTN_ARGS=()
if has_swift_arg "--attn_impl"; then
  ATTN_ARGS+=(--attn_impl "${ATTN_IMPL}")
elif has_swift_arg "--use_flash_attn"; then
  if [[ "${ATTN_IMPL}" == "flash_attn" || "${ATTN_IMPL}" == "flash_attention_2" ]]; then
    ATTN_ARGS+=(--use_flash_attn true)
  else
    ATTN_ARGS+=(--use_flash_attn false)
  fi
fi
if has_swift_arg "--padding_free"; then
  ATTN_ARGS+=(--padding_free "${PADDING_FREE}")
fi

DATA_CACHE_ARGS=()
if has_swift_arg "--load_from_cache_file"; then
  DATA_CACHE_ARGS+=(--load_from_cache_file "${LOAD_FROM_CACHE_FILE}")
fi

DATA_SEED_ARGS=()
if has_swift_arg "--data_seed"; then
  DATA_SEED_ARGS+=(--data_seed "${SEED}")
elif has_swift_arg "--dataset_seed"; then
  DATA_SEED_ARGS+=(--dataset_seed "${SEED}")
fi

WORKER_ARGS=()
if has_swift_arg "--dataset_num_proc"; then
  WORKER_ARGS+=(--dataset_num_proc "${DATASET_NUM_PROC}")
elif has_swift_arg "--preprocess_num_proc"; then
  WORKER_ARGS+=(--preprocess_num_proc "${DATASET_NUM_PROC}")
fi
if has_swift_arg "--dataloader_num_workers"; then
  WORKER_ARGS+=(--dataloader_num_workers "${DATALOADER_NUM_WORKERS}")
fi

USE_HF_ARGS=()
if has_swift_arg "--use_hf"; then
  USE_HF_ARGS+=(--use_hf "${USE_HF}")
fi

swift sft \
  "${MODEL_ARGS[@]}" \
  --task_type generative_reranker \
  --loss_type listwise_reranker \
  "${TRAIN_TYPE_ARGS[@]}" \
  "${LORA_ARGS[@]}" \
  --dataset "${SWIFT_TRAIN_FILE}" \
  "${VAL_ARGS[@]}" \
  "${USE_HF_ARGS[@]}" \
  "${ATTN_ARGS[@]}" \
  "${PRECISION_ARGS[@]}" \
  "${DATA_CACHE_ARGS[@]}" \
  --eval_strategy steps \
  --eval_steps "${EVAL_STEPS}" \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT}" \
  --logging_steps "${LOGGING_STEPS}" \
  --num_train_epochs "${EPOCHS}" \
  --max_length "${MAX_LENGTH}" \
  --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}" \
  --per_device_eval_batch_size "${PER_DEVICE_EVAL_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  "${WORKER_ARGS[@]}" \
  --learning_rate "${LR}" \
  --warmup_ratio "${WARMUP_RATIO}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --dataloader_drop_last "${DATALOADER_DROP_LAST}" \
  --seed "${SEED}" \
  "${DATA_SEED_ARGS[@]}" \
  --output_dir "${OUTPUT_DIR}" \
  "${DEEPSPEED_ARGS[@]}" \
  "$@"
