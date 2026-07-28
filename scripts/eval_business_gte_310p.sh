#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

MODEL_PATH="${MODEL_PATH:-/home/reranker_experiment/model/gte-multilingual-reranker-base}"
DATA_ROOT="${DATA_ROOT:-/home/reranker_experiment/data/latency_delay}"
DATASET="${DATASET:-0428caption}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/business_gte_310p_$(date +%Y%m%d_%H%M%S)}"
BACKEND="${BACKEND:-torch_npu}"
DEVICE="${DEVICE:-npu:0}"
DTYPE="${DTYPE:-fp16}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-pfa}"
JIT_COMPILE="${JIT_COMPILE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-${MAX_LENGTH}}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${BATCH_SIZE}}"
ENFORCE_EAGER="${ENFORCE_EAGER:-1}"
INSTRUCTION="${INSTRUCTION:-}"
PYTHON_BIN="${PYTHON_BIN:-python}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

if [[ "${BACKEND}" != "torch_npu" && "${BACKEND}" != "vllm" ]]; then
  echo "Unsupported BACKEND=${BACKEND}; use torch_npu or vllm." >&2
  exit 2
fi

if [[ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]]; then
  set +u
  source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
  set -u
fi

if [[ ! -e "${MODEL_PATH}" ]]; then
  echo "[missing] ${MODEL_PATH}" >&2
  exit 3
fi

if [[ "${DATASET}" == "all" ]]; then
  if [[ -n "${GT_FILE:-}" || -n "${RECALL_FILE:-}" || -n "${GT_DOC_ID_COL:-}" ]]; then
    echo "GT_FILE, RECALL_FILE, and GT_DOC_ID_COL overrides are only valid for a single DATASET." >&2
    exit 2
  fi
  DATASETS=("0428caption" "0428keyword" "0625caption")
else
  DATASETS=("${DATASET}")
fi

run_dataset() {
  local dataset_name="$1"
  local dataset_dir="${DATA_ROOT}/${dataset_name}"
  local recall_file=""
  local gt_doc_id_col=""
  local gt_file=""

  if [[ ! -d "${dataset_dir}" ]]; then
    echo "[missing] ${dataset_dir}" >&2
    return 3
  fi

  case "${dataset_name}" in
    0428caption)
      recall_file="${DATA_ROOT}/0428caption/retrieve_id_caption_0416.json"
      gt_doc_id_col="PageId_new"
      ;;
    0428keyword)
      recall_file="${DATA_ROOT}/0428keyword/id_keywords_pair_new.json"
      gt_doc_id_col="PageId_new"
      ;;
    0625caption)
      recall_file="${DATA_ROOT}/0625caption/0625_raw_recall_result.json"
      gt_doc_id_col="PageId"
      ;;
    *)
      echo "Unsupported DATASET=${dataset_name}; use all, 0428caption, 0428keyword, or 0625caption." >&2
      return 2
      ;;
  esac

  if [[ "${DATASET}" != "all" ]]; then
    recall_file="${RECALL_FILE:-${recall_file}}"
    gt_doc_id_col="${GT_DOC_ID_COL:-${gt_doc_id_col}}"
    gt_file="${GT_FILE:-}"
  fi
  if [[ -z "${gt_file}" ]]; then
    if [[ "${dataset_name}" == "0625caption" && -f "${dataset_dir}/gtfile-20260617.xlsx" ]]; then
      gt_file="${dataset_dir}/gtfile-20260617.xlsx"
    else
      gt_file="$(find "${dataset_dir}" -maxdepth 1 -type f -name '*.xlsx' -print -quit)"
    fi
  fi

  local required=""
  for required in "${gt_file}" "${recall_file}"; do
    if [[ ! -e "${required}" ]]; then
      echo "[missing] ${required}" >&2
      return 3
    fi
  done

  local run_dir="${OUTPUT_ROOT}/${dataset_name}__gte_multilingual_reranker_base"
  echo "======================================================================"
  echo "[gte] model=${MODEL_PATH}"
  echo "[gte] dataset=${dataset_name} backend=${BACKEND} device=${DEVICE} dtype=${DTYPE}"
  echo "[gte] max_length=${MAX_LENGTH} batch_size=${BATCH_SIZE}"
  if [[ "${BACKEND}" == "torch_npu" ]]; then
    echo "[gte] attention=${ATTENTION_BACKEND} jit_compile=${JIT_COMPILE}"
  else
    echo "[gte] vllm_runner=pooling enforce_eager=${ENFORCE_EAGER}"
  fi
  echo "[gte] output=${run_dir}"
  echo "======================================================================"

  if [[ "${BACKEND}" == "vllm" ]]; then
    local vllm_dtype="${DTYPE}"
    case "${vllm_dtype}" in
      fp16) vllm_dtype="float16" ;;
      bf16) vllm_dtype="bfloat16" ;;
      fp32) vllm_dtype="float32" ;;
    esac
    local eager_args=()
    if [[ "${ENFORCE_EAGER}" == "1" ]]; then
      eager_args+=(--enforce_eager)
    fi
    "${PYTHON_BIN}" business_eval_vllm.py \
      --gt_file "${gt_file}" \
      --recall_file "${recall_file}" \
      --model_path "${MODEL_PATH}" \
      --output_dir "${run_dir}" \
      --model_family gte \
      --scoring_backend pooling \
      --device_backend ascend \
      --instruction "${INSTRUCTION}" \
      --dtype "${vllm_dtype}" \
      --max_length "${MAX_LENGTH}" \
      --batch_size "${BATCH_SIZE}" \
      --gt_doc_id_col "${gt_doc_id_col}" \
      --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
      --max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS}" \
      --max_num_seqs "${MAX_NUM_SEQS}" \
      --no_enable_prefix_caching \
      --sort_by_length \
      --local_files_only \
      "${eager_args[@]}"
  else
    local compile_args=()
    if [[ "${JIT_COMPILE}" == "1" ]]; then
      compile_args+=(--jit_compile)
    fi
    "${PYTHON_BIN}" src/evaluate_business_gte_npu.py \
      --gt_file "${gt_file}" \
      --recall_file "${recall_file}" \
      --model_path "${MODEL_PATH}" \
      --output_dir "${run_dir}" \
      --device "${DEVICE}" \
      --dtype "${DTYPE}" \
      --attention_backend "${ATTENTION_BACKEND}" \
      --instruction "${INSTRUCTION}" \
      --max_length "${MAX_LENGTH}" \
      --batch_size "${BATCH_SIZE}" \
      --gt_doc_id_col "${gt_doc_id_col}" \
      --score_activation sigmoid \
      --local_files_only \
      "${compile_args[@]}"
  fi

  echo "[done] ${run_dir}/metrics.json"
}

mkdir -p "${OUTPUT_ROOT}"
for dataset_name in "${DATASETS[@]}"; do
  run_dataset "${dataset_name}"
done

"${PYTHON_BIN}" src/summarize_business_matrix.py \
  --output_root "${OUTPUT_ROOT}" \
  --summary_csv "${OUTPUT_ROOT}/summary_metrics.csv" \
  --summary_json "${OUTPUT_ROOT}/summary_metrics.json" \
  --summary_xlsx "${OUTPUT_ROOT}/summary_metrics.xlsx"

echo "[done] matrix summary: ${OUTPUT_ROOT}/summary_metrics.xlsx"
