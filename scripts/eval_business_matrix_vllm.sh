#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run vLLM offline business reranker evaluation for the same model x dataset matrix.

Environment overrides:
  OUTPUT_ROOT              Output root. Default: outputs/business_matrix_vllm_<timestamp>
  CUDA_VISIBLE_DEVICES     GPU ids for evaluation. Default: 0
  MAX_LENGTH               Max sequence length. Default: 2048
  BATCH_SIZE               vLLM LLM.score chunk size. Default: 256
  EXPECTED_FBETA_BETA      Beta for dynamic Expected-Fbeta cutoff. Default: 0.3
  DTYPE                    auto, bfloat16, float16, or float32. Default: bfloat16
  GPU_MEMORY_UTILIZATION   vLLM GPU memory utilization. Default: 0.90
  TENSOR_PARALLEL_SIZE     vLLM tensor parallel size. Default: 1
  MAX_NUM_BATCHED_TOKENS   vLLM max_num_batched_tokens. Default: 32768
  MAX_NUM_SEQS             vLLM max_num_seqs. Default: 256
  SORT_BY_LENGTH           Sort pairs by query+doc length before scoring. Default: 1
  ENABLE_PREFIX_CACHING    Enable vLLM prefix caching if supported. Default: 1
  SKIP_EXISTING            Skip a run if metrics.json already exists. Default: 1
  CONTINUE_ON_ERROR        Continue remaining runs after one failure. Default: 0
  POST_RUN_SLEEP           Seconds to wait after each run. Default: 2
  PYTHON_BIN               Python executable. Default: python

Outputs:
  <OUTPUT_ROOT>/<dataset>__<model>/metrics.json
  <OUTPUT_ROOT>/<dataset>__<model>/business_eval.xlsx
  <OUTPUT_ROOT>/<dataset>__<model>/business_eval.csv
  <OUTPUT_ROOT>/<dataset>__<model>_metrics.json
  <OUTPUT_ROOT>/<dataset>__<model>_business_eval.xlsx
  <OUTPUT_ROOT>/<dataset>__<model>_business_eval.csv
  <OUTPUT_ROOT>/summary_metrics.xlsx
  <OUTPUT_ROOT>/summary_metrics.csv
  <OUTPUT_ROOT>/summary_metrics.json

Notes:
  vLLM expects a full model directory with config.json. PEFT/LoRA adapter-only
  paths must be merged first, for example with src/merge_lora.py.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/business_matrix_vllm_${RUN_TAG}}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EXPECTED_FBETA_BETA="${EXPECTED_FBETA_BETA:-0.3}"
DTYPE="${DTYPE:-bfloat16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
SORT_BY_LENGTH="${SORT_BY_LENGTH:-1}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"
INSTRUCTION="${INSTRUCTION:-Given a user query, retrieve relevant documents that answer the query.}"
GT_QUERY_COL="${GT_QUERY_COL:-query}"
TOP_K_LIST="${TOP_K_LIST:-1 3 5 10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"
POST_RUN_SLEEP="${POST_RUN_SLEEP:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN3_RERANKER_4B_LORA_PATH="${QWEN3_RERANKER_4B_LORA_PATH:-/home/c50061497/MemOS/src/memos/reranker/memranker/outputs/qwen3_reranker_4b_8x3090_lora_merged}"
QWEN3_RERANKER_06B_LORA_PATH="${QWEN3_RERANKER_06B_LORA_PATH:-/home/c50061497/MemOS/src/memos/reranker/memranker/outputs/qwen3_reranker_06b_lora_merged}"

MODEL_NAMES=(
  "memreranker_4b"
  "qwen3_reranker_06b"
  "qwen3_reranker_4b"
  "qwen3_reranker_4b_lora"
  "qwen3_reranker_06b_lora"
)
MODEL_PATHS=(
  "/home/c50061497/MemOS/src/memos/reranker/memranker/models/IAAR-Shanghai/MemReranker-4B"
  "/home/c50061497/MemOS/src/memos/reranker/memranker/models/Qwen3-Reranker-0.6B"
  "/home/c50061497/MemOS/src/memos/reranker/memranker/models/Qwen3-Reranker-4B"
  "${QWEN3_RERANKER_4B_LORA_PATH}"
  "${QWEN3_RERANKER_06B_LORA_PATH}"
)

DATASET_NAMES=(
  "0428caption"
  "0428keyword"
  "0625caption"
)
GT_FILES=(
  "/home/c50061497/MemOS/src/memos/reranker/memranker/data/latency_delay/0428caption/测试集刷新-20260423.xlsx"
  "/home/c50061497/MemOS/src/memos/reranker/memranker/data/latency_delay/0428keyword/测试集刷新-20260423.xlsx"
  "/home/c50061497/MemOS/src/memos/reranker/memranker/data/latency_delay/0625caption/gtfile-20260617.xlsx"
)
RECALL_FILES=(
  "/home/c50061497/MemOS/src/memos/reranker/memranker/data/latency_delay/0428caption/retrieve_id_caption_0416.json"
  "/home/c50061497/MemOS/src/memos/reranker/memranker/data/latency_delay/0428keyword/id_keywords_pair_new.json"
  "/home/c50061497/MemOS/src/memos/reranker/memranker/data/latency_delay/0625caption/0625_raw_recall_result.json"
)
GT_DOC_ID_COLS=(
  "PageId_new"
  "PageId_new"
  "PageId"
)

read -r -a TOP_K_ARGS <<< "${TOP_K_LIST}"
mkdir -p "${OUTPUT_ROOT}"

sort_args=()
if [[ "${SORT_BY_LENGTH}" == "1" ]]; then
  sort_args+=(--sort_by_length)
else
  sort_args+=(--no_sort_by_length)
fi

prefix_cache_args=()
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  prefix_cache_args+=(--enable_prefix_caching)
else
  prefix_cache_args+=(--no_enable_prefix_caching)
fi

copy_named_outputs() {
  local run_dir="$1"
  local run_name="$2"
  [[ -f "${run_dir}/metrics.json" ]] && cp "${run_dir}/metrics.json" "${OUTPUT_ROOT}/${run_name}_metrics.json"
  [[ -f "${run_dir}/business_eval.csv" ]] && cp "${run_dir}/business_eval.csv" "${OUTPUT_ROOT}/${run_name}_business_eval.csv"
  [[ -f "${run_dir}/business_eval.xlsx" ]] && cp "${run_dir}/business_eval.xlsx" "${OUTPUT_ROOT}/${run_name}_business_eval.xlsx"
}

run_one() {
  local dataset_name="$1"
  local gt_file="$2"
  local recall_file="$3"
  local gt_doc_id_col="$4"
  local model_name="$5"
  local model_path="$6"
  local run_name="${dataset_name}__${model_name}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"

  if [[ "${SKIP_EXISTING}" == "1" && -f "${run_dir}/metrics.json" ]]; then
    echo "[skip] ${run_name}: ${run_dir}/metrics.json already exists"
    copy_named_outputs "${run_dir}" "${run_name}"
    return 0
  fi

  if [[ ! -e "${model_path}" ]]; then
    echo "[missing] model path does not exist: ${model_path}" >&2
    return 3
  fi
  if [[ ! -f "${gt_file}" ]]; then
    echo "[missing] gt file does not exist: ${gt_file}" >&2
    return 3
  fi
  if [[ ! -f "${recall_file}" ]]; then
    echo "[missing] recall file does not exist: ${recall_file}" >&2
    return 3
  fi

  echo "======================================================================"
  echo "[run:vllm] dataset=${dataset_name} model=${model_name}"
  echo "[run:vllm] output=${run_dir}"
  echo "======================================================================"

  "${PYTHON_BIN}" business_eval_vllm.py \
    --gt_file "${gt_file}" \
    --recall_file "${recall_file}" \
    --model_path "${model_path}" \
    --output_dir "${run_dir}" \
    --instruction "${INSTRUCTION}" \
    --gt_query_col "${GT_QUERY_COL}" \
    --gt_doc_id_col "${gt_doc_id_col}" \
    --max_length "${MAX_LENGTH}" \
    --batch_size "${BATCH_SIZE}" \
    --dtype "${DTYPE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --expected_fbeta_beta "${EXPECTED_FBETA_BETA}" \
    --top_k_list "${TOP_K_ARGS[@]}" \
    "${sort_args[@]}" \
    "${prefix_cache_args[@]}"

  copy_named_outputs "${run_dir}" "${run_name}"
  if [[ "${POST_RUN_SLEEP}" != "0" ]]; then
    sleep "${POST_RUN_SLEEP}"
  fi
}

for dataset_idx in "${!DATASET_NAMES[@]}"; do
  for model_idx in "${!MODEL_NAMES[@]}"; do
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      if ! run_one \
        "${DATASET_NAMES[$dataset_idx]}" \
        "${GT_FILES[$dataset_idx]}" \
        "${RECALL_FILES[$dataset_idx]}" \
        "${GT_DOC_ID_COLS[$dataset_idx]}" \
        "${MODEL_NAMES[$model_idx]}" \
        "${MODEL_PATHS[$model_idx]}"; then
        echo "[failed] ${DATASET_NAMES[$dataset_idx]}__${MODEL_NAMES[$model_idx]}" >&2
      fi
    else
      run_one \
        "${DATASET_NAMES[$dataset_idx]}" \
        "${GT_FILES[$dataset_idx]}" \
        "${RECALL_FILES[$dataset_idx]}" \
        "${GT_DOC_ID_COLS[$dataset_idx]}" \
        "${MODEL_NAMES[$model_idx]}" \
        "${MODEL_PATHS[$model_idx]}"
    fi
  done
done

"${PYTHON_BIN}" src/summarize_business_matrix.py \
  --output_root "${OUTPUT_ROOT}" \
  --summary_csv "${OUTPUT_ROOT}/summary_metrics.csv" \
  --summary_json "${OUTPUT_ROOT}/summary_metrics.json" \
  --summary_xlsx "${OUTPUT_ROOT}/summary_metrics.xlsx"

echo "[done] vLLM matrix outputs: ${OUTPUT_ROOT}"
