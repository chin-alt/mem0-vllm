#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run vLLM-Ascend business reranker evaluation on Ascend Atlas NPUs.

Environment overrides:
  OUTPUT_ROOT                    Output root. Default: outputs/business_matrix_ascend_vllm_<timestamp>
  ASCEND_RT_VISIBLE_DEVICES      NPU chip ids for evaluation. Default: 0
  DATA_ROOT                      Dataset root. Default: data/latency_delay
  MODEL_ROOT                     Model root, or a single full model dir with config.json. Default: models
  OUTPUTS_ROOT                   LoRA merge output root. Default: outputs
  MODEL_NAME                     Optional single model name.
  MODEL_PATH                     Optional single full model path. Defaults to MODEL_ROOT when MODEL_NAME is set.
  MODEL_NAMES                    Optional whitespace-separated model names.
  MODEL_PATHS                    Optional |-separated model paths matching MODEL_NAMES.
  MAX_LENGTH                     Max sequence length. Default: 2048
  BATCH_SIZE                     vLLM LLM.score chunk size. Default: 64
  SCORING_BACKEND                pooling or generate. Default: pooling
  EXPECTED_FBETA_BETA            Beta for dynamic Expected-Fbeta cutoff. Default: 0.3
  DTYPE                          auto, bfloat16, float16, or float32. Default: float16
  GPU_MEMORY_UTILIZATION         vLLM memory utilization knob. Default: 0.85
  TENSOR_PARALLEL_SIZE           vLLM tensor parallel size. Default: 1
  MAX_NUM_BATCHED_TOKENS         vLLM max_num_batched_tokens. Default: 8192
  MAX_NUM_SEQS                   vLLM max_num_seqs. Default: 64
  ENFORCE_EAGER                  Disable ACL Graph capture. Default: 0
  ENABLE_PREFIX_CACHING          Enable vLLM prefix caching if supported. Default: 0
  VLLM_ADDITIONAL_CONFIG         Optional JSON object for LLM(..., additional_config=...).
  VLLM_COMPILATION_CONFIG        Optional JSON object for LLM(..., compilation_config=...).
  VLLM_QUANTIZATION              Optional quantization name, e.g. ascend for W8A8 weights.
  LOCAL_FILES_ONLY               Force offline local model loading. Default: 1
  SKIP_EXISTING                  Skip a run if metrics.json already exists. Default: 1
  CONTINUE_ON_ERROR              Continue remaining runs after one failure. Default: 1
  SKIP_MISSING                   Skip missing model/data paths. Default: 1
  POST_RUN_SLEEP                 Seconds to wait after each run. Default: 2
  PYTHON_BIN                     Python 3.9-3.11 executable. Default: python

Notes:
  Source the CANN/NNAL environment before running this script, or run inside the
  official vllm-ascend container. Prefer building vllm-ascend on the NPU host so
  npu-smi can detect the SoC automatically.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export VLLM_DEVICE_BACKEND="${VLLM_DEVICE_BACKEND:-ascend}"

RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/business_matrix_ascend_vllm_${RUN_TAG}}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data/latency_delay}"
MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
OUTPUTS_ROOT="${OUTPUTS_ROOT:-${REPO_ROOT}/outputs}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SCORING_BACKEND="${SCORING_BACKEND:-pooling}"
EXPECTED_FBETA_BETA="${EXPECTED_FBETA_BETA:-0.3}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
ENFORCE_EAGER="${ENFORCE_EAGER:-0}"
SORT_BY_LENGTH="${SORT_BY_LENGTH:-1}"
SORT_DESCENDING="${SORT_DESCENDING:-0}"
ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
LOCAL_FILES_ONLY="${LOCAL_FILES_ONLY:-1}"
INSTRUCTION="${INSTRUCTION:-Given a user query, retrieve relevant documents that answer the query.}"
GT_QUERY_COL="${GT_QUERY_COL:-query}"
TOP_K_LIST="${TOP_K_LIST:-1 3 5 10}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
SKIP_MISSING="${SKIP_MISSING:-1}"
POST_RUN_SLEEP="${POST_RUN_SLEEP:-2}"
PYTHON_BIN="${PYTHON_BIN:-python}"
QWEN3_RERANKER_4B_LORA_PATH="${QWEN3_RERANKER_4B_LORA_PATH:-${OUTPUTS_ROOT}/qwen3_reranker_4b_8x3090_lora_merged}"
QWEN3_RERANKER_06B_LORA_PATH="${QWEN3_RERANKER_06B_LORA_PATH:-${OUTPUTS_ROOT}/qwen3_reranker_06b_lora_merged}"

if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 12) else 1)'; then
  python_version="$("${PYTHON_BIN}" --version 2>&1 || true)"
  echo "[error] this vLLM-Ascend stack requires Python 3.9-3.11; ${PYTHON_BIN} is ${python_version}." >&2
  echo "[error] run: bash scripts/install_ascend_vllm_910b4.sh" >&2
  exit 2
fi
echo "[env] python=${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"
"${PYTHON_BIN}" src/vllm_py39_compat.py --quiet

is_full_model_dir() {
  local path="$1"
  [[ -f "${path}/config.json" || -f "${path}/params.json" ]]
}

if [[ -n "${MODEL_NAME:-}" || -n "${MODEL_PATH:-}" ]]; then
  single_model_path="${MODEL_PATH:-${MODEL_ROOT}}"
  single_model_name="${MODEL_NAME:-$(basename "${single_model_path}")}"
  MODEL_NAME_ARRAY=("${single_model_name}")
  MODEL_PATH_ARRAY=("${single_model_path}")
elif [[ -n "${MODEL_NAMES:-}" || -n "${MODEL_PATHS:-}" ]]; then
  if [[ -z "${MODEL_NAMES:-}" || -z "${MODEL_PATHS:-}" ]]; then
    echo "MODEL_NAMES and MODEL_PATHS must be set together." >&2
    exit 2
  fi
  read -r -a MODEL_NAME_ARRAY <<< "${MODEL_NAMES}"
  IFS='|' read -r -a MODEL_PATH_ARRAY <<< "${MODEL_PATHS}"
elif is_full_model_dir "${MODEL_ROOT}"; then
  MODEL_NAME_ARRAY=("$(basename "${MODEL_ROOT}")")
  MODEL_PATH_ARRAY=("${MODEL_ROOT}")
else
  MODEL_NAME_ARRAY=(
    "memreranker_4b"
    "qwen3_reranker_06b"
    "qwen3_reranker_4b"
    "qwen3_reranker_4b_lora"
    "qwen3_reranker_06b_lora"
  )
  MODEL_PATH_ARRAY=(
    "${MODEL_ROOT}/IAAR-Shanghai/MemReranker-4B"
    "${MODEL_ROOT}/Qwen3-Reranker-0.6B"
    "${MODEL_ROOT}/Qwen3-Reranker-4B"
    "${QWEN3_RERANKER_4B_LORA_PATH}"
    "${QWEN3_RERANKER_06B_LORA_PATH}"
  )
fi

if [[ "${#MODEL_NAME_ARRAY[@]}" -ne "${#MODEL_PATH_ARRAY[@]}" ]]; then
  echo "MODEL_NAMES count (${#MODEL_NAME_ARRAY[@]}) does not match MODEL_PATHS count (${#MODEL_PATH_ARRAY[@]})." >&2
  exit 2
fi

DATASET_NAMES=(
  "0428caption"
  "0428keyword"
  "0625caption"
)
DATASET_DIRS=(
  "${DATA_ROOT}/0428caption"
  "${DATA_ROOT}/0428keyword"
  "${DATA_ROOT}/0625caption"
)
RECALL_FILES=(
  "${DATA_ROOT}/0428caption/retrieve_id_caption_0416.json"
  "${DATA_ROOT}/0428keyword/id_keywords_pair_new.json"
  "${DATA_ROOT}/0625caption/0625_raw_recall_result.json"
)
GT_DOC_ID_COLS=(
  "PageId_new"
  "PageId_new"
  "PageId"
)
GT_FILE_HINTS=(
  ""
  ""
  "gtfile-20260617.xlsx"
)

read -r -a TOP_K_ARGS <<< "${TOP_K_LIST}"
mkdir -p "${OUTPUT_ROOT}"

sort_args=()
if [[ "${SORT_BY_LENGTH}" == "1" ]]; then
  sort_args+=(--sort_by_length)
else
  sort_args+=(--no_sort_by_length)
fi
if [[ "${SORT_DESCENDING}" == "1" ]]; then
  sort_args+=(--sort_descending)
fi

prefix_cache_args=()
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  prefix_cache_args+=(--enable_prefix_caching)
else
  prefix_cache_args+=(--no_enable_prefix_caching)
fi

local_model_args=()
if [[ "${LOCAL_FILES_ONLY}" == "1" ]]; then
  local_model_args+=(--local_files_only)
fi

optional_vllm_args=()
if [[ "${ENFORCE_EAGER}" == "1" ]]; then
  optional_vllm_args+=(--enforce_eager)
fi
if [[ -n "${VLLM_ADDITIONAL_CONFIG:-}" ]]; then
  optional_vllm_args+=(--additional_config "${VLLM_ADDITIONAL_CONFIG}")
fi
if [[ -n "${VLLM_COMPILATION_CONFIG:-}" ]]; then
  optional_vllm_args+=(--compilation_config "${VLLM_COMPILATION_CONFIG}")
fi
if [[ -n "${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-}" ]]; then
  optional_vllm_args+=(--distributed_executor_backend "${VLLM_DISTRIBUTED_EXECUTOR_BACKEND}")
fi
if [[ -n "${VLLM_QUANTIZATION:-}" ]]; then
  optional_vllm_args+=(--quantization "${VLLM_QUANTIZATION}")
fi

find_gt_file() {
  local dataset_dir="$1"
  local hint="$2"
  if [[ -n "${hint}" && -f "${dataset_dir}/${hint}" ]]; then
    printf '%s\n' "${dataset_dir}/${hint}"
    return 0
  fi
  local found=()
  while IFS= read -r file; do
    found+=("${file}")
  done < <(find "${dataset_dir}" -maxdepth 1 -type f -name "*.xlsx" | sort)
  if [[ "${#found[@]}" -eq 0 ]]; then
    return 1
  fi
  printf '%s\n' "${found[0]}"
}

copy_named_outputs() {
  local run_dir="$1"
  local run_name="$2"
  [[ -f "${run_dir}/metrics.json" ]] && cp "${run_dir}/metrics.json" "${OUTPUT_ROOT}/${run_name}_metrics.json"
  [[ -f "${run_dir}/business_eval.csv" ]] && cp "${run_dir}/business_eval.csv" "${OUTPUT_ROOT}/${run_name}_business_eval.csv"
  [[ -f "${run_dir}/business_eval.xlsx" ]] && cp "${run_dir}/business_eval.xlsx" "${OUTPUT_ROOT}/${run_name}_business_eval.xlsx"
}

run_one() {
  local dataset_name="$1"
  local dataset_dir="$2"
  local recall_file="$3"
  local gt_doc_id_col="$4"
  local gt_file_hint="$5"
  local model_name="$6"
  local model_path="$7"
  local run_name="${dataset_name}__${model_name}"
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local gt_file=""

  if [[ "${SKIP_EXISTING}" == "1" && -f "${run_dir}/metrics.json" ]]; then
    echo "[skip] ${run_name}: ${run_dir}/metrics.json already exists"
    copy_named_outputs "${run_dir}" "${run_name}"
    return 0
  fi

  if [[ ! -e "${model_path}" ]]; then
    echo "[missing] model path does not exist: ${model_path}" >&2
    [[ "${SKIP_MISSING}" == "1" ]] && return 0
    return 3
  fi
  if [[ ! -d "${dataset_dir}" ]]; then
    echo "[missing] dataset dir does not exist: ${dataset_dir}" >&2
    [[ "${SKIP_MISSING}" == "1" ]] && return 0
    return 3
  fi
  if ! gt_file="$(find_gt_file "${dataset_dir}" "${gt_file_hint}")"; then
    echo "[missing] no .xlsx ground-truth file found under: ${dataset_dir}" >&2
    [[ "${SKIP_MISSING}" == "1" ]] && return 0
    return 3
  fi
  if [[ ! -f "${recall_file}" ]]; then
    echo "[missing] recall file does not exist: ${recall_file}" >&2
    [[ "${SKIP_MISSING}" == "1" ]] && return 0
    return 3
  fi

  echo "======================================================================"
  echo "[run:ascend-vllm] dataset=${dataset_name} model=${model_name}"
  echo "[run:ascend-vllm] gt=${gt_file}"
  echo "[run:ascend-vllm] recall=${recall_file}"
  echo "[run:ascend-vllm] output=${run_dir}"
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
    --scoring_backend "${SCORING_BACKEND}" \
    --device_backend ascend \
    --dtype "${DTYPE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --max_num_batched_tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --expected_fbeta_beta "${EXPECTED_FBETA_BETA}" \
    --top_k_list "${TOP_K_ARGS[@]}" \
    "${sort_args[@]}" \
    "${prefix_cache_args[@]}" \
    "${local_model_args[@]}" \
    "${optional_vllm_args[@]}"

  copy_named_outputs "${run_dir}" "${run_name}"
  if [[ "${POST_RUN_SLEEP}" != "0" ]]; then
    sleep "${POST_RUN_SLEEP}"
  fi
}

for dataset_idx in "${!DATASET_NAMES[@]}"; do
  for model_idx in "${!MODEL_NAME_ARRAY[@]}"; do
    if [[ "${CONTINUE_ON_ERROR}" == "1" ]]; then
      if ! run_one \
        "${DATASET_NAMES[$dataset_idx]}" \
        "${DATASET_DIRS[$dataset_idx]}" \
        "${RECALL_FILES[$dataset_idx]}" \
        "${GT_DOC_ID_COLS[$dataset_idx]}" \
        "${GT_FILE_HINTS[$dataset_idx]}" \
        "${MODEL_NAME_ARRAY[$model_idx]}" \
        "${MODEL_PATH_ARRAY[$model_idx]}"; then
        echo "[failed] ${DATASET_NAMES[$dataset_idx]}__${MODEL_NAME_ARRAY[$model_idx]}" >&2
      fi
    else
      run_one \
        "${DATASET_NAMES[$dataset_idx]}" \
        "${DATASET_DIRS[$dataset_idx]}" \
        "${RECALL_FILES[$dataset_idx]}" \
        "${GT_DOC_ID_COLS[$dataset_idx]}" \
        "${GT_FILE_HINTS[$dataset_idx]}" \
        "${MODEL_NAME_ARRAY[$model_idx]}" \
        "${MODEL_PATH_ARRAY[$model_idx]}"
    fi
  done
done

metrics_count="$(find "${OUTPUT_ROOT}" -mindepth 2 -maxdepth 2 -name metrics.json | wc -l | tr -d '[:space:]')"
if [[ "${metrics_count}" == "0" ]]; then
  echo "[error] no metrics.json files were produced under ${OUTPUT_ROOT}" >&2
  echo "[error] check MODEL_ROOT/MODEL_PATH, DATA_ROOT, recall files, and ground-truth Excel files." >&2
  exit 3
fi

"${PYTHON_BIN}" src/summarize_business_matrix.py \
  --output_root "${OUTPUT_ROOT}" \
  --summary_csv "${OUTPUT_ROOT}/summary_metrics.csv" \
  --summary_json "${OUTPUT_ROOT}/summary_metrics.json" \
  --summary_xlsx "${OUTPUT_ROOT}/summary_metrics.xlsx"

echo "[done] Ascend vLLM matrix outputs: ${OUTPUT_ROOT}"
