#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Evaluate one business dataset with Qwen3-Reranker W8A8SC through pure ATB.

The evaluator captures the actual first-token yes/no logits inside ATB and
writes ranking metrics plus latency details. It does not use vLLM or MindIE
Service.

Overrides:
  HOST_REPO_PATH           Default: repository root.
  HOST_DATA_PATH           Default: /home/reranker_experiment/data/latency_delay
  HOST_MODEL_PATH          W8A8SC model directory.
  HOST_OUTPUT_PATH         Host output directory.
  HOST_PYTHON_CACHE        Persistent Python package cache.
  DATASET                  0428caption, 0428keyword, or 0625caption.
  INSTRUCTION              Reranker instruction.
  MAX_QUERIES              First N matching queries; 0 means full dataset.
  MAX_LENGTH               Default: 1024
  BATCH_SIZE               Default: 1
  TP_SIZE                  Must match compression. Default: 1
  EXPECTED_FBETA_BETA      Default: 0.3
  TOP_K_LIST               Default: "1 3 5 10"
  SORT_BY_LENGTH           0 or 1. Default: 1
  SORT_DESCENDING          0 or 1. Default: 0
  SAVE_DOC_TEXT            0 or 1. Default: 0
  IMAGE                    MindIE 2.1.RC1 300I-Duo image.
  PULL_IMAGE               0 or 1. Default: 0
  PIP_INDEX_URL            Default: NJU PyPI mirror.
  ASCEND_RT_VISIBLE_DEVICES Default: 0
  MASTER_PORT              Default: 20039
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -ne 0 ]]; then
  echo "[invalid] this script accepts environment variables, not positional arguments" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_REPO_PATH="${HOST_REPO_PATH:-${REPO_ROOT}}"
HOST_DATA_PATH="${HOST_DATA_PATH:-/home/reranker_experiment/data/latency_delay}"
HOST_MODEL_PATH="${HOST_MODEL_PATH:-/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8SC}"
DATASET="${DATASET:-0625caption}"
HOST_OUTPUT_PATH="${HOST_OUTPUT_PATH:-/home/reranker_experiment/output/qwen3_w8a8sc_atb_${DATASET}}"
HOST_PYTHON_CACHE="${HOST_PYTHON_CACHE:-/home/reranker_experiment/deps/atb-eval-python}"
IMAGE="${IMAGE:-swr.cn-south-1.myhuaweicloud.com/ascendhub/mindie:2.1.RC1-300I-Duo-py311-openeuler24.03-lts}"
PULL_IMAGE="${PULL_IMAGE:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirror.nju.edu.cn/pypi/web/simple/}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MAX_QUERIES="${MAX_QUERIES:-0}"
MASTER_PORT="${MASTER_PORT:-20039}"
SHM_SIZE="${SHM_SIZE:-16g}"
EXPECTED_FBETA_BETA="${EXPECTED_FBETA_BETA:-0.3}"
TOP_K_LIST="${TOP_K_LIST:-1 3 5 10}"
SORT_BY_LENGTH="${SORT_BY_LENGTH:-1}"
SORT_DESCENDING="${SORT_DESCENDING:-0}"
SAVE_DOC_TEXT="${SAVE_DOC_TEXT:-0}"
INSTRUCTION="${INSTRUCTION:-Given a user query, retrieve relevant documents that answer the query.}"

case "${DATASET}" in
  0428caption)
    dataset_dir="${HOST_DATA_PATH}/0428caption"
    recall_file="${dataset_dir}/retrieve_id_caption_0416.json"
    gt_doc_id_col="PageId_new"
    gt_hint=""
    ;;
  0428keyword)
    dataset_dir="${HOST_DATA_PATH}/0428keyword"
    recall_file="${dataset_dir}/id_keywords_pair_new.json"
    gt_doc_id_col="PageId_new"
    gt_hint=""
    ;;
  0625caption)
    dataset_dir="${HOST_DATA_PATH}/0625caption"
    recall_file="${dataset_dir}/0625_raw_recall_result.json"
    gt_doc_id_col="PageId"
    gt_hint="gtfile-20260617.xlsx"
    ;;
  *)
    echo "[invalid] DATASET=${DATASET}; use 0428caption, 0428keyword, or 0625caption" >&2
    exit 2
    ;;
esac

for toggle in PULL_IMAGE SORT_BY_LENGTH SORT_DESCENDING SAVE_DOC_TEXT; do
  value="${!toggle}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "[invalid] ${toggle} must be 0 or 1; got ${value}" >&2
    exit 2
  fi
done
for number in TP_SIZE MAX_LENGTH BATCH_SIZE MASTER_PORT; do
  value="${!number}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "[invalid] ${number} must be a positive integer; got ${value}" >&2
    exit 2
  fi
done
if [[ ! "${MAX_QUERIES}" =~ ^[0-9]+$ ]]; then
  echo "[invalid] MAX_QUERIES must be a non-negative integer; got ${MAX_QUERIES}" >&2
  exit 2
fi
if ! command -v docker >/dev/null 2>&1; then
  echo "[missing] docker is not installed" >&2
  exit 3
fi
if ! command -v npu-smi >/dev/null 2>&1 || ! npu-smi info >/dev/null; then
  echo "[missing] npu-smi cannot access the host NPU/driver" >&2
  exit 3
fi
if [[ ! -f "${HOST_MODEL_PATH}/config.json" ]]; then
  echo "[missing] W8A8SC model config: ${HOST_MODEL_PATH}/config.json" >&2
  exit 3
fi
if [[ ! -d "${dataset_dir}" ]]; then
  echo "[missing] dataset directory: ${dataset_dir}" >&2
  exit 3
fi
if [[ ! -f "${recall_file}" ]]; then
  echo "[missing] recall file: ${recall_file}" >&2
  exit 3
fi
if [[ -n "${gt_hint}" && -f "${dataset_dir}/${gt_hint}" ]]; then
  gt_file="${dataset_dir}/${gt_hint}"
else
  gt_files=()
  while IFS= read -r file; do
    gt_files+=("${file}")
  done < <(find "${dataset_dir}" -maxdepth 1 -type f -name "*.xlsx" | sort)
  if [[ "${#gt_files[@]}" -eq 0 ]]; then
    echo "[missing] no ground-truth .xlsx under ${dataset_dir}" >&2
    exit 3
  fi
  gt_file="${gt_files[0]}"
fi

checker="${HOST_REPO_PATH}/scripts/check_qwen3_reranker_w8a8sc_310p.py"
evaluator="${HOST_REPO_PATH}/scripts/business_eval_atb.py"
for required_file in "${checker}" "${evaluator}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[missing] ${required_file}" >&2
    exit 3
  fi
done
if command -v python3 >/dev/null 2>&1; then
  host_python=python3
elif command -v python >/dev/null 2>&1; then
  host_python=python
else
  echo "[missing] python3 or python is required for model validation" >&2
  exit 3
fi
"${host_python}" "${checker}" \
  --model-path "${HOST_MODEL_PATH}" \
  --expected w8a8sc \
  --expected-parts "${TP_SIZE}"

mkdir -p "${HOST_OUTPUT_PATH}" "${HOST_PYTHON_CACHE}"
if [[ "${PULL_IMAGE}" == "1" ]]; then
  docker pull "${IMAGE}"
elif ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[missing] local image ${IMAGE}; rerun with PULL_IMAGE=1" >&2
  exit 3
fi

NPU_SMI_PATH="$(command -v npu-smi)"
mount_args=(
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro
  -v "${NPU_SMI_PATH}:${NPU_SMI_PATH}:ro"
)
[[ -d /usr/local/Ascend/firmware ]] && mount_args+=(
  -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro
)
[[ -d /usr/local/dcmi ]] && mount_args+=(
  -v /usr/local/dcmi:/usr/local/dcmi:ro
)

echo "[config] dataset=${DATASET}"
echo "[config] gt=${gt_file}"
echo "[config] recall=${recall_file}"
echo "[config] model=${HOST_MODEL_PATH}"
echo "[config] output=${HOST_OUTPUT_PATH}"
echo "[config] max_queries=${MAX_QUERIES} batch=${BATCH_SIZE} max_length=${MAX_LENGTH}"

docker run --rm \
  --network host \
  --shm-size="${SHM_SIZE}" \
  --privileged=true \
  --user root \
  --entrypoint /bin/bash \
  -e ASCEND_RUNTIME_OPTIONS=NODRV \
  -e "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}" \
  -e "TP_SIZE=${TP_SIZE}" \
  -e "MAX_LENGTH=${MAX_LENGTH}" \
  -e "BATCH_SIZE=${BATCH_SIZE}" \
  -e "MAX_QUERIES=${MAX_QUERIES}" \
  -e "MASTER_PORT=${MASTER_PORT}" \
  -e "EXPECTED_FBETA_BETA=${EXPECTED_FBETA_BETA}" \
  -e "TOP_K_LIST=${TOP_K_LIST}" \
  -e "SORT_BY_LENGTH=${SORT_BY_LENGTH}" \
  -e "SORT_DESCENDING=${SORT_DESCENDING}" \
  -e "SAVE_DOC_TEXT=${SAVE_DOC_TEXT}" \
  -e "INSTRUCTION=${INSTRUCTION}" \
  -e "GT_DOC_ID_COL=${gt_doc_id_col}" \
  -e "PIP_INDEX_URL=${PIP_INDEX_URL}" \
  "${mount_args[@]}" \
  -v "${HOST_REPO_PATH}:/workspace/memranker:ro" \
  -v "${HOST_MODEL_PATH}:/models/w8a8sc:ro" \
  -v "${dataset_dir}:/dataset:ro" \
  -v "${gt_file}:/inputs/ground_truth.xlsx:ro" \
  -v "${recall_file}:/inputs/recall.json:ro" \
  -v "${HOST_OUTPUT_PATH}:/outputs:rw" \
  -v "${HOST_PYTHON_CACHE}:/cache/python:rw" \
  "${IMAGE}" -lc '
    set -euo pipefail
    source_if_present() {
      local path="$1"
      if [[ -f "${path}" ]]; then
        set +u
        source "${path}"
        set -u
      fi
    }
    source_if_present /usr/local/Ascend/ascend-toolkit/set_env.sh
    source_if_present /usr/local/Ascend/cann/set_env.sh
    source_if_present /usr/local/Ascend/nnal/atb/set_env.sh

    atb_root="${ATB_SPEED_HOME_PATH:-}"
    if [[ -z "${atb_root}" ]]; then
      for candidate in \
        /usr/local/Ascend/atb-models \
        /usr/local/Ascend/mindie/latest/mindie-llm/atb-models \
        /usr/local/Ascend/mindie/latest/atb-models; do
        if [[ -f "${candidate}/examples/run_pa.py" ]]; then
          atb_root="${candidate}"
          break
        fi
      done
    fi
    if [[ -n "${atb_root}" ]]; then
      source_if_present "${atb_root}/set_env.sh"
    fi
    if [[ -z "${atb_root}" || ! -f "${atb_root}/examples/run_pa.py" ]]; then
      echo "[missing] examples.run_pa in the image" >&2
      exit 4
    fi

    export PYTHONPATH="/cache/python:/workspace/memranker:${PYTHONPATH:-}"
    if ! python3 -c "import openpyxl, tqdm" >/dev/null 2>&1; then
      python3 -m pip install \
        --disable-pip-version-check \
        --no-cache-dir \
        --target /cache/python \
        --index-url "${PIP_INDEX_URL}" \
        openpyxl tqdm
    fi

    sort_args=()
    if [[ "${SORT_BY_LENGTH}" == "1" ]]; then
      sort_args+=(--sort-by-length)
    else
      sort_args+=(--no-sort-by-length)
    fi
    if [[ "${SORT_DESCENDING}" == "1" ]]; then
      sort_args+=(--sort-descending)
    fi
    if [[ "${SAVE_DOC_TEXT}" == "1" ]]; then
      sort_args+=(--save-doc-text)
    fi
    read -r -a top_k_args <<< "${TOP_K_LIST}"

    export MINDIE_LOG_TO_STDOUT=1
    cd "${atb_root}"
    python3 -m torch.distributed.run \
      --nproc_per_node "${TP_SIZE}" \
      --master_port "${MASTER_PORT}" \
      /workspace/memranker/scripts/business_eval_atb.py \
      --model-root /models/w8a8sc \
      --gt-file /inputs/ground_truth.xlsx \
      --recall-file /inputs/recall.json \
      --output-dir /outputs \
      --instruction "${INSTRUCTION}" \
      --gt-doc-id-col "${GT_DOC_ID_COL}" \
      --max-length "${MAX_LENGTH}" \
      --batch-size "${BATCH_SIZE}" \
      --max-queries "${MAX_QUERIES}" \
      --expected-fbeta-beta "${EXPECTED_FBETA_BETA}" \
      --top-k-list "${top_k_args[@]}" \
      "${sort_args[@]}"
  '

echo "[done] ATB W8A8SC evaluation: ${HOST_OUTPUT_PATH}"
echo "[done] metrics: ${HOST_OUTPUT_PATH}/metrics.json"
