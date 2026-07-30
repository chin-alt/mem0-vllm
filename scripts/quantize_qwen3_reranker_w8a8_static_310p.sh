#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Prepare calibration data and export experimental static W8A8 weights for the
Qwen3-Reranker-0.6B / Ascend 310P vLLM-Ascend 0.10.2rc1 path.

The host driver is never installed, upgraded, or modified by this script.

Important environment overrides:
  TRAIN_JSONL              Source reranker JSONL.
                           Default: /home/reranker_experiment/data/split/train.jsonl
  FLOAT_MODEL_PATH         Original Qwen3-Reranker-0.6B directory.
  QUANT_MODEL_PATH         New static-W8A8 model directory.
  CALIB_JSONL              Generated ModelSlim calibration JSONL.
  MAX_LENGTH               Calibration/deployment sequence length. Default: 1024
  CALIB_SAMPLES            Length-stratified sample count. Default: 64
  CALIB_BACKEND            pooling or generate. Default: pooling
  PRODUCTION_INSTRUCTION   Optional instruction override for every calibration row.
  MODELSLIM_DIR            Pinned msit checkout directory.
  MODELSLIM_VENV           Dedicated Python venv directory.
  INSTALL_MODELSLIM        Clone/install the pinned ModelSlim branch if needed. Default: 1
  REINSTALL_MODELSLIM      Re-run ModelSlim installation. Default: 0
  PIP_INDEX_URL            Python package index used by install.sh and pip.
                           Default: https://pypi.tuna.tsinghua.edu.cn/simple
  QUANTIZE_DOWN_PROJ       Also quantize down_proj instead of conservative FP fallback. Default: 0
  RUN_BENCHMARK            Run FP16 and W8A8 vLLM-Ascend A/B after export. Default: 0
  BENCHMARK_DATA_PATH      Business benchmark root used when RUN_BENCHMARK=1.
  BENCHMARK_DATASET        0428caption, 0428keyword, or 0625caption. Default: 0428caption
  BENCHMARK_BACKENDS       Backends to benchmark. Default: pooling
  BENCHMARK_BATCH_SIZES    Batch-size sweep. Default: 1 8 16 32
  BENCHMARK_MAX_LENGTHS    Length sweep. Default: value of MAX_LENGTH
  PULL_IMAGE               Pull the vLLM-Ascend image for benchmark. Default: 0

The output directory must be absent or empty. This script does not delete or
overwrite an existing quantized model.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_JSONL="${TRAIN_JSONL:-/home/reranker_experiment/data/split/train.jsonl}"
FLOAT_MODEL_PATH="${FLOAT_MODEL_PATH:-/home/reranker_experiment/model/Qwen3-Reranker-0.6B}"
QUANT_MODEL_PATH="${QUANT_MODEL_PATH:-/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8-static-safe}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
CALIB_SAMPLES="${CALIB_SAMPLES:-64}"
CALIB_LENGTH_BINS="${CALIB_LENGTH_BINS:-4}"
CALIB_SEED="${CALIB_SEED:-20260730}"
CALIB_BACKEND="${CALIB_BACKEND:-pooling}"
CALIB_JSONL="${CALIB_JSONL:-/home/reranker_experiment/data/calibration/qwen3_reranker_w8a8_static_${CALIB_BACKEND}_len${MAX_LENGTH}_n${CALIB_SAMPLES}.jsonl}"
PRODUCTION_INSTRUCTION="${PRODUCTION_INSTRUCTION:-}"
ALLOW_MULTIPLE_INSTRUCTIONS="${ALLOW_MULTIPLE_INSTRUCTIONS:-0}"
MODELSLIM_BRANCH="${MODELSLIM_BRANCH:-modelslim-VLLM-8.1.RC1.b020_001}"
MODELSLIM_DIR="${MODELSLIM_DIR:-/home/reranker_experiment/deps/msit-modelslim-vllm-8.1}"
MODELSLIM_VENV="${MODELSLIM_VENV:-/home/reranker_experiment/venvs/modelslim-vllm-8.1}"
INSTALL_MODELSLIM="${INSTALL_MODELSLIM:-1}"
REINSTALL_MODELSLIM="${REINSTALL_MODELSLIM:-0}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://pypi.tuna.tsinghua.edu.cn/simple}"
QUANTIZE_DOWN_PROJ="${QUANTIZE_DOWN_PROJ:-0}"
RUN_BENCHMARK="${RUN_BENCHMARK:-0}"
BENCHMARK_DATA_PATH="${BENCHMARK_DATA_PATH:-/home/reranker_experiment/data/latency_delay}"
BENCHMARK_DATASET="${BENCHMARK_DATASET:-0428caption}"
BENCHMARK_BACKENDS="${BENCHMARK_BACKENDS:-pooling}"
BENCHMARK_BATCH_SIZES="${BENCHMARK_BATCH_SIZES:-1 8 16 32}"
BENCHMARK_MAX_LENGTHS="${BENCHMARK_MAX_LENGTHS:-${MAX_LENGTH}}"
BENCHMARK_OUTPUT_BASE="${BENCHMARK_OUTPUT_BASE:-/home/reranker_experiment/output/qwen3_reranker_310p_static_w8a8_ab}"
PULL_IMAGE="${PULL_IMAGE:-0}"

if [[ "${CALIB_BACKEND}" != "pooling" && "${CALIB_BACKEND}" != "generate" ]]; then
  echo "[invalid] CALIB_BACKEND must be pooling or generate" >&2
  exit 2
fi
for flag in INSTALL_MODELSLIM REINSTALL_MODELSLIM QUANTIZE_DOWN_PROJ RUN_BENCHMARK ALLOW_MULTIPLE_INSTRUCTIONS PULL_IMAGE; do
  value="${!flag}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "[invalid] ${flag} must be 0 or 1" >&2
    exit 2
  fi
done
if [[ ! -f "${TRAIN_JSONL}" ]]; then
  echo "[missing] training JSONL: ${TRAIN_JSONL}" >&2
  exit 3
fi
if [[ ! -f "${FLOAT_MODEL_PATH}/config.json" ]]; then
  echo "[missing] float model config: ${FLOAT_MODEL_PATH}/config.json" >&2
  exit 3
fi
existing_quant_file=""
if [[ -d "${QUANT_MODEL_PATH}" ]]; then
  existing_quant_file="$(find "${QUANT_MODEL_PATH}" -mindepth 1 -print -quit)"
fi
if [[ -n "${existing_quant_file}" ]]; then
  echo "[exists] quantized output is not empty: ${QUANT_MODEL_PATH}" >&2
  echo "[exists] choose a new QUANT_MODEL_PATH; this script never deletes model output" >&2
  exit 3
fi

if ! "${PYTHON_BOOTSTRAP}" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 12) else 1)'; then
  echo "[invalid] ModelSlim workflow requires Python 3.9-3.11: $("${PYTHON_BOOTSTRAP}" --version 2>&1)" >&2
  exit 4
fi

if [[ ! -d "${MODELSLIM_DIR}/.git" ]]; then
  if [[ "${INSTALL_MODELSLIM}" != "1" ]]; then
    echo "[missing] ModelSlim checkout: ${MODELSLIM_DIR}" >&2
    echo "[missing] set INSTALL_MODELSLIM=1 or provide MODELSLIM_DIR" >&2
    exit 4
  fi
  mkdir -p "$(dirname "${MODELSLIM_DIR}")"
  git clone --depth 1 --branch "${MODELSLIM_BRANCH}" \
    https://gitee.com/ascend/msit.git "${MODELSLIM_DIR}"
fi

modelslim_commit="$(git -C "${MODELSLIM_DIR}" rev-parse HEAD)"
expected_modelslim_commit=""
for candidate_ref in \
  "${MODELSLIM_BRANCH}^{commit}" \
  "refs/tags/${MODELSLIM_BRANCH}^{commit}" \
  "refs/heads/${MODELSLIM_BRANCH}^{commit}" \
  "refs/remotes/origin/${MODELSLIM_BRANCH}^{commit}"; do
  if expected_modelslim_commit="$(
    git -C "${MODELSLIM_DIR}" rev-parse --verify "${candidate_ref}" 2>/dev/null
  )"; then
    break
  fi
  expected_modelslim_commit=""
done
if [[ -z "${expected_modelslim_commit}" ]]; then
  echo "[invalid] MODELSLIM_DIR does not contain expected ref ${MODELSLIM_BRANCH}" >&2
  exit 4
fi
if [[ "${modelslim_commit}" != "${expected_modelslim_commit}" ]]; then
  echo "[invalid] MODELSLIM_DIR HEAD=${modelslim_commit}" >&2
  echo "[invalid] expected ${MODELSLIM_BRANCH}=${expected_modelslim_commit}" >&2
  exit 4
fi
modelslim_utils="${MODELSLIM_DIR}/msmodelslim/example/common/utils.py"
if [[ ! -f "${modelslim_utils}" ]] || ! grep -q "inputs_pretokenized" "${modelslim_utils}"; then
  echo "[invalid] ModelSlim checkout does not have the expected inputs_pretokenized loader" >&2
  exit 4
fi
echo "[modelslim] ref=${MODELSLIM_BRANCH} commit=${modelslim_commit}"

if [[ ! -x "${MODELSLIM_VENV}/bin/python" ]]; then
  if [[ "${INSTALL_MODELSLIM}" != "1" ]]; then
    echo "[missing] ModelSlim venv: ${MODELSLIM_VENV}" >&2
    exit 4
  fi
  mkdir -p "$(dirname "${MODELSLIM_VENV}")"
  "${PYTHON_BOOTSTRAP}" -m venv --system-site-packages "${MODELSLIM_VENV}"
fi

MODELSLIM_PYTHON="${MODELSLIM_VENV}/bin/python"
install_stamp="${MODELSLIM_VENV}/.memranker_${MODELSLIM_BRANCH}_installed"
if [[ "${REINSTALL_MODELSLIM}" == "1" || ! -f "${install_stamp}" ]]; then
  if [[ "${INSTALL_MODELSLIM}" != "1" ]]; then
    echo "[missing] ModelSlim install stamp: ${install_stamp}" >&2
    exit 4
  fi
  # Activation makes install.sh use this dedicated environment while retaining
  # host torch/torch_npu through --system-site-packages. PIP_INDEX_URL is
  # exported so pip calls made inside install.sh use the same mirror.
  # shellcheck disable=SC1091
  source "${MODELSLIM_VENV}/bin/activate"
  export PIP_INDEX_URL
  echo "[modelslim] pip_index_url=${PIP_INDEX_URL}"
  pushd "${MODELSLIM_DIR}/msmodelslim" >/dev/null
  bash install.sh
  popd >/dev/null
  "${MODELSLIM_PYTHON}" -m pip install \
    --index-url "${PIP_INDEX_URL}" \
    transformers==4.55.2 tokenizers==0.21.4 accelerate safetensors
  "${MODELSLIM_PYTHON}" -c 'import msmodelslim, transformers; print("[modelslim] transformers=" + transformers.__version__)'
  printf '%s\n' "${MODELSLIM_BRANCH}" > "${install_stamp}"
fi

prepare_args=(
  --input "${TRAIN_JSONL}"
  --output "${CALIB_JSONL}"
  --model-path "${FLOAT_MODEL_PATH}"
  --backend "${CALIB_BACKEND}"
  --max-length "${MAX_LENGTH}"
  --sample-count "${CALIB_SAMPLES}"
  --length-bins "${CALIB_LENGTH_BINS}"
  --seed "${CALIB_SEED}"
  --overwrite
)
if [[ -n "${PRODUCTION_INSTRUCTION}" ]]; then
  prepare_args+=(--instruction "${PRODUCTION_INSTRUCTION}")
elif [[ "${ALLOW_MULTIPLE_INSTRUCTIONS}" != "1" ]]; then
  prepare_args+=(--require-single-instruction)
fi

echo "[step 1/3] preparing ModelSlim calibration data"
"${MODELSLIM_PYTHON}" "${REPO_ROOT}/scripts/prepare_qwen3_reranker_calibration.py" \
  "${prepare_args[@]}"

quant_script="${MODELSLIM_DIR}/msmodelslim/example/Qwen/quant_qwen.py"
if [[ ! -f "${quant_script}" ]]; then
  echo "[missing] pinned ModelSlim Qwen script: ${quant_script}" >&2
  exit 4
fi
mkdir -p "$(dirname "${QUANT_MODEL_PATH}")"
quant_args=(
  --model_path "${FLOAT_MODEL_PATH}"
  --save_directory "${QUANT_MODEL_PATH}"
  --calib_file "${CALIB_JSONL}"
  --w_bit 8
  --a_bit 8
  --device_type cpu
  --act_method 1
  --anti_method m2
  --disable_level L0
  --model_type qwen2.5
  --is_dynamic False
  --use_kvcache_quant False
  --use_fa_quant False
  --trust_remote_code True
)
if [[ "${QUANTIZE_DOWN_PROJ}" == "1" ]]; then
  quant_args+=(--disable_names lm_head)
fi

echo "[step 2/3] exporting static W8A8 weights (CPU calibration)"
pushd "$(dirname "${quant_script}")" >/dev/null
"${MODELSLIM_PYTHON}" "${quant_script}" "${quant_args[@]}"
popd >/dev/null

"${MODELSLIM_PYTHON}" - \
  "${QUANT_MODEL_PATH}" \
  "${TRAIN_JSONL}" \
  "${CALIB_JSONL}" \
  "${MODELSLIM_BRANCH}" \
  "${modelslim_commit}" \
  "${MAX_LENGTH}" \
  "${CALIB_SAMPLES}" \
  "${CALIB_BACKEND}" \
  "${QUANTIZE_DOWN_PROJ}" <<'PY'
import json
import sys
from pathlib import Path

(
    model_path,
    train_jsonl,
    calib_jsonl,
    modelslim_branch,
    modelslim_commit,
    max_length,
    calib_samples,
    calib_backend,
    quantize_down_proj,
) = sys.argv[1:]
manifest = {
    "source_train_jsonl": train_jsonl,
    "calibration_jsonl": calib_jsonl,
    "modelslim_branch": modelslim_branch,
    "modelslim_commit": modelslim_commit,
    "quantization": "static_w8a8",
    "is_dynamic": False,
    "max_length": int(max_length),
    "calibration_samples": int(calib_samples),
    "calibration_backend": calib_backend,
    "quantize_down_proj": quantize_down_proj == "1",
    "lm_head_float": True,
    "kv_cache_quantization": False,
    "attention_quantization": False,
}
path = Path(model_path) / "memranker_quantization_workflow.json"
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("[quantization] workflow manifest:", path)
PY

echo "[step 3/3] checking exported quantization description"
"${MODELSLIM_PYTHON}" "${REPO_ROOT}/scripts/check_qwen3_reranker_w8a8_310p.py" \
  --model-path "${QUANT_MODEL_PATH}" \
  --skip-runtime

echo "[done] static W8A8 model: ${QUANT_MODEL_PATH}"
echo "[done] calibration manifest: ${CALIB_JSONL}.manifest.json"

if [[ "${RUN_BENCHMARK}" == "1" ]]; then
  if [[ ! -d "${BENCHMARK_DATA_PATH}" ]]; then
    echo "[missing] benchmark data root: ${BENCHMARK_DATA_PATH}" >&2
    exit 5
  fi
  if [[ -z "${PRODUCTION_INSTRUCTION}" ]]; then
    PRODUCTION_INSTRUCTION="$("${MODELSLIM_PYTHON}" - "${TRAIN_JSONL}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8-sig") as handle:
    for line in handle:
        if not line.strip():
            continue
        value = str(json.loads(line).get("instruction", "")).strip()
        if value:
            print(value)
            break
PY
)"
  fi
  if [[ -z "${PRODUCTION_INSTRUCTION}" ]]; then
    echo "[invalid] could not determine the benchmark instruction" >&2
    exit 5
  fi

  echo "[benchmark] FP16 baseline"
  HOST_REPO_PATH="${REPO_ROOT}" \
  HOST_DATA_PATH="${BENCHMARK_DATA_PATH}" \
  HOST_MODEL_PATH="${FLOAT_MODEL_PATH}" \
  HOST_OUTPUT_BASE="${BENCHMARK_OUTPUT_BASE}/fp16" \
  MODEL_LABEL=fp16 \
  INSTRUCTION="${PRODUCTION_INSTRUCTION}" \
  BACKENDS="${BENCHMARK_BACKENDS}" \
  BATCH_SIZES="${BENCHMARK_BATCH_SIZES}" \
  MAX_LENGTHS="${BENCHMARK_MAX_LENGTHS}" \
  DATASET="${BENCHMARK_DATASET}" \
  PULL_IMAGE="${PULL_IMAGE}" \
  VLLM_QUANTIZATION= \
  bash "${REPO_ROOT}/scripts/benchmark_qwen3_reranker_310p.sh"

  echo "[benchmark] static W8A8 candidate"
  HOST_REPO_PATH="${REPO_ROOT}" \
  HOST_DATA_PATH="${BENCHMARK_DATA_PATH}" \
  HOST_MODEL_PATH="${QUANT_MODEL_PATH}" \
  HOST_OUTPUT_BASE="${BENCHMARK_OUTPUT_BASE}/w8a8" \
  MODEL_LABEL=w8a8 \
  INSTRUCTION="${PRODUCTION_INSTRUCTION}" \
  BACKENDS="${BENCHMARK_BACKENDS}" \
  BATCH_SIZES="${BENCHMARK_BATCH_SIZES}" \
  MAX_LENGTHS="${BENCHMARK_MAX_LENGTHS}" \
  DATASET="${BENCHMARK_DATASET}" \
  PULL_IMAGE=0 \
  VLLM_QUANTIZATION=ascend \
  bash "${REPO_ROOT}/scripts/benchmark_qwen3_reranker_310p.sh"

  echo "[done] FP16/W8A8 benchmark roots: ${BENCHMARK_OUTPUT_BASE}/{fp16,w8a8}"
else
  cat <<EOF
[next] run the FP16 baseline with scripts/benchmark_qwen3_reranker_310p.sh,
then run the same command with:
  HOST_MODEL_PATH='${QUANT_MODEL_PATH}'
  MODEL_LABEL=w8a8
  VLLM_QUANTIZATION=ascend
EOF
fi
