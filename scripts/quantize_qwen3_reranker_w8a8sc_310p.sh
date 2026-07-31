#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Generate a Qwen3-Reranker W8A8SC model inside a 310P MindIE/ATB container.

This is the container-side workflow. Unlike the legacy dense-W8A8 exporter,
W8A8S calibration runs on the NPU and is followed by the official ATB Models
sparse_compressor. The host driver is only mounted and is never modified.

Required/important environment variables:
  TRAIN_JSONL              Reranker calibration source JSONL.
  FLOAT_MODEL_PATH         Original merged Qwen3-Reranker model.
  W8A8S_MODEL_PATH         Empty first-stage W8A8S output directory.
  W8A8SC_MODEL_PATH        Empty final W8A8SC output directory.
  CALIB_JSONL              Generated ModelSlim calibration JSONL.
  MAX_LENGTH               Calibration/deployment input length. Default: 1024
  CALIB_SAMPLES            Length-stratified calibration rows. Default: 64
  CALIB_BACKEND            generate or pooling. Default: generate
  FRACTION                 Protected outlier fraction. Default: 0.011
  SIGMA_FACTOR             Sparse-quant sigma factor. Default: 4.0
  TP_SIZE                  Compression/runtime tensor parallel size. Default: 1
  MULTIPROCESS_NUM         CPU compression workers. Default: 4
  RUN_ATB_SMOKE            Run one-token examples.run_pa after compression. Default: 1
  PIP_INDEX_URL            Default: https://mirror.nju.edu.cn/pypi/web/simple/

Both output directories must be absent or empty. The script never overwrites
an existing model.
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
TRAIN_JSONL="${TRAIN_JSONL:-/inputs/train.jsonl}"
FLOAT_MODEL_PATH="${FLOAT_MODEL_PATH:-/models/float}"
W8A8S_MODEL_PATH="${W8A8S_MODEL_PATH:-/models/w8a8s}"
W8A8SC_MODEL_PATH="${W8A8SC_MODEL_PATH:-/models/w8a8sc}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
CALIB_SAMPLES="${CALIB_SAMPLES:-64}"
CALIB_LENGTH_BINS="${CALIB_LENGTH_BINS:-4}"
CALIB_SEED="${CALIB_SEED:-20260731}"
CALIB_BACKEND="${CALIB_BACKEND:-generate}"
CALIB_JSONL="${CALIB_JSONL:-/calibration/qwen3_reranker_w8a8s_generate_len${MAX_LENGTH}_n${CALIB_SAMPLES}.jsonl}"
PRODUCTION_INSTRUCTION="${PRODUCTION_INSTRUCTION:-}"
ALLOW_MULTIPLE_INSTRUCTIONS="${ALLOW_MULTIPLE_INSTRUCTIONS:-0}"
FRACTION="${FRACTION:-0.011}"
SIGMA_FACTOR="${SIGMA_FACTOR:-4.0}"
TP_SIZE="${TP_SIZE:-1}"
MULTIPROCESS_NUM="${MULTIPROCESS_NUM:-4}"
MASTER_PORT="${MASTER_PORT:-20037}"
RUN_ATB_SMOKE="${RUN_ATB_SMOKE:-1}"
MAX_PREFILL_TOKENS="${MAX_PREFILL_TOKENS:-${MAX_LENGTH}}"
MODELSLIM_BRANCH="${MODELSLIM_BRANCH:-modelslim-VLLM-8.1.RC1.b020_001}"
MODELSLIM_COMMIT="${MODELSLIM_COMMIT:-618633f1efbbcc41eaaeabbdfc624d2fe7264d8d}"
MODELSLIM_DIR="${MODELSLIM_DIR:-/cache/deps/msit-modelslim-vllm-8.1}"
MODELSLIM_VENV="${MODELSLIM_VENV:-/cache/venvs/modelslim-w8a8sc-mindie21}"
REINSTALL_MODELSLIM="${REINSTALL_MODELSLIM:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirror.nju.edu.cn/pypi/web/simple/}"
PYTHON_BOOTSTRAP="${PYTHON_BOOTSTRAP:-python3}"
ATB_SPEED_HOME_PATH="${ATB_SPEED_HOME_PATH:-}"

for flag in ALLOW_MULTIPLE_INSTRUCTIONS REINSTALL_MODELSLIM RUN_ATB_SMOKE; do
  value="${!flag}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "[invalid] ${flag} must be 0 or 1; got ${value}" >&2
    exit 2
  fi
done
for number in MAX_LENGTH CALIB_SAMPLES CALIB_LENGTH_BINS TP_SIZE MULTIPROCESS_NUM MAX_PREFILL_TOKENS; do
  value="${!number}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "[invalid] ${number} must be a positive integer; got ${value}" >&2
    exit 2
  fi
done
if [[ "${CALIB_BACKEND}" != "generate" && "${CALIB_BACKEND}" != "pooling" ]]; then
  echo "[invalid] CALIB_BACKEND must be generate or pooling" >&2
  exit 2
fi
if (( MAX_PREFILL_TOKENS < MAX_LENGTH )); then
  echo "[invalid] MAX_PREFILL_TOKENS must be at least MAX_LENGTH" >&2
  exit 2
fi
for file in "${TRAIN_JSONL}" "${FLOAT_MODEL_PATH}/config.json"; do
  if [[ ! -f "${file}" ]]; then
    echo "[missing] ${file}" >&2
    exit 3
  fi
done
for output_dir in "${W8A8S_MODEL_PATH}" "${W8A8SC_MODEL_PATH}"; do
  existing=""
  if [[ -d "${output_dir}" ]]; then
    existing="$(find "${output_dir}" -mindepth 1 -print -quit)"
  fi
  if [[ -n "${existing}" ]]; then
    echo "[exists] output is not empty: ${output_dir}" >&2
    exit 3
  fi
  mkdir -p "${output_dir}"
done

source_if_present() {
  local path="$1"
  if [[ -f "${path}" ]]; then
    local restore_nounset=0
    if [[ $- == *u* ]]; then
      restore_nounset=1
      set +u
    fi
    # shellcheck disable=SC1090
    source "${path}"
    if [[ "${restore_nounset}" == "1" ]]; then
      set -u
    fi
  fi
}

source_if_present /usr/local/Ascend/ascend-toolkit/set_env.sh
source_if_present /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
source_if_present /usr/local/Ascend/cann/set_env.sh
source_if_present /usr/local/Ascend/nnal/atb/set_env.sh

if [[ -z "${ATB_SPEED_HOME_PATH}" ]]; then
  for candidate in \
    /usr/local/Ascend/atb-models \
    /usr/local/Ascend/mindie/latest/mindie-llm/atb-models \
    /usr/local/Ascend/mindie/latest/atb-models; do
    if [[ -f "${candidate}/examples/run_pa.py" ]]; then
      ATB_SPEED_HOME_PATH="${candidate}"
      break
    fi
  done
fi
if [[ -n "${ATB_SPEED_HOME_PATH}" ]]; then
  source_if_present "${ATB_SPEED_HOME_PATH}/set_env.sh"
fi
if [[ -z "${ATB_SPEED_HOME_PATH}" || ! -f "${ATB_SPEED_HOME_PATH}/examples/run_pa.py" ]]; then
  echo "[missing] ATB Models root; ATB_SPEED_HOME_PATH=${ATB_SPEED_HOME_PATH:-<unset>}" >&2
  exit 4
fi
SPARSE_COMPRESSOR="${ATB_SPEED_HOME_PATH}/examples/convert/model_slim/sparse_compressor.py"
if [[ ! -f "${SPARSE_COMPRESSOR}" ]]; then
  echo "[missing] official ATB sparse compressor: ${SPARSE_COMPRESSOR}" >&2
  exit 4
fi
if ! grep -R -q -i --include='*.py' 'qwen3' \
  "${ATB_SPEED_HOME_PATH}/atb_llm" "${ATB_SPEED_HOME_PATH}/examples" 2>/dev/null; then
  echo "[unsupported] this ATB Models image does not register Qwen3" >&2
  echo "[unsupported] refusing to spend time on calibration before runtime compatibility is known" >&2
  exit 4
fi

if ! "${PYTHON_BOOTSTRAP}" - <<'PY'
import sys
import torch
import torch_npu

assert (3, 9) <= sys.version_info[:2] < (3, 12), sys.version
assert torch.npu.is_available(), "torch.npu.is_available() is false"
print("[runtime] python=" + sys.version.split()[0])
print("[runtime] torch=" + torch.__version__)
print("[runtime] torch_npu=" + torch_npu.__version__)
print("[runtime] npu=" + torch.npu.get_device_name(0))
PY
then
  echo "[missing] usable torch_npu/310P runtime inside the image" >&2
  exit 4
fi

visible_npu_count="$(
  "${PYTHON_BOOTSTRAP}" -c \
    'import torch; import torch_npu; print(torch.npu.device_count())'
)"
if (( TP_SIZE > visible_npu_count )); then
  echo "[invalid] TP_SIZE=${TP_SIZE}, but only ${visible_npu_count} NPU device(s) are visible" >&2
  exit 4
fi
echo "[atb] root=${ATB_SPEED_HOME_PATH}"
echo "[atb] sparse_compressor=${SPARSE_COMPRESSOR}"

modelslim_commit_stamp="${MODELSLIM_DIR}/.memranker_pinned_commit"
if [[ ! -d "${MODELSLIM_DIR}/msmodelslim" ]]; then
  echo "[missing] host-prepared ModelSlim source: ${MODELSLIM_DIR}" >&2
  echo "[missing] use scripts/quantize_qwen3_reranker_w8a8sc_310p_container.sh" >&2
  exit 4
fi
if [[ ! -f "${modelslim_commit_stamp}" ]]; then
  echo "[missing] host ModelSlim commit stamp: ${modelslim_commit_stamp}" >&2
  exit 4
fi
modelslim_commit=""
IFS= read -r modelslim_commit < "${modelslim_commit_stamp}" || true
if [[ "${modelslim_commit}" != "${MODELSLIM_COMMIT}" ]]; then
  echo "[invalid] ModelSlim checkout is not the pinned commit" >&2
  echo "[invalid] stamp=${modelslim_commit:-<empty>} expected=${MODELSLIM_COMMIT}" >&2
  exit 4
fi
echo "[modelslim] ref=${MODELSLIM_BRANCH} commit=${modelslim_commit}"

if [[ ! -x "${MODELSLIM_VENV}/bin/python" ]]; then
  mkdir -p "$(dirname "${MODELSLIM_VENV}")"
  "${PYTHON_BOOTSTRAP}" -m venv --system-site-packages "${MODELSLIM_VENV}"
fi
MODELSLIM_PYTHON="${MODELSLIM_VENV}/bin/python"
install_stamp="${MODELSLIM_VENV}/.memranker_${modelslim_commit}_npu_installed"

check_modelslim_imports() {
  "${MODELSLIM_PYTHON}" - <<'PY'
import torch
import torch_npu
import transformers
from modelslim.pytorch.weight_compression import CompressConfig, Compressor
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig

assert torch.npu.is_available()
print("[modelslim] transformers=" + transformers.__version__)
print("[modelslim] NPU PTQ and weight-compression imports are complete")
PY
}

if [[ "${REINSTALL_MODELSLIM}" == "1" || ! -f "${install_stamp}" ]] || ! check_modelslim_imports; then
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
  check_modelslim_imports
  printf '%s\n' "${modelslim_commit}" > "${install_stamp}"
fi

compress_package_dir="$("${MODELSLIM_PYTHON}" - <<'PY'
from pathlib import Path
import msmodelslim.pytorch.weight_compression.compress_utils as module
print(Path(module.__file__).resolve().parent)
PY
)"
compress_executor="${compress_package_dir}/compress_graph/build/compress_excutor"
if [[ ! -x "${compress_executor}" ]]; then
  build_script="${compress_package_dir}/compress_graph/build.sh"
  if [[ ! -f "${build_script}" || -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "[missing] compressor executable and build prerequisites" >&2
    echo "[missing] executor=${compress_executor}" >&2
    echo "[missing] build_script=${build_script} ASCEND_HOME_PATH=${ASCEND_HOME_PATH:-<unset>}" >&2
    exit 4
  fi
  echo "[modelslim] building weight compressor against ${ASCEND_HOME_PATH}"
  bash "${build_script}" "${ASCEND_HOME_PATH}"
fi
if [[ ! -x "${compress_executor}" ]]; then
  echo "[missing] ModelSlim compression executable after build: ${compress_executor}" >&2
  exit 4
fi
echo "[modelslim] compressor=${compress_executor}"

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
echo "[step 1/5] preparing Qwen3-Reranker calibration prompts"
"${MODELSLIM_PYTHON}" "${REPO_ROOT}/scripts/prepare_qwen3_reranker_calibration.py" \
  "${prepare_args[@]}"

runtime_root="$(mktemp -d /tmp/memranker-w8a8sc.XXXXXX)"
runtime_float_model="${runtime_root}/float-model"
mkdir -p "${runtime_float_model}"
cp -a "${FLOAT_MODEL_PATH}/." "${runtime_float_model}/"
"${MODELSLIM_PYTHON}" - "${runtime_float_model}/config.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
config = json.loads(path.read_text(encoding="utf-8"))
config["torch_dtype"] = "float16"
if "dtype" in config:
    config["dtype"] = "float16"
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("[model] temporary 310P config uses torch_dtype=float16:", path)
PY

quant_script="${MODELSLIM_DIR}/msmodelslim/example/Qwen/quant_qwen.py"
if [[ ! -f "${quant_script}" ]]; then
  echo "[missing] ${quant_script}" >&2
  exit 4
fi
quant_args=(
  --model_path "${runtime_float_model}"
  --save_directory "${W8A8S_MODEL_PATH}"
  --calib_file "${CALIB_JSONL}"
  --w_bit 4
  --a_bit 8
  --device_type npu
  --act_method 1
  --fraction "${FRACTION}"
  --co_sparse True
  --use_sigma True
  --is_lowbit True
  --sigma_factor "${SIGMA_FACTOR}"
  --disable_level L0
  --disable_names lm_head
  --model_type qwen2.5
  --is_dynamic False
  --use_kvcache_quant False
  --use_fa_quant False
  --trust_remote_code True
)

patch_atb_quantize_config() {
  local model_path="$1"
  local quantize="$2"
  "${MODELSLIM_PYTHON}" - "${model_path}/config.json" "${quantize}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
quantize = sys.argv[2]
config = json.loads(path.read_text(encoding="utf-8"))
config["quantize"] = quantize
config["torch_dtype"] = "float16"
if "dtype" in config:
    config["dtype"] = "float16"
path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"[model] ATB config quantize={quantize} torch_dtype=float16: {path}")
PY
}

echo "[step 2/5] exporting complete-body W8A8S on NPU"
echo "[modelslim] w_bit=4 a_bit=8 co_sparse=True is_lowbit=True fraction=${FRACTION}"
pushd "$(dirname "${quant_script}")" >/dev/null
"${MODELSLIM_PYTHON}" "${REPO_ROOT}/scripts/run_modelslim_npu.py" \
  "${quant_script}" "${quant_args[@]}"
popd >/dev/null
patch_atb_quantize_config "${W8A8S_MODEL_PATH}" "w8a8s"

echo "[step 3/5] validating W8A8S coverage before compression"
"${MODELSLIM_PYTHON}" "${REPO_ROOT}/scripts/check_qwen3_reranker_w8a8sc_310p.py" \
  --model-path "${W8A8S_MODEL_PATH}" \
  --expected w8a8s

echo "[step 4/5] splitting and compressing W8A8S -> W8A8SC with ATB Models"
export IGNORE_INFER_ERROR=1
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:False}"
compress_args=(
  --model_path "${W8A8S_MODEL_PATH}"
  --save_directory "${W8A8SC_MODEL_PATH}"
)
if grep -q -- '--multiprocess_num' "${SPARSE_COMPRESSOR}"; then
  compress_args+=(--multiprocess_num "${MULTIPROCESS_NUM}")
else
  echo "[atb] sparse_compressor has no --multiprocess_num option; using its built-in worker count"
fi
pushd "${ATB_SPEED_HOME_PATH}" >/dev/null
"${MODELSLIM_PYTHON}" -m torch.distributed.run \
  --nproc_per_node "${TP_SIZE}" \
  --master_port "${MASTER_PORT}" \
  -m examples.convert.model_slim.sparse_compressor \
  "${compress_args[@]}"
popd >/dev/null
patch_atb_quantize_config "${W8A8SC_MODEL_PATH}" "w8a8sc"

echo "[step 5/5] validating final W8A8SC model"
"${MODELSLIM_PYTHON}" "${REPO_ROOT}/scripts/check_qwen3_reranker_w8a8sc_310p.py" \
  --model-path "${W8A8SC_MODEL_PATH}" \
  --expected w8a8sc \
  --expected-parts "${TP_SIZE}"

"${MODELSLIM_PYTHON}" - \
  "${W8A8SC_MODEL_PATH}" "${TRAIN_JSONL}" "${CALIB_JSONL}" \
  "${MODELSLIM_BRANCH}" "${modelslim_commit}" "${MAX_LENGTH}" \
  "${CALIB_SAMPLES}" "${CALIB_BACKEND}" "${FRACTION}" "${SIGMA_FACTOR}" \
  "${TP_SIZE}" <<'PY'
import json
import sys
from pathlib import Path

(
    output,
    train_jsonl,
    calib_jsonl,
    branch,
    commit,
    max_length,
    samples,
    backend,
    fraction,
    sigma_factor,
    tp_size,
) = sys.argv[1:]
manifest = {
    "quantization": "w8a8sc",
    "stages": ["w8a8s_npu", "atb_sparse_compressor"],
    "source_train_jsonl": train_jsonl,
    "calibration_jsonl": calib_jsonl,
    "modelslim_branch": branch,
    "modelslim_commit": commit,
    "max_length": int(max_length),
    "calibration_samples": int(samples),
    "calibration_backend": backend,
    "w_bit_argument": 4,
    "a_bit": 8,
    "co_sparse": True,
    "is_lowbit": True,
    "fraction": float(fraction),
    "sigma_factor": float(sigma_factor),
    "anti_outlier_method": None,
    "body_projection_fallbacks": [],
    "lm_head_float": True,
    "tp_size": int(tp_size),
}
path = Path(output) / "memranker_quantization_workflow.json"
path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print("[quantization] workflow manifest:", path)
PY

if [[ "${RUN_ATB_SMOKE}" == "1" ]]; then
  smoke_input="$("${MODELSLIM_PYTHON}" - "${CALIB_JSONL}" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    print(json.loads(next(line for line in stream if line.strip()))["inputs_pretokenized"])
PY
)"
  echo "[smoke] ATB one-token Qwen3-Reranker inference"
  export MINDIE_LOG_TO_STDOUT=1
  pushd "${ATB_SPEED_HOME_PATH}" >/dev/null
  "${MODELSLIM_PYTHON}" -m torch.distributed.run \
    --nproc_per_node "${TP_SIZE}" \
    --master_port "$((MASTER_PORT + 1))" \
    "${REPO_ROOT}/scripts/run_atb_sharded_model.py" \
    --model-root "${W8A8SC_MODEL_PATH}" \
    --input_texts "${smoke_input}" \
    --max_input_length "${MAX_LENGTH}" \
    --max_prefill_tokens "${MAX_PREFILL_TOKENS}" \
    --max_output_length 1 \
    --max_batch_size 1
  popd >/dev/null
fi

echo "[done] W8A8S model: ${W8A8S_MODEL_PATH}"
echo "[done] W8A8SC model: ${W8A8SC_MODEL_PATH}"
echo "[done] original float model was not modified"
