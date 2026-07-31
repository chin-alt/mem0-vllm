#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Probe or run Qwen3-Reranker W8A8SC with the first official vLLM-Ascend
release that supports W8A8SC on 310P.

The existing ATB partN-of-M W8A8SC directory is not a vLLM checkpoint.
When CONVERT_MODEL=1, this script loads the W8A8S model with vLLM and invokes
the official examples/save_sharded_state_310.py converter to create a separate
vLLM sharded_state W8A8SC directory before evaluation.

Overrides:
  IMAGE                     Default: quay.nju.edu.cn/ascend/vllm-ascend:v0.18.0-310p
  PULL_IMAGE                0 or 1. Default: 1
  VERSION_ONLY              Only report the actual runtime and NPU versions. Default: 0
  CONVERT_MODEL             Convert W8A8S to vLLM W8A8SC. Default: 0
  RUN_EVAL                  Evaluate the converted vLLM checkpoint. Default: 1
  HOST_REPO_PATH            Default: repository root
  HOST_W8A8S_MODEL_PATH     Input ModelSlim W8A8S model for conversion
  HOST_MODEL_PATH           vLLM sharded_state W8A8SC output/input directory
  HOST_DATA_PATH            Default: /home/reranker_experiment/data/latency_delay
  HOST_OUTPUT_PATH          Business evaluation output root
  TRAIN_JSONL               Used to recover the production instruction
  DATASET                   Default: 0625caption
  MAX_LENGTH                Default: 1024
  BATCH_SIZE                Default: 4
  MAX_NUM_BATCHED_TOKENS    Default: BATCH_SIZE * MAX_LENGTH
  MAX_NUM_SEQS              Default: BATCH_SIZE
  GPU_MEMORY_UTILIZATION    Default: 0.85
  COMPRESS_PROCESS_NUM      Default: 4
  HOST_MODELSLIM_CACHE      Default: /home/reranker_experiment/deps
  HOST_VENV_CACHE           Default: /home/reranker_experiment/venvs-container-vllm018
  MODELSLIM_REPO            Default: https://gitee.com/ascend/msit.git
  MODELSLIM_COMMIT          Pinned official Qwen3 310P W8A8SC commit
  REINSTALL_MODELSLIM       Reinstall the pinned ModelSlim source. Default: 0
  PIP_INDEX_URL             Default: NJU PyPI mirror

Run VERSION_ONLY=1 first. The v0.18.0 image requires a much newer userspace
stack than v0.10.0rc1; the NPU smoke is the authoritative compatibility check
for a fixed host driver.
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
HOST_W8A8S_MODEL_PATH="${HOST_W8A8S_MODEL_PATH:-/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8S-v1}"
HOST_MODEL_PATH="${HOST_MODEL_PATH:-/home/reranker_experiment/model/Qwen3-Reranker-0.6B-W8A8SC-vllm-v1}"
HOST_DATA_PATH="${HOST_DATA_PATH:-/home/reranker_experiment/data/latency_delay}"
HOST_OUTPUT_PATH="${HOST_OUTPUT_PATH:-/home/reranker_experiment/output/qwen3_w8a8sc_vllm_0625}"
TRAIN_JSONL="${TRAIN_JSONL:-/home/reranker_experiment/data/split/train.jsonl}"
IMAGE="${IMAGE:-quay.nju.edu.cn/ascend/vllm-ascend:v0.18.0-310p}"
PULL_IMAGE="${PULL_IMAGE:-1}"
VERSION_ONLY="${VERSION_ONLY:-0}"
CONVERT_MODEL="${CONVERT_MODEL:-0}"
RUN_EVAL="${RUN_EVAL:-1}"
DATASET="${DATASET:-0625caption}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((BATCH_SIZE * MAX_LENGTH))}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-${BATCH_SIZE}}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
COMPRESS_PROCESS_NUM="${COMPRESS_PROCESS_NUM:-4}"
HOST_MODELSLIM_CACHE="${HOST_MODELSLIM_CACHE:-/home/reranker_experiment/deps}"
HOST_VENV_CACHE="${HOST_VENV_CACHE:-/home/reranker_experiment/venvs-container-vllm018}"
MODELSLIM_REPO="${MODELSLIM_REPO:-https://gitee.com/ascend/msit.git}"
MODELSLIM_COMMIT="${MODELSLIM_COMMIT:-6a860e4a7b48b4573a8aeeaa12123d2bbc9ec9b8}"
REINSTALL_MODELSLIM="${REINSTALL_MODELSLIM:-0}"
PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirror.nju.edu.cn/pypi/web/simple/}"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
INSTRUCTION="${INSTRUCTION:-}"

for toggle in PULL_IMAGE VERSION_ONLY CONVERT_MODEL RUN_EVAL REINSTALL_MODELSLIM; do
  value="${!toggle}"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "[invalid] ${toggle} must be 0 or 1; got ${value}" >&2
    exit 2
  fi
done
for number in MAX_LENGTH BATCH_SIZE MAX_NUM_BATCHED_TOKENS MAX_NUM_SEQS COMPRESS_PROCESS_NUM; do
  value="${!number}"
  if [[ ! "${value}" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "[invalid] ${number} must be a positive integer; got ${value}" >&2
    exit 2
  fi
done
if ! command -v docker >/dev/null 2>&1; then
  echo "[missing] docker is not installed" >&2
  exit 3
fi
if ! command -v npu-smi >/dev/null 2>&1 || ! npu-smi info >/dev/null; then
  echo "[missing] npu-smi cannot access the fixed host driver/NPU" >&2
  exit 3
fi
probe="${HOST_REPO_PATH}/scripts/probe_vllm_ascend_w8a8sc_310p.py"
if [[ ! -f "${probe}" ]]; then
  echo "[missing] ${probe}" >&2
  exit 3
fi

if [[ "${VERSION_ONLY}" != "1" ]]; then
  if [[ "${CONVERT_MODEL}" == "1" ]]; then
    if ! command -v git >/dev/null 2>&1; then
      echo "[missing] host git is required to prepare pinned ModelSlim" >&2
      exit 3
    fi
    if [[ ! -f "${HOST_W8A8S_MODEL_PATH}/config.json" ]]; then
      echo "[missing] W8A8S input model: ${HOST_W8A8S_MODEL_PATH}" >&2
      exit 3
    fi
    if find "${HOST_MODEL_PATH}" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null | grep -q .; then
      echo "[exists] vLLM W8A8SC output is not empty: ${HOST_MODEL_PATH}" >&2
      echo "[exists] use a new HOST_MODEL_PATH; the ATB W8A8SC directory is not reusable" >&2
      exit 3
    fi
  elif [[ ! -f "${HOST_MODEL_PATH}/config.json" ]]; then
    echo "[missing] vLLM W8A8SC model: ${HOST_MODEL_PATH}" >&2
    echo "[missing] set CONVERT_MODEL=1 and provide HOST_W8A8S_MODEL_PATH" >&2
    exit 3
  fi
  if [[ "${RUN_EVAL}" == "1" && ! -d "${HOST_DATA_PATH}" ]]; then
    echo "[missing] business data root: ${HOST_DATA_PATH}" >&2
    exit 3
  fi
fi

host_modelslim_dir="${HOST_MODELSLIM_CACHE}/msit-modelslim-vllm-0.18"
if [[ "${VERSION_ONLY}" != "1" && "${CONVERT_MODEL}" == "1" ]]; then
  mkdir -p "${HOST_MODELSLIM_CACHE}" "${HOST_VENV_CACHE}"
  if [[ ! -d "${host_modelslim_dir}/.git" ]]; then
    echo "[modelslim] cloning official source on the host"
    git clone --depth 1 --no-checkout "${MODELSLIM_REPO}" "${host_modelslim_dir}"
  fi
  if ! git -C "${host_modelslim_dir}" cat-file -e "${MODELSLIM_COMMIT}^{commit}" 2>/dev/null; then
    echo "[modelslim] fetching pinned commit ${MODELSLIM_COMMIT}"
    git -C "${host_modelslim_dir}" fetch --depth 1 origin "${MODELSLIM_COMMIT}"
  fi
  if ! git -C "${host_modelslim_dir}" diff --quiet || \
     ! git -C "${host_modelslim_dir}" diff --cached --quiet; then
    echo "[invalid] tracked changes exist in ModelSlim cache: ${host_modelslim_dir}" >&2
    echo "[invalid] preserve or discard those changes before rerunning" >&2
    exit 3
  fi
  git -C "${host_modelslim_dir}" checkout --detach "${MODELSLIM_COMMIT}"
  actual_modelslim_commit="$(git -C "${host_modelslim_dir}" rev-parse HEAD)"
  if [[ "${actual_modelslim_commit}" != "${MODELSLIM_COMMIT}" ]]; then
    echo "[invalid] ModelSlim HEAD=${actual_modelslim_commit}; expected=${MODELSLIM_COMMIT}" >&2
    exit 3
  fi
  printf '%s\n' "${actual_modelslim_commit}" > "${host_modelslim_dir}/.memranker_pinned_commit"
  echo "[modelslim] commit=${actual_modelslim_commit}"
fi

experiment_mount_args=()
if [[ "${VERSION_ONLY}" != "1" ]]; then
  mkdir -p "${HOST_MODEL_PATH}" "${HOST_OUTPUT_PATH}"
  experiment_mount_args+=(
    -v "${HOST_W8A8S_MODEL_PATH}:/models/w8a8s:ro"
    -v "${HOST_MODEL_PATH}:/models/w8a8sc-vllm:rw"
    -v "${HOST_DATA_PATH}:/workspace/data:ro"
    -v "${HOST_OUTPUT_PATH}:/outputs:rw"
  )
  if [[ "${CONVERT_MODEL}" == "1" ]]; then
    experiment_mount_args+=(
      -v "${HOST_MODELSLIM_CACHE}:/cache/deps:rw"
      -v "${HOST_VENV_CACHE}:/cache/venvs:rw"
    )
  fi
fi
if [[ "${PULL_IMAGE}" == "1" ]]; then
  docker pull "${IMAGE}"
elif ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "[missing] local image ${IMAGE}; rerun with PULL_IMAGE=1" >&2
  exit 3
fi

if [[ -z "${INSTRUCTION}" && "${VERSION_ONLY}" != "1" && "${RUN_EVAL}" == "1" ]]; then
  if [[ ! -f "${TRAIN_JSONL}" ]]; then
    echo "[missing] TRAIN_JSONL=${TRAIN_JSONL}; set INSTRUCTION explicitly" >&2
    exit 3
  fi
  if command -v python3 >/dev/null 2>&1; then
    host_python=python3
  else
    host_python=python
  fi
  INSTRUCTION="$("${host_python}" - "${TRAIN_JSONL}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    for line_number, line in enumerate(stream, 1):
        if not line.strip():
            continue
        instruction = json.loads(line).get("instruction")
        if not isinstance(instruction, str) or not instruction.strip():
            raise SystemExit(f"[invalid] no instruction at line {line_number}")
        print(instruction.strip())
        break
    else:
        raise SystemExit("[invalid] empty TRAIN_JSONL")
PY
)"
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

echo "[config] image=${IMAGE}"
echo "[config] fixed_host_driver=$(npu-smi info -t board -i 0 2>/dev/null | head -n 1 || true)"
echo "[config] version_only=${VERSION_ONLY} convert=${CONVERT_MODEL} eval=${RUN_EVAL}"
echo "[config] w8a8s=${HOST_W8A8S_MODEL_PATH}"
echo "[config] vllm_w8a8sc=${HOST_MODEL_PATH}"

docker run --rm \
  --network host \
  --shm-size=16g \
  --privileged=true \
  --user root \
  --entrypoint /bin/bash \
  -e ASCEND_RUNTIME_OPTIONS=NODRV \
  -e "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}" \
  -e "VERSION_ONLY=${VERSION_ONLY}" \
  -e "CONVERT_MODEL=${CONVERT_MODEL}" \
  -e "RUN_EVAL=${RUN_EVAL}" \
  -e "DATASET=${DATASET}" \
  -e "MAX_LENGTH=${MAX_LENGTH}" \
  -e "BATCH_SIZE=${BATCH_SIZE}" \
  -e "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}" \
  -e "MAX_NUM_SEQS=${MAX_NUM_SEQS}" \
  -e "GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}" \
  -e "COMPRESS_PROCESS_NUM=${COMPRESS_PROCESS_NUM}" \
  -e "MODELSLIM_COMMIT=${MODELSLIM_COMMIT}" \
  -e "REINSTALL_MODELSLIM=${REINSTALL_MODELSLIM}" \
  -e "INSTRUCTION=${INSTRUCTION}" \
  -e "PIP_INDEX_URL=${PIP_INDEX_URL}" \
  "${mount_args[@]}" \
  "${experiment_mount_args[@]}" \
  -v "${HOST_REPO_PATH}:/workspace/memranker:ro" \
  -w /workspace/memranker \
  "${IMAGE}" -lc '
    set -euo pipefail
    export VLLM_USE_V1=1
    export VLLM_DEVICE_BACKEND=ascend
    export PYTHONPATH="/workspace/memranker:${PYTHONPATH:-}"

    echo "[step 1/4] exact vLLM-Ascend/CANN/driver version probe"
    python scripts/probe_vllm_ascend_w8a8sc_310p.py \
      --npu-smoke \
      --require-w8a8sc
    if [[ "${VERSION_ONLY}" == "1" ]]; then
      exit 0
    fi

    if [[ "${CONVERT_MODEL}" == "1" ]]; then
      echo "[step 2/4] converting W8A8S to vLLM sharded_state W8A8SC"
      modelslim_dir=/cache/deps/msit-modelslim-vllm-0.18
      modelslim_venv=/cache/venvs/modelslim-vllm-0.18
      commit_stamp="${modelslim_dir}/.memranker_pinned_commit"
      if [[ ! -f "${commit_stamp}" ]] || [[ "$(<"${commit_stamp}")" != "${MODELSLIM_COMMIT}" ]]; then
        echo "[invalid] pinned ModelSlim source/stamp is missing or mismatched" >&2
        exit 4
      fi
      if [[ ! -x "${modelslim_venv}/bin/python" ]]; then
        python -m venv --system-site-packages "${modelslim_venv}"
      fi
      # shellcheck disable=SC1091
      source "${modelslim_venv}/bin/activate"
      install_stamp="${modelslim_venv}/.memranker_${MODELSLIM_COMMIT}_installed"
      check_modelslim() {
        python - <<PY
import torch
import torch_npu
from msmodelslim.pytorch.weight_compression import CompressConfig, Compressor

assert torch.npu.is_available()
print("[modelslim] torch=" + torch.__version__)
print("[modelslim] torch_npu=" + torch_npu.__version__)
print("[modelslim] weight-compression imports are complete")
PY
      }
      if [[ "${REINSTALL_MODELSLIM}" == "1" || ! -f "${install_stamp}" ]] || ! check_modelslim; then
        export PIP_INDEX_URL
        echo "[modelslim] installing commit=${MODELSLIM_COMMIT} pip=${PIP_INDEX_URL}"
        pushd "${modelslim_dir}/msmodelslim" >/dev/null
        bash install.sh
        popd >/dev/null
        check_modelslim
        printf "%s\n" "${MODELSLIM_COMMIT}" > "${install_stamp}"
      fi
      compress_root="$(python - <<PY
from pathlib import Path
import msmodelslim.pytorch.weight_compression.compress_utils as module

print(Path(module.__file__).resolve().parent / "compress_graph")
PY
)"
      compress_executor="${compress_root}/build/compress_excutor"
      if [[ ! -x "${compress_executor}" ]]; then
        cann_root="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
        if [[ ! -f "${compress_root}/build.sh" || ! -d "${cann_root}" ]]; then
          echo "[missing] ModelSlim compressor build prerequisites" >&2
          echo "[missing] build=${compress_root}/build.sh CANN=${cann_root}" >&2
          exit 4
        fi
        echo "[modelslim] building compressor against ${cann_root}"
        bash "${compress_root}/build.sh" "${cann_root}"
      fi
      if [[ ! -x "${compress_executor}" ]]; then
        echo "[missing] ModelSlim compressor executable: ${compress_executor}" >&2
        exit 4
      fi
      python_libdir="$(python -c "import sysconfig; print(sysconfig.get_config_var(\"LIBDIR\") or \"\")")"
      if [[ -n "${python_libdir}" ]]; then
        export LD_LIBRARY_PATH="${python_libdir}:${LD_LIBRARY_PATH:-}"
      fi
      export PYTORCH_NPU_ALLOC_CONF=max_split_size_mb:256
      echo "[modelslim] compressor=${compress_executor}"
      save_script=""
      for candidate in \
        /vllm-workspace/vllm-ascend/examples/save_sharded_state_310.py \
        /workspace/vllm-ascend/examples/save_sharded_state_310.py; do
        if [[ -f "${candidate}" ]]; then
          save_script="${candidate}"
          break
        fi
      done
      if [[ -z "${save_script}" ]]; then
        save_script="$(find / -path "*/examples/save_sharded_state_310.py" -print -quit 2>/dev/null || true)"
      fi
      if [[ -z "${save_script}" || ! -f "${save_script}" ]]; then
        echo "[missing] official examples/save_sharded_state_310.py in ${IMAGE:-container}" >&2
        exit 4
      fi
      python "${save_script}" \
        --model /models/w8a8s \
        --output /models/w8a8sc-vllm \
        --tensor-parallel-size 1 \
        --max-model-len "${MAX_LENGTH}" \
        --max-num-batched-tokens "${MAX_LENGTH}" \
        --max-num-seqs 1 \
        --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
        --dtype float16 \
        --quantization ascend \
        --load-format auto \
        --enforce-eager \
        --enable-compress \
        --compress-process-num "${COMPRESS_PROCESS_NUM}" \
        --trust-remote-code
    else
      echo "[step 2/4] conversion skipped; using existing vLLM W8A8SC"
    fi

    echo "[step 3/4] validating vLLM W8A8SC checkpoint layout"
    python scripts/probe_vllm_ascend_w8a8sc_310p.py \
      --model-path /models/w8a8sc-vllm \
      --require-w8a8sc
    if [[ "${RUN_EVAL}" != "1" ]]; then
      exit 0
    fi

    echo "[step 4/4] vLLM W8A8SC business evaluation"
    if ! python -c "import openpyxl" >/dev/null 2>&1; then
      python -m pip install \
        --disable-pip-version-check \
        --index-url "${PIP_INDEX_URL}" \
        openpyxl==3.1.5
    fi
    export DATA_ROOT=/workspace/data
    export MODEL_NAME=qwen3_reranker_06b_w8a8sc_vllm
    export MODEL_PATH=/models/w8a8sc-vllm
    export OUTPUT_ROOT=/outputs
    export SCORING_BACKEND=generate
    export DTYPE=float16
    export VLLM_QUANTIZATION=ascend
    export VLLM_LOAD_FORMAT=sharded_state
    export TENSOR_PARALLEL_SIZE=1
    export WARMUP_PAIRS="${BATCH_SIZE}"
    export ENFORCE_EAGER=1
    export ENABLE_PREFIX_CACHING=0
    export LOCAL_FILES_ONLY=1
    export SKIP_EXISTING=0
    export CONTINUE_ON_ERROR=0
    export SKIP_MISSING=0
    bash scripts/eval_business_matrix_ascend_vllm.sh
  '

echo "[done] vLLM W8A8SC experiment completed"
