#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Install the local vLLM-Ascend 910B4 reranker stack on restricted Huawei Cloud.

Why this script exists:
  vllm-ascend 0.11.0rc1 officially pairs with torch/torch-npu 2.7.1, but the
  PyPI vllm==0.11.0 wheel metadata depends on torch==2.8.0. Installing all
  packages in one pip resolver pass therefore fails. This script installs the
  Ascend torch stack first and then installs vllm/vllm-ascend with --no-deps.

Environment overrides:
  PYTHON_BIN          Python 3.9-3.11 executable. Default: python
  PYPI_INDEX          Default: https://mirrors.huaweicloud.com/repository/pypi/simple
  ASCEND_INDEX        Default: https://mirrors.huaweicloud.com/ascend/repos/pypi
  VLLM_VERSION        Default: 0.11.0
  VLLM_ASCEND_VERSION Default: 0.11.0rc1
  REQUIREMENTS        Default: requirements-ascend-vllm.txt
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYPI_INDEX="${PYPI_INDEX:-https://mirrors.huaweicloud.com/repository/pypi/simple}"
ASCEND_INDEX="${ASCEND_INDEX:-https://mirrors.huaweicloud.com/ascend/repos/pypi}"
VLLM_VERSION="${VLLM_VERSION:-0.11.0}"
VLLM_ASCEND_VERSION="${VLLM_ASCEND_VERSION:-0.11.0rc1}"
REQUIREMENTS="${REQUIREMENTS:-requirements-ascend-vllm.txt}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[error] PYTHON_BIN was not found: ${PYTHON_BIN}" >&2
  exit 2
fi
if ! "${PYTHON_BIN}" -c 'import sys; raise SystemExit(0 if (3, 9) <= sys.version_info[:2] < (3, 12) else 1)'; then
  echo "[error] Python 3.9, 3.10, or 3.11 is required: $("${PYTHON_BIN}" --version 2>&1)" >&2
  exit 2
fi

echo "[env] installing with ${PYTHON_BIN} ($("${PYTHON_BIN}" --version 2>&1))"

"${PYTHON_BIN}" -m pip install --upgrade "pip>=24" "setuptools==80.9.0" wheel \
  -i "${PYPI_INDEX}" \
  --extra-index-url "${ASCEND_INDEX}"

"${PYTHON_BIN}" -m pip install -r "${REQUIREMENTS}" \
  -i "${PYPI_INDEX}" \
  --extra-index-url "${ASCEND_INDEX}"

"${PYTHON_BIN}" -m pip install "vllm==${VLLM_VERSION}" --no-deps \
  -i "${PYPI_INDEX}" \
  --extra-index-url "${ASCEND_INDEX}"

"${PYTHON_BIN}" -m pip install "vllm-ascend==${VLLM_ASCEND_VERSION}" --no-deps \
  -i "${PYPI_INDEX}" \
  --extra-index-url "${ASCEND_INDEX}"

"${PYTHON_BIN}" src/vllm_py39_compat.py

"${PYTHON_BIN}" - <<'PY'
import sys
import pkg_resources  # noqa: F401
import torch
import torch_npu  # noqa: F401
import vllm
import vllm_ascend  # noqa: F401

print("python", sys.version.split()[0])
print("pkg_resources import ok")
print("torch", torch.__version__)
print("npu_available", torch.npu.is_available())
print("vllm", vllm.__version__)
print("vllm_ascend import ok")
PY
