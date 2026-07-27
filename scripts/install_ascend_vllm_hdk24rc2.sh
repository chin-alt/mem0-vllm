#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REQUIREMENTS="${REQUIREMENTS:-requirements-ascend-vllm-hdk24rc2.txt}"
export VLLM_VERSION="${VLLM_VERSION:-0.8.5.post1}"
export VLLM_ASCEND_VERSION="${VLLM_ASCEND_VERSION:-0.8.5rc1}"

exec bash "${SCRIPT_DIR}/install_ascend_vllm_910b4.sh" "$@"
