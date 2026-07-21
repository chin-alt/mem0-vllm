#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run local Ascend 910B4 business matrix inference.

This wrapper never calls an HTTP/API service. It dispatches to one of the local
NPU backends:

  BACKEND=vllm   vLLM-Ascend LLM.score path. Default.
  BACKEND=torch  torch-npu + Transformers yes/no-logit fallback.

All other environment variables are passed through to the backend script.

Examples:
  BACKEND=vllm bash scripts/eval_business_matrix_ascend_local.sh
  BACKEND=torch BATCH_SIZE=1 bash scripts/eval_business_matrix_ascend_local.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

BACKEND="${BACKEND:-vllm}"

case "${BACKEND}" in
  vllm)
    exec bash scripts/eval_business_matrix_ascend_vllm.sh "$@"
    ;;
  torch)
    exec bash scripts/eval_business_matrix_ascend_torch.sh "$@"
    ;;
  *)
    echo "Unsupported BACKEND=${BACKEND}; use vllm or torch." >&2
    exit 2
    ;;
esac
