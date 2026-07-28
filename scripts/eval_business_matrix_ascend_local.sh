#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run local Ascend business matrix inference.

This wrapper never calls an external model service. It dispatches to a local
NPU backend; MindIE communicates only over the loopback address:

  BACKEND=vllm    vLLM-Ascend path. Default.
  BACKEND=torch   torch-npu + Transformers yes/no-logit fallback.
  BACKEND=mindie  Local MindIE service yes/no-logprob path for 310P.

All other environment variables are passed through to the backend script.

Examples:
  BACKEND=vllm bash scripts/eval_business_matrix_ascend_local.sh
  BACKEND=torch BATCH_SIZE=1 bash scripts/eval_business_matrix_ascend_local.sh
  BACKEND=mindie BATCH_SIZE=32 bash scripts/eval_business_matrix_ascend_local.sh
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
  mindie)
    exec bash scripts/eval_business_matrix_ascend_mindie.sh "$@"
    ;;
  *)
    echo "Unsupported BACKEND=${BACKEND}; use vllm, torch, or mindie." >&2
    exit 2
    ;;
esac
