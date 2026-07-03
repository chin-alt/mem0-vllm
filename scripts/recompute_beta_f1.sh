#!/usr/bin/env bash
set -euo pipefail

BETAS="${BETAS:-1.0 0.7 0.5 0.3 0.2}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Recompute dynamic Expected-Fbeta cutoffs and real F1 from existing predictions.

Examples:
  OUTPUT_ROOT=outputs/business_matrix_xxx bash scripts/recompute_beta_f1.sh
  RUN_DIR=outputs/business_matrix_xxx/0428caption__memreranker_4b bash scripts/recompute_beta_f1.sh
  BETAS="1.0 0.7 0.5 0.3 0.2" OUTPUT_ROOT=outputs/business_matrix_xxx bash scripts/recompute_beta_f1.sh

You can also pass src/recompute_beta_f1.py arguments directly after this script.
EOF
  exit 0
fi

if [[ "$#" -gt 0 ]]; then
  exec "${PYTHON_BIN}" src/recompute_beta_f1.py "$@"
fi

if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  exec "${PYTHON_BIN}" src/recompute_beta_f1.py --output_root "${OUTPUT_ROOT}" --betas ${BETAS}
fi

if [[ -n "${RUN_DIR:-}" ]]; then
  exec "${PYTHON_BIN}" src/recompute_beta_f1.py --run_dir "${RUN_DIR}" --betas ${BETAS}
fi

cat >&2 <<'EOF'
Set OUTPUT_ROOT or RUN_DIR, or pass explicit src/recompute_beta_f1.py arguments.

Examples:
  OUTPUT_ROOT=outputs/business_matrix_xxx bash scripts/recompute_beta_f1.sh
  RUN_DIR=outputs/business_matrix_xxx/0428caption__memreranker_4b bash scripts/recompute_beta_f1.sh
EOF
exit 2
