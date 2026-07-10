#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/cmteb_r/raw}"
DATASETS="${DATASETS:-T2Retrieval MMarcoRetrieval DuRetrieval CovidRetrieval CmedqaRetrieval EcomRetrieval MedicalRetrieval}"
PYTHON_BIN="${PYTHON_BIN:-python}"
INCLUDE_QRELS="${INCLUDE_QRELS:-0}"
QRELS_DATASET_SUFFIX="${QRELS_DATASET_SUFFIX:--qrels}"

read -r -a DATASET_ARGS <<< "${DATASETS}"
EXTRA_ARGS=()
if [[ "${INCLUDE_QRELS}" == "1" ]]; then
  EXTRA_ARGS+=(--include_qrels)
fi

"${PYTHON_BIN}" src/download_cmteb_r.py \
  --output_dir "${OUTPUT_DIR}" \
  --datasets "${DATASET_ARGS[@]}" \
  --qrels_dataset_suffix "${QRELS_DATASET_SUFFIX}" \
  "${EXTRA_ARGS[@]}"
