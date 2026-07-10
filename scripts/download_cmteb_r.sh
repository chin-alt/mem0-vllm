#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="${OUTPUT_DIR:-data/cmteb_r/raw}"
DATASETS="${DATASETS:-T2Retrieval MMarcoRetrieval DuRetrieval CovidRetrieval CmedqaRetrieval EcomRetrieval MedicalRetrieval}"
PYTHON_BIN="${PYTHON_BIN:-python}"

read -r -a DATASET_ARGS <<< "${DATASETS}"

"${PYTHON_BIN}" src/download_cmteb_r.py \
  --output_dir "${OUTPUT_DIR}" \
  --datasets "${DATASET_ARGS[@]}"
