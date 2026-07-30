#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKENDS="${BACKENDS:-generate pooling}"
BATCH_SIZES="${BATCH_SIZES:-1 8 16 32}"
MAX_LENGTHS="${MAX_LENGTHS:-512 1024}"
DATASET="${DATASET:-0428caption}"
PULL_IMAGE="${PULL_IMAGE:-1}"
MODEL_LABEL="${MODEL_LABEL:-fp16}"
HOST_OUTPUT_BASE="${HOST_OUTPUT_BASE:-/home/reranker_experiment/output/qwen3_reranker_310p_ab}"

read -r -a backend_array <<< "${BACKENDS}"
read -r -a batch_array <<< "${BATCH_SIZES}"
read -r -a length_array <<< "${MAX_LENGTHS}"
mkdir -p "${HOST_OUTPUT_BASE}"

pull_next="${PULL_IMAGE}"
for backend in "${backend_array[@]}"; do
  for max_length in "${length_array[@]}"; do
    for batch_size in "${batch_array[@]}"; do
      run_name="${MODEL_LABEL}_${backend}_len${max_length}_bs${batch_size}"
      echo "[benchmark] ${run_name}"
      PULL_IMAGE="${pull_next}" \
      CONTAINER_NAME="qwen3-reranker-${backend}-${max_length}-${batch_size}" \
      HOST_OUTPUT_PATH="${HOST_OUTPUT_BASE}/${run_name}" \
      DATASET="${DATASET}" \
      SCORING_BACKEND="${backend}" \
      MAX_LENGTH="${max_length}" \
      BATCH_SIZE="${batch_size}" \
      MAX_NUM_SEQS="${batch_size}" \
      MAX_NUM_BATCHED_TOKENS="$((max_length * batch_size))" \
      WARMUP_PAIRS="${WARMUP_PAIRS:-${batch_size}}" \
      bash "${SCRIPT_DIR}/run_qwen3_reranker_vllm_310p_container.sh"
      pull_next=0
    done
  done
done

python - "${HOST_OUTPUT_BASE}" <<'PY'
import json
import sys
from pathlib import Path

base = Path(sys.argv[1])
rows = []
for metrics_path in sorted(base.rglob("metrics.json")):
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "run": str(metrics_path.relative_to(base).parents[1]),
            "backend": metrics.get("scoring_backend"),
            "max_length": metrics.get("max_length"),
            "batch_size": metrics.get("batch_size"),
            "pairs_per_second": metrics.get("examples_per_second"),
            "pair_p50_seconds": metrics.get("pair_latency_p50_seconds"),
            "pair_p95_seconds": metrics.get("pair_latency_p95_seconds"),
            "NDCG@10": metrics.get("NDCG@10"),
            "Recall@5": metrics.get("Recall@5"),
        }
    )
summary_path = base / "benchmark_summary.json"
summary_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(rows, ensure_ascii=False, indent=2))
print("[benchmark] wrote", summary_path)
PY
