#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST_REPO_PATH="${HOST_REPO_PATH:-${REPO_ROOT}}"
HOST_DATA_PATH="${HOST_DATA_PATH:-/home/reranker_experiment/data/latency_delay}"
HOST_MODEL_PATH="${HOST_MODEL_PATH:-/home/reranker_experiment/model/qwen3_reranker_06b_lora_merged}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
HOST_OUTPUT_BASE="${HOST_OUTPUT_BASE:-/home/reranker_experiment/output/qwen3_fp16_recall_topk_0625_${RUN_TAG}}"

TOP_K_START="${TOP_K_START:-25}"
TOP_K_END="${TOP_K_END:-10}"
PULL_IMAGE="${PULL_IMAGE:-0}"
INSTRUCTION="${INSTRUCTION:-${RERANK_INSTRUCTION:-Given a user query, retrieve relevant documents that answer the query.}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_BIN=python
fi

for value in "${TOP_K_START}" "${TOP_K_END}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "[invalid] TOP_K_START and TOP_K_END must be positive integers" >&2
    exit 2
  fi
done
if (( TOP_K_START < TOP_K_END )); then
  echo "[invalid] TOP_K_START must be >= TOP_K_END for a descending sweep" >&2
  exit 2
fi

mkdir -p "${HOST_OUTPUT_BASE}"
echo "[sweep] top_k=${TOP_K_START}..${TOP_K_END} (descending)"
echo "[sweep] output=${HOST_OUTPUT_BASE}"

pull_image_this_run="${PULL_IMAGE}"
for ((top_k = TOP_K_START; top_k >= TOP_K_END; top_k--)); do
  run_output="${HOST_OUTPUT_BASE}/top${top_k}"
  echo "======================================================================"
  echo "[sweep] recall_top_k=${top_k} output=${run_output}"
  echo "======================================================================"

  HOST_REPO_PATH="${HOST_REPO_PATH}" \
  HOST_DATA_PATH="${HOST_DATA_PATH}" \
  HOST_MODEL_PATH="${HOST_MODEL_PATH}" \
  HOST_OUTPUT_PATH="${run_output}" \
  DATASET=0625caption \
  SCORING_BACKEND=pooling \
  INSTRUCTION="${INSTRUCTION}" \
  RECALL_TOP_K="${top_k}" \
  PRETOKENIZED_POOLING=1 \
  TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-256}" \
  PREFIX_CACHE_SEEDING=0 \
  RESET_PREFIX_CACHE_AFTER_WARMUP=0 \
  ENABLE_PREFIX_CACHING=0 \
  SUBMIT_ALL_AT_ONCE=1 \
  GROUP_BY_QUERY=1 \
  SHOW_PROGRESS=0 \
  DTYPE="${DTYPE:-float16}" \
  MAX_LENGTH="${MAX_LENGTH:-1024}" \
  BATCH_SIZE="${BATCH_SIZE:-16}" \
  MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}" \
  MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-16384}" \
  WARMUP_PAIRS="${WARMUP_PAIRS:-16}" \
  GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}" \
  TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}" \
  CPU_AFFINITY_CONF="${CPU_AFFINITY_CONF:-1}" \
  PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-max_split_size_mb:256}" \
  PIP_INDEX_URL="${PIP_INDEX_URL:-https://mirror.nju.edu.cn/pypi/web/simple/}" \
  PULL_IMAGE="${pull_image_this_run}" \
  bash "${REPO_ROOT}/scripts/run_qwen3_reranker_vllm_310p_container.sh"

  # Pull or validate the image only on the first sweep iteration.
  pull_image_this_run=0
done

"${PYTHON_BIN}" - "${HOST_OUTPUT_BASE}" "${TOP_K_START}" "${TOP_K_END}" <<'PY'
import csv
import json
import sys
from pathlib import Path


output_root = Path(sys.argv[1])
top_k_start = int(sys.argv[2])
top_k_end = int(sys.argv[3])
rows = []
missing = []

for top_k in range(top_k_start, top_k_end - 1, -1):
    candidates = sorted((output_root / f"top{top_k}").glob("0625caption__*/metrics.json"))
    if len(candidates) != 1:
        missing.append({"top_k": top_k, "matches": [str(path) for path in candidates]})
        continue
    metrics_path = candidates[0]
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "recall_top_k": top_k,
            "Accuracy@GTCount": float(metrics["Accuracy@GTCount"]),
            "score_time_seconds": float(metrics["score_time_seconds"]),
            "num_scored_pairs": int(metrics["num_scored_pairs"]),
            "examples_per_second": float(metrics["examples_per_second"]),
            "recalled_gt_doc_retention_at_top_k": float(
                metrics.get("recalled_gt_doc_retention_at_top_k", 1.0)
            ),
            "metrics_json": str(metrics_path),
        }
    )

if missing:
    raise SystemExit(f"Missing or ambiguous metrics.json files: {json.dumps(missing, ensure_ascii=False)}")

top25_time = rows[0]["score_time_seconds"]
for row in rows:
    row["time_reduction_vs_top25_percent"] = (
        (top25_time - row["score_time_seconds"]) / top25_time * 100.0
        if top25_time > 0
        else 0.0
    )

csv_path = output_root / "recall_topk_sweep.csv"
json_path = output_root / "recall_topk_sweep.json"
fieldnames = [
    "recall_top_k",
    "Accuracy@GTCount",
    "score_time_seconds",
    "time_reduction_vs_top25_percent",
    "num_scored_pairs",
    "examples_per_second",
    "recalled_gt_doc_retention_at_top_k",
    "metrics_json",
]
with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
json_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

print("\nTopK  Accuracy@GTCount  score_time_seconds  reduction_vs_top25  pairs  GT-retention")
for row in rows:
    print(
        f"{row['recall_top_k']:>4}  "
        f"{row['Accuracy@GTCount']:>16.6f}  "
        f"{row['score_time_seconds']:>18.3f}  "
        f"{row['time_reduction_vs_top25_percent']:>17.2f}%  "
        f"{row['num_scored_pairs']:>5}  "
        f"{row['recalled_gt_doc_retention_at_top_k']:>12.6f}"
    )
print(f"[done] csv={csv_path}")
print(f"[done] json={json_path}")
PY
