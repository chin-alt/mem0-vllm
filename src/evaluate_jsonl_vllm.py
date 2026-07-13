from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_eval_vllm import create_vllm_llm, score_with_vllm  # noqa: E402
from data import load_examples, read_json_records, write_jsonl  # noqa: E402
from metrics import add_group_ranks, compute_all_metrics  # noqa: E402
from modeling import DEFAULT_MODEL_NAME  # noqa: E402


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate query-doc-label JSONL with vLLM Qwen3-Reranker scoring.")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", default="outputs/cmteb_r_vllm_eval")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=True)
    parser.add_argument("--no_enable_prefix_caching", dest="enable_prefix_caching", action="store_false")
    parser.add_argument("--sort_by_length", action="store_true", default=True)
    parser.add_argument("--no_sort_by_length", dest="sort_by_length", action="store_false")
    parser.add_argument("--sort_descending", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--expected_fbeta_betas", type=float, nargs="+", default=[0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument("--progress_file", default="", help="Optional JSONL file updated after each scoring batch.")
    return parser.parse_args()


def choose_instruction(args_instruction: str, examples: list) -> str:
    if args_instruction.strip():
        return args_instruction.strip()
    for ex in examples:
        if ex.instruction.strip():
            return ex.instruction.strip()
    return "Given a query, retrieve relevant documents that answer the query."


def choose_expected_fbeta_best_k(score_list: list[float], beta: float) -> tuple[int, float]:
    if not score_list:
        return 0, 0.0
    scores = np.asarray(score_list, dtype=np.float64)
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    cum_gain = np.cumsum(norm_scores)
    total_sum = float(cum_gain[-1]) if len(cum_gain) else 0.0
    k_array = np.arange(1, len(scores) + 1, dtype=np.float64)
    expected_fbeta = (1 + beta**2) * cum_gain / (beta**2 * total_sum + k_array)
    best_idx = int(np.argmax(expected_fbeta))
    return best_idx + 1, float(expected_fbeta[best_idx])


def f1_from_precision_recall(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def compute_dynamic_beta_metrics(
    rows: list[dict],
    betas: list[float],
    relevance_threshold: float,
) -> tuple[list[dict], list[dict], dict[str, float]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("group_key") or row.get("query_id") or row.get("query"))].append(row)

    per_query_rows: list[dict] = []
    summary_rows: list[dict] = []
    overall_updates: dict[str, float] = {}
    true_count_sum = 0
    candidate_relevant_sum = 0
    ideal_top_k_sum = 0
    ideal_topk_precision_sum = 0.0
    ideal_topk_recall_sum = 0.0
    ideal_topk_f1_sum = 0.0
    ideal_topk_selected_sum = 0
    ideal_topk_hit_sum = 0
    ideal_topk_true_sum = 0

    for group_key, group_rows in grouped.items():
        ranked = sorted(group_rows, key=lambda row: (int(row.get("rank_by_score", 10**9)), -float(row.get("score", 0.0))))
        relevant_ids = {str(row["doc_id"]) for row in ranked if float(row.get("label", 0.0)) >= relevance_threshold}
        true_counts = [int(row.get("true_relevant_count") or 0) for row in ranked]
        true_count = max(true_counts) if true_counts else 0
        if true_count <= 0:
            true_count = len(relevant_ids)
        candidate_relevant_count = len(relevant_ids)
        ideal_top_k = candidate_relevant_count
        true_count_sum += true_count
        candidate_relevant_sum += candidate_relevant_count
        ideal_top_k_sum += ideal_top_k
        ideal_selected = ranked[:ideal_top_k]
        ideal_selected_ids = [str(row["doc_id"]) for row in ideal_selected]
        ideal_hit_count = len(set(ideal_selected_ids) & relevant_ids)
        ideal_precision = ideal_hit_count / len(ideal_selected_ids) if ideal_selected_ids else 0.0
        ideal_recall = ideal_hit_count / true_count if true_count else 0.0
        ideal_f1 = f1_from_precision_recall(ideal_precision, ideal_recall)
        ideal_topk_precision_sum += ideal_precision
        ideal_topk_recall_sum += ideal_recall
        ideal_topk_f1_sum += ideal_f1
        ideal_topk_selected_sum += len(ideal_selected_ids)
        ideal_topk_hit_sum += ideal_hit_count
        ideal_topk_true_sum += true_count

        score_list = [float(row.get("score", 0.0)) for row in ranked]
        base = {
            "group_key": group_key,
            "query": ranked[0].get("query", "") if ranked else "",
            "query_id": ranked[0].get("query_id", "") if ranked else "",
            "qid": ranked[0].get("qid", "") if ranked else "",
            "num_candidates": len(ranked),
            "true_relevant_count": true_count,
            "candidate_relevant_count": candidate_relevant_count,
            "IdealTopK": ideal_top_k,
            "CandidateRecall": candidate_relevant_count / true_count if true_count else 0.0,
            "IdealTopKSelectedDocIds": ",".join(ideal_selected_ids),
            "IdealTopKHitCount": ideal_hit_count,
            "Precision@IdealTopK": ideal_precision,
            "Recall@IdealTopK": ideal_recall,
            "F1@IdealTopK": ideal_f1,
        }
        for beta in betas:
            best_k, expected_fbeta = choose_expected_fbeta_best_k(score_list, beta=beta)
            selected = ranked[:best_k]
            selected_ids = [str(row["doc_id"]) for row in selected]
            hit_count = len(set(selected_ids) & relevant_ids)
            precision = hit_count / len(selected_ids) if selected_ids else 0.0
            recall = hit_count / true_count if true_count else 0.0
            per_query_rows.append(
                {
                    **base,
                    "beta": beta,
                    "BestK@ExpectedFbeta": best_k,
                    "ExpectedFbeta@BestK": expected_fbeta,
                    "selected_doc_ids": ",".join(selected_ids),
                    "hit_count": hit_count,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1_from_precision_recall(precision, recall),
                }
            )

    denom = max(1, len(grouped))
    overall_updates["AvgIdealTopK"] = ideal_top_k_sum / denom
    overall_updates["CandidateRecall"] = candidate_relevant_sum / true_count_sum if true_count_sum else 0.0
    overall_updates["Precision@IdealTopK"] = ideal_topk_precision_sum / denom
    overall_updates["Recall@IdealTopK"] = ideal_topk_recall_sum / denom
    overall_updates["F1@IdealTopK"] = ideal_topk_f1_sum / denom
    micro_ideal_precision = ideal_topk_hit_sum / ideal_topk_selected_sum if ideal_topk_selected_sum else 0.0
    micro_ideal_recall = ideal_topk_hit_sum / ideal_topk_true_sum if ideal_topk_true_sum else 0.0
    overall_updates["MicroPrecision@IdealTopK"] = micro_ideal_precision
    overall_updates["MicroRecall@IdealTopK"] = micro_ideal_recall
    overall_updates["MicroF1@IdealTopK"] = f1_from_precision_recall(micro_ideal_precision, micro_ideal_recall)

    for beta in betas:
        beta_rows = [row for row in per_query_rows if float(row["beta"]) == float(beta)]
        beta_denom = max(1, len(beta_rows))
        total_selected = sum(int(row["BestK@ExpectedFbeta"]) for row in beta_rows)
        total_hits = sum(int(row["hit_count"]) for row in beta_rows)
        total_true = sum(int(row["true_relevant_count"]) for row in beta_rows)
        micro_precision = total_hits / total_selected if total_selected else 0.0
        micro_recall = total_hits / total_true if total_true else 0.0
        beta_key = str(beta).replace(".", "_")
        summary = {
            "beta": beta,
            "num_queries": len(beta_rows),
            "avg_ideal_top_k": sum(float(row["IdealTopK"]) for row in beta_rows) / beta_denom,
            "avg_best_k": sum(float(row["BestK@ExpectedFbeta"]) for row in beta_rows) / beta_denom,
            "avg_expected_fbeta": sum(float(row["ExpectedFbeta@BestK"]) for row in beta_rows) / beta_denom,
            "avg_precision": sum(float(row["precision"]) for row in beta_rows) / beta_denom,
            "avg_recall": sum(float(row["recall"]) for row in beta_rows) / beta_denom,
            "avg_f1": sum(float(row["f1"]) for row in beta_rows) / beta_denom,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": f1_from_precision_recall(micro_precision, micro_recall),
        }
        summary_rows.append(summary)
        for key, value in summary.items():
            if key not in {"beta", "num_queries"}:
                overall_updates[f"beta_{beta_key}_{key}"] = float(value)
    return per_query_rows, summary_rows, overall_updates


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_progress_callback(progress_file: str):
    if not progress_file:
        return None
    path = Path(progress_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")

    def callback(event: dict) -> None:
        event = dict(event)
        event["time"] = time.time()
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    return callback


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    examples = load_examples(args.test_file)
    raw_records = read_json_records(args.test_file)
    instruction = choose_instruction(args.instruction, examples)
    llm = create_vllm_llm(args)

    queries = [ex.query for ex in examples]
    docs = [ex.doc for ex in examples]
    logger.info("Scoring %d query-doc pairs with vLLM batch_size=%d", len(examples), args.batch_size)
    start_time = time.perf_counter()
    scores = score_with_vllm(
        llm,
        queries=queries,
        documents=docs,
        batch_size=args.batch_size,
        instruction=instruction,
        sort_by_length=args.sort_by_length,
        sort_descending=args.sort_descending,
        progress_callback=build_progress_callback(args.progress_file),
    )
    score_time = time.perf_counter() - start_time
    sec_per_example = score_time / max(1, len(scores))
    examples_per_sec = len(scores) / score_time if score_time > 0 else 0.0

    rows = []
    for ex, score, raw in zip(examples, scores, raw_records, strict=False):
        rows.append(
            {
                "group_key": ex.group_key,
                "query": ex.query,
                "query_id": ex.query_id,
                "qid": raw.get("qid", ""),
                "doc_id": ex.doc_id,
                "doc": ex.doc,
                "label": ex.label,
                "raw_label": ex.raw_label,
                "score": float(score),
                "retrieval_rank": raw.get("retrieval_rank"),
                "retrieval_score": raw.get("retrieval_score"),
                "true_relevant_count": raw.get("true_relevant_count"),
                "candidate_relevant_count": raw.get("candidate_relevant_count"),
                "reason": ex.reason,
            }
        )
    rows = add_group_ranks(rows, query_key="group_key")
    overall, per_query = compute_all_metrics(
        rows,
        query_key="group_key",
        relevance_threshold=args.relevance_threshold,
    )
    beta_per_query, beta_summary, beta_overall = compute_dynamic_beta_metrics(
        rows,
        betas=args.expected_fbeta_betas,
        relevance_threshold=args.relevance_threshold,
    )
    overall.update(
        {
            "backend": "vllm",
            "vllm_runner": "pooling",
            "vllm_version": getattr(llm, "_memranker_vllm_version", "unknown"),
            "vllm_tokenizer_path": getattr(llm, "_memranker_vllm_tokenizer_path", ""),
            "model_path": args.model_path,
            "test_file": args.test_file,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "sort_by_length": args.sort_by_length,
            "sort_descending": args.sort_descending,
            "local_files_only": args.local_files_only,
            "expected_fbeta_betas": args.expected_fbeta_betas,
            "score_time_seconds": float(score_time),
            "seconds_per_example": float(sec_per_example),
            "examples_per_second": float(examples_per_sec),
        }
    )
    overall.update(beta_overall)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "per_query_metrics.jsonl", per_query)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_jsonl(output_dir / "beta_f1_per_query.jsonl", beta_per_query)
    write_csv(output_dir / "beta_f1_summary.csv", beta_summary)
    (output_dir / "beta_f1_summary.json").write_text(
        json.dumps(beta_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Wrote vLLM JSONL evaluation outputs to %s", output_dir)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
