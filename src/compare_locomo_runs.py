from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from analyze_locomo_results import enrich_predictions_with_test_rows
from data import read_json_records, write_jsonl


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two LoCoMo reranker evaluation runs query by query. "
            "Useful for finding cases where one model fixes another model's bad cases."
        )
    )
    parser.add_argument("--better_eval_dir", required=True, help="Evaluation dir for the model expected to be better.")
    parser.add_argument("--worse_eval_dir", required=True, help="Evaluation dir for the baseline/model to compare against.")
    parser.add_argument("--better_name", default="mem_reranker")
    parser.add_argument("--worse_name", default="qwen_soft_label")
    parser.add_argument("--better_predictions_file", default="", help="Override <better_eval_dir>/predictions.jsonl.")
    parser.add_argument("--worse_predictions_file", default="", help="Override <worse_eval_dir>/predictions.jsonl.")
    parser.add_argument("--test_file", default="", help="Original LoCoMo candidate JSONL for metadata enrichment.")
    parser.add_argument("--output_dir", default="", help="Defaults to <better_eval_dir>/compare_<worse_name>.")
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--ndcg_k", type=int, default=10)
    parser.add_argument("--top_k_docs", type=int, default=5)
    parser.add_argument(
        "--min_delta",
        type=float,
        default=0.2,
        help="Keep better cases whose better_NDCG@k - worse_NDCG@k is at least this value.",
    )
    parser.add_argument(
        "--worse_max_ndcg",
        type=float,
        default=0.5,
        help="Keep better cases only when worse model NDCG@k is <= this value. Set >1 to disable.",
    )
    parser.add_argument(
        "--require_better_positive_rank",
        action="store_true",
        help="Require the better model's first positive rank to be smaller than the worse model's.",
    )
    parser.add_argument("--report_top_n", type=int, default=50)
    return parser.parse_args()


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def normalize_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [stringify(item) for item in value if stringify(item)]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [stringify(item) for item in parsed if stringify(item)]
        return [part.strip() for part in text.split(",") if part.strip()]
    return [stringify(value)]


def row_group_key(row: dict[str, Any]) -> str:
    return str(row.get("group_key") or row.get("query_id") or row.get("qid") or row.get("query") or "")


def normalized_label(row: dict[str, Any]) -> float:
    if "label" in row and "labels" not in row and "raw_label" not in row:
        value = as_float(row.get("label"))
        if 0.0 <= value <= 1.0:
            return value
        return max(0.0, min(1.0, value / 10.0))
    value = row.get("labels", row.get("raw_label", row.get("label", 0.0)))
    return max(0.0, min(1.0, as_float(value) / 10.0))


def dcg(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    gains = labels[order[:k]]
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    return float(np.sum(gains * discounts))


def average_precision(relevant: np.ndarray, order: np.ndarray) -> float:
    ordered = relevant[order].astype(bool)
    total = int(np.sum(ordered))
    if total == 0:
        return 0.0
    hits = 0
    total_precision = 0.0
    for rank, is_relevant in enumerate(ordered, start=1):
        if is_relevant:
            hits += 1
            total_precision += hits / rank
    return total_precision / total


def reciprocal_rank(relevant: np.ndarray, order: np.ndarray) -> float:
    ordered = relevant[order].astype(bool)
    for rank, is_relevant in enumerate(ordered, start=1):
        if is_relevant:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevant: np.ndarray, order: np.ndarray, k: int, true_relevant_count: int) -> float:
    if true_relevant_count <= 0:
        return 0.0
    return float(np.sum(relevant[order[:k]]) / true_relevant_count)


def snippet(text: Any, max_chars: int = 260) -> str:
    clean = " ".join(stringify(text).split())
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def load_run_rows(eval_dir: str, predictions_file: str, test_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    path = Path(predictions_file) if predictions_file else Path(eval_dir) / "predictions.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions file: {path}")
    rows = read_json_records(path)
    if test_rows:
        rows = enrich_predictions_with_test_rows(rows, test_rows)
    enriched = []
    for row in rows:
        item = dict(row)
        item["group_key"] = row_group_key(item)
        item["label"] = normalized_label(item)
        enriched.append(item)
    return enriched


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_group_key(row)].append(row)
    return dict(grouped)


def summarize_group(rows: list[dict[str, Any]], relevance_threshold: float, ndcg_k: int, top_k_docs: int) -> dict[str, Any]:
    ranked = sorted(
        rows,
        key=lambda row: (
            as_int(row.get("rank_by_score"), 10**9),
            -as_float(row.get("score")),
            as_int(row.get("retrieval_rank"), 10**9),
        ),
    )
    labels = np.asarray([normalized_label(row) for row in ranked], dtype=np.float64)
    scores = np.asarray([as_float(row.get("score")) for row in ranked], dtype=np.float64)
    order_score = np.argsort(-scores, kind="mergesort")
    order_label = np.argsort(-labels, kind="mergesort")
    relevant = labels >= relevance_threshold
    true_counts = [as_int(row.get("true_relevant_count")) for row in ranked]
    true_relevant_count = max(true_counts) if true_counts else int(np.sum(relevant))
    if true_relevant_count <= 0:
        true_relevant_count = int(np.sum(relevant))
    ideal = dcg(labels, order_label, ndcg_k)
    ndcg = 0.0 if ideal <= 0 else dcg(labels, order_score, ndcg_k) / ideal
    relevant_positions = [idx + 1 for idx, flag in enumerate(relevant.tolist()) if flag]
    first_positive_rank = min(relevant_positions) if relevant_positions else 0
    positive_rows = [row for row, flag in zip(ranked, relevant.tolist(), strict=False) if flag]
    top_docs = []
    for rank, row in enumerate(ranked[:top_k_docs], start=1):
        top_docs.append(
            {
                "rank": rank,
                "doc_id": row.get("doc_id", ""),
                "dia_id": row.get("dia_id", ""),
                "score": as_float(row.get("score")),
                "label": normalized_label(row),
                "is_relevant": normalized_label(row) >= relevance_threshold,
                "retrieval_rank": row.get("retrieval_rank", ""),
                "session": row.get("session", ""),
                "speaker": row.get("speaker", ""),
                "doc_snippet": snippet(row.get("doc", "")),
            }
        )
    return {
        "query": ranked[0].get("query", "") if ranked else "",
        "query_id": ranked[0].get("query_id", "") if ranked else "",
        "qid": ranked[0].get("qid", "") if ranked else "",
        "sample_id": ranked[0].get("sample_id", "") if ranked else "",
        "category": ranked[0].get("category", "") if ranked else "",
        "evidence": normalize_list(ranked[0].get("evidence")) if ranked else [],
        "num_candidates": len(ranked),
        "true_relevant_count": true_relevant_count,
        "candidate_relevant_count": int(np.sum(relevant)),
        f"NDCG@{ndcg_k}": ndcg,
        "MRR": reciprocal_rank(relevant, order_score),
        "AP": average_precision(relevant, order_score),
        f"Recall@{ndcg_k}": recall_at_k(relevant, order_score, ndcg_k, true_relevant_count),
        "first_positive_rank": first_positive_rank,
        "positive_ranks": relevant_positions,
        "positive_doc_ids": [str(row.get("doc_id", "")) for row in positive_rows],
        "positive_dia_ids": [str(row.get("dia_id", "")) for row in positive_rows],
        "top1_doc_id": ranked[0].get("doc_id", "") if ranked else "",
        "top1_dia_id": ranked[0].get("dia_id", "") if ranked else "",
        "top1_score": as_float(ranked[0].get("score")) if ranked else 0.0,
        "top1_label": normalized_label(ranked[0]) if ranked else 0.0,
        "top_docs": top_docs,
    }


def flatten_case(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: value for key, value in row.items() if key not in {"better_top_docs", "worse_top_docs"}}
    for key in ("evidence", "better_positive_ranks", "worse_positive_ranks"):
        if isinstance(flat.get(key), list):
            flat[key] = ",".join(str(item) for item in flat[key])
    return flat


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def build_comparisons(
    better_rows: list[dict[str, Any]],
    worse_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    better_groups = group_rows(better_rows)
    worse_groups = group_rows(worse_rows)
    common_keys = sorted(set(better_groups) & set(worse_groups))
    metric_key = f"NDCG@{args.ndcg_k}"
    comparisons = []
    missing = {
        "only_in_better": len(set(better_groups) - set(worse_groups)),
        "only_in_worse": len(set(worse_groups) - set(better_groups)),
    }
    if missing["only_in_better"] or missing["only_in_worse"]:
        logger.warning("Run group mismatch: %s", missing)
    for group_key in common_keys:
        better = summarize_group(better_groups[group_key], args.relevance_threshold, args.ndcg_k, args.top_k_docs)
        worse = summarize_group(worse_groups[group_key], args.relevance_threshold, args.ndcg_k, args.top_k_docs)
        delta_ndcg = as_float(better.get(metric_key)) - as_float(worse.get(metric_key))
        delta_mrr = as_float(better.get("MRR")) - as_float(worse.get("MRR"))
        better_rank = as_int(better.get("first_positive_rank"))
        worse_rank = as_int(worse.get("first_positive_rank"))
        rank_improved = bool(better_rank > 0 and (worse_rank == 0 or better_rank < worse_rank))
        comparisons.append(
            {
                "group_key": group_key,
                "query": better.get("query") or worse.get("query"),
                "query_id": better.get("query_id") or worse.get("query_id"),
                "qid": better.get("qid") or worse.get("qid"),
                "sample_id": better.get("sample_id") or worse.get("sample_id"),
                "category": better.get("category") or worse.get("category"),
                "evidence": better.get("evidence") or worse.get("evidence"),
                "num_candidates": max(as_int(better.get("num_candidates")), as_int(worse.get("num_candidates"))),
                "true_relevant_count": max(as_int(better.get("true_relevant_count")), as_int(worse.get("true_relevant_count"))),
                "candidate_relevant_count": max(as_int(better.get("candidate_relevant_count")), as_int(worse.get("candidate_relevant_count"))),
                f"{args.better_name}_NDCG@{args.ndcg_k}": better.get(metric_key),
                f"{args.worse_name}_NDCG@{args.ndcg_k}": worse.get(metric_key),
                f"delta_NDCG@{args.ndcg_k}": delta_ndcg,
                f"{args.better_name}_MRR": better.get("MRR"),
                f"{args.worse_name}_MRR": worse.get("MRR"),
                "delta_MRR": delta_mrr,
                f"{args.better_name}_Recall@{args.ndcg_k}": better.get(f"Recall@{args.ndcg_k}"),
                f"{args.worse_name}_Recall@{args.ndcg_k}": worse.get(f"Recall@{args.ndcg_k}"),
                f"{args.better_name}_first_positive_rank": better_rank,
                f"{args.worse_name}_first_positive_rank": worse_rank,
                "rank_improved": rank_improved,
                f"{args.better_name}_top1_doc_id": better.get("top1_doc_id"),
                f"{args.worse_name}_top1_doc_id": worse.get("top1_doc_id"),
                f"{args.better_name}_top1_dia_id": better.get("top1_dia_id"),
                f"{args.worse_name}_top1_dia_id": worse.get("top1_dia_id"),
                f"{args.better_name}_top1_score": better.get("top1_score"),
                f"{args.worse_name}_top1_score": worse.get("top1_score"),
                f"{args.better_name}_top1_label": better.get("top1_label"),
                f"{args.worse_name}_top1_label": worse.get("top1_label"),
                "better_top_docs": better.get("top_docs", []),
                "worse_top_docs": worse.get("top_docs", []),
                "is_better_case": False,
            }
        )
    return comparisons


def select_better_cases(comparisons: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    delta_key = f"delta_NDCG@{args.ndcg_k}"
    worse_key = f"{args.worse_name}_NDCG@{args.ndcg_k}"
    selected = []
    for row in comparisons:
        if as_float(row.get(delta_key)) < args.min_delta:
            continue
        if args.worse_max_ndcg <= 1.0 and as_float(row.get(worse_key)) > args.worse_max_ndcg:
            continue
        if args.require_better_positive_rank and not row.get("rank_improved"):
            continue
        item = dict(row)
        item["is_better_case"] = True
        selected.append(item)
    selected.sort(
        key=lambda row: (
            -as_float(row.get(delta_key)),
            as_float(row.get(worse_key)),
            -as_float(row.get("delta_MRR")),
            str(row.get("query_id")),
        )
    )
    return selected


def write_markdown_report(path: Path, cases: list[dict[str, Any]], args: argparse.Namespace) -> None:
    metric = f"NDCG@{args.ndcg_k}"
    lines = [
        f"# LoCoMo Run Comparison: {args.better_name} Better Than {args.worse_name}",
        "",
        f"Selection: `{args.better_name}_{metric} - {args.worse_name}_{metric} >= {args.min_delta}` "
        f"and `{args.worse_name}_{metric} <= {args.worse_max_ndcg}`.",
        "",
    ]
    for idx, row in enumerate(cases[: args.report_top_n], start=1):
        lines.append(f"## {idx}. {row.get('query', '')}")
        lines.append("")
        lines.append(
            f"- category: `{row.get('category', '')}`; evidence: `{','.join(row.get('evidence', []))}`"
        )
        lines.append(
            f"- {args.better_name}: {metric}=`{as_float(row.get(f'{args.better_name}_{metric}')):.4f}`, "
            f"first_positive_rank=`{row.get(f'{args.better_name}_first_positive_rank')}`"
        )
        lines.append(
            f"- {args.worse_name}: {metric}=`{as_float(row.get(f'{args.worse_name}_{metric}')):.4f}`, "
            f"first_positive_rank=`{row.get(f'{args.worse_name}_first_positive_rank')}`"
        )
        lines.append(f"- delta_{metric}: `{as_float(row.get(f'delta_{metric}')):.4f}`")
        lines.append("")
        lines.append(f"### {args.better_name} Top Docs")
        for doc in row.get("better_top_docs", [])[: args.top_k_docs]:
            lines.append(
                f"- #{doc.get('rank')} `{doc.get('dia_id')}` score=`{as_float(doc.get('score')):.6f}` "
                f"label=`{as_float(doc.get('label')):.1f}` rel=`{doc.get('is_relevant')}`: {doc.get('doc_snippet', '')}"
            )
        lines.append("")
        lines.append(f"### {args.worse_name} Top Docs")
        for doc in row.get("worse_top_docs", [])[: args.top_k_docs]:
            lines.append(
                f"- #{doc.get('rank')} `{doc.get('dia_id')}` score=`{as_float(doc.get('score')):.6f}` "
                f"label=`{as_float(doc.get('label')):.1f}` rel=`{doc.get('is_relevant')}`: {doc.get('doc_snippet', '')}"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    test_rows = read_json_records(args.test_file) if args.test_file else []
    better_rows = load_run_rows(args.better_eval_dir, args.better_predictions_file, test_rows)
    worse_rows = load_run_rows(args.worse_eval_dir, args.worse_predictions_file, test_rows)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.better_eval_dir) / f"compare_{args.worse_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = build_comparisons(better_rows, worse_rows, args)
    better_cases = select_better_cases(comparisons, args)
    comparisons.sort(
        key=lambda row: (
            -as_float(row.get(f"delta_NDCG@{args.ndcg_k}")),
            str(row.get("query_id")),
        )
    )

    summary = {
        "better_name": args.better_name,
        "worse_name": args.worse_name,
        "better_eval_dir": args.better_eval_dir,
        "worse_eval_dir": args.worse_eval_dir,
        "test_file": args.test_file,
        "relevance_threshold": args.relevance_threshold,
        "ndcg_k": args.ndcg_k,
        "min_delta": args.min_delta,
        "worse_max_ndcg": args.worse_max_ndcg,
        "num_common_queries": len(comparisons),
        "num_better_cases": len(better_cases),
        "mean_delta_ndcg": float(np.mean([as_float(row.get(f"delta_NDCG@{args.ndcg_k}")) for row in comparisons]))
        if comparisons
        else 0.0,
        "mean_delta_mrr": float(np.mean([as_float(row.get("delta_MRR")) for row in comparisons])) if comparisons else 0.0,
        "rank_improved_count": sum(1 for row in comparisons if row.get("rank_improved")),
    }

    write_jsonl(output_dir / "query_comparison.jsonl", comparisons)
    write_csv(output_dir / "query_comparison.csv", [flatten_case(row) for row in comparisons])
    write_jsonl(output_dir / "better_cases.jsonl", better_cases)
    write_csv(output_dir / "better_cases.csv", [flatten_case(row) for row in better_cases])
    write_markdown_report(output_dir / "better_cases_report.md", better_cases, args)
    (output_dir / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info("Wrote LoCoMo comparison outputs to %s", output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
