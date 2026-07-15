from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


logger = logging.getLogger(__name__)

DEFAULT_META_FIELDS = (
    "dataset",
    "sample_id",
    "qid",
    "category",
    "evidence",
    "positive_doc_ids",
    "doc_id",
    "dia_id",
    "session",
    "session_time",
    "speaker",
    "retrieval_rank",
    "retrieval_score",
    "retrieval_backend",
    "true_relevant_count",
    "candidate_relevant_count",
    "reason",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze LoCoMo reranker evaluation outputs by query and category, "
            "and export detailed bad cases."
        )
    )
    parser.add_argument(
        "--eval_dir",
        default="",
        help="Evaluation output directory containing predictions.jsonl. Used if --predictions_file is omitted.",
    )
    parser.add_argument("--predictions_file", default="", help="Path to predictions.jsonl.")
    parser.add_argument(
        "--test_file",
        default="",
        help=(
            "Original LoCoMo candidate JSONL. Recommended because older predictions.jsonl "
            "may not contain category/evidence/sample metadata."
        ),
    )
    parser.add_argument("--output_dir", default="", help="Defaults to <eval_dir>/locomo_analysis.")
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=10, help="Top documents kept in bad-case JSONL.")
    parser.add_argument("--bad_ndcg_k", type=int, default=10)
    parser.add_argument("--bad_ndcg_threshold", type=float, default=0.5)
    parser.add_argument(
        "--bad_rank_threshold",
        type=int,
        default=10,
        help="Mark a query bad if its first relevant document rank is greater than this value.",
    )
    parser.add_argument("--doc_snippet_chars", type=int, default=320)
    parser.add_argument(
        "--category_map_file",
        default="",
        help="Optional JSON mapping from raw LoCoMo category values to display names.",
    )
    return parser.parse_args()


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
                if isinstance(value, dict):
                    rows.append(value)
        return rows
    with path.open("r", encoding="utf-8-sig") as f:
        value = json.load(f)
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("data", "records", "items", "examples"):
            rows = value.get(key)
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        return [value]
    raise ValueError(f"Unsupported JSON root in {path}: {type(value).__name__}")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        if not value.strip():
            return []
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [stringify(item) for item in parsed if stringify(item)]
        return [part.strip() for part in value.split(",") if part.strip()]
    return [stringify(value)]


def row_group_key(row: dict[str, Any]) -> str:
    return str(row.get("group_key") or row.get("query_id") or row.get("qid") or row.get("query") or "")


def row_pair_key(row: dict[str, Any]) -> tuple[str, str]:
    return row_group_key(row), str(row.get("doc_id") or row.get("dia_id") or row.get("doc") or "")


def row_pair_keys(row: dict[str, Any]) -> list[tuple[str, str]]:
    group = row_group_key(row)
    keys: list[tuple[str, str]] = []
    for field in ("doc_id", "dia_id", "doc"):
        value = row.get(field)
        if value not in (None, ""):
            keys.append((group, stringify(value)))
    return keys


def enrich_predictions_with_test_rows(
    predictions: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not test_rows:
        return predictions
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in test_rows:
        for key in row_pair_keys(row):
            by_pair.setdefault(key, row)
    enriched: list[dict[str, Any]] = []
    missing = 0
    for row in predictions:
        merged = dict(row)
        source = None
        for key in row_pair_keys(row):
            source = by_pair.get(key)
            if source is not None:
                break
        if source is None:
            missing += 1
        else:
            for field in DEFAULT_META_FIELDS:
                if field not in merged or merged.get(field) in (None, ""):
                    if field in source:
                        merged[field] = source.get(field)
            if "label" not in merged and "labels" in source:
                merged["label"] = as_float(source.get("labels")) / 10.0
            if "raw_label" not in merged and "labels" in source:
                merged["raw_label"] = source.get("labels")
        enriched.append(merged)
    if missing:
        logger.warning("Could not match %d prediction rows to --test_file metadata.", missing)
    return enriched


def load_category_map(path: str) -> dict[str, str]:
    if not path:
        return {}
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("--category_map_file must contain a JSON object")
    return {str(key): str(item) for key, item in value.items()}


def category_name(raw_category: Any, category_map: dict[str, str]) -> str:
    raw = stringify(raw_category)
    if not raw:
        return "unknown"
    return category_map.get(raw, raw)


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


def snippet(text: Any, max_chars: int) -> str:
    clean = " ".join(stringify(text).split())
    if max_chars <= 0 or len(clean) <= max_chars:
        return clean
    return clean[: max_chars - 3] + "..."


def summarize_query(
    group_key: str,
    rows: list[dict[str, Any]],
    relevance_threshold: float,
    category_map: dict[str, str],
    bad_ndcg_k: int,
    bad_ndcg_threshold: float,
    bad_rank_threshold: int,
    top_k: int,
    doc_snippet_chars: int,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    ranked = sorted(
        rows,
        key=lambda row: (
            as_int(row.get("rank_by_score"), 10**9),
            -as_float(row.get("score")),
            as_int(row.get("retrieval_rank"), 10**9),
        ),
    )
    labels = np.asarray([as_float(row.get("label", as_float(row.get("labels")) / 10.0)) for row in ranked], dtype=np.float64)
    scores = np.asarray([as_float(row.get("score")) for row in ranked], dtype=np.float64)
    order_score = np.argsort(-scores, kind="mergesort")
    order_label = np.argsort(-labels, kind="mergesort")
    relevant = labels >= relevance_threshold
    relevant_positions = [idx + 1 for idx, flag in enumerate(relevant.tolist()) if flag]

    true_counts = [as_int(row.get("true_relevant_count")) for row in ranked]
    true_relevant_count = max(true_counts) if true_counts else int(np.sum(relevant))
    if true_relevant_count <= 0:
        true_relevant_count = int(np.sum(relevant))
    candidate_relevant_count = int(np.sum(relevant))
    ndcg_values: dict[int, float] = {}
    for k in (1, 3, 5, 10, bad_ndcg_k):
        ideal = dcg(labels, order_label, k)
        ndcg_values[k] = 0.0 if ideal <= 0 else dcg(labels, order_score, k) / ideal

    recall_values = {
        k: recall_at_k(relevant, order_score, k, true_relevant_count)
        for k in (1, 3, 5, 10)
    }
    positive_rows = [row for row, flag in zip(ranked, relevant.tolist()) if flag]
    positive_doc_ids = [str(row.get("doc_id", "")) for row in positive_rows if row.get("doc_id") not in (None, "")]
    positive_dia_ids = [str(row.get("dia_id", "")) for row in positive_rows if row.get("dia_id") not in (None, "")]
    top_docs = []
    for rank, row in enumerate(ranked[:top_k], start=1):
        is_relevant = as_float(row.get("label", as_float(row.get("labels")) / 10.0)) >= relevance_threshold
        top_docs.append(
            {
                "rank": rank,
                "doc_id": row.get("doc_id", ""),
                "dia_id": row.get("dia_id", ""),
                "score": as_float(row.get("score")),
                "label": as_float(row.get("label", as_float(row.get("labels")) / 10.0)),
                "is_relevant": is_relevant,
                "retrieval_rank": row.get("retrieval_rank", ""),
                "session": row.get("session", ""),
                "session_time": row.get("session_time", ""),
                "speaker": row.get("speaker", ""),
                "doc_snippet": snippet(row.get("doc", ""), doc_snippet_chars),
            }
        )

    first = ranked[0] if ranked else {}
    raw_category = first.get("category", "")
    best_positive_rank = min(relevant_positions) if relevant_positions else 0
    first_stage_miss = candidate_relevant_count == 0 and true_relevant_count > 0
    ndcg_bad = ndcg_values.get(bad_ndcg_k, 0.0) < bad_ndcg_threshold
    rank_bad = bool(best_positive_rank == 0 or best_positive_rank > bad_rank_threshold)
    bad_reasons: list[str] = []
    if first_stage_miss:
        bad_reasons.append("first_stage_miss")
    if ndcg_bad:
        bad_reasons.append(f"ndcg@{bad_ndcg_k}<{bad_ndcg_threshold:g}")
    if rank_bad:
        bad_reasons.append(f"best_positive_rank>{bad_rank_threshold}")

    summary = {
        "group_key": group_key,
        "query_id": first.get("query_id", ""),
        "qid": first.get("qid", ""),
        "sample_id": first.get("sample_id", ""),
        "category": raw_category,
        "category_name": category_name(raw_category, category_map),
        "query": first.get("query", ""),
        "num_candidates": len(ranked),
        "true_relevant_count": true_relevant_count,
        "candidate_relevant_count": candidate_relevant_count,
        "CandidateRecall": candidate_relevant_count / true_relevant_count if true_relevant_count else 0.0,
        "AP": average_precision(relevant, order_score),
        "MRR": reciprocal_rank(relevant, order_score),
        "NDCG@1": ndcg_values[1],
        "NDCG@3": ndcg_values[3],
        "NDCG@5": ndcg_values[5],
        "NDCG@10": ndcg_values[10],
        f"NDCG@{bad_ndcg_k}": ndcg_values[bad_ndcg_k],
        "Recall@1": recall_values[1],
        "Recall@3": recall_values[3],
        "Recall@5": recall_values[5],
        "Recall@10": recall_values[10],
        "best_positive_rank": best_positive_rank,
        "positive_ranks": ",".join(str(rank) for rank in relevant_positions),
        "positive_doc_ids": ",".join(positive_doc_ids),
        "positive_dia_ids": ",".join(positive_dia_ids),
        "evidence": ",".join(normalize_list(first.get("evidence"))),
        "top1_doc_id": ranked[0].get("doc_id", "") if ranked else "",
        "top1_dia_id": ranked[0].get("dia_id", "") if ranked else "",
        "top1_score": as_float(ranked[0].get("score")) if ranked else 0.0,
        "top1_label": as_float(ranked[0].get("label", 0.0)) if ranked else 0.0,
        "top1_is_relevant": bool(relevant[0]) if len(relevant) else False,
        "top1_snippet": snippet(ranked[0].get("doc", ""), doc_snippet_chars) if ranked else "",
        "first_stage_miss": first_stage_miss,
        "is_bad_case": bool(bad_reasons),
        "bad_reasons": ",".join(bad_reasons),
    }
    bad_case = None
    if bad_reasons:
        bad_case = {
            **summary,
            "top_docs": top_docs,
            "positive_docs_in_candidates": [
                {
                    "rank": rank,
                    "doc_id": row.get("doc_id", ""),
                    "dia_id": row.get("dia_id", ""),
                    "score": as_float(row.get("score")),
                    "label": as_float(row.get("label", 0.0)),
                    "session": row.get("session", ""),
                    "session_time": row.get("session_time", ""),
                    "speaker": row.get("speaker", ""),
                    "doc_snippet": snippet(row.get("doc", ""), doc_snippet_chars),
                }
                for rank, row in zip(relevant_positions, positive_rows)
            ],
        }
    return summary, bad_case


def aggregate_by_category(query_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in query_rows:
        grouped[str(row.get("category_name", "unknown"))].append(row)
    summaries: list[dict[str, Any]] = []
    numeric_keys = (
        "CandidateRecall",
        "AP",
        "MRR",
        "NDCG@1",
        "NDCG@3",
        "NDCG@5",
        "NDCG@10",
        "Recall@1",
        "Recall@3",
        "Recall@5",
        "Recall@10",
    )
    for category, rows in sorted(grouped.items(), key=lambda item: item[0]):
        summary: dict[str, Any] = {
            "category_name": category,
            "category": rows[0].get("category", ""),
            "num_queries": len(rows),
            "num_bad_cases": sum(1 for row in rows if row.get("is_bad_case")),
            "bad_case_rate": sum(1 for row in rows if row.get("is_bad_case")) / max(1, len(rows)),
            "first_stage_miss_rate": sum(1 for row in rows if row.get("first_stage_miss")) / max(1, len(rows)),
            "top1_accuracy": sum(1 for row in rows if row.get("top1_is_relevant")) / max(1, len(rows)),
            "mean_best_positive_rank": float(np.mean([as_float(row.get("best_positive_rank")) for row in rows])),
            "median_best_positive_rank": float(np.median([as_float(row.get("best_positive_rank")) for row in rows])),
        }
        for key in numeric_keys:
            summary[f"mean_{key}"] = float(np.mean([as_float(row.get(key)) for row in rows]))
        summaries.append(summary)
    return summaries


def write_markdown_report(path: Path, category_rows: list[dict[str, Any]], bad_cases: list[dict[str, Any]], top_n: int) -> None:
    lines = ["# LoCoMo Reranker Analysis", ""]
    lines.append("## Category Summary")
    lines.append("")
    lines.append("| category | queries | bad_rate | NDCG@10 | Recall@10 | top1_acc | first_stage_miss |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in category_rows:
        lines.append(
            "| {category} | {queries} | {bad:.4f} | {ndcg:.4f} | {recall:.4f} | {top1:.4f} | {miss:.4f} |".format(
                category=row.get("category_name", ""),
                queries=row.get("num_queries", 0),
                bad=as_float(row.get("bad_case_rate")),
                ndcg=as_float(row.get("mean_NDCG@10")),
                recall=as_float(row.get("mean_Recall@10")),
                top1=as_float(row.get("top1_accuracy")),
                miss=as_float(row.get("first_stage_miss_rate")),
            )
        )
    lines.append("")
    lines.append(f"## Worst {top_n} Bad Cases")
    lines.append("")
    for idx, row in enumerate(bad_cases[:top_n], start=1):
        lines.append(f"### {idx}. {row.get('query', '')}")
        lines.append("")
        lines.append(
            f"- category: `{row.get('category_name', '')}`; reasons: `{row.get('bad_reasons', '')}`; "
            f"NDCG@10: `{as_float(row.get('NDCG@10')):.4f}`; best positive rank: `{row.get('best_positive_rank', '')}`"
        )
        lines.append(f"- evidence: `{row.get('evidence', '')}`")
        lines.append(f"- top1: `{row.get('top1_doc_id', '')}` score=`{as_float(row.get('top1_score')):.6f}`")
        lines.append(f"- top1 snippet: {row.get('top1_snippet', '')}")
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    predictions_file = Path(args.predictions_file) if args.predictions_file else Path(args.eval_dir) / "predictions.jsonl"
    if not predictions_file.exists():
        raise FileNotFoundError("Could not find predictions file. Pass --predictions_file or --eval_dir.")
    output_dir = Path(args.output_dir) if args.output_dir else predictions_file.parent / "locomo_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions = read_json_records(predictions_file)
    test_rows = read_json_records(args.test_file) if args.test_file else []
    rows = enrich_predictions_with_test_rows(predictions, test_rows)
    category_map = load_category_map(args.category_map_file)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row_group_key(row)].append(row)
    if not grouped:
        raise ValueError(f"No query groups found in {predictions_file}")

    query_summaries: list[dict[str, Any]] = []
    bad_cases: list[dict[str, Any]] = []
    for group_key, group_rows in grouped.items():
        summary, bad_case = summarize_query(
            group_key=group_key,
            rows=group_rows,
            relevance_threshold=args.relevance_threshold,
            category_map=category_map,
            bad_ndcg_k=args.bad_ndcg_k,
            bad_ndcg_threshold=args.bad_ndcg_threshold,
            bad_rank_threshold=args.bad_rank_threshold,
            top_k=args.top_k,
            doc_snippet_chars=args.doc_snippet_chars,
        )
        query_summaries.append(summary)
        if bad_case is not None:
            bad_cases.append(bad_case)

    query_summaries.sort(
        key=lambda row: (
            as_float(row.get("NDCG@10")),
            as_float(row.get("MRR")),
            -as_float(row.get("best_positive_rank")),
            str(row.get("query_id", "")),
        )
    )
    bad_cases.sort(
        key=lambda row: (
            not row.get("first_stage_miss", False),
            as_float(row.get("NDCG@10")),
            as_float(row.get("MRR")),
            -as_float(row.get("best_positive_rank")),
        )
    )
    category_summaries = aggregate_by_category(query_summaries)
    overall = {
        "predictions_file": str(predictions_file),
        "test_file": args.test_file,
        "num_pairs": len(rows),
        "num_queries": len(query_summaries),
        "num_bad_cases": len(bad_cases),
        "bad_case_rate": len(bad_cases) / max(1, len(query_summaries)),
        "relevance_threshold": args.relevance_threshold,
        "bad_ndcg_k": args.bad_ndcg_k,
        "bad_ndcg_threshold": args.bad_ndcg_threshold,
        "bad_rank_threshold": args.bad_rank_threshold,
        "mean_NDCG@10": float(np.mean([as_float(row.get("NDCG@10")) for row in query_summaries])),
        "mean_MRR": float(np.mean([as_float(row.get("MRR")) for row in query_summaries])),
        "mean_Recall@10": float(np.mean([as_float(row.get("Recall@10")) for row in query_summaries])),
        "first_stage_miss_rate": sum(1 for row in query_summaries if row.get("first_stage_miss")) / max(1, len(query_summaries)),
    }

    write_jsonl(output_dir / "locomo_query_analysis.jsonl", query_summaries)
    write_csv(output_dir / "locomo_query_analysis.csv", query_summaries)
    write_jsonl(output_dir / "locomo_bad_cases.jsonl", bad_cases)
    bad_case_flat = [{key: value for key, value in row.items() if key not in {"top_docs", "positive_docs_in_candidates"}} for row in bad_cases]
    write_csv(output_dir / "locomo_bad_cases.csv", bad_case_flat)
    write_jsonl(output_dir / "locomo_category_summary.jsonl", category_summaries)
    write_csv(output_dir / "locomo_category_summary.csv", category_summaries)
    (output_dir / "locomo_overall_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown_report(output_dir / "locomo_bad_case_report.md", category_summaries, bad_case_flat, top_n=30)

    logger.info("Wrote LoCoMo analysis to %s", output_dir)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
