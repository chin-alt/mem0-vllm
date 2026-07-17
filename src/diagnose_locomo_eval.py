from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from data import read_json_records
from evaluate_jsonl_vllm import compute_inverted_score_diagnostics
from metrics import compute_all_metrics


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose LoCoMo reranker candidate/evaluation files. It checks label scale, "
            "candidate evidence coverage, retrieval-rank baseline, model-score metrics, "
            "and score direction."
        )
    )
    parser.add_argument("--test_file", required=True, help="LoCoMo candidate JSONL.")
    parser.add_argument("--predictions_file", default="", help="Optional predictions.jsonl from an evaluator.")
    parser.add_argument("--eval_dir", default="", help="Directory containing predictions.jsonl if --predictions_file is omitted.")
    parser.add_argument("--output_file", default="", help="Defaults to <eval_dir>/locomo_diagnostics.json or test-file sidecar.")
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--ndcg_k", type=int, default=10)
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


def group_key(row: dict[str, Any]) -> str:
    return str(row.get("group_key") or row.get("query_id") or row.get("qid") or row.get("query") or "")


def pair_key(row: dict[str, Any]) -> tuple[str, str]:
    doc_key = row.get("doc_id") or row.get("dia_id") or row.get("doc") or ""
    return group_key(row), stringify(doc_key)


def normalized_label(row: dict[str, Any]) -> float:
    if "label" in row and "labels" not in row and "raw_label" not in row:
        value = as_float(row.get("label"))
        if 0.0 <= value <= 1.0:
            return value
        return max(0.0, min(1.0, value / 10.0))
    value = row.get("labels", row.get("raw_label", row.get("label", 0.0)))
    return max(0.0, min(1.0, as_float(value) / 10.0))


def raw_label(row: dict[str, Any]) -> float:
    if "labels" in row:
        return as_float(row.get("labels"))
    if "raw_label" in row:
        return as_float(row.get("raw_label"))
    value = as_float(row.get("label"))
    if 0.0 <= value <= 1.0:
        return value * 10.0
    return value


def add_labels(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    labeled = []
    for row in rows:
        item = dict(row)
        item["label"] = normalized_label(item)
        item["raw_label"] = raw_label(item)
        item["group_key"] = group_key(item)
        labeled.append(item)
    return labeled


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[group_key(row)].append(row)
    return dict(grouped)


def dcg(labels: np.ndarray, order: np.ndarray, k: int) -> float:
    if k <= 0:
        return 0.0
    gains = labels[order[:k]]
    if gains.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, gains.size + 2, dtype=np.float64))
    return float(np.sum(gains * discounts))


def metrics_for_order(rows: list[dict[str, Any]], score_name: str, relevance_threshold: float) -> dict[str, float]:
    if not rows:
        return {}
    metric_rows = []
    for row in rows:
        item = dict(row)
        item["score"] = as_float(row.get(score_name))
        metric_rows.append(item)
    overall, _ = compute_all_metrics(metric_rows, query_key="group_key", relevance_threshold=relevance_threshold)
    return overall


def retrieval_baseline_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted = []
    for row in rows:
        item = dict(row)
        retrieval_rank = as_int(item.get("retrieval_rank"), 10**9)
        retrieval_score = item.get("retrieval_score")
        if retrieval_score not in (None, ""):
            item["retrieval_baseline_score"] = as_float(retrieval_score)
        else:
            item["retrieval_baseline_score"] = -float(retrieval_rank)
        converted.append(item)
    return converted


def summarize_candidates(rows: list[dict[str, Any]], relevance_threshold: float, ndcg_k: int) -> dict[str, Any]:
    grouped = group_rows(rows)
    total_true = 0
    total_candidate = 0
    no_positive_groups = 0
    first_stage_miss_groups = 0
    ideal_ndcg_values = []
    label_values = [raw_label(row) for row in rows]
    normalized_values = [normalized_label(row) for row in rows]

    for group in grouped.values():
        labels = np.asarray([normalized_label(row) for row in group], dtype=np.float64)
        relevant = labels >= relevance_threshold
        candidate_relevant = int(np.sum(relevant))
        true_counts = [as_int(row.get("true_relevant_count")) for row in group]
        true_relevant = max(true_counts) if true_counts else candidate_relevant
        if true_relevant <= 0:
            true_relevant = candidate_relevant
        total_true += true_relevant
        total_candidate += candidate_relevant
        no_positive_groups += int(candidate_relevant == 0)
        first_stage_miss_groups += int(candidate_relevant == 0 and true_relevant > 0)
        order_label = np.argsort(-labels, kind="mergesort")
        ideal = dcg(labels, order_label, ndcg_k)
        ideal_ndcg_values.append(0.0 if ideal <= 0 else 1.0)

    retrieval_rows = retrieval_baseline_scores(rows)
    retrieval_metrics = metrics_for_order(retrieval_rows, "retrieval_baseline_score", relevance_threshold)
    return {
        "num_pairs": len(rows),
        "num_queries": len(grouped),
        "raw_label_min": float(np.min(label_values)) if label_values else 0.0,
        "raw_label_max": float(np.max(label_values)) if label_values else 0.0,
        "normalized_label_min": float(np.min(normalized_values)) if normalized_values else 0.0,
        "normalized_label_max": float(np.max(normalized_values)) if normalized_values else 0.0,
        "positive_pairs": int(sum(1 for row in rows if normalized_label(row) >= relevance_threshold)),
        "groups_without_candidate_positive": no_positive_groups,
        "first_stage_miss_groups": first_stage_miss_groups,
        "total_true_relevant": total_true,
        "total_candidate_relevant": total_candidate,
        "candidate_recall": total_candidate / total_true if total_true else 0.0,
        "mean_oracle_candidate_ndcg_at_k": float(np.mean(ideal_ndcg_values)) if ideal_ndcg_values else 0.0,
        f"retrieval_baseline_NDCG@{ndcg_k}": float(retrieval_metrics.get(f"NDCG@{ndcg_k}", 0.0)),
        "retrieval_baseline_MRR": float(retrieval_metrics.get("MRR", 0.0)),
        "retrieval_baseline_Recall@5": float(retrieval_metrics.get("Recall@5", 0.0)),
    }


def merge_predictions_with_candidates(
    predictions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_pair = {pair_key(row): row for row in candidates}
    merged = []
    missing = 0
    for row in predictions:
        item = dict(row)
        source = by_pair.get(pair_key(item))
        if source is None:
            missing += 1
        else:
            for key in (
                "labels",
                "raw_label",
                "label",
                "query_id",
                "qid",
                "doc_id",
                "dia_id",
                "sample_id",
                "category",
                "evidence",
                "positive_doc_ids",
                "retrieval_rank",
                "retrieval_score",
                "true_relevant_count",
                "candidate_relevant_count",
            ):
                if key not in item or item.get(key) in (None, ""):
                    if key in source:
                        item[key] = source.get(key)
        item["group_key"] = group_key(item)
        item["label"] = normalized_label(item)
        item["raw_label"] = raw_label(item)
        merged.append(item)
    if missing:
        logger.warning("Could not match %d prediction rows to candidate rows.", missing)
    return merged


def summarize_predictions(rows: list[dict[str, Any]], relevance_threshold: float, ndcg_k: int) -> dict[str, Any]:
    if not rows:
        return {}
    overall, _ = compute_all_metrics(rows, query_key="group_key", relevance_threshold=relevance_threshold)
    inverted = compute_inverted_score_diagnostics(rows, relevance_threshold)
    positives = [as_float(row.get("score")) for row in rows if normalized_label(row) >= relevance_threshold]
    negatives = [as_float(row.get("score")) for row in rows if normalized_label(row) < relevance_threshold]
    pos_mean = float(np.mean(positives)) if positives else 0.0
    neg_mean = float(np.mean(negatives)) if negatives else 0.0
    return {
        f"model_NDCG@{ndcg_k}": float(overall.get(f"NDCG@{ndcg_k}", 0.0)),
        "model_MRR": float(overall.get("MRR", 0.0)),
        "model_MAP": float(overall.get("MAP", 0.0)),
        "model_Recall@5": float(overall.get("Recall@5", 0.0)),
        "model_Pearson": float(overall.get("Pearson", 0.0)),
        "model_Spearman": float(overall.get("Spearman", 0.0)),
        f"inverted_NDCG@{ndcg_k}": float(inverted.get(f"inverted_NDCG@{ndcg_k}", 0.0)),
        "inverted_MRR": float(inverted.get("inverted_MRR", 0.0)),
        "positive_score_mean": pos_mean,
        "negative_score_mean": neg_mean,
        "positive_minus_negative_score_mean": pos_mean - neg_mean,
    }


def build_suggestions(candidate_summary: dict[str, Any], prediction_summary: dict[str, Any]) -> list[str]:
    suggestions: list[str] = []
    if candidate_summary.get("candidate_recall", 1.0) < 0.5:
        suggestions.append(
            "CandidateRecall is low. Rebuild LoCoMo candidates with ENSURE_POSITIVES=1 for reranker-only evaluation, "
            "or report this as first-stage retrieval loss."
        )
    if candidate_summary.get("groups_without_candidate_positive", 0) > 0:
        suggestions.append("Some query groups have no positive candidate, so their NDCG/MRR/Recall are capped at 0.")
    if candidate_summary.get("raw_label_max", 0.0) <= 1.0:
        suggestions.append(
            "Labels look normalized already. The evaluator handles predictions, but input candidate JSONL should preferably "
            "use labels=10/0 or raw_label plus label to avoid scale ambiguity."
        )
    if prediction_summary:
        model_ndcg = prediction_summary.get("model_NDCG@10", prediction_summary.get("model_NDCG@5", 0.0))
        inverted_ndcg = prediction_summary.get("inverted_NDCG@10", prediction_summary.get("inverted_NDCG@5", 0.0))
        if inverted_ndcg > model_ndcg:
            suggestions.append("Inverted scores outperform model scores; check score activation or sign direction.")
        if prediction_summary.get("positive_minus_negative_score_mean", 0.0) < 0:
            suggestions.append("Positive examples have lower mean score than negatives; this also suggests score inversion or domain mismatch.")
    return suggestions


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    candidates = add_labels(read_json_records(args.test_file))
    candidate_summary = summarize_candidates(candidates, args.relevance_threshold, args.ndcg_k)

    predictions_file = Path(args.predictions_file) if args.predictions_file else None
    if predictions_file is None and args.eval_dir:
        predictions_file = Path(args.eval_dir) / "predictions.jsonl"
    prediction_summary: dict[str, Any] = {}
    if predictions_file and predictions_file.exists():
        predictions = read_json_records(predictions_file)
        merged_predictions = merge_predictions_with_candidates(predictions, candidates)
        prediction_summary = summarize_predictions(merged_predictions, args.relevance_threshold, args.ndcg_k)
    elif predictions_file:
        logger.warning("Predictions file does not exist: %s", predictions_file)

    report = {
        "test_file": args.test_file,
        "predictions_file": str(predictions_file) if predictions_file else "",
        "relevance_threshold": args.relevance_threshold,
        "ndcg_k": args.ndcg_k,
        "candidate_summary": candidate_summary,
        "prediction_summary": prediction_summary,
        "suggestions": build_suggestions(candidate_summary, prediction_summary),
    }

    if args.output_file:
        output_file = Path(args.output_file)
    elif args.eval_dir:
        output_file = Path(args.eval_dir) / "locomo_diagnostics.json"
    else:
        output_file = Path(args.test_file).with_suffix(".diagnostics.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote LoCoMo diagnostics to %s", output_file)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
