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

DEFAULT_BETAS = [1.0, 0.7, 0.5, 0.3, 0.2]
SUMMARY_COLUMNS = [
    "run_name",
    "beta",
    "num_queries",
    "avg_selected_count",
    "avg_expected_fbeta",
    "avg_precision",
    "avg_recall",
    "avg_f1",
    "micro_precision",
    "micro_recall",
    "micro_f1",
    "total_selected",
    "total_true_ids",
    "total_hits",
    "predictions_file",
    "per_query_file",
]
PER_QUERY_COLUMNS = [
    "run_name",
    "beta",
    "query",
    "true_count",
    "candidate_count",
    "recalled_relevant_count",
    "best_k",
    "expected_fbeta_at_best_k",
    "selected_ids",
    "selected_id_scores",
    "hit_ids",
    "hit_count",
    "precision",
    "recall",
    "f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute dynamic Expected-Fbeta cutoffs and real F1 from existing business predictions."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--output_root",
        help="Matrix output root containing */predictions.jsonl.",
    )
    input_group.add_argument(
        "--run_dir",
        nargs="+",
        help="One or more run directories containing predictions.jsonl.",
    )
    input_group.add_argument(
        "--predictions_file",
        help="A single predictions.jsonl file.",
    )
    parser.add_argument(
        "--per_query_file",
        default=None,
        help="Optional per_query_metrics.jsonl for --predictions_file mode.",
    )
    parser.add_argument("--betas", type=float, nargs="+", default=DEFAULT_BETAS)
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory for combined files. Defaults to output_root or run_dir.",
    )
    parser.add_argument("--no_per_run_files", action="store_true")
    return parser.parse_args()


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if isinstance(row, dict):
                rows.append(row)
    return rows


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: str | Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_xlsx(path: str | Path, sheets: dict[str, tuple[list[dict[str, Any]], list[str]]]) -> bool:
    try:
        from openpyxl import Workbook
    except ImportError:
        logger.warning("openpyxl is not installed; skipped xlsx output: %s", path)
        return False

    workbook = Workbook()
    default_sheet = workbook.active
    workbook.remove(default_sheet)
    for sheet_name, (rows, columns) in sheets.items():
        sheet = workbook.create_sheet(sheet_name[:31])
        sheet.append(columns)
        for row in rows:
            sheet.append([row.get(col, "") for col in columns])
        for column_cells in sheet.columns:
            max_length = max(len(str(cell.value or "")) for cell in column_cells)
            sheet.column_dimensions[column_cells[0].column_letter].width = min(max(12, max_length + 2), 72)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return True


def join_ids(values: list[str]) -> str:
    return "，".join(values)


def format_score(score: float) -> str:
    return f"{float(score):.6f}"


def join_id_scores(rows: list[dict[str, Any]]) -> str:
    return "，".join(f"{row['doc_id']}:{format_score(float(row['score']))}" for row in rows)


def choose_expected_fbeta_best_k(score_list: list[float], beta: float) -> tuple[int, float]:
    if not score_list:
        return 0, 0.0
    scores = np.asarray(score_list, dtype=np.float64)
    norm_scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
    cum_gain = np.cumsum(norm_scores)
    total_sum = float(cum_gain[-1])
    k_array = np.arange(1, len(scores) + 1, dtype=np.float64)
    expected_fbeta = (1 + beta**2) * cum_gain / (beta**2 * total_sum + k_array)
    best_idx = int(np.argmax(expected_fbeta))
    return best_idx + 1, float(expected_fbeta[best_idx])


def f1_from_precision_recall(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def load_true_counts(per_query_file: Path | None) -> dict[str, int]:
    if per_query_file is None or not per_query_file.exists():
        return {}
    counts: dict[str, int] = {}
    for row in load_jsonl(per_query_file):
        query = str(row.get("query", ""))
        if not query:
            continue
        value = (
            row.get("正确标签数量")
            or row.get("num_gt_docs")
            or row.get("true_count")
            or row.get("gt_count")
        )
        try:
            counts[query] = int(value)
        except (TypeError, ValueError):
            continue
    return counts


def group_predictions(predictions_file: Path) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(predictions_file):
        query = str(row.get("query", ""))
        doc_id = str(row.get("doc_id", ""))
        if not query or not doc_id:
            continue
        grouped[query].append(row)

    for query, rows in grouped.items():
        rows.sort(
            key=lambda row: (
                int(row.get("rank", 10**9)),
                -float(row.get("score", 0.0)),
                int(row.get("source_rank", 10**9)),
            )
        )
    return dict(grouped)


def compute_for_run(
    run_name: str,
    predictions_file: Path,
    per_query_file: Path | None,
    betas: list[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped = group_predictions(predictions_file)
    true_counts = load_true_counts(per_query_file)
    if not true_counts:
        logger.warning(
            "No per-query true counts found for %s; recall denominators use recalled relevant count.",
            predictions_file,
        )

    per_query_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for beta in betas:
        beta_rows: list[dict[str, Any]] = []
        for query, preds in grouped.items():
            true_count = true_counts.get(
                query,
                len({str(row["doc_id"]) for row in preds if bool(row.get("is_relevant"))}),
            )
            recalled_relevant_count = len({str(row["doc_id"]) for row in preds if bool(row.get("is_relevant"))})
            score_list = [float(row.get("score", 0.0)) for row in preds]
            best_k, expected_fbeta = choose_expected_fbeta_best_k(score_list, beta=beta)
            selected = preds[:best_k]
            hit_ids = [str(row["doc_id"]) for row in selected if bool(row.get("is_relevant"))]
            hit_count = len(set(hit_ids))
            precision = hit_count / len(selected) if selected else 0.0
            recall = hit_count / true_count if true_count else 0.0
            f1 = f1_from_precision_recall(precision, recall)
            beta_rows.append(
                {
                    "run_name": run_name,
                    "beta": beta,
                    "query": query,
                    "true_count": true_count,
                    "candidate_count": len(preds),
                    "recalled_relevant_count": recalled_relevant_count,
                    "best_k": best_k,
                    "expected_fbeta_at_best_k": expected_fbeta,
                    "selected_ids": join_ids([str(row["doc_id"]) for row in selected]),
                    "selected_id_scores": join_id_scores(selected),
                    "hit_ids": join_ids(hit_ids),
                    "hit_count": hit_count,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )
        per_query_rows.extend(beta_rows)
        denom = max(1, len(beta_rows))
        total_selected = sum(int(row["best_k"]) for row in beta_rows)
        total_true = sum(int(row["true_count"]) for row in beta_rows)
        total_hits = sum(int(row["hit_count"]) for row in beta_rows)
        micro_precision = total_hits / total_selected if total_selected else 0.0
        micro_recall = total_hits / total_true if total_true else 0.0
        summary_rows.append(
            {
                "run_name": run_name,
                "beta": beta,
                "num_queries": len(beta_rows),
                "avg_selected_count": sum(float(row["best_k"]) for row in beta_rows) / denom,
                "avg_expected_fbeta": sum(float(row["expected_fbeta_at_best_k"]) for row in beta_rows) / denom,
                "avg_precision": sum(float(row["precision"]) for row in beta_rows) / denom,
                "avg_recall": sum(float(row["recall"]) for row in beta_rows) / denom,
                "avg_f1": sum(float(row["f1"]) for row in beta_rows) / denom,
                "micro_precision": micro_precision,
                "micro_recall": micro_recall,
                "micro_f1": f1_from_precision_recall(micro_precision, micro_recall),
                "total_selected": total_selected,
                "total_true_ids": total_true,
                "total_hits": total_hits,
                "predictions_file": str(predictions_file),
                "per_query_file": str(per_query_file) if per_query_file else "",
            }
        )
    return summary_rows, per_query_rows


def discover_runs(args: argparse.Namespace) -> list[tuple[str, Path, Path | None, Path]]:
    runs: list[tuple[str, Path, Path | None, Path]] = []
    if args.output_root:
        output_root = Path(args.output_root)
        for predictions_file in sorted(output_root.glob("*/predictions.jsonl")):
            run_dir = predictions_file.parent
            runs.append((run_dir.name, predictions_file, run_dir / "per_query_metrics.jsonl", run_dir))
    elif args.run_dir:
        for raw_run_dir in args.run_dir:
            run_dir = Path(raw_run_dir)
            runs.append((run_dir.name, run_dir / "predictions.jsonl", run_dir / "per_query_metrics.jsonl", run_dir))
    else:
        predictions_file = Path(args.predictions_file)
        per_query_file = Path(args.per_query_file) if args.per_query_file else predictions_file.parent / "per_query_metrics.jsonl"
        runs.append((predictions_file.parent.name, predictions_file, per_query_file, predictions_file.parent))

    missing = [str(path) for _, path, _, _ in runs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing predictions files: {missing}")
    return runs


def write_run_outputs(run_dir: Path, summary_rows: list[dict[str, Any]], per_query_rows: list[dict[str, Any]]) -> None:
    write_csv(run_dir / "beta_f1_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_json(run_dir / "beta_f1_summary.json", summary_rows)
    write_jsonl(run_dir / "beta_f1_per_query.jsonl", per_query_rows)
    write_csv(run_dir / "beta_f1_per_query.csv", per_query_rows, PER_QUERY_COLUMNS)
    write_xlsx(
        run_dir / "beta_f1.xlsx",
        {
            "summary": (summary_rows, SUMMARY_COLUMNS),
            "per_query": (per_query_rows, PER_QUERY_COLUMNS),
        },
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    runs = discover_runs(args)
    if not runs:
        raise ValueError(
            "No predictions.jsonl files found. For matrix mode, pass the directory that contains "
            "per-run subdirectories like 0428caption__model_name/predictions.jsonl."
        )
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.output_root or runs[0][3])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_summary_rows: list[dict[str, Any]] = []
    all_per_query_rows: list[dict[str, Any]] = []
    for run_name, predictions_file, per_query_file, run_dir in runs:
        logger.info("Recomputing beta F1 for run=%s predictions=%s", run_name, predictions_file)
        summary_rows, per_query_rows = compute_for_run(
            run_name,
            predictions_file,
            per_query_file,
            betas=args.betas,
        )
        all_summary_rows.extend(summary_rows)
        all_per_query_rows.extend(per_query_rows)
        if not args.no_per_run_files:
            write_run_outputs(run_dir, summary_rows, per_query_rows)

    write_csv(output_dir / "beta_f1_matrix_summary.csv", all_summary_rows, SUMMARY_COLUMNS)
    write_json(output_dir / "beta_f1_matrix_summary.json", all_summary_rows)
    write_jsonl(output_dir / "beta_f1_matrix_per_query.jsonl", all_per_query_rows)
    wrote_xlsx = write_xlsx(
        output_dir / "beta_f1_matrix.xlsx",
        {
            "summary": (all_summary_rows, SUMMARY_COLUMNS),
            "per_query": (all_per_query_rows, PER_QUERY_COLUMNS),
        },
    )
    logger.info("Wrote beta F1 outputs to %s", output_dir)
    print(
        json.dumps(
            {
                "num_runs": len(runs),
                "betas": args.betas,
                "summary_csv": str(output_dir / "beta_f1_matrix_summary.csv"),
                "summary_json": str(output_dir / "beta_f1_matrix_summary.json"),
                "summary_xlsx": str(output_dir / "beta_f1_matrix.xlsx") if wrote_xlsx else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
