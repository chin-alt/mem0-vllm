from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

PREFERRED_COLUMNS = [
    "dataset",
    "model",
    "run_name",
    "output_dir",
    "backend",
    "model_path",
    "test_file",
    "max_length",
    "batch_size",
    "precision",
    "dtype",
    "attn_implementation",
    "score_activation",
    "score_time_seconds",
    "seconds_per_example",
    "examples_per_second",
    "cuda_peak_allocated_mib",
    "cuda_peak_reserved_mib",
    "MAP",
    "MRR",
    "NDCG@1",
    "NDCG@3",
    "NDCG@10",
    "Recall@1",
    "Recall@3",
    "Recall@5",
    "Pearson",
    "Spearman",
    "CandidateRecall",
    "Precision@IdealTopK",
    "Recall@IdealTopK",
    "F1@IdealTopK",
    "MicroF1@IdealTopK",
    "beta_0_2_avg_f1",
    "beta_0_3_avg_f1",
    "beta_0_5_avg_f1",
    "beta_0_7_avg_f1",
    "beta_1_0_avg_f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize JSONL reranker matrix overall_metrics.json files.")
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--summary_csv", default=None)
    parser.add_argument("--summary_json", default=None)
    parser.add_argument("--summary_xlsx", default=None)
    return parser.parse_args()


def split_run_name(run_name: str) -> tuple[str, str]:
    if "__" not in run_name:
        return "", run_name
    return run_name.split("__", 1)


def jsonable_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def collect_rows(output_root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    columns = list(PREFERRED_COLUMNS)
    for metrics_path in sorted(output_root.glob("*/overall_metrics.json")):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skipped unreadable metrics file %s: %s", metrics_path, exc)
            continue
        run_name = metrics_path.parent.name
        dataset, model = split_run_name(run_name)
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "run_name": run_name,
            "output_dir": str(metrics_path.parent),
        }
        for key, value in metrics.items():
            row[key] = jsonable_cell(value)
            if key not in columns:
                columns.append(key)
        rows.append(row)
    return rows, columns


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_xlsx(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> bool:
    try:
        from openpyxl import Workbook
    except ImportError:
        logger.warning("openpyxl is not installed; skipped xlsx matrix summary output.")
        return False
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "summary"
    sheet.append(columns)
    for row in rows:
        sheet.append([row.get(col, "") for col in columns])
    for column_cells in sheet.columns:
        max_length = max(len(str(cell.value or "")) for cell in column_cells)
        sheet.column_dimensions[column_cells[0].column_letter].width = min(max(12, max_length + 2), 72)
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook.save(path)
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    output_root = Path(args.output_root)
    rows, columns = collect_rows(output_root)
    if not rows:
        raise ValueError(f"No overall_metrics.json files found under {output_root}")
    summary_csv = Path(args.summary_csv) if args.summary_csv else output_root / "summary_metrics.csv"
    summary_json = Path(args.summary_json) if args.summary_json else output_root / "summary_metrics.json"
    summary_xlsx = Path(args.summary_xlsx) if args.summary_xlsx else output_root / "summary_metrics.xlsx"
    write_csv(summary_csv, rows, columns)
    summary_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    wrote_xlsx = write_xlsx(summary_xlsx, rows, columns)
    print(
        json.dumps(
            {
                "num_runs": len(rows),
                "summary_csv": str(summary_csv),
                "summary_json": str(summary_json),
                "summary_xlsx": str(summary_xlsx) if wrote_xlsx else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
