from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from prepare_cmteb_r import (
    DEFAULT_DATASETS,
    dataset_dir,
    find_split_files,
    positives_from_query_row,
    find_qrels_files,
    read_qrels,
    read_split,
    row_id,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect local C-MTEB Retrieval parquet/json files before conversion."
    )
    parser.add_argument("--input_dir", default="data/cmteb_r/raw")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--qrels_input_dir", default="")
    parser.add_argument("--qrels_dataset_suffix", default="-qrels")
    parser.add_argument("--sample_rows", type=int, default=2)
    parser.add_argument("--scan_queries", type=int, default=1000)
    parser.add_argument("--output_file", default="")
    return parser.parse_args()


def compact_row(row: dict[str, Any], max_chars: int = 240) -> dict[str, Any]:
    compact = {}
    for key, value in row.items():
        if isinstance(value, str) and len(value) > max_chars:
            compact[key] = value[:max_chars] + "..."
        else:
            compact[key] = value
    return compact


def inspect_dataset(
    input_dir: Path,
    dataset: str,
    sample_rows: int,
    scan_queries: int,
    qrels_input_dir: str,
    qrels_dataset_suffix: str,
) -> dict[str, Any]:
    root = dataset_dir(input_dir, dataset)
    corpus_files = [str(path) for path in find_split_files(root, "corpus")]
    query_files = [str(path) for path in find_split_files(root, "queries")]
    summary: dict[str, Any] = {
        "dataset": dataset,
        "resolved_root": str(root),
        "corpus_files": corpus_files,
        "query_files": query_files,
        "ok": False,
    }
    if not root.exists() or not corpus_files or not query_files:
        summary["error"] = "Missing corpus or queries files."
        return summary

    corpus_rows = read_split(root, "corpus")
    query_rows = read_split(root, "queries")
    qrels_files = find_qrels_files(root, dataset, qrels_input_dir, qrels_dataset_suffix)
    qrels = read_qrels(root, dataset, qrels_input_dir, qrels_dataset_suffix)
    query_scan = query_rows[: max(0, scan_queries)]
    positive_field_queries = sum(1 for row in query_scan if positives_from_query_row(row))
    corpus_ids = set()
    for row in corpus_rows:
        try:
            corpus_ids.add(row_id(row))
        except ValueError:
            continue
    query_ids = []
    for row in query_rows:
        try:
            query_ids.append(row_id(row))
        except ValueError:
            continue
    id_match_overlap = sum(1 for qid in query_ids if qid in corpus_ids)
    id_match_ratio = id_match_overlap / max(1, len(query_ids))

    summary.update(
        {
            "ok": True,
            "num_corpus_rows": len(corpus_rows),
            "num_query_rows": len(query_rows),
            "corpus_columns": list(corpus_rows[0]) if corpus_rows else [],
            "query_columns": list(query_rows[0]) if query_rows else [],
            "qrels_query_count": len(qrels),
            "qrels_files": [str(path) for path in qrels_files],
            "qrels_input_dir": qrels_input_dir,
            "qrels_dataset_suffix": qrels_dataset_suffix,
            "positive_field_queries_in_scan": positive_field_queries,
            "scan_queries": len(query_scan),
            "id_match_overlap_queries": id_match_overlap,
            "id_match_overlap_ratio": id_match_ratio,
            "has_supervision": bool(qrels) or positive_field_queries > 0 or id_match_overlap > 0,
            "recommended_supervision_strategy": (
                "explicit"
                if bool(qrels) or positive_field_queries > 0
                else "id_match"
                if id_match_ratio >= 0.8
                else "missing"
            ),
            "corpus_samples": [compact_row(row) for row in corpus_rows[:sample_rows]],
            "query_samples": [compact_row(row) for row in query_rows[:sample_rows]],
        }
    )
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    input_dir = Path(args.input_dir)
    summaries = [
        inspect_dataset(
            input_dir,
            dataset,
            args.sample_rows,
            args.scan_queries,
            args.qrels_input_dir,
            args.qrels_dataset_suffix,
        )
        for dataset in args.datasets
    ]
    payload = {"input_dir": str(input_dir), "datasets": summaries}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_file:
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(text, encoding="utf-8")
        logger.info("Wrote inspection summary to %s", output_file)
    print(text)


if __name__ == "__main__":
    main()
