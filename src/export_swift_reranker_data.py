from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from data import RerankerExample, load_examples


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export query-doc-score data to ms-swift native reranker listwise "
            "format: messages + positive_messages + negative_messages."
        )
    )
    parser.add_argument("--input_file", required=True, help="Source JSON/JSONL data.")
    parser.add_argument("--output_file", required=True, help="Output ms-swift JSONL file.")
    parser.add_argument("--stats_file", default=None, help="Optional JSON stats output.")
    parser.add_argument("--default_instruction", default="", help="Fallback instruction if an example has none.")
    parser.add_argument(
        "--positive_threshold",
        type=float,
        default=0.7,
        help="Normalized label threshold for positives. Original label 7 equals 0.7.",
    )
    parser.add_argument(
        "--negative_threshold",
        type=float,
        default=None,
        help=(
            "Optional normalized threshold for negatives. Defaults to values below "
            "--positive_threshold."
        ),
    )
    parser.add_argument(
        "--positive_strategy",
        choices=["threshold", "top1", "threshold_or_top1"],
        default="threshold_or_top1",
        help="How to choose positives from soft labels inside each query group.",
    )
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=2,
        help="Minimum number of docs in a query group before export.",
    )
    parser.add_argument(
        "--max_positive_messages",
        type=int,
        default=0,
        help="Optional cap for exported positives per query. 0 keeps all.",
    )
    parser.add_argument(
        "--max_negative_messages",
        type=int,
        default=0,
        help="Optional cap for exported negatives per query. 0 keeps all.",
    )
    parser.add_argument(
        "--sort_by_label",
        action="store_true",
        help="Sort positives and negatives by label descending before optional caps.",
    )
    parser.add_argument(
        "--include_debug_fields",
        action="store_true",
        help="Keep group key and selected raw labels in exported rows for inspection.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print/export stats; do not write the JSONL dataset.",
    )
    return parser.parse_args()


def group_examples(examples: list[RerankerExample]) -> list[list[RerankerExample]]:
    groups: dict[str, list[RerankerExample]] = {}
    for ex in examples:
        groups.setdefault(ex.group_key, []).append(ex)
    return list(groups.values())


def choose_instruction(group: list[RerankerExample], default_instruction: str) -> str:
    values = [ex.instruction.strip() for ex in group if ex.instruction.strip()]
    if not values:
        return default_instruction.strip()
    return Counter(values).most_common(1)[0][0]


def cap_examples(examples: list[RerankerExample], cap: int) -> list[RerankerExample]:
    if cap <= 0 or len(examples) <= cap:
        return examples
    return examples[:cap]


def select_positive_negative(
    group: list[RerankerExample],
    positive_threshold: float,
    negative_threshold: float | None,
    positive_strategy: str,
    sort_by_label: bool,
    max_positive_messages: int,
    max_negative_messages: int,
) -> tuple[list[RerankerExample], list[RerankerExample]]:
    if sort_by_label:
        group = sorted(group, key=lambda ex: ex.label, reverse=True)

    positives: list[RerankerExample]
    if positive_strategy == "top1":
        positives = [max(group, key=lambda ex: ex.label)]
    else:
        positives = [ex for ex in group if ex.label >= positive_threshold]
        if not positives and positive_strategy == "threshold_or_top1":
            positives = [max(group, key=lambda ex: ex.label)]

    positive_ids = {id(ex) for ex in positives}
    neg_threshold = positive_threshold if negative_threshold is None else negative_threshold
    negatives = [
        ex
        for ex in group
        if id(ex) not in positive_ids and ex.label < neg_threshold
    ]
    if not negatives:
        negatives = [ex for ex in group if id(ex) not in positive_ids]

    positives = cap_examples(positives, max_positive_messages)
    positive_ids = {id(ex) for ex in positives}
    negatives = [ex for ex in negatives if id(ex) not in positive_ids]
    negatives = cap_examples(negatives, max_negative_messages)
    return positives, negatives


def message_sequence(role: str, content: str) -> list[dict[str, str]]:
    return [{"role": role, "content": content.strip()}]


def build_swift_row(
    group: list[RerankerExample],
    positives: list[RerankerExample],
    negatives: list[RerankerExample],
    default_instruction: str,
    include_debug_fields: bool,
) -> dict[str, Any]:
    instruction = choose_instruction(group, default_instruction)
    query = group[0].query
    messages = []
    if instruction:
        messages.append({"role": "system", "content": instruction})
    messages.append({"role": "user", "content": query})

    row: dict[str, Any] = {
        "messages": messages,
        "positive_messages": [message_sequence("assistant", ex.doc) for ex in positives],
        "negative_messages": [message_sequence("assistant", ex.doc) for ex in negatives],
    }
    if include_debug_fields:
        row.update(
            {
                "group_key": group[0].group_key,
                "query_id": group[0].query_id,
                "positive_labels": [ex.raw_label for ex in positives],
                "negative_labels": [ex.raw_label for ex in negatives],
            }
        )
    return row


def export_swift_rows(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    examples = load_examples(
        args.input_file,
        default_instruction=args.default_instruction,
        require_label=True,
    )
    groups = group_examples(examples)
    rows: list[dict[str, Any]] = []
    skipped_small = 0
    skipped_no_pos = 0
    skipped_no_neg = 0
    group_sizes = []
    positive_sizes = []
    negative_sizes = []

    for group in tqdm(groups, desc="Exporting swift groups", unit="query", dynamic_ncols=True, ascii=True):
        if len(group) < args.min_group_size:
            skipped_small += 1
            continue
        positives, negatives = select_positive_negative(
            group,
            positive_threshold=args.positive_threshold,
            negative_threshold=args.negative_threshold,
            positive_strategy=args.positive_strategy,
            sort_by_label=args.sort_by_label,
            max_positive_messages=args.max_positive_messages,
            max_negative_messages=args.max_negative_messages,
        )
        if not positives:
            skipped_no_pos += 1
            continue
        if not negatives:
            skipped_no_neg += 1
            continue
        rows.append(
            build_swift_row(
                group,
                positives,
                negatives,
                default_instruction=args.default_instruction,
                include_debug_fields=args.include_debug_fields,
            )
        )
        group_sizes.append(len(group))
        positive_sizes.append(len(positives))
        negative_sizes.append(len(negatives))

    stats = {
        "input_file": str(args.input_file),
        "output_file": str(args.output_file),
        "num_examples": len(examples),
        "num_query_groups": len(groups),
        "num_exported_groups": len(rows),
        "skipped_small_groups": skipped_small,
        "skipped_no_positive": skipped_no_pos,
        "skipped_no_negative": skipped_no_neg,
        "positive_threshold": args.positive_threshold,
        "negative_threshold": args.negative_threshold,
        "positive_strategy": args.positive_strategy,
        "min_group_size": args.min_group_size,
        "avg_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
        "avg_positive_messages": float(np.mean(positive_sizes)) if positive_sizes else 0.0,
        "avg_negative_messages": float(np.mean(negative_sizes)) if negative_sizes else 0.0,
        "max_group_size": int(max(group_sizes)) if group_sizes else 0,
        "max_positive_messages": int(max(positive_sizes)) if positive_sizes else 0,
        "max_negative_messages": int(max(negative_sizes)) if negative_sizes else 0,
    }
    return rows, stats


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_stats(path: str | Path, stats: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.positive_threshold < 0 or args.positive_threshold > 1:
        raise ValueError("--positive_threshold must be normalized to [0, 1].")
    if args.negative_threshold is not None and not (0 <= args.negative_threshold <= 1):
        raise ValueError("--negative_threshold must be normalized to [0, 1].")
    if args.min_group_size < 2:
        raise ValueError("Swift listwise reranker training needs --min_group_size >= 2.")

    rows, stats = export_swift_rows(args)
    logger.info("Swift export stats: %s", json.dumps(stats, ensure_ascii=False))
    if not rows:
        raise ValueError("No Swift listwise rows were exported. Check thresholds and query grouping.")

    stats_file = args.stats_file or str(Path(args.output_file).with_suffix(".stats.json"))
    if not args.dry_run:
        write_jsonl(args.output_file, rows)
        write_stats(stats_file, stats)
        logger.info("Wrote %d rows to %s", len(rows), args.output_file)
        logger.info("Wrote stats to %s", stats_file)


if __name__ == "__main__":
    main()
