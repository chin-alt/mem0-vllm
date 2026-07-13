from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from data import read_json_records, write_jsonl


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a faster reranker evaluation subset by query group. "
            "It samples groups, truncates long text fields, optionally drops overlong pairs, "
            "and writes a JSONL plus metadata."
        )
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--metadata_file", default="")
    parser.add_argument("--sample_ratio", type=float, default=0.10, help="Query-group sampling ratio.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_doc_chars", type=int, default=2048, help="0 keeps full docs.")
    parser.add_argument("--max_query_chars", type=int, default=0, help="0 keeps full queries.")
    parser.add_argument(
        "--drop_if_pair_chars_gt",
        type=int,
        default=0,
        help="Drop a row if len(query)+len(doc) is still larger than this after truncation. 0 disables dropping.",
    )
    parser.add_argument(
        "--max_docs_per_query",
        type=int,
        default=0,
        help="Keep at most this many docs per sampled query. 0 keeps all docs.",
    )
    parser.add_argument(
        "--keep_relevant_when_capping",
        action="store_true",
        help="When --max_docs_per_query is set, append relevant docs that would otherwise be capped out.",
    )
    parser.add_argument("--relevance_threshold", type=float, default=7.0)
    parser.add_argument(
        "--group_key_fields",
        nargs="+",
        default=["query_id", "qid", "query"],
        help="First non-empty field is used as group key.",
    )
    return parser.parse_args()


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def group_key(row: dict[str, Any], fields: list[str]) -> str:
    for field in fields:
        value = row.get(field)
        if value not in (None, ""):
            return f"{field}:{stringify(value)}"
    return f"__missing__:{id(row)}"


def raw_label(row: dict[str, Any]) -> float:
    value = row.get("labels", row.get("raw_label", row.get("label", 0.0)))
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    if "label" in row and "labels" not in row and "raw_label" not in row and score <= 1.0:
        return score * 10.0
    return score


def sort_key(row: dict[str, Any]) -> tuple[float, float, int]:
    rank = row.get("retrieval_rank")
    try:
        rank_value = float(rank)
    except (TypeError, ValueError):
        rank_value = 10**12
    score = row.get("retrieval_score", row.get("score", 0.0))
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = 0.0
    return rank_value, -score_value, int(row.get("_source_index", 0))


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def process_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any] | None, dict[str, int]]:
    stats = {
        "doc_truncated": 0,
        "query_truncated": 0,
        "pair_dropped_overlong": 0,
    }
    item = dict(row)
    query = stringify(item.get("query", item.get("question", item.get("q", ""))))
    doc = stringify(item.get("doc", item.get("text", "")))
    query, query_truncated = truncate_text(query, args.max_query_chars)
    doc, doc_truncated = truncate_text(doc, args.max_doc_chars)
    if query_truncated:
        stats["query_truncated"] += 1
    if doc_truncated:
        stats["doc_truncated"] += 1
    if args.drop_if_pair_chars_gt > 0 and len(query) + len(doc) > args.drop_if_pair_chars_gt:
        stats["pair_dropped_overlong"] += 1
        return None, stats
    if query:
        item["query"] = query
    if doc:
        item["doc"] = doc
    item["truncated_for_fast_eval"] = bool(query_truncated or doc_truncated)
    return item, stats


def cap_group(rows: list[dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], int]:
    if args.max_docs_per_query <= 0 or len(rows) <= args.max_docs_per_query:
        return rows, 0
    ranked = sorted(rows, key=sort_key)
    kept = ranked[: args.max_docs_per_query]
    if args.keep_relevant_when_capping:
        seen = {int(row["_source_index"]) for row in kept}
        for row in ranked[args.max_docs_per_query :]:
            if raw_label(row) >= args.relevance_threshold and int(row["_source_index"]) not in seen:
                kept.append(row)
                seen.add(int(row["_source_index"]))
    kept = sorted(kept, key=lambda row: int(row["_source_index"]))
    return kept, len(rows) - len(kept)


def sample_group_keys(keys: list[str], ratio: float, seed: int) -> list[str]:
    if not keys:
        return []
    ratio = max(0.0, min(1.0, ratio))
    if ratio >= 1.0:
        return list(keys)
    sample_count = int(round(len(keys) * ratio))
    if ratio > 0.0:
        sample_count = max(1, sample_count)
    sample_count = min(sample_count, len(keys))
    rng = random.Random(seed)
    return sorted(rng.sample(keys, sample_count))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    metadata_file = Path(args.metadata_file) if args.metadata_file else output_file.with_suffix(".metadata.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    records = read_json_records(input_file)
    for idx, row in enumerate(records):
        row["_source_index"] = idx
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        groups[group_key(row, args.group_key_fields)].append(row)

    selected_keys = set(sample_group_keys(list(groups), args.sample_ratio, args.seed))
    output_rows: list[dict[str, Any]] = []
    stats = {
        "input_records": len(records),
        "input_groups": len(groups),
        "sampled_groups": len(selected_keys),
        "sample_ratio": args.sample_ratio,
        "seed": args.seed,
        "doc_truncated": 0,
        "query_truncated": 0,
        "pair_dropped_overlong": 0,
        "docs_dropped_by_cap": 0,
        "empty_groups_after_processing": 0,
    }

    for key in selected_keys:
        capped_rows, dropped_by_cap = cap_group(groups[key], args)
        stats["docs_dropped_by_cap"] += dropped_by_cap
        processed_group: list[dict[str, Any]] = []
        for row in capped_rows:
            item, row_stats = process_row(row, args)
            for stat_key, value in row_stats.items():
                stats[stat_key] += value
            if item is not None:
                item.pop("_source_index", None)
                processed_group.append(item)
        if not processed_group:
            stats["empty_groups_after_processing"] += 1
            continue
        output_rows.extend(processed_group)

    write_jsonl(output_file, output_rows)
    stats.update(
        {
            "output_records": len(output_rows),
            "output_groups": len({group_key(row, args.group_key_fields) for row in output_rows}),
            "max_doc_chars": args.max_doc_chars,
            "max_query_chars": args.max_query_chars,
            "drop_if_pair_chars_gt": args.drop_if_pair_chars_gt,
            "max_docs_per_query": args.max_docs_per_query,
            "keep_relevant_when_capping": args.keep_relevant_when_capping,
            "output_file": str(output_file),
        }
    )
    metadata_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d records from %d groups to %s", stats["output_records"], stats["output_groups"], output_file)
    logger.info("Wrote metadata to %s", metadata_file)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
