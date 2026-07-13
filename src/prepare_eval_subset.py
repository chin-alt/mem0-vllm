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
            "It filters overlong pairs first, samples groups to a target record ratio, "
            "and writes a JSONL plus metadata."
        )
    )
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_file", required=True)
    parser.add_argument("--metadata_file", default="")
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.10,
        help="Target output record ratio relative to the original input. Sampling is still query-group based.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_pair_chars",
        type=int,
        default=2048,
        help="Drop rows whose formatted instruction+query+doc length exceeds this value. 0 disables filtering.",
    )
    parser.add_argument("--max_doc_chars", type=int, default=0, help="Optional post-filter doc truncation. 0 keeps full docs.")
    parser.add_argument("--max_query_chars", type=int, default=0, help="0 keeps full queries.")
    parser.add_argument(
        "--drop_if_pair_chars_gt",
        type=int,
        default=0,
        help="Deprecated alias for --max_pair_chars when --max_pair_chars is 0.",
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
        "--allow_no_relevant_groups",
        action="store_true",
        help="Keep groups that have no remaining relevant docs after length filtering.",
    )
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


def get_query(row: dict[str, Any]) -> str:
    return stringify(row.get("query", row.get("question", row.get("q", ""))))


def get_doc(row: dict[str, Any]) -> str:
    return stringify(row.get("doc", row.get("text", "")))


def formatted_pair_length(row: dict[str, Any]) -> int:
    instruction = stringify(row.get("instruction", ""))
    query = get_query(row)
    doc = get_doc(row)
    return len(f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}")


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0 or len(text) <= max_chars:
        return text, False
    return text[:max_chars], True


def process_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, int]]:
    stats = {
        "doc_truncated": 0,
        "query_truncated": 0,
    }
    item = dict(row)
    query = get_query(item)
    doc = get_doc(item)
    query, query_truncated = truncate_text(query, args.max_query_chars)
    doc, doc_truncated = truncate_text(doc, args.max_doc_chars)
    if query_truncated:
        stats["query_truncated"] += 1
    if doc_truncated:
        stats["doc_truncated"] += 1
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


def group_relevant_count(rows: list[dict[str, Any]], threshold: float) -> int:
    return sum(1 for row in rows if raw_label(row) >= threshold)


def update_group_candidate_metadata(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    relevant_count = group_relevant_count(rows, args.relevance_threshold)
    updated: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item["candidate_relevant_count"] = relevant_count
        item["fast_eval_candidate_count"] = len(rows)
        updated.append(item)
    return updated


def filter_groups_before_sampling(
    groups: dict[str, list[dict[str, Any]]],
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, int]]:
    max_pair_chars = args.max_pair_chars
    if max_pair_chars <= 0 and args.drop_if_pair_chars_gt > 0:
        max_pair_chars = args.drop_if_pair_chars_gt
    stats = {
        "pair_dropped_overlong": 0,
        "groups_dropped_empty_after_length_filter": 0,
        "groups_dropped_no_relevant_after_length_filter": 0,
        "docs_dropped_by_cap": 0,
        "doc_truncated": 0,
        "query_truncated": 0,
    }
    filtered: dict[str, list[dict[str, Any]]] = {}
    for key, rows in groups.items():
        kept_rows: list[dict[str, Any]] = []
        for row in rows:
            original_length = formatted_pair_length(row)
            if max_pair_chars > 0 and original_length > max_pair_chars:
                stats["pair_dropped_overlong"] += 1
                continue
            item, row_stats = process_row(row, args)
            item["fast_eval_original_pair_chars"] = original_length
            for stat_key, value in row_stats.items():
                stats[stat_key] += value
            kept_rows.append(item)
        if not kept_rows:
            stats["groups_dropped_empty_after_length_filter"] += 1
            continue
        capped_rows, dropped_by_cap = cap_group(kept_rows, args)
        stats["docs_dropped_by_cap"] += dropped_by_cap
        if not args.allow_no_relevant_groups and group_relevant_count(capped_rows, args.relevance_threshold) <= 0:
            stats["groups_dropped_no_relevant_after_length_filter"] += 1
            continue
        filtered[key] = update_group_candidate_metadata(capped_rows, args)
    return filtered, stats


def sample_groups_to_record_target(
    groups: dict[str, list[dict[str, Any]]],
    target_records: int,
    seed: int,
) -> list[str]:
    if not groups or target_records <= 0:
        return []
    rng = random.Random(seed)
    keys = list(groups)
    rng.shuffle(keys)
    selected: list[str] = []
    total = 0
    for key in keys:
        selected.append(key)
        total += len(groups[key])
        if total >= target_records:
            break
    return selected


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

    filtered_groups, filter_stats = filter_groups_before_sampling(groups, args)
    target_records = int(round(len(records) * max(0.0, min(1.0, args.sample_ratio))))
    if args.sample_ratio > 0.0:
        target_records = max(1, target_records)
    target_records = min(target_records, sum(len(rows) for rows in filtered_groups.values()))
    selected_keys = set(sample_groups_to_record_target(filtered_groups, target_records, args.seed))
    output_rows: list[dict[str, Any]] = []
    stats = {
        "input_records": len(records),
        "input_groups": len(groups),
        "records_after_length_filter": sum(len(rows) for rows in filtered_groups.values()),
        "groups_after_length_filter": len(filtered_groups),
        "target_output_records": target_records,
        "sampled_groups": len(selected_keys),
        "sample_ratio": args.sample_ratio,
        "seed": args.seed,
        **filter_stats,
    }

    for key in selected_keys:
        for row in filtered_groups[key]:
            item = dict(row)
            item.pop("_source_index", None)
            output_rows.append(item)

    write_jsonl(output_file, output_rows)
    stats.update(
        {
            "output_records": len(output_rows),
            "output_groups": len({group_key(row, args.group_key_fields) for row in output_rows}),
            "max_pair_chars": args.max_pair_chars,
            "max_doc_chars": args.max_doc_chars,
            "max_query_chars": args.max_query_chars,
            "drop_if_pair_chars_gt": args.drop_if_pair_chars_gt,
            "max_docs_per_query": args.max_docs_per_query,
            "keep_relevant_when_capping": args.keep_relevant_when_capping,
            "allow_no_relevant_groups": args.allow_no_relevant_groups,
            "output_file": str(output_file),
        }
    )
    metadata_file.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d records from %d groups to %s", stats["output_records"], stats["output_groups"], output_file)
    logger.info("Wrote metadata to %s", metadata_file)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
