from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from data import write_jsonl


logger = logging.getLogger(__name__)

DEFAULT_INSTRUCTION = "Given a Chinese search query, retrieve relevant passages that answer the query."
DEFAULT_DATASETS = [
    "T2Retrieval",
    "MMarcoRetrieval",
    "DuRetrieval",
    "CovidRetrieval",
    "CmedqaRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
]
POSITIVE_FIELDS = (
    "positive_doc_ids",
    "positive_docids",
    "positive_docs",
    "positive",
    "positives",
    "relevant_doc_ids",
    "relevant_docs",
    "relevant",
    "doc_ids",
    "docids",
    "doc_id",
    "docid",
    "corpus_id",
    "pid",
    "pids",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert C-MTEB Retrieval data to MemReranker query-doc-label JSONL."
    )
    parser.add_argument("--input_dir", default="data/cmteb_r/raw")
    parser.add_argument("--output_file", default="data/cmteb_r/cmteb_r_eval.jsonl")
    parser.add_argument("--metadata_file", default=None)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--negatives_per_query", type=int, default=15)
    parser.add_argument("--max_queries_per_dataset", type=int, default=1000)
    parser.add_argument("--max_docs_per_query", type=int, default=32)
    parser.add_argument("--doc_max_chars", type=int, default=0, help="0 keeps full document text.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--positive_label", type=float, default=10.0)
    parser.add_argument("--negative_label", type=float, default=0.0)
    parser.add_argument(
        "--mode",
        choices=["qrels_random", "positives_only"],
        default="qrels_random",
        help="qrels_random keeps positives and samples random corpus negatives; positives_only is a smoke-test mode.",
    )
    parser.add_argument(
        "--skip_missing_qrels",
        action="store_true",
        help="Skip datasets where no positives/qrels can be discovered instead of raising.",
    )
    return parser.parse_args()


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def dataset_dir(input_dir: Path, name: str) -> Path:
    if "/" in name:
        name = name.split("/", 1)[1]
    return input_dir / name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_parquet_files(paths: list[Path]) -> list[dict[str, Any]]:
    import pandas as pd

    frames = [pd.read_parquet(path) for path in paths]
    if not frames:
        return []
    data = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return data.to_dict(orient="records")


def find_split_files(root: Path, split: str) -> list[Path]:
    patterns = [
        f"{split}.jsonl",
        f"{split}.json",
        f"{split}-*.jsonl",
        f"{split}-*.json",
        f"{split}*.parquet",
        f"data/{split}*.parquet",
        f"data/{split}-*.parquet",
    ]
    files: list[Path] = []
    for pattern in patterns:
        files.extend(sorted(root.glob(pattern)))
    seen = set()
    unique = []
    for path in files:
        if path not in seen and path.is_file():
            seen.add(path)
            unique.append(path)
    return unique


def read_split(root: Path, split: str) -> list[dict[str, Any]]:
    files = find_split_files(root, split)
    if not files:
        raise FileNotFoundError(f"Could not find {split} files under {root}")
    suffixes = {path.suffix.lower() for path in files}
    if suffixes == {".parquet"}:
        return read_parquet_files(files)
    rows: list[dict[str, Any]] = []
    for path in files:
        if path.suffix.lower() == ".jsonl":
            rows.extend(read_jsonl(path))
        elif path.suffix.lower() == ".json":
            obj = json.loads(path.read_text(encoding="utf-8-sig"))
            if isinstance(obj, list):
                rows.extend(item for item in obj if isinstance(item, dict))
            elif isinstance(obj, dict):
                rows.extend(item for item in obj.values() if isinstance(item, dict))
        else:
            raise ValueError(f"Unsupported split file type: {path}")
    return rows


def row_id(row: dict[str, Any]) -> str:
    for key in ("id", "_id", "qid", "query_id", "doc_id", "docid", "pid", "corpus_id"):
        value = row.get(key)
        if not is_missing(value):
            return stringify(value).strip()
    raise ValueError(f"Missing id field in row keys={list(row)[:10]}")


def row_text(row: dict[str, Any], *, is_doc: bool) -> str:
    if not is_missing(row.get("text")):
        text = stringify(row.get("text")).strip()
    elif not is_missing(row.get("contents")):
        text = stringify(row.get("contents")).strip()
    elif not is_missing(row.get("query")):
        text = stringify(row.get("query")).strip()
    elif not is_missing(row.get("question")):
        text = stringify(row.get("question")).strip()
    else:
        text = ""

    title = stringify(row.get("title")).strip() if not is_missing(row.get("title")) else ""
    if is_doc and title and title not in text:
        return f"title: {title}\ntext: {text}".strip()
    return text


def normalize_doc_id(value: Any) -> str | None:
    if is_missing(value):
        return None
    text = stringify(value).strip()
    return text or None


def extract_doc_ids(value: Any) -> list[str]:
    if is_missing(value):
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") or text.startswith("{"):
            try:
                return extract_doc_ids(json.loads(text))
            except json.JSONDecodeError:
                pass
        if "," in text:
            return [item for item in (part.strip() for part in text.split(",")) if item]
        return [text]
    if isinstance(value, dict):
        for key in ("id", "_id", "doc_id", "docid", "pid", "corpus_id"):
            if key in value:
                doc_id = normalize_doc_id(value[key])
                return [doc_id] if doc_id else []
        ids = []
        for key, score in value.items():
            try:
                keep = float(score) > 0
            except (TypeError, ValueError):
                keep = True
            if keep:
                doc_id = normalize_doc_id(key)
                if doc_id:
                    ids.append(doc_id)
        return ids
    if isinstance(value, Iterable):
        ids = []
        for item in value:
            ids.extend(extract_doc_ids(item))
        return ids
    return [stringify(value).strip()]


def positives_from_query_row(row: dict[str, Any]) -> list[str]:
    ids: list[str] = []
    for field in POSITIVE_FIELDS:
        if field in row:
            ids.extend(extract_doc_ids(row[field]))
    seen = set()
    unique = []
    for doc_id in ids:
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            unique.append(doc_id)
    return unique


def read_qrels(root: Path) -> dict[str, dict[str, float]]:
    candidates = []
    for pattern in ("qrels/*.tsv", "qrels/*.txt", "qrels/*.csv", "*qrels*.tsv", "*qrels*.txt", "*qrels*.csv"):
        candidates.extend(root.glob(pattern))
    qrels: dict[str, dict[str, float]] = {}
    for path in sorted(set(candidates)):
        delimiter = "," if path.suffix.lower() == ".csv" else "\t"
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            first_line = f.readline()
            f.seek(0)
            first_cells = {cell.strip().lower() for cell in first_line.strip().split(delimiter)}
            header_names = {
                "query-id",
                "query_id",
                "qid",
                "query",
                "corpus-id",
                "corpus_id",
                "docid",
                "doc_id",
                "score",
                "relevance",
                "label",
            }
            has_header = bool(first_cells & header_names)
            reader = csv.DictReader(f, delimiter=delimiter) if has_header else None
            if reader is not None:
                for row in reader:
                    qid = row.get("query-id") or row.get("query_id") or row.get("qid") or row.get("query")
                    docid = row.get("corpus-id") or row.get("corpus_id") or row.get("docid") or row.get("doc_id")
                    score_raw = row.get("score") or row.get("relevance") or row.get("label") or 1
                    add_qrel(qrels, qid, docid, score_raw)
            else:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        qid, docid, score_raw = parts[0], parts[2], parts[3]
                    elif len(parts) >= 3:
                        qid, docid, score_raw = parts[0], parts[1], parts[2]
                    else:
                        continue
                    add_qrel(qrels, qid, docid, score_raw)
    return qrels


def add_qrel(qrels: dict[str, dict[str, float]], qid: Any, docid: Any, score_raw: Any) -> None:
    qid_s = normalize_doc_id(qid)
    docid_s = normalize_doc_id(docid)
    if not qid_s or not docid_s:
        return
    try:
        score = float(score_raw)
    except (TypeError, ValueError):
        score = 1.0
    if score <= 0:
        return
    qrels.setdefault(qid_s, {})[docid_s] = score


def truncate_doc(text: str, max_chars: int) -> str:
    if max_chars and max_chars > 0 and len(text) > max_chars:
        return text[:max_chars]
    return text


def build_dataset_records(
    name: str,
    root: Path,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus_rows = read_split(root, "corpus")
    query_rows = read_split(root, "queries")
    corpus = {
        row_id(row): truncate_doc(row_text(row, is_doc=True), args.doc_max_chars)
        for row in corpus_rows
    }
    corpus = {doc_id: doc for doc_id, doc in corpus.items() if doc}
    corpus_ids = list(corpus)
    qrels = read_qrels(root)

    records: list[dict[str, Any]] = []
    query_count = 0
    skipped_no_positive = 0
    skipped_no_doc = 0
    for query_row in tqdm(query_rows, desc=f"Preparing {name}", unit="query", dynamic_ncols=True, ascii=True):
        qid = row_id(query_row)
        query = row_text(query_row, is_doc=False)
        if not query:
            continue
        positives = list(qrels.get(qid, {}))
        if not positives:
            positives = positives_from_query_row(query_row)
        positives = [doc_id for doc_id in positives if doc_id in corpus]
        if not positives:
            skipped_no_positive += 1
            continue

        query_count += 1
        if args.max_queries_per_dataset > 0 and query_count > args.max_queries_per_dataset:
            break

        candidate_ids = list(positives)
        if args.mode == "qrels_random" and args.negatives_per_query > 0:
            positive_set = set(positives)
            available_negatives = [doc_id for doc_id in corpus_ids if doc_id not in positive_set]
            negative_count = min(args.negatives_per_query, len(available_negatives))
            candidate_ids.extend(rng.sample(available_negatives, negative_count))
        if args.max_docs_per_query > 0:
            positive_set = set(positives)
            positives_kept = [doc_id for doc_id in candidate_ids if doc_id in positive_set]
            negatives_kept = [doc_id for doc_id in candidate_ids if doc_id not in positive_set]
            remaining = max(0, args.max_docs_per_query - len(positives_kept))
            candidate_ids = positives_kept + negatives_kept[:remaining]
        rng.shuffle(candidate_ids)

        for doc_id in candidate_ids:
            doc = corpus.get(doc_id)
            if not doc:
                skipped_no_doc += 1
                continue
            is_positive = doc_id in set(positives)
            records.append(
                {
                    "instruction": args.instruction,
                    "dataset": name,
                    "query_id": f"{name}:{qid}",
                    "qid": qid,
                    "doc_id": doc_id,
                    "query": query,
                    "doc": doc,
                    "labels": args.positive_label if is_positive else args.negative_label,
                    "reason": "cmteb_r_positive" if is_positive else "cmteb_r_random_negative",
                }
            )

    meta = {
        "dataset": name,
        "root": str(root),
        "num_corpus": len(corpus),
        "num_queries_raw": len(query_rows),
        "num_queries_exported": len({row["query_id"] for row in records}),
        "num_records": len(records),
        "num_positive_records": sum(1 for row in records if float(row["labels"]) > 0),
        "skipped_no_positive": skipped_no_positive,
        "skipped_no_doc": skipped_no_doc,
        "qrels_queries": len(qrels),
        "query_columns": list(query_rows[0]) if query_rows else [],
        "corpus_columns": list(corpus_rows[0]) if corpus_rows else [],
    }
    return records, meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file = Path(args.metadata_file) if args.metadata_file else output_file.with_suffix(".metadata.json")
    rng = random.Random(args.seed)

    all_records: list[dict[str, Any]] = []
    metadata = {
        "input_dir": str(input_dir),
        "output_file": str(output_file),
        "seed": args.seed,
        "mode": args.mode,
        "negatives_per_query": args.negatives_per_query,
        "max_queries_per_dataset": args.max_queries_per_dataset,
        "max_docs_per_query": args.max_docs_per_query,
        "datasets": [],
    }
    for dataset in args.datasets:
        root = dataset_dir(input_dir, dataset)
        if not root.exists():
            message = f"Dataset directory not found: {root}"
            if args.skip_missing_qrels:
                logger.warning(message)
                continue
            raise FileNotFoundError(message)
        records, meta = build_dataset_records(dataset.split("/", 1)[-1], root, args, rng)
        if not records:
            message = f"No records exported for {dataset}; qrels/positive doc ids may be absent."
            if args.skip_missing_qrels:
                logger.warning(message)
                metadata["datasets"].append(meta)
                continue
            raise ValueError(message)
        all_records.extend(records)
        metadata["datasets"].append(meta)

    write_jsonl(output_file, all_records)
    metadata["num_records"] = len(all_records)
    metadata["num_query_groups"] = len({row["query_id"] for row in all_records})
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d records to %s", len(all_records), output_file)
    logger.info("Wrote metadata to %s", metadata_file)


if __name__ == "__main__":
    main()
