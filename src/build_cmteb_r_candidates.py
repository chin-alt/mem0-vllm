from __future__ import annotations

import argparse
import heapq
import json
import logging
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from data import write_jsonl
from prepare_cmteb_r import (
    DEFAULT_DATASETS,
    DEFAULT_INSTRUCTION,
    dataset_dir,
    find_qrels_files,
    read_qrels,
    read_split,
    row_id,
    row_text,
    truncate_doc,
)


logger = logging.getLogger(__name__)

WORD_RE = re.compile(r"[a-zA-Z0-9]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build CMTEB-R first-stage candidate lists from corpus/queries before reranking."
    )
    parser.add_argument("--input_dir", default="data/cmteb_r")
    parser.add_argument("--qrels_input_dir", default="")
    parser.add_argument("--qrels_dataset_suffix", default="-qrels")
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--output_file", default="data/cmteb_r/cmteb_r_bm25_candidates.jsonl")
    parser.add_argument("--metadata_file", default="")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--candidate_top_k", type=int, default=100)
    parser.add_argument("--max_queries_per_dataset", type=int, default=1000)
    parser.add_argument("--doc_max_chars", type=int, default=0, help="0 keeps full document text in output.")
    parser.add_argument(
        "--index_doc_max_chars",
        type=int,
        default=2048,
        help="Characters indexed per document for BM25. 0 indexes full documents.",
    )
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument(
        "--ensure_positives",
        action="store_true",
        help="Append qrels positives missed by BM25. Off by default for realistic first-stage recall.",
    )
    return parser.parse_args()


def tokenize(text: str) -> list[str]:
    text = (text or "").lower()
    tokens: list[str] = []
    tokens.extend(match.group(0) for match in WORD_RE.finditer(text))
    cjk_chars = CJK_RE.findall(text)
    tokens.extend(cjk_chars)
    tokens.extend("".join(cjk_chars[idx : idx + 2]) for idx in range(max(0, len(cjk_chars) - 1)))
    return tokens


class BM25Index:
    def __init__(
        self,
        doc_ids: list[str],
        docs: list[str],
        k1: float = 1.2,
        b: float = 0.75,
        index_doc_max_chars: int = 2048,
    ):
        self.doc_ids = doc_ids
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_lengths: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)

        for doc_idx, doc in enumerate(tqdm(docs, desc="Indexing BM25", unit="doc", dynamic_ncols=True, ascii=True)):
            index_text = doc if index_doc_max_chars <= 0 else doc[:index_doc_max_chars]
            term_counts = Counter(tokenize(index_text))
            doc_len = sum(term_counts.values())
            self.doc_lengths.append(doc_len)
            for term, tf in term_counts.items():
                postings[term].append((doc_idx, int(tf)))

        self.postings = dict(postings)
        self.num_docs = len(docs)
        self.avgdl = sum(self.doc_lengths) / max(1, self.num_docs)
        self.idf = {
            term: math.log(1.0 + (self.num_docs - len(term_postings) + 0.5) / (len(term_postings) + 0.5))
            for term, term_postings in self.postings.items()
        }

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term, qtf in Counter(tokenize(query)).items():
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, tf in postings:
                doc_len = self.doc_lengths[doc_idx]
                denom = tf + self.k1 * (1.0 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                scores[doc_idx] += idf * (tf * (self.k1 + 1.0) / max(denom, 1e-9)) * float(qtf)
        if not scores:
            return []
        return heapq.nlargest(top_k, scores.items(), key=lambda item: (item[1], -item[0]))


def build_dataset_candidates(
    name: str,
    root: Path,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    corpus_rows = read_split(root, "corpus")
    query_rows = read_split(root, "queries")
    corpus: dict[str, str] = {}
    for row in corpus_rows:
        doc_id = row_id(row)
        doc = truncate_doc(row_text(row, is_doc=True), args.doc_max_chars)
        if doc:
            corpus[doc_id] = doc

    qrels_files = find_qrels_files(root, name, args.qrels_input_dir, args.qrels_dataset_suffix)
    qrels = read_qrels(root, name, args.qrels_input_dir, args.qrels_dataset_suffix)
    corpus_ids = list(corpus)
    docs = [corpus[doc_id] for doc_id in corpus_ids]
    doc_id_to_index = {doc_id: idx for idx, doc_id in enumerate(corpus_ids)}
    bm25 = BM25Index(
        corpus_ids,
        docs,
        k1=args.k1,
        b=args.b,
        index_doc_max_chars=args.index_doc_max_chars,
    )

    records: list[dict[str, Any]] = []
    query_count = 0
    skipped_no_qrels = 0
    skipped_no_candidates = 0
    total_true_relevant = 0
    total_candidate_relevant = 0

    for query_row in tqdm(query_rows, desc=f"Building candidates {name}", unit="query", dynamic_ncols=True, ascii=True):
        qid = row_id(query_row)
        query = row_text(query_row, is_doc=False)
        positive_ids = [doc_id for doc_id in qrels.get(qid, {}) if doc_id in corpus]
        if not query or not positive_ids:
            skipped_no_qrels += 1
            continue

        query_count += 1
        if args.max_queries_per_dataset > 0 and query_count > args.max_queries_per_dataset:
            break

        ranked = bm25.search(query, args.candidate_top_k)
        candidate_doc_indices = [doc_idx for doc_idx, _score in ranked]
        retrieval_scores = {doc_idx: float(score) for doc_idx, score in ranked}
        if args.ensure_positives:
            seen = set(candidate_doc_indices)
            for doc_id in positive_ids:
                doc_idx = doc_id_to_index[doc_id]
                if doc_idx not in seen:
                    candidate_doc_indices.append(doc_idx)
                    retrieval_scores.setdefault(doc_idx, 0.0)
                    seen.add(doc_idx)
        if not candidate_doc_indices:
            skipped_no_candidates += 1
            continue

        positive_set = set(positive_ids)
        candidate_doc_ids = [corpus_ids[doc_idx] for doc_idx in candidate_doc_indices]
        candidate_relevant_count = len(set(candidate_doc_ids) & positive_set)
        for rank, doc_idx in enumerate(candidate_doc_indices, start=1):
            doc_id = corpus_ids[doc_idx]
            is_positive = doc_id in positive_set
            records.append(
                {
                    "instruction": args.instruction,
                    "dataset": name,
                    "query_id": f"{name}:{qid}",
                    "qid": qid,
                    "doc_id": doc_id,
                    "query": query,
                    "doc": corpus[doc_id],
                    "labels": 10.0 if is_positive else 0.0,
                    "reason": "cmteb_r_retrieved_positive" if is_positive else "cmteb_r_retrieved_negative",
                    "retrieval_rank": rank,
                    "retrieval_score": retrieval_scores.get(doc_idx, 0.0),
                    "true_relevant_count": len(positive_ids),
                    "candidate_relevant_count": candidate_relevant_count,
                }
            )
        total_true_relevant += len(positive_ids)
        total_candidate_relevant += candidate_relevant_count

    meta = {
        "dataset": name,
        "root": str(root),
        "qrels_files": [str(path) for path in qrels_files],
        "num_corpus": len(corpus),
        "num_queries_raw": len(query_rows),
        "num_qrels_queries": len(qrels),
        "num_queries_exported": len({row["query_id"] for row in records}),
        "num_records": len(records),
        "candidate_top_k": args.candidate_top_k,
        "ensure_positives": args.ensure_positives,
        "skipped_no_qrels": skipped_no_qrels,
        "skipped_no_candidates": skipped_no_candidates,
        "total_true_relevant": total_true_relevant,
        "total_candidate_relevant": total_candidate_relevant,
        "candidate_recall": total_candidate_relevant / total_true_relevant if total_true_relevant else 0.0,
    }
    return records, meta


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)
    metadata_file = Path(args.metadata_file) if args.metadata_file else output_file.with_suffix(".metadata.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_records: list[dict[str, Any]] = []
    metadata = {
        "input_dir": str(input_dir),
        "qrels_input_dir": args.qrels_input_dir,
        "output_file": str(output_file),
        "candidate_top_k": args.candidate_top_k,
        "max_queries_per_dataset": args.max_queries_per_dataset,
        "datasets": [],
    }
    for dataset in args.datasets:
        root = dataset_dir(input_dir, dataset)
        records, meta = build_dataset_candidates(dataset.split("/", 1)[-1], root, args)
        all_records.extend(records)
        metadata["datasets"].append(meta)

    write_jsonl(output_file, all_records)
    metadata["num_records"] = len(all_records)
    metadata["num_query_groups"] = len({row["query_id"] for row in all_records})
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d candidate records to %s", len(all_records), output_file)
    logger.info("Wrote metadata to %s", metadata_file)


if __name__ == "__main__":
    main()
