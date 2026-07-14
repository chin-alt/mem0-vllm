from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from build_cmteb_r_candidates import BM25Index, EmbeddingEncoder, EmbeddingRetriever
from data import write_jsonl
from prepare_cmteb_r import DEFAULT_INSTRUCTION, truncate_doc


logger = logging.getLogger(__name__)

SESSION_RE = re.compile(r"^session_(\d+)$")

DEFAULT_LOCOMO_INSTRUCTION = (
    "Given a long-term conversation memory question, retrieve dialogue turns "
    "that contain evidence needed to answer the question."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert LoCoMo locomo10.json into MemReranker candidate-ranking JSONL. "
            "Each QA question is a query, each dialogue turn in the same conversation "
            "is a candidate document, and qa.evidence dia_id values mark positives."
        )
    )
    parser.add_argument("--input_file", default="data/locomo/locomo10.json")
    parser.add_argument("--output_file", default="data/locomo/locomo_qwen3_embedding_candidates.jsonl")
    parser.add_argument("--metadata_file", default="")
    parser.add_argument("--instruction", default=DEFAULT_LOCOMO_INSTRUCTION)
    parser.add_argument("--candidate_top_k", type=int, default=100)
    parser.add_argument(
        "--retrieval_backend",
        choices=["embedding", "bm25"],
        default="embedding",
        help="First-stage retriever used to build per-question candidate lists.",
    )
    parser.add_argument("--max_samples", type=int, default=0, help="0 keeps all samples.")
    parser.add_argument("--max_queries", type=int, default=0, help="0 keeps all QA queries after filtering.")
    parser.add_argument("--max_qa_per_sample", type=int, default=0, help="0 keeps all QA entries per sample.")
    parser.add_argument("--doc_max_chars", type=int, default=0, help="0 keeps full dialogue-turn document text.")
    parser.add_argument(
        "--index_doc_max_chars",
        type=int,
        default=2048,
        help="Characters used per document for first-stage retrieval. 0 uses full documents.",
    )
    parser.add_argument("--positive_label", type=float, default=10.0)
    parser.add_argument("--negative_label", type=float, default=0.0)
    parser.add_argument(
        "--ensure_positives",
        action="store_true",
        help=(
            "Append evidence turns missed by first-stage retrieval. Off by default, "
            "so CandidateRecall reflects first-stage misses."
        ),
    )
    parser.add_argument(
        "--include_no_evidence",
        action="store_true",
        help="Keep QA entries with no evidence as all-negative groups. Usually not useful for reranker metrics.",
    )
    parser.add_argument(
        "--include_answer_metadata",
        action="store_true",
        help="Include answer/adversarial_answer as metadata fields only. They are never inserted into doc text.",
    )
    parser.add_argument("--embedding_model_name_or_path", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--embedding_query_instruction",
        default="Given a long-term conversation memory question, retrieve relevant dialogue turns.",
    )
    parser.add_argument(
        "--embedding_query_template",
        default="Instruct: {instruction}\nQuery: {query}",
        help="Prompt template used only for embedding queries. Available fields: instruction, query.",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--embedding_search_batch_size", type=int, default=64)
    parser.add_argument("--embedding_max_length", type=int, default=512)
    parser.add_argument("--embedding_device", default="auto")
    parser.add_argument("--embedding_multi_process", action="store_true")
    parser.add_argument("--embedding_devices", default="")
    parser.add_argument("--embedding_chunk_size", type=int, default=0)
    parser.add_argument("--embedding_search_device", default="auto")
    parser.add_argument(
        "--embedding_search_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument(
        "--embedding_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
    )
    parser.add_argument("--embedding_cache_dir", default="data/locomo/embedding_cache")
    parser.add_argument("--embedding_local_files_only", action="store_true")
    parser.add_argument("--no_embedding_normalize", dest="embedding_normalize", action="store_false")
    parser.set_defaults(embedding_normalize=True)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    return parser.parse_args()


def stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def clean_text(value: Any) -> str:
    return " ".join(stringify(value).replace("\r", " ").replace("\n", " ").split()).strip()


def read_locomo_samples(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as f:
        data = json.load(f)
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("data", "samples", "items", "records"):
            value = data.get(key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
        return [data]
    raise ValueError(f"Unsupported LoCoMo JSON root in {path}: {type(data).__name__}")


def sample_id_of(sample: dict[str, Any], index: int) -> str:
    for key in ("sample_id", "id", "conversation_id", "conv_id"):
        value = clean_text(sample.get(key))
        if value:
            return value
    return f"sample_{index:04d}"


def session_keys(conversation: dict[str, Any]) -> list[str]:
    keys: list[tuple[int, str]] = []
    for key, value in conversation.items():
        match = SESSION_RE.match(str(key))
        if match and isinstance(value, list):
            keys.append((int(match.group(1)), key))
    return [key for _num, key in sorted(keys)]


def first_non_empty(turn: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = clean_text(turn.get(key))
        if value:
            return value
    return ""


def format_turn_doc(
    turn: dict[str, Any],
    session_key: str,
    session_time: str,
    speaker_a: str,
    speaker_b: str,
) -> str:
    speaker = clean_text(turn.get("speaker"))
    text = first_non_empty(turn, ("compressed_text", "clean_text", "text", "content"))
    caption = first_non_empty(turn, ("blip_caption", "caption", "image_caption"))
    image_query = first_non_empty(turn, ("image_search_query", "search_query", "img_query"))

    parts: list[str] = []
    if session_time:
        parts.append(f"time: {session_time}")
    if speaker:
        parts.append(f"speaker: {speaker}")
    if speaker_a or speaker_b:
        participants = ", ".join(value for value in (speaker_a, speaker_b) if value)
        if participants:
            parts.append(f"participants: {participants}")
    if text:
        parts.append(f"text: {text}")
    if caption:
        parts.append(f"image_caption: {caption}")
    if image_query:
        parts.append(f"image_search_query: {image_query}")

    if not parts:
        return ""
    return f"session: {session_key}, " + ", ".join(parts)


def extract_dialog_docs(sample: dict[str, Any], sample_id: str, args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, str]]:
    conversation = sample.get("conversation")
    if not isinstance(conversation, dict):
        return [], {}

    speaker_a = clean_text(conversation.get("speaker_a"))
    speaker_b = clean_text(conversation.get("speaker_b"))
    docs: list[dict[str, Any]] = []
    dia_to_doc_id: dict[str, str] = {}

    for session_key in session_keys(conversation):
        session = conversation.get(session_key)
        if not isinstance(session, list):
            continue
        session_time = clean_text(conversation.get(f"{session_key}_date_time"))
        for turn_idx, turn in enumerate(session, start=1):
            if not isinstance(turn, dict):
                continue
            dia_id = clean_text(turn.get("dia_id")) or f"{session_key}:{turn_idx}"
            doc_text = format_turn_doc(turn, session_key, session_time, speaker_a, speaker_b)
            doc_text = truncate_doc(doc_text, args.doc_max_chars)
            if not doc_text:
                continue
            doc_id = f"{sample_id}::{dia_id}"
            dia_to_doc_id[dia_id] = doc_id
            docs.append(
                {
                    "doc_id": doc_id,
                    "dia_id": dia_id,
                    "doc": doc_text,
                    "session": session_key,
                    "session_time": session_time,
                    "speaker": clean_text(turn.get("speaker")),
                }
            )
    return docs, dia_to_doc_id


def extract_evidence_ids(value: Any) -> list[str]:
    evidence: list[str] = []
    if value is None:
        return evidence
    if isinstance(value, str):
        chunks = [part.strip() for part in re.split(r"[,;]", value) if part.strip()]
        return chunks if chunks else [value.strip()]
    if isinstance(value, (int, float)):
        return [str(value)]
    if isinstance(value, list):
        for item in value:
            evidence.extend(extract_evidence_ids(item))
        return evidence
    if isinstance(value, dict):
        for key in ("dia_id", "dialog_id", "turn_id", "id"):
            if key in value:
                evidence.extend(extract_evidence_ids(value.get(key)))
        if not evidence:
            for key in ("evidence", "evidences"):
                if key in value:
                    evidence.extend(extract_evidence_ids(value.get(key)))
        return evidence
    return []


def unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def qa_entries(sample: dict[str, Any]) -> list[dict[str, Any]]:
    value = sample.get("qa")
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    return []


def build_retriever(
    sample_id: str,
    docs: list[dict[str, Any]],
    args: argparse.Namespace,
    embedding_encoder: EmbeddingEncoder | None,
) -> Any:
    doc_ids = [row["doc_id"] for row in docs]
    doc_texts = [row["doc"] for row in docs]
    if args.retrieval_backend == "embedding":
        if embedding_encoder is None:
            raise ValueError("embedding_encoder is required for embedding retrieval")
        safe_sample_id = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("_") or "sample"
        return EmbeddingRetriever(f"locomo_{safe_sample_id}", doc_ids, doc_texts, embedding_encoder, args)
    return BM25Index(
        doc_ids,
        doc_texts,
        k1=args.k1,
        b=args.b,
        index_doc_max_chars=args.index_doc_max_chars,
    )


def build_sample_candidates(
    sample: dict[str, Any],
    sample_index: int,
    args: argparse.Namespace,
    embedding_encoder: EmbeddingEncoder | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sample_id = sample_id_of(sample, sample_index)
    docs, dia_to_doc_id = extract_dialog_docs(sample, sample_id, args)
    doc_id_to_index = {row["doc_id"]: idx for idx, row in enumerate(docs)}
    qas = qa_entries(sample)

    records: list[dict[str, Any]] = []
    meta: dict[str, Any] = {
        "sample_id": sample_id,
        "num_docs": len(docs),
        "num_qa_raw": len(qas),
        "num_queries_exported": 0,
        "num_records": 0,
        "skipped_no_question": 0,
        "skipped_no_evidence": 0,
        "skipped_missing_evidence": 0,
        "skipped_no_candidates": 0,
        "total_true_relevant": 0,
        "total_candidate_relevant": 0,
    }
    if not docs or not qas:
        return records, meta

    eligible: list[tuple[int, dict[str, Any], str, list[str], list[str]]] = []
    for qa_idx, qa in enumerate(qas):
        if args.max_qa_per_sample > 0 and len(eligible) >= args.max_qa_per_sample:
            break
        question = clean_text(qa.get("question"))
        if not question:
            meta["skipped_no_question"] += 1
            continue
        evidence_ids = unique_keep_order(extract_evidence_ids(qa.get("evidence")))
        if not evidence_ids and not args.include_no_evidence:
            meta["skipped_no_evidence"] += 1
            continue
        positive_ids = unique_keep_order([dia_to_doc_id[eid] for eid in evidence_ids if eid in dia_to_doc_id])
        if evidence_ids and not positive_ids:
            meta["skipped_missing_evidence"] += 1
            continue
        eligible.append((qa_idx, qa, question, evidence_ids, positive_ids))

    if not eligible:
        return records, meta

    retriever = build_retriever(sample_id, docs, args, embedding_encoder)
    queries = [question for _qa_idx, _qa, question, _evidence_ids, _positive_ids in eligible]
    if args.retrieval_backend == "embedding":
        ranked_by_query = retriever.search_batch(queries, args.candidate_top_k)
    else:
        ranked_by_query = [
            retriever.search(query, args.candidate_top_k)
            for query in tqdm(
                queries,
                desc=f"Building LoCoMo BM25 candidates {sample_id}",
                unit="query",
                dynamic_ncols=True,
                ascii=True,
            )
        ]

    for (qa_idx, qa, question, evidence_ids, positive_ids), ranked in zip(eligible, ranked_by_query, strict=False):
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
            meta["skipped_no_candidates"] += 1
            continue

        positive_set = set(positive_ids)
        candidate_doc_ids = [docs[doc_idx]["doc_id"] for doc_idx in candidate_doc_indices]
        candidate_relevant_count = len(set(candidate_doc_ids) & positive_set)
        query_id = f"locomo:{sample_id}:qa_{qa_idx}"
        category = qa.get("category", "")
        for rank, doc_idx in enumerate(candidate_doc_indices, start=1):
            doc_row = docs[doc_idx]
            doc_id = doc_row["doc_id"]
            is_positive = doc_id in positive_set
            row: dict[str, Any] = {
                "instruction": args.instruction or DEFAULT_INSTRUCTION,
                "dataset": "LoCoMo",
                "sample_id": sample_id,
                "query_id": query_id,
                "qid": f"{sample_id}:qa_{qa_idx}",
                "doc_id": doc_id,
                "dia_id": doc_row["dia_id"],
                "query": question,
                "doc": doc_row["doc"],
                "labels": args.positive_label if is_positive else args.negative_label,
                "reason": (
                    f"locomo_{args.retrieval_backend}_retrieved_positive"
                    if is_positive
                    else f"locomo_{args.retrieval_backend}_retrieved_negative"
                ),
                "category": category,
                "evidence": evidence_ids,
                "positive_doc_ids": positive_ids,
                "retrieval_rank": rank,
                "retrieval_score": retrieval_scores.get(doc_idx, 0.0),
                "retrieval_backend": args.retrieval_backend,
                "true_relevant_count": len(positive_ids),
                "candidate_relevant_count": candidate_relevant_count,
                "session": doc_row["session"],
                "session_time": doc_row["session_time"],
                "speaker": doc_row["speaker"],
            }
            if args.include_answer_metadata:
                row["answer"] = stringify(qa.get("answer"))
                row["adversarial_answer"] = stringify(qa.get("adversarial_answer"))
            records.append(row)
        meta["total_true_relevant"] += len(positive_ids)
        meta["total_candidate_relevant"] += candidate_relevant_count

    meta["num_queries_exported"] = len({row["query_id"] for row in records})
    meta["num_records"] = len(records)
    meta["candidate_recall"] = (
        meta["total_candidate_relevant"] / meta["total_true_relevant"]
        if meta["total_true_relevant"]
        else 0.0
    )
    return records, meta


def maybe_trim_queries(records: list[dict[str, Any]], max_queries: int) -> list[dict[str, Any]]:
    if max_queries <= 0:
        return records
    kept_query_ids: set[str] = set()
    trimmed: list[dict[str, Any]] = []
    for row in records:
        query_id = str(row.get("query_id", ""))
        if query_id not in kept_query_ids:
            if len(kept_query_ids) >= max_queries:
                break
            kept_query_ids.add(query_id)
        trimmed.append(row)
    return trimmed


def aggregate_exported_counts(records: list[dict[str, Any]]) -> tuple[int, int]:
    by_query: dict[str, list[dict[str, Any]]] = {}
    for row in records:
        by_query.setdefault(str(row.get("query_id", "")), []).append(row)
    total_true = 0
    total_candidate = 0
    for rows in by_query.values():
        true_counts = [int(row.get("true_relevant_count") or 0) for row in rows]
        candidate_counts = [int(row.get("candidate_relevant_count") or 0) for row in rows]
        total_true += max(true_counts) if true_counts else 0
        total_candidate += max(candidate_counts) if candidate_counts else 0
    return total_true, total_candidate


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    metadata_file = Path(args.metadata_file) if args.metadata_file else output_file.with_suffix(".metadata.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    samples = read_locomo_samples(input_file)
    if args.max_samples > 0:
        samples = samples[: args.max_samples]
    logger.info("Loaded %d LoCoMo samples from %s", len(samples), input_file)

    all_records: list[dict[str, Any]] = []
    sample_metadata: list[dict[str, Any]] = []
    embedding_encoder = EmbeddingEncoder(args) if args.retrieval_backend == "embedding" else None
    try:
        for sample_index, sample in enumerate(
            tqdm(samples, desc="Preparing LoCoMo samples", unit="sample", dynamic_ncols=True, ascii=True)
        ):
            records, meta = build_sample_candidates(sample, sample_index, args, embedding_encoder)
            all_records.extend(records)
            sample_metadata.append(meta)
            if args.max_queries > 0 and len({row["query_id"] for row in all_records}) >= args.max_queries:
                all_records = maybe_trim_queries(all_records, args.max_queries)
                break
    finally:
        if embedding_encoder is not None:
            embedding_encoder.close()

    all_records = maybe_trim_queries(all_records, args.max_queries)
    write_jsonl(output_file, all_records)

    total_true, total_candidate = aggregate_exported_counts(all_records)
    metadata = {
        "input_file": str(input_file),
        "output_file": str(output_file),
        "retrieval_backend": args.retrieval_backend,
        "embedding_model_name_or_path": args.embedding_model_name_or_path
        if args.retrieval_backend == "embedding"
        else None,
        "candidate_top_k": args.candidate_top_k,
        "ensure_positives": args.ensure_positives,
        "num_samples": len(samples),
        "num_query_groups": len({row["query_id"] for row in all_records}),
        "num_records": len(all_records),
        "total_true_relevant": total_true,
        "total_candidate_relevant": total_candidate,
        "candidate_recall": total_candidate / total_true if total_true else 0.0,
        "samples": sample_metadata,
    }
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d LoCoMo candidate records to %s", len(all_records), output_file)
    logger.info("Wrote metadata to %s", metadata_file)


if __name__ == "__main__":
    main()
