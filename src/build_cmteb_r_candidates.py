from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import logging
import math
import os
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
    parser.add_argument("--output_file", default="data/cmteb_r/cmteb_r_qwen3_embedding_candidates.jsonl")
    parser.add_argument("--metadata_file", default="")
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--candidate_top_k", type=int, default=100)
    parser.add_argument(
        "--retrieval_backend",
        choices=["embedding", "bm25"],
        default="embedding",
        help="First-stage retriever. Use embedding to mirror dense-retrieval + reranker refinement.",
    )
    parser.add_argument("--max_queries_per_dataset", type=int, default=1000)
    parser.add_argument("--doc_max_chars", type=int, default=0, help="0 keeps full document text in output.")
    parser.add_argument(
        "--index_doc_max_chars",
        type=int,
        default=2048,
        help="Characters used per document for first-stage retrieval. 0 uses full documents.",
    )
    parser.add_argument("--embedding_model_name_or_path", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument(
        "--embedding_query_instruction",
        default="Given a Chinese search query, retrieve relevant passages that answer the query.",
    )
    parser.add_argument(
        "--embedding_query_template",
        default="Instruct: {instruction}\nQuery: {query}",
        help="Prompt template used only for embedding queries. Available fields: instruction, query.",
    )
    parser.add_argument("--embedding_batch_size", type=int, default=32)
    parser.add_argument("--embedding_search_batch_size", type=int, default=64)
    parser.add_argument("--embedding_max_length", type=int, default=512)
    parser.add_argument("--embedding_device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument(
        "--embedding_multi_process",
        action="store_true",
        help="Use sentence-transformers multi-process encoding, typically one worker per GPU.",
    )
    parser.add_argument(
        "--embedding_devices",
        default="",
        help="Comma-separated devices for multi-process encoding, e.g. cuda:0,cuda:1. Empty lets sentence-transformers decide.",
    )
    parser.add_argument(
        "--embedding_chunk_size",
        type=int,
        default=0,
        help="Chunk size for multi-process encoding. 0 lets sentence-transformers choose.",
    )
    parser.add_argument(
        "--embedding_search_device",
        default="auto",
        help="Device for dense top-k search: auto, cpu, cuda, cuda:0, etc. auto uses CUDA when available.",
    )
    parser.add_argument(
        "--embedding_search_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype for GPU dense top-k search. auto uses float16 on CUDA and float32 on CPU.",
    )
    parser.add_argument(
        "--embedding_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Torch dtype hint for loading the embedding model when supported by sentence-transformers.",
    )
    parser.add_argument("--embedding_cache_dir", default="", help="Optional directory for cached corpus embeddings.")
    parser.add_argument(
        "--embedding_local_files_only",
        action="store_true",
        help="Load the embedding model from local files/cache only.",
    )
    parser.add_argument(
        "--no_embedding_normalize",
        dest="embedding_normalize",
        action="store_false",
        help="Disable L2-normalization before dense dot-product retrieval.",
    )
    parser.set_defaults(embedding_normalize=True)
    parser.add_argument("--k1", type=float, default=1.2)
    parser.add_argument("--b", type=float, default=0.75)
    parser.add_argument(
        "--ensure_positives",
        action="store_true",
        help="Append qrels positives missed by retrieval. Off by default for realistic first-stage recall.",
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


class EmbeddingEncoder:
    def __init__(self, args: argparse.Namespace):
        if args.embedding_local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "Embedding retrieval requires sentence-transformers. "
                "Install requirements-cmteb.txt or set --retrieval_backend bm25."
            ) from exc

        device = None if args.embedding_device == "auto" else args.embedding_device
        model_kwargs: dict[str, Any] = {}
        if args.embedding_dtype != "auto":
            try:
                import torch

                model_kwargs["torch_dtype"] = {
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "float32": torch.float32,
                }[args.embedding_dtype]
            except ImportError:
                logger.warning("torch is not importable; ignoring --embedding_dtype=%s", args.embedding_dtype)

        kwargs: dict[str, Any] = {"trust_remote_code": True}
        if device is not None:
            kwargs["device"] = device
        if args.embedding_local_files_only:
            kwargs["local_files_only"] = True
        if model_kwargs:
            kwargs["model_kwargs"] = model_kwargs

        logger.info("Loading embedding model: %s", args.embedding_model_name_or_path)
        try:
            self.model = SentenceTransformer(args.embedding_model_name_or_path, **kwargs)
        except TypeError:
            # Older sentence-transformers releases do not accept every loading kwarg.
            kwargs.pop("local_files_only", None)
            kwargs.pop("model_kwargs", None)
            self.model = SentenceTransformer(args.embedding_model_name_or_path, **kwargs)

        if args.embedding_max_length > 0 and hasattr(self.model, "max_seq_length"):
            self.model.max_seq_length = args.embedding_max_length
        self.args = args
        self._pool: Any | None = None

    def format_query(self, query: str) -> str:
        instruction = self.args.embedding_query_instruction or self.args.instruction
        try:
            return self.args.embedding_query_template.format(instruction=instruction, query=query)
        except KeyError as exc:
            raise ValueError(f"Unknown field in --embedding_query_template: {exc}") from exc

    def _target_devices(self) -> list[str] | None:
        if self.args.embedding_devices.strip():
            return [device.strip() for device in self.args.embedding_devices.split(",") if device.strip()]
        return None

    def _get_pool(self) -> Any:
        if self._pool is None:
            devices = self._target_devices()
            logger.info("Starting multi-process embedding pool on devices: %s", devices or "auto")
            self._pool = self.model.start_multi_process_pool(target_devices=devices)
        return self._pool

    def close(self) -> None:
        if self._pool is not None:
            logger.info("Stopping multi-process embedding pool")
            self.model.stop_multi_process_pool(self._pool)
            self._pool = None

    def _maybe_normalize(self, embeddings: Any) -> Any:
        if not self.args.embedding_normalize:
            return embeddings
        import numpy as np

        array = np.asarray(embeddings, dtype="float32")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, 1e-12)

    def encode(self, texts: list[str], batch_size: int, desc: str) -> Any:
        logger.info("%s: encoding %d texts", desc, len(texts))
        if self.args.embedding_multi_process:
            kwargs: dict[str, Any] = {
                "batch_size": batch_size,
                "normalize_embeddings": self.args.embedding_normalize,
            }
            if self.args.embedding_chunk_size > 0:
                kwargs["chunk_size"] = self.args.embedding_chunk_size
            try:
                return self._maybe_normalize(self.model.encode_multi_process(texts, self._get_pool(), **kwargs))
            except TypeError:
                kwargs.pop("normalize_embeddings", None)
                return self._maybe_normalize(self.model.encode_multi_process(texts, self._get_pool(), **kwargs))

        return self._maybe_normalize(
            self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=self.args.embedding_normalize,
            )
        )


class EmbeddingRetriever:
    def __init__(
        self,
        dataset_name: str,
        doc_ids: list[str],
        docs: list[str],
        encoder: EmbeddingEncoder,
        args: argparse.Namespace,
    ):
        self.dataset_name = dataset_name
        self.doc_ids = doc_ids
        self.docs = docs
        self.encoder = encoder
        self.args = args
        self.doc_embeddings = self._load_or_encode_doc_embeddings()

    def _cache_paths(self) -> tuple[Path, Path] | None:
        if not self.args.embedding_cache_dir:
            return None
        cache_dir = Path(self.args.embedding_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_payload = {
            "dataset": self.dataset_name,
            "model": self.args.embedding_model_name_or_path,
            "max_length": self.args.embedding_max_length,
            "index_doc_max_chars": self.args.index_doc_max_chars,
            "normalize": self.args.embedding_normalize,
            "num_docs": len(self.doc_ids),
        }
        digest = hashlib.sha1(json.dumps(cache_payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
        model_name = re.sub(r"[^A-Za-z0-9._-]+", "_", self.args.embedding_model_name_or_path).strip("_")
        prefix = f"{self.dataset_name}.{model_name[-80:]}.{digest}"
        return cache_dir / f"{prefix}.embeddings.npy", cache_dir / f"{prefix}.doc_ids.json"

    def _load_or_encode_doc_embeddings(self) -> Any:
        import numpy as np

        cache_paths = self._cache_paths()
        if cache_paths is not None:
            emb_path, ids_path = cache_paths
            if emb_path.exists() and ids_path.exists():
                cached_ids = json.loads(ids_path.read_text(encoding="utf-8"))
                if cached_ids == self.doc_ids:
                    logger.info("Loading cached corpus embeddings from %s", emb_path)
                    return np.load(emb_path).astype("float32", copy=False)
                logger.warning("Ignoring embedding cache with mismatched document ids: %s", ids_path)

        index_docs = [
            doc if self.args.index_doc_max_chars <= 0 else doc[: self.args.index_doc_max_chars]
            for doc in self.docs
        ]
        embeddings = self.encoder.encode(
            index_docs,
            batch_size=self.args.embedding_batch_size,
            desc=f"Encoding corpus {self.dataset_name}",
        ).astype("float32", copy=False)
        if cache_paths is not None:
            emb_path, ids_path = cache_paths
            np.save(emb_path, embeddings)
            ids_path.write_text(json.dumps(self.doc_ids, ensure_ascii=False), encoding="utf-8")
            logger.info("Saved corpus embedding cache to %s", emb_path)
        return embeddings

    def search_batch(self, queries: list[str], top_k: int) -> list[list[tuple[int, float]]]:
        import numpy as np

        if top_k <= 0 or not queries:
            return [[] for _ in queries]
        top_k = min(top_k, len(self.doc_ids))
        query_texts = [self.encoder.format_query(query) for query in queries]
        query_embeddings = self.encoder.encode(
            query_texts,
            batch_size=self.args.embedding_batch_size,
            desc=f"Encoding queries {self.dataset_name}",
        ).astype("float32", copy=False)
        if self.args.embedding_multi_process:
            # Free worker model memory before optional GPU matrix search.
            self.encoder.close()

        return self._search_embeddings(query_embeddings, top_k)

    def _search_embeddings(self, query_embeddings: Any, top_k: int) -> list[list[tuple[int, float]]]:
        if self.args.embedding_search_device == "cpu":
            return self._search_embeddings_numpy(query_embeddings, top_k)
        try:
            return self._search_embeddings_torch(query_embeddings, top_k)
        except ImportError:
            logger.warning("torch is not importable; falling back to CPU numpy dense retrieval")
            return self._search_embeddings_numpy(query_embeddings, top_k)
        except RuntimeError as exc:
            if "CUDA" not in str(exc) and "cuda" not in str(exc):
                raise
            logger.warning("GPU dense retrieval failed (%s); falling back to CPU numpy dense retrieval", exc)
            return self._search_embeddings_numpy(query_embeddings, top_k)

    def _search_embeddings_numpy(self, query_embeddings: Any, top_k: int) -> list[list[tuple[int, float]]]:
        import numpy as np

        logger.info("Running dense top-k search on CPU with numpy")
        results: list[list[tuple[int, float]]] = []
        batch_size = max(1, self.args.embedding_search_batch_size)
        iterator = range(0, len(query_embeddings), batch_size)
        for start in tqdm(
            iterator,
            desc=f"Dense retrieval {self.dataset_name}",
            unit="batch",
            dynamic_ncols=True,
            ascii=True,
        ):
            batch = query_embeddings[start : start + batch_size]
            scores = batch @ self.doc_embeddings.T
            if top_k >= scores.shape[1]:
                top_indices = np.argsort(-scores, axis=1)
            else:
                partial = np.argpartition(-scores, kth=top_k - 1, axis=1)[:, :top_k]
                partial_scores = np.take_along_axis(scores, partial, axis=1)
                order = np.argsort(-partial_scores, axis=1)
                top_indices = np.take_along_axis(partial, order, axis=1)
            top_scores = np.take_along_axis(scores, top_indices, axis=1)
            for idx_row, score_row in zip(top_indices, top_scores):
                results.append([(int(doc_idx), float(score)) for doc_idx, score in zip(idx_row, score_row)])
        return results

    def _search_embeddings_torch(self, query_embeddings: Any, top_k: int) -> list[list[tuple[int, float]]]:
        import torch

        if self.args.embedding_search_device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.args.embedding_search_device
        if device == "cpu":
            return self._search_embeddings_numpy(query_embeddings, top_k)

        dtype_name = self.args.embedding_search_dtype
        if dtype_name == "auto":
            dtype = torch.float16 if str(device).startswith("cuda") else torch.float32
        else:
            dtype = {
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "float32": torch.float32,
            }[dtype_name]

        logger.info("Running dense top-k search on %s with torch dtype=%s", device, dtype)
        doc_embeddings = torch.as_tensor(self.doc_embeddings, device=device, dtype=dtype).T.contiguous()
        results: list[list[tuple[int, float]]] = []
        batch_size = max(1, self.args.embedding_search_batch_size)
        iterator = range(0, len(query_embeddings), batch_size)
        with torch.no_grad():
            for start in tqdm(
                iterator,
                desc=f"Dense retrieval {self.dataset_name}",
                unit="batch",
                dynamic_ncols=True,
                ascii=True,
            ):
                batch = torch.as_tensor(query_embeddings[start : start + batch_size], device=device, dtype=dtype)
                scores = batch @ doc_embeddings
                top_scores, top_indices = torch.topk(scores, k=top_k, dim=1, largest=True, sorted=True)
                for idx_row, score_row in zip(top_indices.cpu().tolist(), top_scores.float().cpu().tolist()):
                    results.append([(int(doc_idx), float(score)) for doc_idx, score in zip(idx_row, score_row)])
                del batch, scores, top_scores, top_indices
        del doc_embeddings
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        return results


def build_dataset_candidates(
    name: str,
    root: Path,
    args: argparse.Namespace,
    embedding_encoder: EmbeddingEncoder | None = None,
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
    if args.retrieval_backend == "embedding":
        if embedding_encoder is None:
            raise ValueError("embedding_encoder is required when --retrieval_backend embedding")
        retriever = EmbeddingRetriever(name, corpus_ids, docs, embedding_encoder, args)
    else:
        retriever = BM25Index(
            corpus_ids,
            docs,
            k1=args.k1,
            b=args.b,
            index_doc_max_chars=args.index_doc_max_chars,
        )

    records: list[dict[str, Any]] = []
    eligible_queries: list[tuple[str, str, list[str]]] = []
    skipped_no_qrels = 0
    skipped_no_candidates = 0
    total_true_relevant = 0
    total_candidate_relevant = 0

    for query_row in query_rows:
        qid = row_id(query_row)
        query = row_text(query_row, is_doc=False)
        positive_ids = [doc_id for doc_id in qrels.get(qid, {}) if doc_id in corpus]
        if not query or not positive_ids:
            skipped_no_qrels += 1
            continue
        if args.max_queries_per_dataset > 0 and len(eligible_queries) >= args.max_queries_per_dataset:
            break
        eligible_queries.append((qid, query, positive_ids))

    queries = [query for _qid, query, _positive_ids in eligible_queries]
    if args.retrieval_backend == "embedding":
        ranked_by_query = retriever.search_batch(queries, args.candidate_top_k)
    else:
        ranked_by_query = [
            retriever.search(query, args.candidate_top_k)
            for query in tqdm(queries, desc=f"Building BM25 candidates {name}", unit="query", dynamic_ncols=True, ascii=True)
        ]

    for (qid, query, positive_ids), ranked in zip(eligible_queries, ranked_by_query):
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
                    "reason": (
                        f"cmteb_r_{args.retrieval_backend}_retrieved_positive"
                        if is_positive
                        else f"cmteb_r_{args.retrieval_backend}_retrieved_negative"
                    ),
                    "retrieval_rank": rank,
                    "retrieval_score": retrieval_scores.get(doc_idx, 0.0),
                    "retrieval_backend": args.retrieval_backend,
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
        "retrieval_backend": args.retrieval_backend,
        "embedding_model_name_or_path": args.embedding_model_name_or_path
        if args.retrieval_backend == "embedding"
        else None,
        "embedding_multi_process": args.embedding_multi_process if args.retrieval_backend == "embedding" else None,
        "embedding_devices": args.embedding_devices if args.retrieval_backend == "embedding" else None,
        "embedding_search_device": args.embedding_search_device if args.retrieval_backend == "embedding" else None,
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
        "retrieval_backend": args.retrieval_backend,
        "embedding_model_name_or_path": args.embedding_model_name_or_path
        if args.retrieval_backend == "embedding"
        else None,
        "embedding_multi_process": args.embedding_multi_process if args.retrieval_backend == "embedding" else None,
        "embedding_devices": args.embedding_devices if args.retrieval_backend == "embedding" else None,
        "embedding_search_device": args.embedding_search_device if args.retrieval_backend == "embedding" else None,
        "candidate_top_k": args.candidate_top_k,
        "max_queries_per_dataset": args.max_queries_per_dataset,
        "datasets": [],
    }
    embedding_encoder = EmbeddingEncoder(args) if args.retrieval_backend == "embedding" else None
    try:
        for dataset in args.datasets:
            root = dataset_dir(input_dir, dataset)
            records, meta = build_dataset_candidates(dataset.split("/", 1)[-1], root, args, embedding_encoder)
            all_records.extend(records)
            metadata["datasets"].append(meta)
    finally:
        if embedding_encoder is not None:
            embedding_encoder.close()

    write_jsonl(output_file, all_records)
    metadata["num_records"] = len(all_records)
    metadata["num_query_groups"] = len({row["query_id"] for row in all_records})
    metadata_file.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %d candidate records to %s", len(all_records), output_file)
    logger.info("Wrote metadata to %s", metadata_file)


if __name__ == "__main__":
    main()
