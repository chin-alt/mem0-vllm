from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

from data import RerankerExample, record_to_doc, read_json_records, write_jsonl
from metrics import add_group_ranks, compute_all_metrics


logger = logging.getLogger(__name__)

DEFAULT_MODERNBERT_MODEL_NAME = "answerdotai/ModernBERT-base"


try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]


def format_instruction_query(instruction: str, query: str) -> str:
    instruction = (instruction or "").strip()
    query = (query or "").strip()
    if instruction:
        return f"{instruction}\n\nQuery: {query}"
    return f"Query: {query}"


def tokenize_modernbert_examples(
    tokenizer: Any,
    examples: list[RerankerExample],
    max_length: int,
    device: Any | None = None,
) -> dict[str, Any]:
    texts_a = [format_instruction_query(ex.instruction, ex.query) for ex in examples]
    texts_b = [ex.doc for ex in examples]
    batch = tokenizer(
        texts_a,
        texts_b,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch


def tokenize_modernbert_pairs(
    tokenizer: Any,
    instruction: str,
    queries: list[str],
    docs: list[str],
    max_length: int,
    device: Any | None = None,
) -> dict[str, Any]:
    texts_a = [format_instruction_query(instruction, query) for query in queries]
    batch = tokenizer(
        texts_a,
        docs,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if device is not None:
        batch = {key: value.to(device) for key, value in batch.items()}
    return batch


def sigmoid_array(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def torch_dtype_from_flags(bf16: bool = False, fp16: bool = False) -> Any | None:
    if torch is None:
        return None
    if bf16:
        return torch.bfloat16
    if fp16:
        return torch.float16
    return None


def load_modernbert_model_and_tokenizer(
    model_name_or_path: str,
    bf16: bool = False,
    fp16: bool = False,
    gradient_checkpointing: bool = False,
    local_files_only: bool = False,
    attn_implementation: str | None = None,
) -> tuple[Any, Any]:
    if torch is None:
        raise RuntimeError("torch is required for ModernBERT training/evaluation")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dtype = torch_dtype_from_flags(bf16=bf16, fp16=fp16)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    model_kwargs = {
        "num_labels": 1,
        "problem_type": "regression",
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": local_files_only,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, **model_kwargs)
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False
    return model, tokenizer


def predict_modernbert_examples(
    model: Any,
    tokenizer: Any,
    examples: list[RerankerExample],
    max_length: int,
    batch_size: int,
    device: str | None = None,
) -> list[float]:
    if torch is None:
        raise RuntimeError("torch is required for ModernBERT prediction")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    scores: list[float] = []
    total_batches = math.ceil(len(examples) / batch_size) if examples else 0
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(examples), batch_size),
            total=total_batches,
            desc="ModernBERT scoring",
            unit="batch",
            dynamic_ncols=True,
            ascii=True,
        ):
            batch_examples = examples[start : start + batch_size]
            encoded = tokenize_modernbert_examples(
                tokenizer,
                batch_examples,
                max_length=max_length,
                device=device,
            )
            logits = model(**encoded).logits.squeeze(-1).detach().float().cpu().numpy()
            scores.extend(sigmoid_array(logits).tolist())
    return [float(score) for score in scores]


def predict_modernbert_pairs(
    model: Any,
    tokenizer: Any,
    instruction: str,
    queries: list[str],
    docs: list[str],
    max_length: int,
    batch_size: int,
    device: str | None = None,
) -> list[float]:
    if torch is None:
        raise RuntimeError("torch is required for ModernBERT prediction")
    if len(queries) != len(docs):
        raise ValueError(f"queries/docs length mismatch: {len(queries)} != {len(docs)}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    scores: list[float] = []
    total_batches = math.ceil(len(queries) / batch_size) if queries else 0
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(queries), batch_size),
            total=total_batches,
            desc="ModernBERT scoring",
            unit="batch",
            dynamic_ncols=True,
            ascii=True,
        ):
            batch_queries = queries[start : start + batch_size]
            batch_docs = docs[start : start + batch_size]
            encoded = tokenize_modernbert_pairs(
                tokenizer,
                instruction=instruction,
                queries=batch_queries,
                docs=batch_docs,
                max_length=max_length,
                device=device,
            )
            logits = model(**encoded).logits.squeeze(-1).detach().float().cpu().numpy()
            scores.extend(sigmoid_array(logits).tolist())
    return [float(score) for score in scores]


def examples_to_prediction_rows(examples: list[RerankerExample], scores: list[float]) -> list[dict[str, Any]]:
    rows = []
    for ex, score in zip(examples, scores, strict=False):
        rows.append(
            {
                "group_key": ex.group_key,
                "query": ex.query,
                "query_id": ex.query_id,
                "doc_id": ex.doc_id,
                "doc": ex.doc,
                "label": ex.label,
                "raw_label": ex.raw_label,
                "score": float(score),
                "reason": ex.reason,
            }
        )
    return add_group_ranks(rows, query_key="group_key")


def evaluate_modernbert_examples(
    model: Any,
    tokenizer: Any,
    examples: list[RerankerExample],
    max_length: int,
    batch_size: int,
    relevance_threshold: float,
    device: str | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    scores = predict_modernbert_examples(
        model,
        tokenizer,
        examples,
        max_length=max_length,
        batch_size=batch_size,
        device=device,
    )
    rows = examples_to_prediction_rows(examples, scores)
    overall, per_query = compute_all_metrics(
        rows,
        query_key="group_key",
        relevance_threshold=relevance_threshold,
    )
    return overall, per_query, rows


def docs_file_to_docs(path: str | Path) -> list[str]:
    docs = []
    for row in read_json_records(path):
        doc = record_to_doc(row)
        if doc:
            docs.append(doc)
    return docs


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_eval_outputs(
    output_dir: str | Path,
    overall: dict[str, Any],
    per_query: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "overall_metrics.json", overall)
    write_jsonl(output_dir / "per_query_metrics.jsonl", per_query)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
