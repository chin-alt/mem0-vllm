from __future__ import annotations

import argparse
import json
import logging
import math
import time

from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from data import write_jsonl
from evaluate_business import (
    attach_scores_and_ranks,
    build_scoring_inputs,
    cleanup_accelerator_memory,
    compute_business_metrics,
    get_torch_accelerator_memory_mib,
    load_ground_truth,
    load_recall_results,
    reset_torch_accelerator_memory_stats,
    write_summary_csv,
    write_summary_xlsx,
)


logger = logging.getLogger(__name__)
DEFAULT_GTE_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GTE sequence-classification reranker locally on an Ascend NPU."
    )
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--recall_file", required=True)
    parser.add_argument("--model_path", default=DEFAULT_GTE_MODEL)
    parser.add_argument("--output_dir", default="outputs/business_gte_310p")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--gt_query_col", default="query")
    parser.add_argument("--gt_doc_id_col", default="PageId")
    parser.add_argument("--gt_sheet", default=None)
    parser.add_argument("--recall_id_key", default="id")
    parser.add_argument("--recall_text_key", default="text")
    parser.add_argument("--top_k_list", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--expected_fbeta_beta", type=float, default=0.3)
    parser.add_argument(
        "--score_activation",
        choices=["sigmoid", "identity"],
        default="sigmoid",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--save_doc_text", action="store_true")
    return parser.parse_args()


def format_gte_query(query: str, instruction: str = "") -> str:
    query = str(query).strip()
    instruction = str(instruction).strip()
    return f"{instruction}\n\n{query}" if instruction else query


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64, copy=False), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_npu_stack() -> tuple[Any, Any]:
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "GTE Ascend inference requires matching torch and torch-npu wheels. "
            "Install requirements-ascend-gte-310p-py39.txt in an isolated environment."
        ) from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch_npu imported, but torch.npu.is_available() is False")
    return torch, torch_npu


def resolve_dtype(torch: Any, name: str) -> Any:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def npu_synchronize(torch: Any) -> None:
    synchronize = getattr(torch.npu, "synchronize", None)
    if synchronize is not None:
        synchronize()


def load_model_and_tokenizer(args: argparse.Namespace, torch: Any) -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=resolve_dtype(torch, args.dtype),
    )
    model.eval()
    model.to(args.device)
    return model, tokenizer


def validate_rope_buffers(model: Any, torch: Any) -> None:
    damaged: list[str] = []
    checked = 0
    for name, value in model.named_buffers():
        if value.numel() == 0 or not any(key in name for key in ("inv_freq", "cos_cached", "sin_cached")):
            continue
        checked += 1
        if not bool(torch.isfinite(value).all().item()):
            damaged.append(name)
    if damaged:
        raise RuntimeError(
            "GTE RoPE buffers contain NaN/Inf: " + ", ".join(damaged[:8])
        )
    logger.info("GTE RoPE buffer health check passed (%d buffers checked)", checked)


def score_pairs(
    model: Any,
    tokenizer: Any,
    queries: list[str],
    docs: list[str],
    args: argparse.Namespace,
    torch: Any,
) -> list[float]:
    if len(queries) != len(docs):
        raise ValueError(f"queries/docs length mismatch: {len(queries)} != {len(docs)}")

    scores: list[float] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(queries), args.batch_size),
            total=math.ceil(len(queries) / args.batch_size) if queries else 0,
            desc="GTE NPU scoring",
            unit="batch",
            dynamic_ncols=True,
            ascii=True,
        ):
            batch_queries = [
                format_gte_query(value, args.instruction)
                for value in queries[start : start + args.batch_size]
            ]
            batch_docs = docs[start : start + args.batch_size]
            encoded = tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                pad_to_multiple_of=args.pad_to_multiple_of or None,
                return_tensors="pt",
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            logits = model(**encoded, return_dict=True).logits.reshape(-1).float().cpu().numpy()
            if args.score_activation == "sigmoid":
                logits = sigmoid(logits)
            scores.extend(float(value) for value in logits.tolist())
    npu_synchronize(torch)
    return scores


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("--batch_size and --max_length must be positive")

    torch, _torch_npu = load_npu_stack()
    torch.npu.set_device(args.device)
    logger.info(
        "Ascend runtime torch=%s torch_npu=%s device=%s",
        torch.__version__,
        getattr(_torch_npu, "__version__", "unknown"),
        args.device,
    )

    ground_truth = load_ground_truth(
        args.gt_file,
        query_col=args.gt_query_col,
        doc_id_col=args.gt_doc_id_col,
        sheet_name=args.gt_sheet,
    )
    recall_results = load_recall_results(
        args.recall_file,
        id_key=args.recall_id_key,
        text_key=args.recall_text_key,
    )
    _unused, mapping, skipped_queries = build_scoring_inputs(
        recall_results,
        ground_truth,
        instruction="",
    )
    if not mapping:
        raise ValueError("No query-document pairs remain after matching recall data to ground truth")

    reset_torch_accelerator_memory_stats()
    model, tokenizer = load_model_and_tokenizer(args, torch)
    validate_rope_buffers(model, torch)
    queries = [str(row["query"]) for row in mapping]
    docs = [str(row["doc"]) for row in mapping]

    # Compile common kernels outside the measured pass. The measured pass still
    # includes any additional dynamic-shape compilation required by later batches.
    warmup_count = min(args.batch_size, len(queries))
    warmup_start = time.perf_counter()
    score_pairs(
        model,
        tokenizer,
        queries[:warmup_count],
        docs[:warmup_count],
        args,
        torch,
    )
    warmup_seconds = time.perf_counter() - warmup_start

    start = time.perf_counter()
    scores = score_pairs(model, tokenizer, queries, docs, args, torch)
    score_seconds = time.perf_counter() - start
    seconds_per_example = score_seconds / max(1, len(scores))

    ranked_predictions = attach_scores_and_ranks(
        mapping,
        scores,
        ground_truth,
        save_doc_text=args.save_doc_text,
    )
    metrics, per_query = compute_business_metrics(
        ranked_predictions,
        ground_truth,
        args.top_k_list,
        seconds_per_example=seconds_per_example,
        expected_fbeta_beta=args.expected_fbeta_beta,
    )
    memory = get_torch_accelerator_memory_mib()
    metrics.update(
        {
            "backend": "gte_torch_npu",
            "model_path": args.model_path,
            "device": args.device,
            "dtype": args.dtype,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "score_activation": args.score_activation,
            "instruction": args.instruction,
            "warmup_seconds": warmup_seconds,
            "score_time_seconds": score_seconds,
            "seconds_per_example": seconds_per_example,
            "examples_per_second": len(scores) / score_seconds if score_seconds else 0.0,
            "skipped_recall_queries_without_gt": skipped_queries,
            **memory,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_query_metrics.jsonl", per_query)
    write_jsonl(output_dir / "predictions.jsonl", ranked_predictions)
    write_summary_csv(output_dir / "business_eval.csv", per_query)
    wrote_xlsx = write_summary_xlsx(output_dir / "business_eval.xlsx", per_query)
    metrics["summary_csv"] = str(output_dir / "business_eval.csv")
    metrics["summary_xlsx"] = str(output_dir / "business_eval.xlsx") if wrote_xlsx else ""
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    del model
    cleanup_accelerator_memory()
    logger.info("Wrote GTE Ascend evaluation outputs to %s", output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
