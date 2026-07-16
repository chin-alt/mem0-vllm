from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import time

from pathlib import Path
from typing import Any

import numpy as np

from evaluate_business import (
    attach_scores_and_ranks,
    build_scoring_inputs,
    cleanup_cuda_memory,
    compute_business_metrics,
    get_torch_cuda_memory_mib,
    load_ground_truth,
    load_recall_results,
    write_summary_csv,
    write_summary_xlsx,
)
from modernbert_utils import format_instruction_query, torch
from data import write_jsonl


logger = logging.getLogger(__name__)

DEFAULT_BUSINESS_INSTRUCTION = (
    "Given a user query, retrieve relevant documents that answer the query."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Sentence-Transformers CrossEncoder rerankers on a business recall dataset."
    )
    parser.add_argument("--gt_file", required=True, help="Excel/CSV file with query-doc ground truth.")
    parser.add_argument("--recall_file", required=True, help="JSON file with recalled docs per query.")
    parser.add_argument("--model_path", required=True, help="CrossEncoder model path or Hugging Face id.")
    parser.add_argument("--output_dir", default="outputs/business_crossencoder_eval")
    parser.add_argument("--instruction", default=DEFAULT_BUSINESS_INSTRUCTION)
    parser.add_argument("--gt_query_col", default="query")
    parser.add_argument("--gt_doc_id_col", default="PageId")
    parser.add_argument("--gt_sheet", default=None, help="Optional Excel sheet name. Defaults to active sheet.")
    parser.add_argument("--recall_id_key", default="id")
    parser.add_argument("--recall_text_key", default="text")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--top_k_list", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--expected_fbeta_beta",
        type=float,
        default=0.3,
        help="Beta used to choose the dynamic cutoff from normalized score prefix sums.",
    )
    parser.add_argument(
        "--score_activation",
        choices=["sigmoid", "identity", "default"],
        default="sigmoid",
        help=(
            "sigmoid uses raw CrossEncoder logits and applies sigmoid, matching BCE soft-label training. "
            "identity keeps raw logits. default uses CrossEncoder.predict default activation."
        ),
    )
    parser.add_argument("--attn_implementation", default=None)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--save_doc_text",
        action="store_true",
        help="Include full document text in predictions.jsonl for debugging.",
    )
    return parser.parse_args()


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return {key: value for key, value in kwargs.items() if value is not None}
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if key in signature.parameters and value is not None}


def torch_dtype_from_flags(args: argparse.Namespace) -> Any | None:
    if torch is None:
        return None
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def load_cross_encoder(args: argparse.Namespace) -> Any:
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
    except ImportError as exc:
        raise RuntimeError(
            "sentence-transformers is required for CrossEncoder business evaluation. "
            "Install requirements.txt first."
        ) from exc

    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_from_flags(args),
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    tokenizer_kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    base_kwargs = {
        "max_length": args.max_length,
        "max_seq_length": args.max_length,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    candidates = [
        {**base_kwargs, "automodel_args": model_kwargs, "tokenizer_args": tokenizer_kwargs},
        {**base_kwargs, "model_kwargs": model_kwargs, "tokenizer_kwargs": tokenizer_kwargs},
        base_kwargs,
    ]
    last_type_error: TypeError | None = None
    for kwargs in candidates:
        try:
            return CrossEncoder(args.model_path, **filter_kwargs(CrossEncoder.__init__, kwargs))
        except TypeError as exc:
            last_type_error = exc
    raise last_type_error or RuntimeError(f"Failed to load CrossEncoder from {args.model_path}")


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def flatten_scores(values: Any) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float64)
    if scores.ndim == 0:
        scores = scores.reshape(1)
    if scores.ndim > 1:
        if scores.shape[-1] == 1:
            scores = scores.reshape(-1)
        else:
            scores = scores[:, -1]
    return scores.astype(np.float64, copy=False)


def predict_cross_encoder(
    model: Any,
    pairs: list[tuple[str, str]],
    batch_size: int,
    score_activation: str,
) -> list[float]:
    predict_kwargs = {
        "batch_size": batch_size,
        "show_progress_bar": True,
    }
    if score_activation in {"sigmoid", "identity"}:
        if torch is not None:
            predict_kwargs["activation_fn"] = torch.nn.Identity()
        else:
            logger.warning("torch is unavailable; falling back to CrossEncoder default activation.")
    filtered_kwargs = filter_kwargs(model.predict, predict_kwargs)
    raw_scores = flatten_scores(model.predict(pairs, **filtered_kwargs))
    if score_activation == "sigmoid" and "activation_fn" in filtered_kwargs:
        raw_scores = sigmoid(raw_scores)
    if not np.all(np.isfinite(raw_scores)):
        bad_count = int((~np.isfinite(raw_scores)).sum())
        logger.warning("CrossEncoder scores contain %d NaN/inf values; replacing with 0.0.", bad_count)
        raw_scores = np.nan_to_num(raw_scores, nan=0.0, posinf=1.0, neginf=0.0)
    return [float(score) for score in raw_scores.tolist()]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")

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
    _input_texts, mapping, skipped_queries = build_scoring_inputs(
        recall_results,
        ground_truth,
        instruction=args.instruction,
    )
    if skipped_queries:
        logger.warning("Skipped %d recall queries not found in ground truth", skipped_queries)
    if not mapping:
        raise ValueError("No query-document pairs to score after matching recall data to ground truth.")

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = load_cross_encoder(args)
    pairs = [
        (format_instruction_query(args.instruction, str(row["query"])), str(row["doc"]))
        for row in mapping
    ]

    logger.info(
        "Scoring %d business query-document pairs with CrossEncoder model=%s",
        len(pairs),
        args.model_path,
    )
    start_time = time.perf_counter()
    scores = predict_cross_encoder(
        model,
        pairs,
        batch_size=args.batch_size,
        score_activation=args.score_activation,
    )
    score_time = time.perf_counter() - start_time
    sec_per_example = score_time / max(1, len(scores))
    examples_per_sec = len(scores) / score_time if score_time > 0 else 0.0
    cuda_peak_memory = get_torch_cuda_memory_mib()

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
        seconds_per_example=sec_per_example,
        expected_fbeta_beta=args.expected_fbeta_beta,
    )
    metrics.update(
        {
            "backend": "crossencoder",
            "model_path": args.model_path,
            "gt_file": args.gt_file,
            "recall_file": args.recall_file,
            "gt_query_col": args.gt_query_col,
            "gt_doc_id_col": args.gt_doc_id_col,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "expected_fbeta_beta": args.expected_fbeta_beta,
            "precision": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
            "dtype": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
            "attn_implementation": args.attn_implementation or "",
            "score_activation": args.score_activation,
            "score_time_seconds": float(score_time),
            "seconds_per_example": float(sec_per_example),
            "examples_per_second": float(examples_per_sec),
            "skipped_recall_queries_without_gt": skipped_queries,
            "cuda_peak_allocated_mib": cuda_peak_memory["cuda_peak_allocated_mib"],
            "cuda_peak_reserved_mib": cuda_peak_memory["cuda_peak_reserved_mib"],
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
    cleanup_cuda_memory()

    logger.info("Wrote CrossEncoder business evaluation outputs to %s", output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
