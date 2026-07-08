from __future__ import annotations

import argparse
import inspect
import json
import logging
import math
import sys
import time

from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import write_jsonl  # noqa: E402
from evaluate_business import (  # noqa: E402
    DEFAULT_BUSINESS_INSTRUCTION,
    attach_scores_and_ranks,
    build_scoring_inputs,
    compute_business_metrics,
    load_ground_truth,
    load_recall_results,
    write_summary_csv,
    write_summary_xlsx,
)


logger = logging.getLogger(__name__)

DEFAULT_VLLM_MODEL = "Qwen/Qwen3-Reranker-4B"
QWEN3_RERANKER_HF_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}


def build_qwen3_reranker_chat_template(instruction: str) -> str:
    """Jinja chat template matching Qwen3-Reranker yes/no judgement style.

    vLLM versions that support `chat_template` can use this with `LLM.score`.
    If a vLLM build ignores or does not expose `chat_template`, the script logs
    a warning and relies on the model/tokenizer built-in template.
    """
    escaped_instruction = json.dumps(instruction, ensure_ascii=False)
    return (
        "{% set instruction = "
        + escaped_instruction
        + " %}\n"
        "{% if messages is defined %}\n"
        "{% for message in messages %}\n"
        "{% if message['role'] == 'system' %}<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'user' %}<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
        "{% elif message['role'] == 'assistant' %}<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
        "{% endif %}\n"
        "{% endfor %}\n"
        "{% elif query is defined and document is defined %}\n"
        "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and "
        'the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n'
        "<|im_start|>user\n<Instruct>: {{ instruction }}\n<Query>: {{ query }}\n<Document>: {{ document }}<|im_end|>\n"
        "{% endif %}\n"
        "{% if add_generation_prompt is not defined or add_generation_prompt %}"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n{% endif %}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate business recall data with vLLM offline LLM.score reranking."
    )
    parser.add_argument("--gt_file", required=True, help="Excel/CSV file with query-doc ground truth.")
    parser.add_argument("--recall_file", required=True, help="JSON file with recalled docs per query.")
    parser.add_argument("--model_path", default=DEFAULT_VLLM_MODEL, help="Qwen3-Reranker model path or HF id.")
    parser.add_argument("--output_dir", default="outputs/business_eval_vllm")
    parser.add_argument("--instruction", default=DEFAULT_BUSINESS_INSTRUCTION)
    parser.add_argument("--gt_query_col", default="query")
    parser.add_argument("--gt_doc_id_col", default="PageId")
    parser.add_argument("--gt_sheet", default=None, help="Optional Excel sheet name. Defaults to active sheet.")
    parser.add_argument("--recall_id_key", default="id")
    parser.add_argument("--recall_text_key", default="text")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=256, help="Chunk size for vLLM score().")
    parser.add_argument("--top_k_list", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--expected_fbeta_beta",
        type=float,
        default=0.3,
        help="Beta used to choose the dynamic cutoff from normalized score prefix sums.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=True)
    parser.add_argument("--no_enable_prefix_caching", dest="enable_prefix_caching", action="store_false")
    parser.add_argument("--sort_by_length", action="store_true", default=True)
    parser.add_argument("--no_sort_by_length", dest="sort_by_length", action="store_false")
    parser.add_argument(
        "--save_doc_text",
        action="store_true",
        help="Include full document text in predictions.jsonl for debugging.",
    )
    return parser.parse_args()


def filter_supported_kwargs(
    callable_obj: Callable[..., Any],
    kwargs: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        logger.warning("Could not inspect %s signature; passing all kwargs.", context)
        return kwargs

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs

    supported: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in parameters:
            supported[key] = value
        else:
            logger.warning("Current vLLM does not support %s kwarg %r; skipped.", context, key)
    return supported


def create_vllm_llm(args: argparse.Namespace) -> Any:
    try:
        from vllm import LLM
    except ImportError as exc:
        raise RuntimeError(
            "business_eval_vllm.py requires vLLM. Install it on the Linux GPU "
            "machine with a command such as: pip install vllm"
        ) from exc

    llm_kwargs: dict[str, Any] = {
        "model": args.model_path,
        "runner": "pooling",
        "hf_overrides": QWEN3_RERANKER_HF_OVERRIDES,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_length,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": args.enable_prefix_caching,
        "chat_template": build_qwen3_reranker_chat_template(args.instruction),
    }
    filtered_kwargs = filter_supported_kwargs(LLM, llm_kwargs, context="LLM")
    if "chat_template" not in filtered_kwargs:
        logger.warning(
            "This vLLM version does not expose LLM(..., chat_template=...). "
            "Falling back to the model/tokenizer built-in template."
        )
    logger.info("Initializing vLLM with kwargs: %s", json.dumps(_jsonable(filtered_kwargs), ensure_ascii=False))
    return LLM(**filtered_kwargs)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def maybe_extract_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "scores", "data", "value", "outputs"):
            if key in value:
                score = maybe_extract_numeric(value[key])
                if score is not None:
                    return score
    if isinstance(value, np.ndarray):
        return maybe_extract_numeric(value.tolist())
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (int, float, np.floating, np.integer)) for item in value):
            if len(value) == 2:
                return float(value[1])
            if len(value) == 1:
                return float(value[0])
        if len(value) == 1:
            return maybe_extract_numeric(value[0])
        for item in value:
            score = maybe_extract_numeric(item)
            if score is not None:
                return score
    for attr in ("score", "scores", "data", "value"):
        if hasattr(value, attr):
            score = maybe_extract_numeric(getattr(value, attr))
            if score is not None:
                return score
    return None


def extract_vllm_score(output: Any) -> float:
    for candidate in (
        getattr(getattr(output, "outputs", None), "score", None),
        getattr(output, "score", None),
        getattr(output, "outputs", None),
        output,
    ):
        score = maybe_extract_numeric(candidate)
        if score is not None:
            return float(score)
    raise ValueError(f"Could not extract a numeric score from vLLM output: {output!r}")


def build_score_call_kwargs(llm: Any, instruction: str) -> dict[str, Any]:
    requested_kwargs = {
        "use_tqdm": False,
        "instruction": instruction,
    }
    return filter_supported_kwargs(llm.score, requested_kwargs, context="LLM.score")


def score_with_vllm(
    llm: Any,
    queries: list[str],
    documents: list[str],
    batch_size: int,
    instruction: str,
    sort_by_length: bool,
) -> list[float]:
    if len(queries) != len(documents):
        raise ValueError(f"queries/documents length mismatch: {len(queries)} != {len(documents)}")
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if not queries:
        return []

    indexed = [
        (idx, query, document, len(query) + len(document))
        for idx, (query, document) in enumerate(zip(queries, documents, strict=True))
    ]
    if sort_by_length:
        indexed.sort(key=lambda item: item[3])

    scores: list[float | None] = [None] * len(indexed)
    score_kwargs = build_score_call_kwargs(llm, instruction=instruction)
    total_batches = math.ceil(len(indexed) / batch_size)
    progress = tqdm(
        range(0, len(indexed), batch_size),
        total=total_batches,
        desc="vLLM scoring",
        unit="batch",
        dynamic_ncols=True,
        ascii=True,
    )
    for start in progress:
        batch = indexed[start : start + batch_size]
        batch_queries = [item[1] for item in batch]
        batch_documents = [item[2] for item in batch]
        outputs = llm.score(batch_queries, batch_documents, **score_kwargs)
        if len(outputs) != len(batch):
            raise ValueError(f"vLLM returned {len(outputs)} outputs for {len(batch)} input pairs")
        for (original_idx, _, _, _), output in zip(batch, outputs, strict=True):
            scores[original_idx] = extract_vllm_score(output)
        progress.set_postfix(scored=min(start + len(batch), len(indexed)))

    final_scores: list[float] = []
    bad_values = 0
    for idx, score in enumerate(scores):
        if score is None:
            raise ValueError(f"Missing score for pair index {idx}")
        score_float = float(score)
        if not math.isfinite(score_float):
            bad_values += 1
        final_scores.append(score_float)
    if len(final_scores) != len(queries):
        raise ValueError(f"Returned score count mismatch: {len(final_scores)} != {len(queries)}")
    if bad_values:
        logger.warning("vLLM scores contain %d NaN/inf values.", bad_values)
    return final_scores


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()

    ground_truth = load_ground_truth(
        args.gt_file,
        query_col=args.gt_query_col,
        doc_id_col=args.gt_doc_id_col,
        sheet_name=args.gt_sheet,
    )
    logger.info("Loaded ground truth query count: %d", len(ground_truth))
    recall_results = load_recall_results(
        args.recall_file,
        id_key=args.recall_id_key,
        text_key=args.recall_text_key,
    )
    logger.info("Loaded recall query count: %d", len(recall_results))
    _input_texts, mapping, skipped_queries = build_scoring_inputs(
        recall_results,
        ground_truth,
        instruction=args.instruction,
    )
    if skipped_queries:
        logger.warning("Skipped %d recall queries not found in ground truth", skipped_queries)
    if not mapping:
        raise ValueError("No query-document pairs to score after matching recall data to ground truth.")
    logger.info("Total query-doc pairs to score: %d", len(mapping))

    llm = create_vllm_llm(args)
    queries = [row["query"] for row in mapping]
    documents = [row["doc"] for row in mapping]

    start_time = time.perf_counter()
    scores = score_with_vllm(
        llm,
        queries=queries,
        documents=documents,
        batch_size=args.batch_size,
        instruction=args.instruction,
        sort_by_length=args.sort_by_length,
    )
    score_time = time.perf_counter() - start_time
    sec_per_example = score_time / max(1, len(scores))
    examples_per_sec = len(scores) / score_time if score_time > 0 else 0.0
    logger.info(
        "vLLM scoring finished: pairs=%d time=%.3fs examples/s=%.3f",
        len(scores),
        score_time,
        examples_per_sec,
    )

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
            "backend": "vllm",
            "vllm_runner": "pooling",
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "sort_by_length": args.sort_by_length,
            "score_time_seconds": float(score_time),
            "seconds_per_example": float(sec_per_example),
            "examples_per_second": float(examples_per_sec),
            "num_scored_pairs": len(scores),
            "num_scored_queries": len({row["query"] for row in mapping}),
            "model_path": args.model_path,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gt_file": args.gt_file,
            "recall_file": args.recall_file,
            "gt_query_col": args.gt_query_col,
            "gt_doc_id_col": args.gt_doc_id_col,
            "expected_fbeta_beta": args.expected_fbeta_beta,
            "skipped_recall_queries_without_gt": skipped_queries,
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
    logger.info("Wrote vLLM business evaluation outputs to %s", output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
