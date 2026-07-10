from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from business_eval_vllm import create_vllm_llm, score_with_vllm  # noqa: E402
from data import load_examples, write_jsonl  # noqa: E402
from metrics import add_group_ranks, compute_all_metrics  # noqa: E402
from modeling import DEFAULT_MODEL_NAME  # noqa: E402


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate query-doc-label JSONL with vLLM Qwen3-Reranker scoring.")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", default="outputs/cmteb_r_vllm_eval")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=True)
    parser.add_argument("--no_enable_prefix_caching", dest="enable_prefix_caching", action="store_false")
    parser.add_argument("--sort_by_length", action="store_true", default=True)
    parser.add_argument("--no_sort_by_length", dest="sort_by_length", action="store_false")
    return parser.parse_args()


def choose_instruction(args_instruction: str, examples: list) -> str:
    if args_instruction.strip():
        return args_instruction.strip()
    for ex in examples:
        if ex.instruction.strip():
            return ex.instruction.strip()
    return "Given a query, retrieve relevant documents that answer the query."


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    examples = load_examples(args.test_file)
    instruction = choose_instruction(args.instruction, examples)
    llm = create_vllm_llm(args)

    queries = [ex.query for ex in examples]
    docs = [ex.doc for ex in examples]
    logger.info("Scoring %d query-doc pairs with vLLM batch_size=%d", len(examples), args.batch_size)
    start_time = time.perf_counter()
    scores = score_with_vllm(
        llm,
        queries=queries,
        documents=docs,
        batch_size=args.batch_size,
        instruction=instruction,
        sort_by_length=args.sort_by_length,
    )
    score_time = time.perf_counter() - start_time
    sec_per_example = score_time / max(1, len(scores))
    examples_per_sec = len(scores) / score_time if score_time > 0 else 0.0

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
    rows = add_group_ranks(rows, query_key="group_key")
    overall, per_query = compute_all_metrics(
        rows,
        query_key="group_key",
        relevance_threshold=args.relevance_threshold,
    )
    overall.update(
        {
            "backend": "vllm",
            "vllm_runner": "pooling",
            "vllm_version": getattr(llm, "_memranker_vllm_version", "unknown"),
            "vllm_tokenizer_path": getattr(llm, "_memranker_vllm_tokenizer_path", ""),
            "model_path": args.model_path,
            "test_file": args.test_file,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "dtype": args.dtype,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "score_time_seconds": float(score_time),
            "seconds_per_example": float(sec_per_example),
            "examples_per_second": float(examples_per_sec),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "per_query_metrics.jsonl", per_query)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    logger.info("Wrote vLLM JSONL evaluation outputs to %s", output_dir)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
