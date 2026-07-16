from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from data import load_examples
from modernbert_utils import (
    DEFAULT_MODERNBERT_MODEL_NAME,
    evaluate_modernbert_examples,
    load_modernbert_model_and_tokenizer,
    torch,
    write_eval_outputs,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a ModernBERT pointwise reranker checkpoint.")
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--model_path", default=DEFAULT_MODERNBERT_MODEL_NAME)
    parser.add_argument("--output_dir", default="outputs/modernbert_eval")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--default_instruction", default="")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional transformers attention backend, for example sdpa, eager, or flash_attention_2.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    if torch is None:
        raise RuntimeError("torch is required for ModernBERT evaluation")

    examples = load_examples(args.test_file, default_instruction=args.default_instruction)
    model, tokenizer = load_modernbert_model_and_tokenizer(
        args.model_path,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=False,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    start_time = time.perf_counter()
    overall, per_query, predictions = evaluate_modernbert_examples(
        model,
        tokenizer,
        examples,
        max_length=args.max_length,
        batch_size=args.batch_size,
        relevance_threshold=args.relevance_threshold,
        device=device,
    )
    score_time = time.perf_counter() - start_time
    overall.update(
        {
            "backend": "modernbert",
            "model_path": args.model_path,
            "test_file": args.test_file,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "attn_implementation": args.attn_implementation,
            "score_time_seconds": float(score_time),
            "seconds_per_example": score_time / max(1, len(examples)),
            "examples_per_second": len(examples) / score_time if score_time > 0 else 0.0,
        }
    )
    write_eval_outputs(args.output_dir, overall, per_query, predictions)
    logger.info("Wrote ModernBERT evaluation outputs to %s", args.output_dir)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
