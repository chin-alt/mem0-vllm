from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import torch

from data import read_json_records, record_to_doc
from modernbert_utils import (
    DEFAULT_MODERNBERT_MODEL_NAME,
    load_modernbert_model_and_tokenizer,
    predict_modernbert_pairs,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank documents for one query with a ModernBERT BCE reranker.")
    parser.add_argument("--model_path", default=DEFAULT_MODERNBERT_MODEL_NAME)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--docs_file", required=True, help="JSONL/JSON docs. Each row may contain doc or title/abstract.")
    parser.add_argument("--output_file", default="predictions_ranked.json")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional transformers attention backend, for example sdpa, eager, or flash_attention_2.",
    )
    return parser.parse_args()


def load_docs(path: str | Path) -> list[dict[str, Any]]:
    rows = read_json_records(path)
    docs = []
    skipped = 0
    for idx, row in enumerate(rows):
        doc = record_to_doc(row)
        if not doc:
            skipped += 1
            continue
        docs.append({"doc": doc, "source_index": idx, "raw": row})
    if skipped:
        logger.warning("Skipped %d docs without usable text", skipped)
    return docs


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")

    docs = load_docs(args.docs_file)
    if not docs:
        raise ValueError(f"No usable docs found in {args.docs_file}")

    model, tokenizer = load_modernbert_model_and_tokenizer(
        args.model_path,
        bf16=args.bf16,
        fp16=args.fp16,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    queries = [args.query] * len(docs)
    scores = predict_modernbert_pairs(
        model,
        tokenizer,
        instruction=args.instruction,
        queries=queries,
        docs=[row["doc"] for row in docs],
        max_length=args.max_length,
        batch_size=args.batch_size,
        device=device,
    )

    ranked = []
    for row, score in zip(docs, scores):
        ranked.append(
            {
                "doc": row["doc"],
                "score": float(score),
                "source_index": row["source_index"],
                "raw": row["raw"],
            }
        )
    ranked.sort(key=lambda item: item["score"], reverse=True)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank

    top_k = ranked[: max(0, args.top_k)]
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(top_k, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote top-%d predictions to %s", len(top_k), output_path)
    print(json.dumps(top_k, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
