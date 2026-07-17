from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from data import load_examples, read_json_records, write_jsonl
from evaluate_business import get_torch_cuda_memory_mib, cleanup_cuda_memory
from evaluate_business_crossencoder import load_cross_encoder, predict_cross_encoder
from evaluate_jsonl_vllm import (
    PASSTHROUGH_METADATA_FIELDS,
    choose_instruction,
    compute_dynamic_beta_metrics,
    compute_inverted_score_diagnostics,
    write_csv,
)
from metrics import add_group_ranks, compute_all_metrics
from modernbert_utils import format_instruction_query, torch


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate query-doc-label JSONL with a Sentence-Transformers CrossEncoder reranker."
    )
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_dir", default="outputs/jsonl_crossencoder_eval")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--expected_fbeta_betas", type=float, nargs="+", default=[0.2, 0.3, 0.5, 0.7, 1.0])
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
    parser.add_argument("--save_doc_text", action="store_true", default=True)
    parser.add_argument("--no_save_doc_text", dest="save_doc_text", action="store_false")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")

    examples = load_examples(args.test_file)
    raw_records = read_json_records(args.test_file)
    instruction = choose_instruction(args.instruction, examples)

    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = load_cross_encoder(args)
    pairs = [(format_instruction_query(instruction, ex.query), ex.doc) for ex in examples]
    logger.info(
        "Scoring %d query-doc pairs with CrossEncoder model=%s batch_size=%d",
        len(pairs),
        args.model_path,
        args.batch_size,
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

    rows = []
    for ex, score, raw in zip(examples, scores, raw_records, strict=False):
        row = {
            "group_key": ex.group_key,
            "query": ex.query,
            "query_id": ex.query_id,
            "qid": raw.get("qid", ""),
            "doc_id": ex.doc_id,
            "doc": ex.doc if args.save_doc_text else "",
            "label": ex.label,
            "raw_label": ex.raw_label,
            "score": float(score),
            "retrieval_rank": raw.get("retrieval_rank"),
            "retrieval_score": raw.get("retrieval_score"),
            "true_relevant_count": raw.get("true_relevant_count"),
            "candidate_relevant_count": raw.get("candidate_relevant_count"),
            "reason": ex.reason,
        }
        for field in PASSTHROUGH_METADATA_FIELDS:
            if field in raw:
                row[field] = raw.get(field)
        rows.append(row)

    rows = add_group_ranks(rows, query_key="group_key")
    overall, per_query = compute_all_metrics(
        rows,
        query_key="group_key",
        relevance_threshold=args.relevance_threshold,
    )
    overall.update(compute_inverted_score_diagnostics(rows, args.relevance_threshold))
    beta_per_query, beta_summary, beta_overall = compute_dynamic_beta_metrics(
        rows,
        betas=args.expected_fbeta_betas,
        relevance_threshold=args.relevance_threshold,
    )
    overall.update(
        {
            "backend": "crossencoder",
            "model_path": args.model_path,
            "test_file": args.test_file,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "precision": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
            "dtype": "bf16" if args.bf16 else "fp16" if args.fp16 else "fp32",
            "attn_implementation": args.attn_implementation or "",
            "score_activation": args.score_activation,
            "local_files_only": args.local_files_only,
            "expected_fbeta_betas": args.expected_fbeta_betas,
            "score_time_seconds": float(score_time),
            "seconds_per_example": float(sec_per_example),
            "examples_per_second": float(examples_per_sec),
            "cuda_peak_allocated_mib": cuda_peak_memory["cuda_peak_allocated_mib"],
            "cuda_peak_reserved_mib": cuda_peak_memory["cuda_peak_reserved_mib"],
        }
    )
    overall.update(beta_overall)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overall_metrics.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "per_query_metrics.jsonl", per_query)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_jsonl(output_dir / "beta_f1_per_query.jsonl", beta_per_query)
    write_csv(output_dir / "beta_f1_summary.csv", beta_summary)
    (output_dir / "beta_f1_summary.json").write_text(
        json.dumps(beta_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    del model
    cleanup_cuda_memory()

    logger.info("Wrote CrossEncoder JSONL evaluation outputs to %s", output_dir)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
