#!/usr/bin/env python3
"""Evaluate business reranking data with pure ATB yes/no first-token logits."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
import os
import shutil
import sys
import time

from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for import_path in (PROJECT_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

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
from scripts.prepare_qwen3_reranker_calibration import (  # noqa: E402
    truncate_prompt_document,
)
from scripts.run_atb_sharded_model import prepare_flat_model  # noqa: E402


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Qwen3-Reranker W8A8SC through pure ATB first-token "
            "yes/no logits."
        )
    )
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--gt-file", required=True)
    parser.add_argument("--recall-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--instruction", default=DEFAULT_BUSINESS_INSTRUCTION)
    parser.add_argument("--gt-query-col", default="query")
    parser.add_argument("--gt-doc-id-col", default="PageId")
    parser.add_argument("--gt-sheet", default=None)
    parser.add_argument("--recall-id-key", default="id")
    parser.add_argument("--recall-text-key", default="text")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-queries",
        type=int,
        default=0,
        help="Evaluate the first N matching queries; 0 evaluates the full dataset.",
    )
    parser.add_argument("--top-k-list", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--expected-fbeta-beta", type=float, default=0.3)
    parser.add_argument("--sort-by-length", action="store_true", default=True)
    parser.add_argument(
        "--no-sort-by-length",
        dest="sort_by_length",
        action="store_false",
    )
    parser.add_argument("--sort-descending", action="store_true")
    parser.add_argument("--save-doc-text", action="store_true")
    return parser.parse_args()


def single_token_id(tokenizer: Any, text: str) -> int:
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    if len(token_ids) != 1:
        raise ValueError(
            f"expected {text!r} to use one tokenizer id, found {token_ids}"
        )
    return int(token_ids[0])


def yes_no_probabilities(
    logits: Any,
    yes_token_id: int,
    no_token_id: int,
) -> list[float]:
    import torch

    yes_logits = logits[..., yes_token_id].float()
    no_logits = logits[..., no_token_id].float()
    probabilities = torch.sigmoid(yes_logits - no_logits)
    return [
        float(value)
        for value in probabilities.detach().reshape(-1).cpu().tolist()
    ]


class YesNoLogitCapture:
    """Capture normalized yes probability without changing ATB token choice."""

    def __init__(
        self,
        original_chooser: Callable[..., Any],
        yes_token_id: int,
        no_token_id: int,
    ) -> None:
        self.original_chooser = original_chooser
        self.yes_token_id = yes_token_id
        self.no_token_id = no_token_id
        self.scores: list[float] = []

    def __call__(self, logits: Any, *args: Any, **kwargs: Any) -> Any:
        self.scores.extend(
            yes_no_probabilities(
                logits,
                yes_token_id=self.yes_token_id,
                no_token_id=self.no_token_id,
            )
        )
        return self.original_chooser(logits, *args, **kwargs)


def load_saved_logit_scores(
    logits_dir: Path,
    yes_token_id: int,
    no_token_id: int,
) -> list[float]:
    import torch

    logit_files = sorted(logits_dir.glob("*.pth"))
    if not logit_files:
        raise RuntimeError(
            "ATB logits-save fallback produced no .pth files under "
            f"{logits_dir}"
        )
    scores: list[float] = []
    for path in logit_files:
        logits = torch.load(path, map_location="cpu")
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                f"ATB logits file must contain a tensor: {path} "
                f"({type(logits).__name__})"
            )
        scores.extend(
            yes_no_probabilities(
                logits,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
            )
        )
    return scores


def limit_to_matching_queries(
    ground_truth: dict[str, Any],
    recall_results: dict[str, list[dict[str, str]]],
    max_queries: int,
) -> dict[str, Any]:
    if max_queries == 0:
        return ground_truth
    selected = [
        query
        for query in ground_truth
        if query in recall_results
    ][:max_queries]
    return {query: ground_truth[query] for query in selected}


def prepare_prompts(
    tokenizer: Any,
    mapping: list[dict[str, Any]],
    instruction: str,
    max_length: int,
) -> tuple[list[str], list[int], int]:
    prompts: list[str] = []
    token_lengths: list[int] = []
    truncated_count = 0
    for row in mapping:
        prompt, token_length, _original_length, truncated = (
            truncate_prompt_document(
                tokenizer=tokenizer,
                instruction=instruction,
                query=row["query"],
                document=row["doc"],
                backend="generate",
                max_length=max_length,
            )
        )
        prompts.append(prompt)
        token_lengths.append(token_length)
        truncated_count += int(truncated)
    return prompts, token_lengths, truncated_count


def score_prompts(
    pa_runner: Any,
    generate_module: Any,
    prompts: list[str],
    token_lengths: list[int],
    batch_size: int,
    sort_by_length: bool,
    sort_descending: bool,
    yes_token_id: int,
    no_token_id: int,
    rank: int,
    logits_dir: Path,
) -> tuple[list[float], float, list[dict[str, Any]]]:
    original_chooser = getattr(generate_module, "next_token_chooser", None)
    capture = (
        YesNoLogitCapture(
            original_chooser=original_chooser,
            yes_token_id=yes_token_id,
            no_token_id=no_token_id,
        )
        if original_chooser is not None
        else None
    )
    capture_mode = "memory-hook" if capture is not None else "atb-logits-save"
    if capture is None:
        logits_dir.mkdir(parents=True, exist_ok=True)
        os.environ["ATB_LLM_LOGITS_SAVE_ENABLE"] = "1"
        os.environ["ATB_LLM_LOGITS_SAVE_FOLDER"] = str(logits_dir)
        atb_env = getattr(generate_module, "ENV", None)
        update_env = getattr(atb_env, "update", None)
        if callable(update_env):
            update_env()
        if atb_env is not None:
            atb_env.logits_save_enable = True
            atb_env.logits_save_folder = str(logits_dir)
    if rank == 0:
        print(f"[atb-eval] logits_capture={capture_mode}", flush=True)
    indexed = list(enumerate(zip(prompts, token_lengths)))
    if sort_by_length:
        indexed.sort(
            key=lambda item: item[1][1],
            reverse=sort_descending,
        )

    scores: list[float | None] = [None] * len(prompts)
    batch_events: list[dict[str, Any]] = []
    if capture is not None:
        generate_module.next_token_chooser = capture
    total_start = time.perf_counter()
    try:
        for batch_start in range(0, len(indexed), batch_size):
            batch = indexed[batch_start : batch_start + batch_size]
            batch_prompts = [item[1][0] for item in batch]
            if capture is not None:
                captured_before = len(capture.scores)
            else:
                for path in logits_dir.iterdir():
                    if path.is_file():
                        path.unlink()
            wall_start = time.perf_counter()
            _texts, _token_nums, atb_seconds = pa_runner.infer(
                batch_prompts,
                len(batch_prompts),
                1,
                False,
                None,
            )
            wall_seconds = time.perf_counter() - wall_start
            batch_scores = (
                capture.scores[captured_before:]
                if capture is not None
                else load_saved_logit_scores(
                    logits_dir,
                    yes_token_id=yes_token_id,
                    no_token_id=no_token_id,
                )
            )
            if len(batch_scores) != len(batch):
                raise RuntimeError(
                    "ATB first-token score count mismatch for batch "
                    f"{batch_start // batch_size}: "
                    f"{len(batch_scores)} != {len(batch)}"
                )
            for (original_index, _item), score in zip(batch, batch_scores):
                scores[original_index] = score
            event = {
                "batch_index": batch_start // batch_size,
                "batch_size": len(batch),
                "max_tokens": max(item[1][1] for item in batch),
                "atb_seconds": float(atb_seconds),
                "wall_seconds": float(wall_seconds),
            }
            batch_events.append(event)
            if rank == 0:
                completed = min(batch_start + len(batch), len(indexed))
                print(
                    "[atb-eval] scored=%d/%d batch=%d tokens_max=%d "
                    "atb=%.4fs wall=%.4fs"
                    % (
                        completed,
                        len(indexed),
                        len(batch),
                        event["max_tokens"],
                        event["atb_seconds"],
                        event["wall_seconds"],
                    ),
                    flush=True,
                )
    finally:
        if capture is not None:
            generate_module.next_token_chooser = original_chooser
    total_seconds = time.perf_counter() - total_start

    missing = [index for index, score in enumerate(scores) if score is None]
    if missing:
        raise RuntimeError(f"missing ATB scores for indices: {missing[:10]}")
    return [float(score) for score in scores], total_seconds, batch_events


def main() -> None:
    args = parse_args()
    if args.max_length < 64:
        raise ValueError("--max-length must be at least 64")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.max_queries < 0:
        raise ValueError("--max-queries must be non-negative")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
    logits_dir = Path(f"/tmp/memranker-atb-logits-rank{local_rank}")

    runtime_model, temporary_model = prepare_flat_model(
        args.model_root,
        local_rank,
        local_world_size,
        quantize="w8a8sc",
    )
    if temporary_model is not None:
        atexit.register(
            shutil.rmtree,
            temporary_model,
            ignore_errors=True,
        )
    print(
        f"[atb-eval] rank={rank}/{world_size} model={runtime_model}",
        flush=True,
    )

    from examples.run_pa import PARunner
    from examples.server import generate as generate_module

    runner_kwargs = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "model_path": str(runtime_model),
        "input_texts": [""],
        "max_input_length": args.max_length,
        "max_output_length": 1,
        "max_prefill_tokens": args.batch_size * args.max_length,
        "max_batch_size": args.batch_size,
        "block_size": 128,
        "load_tokenizer": True,
        "trust_remote_code": True,
    }
    pa_runner = PARunner(**runner_kwargs)
    tokenizer = pa_runner.tokenizer
    yes_token_id = single_token_id(tokenizer, "yes")
    no_token_id = single_token_id(tokenizer, "no")
    if rank == 0:
        print(
            f"[atb-eval] token_ids yes={yes_token_id} no={no_token_id}",
            flush=True,
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
    ground_truth = limit_to_matching_queries(
        ground_truth,
        recall_results,
        args.max_queries,
    )
    _input_texts, mapping, skipped_queries = build_scoring_inputs(
        recall_results,
        ground_truth,
        instruction=args.instruction,
    )
    if not mapping:
        raise ValueError("no matching query-document pairs to evaluate")
    prompts, token_lengths, truncated_count = prepare_prompts(
        tokenizer,
        mapping,
        instruction=args.instruction,
        max_length=args.max_length,
    )
    if rank == 0:
        print(
            "[atb-eval] queries=%d pairs=%d truncated=%d token_max=%d"
            % (
                len(ground_truth),
                len(prompts),
                truncated_count,
                max(token_lengths),
            ),
            flush=True,
        )

    pa_runner.warm_up()
    scores, score_time, batch_events = score_prompts(
        pa_runner=pa_runner,
        generate_module=generate_module,
        prompts=prompts,
        token_lengths=token_lengths,
        batch_size=args.batch_size,
        sort_by_length=args.sort_by_length,
        sort_descending=args.sort_descending,
        yes_token_id=yes_token_id,
        no_token_id=no_token_id,
        rank=rank,
        logits_dir=logits_dir,
    )

    if rank != 0:
        return
    seconds_per_example = score_time / max(1, len(scores))
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
    metrics.update(
        {
            "backend": "pure_atb",
            "scoring_backend": "generate_yes_no_logits",
            "quantization": "w8a8sc",
            "model_path": str(args.model_root),
            "gt_file": args.gt_file,
            "recall_file": args.recall_file,
            "gt_query_col": args.gt_query_col,
            "gt_doc_id_col": args.gt_doc_id_col,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "max_queries": args.max_queries,
            "num_scored_pairs": len(scores),
            "num_scored_queries": len(ground_truth),
            "truncated_pairs": truncated_count,
            "yes_token_id": yes_token_id,
            "no_token_id": no_token_id,
            "score_time_seconds": float(score_time),
            "seconds_per_example": float(seconds_per_example),
            "examples_per_second": (
                float(len(scores) / score_time) if score_time > 0 else 0.0
            ),
            "sort_by_length": args.sort_by_length,
            "sort_descending": args.sort_descending,
            "skipped_recall_queries_without_gt": skipped_queries,
            "batch_count": len(batch_events),
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_query_metrics.jsonl", per_query)
    write_jsonl(output_dir / "predictions.jsonl", ranked_predictions)
    write_jsonl(output_dir / "batch_latency.jsonl", batch_events)
    write_summary_csv(output_dir / "business_eval.csv", per_query)
    wrote_xlsx = write_summary_xlsx(
        output_dir / "business_eval.xlsx",
        per_query,
    )
    metrics["summary_csv"] = str(output_dir / "business_eval.csv")
    metrics["summary_xlsx"] = (
        str(output_dir / "business_eval.xlsx") if wrote_xlsx else ""
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
