#!/usr/bin/env python3
"""Use pure ATB as the scorer for the existing business evaluator."""

from __future__ import annotations

import atexit
import logging
import math
import os
import shutil
import sys
import tempfile
import time

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for import_path in (PROJECT_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import evaluate_business  # noqa: E402
from modeling import resolve_yes_no_token_ids  # noqa: E402
from scripts.prepare_qwen3_reranker_calibration import (  # noqa: E402
    truncate_prompt_document,
)
from scripts.run_atb_sharded_model import prepare_flat_model  # noqa: E402


INPUT_PREFIX = "<Instruct>: "
QUERY_SEPARATOR = "\n<Query>: "
DOCUMENT_SEPARATOR = "\n<Document>: "


@dataclass(frozen=True)
class AtbPrompt:
    prompt: str
    input_ids: tuple[int, ...]
    token_length: int
    original_token_length: int
    truncated: bool


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

    files = sorted(logits_dir.glob("*.pth"))
    if not files:
        raise RuntimeError(f"ATB produced no saved logits under {logits_dir}")
    scores: list[float] = []
    for path in files:
        logits = torch.load(path, map_location="cpu")
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"ATB logits file is not a tensor: {path}")
        scores.extend(
            yes_no_probabilities(
                logits,
                yes_token_id=yes_token_id,
                no_token_id=no_token_id,
            )
        )
    return scores


def format_atb_prompts(
    tokenizer: Any,
    input_texts: list[str],
    max_length: int,
) -> list[AtbPrompt]:
    """Rebuild the exact generate prompt used for calibration and vLLM.

    ``evaluate_business`` passes the scorer its historical single-newline
    intermediate format.  ATB must not wrap that string directly: W8A8SC was
    calibrated with the official Qwen3 generate prompt, which uses double
    newlines, and only the document may be truncated so the answer suffix is
    always retained.
    """
    prompts: list[AtbPrompt] = []
    for text in input_texts:
        instruction, query, document = parse_business_input(text)
        prompt, token_length, original_length, truncated = truncate_prompt_document(
            tokenizer=tokenizer,
            instruction=instruction,
            query=query,
            document=document,
            backend="generate",
            max_length=max_length,
        )
        prompts.append(
            AtbPrompt(
                prompt=prompt,
                input_ids=tuple(
                    tokenizer.encode(prompt, add_special_tokens=True)
                ),
                token_length=token_length,
                original_token_length=original_length,
                truncated=truncated,
            )
        )
    return prompts


def parse_business_input(text: str) -> tuple[str, str, str]:
    text = text.strip()
    if not text.startswith(INPUT_PREFIX):
        raise ValueError("business scorer input is missing '<Instruct>: '")
    instruction, query_marker, remainder = text[len(INPUT_PREFIX) :].partition(
        QUERY_SEPARATOR
    )
    if not query_marker:
        raise ValueError("business scorer input is missing '\\n<Query>: '")
    query, document_marker, document = remainder.partition(DOCUMENT_SEPARATOR)
    if not document_marker:
        raise ValueError("business scorer input is missing '\\n<Document>: '")
    return instruction.strip(), query.strip(), document.strip()


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


class AtbCausalLMScorer:
    """Match the existing business scorer's predict() interface with ATB."""

    def __init__(self, model_path: str, max_length: int) -> None:
        self.evaluation_metadata = {
            "backend": "pure_atb",
            "scoring_backend": "generate_yes_no_logits",
            "quantization": "w8a8sc",
        }
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", str(world_size)))
        if world_size != 1:
            raise ValueError(
                "the business ATB scorer currently requires TP_SIZE=1; "
                f"found WORLD_SIZE={world_size}"
            )

        self.rank = rank
        self.max_length = max_length
        self.configured_batch_size = int(
            os.environ.get("ATB_EVAL_BATCH_SIZE", "1")
        )
        self.sort_by_length = (
            os.environ.get("ATB_EVAL_SORT_BY_LENGTH", "1") == "1"
        )
        self.progress_every = max(
            0,
            int(os.environ.get("ATB_EVAL_PROGRESS_EVERY", "0")),
        )
        runtime_model, temporary_model = prepare_flat_model(
            Path(model_path),
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

        from examples.run_pa import PARunner
        from examples.server import generate as generate_module

        # PARunner is an example CLI runner rather than a serving scheduler.
        # Its infer() logs begin/end for every micro-batch.  At thousands of
        # batches that terminal I/O is both noisy and measurable, so keep the
        # default at WARNING while allowing explicit overrides for diagnosis.
        atb_log_level = os.environ.get("ATB_EVAL_LOG_LEVEL", "WARNING").upper()
        try:
            from atb_llm.utils.log import logger as atb_logger

            atb_logger.setLevel(getattr(logging, atb_log_level))
        except (ImportError, AttributeError):
            pass
        self.evaluation_metadata.update(
            {
                "atb_input_mode": "pretokenized_ids",
                "atb_log_level": atb_log_level,
                "atb_sort_by_length": self.sort_by_length,
            }
        )

        self.generate_module = generate_module
        self.runner = PARunner(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            model_path=str(runtime_model),
            max_input_length=max_length,
            max_output_length=1,
            max_prefill_tokens=self.configured_batch_size * max_length,
            max_batch_size=self.configured_batch_size,
            block_size=128,
            load_tokenizer=True,
            trust_remote_code=True,
        )
        self.tokenizer = self.runner.tokenizer
        self.yes_token_id, self.no_token_id = resolve_yes_no_token_ids(
            self.tokenizer
        )
        self.logits_dir = Path(
            tempfile.mkdtemp(prefix=f"memranker-atb-logits-rank{local_rank}-")
        )
        atexit.register(shutil.rmtree, self.logits_dir, ignore_errors=True)
        self.runner.warm_up()

    def _score_batch(self, input_ids: list[list[int]]) -> tuple[list[float], float]:
        original_chooser = getattr(
            self.generate_module,
            "next_token_chooser",
            None,
        )
        capture = (
            YesNoLogitCapture(
                original_chooser,
                self.yes_token_id,
                self.no_token_id,
            )
            if original_chooser is not None
            else None
        )
        if capture is not None:
            self.generate_module.next_token_chooser = capture
        else:
            for path in self.logits_dir.iterdir():
                if path.is_file():
                    path.unlink()
            os.environ["ATB_LLM_LOGITS_SAVE_ENABLE"] = "1"
            os.environ["ATB_LLM_LOGITS_SAVE_FOLDER"] = str(self.logits_dir)
            atb_env = getattr(self.generate_module, "ENV", None)
            update_env = getattr(atb_env, "update", None)
            if callable(update_env):
                update_env()
            if atb_env is not None:
                atb_env.logits_save_enable = True
                atb_env.logits_save_folder = str(self.logits_dir)

        try:
            _texts, _token_nums, atb_seconds = self.runner.infer(
                [],
                len(input_ids),
                1,
                False,
                input_ids,
            )
        finally:
            if capture is not None:
                self.generate_module.next_token_chooser = original_chooser

        scores = (
            capture.scores
            if capture is not None
            else load_saved_logit_scores(
                self.logits_dir,
                self.yes_token_id,
                self.no_token_id,
            )
        )
        if len(scores) != len(input_ids):
            raise RuntimeError(
                f"ATB score count mismatch: {len(scores)} != {len(input_ids)}"
            )
        return scores, float(atb_seconds)

    def predict(
        self,
        input_texts: list[str],
        batch_size: int = 1,
    ) -> list[float]:
        if batch_size != self.configured_batch_size:
            raise ValueError(
                "ATB scorer batch size must match container configuration: "
                f"{batch_size} != {self.configured_batch_size}"
            )
        formatted_prompts = format_atb_prompts(
            self.tokenizer,
            input_texts,
            self.max_length,
        )
        token_lengths = [item.token_length for item in formatted_prompts]
        original_lengths = [item.original_token_length for item in formatted_prompts]
        truncated_count = sum(item.truncated for item in formatted_prompts)
        self.evaluation_metadata.update(
            {
                "prompt_format": "qwen3_generate_calibration_parity",
                "prompt_token_p50": percentile(token_lengths, 0.50),
                "prompt_token_p90": percentile(token_lengths, 0.90),
                "prompt_token_max": max(token_lengths, default=0),
                "original_prompt_token_p50": percentile(original_lengths, 0.50),
                "original_prompt_token_p90": percentile(original_lengths, 0.90),
                "original_prompt_token_max": max(original_lengths, default=0),
                "num_truncated_pairs": truncated_count,
                "truncated_pair_ratio": (
                    truncated_count / len(formatted_prompts)
                    if formatted_prompts
                    else 0.0
                ),
            }
        )

        indexed = list(enumerate(formatted_prompts))
        if self.sort_by_length:
            indexed.sort(key=lambda item: item[1].token_length)

        scores: list[float | None] = [None] * len(indexed)
        predict_started = time.perf_counter()
        total_atb_seconds = 0.0
        num_batches = 0
        for start in range(0, len(indexed), batch_size):
            batch = indexed[start : start + batch_size]
            batch_scores, atb_seconds = self._score_batch(
                [list(item[1].input_ids) for item in batch]
            )
            total_atb_seconds += atb_seconds
            num_batches += 1
            for (original_index, _prompt), score in zip(batch, batch_scores):
                scores[original_index] = score
            scored = min(start + len(batch), len(indexed))
            if self.rank == 0 and self.progress_every > 0 and (
                num_batches == 1
                or num_batches % self.progress_every == 0
                or scored == len(indexed)
            ):
                print(
                    "[atb-eval] scored=%d/%d batch=%d atb=%.4fs"
                    % (
                        scored,
                        len(indexed),
                        len(batch),
                        atb_seconds,
                    ),
                    flush=True,
                )
        if any(score is None for score in scores):
            raise RuntimeError("ATB scoring left one or more inputs without a score")
        predict_seconds = time.perf_counter() - predict_started
        example_count = len(indexed)
        self.evaluation_metadata.update(
            {
                "num_atb_batches": num_batches,
                "atb_infer_time_seconds": total_atb_seconds,
                "atb_seconds_per_example": (
                    total_atb_seconds / example_count if example_count else 0.0
                ),
                "atb_examples_per_second": (
                    example_count / total_atb_seconds
                    if total_atb_seconds > 0
                    else 0.0
                ),
                "atb_adapter_time_seconds": predict_seconds,
                "atb_adapter_overhead_seconds": max(
                    0.0,
                    predict_seconds - total_atb_seconds,
                ),
            }
        )
        return [float(score) for score in scores if score is not None]


def load_atb_scorer(
    model_path: str,
    max_length: int = 4096,
    **_kwargs: Any,
) -> AtbCausalLMScorer:
    return AtbCausalLMScorer(model_path=model_path, max_length=max_length)


def main() -> None:
    evaluate_business.load_scorer = load_atb_scorer
    evaluate_business.main()


if __name__ == "__main__":
    main()
