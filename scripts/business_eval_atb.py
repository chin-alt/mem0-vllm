#!/usr/bin/env python3
"""Use pure ATB as the scorer for the existing business evaluator."""

from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile

from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
for import_path in (PROJECT_ROOT, SRC_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

import evaluate_business  # noqa: E402
from modeling import (  # noqa: E402
    RERANKER_PREFIX,
    RERANKER_SUFFIX,
    resolve_yes_no_token_ids,
)
from scripts.run_atb_sharded_model import prepare_flat_model  # noqa: E402


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
) -> list[tuple[str, int]]:
    prefix_ids = list(tokenizer.encode(RERANKER_PREFIX, add_special_tokens=False))
    suffix_ids = list(tokenizer.encode(RERANKER_SUFFIX, add_special_tokens=False))
    pair_max_length = max(1, max_length - len(prefix_ids) - len(suffix_ids))
    prompts: list[tuple[str, int]] = []
    for text in input_texts:
        pair_ids = list(tokenizer.encode(text.strip(), add_special_tokens=False))
        prompt_ids = prefix_ids + pair_ids[:pair_max_length] + suffix_ids
        prompt = tokenizer.decode(
            prompt_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        prompts.append((prompt, len(prompt_ids)))
    return prompts


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

    def _score_batch(self, prompts: list[str]) -> tuple[list[float], float]:
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
                prompts,
                len(prompts),
                1,
                False,
                None,
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
        if len(scores) != len(prompts):
            raise RuntimeError(
                f"ATB score count mismatch: {len(scores)} != {len(prompts)}"
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
        indexed = list(enumerate(format_atb_prompts(
            self.tokenizer,
            input_texts,
            self.max_length,
        )))
        if self.sort_by_length:
            indexed.sort(key=lambda item: item[1][1])

        scores: list[float | None] = [None] * len(indexed)
        for start in range(0, len(indexed), batch_size):
            batch = indexed[start : start + batch_size]
            batch_scores, atb_seconds = self._score_batch(
                [item[1][0] for item in batch]
            )
            for (original_index, _prompt), score in zip(batch, batch_scores):
                scores[original_index] = score
            if self.rank == 0:
                print(
                    "[atb-eval] scored=%d/%d batch=%d atb=%.4fs"
                    % (
                        min(start + len(batch), len(indexed)),
                        len(indexed),
                        len(batch),
                        atb_seconds,
                    ),
                    flush=True,
                )
        if any(score is None for score in scores):
            raise RuntimeError("ATB scoring left one or more inputs without a score")
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
