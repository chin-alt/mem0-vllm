from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import shutil

from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from data import RerankerExample
from metrics import compute_all_metrics, is_better_metric
from modeling import (
    DEFAULT_MODEL_NAME,
    load_causal_training_model,
    model_load_help,
    normalize_model_name_or_path,
    prepare_qwen3_reranker_inputs,
    predict_causal_model,
    save_reranker_config,
    torch,
)
from train_pointwise import (
    add_bool_arg,
    create_accelerator,
    load_train_dev_test,
    make_optimizer,
    make_scheduler,
    prepare_output_dir,
    save_json,
    set_seed,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listwise soft-label distillation for MemReranker-style Qwen3 reranking."
    )
    parser.add_argument("--train_file", required=True, help="JSON/JSONL data file, or train split if dev/test files are set.")
    parser.add_argument("--dev_file", default=None, help="Optional explicit dev JSON/JSONL split.")
    parser.add_argument("--test_file", default=None, help="Optional explicit test JSON/JSONL split.")
    parser.add_argument("--split_file", default=None, help="Optional JSON split file with train/dev/test group keys.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--per_device_train_group_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    add_bool_arg(parser, "use_lora", default=True, help_text="Use LoRA adapters")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--default_instruction", default="")
    add_bool_arg(parser, "gradient_checkpointing", default=True, help_text="Enable gradient checkpointing")
    parser.add_argument("--load_in_4bit", action="store_true", help="Enable QLoRA-style 4-bit loading.")
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional transformers attention backend, for example flash_attention_2 or eager.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--disable_tqdm", action="store_true", help="Disable tqdm progress bars.")
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Set DDP find_unused_parameters=True. Usually keep this off for LoRA.",
    )
    parser.add_argument(
        "--teacher_score_scale",
        choices=["normalized", "raw"],
        default="normalized",
        help="Use labels/10 or raw 0-10 labels before building the teacher listwise distribution.",
    )
    parser.add_argument("--teacher_temperature", type=float, default=1.0)
    parser.add_argument("--model_temperature", type=float, default=1.0)
    parser.add_argument("--loss_type", choices=["kl", "ce"], default="kl")
    parser.add_argument("--min_group_size", type=int, default=2)
    parser.add_argument(
        "--max_group_size",
        type=int,
        default=16,
        help="Maximum docs per query group during training. Use <=0 to keep all docs.",
    )
    parser.add_argument(
        "--group_truncation",
        choices=["input_order", "label_desc", "random"],
        default="input_order",
    )
    return parser.parse_args()


def group_examples(examples: list[RerankerExample], min_group_size: int = 1) -> list[list[RerankerExample]]:
    grouped: dict[str, list[RerankerExample]] = {}
    for ex in examples:
        grouped.setdefault(ex.group_key, []).append(ex)
    return [items for items in grouped.values() if len(items) >= min_group_size]


def truncate_group(
    group: list[RerankerExample],
    max_group_size: int,
    strategy: str,
    rng: random.Random,
) -> list[RerankerExample]:
    if max_group_size <= 0 or len(group) <= max_group_size:
        return list(group)
    if strategy == "label_desc":
        return sorted(group, key=lambda ex: ex.raw_label, reverse=True)[:max_group_size]
    if strategy == "random":
        return rng.sample(group, max_group_size)
    return list(group[:max_group_size])


class ListwiseGroupDataset:
    def __init__(
        self,
        groups: list[list[RerankerExample]],
        max_group_size: int,
        group_truncation: str,
        seed: int,
    ):
        self.groups = groups
        self.max_group_size = max_group_size
        self.group_truncation = group_truncation
        self.seed = seed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.groups)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __getitem__(self, index: int) -> list[RerankerExample]:
        rng = random.Random(self.seed + self.epoch * 1_000_003 + index)
        return truncate_group(
            self.groups[index],
            max_group_size=self.max_group_size,
            strategy=self.group_truncation,
            rng=rng,
        )


def example_teacher_score(ex: RerankerExample, score_scale: str) -> float:
    return float(ex.label if score_scale == "normalized" else ex.raw_label)


def compute_listwise_loss(
    logits: Any,
    teacher_scores: Any,
    group_sizes: list[int],
    teacher_temperature: float,
    model_temperature: float,
    loss_type: str,
) -> Any:
    """Compute listwise loss from raw yes-minus-no logits.

    Do not pass sigmoid probabilities here. The model distribution should be
    formed from unbounded logits with log_softmax.
    """
    if torch is None:
        raise RuntimeError("torch is required for listwise loss")
    if teacher_temperature <= 0 or model_temperature <= 0:
        raise ValueError("Temperatures must be > 0")

    losses = []
    offset = 0
    for group_size in group_sizes:
        group_logits = logits[offset : offset + group_size].float() / model_temperature
        group_teacher_scores = teacher_scores[offset : offset + group_size].float() / teacher_temperature
        offset += group_size

        teacher_probs = torch.softmax(group_teacher_scores, dim=0)
        log_probs = torch.log_softmax(group_logits, dim=0)
        if loss_type == "ce":
            loss = -(teacher_probs * log_probs).sum()
        else:
            loss = torch.nn.functional.kl_div(log_probs, teacher_probs, reduction="sum")
        losses.append(loss)

    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def logits_from_scores(scores: list[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(scores, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def softmax_numpy(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    exp_values = np.exp(shifted)
    return exp_values / max(exp_values.sum(), 1e-12)


def log_softmax_numpy(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max()
    return shifted - np.log(np.exp(shifted).sum())


def compute_listwise_loss_numpy(
    groups: list[list[RerankerExample]],
    scores_by_example: dict[int, float],
    args: argparse.Namespace,
) -> float:
    losses: list[float] = []
    for group in groups:
        valid = [ex for ex in group if ex.source_index in scores_by_example]
        if len(valid) < max(1, args.min_group_size):
            continue
        logits = logits_from_scores([scores_by_example[int(ex.source_index)] for ex in valid])
        teacher_scores = np.asarray(
            [example_teacher_score(ex, args.teacher_score_scale) for ex in valid],
            dtype=np.float64,
        )
        teacher_probs = softmax_numpy(teacher_scores / args.teacher_temperature)
        log_probs = log_softmax_numpy(logits / args.model_temperature)
        if args.loss_type == "ce":
            losses.append(float(-(teacher_probs * log_probs).sum()))
        else:
            losses.append(float((teacher_probs * (np.log(np.clip(teacher_probs, 1e-12, 1.0)) - log_probs)).sum()))
    return float(np.mean(losses)) if losses else 0.0


def examples_to_records(examples: list[RerankerExample], scores: list[float]) -> list[dict[str, Any]]:
    rows = []
    for ex, score in zip(examples, scores):
        rows.append(
            {
                "group_key": ex.group_key,
                "query": ex.query,
                "query_id": ex.query_id,
                "doc": ex.doc,
                "label": ex.label,
                "raw_label": ex.raw_label,
                "score": float(score),
                "reason": ex.reason,
            }
        )
    return rows


def evaluate_examples(
    examples: list[RerankerExample],
    predict_fn: Any,
    args: argparse.Namespace,
) -> dict[str, float]:
    input_texts = [ex.input_text for ex in examples]
    scores = predict_fn(input_texts)
    records = examples_to_records(examples, scores)
    overall, _ = compute_all_metrics(
        records,
        query_key="group_key",
        relevance_threshold=args.relevance_threshold,
    )
    score_lookup = {
        int(ex.source_index): float(score)
        for ex, score in zip(examples, scores)
        if ex.source_index is not None
    }
    eval_groups = group_examples(examples, min_group_size=args.min_group_size)
    overall["ListwiseLoss"] = compute_listwise_loss_numpy(eval_groups, score_lookup, args)
    return overall


def save_best_listwise(
    wrapper: Any,
    tokenizer: Any,
    args: argparse.Namespace,
    accelerator: Any,
    metrics: dict[str, float],
) -> None:
    best_dir = Path(args.output_dir) / "best"
    if best_dir.exists():
        shutil.rmtree(best_dir)
    best_dir.mkdir(parents=True, exist_ok=True)
    unwrapped = accelerator.unwrap_model(wrapper)
    unwrapped.model.save_pretrained(best_dir, save_function=accelerator.save)
    tokenizer.save_pretrained(best_dir)
    save_reranker_config(
        best_dir,
        {
            "base_model_name_or_path": args.model_name_or_path,
            "max_length": args.max_length,
            "prompt_format": "Qwen3-Reranker chat prefix + <Instruct>/<Query>/<Document> + assistant think suffix",
            "score": "softmax([logit_no, logit_yes])[yes]",
            "logit_equivalent": "sigmoid(logit_yes - logit_no)",
            "loss": f"listwise_{args.loss_type}",
            "listwise_teacher_distribution": (
                f"softmax({args.teacher_score_scale}_labels / {args.teacher_temperature}) within each query"
            ),
            "model_distribution": f"softmax(logit_yes_minus_no / {args.model_temperature}) within each query",
            "label_normalization": "labels / 10 clipped to [0, 1]",
            "use_lora": args.use_lora,
            "load_in_4bit": args.load_in_4bit,
            "attn_implementation": args.attn_implementation,
            "min_group_size": args.min_group_size,
            "max_group_size": args.max_group_size,
            "group_truncation": args.group_truncation,
            "distributed": {
                "num_processes": accelerator.num_processes,
                "mixed_precision": accelerator.mixed_precision,
            },
        },
    )
    save_json(Path(args.output_dir) / "best_metrics.json", metrics)
    logger.info("Saved new best checkpoint to %s", best_dir)


def _kbit_device_map_for_process(args: argparse.Namespace, accelerator: Any) -> Any | None:
    if not args.load_in_4bit or accelerator.num_processes <= 1:
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", accelerator.local_process_index))
    return {"": local_rank}


def train_listwise(
    args: argparse.Namespace,
    splits: dict[str, list[RerankerExample]],
    accelerator: Any,
) -> dict[str, float]:
    if torch is None:
        raise RuntimeError("torch is required for listwise training")
    from torch.utils.data import DataLoader

    if accelerator.is_main_process:
        logger.info("Using Qwen3 listwise soft-label training with model %s", args.model_name_or_path)
    try:
        wrapper, tokenizer = load_causal_training_model(
            args.model_name_or_path,
            bf16=args.bf16,
            fp16=args.fp16,
            use_lora=args.use_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            load_in_4bit=args.load_in_4bit,
            device_map=_kbit_device_map_for_process(args, accelerator),
            gradient_checkpointing=args.gradient_checkpointing,
            attn_implementation=args.attn_implementation,
        )
    except Exception as exc:
        raise RuntimeError(model_load_help(args.model_name_or_path, exc)) from exc

    train_groups = group_examples(splits["train"], min_group_size=args.min_group_size)
    eval_examples = splits["dev"] or splits["test"] or splits["train"]
    if not train_groups:
        raise ValueError(
            f"No training groups with at least min_group_size={args.min_group_size}. "
            "Listwise training requires multiple docs per query."
        )
    if accelerator.is_main_process:
        sizes = [len(group) for group in train_groups]
        logger.info(
            "Listwise train groups=%d avg_group_size=%.2f max_group_size=%d",
            len(train_groups),
            float(np.mean(sizes)),
            max(sizes),
        )

    dataset = ListwiseGroupDataset(
        train_groups,
        max_group_size=args.max_group_size,
        group_truncation=args.group_truncation,
        seed=args.seed,
    )

    def collate(batch_groups: list[list[RerankerExample]]) -> dict[str, Any]:
        input_texts: list[str] = []
        teacher_scores: list[float] = []
        group_sizes: list[int] = []
        for group in batch_groups:
            if len(group) < args.min_group_size:
                continue
            group_sizes.append(len(group))
            for ex in group:
                input_texts.append(ex.input_text)
                teacher_scores.append(example_teacher_score(ex, args.teacher_score_scale))
        if not input_texts:
            raise ValueError("Empty listwise batch after grouping/truncation")
        encoded = prepare_qwen3_reranker_inputs(
            tokenizer,
            input_texts,
            max_length=args.max_length,
        )
        encoded["teacher_scores"] = torch.tensor(teacher_scores, dtype=torch.float32)
        encoded["group_sizes"] = group_sizes
        return encoded

    loader = DataLoader(
        dataset,
        batch_size=args.per_device_train_group_batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    if len(loader) == 0:
        raise ValueError("Empty listwise training dataloader")

    optimizer = make_optimizer(wrapper, args)
    wrapper, optimizer, loader = accelerator.prepare(wrapper, optimizer, loader)
    scheduler = make_scheduler(optimizer, args, len(loader))
    best_metrics: dict[str, float] | None = None
    history_path = Path(args.output_dir) / "metrics_history.jsonl"

    for epoch in range(1, args.epochs + 1):
        dataset.set_epoch(epoch)
        wrapper.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        progress = tqdm(
            enumerate(loader, start=1),
            total=len(loader),
            desc=f"Listwise epoch {epoch}/{args.epochs}",
            unit="group_batch",
            dynamic_ncols=True,
            ascii=True,
            disable=args.disable_tqdm or not accelerator.is_main_process,
        )
        for step, batch in progress:
            teacher_scores = batch.pop("teacher_scores").to(accelerator.device)
            group_sizes = batch.pop("group_sizes")
            with accelerator.accumulate(wrapper):
                outputs = wrapper(**batch)
                loss = compute_listwise_loss(
                    outputs["logits"],
                    teacher_scores,
                    group_sizes=group_sizes,
                    teacher_temperature=args.teacher_temperature,
                    model_temperature=args.model_temperature,
                    loss_type=args.loss_type,
                )
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            running_loss += float(accelerator.gather(loss.detach()).mean().cpu())
            if accelerator.is_main_process and not args.disable_tqdm:
                progress.set_postfix(loss=f"{running_loss / step:.4f}")
            if accelerator.is_main_process and args.logging_steps > 0 and step % args.logging_steps == 0:
                logger.info("epoch=%d step=%d train_listwise_loss=%.6f", epoch, step, running_loss / step)

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            eval_wrapper = accelerator.unwrap_model(wrapper)
            predict_fn = lambda texts: predict_causal_model(
                eval_wrapper,
                tokenizer,
                texts,
                max_length=args.max_length,
                batch_size=args.eval_batch_size,
                device=str(accelerator.device),
            )
            metrics = evaluate_examples(eval_examples, predict_fn, args)
            metrics["epoch"] = float(epoch)
            metrics["TrainListwiseLoss"] = running_loss / max(1, len(loader))
            logger.info("epoch=%d dev_metrics=%s", epoch, json.dumps(metrics, ensure_ascii=False))
            with history_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")

            if is_better_metric(metrics, best_metrics):
                best_metrics = metrics
                save_best_listwise(wrapper, tokenizer, args, accelerator, metrics)
        accelerator.wait_for_everyone()

    return best_metrics or {}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    if args.epochs < 1:
        raise ValueError("--epochs must be >= 1")
    if args.teacher_temperature <= 0 or args.model_temperature <= 0:
        raise ValueError("Temperatures must be > 0")
    args.model_name_or_path = normalize_model_name_or_path(args.model_name_or_path)

    accelerator = create_accelerator(args)
    set_seed(args.seed)

    splits, split_info = load_train_dev_test(args)
    if not splits["train"]:
        raise ValueError("Train split is empty.")
    if accelerator.is_main_process:
        logger.info(
            "Split sizes: train=%d dev=%d test=%d",
            len(splits["train"]),
            len(splits["dev"]),
            len(splits["test"]),
        )
        logger.info(
            "Accelerate: num_processes=%d mixed_precision=%s device=%s",
            accelerator.num_processes,
            accelerator.mixed_precision,
            accelerator.device,
        )
    prepare_output_dir(args, split_info, accelerator)
    best_metrics = train_listwise(args, splits, accelerator)
    if accelerator.is_main_process:
        logger.info("Best metrics: %s", json.dumps(best_metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
