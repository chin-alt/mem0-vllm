from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from data import RerankerExample, load_dataset_splits, load_examples
from modernbert_utils import DEFAULT_MODERNBERT_MODEL_NAME, evaluate_modernbert_examples, format_instruction_query, save_json, torch


logger = logging.getLogger(__name__)


def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    dest = name.replace("-", "_")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=dest, action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=f"Disable {name.replace('_', ' ')}")
    parser.set_defaults(**{dest: default})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ModernBERT pointwise BCE soft-label reranker baseline using sentence-transformers CrossEncoderTrainer."
    )
    parser.add_argument("--train_file", required=True, help="JSON/JSONL data file, or train split if dev/test files are set.")
    parser.add_argument("--dev_file", default=None)
    parser.add_argument("--test_file", default=None)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_name_or_path", default=DEFAULT_MODERNBERT_MODEL_NAME)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--eval_ratio", type=float, default=0.1)
    parser.add_argument("--test_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--default_instruction", default="")
    add_bool_arg(parser, "gradient_checkpointing", default=True, help_text="Enable gradient checkpointing")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--attn_implementation",
        default=None,
        help="Optional transformers attention backend, for example sdpa, eager, or flash_attention_2.",
    )
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--pos_weight", type=float, default=None, help="Optional BCE positive weight. Leave unset for soft labels.")
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action="store_true",
        help="Pass DDP find_unused_parameters=True when supported by TrainingArguments.",
    )
    parser.add_argument(
        "--report_to",
        default="none",
        help="Trainer reporting target, for example none, tensorboard, wandb.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def get_rank() -> int:
    for name in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(name)
        if value not in (None, ""):
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def is_main_process() -> bool:
    return get_rank() in (-1, 0)


def destroy_distributed_if_needed() -> None:
    if torch is None:
        return
    if hasattr(torch, "distributed") and torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def torch_dtype_from_flags(args: argparse.Namespace) -> Any | None:
    if torch is None:
        return None
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return None


def load_train_dev_test(args: argparse.Namespace) -> tuple[dict[str, list[RerankerExample]], dict[str, Any]]:
    if args.dev_file or args.test_file:
        train = load_examples(args.train_file, default_instruction=args.default_instruction)
        dev = load_examples(args.dev_file, default_instruction=args.default_instruction) if args.dev_file else []
        test = load_examples(args.test_file, default_instruction=args.default_instruction) if args.test_file else []
        split_info = {
            "strategy": "explicit_files",
            "splits": {
                "train": {"num_examples": len(train), "num_groups": len({ex.group_key for ex in train})},
                "dev": {"num_examples": len(dev), "num_groups": len({ex.group_key for ex in dev})},
                "test": {"num_examples": len(test), "num_groups": len({ex.group_key for ex in test})},
            },
        }
        return {"train": train, "dev": dev, "test": test}, split_info
    return load_dataset_splits(
        args.train_file,
        eval_ratio=args.eval_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_file=args.split_file,
        default_instruction=args.default_instruction,
    )


def examples_to_hf_dataset(examples: list[RerankerExample]) -> Any:
    try:
        from datasets import Dataset
    except ImportError as exc:
        raise RuntimeError("datasets is required for CrossEncoderTrainer. Install requirements.txt first.") from exc

    rows = [
        {
            "query": format_instruction_query(ex.instruction, ex.query),
            "document": ex.doc,
            "label": float(ex.label),
        }
        for ex in examples
    ]
    return Dataset.from_list(rows)


def filter_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return {key: value for key, value in kwargs.items() if value is not None}
    return {key: value for key, value in kwargs.items() if key in signature.parameters and value is not None}


def import_cross_encoder_stack() -> tuple[Any, Any, Any, Any]:
    try:
        from sentence_transformers.cross_encoder import CrossEncoder
    except ImportError as exc:
        raise RuntimeError("sentence-transformers is required for ModernBERT CrossEncoder training.") from exc

    try:
        from sentence_transformers.cross_encoder import CrossEncoderTrainer, CrossEncoderTrainingArguments
    except ImportError:
        from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
        from sentence_transformers.cross_encoder.training_args import CrossEncoderTrainingArguments

    try:
        from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
    except ImportError:
        from sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss import BinaryCrossEntropyLoss

    return CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments, BinaryCrossEntropyLoss


def build_cross_encoder(args: argparse.Namespace, CrossEncoder: Any) -> Any:
    model_load_kwargs: dict[str, Any] = {
        "torch_dtype": torch_dtype_from_flags(args),
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.attn_implementation:
        model_load_kwargs["attn_implementation"] = args.attn_implementation

    tokenizer_load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    base_kwargs = {
        "num_labels": 1,
        "max_length": args.max_length,
        "max_seq_length": args.max_length,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    candidate_kwargs = [
        {**base_kwargs, "automodel_args": model_load_kwargs, "tokenizer_args": tokenizer_load_kwargs},
        {**base_kwargs, "model_kwargs": model_load_kwargs, "tokenizer_kwargs": tokenizer_load_kwargs},
        base_kwargs,
    ]
    last_type_error: TypeError | None = None
    for constructor_kwargs in candidate_kwargs:
        filtered_kwargs = filter_kwargs(CrossEncoder.__init__, constructor_kwargs)
        try:
            model = CrossEncoder(args.model_name_or_path, **filtered_kwargs)
            break
        except TypeError as exc:
            last_type_error = exc
    else:
        raise last_type_error or RuntimeError("Failed to initialize CrossEncoder")

    underlying = get_underlying_model(model)
    if args.gradient_checkpointing and hasattr(underlying, "gradient_checkpointing_enable"):
        underlying.gradient_checkpointing_enable()
        if hasattr(underlying.config, "use_cache"):
            underlying.config.use_cache = False
    return model


def build_training_args(args: argparse.Namespace, CrossEncoderTrainingArguments: Any, has_eval: bool) -> Any:
    output_dir = Path(args.output_dir)
    training_kwargs = {
        "output_dir": str(output_dir / "trainer"),
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.lr,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "fp16": args.fp16,
        "bf16": args.bf16,
        "dataloader_num_workers": args.dataloader_num_workers,
        "logging_steps": args.logging_steps,
        "logging_first_step": True,
        "save_strategy": "epoch",
        "save_total_limit": args.save_total_limit,
        "seed": args.seed,
        "disable_tqdm": args.disable_tqdm,
        "report_to": [] if args.report_to == "none" else [args.report_to],
        "ddp_find_unused_parameters": bool(args.ddp_find_unused_parameters),
    }
    if has_eval:
        training_kwargs.update(
            {
                "eval_strategy": "epoch",
                "evaluation_strategy": "epoch",
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    else:
        training_kwargs.update({"eval_strategy": "no", "evaluation_strategy": "no", "load_best_model_at_end": False})

    filtered_kwargs = filter_kwargs(CrossEncoderTrainingArguments.__init__, training_kwargs)
    return CrossEncoderTrainingArguments(**filtered_kwargs)


def build_bce_loss(args: argparse.Namespace, model: Any, BinaryCrossEntropyLoss: Any) -> Any:
    loss_kwargs: dict[str, Any] = {"model": model}
    if args.pos_weight is not None:
        if torch is None:
            raise RuntimeError("torch is required for pos_weight")
        loss_kwargs["pos_weight"] = torch.tensor(float(args.pos_weight))
    return BinaryCrossEntropyLoss(**filter_kwargs(BinaryCrossEntropyLoss.__init__, loss_kwargs))


def get_underlying_model(cross_encoder: Any) -> Any:
    return getattr(cross_encoder, "model", cross_encoder)


def get_tokenizer(cross_encoder: Any) -> Any:
    return getattr(cross_encoder, "tokenizer", None)


def count_parameters(model: Any) -> dict[str, int | float]:
    total = 0
    trainable = 0
    for param in model.parameters():
        numel = int(param.numel())
        total += numel
        if param.requires_grad:
            trainable += numel
    ratio = trainable / total if total else 0.0
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_ratio": ratio,
    }


def model_device(model: Any) -> str:
    try:
        return str(next(model.parameters()).device)
    except StopIteration:
        return "cuda" if torch is not None and torch.cuda.is_available() else "cpu"


def save_metadata(args: argparse.Namespace, split_info: dict[str, Any], parameter_counts: dict[str, Any]) -> None:
    if not is_main_process():
        return
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "training_args.json", vars(args))
    save_json(out / "split_info.json", split_info)
    save_json(out / "parameter_counts.json", parameter_counts)


def save_cross_encoder_checkpoint(
    cross_encoder: Any,
    args: argparse.Namespace,
    metrics: dict[str, Any],
    parameter_counts: dict[str, Any],
) -> None:
    output_dir = Path(args.output_dir)
    best_dir = output_dir / "best"
    final_dir = output_dir / "final"
    for path in (best_dir, final_dir):
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        cross_encoder.save_pretrained(str(path))
        save_json(
            path / "modernbert_reranker_config.json",
            {
                "base_model_name_or_path": args.model_name_or_path,
                "trainer": "sentence_transformers.CrossEncoderTrainer",
                "model": "sentence_transformers.CrossEncoder",
                "score": "sigmoid(sequence_classification_logit)",
                "loss": "sentence_transformers.cross_encoder.losses.BinaryCrossEntropyLoss",
                "input_format": "pair(text_a=instruction + query, text_b=document)",
                "max_length": args.max_length,
                "label_normalization": "labels / 10 clipped to [0, 1]",
                "parameter_counts": parameter_counts,
            },
        )
    save_json(output_dir / "best_metrics.json", metrics)
    save_json(output_dir / "final_metrics.json", metrics)
    logger.info("Saved CrossEncoder checkpoint to %s and %s", best_dir, final_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.bf16 and args.fp16:
        raise ValueError("--bf16 and --fp16 are mutually exclusive")
    if torch is None:
        raise RuntimeError("torch is required for ModernBERT training")

    set_seed(args.seed)
    splits, split_info = load_train_dev_test(args)
    if not splits["train"]:
        raise ValueError("Train split is empty.")
    if is_main_process():
        logger.info(
            "Split sizes: train=%d dev=%d test=%d",
            len(splits["train"]),
            len(splits["dev"]),
            len(splits["test"]),
        )

    CrossEncoder, CrossEncoderTrainer, CrossEncoderTrainingArguments, BinaryCrossEntropyLoss = import_cross_encoder_stack()
    cross_encoder = build_cross_encoder(args, CrossEncoder)
    underlying_model = get_underlying_model(cross_encoder)
    tokenizer = get_tokenizer(cross_encoder)
    if tokenizer is None:
        raise RuntimeError("CrossEncoder did not expose a tokenizer; cannot run project evaluation.")

    parameter_counts = count_parameters(underlying_model)
    if is_main_process():
        logger.info(
            "ModernBERT parameter counts: total=%d trainable=%d frozen=%d trainable_ratio=%.6f",
            parameter_counts["total_parameters"],
            parameter_counts["trainable_parameters"],
            parameter_counts["frozen_parameters"],
            parameter_counts["trainable_ratio"],
        )
    save_metadata(args, split_info, parameter_counts)

    train_dataset = examples_to_hf_dataset(splits["train"])
    eval_examples = splits["dev"] or splits["test"] or []
    eval_dataset = examples_to_hf_dataset(eval_examples) if eval_examples else None
    training_args = build_training_args(args, CrossEncoderTrainingArguments, has_eval=eval_dataset is not None)
    loss = build_bce_loss(args, cross_encoder, BinaryCrossEntropyLoss)

    trainer_kwargs = {
        "model": cross_encoder,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "loss": loss,
    }
    trainer = CrossEncoderTrainer(**filter_kwargs(CrossEncoderTrainer.__init__, trainer_kwargs))
    trainer.train()

    if not is_main_process():
        logger.info("Rank %d finished training; skipping final evaluation and checkpoint export.", get_rank())
        destroy_distributed_if_needed()
        return

    metrics: dict[str, Any] = {}
    eval_for_metrics = eval_examples or splits["train"]
    if eval_for_metrics:
        overall, _per_query, _predictions = evaluate_modernbert_examples(
            underlying_model,
            tokenizer,
            eval_for_metrics,
            max_length=args.max_length,
            batch_size=args.eval_batch_size,
            relevance_threshold=args.relevance_threshold,
            device=model_device(underlying_model),
        )
        metrics.update(overall)
    metrics.update(
        {
            "trainer": "sentence_transformers.CrossEncoderTrainer",
            "loss": "BinaryCrossEntropyLoss",
            "base_model_name_or_path": args.model_name_or_path,
            "eval_split": "dev" if splits["dev"] else ("test" if splits["test"] else "train"),
            **parameter_counts,
        }
    )
    save_cross_encoder_checkpoint(cross_encoder, args, metrics, parameter_counts)
    logger.info("Training complete. Metrics: %s", json.dumps(metrics, ensure_ascii=False))
    destroy_distributed_if_needed()


if __name__ == "__main__":
    main()
