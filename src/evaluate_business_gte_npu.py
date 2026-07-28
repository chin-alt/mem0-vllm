from __future__ import annotations

import argparse
import json
import logging
import math
import time

from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from data import write_jsonl
from evaluate_business import (
    attach_scores_and_ranks,
    build_scoring_inputs,
    cleanup_accelerator_memory,
    compute_business_metrics,
    get_torch_accelerator_memory_mib,
    load_ground_truth,
    load_recall_results,
    reset_torch_accelerator_memory_stats,
    write_summary_csv,
    write_summary_xlsx,
)


logger = logging.getLogger(__name__)
DEFAULT_GTE_MODEL = "Alibaba-NLP/gte-multilingual-reranker-base"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a GTE sequence-classification reranker locally on an Ascend NPU."
    )
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--recall_file", required=True)
    parser.add_argument("--model_path", default=DEFAULT_GTE_MODEL)
    parser.add_argument("--output_dir", default="outputs/business_gte_310p")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    parser.add_argument(
        "--attention_backend",
        choices=["pfa", "eager", "sdpa"],
        default="pfa",
        help="pfa uses the 310P PromptFlashAttention operator; eager is the compatibility fallback.",
    )
    parser.add_argument("--jit_compile", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pad_to_multiple_of", type=int, default=8)
    parser.add_argument("--instruction", default="")
    parser.add_argument("--gt_query_col", default="query")
    parser.add_argument("--gt_doc_id_col", default="PageId")
    parser.add_argument("--gt_sheet", default=None)
    parser.add_argument("--recall_id_key", default="id")
    parser.add_argument("--recall_text_key", default="text")
    parser.add_argument("--top_k_list", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--expected_fbeta_beta", type=float, default=0.3)
    parser.add_argument(
        "--score_activation",
        choices=["sigmoid", "identity"],
        default="sigmoid",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--save_doc_text", action="store_true")
    return parser.parse_args()


def format_gte_query(query: str, instruction: str = "") -> str:
    query = str(query).strip()
    instruction = str(instruction).strip()
    return f"{instruction}\n\n{query}" if instruction else query


def sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values.astype(np.float64, copy=False), -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def load_npu_stack() -> tuple[Any, Any]:
    try:
        import torch
        import torch_npu  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "GTE Ascend inference requires matching torch and torch-npu wheels. "
            "Install requirements-ascend-gte-310p-py39.txt in an isolated environment."
        ) from exc
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("torch_npu imported, but torch.npu.is_available() is False")
    return torch, torch_npu


def resolve_dtype(torch: Any, name: str) -> Any:
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def npu_synchronize(torch: Any) -> None:
    synchronize = getattr(torch.npu, "synchronize", None)
    if synchronize is not None:
        synchronize()


def validate_attention_backend(attention_backend: str, dtype: str) -> None:
    if attention_backend == "pfa" and dtype != "fp16":
        raise ValueError("Ascend 310P PromptFlashAttention only supports --dtype fp16")


def _build_pfa_attention_mask(
    attention_bias: Any,
    query_length: int,
    torch: Any,
) -> Any:
    if attention_bias is None:
        return None
    if attention_bias.ndim != 4:
        raise ValueError(
            "GTE PFA expects a four-dimensional attention bias, got "
            f"shape={tuple(attention_bias.shape)}"
        )

    if attention_bias.dtype == torch.bool:
        mask = attention_bias
    else:
        # Hugging Face extended masks use zero for valid keys and a large
        # negative value for padding. PFA uses True/1 for masked positions.
        mask = attention_bias < 0

    if mask.shape[-2] == 1:
        mask = mask.expand(mask.shape[0], mask.shape[1], query_length, mask.shape[-1])
    elif mask.shape[-2] != query_length:
        raise ValueError(
            "GTE PFA attention mask query dimension does not match Q: "
            f"mask={tuple(mask.shape)} query_length={query_length}"
        )
    return mask.contiguous()


def install_gte_pfa_attention(model: Any, torch: Any, torch_npu: Any) -> int:
    pfa = getattr(torch_npu, "npu_prompt_flash_attention", None)
    if pfa is None:
        raise RuntimeError(
            "torch_npu.npu_prompt_flash_attention is unavailable. "
            "Use CANN 8.0.RC2 with torch 2.1.0 and torch-npu 2.1.0.post6."
        )

    mask_cache: dict[str, Any] = {
        "source": None,
        "query_length": None,
        "mask": None,
    }

    def pfa_attention(
        self: Any,
        query_states: Any,
        key_states: Any,
        value_states: Any,
        attention_bias: Any,
        head_mask: Any,
    ) -> tuple[Any, None]:
        if head_mask is not None:
            raise RuntimeError("GTE PFA does not support head_mask")
        if query_states.dtype != torch.float16:
            raise RuntimeError(
                f"Ascend 310P PFA requires fp16 Q/K/V, got {query_states.dtype}"
            )

        query = query_states.transpose(1, 2).contiguous()
        key = key_states.transpose(1, 2).contiguous()
        value = value_states.transpose(1, 2).contiguous()
        query_length = int(query.shape[2])

        if (
            mask_cache["source"] is not attention_bias
            or mask_cache["query_length"] != query_length
        ):
            mask_cache["source"] = attention_bias
            mask_cache["query_length"] = query_length
            mask_cache["mask"] = _build_pfa_attention_mask(
                attention_bias,
                query_length,
                torch,
            )

        output = pfa(
            query,
            key,
            value,
            atten_mask=mask_cache["mask"],
            num_heads=int(self.num_attention_heads),
            scale_value=1.0 / math.sqrt(int(self.attention_head_size)),
            pre_tokens=2147483647,
            next_tokens=2147483647,
            input_layout="BNSD",
            sparse_mode=0,
        )
        return output.transpose(1, 2).contiguous(), None

    patched = 0
    for module in model.modules():
        if not all(
            hasattr(module, name)
            for name in ("_attention", "num_attention_heads", "attention_head_size")
        ):
            continue
        module._attention = MethodType(pfa_attention, module)
        patched += 1
    if patched == 0:
        raise RuntimeError(
            "Could not find GTE attention modules to patch. "
            "Check that MODEL_PATH is gte-multilingual-reranker-base."
        )
    return patched


def load_model_and_tokenizer(
    args: argparse.Namespace,
    torch: Any,
    torch_npu: Any,
) -> tuple[Any, Any]:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    validate_attention_backend(args.attention_backend, args.dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
        torch_dtype=resolve_dtype(torch, args.dtype),
        # PFA replaces the eager implementation after loading. Instantiating
        # eager here prevents Transformers from selecting SDPA automatically.
        attn_implementation=(
            "eager" if args.attention_backend == "pfa" else args.attention_backend
        ),
    )
    if args.attention_backend == "pfa":
        patched = install_gte_pfa_attention(model, torch, torch_npu)
        logger.info("Installed Ascend 310P PFA backend on %d GTE layers", patched)
    model.eval()
    model.to(args.device)
    return model, tokenizer


def validate_rope_buffers(model: Any, torch: Any) -> None:
    damaged: list[str] = []
    checked = 0
    for name, value in model.named_buffers():
        if value.numel() == 0 or not any(key in name for key in ("inv_freq", "cos_cached", "sin_cached")):
            continue
        checked += 1
        if not bool(torch.isfinite(value).all().item()):
            damaged.append(name)
    if damaged:
        raise RuntimeError(
            "GTE RoPE buffers contain NaN/Inf: " + ", ".join(damaged[:8])
        )
    logger.info("GTE RoPE buffer health check passed (%d buffers checked)", checked)


def score_pairs(
    model: Any,
    tokenizer: Any,
    queries: list[str],
    docs: list[str],
    args: argparse.Namespace,
    torch: Any,
) -> list[float]:
    if len(queries) != len(docs):
        raise ValueError(f"queries/docs length mismatch: {len(queries)} != {len(docs)}")

    scores: list[float] = []
    with torch.inference_mode():
        for start in tqdm(
            range(0, len(queries), args.batch_size),
            total=math.ceil(len(queries) / args.batch_size) if queries else 0,
            desc="GTE NPU scoring",
            unit="batch",
            dynamic_ncols=True,
            ascii=True,
        ):
            batch_queries = [
                format_gte_query(value, args.instruction)
                for value in queries[start : start + args.batch_size]
            ]
            batch_docs = docs[start : start + args.batch_size]
            encoded = tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                pad_to_multiple_of=args.pad_to_multiple_of or None,
                return_tensors="pt",
            )
            encoded = {key: value.to(args.device) for key, value in encoded.items()}
            logits = model(**encoded, return_dict=True).logits.reshape(-1).float().cpu().numpy()
            if args.score_activation == "sigmoid":
                logits = sigmoid(logits)
            scores.extend(float(value) for value in logits.tolist())
    npu_synchronize(torch)
    return scores


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    if args.batch_size < 1 or args.max_length < 1:
        raise ValueError("--batch_size and --max_length must be positive")

    validate_attention_backend(args.attention_backend, args.dtype)
    torch, _torch_npu = load_npu_stack()
    torch.npu.set_device(args.device)
    set_compile_mode = getattr(torch.npu, "set_compile_mode", None)
    if set_compile_mode is not None:
        set_compile_mode(jit_compile=args.jit_compile)
    logger.info(
        "Ascend runtime torch=%s torch_npu=%s device=%s attention=%s jit_compile=%s",
        torch.__version__,
        getattr(_torch_npu, "__version__", "unknown"),
        args.device,
        args.attention_backend,
        args.jit_compile,
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
    _unused, mapping, skipped_queries = build_scoring_inputs(
        recall_results,
        ground_truth,
        instruction="",
    )
    if not mapping:
        raise ValueError("No query-document pairs remain after matching recall data to ground truth")

    reset_torch_accelerator_memory_stats()
    model, tokenizer = load_model_and_tokenizer(args, torch, _torch_npu)
    validate_rope_buffers(model, torch)
    queries = [str(row["query"]) for row in mapping]
    docs = [str(row["doc"]) for row in mapping]

    # Compile common kernels outside the measured pass. The measured pass still
    # includes any additional dynamic-shape compilation required by later batches.
    warmup_count = min(args.batch_size, len(queries))
    warmup_start = time.perf_counter()
    score_pairs(
        model,
        tokenizer,
        queries[:warmup_count],
        docs[:warmup_count],
        args,
        torch,
    )
    warmup_seconds = time.perf_counter() - warmup_start

    start = time.perf_counter()
    scores = score_pairs(model, tokenizer, queries, docs, args, torch)
    score_seconds = time.perf_counter() - start
    seconds_per_example = score_seconds / max(1, len(scores))

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
    memory = get_torch_accelerator_memory_mib()
    metrics.update(
        {
            "backend": "gte_torch_npu",
            "model_path": args.model_path,
            "gt_file": args.gt_file,
            "recall_file": args.recall_file,
            "gt_doc_id_col": args.gt_doc_id_col,
            "device": args.device,
            "dtype": args.dtype,
            "precision": args.dtype,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "score_activation": args.score_activation,
            "attention_backend": args.attention_backend,
            "jit_compile": args.jit_compile,
            "instruction": args.instruction,
            "warmup_seconds": warmup_seconds,
            "score_time_seconds": score_seconds,
            "seconds_per_example": seconds_per_example,
            "examples_per_second": len(scores) / score_seconds if score_seconds else 0.0,
            "skipped_recall_queries_without_gt": skipped_queries,
            **memory,
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

    del model
    cleanup_accelerator_memory()
    logger.info("Wrote GTE Ascend evaluation outputs to %s", output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
