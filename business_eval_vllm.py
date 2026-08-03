from __future__ import annotations

import argparse
import inspect
import importlib.metadata
import importlib.util
import json
import logging
import math
import os
import shutil
import sys
import time

from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm.auto import tqdm
from packaging.version import Version


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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


logger = logging.getLogger(__name__)

DEFAULT_VLLM_MODEL = "Qwen/Qwen3-Reranker-4B"
QWEN3_RERANKER_HF_OVERRIDES = {
    "architectures": ["Qwen3ForSequenceClassification"],
    "classifier_from_token": ["no", "yes"],
    "is_original_qwen3_reranker": True,
}
GTE_RERANKER_HF_OVERRIDES = {
    "architectures": ["GteNewForSequenceClassification"],
}

MIN_TRANSFORMERS_FOR_VLLM_0102 = "4.55.2"
MAX_TRANSFORMERS_FOR_VLLM_0102 = "5.0.0"
MIN_TOKENIZERS_FOR_VLLM_0102 = "0.21.1"
MAX_TOKENIZERS_FOR_VLLM_0102 = "0.22.0"
TOKENIZER_CONFIG_MAX_COPY_BYTES = 128 * 1024 * 1024
TOKENIZER_LARGE_SUFFIXES = {".bin", ".ckpt", ".pt", ".pth", ".safetensors"}
MIN_ASCEND_VLLM_PYTHON = (3, 9)
MAX_ASCEND_VLLM_PYTHON = (3, 12)


QWEN3_RERANKER_PREFIX = (
    '<|im_start|>system\n'
    ' Judge whether the Document meets the requirements based on the Query and '
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    '<|im_end|>\n<|im_start|>user\n'
)
QWEN3_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_qwen3_score_inputs(
    queries: list[str],
    documents: list[str],
    instruction: str,
) -> tuple[list[str], list[str]]:
    """Format inputs as the official vLLM Qwen3-Reranker score example does.

    vLLM 0.10.x documents Qwen3-Reranker with already formatted query/document
    strings before calling ``llm.score``. Doing it explicitly here keeps the
    scorer aligned with the official yes/no reranker prompt.
    """
    formatted_queries = [
        f"{QWEN3_RERANKER_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n"
        for query in queries
    ]
    formatted_documents = [
        f"<Document>: {document}{QWEN3_RERANKER_SUFFIX}"
        for document in documents
    ]
    return formatted_queries, formatted_documents


def format_qwen3_generate_prompts(
    queries: list[str],
    documents: list[str],
    instruction: str,
) -> list[str]:
    """Format full prompts for the official Qwen3-Reranker vLLM logprob path."""
    if len(queries) != len(documents):
        raise ValueError(f"queries/documents length mismatch: {len(queries)} != {len(documents)}")
    return [
        (
            f"{QWEN3_RERANKER_PREFIX}<Instruct>: {instruction}\n\n"
            f"<Query>: {query}\n\n<Document>: {document}{QWEN3_RERANKER_SUFFIX}"
        )
        for query, document in zip(queries, documents)
    ]


def format_gte_score_inputs(
    queries: list[str],
    documents: list[str],
    instruction: str,
) -> tuple[list[str], list[str]]:
    if len(queries) != len(documents):
        raise ValueError(f"queries/documents length mismatch: {len(queries)} != {len(documents)}")
    clean_instruction = instruction.strip()
    formatted_queries = [
        f"{clean_instruction}\n\n{query.strip()}" if clean_instruction else query.strip()
        for query in queries
    ]
    return formatted_queries, [document.strip() for document in documents]


def pooling_hf_overrides(model_family: str) -> dict[str, Any]:
    if model_family == "qwen3":
        return QWEN3_RERANKER_HF_OVERRIDES
    if model_family == "gte":
        return GTE_RERANKER_HF_OVERRIDES
    raise ValueError(f"Unsupported pooling model family: {model_family}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate business recall data with vLLM offline LLM.score reranking."
    )
    parser.add_argument("--gt_file", required=True, help="Excel/CSV file with query-doc ground truth.")
    parser.add_argument("--recall_file", required=True, help="JSON file with recalled docs per query.")
    parser.add_argument("--model_path", default=DEFAULT_VLLM_MODEL, help="Reranker model path or HF id.")
    parser.add_argument("--output_dir", default="outputs/business_eval_vllm")
    parser.add_argument("--instruction", default=DEFAULT_BUSINESS_INSTRUCTION)
    parser.add_argument("--gt_query_col", default="query")
    parser.add_argument("--gt_doc_id_col", default="PageId")
    parser.add_argument("--gt_sheet", default=None, help="Optional Excel sheet name. Defaults to active sheet.")
    parser.add_argument("--recall_id_key", default="id")
    parser.add_argument("--recall_text_key", default="text")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help=(
            "Compatibility chunk size used only with --no_submit_all_at_once. "
            "The default submits the complete dataset once and lets vLLM enforce "
            "max_num_seqs/max_num_batched_tokens internally."
        ),
    )
    parser.add_argument(
        "--model_family",
        choices=["qwen3", "gte"],
        default="qwen3",
        help="Select the native vLLM reranker architecture and input formatting.",
    )
    parser.add_argument(
        "--scoring_backend",
        choices=["pooling", "generate"],
        default="pooling",
        help=(
            "pooling uses vLLM LLM.score with the selected model-family override. "
            "generate is available only for the Qwen3-Reranker yes/no logprob path."
        ),
    )
    parser.add_argument("--top_k_list", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--expected_fbeta_beta",
        type=float,
        default=0.3,
        help="Beta used to choose the dynamic cutoff from normalized score prefix sums.",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="bfloat16",
    )
    parser.add_argument(
        "--device_backend",
        choices=["auto", "cuda", "ascend"],
        default=os.environ.get("VLLM_DEVICE_BACKEND", "auto"),
        help=(
            "Inference backend hint. Use ascend on Huawei Ascend/vllm-ascend "
            "machines; auto selects ascend when vllm_ascend is installed."
        ),
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_num_batched_tokens", type=int, default=32768)
    parser.add_argument("--max_num_seqs", type=int, default=256)
    parser.add_argument(
        "--warmup_pairs",
        type=int,
        default=0,
        help=(
            "Run this many query-document pairs once before timing. This is useful "
            "for stable latency measurements on Ascend eager execution."
        ),
    )
    parser.add_argument(
        "--enforce_eager",
        action="store_true",
        help="Disable vLLM graph capture and execute the model in eager mode.",
    )
    parser.add_argument(
        "--additional_config",
        default=os.environ.get("VLLM_ADDITIONAL_CONFIG", ""),
        help="Optional JSON object passed to vLLM LLM(..., additional_config=...).",
    )
    parser.add_argument(
        "--compilation_config",
        default=os.environ.get("VLLM_COMPILATION_CONFIG", ""),
        help="Optional JSON object passed to vLLM LLM(..., compilation_config=...).",
    )
    parser.add_argument(
        "--distributed_executor_backend",
        default=os.environ.get("VLLM_DISTRIBUTED_EXECUTOR_BACKEND", ""),
        help="Optional vLLM distributed executor backend, for example mp or ray.",
    )
    parser.add_argument(
        "--quantization",
        default=os.environ.get("VLLM_QUANTIZATION", ""),
        help="Optional vLLM quantization name, for example ascend for W8A8 Ascend weights.",
    )
    parser.add_argument(
        "--load_format",
        default=os.environ.get("VLLM_LOAD_FORMAT", ""),
        help=(
            "Optional vLLM load format. W8A8SC checkpoints produced by "
            "vLLM-Ascend save_sharded_state_310.py require sharded_state."
        ),
    )
    parser.add_argument("--enable_prefix_caching", action="store_true", default=True)
    parser.add_argument("--no_enable_prefix_caching", dest="enable_prefix_caching", action="store_false")
    parser.add_argument(
        "--submit_all_at_once",
        action="store_true",
        default=True,
        help="Submit the complete grouped dataset in one vLLM call for continuous batching.",
    )
    parser.add_argument(
        "--no_submit_all_at_once",
        dest="submit_all_at_once",
        action="store_false",
        help="Restore compatibility chunking with --batch_size.",
    )
    parser.add_argument(
        "--group_by_query",
        action="store_true",
        default=True,
        help="Keep all documents for the same query contiguous before submission.",
    )
    parser.add_argument(
        "--no_group_by_query",
        dest="group_by_query",
        action="store_false",
    )
    parser.add_argument(
        "--show_progress",
        action="store_true",
        help="Show per-submission tqdm progress. Disabled by default to avoid log I/O overhead.",
    )
    parser.add_argument(
        "--pretokenized_pooling",
        action="store_true",
        help=(
            "For pooling models, batch-tokenize every complete prompt with the "
            "Hugging Face fast tokenizer, verify every token-id sequence against "
            "the legacy LLM.score tokenizer path, then submit TokensPrompt inputs "
            "to LLM.encode with PoolingParams(task='score')."
        ),
    )
    parser.add_argument(
        "--tokenizer_batch_size",
        type=int,
        default=256,
        help="Fast-tokenizer batch size for --pretokenized_pooling. No padding is applied.",
    )
    parser.add_argument(
        "--prefix_cache_seeding",
        action="store_true",
        help=(
            "Submit one global seed, then one shortest-document seed per remaining "
            "query, then all remaining pairs so completed KV blocks exist before reuse."
        ),
    )
    parser.add_argument(
        "--reset_prefix_cache_after_warmup",
        action="store_true",
        help="Reset APC after warm-up so measured prefix seeding starts from a cold cache.",
    )
    parser.add_argument("--sort_by_length", action="store_true", default=True)
    parser.add_argument("--no_sort_by_length", dest="sort_by_length", action="store_false")
    parser.add_argument("--sort_descending", action="store_true", help="Score longer pairs first when length sorting.")
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Force offline Hugging Face/Transformers loading and fail if --model_path is not local.",
    )
    parser.add_argument(
        "--save_doc_text",
        action="store_true",
        help="Include full document text in predictions.jsonl for debugging.",
    )
    return parser.parse_args()


def package_version(package_name: str) -> str:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def resolve_device_backend(requested_backend: str) -> str:
    if requested_backend != "auto":
        return requested_backend
    if importlib.util.find_spec("vllm_ascend") is not None:
        return "ascend"
    return "cuda"


def prepare_vllm_platform(device_backend: str) -> str:
    resolved_backend = resolve_device_backend(device_backend)
    if resolved_backend == "ascend":
        try:
            import vllm_ascend  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "--device_backend ascend requires vllm-ascend. "
                "Install the Ascend environment with requirements-ascend-vllm.txt "
                "or run inside the official vllm-ascend image."
            ) from exc
        if not os.environ.get("ASCEND_RT_VISIBLE_DEVICES"):
            logger.warning(
                "ASCEND_RT_VISIBLE_DEVICES is not set; vLLM Ascend will use the runtime-visible NPUs."
            )
    return resolved_backend


def validate_vllm_python_runtime(device_backend: str) -> None:
    if device_backend != "ascend":
        return
    current = sys.version_info[:2]
    if MIN_ASCEND_VLLM_PYTHON <= current < MAX_ASCEND_VLLM_PYTHON:
        return
    current_text = ".".join(str(part) for part in sys.version_info[:3])
    raise RuntimeError(
        "This Ascend vLLM evaluator requires Python 3.9, 3.10, or 3.11; "
        f"this process is using Python {current_text} at {sys.executable}. "
        "Use the Python environment prepared by scripts/install_ascend_vllm_910b4.sh."
    )


def parse_json_object(text: str, option_name: str) -> dict[str, Any] | None:
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{option_name} must be a JSON object.")
    return value


def validate_vllm_python_stack(
    device_backend: str,
    vllm_version: str,
    transformers_version: str,
    tokenizers_version: str,
) -> None:
    if device_backend == "ascend":
        logger.info(
            "Using vLLM Ascend stack: vllm=%s vllm-ascend=%s transformers=%s tokenizers=%s",
            vllm_version,
            package_version("vllm-ascend") or package_version("vllm_ascend") or "unknown",
            transformers_version,
            tokenizers_version,
        )
        return

    if Version(vllm_version) != Version("0.10.2"):
        logger.warning(
            "The CUDA vLLM evaluator was validated with vllm==0.10.2; current vllm=%s. "
            "Continuing with feature-based kwarg filtering.",
            vllm_version,
        )
        return

    if Version(transformers_version) < Version(MIN_TRANSFORMERS_FOR_VLLM_0102):
        raise RuntimeError(
            f"transformers=={transformers_version} is too old for vllm==0.10.2. "
            f"Please upgrade to transformers>={MIN_TRANSFORMERS_FOR_VLLM_0102}. "
            "The Qwen2Tokenizer all_special_tokens_extended error is caused by this mismatch."
        )
    if Version(transformers_version) >= Version(MAX_TRANSFORMERS_FOR_VLLM_0102):
        raise RuntimeError(
            f"transformers=={transformers_version} is too new for vllm==0.10.2. "
            "Please use transformers>=4.55.2,<5.0.0 in the dedicated vLLM eval environment. "
            "The Qwen2Tokenizer all_special_tokens_extended error is caused by this mismatch."
        )
    if Version(tokenizers_version) < Version(MIN_TOKENIZERS_FOR_VLLM_0102):
        raise RuntimeError(
            f"tokenizers=={tokenizers_version} is too old for vllm==0.10.2. "
            f"Please upgrade to tokenizers>={MIN_TOKENIZERS_FOR_VLLM_0102}."
        )
    if Version(tokenizers_version) >= Version(MAX_TOKENIZERS_FOR_VLLM_0102):
        raise RuntimeError(
            f"tokenizers=={tokenizers_version} is too new for vllm==0.10.2. "
            "Please use tokenizers>=0.21.1,<0.22.0 in the dedicated vLLM eval environment."
        )


def filter_supported_kwargs(
    callable_obj: Callable[..., Any],
    kwargs: dict[str, Any],
    context: str,
) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        logger.warning("Could not inspect %s signature; passing all kwargs.", context)
        return kwargs

    parameters = signature.parameters
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        return kwargs

    supported: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in parameters:
            supported[key] = value
        else:
            logger.warning("Current vLLM does not support %s kwarg %r; skipped.", context, key)
    return supported


def pooling_api_kwargs(llm_cls: Callable[..., Any]) -> dict[str, str]:
    """Select the pooling API used by the installed vLLM.

    vLLM 0.10.0 exposes ``task="score"`` on ``LLM``. In 0.10.2 this was
    replaced by ``runner="pooling"``. Inspect the public constructor instead
    of relying on package-version strings, which may contain vendor suffixes.
    """
    try:
        parameters = inspect.signature(llm_cls).parameters
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Could not inspect the installed vLLM LLM constructor to select "
            "its pooling API."
        ) from exc

    if "runner" in parameters:
        return {"runner": "pooling"}
    if "task" in parameters:
        return {"task": "score"}
    raise RuntimeError(
        "The installed vLLM exposes neither LLM(..., runner='pooling') nor "
        "LLM(..., task='score'); this evaluator cannot select the score runner."
    )


def looks_like_local_path(model_path: str) -> bool:
    text = model_path.strip()
    return (
        text.startswith(("/", "./", "../", "~"))
        or "\\" in text
        or Path(text).is_absolute()
    )


def validate_vllm_model_path(model_path: str, require_local: bool = False) -> None:
    path = Path(model_path).expanduser()
    if not path.exists():
        if require_local or looks_like_local_path(model_path):
            raise FileNotFoundError(
                f"--model_path looks like a local path but does not exist: {model_path}"
            )
        return
    if path.is_file():
        return

    has_config = (path / "config.json").is_file() or (path / "params.json").is_file()
    is_adapter_only = (path / "adapter_config.json").is_file() and not has_config
    if is_adapter_only:
        raise RuntimeError(
            "vLLM cannot load a PEFT/LoRA adapter-only directory as --model_path. "
            f"The path has adapter_config.json but no config.json: {path}\n"
            "Merge the adapter into a full model first, for example:\n"
            "python src/merge_lora.py \\\n"
            f"  --adapter_path {path} \\\n"
            "  --base_model_path /path/to/base/Qwen3-Reranker-4B \\\n"
            f"  --output_dir {path.parent / (path.name + '_merged')} \\\n"
            "  --torch_dtype float16 \\\n"
            "  --overwrite\n"
            "Then pass the merged output directory to business_eval_vllm.py."
        )
    if not has_config:
        raise RuntimeError(
            "vLLM expects --model_path to be a full Hugging Face model directory "
            f"with config.json, but none was found in: {path}"
        )


def prepare_vllm_tokenizer_path(model_path: str, output_dir: str) -> str | None:
    path = Path(model_path)
    if not path.is_dir():
        return None

    tokenizer_config_path = path / "tokenizer_config.json"
    if not tokenizer_config_path.is_file():
        return None

    try:
        tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Could not parse tokenizer_config.json in {path}: {exc}") from exc

    extra_special_tokens = tokenizer_config.get("extra_special_tokens")
    if not isinstance(extra_special_tokens, list):
        return None

    tokenizer_dir = Path(output_dir) / "_vllm_tokenizer"
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for src in path.iterdir():
        if not src.is_file() or src.suffix in TOKENIZER_LARGE_SUFFIXES:
            continue
        if src.stat().st_size > TOKENIZER_CONFIG_MAX_COPY_BYTES:
            continue
        shutil.copy2(src, tokenizer_dir / src.name)
        copied += 1

    tokenizer_config["extra_special_tokens"] = {}
    (tokenizer_dir / "tokenizer_config.json").write_text(
        json.dumps(tokenizer_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.warning(
        "Patched tokenizer_config extra_special_tokens from list to dict for vLLM/transformers compatibility. "
        "Using tokenizer copy at %s (%d small config/tokenizer files copied).",
        tokenizer_dir,
        copied,
    )
    return str(tokenizer_dir)


def create_vllm_llm(args: argparse.Namespace) -> Any:
    if getattr(args, "local_files_only", False):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_DATASETS_OFFLINE"] = "1"
    device_backend = resolve_device_backend(getattr(args, "device_backend", "auto"))
    validate_vllm_python_runtime(device_backend)
    if device_backend == "ascend" and sys.version_info[:2] == (3, 9):
        from vllm_py39_compat import patch_vllm_011_for_python39

        changed = patch_vllm_011_for_python39()
        if changed:
            logger.info(
                "Applied Python 3.9 compatibility patch to %d vLLM files.",
                len(changed),
            )
    device_backend = prepare_vllm_platform(device_backend)
    setattr(args, "device_backend", device_backend)
    try:
        import tokenizers
        import transformers
        from vllm import LLM
        import vllm
    except ImportError as exc:
        raise RuntimeError(
            "business_eval_vllm.py requires vLLM and the selected platform plugin. "
            "For Ascend, install them with scripts/install_ascend_vllm_910b4.sh."
        ) from exc
    validate_vllm_python_stack(
        device_backend=device_backend,
        vllm_version=getattr(vllm, "__version__", "0"),
        transformers_version=transformers.__version__,
        tokenizers_version=tokenizers.__version__,
    )
    validate_vllm_model_path(args.model_path, require_local=getattr(args, "local_files_only", False))
    tokenizer_path = prepare_vllm_tokenizer_path(args.model_path, args.output_dir)
    additional_config = parse_json_object(getattr(args, "additional_config", ""), "--additional_config")
    compilation_config = parse_json_object(
        getattr(args, "compilation_config", ""), "--compilation_config"
    )

    scoring_backend = getattr(args, "scoring_backend", "pooling")
    model_family = getattr(args, "model_family", "qwen3")
    if model_family == "gte" and scoring_backend != "pooling":
        raise ValueError("GTE reranking requires --scoring_backend pooling")
    llm_kwargs: dict[str, Any] = {
        "model": args.model_path,
        "dtype": args.dtype,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_length,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_num_seqs": args.max_num_seqs,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": getattr(args, "enforce_eager", False),
    }
    if additional_config is not None:
        llm_kwargs["additional_config"] = additional_config
    if compilation_config is not None:
        llm_kwargs["compilation_config"] = compilation_config
    if getattr(args, "distributed_executor_backend", ""):
        llm_kwargs["distributed_executor_backend"] = args.distributed_executor_backend
    if getattr(args, "quantization", ""):
        llm_kwargs["quantization"] = args.quantization
    if getattr(args, "load_format", ""):
        llm_kwargs["load_format"] = args.load_format
    if scoring_backend == "pooling":
        llm_kwargs.update(pooling_api_kwargs(LLM))
        llm_kwargs["hf_overrides"] = pooling_hf_overrides(model_family)
        if model_family == "gte":
            # GTE's local tokenizer/config files use custom Auto classes. The
            # model itself is still the native vLLM implementation selected by
            # the architecture override above.
            llm_kwargs["trust_remote_code"] = True
    if tokenizer_path:
        llm_kwargs["tokenizer"] = tokenizer_path
    filtered_kwargs = filter_supported_kwargs(LLM, llm_kwargs, context="LLM")
    if tokenizer_path and "tokenizer" not in filtered_kwargs:
        raise RuntimeError(
            "This model tokenizer_config needs compatibility patching, but the installed vLLM "
            "does not support the LLM(..., tokenizer=...) argument."
        )
    if scoring_backend == "pooling" and not (
        filtered_kwargs.get("runner") == "pooling"
        or filtered_kwargs.get("task") == "score"
    ):
        raise RuntimeError(
            "This evaluator requires a vLLM score runner selected through "
            "LLM(..., task='score') or LLM(..., runner='pooling')."
        )
    logger.info("Initializing vLLM with kwargs: %s", json.dumps(_jsonable(filtered_kwargs), ensure_ascii=False))
    llm = LLM(**filtered_kwargs)
    setattr(llm, "_memranker_vllm_version", getattr(vllm, "__version__", "unknown"))
    setattr(llm, "_memranker_vllm_ascend_version", package_version("vllm-ascend") or "")
    setattr(llm, "_memranker_vllm_tokenizer_path", tokenizer_path or "")
    setattr(llm, "_memranker_scoring_backend", scoring_backend)
    setattr(llm, "_memranker_model_family", model_family)
    setattr(llm, "_memranker_device_backend", device_backend)
    setattr(
        llm,
        "_memranker_pooling_selector",
        "runner=pooling"
        if filtered_kwargs.get("runner") == "pooling"
        else "task=score"
        if filtered_kwargs.get("task") == "score"
        else "",
    )
    return llm


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def maybe_extract_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, dict):
        for key in ("score", "scores", "data", "value", "outputs"):
            if key in value:
                score = maybe_extract_numeric(value[key])
                if score is not None:
                    return score
    if isinstance(value, np.ndarray):
        return maybe_extract_numeric(value.tolist())
    # LLM.encode returns the base PoolingRequestOutput rather than the
    # ScoringRequestOutput wrapper produced by LLM.score. Its ``outputs.data``
    # can therefore still be a CPU torch.Tensor. Avoid importing torch in this
    # evaluator and normalize any tensor-like value through its public tolist.
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        converted = tolist()
        if converted is not value:
            return maybe_extract_numeric(converted)
    if isinstance(value, (list, tuple)):
        if value and all(isinstance(item, (int, float, np.floating, np.integer)) for item in value):
            if len(value) == 2:
                return float(value[1])
            if len(value) == 1:
                return float(value[0])
        if len(value) == 1:
            return maybe_extract_numeric(value[0])
        for item in value:
            score = maybe_extract_numeric(item)
            if score is not None:
                return score
    for attr in ("score", "scores", "data", "value"):
        if hasattr(value, attr):
            score = maybe_extract_numeric(getattr(value, attr))
            if score is not None:
                return score
    return None


def extract_vllm_score(output: Any) -> float:
    for candidate in (
        getattr(getattr(output, "outputs", None), "score", None),
        getattr(output, "score", None),
        getattr(output, "outputs", None),
        output,
    ):
        score = maybe_extract_numeric(candidate)
        if score is not None:
            return float(score)
    raise ValueError(f"Could not extract a numeric score from vLLM output: {output!r}")


def build_score_call_kwargs(llm: Any, max_length: int) -> dict[str, Any]:
    requested_kwargs = {
        "truncate_prompt_tokens": max_length,
        "use_tqdm": False,
    }
    return filter_supported_kwargs(llm.score, requested_kwargs, context="LLM.score")


def build_score_pooling_params() -> Any:
    try:
        from vllm import PoolingParams
    except ImportError:
        # Some vendor builds do not re-export PoolingParams from vllm.__init__.
        from vllm.pooling_params import PoolingParams

    return PoolingParams(task="score")


def build_pretokenized_encode_kwargs(llm: Any) -> dict[str, Any]:
    requested_kwargs = {
        "use_tqdm": False,
        "pooling_task": "score",
        # TokensPrompt already contains truncated IDs. Passing an explicit empty
        # mapping prevents encode() from applying text-tokenization options.
        "tokenization_kwargs": {},
    }
    supported = filter_supported_kwargs(
        llm.encode,
        requested_kwargs,
        context="LLM.encode",
    )
    if supported.get("pooling_task") != "score":
        raise RuntimeError(
            "PRETOKENIZED_POOLING requires LLM.encode(..., pooling_task='score'); "
            "the installed vLLM does not expose that API."
        )
    return supported


def first_token_mismatch(expected: list[int], actual: list[int]) -> int | None:
    for position, (expected_id, actual_id) in enumerate(zip(expected, actual)):
        if expected_id != actual_id:
            return position
    if len(expected) != len(actual):
        return min(len(expected), len(actual))
    return None


def prepare_pretokenized_pooling_inputs(
    llm: Any,
    indexed: list[tuple[int, str, str, int]],
    *,
    instruction: str,
    model_family: str,
    max_length: int,
    tokenizer_batch_size: int,
) -> tuple[list[dict[str, list[int]]], dict[str, float | int | bool]]:
    """Build, batch-tokenize, and fully validate buffered TokensPrompt inputs.

    The reference side deliberately reproduces vLLM 0.10.0 LLM.score's scalar
    tokenizer call for every prompt. All comparisons finish before the caller
    is allowed to enqueue any request, so a mismatch cannot produce a partial
    NPU run.
    """
    if tokenizer_batch_size < 1:
        raise ValueError("--tokenizer_batch_size must be >= 1")
    if max_length < 1:
        raise ValueError("--max_length must be >= 1")
    if model_family != "qwen3":
        raise ValueError("PRETOKENIZED_POOLING currently supports --model_family qwen3 only")

    tokenizer = llm.get_tokenizer()
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError(
            "PRETOKENIZED_POOLING requires a Hugging Face fast tokenizer, but "
            f"LLM.get_tokenizer() returned {type(tokenizer).__name__} with is_fast=False."
        )

    model_config = getattr(getattr(llm, "llm_engine", None), "model_config", None)
    if model_config is None:
        raise RuntimeError("PRETOKENIZED_POOLING could not read llm.llm_engine.model_config")
    if bool(getattr(model_config, "use_pad_token", False)):
        raise RuntimeError(
            "PRETOKENIZED_POOLING's complete-prompt path requires "
            "model_config.use_pad_token=False. Refusing to replace the legacy "
            "text/text_pair tokenizer semantics."
        )

    preparation_start = time.perf_counter()
    format_start = time.perf_counter()
    formatted_queries, formatted_documents = format_qwen3_score_inputs(
        [item[1] for item in indexed],
        [item[2] for item in indexed],
        instruction=instruction,
    )
    full_prompts = [
        formatted_query + formatted_document
        for formatted_query, formatted_document in zip(
            formatted_queries,
            formatted_documents,
        )
    ]
    format_seconds = time.perf_counter() - format_start

    token_ids_buffer: list[list[int]] = []
    tokenizer_seconds = 0.0
    for start in range(0, len(full_prompts), tokenizer_batch_size):
        prompt_buffer = full_prompts[start : start + tokenizer_batch_size]
        tokenize_start = time.perf_counter()
        encoded = tokenizer(
            text=prompt_buffer,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        tokenizer_seconds += time.perf_counter() - tokenize_start
        batch_token_ids = encoded.get("input_ids")
        if not isinstance(batch_token_ids, list) or len(batch_token_ids) != len(prompt_buffer):
            raise RuntimeError(
                "Fast tokenizer returned an invalid input_ids batch: "
                f"expected {len(prompt_buffer)} rows, got {type(batch_token_ids).__name__}."
            )
        token_ids_buffer.extend([list(map(int, token_ids)) for token_ids in batch_token_ids])

    validation_start = time.perf_counter()
    for ordered_idx, (full_prompt, actual_ids) in enumerate(
        zip(full_prompts, token_ids_buffer)
    ):
        reference = tokenizer(
            text=full_prompt,
            truncation=True,
            max_length=max_length,
        )
        expected_ids = list(map(int, reference["input_ids"]))
        mismatch_at = first_token_mismatch(expected_ids, actual_ids)
        if mismatch_at is not None:
            original_idx = indexed[ordered_idx][0]
            expected_id = expected_ids[mismatch_at] if mismatch_at < len(expected_ids) else None
            actual_id = actual_ids[mismatch_at] if mismatch_at < len(actual_ids) else None
            raise RuntimeError(
                "PRETOKENIZED_POOLING token parity check failed before vLLM submission: "
                f"ordered_index={ordered_idx} original_index={original_idx} "
                f"token_position={mismatch_at} expected_id={expected_id} "
                f"actual_id={actual_id} expected_length={len(expected_ids)} "
                f"actual_length={len(actual_ids)}. No request was submitted."
            )
    validation_seconds = time.perf_counter() - validation_start

    token_prompts = [
        {"prompt_token_ids": token_ids}
        for token_ids in token_ids_buffer
    ]
    preparation_seconds = time.perf_counter() - preparation_start
    timings: dict[str, float | int | bool] = {
        "prompt_format_time_seconds": float(format_seconds),
        "tokenizer_time_seconds": float(tokenizer_seconds),
        "token_id_validation_time_seconds": float(validation_seconds),
        "pretokenized_preparation_time_seconds": float(preparation_seconds),
        "num_token_ids_validated": len(token_prompts),
        "token_id_parity_passed": True,
        "tokenizer_batch_size": tokenizer_batch_size,
    }
    return token_prompts, timings


def load_yes_no_token_ids(model_path: str, tokenizer_path: str = "", local_files_only: bool = False) -> tuple[int, int]:
    from transformers import AutoTokenizer

    load_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(
        load_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    yes_id = tokenizer("yes", add_special_tokens=False).input_ids[0]
    no_id = tokenizer("no", add_special_tokens=False).input_ids[0]
    logger.info("Resolved vLLM generate yes/no token ids: yes=%s no=%s", yes_id, no_id)
    return int(yes_id), int(no_id)


def extract_yes_no_score_from_generate_output(output: Any, yes_token_id: int, no_token_id: int) -> float:
    generated = output.outputs[0]
    if not getattr(generated, "logprobs", None):
        raise ValueError(
            "vLLM generate output has no logprobs. The generate scorer requires SamplingParams(logprobs=...)."
        )
    final_logprobs = generated.logprobs[-1]
    yes_item = final_logprobs.get(yes_token_id)
    no_item = final_logprobs.get(no_token_id)
    yes_logprob = -10.0 if yes_item is None else float(yes_item.logprob)
    no_logprob = -10.0 if no_item is None else float(no_item.logprob)
    yes_prob = math.exp(yes_logprob)
    no_prob = math.exp(no_logprob)
    return float(yes_prob / max(yes_prob + no_prob, 1e-12))


def build_generate_sampling_params(llm: Any, yes_token_id: int, no_token_id: int) -> Any:
    from vllm import SamplingParams

    requested_kwargs = {
        "temperature": 0,
        "max_tokens": 1,
        "logprobs": 20,
        "allowed_token_ids": [yes_token_id, no_token_id],
    }
    filtered_kwargs = filter_supported_kwargs(SamplingParams, requested_kwargs, context="SamplingParams")
    # When allowed_token_ids is supported, the only possible output tokens are
    # yes and no. Returning 20 logprobs adds CPU serialization/post-processing
    # work without adding information. Keep 20 only for older vLLM releases
    # where allowed_token_ids is unavailable and yes/no must be found among the
    # regular top logprobs.
    if "allowed_token_ids" in filtered_kwargs and "logprobs" in filtered_kwargs:
        filtered_kwargs["logprobs"] = 2
    return SamplingParams(**filtered_kwargs)


def summarize_batch_latency(events: list[dict[str, Any]]) -> dict[str, float | int]:
    valid_events = [
        event
        for event in events
        if int(event.get("batch_size", 0)) > 0 and float(event.get("batch_seconds", -1.0)) >= 0
    ]
    if not valid_events:
        return {
            "num_timed_batches": 0,
            "batch_latency_p50_seconds": 0.0,
            "batch_latency_p95_seconds": 0.0,
            "pair_latency_p50_seconds": 0.0,
            "pair_latency_p95_seconds": 0.0,
        }

    batch_seconds = np.asarray(
        [float(event["batch_seconds"]) for event in valid_events],
        dtype=np.float64,
    )
    pair_seconds = np.asarray(
        [
            float(event["batch_seconds"]) / int(event["batch_size"])
            for event in valid_events
        ],
        dtype=np.float64,
    )
    return {
        "num_timed_batches": len(valid_events),
        "batch_latency_p50_seconds": float(np.percentile(batch_seconds, 50)),
        "batch_latency_p95_seconds": float(np.percentile(batch_seconds, 95)),
        "pair_latency_p50_seconds": float(np.percentile(pair_seconds, 50)),
        "pair_latency_p95_seconds": float(np.percentile(pair_seconds, 95)),
    }


def order_score_pairs(
    queries: list[str],
    documents: list[str],
    *,
    group_by_query: bool,
    sort_by_length: bool,
    sort_descending: bool,
) -> list[tuple[int, str, str, int]]:
    """Order pairs for cache locality while retaining their original indices.

    Recall JSON is normally a mapping from query to a list of documents, so it
    already arrives query-grouped. Rebuilding the groups here also handles
    callers that provide interleaved pairs. Length sorting is deliberately
    limited to each query group so it cannot destroy query-prefix locality.
    """
    indexed = [
        (idx, query, document, len(query) + len(document))
        for idx, (query, document) in enumerate(zip(queries, documents))
    ]
    if not group_by_query:
        if sort_by_length:
            indexed.sort(key=lambda item: item[3], reverse=sort_descending)
        return indexed

    query_order: list[str] = []
    query_groups: dict[str, list[tuple[int, str, str, int]]] = {}
    for item in indexed:
        query = item[1]
        if query not in query_groups:
            query_order.append(query)
            query_groups[query] = []
        query_groups[query].append(item)

    ordered: list[tuple[int, str, str, int]] = []
    for query in query_order:
        group = query_groups[query]
        if sort_by_length:
            group.sort(key=lambda item: item[3], reverse=sort_descending)
        ordered.extend(group)
    return ordered


def build_prefix_cache_seed_phases(
    indexed: list[tuple[int, str, str, int]],
) -> list[tuple[str, list[tuple[int, str, str, int]]]]:
    """Create dependency-ordered APC phases without adding model work.

    Every selected seed is an ordinary pair that must be scored anyway. The
    shortest pair for each query minimizes the time until its query prefix is
    committed to APC. One globally shortest query seed is completed first so
    the shared system/instruction blocks exist before the other query seeds.
    """
    if not indexed:
        return []

    query_order: list[str] = []
    query_groups: dict[str, list[tuple[int, str, str, int]]] = {}
    for item in indexed:
        query = item[1]
        if query not in query_groups:
            query_order.append(query)
            query_groups[query] = []
        query_groups[query].append(item)

    query_seeds = [
        min(query_groups[query], key=lambda item: (item[3], item[0]))
        for query in query_order
    ]
    global_seed = min(query_seeds, key=lambda item: (item[3], item[0]))
    remaining_query_seeds = [item for item in query_seeds if item[0] != global_seed[0]]
    seed_indices = {item[0] for item in query_seeds}
    remainder = [item for item in indexed if item[0] not in seed_indices]

    phases: list[tuple[str, list[tuple[int, str, str, int]]]] = [
        ("global_seed", [global_seed]),
    ]
    if remaining_query_seeds:
        phases.append(("query_seeds", remaining_query_seeds))
    if remainder:
        phases.append(("remainder", remainder))

    flattened_indices = [item[0] for _, phase in phases for item in phase]
    if len(flattened_indices) != len(indexed) or set(flattened_indices) != {
        item[0] for item in indexed
    }:
        raise RuntimeError("Internal error: prefix-cache seed plan lost or duplicated pairs")
    return phases


def score_with_vllm(
    llm: Any,
    queries: list[str],
    documents: list[str],
    batch_size: int,
    instruction: str,
    sort_by_length: bool,
    sort_descending: bool = False,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    scoring_backend: str | None = None,
    model_path: str = "",
    local_files_only: bool = False,
    model_family: str | None = None,
    max_length: int | None = None,
    submit_all_at_once: bool = True,
    group_by_query: bool = True,
    show_progress: bool = False,
    pretokenized_pooling: bool = False,
    tokenizer_batch_size: int = 256,
    prefix_cache_seeding: bool = False,
    timing_metrics: dict[str, Any] | None = None,
) -> list[float]:
    score_call_start = time.perf_counter()
    if len(queries) != len(documents):
        raise ValueError(f"queries/documents length mismatch: {len(queries)} != {len(documents)}")
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if not queries:
        return []

    indexed = order_score_pairs(
        queries,
        documents,
        group_by_query=group_by_query,
        sort_by_length=sort_by_length,
        sort_descending=sort_descending,
    )

    scores: list[float | None] = [None] * len(indexed)
    backend = scoring_backend or getattr(llm, "_memranker_scoring_backend", "pooling")
    resolved_model_family = model_family or getattr(llm, "_memranker_model_family", "qwen3")
    if backend not in {"pooling", "generate"}:
        raise ValueError(f"Unsupported scoring backend: {backend}")
    if pretokenized_pooling:
        if backend != "pooling":
            raise ValueError("PRETOKENIZED_POOLING requires --scoring_backend pooling")
        if not submit_all_at_once:
            raise ValueError("PRETOKENIZED_POOLING requires --submit_all_at_once")
        if max_length is None:
            raise ValueError("PRETOKENIZED_POOLING requires --max_length")
    if prefix_cache_seeding:
        if backend != "pooling":
            raise ValueError("PREFIX_CACHE_SEEDING requires --scoring_backend pooling")
        if not submit_all_at_once:
            raise ValueError("PREFIX_CACHE_SEEDING requires --submit_all_at_once")
        if not group_by_query:
            raise ValueError("PREFIX_CACHE_SEEDING requires --group_by_query")
    score_kwargs = (
        build_score_call_kwargs(llm, max_length=max_length)
        if backend == "pooling" and max_length is not None
        else {"use_tqdm": False} if backend == "pooling" else {}
    )
    sampling_params = None
    yes_token_id = None
    no_token_id = None
    if backend == "generate":
        tokenizer_path = getattr(llm, "_memranker_vllm_tokenizer_path", "")
        yes_token_id, no_token_id = load_yes_no_token_ids(
            model_path or getattr(llm, "llm_engine", None) or "",
            tokenizer_path=tokenizer_path,
            local_files_only=local_files_only,
        )
        sampling_params = build_generate_sampling_params(llm, yes_token_id, no_token_id)
    if pretokenized_pooling:
        token_prompts, pretokenized_timings = prepare_pretokenized_pooling_inputs(
            llm,
            indexed,
            instruction=instruction,
            model_family=resolved_model_family,
            max_length=max_length,
            tokenizer_batch_size=tokenizer_batch_size,
        )
        pooling_params = build_score_pooling_params()
        encode_kwargs = build_pretokenized_encode_kwargs(llm)
        token_prompt_by_index = {
            item[0]: token_prompt for item, token_prompt in zip(indexed, token_prompts)
        }
        phases = (
            build_prefix_cache_seed_phases(indexed)
            if prefix_cache_seeding
            else [("all", indexed)]
        )
        encode_seconds = 0.0
        completed = 0
        phase_timings: dict[str, float] = {}
        for phase_name, phase in phases:
            phase_prompts = [token_prompt_by_index[item[0]] for item in phase]
            encode_start = time.perf_counter()
            outputs = llm.encode(phase_prompts, pooling_params, **encode_kwargs)
            phase_seconds = time.perf_counter() - encode_start
            encode_seconds += phase_seconds
            phase_timings[phase_name] = float(phase_seconds)
            if len(outputs) != len(phase):
                raise ValueError(
                    f"vLLM returned {len(outputs)} outputs for {len(phase)} "
                    f"input pairs in prefix-cache phase {phase_name}"
                )
            for (original_idx, _, _, _), output in zip(phase, outputs):
                scores[original_idx] = extract_vllm_score(output)
            completed += len(phase)
            if progress_callback is not None:
                progress_callback(
                    {
                        "completed": completed,
                        "total": len(indexed),
                        "batch_size": len(phase),
                        "batch_seconds": phase_seconds,
                        "max_chars": max(item[3] for item in phase),
                        "phase": phase_name,
                    }
                )
        pretokenized_timings["vllm_enqueue_and_npu_execute_time_seconds"] = float(
            encode_seconds
        )
        pretokenized_timings["pretokenized_pipeline_time_seconds"] = float(
            pretokenized_timings["prompt_format_time_seconds"]
            + pretokenized_timings["tokenizer_time_seconds"]
            + encode_seconds
        )
        if prefix_cache_seeding:
            pretokenized_timings.update(
                {
                    f"prefix_cache_{phase_name}_time_seconds": phase_seconds
                    for phase_name, phase_seconds in phase_timings.items()
                }
            )
            pretokenized_timings["prefix_cache_seed_total_time_seconds"] = float(
                encode_seconds
            )
            pretokenized_timings["prefix_cache_seed_num_queries"] = len(
                {item[1] for item in indexed}
            )
            pretokenized_timings["prefix_cache_seed_num_pairs"] = len(
                {item[1] for item in indexed}
            )
            pretokenized_timings["prefix_cache_remainder_num_pairs"] = (
                len(indexed) - int(pretokenized_timings["prefix_cache_seed_num_pairs"])
            )
            pretokenized_timings["prefix_cache_seed_num_phases"] = len(phases)
        if timing_metrics is not None:
            timing_metrics.update(pretokenized_timings)
    else:
        if prefix_cache_seeding:
            planned_submissions = build_prefix_cache_seed_phases(indexed)
        else:
            submission_size = len(indexed) if submit_all_at_once else batch_size
            planned_submissions = [
                ("all" if submit_all_at_once else "chunk", indexed[start : start + submission_size])
                for start in range(0, len(indexed), submission_size)
            ]
        progress = tqdm(
            planned_submissions,
            total=len(planned_submissions),
            desc="vLLM submissions",
            unit="submission",
            dynamic_ncols=True,
            ascii=True,
            disable=not show_progress,
        )
        completed = 0
        phase_timings = {}
        for phase_name, batch in progress:
            batch_queries = [item[1] for item in batch]
            batch_documents = [item[2] for item in batch]
            batch_max_chars = max(item[3] for item in batch)
            batch_start_time = time.perf_counter()
            if backend == "pooling":
                if resolved_model_family == "gte":
                    formatted_queries, formatted_documents = format_gte_score_inputs(
                        batch_queries,
                        batch_documents,
                        instruction=instruction,
                    )
                elif resolved_model_family == "qwen3":
                    formatted_queries, formatted_documents = format_qwen3_score_inputs(
                        batch_queries,
                        batch_documents,
                        instruction=instruction,
                    )
                else:
                    raise ValueError(f"Unsupported vLLM model family: {resolved_model_family}")
                outputs = llm.score(formatted_queries, formatted_documents, **score_kwargs)
            else:
                prompts = format_qwen3_generate_prompts(
                    batch_queries,
                    batch_documents,
                    instruction=instruction,
                )
                outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
            batch_seconds = time.perf_counter() - batch_start_time
            if len(outputs) != len(batch):
                raise ValueError(
                    f"vLLM returned {len(outputs)} outputs for {len(batch)} "
                    f"input pairs in submission phase {phase_name}"
                )
            for (original_idx, _, _, _), output in zip(batch, outputs):
                if backend == "pooling":
                    scores[original_idx] = extract_vllm_score(output)
                else:
                    if yes_token_id is None or no_token_id is None:
                        raise RuntimeError("Internal error: missing yes/no token ids for generate scoring")
                    scores[original_idx] = extract_yes_no_score_from_generate_output(output, yes_token_id, no_token_id)
            completed += len(batch)
            phase_timings[phase_name] = phase_timings.get(phase_name, 0.0) + float(
                batch_seconds
            )
            if show_progress:
                progress.set_postfix(
                    scored=completed,
                    max_chars=batch_max_chars,
                    sec=f"{batch_seconds:.2f}",
                )
            if progress_callback is not None:
                progress_callback(
                    {
                        "completed": completed,
                        "total": len(indexed),
                        "batch_size": len(batch),
                        "batch_seconds": batch_seconds,
                        "max_chars": batch_max_chars,
                        "phase": phase_name,
                    }
                )
        if prefix_cache_seeding and timing_metrics is not None:
            timing_metrics.update(
                {
                    f"prefix_cache_{phase_name}_time_seconds": phase_seconds
                    for phase_name, phase_seconds in phase_timings.items()
                }
            )
            timing_metrics["prefix_cache_seed_total_time_seconds"] = float(
                sum(phase_timings.values())
            )
            timing_metrics["prefix_cache_seed_num_queries"] = len(
                {item[1] for item in indexed}
            )
            timing_metrics["prefix_cache_seed_num_pairs"] = len(
                {item[1] for item in indexed}
            )
            timing_metrics["prefix_cache_remainder_num_pairs"] = (
                len(indexed) - timing_metrics["prefix_cache_seed_num_pairs"]
            )
            timing_metrics["prefix_cache_seed_num_phases"] = len(planned_submissions)

    final_scores: list[float] = []
    bad_values = 0
    for idx, score in enumerate(scores):
        if score is None:
            raise ValueError(f"Missing score for pair index {idx}")
        score_float = float(score)
        if not math.isfinite(score_float):
            bad_values += 1
        final_scores.append(score_float)
    if len(final_scores) != len(queries):
        raise ValueError(f"Returned score count mismatch: {len(final_scores)} != {len(queries)}")
    if bad_values:
        logger.warning("vLLM scores contain %d NaN/inf values.", bad_values)
    if pretokenized_pooling and timing_metrics is not None:
        timing_metrics["pretokenized_total_time_seconds"] = float(
            time.perf_counter() - score_call_start
        )
    return final_scores


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()

    ground_truth = load_ground_truth(
        args.gt_file,
        query_col=args.gt_query_col,
        doc_id_col=args.gt_doc_id_col,
        sheet_name=args.gt_sheet,
    )
    logger.info("Loaded ground truth query count: %d", len(ground_truth))
    recall_results = load_recall_results(
        args.recall_file,
        id_key=args.recall_id_key,
        text_key=args.recall_text_key,
    )
    logger.info("Loaded recall query count: %d", len(recall_results))
    _input_texts, mapping, skipped_queries = build_scoring_inputs(
        recall_results,
        ground_truth,
        instruction=args.instruction,
    )
    if skipped_queries:
        logger.warning("Skipped %d recall queries not found in ground truth", skipped_queries)
    if not mapping:
        raise ValueError("No query-document pairs to score after matching recall data to ground truth.")
    logger.info("Total query-doc pairs to score: %d", len(mapping))
    logger.info(
        "Submission plan: query_groups=%d submit_all_at_once=%s "
        "sort_within_query=%s prefix_caching=%s pretokenized_pooling=%s "
        "prefix_cache_seeding=%s",
        len({row["query"] for row in mapping}),
        args.submit_all_at_once,
        args.sort_by_length if args.group_by_query else False,
        args.enable_prefix_caching,
        args.pretokenized_pooling,
        args.prefix_cache_seeding,
    )

    if args.tokenizer_batch_size < 1:
        raise ValueError("--tokenizer_batch_size must be >= 1")
    if args.pretokenized_pooling and args.scoring_backend != "pooling":
        raise ValueError("--pretokenized_pooling requires --scoring_backend pooling")
    if args.pretokenized_pooling and not args.submit_all_at_once:
        raise ValueError("--pretokenized_pooling requires --submit_all_at_once")
    if args.prefix_cache_seeding and args.scoring_backend != "pooling":
        raise ValueError("--prefix_cache_seeding requires --scoring_backend pooling")
    if args.prefix_cache_seeding and not args.enable_prefix_caching:
        raise ValueError("--prefix_cache_seeding requires --enable_prefix_caching")
    if args.prefix_cache_seeding and not args.submit_all_at_once:
        raise ValueError("--prefix_cache_seeding requires --submit_all_at_once")
    if args.prefix_cache_seeding and not args.group_by_query:
        raise ValueError("--prefix_cache_seeding requires --group_by_query")
    if args.reset_prefix_cache_after_warmup and not args.enable_prefix_caching:
        raise ValueError(
            "--reset_prefix_cache_after_warmup requires --enable_prefix_caching"
        )

    llm = create_vllm_llm(args)
    queries = [row["query"] for row in mapping]
    documents = [row["doc"] for row in mapping]

    if args.warmup_pairs < 0:
        raise ValueError("--warmup_pairs must be >= 0")
    warmup_count = min(args.warmup_pairs, len(queries))
    if warmup_count:
        logger.info("Warming up vLLM with %d query-document pairs", warmup_count)
        score_with_vllm(
            llm,
            queries=queries[:warmup_count],
            documents=documents[:warmup_count],
            batch_size=min(args.batch_size, warmup_count),
            instruction=args.instruction,
            sort_by_length=args.sort_by_length,
            sort_descending=args.sort_descending,
            scoring_backend=args.scoring_backend,
            model_path=args.model_path,
            local_files_only=args.local_files_only,
            model_family=args.model_family,
            max_length=args.max_length,
            submit_all_at_once=args.submit_all_at_once,
            group_by_query=args.group_by_query,
            show_progress=args.show_progress,
            pretokenized_pooling=args.pretokenized_pooling,
            tokenizer_batch_size=args.tokenizer_batch_size,
            prefix_cache_seeding=False,
        )

    prefix_cache_reset_after_warmup = False
    if warmup_count and args.reset_prefix_cache_after_warmup:
        reset_prefix_cache = getattr(llm, "reset_prefix_cache", None)
        if not callable(reset_prefix_cache):
            raise RuntimeError(
                "The installed vLLM does not expose LLM.reset_prefix_cache(), "
                "which is required for a cold prefix-seeding measurement."
            )
        if not bool(reset_prefix_cache()):
            raise RuntimeError("vLLM refused to reset the prefix cache after warm-up")
        prefix_cache_reset_after_warmup = True
        logger.info("Reset vLLM prefix cache after warm-up for a cold seeded measurement")

    batch_latency_events: list[dict[str, Any]] = []
    scoring_timing_metrics: dict[str, Any] = {}
    start_time = time.perf_counter()
    scores = score_with_vllm(
        llm,
        queries=queries,
        documents=documents,
        batch_size=args.batch_size,
        instruction=args.instruction,
        sort_by_length=args.sort_by_length,
        sort_descending=args.sort_descending,
        scoring_backend=args.scoring_backend,
        model_path=args.model_path,
        local_files_only=args.local_files_only,
        model_family=args.model_family,
        max_length=args.max_length,
        progress_callback=batch_latency_events.append,
        submit_all_at_once=args.submit_all_at_once,
        group_by_query=args.group_by_query,
        show_progress=args.show_progress,
        pretokenized_pooling=args.pretokenized_pooling,
        tokenizer_batch_size=args.tokenizer_batch_size,
        prefix_cache_seeding=args.prefix_cache_seeding,
        timing_metrics=scoring_timing_metrics,
    )
    score_time = time.perf_counter() - start_time
    sec_per_example = score_time / max(1, len(scores))
    examples_per_sec = len(scores) / score_time if score_time > 0 else 0.0
    logger.info(
        "vLLM scoring finished: pairs=%d time=%.3fs examples/s=%.3f",
        len(scores),
        score_time,
        examples_per_sec,
    )
    if args.pretokenized_pooling:
        logger.info(
            "Pretokenized timing: format=%.3fs tokenizer=%.3fs validation=%.3fs "
            "vllm_enqueue_and_npu_execute=%.3fs pipeline=%.3fs total=%.3fs",
            scoring_timing_metrics["prompt_format_time_seconds"],
            scoring_timing_metrics["tokenizer_time_seconds"],
            scoring_timing_metrics["token_id_validation_time_seconds"],
            scoring_timing_metrics["vllm_enqueue_and_npu_execute_time_seconds"],
            scoring_timing_metrics["pretokenized_pipeline_time_seconds"],
            scoring_timing_metrics["pretokenized_total_time_seconds"],
        )
    if args.prefix_cache_seeding:
        logger.info(
            "Prefix-cache seeded timing: global=%.3fs query_seeds=%.3fs "
            "remainder=%.3fs total=%.3fs phases=%d",
            scoring_timing_metrics.get("prefix_cache_global_seed_time_seconds", 0.0),
            scoring_timing_metrics.get("prefix_cache_query_seeds_time_seconds", 0.0),
            scoring_timing_metrics.get("prefix_cache_remainder_time_seconds", 0.0),
            scoring_timing_metrics["prefix_cache_seed_total_time_seconds"],
            scoring_timing_metrics["prefix_cache_seed_num_phases"],
        )

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
        seconds_per_example=sec_per_example,
        expected_fbeta_beta=args.expected_fbeta_beta,
    )
    metrics.update(
        {
            "backend": "vllm",
            "device_backend": args.device_backend,
            "vllm_runner": "pooling" if args.scoring_backend == "pooling" else "generate",
            "vllm_pooling_selector": getattr(llm, "_memranker_pooling_selector", ""),
            "scoring_backend": args.scoring_backend,
            "model_family": args.model_family,
            "vllm_version": getattr(llm, "_memranker_vllm_version", "unknown"),
            "vllm_ascend_version": getattr(llm, "_memranker_vllm_ascend_version", ""),
            "vllm_tokenizer_path": getattr(llm, "_memranker_vllm_tokenizer_path", ""),
            "dtype": args.dtype,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "warmup_pairs": warmup_count,
            "enforce_eager": args.enforce_eager,
            "additional_config": args.additional_config,
            "compilation_config": args.compilation_config,
            "distributed_executor_backend": args.distributed_executor_backend,
            "quantization": args.quantization,
            "load_format": args.load_format,
            "sort_by_length": args.sort_by_length,
            "sort_descending": args.sort_descending,
            "submit_all_at_once": args.submit_all_at_once,
            "group_by_query": args.group_by_query,
            "show_progress": args.show_progress,
            "pretokenized_pooling": args.pretokenized_pooling,
            "tokenizer_batch_size": args.tokenizer_batch_size,
            "prefix_cache_seeding": args.prefix_cache_seeding,
            "reset_prefix_cache_after_warmup": args.reset_prefix_cache_after_warmup,
            "prefix_cache_reset_after_warmup": prefix_cache_reset_after_warmup,
            "num_query_groups": len(set(queries)),
            "num_submission_calls": len(batch_latency_events),
            "local_files_only": args.local_files_only,
            "score_time_seconds": float(score_time),
            "seconds_per_example": float(sec_per_example),
            "examples_per_second": float(examples_per_sec),
            "num_scored_pairs": len(scores),
            "num_scored_queries": len({row["query"] for row in mapping}),
            "model_path": args.model_path,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "gt_file": args.gt_file,
            "recall_file": args.recall_file,
            "gt_query_col": args.gt_query_col,
            "gt_doc_id_col": args.gt_doc_id_col,
            "expected_fbeta_beta": args.expected_fbeta_beta,
            "skipped_recall_queries_without_gt": skipped_queries,
        }
    )
    metrics.update(scoring_timing_metrics)
    metrics.update(summarize_batch_latency(batch_latency_events))

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
    logger.info("Wrote vLLM business evaluation outputs to %s", output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
