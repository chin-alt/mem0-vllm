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
    parser.add_argument("--batch_size", type=int, default=256, help="Chunk size for vLLM score().")
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
    parser.add_argument("--enable_prefix_caching", action="store_true", default=True)
    parser.add_argument("--no_enable_prefix_caching", dest="enable_prefix_caching", action="store_false")
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
    if scoring_backend == "pooling":
        llm_kwargs["runner"] = "pooling"
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
    if scoring_backend == "pooling" and filtered_kwargs.get("runner") != "pooling":
        raise RuntimeError(
            "This evaluator requires vLLM with LLM(..., runner='pooling'), "
            "which is supported by vllm==0.10.2. Please install requirements-vllm.txt."
        )
    logger.info("Initializing vLLM with kwargs: %s", json.dumps(_jsonable(filtered_kwargs), ensure_ascii=False))
    llm = LLM(**filtered_kwargs)
    setattr(llm, "_memranker_vllm_version", getattr(vllm, "__version__", "unknown"))
    setattr(llm, "_memranker_vllm_ascend_version", package_version("vllm-ascend") or "")
    setattr(llm, "_memranker_vllm_tokenizer_path", tokenizer_path or "")
    setattr(llm, "_memranker_scoring_backend", scoring_backend)
    setattr(llm, "_memranker_model_family", model_family)
    setattr(llm, "_memranker_device_backend", device_backend)
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
    return SamplingParams(**filtered_kwargs)


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
) -> list[float]:
    if len(queries) != len(documents):
        raise ValueError(f"queries/documents length mismatch: {len(queries)} != {len(documents)}")
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    if not queries:
        return []

    indexed = [
        (idx, query, document, len(query) + len(document))
        for idx, (query, document) in enumerate(zip(queries, documents))
    ]
    if sort_by_length:
        indexed.sort(key=lambda item: item[3], reverse=sort_descending)

    scores: list[float | None] = [None] * len(indexed)
    backend = scoring_backend or getattr(llm, "_memranker_scoring_backend", "pooling")
    resolved_model_family = model_family or getattr(llm, "_memranker_model_family", "qwen3")
    if backend not in {"pooling", "generate"}:
        raise ValueError(f"Unsupported scoring backend: {backend}")
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
    total_batches = math.ceil(len(indexed) / batch_size)
    progress = tqdm(
        range(0, len(indexed), batch_size),
        total=total_batches,
        desc="vLLM scoring",
        unit="batch",
        dynamic_ncols=True,
        ascii=True,
    )
    for start in progress:
        batch = indexed[start : start + batch_size]
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
            raise ValueError(f"vLLM returned {len(outputs)} outputs for {len(batch)} input pairs")
        for (original_idx, _, _, _), output in zip(batch, outputs):
            if backend == "pooling":
                scores[original_idx] = extract_vllm_score(output)
            else:
                if yes_token_id is None or no_token_id is None:
                    raise RuntimeError("Internal error: missing yes/no token ids for generate scoring")
                scores[original_idx] = extract_yes_no_score_from_generate_output(output, yes_token_id, no_token_id)
        completed = min(start + len(batch), len(indexed))
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
                }
            )

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

    llm = create_vllm_llm(args)
    queries = [row["query"] for row in mapping]
    documents = [row["doc"] for row in mapping]

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
            "enforce_eager": args.enforce_eager,
            "additional_config": args.additional_config,
            "compilation_config": args.compilation_config,
            "distributed_executor_backend": args.distributed_executor_backend,
            "quantization": args.quantization,
            "sort_by_length": args.sort_by_length,
            "sort_descending": args.sort_descending,
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
