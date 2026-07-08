from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge a PEFT LoRA adapter into a full Hugging Face causal LM directory for vLLM."
    )
    parser.add_argument("--adapter_path", required=True, help="LoRA adapter directory with adapter_config.json.")
    parser.add_argument("--output_dir", required=True, help="Directory to write the merged full model.")
    parser.add_argument(
        "--base_model_path",
        default=None,
        help="Optional local base model path. Defaults to base_model_name_or_path from adapter_config.json.",
    )
    parser.add_argument(
        "--torch_dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help="Dtype used while loading and saving the merged model. RTX 3090 usually prefers float16.",
    )
    parser.add_argument(
        "--device_map",
        default="auto",
        help='Transformers device_map. Use "auto" for GPU merge, "cpu" for CPU, or "none" to omit.',
    )
    parser.add_argument("--max_shard_size", default="4GB")
    parser.add_argument("--safe_serialization", action="store_true", default=True)
    parser.add_argument("--no_safe_serialization", dest="safe_serialization", action="store_false")
    parser.add_argument("--overwrite", action="store_true", help="Remove output_dir first if it already exists.")
    return parser.parse_args()


def dtype_from_name(name: str) -> Any:
    import torch

    if name == "auto":
        return "auto"
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def read_adapter_base(adapter_path: Path) -> str:
    config_path = adapter_path / "adapter_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing adapter_config.json in adapter path: {adapter_path}")
    data = json.loads(config_path.read_text(encoding="utf-8"))
    base_model = data.get("base_model_name_or_path")
    if not base_model:
        raise ValueError(
            f"adapter_config.json in {adapter_path} does not contain base_model_name_or_path; "
            "pass --base_model_path explicitly."
        )
    return str(base_model)


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is not empty: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()

    adapter_path = Path(args.adapter_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    base_model_path = args.base_model_path or read_adapter_base(adapter_path)
    tokenizer_path = adapter_path if (adapter_path / "tokenizer_config.json").is_file() else base_model_path
    dtype = dtype_from_name(args.torch_dtype)

    prepare_output_dir(output_dir, args.overwrite)
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading base model: %s", base_model_path)
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if args.device_map == "cpu":
        model_kwargs["device_map"] = {"": "cpu"}
    elif args.device_map != "none":
        model_kwargs["device_map"] = args.device_map

    base_model = AutoModelForCausalLM.from_pretrained(base_model_path, **model_kwargs)
    logger.info("Loading LoRA adapter: %s", adapter_path)
    peft_model = PeftModel.from_pretrained(base_model, str(adapter_path))

    logger.info("Merging LoRA weights into base model")
    merged_model = peft_model.merge_and_unload()
    if hasattr(merged_model.config, "use_cache"):
        merged_model.config.use_cache = True

    logger.info("Saving merged model to: %s", output_dir)
    merged_model.save_pretrained(
        output_dir,
        safe_serialization=args.safe_serialization,
        max_shard_size=args.max_shard_size,
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, padding_side="left")
    tokenizer.save_pretrained(output_dir)

    for filename in ("reranker_config.json", "training_args.json"):
        src = adapter_path / filename
        if src.is_file():
            shutil.copy2(src, output_dir / filename)

    merge_info = {
        "adapter_path": str(adapter_path),
        "base_model_path": str(base_model_path),
        "torch_dtype": args.torch_dtype,
        "device_map": args.device_map,
    }
    (output_dir / "merge_info.json").write_text(json.dumps(merge_info, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Done. vLLM can now use --model_path %s", output_dir)


if __name__ == "__main__":
    main()
