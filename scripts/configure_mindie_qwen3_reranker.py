from __future__ import annotations

import argparse
import json
import shutil

from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Configure MindIE for one local Qwen3 reranker model.")
    parser.add_argument("--config", required=True, help="MindIE service config.json")
    parser.add_argument("--model_path", required=True, help="Model path as seen inside the container")
    parser.add_argument("--model_name", default="qwen3-reranker-4b")
    parser.add_argument("--npu_devices", default="0", help="Comma-separated logical NPU ids")
    parser.add_argument("--max_length", type=int, default=8192)
    parser.add_argument("--max_batch_size", type=int, default=32)
    parser.add_argument("--max_prefill_tokens", type=int, default=32768)
    parser.add_argument("--port", type=int, default=1025)
    parser.add_argument("--management_port", type=int, default=1026)
    parser.add_argument("--metrics_port", type=int, default=1027)
    parser.add_argument("--listen_address", default="127.0.0.1")
    parser.add_argument(
        "--patch_model_dtype",
        action="store_true",
        help="Back up model config.json and set torch_dtype/dtype to float16 for 310P.",
    )
    parser.add_argument("--output", default="", help="Write a new config instead of editing --config")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json_with_backup(path: Path, value: dict[str, Any], backup: bool) -> None:
    if backup and path.exists():
        backup_path = path.with_suffix(path.suffix + ".memranker.bak")
        if not backup_path.exists():
            shutil.copy2(path, backup_path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")


def parse_devices(text: str) -> list[int]:
    devices = [int(item.strip()) for item in text.split(",") if item.strip()]
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--npu_devices must contain one or more non-negative ids")
    if len(set(devices)) != len(devices):
        raise ValueError("--npu_devices contains duplicate ids")
    return devices


def patch_model_dtype(model_path: Path) -> None:
    config_path = model_path / "config.json"
    model_config = read_json(config_path)
    changed = False
    for key in ("torch_dtype", "dtype"):
        if key in model_config and model_config[key] != "float16":
            print(f"[model] {key}: {model_config[key]!r} -> 'float16'")
            model_config[key] = "float16"
            changed = True
    if "torch_dtype" not in model_config:
        model_config["torch_dtype"] = "float16"
        changed = True
    if changed:
        write_json_with_backup(config_path, model_config, backup=True)
        print(f"[model] updated 310P dtype in {config_path}")
    else:
        print(f"[model] dtype already float16 in {config_path}")


def configure(args: argparse.Namespace) -> Path:
    if args.max_length < 1:
        raise ValueError("--max_length must be >= 1")
    if args.max_batch_size < 1:
        raise ValueError("--max_batch_size must be >= 1")
    if args.max_prefill_tokens < args.max_length:
        raise ValueError("--max_prefill_tokens must be >= --max_length")

    config_path = Path(args.config)
    output_path = Path(args.output) if args.output else config_path
    model_path = Path(args.model_path)
    if not (model_path / "config.json").is_file():
        raise FileNotFoundError(f"Model directory has no config.json: {model_path}")
    if args.patch_model_dtype:
        patch_model_dtype(model_path)

    devices = parse_devices(args.npu_devices)
    config = read_json(config_path)
    server = config.setdefault("ServerConfig", {})
    backend = config.setdefault("BackendConfig", {})
    deploy = backend.setdefault("ModelDeployConfig", {})
    schedule = backend.setdefault("ScheduleConfig", {})
    if not all(isinstance(item, dict) for item in (server, backend, deploy, schedule)):
        raise ValueError(
            "ServerConfig, BackendConfig, ModelDeployConfig, and ScheduleConfig "
            "must be JSON objects"
        )

    server.update(
        {
            "ipAddress": args.listen_address,
            "managementIpAddress": args.listen_address,
            "port": args.port,
            "managementPort": args.management_port,
            "metricsPort": args.metrics_port,
            "httpsEnabled": False,
            "openAiSupport": "vllm",
        }
    )
    backend["npuDeviceIds"] = [devices]
    schedule["enablePrefixCache"] = False
    deploy.update(
        {
            "maxSeqLen": args.max_length + 1,
            "maxInputTokenLen": args.max_length,
            "truncation": True,
            "maxBatchSize": args.max_batch_size,
            "maxPrefillBatchSize": args.max_batch_size,
            "maxPrefillTokens": args.max_prefill_tokens,
            "maxIterTimes": 1,
        }
    )

    existing_models = deploy.get("ModelConfig")
    if isinstance(existing_models, list) and existing_models and isinstance(existing_models[0], dict):
        model_config = dict(existing_models[0])
    elif isinstance(existing_models, dict):
        model_config = dict(existing_models)
    else:
        model_config = {}
    model_config.update(
        {
            "modelInstanceType": "Standard",
            "modelName": args.model_name,
            "modelWeightPath": str(model_path),
            "worldSize": len(devices),
            "backendType": "atb",
            "trustRemoteCode": False,
        }
    )
    model_config.pop("plugin_params", None)
    deploy["ModelConfig"] = [model_config]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_with_backup(output_path, config, backup=output_path == config_path)
    return output_path


def main() -> None:
    args = parse_args()
    output_path = configure(args)
    print(f"[mindie] configured {output_path}")
    print(f"[mindie] model={args.model_name} path={args.model_path}")
    print(f"[mindie] endpoint=http://{args.listen_address}:{args.port}/v1/completions")


if __name__ == "__main__":
    main()
