#!/usr/bin/env python3
"""Preflight a static W8A8 Qwen3-Reranker model for 310P/vLLM-Ascend.

This checker is intentionally read-only. It validates the ModelSlim model
description and, unless ``--skip-runtime`` is used, checks the exact legacy
runtime and operators used by the supported 310P paths.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any


SUPPORTED_RUNTIME_PAIRS = (
    ("0.10.0", "0.10.0rc1"),
    ("0.10.2", "0.10.2rc1"),
)
QUANT_DESCRIPTION_NAME = "quant_model_description.json"
BODY_WEIGHT_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
FLOAT_HEAD_SUFFIXES = (
    "embed_tokens.weight",
    "lm_head.weight",
)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return ""


def load_quant_description(model_path: Path) -> tuple[Path, dict[str, Any]]:
    description_path = model_path / QUANT_DESCRIPTION_NAME
    if not description_path.is_file():
        raise FileNotFoundError(
            f"missing {QUANT_DESCRIPTION_NAME} in quantized model: {model_path}"
        )
    value = json.loads(description_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{description_path} must contain a JSON object")
    return description_path, value


def keys_ending_with(description: dict[str, Any], suffix: str) -> list[str]:
    return [key for key in description if str(key).endswith(suffix)]


def validate_quant_description(
    description: dict[str, Any],
) -> tuple[list[str], list[str], dict[str, int]]:
    failures: list[str] = []
    warnings: list[str] = []
    value_counts: dict[str, int] = {}
    for value in description.values():
        normalized = str(value).upper()
        value_counts[normalized] = value_counts.get(normalized, 0) + 1

    static_w8a8_count = value_counts.get("W8A8", 0)
    if static_w8a8_count == 0:
        failures.append("model description contains no static W8A8 layers")
    dynamic_types = sorted(
        quant_type
        for quant_type in value_counts
        if "DYNAMIC" in quant_type or "PERTOKEN" in quant_type
    )
    if dynamic_types:
        failures.append(
            "dynamic/per-token quantization is not enabled for this fixed 310P stack: "
            + ", ".join(dynamic_types)
        )

    for suffix in BODY_WEIGHT_SUFFIXES:
        matching_keys = keys_ending_with(description, suffix)
        if not matching_keys:
            warnings.append(f"no layer matched expected Qwen3 weight suffix {suffix}")
            continue
        non_w8a8 = [
            key for key in matching_keys if str(description[key]).upper() != "W8A8"
        ]
        if non_w8a8:
            warnings.append(
                f"{len(non_w8a8)}/{len(matching_keys)} {suffix} layers are not W8A8"
            )

    for suffix in FLOAT_HEAD_SUFFIXES:
        matching_keys = keys_ending_with(description, suffix)
        if not matching_keys:
            failures.append(
                f"model description has no {suffix}; export embeddings/head explicitly as FLOAT"
            )
            continue
        non_float = [
            key for key in matching_keys if str(description[key]).upper() != "FLOAT"
        ]
        if non_float:
            failures.append(
                f"{suffix} must remain FLOAT, found: "
                + ", ".join(f"{key}={description[key]}" for key in non_float)
            )

    score_keys = keys_ending_with(description, "score.weight")
    non_float_score = [
        key for key in score_keys if str(description[key]).upper() != "FLOAT"
    ]
    if non_float_score:
        failures.append("score.weight must remain FLOAT for Qwen3 pooling")
    elif not score_keys:
        warnings.append(
            "score.weight is absent; apply patch_vllm_ascend_0102_310p.py so "
            "the synthetic pooling score head stays unquantized FP32"
        )

    return failures, warnings, value_counts


def validate_runtime(device: int) -> tuple[list[str], list[str], dict[str, str]]:
    failures: list[str] = []
    warnings: list[str] = []
    versions = {
        "torch": package_version("torch"),
        "torch-npu": package_version("torch-npu"),
        "vllm": package_version("vllm"),
        "vllm-ascend": package_version("vllm-ascend"),
    }
    if not any(
        versions["vllm"].startswith(expected_vllm)
        and versions["vllm-ascend"].startswith(expected_ascend)
        for expected_vllm, expected_ascend in SUPPORTED_RUNTIME_PAIRS
    ):
        supported = ", ".join(
            f"{vllm}/{ascend}" for vllm, ascend in SUPPORTED_RUNTIME_PAIRS
        )
        failures.append(
            "unsupported vllm/vllm-ascend runtime pair %s/%s; expected %s"
            % (
                versions["vllm"] or "not installed",
                versions["vllm-ascend"] or "not installed",
                supported,
            )
        )

    try:
        import torch
        import torch_npu
    except Exception as exc:
        failures.append(f"torch/torch_npu import failed: {exc}")
        return failures, warnings, versions

    if not torch.npu.is_available():
        failures.append("torch.npu.is_available() is false")
        return failures, warnings, versions
    if device < 0 or device >= torch.npu.device_count():
        failures.append(
            f"device {device} is outside visible NPU range 0..{torch.npu.device_count() - 1}"
        )
        return failures, warnings, versions

    device_name = str(torch.npu.get_device_name(device))
    versions["device"] = device_name
    if "310P" not in device_name.upper():
        warnings.append(f"device name does not look like Ascend 310P: {device_name}")

    for operator in ("npu_quantize", "npu_quant_matmul", "npu_format_cast"):
        if not hasattr(torch_npu, operator):
            failures.append(f"torch_npu has no required static-W8A8 operator {operator}")

    return failures, warnings, versions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a ModelSlim static-W8A8 Qwen3-Reranker for Ascend 310P."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Validate only model files; useful on a non-Ascend preparation host.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    description_path, description = load_quant_description(model_path)
    failures, warnings, value_counts = validate_quant_description(description)
    print("model:", model_path)
    print("quant description:", description_path)
    print("quant types:", json.dumps(value_counts, ensure_ascii=False, sort_keys=True))

    if not args.skip_runtime:
        runtime_failures, runtime_warnings, versions = validate_runtime(args.device)
        failures.extend(runtime_failures)
        warnings.extend(runtime_warnings)
        print("runtime:", json.dumps(versions, ensure_ascii=False, sort_keys=True))

    for warning in warnings:
        print("WARNING:", warning)
    if failures:
        for failure in failures:
            print("ERROR:", failure)
        raise SystemExit(1)
    print("PASS: model and runtime are ready for the experimental 310P static-W8A8 path")


if __name__ == "__main__":
    main()
