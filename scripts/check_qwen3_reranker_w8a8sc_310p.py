#!/usr/bin/env python3
"""Validate ModelSlim W8A8S/W8A8SC files for the 310P ATB path."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


BODY_WEIGHT_SUFFIXES = (
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
)
FLOAT_WEIGHT_SUFFIXES = (
    "embed_tokens.weight",
    "lm_head.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a Qwen3-Reranker ModelSlim sparse-quantized model."
    )
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--expected",
        choices=("w8a8s", "w8a8sc"),
        required=True,
        help="Expected ModelSlim model_quant_type.",
    )
    return parser.parse_args()


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def find_description(model_path: Path, expected: str) -> tuple[Path, dict[str, Any]]:
    preferred = (
        model_path / f"quant_model_description_{expected}.json",
        model_path / "quant_model_description.json",
    )
    candidates = list(preferred)
    candidates.extend(sorted(model_path.glob("quant_model_description*.json")))
    seen: set[Path] = set()
    mismatches: list[str] = []
    for path in candidates:
        if path in seen or not path.is_file():
            continue
        seen.add(path)
        description = read_json_object(path)
        actual = str(description.get("model_quant_type", "")).lower()
        if actual == expected:
            return path, description
        mismatches.append(f"{path.name}={actual or '<missing>'}")
    detail = ", ".join(mismatches) if mismatches else "no description files"
    raise FileNotFoundError(
        f"no {expected.upper()} quantization description under {model_path}: {detail}"
    )


def keys_ending_with(description: dict[str, Any], suffix: str) -> list[str]:
    return [key for key in description if str(key).endswith(suffix)]


def validate(
    model_path: Path,
    description: dict[str, Any],
    expected: str,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    expected_upper = expected.upper()
    num_hidden_layers = 0

    config_path = model_path / "config.json"
    if not config_path.is_file():
        failures.append("config.json is missing")
    else:
        config = read_json_object(config_path)
        dtype = str(config.get("torch_dtype", config.get("dtype", ""))).lower()
        if dtype not in ("float16", "torch.float16"):
            failures.append(
                f"310P ATB requires config torch_dtype=float16; found {dtype or '<missing>'}"
            )
        model_type = str(config.get("model_type", "")).lower()
        if model_type != "qwen3":
            warnings.append(
                f"expected Qwen3 model_type for this workflow; found {model_type or '<missing>'}"
            )
        raw_num_hidden_layers = config.get("num_hidden_layers", 0)
        if not isinstance(raw_num_hidden_layers, int) or raw_num_hidden_layers < 1:
            failures.append(
                "config num_hidden_layers must be a positive integer; found "
                f"{raw_num_hidden_layers!r}"
            )
        else:
            num_hidden_layers = raw_num_hidden_layers

    weights = sorted(model_path.rglob("*.safetensors"))
    if not weights:
        failures.append("no safetensors weight files found")

    counts = Counter(str(value).upper() for value in description.values())
    if counts[expected_upper] == 0:
        failures.append(f"description contains no {expected_upper} tensors")

    for suffix in BODY_WEIGHT_SUFFIXES:
        keys = keys_ending_with(description, suffix)
        if not keys:
            failures.append(f"no Qwen3 body weight matched {suffix}")
            continue
        layer_matches = [
            re.search(r"(?:^|\.)layers\.(\d+)\.", key)
            for key in keys
        ]
        layer_indices = {
            int(match.group(1)) for match in layer_matches if match is not None
        }
        if num_hidden_layers:
            expected_indices = set(range(num_hidden_layers))
            missing_indices = sorted(expected_indices - layer_indices)
            unexpected_indices = sorted(layer_indices - expected_indices)
            unindexed_count = sum(match is None for match in layer_matches)
            if missing_indices or unexpected_indices or unindexed_count:
                details: list[str] = []
                if missing_indices:
                    details.append(f"missing layers={missing_indices}")
                if unexpected_indices:
                    details.append(f"unexpected layers={unexpected_indices}")
                if unindexed_count:
                    details.append(f"unindexed keys={unindexed_count}")
                failures.append(f"incomplete {suffix} coverage: " + "; ".join(details))
        wrong = [key for key in keys if str(description[key]).upper() != expected_upper]
        if wrong:
            failures.append(
                f"{len(wrong)}/{len(keys)} {suffix} weights are not {expected_upper}"
            )

    for suffix in FLOAT_WEIGHT_SUFFIXES:
        keys = keys_ending_with(description, suffix)
        if not keys:
            failures.append(f"no required floating-point weight matched {suffix}")
            continue
        wrong = [key for key in keys if str(description[key]).upper() != "FLOAT"]
        if wrong:
            failures.append(
                f"{suffix} must remain FLOAT; found "
                + ", ".join(f"{key}={description[key]}" for key in wrong)
            )

    if expected == "w8a8sc":
        index_keys = keys_ending_with(description, ".index")
        info_keys = keys_ending_with(description, ".info")
        if not index_keys or not info_keys:
            failures.append("compressed model has no W8A8SC index/info tensors")
        for keys, label in ((index_keys, "index"), (info_keys, "info")):
            wrong = [key for key in keys if str(description[key]).upper() != "W8A8SC"]
            if wrong:
                failures.append(f"{label} tensors are not marked W8A8SC")

    print("model:", model_path)
    print("quant types:", json.dumps(dict(sorted(counts.items())), ensure_ascii=False))
    print("safetensors:", len(weights))
    return failures, warnings


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")
    description_path, description = find_description(model_path, args.expected)
    print("description:", description_path)
    failures, warnings = validate(model_path, description, args.expected)
    for warning in warnings:
        print("WARNING:", warning)
    if failures:
        for failure in failures:
            print("ERROR:", failure)
        raise SystemExit(1)
    print(f"PASS: complete {args.expected.upper()} Qwen3 body for the 310P ATB path")


if __name__ == "__main__":
    main()
