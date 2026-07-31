#!/usr/bin/env python3
"""Validate ModelSlim W8A8S/W8A8SC files for the 310P ATB path."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


W8A8S_BODY_GROUPS = (
    ("q_proj", ("self_attn.q_proj.weight",)),
    ("k_proj", ("self_attn.k_proj.weight",)),
    ("v_proj", ("self_attn.v_proj.weight",)),
    ("o_proj", ("self_attn.o_proj.weight",)),
    ("gate_proj", ("mlp.gate_proj.weight",)),
    ("up_proj", ("mlp.up_proj.weight",)),
    ("down_proj", ("mlp.down_proj.weight",)),
)

# ATB Models fuses Q/K/V and gate/up while splitting and compressing W8A8S.
# MindIE releases use either the public Qwen names or the older internal names
# printed by the 2.1.RC1 300I-Duo sparse_compressor.
W8A8SC_BODY_GROUPS = (
    (
        "fused_qkv",
        (
            "self_attn.query_key_value.weight",
            "attn.c_attn.weight",
        ),
    ),
    (
        "attention_output",
        (
            "self_attn.dense.weight",
            "attn.c_proj.weight",
        ),
    ),
    (
        "fused_gate_up",
        (
            "mlp.dense_h_to_4h.weight",
            "mlp.w2_w1.weight",
        ),
    ),
    (
        "mlp_output",
        (
            "mlp.dense_4h_to_h.weight",
            "mlp.c_proj.weight",
        ),
    ),
)

EMBEDDING_SUFFIXES = (
    "embed_tokens.weight",
    "word_embeddings.weight",
    "wte.weight",
)
LM_HEAD_SUFFIXES = (
    "lm_head.weight",
    "lm_head.linear.weight",
)
LAYER_PATTERN = re.compile(r"(?:^|\.)(?:layers|h)\.(\d+)\.")
PART_PATTERN = re.compile(r"^part(\d+)-of-(\d+)$")


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
    parser.add_argument(
        "--expected-parts",
        type=int,
        default=0,
        help="Expected ATB part count for W8A8SC; normally equal to TP_SIZE.",
    )
    args = parser.parse_args()
    if args.expected_parts < 0:
        parser.error("--expected-parts must be zero or a positive integer")
    return args


def read_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def description_priority(path: Path, expected: str) -> tuple[int, str]:
    preferred_names = (
        f"quant_model_description_{expected}.json",
        "quant_model_description.json",
    )
    try:
        priority = preferred_names.index(path.name)
    except ValueError:
        priority = len(preferred_names)
    return priority, path.name


def find_descriptions(
    model_path: Path,
    expected: str,
) -> list[tuple[Path, dict[str, Any]]]:
    """Find one matching description per root or ATB part directory."""

    by_parent: dict[Path, list[tuple[Path, dict[str, Any]]]] = {}
    mismatches: list[str] = []
    for path in sorted(model_path.rglob("quant_model_description*.json")):
        description = read_json_object(path)
        actual = str(description.get("model_quant_type", "")).lower()
        if actual == expected:
            by_parent.setdefault(path.parent, []).append((path, description))
        else:
            relative = path.relative_to(model_path)
            mismatches.append(f"{relative}={actual or '<missing>'}")

    selected: list[tuple[Path, dict[str, Any]]] = []
    for entries in by_parent.values():
        selected.append(
            min(entries, key=lambda item: description_priority(item[0], expected))
        )

    if expected == "w8a8s":
        root_selected = [item for item in selected if item[0].parent == model_path]
        if root_selected:
            selected = root_selected

    if not selected:
        detail = ", ".join(mismatches) if mismatches else "no description files"
        raise FileNotFoundError(
            f"no {expected.upper()} quantization description under {model_path}: {detail}"
        )
    return sorted(selected, key=lambda item: str(item[0]))


def keys_ending_with_any(
    description: dict[str, Any],
    suffixes: tuple[str, ...],
) -> list[str]:
    return [
        str(key)
        for key in description
        if any(str(key).endswith(suffix) for suffix in suffixes)
    ]


def validate_config(model_path: Path) -> tuple[int, list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    num_hidden_layers = 0
    config_path = model_path / "config.json"
    if not config_path.is_file():
        failures.append("config.json is missing from the W8A8SC root")
        return num_hidden_layers, failures, warnings

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
    return num_hidden_layers, failures, warnings


def validate_part_layout(
    model_path: Path,
    descriptions: list[tuple[Path, dict[str, Any]]],
    expected: str,
    expected_parts: int,
) -> list[str]:
    failures: list[str] = []
    if expected != "w8a8sc":
        return failures

    parts: dict[int, Path] = {}
    declared_totals: set[int] = set()
    for path, _description in descriptions:
        match = PART_PATTERN.fullmatch(path.parent.name)
        if match is None:
            failures.append(
                f"W8A8SC description is not inside partN-of-M: "
                f"{path.relative_to(model_path)}"
            )
            continue
        part_index = int(match.group(1))
        part_total = int(match.group(2))
        declared_totals.add(part_total)
        if part_index in parts:
            failures.append(f"duplicate W8A8SC part index {part_index}")
        parts[part_index] = path

    if len(declared_totals) != 1:
        failures.append(
            f"inconsistent W8A8SC part totals: {sorted(declared_totals)}"
        )
        return failures

    declared_total = next(iter(declared_totals))
    if expected_parts and declared_total != expected_parts:
        failures.append(
            f"W8A8SC was split into {declared_total} parts, expected {expected_parts}"
        )
    missing_parts = sorted(set(range(declared_total)) - set(parts))
    unexpected_parts = sorted(set(parts) - set(range(declared_total)))
    if missing_parts:
        failures.append(f"missing W8A8SC parts: {missing_parts}")
    if unexpected_parts:
        failures.append(f"unexpected W8A8SC parts: {unexpected_parts}")
    return failures


def layer_coverage_failures(
    keys: list[str],
    label: str,
    num_hidden_layers: int,
) -> list[str]:
    failures: list[str] = []
    matches = [LAYER_PATTERN.search(key) for key in keys]
    layer_indices = {
        int(match.group(1)) for match in matches if match is not None
    }
    expected_indices = set(range(num_hidden_layers))
    missing_indices = sorted(expected_indices - layer_indices)
    unexpected_indices = sorted(layer_indices - expected_indices)
    unindexed_count = sum(match is None for match in matches)
    if missing_indices:
        failures.append(f"{label} missing layers={missing_indices}")
    if unexpected_indices:
        failures.append(f"{label} unexpected layers={unexpected_indices}")
    if unindexed_count:
        failures.append(f"{label} has {unindexed_count} unindexed keys")
    if len(keys) != num_hidden_layers:
        failures.append(
            f"{label} has {len(keys)} weights; expected {num_hidden_layers}"
        )
    return failures


def validate_weight_group(
    description: dict[str, Any],
    label: str,
    suffixes: tuple[str, ...],
    num_hidden_layers: int,
    expected_type: str,
    require_compression_metadata: bool,
) -> list[str]:
    failures: list[str] = []
    keys = keys_ending_with_any(description, suffixes)
    if not keys:
        return [f"no Qwen3/ATB body weight matched {label}: {suffixes}"]

    failures.extend(layer_coverage_failures(keys, label, num_hidden_layers))
    wrong = [
        key for key in keys if str(description.get(key, "")).upper() != expected_type
    ]
    if wrong:
        failures.append(
            f"{len(wrong)}/{len(keys)} {label} weights are not {expected_type}"
        )

    if require_compression_metadata:
        for weight_key in keys:
            base = weight_key[: -len(".weight")]
            for metadata_name in ("index", "info"):
                metadata_key = f"{base}.{metadata_name}"
                actual = str(description.get(metadata_key, "")).upper()
                if actual != "W8A8SC":
                    failures.append(
                        f"{metadata_key} must be W8A8SC; found {actual or '<missing>'}"
                    )
    return failures


def validate_float_weights(
    description: dict[str, Any],
    expected: str,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    embedding_keys = keys_ending_with_any(description, EMBEDDING_SUFFIXES)
    if not embedding_keys:
        failures.append(
            f"no floating embedding weight matched {EMBEDDING_SUFFIXES}"
        )
    else:
        wrong = [
            key
            for key in embedding_keys
            if str(description.get(key, "")).upper() != "FLOAT"
        ]
        if wrong:
            failures.append("embedding weights must remain FLOAT: " + ", ".join(wrong))

    lm_head_keys = keys_ending_with_any(description, LM_HEAD_SUFFIXES)
    if not lm_head_keys:
        message = f"no lm_head weight matched {LM_HEAD_SUFFIXES}"
        if expected == "w8a8s":
            failures.append(message)
        else:
            # ATB may omit a tied lm_head from a TP part description.
            warnings.append(message + "; accepting tied-head ATB output")
    else:
        wrong = [
            key
            for key in lm_head_keys
            if str(description.get(key, "")).upper() != "FLOAT"
        ]
        if wrong:
            failures.append("lm_head weights must remain FLOAT: " + ", ".join(wrong))
    return failures, warnings


def validate_description(
    description_path: Path,
    description: dict[str, Any],
    expected: str,
    num_hidden_layers: int,
) -> tuple[list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    expected_upper = expected.upper()
    counts = Counter(str(value).upper() for value in description.values())
    if counts[expected_upper] == 0:
        failures.append(f"description contains no {expected_upper} tensors")

    body_groups = (
        W8A8S_BODY_GROUPS if expected == "w8a8s" else W8A8SC_BODY_GROUPS
    )
    for label, suffixes in body_groups:
        failures.extend(
            validate_weight_group(
                description=description,
                label=label,
                suffixes=suffixes,
                num_hidden_layers=num_hidden_layers,
                expected_type=expected_upper,
                require_compression_metadata=expected == "w8a8sc",
            )
        )

    float_failures, float_warnings = validate_float_weights(description, expected)
    failures.extend(float_failures)
    warnings.extend(float_warnings)

    weights = sorted(description_path.parent.glob("*.safetensors"))
    if not weights:
        failures.append(
            f"no safetensors weight file beside {description_path.name}"
        )
    print("description:", description_path)
    print(
        "quant types:",
        json.dumps(dict(sorted(counts.items())), ensure_ascii=False),
    )
    print("part safetensors:", len(weights))
    return failures, warnings


def main() -> None:
    args = parse_args()
    model_path = args.model_path.expanduser().resolve()
    if not model_path.is_dir():
        raise FileNotFoundError(f"model directory does not exist: {model_path}")

    descriptions = find_descriptions(model_path, args.expected)
    num_hidden_layers, failures, warnings = validate_config(model_path)
    failures.extend(
        validate_part_layout(
            model_path,
            descriptions,
            args.expected,
            args.expected_parts,
        )
    )
    for description_path, description in descriptions:
        part_failures, part_warnings = validate_description(
            description_path,
            description,
            args.expected,
            num_hidden_layers,
        )
        failures.extend(
            f"{description_path.parent.name}: {failure}"
            for failure in part_failures
        )
        warnings.extend(
            f"{description_path.parent.name}: {warning}"
            for warning in part_warnings
        )

    print("model:", model_path)
    print("description parts:", len(descriptions))
    for warning in warnings:
        print("WARNING:", warning)
    if failures:
        for failure in failures:
            print("ERROR:", failure)
        raise SystemExit(1)
    print(
        f"PASS: complete {args.expected.upper()} Qwen3 body "
        f"for the 310P ATB path"
    )


if __name__ == "__main__":
    main()
