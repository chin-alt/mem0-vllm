#!/usr/bin/env python3
"""Build ModelSlim calibration JSONL from reranker training records.

The pinned ModelSlim loader reads the ``inputs_pretokenized`` field and then
tokenizes that string.  This script deliberately excludes labels/reasons from
the model input and preserves the Qwen3-Reranker answer-position suffix.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT = Path("/home/reranker_experiment/data/split/train.jsonl")
DEFAULT_MODEL = Path("/home/reranker_experiment/model/Qwen3-Reranker-0.6B")
DEFAULT_OUTPUT = Path(
    "/home/reranker_experiment/data/calibration/"
    "qwen3_reranker_w8a8_static_pooling_len1024_n64.jsonl"
)
QWEN3_RERANKER_PREFIX = (
    "<|im_start|>system\n"
    " Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN3_RERANKER_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


@dataclass(frozen=True)
class CalibrationCandidate:
    source_index: int
    prompt: str
    token_length: int
    original_token_length: int
    truncated: bool


def prompt_header(instruction: str, query: str, backend: str) -> str:
    instruction = instruction.strip()
    query = query.strip()
    if backend == "generate":
        return (
            f"{QWEN3_RERANKER_PREFIX}<Instruct>: {instruction}\n\n"
            f"<Query>: {query}\n\n<Document>: "
        )
    if backend == "pooling":
        return (
            f"{QWEN3_RERANKER_PREFIX}<Instruct>: {instruction}\n"
            f"<Query>: {query}\n<Document>: "
        )
    raise ValueError(f"unsupported backend: {backend}")


def format_prompt(instruction: str, query: str, document: str, backend: str) -> str:
    return prompt_header(instruction, query, backend) + document.strip() + QWEN3_RERANKER_SUFFIX


def _encode(tokenizer: Any, text: str, *, add_special_tokens: bool) -> list[int]:
    return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))


def _decode(tokenizer: Any, token_ids: list[int]) -> str:
    return str(
        tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )


def truncate_prompt_document(
    tokenizer: Any,
    instruction: str,
    query: str,
    document: str,
    backend: str,
    max_length: int,
) -> tuple[str, int, int, bool]:
    """Truncate only the document while retaining the classifier suffix."""
    header = prompt_header(instruction, query, backend)
    document = document.strip()
    original_prompt = header + document + QWEN3_RERANKER_SUFFIX
    original_length = len(_encode(tokenizer, original_prompt, add_special_tokens=True))
    if original_length <= max_length:
        return original_prompt, original_length, original_length, False

    empty_prompt = header + QWEN3_RERANKER_SUFFIX
    empty_length = len(_encode(tokenizer, empty_prompt, add_special_tokens=True))
    if empty_length >= max_length:
        raise ValueError(
            "instruction/query plus Qwen3 suffix already use "
            f"{empty_length} tokens, which does not fit max_length={max_length}; "
            "shorten the production instruction or query"
        )

    document_ids = _encode(tokenizer, document, add_special_tokens=False)
    keep = min(len(document_ids), max_length - empty_length)
    while keep >= 0:
        truncated_document = _decode(tokenizer, document_ids[:keep])
        prompt = header + truncated_document + QWEN3_RERANKER_SUFFIX
        token_length = len(_encode(tokenizer, prompt, add_special_tokens=True))
        if token_length <= max_length:
            return prompt, token_length, original_length, True
        keep -= max(1, token_length - max_length)

    raise RuntimeError("failed to truncate calibration prompt to max_length")


def read_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for source_index, line in enumerate(handle):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{source_index + 1}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"expected a JSON object at {path}:{source_index + 1}")
            yield source_index, row


def build_candidates(
    rows: Iterable[tuple[int, dict[str, Any]]],
    tokenizer: Any,
    backend: str,
    max_length: int,
    instruction_override: str = "",
) -> tuple[list[CalibrationCandidate], int, set[str]]:
    candidates: list[CalibrationCandidate] = []
    skipped = 0
    instructions: set[str] = set()
    for source_index, row in rows:
        instruction = (instruction_override or str(row.get("instruction", ""))).strip()
        query = str(row.get("query", row.get("question", ""))).strip()
        document = str(row.get("doc", row.get("document", ""))).strip()
        if not instruction or not query or not document:
            skipped += 1
            continue
        instructions.add(instruction)
        try:
            prompt, token_length, original_length, truncated = truncate_prompt_document(
                tokenizer=tokenizer,
                instruction=instruction,
                query=query,
                document=document,
                backend=backend,
                max_length=max_length,
            )
        except ValueError as exc:
            raise ValueError(f"source record {source_index}: {exc}") from exc
        candidates.append(
            CalibrationCandidate(
                source_index=source_index,
                prompt=prompt,
                token_length=token_length,
                original_token_length=original_length,
                truncated=truncated,
            )
        )
    return candidates, skipped, instructions


def select_length_stratified(
    candidates: list[CalibrationCandidate],
    sample_count: int,
    length_bins: int,
    seed: int,
) -> list[CalibrationCandidate]:
    if sample_count <= 0 or sample_count >= len(candidates):
        return list(candidates)
    bin_count = min(length_bins, sample_count, len(candidates))
    if bin_count < 1:
        raise ValueError("length_bins must be positive")

    ordered = sorted(candidates, key=lambda item: (item.original_token_length, item.source_index))
    bins = [
        ordered[index * len(ordered) // bin_count : (index + 1) * len(ordered) // bin_count]
        for index in range(bin_count)
    ]
    rng = random.Random(seed)
    base_quota, remainder = divmod(sample_count, bin_count)
    selected: list[CalibrationCandidate] = []
    for index, items in enumerate(bins):
        quota = base_quota + (1 if index < remainder else 0)
        selected.extend(rng.sample(items, min(quota, len(items))))

    if len(selected) < sample_count:
        selected_indices = {item.source_index for item in selected}
        remaining = [item for item in candidates if item.source_index not in selected_indices]
        selected.extend(rng.sample(remaining, sample_count - len(selected)))
    rng.shuffle(selected)
    return selected


def percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_outputs(
    output_path: Path,
    selected: list[CalibrationCandidate],
    manifest: dict[str, Any],
    overwrite: bool,
) -> Path:
    manifest_path = output_path.with_suffix(output_path.suffix + ".manifest.json")
    for path in (output_path, manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"output exists; pass --overwrite to replace it: {path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in selected:
            record = {
                "inputs_pretokenized": item.prompt,
                "source_index": item.source_index,
                "token_length": item.token_length,
                "original_token_length": item.original_token_length,
                "truncated": item.truncated,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest["output_sha256"] = sha256_file(output_path)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare static-W8A8 ModelSlim data for Qwen3-Reranker."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=("pooling", "generate"), default="pooling")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--sample-count", type=int, default=64, help="0 means use every valid row")
    parser.add_argument("--length-bins", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--instruction",
        default="",
        help="Optional production instruction override applied to every record.",
    )
    parser.add_argument(
        "--require-single-instruction",
        action="store_true",
        help="Fail if the source contains more than one instruction and no override is set.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_length < 64:
        raise ValueError("--max-length must be at least 64")
    if args.sample_count < 0:
        raise ValueError("--sample-count must be non-negative")
    if args.length_bins < 1:
        raise ValueError("--length-bins must be positive")

    input_path = args.input.expanduser().resolve()
    model_path = args.model_path.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"training JSONL does not exist: {input_path}")
    if not model_path.is_dir():
        raise FileNotFoundError(f"tokenizer/model directory does not exist: {model_path}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=False,
    )
    candidates, skipped, instructions = build_candidates(
        rows=read_jsonl(input_path),
        tokenizer=tokenizer,
        backend=args.backend,
        max_length=args.max_length,
        instruction_override=args.instruction,
    )
    if not candidates:
        raise ValueError(f"no valid instruction/query/doc records found in {input_path}")
    if args.require_single_instruction and not args.instruction and len(instructions) != 1:
        raise ValueError(
            "expected exactly one source instruction, found "
            f"{len(instructions)}; pass --instruction with the production value"
        )

    selected = select_length_stratified(
        candidates=candidates,
        sample_count=args.sample_count,
        length_bins=args.length_bins,
        seed=args.seed,
    )
    token_lengths = [item.token_length for item in selected]
    original_lengths = [item.original_token_length for item in selected]
    manifest = {
        "format": "modelslim.inputs_pretokenized",
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "model_path": str(model_path),
        "backend": args.backend,
        "max_length": args.max_length,
        "sample_count_requested": args.sample_count,
        "sample_count_written": len(selected),
        "length_bins": args.length_bins,
        "seed": args.seed,
        "valid_source_records": len(candidates),
        "skipped_source_records": skipped,
        "unique_source_instructions": len(instructions),
        "instruction_override": bool(args.instruction),
        "truncated_samples": sum(item.truncated for item in selected),
        "token_length": {
            "min": min(token_lengths),
            "p50": percentile(token_lengths, 0.50),
            "p90": percentile(token_lengths, 0.90),
            "max": max(token_lengths),
        },
        "original_token_length": {
            "min": min(original_lengths),
            "p50": percentile(original_lengths, 0.50),
            "p90": percentile(original_lengths, 0.90),
            "max": max(original_lengths),
        },
        "records": [
            {
                "source_index": item.source_index,
                "token_length": item.token_length,
                "original_token_length": item.original_token_length,
                "truncated": item.truncated,
            }
            for item in selected
        ],
    }
    manifest_path = write_outputs(output_path, selected, manifest, overwrite=args.overwrite)
    print(f"[calibration] source valid={len(candidates)} skipped={skipped}")
    print(
        "[calibration] wrote=%d truncated=%d token_p50=%d token_p90=%d token_max=%d"
        % (
            len(selected),
            manifest["truncated_samples"],
            manifest["token_length"]["p50"],
            manifest["token_length"]["p90"],
            manifest["token_length"]["max"],
        )
    )
    print(f"[calibration] jsonl={output_path}")
    print(f"[calibration] manifest={manifest_path}")


if __name__ == "__main__":
    main()
