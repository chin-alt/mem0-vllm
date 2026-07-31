#!/usr/bin/env python3
"""Run ATB examples.run_pa with a rank-local flat view of sharded weights."""

from __future__ import annotations

import argparse
import atexit
import os
import re
import runpy
import shutil
import sys
import tempfile
from pathlib import Path


PART_PATTERN = re.compile(r"^part(\d+)-of-(\d+)$")


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=(
            "Expose root config/tokenizer files and the current ATB rank's "
            "partN-of-M weights in one temporary directory."
        )
    )
    parser.add_argument("--model-root", type=Path, required=True)
    args, run_pa_args = parser.parse_known_args()
    return args, run_pa_args


def symlink_entry(source: Path, destination: Path) -> None:
    if destination.is_symlink():
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(
            f"refusing to replace non-symlink runtime entry: {destination}"
        )
    destination.symlink_to(source, target_is_directory=source.is_dir())


def prepare_flat_model(
    model_root: Path,
    local_rank: int,
    local_world_size: int,
) -> tuple[Path, Path | None]:
    model_root = model_root.expanduser().resolve()
    if not (model_root / "config.json").is_file():
        raise FileNotFoundError(f"model root has no config.json: {model_root}")

    direct_weights = sorted(model_root.glob("*.safetensors"))
    if direct_weights:
        return model_root, None

    part_dir = model_root / f"part{local_rank}-of-{local_world_size}"
    if not part_dir.is_dir():
        available_parts = sorted(
            child.name
            for child in model_root.iterdir()
            if child.is_dir() and PART_PATTERN.fullmatch(child.name)
        )
        raise FileNotFoundError(
            f"missing rank-local ATB weight directory {part_dir}; "
            f"available={available_parts}"
        )
    part_weights = sorted(part_dir.glob("*.safetensors"))
    if not part_weights:
        raise FileNotFoundError(f"no safetensors weights under {part_dir}")

    runtime_model = Path(
        tempfile.mkdtemp(prefix=f"memranker-atb-rank{local_rank}-")
    )
    for source in model_root.iterdir():
        if source.is_dir() and PART_PATTERN.fullmatch(source.name):
            continue
        symlink_entry(source, runtime_model / source.name)
    for source in part_dir.iterdir():
        symlink_entry(source, runtime_model / source.name)
    return runtime_model, runtime_model


def main() -> None:
    args, run_pa_args = parse_args()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", os.environ.get("WORLD_SIZE", "1"))
    )
    if local_rank < 0 or local_world_size < 1 or local_rank >= local_world_size:
        raise ValueError(
            f"invalid distributed layout: rank={local_rank} world={local_world_size}"
        )

    runtime_model, temporary_model = prepare_flat_model(
        args.model_root,
        local_rank,
        local_world_size,
    )
    if temporary_model is not None:
        atexit.register(shutil.rmtree, temporary_model, ignore_errors=True)
    print(
        "[atb-shard] rank=%d/%d root=%s runtime=%s"
        % (
            local_rank,
            local_world_size,
            args.model_root.resolve(),
            runtime_model,
        ),
        flush=True,
    )

    sys.argv = [
        "examples.run_pa",
        "--model_path",
        str(runtime_model),
        *run_pa_args,
    ]
    # torch.distributed.run launches this file by absolute path, making the
    # repository scripts directory sys.path[0]. The caller has already changed
    # into ATB_SPEED_HOME_PATH, so expose that working directory for the
    # ``examples.run_pa`` module import.
    sys.path.insert(0, str(Path.cwd()))
    runpy.run_module("examples.run_pa", run_name="__main__")


if __name__ == "__main__":
    main()
