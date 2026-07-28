#!/usr/bin/env python3
"""Disable the vLLM-Ascend 0.10.2rc1 ATB warm-up that fails on 310P."""

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path


SUPPORTED_VERSION = "0.10.2rc1"
CALL = "        self._warm_up_atb()"
MARKER = "        pass  # MEMRANKER_310P_SKIP_ATB_WARMUP"


def patch_worker(worker_path: Path, version: str) -> bool:
    if version != SUPPORTED_VERSION:
        raise RuntimeError(
            "Refusing to patch vllm-ascend version %s; expected %s"
            % (version, SUPPORTED_VERSION)
        )

    source = worker_path.read_text(encoding="utf-8")
    if MARKER in source:
        return False
    if source.count(CALL) != 1:
        raise RuntimeError(
            "Expected exactly one _warm_up_atb call in %s; found %d"
            % (worker_path, source.count(CALL))
        )

    backup = worker_path.with_suffix(worker_path.suffix + ".memranker.bak")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    worker_path.write_text(source.replace(CALL, MARKER), encoding="utf-8")
    return True


def restore_worker(worker_path: Path) -> bool:
    backup = worker_path.with_suffix(worker_path.suffix + ".memranker.bak")
    if not backup.exists():
        return False
    worker_path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def installed_worker_path() -> Path:
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm_ascend is not installed")
    return Path(next(iter(spec.submodule_search_locations))) / "worker" / "worker_v1.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    worker_path = installed_worker_path()
    if args.restore:
        changed = restore_worker(worker_path)
        print("[patch] restored %s" % worker_path if changed else "[patch] no backup found")
        return

    version = importlib.metadata.version("vllm-ascend")
    changed = patch_worker(worker_path, version)
    if changed:
        print("[patch] disabled unsupported 310P ATB warm-up in %s" % worker_path)
    else:
        print("[patch] 310P ATB warm-up patch already applied")


if __name__ == "__main__":
    main()
