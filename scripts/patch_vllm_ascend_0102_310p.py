#!/usr/bin/env python3
"""Apply narrowly scoped vLLM-Ascend 0.10.2rc1 fixes for 310P pooling."""

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path


SUPPORTED_VERSION = "0.10.2rc1"
CALL = "        self._warm_up_atb()"
MARKER = "        pass  # MEMRANKER_310P_SKIP_ATB_WARMUP"
POOLING_CONDITION = "            if not model_config.is_multimodal_model and \\\n"
POOLING_REPLACEMENT = (
    "            if model_config.runner_type != \"pooling\" and \\\n"
    "                not model_config.is_multimodal_model and \\\n"
)
POOLING_MARKER = "model_config.runner_type != \"pooling\""


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


def patch_platform(platform_path: Path, version: str) -> bool:
    if version != SUPPORTED_VERSION:
        raise RuntimeError(
            "Refusing to patch vllm-ascend version %s; expected %s"
            % (version, SUPPORTED_VERSION)
        )

    source = platform_path.read_text(encoding="utf-8")
    if POOLING_MARKER in source:
        return False
    if source.count(POOLING_CONDITION) != 1:
        raise RuntimeError(
            "Expected exactly one Ascend scheduler condition in %s; found %d"
            % (platform_path, source.count(POOLING_CONDITION))
        )

    backup = platform_path.with_suffix(platform_path.suffix + ".memranker.bak")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    platform_path.write_text(
        source.replace(POOLING_CONDITION, POOLING_REPLACEMENT), encoding="utf-8"
    )
    return True


def restore_worker(worker_path: Path) -> bool:
    backup = worker_path.with_suffix(worker_path.suffix + ".memranker.bak")
    if not backup.exists():
        return False
    worker_path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def installed_package_path() -> Path:
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm_ascend is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    package_path = installed_package_path()
    worker_path = package_path / "worker" / "worker_v1.py"
    platform_path = package_path / "platform.py"
    if args.restore:
        restored = [
            path
            for path in (worker_path, platform_path)
            if restore_worker(path)
        ]
        if restored:
            for path in restored:
                print("[patch] restored %s" % path)
        else:
            print("[patch] no backup found")
        return

    version = importlib.metadata.version("vllm-ascend")
    worker_changed = patch_worker(worker_path, version)
    platform_changed = patch_platform(platform_path, version)
    if worker_changed:
        print("[patch] disabled unsupported 310P ATB warm-up in %s" % worker_path)
    else:
        print("[patch] 310P ATB warm-up patch already applied")
    if platform_changed:
        print("[patch] selected the native vLLM scheduler for pooling models in %s" % platform_path)
    else:
        print("[patch] pooling scheduler patch already applied")


if __name__ == "__main__":
    main()
