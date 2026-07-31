#!/usr/bin/env python3
"""Run a ModelSlim entry point on an available Ascend NPU."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

from run_modelslim_cpu import install_safetensors_shared_storage_guard


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_modelslim_npu.py TARGET_SCRIPT [TARGET_ARGS...]", file=sys.stderr)
        return 2

    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        print(f"[missing] ModelSlim target script: {target}", file=sys.stderr)
        return 2

    import torch
    import torch_npu  # noqa: F401

    if not torch.npu.is_available():
        print("[missing] torch.npu.is_available() is false", file=sys.stderr)
        return 3

    install_safetensors_shared_storage_guard()
    print(
        "[npu] device_count=%d device=%s"
        % (torch.npu.device_count(), torch.npu.get_device_name(0)),
        flush=True,
    )
    print("[npu] safetensors shared-storage guard installed", flush=True)

    target_args = sys.argv[2:]
    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *target_args]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
