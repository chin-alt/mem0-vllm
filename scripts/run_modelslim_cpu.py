#!/usr/bin/env python3
"""Run a ModelSlim entry point without probing the unavailable NPU driver."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any


def disable_transformers_npu_probe() -> None:
    """Make Transformers treat torch_npu as unavailable in this CPU process.

    Transformers 4.55.2 imports torch_npu whenever the distribution is visible,
    even when all model tensors use CPU. In the containerized CPU workflow the
    package comes from the CANN image but the host driver is intentionally not
    mounted, so that probe raises while loading libascend_hal.so.
    """

    import transformers.utils as transformers_utils
    from transformers.utils import import_utils

    def npu_unavailable(*_args: Any, **_kwargs: Any) -> bool:
        return False

    # Some Transformers modules import the public re-export from utils while
    # others import the implementation directly. Patch both before loading any
    # PreTrainedModel/modeling module.
    import_utils.is_torch_npu_available = npu_unavailable
    transformers_utils.is_torch_npu_available = npu_unavailable


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: run_modelslim_cpu.py TARGET_SCRIPT [TARGET_ARGS...]", file=sys.stderr)
        return 2

    target = Path(sys.argv[1]).resolve()
    if not target.is_file():
        print(f"[missing] ModelSlim target script: {target}", file=sys.stderr)
        return 2

    target_args = sys.argv[2:]
    disable_transformers_npu_probe()
    print("[cpu] Transformers torch_npu probe disabled", flush=True)

    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *target_args]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
