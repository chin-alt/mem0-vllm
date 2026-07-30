#!/usr/bin/env python3
"""Run a ModelSlim entry point without probing the unavailable NPU driver."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path
from typing import Any, Mapping


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


def clone_shared_tensors_for_safetensors(
    tensors: Mapping[str, Any],
) -> tuple[dict[str, Any], list[tuple[str, list[str]]]]:
    """Clone tensor aliases that the safetensors dict API cannot serialize.

    Qwen3 commonly ties ``lm_head.weight`` to ``embed_tokens.weight``. The
    legacy ModelSlim saver passes both names to ``safetensors.torch.save_file``
    instead of using its model-aware API, so save_file rejects the shared
    storage after an otherwise successful calibration. Keep the embedding
    tensor and clone only its aliases in the serialization copy; the live model
    remains tied and numerically unchanged.
    """

    import torch

    output = dict(tensors)
    storage_groups: dict[tuple[str, int, int], list[str]] = {}
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor) or tensor.device.type == "meta":
            continue
        storage = tensor.untyped_storage()
        data_ptr = storage.data_ptr()
        storage_bytes = storage.nbytes()
        if data_ptr == 0 or storage_bytes == 0:
            continue
        key = (str(tensor.device), data_ptr, storage_bytes)
        storage_groups.setdefault(key, []).append(name)

    cloned_groups: list[tuple[str, list[str]]] = []
    for names in storage_groups.values():
        if len(names) < 2:
            continue
        ordered = sorted(
            names,
            key=lambda name: (not name.endswith("embed_tokens.weight"), name),
        )
        keep_name = ordered[0]
        clone_names = ordered[1:]
        for name in clone_names:
            output[name] = output[name].clone()
        cloned_groups.append((keep_name, clone_names))

    return output, cloned_groups


def install_safetensors_shared_storage_guard() -> None:
    """Make the legacy ModelSlim dict saver handle tied model weights."""

    import safetensors.torch as safetensors_torch

    original_save_file = safetensors_torch.save_file
    if getattr(original_save_file, "_memranker_shared_storage_guard", False):
        return

    def save_file_without_aliases(
        tensors: Mapping[str, Any],
        filename: str,
        metadata: dict[str, str] | None = None,
    ) -> None:
        serializable, cloned_groups = clone_shared_tensors_for_safetensors(tensors)
        for keep_name, clone_names in cloned_groups:
            print(
                "[cpu] safetensors shared storage: "
                f"keep={keep_name} clone={','.join(clone_names)}",
                flush=True,
            )
        original_save_file(serializable, filename, metadata=metadata)

    save_file_without_aliases._memranker_shared_storage_guard = True  # type: ignore[attr-defined]
    safetensors_torch.save_file = save_file_without_aliases


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
    install_safetensors_shared_storage_guard()
    print("[cpu] Transformers torch_npu probe disabled", flush=True)
    print("[cpu] safetensors shared-storage guard installed", flush=True)

    sys.path.insert(0, str(target.parent))
    sys.argv = [str(target), *target_args]
    runpy.run_path(str(target), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
