#!/usr/bin/env python3
"""Report whether an installed vLLM-Ascend stack can load 310P W8A8SC."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import re

from pathlib import Path
from typing import Any


W8A8SC_MODULE = "vllm_ascend._310p.quantization.methods.w8a8sc"
SHARDED_STATE_PATTERN = "model-rank-*-part-*.safetensors"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def read_version(path: str, keys: tuple[str, ...]) -> str:
    version_path = Path(path)
    if not version_path.is_file():
        return "missing"
    text = version_path.read_text(encoding="utf-8", errors="replace")
    for key in keys:
        match = re.search(
            rf"^{re.escape(key)}\s*=\s*\[?([^\]\n:]+)",
            text,
            flags=re.MULTILINE,
        )
        if match:
            return match.group(1).strip()
    return "unknown"


def probe_module(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, ""
    except Exception as exc:  # Runtime-linked imports may fail on old CANN.
        return False, f"{type(exc).__name__}: {exc}"


def inspect_model_layout(model_path: Path | None) -> dict[str, Any]:
    if model_path is None:
        return {}
    if not model_path.is_dir():
        return {
            "model_path": str(model_path),
            "model_path_exists": False,
        }

    root_descriptions = sorted(model_path.glob("quant_model_description*.json"))
    nested_descriptions = sorted(model_path.glob("part*-of-*/quant_model_description*.json"))
    sharded_states = sorted(model_path.glob(SHARDED_STATE_PATTERN))
    atb_parts = sorted(path for path in model_path.glob("part*-of-*") if path.is_dir())
    return {
        "model_path": str(model_path),
        "model_path_exists": True,
        "root_quant_descriptions": [path.name for path in root_descriptions],
        "nested_quant_descriptions": [
            str(path.relative_to(model_path)) for path in nested_descriptions
        ],
        "vllm_sharded_state_files": [path.name for path in sharded_states],
        "atb_part_directories": [path.name for path in atb_parts],
        "model_layout": (
            "vllm_sharded_state"
            if sharded_states and any(path.name == "quant_model_description.json" for path in root_descriptions)
            else "atb_part_layout"
            if atb_parts and nested_descriptions
            else "unknown"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Probe vLLM-Ascend 310P W8A8SC runtime and checkpoint layout."
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--npu-smoke", action="store_true")
    parser.add_argument("--require-w8a8sc", action="store_true")
    args = parser.parse_args()

    import torch
    import torch_npu
    import vllm
    import vllm_ascend

    del vllm_ascend
    w8a8sc_module, w8a8sc_import_error = probe_module(W8A8SC_MODULE)
    compress_op = hasattr(torch_npu, "npu_matmul_compress_dequant")
    model_layout = inspect_model_layout(
        args.model_path.expanduser().resolve() if args.model_path else None
    )
    report: dict[str, Any] = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_npu": package_version("torch-npu"),
        "vllm": getattr(vllm, "__version__", package_version("vllm")),
        "vllm_ascend": package_version("vllm-ascend"),
        "transformers": package_version("transformers"),
        "tokenizers": package_version("tokenizers"),
        "host_driver": read_version(
            "/usr/local/Ascend/driver/version.info",
            ("Version", "version"),
        ),
        "cann_toolkit": read_version(
            "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
            ("toolkit_running_version", "version"),
        ),
        "npu_available": bool(torch.npu.is_available()),
        "npu_count": int(torch.npu.device_count()),
        "w8a8sc_module": W8A8SC_MODULE,
        "w8a8sc_loader_available": w8a8sc_module,
        "w8a8sc_loader_import_error": w8a8sc_import_error,
        "npu_matmul_compress_dequant_available": compress_op,
        **model_layout,
    }

    failures: list[str] = []
    if args.require_w8a8sc:
        if not w8a8sc_module:
            failures.append(
                f"installed vllm-ascend has no {W8A8SC_MODULE} loader"
            )
        if not compress_op:
            failures.append(
                "installed torch-npu has no npu_matmul_compress_dequant operator"
            )
        if args.model_path and model_layout.get("model_layout") != "vllm_sharded_state":
            failures.append(
                "checkpoint is not vLLM W8A8SC sharded_state; the ATB "
                "partN-of-M output must be regenerated with "
                "examples/save_sharded_state_310.py"
            )

    if args.npu_smoke:
        if not report["npu_available"]:
            failures.append("torch reports that the NPU is unavailable")
        else:
            torch.npu.set_device(0)
            value = torch.ones(4, dtype=torch.float16, device="npu:0")
            report["npu_smoke"] = (value + value).cpu().tolist()

    report["failures"] = failures
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("vLLM W8A8SC preflight failed:\n- " + "\n- ".join(failures))


if __name__ == "__main__":
    main()
