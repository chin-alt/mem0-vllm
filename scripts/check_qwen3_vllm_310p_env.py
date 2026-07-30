from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import sys

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.check_qwen3_reranker_w8a8_310p import (
    QUANT_DESCRIPTION_NAME,
    load_quant_description,
    validate_quant_description,
)


EXPECTED_VLLM = "0.10.2"
EXPECTED_VLLM_ASCEND = "0.10.2rc1"
QWEN3_POOLING_ARCHITECTURE = "Qwen3ForSequenceClassification"
QWEN3_BASE_ARCHITECTURE = "Qwen3ForCausalLM"


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def read_ascend_version(path: str, key: str) -> str:
    version_file = Path(path)
    if not version_file.is_file():
        return "missing"
    text = version_file.read_text(encoding="utf-8", errors="replace")
    match = re.search(rf"^{re.escape(key)}=\[?([^\]\n:]+)", text, flags=re.MULTILINE)
    return match.group(1).strip() if match else "unknown"


def qwen3_pooling_registry_supported(supported_architectures: set[str]) -> bool:
    """Check the native or adapter-backed Qwen3 classification path.

    vLLM 0.10.2 registers Qwen3ForCausalLM and normalizes the synthetic
    Qwen3ForSequenceClassification override back to that base architecture
    before applying its classification adapter. Newer vLLM releases may
    register the sequence-classification architecture directly.
    """

    return bool(
        {QWEN3_POOLING_ARCHITECTURE, QWEN3_BASE_ARCHITECTURE}
        & supported_architectures
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the 310P vLLM Qwen3 reranker runtime.")
    parser.add_argument("--model-path", required=True)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Missing model config: {model_path / 'config.json'}")

    import torch
    import torch_npu  # noqa: F401
    import vllm
    import vllm_ascend  # noqa: F401
    from vllm import ModelRegistry

    quantized = (model_path / QUANT_DESCRIPTION_NAME).is_file()
    supported_architectures = set(ModelRegistry.get_supported_archs())
    versions = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_npu": package_version("torch-npu"),
        "vllm": vllm.__version__,
        "vllm_ascend": package_version("vllm-ascend"),
        "npu_available": torch.npu.is_available(),
        "npu_count": torch.npu.device_count(),
        "qwen3_pooling_architecture": QWEN3_POOLING_ARCHITECTURE,
        "qwen3_pooling_directly_registered": QWEN3_POOLING_ARCHITECTURE
        in supported_architectures,
        "qwen3_base_architecture_registered": QWEN3_BASE_ARCHITECTURE
        in supported_architectures,
        "qwen3_pooling_supported": qwen3_pooling_registry_supported(
            supported_architectures
        ),
        "quantized_model": quantized,
        "host_driver": read_ascend_version(
            "/usr/local/Ascend/driver/version.info", "Version"
        ),
        "cann_toolkit": read_ascend_version(
            "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
            "toolkit_running_version",
        ),
    }
    print(json.dumps(versions, ensure_ascii=False, indent=2))

    failures: list[str] = []
    warnings: list[str] = []
    if not str(versions["vllm"]).startswith(EXPECTED_VLLM):
        failures.append(f"expected vllm=={EXPECTED_VLLM}, got {versions['vllm']}")
    if not str(versions["vllm_ascend"]).startswith(EXPECTED_VLLM_ASCEND):
        failures.append(
            f"expected vllm-ascend=={EXPECTED_VLLM_ASCEND}, got {versions['vllm_ascend']}"
        )
    if not versions["npu_available"]:
        failures.append("torch reports that the NPU is unavailable")
    if not versions["qwen3_pooling_supported"]:
        failures.append(
            "vLLM registry contains neither the direct pooling architecture "
            f"{QWEN3_POOLING_ARCHITECTURE} nor its adapter base "
            f"{QWEN3_BASE_ARCHITECTURE}"
        )
    if not str(versions["host_driver"]).lower().startswith("24.1.rc2"):
        failures.append(
            "this image was selected for host driver 24.1.RC2.x, got "
            f"{versions['host_driver']}"
        )
    if versions["cann_toolkit"] not in {"missing", "unknown"} and not str(
        versions["cann_toolkit"]
    ).startswith("8.2"):
        failures.append(
            f"expected CANN 8.2 container userspace, got {versions['cann_toolkit']}"
        )

    if quantized:
        _path, description = load_quant_description(model_path)
        quant_failures, quant_warnings, _counts = validate_quant_description(description)
        failures.extend(quant_failures)
        warnings.extend(quant_warnings)
        for operator in ("npu_quantize", "npu_quant_matmul", "npu_format_cast"):
            if not hasattr(torch_npu, operator):
                failures.append(f"torch_npu has no required static-W8A8 operator {operator}")

    for warning in warnings:
        print("WARNING:", warning)
    if failures:
        raise SystemExit("Qwen3 vLLM preflight failed:\n- " + "\n- ".join(failures))

    torch.npu.set_device(0)
    value = torch.ones(4, dtype=torch.float16, device="npu:0")
    print("npu_smoke:", (value + value).cpu().tolist())
    print("Qwen3 vLLM 310P preflight: PASS")


if __name__ == "__main__":
    main()
