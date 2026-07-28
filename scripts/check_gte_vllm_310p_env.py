from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re

from pathlib import Path


EXPECTED_VLLM = "0.10.2"
EXPECTED_VLLM_ASCEND = "0.10.2rc1"
GTE_ARCHITECTURE = "GteNewForSequenceClassification"


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the 310P vLLM GTE runtime.")
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    if not (model_path / "config.json").is_file():
        raise SystemExit(f"Missing model config: {model_path / 'config.json'}")

    import torch
    import torch_npu  # noqa: F401
    import vllm
    import vllm_ascend  # noqa: F401
    from vllm import ModelRegistry

    versions = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "torch_npu": package_version("torch-npu"),
        "vllm": vllm.__version__,
        "vllm_ascend": package_version("vllm-ascend"),
        "npu_available": torch.npu.is_available(),
        "npu_count": torch.npu.device_count(),
        "gte_architecture": GTE_ARCHITECTURE,
        "gte_supported": GTE_ARCHITECTURE in ModelRegistry.get_supported_archs(),
        "host_driver": read_ascend_version(
            "/usr/local/Ascend/driver/version.info", "Version"
        ),
        "cann_toolkit": read_ascend_version(
            "/usr/local/Ascend/ascend-toolkit/latest/version.cfg",
            "toolkit_running_version",
        ),
    }
    print(json.dumps(versions, ensure_ascii=False, indent=2))

    failures = []
    if not str(versions["vllm"]).startswith(EXPECTED_VLLM):
        failures.append(f"expected vllm=={EXPECTED_VLLM}, got {versions['vllm']}")
    if not str(versions["vllm_ascend"]).startswith(EXPECTED_VLLM_ASCEND):
        failures.append(
            f"expected vllm-ascend=={EXPECTED_VLLM_ASCEND}, got {versions['vllm_ascend']}"
        )
    if not versions["npu_available"]:
        failures.append("torch reports that the NPU is unavailable")
    if not versions["gte_supported"]:
        failures.append(f"vLLM registry does not contain {GTE_ARCHITECTURE}")
    if not str(versions["host_driver"]).lower().startswith("24.1.rc2"):
        failures.append(
            "this image was selected for host driver 24.1.RC2.x, got "
            f"{versions['host_driver']}"
        )
    if versions["cann_toolkit"] not in {"missing", "unknown"} and not str(
        versions["cann_toolkit"]
    ).startswith("8.2"):
        failures.append(f"expected CANN 8.2 container userspace, got {versions['cann_toolkit']}")
    if failures:
        raise SystemExit("GTE vLLM preflight failed:\n- " + "\n- ".join(failures))

    torch.npu.set_device(0)
    value = torch.ones(4, dtype=torch.float16, device="npu:0")
    print("npu_smoke:", (value + value).cpu().tolist())
    print("GTE vLLM preflight: PASS")


if __name__ == "__main__":
    main()
