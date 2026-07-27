#!/usr/bin/env python3
"""Verify the fixed HDK 24.1.RC2.x vLLM-Ascend environment."""

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Dict, List


EXPECTED_PACKAGES: Dict[str, str] = {
    "torch": "2.5.1",
    "torch-npu": "2.5.1",
    "vllm": "0.8.5.post1",
    "vllm-ascend": "0.8.5rc1",
    "transformers": "4.51.3",
    "tokenizers": "0.21.1",
    "protobuf": "4.25.8",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check HDK 24.1.RC2.x, CANN 8.1.RC1, torch-npu, and vLLM-Ascend."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        help="Optional local model directory for an end-to-end vLLM generation test.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    return parser.parse_args()


def normalized_version(version: str) -> str:
    return version.split("+", 1)[0]


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def verify_system_versions(failures: List[str]) -> None:
    driver_file = Path("/usr/local/Ascend/driver/version.info")
    driver_text = read_text(driver_file)
    if driver_text:
        driver_line = next(
            (line for line in driver_text.splitlines() if line.startswith("Version=")),
            "Version=unknown",
        )
        print("driver:", driver_line.split("=", 1)[-1])
        if "24.1.rc2" not in driver_text.lower():
            failures.append("driver is not in the expected 24.1.RC2.x line")
    else:
        failures.append("cannot read /usr/local/Ascend/driver/version.info")

    cann_file = Path("/usr/local/Ascend/ascend-toolkit/latest/version.cfg")
    cann_text = read_text(cann_file)
    if cann_text:
        cann_lines = [line for line in cann_text.splitlines() if "running_version" in line]
        print("CANN:", cann_lines[0] if cann_lines else "version.cfg found")
        if "8.1.rc1" not in cann_text.lower():
            failures.append("CANN is not 8.1.RC1")
    else:
        failures.append("cannot read CANN latest/version.cfg")

    nnal_env = Path("/usr/local/Ascend/nnal/atb/set_env.sh")
    if nnal_env.is_file():
        print("NNAL/ATB: set_env.sh found")
    else:
        failures.append("NNAL/ATB set_env.sh was not found")


def verify_packages(failures: List[str]) -> None:
    print("python:", sys.version.split()[0])
    if not ((3, 9) <= sys.version_info[:2] < (3, 12)):
        failures.append("Python must be 3.9, 3.10, or 3.11")

    for package, expected in EXPECTED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            failures.append("%s is not installed" % package)
            continue
        print("%s: %s" % (package, actual))
        if normalized_version(actual) != expected:
            failures.append("%s must be %s, found %s" % (package, expected, actual))

    modules = ("vllm", "vllm_ascend", "transformers", "tokenizers", "google.protobuf")
    for module in modules:
        try:
            __import__(module)
            print("import %s: ok" % module)
        except Exception as exc:
            failures.append("import %s failed: %s" % (module, exc))


def verify_npu(device: int, failures: List[str]) -> None:
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as exc:
        failures.append("torch/torch_npu import failed: %s" % exc)
        return

    print("npu device_count:", torch.npu.device_count())
    print("npu available:", torch.npu.is_available())
    if not torch.npu.is_available():
        failures.append("torch.npu.is_available() is false")
        return
    if device < 0 or device >= torch.npu.device_count():
        failures.append("requested NPU device %d is not visible" % device)
        return

    try:
        torch.npu.set_device(device)
        left = torch.arange(4, dtype=torch.float16, device="npu")
        result = (left + left).cpu()
        expected = torch.tensor([0.0, 2.0, 4.0, 6.0], dtype=torch.float16)
        if not torch.equal(result, expected):
            failures.append("NPU tensor result is incorrect: %s" % result)
            return
        print("npu tensor test:", result.tolist())
    except Exception as exc:
        failures.append("NPU tensor test failed: %s" % exc)


def verify_vllm_model(args: argparse.Namespace, failures: List[str]) -> None:
    if args.model_path is None:
        print("vLLM model test: skipped (pass --model-path to enable)")
        return
    model_path = args.model_path.expanduser().resolve()
    if not (model_path / "config.json").is_file():
        failures.append("model path has no config.json: %s" % model_path)
        return

    os.environ.setdefault("VLLM_USE_V1", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    try:
        from vllm import LLM, SamplingParams

        print("vLLM model test: loading", model_path)
        llm = LLM(
            model=str(model_path),
            dtype="float16",
            enforce_eager=True,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_model_len,
            max_num_seqs=1,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        params = SamplingParams(temperature=0, max_tokens=1, logprobs=5)
        outputs = llm.generate(["The capital of China is"], params, use_tqdm=False)
        if len(outputs) != 1 or not outputs[0].outputs:
            failures.append("vLLM returned no generation output")
            return
        generated = outputs[0].outputs[0].text
        print("vLLM generation test:", repr(generated))
    except Exception as exc:
        failures.append("vLLM model test failed: %s" % exc)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("VLLM_USE_V1", "0")
    failures: List[str] = []
    verify_system_versions(failures)
    verify_packages(failures)
    verify_npu(args.device, failures)
    if not failures:
        verify_vllm_model(args, failures)

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)
    print("\nPASS: Ascend vLLM environment is ready")


if __name__ == "__main__":
    main()
