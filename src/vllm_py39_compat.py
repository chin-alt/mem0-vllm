from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Iterable


SUPPORTED_VLLM_VERSION = "0.11.0"

# vLLM 0.11.0 declares Python 3.9 support and ships an abi3 wheel, but these
# three source files contain a few PEP 604 annotations that are evaluated by
# Python 3.9 at import time. The replacements use typing names already imported
# by the affected modules.
PATCHES = {
    Path("model_executor/models/registry.py"): (
        (
            "module_hash: str) -> _ModelInfo | None:",
            "module_hash: str) -> Optional[_ModelInfo]:",
            1,
        ),
    ),
    Path("model_executor/models/qwen3_vl.py"): (
        (
            "indices: list[int] | torch.Tensor,",
            "indices: Union[list[int], torch.Tensor],",
            1,
        ),
    ),
    Path("model_executor/models/step3_vl.py"): (
        (
            "ImageWithPatches = tuple[Image.Image, list[Image.Image], list[int] | None]",
            "ImageWithPatches = tuple[Image.Image, list[Image.Image], Optional[list[int]]]",
            1,
        ),
        (
            ") -> tuple[Image.Image, list[Image.Image], list[bool] | None]:",
            ") -> tuple[Image.Image, list[Image.Image], Optional[list[bool]]]:",
            1,
        ),
        (
            "patch_newline_mask: list[bool] | None,",
            "patch_newline_mask: Optional[list[bool]],",
            1,
        ),
    ),
}


def installed_vllm_root() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError(
            "vLLM is not installed in this Python environment. Install "
            "vllm==0.11.0 first, then rerun this compatibility patch."
        )
    return Path(next(iter(spec.submodule_search_locations))).resolve()


def installed_vllm_version() -> str:
    try:
        return importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("Could not read the installed vLLM version.") from exc


def _patch_file(path: Path, replacements: Iterable[tuple[str, str, int]]) -> bool:
    if not path.is_file():
        raise RuntimeError(f"Expected vLLM source file does not exist: {path}")

    original = path.read_text(encoding="utf-8")
    updated = original
    for old, new, expected_count in replacements:
        old_count = updated.count(old)
        new_count = updated.count(new)
        if old_count == expected_count:
            updated = updated.replace(old, new)
        elif old_count == 0 and new_count == expected_count:
            continue
        else:
            raise RuntimeError(
                f"Unexpected vLLM source layout in {path}: expected "
                f"{expected_count} occurrence(s) of {old!r}, found {old_count}; "
                f"already-patched count is {new_count}. Refusing a blind edit."
            )

    compile(updated, str(path), "exec")
    if updated == original:
        return False

    backup = path.with_name(path.name + ".memranker-py39.orig")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = path.with_name(path.name + ".memranker-py39.tmp")
    temporary.write_text(updated, encoding="utf-8")
    shutil.copymode(path, temporary)
    temporary.replace(path)
    return True


def patch_vllm_011_for_python39(
    vllm_root: Path | None = None,
    *,
    force: bool = False,
) -> list[Path]:
    if sys.version_info[:2] != (3, 9) and not force:
        return []

    if vllm_root is None:
        version = installed_vllm_version()
        if version.split("+", 1)[0] != SUPPORTED_VLLM_VERSION:
            return []
        vllm_root = installed_vllm_root()
    else:
        vllm_root = vllm_root.resolve()

    changed = []
    for relative_path, replacements in PATCHES.items():
        path = vllm_root / relative_path
        if _patch_file(path, replacements):
            changed.append(path)
    return changed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patch the vLLM 0.11.0 Python 3.9 annotation regression."
    )
    parser.add_argument(
        "--vllm_root",
        type=Path,
        help="Optional vLLM package root for verification/testing.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    changed = patch_vllm_011_for_python39(
        args.vllm_root,
        force=args.vllm_root is not None,
    )
    if args.quiet:
        return
    if sys.version_info[:2] != (3, 9) and args.vllm_root is None:
        print(f"[vllm-py39] skipped on Python {sys.version.split()[0]}")
    elif changed:
        print(f"[vllm-py39] patched {len(changed)} vLLM source file(s)")
        for path in changed:
            print(f"[vllm-py39] patched: {path}")
    else:
        print("[vllm-py39] compatibility patch already applied")


if __name__ == "__main__":
    main()
