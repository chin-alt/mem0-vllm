from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from tqdm.auto import tqdm


logger = logging.getLogger(__name__)

DEFAULT_CMTEB_RETRIEVAL_DATASETS = [
    "T2Retrieval",
    "MMarcoRetrieval",
    "DuRetrieval",
    "CovidRetrieval",
    "CmedqaRetrieval",
    "EcomRetrieval",
    "MedicalRetrieval",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download C-MTEB Retrieval datasets from Hugging Face."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_CMTEB_RETRIEVAL_DATASETS,
        help="Dataset names, either short names like T2Retrieval or full repo ids like C-MTEB/T2Retrieval.",
    )
    parser.add_argument("--output_dir", default="data/cmteb_r/raw")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for gated/private mirrors.")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--allow_patterns",
        nargs="*",
        default=None,
        help="Optional snapshot_download allow patterns. By default the full dataset repo is downloaded.",
    )
    return parser.parse_args()


def repo_id_from_name(name: str) -> str:
    return name if "/" in name else f"C-MTEB/{name}"


def local_name_from_repo(repo_id: str) -> str:
    return repo_id.split("/", 1)[-1]


def snapshot_dataset(repo_id: str, local_dir: Path, args: argparse.Namespace) -> str:
    from huggingface_hub import snapshot_download

    kwargs = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "revision": args.revision,
        "local_dir": str(local_dir),
        "token": args.token,
        "local_files_only": args.local_files_only,
    }
    if args.allow_patterns:
        kwargs["allow_patterns"] = args.allow_patterns
    try:
        return snapshot_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        return snapshot_download(**kwargs)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "output_dir": str(output_dir),
        "revision": args.revision,
        "datasets": [],
    }
    for name in tqdm(args.datasets, desc="Downloading C-MTEB/R", unit="dataset", dynamic_ncols=True, ascii=True):
        repo_id = repo_id_from_name(name)
        local_dir = output_dir / local_name_from_repo(repo_id)
        local_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading %s -> %s", repo_id, local_dir)
        snapshot_path = snapshot_dataset(repo_id, local_dir, args)
        manifest["datasets"].append(
            {
                "name": local_name_from_repo(repo_id),
                "repo_id": repo_id,
                "local_dir": str(local_dir),
                "snapshot_path": snapshot_path,
            }
        )

    manifest_path = output_dir / "download_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote manifest to %s", manifest_path)


if __name__ == "__main__":
    main()
