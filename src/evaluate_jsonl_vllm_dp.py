from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import load_examples, read_json_records, write_jsonl  # noqa: E402
from evaluate_jsonl_vllm import compute_dynamic_beta_metrics, write_csv  # noqa: E402
from metrics import add_group_ranks, compute_all_metrics  # noqa: E402
from modeling import DEFAULT_MODEL_NAME  # noqa: E402


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Data-parallel vLLM JSONL evaluation. It splits the test file by query group, "
            "runs one single-GPU vLLM worker per shard, then merges predictions and recomputes metrics."
        )
    )
    parser.add_argument("--test_file", required=True)
    parser.add_argument("--model_path", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--output_dir", default="outputs/cmteb_r_vllm_dp_eval")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--relevance_threshold", type=float, default=0.7)
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="float16")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.80)
    parser.add_argument("--max_num_batched_tokens", type=int, default=8192)
    parser.add_argument("--max_num_seqs", type=int, default=64)
    parser.add_argument("--enable_prefix_caching", action="store_true", default=True)
    parser.add_argument("--no_enable_prefix_caching", dest="enable_prefix_caching", action="store_false")
    parser.add_argument("--sort_by_length", action="store_true", default=True)
    parser.add_argument("--no_sort_by_length", dest="sort_by_length", action="store_false")
    parser.add_argument("--sort_descending", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--expected_fbeta_betas", type=float, nargs="+", default=[0.2, 0.3, 0.5, 0.7, 1.0])
    parser.add_argument(
        "--devices",
        default="auto",
        help="Comma-separated physical GPU ids, e.g. 0,1,2,3. auto uses CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--num_shards",
        default="auto",
        help="Number of data-parallel shards. auto equals the number of selected devices.",
    )
    parser.add_argument("--python_bin", default=sys.executable)
    return parser.parse_args()


def parse_devices(devices_arg: str) -> list[str]:
    if devices_arg != "auto":
        devices = [device.strip() for device in devices_arg.split(",") if device.strip()]
        if not devices:
            raise ValueError("--devices was provided but no valid device ids were found")
        return devices

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible and visible != "-1":
        devices = [device.strip() for device in visible.split(",") if device.strip()]
        if devices:
            return devices

    return ["0"]


def parse_num_shards(num_shards_arg: str, device_count: int) -> int:
    if num_shards_arg == "auto":
        return device_count
    value = int(num_shards_arg)
    if value < 1:
        raise ValueError("--num_shards must be >= 1")
    return value


def split_records_by_group(test_file: str, num_shards: int) -> list[list[dict[str, Any]]]:
    raw_records = read_json_records(test_file)
    examples = load_examples(test_file)
    groups: dict[str, list[int]] = defaultdict(list)
    for ex in examples:
        if ex.source_index is None:
            continue
        groups[ex.group_key].append(ex.source_index)
    if not groups:
        raise ValueError(f"No valid query groups found in {test_file}")

    shard_indices: list[list[int]] = [[] for _ in range(num_shards)]
    shard_sizes = [0] * num_shards
    for _group_key, indices in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        shard_id = min(range(num_shards), key=lambda idx: (shard_sizes[idx], idx))
        shard_indices[shard_id].extend(indices)
        shard_sizes[shard_id] += len(indices)

    shards: list[list[dict[str, Any]]] = []
    for indices in shard_indices:
        indices.sort()
        shards.append([raw_records[idx] for idx in indices])
    return shards


def build_worker_command(args: argparse.Namespace, shard_input: Path, shard_output: Path) -> list[str]:
    cmd = [
        args.python_bin,
        str(PROJECT_ROOT / "src" / "evaluate_jsonl_vllm.py"),
        "--test_file",
        str(shard_input),
        "--model_path",
        args.model_path,
        "--output_dir",
        str(shard_output),
        "--max_length",
        str(args.max_length),
        "--batch_size",
        str(args.batch_size),
        "--relevance_threshold",
        str(args.relevance_threshold),
        "--dtype",
        args.dtype,
        "--gpu_memory_utilization",
        str(args.gpu_memory_utilization),
        "--tensor_parallel_size",
        "1",
        "--max_num_batched_tokens",
        str(args.max_num_batched_tokens),
        "--max_num_seqs",
        str(args.max_num_seqs),
        "--expected_fbeta_betas",
        *[str(beta) for beta in args.expected_fbeta_betas],
    ]
    if args.instruction.strip():
        cmd.extend(["--instruction", args.instruction])
    if args.enable_prefix_caching:
        cmd.append("--enable_prefix_caching")
    else:
        cmd.append("--no_enable_prefix_caching")
    if args.sort_by_length:
        cmd.append("--sort_by_length")
    else:
        cmd.append("--no_sort_by_length")
    if args.sort_descending:
        cmd.append("--sort_descending")
    if args.local_files_only:
        cmd.append("--local_files_only")
    return cmd


def tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def run_workers(args: argparse.Namespace, devices: list[str], shards: list[list[dict[str, Any]]], output_dir: Path) -> float:
    shard_root = output_dir / "_dp_shards"
    shard_root.mkdir(parents=True, exist_ok=True)
    processes: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for shard_id, records in enumerate(shards):
        if not records:
            logger.warning("Skipping empty shard %d", shard_id)
            continue
        device = devices[shard_id % len(devices)]
        shard_dir = shard_root / f"shard_{shard_id:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        shard_input = shard_dir / "input.jsonl"
        shard_output = shard_dir / "output"
        log_path = shard_dir / "worker.log"
        write_jsonl(shard_input, records)
        cmd = build_worker_command(args, shard_input, shard_output)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = device
        env["MEMRANKER_DP_SHARD_ID"] = str(shard_id)
        env["MEMRANKER_DP_DEVICE"] = device
        logger.info("Launching shard %d on GPU %s with %d records", shard_id, device, len(records))
        log_f = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append(
            {
                "shard_id": shard_id,
                "device": device,
                "records": len(records),
                "process": process,
                "log_f": log_f,
                "log_path": log_path,
                "output_dir": shard_output,
            }
        )

    failed: list[dict[str, Any]] = []
    for item in processes:
        return_code = item["process"].wait()
        item["log_f"].close()
        if return_code != 0:
            item["return_code"] = return_code
            failed.append(item)
        else:
            logger.info("Shard %d finished on GPU %s", item["shard_id"], item["device"])

    if failed:
        messages = []
        for item in failed:
            messages.append(
                f"shard={item['shard_id']} device={item['device']} return_code={item['return_code']}\n"
                f"{tail_text(item['log_path'])}"
            )
        raise RuntimeError("One or more data-parallel vLLM workers failed:\n\n" + "\n\n".join(messages))

    return time.perf_counter() - start_time


def read_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def merge_outputs(
    args: argparse.Namespace,
    devices: list[str],
    output_dir: Path,
    wall_time_seconds: float,
) -> None:
    shard_dirs = sorted((output_dir / "_dp_shards").glob("shard_*/output"))
    rows: list[dict[str, Any]] = []
    shard_metrics: list[dict[str, Any]] = []
    for shard_dir in shard_dirs:
        pred_path = shard_dir / "predictions.jsonl"
        if not pred_path.exists():
            continue
        rows.extend(read_json_records(pred_path))
        shard_overall = read_json_file(shard_dir / "overall_metrics.json")
        shard_metrics.append(
            {
                "shard": shard_dir.parent.name,
                "output_dir": str(shard_dir),
                **shard_overall,
            }
        )
    if not rows:
        raise ValueError(f"No shard predictions found under {output_dir / '_dp_shards'}")

    rows = add_group_ranks(rows, query_key="group_key")
    overall, per_query = compute_all_metrics(
        rows,
        query_key="group_key",
        relevance_threshold=args.relevance_threshold,
    )
    beta_per_query, beta_summary, beta_overall = compute_dynamic_beta_metrics(
        rows,
        betas=args.expected_fbeta_betas,
        relevance_threshold=args.relevance_threshold,
    )
    sum_shard_score_time = sum(float(row.get("score_time_seconds", 0.0)) for row in shard_metrics)
    overall.update(
        {
            "backend": "vllm_data_parallel",
            "vllm_runner": "pooling",
            "model_path": args.model_path,
            "test_file": args.test_file,
            "max_length": args.max_length,
            "batch_size_per_worker": args.batch_size,
            "dtype": args.dtype,
            "tensor_parallel_size_per_worker": 1,
            "data_parallel_shards": len(shard_metrics),
            "devices": devices,
            "max_num_batched_tokens": args.max_num_batched_tokens,
            "max_num_seqs": args.max_num_seqs,
            "sort_by_length": args.sort_by_length,
            "sort_descending": args.sort_descending,
            "local_files_only": args.local_files_only,
            "expected_fbeta_betas": args.expected_fbeta_betas,
            "wall_time_seconds": float(wall_time_seconds),
            "sum_shard_score_time_seconds": float(sum_shard_score_time),
            "examples_per_second": len(rows) / wall_time_seconds if wall_time_seconds > 0 else 0.0,
        }
    )
    overall.update(beta_overall)

    (output_dir / "overall_metrics.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(output_dir / "per_query_metrics.jsonl", per_query)
    write_jsonl(output_dir / "predictions.jsonl", rows)
    write_jsonl(output_dir / "beta_f1_per_query.jsonl", beta_per_query)
    write_jsonl(output_dir / "shard_metrics.jsonl", shard_metrics)
    write_csv(output_dir / "beta_f1_summary.csv", beta_summary)
    (output_dir / "beta_f1_summary.json").write_text(
        json.dumps(beta_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Merged %d predictions from %d shards into %s", len(rows), len(shard_metrics), output_dir)
    print(json.dumps(overall, ensure_ascii=False, indent=2))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    devices = parse_devices(args.devices)
    num_shards = parse_num_shards(args.num_shards, len(devices))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_shards > len(devices):
        raise ValueError(
            f"--num_shards={num_shards} is larger than selected device count={len(devices)}. "
            "Use at most one worker per GPU to avoid loading multiple model copies on the same card."
        )
    devices = devices[:num_shards]
    shards = split_records_by_group(args.test_file, num_shards)
    logger.info(
        "Prepared %d data-parallel shards from %s: %s",
        len(shards),
        args.test_file,
        [len(shard) for shard in shards],
    )
    wall_time_seconds = run_workers(args, devices, shards, output_dir)
    merge_outputs(args, devices, output_dir, wall_time_seconds)


if __name__ == "__main__":
    main()
