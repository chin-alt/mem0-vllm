from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import math
import sys
import threading
import time

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional
from urllib import error, parse, request

from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data import write_jsonl  # noqa: E402
from evaluate_business import (  # noqa: E402
    DEFAULT_BUSINESS_INSTRUCTION,
    attach_scores_and_ranks,
    build_scoring_inputs,
    compute_business_metrics,
    load_ground_truth,
    load_recall_results,
    write_summary_csv,
    write_summary_xlsx,
)


logger = logging.getLogger(__name__)

QWEN3_RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and "
    'the Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN3_RERANKER_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def format_qwen3_reranker_prompt(query: str, document: str, instruction: str) -> str:
    return (
        f"{QWEN3_RERANKER_PREFIX}<Instruct>: {instruction}\n\n"
        f"<Query>: {query}\n\n<Document>: {document}{QWEN3_RERANKER_SUFFIX}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate business recall data through a local MindIE OpenAI-compatible "
            "completion endpoint. Model inference stays on the local Ascend NPU."
        )
    )
    parser.add_argument("--gt_file", required=True)
    parser.add_argument("--recall_file", required=True)
    parser.add_argument("--model_name", default="qwen3-reranker-4b")
    parser.add_argument("--model_path", default="", help="Metadata only; MindIE already owns model loading.")
    parser.add_argument("--output_dir", default="outputs/business_eval_mindie")
    parser.add_argument("--instruction", default=DEFAULT_BUSINESS_INSTRUCTION)
    parser.add_argument("--gt_query_col", default="query")
    parser.add_argument("--gt_doc_id_col", default="PageId")
    parser.add_argument("--gt_sheet", default=None)
    parser.add_argument("--recall_id_key", default="id")
    parser.add_argument("--recall_text_key", default="text")
    parser.add_argument("--max_length", type=int, default=8192, help="Configured MindIE input-token limit.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--max_request_chars",
        type=int,
        default=32000,
        help=(
            "Maximum characters per prompt. In list request mode this is also the "
            "maximum total prompt characters in one HTTP request."
        ),
    )
    parser.add_argument("--top_logprobs", type=int, default=5, choices=range(1, 6))
    parser.add_argument(
        "--request_mode",
        choices=("concurrent", "list"),
        default="concurrent",
        help=(
            "concurrent sends one prompt per HTTP request and lets MindIE dynamically batch "
            "them; list uses a prompt list and requires a MindIE version that supports it."
        ),
    )
    parser.add_argument("--missing_logprob_floor", type=float, default=-20.0)
    parser.add_argument("--request_timeout", type=float, default=600.0)
    parser.add_argument("--request_retries", type=int, default=2)
    parser.add_argument(
        "--endpoint",
        default="http://127.0.0.1:1025/v1/completions",
        help="Local MindIE v1/completions endpoint.",
    )
    parser.add_argument("--api_key", default="")
    parser.add_argument(
        "--allow_remote_endpoint",
        action="store_true",
        help="Allow a non-loopback MindIE endpoint. Off by default to guarantee local inference.",
    )
    parser.add_argument(
        "--extra_request_json",
        default="",
        help="Optional JSON object merged into every MindIE request.",
    )
    parser.add_argument("--sort_by_length", action="store_true", default=True)
    parser.add_argument("--no_sort_by_length", dest="sort_by_length", action="store_false")
    parser.add_argument("--sort_descending", action="store_true")
    parser.add_argument("--top_k_list", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--expected_fbeta_beta", type=float, default=0.3)
    parser.add_argument("--save_doc_text", action="store_true")
    return parser.parse_args()


def ensure_local_endpoint(endpoint: str, allow_remote: bool) -> None:
    parsed = parse.urlparse(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"Invalid MindIE endpoint: {endpoint!r}")
    if allow_remote:
        return
    hostname = parsed.hostname.lower()
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise ValueError(
            "MindIE endpoint must be localhost/loopback so inference remains on this machine. "
            "Pass --allow_remote_endpoint only for an explicitly approved private deployment."
        )


def parse_json_object(text: str, option_name: str) -> dict[str, Any]:
    if not text:
        return {}
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{option_name} must be valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{option_name} must be a JSON object")
    return value


def parse_mindie_body(body: bytes) -> dict[str, Any]:
    text = body.decode("utf-8", errors="replace").strip()
    if not text:
        raise ValueError("MindIE returned an empty response")
    if text.startswith("data:"):
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload and payload != "[DONE]":
                value = json.loads(payload)
                if isinstance(value, dict) and value.get("choices"):
                    return value
        raise ValueError("MindIE SSE response contained no choices")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("MindIE response must be a JSON object")
    return value


def normalized_token(token: Any) -> str:
    return str(token or "").strip().lower()


def first_token_logprobs(choice: dict[str, Any]) -> tuple[dict[str, float], str]:
    result: dict[str, float] = {}
    logprobs = choice.get("logprobs") or {}
    generated_token = ""

    tokens = logprobs.get("tokens") or []
    token_logprobs = logprobs.get("token_logprobs") or []
    if tokens:
        generated_token = normalized_token(tokens[0])
        if token_logprobs and token_logprobs[0] is not None:
            result[generated_token] = float(token_logprobs[0])

    top_logprobs = logprobs.get("top_logprobs") or []
    first_top = top_logprobs[0] if top_logprobs else {}
    if isinstance(first_top, dict):
        for token, value in first_top.items():
            if value is not None:
                result[normalized_token(token)] = float(value)
    elif isinstance(first_top, list):
        for item in first_top:
            if isinstance(item, dict) and item.get("logprob") is not None:
                result[normalized_token(item.get("token"))] = float(item["logprob"])

    if not generated_token:
        generated_token = normalized_token(choice.get("text"))
    return result, generated_token


def score_completion_choice(
    choice: dict[str, Any],
    missing_logprob_floor: float,
) -> tuple[float, int, bool]:
    token_logprobs, generated_token = first_token_logprobs(choice)
    yes_logprob = token_logprobs.get("yes")
    no_logprob = token_logprobs.get("no")
    missing = int(yes_logprob is None) + int(no_logprob is None)
    if yes_logprob is None:
        yes_logprob = missing_logprob_floor
    if no_logprob is None:
        no_logprob = missing_logprob_floor
    delta = max(-80.0, min(80.0, yes_logprob - no_logprob))
    score = 1.0 / (1.0 + math.exp(-delta))
    return float(score), missing, generated_token not in {"yes", "no"}


@dataclass
class MindIEStats:
    request_count: int = 0
    retry_count: int = 0
    missing_yes_no_logprobs: int = 0
    unexpected_generated_tokens: int = 0
    max_request_chars: int = 0
    prefill_time_ms_sum: float = 0.0
    decode_time_ms_sum: float = 0.0
    queue_wait_time_us_sum: float = 0.0


class MindIEClient:
    def __init__(
        self,
        endpoint: str,
        model_name: str,
        api_key: str,
        timeout: float,
        retries: int,
        top_logprobs: int,
        missing_logprob_floor: float,
        extra_request: dict[str, Any],
        request_mode: str,
    ) -> None:
        self.endpoint = endpoint
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.retries = retries
        self.top_logprobs = top_logprobs
        self.missing_logprob_floor = missing_logprob_floor
        self.extra_request = extra_request
        self.request_mode = request_mode
        self.stats = MindIEStats()
        self._stats_lock = threading.Lock()

    def _add_stat(self, name: str, value: float) -> None:
        with self._stats_lock:
            setattr(self.stats, name, getattr(self.stats, name) + value)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Optional[BaseException] = None
        for attempt in range(self.retries + 1):
            try:
                req = request.Request(self.endpoint, data=body, headers=headers, method="POST")
                with request.urlopen(req, timeout=self.timeout) as response:
                    self._add_stat("request_count", 1)
                    return parse_mindie_body(response.read())
            except error.HTTPError as exc:
                response_body = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"MindIE HTTP {exc.code}: {response_body[:2000]}")
                retryable = exc.code in {408, 429, 500, 502, 503, 504}
                if not retryable or attempt >= self.retries:
                    raise last_error from exc
            except (error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
                if attempt >= self.retries:
                    break
            self._add_stat("retry_count", 1)
            time.sleep(min(8.0, 2.0**attempt))
        raise RuntimeError(f"MindIE request failed after {self.retries + 1} attempts: {last_error}")

    def _payload(self, prompt: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 0,
            "max_tokens": 1,
            "ignore_eos": True,
            "n": 1,
            "best_of": 1,
            "logprobs": self.top_logprobs,
        }
        payload.update(self.extra_request)
        return payload

    def _scores_from_response(self, response: dict[str, Any], expected: int) -> list[float]:
        choices = response.get("choices")
        if not isinstance(choices, list):
            raise ValueError(f"MindIE response has no choices list: {response}")
        if len(choices) != expected:
            raise ValueError(f"MindIE returned {len(choices)} choices for {expected} prompts")
        choices = sorted(choices, key=lambda item: int(item.get("index", 0)))

        scores: list[float] = []
        missing_count = 0
        unexpected_count = 0
        for choice in choices:
            score, missing, unexpected = score_completion_choice(
                choice,
                missing_logprob_floor=self.missing_logprob_floor,
            )
            missing_count += missing
            unexpected_count += int(unexpected)
            scores.append(score)
        self._add_stat("missing_yes_no_logprobs", missing_count)
        self._add_stat("unexpected_generated_tokens", unexpected_count)

        self._add_stat("prefill_time_ms_sum", float(response.get("prefill_time") or 0.0))
        decode_times = response.get("decode_time_arr") or []
        if isinstance(decode_times, list):
            self._add_stat(
                "decode_time_ms_sum",
                sum(float(value or 0.0) for value in decode_times),
            )
        queue_wait = (response.get("usage") or {}).get("queue_wait_time") or []
        if isinstance(queue_wait, list):
            self._add_stat(
                "queue_wait_time_us_sum",
                sum(float(value or 0.0) for value in queue_wait),
            )
        return scores

    def score_prompt(self, prompt: str) -> float:
        response = self._post(self._payload(prompt))
        return self._scores_from_response(response, expected=1)[0]

    def score_batch(self, prompts: list[str]) -> list[float]:
        if self.request_mode == "list":
            response = self._post(self._payload(prompts))
            return self._scores_from_response(response, expected=len(prompts))
        with ThreadPoolExecutor(max_workers=len(prompts), thread_name_prefix="mindie") as executor:
            return list(executor.map(self.score_prompt, prompts))


def packed_batches(
    indexed_prompts: list[tuple[int, str]],
    batch_size: int,
    max_request_chars: int,
    enforce_total_chars: bool = True,
) -> Iterable[list[tuple[int, str]]]:
    batch: list[tuple[int, str]] = []
    char_count = 0
    for item in indexed_prompts:
        prompt_chars = len(item[1])
        if prompt_chars > max_request_chars:
            raise ValueError(
                f"One reranker prompt has {prompt_chars} characters, exceeding "
                f"--max_request_chars={max_request_chars}. Reduce document length or raise the "
                "limit only after confirming the MindIE version accepts it."
            )
        exceeds_total = enforce_total_chars and char_count + prompt_chars > max_request_chars
        if batch and (len(batch) >= batch_size or exceeds_total):
            yield batch
            batch = []
            char_count = 0
        batch.append(item)
        char_count += prompt_chars
    if batch:
        yield batch


def score_with_mindie(
    client: MindIEClient,
    prompts: list[str],
    batch_size: int,
    max_request_chars: int,
    sort_by_length: bool,
    sort_descending: bool,
) -> list[float]:
    if batch_size < 1:
        raise ValueError("--batch_size must be >= 1")
    indexed = list(enumerate(prompts))
    if sort_by_length:
        indexed.sort(key=lambda item: len(item[1]), reverse=sort_descending)
    batches = list(
        packed_batches(
            indexed,
            batch_size,
            max_request_chars,
            enforce_total_chars=client.request_mode == "list",
        )
    )
    scores: list[Optional[float]] = [None] * len(prompts)
    progress = tqdm(batches, desc="MindIE scoring", unit="request", dynamic_ncols=True, ascii=True)
    for batch in progress:
        batch_prompts = [item[1] for item in batch]
        if client.request_mode == "list":
            request_chars = sum(len(prompt) for prompt in batch_prompts)
        else:
            request_chars = max(len(prompt) for prompt in batch_prompts)
        client.stats.max_request_chars = max(client.stats.max_request_chars, request_chars)
        started = time.perf_counter()
        batch_scores = client.score_batch(batch_prompts)
        elapsed = time.perf_counter() - started
        for (original_index, _), score in zip(batch, batch_scores):
            scores[original_index] = score
        progress.set_postfix(size=len(batch), chars=request_chars, sec=f"{elapsed:.2f}")
    if any(score is None for score in scores):
        raise RuntimeError("MindIE scoring left one or more prompts without a score")
    return [float(score) for score in scores if score is not None]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    args = parse_args()
    ensure_local_endpoint(args.endpoint, args.allow_remote_endpoint)
    extra_request = parse_json_object(args.extra_request_json, "--extra_request_json")

    ground_truth = load_ground_truth(
        args.gt_file,
        query_col=args.gt_query_col,
        doc_id_col=args.gt_doc_id_col,
        sheet_name=args.gt_sheet,
    )
    recall_results = load_recall_results(
        args.recall_file,
        id_key=args.recall_id_key,
        text_key=args.recall_text_key,
    )
    _input_texts, mapping, skipped_queries = build_scoring_inputs(
        recall_results,
        ground_truth,
        instruction=args.instruction,
    )
    if skipped_queries:
        logger.warning("Skipped %d recall queries not found in ground truth", skipped_queries)
    if not mapping:
        raise ValueError("No query-document pairs to score after matching recall data to ground truth")

    prompts = [
        format_qwen3_reranker_prompt(row["query"], row["doc"], args.instruction)
        for row in mapping
    ]
    client = MindIEClient(
        endpoint=args.endpoint,
        model_name=args.model_name,
        api_key=args.api_key,
        timeout=args.request_timeout,
        retries=args.request_retries,
        top_logprobs=args.top_logprobs,
        missing_logprob_floor=args.missing_logprob_floor,
        extra_request=extra_request,
        request_mode=args.request_mode,
    )
    started = time.perf_counter()
    scores = score_with_mindie(
        client,
        prompts,
        batch_size=args.batch_size,
        max_request_chars=args.max_request_chars,
        sort_by_length=args.sort_by_length,
        sort_descending=args.sort_descending,
    )
    score_time = time.perf_counter() - started
    seconds_per_example = score_time / max(1, len(scores))

    if client.stats.missing_yes_no_logprobs:
        logger.warning(
            "MindIE omitted %d yes/no alternatives from top_logprobs; floor=%s was used.",
            client.stats.missing_yes_no_logprobs,
            args.missing_logprob_floor,
        )

    ranked_predictions = attach_scores_and_ranks(
        mapping,
        scores,
        ground_truth,
        save_doc_text=args.save_doc_text,
    )
    metrics, per_query = compute_business_metrics(
        ranked_predictions,
        ground_truth,
        args.top_k_list,
        seconds_per_example=seconds_per_example,
        expected_fbeta_beta=args.expected_fbeta_beta,
    )
    metrics.update(
        {
            "backend": "mindie",
            "mindie_endpoint": args.endpoint,
            "mindie_model_name": args.model_name,
            "model_path": args.model_path,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "max_request_chars": args.max_request_chars,
            "top_logprobs": args.top_logprobs,
            "request_mode": args.request_mode,
            "missing_logprob_floor": args.missing_logprob_floor,
            "sort_by_length": args.sort_by_length,
            "sort_descending": args.sort_descending,
            "score_time_seconds": float(score_time),
            "seconds_per_example": float(seconds_per_example),
            "examples_per_second": len(scores) / score_time if score_time > 0 else 0.0,
            "num_scored_pairs": len(scores),
            "num_scored_queries": len({row["query"] for row in mapping}),
            "skipped_recall_queries_without_gt": skipped_queries,
            "mindie_request_count": client.stats.request_count,
            "mindie_retry_count": client.stats.retry_count,
            "mindie_max_request_chars_observed": client.stats.max_request_chars,
            "mindie_missing_yes_no_logprobs": client.stats.missing_yes_no_logprobs,
            "mindie_unexpected_generated_tokens": client.stats.unexpected_generated_tokens,
            "mindie_prefill_time_ms_sum": client.stats.prefill_time_ms_sum,
            "mindie_decode_time_ms_sum": client.stats.decode_time_ms_sum,
            "mindie_queue_wait_time_us_sum": client.stats.queue_wait_time_us_sum,
        }
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_query_metrics.jsonl", per_query)
    write_jsonl(output_dir / "predictions.jsonl", ranked_predictions)
    write_summary_csv(output_dir / "business_eval.csv", per_query)
    wrote_xlsx = write_summary_xlsx(output_dir / "business_eval.xlsx", per_query)
    metrics["summary_csv"] = str(output_dir / "business_eval.csv")
    metrics["summary_xlsx"] = str(output_dir / "business_eval.xlsx") if wrote_xlsx else ""
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(
        "MindIE scoring finished: pairs=%d time=%.3fs examples/s=%.3f",
        len(scores),
        score_time,
        metrics["examples_per_second"],
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
