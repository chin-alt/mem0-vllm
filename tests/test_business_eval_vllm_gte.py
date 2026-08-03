import sys
import unittest

from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from business_eval_vllm import (  # noqa: E402
    GTE_RERANKER_HF_OVERRIDES,
    QWEN3_RERANKER_HF_OVERRIDES,
    format_gte_score_inputs,
    format_qwen3_score_inputs,
    order_score_pairs,
    pooling_api_kwargs,
    pooling_hf_overrides,
    score_with_vllm,
    summarize_batch_latency,
)


class GteVllmTests(unittest.TestCase):
    def test_pooling_api_uses_task_score_on_vllm_0100(self):
        class LegacyLlm:
            def __init__(self, model, *, task="auto", **kwargs):
                pass

        self.assertEqual(pooling_api_kwargs(LegacyLlm), {"task": "score"})

    def test_pooling_api_uses_runner_on_vllm_0102(self):
        class RunnerLlm:
            def __init__(self, model, *, runner="auto", **kwargs):
                pass

        self.assertEqual(pooling_api_kwargs(RunnerLlm), {"runner": "pooling"})

    def test_pooling_api_rejects_unknown_constructor(self):
        class UnsupportedLlm:
            def __init__(self, model, **kwargs):
                pass

        with self.assertRaisesRegex(RuntimeError, "neither"):
            pooling_api_kwargs(UnsupportedLlm)

    def test_qwen3_pooling_uses_two_token_classifier(self):
        self.assertEqual(
            pooling_hf_overrides("qwen3"),
            QWEN3_RERANKER_HF_OVERRIDES,
        )
        queries, documents = format_qwen3_score_inputs(
            ["query"], ["document"], "instruction"
        )
        self.assertIn("<Instruct>: instruction", queries[0])
        self.assertIn("<Query>: query", queries[0])
        self.assertIn("<Document>: document", documents[0])

    def test_gte_uses_native_vllm_architecture(self):
        self.assertEqual(
            pooling_hf_overrides("gte"),
            GTE_RERANKER_HF_OVERRIDES,
        )

    def test_gte_formats_tokenizer_pairs_without_qwen_prompt(self):
        queries, documents = format_gte_score_inputs(
            [" query "],
            [" document "],
            "instruction",
        )
        self.assertEqual(queries, ["instruction\n\nquery"])
        self.assertEqual(documents, ["document"])

    def test_gte_score_uses_truncation_and_restores_input_order(self):
        class FakeLlm:
            _memranker_scoring_backend = "pooling"
            _memranker_model_family = "gte"

            def __init__(self):
                self.calls = []

            def score(self, queries, documents, *, truncate_prompt_tokens=None, use_tqdm=True):
                self.calls.append((queries, documents, truncate_prompt_tokens, use_tqdm))
                return [
                    SimpleNamespace(outputs=SimpleNamespace(score=float(len(query))))
                    for query in queries
                ]

        llm = FakeLlm()
        scores = score_with_vllm(
            llm,
            queries=["long query", "q"],
            documents=["doc", "doc"],
            batch_size=2,
            instruction="",
            sort_by_length=True,
            max_length=512,
        )

        self.assertEqual(scores, [10.0, 1.0])
        self.assertEqual(llm.calls[0][2:], (512, False))

    def test_submit_all_groups_same_query_and_restores_input_order(self):
        class FakeLlm:
            _memranker_scoring_backend = "pooling"
            _memranker_model_family = "gte"

            def __init__(self):
                self.calls = []

            def score(self, queries, documents, *, truncate_prompt_tokens=None, use_tqdm=True):
                self.calls.append((queries, documents, truncate_prompt_tokens, use_tqdm))
                return [
                    SimpleNamespace(outputs=SimpleNamespace(score=float(document)))
                    for document in documents
                ]

        llm = FakeLlm()
        scores = score_with_vllm(
            llm,
            queries=["q1", "q2", "q1"],
            documents=["333", "22", "1"],
            batch_size=1,
            instruction="",
            sort_by_length=True,
            max_length=512,
            submit_all_at_once=True,
            group_by_query=True,
        )

        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(llm.calls[0][0], ["q1", "q1", "q2"])
        self.assertEqual(llm.calls[0][1], ["1", "333", "22"])
        self.assertEqual(scores, [333.0, 22.0, 1.0])

    def test_query_grouping_preserves_first_query_order(self):
        ordered = order_score_pairs(
            ["q2", "q1", "q2", "q1"],
            ["bbbb", "c", "a", "dd"],
            group_by_query=True,
            sort_by_length=True,
            sort_descending=False,
        )

        self.assertEqual([(item[1], item[2]) for item in ordered], [
            ("q2", "a"),
            ("q2", "bbbb"),
            ("q1", "c"),
            ("q1", "dd"),
        ])

    def test_batch_latency_summary_reports_batch_and_pair_percentiles(self):
        summary = summarize_batch_latency(
            [
                {"batch_seconds": 1.0, "batch_size": 2},
                {"batch_seconds": 2.0, "batch_size": 4},
            ]
        )
        self.assertEqual(summary["num_timed_batches"], 2)
        self.assertEqual(summary["batch_latency_p50_seconds"], 1.5)
        self.assertEqual(summary["pair_latency_p50_seconds"], 0.5)


if __name__ == "__main__":
    unittest.main()
