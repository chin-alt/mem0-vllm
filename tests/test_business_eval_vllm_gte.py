import sys
import unittest

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from business_eval_vllm import (  # noqa: E402
    GTE_RERANKER_HF_OVERRIDES,
    QWEN3_RERANKER_HF_OVERRIDES,
    build_prefix_cache_seed_phases,
    format_gte_score_inputs,
    format_qwen3_score_inputs,
    extract_vllm_score,
    order_score_pairs,
    pooling_api_kwargs,
    pooling_hf_overrides,
    prefix_cache_pooler_override,
    prepare_pretokenized_pooling_inputs,
    inspect_vllm_prefix_cache_state,
    reset_vllm_prefix_cache,
    score_with_vllm,
    summarize_batch_latency,
)


class GteVllmTests(unittest.TestCase):
    @staticmethod
    def _fake_vllm_module():
        module = ModuleType("vllm")

        class FakePoolingParams:
            def __init__(self, task=None):
                self.task = task

        module.PoolingParams = FakePoolingParams
        return module

    def test_pooling_api_uses_task_score_on_vllm_0100(self):
        class LegacyLlm:
            def __init__(self, model, *, task="auto", **kwargs):
                pass

        self.assertEqual(pooling_api_kwargs(LegacyLlm), {"task": "score"})

    def test_extract_score_accepts_base_pooling_tensor_like_output(self):
        class TensorLike:
            def tolist(self):
                return [0.75]

        output = SimpleNamespace(outputs=SimpleNamespace(data=TensorLike()))
        self.assertEqual(extract_vllm_score(output), 0.75)

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

    def test_qwen3_prefix_cache_forces_last_pooler_only_when_needed(self):
        self.assertEqual(
            prefix_cache_pooler_override("qwen3", "pooling", True),
            {"pooling_type": "LAST"},
        )
        self.assertIsNone(prefix_cache_pooler_override("qwen3", "pooling", False))
        self.assertIsNone(prefix_cache_pooler_override("gte", "pooling", True))
        self.assertIsNone(prefix_cache_pooler_override("qwen3", "generate", True))

    def test_inspect_vllm_prefix_cache_state_reads_initialized_config(self):
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                vllm_config=SimpleNamespace(
                    cache_config=SimpleNamespace(enable_prefix_caching=True),
                    model_config=SimpleNamespace(
                        pooler_config=SimpleNamespace(pooling_type="LAST")
                    ),
                )
            )
        )
        self.assertEqual(inspect_vllm_prefix_cache_state(llm), (True, "LAST"))

    def test_reset_prefix_cache_accepts_v1_none_success_return(self):
        llm = SimpleNamespace(reset_prefix_cache=lambda: None)
        self.assertIsNone(reset_vllm_prefix_cache(llm))

    def test_reset_prefix_cache_rejects_explicit_false(self):
        llm = SimpleNamespace(reset_prefix_cache=lambda: False)
        with self.assertRaisesRegex(RuntimeError, "refused"):
            reset_vllm_prefix_cache(llm)

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

    def test_prefix_cache_seed_plan_uses_shortest_pair_per_query_once(self):
        indexed = [
            (0, "q1", "333", 5),
            (2, "q1", "1", 3),
            (1, "q2", "22", 4),
            (3, "q2", "4444", 6),
            (4, "q3", "x", 3),
        ]

        phases = build_prefix_cache_seed_phases(indexed)

        self.assertEqual([name for name, _ in phases], [
            "global_seed",
            "query_seeds",
            "remainder",
        ])
        self.assertEqual([item[0] for item in phases[0][1]], [2])
        self.assertEqual([item[0] for item in phases[1][1]], [1, 4])
        self.assertEqual([item[0] for item in phases[2][1]], [0, 3])
        self.assertEqual(
            sorted(item[0] for _, phase in phases for item in phase),
            [0, 1, 2, 3, 4],
        )

    def test_prefix_cache_seeding_scores_three_dependency_phases(self):
        class FakeLlm:
            _memranker_scoring_backend = "pooling"
            _memranker_model_family = "gte"

            def __init__(self):
                self.calls = []

            def score(self, queries, documents, *, truncate_prompt_tokens=None, use_tqdm=True):
                self.calls.append((list(queries), list(documents)))
                return [
                    SimpleNamespace(outputs=SimpleNamespace(score=float(document)))
                    for document in documents
                ]

        llm = FakeLlm()
        events = []
        timings = {}
        scores = score_with_vllm(
            llm,
            queries=["q1", "q2", "q1", "q2"],
            documents=["333", "22", "1", "4444"],
            batch_size=16,
            instruction="",
            sort_by_length=True,
            max_length=512,
            submit_all_at_once=True,
            group_by_query=True,
            prefix_cache_seeding=True,
            progress_callback=events.append,
            timing_metrics=timings,
        )

        self.assertEqual(len(llm.calls), 3)
        self.assertEqual(llm.calls[0][1], ["1"])
        self.assertEqual(llm.calls[1][1], ["22"])
        self.assertEqual(llm.calls[2][1], ["333", "4444"])
        self.assertEqual(scores, [333.0, 22.0, 1.0, 4444.0])
        self.assertEqual([event["phase"] for event in events], [
            "global_seed",
            "query_seeds",
            "remainder",
        ])
        self.assertEqual(timings["prefix_cache_seed_num_queries"], 2)
        self.assertEqual(timings["prefix_cache_seed_num_phases"], 3)
        self.assertIn("prefix_cache_global_seed_time_seconds", timings)
        self.assertIn("prefix_cache_query_seeds_time_seconds", timings)
        self.assertIn("prefix_cache_remainder_time_seconds", timings)

    def test_pretokenized_pooling_batches_validates_and_uses_encode_score(self):
        class FakeFastTokenizer:
            is_fast = True

            def __init__(self):
                self.batch_calls = []
                self.scalar_calls = []

            @staticmethod
            def token_ids(text, max_length):
                return [ord(char) % 97 for char in text][:max_length]

            def __call__(self, *, text, max_length, truncation, **kwargs):
                self.assert_common(max_length, truncation)
                if isinstance(text, list):
                    self.batch_calls.append((list(text), dict(kwargs)))
                    return {
                        "input_ids": [self.token_ids(item, max_length) for item in text]
                    }
                self.scalar_calls.append((text, dict(kwargs)))
                return {"input_ids": self.token_ids(text, max_length)}

            @staticmethod
            def assert_common(max_length, truncation):
                if max_length != 512 or truncation is not True:
                    raise AssertionError("unexpected tokenizer truncation settings")

        class FakeLlm:
            _memranker_scoring_backend = "pooling"
            _memranker_model_family = "qwen3"

            def __init__(self):
                self.tokenizer = FakeFastTokenizer()
                self.llm_engine = SimpleNamespace(
                    model_config=SimpleNamespace(use_pad_token=False)
                )
                self.encode_calls = []

            def get_tokenizer(self):
                return self.tokenizer

            def encode(self, prompts, pooling_params, **kwargs):
                self.encode_calls.append((prompts, pooling_params, kwargs))
                return [
                    SimpleNamespace(outputs=SimpleNamespace(data=[float(index)]))
                    for index in range(len(prompts))
                ]

            def score(self, *args, **kwargs):
                raise AssertionError("PRETOKENIZED_POOLING must not call LLM.score")

        llm = FakeLlm()
        timings = {}
        events = []
        with patch.dict(sys.modules, {"vllm": self._fake_vllm_module()}):
            scores = score_with_vllm(
                llm,
                queries=["q1", "q2", "q1"],
                documents=["333", "22", "1"],
                batch_size=16,
                instruction="instruction",
                sort_by_length=True,
                max_length=512,
                submit_all_at_once=True,
                group_by_query=True,
                pretokenized_pooling=True,
                tokenizer_batch_size=2,
                timing_metrics=timings,
                progress_callback=events.append,
            )

        self.assertEqual(len(llm.encode_calls), 1)
        prompts, pooling_params, encode_kwargs = llm.encode_calls[0]
        self.assertEqual(pooling_params.task, "score")
        self.assertEqual(encode_kwargs["pooling_task"], "score")
        self.assertEqual(encode_kwargs["tokenization_kwargs"], {})
        self.assertEqual(len(prompts), 3)
        self.assertTrue(all(set(prompt) == {"prompt_token_ids"} for prompt in prompts))
        self.assertEqual(len(llm.tokenizer.batch_calls), 2)
        self.assertEqual(len(llm.tokenizer.scalar_calls), 3)
        self.assertTrue(
            all(call_kwargs["padding"] is False for _, call_kwargs in llm.tokenizer.batch_calls)
        )
        self.assertEqual(scores, [1.0, 2.0, 0.0])
        self.assertEqual(timings["num_token_ids_validated"], 3)
        self.assertTrue(timings["token_id_parity_passed"])
        self.assertIn("prompt_format_time_seconds", timings)
        self.assertIn("tokenizer_time_seconds", timings)
        self.assertIn("vllm_enqueue_and_npu_execute_time_seconds", timings)
        self.assertIn("pretokenized_total_time_seconds", timings)
        self.assertEqual(events[0]["batch_size"], 3)

    def test_pretokenized_pooling_refuses_mismatch_before_encode(self):
        class MismatchingFastTokenizer:
            is_fast = True

            def __init__(self):
                self.encode_called = False

            def __call__(self, *, text, max_length, truncation, **kwargs):
                if isinstance(text, list):
                    return {"input_ids": [[1, 999] for _ in text]}
                return {"input_ids": [1, 2]}

        tokenizer = MismatchingFastTokenizer()
        llm = SimpleNamespace(
            get_tokenizer=lambda: tokenizer,
            llm_engine=SimpleNamespace(
                model_config=SimpleNamespace(use_pad_token=False)
            ),
        )
        indexed = [(0, "query", "document", 13)]
        with self.assertRaisesRegex(RuntimeError, "No request was submitted"):
            prepare_pretokenized_pooling_inputs(
                llm,
                indexed,
                instruction="instruction",
                model_family="qwen3",
                max_length=512,
                tokenizer_batch_size=256,
            )

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
