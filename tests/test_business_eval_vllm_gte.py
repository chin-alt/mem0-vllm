import sys
import unittest

from pathlib import Path
from types import SimpleNamespace


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from business_eval_vllm import (  # noqa: E402
    GTE_RERANKER_HF_OVERRIDES,
    format_gte_score_inputs,
    pooling_hf_overrides,
    score_with_vllm,
)


class GteVllmTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
