import math
import unittest

import torch

from scripts.business_eval_atb import (
    YesNoLogitCapture,
    limit_to_matching_queries,
    single_token_id,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"yes": [7], "no": [9], "many": [1, 2]}[text]


class AtbBusinessEvaluationTests(unittest.TestCase):
    def test_single_token_id_rejects_multi_token_label(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(single_token_id(tokenizer, "yes"), 7)
        with self.assertRaisesRegex(ValueError, "one tokenizer id"):
            single_token_id(tokenizer, "many")

    def test_capture_uses_normalized_yes_no_logits(self):
        logits = torch.zeros((2, 12), dtype=torch.float32)
        logits[0, 7] = 2.0
        logits[0, 9] = 0.0
        logits[1, 7] = -1.0
        logits[1, 9] = 1.0
        capture = YesNoLogitCapture(
            original_chooser=lambda value: torch.argmax(value, dim=-1),
            yes_token_id=7,
            no_token_id=9,
        )

        chosen = capture(logits)

        self.assertEqual(chosen.tolist(), [7, 9])
        self.assertAlmostEqual(capture.scores[0], 1.0 / (1.0 + math.exp(-2.0)))
        self.assertAlmostEqual(capture.scores[1], 1.0 / (1.0 + math.exp(2.0)))

    def test_query_limit_keeps_only_queries_with_recall(self):
        ground_truth = {"q1": object(), "q2": object(), "q3": object()}
        recall = {"q2": [], "q3": []}
        selected = limit_to_matching_queries(ground_truth, recall, max_queries=1)
        self.assertEqual(list(selected), ["q2"])
        self.assertIs(
            limit_to_matching_queries(ground_truth, recall, max_queries=0),
            ground_truth,
        )


if __name__ == "__main__":
    unittest.main()
