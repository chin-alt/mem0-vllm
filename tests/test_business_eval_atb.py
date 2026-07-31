import math
import unittest

import torch

from scripts.business_eval_atb import (
    YesNoLogitCapture,
    format_atb_prompts,
)
from modeling import RERANKER_PREFIX, RERANKER_SUFFIX


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == RERANKER_PREFIX:
            return [10, 11]
        if text == RERANKER_SUFFIX:
            return [20, 21]
        return [1, 2, 3, 4]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return ",".join(str(token_id) for token_id in token_ids)


class AtbBusinessEvaluationTests(unittest.TestCase):
    def test_formats_existing_business_input_for_atb(self):
        prompts = format_atb_prompts(
            FakeTokenizer(),
            ["business input"],
            max_length=7,
        )
        self.assertEqual(prompts, [("10,11,1,2,3,20,21", 7)])

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

if __name__ == "__main__":
    unittest.main()
