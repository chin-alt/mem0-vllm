import math
import unittest

import torch

from scripts.business_eval_atb import (
    YesNoLogitCapture,
    format_atb_prompts,
    parse_business_input,
)
from scripts.prepare_qwen3_reranker_calibration import (
    QWEN3_RERANKER_SUFFIX,
    format_prompt,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        del skip_special_tokens, clean_up_tokenization_spaces
        return "".join(chr(token_id) for token_id in token_ids)


class AtbBusinessEvaluationTests(unittest.TestCase):
    def test_formats_existing_business_input_for_atb(self):
        expected = format_prompt(
            "instruction",
            "query",
            "document",
            backend="generate",
        )
        prompts = format_atb_prompts(
            FakeTokenizer(),
            ["<Instruct>: instruction\n<Query>: query\n<Document>: document"],
            max_length=len(expected),
        )
        self.assertEqual(prompts[0].prompt, expected)
        self.assertEqual(
            list(prompts[0].input_ids),
            FakeTokenizer().encode(expected),
        )
        self.assertEqual(prompts[0].token_length, len(expected))
        self.assertFalse(prompts[0].truncated)

    def test_truncates_only_document_and_preserves_generate_suffix(self):
        prompts = format_atb_prompts(
            FakeTokenizer(),
            [
                "<Instruct>: instruction\n<Query>: query\n<Document>: "
                + "d" * 1000
            ],
            max_length=512,
        )
        self.assertEqual(prompts[0].token_length, 512)
        self.assertTrue(prompts[0].truncated)
        self.assertTrue(prompts[0].prompt.endswith(QWEN3_RERANKER_SUFFIX))

    def test_parses_multiline_instruction(self):
        self.assertEqual(
            parse_business_input(
                "<Instruct>: first\nsecond\n<Query>: q\n<Document>: d"
            ),
            ("first\nsecond", "q", "d"),
        )

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
