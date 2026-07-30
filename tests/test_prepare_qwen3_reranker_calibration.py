import json
import tempfile
import unittest

from pathlib import Path

from business_eval_vllm import (
    format_qwen3_generate_prompts,
    format_qwen3_score_inputs,
)
from scripts.prepare_qwen3_reranker_calibration import (
    QWEN3_RERANKER_SUFFIX,
    CalibrationCandidate,
    build_candidates,
    format_prompt,
    select_length_stratified,
    write_outputs,
)


class CharacterTokenizer:
    def encode(self, text, add_special_tokens=True):
        return [ord(character) for character in text]

    def decode(
        self,
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    ):
        return "".join(chr(token_id) for token_id in token_ids)


class PrepareQwen3RerankerCalibrationTests(unittest.TestCase):
    def test_prompt_uses_online_pooling_shape_without_label_or_reason(self):
        prompt = format_prompt("instruction", "query", "document", "pooling")
        self.assertIn("<Instruct>: instruction\n<Query>: query\n<Document>: document", prompt)
        self.assertTrue(prompt.endswith(QWEN3_RERANKER_SUFFIX))
        self.assertNotIn("labels", prompt)
        self.assertNotIn("reason", prompt)

    def test_calibration_formats_match_online_vllm_formats(self):
        queries, documents = format_qwen3_score_inputs(
            ["query"], ["document"], "instruction"
        )
        self.assertEqual(
            format_prompt("instruction", "query", "document", "pooling"),
            queries[0] + documents[0],
        )
        self.assertEqual(
            format_prompt("instruction", "query", "document", "generate"),
            format_qwen3_generate_prompts(
                ["query"], ["document"], "instruction"
            )[0],
        )

    def test_long_document_is_truncated_but_suffix_is_preserved(self):
        tokenizer = CharacterTokenizer()
        rows = [
            (
                7,
                {
                    "instruction": "rank",
                    "query": "q",
                    "doc": "d" * 1000,
                    "reason": "must not be calibrated",
                    "labels": 8.6,
                },
            )
        ]
        candidates, skipped, instructions = build_candidates(
            rows,
            tokenizer=tokenizer,
            backend="pooling",
            max_length=512,
        )
        self.assertEqual(skipped, 0)
        self.assertEqual(instructions, {"rank"})
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].source_index, 7)
        self.assertTrue(candidates[0].truncated)
        self.assertLessEqual(candidates[0].token_length, 512)
        self.assertTrue(candidates[0].prompt.endswith(QWEN3_RERANKER_SUFFIX))
        self.assertNotIn("must not be calibrated", candidates[0].prompt)

    def test_length_stratification_is_deterministic_and_covers_bins(self):
        candidates = [
            CalibrationCandidate(index, str(index), index + 10, index + 10, False)
            for index in range(20)
        ]
        first = select_length_stratified(candidates, sample_count=8, length_bins=4, seed=3)
        second = select_length_stratified(candidates, sample_count=8, length_bins=4, seed=3)
        self.assertEqual(first, second)
        selected_lengths = sorted(item.original_token_length for item in first)
        self.assertLess(selected_lengths[0], 15)
        self.assertGreaterEqual(selected_lengths[-1], 25)

    def test_modelslim_output_uses_inputs_pretokenized_key(self):
        selected = [CalibrationCandidate(2, "prompt", 6, 6, False)]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "calib.jsonl"
            manifest = {"test": True}
            manifest_path = write_outputs(output, selected, manifest, overwrite=False)
            row = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(row["inputs_pretokenized"], "prompt")
            self.assertNotIn("labels", row)
            self.assertTrue(manifest_path.is_file())


if __name__ == "__main__":
    unittest.main()
