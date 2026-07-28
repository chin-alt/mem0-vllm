import sys
import unittest

from pathlib import Path


SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluate_business_gte_npu import (  # noqa: E402
    format_gte_query,
    install_gte_pfa_attention,
    sigmoid,
    validate_attention_backend,
)


class GteNpuHelpersTests(unittest.TestCase):
    def test_query_is_unchanged_without_instruction(self):
        self.assertEqual(format_gte_query("  原始查询  "), "原始查询")

    def test_instruction_is_optional_and_explicit(self):
        self.assertEqual(format_gte_query("查询", "任务"), "任务\n\n查询")

    def test_sigmoid_is_stable(self):
        values = sigmoid(__import__("numpy").array([-1000.0, 0.0, 1000.0]))
        self.assertEqual(values[0], 1.8048513878454153e-35)
        self.assertEqual(values[1], 0.5)
        self.assertEqual(values[2], 1.0)

    def test_pfa_requires_fp16_on_310p(self):
        validate_attention_backend("pfa", "fp16")
        with self.assertRaisesRegex(ValueError, "only supports --dtype fp16"):
            validate_attention_backend("pfa", "bf16")

    def test_pfa_requires_torch_npu_operator(self):
        class FakeTorchNpu:
            pass

        with self.assertRaisesRegex(RuntimeError, "npu_prompt_flash_attention"):
            install_gte_pfa_attention(object(), object(), FakeTorchNpu())


if __name__ == "__main__":
    unittest.main()
