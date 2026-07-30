import tempfile
import unittest
from pathlib import Path

from scripts.patch_vllm_ascend_0102_310p import (
    VLLM_LM_HEAD_BLOCK,
    VLLM_LM_HEAD_MARKER,
    has_supported_public_version,
    patch_vllm_qwen3_lm_head_prefix,
)


class PatchVllmAscend310PTests(unittest.TestCase):
    def test_accepts_local_wheel_suffix_for_exact_public_version(self):
        self.assertTrue(has_supported_public_version("0.10.2+empty", "0.10.2"))
        self.assertTrue(
            has_supported_public_version("0.10.2rc1+310p", "0.10.2rc1")
        )

    def test_rejects_different_or_invalid_version(self):
        self.assertFalse(has_supported_public_version("0.10.3", "0.10.2"))
        self.assertFalse(has_supported_public_version("not-a-version", "0.10.2"))

    def test_lm_head_patch_accepts_image_version(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapters_path = Path(temp_dir) / "adapters.py"
            adapters_path.write_text(
                VLLM_LM_HEAD_BLOCK + "\n" + VLLM_LM_HEAD_BLOCK,
                encoding="utf-8",
            )

            changed = patch_vllm_qwen3_lm_head_prefix(
                adapters_path, "0.10.2+empty"
            )

            source = adapters_path.read_text(encoding="utf-8")
            self.assertTrue(changed)
            self.assertEqual(source.count(VLLM_LM_HEAD_MARKER), 2)
            self.assertTrue(adapters_path.with_suffix(".py.memranker.bak").is_file())


if __name__ == "__main__":
    unittest.main()
