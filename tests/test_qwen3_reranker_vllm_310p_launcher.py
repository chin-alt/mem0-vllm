import unittest

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = PROJECT_ROOT / "scripts" / "run_qwen3_reranker_vllm_310p_container.sh"


class Qwen3RerankerVllm310pLauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = LAUNCHER.read_text(encoding="utf-8")

    def test_default_token_capacity_matches_requested_batch(self):
        self.assertIn('BATCH_SIZE="${BATCH_SIZE:-16}"', self.script)
        self.assertIn('MAX_NUM_SEQS="${MAX_NUM_SEQS:-${BATCH_SIZE}}"', self.script)
        self.assertIn(
            'MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$((MAX_LENGTH * BATCH_SIZE))}"',
            self.script,
        )

    def test_official_310p_qwen3_fused_ops_are_forwarded(self):
        self.assertIn(
            "VLLM_COMPILATION_CONFIG="
            "'{\"custom_ops\":[\"none\",\"+rms_norm\",\"+rotary_embedding\"]}'",
            self.script,
        )
        self.assertIn(
            '-e "VLLM_COMPILATION_CONFIG=${VLLM_COMPILATION_CONFIG}"',
            self.script,
        )


if __name__ == "__main__":
    unittest.main()
