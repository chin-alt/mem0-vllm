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

    def test_pretokenized_pooling_is_opt_in_and_buffered(self):
        self.assertIn('PRETOKENIZED_POOLING="${PRETOKENIZED_POOLING:-0}"', self.script)
        self.assertIn('TOKENIZER_BATCH_SIZE="${TOKENIZER_BATCH_SIZE:-256}"', self.script)
        self.assertIn('-e "PRETOKENIZED_POOLING=${PRETOKENIZED_POOLING}"', self.script)
        self.assertIn('-e "TOKENIZER_BATCH_SIZE=${TOKENIZER_BATCH_SIZE}"', self.script)

    def test_query_grouping_and_prefix_cache_remain_enabled(self):
        self.assertIn('ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-1}"', self.script)
        self.assertIn('GROUP_BY_QUERY="${GROUP_BY_QUERY:-1}"', self.script)
        self.assertIn('-e "ENABLE_PREFIX_CACHING=${ENABLE_PREFIX_CACHING}"', self.script)
        self.assertIn('-e "GROUP_BY_QUERY=${GROUP_BY_QUERY}"', self.script)
        self.assertIn(
            '-e "VLLM_COMPILATION_CONFIG=${VLLM_COMPILATION_CONFIG}"',
            self.script,
        )

    def test_official_ascend_runtime_settings_are_forwarded(self):
        expected_defaults = {
            "TASK_QUEUE_ENABLE": "2",
            "CPU_AFFINITY_CONF": "1",
            "PYTORCH_NPU_ALLOC_CONF": "max_split_size_mb:256",
        }
        for name, default in expected_defaults.items():
            with self.subTest(name=name):
                operator = "-" if name == "PYTORCH_NPU_ALLOC_CONF" else ":-"
                self.assertIn(
                    f'{name}="${{{name}{operator}{default}}}"',
                    self.script,
                )
                self.assertIn(f'-e "{name}=${{{name}}}"', self.script)


if __name__ == "__main__":
    unittest.main()
