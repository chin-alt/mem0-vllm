import unittest

from scripts.check_qwen3_vllm_310p_env import (
    QWEN3_BASE_ARCHITECTURE,
    QWEN3_POOLING_ARCHITECTURE,
    qwen3_pooling_registry_supported,
    runtime_pair_supported,
)


class Qwen3Vllm310PEnvTests(unittest.TestCase):
    def test_accepts_stable_0100_runtime_pair(self):
        self.assertTrue(runtime_pair_supported("0.10.0", "0.10.0rc1"))

    def test_rejects_mixed_runtime_pair(self):
        self.assertFalse(runtime_pair_supported("0.10.0", "0.10.2rc1"))

    def test_accepts_vllm_0102_adapter_backed_registry(self):
        self.assertTrue(
            qwen3_pooling_registry_supported({QWEN3_BASE_ARCHITECTURE})
        )

    def test_accepts_direct_sequence_classification_registration(self):
        self.assertTrue(
            qwen3_pooling_registry_supported({QWEN3_POOLING_ARCHITECTURE})
        )

    def test_rejects_registry_without_qwen3(self):
        self.assertFalse(qwen3_pooling_registry_supported({"Qwen2ForCausalLM"}))


if __name__ == "__main__":
    unittest.main()
