import unittest

from scripts.check_qwen3_reranker_w8a8_310p import validate_quant_description


class Qwen3RerankerW8A8CheckTests(unittest.TestCase):
    def make_description(self):
        return {
            "model.embed_tokens.weight": "FLOAT",
            "lm_head.weight": "FLOAT",
            "model.layers.0.self_attn.q_proj.weight": "W8A8",
            "model.layers.0.self_attn.k_proj.weight": "W8A8",
            "model.layers.0.self_attn.v_proj.weight": "W8A8",
            "model.layers.0.self_attn.o_proj.weight": "W8A8",
            "model.layers.0.mlp.gate_proj.weight": "W8A8",
            "model.layers.0.mlp.up_proj.weight": "W8A8",
            "model.layers.0.mlp.down_proj.weight": "W8A8",
        }

    def test_accepts_static_body_with_float_embedding_and_head(self):
        failures, warnings, counts = validate_quant_description(
            self.make_description()
        )
        self.assertEqual(failures, [])
        self.assertEqual(counts["W8A8"], 7)
        self.assertTrue(any("score.weight is absent" in item for item in warnings))

    def test_rejects_quantized_lm_head(self):
        description = self.make_description()
        description["lm_head.weight"] = "W8A8"
        failures, _warnings, _counts = validate_quant_description(description)
        self.assertTrue(any("lm_head.weight must remain FLOAT" in item for item in failures))

    def test_rejects_dynamic_quantization(self):
        description = self.make_description()
        description["model.layers.0.mlp.down_proj.weight"] = "W8A8_DYNAMIC"
        failures, _warnings, _counts = validate_quant_description(description)
        self.assertTrue(any("dynamic/per-token" in item for item in failures))


if __name__ == "__main__":
    unittest.main()
