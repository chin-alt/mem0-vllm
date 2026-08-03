import unittest

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Qwen3RecallTopKSweepTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (
            PROJECT_ROOT / "scripts" / "sweep_qwen3_reranker_recall_topk_0625_310p.sh"
        ).read_text(encoding="utf-8")

    def test_sweeps_every_integer_from_25_to_10_descending(self):
        self.assertIn('TOP_K_START="${TOP_K_START:-25}"', self.script)
        self.assertIn('TOP_K_END="${TOP_K_END:-10}"', self.script)
        self.assertIn(
            "for ((top_k = TOP_K_START; top_k >= TOP_K_END; top_k--))",
            self.script,
        )

    def test_uses_final_non_apc_pretokenized_configuration(self):
        self.assertIn("PRETOKENIZED_POOLING=1", self.script)
        self.assertIn("PREFIX_CACHE_SEEDING=0", self.script)
        self.assertIn("ENABLE_PREFIX_CACHING=0", self.script)
        self.assertIn('RECALL_TOP_K="${top_k}"', self.script)

    def test_writes_requested_metrics_and_comparison_files(self):
        self.assertIn('metrics["Accuracy@GTCount"]', self.script)
        self.assertIn('metrics["score_time_seconds"]', self.script)
        self.assertIn('output_root / "recall_topk_sweep.csv"', self.script)
        self.assertIn('output_root / "recall_topk_sweep.json"', self.script)


if __name__ == "__main__":
    unittest.main()
