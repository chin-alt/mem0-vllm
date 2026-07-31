import tempfile
import unittest

from pathlib import Path

from scripts.probe_vllm_ascend_w8a8sc_310p import inspect_model_layout


class VllmAscendW8A8SCProbeTests(unittest.TestCase):
    def test_detects_vllm_sharded_state_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            (model_path / "quant_model_description.json").write_text("{}")
            (model_path / "model-rank-0-part-0.safetensors").touch()

            report = inspect_model_layout(model_path)

        self.assertEqual(report["model_layout"], "vllm_sharded_state")
        self.assertEqual(
            report["vllm_sharded_state_files"],
            ["model-rank-0-part-0.safetensors"],
        )

    def test_rejects_atb_part_layout_as_vllm_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir)
            part_path = model_path / "part0-of-1"
            part_path.mkdir()
            (part_path / "quant_model_description_w8a8sc.json").write_text("{}")
            (part_path / "quant_model_weight_w8a8sc.safetensors").touch()

            report = inspect_model_layout(model_path)

        self.assertEqual(report["model_layout"], "atb_part_layout")
        self.assertEqual(report["atb_part_directories"], ["part0-of-1"])

    def test_reports_missing_model_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = inspect_model_layout(Path(temp_dir) / "missing")

        self.assertFalse(report["model_path_exists"])


if __name__ == "__main__":
    unittest.main()
