import argparse
import json
import tempfile
import unittest

from pathlib import Path

from scripts.configure_mindie_qwen3_reranker import configure


class ConfigureMindIETests(unittest.TestCase):
    def test_configures_single_card_atb_service(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "model"
            model_path.mkdir()
            (model_path / "config.json").write_text(
                json.dumps({"torch_dtype": "float16"}), encoding="utf-8"
            )
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "ServerConfig": {},
                        "BackendConfig": {
                            "ScheduleConfig": {"enablePrefixCache": True},
                            "ModelDeployConfig": {
                                "ModelConfig": [{"plugin_params": "prefix_cache"}]
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                config=str(config_path),
                model_path=str(model_path),
                model_name="qwen3-reranker-4b",
                npu_devices="0",
                max_length=8192,
                max_batch_size=32,
                max_prefill_tokens=32768,
                port=1025,
                management_port=1026,
                metrics_port=1027,
                listen_address="127.0.0.1",
                patch_model_dtype=False,
                output="",
            )
            configure(args)
            configured = json.loads(config_path.read_text(encoding="utf-8"))
            backend = configured["BackendConfig"]
            deploy = backend["ModelDeployConfig"]
            model = deploy["ModelConfig"][0]
            self.assertEqual(backend["npuDeviceIds"], [[0]])
            self.assertFalse(backend["ScheduleConfig"]["enablePrefixCache"])
            self.assertEqual(deploy["maxIterTimes"], 1)
            self.assertEqual(model["backendType"], "atb")
            self.assertEqual(model["worldSize"], 1)
            self.assertNotIn("plugin_params", model)


if __name__ == "__main__":
    unittest.main()
