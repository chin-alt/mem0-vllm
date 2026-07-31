import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_atb_sharded_model


def copy_entry(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, destination)
    else:
        shutil.copy2(source, destination)


class AtbShardedModelRunnerTests(unittest.TestCase):
    def test_main_overlays_and_forwards_w8a8sc_quantize_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "model"
            part = root / "part0-of-1"
            part.mkdir(parents=True)
            (root / "config.json").write_text(
                json.dumps({"model_type": "qwen3", "torch_dtype": "bfloat16"}),
                encoding="utf-8",
            )
            (part / "model.safetensors").write_bytes(b"weights")
            runtime = Path(directory) / "runtime"

            observed = {}

            def fake_run_module(module_name, run_name):
                observed["module_name"] = module_name
                observed["run_name"] = run_name
                observed["argv"] = list(sys.argv)
                model_path = Path(
                    sys.argv[sys.argv.index("--model_path") + 1]
                )
                observed["config"] = json.loads(
                    (model_path / "config.json").read_text(encoding="utf-8")
                )
                observed["has_weights"] = (
                    model_path / "model.safetensors"
                ).is_file()

            argv = [
                "run_atb_sharded_model.py",
                "--model-root",
                str(root),
                "--input_texts",
                "hello",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch.dict(
                    run_atb_sharded_model.os.environ,
                    {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1"},
                ),
                mock.patch.object(
                    run_atb_sharded_model.tempfile,
                    "mkdtemp",
                    side_effect=lambda **_kwargs: (
                        runtime.mkdir() or str(runtime)
                    ),
                ),
                mock.patch.object(
                    run_atb_sharded_model,
                    "symlink_entry",
                    side_effect=copy_entry,
                ),
                mock.patch.object(run_atb_sharded_model.atexit, "register"),
                mock.patch.object(
                    run_atb_sharded_model.runpy,
                    "run_module",
                    side_effect=fake_run_module,
                ),
            ):
                run_atb_sharded_model.main()

            self.assertEqual(observed["module_name"], "examples.run_pa")
            self.assertEqual(observed["run_name"], "__main__")
            self.assertIn("--input_texts", observed["argv"])
            quantize_index = observed["argv"].index("--quantize")
            self.assertEqual(observed["argv"][quantize_index + 1], "w8a8sc")
            self.assertTrue(observed["has_weights"])
            self.assertEqual(observed["config"]["quantize"], "w8a8sc")
            self.assertEqual(observed["config"]["torch_dtype"], "float16")

            source_config = json.loads(
                (root / "config.json").read_text(encoding="utf-8")
            )
            self.assertNotIn("quantize", source_config)
            self.assertEqual(source_config["torch_dtype"], "bfloat16")


if __name__ == "__main__":
    unittest.main()
