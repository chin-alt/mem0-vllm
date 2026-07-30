import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_modelslim_cpu.py"


class RunModelSlimCpuTests(unittest.TestCase):
    def test_patches_both_transformers_npu_probe_references_before_target(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            utils_dir = root / "transformers" / "utils"
            utils_dir.mkdir(parents=True)
            (root / "transformers" / "__init__.py").write_text("", encoding="utf-8")
            (utils_dir / "import_utils.py").write_text(
                "def is_torch_npu_available(*args, **kwargs): return True\n",
                encoding="utf-8",
            )
            (utils_dir / "__init__.py").write_text(
                "from .import_utils import is_torch_npu_available\n",
                encoding="utf-8",
            )
            target = root / "target.py"
            target.write_text(
                textwrap.dedent(
                    """
                    import sys
                    import transformers.utils as public_utils
                    from transformers.utils import import_utils

                    assert public_utils.is_torch_npu_available() is False
                    assert import_utils.is_torch_npu_available() is False
                    assert sys.argv[1:] == ["--model_path", "/models/float"]
                    print("target-ok")
                    """
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root)
            result = subprocess.run(
                [
                    sys.executable,
                    str(RUNNER),
                    str(target),
                    "--model_path",
                    "/models/float",
                ],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Transformers torch_npu probe disabled", result.stdout)
        self.assertIn("target-ok", result.stdout)

    def test_clones_tied_weights_for_legacy_safetensors_dict_saver(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            utils_dir = root / "transformers" / "utils"
            utils_dir.mkdir(parents=True)
            (root / "transformers" / "__init__.py").write_text("", encoding="utf-8")
            (utils_dir / "import_utils.py").write_text(
                "def is_torch_npu_available(*args, **kwargs): return True\n",
                encoding="utf-8",
            )
            (utils_dir / "__init__.py").write_text(
                "from .import_utils import is_torch_npu_available\n",
                encoding="utf-8",
            )
            output = root / "tied.safetensors"
            target = root / "save_tied.py"
            target.write_text(
                textwrap.dedent(
                    """
                    import sys
                    import torch
                    from safetensors.torch import load_file, save_file

                    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)
                    save_file(
                        {
                            "model.embed_tokens.weight": weight,
                            "lm_head.weight": weight,
                        },
                        sys.argv[1],
                    )
                    loaded = load_file(sys.argv[1])
                    assert set(loaded) == {
                        "model.embed_tokens.weight",
                        "lm_head.weight",
                    }
                    assert torch.equal(
                        loaded["model.embed_tokens.weight"], loaded["lm_head.weight"]
                    )
                    print("tied-save-ok")
                    """
                ),
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = str(root)
            result = subprocess.run(
                [sys.executable, str(RUNNER), str(target), str(output)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("keep=model.embed_tokens.weight clone=lm_head.weight", result.stdout)
        self.assertIn("tied-save-ok", result.stdout)


if __name__ == "__main__":
    unittest.main()
