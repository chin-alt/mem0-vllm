from pathlib import Path
import tempfile
import unittest

from scripts.patch_vllm_ascend_0102_310p import (
    CALL,
    MARKER,
    POOLING_CONDITION,
    POOLING_MARKER,
    patch_platform,
    patch_worker,
    restore_worker,
)


class PatchVllmAscendTests(unittest.TestCase):
    def test_patch_is_idempotent_and_restorable(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "worker_v1.py"
            original = "def compile_or_warm_up_model(self):\n%s\n" % CALL
            worker.write_text(original, encoding="utf-8")

            self.assertTrue(patch_worker(worker, "0.10.2rc1"))
            self.assertIn(MARKER, worker.read_text(encoding="utf-8"))
            self.assertFalse(patch_worker(worker, "0.10.2rc1"))
            self.assertTrue(restore_worker(worker))
            self.assertEqual(worker.read_text(encoding="utf-8"), original)

    def test_patch_rejects_unknown_version(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "worker_v1.py"
            worker.write_text(CALL, encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Refusing to patch"):
                patch_worker(worker, "0.10.3")

    def test_pooling_uses_native_scheduler(self):
        with tempfile.TemporaryDirectory() as directory:
            platform = Path(directory) / "platform.py"
            original = "before\n%safter\n" % POOLING_CONDITION
            platform.write_text(original, encoding="utf-8")

            self.assertTrue(patch_platform(platform, "0.10.2rc1"))
            self.assertIn(POOLING_MARKER, platform.read_text(encoding="utf-8"))
            self.assertFalse(patch_platform(platform, "0.10.2rc1"))
            self.assertTrue(restore_worker(platform))
            self.assertEqual(platform.read_text(encoding="utf-8"), original)
