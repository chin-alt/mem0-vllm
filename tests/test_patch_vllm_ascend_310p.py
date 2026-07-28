from pathlib import Path
import tempfile
import unittest

from scripts.patch_vllm_ascend_0102_310p import (
    CALL,
    ATTENTION_GUARD,
    ENCODER_FORWARD_ANCHOR,
    ENCODER_FORWARD_MARKER,
    ENCODER_IMPORT,
    ENCODER_INIT,
    ENCODER_MARKER,
    ENCODER_METHOD_ANCHOR,
    MARKER,
    METADATA_BLOCK,
    METADATA_FIELDS,
    POOLING_CONDITION,
    POOLING_MARKER,
    BUILDER_BRANCH,
    patch_encoder_pooling,
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

    def test_encoder_pooling_backport_is_idempotent_and_restorable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runner = root / "model_runner_v1.py"
            attention = root / "attention_v1.py"
            runner_original = "\n".join(
                (
                    ENCODER_IMPORT,
                    ENCODER_INIT,
                    ENCODER_METHOD_ANCHOR,
                    METADATA_BLOCK,
                    METADATA_FIELDS,
                    BUILDER_BRANCH,
                )
            )
            attention_original = "\n".join(
                (ATTENTION_GUARD, ENCODER_FORWARD_ANCHOR)
            )
            runner.write_text(runner_original, encoding="utf-8")
            attention.write_text(attention_original, encoding="utf-8")

            self.assertTrue(
                patch_encoder_pooling(runner, attention, "0.10.2rc1")
            )
            self.assertIn(ENCODER_MARKER, runner.read_text(encoding="utf-8"))
            self.assertIn(
                ENCODER_FORWARD_MARKER, attention.read_text(encoding="utf-8")
            )
            self.assertFalse(
                patch_encoder_pooling(runner, attention, "0.10.2rc1")
            )
            self.assertTrue(restore_worker(runner))
            self.assertTrue(restore_worker(attention))
            self.assertEqual(runner.read_text(encoding="utf-8"), runner_original)
            self.assertEqual(
                attention.read_text(encoding="utf-8"), attention_original
            )
