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
    QUANT_SCORE_ANCHOR,
    QUANT_SCORE_MARKER,
    STABLE_PLATFORM_SCHEDULER,
    STABLE_WORKER_DUMMY_RUN,
    STABLE_WORKER_FINGERPRINT,
    VLLM_LM_HEAD_BLOCK,
    VLLM_LM_HEAD_MARKER,
    BUILDER_BRANCH,
    patch_encoder_pooling,
    patch_platform,
    patch_quant_score_head,
    patch_vllm_qwen3_lm_head_prefix,
    patch_worker,
    restore_worker,
)


class PatchVllmAscendTests(unittest.TestCase):
    def test_stable_0100_needs_no_worker_or_scheduler_patch(self):
        with tempfile.TemporaryDirectory() as directory:
            worker = Path(directory) / "worker_v1.py"
            platform = Path(directory) / "platform.py"
            worker_source = (
                STABLE_WORKER_FINGERPRINT
                + "        for size in warmup_sizes:\n"
                + STABLE_WORKER_DUMMY_RUN
            )
            platform_source = (
                "def check_and_update_config():\n"
                + STABLE_PLATFORM_SCHEDULER
                + "            pass\n"
            )
            worker.write_text(worker_source, encoding="utf-8")
            platform.write_text(platform_source, encoding="utf-8")

            self.assertFalse(patch_worker(worker, "0.10.0rc1"))
            self.assertFalse(patch_platform(platform, "0.10.0rc1"))
            self.assertEqual(worker.read_text(encoding="utf-8"), worker_source)
            self.assertEqual(platform.read_text(encoding="utf-8"), platform_source)

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

    def test_quantized_reranker_score_head_stays_float(self):
        with tempfile.TemporaryDirectory() as directory:
            quant_config = Path(directory) / "quant_config.py"
            original = "def is_layer_skipped(self, prefix):\n%s    return False\n" % QUANT_SCORE_ANCHOR
            quant_config.write_text(original, encoding="utf-8")

            self.assertTrue(
                patch_quant_score_head(quant_config, "0.10.2rc1")
            )
            self.assertIn(
                QUANT_SCORE_MARKER,
                quant_config.read_text(encoding="utf-8"),
            )
            self.assertFalse(
                patch_quant_score_head(quant_config, "0.10.2rc1")
            )
            self.assertTrue(restore_worker(quant_config))
            self.assertEqual(
                quant_config.read_text(encoding="utf-8"), original
            )

    def test_qwen3_pooling_lm_head_has_quantization_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            adapters = Path(directory) / "adapters.py"
            original = VLLM_LM_HEAD_BLOCK + "between\n" + VLLM_LM_HEAD_BLOCK
            adapters.write_text(original, encoding="utf-8")

            self.assertTrue(
                patch_vllm_qwen3_lm_head_prefix(adapters, "0.10.2")
            )
            self.assertEqual(
                adapters.read_text(encoding="utf-8").count(VLLM_LM_HEAD_MARKER),
                2,
            )
            self.assertFalse(
                patch_vllm_qwen3_lm_head_prefix(adapters, "0.10.2")
            )
            self.assertTrue(restore_worker(adapters))
            self.assertEqual(adapters.read_text(encoding="utf-8"), original)
