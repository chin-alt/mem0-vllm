#!/usr/bin/env python3
"""Apply narrowly scoped legacy vLLM-Ascend fixes for 310P pooling."""

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path

from packaging.version import InvalidVersion, Version


STABLE_ASCEND_VERSION = "0.10.0rc1"
REGRESSION_ASCEND_VERSION = "0.10.2rc1"
SUPPORTED_ASCEND_VERSIONS = (
    STABLE_ASCEND_VERSION,
    REGRESSION_ASCEND_VERSION,
)
SUPPORTED_VLLM_VERSIONS = ("0.10.0", "0.10.2")
CALL = "        self._warm_up_atb()"
MARKER = "        pass  # MEMRANKER_310P_SKIP_ATB_WARMUP"
STABLE_WORKER_FINGERPRINT = "    def compile_or_warm_up_model(self) -> None:\n"
STABLE_WORKER_DUMMY_RUN = "            self.model_runner._dummy_run(size)\n"
POOLING_CONDITION = "            if not model_config.is_multimodal_model and \\\n"
POOLING_REPLACEMENT = (
    "            if model_config.runner_type != \"pooling\" and \\\n"
    "                not model_config.is_multimodal_model and \\\n"
)
POOLING_MARKER = "model_config.runner_type != \"pooling\""
STABLE_PLATFORM_SCHEDULER = (
    "        if ascend_config.ascend_scheduler_config.enabled:\n"
)
ENCODER_IMPORT = (
    "from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,\n"
    "                                        KVCacheConfig, KVCacheSpec, MambaSpec)"
)
ENCODER_IMPORT_REPLACEMENT = (
    "from vllm.v1.kv_cache_interface import (AttentionSpec, EncoderOnlyAttentionSpec,\n"
    "                                        FullAttentionSpec, KVCacheConfig,\n"
    "                                        KVCacheGroupSpec, KVCacheSpec, MambaSpec)"
)
ENCODER_INIT = (
    "        self.may_reinitialize_input_batch(kv_cache_config)\n"
    "        self.initialize_attn_backend(kv_cache_config)"
)
ENCODER_INIT_REPLACEMENT = (
    "        self.may_reinitialize_input_batch(kv_cache_config)\n"
    "        self.may_add_encoder_only_layers_to_kv_cache_config()\n"
    "        self.initialize_attn_backend(kv_cache_config)"
)
ENCODER_METHOD_ANCHOR = (
    "    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:\n"
)
ENCODER_METHOD = '''    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """Register encoder-only layers for metadata without allocating KV cache."""
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        encoder_only_attn_specs = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype,
                    use_mla=use_mla)
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        if encoder_only_attn_specs:
            if len(encoder_only_attn_specs) != 1:
                raise RuntimeError("Only one encoder-only attention spec is supported")
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec))

'''
METADATA_BLOCK = '''            blk_table = self.input_batch.block_table[kv_cache_group_id]
            blk_table_tensor = blk_table.get_device_tensor()
            slot_mapping = blk_table.slot_mapping_cpu[:
                                                      total_num_scheduled_tokens]
            self.slot_mapping_cpu[:total_num_scheduled_tokens].copy_(
                slot_mapping)
            # # Fill unused with -1. Needed for reshape_and_cache in full cuda
            # # graph mode.
            # blk_table.slot_mapping[total_num_scheduled_tokens:].fill_(-1)

            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
'''
METADATA_BLOCK_REPLACEMENT = '''            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only attention has no KV cache. These tensors only
                # satisfy the common metadata contract used by the backend.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1), dtype=torch.int32, device=self.device)
                current_slot_mapping = torch.zeros(
                    total_num_scheduled_tokens, dtype=torch.int64)
                current_attn_mask = torch.zeros_like(self.attn_mask)
            else:
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor()
                slot_mapping = blk_table.slot_mapping_cpu[:
                                                          total_num_scheduled_tokens]
                self.slot_mapping_cpu[:total_num_scheduled_tokens].copy_(
                    slot_mapping)
                current_slot_mapping = self.slot_mapping_cpu
                current_attn_mask = self.attn_mask

            # Make AscendCommonAttentionMetadata
            common_attn_metadata = AscendCommonAttentionMetadata(
'''
METADATA_FIELDS = '''                slot_mapping_cpu=self.slot_mapping_cpu,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                positions=self.positions,
                attn_mask=self.attn_mask,
'''
METADATA_FIELDS_REPLACEMENT = '''                slot_mapping_cpu=current_slot_mapping,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                positions=self.positions,
                attn_mask=current_attn_mask,
'''
BUILDER_BRANCH = '''                if isinstance(builder, GDNAttentionMetadataBuilder):
'''
BUILDER_BRANCH_REPLACEMENT = '''                if isinstance(kv_cache_group_spec.kv_cache_spec,
                              EncoderOnlyAttentionSpec):
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata)
                elif isinstance(builder, GDNAttentionMetadataBuilder):
'''
ATTENTION_GUARD = '''            if attn_type != AttentionType.DECODER:
                raise NotImplementedError("Encoder self-attention and "
                                          "encoder/decoder cross-attention "
                                          "are not implemented for "
                                          "PallasAttentionBackendImpl")
'''
ATTENTION_GUARD_REPLACEMENT = '''            if attn_type not in (AttentionType.DECODER,
                                  AttentionType.ENCODER_ONLY):
                raise NotImplementedError("Encoder/decoder cross-attention is not implemented")
'''
ENCODER_FORWARD_ANCHOR = '''            # TODO: Remove this contiguous in the future.
            value = value.contiguous()

            if len(kv_cache) > 1:
'''
ENCODER_FORWARD_REPLACEMENT = '''            # TODO: Remove this contiguous in the future.
            value = value.contiguous()

            if attn_type == AttentionType.ENCODER_ONLY:
                output = self._forward_prefill_no_cache(
                    query, key, value, attn_metadata, output, num_tokens)
                ori_output[:, :, :] = output[:num_tokens, :, :]
                return ori_output.view(num_tokens, self.hidden_size)

            if len(kv_cache) > 1:
'''
ENCODER_MARKER = "may_add_encoder_only_layers_to_kv_cache_config"
ENCODER_FORWARD_MARKER = "attn_type == AttentionType.ENCODER_ONLY"
QUANT_SCORE_ANCHOR = '        proj_name = prefix.split(".")[-1]\n'
QUANT_SCORE_MARKER = "MEMRANKER_KEEP_SCORE_HEAD_FLOAT"
QUANT_SCORE_GUARD = (
    '        if proj_name == "score":\n'
    "            # The Qwen3 reranker pooling conversion builds a tiny FP32 score\n"
    "            # head from the original lm_head yes/no rows. Quantizing\n"
    "            # this synthetic head is both unnecessary and incompatible\n"
    "            # with older ModelSlim descriptions that have no score key.\n"
    "            return True  # MEMRANKER_KEEP_SCORE_HEAD_FLOAT\n"
)
VLLM_LM_HEAD_BLOCK = '''        model.lm_head = ParallelLMHead(model.config.vocab_size,
                                       model.config.hidden_size,
                                       quant_config=quant_config)
'''
VLLM_LM_HEAD_BLOCK_REPLACEMENT = '''        model.lm_head = ParallelLMHead(model.config.vocab_size,
                                       model.config.hidden_size,
                                       quant_config=quant_config,
                                       prefix="lm_head")  # MEMRANKER_QWEN3_LM_HEAD_PREFIX
'''
VLLM_LM_HEAD_MARKER = "MEMRANKER_QWEN3_LM_HEAD_PREFIX"
VLLM_0100_LM_HEAD_BLOCK = '''        model.lm_head = ParallelLMHead(model.config.vocab_size,
                                       model.config.hidden_size,
                                       quant_config=model.quant_config)
'''
VLLM_0100_LM_HEAD_BLOCK_REPLACEMENT = '''        model.lm_head = ParallelLMHead(model.config.vocab_size,
                                       model.config.hidden_size,
                                       quant_config=model.quant_config,
                                       prefix="lm_head")  # MEMRANKER_QWEN3_LM_HEAD_PREFIX
'''
QWEN3_NORM_IMPORT = (
    "from vllm_ascend.ops.layernorm import AddRMSNormW8A8Quant\n"
)
QWEN3_NORM_IMPORT_REPLACEMENT = (
    "from vllm_ascend.ops.layernorm import AddRMSNormW8A8Quant\n"
    "from vllm_ascend.utils import is_310p\n"
)
QWEN3_QKV_NORM_BLOCK = '''        if isinstance(self.self_attn.qkv_proj.quant_method.quant_method,
                      AscendW8A8LinearMethod):
'''
QWEN3_QKV_NORM_BLOCK_REPLACEMENT = '''        if (not is_310p() and  # MEMRANKER_310P_DISABLE_ADD_RMS_NORM_QUANT
                isinstance(self.self_attn.qkv_proj.quant_method.quant_method,
                           AscendW8A8LinearMethod)):
'''
QWEN3_MLP_NORM_BLOCK = '''        if isinstance(self.mlp.gate_up_proj.quant_method.quant_method,
                      AscendW8A8LinearMethod):
'''
QWEN3_MLP_NORM_BLOCK_REPLACEMENT = '''        if (not is_310p() and  # MEMRANKER_310P_DISABLE_ADD_RMS_NORM_QUANT
                isinstance(self.mlp.gate_up_proj.quant_method.quant_method,
                           AscendW8A8LinearMethod)):
'''
QWEN3_NORM_MARKER = "MEMRANKER_310P_DISABLE_ADD_RMS_NORM_QUANT"


def has_supported_public_version(actual: str, expected: str) -> bool:
    """Accept PEP 440 local builds of the exact supported public version.

    The legacy 310P images can label a vLLM wheel with an ``+empty`` local
    suffix. The suffix identifies the local wheel build; the public API/source
    version remains the version validated below.
    """

    try:
        return Version(actual).public == Version(expected).public
    except InvalidVersion:
        return False


def require_supported_version(package: str, actual: str, expected: str) -> None:
    if not has_supported_public_version(actual, expected):
        raise RuntimeError(
            "Refusing to patch %s version %s; expected %s (local build suffix allowed)"
            % (package, actual, expected)
        )


def require_one_of_supported_versions(
    package: str, actual: str, expected_versions: tuple[str, ...]
) -> str:
    for expected in expected_versions:
        if has_supported_public_version(actual, expected):
            return expected
    raise RuntimeError(
        "Refusing to patch %s version %s; expected one of %s "
        "(local build suffix allowed)"
        % (package, actual, ", ".join(expected_versions))
    )


def require_supported_ascend_version(actual: str) -> str:
    return require_one_of_supported_versions(
        "vllm-ascend", actual, SUPPORTED_ASCEND_VERSIONS
    )


def patch_worker(worker_path: Path, version: str) -> bool:
    supported_version = require_supported_ascend_version(version)

    source = worker_path.read_text(encoding="utf-8")
    if supported_version == STABLE_ASCEND_VERSION:
        if (
            source.count(STABLE_WORKER_FINGERPRINT) != 1
            or source.count(STABLE_WORKER_DUMMY_RUN) != 1
            or CALL in source
        ):
            raise RuntimeError(
                "Stable vLLM-Ascend worker source does not match %s"
                % STABLE_ASCEND_VERSION
            )
        # 0.10.0rc1 never calls the unsupported _warm_up_atb helper.
        return False

    if MARKER in source:
        return False
    if source.count(CALL) != 1:
        raise RuntimeError(
            "Expected exactly one _warm_up_atb call in %s; found %d"
            % (worker_path, source.count(CALL))
        )

    backup = worker_path.with_suffix(worker_path.suffix + ".memranker.bak")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    worker_path.write_text(source.replace(CALL, MARKER), encoding="utf-8")
    return True


def patch_platform(platform_path: Path, version: str) -> bool:
    supported_version = require_supported_ascend_version(version)

    source = platform_path.read_text(encoding="utf-8")
    if supported_version == STABLE_ASCEND_VERSION:
        if (
            source.count(STABLE_PLATFORM_SCHEDULER) != 1
            or POOLING_CONDITION in source
        ):
            raise RuntimeError(
                "Stable vLLM-Ascend platform source does not match %s"
                % STABLE_ASCEND_VERSION
            )
        # 0.10.0rc1 uses the native scheduler unless its optional Ascend
        # scheduler is explicitly enabled.
        return False

    if POOLING_MARKER in source:
        return False
    if source.count(POOLING_CONDITION) != 1:
        raise RuntimeError(
            "Expected exactly one Ascend scheduler condition in %s; found %d"
            % (platform_path, source.count(POOLING_CONDITION))
        )

    backup = platform_path.with_suffix(platform_path.suffix + ".memranker.bak")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    platform_path.write_text(
        source.replace(POOLING_CONDITION, POOLING_REPLACEMENT), encoding="utf-8"
    )
    return True


def replace_exact(source: str, old: str, new: str, label: str) -> str:
    count = source.count(old)
    if count != 1:
        raise RuntimeError("Expected exactly one %s; found %d" % (label, count))
    return source.replace(old, new)


def patch_encoder_pooling(model_runner_path: Path, attention_path: Path, version: str) -> bool:
    require_supported_version(
        "vllm-ascend", version, REGRESSION_ASCEND_VERSION
    )

    runner_source = model_runner_path.read_text(encoding="utf-8")
    attention_source = attention_path.read_text(encoding="utf-8")
    if ENCODER_MARKER in runner_source and ENCODER_FORWARD_MARKER in attention_source:
        return False

    runner_source = replace_exact(
        runner_source, ENCODER_IMPORT, ENCODER_IMPORT_REPLACEMENT, "encoder-only import"
    )
    runner_source = replace_exact(
        runner_source, ENCODER_INIT, ENCODER_INIT_REPLACEMENT, "encoder-only initialization"
    )
    runner_source = replace_exact(
        runner_source,
        ENCODER_METHOD_ANCHOR,
        ENCODER_METHOD + ENCODER_METHOD_ANCHOR,
        "initialize_kv_cache anchor",
    )
    runner_source = replace_exact(
        runner_source, METADATA_BLOCK, METADATA_BLOCK_REPLACEMENT, "metadata block"
    )
    runner_source = replace_exact(
        runner_source, METADATA_FIELDS, METADATA_FIELDS_REPLACEMENT, "metadata fields"
    )
    runner_source = replace_exact(
        runner_source, BUILDER_BRANCH, BUILDER_BRANCH_REPLACEMENT, "metadata builder branch"
    )
    attention_source = replace_exact(
        attention_source, ATTENTION_GUARD, ATTENTION_GUARD_REPLACEMENT, "attention type guard"
    )
    attention_source = replace_exact(
        attention_source,
        ENCODER_FORWARD_ANCHOR,
        ENCODER_FORWARD_REPLACEMENT,
        "encoder attention forward",
    )

    for path, source in (
        (model_runner_path, runner_source),
        (attention_path, attention_source),
    ):
        backup = path.with_suffix(path.suffix + ".memranker.bak")
        if not backup.exists():
            backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.write_text(source, encoding="utf-8")
    return True


def patch_quant_score_head(quant_config_path: Path, version: str) -> bool:
    """Keep the converted Qwen3 sequence-classification head unquantized.

    vLLM creates ``score`` dynamically from two rows of ``lm_head`` for the
    original Qwen3 reranker. The supported legacy vllm-ascend releases
    otherwise look up
    ``score.weight`` in ``quant_model_description.json`` and may either raise a
    KeyError or try to quantize the synthetic one-row head.
    """
    require_supported_ascend_version(version)

    source = quant_config_path.read_text(encoding="utf-8")
    if QUANT_SCORE_MARKER in source:
        return False
    if source.count(QUANT_SCORE_ANCHOR) != 1:
        raise RuntimeError(
            "Expected exactly one quant score anchor in %s; found %d"
            % (quant_config_path, source.count(QUANT_SCORE_ANCHOR))
        )

    backup = quant_config_path.with_suffix(quant_config_path.suffix + ".memranker.bak")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    quant_config_path.write_text(
        source.replace(QUANT_SCORE_ANCHOR, QUANT_SCORE_ANCHOR + QUANT_SCORE_GUARD),
        encoding="utf-8",
    )
    return True


def patch_vllm_qwen3_lm_head_prefix(adapters_path: Path, version: str) -> bool:
    """Give temporary Qwen3 pooling LM heads their real quantization key.

    vLLM 0.10.0/0.10.2 recreates ``lm_head`` while deriving the Qwen3 score
    head but omits ``prefix="lm_head"``. AscendQuantConfig consequently looks up
    ``".weight"`` and raises a KeyError before it can see that the exported
    ``lm_head.weight`` is FLOAT.
    """
    supported_version = require_one_of_supported_versions(
        "vllm", version, SUPPORTED_VLLM_VERSIONS
    )

    source = adapters_path.read_text(encoding="utf-8")
    if VLLM_LM_HEAD_MARKER in source:
        return False
    if supported_version == "0.10.0":
        old_block = VLLM_0100_LM_HEAD_BLOCK
        new_block = VLLM_0100_LM_HEAD_BLOCK_REPLACEMENT
    else:
        old_block = VLLM_LM_HEAD_BLOCK
        new_block = VLLM_LM_HEAD_BLOCK_REPLACEMENT
    if source.count(old_block) != 2:
        raise RuntimeError(
            "Expected two Qwen3 pooling LM-head blocks in %s; found %d"
            % (adapters_path, source.count(old_block))
        )

    backup = adapters_path.with_suffix(adapters_path.suffix + ".memranker.bak")
    if not backup.exists():
        backup.write_text(source, encoding="utf-8")
    adapters_path.write_text(
        source.replace(old_block, new_block),
        encoding="utf-8",
    )
    return True


def patch_qwen3_310p_norm_quant(qwen3_path: Path, version: str) -> bool:
    """Avoid AddRmsNormQuant, whose binary is absent on the pinned 310P stack.

    Leaving the base RMSNorm modules in place makes AscendW8A8LinearMethod use
    its existing FP16 -> npu_quantize -> npu_quant_matmul fallback. This keeps
    static W8A8 matmuls while replacing only the unsupported fused norm op.
    """

    require_supported_ascend_version(version)
    source = qwen3_path.read_text(encoding="utf-8")
    if QWEN3_NORM_MARKER in source:
        return False

    source = replace_exact(
        source,
        QWEN3_NORM_IMPORT,
        QWEN3_NORM_IMPORT_REPLACEMENT,
        "Qwen3 AddRMSNormW8A8Quant import",
    )
    source = replace_exact(
        source,
        QWEN3_QKV_NORM_BLOCK,
        QWEN3_QKV_NORM_BLOCK_REPLACEMENT,
        "Qwen3 QKV fused norm quant block",
    )
    source = replace_exact(
        source,
        QWEN3_MLP_NORM_BLOCK,
        QWEN3_MLP_NORM_BLOCK_REPLACEMENT,
        "Qwen3 MLP fused norm quant block",
    )

    backup = qwen3_path.with_suffix(qwen3_path.suffix + ".memranker.bak")
    if not backup.exists():
        backup.write_text(qwen3_path.read_text(encoding="utf-8"), encoding="utf-8")
    qwen3_path.write_text(source, encoding="utf-8")
    return True


def restore_worker(worker_path: Path) -> bool:
    backup = worker_path.with_suffix(worker_path.suffix + ".memranker.bak")
    if not backup.exists():
        return False
    worker_path.write_text(backup.read_text(encoding="utf-8"), encoding="utf-8")
    return True


def installed_package_path() -> Path:
    spec = importlib.util.find_spec("vllm_ascend")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm_ascend is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def installed_vllm_path() -> Path:
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("vllm is not installed")
    return Path(next(iter(spec.submodule_search_locations)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    parser.add_argument(
        "--decoder-pooling-only",
        action="store_true",
        help=(
            "Apply only the shared 310P pooling fixes needed by decoder-only "
            "models such as Qwen3-Reranker; skip the GTE encoder attention backport."
        ),
    )
    args = parser.parse_args()

    package_path = installed_package_path()
    vllm_path = installed_vllm_path()
    worker_path = package_path / "worker" / "worker_v1.py"
    platform_path = package_path / "platform.py"
    model_runner_path = package_path / "worker" / "model_runner_v1.py"
    attention_path = package_path / "attention" / "attention_v1.py"
    quant_config_path = package_path / "quantization" / "quant_config.py"
    qwen3_path = package_path / "models" / "qwen3.py"
    adapters_path = vllm_path / "model_executor" / "models" / "adapters.py"
    if args.restore:
        restored = [
            path
            for path in (
                worker_path,
                platform_path,
                model_runner_path,
                attention_path,
                quant_config_path,
                qwen3_path,
                adapters_path,
            )
            if restore_worker(path)
        ]
        if restored:
            for path in restored:
                print("[patch] restored %s" % path)
        else:
            print("[patch] no backup found")
        return

    version = importlib.metadata.version("vllm-ascend")
    vllm_version = importlib.metadata.version("vllm")
    worker_changed = patch_worker(worker_path, version)
    platform_changed = patch_platform(platform_path, version)
    encoder_changed = False
    if not args.decoder_pooling_only:
        encoder_changed = patch_encoder_pooling(
            model_runner_path, attention_path, version
        )
    quant_score_changed = False
    lm_head_changed = False
    qwen3_norm_changed = patch_qwen3_310p_norm_quant(qwen3_path, version)
    if args.decoder_pooling_only:
        quant_score_changed = patch_quant_score_head(quant_config_path, version)
        lm_head_changed = patch_vllm_qwen3_lm_head_prefix(
            adapters_path, vllm_version
        )
    stable_runtime = has_supported_public_version(
        version, STABLE_ASCEND_VERSION
    )
    if stable_runtime:
        print("[patch] stable 0.10.0 worker has no unsupported ATB warm-up")
    elif worker_changed:
        print("[patch] disabled unsupported 310P ATB warm-up in %s" % worker_path)
    else:
        print("[patch] 310P ATB warm-up patch already applied")
    if stable_runtime:
        print("[patch] stable 0.10.0 uses the native scheduler by default")
    elif platform_changed:
        print("[patch] selected the native vLLM scheduler for pooling models in %s" % platform_path)
    else:
        print("[patch] pooling scheduler patch already applied")
    if args.decoder_pooling_only:
        print("[patch] skipped encoder-only attention backport for Qwen3 pooling")
    elif encoder_changed:
        print("[patch] backported 310P encoder-only attention for pooling models")
    else:
        print("[patch] encoder-only attention patch already applied")
    if args.decoder_pooling_only:
        if quant_score_changed:
            print("[patch] kept the converted Qwen3 reranker score head unquantized")
        else:
            print("[patch] Qwen3 reranker score-head patch already applied")
        if lm_head_changed:
            print("[patch] fixed the temporary Qwen3 pooling lm_head quantization key")
        else:
            print("[patch] Qwen3 pooling lm_head-prefix patch already applied")
    if qwen3_norm_changed:
        print("[patch] disabled unsupported Qwen3 AddRmsNormQuant on 310P")
    else:
        print("[patch] Qwen3 310P norm-quant fallback already applied")


if __name__ == "__main__":
    main()
