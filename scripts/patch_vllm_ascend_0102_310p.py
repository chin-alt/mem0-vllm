#!/usr/bin/env python3
"""Apply narrowly scoped vLLM-Ascend 0.10.2rc1 fixes for 310P pooling."""

import argparse
import importlib.metadata
import importlib.util
from pathlib import Path


SUPPORTED_VERSION = "0.10.2rc1"
CALL = "        self._warm_up_atb()"
MARKER = "        pass  # MEMRANKER_310P_SKIP_ATB_WARMUP"
POOLING_CONDITION = "            if not model_config.is_multimodal_model and \\\n"
POOLING_REPLACEMENT = (
    "            if model_config.runner_type != \"pooling\" and \\\n"
    "                not model_config.is_multimodal_model and \\\n"
)
POOLING_MARKER = "model_config.runner_type != \"pooling\""
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


def patch_worker(worker_path: Path, version: str) -> bool:
    if version != SUPPORTED_VERSION:
        raise RuntimeError(
            "Refusing to patch vllm-ascend version %s; expected %s"
            % (version, SUPPORTED_VERSION)
        )

    source = worker_path.read_text(encoding="utf-8")
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
    if version != SUPPORTED_VERSION:
        raise RuntimeError(
            "Refusing to patch vllm-ascend version %s; expected %s"
            % (version, SUPPORTED_VERSION)
        )

    source = platform_path.read_text(encoding="utf-8")
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
    if version != SUPPORTED_VERSION:
        raise RuntimeError(
            "Refusing to patch vllm-ascend version %s; expected %s"
            % (version, SUPPORTED_VERSION)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()

    package_path = installed_package_path()
    worker_path = package_path / "worker" / "worker_v1.py"
    platform_path = package_path / "platform.py"
    model_runner_path = package_path / "worker" / "model_runner_v1.py"
    attention_path = package_path / "attention" / "attention_v1.py"
    if args.restore:
        restored = [
            path
            for path in (worker_path, platform_path, model_runner_path, attention_path)
            if restore_worker(path)
        ]
        if restored:
            for path in restored:
                print("[patch] restored %s" % path)
        else:
            print("[patch] no backup found")
        return

    version = importlib.metadata.version("vllm-ascend")
    worker_changed = patch_worker(worker_path, version)
    platform_changed = patch_platform(platform_path, version)
    encoder_changed = patch_encoder_pooling(
        model_runner_path, attention_path, version
    )
    if worker_changed:
        print("[patch] disabled unsupported 310P ATB warm-up in %s" % worker_path)
    else:
        print("[patch] 310P ATB warm-up patch already applied")
    if platform_changed:
        print("[patch] selected the native vLLM scheduler for pooling models in %s" % platform_path)
    else:
        print("[patch] pooling scheduler patch already applied")
    if encoder_changed:
        print("[patch] backported 310P encoder-only attention for pooling models")
    else:
        print("[patch] encoder-only attention patch already applied")


if __name__ == "__main__":
    main()
