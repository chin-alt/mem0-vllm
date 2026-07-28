from __future__ import annotations

import argparse
import platform
import sys

from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the Ascend 310P GTE inference environment.")
    parser.add_argument("--model_path", default="")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--attention_backend", choices=["pfa", "eager", "sdpa"], default="pfa")
    parser.add_argument("--jit_compile", action="store_true")
    args = parser.parse_args()

    import torch
    import torch_npu
    import transformers
    import tokenizers

    print("python:", platform.python_version())
    print("machine:", platform.machine())
    print("torch:", torch.__version__)
    print("torch_npu:", torch_npu.__version__)
    print("transformers:", transformers.__version__)
    print("tokenizers:", tokenizers.__version__)
    print("npu_available:", torch.npu.is_available())
    print("npu_count:", torch.npu.device_count())
    if not torch.npu.is_available():
        raise SystemExit("NPU is not available")

    torch.npu.set_device(args.device)
    set_compile_mode = getattr(torch.npu, "set_compile_mode", None)
    if set_compile_mode is not None:
        set_compile_mode(jit_compile=args.jit_compile)
    print("attention_backend:", args.attention_backend)
    print("jit_compile:", args.jit_compile)
    value = torch.arange(8, dtype=torch.float16, device=args.device)
    print("npu_smoke:", (value * 2).cpu().tolist())

    if not args.model_path:
        print("model_smoke: skipped (pass --model_path to enable)")
        return

    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="eager" if args.attention_backend == "pfa" else args.attention_backend,
    )
    if args.attention_backend == "pfa":
        src_dir = Path(__file__).resolve().parents[1] / "src"
        sys.path.insert(0, str(src_dir))
        from evaluate_business_gte_npu import install_gte_pfa_attention

        patched = install_gte_pfa_attention(model, torch, torch_npu)
        print("pfa_layers:", patched)
    model = model.eval().to(args.device)
    pairs = [
        ("中国的首都是哪里？", "北京是中华人民共和国的首都。"),
        ("中国的首都是哪里？", "木星是太阳系中最大的行星。"),
    ]
    encoded = tokenizer(
        [query for query, _ in pairs],
        [doc for _, doc in pairs],
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt",
    )
    encoded = {key: value.to(args.device) for key, value in encoded.items()}
    with torch.inference_mode():
        scores = model(**encoded, return_dict=True).logits.reshape(-1).float().cpu()
    torch.npu.synchronize()
    print("model_logits:", scores.tolist())
    if not bool(torch.isfinite(scores).all()):
        raise SystemExit("Model produced NaN/Inf logits")
    if not scores[0] > scores[1]:
        raise SystemExit("GTE semantic smoke test failed: relevant score is not greater than irrelevant score")
    print("model_smoke: PASS")


if __name__ == "__main__":
    main()
