"""Run the reasoning-on/off GSM8K eval on a LOCAL SFT checkpoint (MPS/CPU).

Recovers the step-2000 eval the training loop never fired, and lets us compare
checkpoints head-to-head. Builds the model exactly like lightning_sft.py
(v6 preset + v6 tokenizer), loads <ckpt>['model_state_dict'] (native HRA), and
calls run_reasoning_eval.

Run: HF_TOKEN=... PYTHONPATH=src python scripts/local_sft_eval.py \
        --ckpt checkpoints/v5/osrt_v5_sft_v1_final.pt --n 100
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, _, v = _l.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v5/osrt_v5_sft_v1_final.pt")
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config
    from osrt.sft_eval import run_reasoning_eval

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = build_config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
    )
    model = OSRTForCausalLM(cfg)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(
            f"  load: {len(missing)} missing, {len(unexpected)} unexpected", flush=True
        )
        if missing:
            print("   missing[:5]:", missing[:5], flush=True)
        if unexpected:
            print("   unexpected[:5]:", unexpected[:5], flush=True)
    else:
        print("  clean load: all keys matched.", flush=True)

    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    model = model.to(device).eval()
    print(
        f"loaded {args.ckpt}  (stage={ck.get('training_stage')}, "
        f"steps={ck.get('total_steps')})",
        flush=True,
    )

    res = run_reasoning_eval(
        model,
        tok,
        device,
        n_problems=args.n,
        max_new_tokens=args.max_new_tokens,
    )
    print("\n=== reasoning eval ===", flush=True)
    for k, v in res.items():
        print(f"  {k:>28}: {v}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
