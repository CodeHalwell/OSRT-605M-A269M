"""Local CPU smoke test for the v6 midtrain checkpoint.

Builds the model exactly as app.py::pretrain does (OSRT_605M_A288M preset +
tokenizer ids + fused_ce), loads the checkpoint with native HRA (NO inject_hra
— hra_native path), and greedy-generates on a few math/STEM/code prompts to
confirm the checkpoint loads cleanly and produces coherent continuations.

Run: PYTHONPATH=src python scripts/smoke_midtrain_ckpt.py
CPU/MPS, no network, no Modal. ~601M params in bf16 ≈ 1.2GB RAM for weights.
"""

from __future__ import annotations

import os
import sys
import time

import torch
from transformers import AutoTokenizer

from osrt.model import OSRTForCausalLM
from osrt.presets import build_config
from osrt.train import load_model_state_or_raise

CKPT = "checkpoints/v5/osrt_v5_midtrain_rescue_step_3978.pt"
TOK = "v6_tokenizer_export"

PROMPTS = [
    "The derivative of f(x) = x^2 is",
    "To solve the equation 2x + 6 = 14, we",
    "Photosynthesis is the process by which plants",
    "def fibonacci(n):\n    ",
    "The capital of France is",
]


def _pick_device() -> torch.device:
    # SMOKE_DEVICE env overrides: "mps" | "cpu" | "cuda". Default: prefer MPS
    # on Apple silicon (much faster for generation), else CPU. MPS sometimes
    # lacks an op for this arch; if so we catch and fall back to CPU in main.
    want = os.environ.get("SMOKE_DEVICE", "").lower()
    if want:
        return torch.device(want)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    dev = _pick_device()
    # MPS needs this for any op without a native kernel to fall back to CPU
    # instead of erroring mid-generation.
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"device={dev} (MPS fallback enabled)")

    tok = AutoTokenizer.from_pretrained(TOK)
    print(
        f"tokenizer: vocab={len(tok)} bos={tok.bos_token_id} "
        f"eos={tok.eos_token_id} pad={tok.pad_token_id}"
    )

    cfg = build_config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
    )
    print("building model (OSRT_605M_A288M, native HRA rank=256)...")
    model = OSRTForCausalLM(cfg).to(device=dev)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,} ({n_params / 1e6:.0f}M)")

    print(f"loading checkpoint {CKPT} ...")
    t = time.time()
    ckpt = torch.load(CKPT, map_location=dev, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    saved_step = ckpt.get("step", "?") if isinstance(ckpt, dict) else "?"
    # native HRA: keys must match with NO inject_hra. This raises on any drift.
    load_model_state_or_raise(model, state, context=f"smoke load {CKPT}")
    print(
        f"  CLEAN LOAD: all keys matched (saved step={saved_step}, "
        f"{time.time() - t:.1f}s)"
    )

    model.eval()
    model.train(False)  # disable MoE capacity drops for inference

    print("\n=== greedy generation (max_new_tokens=40) ===")
    ok = 0
    for p in PROMPTS:
        ids = torch.tensor([tok(p).input_ids], dtype=torch.long, device=dev)
        t = time.time()
        with torch.no_grad():
            out = model.generate(
                ids, max_new_tokens=40, temperature=0.0, eos_token_id=tok.eos_token_id
            )
        gen = tok.decode(out[0, ids.shape[1] :], skip_special_tokens=True)
        dt = time.time() - t
        coherent = len(gen.strip()) > 0
        ok += coherent
        print(f"\nPROMPT: {p!r}")
        print(f"  -> {gen!r}")
        print(f"  ({dt:.1f}s, {'non-empty' if coherent else 'EMPTY'})")

    print(
        f"\n=== SMOKE RESULT: loaded clean + {ok}/{len(PROMPTS)} "
        f"non-empty generations ==="
    )
    return 0 if ok == len(PROMPTS) else 1


if __name__ == "__main__":
    sys.exit(main())
