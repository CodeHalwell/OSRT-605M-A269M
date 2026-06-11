"""Fairer base-model probes for the v6 midtrain checkpoint.

(a) Perplexity on held-out text snippets (math/STEM/code/general) — the right
    signal for a NON-instruction-tuned base model (greedy prompt-following is
    not what it was trained for).
(b) Sampling (temp 0.7, top_p 0.95) on the same prompts as the greedy smoke —
    to see whether the "circle-area template" lock-in is a greedy artifact.

Run: PYTORCH_ENABLE_MPS_FALLBACK=1 SMOKE_DEVICE=mps PYTHONPATH=src \
        python -u scripts/smoke_midtrain_probe.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from osrt.presets import build_config
from osrt.model import OSRTForCausalLM
from osrt.train import load_model_state_or_raise

CKPT = os.environ.get("PROBE_CKPT",
                      "checkpoints/v5/osrt_v5_midtrain_final.pt")
TOK = "v6_tokenizer_export"

# (a) Held-out snippets — coherent natural text the model should assign
# reasonable probability to. Hand-written (not from the training stream).
PPL_TEXTS = {
    "math": "The derivative of x squared is 2x. To differentiate a polynomial, "
            "multiply each term by its exponent and reduce the exponent by one.",
    "stem": "Photosynthesis converts carbon dioxide and water into glucose and "
            "oxygen using energy from sunlight, captured by chlorophyll in the "
            "chloroplasts of plant cells.",
    "code": "def factorial(n):\n    if n <= 1:\n        return 1\n    "
            "return n * factorial(n - 1)",
    "general": "France is a country in Western Europe. Its capital and largest "
               "city is Paris, which sits on the river Seine.",
}

GEN_PROMPTS = [
    "The derivative of f(x) = x^2 is",
    "To solve the equation 2x + 6 = 14, we",
    "The capital of France is",
]


def _pick_device():
    want = os.environ.get("SMOKE_DEVICE", "").lower()
    if want:
        return torch.device(want)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def perplexity(model, tok, text, dev) -> tuple[float, int]:
    """Teacher-forced perplexity on a single text (no special tokens added
    beyond BOS) using the model's own shift-aligned LM loss path."""
    ids = tok(text).input_ids
    if len(ids) < 2:
        return float("nan"), len(ids)
    x = torch.tensor([ids], dtype=torch.long, device=dev)
    out = model(x, labels=x)  # model shifts internally; returns mean CE loss
    loss = out.loss.item()
    return math.exp(min(loss, 20.0)), len(ids)


def main() -> int:
    dev = _pick_device()
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    print(f"device={dev}")

    tok = AutoTokenizer.from_pretrained(TOK)
    cfg = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )
    model = OSRTForCausalLM(cfg).to(device=dev)
    ckpt = torch.load(CKPT, map_location=dev, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    load_model_state_or_raise(model, state, context="probe load")
    model.eval(); model.train(False)
    print(f"loaded {CKPT} (step {ckpt.get('step','?')})\n")

    # ── (a) Perplexity on held-out snippets ──────────────────────────
    print("=== (a) PERPLEXITY on held-out text (lower = better) ===")
    ppls = []
    for name, text in PPL_TEXTS.items():
        t = time.time()
        ppl, ntok = perplexity(model, tok, text, dev)
        ppls.append(ppl)
        print(f"  {name:8s}: ppl={ppl:6.2f}  ({ntok} tok, {time.time()-t:.1f}s)")
    print(f"  mean ppl = {sum(ppls)/len(ppls):.2f}")
    print("  (foundation held-out eval was ppl ~43 @ seq2048; midtrain eval "
          "fell to ~32. These short hand-written snippets aren't the eval set "
          "— treat as a coherence sanity check, not the official number.)\n")

    # ── (b) Sampling vs the greedy lock-in ───────────────────────────
    print("=== (b) SAMPLING (temp=0.7, top_p=0.95, max_new=40) ===")
    for p in GEN_PROMPTS:
        ids = torch.tensor([tok(p).input_ids], dtype=torch.long, device=dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=40, temperature=0.7,
                                  top_p=0.95, eos_token_id=tok.eos_token_id)
        gen = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        print(f"\nPROMPT: {p!r}")
        print(f"  -> {gen!r}")
    print("\n=== PROBE DONE ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
