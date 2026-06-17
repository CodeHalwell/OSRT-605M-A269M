"""Variable-loop output-perturbation probe — how much does dropping recursive
loops change the model's predictions?

Companion to scripts/probe_cross_loop_kv.py. The KV probe showed the recursion
is a contracting iteration that has largely converged by loops 4-5 — which
*motivates* the variable-loop inference knob (run K<L loops to save decode
compute). This probe measures the knob's cost directly, forward-only, on the
base model (no reasoning/SFT model required):

  for K in {3,4,5}: KL( P(full L=6 loops) || P(K loops) ) and top-1 agreement,
  plus the held-out next-token CE/ppl at each K.

Small KL + high top-1 agreement at K=4/5 => dropping loops barely moves the
output distribution => the compute saving is nearly free distributionally. The
ACCURACY validation (does on>off survive fewer loops?) waits for a checkpoint
with a measurable reasoning delta — this is the distributional half, bankable now.

Run:
  HF_TOKEN=... PYTHONPATH=src python scripts/probe_loop_kl.py \
      --ckpt checkpoints/v5/osrt_v5_midtrain_final.pt --texts general --out /tmp/loop_kl.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, _, v = _l.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# Reuse the exact probe batches from the KV probe (scripts/ is on sys.path[0]
# when run as a script). DEFAULT_PROBE_TEXTS is the math-heavy set.
from probe_cross_loop_kv import (  # noqa: E402
    DEFAULT_PROBE_TEXTS as MATH_PROBE_TEXTS,
    GENERAL_PROBE_TEXTS,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v5/osrt_v5_midtrain_final.pt")
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument("--texts", choices=["math", "general", "mixed"], default="general")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from osrt.presets import build_config
    from osrt.model import OSRTForCausalLM
    from osrt.train import load_model_state_or_raise

    device = (torch.device("cuda") if torch.cuda.is_available()
              else torch.device("mps") if torch.backends.mps.is_available()
              else torch.device("cpu"))
    print(f"device={device}", flush=True)

    texts = {"math": MATH_PROBE_TEXTS, "general": GENERAL_PROBE_TEXTS,
             "mixed": MATH_PROBE_TEXTS + GENERAL_PROBE_TEXTS}[args.texts]
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = build_config(vocab_size=len(tok), real_vocab_size=len(tok),
                       bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                       pad_token_id=tok.pad_token_id)
    batch = (texts * ((args.batch // len(texts)) + 1))[: args.batch]
    enc = tok(batch, return_tensors="pt", padding="max_length",
              truncation=True, max_length=args.seq_len)
    input_ids = enc["input_ids"].to(device)
    attn = enc["attention_mask"].to(device)

    model = OSRTForCausalLM(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    load_model_state_or_raise(model, ck.get("model_state_dict", ck),
                              context=f"loop-kl {args.ckpt}")
    model = model.to(device).eval()

    full_loops = cfg.recursive_loops
    rv = cfg.real_vocab_size

    @torch.no_grad()
    def logits_at(k):
        out = model(input_ids=input_ids, attention_mask=attn, num_loops=k)
        return out.logits[..., :rv].float()

    # Valid NEXT-token positions: predict token t+1 from position t, both real.
    valid = (attn[:, 1:].bool()).reshape(-1)            # (B*(S-1),)
    gold = input_ids[:, 1:].reshape(-1)[valid]

    def ce_ppl(lg):
        lp = F.log_softmax(lg[:, :-1, :], dim=-1).reshape(-1, rv)[valid]
        ce = F.nll_loss(lp, gold)
        return float(ce), float(torch.exp(ce))

    full = logits_at(full_loops)
    full_lp = F.log_softmax(full[:, :-1, :], dim=-1).reshape(-1, rv)[valid]
    full_p = full_lp.exp()
    full_argmax = full[:, :-1, :].reshape(-1, rv)[valid].argmax(-1)
    ce_full, ppl_full = ce_ppl(full)

    rows = []
    for k in range(2, full_loops):
        lg = logits_at(k)
        lp = F.log_softmax(lg[:, :-1, :], dim=-1).reshape(-1, rv)[valid]
        kl = float((full_p * (full_lp - lp)).sum(-1).mean())   # KL(full || k)
        top1 = float((lg[:, :-1, :].reshape(-1, rv)[valid].argmax(-1)
                      == full_argmax).float().mean())
        ce_k, ppl_k = ce_ppl(lg)
        rows.append({"loops": k, "kl_full_given_k": round(kl, 4),
                     "top1_agree": round(top1, 4),
                     "ce": round(ce_k, 4), "ppl": round(ppl_k, 3)})

    report = {"ckpt": args.ckpt, "texts": args.texts, "full_loops": full_loops,
              "full_ce": round(ce_full, 4), "full_ppl": round(ppl_full, 3),
              "rows": rows}
    print("\n" + "=" * 60)
    print(f"VARIABLE-LOOP OUTPUT PERTURBATION (full L={full_loops}, "
          f"ppl={ppl_full:.2f})")
    print("=" * 60)
    print(f"{'loops':>5} {'KL(full||k)':>12} {'top1-agree':>11} {'ppl':>8}")
    for r in rows:
        print(f"{r['loops']:>5} {r['kl_full_given_k']:>12.4f} "
              f"{r['top1_agree']:>11.3f} {r['ppl']:>8.2f}")
    print(f"{full_loops:>5} {0.0:>12.4f} {1.0:>11.3f} {ppl_full:>8.2f}  (full)")
    print("=" * 60)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
