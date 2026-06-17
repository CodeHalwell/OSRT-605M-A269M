"""Lightning AI entry for v6 midtrain phase 2 — extended continued-pretraining.

The base is badly undertrained (~1.7B tokens ≈ 0.3x Chinchilla for 278M active).
This adds ~1.1B tokens (~4000 steps @ seq 4096 ≈ ~$110) with a fresh re-warm
cosine on a reasoning/instruction-heavy mix (the existing knowledge sources
reweighted toward Nemotron STEM-SFT + InfiniByte-Reasoning + math) — the modern
annealing/decay phase. Full-sequence LM, resuming from midtrain_final. See
MidtrainExtendConfig.

USAGE (on the box):
    # sanity gate (30 steps) first:
    HF_TOKEN=... WANDB_API_KEY=... PYTHONPATH=src \
    python scripts/lightning_midtrain2.py --sanity
    # full run (~4000 steps):
    HF_TOKEN=... WANDB_API_KEY=... PYTHONPATH=src \
    python scripts/lightning_midtrain2.py --total-steps 4000

Prereqs: CUDA torch; checkpoints/v5/osrt_v5_midtrain_final.pt present;
v6_tokenizer_export/ present. Streams the Nemotron/FineWeb/Cosmopedia mix from
HF (set HF_TOKEN; the Nemotron configs are gated).
"""
from __future__ import annotations

import argparse
import os
import sys


class _LocalVol:
    def commit(self) -> None:  # noqa: D401
        pass

    def reload(self) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/v5")
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument("--total-steps", type=int, default=4000,
                    help="cosine target; ~4000 ≈ ~1.1B tokens ≈ ~$110")
    ap.add_argument("--peak-lr", type=float, default=None,
                    help="override AdamW peak LR (Muon scales proportionally). "
                         "The default 1e-4 re-warm was too hot for an annealed "
                         "base (ppl 30→34, flat); ~3e-5 is a gentler continue. "
                         "Applies on resume via the cosine at the resumed step.")
    ap.add_argument("--min-lr", type=float, default=None,
                    help="override cosine floor (default 1e-5)")
    ap.add_argument("--sanity", action="store_true",
                    help="run the 30-step MidtrainExtendSanityConfig gate")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from osrt.presets import build_config
    from osrt.train import run_pretrain_extend
    from osrt.train_config import MidtrainExtendConfig, MidtrainExtendSanityConfig

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — this is meant for a GPU box. "
              "Aborting.", flush=True)
        return 2
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '0')}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer: vocab={len(tok)}", flush=True)

    model_config = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )

    cfg = MidtrainExtendSanityConfig() if args.sanity else MidtrainExtendConfig()
    cfg.ckpt_dir = args.ckpt_dir
    cfg.pretrained_checkpoint = os.path.join(
        args.ckpt_dir, "osrt_v5_midtrain_final.pt")
    if not args.sanity:
        cfg.total_steps = args.total_steps

    # Optional LR override. The Muon group is kept proportional to the AdamW
    # peak (the config's fixed ratio), and the cosine floor scales too — so
    # lowering --peak-lr gives a gentler continue without yanking the annealed
    # base out of its basin. On resume the schedule is evaluated at the
    # resumed step, so the LR simply picks up lower from there.
    muon_ratio = cfg.muon_lr / cfg.peak_lr  # fixed in the config (~33)
    if args.peak_lr is not None:
        cfg.peak_lr = args.peak_lr
        cfg.muon_lr = args.peak_lr * muon_ratio
    if args.min_lr is not None:
        cfg.min_lr = args.min_lr
    cfg.muon_min_lr = cfg.min_lr * muon_ratio

    if args.sanity:
        cfg.wandb_log = False

    ph = cfg.phases["extend"]
    print(
        f"{cfg.__class__.__name__}: {cfg.total_steps} steps @ seq {ph['seq_len']}, "
        f"eff batch {ph['batch_size'] * ph['grad_accum_steps']}, peak_lr "
        f"{cfg.peak_lr} (→{cfg.min_lr}), reasoning/STEM/math share 0.75, "
        f"resume {cfg.pretrained_checkpoint}", flush=True,
    )

    run_pretrain_extend(model_config, cfg, _LocalVol(), args.tokenizer,
                        ckpt_dir=args.ckpt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
