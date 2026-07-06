"""Modal-free entry for v6 midtrain phase 3 — the LONG capability push.

Runs run_pretrain_extend off-Modal (Colab H100, Lightning, EC2, on-prem). SFT v2
confirmed the base is UNDERTRAINED not capped (format_ok 1.0 but GSM8K ~0.05,
fluent-but-wrong output); the fix is more pretraining. This is the 12,600-step
cosine (+3.4B tok → ~1x Chinchilla) on the reasoning-dense mix, resuming from
midtrain2_step_1750. See MidtrainExtend3Config.

Designed to CHAIN across sessions: the resume-scan loads the highest
`osrt_v5_midtrain3_step_*.pt` in --ckpt-dir, so on Colab point --ckpt-dir at a
mounted Google Drive folder and every reconnect continues the same cosine.
ckpt_interval is 500 (a disconnect loses ≤500 steps); lower it with the config
if your sessions are short.

USAGE (Colab H100 — identical GPU to Modal, so the config runs as-is):
    # one-time in the notebook: mount Drive, clone repo, install deps, and put
    #   osrt_v5_midtrain2_step_1750.pt + v6_tokenizer_export/ under --ckpt-dir.
    # sanity gate (30 steps) first:
    HF_TOKEN=... WANDB_API_KEY=... PYTHONPATH=src \
    python scripts/lightning_midtrain3.py --sanity --ckpt-dir /content/drive/MyDrive/osrt/ckpt
    # full burst (resumes automatically across reconnects):
    HF_TOKEN=... WANDB_API_KEY=... PYTHONPATH=src \
    python scripts/lightning_midtrain3.py --ckpt-dir /content/drive/MyDrive/osrt/ckpt

Prereqs: CUDA torch; <ckpt-dir>/osrt_v5_midtrain2_step_1750.pt present;
v6_tokenizer_export/ present. Streams the Nemotron/FineWeb/Cosmopedia mix from
HF (set HF_TOKEN; the Nemotron configs are gated).
"""
from __future__ import annotations

import argparse
import os
import sys


class _LocalVol:
    """Stub Modal Volume — a Colab/Drive/box disk is persistent; commit() and
    reload() are no-ops. run_pretrain_extend only calls vol.commit()."""

    def commit(self) -> None:  # noqa: D401
        pass

    def reload(self) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/v5",
                    help="on Colab, a mounted Drive folder so checkpoints "
                         "survive disconnects and the run auto-resumes")
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument("--total-steps", type=int, default=None,
                    help="override the 12,600-step cosine target (default from "
                         "the config = +3.4B tok → ~1x Chinchilla)")
    ap.add_argument("--peak-lr", type=float, default=None,
                    help="override AdamW peak (Muon scales ×ratio). Default "
                         "5e-5 sustained capability LR; the long cosine barely "
                         "moves early, so it's effectively sustained.")
    ap.add_argument("--min-lr", type=float, default=None)
    ap.add_argument("--ckpt-interval", type=int, default=None,
                    help="lower (e.g. 250) if Colab sessions are short")
    ap.add_argument("--micro-batch", type=int, default=None,
                    help="override extend-phase batch_size (A100-40GB needs ~3 "
                         "vs the H100 default 6; keep eff-batch constant by "
                         "raising --grad-accum)")
    ap.add_argument("--grad-accum", type=int, default=None,
                    help="override extend-phase grad_accum_steps "
                         "(e.g. 22 to hold eff-batch 66 at --micro-batch 3)")
    ap.add_argument("--hf-repo", default=None,
                    help="private HF repo (e.g. USER/osrt-v6-ckpt) for "
                         "cross-session persistence: pull latest midtrain3 ckpt "
                         "(+ base) on start, push each new ckpt as it saves. "
                         "Essential on Colab (ephemeral VM disk + 24h cap). "
                         "Needs HF_TOKEN.")
    ap.add_argument("--sanity", action="store_true",
                    help="run the 30-step MidtrainExtend3SanityConfig gate")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from osrt.presets import build_config
    from osrt.train import run_pretrain_extend
    from osrt.train_config import (
        MidtrainExtend3Config,
        MidtrainExtend3SanityConfig,
    )

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — this is meant for a GPU box. "
              "Aborting.", flush=True)
        return 2
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer: vocab={len(tok)}", flush=True)

    model_config = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )

    cfg = (MidtrainExtend3SanityConfig() if args.sanity
           else MidtrainExtend3Config())
    cfg.ckpt_dir = args.ckpt_dir
    # GPU-fit override: A100-40GB can't hold the H100 batch-6; drop to
    # --micro-batch 3 --grad-accum 22 (same eff-batch 66). No-op on H100/A100-80.
    if args.micro_batch is not None:
        cfg.phases["extend"]["batch_size"] = args.micro_batch
    if args.grad_accum is not None:
        cfg.phases["extend"]["grad_accum_steps"] = args.grad_accum
    cfg.pretrained_checkpoint = os.path.join(
        args.ckpt_dir, "osrt_v5_midtrain2_step_1750.pt")
    if not args.sanity and args.total_steps is not None:
        cfg.total_steps = args.total_steps
    if args.ckpt_interval is not None:
        cfg.ckpt_interval = args.ckpt_interval

    # LR override — Muon kept proportional to the AdamW peak; cosine floor
    # scales too. On resume the schedule is evaluated at the resumed step.
    muon_ratio = cfg.muon_lr / cfg.peak_lr  # fixed in the config (~33)
    if args.peak_lr is not None:
        cfg.peak_lr = args.peak_lr
        cfg.muon_lr = args.peak_lr * muon_ratio
    if args.min_lr is not None:
        cfg.min_lr = args.min_lr
    cfg.muon_min_lr = cfg.min_lr * muon_ratio

    if args.sanity:
        cfg.wandb_log = False

    # HF cross-session persistence (Colab): pull the latest midtrain3 ckpt (and
    # the base if absent) BEFORE the resume-scan runs, and start a background
    # daemon that pushes each new ckpt to the repo as it saves.
    if args.hf_repo:
        from hf_ckpt_sync import pull_latest, start_push_daemon
        pull_latest(args.hf_repo, args.ckpt_dir, "osrt_v5_midtrain3",
                    base_name="osrt_v5_midtrain2_step_1750.pt")
        if not args.sanity:
            start_push_daemon(args.hf_repo, args.ckpt_dir, "osrt_v5_midtrain3")

    ph = cfg.phases["extend"]
    print(
        f"{cfg.__class__.__name__}: {cfg.total_steps} steps @ seq "
        f"{ph['seq_len']}, eff batch {ph['batch_size'] * ph['grad_accum_steps']}"
        f", peak_lr {cfg.peak_lr} (→{cfg.min_lr}), reasoning/STEM/math 0.75, "
        f"ckpt every {cfg.ckpt_interval}, resume {cfg.pretrained_checkpoint}",
        flush=True,
    )

    run_pretrain_extend(model_config, cfg, _LocalVol(), args.tokenizer,
                        ckpt_dir=args.ckpt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
