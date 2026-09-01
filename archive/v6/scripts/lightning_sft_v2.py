"""Lightning AI entry for SFT v2 — reasoning distillation (Modal-free).

Runs run_pretrain_extend (the MOPD rollout-loader path) on a single-GPU box,
training the v6 midtrain base on rollouts/sft_v2.jsonl (coherent long teacher
CoT + reasoning-on/off personas; built by scripts/build_sft_v2_data.py). Native
HRA, seq 4096, gradient checkpointing — MidtrainConfig's proven v6 plumbing —
with a gentle SFT schedule (peak 1e-5).

USAGE (on the box, after env + checkpoint + tokenizer + data in place):
    # 0. regenerate the corpus deterministically (gitignored, not shipped):
    PYTHONPATH=src python scripts/build_sft_v2_data.py
    # 1. sanity gate (30 steps) — MUST pass before the paid run:
    HF_TOKEN=... WANDB_API_KEY=... PYTHONPATH=src \
    python scripts/lightning_sft_v2.py --sanity
    # 2. full run (~1500 steps):
    HF_TOKEN=... WANDB_API_KEY=... PYTHONPATH=src \
    python scripts/lightning_sft_v2.py

Prereqs on the box:
  - CUDA torch (not MPS); deps synced from the pinned uv.lock (datasets 4.6.1).
  - checkpoints/v5/osrt_v5_midtrain_final.pt present (the SFT-v2 base).
  - v6_tokenizer_export/ present.
  - rollouts/sft_v2.jsonl present (run build_sft_v2_data.py — it needs the v6
    tokenizer + rollouts/mopd_v1.jsonl + rollouts/system_prompt_sft.jsonl).

The reasoning-on/off GSM8K eval is NOT run in-loop (perplexity eval is disabled
in SFTv2Config). Eval the periodic checkpoints offline with:
    PYTHONPATH=src python scripts/local_sft_eval.py \
        --ckpt checkpoints/v5/osrt_v5_sft_v2_step_300.pt --n 100
"""

from __future__ import annotations

import argparse
import os
import sys


class _LocalVol:
    """Stub for the Modal Volume — a Lightning box's disk is persistent, so
    commit()/reload() are no-ops. run_pretrain_extend only calls vol.commit()."""

    def commit(self) -> None:  # noqa: D401
        pass

    def reload(self) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/v5")
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument(
        "--rollout",
        default="rollouts/sft_v2.jsonl",
        help="local path to the SFT-v2 corpus",
    )
    ap.add_argument(
        "--sanity", action="store_true", help="run the 30-step SFTv2SanityConfig gate"
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from osrt.presets import build_config
    from osrt.train import run_pretrain_extend
    from osrt.train_config import SFTv2Config, SFTv2SanityConfig

    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA not available — SFT v2 is meant for a GPU box. Aborting.",
            flush=True,
        )
        return 2
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    if not os.path.exists(args.rollout):
        print(
            f"ERROR: rollout corpus not found at {args.rollout}. "
            f"Run: PYTHONPATH=src python scripts/build_sft_v2_data.py",
            flush=True,
        )
        return 2

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer: vocab={len(tok)}", flush=True)

    model_config = build_config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
    )

    cfg = SFTv2SanityConfig() if args.sanity else SFTv2Config()
    cfg.ckpt_dir = args.ckpt_dir
    # midtrain2 step_1750 — best intact midtrain2 artifact (final save was
    # truncated on the volume); see SFTv2Config lineage note.
    cfg.pretrained_checkpoint = os.path.join(
        args.ckpt_dir, "osrt_v5_midtrain2_step_1750.pt"
    )
    cfg.rollout_dataset_path = args.rollout
    if args.sanity:
        cfg.wandb_log = False

    effective_batch = (
        cfg.phases["extend"]["batch_size"] * cfg.phases["extend"]["grad_accum_steps"]
    )
    print(
        f"{cfg.__class__.__name__}: {cfg.total_steps} steps @ seq "
        f"{cfg.phases['extend']['seq_len']}, eff batch "
        f"{effective_batch}, "
        f"peak_lr {cfg.peak_lr}, hra_native={cfg.hra_native}, "
        f"base={cfg.pretrained_checkpoint}, rollout={cfg.rollout_dataset_path}",
        flush=True,
    )

    run_pretrain_extend(
        model_config, cfg, _LocalVol(), args.tokenizer, ckpt_dir=args.ckpt_dir
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
