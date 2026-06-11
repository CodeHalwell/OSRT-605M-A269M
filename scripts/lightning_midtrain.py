"""Lightning AI entry point for finishing v6 mid-training (Modal-free).

Runs run_pretrain_extend directly on a Lightning box (or any single-GPU host),
bypassing the Modal @app.function wrapper. The training loop is plain PyTorch;
the only Modal-ism it uses is `vol.commit()` (flush volume), which we stub since
a Lightning box has a persistent local disk.

USAGE (on the box, after env + checkpoint + tokenizer are in place):
    HF_TOKEN=<halld token> \
    HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=/teamspace/studios/this_studio/hf_cache \
    PYTHONPATH=src python scripts/lightning_midtrain.py \
        --ckpt-dir checkpoints/v5 \
        --tokenizer v6_tokenizer_export \
        --total-steps 4500

Prereqs on the box:
  1. CUDA torch (NOT the mac/MPS build) — install from pyproject/uv.lock.
  2. checkpoints/v5/osrt_v5_midtrain_rescue_step_3978.pt present (the resume
     point). run_pretrain_extend's stage_prefix scan picks the highest
     midtrain[_rescue]_step_*.pt automatically.
  3. v6_tokenizer_export/ present (in the repo).
  4. (recommended) gated Nemotron configs snapshotted to HF_DATASETS_CACHE via
     scripts/snapshot_gated_datasets.py, with HF_HUB_OFFLINE=1 set — kills the
     cold-start connection storm. Without it, set HF_HUB_OFFLINE=0 and live-
     stream (re-exposes the storm risk).

--total-steps re-targets the cosine: with the checkpoint at step 3978, passing
4500 anneals LR to min_lr (2e-5) exactly at 4500 (a properly-finished base),
~522 steps. Pass 5500 to ride the original full schedule.
"""
from __future__ import annotations

import argparse
import os
import sys


class _LocalVol:
    """Stub for the Modal Volume: a Lightning box's disk is already persistent,
    so commit()/reload() are no-ops. run_pretrain_extend only ever calls
    vol.commit()."""

    def commit(self) -> None:  # noqa: D401
        pass

    def reload(self) -> None:
        pass


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="checkpoints/v5",
                    help="dir holding osrt_v5_midtrain*_step_*.pt (resume scan)")
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument("--total-steps", type=int, default=4500,
                    help="cosine target; 4500 anneals to min at 4500 from the "
                         "3978 resume point (~522 steps). 5500 = full schedule.")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from osrt.presets import build_config
    from osrt.train import run_pretrain_extend
    from osrt.train_config import MidtrainConfig

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — this is meant for a GPU box. "
              "Aborting to avoid a multi-day CPU run.", flush=True)
        return 2
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    print(f"HF_HUB_OFFLINE={os.environ.get('HF_HUB_OFFLINE', '0')} "
          f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE', '(default)')}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer: vocab={len(tok)}", flush=True)

    model_config = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )

    cfg = MidtrainConfig()
    cfg.total_steps = args.total_steps  # re-target cosine for the budget
    # pretrained_checkpoint is the startup existence-check; the resume scan
    # then loads the highest midtrain[_rescue] ckpt in ckpt_dir. Point the
    # existence-check at the rescue ckpt we're resuming from.
    cfg.pretrained_checkpoint = os.path.join(
        args.ckpt_dir, "osrt_v5_midtrain_rescue_step_3978.pt")
    cfg.wandb_run_name = "osrt-v6-midtrain-lightning"

    print(f"Resuming midtrain → total_steps={cfg.total_steps} "
          f"(cosine anneals to {cfg.min_lr} at {cfg.total_steps}), "
          f"ckpt_dir={args.ckpt_dir}", flush=True)

    run_pretrain_extend(model_config, cfg, _LocalVol(), args.tokenizer,
                        ckpt_dir=args.ckpt_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
