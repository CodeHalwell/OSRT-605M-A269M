"""Lightning AI entry for SFT v1 (Modal-free) — system-prompt instruction tuning.

Runs run_sft on a Lightning box (or any single-GPU host) without Modal. The SFT
loop is plain PyTorch; the only Modal-ism it uses is `vol.commit()`, stubbed by
_LocalVol (Lightning disk is persistent). Builds the model from the v6 preset,
loads midtrain_final with NATIVE HRA (hra_native=True → no inject_hra), and runs
the reasoning-on/off GSM8K eval every eval_interval.

USAGE (on the box, after env + checkpoint + tokenizer in place):
    HF_TOKEN=... WANDB_API_KEY=... PYTHONPATH=src \
    python scripts/lightning_sft.py --sanity        # 30-step gate first
    HF_TOKEN=... WANDB_API_KEY=... PYTHONPATH=src \
    python scripts/lightning_sft.py                 # full SFT v1 (2000 steps)

Prereqs:
  - CUDA torch (not MPS).
  - checkpoints/v5/osrt_v5_midtrain_final.pt present (the SFT base).
  - v6_tokenizer_export/ present.
  - HF_TOKEN (some SFT datasets stream from gated/large repos).

--sanity runs the 30-step SFTv1SanityConfig (no final ckpt, no eval) — MUST pass
before the paid full run: confirms the <|system|> turn builds, native-HRA loads
clean (all keys matched), masking is right, and VRAM fits.
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--sanity", action="store_true",
                    help="run the 30-step SFTv1SanityConfig gate")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer

    from osrt.presets import build_config
    from osrt.sft_train import run_sft
    from osrt.train_config import SFTv1Config, SFTv1SanityConfig

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — SFT is meant for a GPU box. Aborting.",
              flush=True)
        return 2
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"tokenizer: vocab={len(tok)}", flush=True)

    model_config = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )

    cfg = SFTv1SanityConfig() if args.sanity else SFTv1Config()
    cfg.ckpt_dir = args.ckpt_dir
    cfg.pretrained_checkpoint = (
        f"{args.ckpt_dir}/osrt_v5_midtrain_final.pt"
    )
    if args.sanity:
        cfg.wandb_log = False  # no W&B for the 30-step gate

    print(
        f"{cfg.__class__.__name__}: {cfg.total_steps} steps @ seq {cfg.seq_len}, "
        f"system_tag={cfg.system_tag}, hra_native={cfg.hra_native}, "
        f"base={cfg.pretrained_checkpoint}", flush=True,
    )
    run_sft(model_config, cfg, _LocalVol(), tok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
