"""Stream N rows per SFT-v1 dataset, run each through its format_fn, and show
the (question, reasoning, answer) triple — confirms the parsers produce sane
SFT examples on REAL rows before a long training run.

Run: HF_TOKEN=... PYTHONPATH=src python scripts/probe_sft_rows.py --n 10
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

# Mirror SFTv1Config.datasets (hf_id, hf_config, split, format, reasoning_mode)
DATASETS = [
    ("tulu3-sft", "allenai/tulu-3-sft-mixture", None, "train", "tulu", "off"),
    ("openhermes", "teknium/OpenHermes-2.5", None, "train", "openhermes", "off"),
    ("gsm8k", "openai/gsm8k", "main", "train", "gsm8k", "on"),
    ("numina_math", "AI-MO/NuminaMath-CoT", None, "train", "numina_math", "on"),
    (
        "evol_code",
        "nickrosh/Evol-Instruct-Code-80k-v1",
        None,
        "train",
        "evol_code",
        "off",
    ),
]


def _clip(s: str, n: int = 220) -> str:
    s = (s or "").replace("\n", "\\n")
    return s if len(s) <= n else s[:n] + f"…(+{len(s) - n} chars)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    from datasets import load_dataset

    from osrt.sft_data import FORMAT_FN

    for name, repo, cfg, split, fmt, mode in DATASETS:
        print("\n" + "=" * 78)
        repo_label = f"{repo}{'/' + cfg if cfg else ''}"
        print(f"{name}  [{repo_label}]  format={fmt}  reasoning={mode}")
        print("=" * 78)
        fn = FORMAT_FN[fmt]
        try:
            ds = (
                load_dataset(repo, cfg, split=split, streaming=True)
                if cfg
                else load_dataset(repo, split=split, streaming=True)
            )
        except Exception as e:
            print(f"  STREAM FAILED: {type(e).__name__}: {str(e)[:150]}")
            continue
        shown = 0
        skipped = 0
        for row in ds:
            try:
                q, r, a = fn(row)
            except Exception:  # noqa: BLE001
                skipped += 1
                continue
            if not q or not a:
                skipped += 1  # SFTStream would drop these (empty q/a)
                continue
            shown += 1
            think_state = "(empty)" if not r.strip() else f"{len(r)} chars"
            print(f"\n  [{shown}] think={think_state}")
            print(f"      Q: {_clip(q)}")
            if r.strip():
                print(f"      R: {_clip(r)}")
            print(f"      A: {_clip(a)}")
            if shown >= args.n:
                break
        print(f"\n  -> shown {shown}, skipped {skipped} (empty/multi-turn/parse-fail)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
