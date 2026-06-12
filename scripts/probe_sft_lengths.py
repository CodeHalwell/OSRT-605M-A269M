"""Measure assembled-sequence token lengths for the SFT-v1 math sets, to
quantify how many HARD examples the seq_len filter (sft_data.py:642) drops.

Replicates the exact SFTStream assembly:
  <|system|>{persona}<|user|>{q}<|assistant|><|think|>{r}<|/think|><|answer|>{a}<|/answer|><eos>

Reports: count, length percentiles, and % dropped at seq_len thresholds
(2048 current, 3072, 4096) so we can see the hard-tail loss.

Run: HF_TOKEN=... PYTHONPATH=src python scripts/probe_sft_lengths.py --n 3000
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

# The two reasoning-on (math) sets — the "hard example" carriers.
DATASETS = [
    ("gsm8k", "openai/gsm8k", "main", "train", "gsm8k"),
    ("numina_math", "AI-MO/NuminaMath-CoT", None, "train", "numina_math"),
    ("evol_code", "nickrosh/Evol-Instruct-Code-80k-v1", None, "train", "evol_code"),
]
THRESHOLDS = [2048, 3072, 4096]


def _pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    i = min(len(sorted_vals) - 1, int(p / 100 * len(sorted_vals)))
    return sorted_vals[i]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer
    from osrt.sft_data import FORMAT_FN
    from osrt.system_prompts import sample_system_prompt
    import random

    tok = AutoTokenizer.from_pretrained("v6_tokenizer_export")
    rng = random.Random(0)
    persona_on = sample_system_prompt(rng, "on")[1]  # representative on-persona
    eos = tok.eos_token

    for name, repo, cfg, split, fmt in DATASETS:
        fn = FORMAT_FN[fmt]
        ds = (load_dataset(repo, cfg, split=split, streaming=True) if cfg
              else load_dataset(repo, split=split, streaming=True))
        lens, n, skipped_fmt = [], 0, 0
        for row in ds:
            try:
                q, r, a = fn(row)
            except Exception:
                continue
            if not q or not a:
                skipped_fmt += 1
                continue
            seq = (f"<|system|>{persona_on}<|user|>{q}<|assistant|>"
                   f"<|think|>{r}<|/think|><|answer|>{a}<|/answer|>{eos}")
            lens.append(len(tok.encode(seq, add_special_tokens=False)))
            n += 1
            if n >= args.n:
                break
        lens.sort()
        print(f"\n=== {name}  (n={n}) ===")
        print(f"  len p50={_pct(lens,50)}  p90={_pct(lens,90)}  "
              f"p95={_pct(lens,95)}  p99={_pct(lens,99)}  max={lens[-1] if lens else 0}")
        for t in THRESHOLDS:
            dropped = sum(1 for x in lens if x > t)
            print(f"  > {t:>4} tok (DROPPED): {dropped:>5} / {n}  = {100*dropped/max(n,1):5.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
