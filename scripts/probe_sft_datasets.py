"""Schema + length + license probe for candidate SFT datasets.

For a 601M model the load-bearing question is RESPONSE LENGTH (long-CoT blends
built for 30B+ teachers don't fit a 601M student) and FIELD SCHEMA (what the
reformatter must map into osrt's <|user|>/<|assistant|>/<|think|>/<|answer|>
template). This streams a sample of each candidate, reports field names, a
median/p90 response-token estimate (chars/4 heuristic — no tokenizer dep), and
the license off the dataset card.

Run: HF_TOKEN=... python scripts/probe_sft_datasets.py
Network-only, no GPU, ~1-2 min. Bounded by --n samples per dataset.
"""
from __future__ import annotations

import argparse
import os
import signal
import statistics
import sys
from pathlib import Path

# load .env for HF_TOKEN
_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, _, v = _l.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

# (key, repo, config-or-None, split) — the top-2 SFT candidates for a 601M
# instruction-following + general-chat SFT v1.
CANDIDATES = [
    # Scale-appropriate SFT candidates (built for sub-2B models or human-written).
    # The length probe decides fit for our 601M / 278M-active student.
    ("smoltalk", "HuggingFaceTB/smoltalk", "all", "train"),
    ("dolly-15k", "databricks/databricks-dolly-15k", None, "train"),
    ("tulu3-sft", "allenai/tulu-3-sft-mixture", None, "train"),
    ("openhermes2.5", "teknium/OpenHermes-2.5", None, "train"),
]

# likely response/output field names to look for, in priority order
_RESP_FIELDS = ("response", "output", "answer", "completion", "assistant",
                "messages", "conversations", "chosen", "target")
_PROMPT_FIELDS = ("prompt", "instruction", "input", "question", "context",
                  "messages", "conversations")


def _text_of(v) -> str:
    """Best-effort flatten of a field value to text for length estimation."""
    if isinstance(v, str):
        return v
    if isinstance(v, list):  # messages / conversations
        parts = []
        for m in v:
            if isinstance(m, dict):
                parts.append(str(m.get("content") or m.get("value") or ""))
            else:
                parts.append(str(m))
        return "\n".join(parts)
    if isinstance(v, dict):
        return str(v.get("content") or v.get("value") or v)
    return str(v)


def _resp_len_chars(ex: dict) -> int | None:
    for f in _RESP_FIELDS:
        if f in ex and ex[f]:
            return len(_text_of(ex[f]))
    return None


def probe_one(key, repo, cfg, split, n) -> None:
    from datasets import load_dataset
    print(f"\n{'='*70}\n{key}: {repo}" + (f" [{cfg}]" if cfg else ""))
    try:
        ds = (load_dataset(repo, cfg, split=split, streaming=True) if cfg
              else load_dataset(repo, split=split, streaming=True))
    except Exception as e:
        print(f"  STREAM FAILED: {type(e).__name__}: {str(e)[:160]}")
        return
    rows = []
    it = iter(ds)
    for _ in range(n):
        try:
            rows.append(next(it))
        except StopIteration:
            break
    if not rows:
        print("  no rows")
        return
    keys = list(rows[0].keys())
    print(f"  FIELDS: {keys}")
    # identify likely prompt/response fields present
    p_field = next((f for f in _PROMPT_FIELDS if f in keys), "(?)")
    r_field = next((f for f in _RESP_FIELDS if f in keys), "(?)")
    print(f"  likely prompt field: {p_field!r} | response field: {r_field!r}")
    # response length distribution (chars → ~tokens via /4)
    lens = [c for c in (_resp_len_chars(r) for r in rows) if c is not None]
    if lens:
        lens.sort()
        med = statistics.median(lens)
        p90 = lens[int(0.9 * (len(lens) - 1))]
        print(f"  response chars: median={med:.0f} (~{med/4:.0f} tok)  "
              f"p90={p90} (~{p90/4:.0f} tok)  max={max(lens)} (~{max(lens)/4:.0f} tok)")
        # 601M capacity-fit verdict
        med_tok = med / 4
        verdict = ("SHORT — good fit for 601M" if med_tok < 400
                   else "MEDIUM — usable, watch p90" if med_tok < 1200
                   else "LONG-CoT — likely too long for 601M, filter/subsample")
        print(f"  601M fit: {verdict}")
    else:
        print("  (could not locate a response field for length estimate — "
              "inspect FIELDS above; may be messages-only)")
    # show one short sample
    ex0 = rows[0]
    sample = {k: (str(v)[:120] + "...") if len(str(v)) > 120 else v
              for k, v in ex0.items()}
    print(f"  sample[0]: {sample}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200, help="samples per dataset")
    args = ap.parse_args()

    def _boom(*_a):
        print("\nTIMEOUT 180s"); raise SystemExit(2)
    signal.signal(signal.SIGALRM, _boom); signal.alarm(180)

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("WARN: no HF_TOKEN — gated nvidia repos may 403", flush=True)

    for key, repo, cfg, split in CANDIDATES:
        probe_one(key, repo, cfg, split, args.n)

    print(f"\n{'='*70}\nNOTE: license is on each dataset's HF card "
          "(check CC-BY-4.0 vs NVIDIA Open License before training).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
