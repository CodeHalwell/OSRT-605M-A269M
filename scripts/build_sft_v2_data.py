"""Build the SFT-v2 reasoning-distillation corpus (rollouts/sft_v2.jsonl).

Combines the two existing rollout files into one persona-tagged corpus so the
RolloutDataset loader (which reads `system`/`prompt`/`thinking`/`response`) can
train the v6 model on COHERENT long reasoning WITH the reasoning-on/off system
toggle that is the project north star.

Composition (reasoning-focused 65/20/15 of the final corpus):
  ON   65%  — mopd_v1 record + sampled REASONING_ON persona, keep thinking+response
  OFF  20%  — mopd_v1 subset, thinking DROPPED + REASONING_OFF persona, direct answer
              (matched questions → clean on/off contrast)
  CHAT 15%  — system_prompt_sft as-is (its character persona, answer-only)

All mopd_v1 reasoning rollouts are used (split ON/OFF); chat is sampled to hit
the ratio. Output records carry a `mode` tag for inspection. Deterministic.

Run: PYTHONPATH=src python scripts/build_sft_v2_data.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

from osrt.system_prompts import sample_system_prompt

ROOT = Path(__file__).parent.parent
MOPD = ROOT / "rollouts" / "mopd_v1.jsonl"
CHAT = ROOT / "rollouts" / "system_prompt_sft.jsonl"
OUT = ROOT / "rollouts" / "sft_v2.jsonl"

SEED = 42
MIN_RESPONSE_CHARS = 100          # drop degenerate-short answers
OFF_FRACTION_OF_MOPD = 0.235      # → ON:OFF ≈ 65:20 within the mopd pool
CHAT_FRACTION_OF_TOTAL = 0.15     # → 15% chat in the final corpus
# Drop records whose full assembled sequence exceeds the SFT-v2 seq_len, so
# we never train on a CoT truncated before its <|answer|> (which would teach
# the model to ramble without concluding — the exact failure we're fixing).
MAX_SEQ_TOKENS = 4096
TOKENIZER = "v6_tokenizer_export"


def _load(path: Path) -> list[dict]:
    # Iterate the file directly: str.splitlines() also breaks on exotic
    # Unicode line separators (  etc.) that appear inside the teacher
    # text, which would split a record mid-JSON-string.
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _too_long(tok, system: str, prompt: str, thinking: str,
              response: str) -> bool:
    seq = (f"<|system|>{system}<|user|>{prompt}<|assistant|>"
           f"<|think|>{thinking}<|/think|><|answer|>{response}<|/answer|>")
    return len(tok.encode(seq, add_special_tokens=False)) > MAX_SEQ_TOKENS


def main() -> int:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rng = random.Random(SEED)
    mopd = _load(MOPD)
    chat = _load(CHAT)

    records: list[dict] = []
    n_on = n_off = n_toolong = 0
    for rec in mopd:
        prompt = (rec.get("prompt") or "").strip()
        thinking = (rec.get("thinking") or "").strip()
        response = (rec.get("response") or "").strip()
        if not prompt or len(response) < MIN_RESPONSE_CHARS:
            continue
        # Assign OFF to a subset; ON requires real thinking (≈98% have it —
        # the rest fall back to OFF since they have no CoT to learn from).
        make_off = (rng.random() < OFF_FRACTION_OF_MOPD) or (not thinking)
        if make_off:
            _, persona = sample_system_prompt(rng, "off")
            think_out = ""
        else:
            _, persona = sample_system_prompt(rng, "on")
            think_out = thinking
        if _too_long(tok, persona, prompt, think_out, response):
            n_toolong += 1
            continue
        records.append({
            "system": persona, "prompt": prompt,
            "thinking": think_out, "response": response,
            "mode": "off" if make_off else "on",
            "source": rec.get("source", "?"),
        })
        if make_off:
            n_off += 1
        else:
            n_on += 1

    # Chat: sample to hit CHAT_FRACTION_OF_TOTAL of the final corpus.
    # total = (n_on + n_off) / (1 - chat_frac)  →  n_chat = total * chat_frac
    reasoning_n = n_on + n_off
    target_chat = int(round(
        reasoning_n * CHAT_FRACTION_OF_TOTAL / (1.0 - CHAT_FRACTION_OF_TOTAL)
    ))
    chat_ok = [c for c in chat
               if (c.get("prompt") or "").strip()
               and len((c.get("response") or "").strip()) >= MIN_RESPONSE_CHARS]
    rng.shuffle(chat_ok)
    n_chat = 0
    for c in chat_ok:
        if n_chat >= target_chat:
            break
        csys = (c.get("system") or "").strip()
        cprompt = (c.get("prompt") or "").strip()
        cresp = (c.get("response") or "").strip()
        if _too_long(tok, csys, cprompt, "", cresp):
            n_toolong += 1
            continue
        records.append({
            "system": csys, "prompt": cprompt, "thinking": "",
            "response": cresp, "mode": "chat", "source": c.get("source", "chat"),
        })
        n_chat += 1

    rng.shuffle(records)  # interleave modes
    with OUT.open("w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(records)
    print(f"wrote {total} records → {OUT}")
    print(f"  ON   {n_on:>6}  ({100*n_on/total:.1f}%)")
    print(f"  OFF  {n_off:>6}  ({100*n_off/total:.1f}%)")
    print(f"  CHAT {n_chat:>6}  ({100*n_chat/total:.1f}%)")
    print(f"  dropped (> {MAX_SEQ_TOKENS} tok): {n_toolong}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
