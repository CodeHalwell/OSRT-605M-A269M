"""Build the SFT-v2 VERIFIED reasoning corpus (rollouts/sft_v2.jsonl).

v2 of this builder. The first version blended the raw mopd teacher rollouts —
which turned out to be only 81.4% correct on math (743/4000 confidently-wrong
traces, verified against GSM8K gold). This version is VERIFIED-ONLY:

  ON   ~45%  OpenR1-Math-220k: rows with a math_verify-CORRECT R1 generation,
             assembled seq <=4096. The community-standard verified math set.
  ON   ~15%  Bespoke-Stratos-17k: rejection-sampled R1 traces (math/code/
             science/puzzle diversity), <=4096.
  ON   ~5%   mopd math rollouts RE-VERIFIED against GSM8K-train gold — only
             the correct 81.4% are eligible.
  OFF  ~20%  DISJOINT problems from the same verified pools, CoT stripped,
             off-persona → the reasoning-on/off toggle contrast on verified
             answers (unique problems maximised: no ON/OFF overlap).
  CHAT ~15%  existing system_prompt_sft persona/chat slice (correctness N/A).

Safeguards baked in:
  - GSM8K TEST decontamination (normalized-prefix match) on every problem.
  - Within-corpus dedup by problem hash.
  - Assembled-length filter <=4096 (never train a CoT truncated pre-answer).
  - Char-prefilter before tokenizing (skip obvious >>4096 rows cheaply).

Targets ~60k rows → ~1.3 epochs at 1200 steps x eff-batch 64. Deterministic.

Run: HF_TOKEN=... PYTHONPATH=src python scripts/build_sft_v2_data.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, _, v = _l.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import random  # noqa: E402

from osrt.system_prompts import get_by_name  # noqa: E402

ROOT = Path(__file__).parent.parent

# ── Domain-NEUTRAL persona pools ──────────────────────────────────────
# We deliberately use ONLY the domain-agnostic reasoners, and NOT the
# domain-specific personas ("Python expert", "word-problem assistant",
# "scientific"). Reasons: (1) domain-specific personas require classifying
# each problem's domain to place them coherently, and that classification is
# fuzzy (e.g. "Return your final response..." in a MATH prompt reads as code)
# — it produced ~1k incoherent persona/problem pairs. (2) At 601M the domain
# identity buys ~nothing: the <|think|>/<|answer|> tokens already carry the
# format. These neutral reasoners are coherent on ANY problem (math, code,
# science, chat), so mismatches are structurally impossible while still
# teaching system-prompt-following and the reasoning ON/OFF toggle.
_GENERAL_ON = ["minimal_format", "concise_direct", "reasoning_3shot",
               "instruction_strict", "verbose_teaching", "casual_helpful",
               "general_default"]
_GENERAL_OFF = ["direct_concise", "no_reasoning", "assistant_plain",
                "instruction_direct", "chat_direct"]
MOPD = ROOT / "rollouts" / "mopd_v1.jsonl"
CHAT = ROOT / "rollouts" / "system_prompt_sft.jsonl"
OUT = ROOT / "rollouts" / "sft_v2.jsonl"

SEED = 42
MAX_SEQ_TOKENS = 4096
# chars/token ~3.5-4 for this tokenizer; 5x gives a safe over-estimate bound.
MAX_SEQ_CHARS = MAX_SEQ_TOKENS * 5
MIN_RESPONSE_CHARS = 40
TOKENIZER = "v6_tokenizer_export"

ON_OPENR1_TARGET = 27_000
OFF_OPENR1_TARGET = 10_500
ON_STRATOS_TARGET = 9_000
OFF_STRATOS_TARGET = 1_500
CHAT_TARGET = 8_700
# mopd: all verified-correct that fit (~3.2k)


def _norm_prefix(text: str, n: int = 64) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())[:n]


def _phash(text: str) -> str:
    return hashlib.sha1(_norm_prefix(text, 128).encode()).hexdigest()


def main() -> int:  # noqa: PLR0915
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rng = random.Random(SEED)

    def assembled_len(system: str, q: str, think: str, ans: str) -> int:
        seq = (f"<|system|>{system}<|user|>{q}<|assistant|>"
               f"<|think|>{think}<|/think|><|answer|>{ans}<|/answer|>")
        if len(seq) > MAX_SEQ_CHARS:
            return 10**9  # obvious over-length; skip tokenizing
        return len(tok.encode(seq, add_special_tokens=False))

    # ── GSM8K test decontamination set ────────────────────────────────
    print("building GSM8K-test decontamination set...", flush=True)
    contam = set()
    for row in load_dataset("openai/gsm8k", "main", split="test",
                            streaming=True):
        contam.add(_norm_prefix(row["question"]))
    print(f"  {len(contam)} test prefixes", flush=True)

    seen: set[str] = set()          # within-corpus problem dedup
    records: list[dict] = []
    stats = {"contaminated": 0, "toolong": 0, "dup": 0}

    def admit(problem: str) -> bool:
        if _norm_prefix(problem) in contam:
            stats["contaminated"] += 1
            return False
        h = _phash(problem)
        if h in seen:
            stats["dup"] += 1
            return False
        seen.add(h)
        return True

    def add(mode: str, source: str, q: str, think: str, ans: str) -> bool:
        names = _GENERAL_ON if mode == "on" else _GENERAL_OFF
        persona = get_by_name(rng.choice(names))
        L = assembled_len(persona, q, think if mode == "on" else "", ans)
        if L > MAX_SEQ_TOKENS:
            stats["toolong"] += 1
            return False
        records.append({
            "system": persona, "prompt": q,
            "thinking": think if mode == "on" else "",
            "response": ans, "mode": mode, "source": source,
        })
        return True

    # ── 1. mopd math, re-verified against GSM8K-train gold ───────────
    print("verifying mopd math rollouts vs GSM8K-train gold...", flush=True)
    from osrt.rewards import extract_numeric_answer
    gold: dict[int, str] = {}
    for i, row in enumerate(load_dataset("openai/gsm8k", "main", split="train",
                                         streaming=True)):
        if i >= 4200:
            break
        m = re.search(r"####\s*([\-0-9\.,]+)", row["answer"])
        if m:
            gold[i] = m.group(1).replace(",", "").strip().rstrip(".")
    n_mopd = 0
    with MOPD.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] != "math":
                continue
            idx = int(r["id"].split(":")[1])
            g = gold.get(idx)
            pred = extract_numeric_answer(r.get("response") or "")
            if g is None or pred is None:
                continue
            if str(pred).replace(",", "").strip().rstrip(".") != g:
                continue  # verified-INCORRECT: drop
            q = (r.get("prompt") or "").strip()
            think = (r.get("thinking") or "").strip()
            ans = (r.get("response") or "").strip()
            if not q or len(ans) < MIN_RESPONSE_CHARS or not admit(q):
                continue
            if add("on", "mopd-verified", q, think, ans):
                n_mopd += 1
    print(f"  mopd verified-correct kept: {n_mopd}", flush=True)

    # ── 2. OpenR1-Math-220k: verified-correct generations ────────────
    print("streaming OpenR1-Math-220k (verified rows)...", flush=True)
    n_on_r1 = n_off_r1 = 0
    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train",
                      streaming=True)
    for row in ds:
        if n_on_r1 >= ON_OPENR1_TARGET and n_off_r1 >= OFF_OPENR1_TARGET:
            break
        gens = row.get("generations") or []
        ver = row.get("correctness_math_verify") or []
        pick = next((g for g, v in zip(gens, ver) if v), None)
        if pick is None:
            continue
        q = (row.get("problem") or "").strip()
        if not q or not admit(q):
            continue
        # R1 output: <think>…</think> then the worked final answer.
        think, post = pick, ""
        if "</think>" in pick:
            head, _, post = pick.partition("</think>")
            think = head.replace("<think>", "").strip()
            post = post.strip()
        ans = post if len(post) >= MIN_RESPONSE_CHARS else str(
            row.get("answer") or "").strip()
        if not ans:
            continue
        # INTERLEAVED ON/OFF assignment: roll per admitted row at the target
        # ratio, so both modes fill proportionally even if the stream ends
        # early. (The first sequential version exhausted the stream while
        # filling ON and produced ZERO off rows.)
        off_frac = OFF_OPENR1_TARGET / (ON_OPENR1_TARGET + OFF_OPENR1_TARGET)
        go_off = rng.random() < off_frac
        if go_off and n_off_r1 < OFF_OPENR1_TARGET:
            if add("off", "openr1", q, "", ans):
                n_off_r1 += 1
        elif n_on_r1 < ON_OPENR1_TARGET:
            if add("on", "openr1", q, think, ans):
                n_on_r1 += 1
        elif n_off_r1 < OFF_OPENR1_TARGET and add("off", "openr1", q, "", ans):
            n_off_r1 += 1
    print(f"  openr1 ON: {n_on_r1} | OFF: {n_off_r1}", flush=True)

    # ── 3. Bespoke-Stratos-17k: rejection-sampled R1 ─────────────────
    print("streaming Bespoke-Stratos-17k...", flush=True)
    n_on_bs = n_off_bs = 0
    ds = load_dataset("bespokelabs/Bespoke-Stratos-17k", split="train",
                      streaming=True)
    for row in ds:
        if n_on_bs >= ON_STRATOS_TARGET and n_off_bs >= OFF_STRATOS_TARGET:
            break
        conv = row.get("conversations") or []
        q = next((c["value"] for c in conv if c.get("from") == "user"), "")
        a = next((c["value"] for c in conv if c.get("from") == "assistant"), "")
        q = q.strip()
        if not q or not a or not admit(q):
            continue
        if "<|begin_of_thought|>" in a:
            think = a.split("<|begin_of_thought|>")[1].split(
                "<|end_of_thought|>")[0].strip()
            ans = a.split("<|begin_of_solution|>")[-1].split(
                "<|end_of_solution|>")[0].strip()
        else:
            think, ans = "", a.strip()
        if len(ans) < MIN_RESPONSE_CHARS or not think:
            continue
        off_frac = OFF_STRATOS_TARGET / (ON_STRATOS_TARGET + OFF_STRATOS_TARGET)
        go_off = rng.random() < off_frac
        if go_off and n_off_bs < OFF_STRATOS_TARGET:
            if add("off", "stratos", q, "", ans):
                n_off_bs += 1
        elif n_on_bs < ON_STRATOS_TARGET:
            if add("on", "stratos", q, think, ans):
                n_on_bs += 1
        elif n_off_bs < OFF_STRATOS_TARGET and add("off", "stratos", q, "", ans):
            n_off_bs += 1
    print(f"  stratos ON: {n_on_bs} | OFF: {n_off_bs}", flush=True)

    # ── 4. chat/persona slice (correctness N/A) ──────────────────────
    print("sampling chat slice...", flush=True)
    chat_rows = []
    with CHAT.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            q = (c.get("prompt") or "").strip()
            ans = (c.get("response") or "").strip()
            sysp = (c.get("system") or "").strip()
            if not q or len(ans) < MIN_RESPONSE_CHARS or not sysp:
                continue
            chat_rows.append((sysp, q, ans))
    rng.shuffle(chat_rows)
    n_chat = 0
    for sysp, q, ans in chat_rows:
        if n_chat >= CHAT_TARGET:
            break
        L = assembled_len(sysp, q, "", ans)
        if L > MAX_SEQ_TOKENS:
            stats["toolong"] += 1
            continue
        # Gate through the same GSM8K-test decontamination + problem-hash dedup
        # as slices 1-3. OpenHermes-2.5 (a chat source) contains GSM8K-style
        # math, so skipping admit() let test contamination and cross-slice
        # duplicates into the corpus, silently invalidating any GSM8K claim.
        # (docs/specs/2026-07-26-ckpt-sync §3)
        if not admit(q):
            continue
        records.append({"system": sysp, "prompt": q, "thinking": "",
                        "response": ans, "mode": "chat", "source": "chat"})
        n_chat += 1
    print(f"  chat kept: {n_chat}", flush=True)

    # ── write ─────────────────────────────────────────────────────────
    rng.shuffle(records)
    with OUT.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(records)
    by_mode: dict[str, int] = {}
    for r in records:
        by_mode[r["mode"]] = by_mode.get(r["mode"], 0) + 1
    print(f"\nwrote {total} records → {OUT}")
    for m, c in sorted(by_mode.items()):
        print(f"  {m:>4}: {c:>6}  ({100 * c / total:.1f}%)")
    print(f"  dropped: contaminated={stats['contaminated']} "
          f"dup={stats['dup']} toolong={stats['toolong']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
