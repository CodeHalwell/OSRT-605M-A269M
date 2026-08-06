"""Build the SFT-v4 corpus — LENGTH-MATCHED reasoning + broadened task mix.

WHY v4 EXISTS (measured on rollouts/sft_v3.jsonl, 2026-08-06):
  v3 gave the model exactly two kinds of reasoning exemplar and neither was
  usable:
    - 4-8k-char R1/Qwen3 traces (median ~4.5k) which a 601M model CANNOT
      execute: only 1% of nemotron:math and 0% of openr1 rows were <=1500c.
    - 456-char mopd (Gemini) rows that are 83.5% META-NARRATION — the right
      length, no content ("I'm now zeroing in on the core calculation...").
  The model produces 250-440c of thinking, so it copied the only short
  exemplar it had: the narration. GSM8K came back 0/20, and ON vs OFF
  agreed on the same wrong answers — the failure was baked in.

  v4's core change: teach BRIEF CORRECT DERIVATION, using exemplars at the
  length the model actually emits. GSM8K-TRAIN's human solutions are a dead
  match (7,473 rows, median 249 chars, p90 472) and were never used as a
  thinking target before — v2/v3 only used them as verification gold.
  orca-math adds short worked solutions for volume.

  Secondary changes:
    - ON mode was 100% math/science in v3, so the ON persona meant "this is
      a maths problem" rather than "think first". Given a general prompt it
      collapsed (no think, unclosed answer block, ran to the token cap).
      Fixed with smoltalk2 `*_think` general splits.
    - Broadened toward what a 601M does WELL: rewriting, summarising,
      instruction-following, chat, and TOOL CALLING (plain-text
      <tool_call>{json}</tool_call>, no new special tokens — the honest
      answer for a model that cannot do arithmetic is to call a calculator).
    - mopd rows: `thinking` STRIPPED, admitted as OFF (their gold-verified
      answers are still worth having). Most dedup away against GSM8K-train
      anyway, which is processed first so the human solution wins.

  Guards that would have caught v3's defects automatically:
    - ON rows REJECTED when think:answer ratio < 1.0 (kills narration rows).
    - ON thinking hard-capped at MAX_THINK_CHARS so impossible-length trace
      imitation can never dominate the mix again.

Also writes a HELD-OUT VALIDATION split (sft_v4_val.jsonl). v3 was sized
from a training-loss plateau, which midtrain3 already proved unreliable
(loss flat while held-out ppl kept falling). Held-out SFT loss is the
signal that says when to stop.

Safeguards retained from v3: 8-gram + prefix decontamination vs GSM8K TEST
and MATH-500, cross-source problem-hash dedup, assembled length <=4096 with
the real v6 tokenizer, single-turn only, deterministic (SEED=42).

Run:  HF_TOKEN=... PYTHONPATH=src python scripts/build_sft_v4_data.py
      (--smoke for a 1/100-scale end-to-end check)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
ANCHOR = ROOT / "rollouts" / "sft_v2.jsonl"
OUT = ROOT / "rollouts" / "sft_v4.jsonl"
OUT_VAL = ROOT / "rollouts" / "sft_v4_val.jsonl"
REPORT = ROOT / "rollouts" / "sft_v4_report.md"

SEED = 42
MAX_SEQ_TOKENS = 4096
MAX_SEQ_CHARS = MAX_SEQ_TOKENS * 5
MIN_RESPONSE_CHARS = 40      # prose slices
MIN_RESPONSE_CHARS_MATH = 1  # math answers are bare numbers ("72")
# Hard cap on ON thinking. The model emits 250-440c; 2,000c keeps exemplars
# within reach while still allowing a genuinely multi-step derivation.
MAX_THINK_CHARS = 2_000
TOKENIZER = "v6_tokenizer_export"
NGRAM = 8
N_VAL = 1_000

_GENERAL_ON = ["minimal_format", "concise_direct", "reasoning_3shot",
               "instruction_strict", "verbose_teaching", "casual_helpful",
               "general_default"]
_GENERAL_OFF = ["direct_concise", "no_reasoning", "assistant_plain",
                "instruction_direct", "chat_direct"]

# Order matters: earlier slices WIN the cross-source problem dedup. Short-CoT
# math is first so GSM8K-train's human solution beats the mopd narration on
# the same problem.
TARGETS = {
    # ── ON: short-CoT math — the new core ────────────────────────────
    "gsm8k_train_on": 6_500,
    "orca_math_on": 6_000,
    # ── ON: general reasoning (fixes v3's domain entanglement) ───────
    "smol_everyday_think": 2_000,
    "smol_systemchats_think": 2_000,
    "smol_aya_think": 1_200,
    "smol_multiturn_if_think": 1_200,
    # ── ON: short science, for topical breadth ───────────────────────
    "nemotron_science_short_on": 1_500,
    # ── OFF: harder math/science, ANSWER-ONLY (coverage, no long CoT) ─
    "nemotron_math_off": 2_000,
    "openr1_off": 1_500,
    "mopd_off": 10**9,  # all that survive dedup (thinking stripped)
    # ── OFF: general instruct — what a 601M does well ────────────────
    "smol_magpie_off": 2_500,
    "smol_tulu_if_off": 2_000,
    "smol_rewrite_off": 2_000,
    "smol_summarize_off": 2_000,
    "smol_explore_rewrite_off": 1_500,
    # ── OFF: tool calling (plain-text delimiters) ────────────────────
    "smol_xlam_tool": 1_500,
    "smol_hermes_tool": 1_000,
    # ── CHAT ─────────────────────────────────────────────────────────
    "smol_everyday_chat": 2_000,
    "smol_systemchats_chat": 2_000,
    "v2_chat": 1_500,
}

TOOL_INSTRUCTION = (
    "\n\nYou have access to these tools:\n{tools}\n"
    "When a tool is needed, emit the call as "
    "<tool_call>{{\"name\": ..., \"arguments\": {{...}}}}</tool_call> "
    "inside <|answer|>...<|/answer|>. Call a tool rather than guessing at "
    "a calculation or a fact you cannot verify."
)


def _norm_prefix(text: str, n: int = 64) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())[:n]


def _phash(text: str) -> str:
    return hashlib.sha1(_norm_prefix(text, 128).encode()).hexdigest()


def _ngrams(text: str, n: int = NGRAM) -> set[str]:
    w = re.findall(r"[a-z0-9]+", (text or "").lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def _clean_gsm8k_solution(answer: str) -> tuple[str, str] | None:
    """GSM8K answer -> (worked steps, final number).

    Strips the <<48/2=24>> calculator annotations (an artifact of the
    collection tool, not something we want the model emitting)."""
    if "####" not in answer:
        return None
    steps, _, final = answer.rpartition("####")
    steps = re.sub(r"<<[^>]*>>", "", steps).strip()
    final = final.replace(",", "").strip().rstrip(".")
    if not steps or not final:
        return None
    return steps, final


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="1/100-scale targets: end-to-end pipe check")
    args = ap.parse_args()

    import pandas as pd
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from osrt.rewards import extract_numeric_answer

    targets = dict(TARGETS)
    if args.smoke:
        targets = {k: (40 if v >= 10**9 else max(5, v // 100))
                   for k, v in TARGETS.items()}

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rng = random.Random(SEED)

    def assembled_len(system: str, q: str, think: str, ans: str) -> int:
        seq = (f"<|system|>{system}<|user|>{q}<|assistant|>"
               f"<|think|>{think}<|/think|><|answer|>{ans}<|/answer|>")
        if len(seq) > MAX_SEQ_CHARS:
            return 10**9
        return len(tok.encode(seq, add_special_tokens=False))

    # ── decontamination: GSM8K TEST + MATH-500 (train splits are fair) ─
    print("building decontamination sets (GSM8K test + MATH-500)...",
          flush=True)
    contam_ngrams: set[str] = set()
    contam_prefix: set[str] = set()
    gsm_test = pd.read_parquet(hf_hub_download(
        "openai/gsm8k", "main/test-00000-of-00001.parquet",
        repo_type="dataset"))
    for q in gsm_test["question"]:
        contam_ngrams |= _ngrams(q)
        contam_prefix.add(_norm_prefix(q))
    n_gsm = len(contam_prefix)
    for row in load_dataset("HuggingFaceH4/MATH-500", split="test",
                            streaming=True):
        contam_ngrams |= _ngrams(row["problem"])
        contam_prefix.add(_norm_prefix(row["problem"]))
    print(f"  {n_gsm} GSM8K-test + {len(contam_prefix) - n_gsm} MATH-500 "
          f"-> {len(contam_ngrams)} 8-grams", flush=True)

    seen: set[str] = set()
    records: list[dict] = []
    lengths: list[int] = []
    think_lens: list[int] = []
    stats = {"contaminated": 0, "dup": 0, "toolong": 0, "parsefail": 0,
             "narration_reject": 0, "think_truncated": 0}
    kept: dict[str, int] = {k: 0 for k in targets}

    def admit(problem: str) -> bool:
        if _norm_prefix(problem) in contam_prefix:
            stats["contaminated"] += 1
            return False
        grams = _ngrams(problem)
        if grams and not grams.isdisjoint(contam_ngrams):
            stats["contaminated"] += 1
            return False
        h = _phash(problem)
        if h in seen:
            stats["dup"] += 1
            return False
        seen.add(h)
        return True

    def add(mode: str, source: str, system: str, q: str, think: str,
            ans: str, *, math: bool = False) -> bool:
        """Append one record. Enforces the two v4 guards on ON rows."""
        think = (think or "").strip()
        min_resp = MIN_RESPONSE_CHARS_MATH if math else MIN_RESPONSE_CHARS
        if len(ans) < min_resp:
            stats["parsefail"] += 1
            return False
        if mode == "on":
            if len(think) > MAX_THINK_CHARS:
                # Truncating a derivation mid-way would teach an unfinished
                # chain — drop instead.
                stats["think_truncated"] += 1
                return False
            # GUARD: v3's defect. A "reasoning" row whose think is shorter
            # than its answer is narration, not reasoning.
            if len(think) < len(ans):
                stats["narration_reject"] += 1
                return False
        L = assembled_len(system, q, think, ans)
        if L > MAX_SEQ_TOKENS:
            stats["toolong"] += 1
            return False
        records.append({"system": system, "prompt": q, "thinking": think,
                        "response": ans, "mode": mode, "source": source})
        lengths.append(L)
        if think:
            think_lens.append(len(think))
        return True

    def persona(mode: str) -> str:
        pool = _GENERAL_ON if mode == "on" else _GENERAL_OFF
        return get_by_name(rng.choice(pool))

    # ── 1. GSM8K-TRAIN: human short solutions as the thinking target ──
    # THE headline change. Processed FIRST so it wins dedup over mopd rows
    # covering the same problems.
    print("GSM8K-train (human solutions -> thinking)...", flush=True)
    key = "gsm8k_train_on"
    gsm_train = pd.read_parquet(hf_hub_download(
        "openai/gsm8k", "main/train-00000-of-00001.parquet",
        repo_type="dataset"))
    idx = list(range(len(gsm_train)))
    rng.shuffle(idx)
    for i in idx:
        if kept[key] >= targets[key]:
            break
        row = gsm_train.iloc[i]
        parsed = _clean_gsm8k_solution(row["answer"])
        if parsed is None:
            stats["parsefail"] += 1
            continue
        steps, final = parsed
        q = str(row["question"]).strip()
        if not q or not admit(q):
            continue
        if add("on", "gsm8k-train", persona("on"), q, steps, final, math=True):
            kept[key] += 1
    print(f"  {key}: {kept[key]}", flush=True)

    # ── 2. orca-math: short worked solutions ─────────────────────────
    print("orca-math-word-problems-200k...", flush=True)
    key = "orca_math_on"
    scanned = 0
    cap = 60_000 // (50 if args.smoke else 1)
    for row in load_dataset("microsoft/orca-math-word-problems-200k",
                            split="train", streaming=True):
        scanned += 1
        if kept[key] >= targets[key] or scanned > cap:
            break
        q = (row.get("question") or "").strip()
        sol = (row.get("answer") or "").strip()
        if not q or not sol:
            stats["parsefail"] += 1
            continue
        final = extract_numeric_answer(sol)
        if final is None:
            stats["parsefail"] += 1
            continue
        if not admit(q):
            continue
        if add("on", "orca-math", persona("on"), q, sol,
               str(final).strip(), math=True):
            kept[key] += 1
    print(f"  {key}: {kept[key]}", flush=True)

    # ── 3. smoltalk2 helpers ─────────────────────────────────────────
    def parse_smol(row) -> tuple[str, str, str, str] | None:
        """-> (own_system, q, think, answer) from the first SUBSTANTIVE
        user->assistant exchange. R1-style <think> blocks are extracted.

        "Substantive" means the reply clears MIN_RESPONSE_CHARS: the
        everyday-conversations splits open with a greeting whose reply is a
        one-liner ("Hello! How can I assist you today?"), so returning
        strictly the first exchange yields almost nothing admissible — and
        the greeting also collapses every row onto the same problem hash,
        so cross-source dedup eats the rest. (Same bug bit the v3 builder.)
        """
        msgs = row.get("messages") or []
        own_sys = ""
        if msgs and msgs[0].get("role") == "system":
            own_sys = (msgs[0].get("content") or "").strip()
            msgs = msgs[1:]
        for i in range(len(msgs) - 1):
            if (msgs[i].get("role") != "user"
                    or msgs[i + 1].get("role") != "assistant"):
                continue
            q = (msgs[i].get("content") or "").strip()
            a = (msgs[i + 1].get("content") or "").strip()
            if not q or not a:
                continue
            think = ""
            if "</think>" in a:
                head, _, post = a.partition("</think>")
                think = head.replace("<think>", "").strip()
                a = post.strip()
            if len(a) < MIN_RESPONSE_CHARS:
                continue  # greeting/one-liner — try the next exchange
            return own_sys, q, think, a
        return None

    def smol_slice(split: str, key: str, mode: str, *,
                   keep_system: bool = False, want_think: bool = False,
                   tools: bool = False) -> None:
        """Stream one smoltalk2 SFT split into the corpus."""
        print(f"smoltalk2 [{split}]...", flush=True)
        ds = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=split,
                          streaming=True)
        scanned = 0
        cap = 60_000 // (50 if args.smoke else 1)
        for row in ds:
            scanned += 1
            if kept[key] >= targets[key] or scanned > cap:
                break
            parsed = parse_smol(row)
            if parsed is None:
                stats["parsefail"] += 1
                continue
            own_sys, q, think, ans = parsed
            if want_think and not think:
                stats["parsefail"] += 1
                continue
            if not want_think:
                think = ""
            # Tool specs live in chat_template_kwargs.xml_tools; the model must
            # see them to know what it may call. Plain text — no new special
            # tokens at 601M (see project backlog #14). Resolved BEFORE
            # admit() so a spec-less row doesn't burn its problem hash.
            spec_text = ""
            if tools:
                kw = row.get("chat_template_kwargs") or {}
                spec = kw.get("xml_tools") or kw.get("python_tools") or []
                spec_text = "\n".join(str(s) for s in spec).strip()
                if not spec_text:
                    stats["parsefail"] += 1
                    continue
            if not admit(q):
                continue
            system = own_sys if (keep_system and own_sys) else persona(
                "on" if mode == "on" else "off")
            if spec_text:
                system = system + TOOL_INSTRUCTION.format(tools=spec_text)
            if add(mode, f"smoltalk2:{split.replace('_no_think','')}",
                   system, q, think, ans):
                kept[key] += 1
        print(f"  {key}: {kept[key]}", flush=True)

    # ON — general reasoning (the v3 domain-entanglement fix)
    smol_slice("smoltalk_everyday_convs_reasoning_Qwen3_32B_think",
               "smol_everyday_think", "on", want_think=True)
    smol_slice("smoltalk_systemchats_Qwen3_32B_think",
               "smol_systemchats_think", "on", want_think=True,
               keep_system=True)
    smol_slice("aya_dataset_Qwen3_32B_think", "smol_aya_think", "on",
               want_think=True)
    smol_slice("multi_turn_reasoning_if_think", "smol_multiturn_if_think",
               "on", want_think=True)

    # OFF — general instruct: what a 601M actually does well
    smol_slice("smoltalk_smollm3_smol_magpie_ultra_no_think",
               "smol_magpie_off", "off")
    smol_slice("tulu_3_sft_personas_instruction_following_no_think",
               "smol_tulu_if_off", "off")
    smol_slice("smoltalk_smollm3_smol_rewrite_no_think",
               "smol_rewrite_off", "off", keep_system=True)
    smol_slice("smoltalk_smollm3_smol_summarize_no_think",
               "smol_summarize_off", "off", keep_system=True)
    smol_slice("smoltalk_smollm3_explore_instruct_rewriting_no_think",
               "smol_explore_rewrite_off", "off")

    # OFF — tool calling
    smol_slice("xlam_traces_no_think", "smol_xlam_tool", "off", tools=True)
    smol_slice("hermes_function_calling_v1_no_think", "smol_hermes_tool",
               "off", tools=True)

    # CHAT
    smol_slice("smoltalk_smollm3_everyday_conversations_no_think",
               "smol_everyday_chat", "chat")
    smol_slice("smoltalk_smollm3_systemchats_30k_no_think",
               "smol_systemchats_chat", "chat", keep_system=True)

    # ── 4. Nemotron: SHORT science ON + answer-only math OFF ──────────
    def parse_nemotron(row) -> tuple[str, str, str] | None:
        inp = row.get("input") or []
        if len(inp) != 1 or inp[0].get("role") != "user":
            return None
        q = (inp[0].get("content") or "").strip()
        out = (row.get("output") or "").strip()
        if not q or not out:
            return None
        if row.get("reasoning") == "on":
            if "</think>" not in out:
                return None
            head, _, post = out.partition("</think>")
            return q, head.replace("<think>", "").strip(), post.strip()
        return q, "", out

    for split, key, mode in (("science", "nemotron_science_short_on", "on"),
                             ("math", "nemotron_math_off", "off")):
        print(f"nemotron-pt [{split}] -> {mode}...", flush=True)
        ds = load_dataset("nvidia/Llama-Nemotron-Post-Training-Dataset",
                          "SFT", split=split, streaming=True)
        scanned = 0
        cap = 120_000 // (50 if args.smoke else 1)
        for row in ds:
            scanned += 1
            if kept[key] >= targets[key] or scanned > cap:
                break
            parsed = parse_nemotron(row)
            if parsed is None:
                stats["parsefail"] += 1
                continue
            q, think, ans = parsed
            # ON science keeps only SHORT derivations; OFF math drops CoT
            # entirely (topical coverage without impossible-length traces).
            think = think if mode == "on" else ""
            if not admit(q):
                continue
            if add(mode, f"nemotron:{split}", persona(mode), q, think, ans):
                kept[key] += 1
        print(f"  {key}: {kept[key]}", flush=True)

    # ── 5. v2 anchor: openr1 answer-only, mopd stripped, chat ─────────
    print("anchor slices from sft_v2.jsonl...", flush=True)
    by_src: dict[str, list[dict]] = {}
    with ANCHOR.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_src.setdefault(r["source"], []).append(r)

    for key, src, mode in (("openr1_off", "openr1", "off"),
                           ("mopd_off", "mopd-verified", "off"),
                           ("v2_chat", "chat", "chat")):
        rows = by_src.get(src, [])
        rng.shuffle(rows)
        for r in rows:
            if kept[key] >= targets[key]:
                break
            q = (r.get("prompt") or "").strip()
            if not q or not admit(q):
                continue
            # thinking DROPPED for openr1 (too long) and mopd (narration).
            system = r["system"] if mode == "chat" else persona("off")
            if add(mode, f"v2:{src}", system, q, "",
                   (r.get("response") or "").strip()):
                kept[key] += 1
        print(f"  {key}: {kept[key]}", flush=True)

    # ── write train + held-out val ────────────────────────────────────
    order = list(range(len(records)))
    rng.shuffle(order)
    n_val = min(N_VAL // (100 if args.smoke else 1), max(1, len(order) // 20))
    val_idx, train_idx = set(order[:n_val]), order[n_val:]
    with OUT.open("w", encoding="utf-8") as f:
        for i in train_idx:
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")
    with OUT_VAL.open("w", encoding="utf-8") as f:
        for i in sorted(val_idx):
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")

    # ── report ────────────────────────────────────────────────────────
    total = len(records)
    by_mode: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for r in records:
        by_mode[r["mode"]] = by_mode.get(r["mode"], 0) + 1
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    lengths.sort()
    think_lens.sort()

    def pct(xs: list[int], p: float) -> int:
        return xs[min(len(xs) - 1, int(p * (len(xs) - 1)))] if xs else 0

    MATHY = ("gsm8k-train", "orca-math", "nemotron:math", "nemotron:science",
             "v2:openr1", "v2:mopd-verified", "v2:stratos")
    mathy = sum(c for s, c in by_source.items() if s.startswith(MATHY))
    tool = sum(c for s, c in by_source.items()
               if "xlam" in s or "hermes" in s)
    lines = [
        "# SFT-v4 corpus report",
        "",
        f"seed={SEED} max_seq={MAX_SEQ_TOKENS} max_think={MAX_THINK_CHARS}c "
        f"tokenizer={TOKENIZER}"
        + (" **SMOKE RUN — 1/100 targets**" if args.smoke else ""),
        "",
        f"**{total} records** ({len(train_idx)} train + {n_val} held-out val), "
        f"{sum(lengths) / 1e6:.1f}M assembled tokens",
        "",
        f"math/science: {mathy} ({100 * mathy / total:.0f}%) — v3 was 64%",
        f"tool-calling: {tool} ({100 * tool / total:.0f}%) — new in v4",
        "",
        "## By mode",
    ]
    for m, c in sorted(by_mode.items()):
        lines.append(f"- {m}: {c} ({100 * c / total:.1f}%)")
    lines += ["", "## By source"]
    for s, c in sorted(by_source.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {s}: {c}")
    lines += [
        "",
        "## THINKING length — the v4 headline "
        "(model emits 250-440c; v3 median was ~4,500c)",
        f"- rows with thinking: {len(think_lens)}",
        f"- p10={pct(think_lens, .10)} p50={pct(think_lens, .50)} "
        f"p90={pct(think_lens, .90)} max={think_lens[-1] if think_lens else 0}",
        "",
        "### Histogram (250-char buckets)",
    ]
    for lo in range(0, MAX_THINK_CHARS, 250):
        c = sum(1 for x in think_lens if lo <= x < lo + 250)
        bar = "#" * round(60 * c / max(1, len(think_lens)))
        lines.append(f"- {lo:>4}-{lo + 250:<4}: {c:>6} {bar}")
    lines += [
        "", "## Assembled length (tokens)",
        f"- p50={pct(lengths, .50)} p90={pct(lengths, .90)} "
        f"p99={pct(lengths, .99)} max={lengths[-1]}",
        "", "## Drops",
        f"- contaminated (8-gram/prefix vs GSM8K-test + MATH-500): "
        f"{stats['contaminated']}",
        f"- duplicate problem (cross-source hash): {stats['dup']}",
        f"- **narration-rejected (ON, think < answer)**: "
        f"{stats['narration_reject']}  <- the v3 defect, now auto-caught",
        f"- **think > {MAX_THINK_CHARS}c (impossible-length trace)**: "
        f"{stats['think_truncated']}",
        f"- assembled > {MAX_SEQ_TOKENS} tokens: {stats['toolong']}",
        f"- parse-fail / too-short / no extractable answer: "
        f"{stats['parsefail']}",
        "", "## Slice fill vs target",
    ]
    for k in targets:
        t = targets[k] if targets[k] < 10**9 else "ALL"
        lines.append(f"- {k}: {kept[k]} / {t}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {len(train_idx)} train -> {OUT}")
    print(f"wrote {n_val} val -> {OUT_VAL}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
