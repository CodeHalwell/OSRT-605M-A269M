"""Build the SFT-v3 corpus (rollouts/sft_v3.jsonl) for the post-midtrain3 SFT.

v3 recipe (agreed 2026-08-06): anchor on our existing VERIFIED corpus, refresh
with two newer post-training sets, keep long CoT — the trainer packs at
seq_len 4096 (SFTStream concatenates examples per row, pads only the tail).

  ANCHOR  rollouts/sft_v2.jsonl — already verified/decontaminated, our schema:
          ALL mopd-verified (gold-checked vs GSM8K-train) + openr1/stratos/chat
          subsamples for continuity.
  ON/OFF  nvidia/Llama-Nemotron-Post-Training-Dataset [SFT]: math+science
          reasoning-on (long R1-style CoT) + math/chat reasoning-off. Its
          "detailed thinking on/off" contract maps 1:1 onto ours — we discard
          their system line and re-persona with DOMAIN-NEUTRAL prompts
          (v2 lesson: domain-persona mismatch produced incoherent pairs).
  CHAT/OFF HuggingFaceTB/smoltalk2 [SFT] no_think splits: magpie-ultra +
          tulu3-personas-IF instruct (OFF), everyday-convs + systemchats
          (CHAT; systemchats keep their OWN system prompt — that slice IS
          system-prompt-following training).

Targets ~42K rows, ON/OFF/CHAT ~= 60/25/15.

Safeguards (v2 safeguards kept, decon upgraded):
  - 8-GRAM decontamination vs GSM8K test + MATH-500 (v2 used prefix-match
    only; 8-gram also catches embedded/reworded test problems).
  - Cross-source dedup by problem hash (OpenR1 problems appear in BOTH our
    anchor and Nemotron-PT).
  - Assembled-length filter <=4096 with the real v6 tokenizer, char
    prefilter first (never train a CoT truncated before its answer).
  - Single-turn only: first user->assistant exchange (our schema).
  - Deterministic (SEED=42); report written to rollouts/sft_v3_report.md
    for review BEFORE any GPU spend.

Run:  HF_TOKEN=... PYTHONPATH=src python scripts/build_sft_v3_data.py
      (add --smoke for a 1/100-scale end-to-end pipe check)
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
OUT = ROOT / "rollouts" / "sft_v3.jsonl"
REPORT = ROOT / "rollouts" / "sft_v3_report.md"

SEED = 42
MAX_SEQ_TOKENS = 4096
MAX_SEQ_CHARS = MAX_SEQ_TOKENS * 5  # chars/token ~3.5-4; 5x = safe bound
MIN_RESPONSE_CHARS = 40
TOKENIZER = "v6_tokenizer_export"
NGRAM = 8

# Domain-NEUTRAL persona pools (see build_sft_v2_data.py for the rationale:
# domain-specific personas needed fuzzy problem classification and produced
# ~1k incoherent pairs; neutral reasoners are coherent on any problem).
_GENERAL_ON = [
    "minimal_format",
    "concise_direct",
    "reasoning_3shot",
    "instruction_strict",
    "verbose_teaching",
    "casual_helpful",
    "general_default",
]
_GENERAL_OFF = [
    "direct_concise",
    "no_reasoning",
    "assistant_plain",
    "instruction_direct",
    "chat_direct",
]

# ── slice targets (ON 25,257 / OFF 10,500 / CHAT 6,500 = 42,257) ──────
TARGETS = {
    # anchor (rollouts/sft_v2.jsonl)
    "anchor_mopd": 10**9,  # ALL gold-checked rows (~3,257)
    "anchor_openr1": 6_000,  # ON
    "anchor_stratos": 2_000,  # ON
    "anchor_chat": 3_000,  # CHAT
    # nemotron-pt [SFT]
    "nemotron_math_on": 9_000,
    "nemotron_science_on": 5_000,
    "nemotron_math_off": 2_000,
    "nemotron_chat_off": 4_000,
    # smoltalk2 [SFT]
    "smol_magpie_off": 2_500,
    "smol_tulu_if_off": 2_000,
    "smol_everyday_chat": 1_500,
    "smol_systemchats_chat": 2_000,
}


def _norm_prefix(text: str, n: int = 64) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())[:n]


def _phash(text: str) -> str:
    return hashlib.sha1(_norm_prefix(text, 128).encode()).hexdigest()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _ngrams(text: str, n: int = NGRAM) -> set[str]:
    w = _words(text)
    return {" ".join(w[i : i + n]) for i in range(len(w) - n + 1)}


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1/100-scale targets: end-to-end pipe check",
    )
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    targets = dict(TARGETS)
    if args.smoke:
        targets = {k: max(5, v // 100) if v < 10**9 else 40 for k, v in TARGETS.items()}

    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rng = random.Random(SEED)

    def assembled_len(system: str, q: str, think: str, ans: str) -> int:
        seq = (
            f"<|system|>{system}<|user|>{q}<|assistant|>"
            f"<|think|>{think}<|/think|><|answer|>{ans}<|/answer|>"
        )
        if len(seq) > MAX_SEQ_CHARS:
            return 10**9
        return len(tok.encode(seq, add_special_tokens=False))

    # ── decontamination sets: 8-grams + prefixes of eval questions ────
    print("building decontamination sets (GSM8K test + MATH-500)...", flush=True)
    contam_ngrams: set[str] = set()
    contam_prefix: set[str] = set()
    for row in load_dataset("openai/gsm8k", "main", split="test", streaming=True):
        contam_ngrams |= _ngrams(row["question"])
        contam_prefix.add(_norm_prefix(row["question"]))
    n_gsm = len(contam_prefix)
    for row in load_dataset("HuggingFaceH4/MATH-500", split="test", streaming=True):
        contam_ngrams |= _ngrams(row["problem"])
        contam_prefix.add(_norm_prefix(row["problem"]))
    print(
        f"  {n_gsm} GSM8K + {len(contam_prefix) - n_gsm} MATH-500 "
        f"questions -> {len(contam_ngrams)} 8-grams",
        flush=True,
    )

    seen: set[str] = set()  # cross-source problem dedup
    records: list[dict] = []
    stats = {"contaminated": 0, "dup": 0, "toolong": 0, "parsefail": 0}
    lengths: list[int] = []

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

    def add(mode: str, source: str, system: str, q: str, think: str, ans: str) -> bool:
        L = assembled_len(system, q, think, ans)
        if L > MAX_SEQ_TOKENS:
            stats["toolong"] += 1
            return False
        records.append(
            {
                "system": system,
                "prompt": q,
                "thinking": think,
                "response": ans,
                "mode": mode,
                "source": source,
            }
        )
        lengths.append(L)
        return True

    def persona(mode: str) -> str:
        names = _GENERAL_ON if mode == "on" else _GENERAL_OFF
        return get_by_name(rng.choice(names))

    kept: dict[str, int] = {k: 0 for k in targets}

    # ── 1. ANCHOR: our verified sft_v2 corpus ─────────────────────────
    print("loading anchor slices from rollouts/sft_v2.jsonl...", flush=True)
    by_src: dict[str, list[dict]] = {}
    with ANCHOR.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_src.setdefault(r["source"], []).append(r)
    anchor_plan = [
        ("anchor_mopd", "mopd-verified"),
        ("anchor_openr1", "openr1"),
        ("anchor_stratos", "stratos"),
        ("anchor_chat", "chat"),
    ]
    for key, src in anchor_plan:
        rows = by_src.get(src, [])
        rng.shuffle(rows)
        for r in rows:
            if kept[key] >= targets[key]:
                break
            q = (r.get("prompt") or "").strip()
            if not q or not admit(q):
                continue
            # rows carry their v2 persona/system; keep it (already coherent)
            if add(
                r["mode"],
                f"v2:{src}",
                r["system"],
                q,
                (r.get("thinking") or "").strip(),
                (r.get("response") or "").strip(),
            ):
                kept[key] += 1
        print(f"  {key}: {kept[key]}", flush=True)

    # ── 2. NEMOTRON-PT: math/science ON + math/chat OFF ───────────────
    def parse_nemotron(row) -> tuple[str, str, str] | None:
        """-> (q, think, ans) or None. Single-turn user rows only."""
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
            think = head.replace("<think>", "").strip()
            ans = post.strip()
            if not think or len(ans) < MIN_RESPONSE_CHARS:
                return None
            return q, think, ans
        if len(out) < MIN_RESPONSE_CHARS:
            return None
        return q, "", out

    # Math OFF rows are CONSTRUCTED by stripping the CoT from disjoint ON
    # rows (v2's OFF recipe: toggle contrast on the same distribution). The
    # native off rows sit gigabytes deep in the math files — v1.1 ON rows
    # come first — so streaming for them stalls the build for hours. Native
    # off rows are still accepted when they do appear (chat split is all-off).
    nemotron_plan = [
        # (split, on-target-key, off-target-key, max rows scanned)
        ("math", "nemotron_math_on", "nemotron_math_off", 250_000),
        ("science", "nemotron_science_on", None, 100_000),
        ("chat", None, "nemotron_chat_off", 100_000),
    ]
    prog_every = 2_000 if args.smoke else 20_000
    for split, on_key, off_key, max_scan in nemotron_plan:
        if args.smoke:
            max_scan //= 50
        print(f"streaming nemotron-pt [{split}]...", flush=True)
        ds = load_dataset(
            "nvidia/Llama-Nemotron-Post-Training-Dataset",
            "SFT",
            split=split,
            streaming=True,
        )
        on_t = targets[on_key] if on_key else 0
        off_t = targets[off_key] if off_key else 0
        off_frac = off_t / max(1, on_t + off_t)
        scanned = 0
        for row in ds:
            scanned += 1
            if scanned > max_scan:
                print(f"  scan cap {max_scan} hit — moving on", flush=True)
                break
            if scanned % prog_every == 0:
                fills = " ".join(f"{k}={kept[k]}" for k in (on_key, off_key) if k)
                print(f"  ...scanned {scanned}: {fills}", flush=True)
            done_on = on_key is None or kept[on_key] >= on_t
            done_off = off_key is None or kept[off_key] >= off_t
            if done_on and done_off:
                break
            parsed = parse_nemotron(row)
            if parsed is None:
                stats["parsefail"] += 1
                continue
            q, think, ans = parsed
            # route before admit() so dedup doesn't eat rows we won't use
            if think and not done_off and (done_on or rng.random() < off_frac):
                mode, key, think = "off", off_key, ""  # strip CoT -> OFF
            elif think and not done_on:
                mode, key = "on", on_key
            elif not think and not done_off:
                mode, key = "off", off_key
            else:
                continue
            if not admit(q):
                continue
            if add(mode, f"nemotron:{split}", persona(mode), q, think, ans):
                kept[key] += 1
        for k in (on_key, off_key):
            if k:
                print(f"  {k}: {kept[k]}", flush=True)

    # ── 3. SMOLTALK2: OFF instruct + CHAT ──────────────────────────────
    def parse_smol(row) -> tuple[str, str, str] | None:
        """-> (own_system, q, ans): first exchange with a SUBSTANTIVE reply.
        everyday-convs open with a greeting exchange ("Hi there" -> one-line
        hello) — taking strictly the first pair yields zero admissible rows.
        """
        msgs = row.get("messages") or []
        system = ""
        if msgs and msgs[0].get("role") == "system":
            system = (msgs[0].get("content") or "").strip()
            msgs = msgs[1:]
        for i in range(len(msgs) - 1):
            if msgs[i].get("role") == "user" and msgs[i + 1].get("role") == "assistant":
                q = (msgs[i].get("content") or "").strip()
                ans = (msgs[i + 1].get("content") or "").strip()
                if q and len(ans) >= MIN_RESPONSE_CHARS:
                    return system, q, ans
        return None

    smol_plan = [
        (
            "smoltalk_smollm3_smol_magpie_ultra_no_think",
            "smol_magpie_off",
            "off",
            False,
        ),
        (
            "tulu_3_sft_personas_instruction_following_no_think",
            "smol_tulu_if_off",
            "off",
            False,
        ),
        (
            "smoltalk_smollm3_everyday_conversations_no_think",
            "smol_everyday_chat",
            "chat",
            False,
        ),
        (
            "smoltalk_smollm3_systemchats_30k_no_think",
            "smol_systemchats_chat",
            "chat",
            True,
        ),  # keep OWN system prompt
    ]
    for split, key, mode, keep_system in smol_plan:
        print(f"streaming smoltalk2 [{split}]...", flush=True)
        ds = load_dataset("HuggingFaceTB/smoltalk2", "SFT", split=split, streaming=True)
        scanned = 0
        max_scan = 60_000 // (50 if args.smoke else 1)
        for row in ds:
            scanned += 1
            if scanned > max_scan:
                print(f"  scan cap {max_scan} hit — moving on", flush=True)
                break
            if scanned % prog_every == 0:
                print(f"  ...scanned {scanned}: {key}={kept[key]}", flush=True)
            if kept[key] >= targets[key]:
                break
            parsed = parse_smol(row)
            if parsed is None:
                stats["parsefail"] += 1
                continue
            own_sys, q, ans = parsed
            if not admit(q):
                continue
            system = own_sys if (keep_system and own_sys) else persona("off")
            if add(
                mode, f"smoltalk2:{split.split('_no_think')[0]}", system, q, "", ans
            ):
                kept[key] += 1
        print(f"  {key}: {kept[key]}", flush=True)

    # ── write corpus ───────────────────────────────────────────────────
    order = list(range(len(records)))
    rng.shuffle(order)
    with OUT.open("w", encoding="utf-8") as f:
        for i in order:
            f.write(json.dumps(records[i], ensure_ascii=False) + "\n")

    # ── report ─────────────────────────────────────────────────────────
    total = len(records)
    by_mode: dict[str, int] = {}
    by_source: dict[str, int] = {}
    for r in records:
        by_mode[r["mode"]] = by_mode.get(r["mode"], 0) + 1
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    lengths.sort()

    def pct(p: float) -> int:
        return lengths[min(len(lengths) - 1, int(p * (len(lengths) - 1)))]

    total_tokens = sum(lengths)
    packed_rows = total_tokens / MAX_SEQ_TOKENS
    lines = [
        "# SFT-v3 corpus report",
        "",
        f"seed={SEED} max_seq={MAX_SEQ_TOKENS} tokenizer={TOKENIZER}"
        + (" **SMOKE RUN — 1/100 targets**" if args.smoke else ""),
        "",
        f"**{total} records, {total_tokens / 1e6:.1f}M assembled tokens** "
        f"(~{packed_rows:,.0f} packed rows at {MAX_SEQ_TOKENS})",
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
        "## Assembled length (tokens)",
        f"- p10={pct(0.10)} p25={pct(0.25)} p50={pct(0.50)} "
        f"p75={pct(0.75)} p90={pct(0.90)} p99={pct(0.99)} "
        f"max={lengths[-1]}",
        "",
        "### Histogram (512-token buckets)",
    ]
    for lo in range(0, MAX_SEQ_TOKENS, 512):
        hi = lo + 512
        c = sum(1 for x in lengths if lo <= x < hi)
        bar = "#" * round(60 * c / max(1, total))
        lines.append(f"- {lo:>4}-{hi:<4}: {c:>6} {bar}")
    lines += [
        "",
        "## Drops",
        f"- contaminated (8-gram/prefix vs GSM8K-test + MATH-500): "
        f"{stats['contaminated']}",
        f"- duplicate problem (cross-source hash): {stats['dup']}",
        f"- assembled > {MAX_SEQ_TOKENS} tokens: {stats['toolong']}",
        f"- parse-fail / multi-turn / too-short: {stats['parsefail']}",
        "",
        "## Slice fill vs target",
    ]
    for k in targets:
        t = targets[k] if targets[k] < 10**9 else "ALL"
        lines.append(f"- {k}: {kept[k]} / {t}")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nwrote {total} records -> {OUT}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
