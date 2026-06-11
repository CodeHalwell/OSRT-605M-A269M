"""Reasoning-on/off GSM8K eval for SFT — the project's north-star metric.

Generates answers to held-out GSM8K problems TWICE per problem: once with a
REASONING_ON system persona, once with REASONING_OFF (same problem, same user
turn). Reports accuracy_on, accuracy_off, mean response length on/off, and
format-compliance rate.

The win condition for the WHOLE project is accuracy_on > accuracy_off (the long
reasoning must earn its tokens). At SFT-v1 this is the BASELINE measurement —
don't expect on>off yet; later stages (CoT-SFT, GRPO) must move it.

Reuses rewards.extract_numeric_answer (parses <|answer|>) and
extract_gsm8k_answer (ground-truth ####) — no new extraction logic. The held-out
slice is a fixed GSM8K test split sample, cached once per process.
"""
from __future__ import annotations

import re

import torch
import torch.nn as nn

from osrt.rewards import extract_gsm8k_answer, extract_numeric_answer
from osrt.system_prompts import sample_system_prompt

# cache the held-out GSM8K eval batch (prompts + gold) once per process
_GSM8K_CACHE: list[tuple[str, str]] | None = None


def _load_gsm8k_heldout(n: int) -> list[tuple[str, str]]:
    """Return n (question, gold_answer) from GSM8K test split. Cached."""
    global _GSM8K_CACHE
    if _GSM8K_CACHE is not None and len(_GSM8K_CACHE) >= n:
        return _GSM8K_CACHE[:n]
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test", streaming=True)
    out: list[tuple[str, str]] = []
    for row in ds:
        gold = extract_gsm8k_answer(row["answer"])
        if gold is None:
            continue
        out.append((row["question"], gold))
        if len(out) >= n:
            break
    _GSM8K_CACHE = out
    return out


_WELL_FORMED = re.compile(
    r"<\|think\|>.*?<\|/think\|>.*?<\|answer\|>.*?<\|/answer\|>", re.S
)


def _norm(s: str | None) -> str | None:
    if s is None:
        return None
    return s.replace(",", "").strip().rstrip(".")


@torch.no_grad()
def _gen_one(model, tok, system_text: str, question: str, device,
             max_new_tokens: int) -> str:
    prompt = f"<|system|>{system_text}<|user|>{question}<|assistant|>"
    ids = torch.tensor([tok.encode(prompt, add_special_tokens=False)],
                       dtype=torch.long, device=device)
    out = model.generate(ids, max_new_tokens=max_new_tokens, temperature=0.0,
                         eos_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)


def run_reasoning_eval(
    model: nn.Module, tok, device, *,
    n_problems: int = 50, max_new_tokens: int = 512, seed: int = 0,
) -> dict:
    """Reasoning-on vs -off accuracy on a held-out GSM8K slice.

    Fixed persona per side (the first ON / first OFF persona) so the A/B isolates
    the reasoning-mode instruction, not persona variance. Returns a wandb-loggable
    dict. Switches the model to eval mode and restores it.
    """
    import random
    rng = random.Random(seed)
    was_training = model.training
    model.train(False)

    # fixed personas for a clean A/B (not sampled — we want the contrast to be
    # the reasoning instruction, not noise across personas)
    on_name, on_sys = sample_system_prompt(random.Random(0), "on")
    off_name, off_sys = sample_system_prompt(random.Random(0), "off")

    problems = _load_gsm8k_heldout(n_problems)
    stats = {"on": {"correct": 0, "len": 0, "fmt": 0},
             "off": {"correct": 0, "len": 0, "fmt": 0}}

    for q, gold in problems:
        for side, sys_text in (("on", on_sys), ("off", off_sys)):
            gen = _gen_one(model, tok, sys_text, q, device, max_new_tokens)
            pred = _norm(extract_numeric_answer(gen))
            if pred is not None and pred == _norm(gold):
                stats[side]["correct"] += 1
            stats[side]["len"] += len(tok.encode(gen, add_special_tokens=False))
            if _WELL_FORMED.search(gen):
                stats[side]["fmt"] += 1

    n = max(len(problems), 1)
    if was_training:
        model.train(True)

    acc_on = stats["on"]["correct"] / n
    acc_off = stats["off"]["correct"] / n
    return {
        "sft_eval/acc_on": acc_on,
        "sft_eval/acc_off": acc_off,
        "sft_eval/acc_delta_on_minus_off": acc_on - acc_off,  # the north star
        "sft_eval/resp_len_on": stats["on"]["len"] / n,
        "sft_eval/resp_len_off": stats["off"]["len"] / n,
        "sft_eval/format_ok_on": stats["on"]["fmt"] / n,
        "sft_eval/format_ok_off": stats["off"]["fmt"] / n,
        "sft_eval/n": n,
        "sft_eval/persona_on": on_name,
        "sft_eval/persona_off": off_name,
    }
