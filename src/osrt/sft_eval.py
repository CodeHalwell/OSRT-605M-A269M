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


@torch.no_grad()
def _gen_batch(model, tok, prompts: list[str], device, max_new_tokens: int,
               repetition_penalty: float = 1.0) -> list[str]:
    """Left-pad a list of prompt strings into ONE batch and decode each
    completion. Batched greedy is token-identical to per-prompt generation
    (see tests/test_batched_generate.py), so this is a pure speedup.

    NOTE: when repetition_penalty != 1.0 the pad token id is also penalised
    (generate() scores the whole row incl. left pad); harmless for the default
    greedy path and negligible otherwise.
    """
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    enc = [tok.encode(p, add_special_tokens=False) for p in prompts]
    width = max(len(e) for e in enc)
    ids_rows, mask_rows = [], []
    for e in enc:
        npad = width - len(e)
        ids_rows.append([pad_id] * npad + e)
        mask_rows.append([0] * npad + [1] * len(e))
    input_ids = torch.tensor(ids_rows, dtype=torch.long, device=device)
    attn = torch.tensor(mask_rows, dtype=torch.long, device=device)
    out = model.generate(
        input_ids, attention_mask=attn, max_new_tokens=max_new_tokens,
        temperature=0.0, repetition_penalty=repetition_penalty,
        eos_token_id=tok.eos_token_id,
    )
    # Every row's prompt occupies [0:width] (left-padded), so completions all
    # start at `width`.
    return [tok.decode(out[i, width:], skip_special_tokens=False)
            for i in range(out.shape[0])]


def run_reasoning_eval(
    model: nn.Module, tok, device, *,
    n_problems: int = 50, max_new_tokens: int = 512, seed: int = 0,
    batch_size: int = 16, repetition_penalty: float = 1.0,
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

    # One request per (problem, side); batched left-padded generation.
    requests = []  # (side, gold, prompt_str)
    for q, gold in problems:
        for side, sys_text in (("on", on_sys), ("off", off_sys)):
            prompt = f"<|system|>{sys_text}<|user|>{q}<|assistant|>"
            requests.append((side, gold, prompt))

    for start in range(0, len(requests), batch_size):
        chunk = requests[start:start + batch_size]
        gens = _gen_batch(model, tok, [r[2] for r in chunk], device,
                          max_new_tokens, repetition_penalty)
        for (side, gold, _), gen in zip(chunk, gens):
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
