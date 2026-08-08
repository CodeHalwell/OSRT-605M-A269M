"""Venue-agnostic GRPO training loop with BATCHED rollouts and log-probs.

Why this module exists
----------------------
The original loop in `app.py::grpo()` is correct but generates and scores one
sequence at a time:

    prompt_batch = prompt_tensor.expand(cfg.group_size, -1)   # batch 16
    ...
    out = model(comp_ids.unsqueeze(0))                        # batch 1, x2

Per optimiser step that is `grad_accum_steps` (32) generate calls at batch 16,
then ~512 policy forwards and ~512 reference forwards at batch 1. Measured on
an H100: **~5.5 minutes per step**, so 900 steps would be ~75 hours.

That is not a small inefficiency, it is the dominant cost, and it is the same
lesson this model has taught three times now: decode here is LAUNCH-BOUND
(18 effective layers of MoE plus 20 sequential Sinkhorn iterations means
thousands of kernel launches per token, a cost that is fixed regardless of
batch size). Measured throughput by batch:

    batch  16-64 : ~620-690 tok/s
    batch    128 : 3,747 tok/s
    batch    256 : 7,504 tok/s
    batch   1024 : 35,631 tok/s

So the fix is not micro-optimisation, it is simply *stop feeding it small
batches*. This module:

  1. Generates ALL `num_prompts x group_size` rollouts in ONE generate() call.
  2. Runs the policy and reference log-prob passes in chunks of many
     sequences rather than one at a time.

Correctness notes
-----------------
* Generation uses LEFT padding (all completions then start at the same index),
  which is what batched sampling requires when prompts differ in length.
* The log-prob passes use RIGHT padding. With causal attention a real token at
  position i only attends to positions <= i, so it never sees the pad tail —
  the logits for real tokens are identical to an unpadded forward. Padded
  positions produce garbage, which we simply never index.
* Advantages are computed PER PROMPT GROUP (`compute_group_advantages` over the
  group's rewards), then attached to each rollout. A prompt whose rollouts all
  succeed or all fail yields zero advantage everywhere and contributes nothing
  — that is expected, and is why prompt difficulty matters so much.
* The loss matches the original formulation exactly: a direct policy gradient
  weighted by the group-normalised advantage, plus Schulman's non-negative KL
  approximation `exp(log_ratio) - log_ratio - 1`. Only the batching differs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from osrt.rewards import compute_group_advantages, compute_reward


@dataclass
class Rollout:
    """One sampled completion and everything needed to train on it."""

    ids: Tensor          # full sequence: prompt + completion (1-D, on device)
    prompt_len: int
    advantage: float
    reward: float
    correct: bool
    text: str            # decoded completion only (for printing / rewards)


def _left_pad(seqs: list[list[int]], pad_id: int, device) -> tuple[Tensor, Tensor, int]:
    """Left-pad for GENERATION so every completion starts at the same index."""
    width = max(len(s) for s in seqs)
    ids = torch.tensor(
        [[pad_id] * (width - len(s)) + s for s in seqs],
        dtype=torch.long, device=device,
    )
    attn = torch.tensor(
        [[0] * (width - len(s)) + [1] * len(s) for s in seqs],
        dtype=torch.long, device=device,
    )
    return ids, attn, width


def _seq_logprobs(
    model: nn.Module,
    batch_ids: Tensor,          # (B, L) right-padded
    prompt_lens: list[int],
    seq_lens: list[int],
    real_vocab_size: int,
    grad: bool,
) -> list[Tensor]:
    """Per-sequence token log-probs over the COMPLETION span only.

    Right padding is safe here: causal attention means a real token never
    attends to the pad tail, so its logits match an unpadded forward.
    """
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx, torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(batch_ids).logits[:, :, :real_vocab_size].float()
    out: list[Tensor] = []
    for i, (p_len, s_len) in enumerate(zip(prompt_lens, seq_lens)):
        # predict token t from position t-1
        shift_logits = logits[i, p_len - 1:s_len - 1]
        shift_labels = batch_ids[i, p_len:s_len]
        lp = F.log_softmax(shift_logits, dim=-1).gather(
            1, shift_labels.unsqueeze(1)
        ).squeeze(1)
        out.append(lp)
    return out


def generate_rollouts(
    model: nn.Module,
    tok: Any,
    prompts: list[tuple[str, str]],      # (prompt_text, gold_answer)
    cfg: Any,
    device,
    stop_token_ids: list[int] | None = None,
) -> list[list[Rollout]]:
    """Sample `cfg.group_size` rollouts for EVERY prompt in ONE generate call.

    Returns one list of Rollouts per prompt (the group), rewards and advantages
    filled in.
    """
    g = cfg.group_size
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Every prompt repeated group_size times -> a single big batch.
    enc = [tok.encode(p, add_special_tokens=False) for p, _ in prompts]
    flat = [e for e in enc for _ in range(g)]
    ids, attn, width = _left_pad(flat, pad_id, device)

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(
            ids, attention_mask=attn,
            max_new_tokens=cfg.max_gen_len,
            temperature=cfg.temperature,
            top_p=getattr(cfg, "top_p", 1.0),
            eos_token_id=tok.eos_token_id,
            stop_token_ids=stop_token_ids,
        )

    groups: list[list[Rollout]] = []
    for pi, (_, gold) in enumerate(prompts):
        p_len = len(enc[pi])
        rollouts: list[Rollout] = []
        rewards: list[float] = []
        for j in range(g):
            row = out[pi * g + j]
            comp_ids = row[width:]
            # drop right padding introduced by shorter siblings finishing early
            if tok.eos_token_id is not None:
                nz = (comp_ids == tok.eos_token_id).nonzero()
                if len(nz):
                    comp_ids = comp_ids[: int(nz[0]) + 1]
            text = tok.decode(comp_ids, skip_special_tokens=False)
            reward, breakdown = compute_reward(
                text, gold,
                correctness_weight=cfg.correctness_reward,
                format_weight=cfg.format_reward,
                length_penalty=cfg.length_penalty,
                think_open=cfg.think_open, think_close=cfg.think_close,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
                max_tokens=cfg.max_gen_len,
                completion_tokens=len(comp_ids),
                reasoning_bonus=cfg.reasoning_bonus,
                truncation_penalty=cfg.truncation_penalty,
                empty_think_penalty=cfg.empty_think_penalty,
            )
            rewards.append(reward)
            # full sequence = the UNPADDED prompt + this completion
            full = torch.cat([
                torch.tensor(enc[pi], dtype=torch.long, device=device),
                comp_ids.to(device),
            ])
            rollouts.append(Rollout(
                ids=full[: cfg.seq_len], prompt_len=p_len, advantage=0.0,
                reward=reward, correct=bool(breakdown.get("correct")),
                text=text,
            ))
        advs = compute_group_advantages(rewards)
        for r, a in zip(rollouts, advs):
            r.advantage = float(a)
        groups.append(rollouts)
    return groups


def train_on_rollouts(
    model: nn.Module,
    ref_model: nn.Module,
    rollouts: list[Rollout],
    cfg: Any,
    real_vocab_size: int,
    device,
    pad_id: int,
    micro_batch: int = 8,
) -> tuple[float, float]:
    """Batched policy-gradient + KL update over all rollouts of one step.

    Returns (summed loss value, mean approx_kl). Backward is called per
    micro-batch; the caller owns optimizer.step().
    """
    live = [r for r in rollouts if abs(r.advantage) > 1e-8
            and len(r.ids) - r.prompt_len > 0]
    if not live:
        return 0.0, 0.0

    total_loss = 0.0
    total_kl = 0.0
    n = len(live)
    # Longest-first keeps padding waste down within each micro-batch.
    live.sort(key=lambda r: len(r.ids), reverse=True)

    for i in range(0, n, micro_batch):
        chunk = live[i:i + micro_batch]
        max_len = max(len(r.ids) for r in chunk)
        batch = torch.full((len(chunk), max_len), pad_id,
                           dtype=torch.long, device=device)
        for k, r in enumerate(chunk):
            batch[k, : len(r.ids)] = r.ids
        p_lens = [r.prompt_len for r in chunk]
        s_lens = [len(r.ids) for r in chunk]

        pol = _seq_logprobs(model, batch, p_lens, s_lens, real_vocab_size, True)
        ref = _seq_logprobs(ref_model, batch, p_lens, s_lens, real_vocab_size, False)

        loss = torch.zeros((), device=device)
        for lp, rlp, r in zip(pol, ref, chunk):
            adv = torch.tensor(r.advantage, device=device, dtype=torch.float32)
            policy_loss = -(lp * adv).mean()
            log_ratio = rlp.detach() - lp
            approx_kl = (torch.exp(log_ratio) - log_ratio - 1).mean()
            loss = loss + (policy_loss + cfg.kl_coeff * approx_kl)
            total_kl += float(approx_kl.detach())
        loss = loss / n          # mean over ALL live rollouts in the step
        loss.backward()
        total_loss += float(loss.detach())

    return total_loss, total_kl / max(n, 1)


def lr_at_step(step: int, cfg: Any) -> float:
    """Warmup then cosine, honouring lr_anchor_step for re-warmed extensions."""
    anchor = getattr(cfg, "lr_anchor_step", 0)
    eff = max(step - anchor, 0)
    total = max(cfg.total_steps - anchor, 1)
    if eff < cfg.warmup_steps:
        return cfg.peak_lr * eff / max(cfg.warmup_steps, 1)
    prog = (eff - cfg.warmup_steps) / max(total - cfg.warmup_steps, 1)
    return cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
        1 + math.cos(math.pi * min(prog, 1.0))
    )
