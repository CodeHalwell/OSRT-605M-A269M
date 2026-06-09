# v6 Midtraining (`midtrain` stage) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a v6 mid-training stage — continued pretraining on the annealed foundation base (step 3500), seq 2048→4096, math-heavy mix, ~9k steps at peak LR 2e-4, with native+trainable HRA.

**Architecture:** Generalize the existing `run_pretrain_extend` loop rather than writing a new one. Three gated changes (a `hra_native` flag that skips the v5 `inject_hra`, a gradient-checkpointing trigger at seq≥4096, a ported periodic `run_eval` call), plus a new `MidtrainConfig`/`MidtrainSanityConfig` and `midtrain`/`midtrain_sanity` Modal entrypoints. Every loop change defaults to the existing v5 behaviour, so legacy stages are untouched.

**Tech Stack:** PyTorch, Modal (H100), Muon+AdamW hybrid optimizer, HF streaming datasets, Weights & Biases, pytest (CPU unit tests with a tiny config).

**Spec:** `docs/superpowers/specs/2026-06-09-v6-midtraining-design.md`

**Key reference facts (verified against code at plan time):**
- Foundation finished: `osrt_v5_final.pt` written at `/vol/checkpoints/v5/` (step 3500, eval loss 3.76 / ppl 43.0).
- Native HRA is built unconditionally in `model.py:1399-1407` (`adapters_a`/`adapters_b` ParameterList from `config.adapter_rank`). No flag.
- `run_pretrain_extend(model_config, extend_cfg, vol, tokenizer_name, ckpt_dir=...)` — `model_config` is param 1 (in scope for the eval port).
- `inject_hra` (v5 path) wraps `nn.Linear` in `HRALinear` (`original.weight`, `adapter_a`, `adapter_b`) — a DIFFERENT layout from native HRA.
- `load_model_state_or_raise` (`train.py:150`) uses `strict=False` then RAISES `RuntimeError` on any missing/unexpected key.
- `PretrainConfig` (`train_config.py:98`) has NO `__init__` — pure class-level attributes, so subclass-override works.
- The extend loop's HRA-inject block is at `train.py:1447-1459`; the seq trigger at `train.py:1679-1685`; the ckpt-save block at `train.py:1906`.
- Test pattern: `tests/test_model.py` has `tiny_config(**overrides)` → `OSRTConfig(dim=128, heads=4, head_dim=32, vocab_size=512, num_blocks=2, recursive_loops=2, num_routed_experts=8, top_k_experts=2, expert_hidden=64, shared_expert_hidden=128, max_position_embeddings=64)`.

---

## File Structure

| File | Responsibility | Action |
|---|---|---|
| `src/osrt/train.py` | `run_pretrain_extend`: gate `inject_hra` behind `hra_native`; lower ckpt trigger to seq≥4096; port periodic `run_eval` | Modify (3 spots) |
| `src/osrt/train_config.py` | `MidtrainConfig` + `MidtrainSanityConfig` | Add (append after the extend configs) |
| `app.py` | `midtrain()` / `midtrain_sanity()` functions + `run_midtrain` / `run_midtrain_sanity` entrypoints | Add |
| `tests/test_midtrain_config.py` | Unit tests: config values, native-HRA clean load, inject-gate behaviour | Create |

---

## Task 1: `hra_native` flag — gate `inject_hra` in `run_pretrain_extend`

**Files:**
- Modify: `src/osrt/train.py:1447-1459`
- Test: `tests/test_midtrain_config.py`

The core fix. v6's foundation checkpoint already carries native `adapters_a`/`adapters_b`. The v5 loop unconditionally calls `inject_hra`, which would add mismatched `HRALinear` keys and make `load_model_state_or_raise` throw. Gate it.

- [ ] **Step 1: Write the failing test**

Create `tests/test_midtrain_config.py`:

```python
"""Unit tests for the v6 midtrain stage: config + native-HRA load gate."""

import pytest
import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM
from osrt.train import load_model_state_or_raise
from osrt.hra import inject_hra


def tiny_config(**overrides) -> OSRTConfig:
    """Small config for fast CPU tests (mirrors tests/test_model.py)."""
    defaults = dict(
        dim=128, heads=4, head_dim=32,
        vocab_size=512, real_vocab_size=512,
        num_blocks=2, recursive_loops=2,
        num_routed_experts=8, top_k_experts=2,
        expert_hidden=64, shared_expert_hidden=128,
        adapter_rank=16, adapter_alpha=16.0,
        max_position_embeddings=64,
    )
    defaults.update(overrides)
    return OSRTConfig(**defaults)


def test_native_hra_checkpoint_loads_without_injection():
    """A model built straight from config (native HRA) round-trips its
    own state_dict with no inject_hra — the hra_native=True path."""
    cfg = tiny_config()
    src = OSRTForCausalLM(cfg)
    state = src.state_dict()

    dst = OSRTForCausalLM(cfg)  # native HRA present, NO inject_hra
    # Must not raise: keys match exactly.
    load_model_state_or_raise(dst, state, context="native-hra test")


def test_inject_hra_on_native_model_breaks_load():
    """Proves WHY the gate is needed: injecting HRALinear onto a model
    that already has native HRA changes the key namespace, so loading a
    native checkpoint then raises."""
    cfg = tiny_config()
    native_state = OSRTForCausalLM(cfg).state_dict()

    injected = OSRTForCausalLM(cfg)
    inject_hra(injected, rank=cfg.adapter_rank, scale=1.0,
               freeze_pretrained=False)

    with pytest.raises(RuntimeError, match="key mismatch"):
        load_model_state_or_raise(
            injected, native_state, context="inject-breaks-load test"
        )
```

- [ ] **Step 2: Run the tests to verify the first passes and the second documents the failure mode**

Run: `PYTHONPATH=src python -m pytest tests/test_midtrain_config.py -v -k "native_hra or inject_hra"`
Expected: `test_native_hra_checkpoint_loads_without_injection` PASSES (native HRA needs no injection — this is the invariant the gate relies on); `test_inject_hra_on_native_model_breaks_load` PASSES (confirms inject-on-native raises). If the second does NOT raise, the `inject_hra` layout overlaps native keys — STOP and re-examine `hra.py` before proceeding.

> Note: these two tests pass on the *current* code because they test model/load invariants, not the gate. They lock the behaviour the gate depends on. The gate itself is exercised by Task 5's sanity run (real checkpoint load).

- [ ] **Step 3: Apply the gate in `run_pretrain_extend`**

In `src/osrt/train.py`, replace the block at lines 1447-1459 (the `# ── HRA injection (BEFORE state_dict load) ──` block) with:

```python
    # ── HRA injection (BEFORE state_dict load) ─────────────────────
    # v5 path: the foundation model had NO HRA, so this stage injects
    #   HRALinear wrappers before loading an SFT checkpoint whose
    #   state_dict contains them.
    # v6 path (hra_native=True): HRA is built inline from config
    #   (model.py adapters_a/adapters_b ParameterList) and is ALREADY in
    #   the foundation checkpoint. Injecting HRALinear here would graft a
    #   second, mismatched layout and make load_model_state_or_raise throw.
    hra_native = getattr(extend_cfg, "hra_native", False)
    if extend_cfg.hra_enabled and not hra_native:
        from osrt.hra import inject_hra
        print(f"Injecting HRA before load (rank={extend_cfg.hra_rank})...")
        inject_hra(
            model,
            rank=extend_cfg.hra_rank,
            scale=getattr(extend_cfg, "hra_scale", 1.0),
            freeze_pretrained=False,
        )
        with_hra_params = sum(p.numel() for p in model.parameters())
        added = with_hra_params - base_params
        print(f"  HRA injected: +{added:,} params ({added / 1e6:.1f}M)")
    elif hra_native:
        print(
            "HRA is native (built from config) — skipping inject_hra; "
            "foundation checkpoint already carries adapters_a/adapters_b."
        )
```

- [ ] **Step 4: Re-run the tests + full suite to confirm no regression**

Run: `PYTHONPATH=src python -m pytest tests/test_midtrain_config.py -v && PYTHONPATH=src python -m pytest tests/ -q`
Expected: midtrain tests PASS; full suite shows no NEW failures vs baseline.

- [ ] **Step 5: Commit**

```bash
git add src/osrt/train.py tests/test_midtrain_config.py
git commit -m "feat(midtrain): gate inject_hra behind hra_native flag

v6 foundation HRA is native (model.py adapters_a/b from config) and
already in the checkpoint. hra_native=True skips the v5 inject_hra so
the load is a clean key-match instead of a grafted, mismatched layout.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Gradient-checkpointing trigger at seq ≥ 4096

**Files:**
- Modify: `src/osrt/train.py:1679-1685`

The extend loop enables activation checkpointing only at seq≥8192. Foundation needs it at seq 2048 (39.5GB). At seq 4096 without it, the v6 model (MTP + mHC 4-stream + 8 experts) would OOM. Drive it from the config, else trigger at seq≥4096.

- [ ] **Step 1: Apply the change**

In `src/osrt/train.py`, replace lines 1679-1685 (the `# Activation checkpointing only for very long seq.` block) with:

```python
    # Activation checkpointing: foundation needs it at seq 2048 (39.5GB);
    # the v6 model (MTP + mHC 4-stream + 8 experts) is heavier than the v5
    # extend model this loop was written for. Drive from the config when
    # set, else trigger at seq>=4096 (was 8192 — too high for v6).
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    base = inner.model if hasattr(inner, "model") else inner
    need_ckpt = bool(
        getattr(extend_cfg, "gradient_checkpointing", seq_len >= 4096)
    )
    if (hasattr(base, "gradient_checkpointing")
            and base.gradient_checkpointing != need_ckpt):
        base.gradient_checkpointing = need_ckpt
    print(f"    Gradient checkpointing: {need_ckpt} (seq_len={seq_len})")
```

> Note for reviewer: this is the ONE change that alters a path a v5 stage could hit. v5 extend1 ran seq 4096 uncheckpointed; after this it would checkpoint. extend1 is a completed, non-rerun stage, so this is acceptable — a conscious choice, not a silent regression.

- [ ] **Step 2: Verify it imports / parses**

Run: `PYTHONPATH=src python -c "import osrt.train; print('train.py OK')"`
Expected: `train.py OK` (no SyntaxError).

- [ ] **Step 3: Commit**

```bash
git add src/osrt/train.py
git commit -m "feat(midtrain): checkpoint at seq>=4096 (was 8192)

The v6 model OOMs at seq 4096 without activation checkpointing. Drive
the trigger from the train config when set, else seq>=4096.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Port periodic `run_eval` into the extend loop

**Files:**
- Modify: `src/osrt/train.py:1906` (insert before the checkpoint-save block)

`run_eval` is a clean module-level fn (`train.py:207`). A 9k-step run needs an eval trend (the scratchpad gate depends on it). The block is a no-op for v5 stages (their `eval_interval` defaults to 9_999_999).

- [ ] **Step 1: Insert the eval block**

In `src/osrt/train.py`, find the checkpoint-save block that begins at line 1906 (`if step > 0 and step % extend_cfg.ckpt_interval == 0:`). Immediately **before** that `if`, insert:

```python
        # ── Periodic held-out eval ──────────────────────────────────
        # Ported from run_training. Required for a 9k-step run: the
        # pre->midtrain gate (review/learnings-scratchpad.md) needs an
        # eval trend. No-op for v5 stages (eval_interval defaults to
        # 9_999_999 there).
        eval_interval = getattr(extend_cfg, "eval_interval", 0)
        if eval_interval and step > 0 and step % eval_interval == 0:
            eval_metrics = run_eval(
                model,
                tokenizer_name,
                seq_len,
                batch_size,
                getattr(extend_cfg, "eval_steps", 20),
                device,
                model_config.real_vocab_size,
            )
            print(
                f"  EVAL step {step} | "
                f"loss {eval_metrics['eval/loss']:.4f} | "
                f"ppl {eval_metrics['eval/perplexity']:.1f}",
                flush=True,
            )
            if use_wandb:
                wandb.log(eval_metrics, step=step)
```

> The args mirror `run_training`'s existing positional call (`train.py:1254-1258`): `run_eval(model, tokenizer_name, seq_len, batch_size, eval_steps, device, real_vocab_size)`. `model_config` is `run_pretrain_extend`'s first parameter — in scope here.

- [ ] **Step 2: Verify it imports / parses**

Run: `PYTHONPATH=src python -c "import osrt.train; print('train.py OK')"`
Expected: `train.py OK`.

- [ ] **Step 3: Commit**

```bash
git add src/osrt/train.py
git commit -m "feat(midtrain): port periodic run_eval into the extend loop

A 9k-step midtrain run needs a held-out eval trend (the pre->midtrain
gate depends on it). Gated on eval_interval, so v5 stages (default
9_999_999) stay unaffected.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: `MidtrainConfig` + `MidtrainSanityConfig`

**Files:**
- Modify: `src/osrt/train_config.py` (append after the last `PretrainExtend*`/`MOPD`/`SystemSFT` config, before the `SFTConfig` block — i.e. with the pretraining-family configs)
- Test: `tests/test_midtrain_config.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_midtrain_config.py`:

```python
def test_midtrain_config_values():
    """MidtrainConfig encodes the locked decisions (spec §2/§4.4)."""
    from osrt.train_config import MidtrainConfig

    cfg = MidtrainConfig()
    assert cfg.total_steps == 9_000
    assert cfg.peak_lr == 2e-4
    assert cfg.min_lr == 2e-5
    assert cfg.warmup_steps == 150
    assert cfg.lr_anchor_step == 0
    # native + trainable HRA
    assert cfg.hra_native is True
    assert cfg.hra_frozen is False
    assert cfg.hra_enabled is True
    # router exploration off
    assert cfg.router_gumbel_tau_init == 0.0
    # gate disabled
    assert cfg.early_stop_check_step > 9_000
    # resume + prefix
    assert cfg.pretrained_checkpoint.endswith("osrt_v5_final.pt")
    assert cfg.stage_prefix == "midtrain"
    assert cfg.gradient_checkpointing is True


def test_midtrain_phase_is_seq4096_math_mix():
    """The single 'extend' phase is seq 4096 with the knowledge mix."""
    from osrt.train_config import MidtrainConfig

    phase = MidtrainConfig().phases["extend"]
    assert phase["seq_len"] == 4096
    names = {d["name"] for d in phase["datasets"]}
    assert "nemotron-cc-math-4plus" in names
    assert "fineweb-edu" in names           # general anchor retained
    assert "cosmopedia-openstax" in names
    # weights: math/STEM/reasoning should dominate (~0.65)
    math_sci = sum(
        d["weight"] for d in phase["datasets"]
        if d["name"] in {
            "nemotron-cc-math-4plus", "nemotron-stem",
            "nemotron-math-textbooks", "nemotron-reasoning",
        }
    )
    assert 0.60 <= math_sci <= 0.70


def test_midtrain_sanity_writes_no_final():
    """Sanity config is a short probe that won't clobber a real final."""
    from osrt.train_config import MidtrainSanityConfig

    cfg = MidtrainSanityConfig()
    assert cfg.total_steps == 30
    assert cfg.save_final_checkpoint is False
    assert cfg.stage_prefix == "midtrain-sanity"
    assert cfg.compile_enabled is False
    # inherits the real seq/mix so VRAM is measured at production size
    assert cfg.phases["extend"]["seq_len"] == 4096
```

- [ ] **Step 2: Run to verify it fails**

Run: `PYTHONPATH=src python -m pytest tests/test_midtrain_config.py -v -k "midtrain_config or midtrain_phase or midtrain_sanity"`
Expected: FAIL with `ImportError: cannot import name 'MidtrainConfig'`.

- [ ] **Step 3: Add the configs**

In `src/osrt/train_config.py`, after the `SystemSFTConfig` class (the last pretraining-family config, ~line 1050) and before `class SFTConfig`, add:

```python
class MidtrainConfig(PretrainConfig):
    """v6 mid-training: continued PRETRAINING on the foundation base.

    Resumes from the annealed v6 foundation checkpoint (step 3500),
    re-warms a fresh cosine at a real continued-pretraining LR (2e-4),
    doubles context to seq 4096, and trains the math-heavy knowledge mix.

    Unlike the v5 PretrainExtend* stages this does NOT resume from an
    SFT/GRPO checkpoint, so there is no chat-format investment to protect:
    HRA stays TRAINABLE and the LR is ~33% of foundation peak, not the
    2.5% the v5 stages used.

    HRA is NATIVE here (built inline from the preset's adapter_rank=256
    and already present in the foundation checkpoint), so hra_native=True
    skips inject_hra — see run_pretrain_extend.

    See docs/superpowers/specs/2026-06-09-v6-midtraining-design.md.
    """

    # ── Schedule (fresh re-warm cosine) ──────────────────────────────
    total_steps: int = 9_000
    warmup_steps: int = 150          # re-warm from the annealed base
    lr_anchor_step: int = 0          # fresh cosine (foundation already cooled)
    peak_lr: float = 2e-4            # ~33% of foundation's 6e-4
    min_lr: float = 2e-5
    weight_decay: float = 0.1        # softer than foundation's 0.3
    grad_clip: float = 1.0

    optimizer_name: str = "muon"
    muon_lr: float = 6.6e-3          # proportional: (2e-4/6e-4) * 0.02
    muon_min_lr: float = 6.6e-4

    log_interval: int = 50
    ckpt_interval: int = 500         # ~18 ckpts; bounds Modal-kill loss
    eval_interval: int = 750         # ported eval (run_pretrain_extend)
    eval_steps: int = 20

    # ── Router exploration: off (router is well-formed) ──────────────
    router_gumbel_tau_init: float = 0.0
    router_gumbel_tau_final: float = 0.0
    router_gumbel_anneal_steps: int = 1

    # ── Early-stop gate: disabled (cold-start gate doesn't apply) ────
    early_stop_check_step: int = 9_999_999

    # ── HRA: native + trainable ──────────────────────────────────────
    hra_enabled: bool = True
    hra_rank: int = 256
    hra_scale: float = 1.0
    hra_native: bool = True          # skip inject_hra (run_pretrain_extend)
    hra_frozen: bool = False         # trainable

    # ── Resume / lineage ─────────────────────────────────────────────
    # Foundation final ckpt (run_training writes osrt_v5_final.pt; the
    # 500-step interval also leaves osrt_v5_step_3500.pt). If a run was
    # killed before the final save, repoint at osrt_v5_step_3500.pt.
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_final.pt"
    gradient_checkpointing: bool = True

    # Distinct prefix — osrt_v5_midtrain_step_*.pt, no collision with
    # foundation's osrt_v5_step_*.pt resume scan.
    stage_prefix: str = "midtrain"

    wandb_run_name: str = "osrt-v6-midtrain"
    wandb_run_id: str = ""

    # ── Data mix: the knowledge phase (seq 4096, math-heavy) ─────────
    # Single phase keyed "extend" (run_pretrain_extend reads
    # phases["extend"]). Content mirrors PretrainConfig.phases["knowledge"].
    phases: dict = {  # noqa: RUF012
        "extend": {
            "start": 0,
            "end": 9_000,
            "seq_len": 4096,
            "batch_size": 6,         # knowledge-phase sizing; sanity-gated
            "grad_accum_steps": 11,
            "datasets": [
                {"name": "nemotron-cc-math-4plus",
                 "hf_id": "nvidia/Nemotron-CC-Math-v1",
                 "hf_config": "4plus", "weight": 0.25},
                {"name": "nemotron-stem",
                 "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                 "hf_config": "Nemotron-Pretraining-STEM-SFT", "weight": 0.15},
                {"name": "nemotron-math-textbooks",
                 "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                 "hf_config": "Nemotron-Pretraining-Math-Textbooks",
                 "weight": 0.15},
                {"name": "nemotron-reasoning",
                 "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                 "hf_config": "Nemotron-Pretraining-InfiniByte-Reasoning",
                 "weight": 0.10},
                {"name": "fineweb-edu",
                 "hf_id": "HuggingFaceFW/fineweb-edu", "weight": 0.15},
                {"name": "nemotron-code-syn-qa",
                 "hf_id": "nvidia/Nemotron-Pretraining-Code-v2",
                 "hf_config": "Synthetic-Question-Answering", "weight": 0.10},
                {"name": "cosmopedia-openstax",
                 "hf_id": "HuggingFaceTB/cosmopedia",
                 "hf_config": "openstax", "weight": 0.10},
            ],
        },
    }

    # DataLoader: 7 streams. Keep workers modest to stay under HF Hub's
    # per-client connection ceiling (extend2 hit resets at 4x9=36 conns).
    dataloader_num_workers: int = 2
    dataloader_prefetch_factor: int = 2
    compile_enabled: bool = True


class MidtrainSanityConfig(MidtrainConfig):
    """30-step VRAM/throughput probe at the REAL seq/batch before the
    $150 launch. Writes no final checkpoint, distinct prefix, eager mode
    so step events appear immediately."""

    total_steps: int = 30
    warmup_steps: int = 5
    ckpt_interval: int = 9_999_999
    eval_interval: int = 9_999_999
    save_final_checkpoint: bool = False
    stage_prefix: str = "midtrain-sanity"
    wandb_run_name: str = "osrt-v6-midtrain-sanity"
    compile_enabled: bool = False
```

- [ ] **Step 4: Run to verify pass**

Run: `PYTHONPATH=src python -m pytest tests/test_midtrain_config.py -v`
Expected: all tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/osrt/train_config.py tests/test_midtrain_config.py
git commit -m "feat(midtrain): add MidtrainConfig + MidtrainSanityConfig

v6 mid-training config: 9k steps, seq 4096, peak LR 2e-4 re-warm cosine,
native+trainable HRA, math-heavy knowledge mix, resume from
osrt_v5_final.pt. Sanity subclass is a 30-step eager VRAM probe.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Modal entrypoints `midtrain` / `midtrain_sanity`

**Files:**
- Modify: `app.py` (add after `pretrain_extend` / its entrypoints; mirror `pretrain()`'s v6 setup)

These build the model from the `OSRT_605M_A288M` preset (a bare `OSRTConfig()` would silently fall back to the v5 363M shape), use the v6 tokenizer volume, and pass `fused_cross_entropy_chunks=8` like `pretrain()`.

- [ ] **Step 1: Verify the module-level names this code depends on exist**

Run: `cd /Users/danielhalwell/nano-osrt-100m && grep -nE "^image = |^vol = |^v6_tokenizer_vol = |^hf_cache_vol = " app.py`
Expected: all four module-level names are defined (used by the decorator). If `hf_cache_vol` is named differently, use the actual name from `pretrain()`'s decorator (`app.py:352-359`).

- [ ] **Step 2: Add the two functions + two entrypoints**

In `app.py`, after the `pretrain_extend` block (and its section), add:

```python
# =============================================================================
# MIDTRAIN — v6 mid-training (continued pretraining, seq 4096, math mix)
# =============================================================================
# Generalizes run_pretrain_extend via hra_native=True (skip inject_hra: v6
# HRA is native + already in the foundation ckpt). See MidtrainConfig and
# docs/superpowers/specs/2026-06-09-v6-midtraining-design.md.


def _run_midtrain(cfg_cls):
    """Shared body for midtrain + midtrain_sanity (differ only by config)."""
    from transformers import AutoTokenizer

    from osrt.presets import build_config
    from osrt.train import run_pretrain_extend

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    model_config = build_config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
    )
    cfg = cfg_cls()
    print(
        f"{cfg.__class__.__name__}: {cfg.total_steps} steps @ seq "
        f"{cfg.phases['extend']['seq_len']}, peak LR {cfg.peak_lr}, "
        f"HRA native+trainable, resume {cfg.pretrained_checkpoint}"
    )
    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def midtrain():
    """v6 mid-training: continued pretraining from the foundation base,
    seq 4096, math-heavy mix, ~9k steps. See MidtrainConfig."""
    from osrt.train_config import MidtrainConfig
    _run_midtrain(MidtrainConfig)


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def midtrain_sanity():
    """30-step VRAM/throughput probe at real seq 4096 / batch 6 before
    the $150 launch. See MidtrainSanityConfig."""
    from osrt.train_config import MidtrainSanityConfig
    _run_midtrain(MidtrainSanityConfig)


@app.local_entrypoint()
def run_midtrain():
    """Spawn v6 mid-training (fire-and-forget)."""
    call = midtrain.spawn()
    print(f"Spawned v6 midtrain — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_midtrain_sanity():
    """Spawn the 30-step v6 midtrain VRAM/throughput sanity probe."""
    call = midtrain_sanity.spawn()
    print(f"Spawned v6 midtrain sanity — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")
```

> If Step 1 showed `hf_cache_vol` has a different name, substitute it in both `@app.function` decorators.

- [ ] **Step 3: Verify the app imports (syntax + Modal graph build)**

Run: `cd /Users/danielhalwell/nano-osrt-100m && python -c "import ast; ast.parse(open('app.py').read()); print('app.py parses')"`
Expected: `app.py parses`.

- [ ] **Step 4: Commit**

```bash
git add app.py
git commit -m "feat(midtrain): add midtrain + midtrain_sanity Modal entrypoints

Build from the OSRT_605M_A288M preset (not bare OSRTConfig, which falls
back to v5 363M), v6 tokenizer volume, fused-CE chunks. Shared _run_midtrain
body; .spawn() entrypoints per the project launch rule.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Pre-launch gate — sanity run on Modal (manual, not a unit test)

**Files:** none (operational). This is the non-negotiable gate from spec §6.

- [ ] **Step 1: Confirm the foundation checkpoint is present**

Run: `cd /Users/danielhalwell/nano-osrt-100m && modal volume ls osrt-checkpoints v5 | grep -E "osrt_v5_final|osrt_v5_step_3500"`
Expected: `osrt_v5_final.pt` exists. (Foundation logs already confirmed it was written.) If only `osrt_v5_step_3500.pt` exists, set `MidtrainConfig.pretrained_checkpoint` to that path.

- [ ] **Step 2: Switch git identity for any push, then launch the sanity probe**

Run: `gh auth switch --user CodeHalwell` (per the project rule before git push; harmless if no push follows).
Run: `cd /Users/danielhalwell/nano-osrt-100m && modal run --detach app.py::run_midtrain_sanity`
Expected: prints a `call_id`. Then watch logs: `modal app logs <app-id>`.

- [ ] **Step 3: Read the sanity output against the pass criteria**

Confirm in the logs:
- `HRA is native (built from config) — skipping inject_hra` (Task 1 gate fired).
- `Clean load: all keys matched.` (native checkpoint loaded with no mismatch).
- `Gradient checkpointing: True (seq_len=4096)` (Task 2 trigger fired).
- VRAM stays under 80GB (target ≲ 70GB) across the 30 steps.
- `tok/s` is reported and non-degenerate; no OOM.

**If it OOMs:** edit `MidtrainConfig.phases["extend"]` → `batch_size=4, grad_accum_steps=16`, re-run the sanity. Record the working batch in the spec.

- [ ] **Step 4: Launch the full run (only after the sanity passes)**

Run: `cd /Users/danielhalwell/nano-osrt-100m && modal run --detach app.py::run_midtrain`
Expected: prints a `call_id`; W&B run `osrt-v6-midtrain` appears. Checkpoints land at `/vol/checkpoints/v5/osrt_v5_midtrain_step_*.pt` every 500 steps; eval logs every 750 steps.

---

## Self-Review

**1. Spec coverage:**
- §3 HRA constraint → Task 1 (gate) + tests. ✓
- §4.1 inject gate → Task 1. ✓
- §4.2 ckpt trigger → Task 2. ✓
- §4.3 eval port → Task 3. ✓
- §4.4 configs → Task 4 + tests. ✓
- §4.5 entrypoints → Task 5. ✓
- §5 data mix → encoded in Task 4 config + asserted in `test_midtrain_phase_is_seq4096_math_mix`. ✓
- §6 pre-launch gate → Task 6. ✓
- §7 lineage/resume → `stage_prefix="midtrain"` (Task 4) + Task 6 step 4. ✓
- §8 blast radius → each loop change gated (Tasks 1-3 default to v5 behaviour). ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows full code; every command has expected output. ✓

**3. Type/name consistency:** `hra_native` (Task 1 read via `getattr` ↔ Task 4 set as attr), `gradient_checkpointing` (Task 2 ↔ Task 4), `eval_interval`/`eval_steps` (Task 3 ↔ Task 4), `stage_prefix="midtrain"` (Task 4 ↔ Task 6), `_run_midtrain(cfg_cls)` defined and called consistently (Task 5). `run_eval` arg order matches `train.py:1254`. ✓

**Coverage gap found + closed:** none. One conscious risk flagged (Task 2's effect on v5 extend1) is documented inline, not silent.

---

## Notes for the executor

- **Run unit tests on CPU** with the tiny config — never instantiate the full 601M model in a test.
- **Baseline the suite first:** `PYTHONPATH=src python -m pytest tests/ -q` before Task 1, so "no new failures" is meaningful (the repo reports 144 passing).
- **Tasks 1-5 are local/offline** (code + CPU tests). **Task 6 is Modal** ($ cost) — do not run it without the user's go-ahead on spend.
- **Launch rule:** all Modal runs via `.spawn()` entrypoints; `gh auth switch --user CodeHalwell` before any git push.
