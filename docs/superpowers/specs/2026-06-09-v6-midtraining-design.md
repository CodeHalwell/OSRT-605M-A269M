# v6 Midtraining (`midtrain` stage) — Design

**Date:** 2026-06-09
**Status:** Approved (design); implementation plan pending
**Author:** training session (OSRT-605M v6)

---

## 1. Goal

Continued **pretraining** on top of the annealed v6 foundation base
(step 3500) to:

1. **Inject math / STEM / reasoning capability** via the math-heavy
   `knowledge`-phase data mix.
2. **Double the context window** from seq 2048 → 4096.

This runs **before any SFT**. It is *not* format-preserving fine-tuning —
there is no chat/answer-format investment to protect (unlike the v5
`PretrainExtend*` stages, which resumed from SFT/GRPO checkpoints). That
single distinction drives most of the decisions below.

## 2. Decisions (locked)

| Decision | Value | Rationale |
|---|---|---|
| Compute budget | ~$150 / **~9,000 steps** | Solid capability-injection pass at seq 4096 (~2.4B tokens). |
| Sequence length | **4096** | Doubles context + switches to the math mix. |
| HRA adapters | **Trainable** | v6 HRA is native + *pretrained* (never SFT'd) — keep learning it. v5 froze HRA only because it held an SFT delta. |
| Handoff | **Straight from step 3500** | The cosine-annealed-to-`min_lr` foundation checkpoint is clean; re-warm directly. |
| Re-warm peak LR | **2e-4** (~33% of foundation's 6e-4) | Genuine new-capability learning, gentler than a cold start. Cools to 2e-5 over 9k steps. The v5 extend LR (1.5e-5) would badly under-train a continued-pretraining pass. |

## 3. The core technical constraint (why this is new code, not config reuse)

There are **two completely different HRA mechanisms** in the codebase:

1. **v6 foundation HRA is native / inline.** `model.py:1400-1408` builds
   `self.adapters_a` / `self.adapters_b` as `nn.ParameterList`s directly
   from `config.adapter_rank=256`. The step-3500 checkpoint already
   contains these trained tensors under native ParameterList keys.
   `run_training` (the foundation loop) has **no** `inject_hra` call.

2. **`inject_hra` (the v5 path, `hra.py`)** wraps each `nn.Linear` in an
   `HRALinear` module — a *different* parameter layout
   (`...original.weight`, `adapter_a`, `adapter_b`).

`run_pretrain_extend` (the v5 mid-training loop) **unconditionally calls
`inject_hra` before load** (`train.py:1448`). Pointed at a v6 checkpoint,
it would graft a second, redundant, mismatched HRA layout on top of the
native one and the `state_dict` load would fail on key mismatch.

**Therefore v6 midtraining cannot reuse `run_pretrain_extend` as-is.**
We generalize that loop (Approach B, chosen over a from-scratch
`run_midtrain` because the extend loop already owns the load-bearing,
hard-to-rebuild machinery: load-specific-checkpoint-as-init, distinct
`stage_prefix` resume scan, and `lr_anchor_step` re-warm).

## 4. Implementation

### 4.1 Generalize `run_pretrain_extend` — gate `inject_hra` behind `hra_native`

The one **conflict** with v6 is a single unconditional call. Gate it.

`src/osrt/train.py`, in `run_pretrain_extend`, replace the HRA-injection
block (currently `train.py:1447-1459`):

```python
    # ── HRA injection (BEFORE state_dict load) ─────────────────────
    # v5 path: the foundation model had NO HRA, so the extend stage
    #   injects HRALinear wrappers before loading the SFT checkpoint
    #   whose state_dict contains them.
    # v6 path (hra_native=True): HRA is built inline from config
    #   (model.py adapters_a/adapters_b ParameterList) and is ALREADY
    #   present in the foundation checkpoint. Injecting HRALinear here
    #   would graft a second, mismatched layout and break the load.
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
        print("HRA is native (built from config) — skipping inject_hra; "
              "foundation checkpoint already carries adapters_a/adapters_b.")
```

The post-load freeze block (`train.py:1480`) is already gated on
`hra_frozen`; with `hra_frozen=False` (our config) it is a no-op, so HRA
stays trainable. Native HRA params are 2D, so `build_param_groups`
already routes them to **Muon** — exactly how foundation trained them. No
differential HRA LR is added.

### 4.2 Lower the gradient-checkpointing trigger to seq ≥ 4096

`run_pretrain_extend` currently enables activation checkpointing only at
seq ≥ 8192 (`train.py:1682`). Foundation needs checkpointing already at
seq 2048 (39.5GB). At seq 4096 *without* it, the v6 model (MTP heads +
mHC 4-stream + 8 experts) would very likely OOM mid-run.

Replace `train.py:1679-1685`:

```python
    # Activation checkpointing: foundation needs it at seq 2048 (39.5GB);
    # the v6 model (MTP + mHC 4-stream + 8 experts) is heavier than the v5
    # extend model this loop was written for. Drive it from the config when
    # set, else fall back to a seq>=4096 trigger (was 8192 — too high for v6).
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    base = inner.model if hasattr(inner, "model") else inner
    need_ckpt = bool(getattr(extend_cfg, "gradient_checkpointing", seq_len >= 4096))
    if (hasattr(base, "gradient_checkpointing")
            and base.gradient_checkpointing != need_ckpt):
        base.gradient_checkpointing = need_ckpt
    print(f"    Gradient checkpointing: {need_ckpt} (seq_len={seq_len})")
```

### 4.3 Port periodic eval into the extend loop

`run_eval` (`train.py:207`) is a clean, self-contained module-level
function — no infra to lift, just call it on an interval. The held-out
FineWeb-Edu slice (100M-record skip) is seq-agnostic and cached on first
call. A 9k-step run with no eval flies blind; the pre→midtrain gate in
`review/learnings-scratchpad.md` depends on eval trend.

Insert into the loop, immediately **before** the checkpoint-save block
(`train.py:1906`):

```python
        # ── Periodic held-out eval ──────────────────────────────────
        # Ported from run_training. Required for a 9k-step run: the
        # pre->midtrain gate (review/learnings-scratchpad.md) needs an
        # eval trend, and the extend loop historically skipped eval
        # (fine for 1.8k steps, not for 9k).
        eval_interval = getattr(extend_cfg, "eval_interval", 0)
        if eval_interval and step > 0 and step % eval_interval == 0:
            eval_metrics = run_eval(
                model,
                tokenizer_name,
                seq_len=seq_len,
                batch_size=batch_size,
                eval_steps=getattr(extend_cfg, "eval_steps", 20),
                device=device,
                real_vocab_size=model_config.real_vocab_size,
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

(`eval_interval` defaults to `9_999_999` in the v5 extend configs, so this
block stays a no-op for them — zero blast radius on legacy stages.)

### 4.4 New config: `MidtrainConfig`

`src/osrt/train_config.py`. Subclass `PretrainConfig` (NOT
`PretrainExtendConfig` — we want the foundation defaults, then override).

```python
class MidtrainConfig(PretrainConfig):
    """v6 mid-training: continued PRETRAINING on the foundation base.

    Resumes from the annealed v6 foundation checkpoint (step 3500),
    re-warms a fresh cosine at a real continued-pretraining LR (2e-4),
    doubles context to seq 4096, and trains the math-heavy knowledge mix.

    Unlike the v5 PretrainExtend* stages this does NOT resume from an
    SFT/GRPO checkpoint, so there is no chat-format investment to
    protect: HRA stays TRAINABLE and the LR is ~33% of foundation peak,
    not the 2.5% the v5 stages used.

    HRA is NATIVE here (built inline from the preset's adapter_rank=256
    and already present in the foundation checkpoint), so hra_native=True
    skips inject_hra — see run_pretrain_extend.
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
    muon_lr: float = 6.6e-3          # proportional: 2e-4/6e-4 * 0.02
    muon_min_lr: float = 6.6e-4

    log_interval: int = 50
    ckpt_interval: int = 500         # ~18 ckpts; bounds Modal-kill loss
    eval_interval: int = 750         # ported eval (4.3)
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
    hra_native: bool = True          # NEW flag — skip inject_hra (4.1)
    hra_frozen: bool = False         # trainable

    # ── Resume / lineage ─────────────────────────────────────────────
    # Foundation final ckpt. run_training writes osrt_v5_final.pt at the
    # end of a completed run (train.py:1347) AND osrt_v5_step_3500.pt at
    # the 500-step interval. Point at _final; if the run was killed before
    # the final save, repoint at osrt_v5_step_3500.pt.
    pretrained_checkpoint: str = "/vol/checkpoints/v5/osrt_v5_final.pt"
    gradient_checkpointing: bool = True   # drives 4.2

    # Distinct prefix — osrt_v5_midtrain_step_*.pt, no collision with
    # foundation's osrt_v5_step_*.pt resume scan.
    stage_prefix: str = "midtrain"

    wandb_run_name: str = "osrt-v6-midtrain"
    wandb_run_id: str = ""

    # ── Data mix: the knowledge phase (seq 4096, math-heavy) ─────────
    # Single phase keyed "extend" (the loop reads phases["extend"]).
    # Content is PretrainConfig.phases["knowledge"] verbatim.
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
                 "hf_config": "Nemotron-Pretraining-Math-Textbooks", "weight": 0.15},
                {"name": "nemotron-reasoning",
                 "hf_id": "nvidia/Nemotron-Pretraining-Specialized-v1",
                 "hf_config": "Nemotron-Pretraining-InfiniByte-Reasoning", "weight": 0.10},
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
    $150 launch. Writes no final checkpoint, distinct prefix."""
    total_steps: int = 30
    warmup_steps: int = 5
    ckpt_interval: int = 9_999_999
    eval_interval: int = 9_999_999
    save_final_checkpoint: bool = False
    stage_prefix: str = "midtrain-sanity"
    wandb_run_name: str = "osrt-v6-midtrain-sanity"
    compile_enabled: bool = False     # start emitting step events immediately
```

### 4.5 New Modal entrypoints in `app.py`

Mirror `pretrain_extend()` but (a) use the v6 tokenizer volume
(`osrt-v6-tokenizer`, not `osrt-v4-tokenizer`), (b) build the model from
the `OSRT_605M_A288M` preset (NOT a bare `OSRTConfig()`, which would fall
back to the v5 363M shape), and (c) pass `fused_cross_entropy_chunks=8`
like `pretrain()` does.

```python
@app.function(
    gpu="H100", image=image,
    volumes={"/vol/checkpoints": vol,
             "/vol/tokenizer": v6_tokenizer_vol,
             "/vol/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("wandb-secret"),
             modal.Secret.from_name("hf-secret")],
    timeout=86400,
)
def midtrain():
    """v6 mid-training: continued pretraining from the foundation base,
    seq 4096, math-heavy mix, ~9k steps. See MidtrainConfig."""
    import os
    from transformers import AutoTokenizer
    from osrt.presets import build_config
    from osrt.train import run_pretrain_extend
    from osrt.train_config import MidtrainConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")
    model_config = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
    )
    cfg = MidtrainConfig()
    print(f"midtrain: {cfg.total_steps} steps @ seq 4096, peak {cfg.peak_lr}, "
          f"HRA native+trainable, resume {cfg.pretrained_checkpoint}")
    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(... same decorators ...)
def midtrain_sanity():
    """30-step VRAM/throughput probe at real seq 4096 / batch 6."""
    # identical body, MidtrainSanityConfig instead of MidtrainConfig

@app.local_entrypoint()
def run_midtrain():
    call = midtrain.spawn()
    print(f"Spawned v6 midtrain — call_id={call.object_id}")

@app.local_entrypoint()
def run_midtrain_sanity():
    call = midtrain_sanity.spawn()
    print(f"Spawned v6 midtrain sanity — call_id={call.object_id}")
```

> **Note on `run_pretrain_extend`'s `model_config` reference.** The ported
> eval block (4.3) references `model_config.real_vocab_size`. The function
> signature already receives `model_config` as its first arg, so this is in
> scope — confirm during implementation that no shadowing occurs.

## 5. Data mix — verified

Reuses `PretrainConfig.phases["knowledge"]` verbatim. **No data-code
change required** — all fields resolve through the generic
`_extract_text` path (`data.py:560`); none needs a `format` key.

| dataset | hf_config | weight | category | field (extractor) |
|---|---|---|---|---|
| Nemotron-CC-Math-v1 | `4plus` | 0.25 | math | `text` (data.py:597) |
| Nemotron-Specialized | `STEM-SFT` | 0.15 | STEM | `text` |
| Nemotron-Specialized | `Math-Textbooks` | 0.15 | math | `text` |
| Nemotron-Specialized | `InfiniByte-Reasoning` | 0.10 | reasoning | `text` |
| FineWeb-Edu | — | 0.15 | general anchor | `text` |
| Nemotron-Code-v2 | `Synthetic-Question-Answering` | 0.10 | code | `content` (data.py:586) |
| Cosmopedia | `openstax` | 0.10 | textbook | `text` |

**Mix: ~65% math/STEM/reasoning · 10% code · 25% general/textbook anchor.**

`make_loader` honors `hf_config` (data.py:384), `split` (381), `skip`
(390). All 9 candidates (this set + the foundation web mix) already
passed the `smoke_new_datasets` Modal stream-test with the **HallD
`hf-secret` token** (which carries the gated Nemotron grants). The
general FineWeb-Edu anchor is retained to prevent catastrophic
forgetting of broad web text during the math-heavy pass.

## 6. Pre-launch gate (non-negotiable)

1. **Foundation must have finished** and written `osrt_v5_final.pt` (or
   confirm `osrt_v5_step_3500.pt` exists and repoint `pretrained_checkpoint`).
2. **`run_midtrain_sanity` must pass** — 30 steps at the real seq 4096 /
   batch 6, confirming: (a) clean state-dict load (HRA native, all keys
   matched), (b) VRAM fits 80GB with checkpointing on, (c) throughput is
   sane. If it OOMs, drop to `batch_size=4, grad_accum_steps=16` in
   `MidtrainConfig.phases["extend"]` and re-run the sanity.
3. Only then `run_midtrain`.

## 7. Lineage & resume

- **Input:** `/vol/checkpoints/v5/osrt_v5_final.pt` (v6 foundation, step 3500).
- **Output:** `osrt_v5_midtrain_step_*.pt` + `osrt_v5_midtrain_final.pt`.
- **Chunked resume:** the existing `stage_prefix` glob in
  `run_pretrain_extend` (train.py:1603) finds the latest
  `osrt_v5_midtrain_step_*.pt` and resumes optimizer + step; launch via
  `.spawn()` per the project rule. Same chunk-across-sessions pattern as
  the foundation run.
- The `v5` directory label is legacy; this is the v6 model. The dir is
  **not** renamed (renaming breaks the foundation run's own resume scan).

## 8. Code-change summary (blast radius)

All loop changes are in `run_pretrain_extend`, which the **legacy v5
stages still call** — but every change is gated behind a new flag that
defaults to the v5 behaviour:

| Change | File | Gated by | v5 stages affected? |
|---|---|---|---|
| `hra_native` skips `inject_hra` | train.py:~1448 | `hra_native` (default False) | No |
| checkpointing trigger ≥4096 | train.py:~1682 | `gradient_checkpointing` attr, else seq≥4096 | Only if a v5 stage ran seq 4096-8191 uncheckpointed — none do at >8GB headroom |
| periodic eval | train.py:~1906 | `eval_interval` (v5 default 9_999_999) | No |
| `MidtrainConfig`/`MidtrainSanityConfig` | train_config.py | new classes | No |
| `midtrain*` entrypoints | app.py | new functions | No |

**One review note:** the seq≥4096 trigger change is the only one that
alters a code path a v5 stage *could* hit. v5 extend stages run seq 2048
(extend2/3) or 4096 (extend1). extend1 at seq 4096 currently runs
**uncheckpointed**; after this change it would checkpoint. Since extend1
is a completed, non-rerun stage, this is acceptable — but call it out in
the plan so it's a conscious choice, not a silent regression.

## 9. Out of scope

- SFT / MOPD / GRPO (later pipeline stages, separate specs).
- Seq 8192 long-context (a later phase; this stage stops at 4096).
- Trust-remote-code self-contained modeling files (separate follow-up).
- Renaming the `v5` checkpoint dir to `v6` (deferred; breaks live resume).
