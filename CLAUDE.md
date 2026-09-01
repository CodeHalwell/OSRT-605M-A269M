# CLAUDE.md

Guidance for AI assistants (Claude Code et al.) working in this repository.

## What this project is

**OSRT** = **Optimized Sparse Recursive Transformer** — a small language model
that combines three ideas no released frontier model puts together at once:
**sparse MoE** + **depth recurrence** + **Muon optimization**.

- **3 physical decoder blocks**, applied **6 times via a loop** (weights reused
  each pass) → **18 effective layers** from one-third the parameters.
- Per block: **1 shared expert + 28 routed experts (top-4)**, GQA attention with
  an MLA-style compressed-latent KV cache, rank-256 HRA adapters, and a 4-channel
  a plain residual stream (mHC removed in v7 — roadmap §12.3).
- **968,468,355 physical / 263,035,779 active per token** (the "605M/A269M/A288M"
  numbers in the repo name and presets are *stale* — see "Naming" below).
- Trained with **Muon (hidden 2D matrices) + AdamW (embeddings/norms/biases)**.

**Status:** the architecture is fully implemented in `src/osrt/`; the unit-test
suite plus CPU smoke tests pass. **GPU training is well underway:** the pipeline
has run pretrain → midtrain → midtrain2 → SFT v1/v2 (GSM8K measured, near-floor:
diagnosed as an undertrained base, not an architecture ceiling), and is currently
extending the base via **midtrain3** — a long continued-pretraining push toward
Chinchilla-optimal, driven from a Colab GPU notebook with cross-session
checkpointing to a private HF repo (`notebooks/midtrain3_colab.ipynb`,
`scripts/lightning_midtrain3.py`, `scripts/hf_ckpt_sync.py`). See
`docs/AGENT_HANDOFF.md` for the live state and
`docs/specs/2026-07-26-*.md` for the latest investigation notes.

## Repository map

```
src/osrt/             # THE implementation — ground truth for exact behaviour
  config.py           # OSRTConfig dataclass (all model hyperparameters)
  presets.py          # OSRT_605M_A288M canonical preset + build_config()
  model.py            # OSRTForCausalLM/OSRTModel, RecursiveBlock, MoELayer, ExpertFFN
  hra.py              # HRA rank-256 attention adapters
  mhc.py              # manifold-constrained hyper-connection mixers (Sinkhorn/Birkhoff)
  muon.py             # Muon optimizer + HybridMuonAdamW + build_param_groups
  fused_ce.py         # chunked/fused cross-entropy (mandatory for the 80GB fit)
  quant.py            # int4 KV-cache quantization (deployment)
  monitoring.py       # loop/MoE collapse telemetry
  train.py            # run_training(): the pure-PyTorch pretraining loop
  train_config.py     # ALL training-phase configs (Pretrain/Midtrain/SFT/GRPO...)
  train_main.py       # non-Modal CLI entry: python -m osrt.train_main
  data.py / sft_data.py   # pretraining + SFT data pipelines
  sft_train.py / sft_eval.py
  rewards.py          # GRPO verifiable-reward functions
  system_prompts.py   # system-prompt contract
  lm_eval_wrapper.py  # lm-eval-harness integration

app.py                # Modal deployment entry point — all training stages live here
scripts/              # CPU smoke tests, probes, data builders, tokenizer training
tests/                # pytest suite (run on CPU)
configs/              # exported HF config.json
tokenizer/            # OSRT-Ostinato tokenizer (SmolLM2 base + 32 specials) — USE THIS
v6_tokenizer_export/  # 65K v6 BPE — superseded at gate G2, do not use for v7
tokenizer/            # STALE 32K artefact (pre-v6) — do not load for v6 ckpts
docs/                 # numbered architecture chapters (00-overview → 10-...)
docs/ARCHITECTURE.md       # terse technical spec (config-value source of truth)
README.md             # design philosophy + the "why" / integrated training plan
docs/LEARNINGS.md          # v5 (363M) failure modes this design avoids
docs/RESEARCH.md           # external paper bibliography behind each technique
archive/              # v3/v4/v5 historical code + docs — NOT importable, reference only
```

## Environment & common commands

Python **3.11**, dependency management via **uv** (`uv.lock` is committed).

```bash
uv sync                         # install deps (incl. dev group: pytest, ruff)

# Tests (CPU, fast)
uv run pytest                   # full suite (testpaths=tests)
uv run pytest tests/test_model.py -q

# Lint / format (ruff, line-length 88, rules E/F/I)
uv run ruff check .
uv run ruff format .

# Parameter budget — the canonical source for param counts
PYTHONPATH=src python scripts/compute_budget.py

# CPU smoke tests that exercise the FULL stack before any GPU spend
PYTHONPATH=src python scripts/sanity_overfit.py   # overfit one batch → loss ~0
PYTHONPATH=src python scripts/dummy_train.py       # synthetic copy task, fresh batches
```

Scripts use `PYTHONPATH=src` (the package lives under `src/osrt`). If you add a
runnable script, follow that convention or run via `uv run`.

## Training workflow

Training has two front-ends over the same `osrt.train.run_training` loop:

1. **Modal** (`app.py`) — the production path. One `@app.local_entrypoint` named
   `main(stage=...)` dispatches a central `registry` of stages. Long runs use
   `SPAWN` (fire-and-forget), short runs (eval/sanity) use `REMOTE`.
   ```bash
   modal run app.py --stage sanity        # ~200-step smoke test
   modal run app.py --stage pretrain      # full pretrain
   modal run app.py --stage midtrain      # v6 mid-training
   modal run app.py --stage sft_v2
   modal run app.py --stage grpo          # GRPO RL with verifiable rewards
   modal run app.py --stage evaluate      # lm-eval-harness suite
   ```
   The pipeline lineage is roughly: **pretrain → midtrain → SFT (v1/v2/long/…) →
   GRPO**. Each stage has a matching `*_sanity` smoke variant — run it first.
   The `registry` dict in `main()` is the single source of truth for stages.

2. **Plain PyTorch** (`osrt.train_main`) — for Lightning Studio / EC2 / on-prem
   GPUs without Modal:
   ```bash
   # Required env: WANDB_API_KEY, HF_TOKEN
   python -m osrt.train_main --tokenizer-path ./tokenizer --ckpt-dir ./checkpoints/v6
   ```
   Resumes automatically from the highest-step checkpoint in `--ckpt-dir`.
   **Pretraining requires a CUDA GPU** (bf16 autocast + torch.compile throughout);
   it exits on CPU by design.

**Training-phase configs** all live in `src/osrt/train_config.py` as dataclasses
that subclass each other (e.g. `MidtrainConfig(PretrainConfig)`, `SFTv2Config`,
`GRPOConfig`). Change hyperparameters there, not inline in `app.py`.

## Conventions & gotchas

- **`compute_budget.py` is the only trusted source for param counts.** Do not
  hand-edit param tables in README/ARCHITECTURE; regenerate them. Numbers come
  from instantiating the real model on a `meta` device.
- **Naming is stale on purpose.** The repo `OSRT-605M-A269M`, the preset
  `OSRT_605M_A288M`, and `OSRT_605M_A279M` (a back-compat alias → same preset)
  all predate v7 entirely. `build_v7_config()` instantiates **968,468,355
  physical / 263,035,779 active**. Trust `compute_budget.py` and
  `presets.py`, never a name.
- **Doc precedence.** When `docs/` chapters and `docs/ARCHITECTURE.md` disagree, the
  chapters cite the code (`file:line`) and **the code wins**. `src/osrt/` is
  ground truth. When you change behaviour, update the relevant `docs/0X-*.md`
  chapter and the `docs/ARCHITECTURE.md` section it maps to.
- **`archive/` (v3/v4/v5) is reference only** — not importable, do not edit as
  if it were live code. Only `src/osrt/` is the current architecture.
- **Checkpoints, data, tokenizer binaries, wandb runs are git-ignored**
  (`*.pt`, `*.safetensors`, `data/*.bin`, `checkpoints/`, `wandb/`). Don't commit
  weights; commit code and configs.
- **Run a `*_sanity` / smoke variant before any real GPU spend.** The repo's
  whole philosophy (see README §0, docs/LEARNINGS.md) is "measure before you build" —
  loop collapse, router collapse, and reward hacking all bit the v5 lineage late.
- **Stability features are load-bearing**, not optional: QK-norm, sandwich
  RMSNorm, per-loop routing accounting, aux-loss-free balancing, SwiGLU clamp.
  Don't remove them to "simplify" without understanding the failure they prevent.

## Git workflow

- Develop on the designated feature branch; create it locally if missing.
- Commit with clear messages; **push with `git push -u origin <branch>`**, retry
  with exponential backoff (2s/4s/8s/16s) on network errors only.
- **Do not open a pull request unless explicitly asked.**
- Never push to a different branch without explicit permission.

## Where to read more

- Architecture deep-dive: start at `docs/00-overview.md`, then chapters 01–10.
- The "why" and training plan: `README.md`.
- Config-value spec: `docs/ARCHITECTURE.md`.
- What went wrong before (and why the design is shaped this way): `docs/LEARNINGS.md`.
- Papers behind each technique: `docs/RESEARCH.md`.
