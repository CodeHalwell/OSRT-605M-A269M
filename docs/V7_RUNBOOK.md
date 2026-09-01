# OSRT v7 runbook

**Status (2026-09-01): preflight-ready; paid trunk is NO-GO.**

The committed v7 model and its 30-step launch gate are implemented. A full
`v7` stage is intentionally absent: the roadmap explicitly makes G3a block the
trunk, while G2 and G7 still determine the final tokenizer and expert-kernel
economics.

## What is frozen in code

- 1,536 hidden width; 3 physical blocks × 6 recursive loops.
- 28 routed experts × h2,112, top-4; one h2,816 shared expert.
- 993,437,571 physical parameters / 288,004,995 active per token with the
  current 65,536-token vocabulary and two MTP heads.
- mHC off; SiTU-GLU on.
- sqrt(softplus) routing, Quantile Balancing, z-loss, sequence balance at
  `1e-4`, and the existing `0.10` learned-router auxiliary loss. The auxiliary
  loss is retained because removing it collapsed the small-model router in the
  repo's prior controlled ablation; remove it only after a matched v7 ladder
  arm demonstrates that QB makes it unnecessary.
- Per-head Muon with the V4 8-fast + 2-stabilising Newton-Schulz recipe,
  update RMS 0.18, Nesterov momentum 0.95, AdamW `(0.9, 0.95)`, epsilon
  `1e-20`, and weight decay 0.1.
- WSD scheduling for the v7 sanity recipe.
- Strict checkpoint resume: semantic model-config drift, optimizer-group
  drift, missing v7 metadata, and incompatible optimizer state all fail
  closed. v7 checkpoint names cannot collide with the v5/v6 lineage.

## Local preflight

Run from the repository root:

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
PYTHONPATH=src python scripts/compute_budget.py --preset v7
PYTHONPATH=src python scripts/sanity_overfit.py
PYTHONPATH=src python scripts/dummy_train.py
```

The local non-Modal entrypoint exposes v7 only as a capped sanity run:

```bash
python -m osrt.train_main \
  --preset v7-sanity \
  --tokenizer-path ./v6_tokenizer_export
```

It requires CUDA. The stale local `./tokenizer` directory is 32K and is now
rejected instead of silently changing the embedding shape.

## GPU launch gate

```bash
modal run app.py --stage v7_sanity
```

The stage uses the full v7 model, batch 8 × sequence 2,048, gradient
checkpointing, fused CE, grouped-GEMM MoE, QB, WSD, and the production
optimizer. It writes only under `/vol/checkpoints/v7-sanity` with the
`osrt_v7_sanity` prefix.

Pass only if all of the following hold:

1. All 30 steps are finite and task loss has a credible downward trend.
2. Peak allocated memory leaves at least 10% device headroom.
3. All 28 experts receive assignments in every effective layer after QB has
   had time to react; no persistent router or loop collapse is visible.
4. `torch.compile` does not repeatedly recompile or regress below an agreed
   BF16 baseline; record tokens/s and peak memory.
5. Checkpoint `step_15` and the final checkpoint save atomically.
6. Invoke the stage a second time. It must strictly resume from step 15, and
   the logged `data/first_batch_sha` must differ between resume steps. This is
   the cross-session stream-position observability check; a repeated hash is a
   hard failure.

## Gates that still block the paid trunk

- **G2 — tokenizer:** compare the current custom 65K tokenizer with SmolLM3
  49,152 and LFM2 65,536 on the real math/code mix, verify structural-token
  mappings, and perform a teacher-KD dry run. `OSRT_V7` currently preserves
  the verified v6 contract; that is an explicit provisional choice.
- **G3a — token yardstick:** at fixed ~150M active compute, sweep total
  parameters and determine whether loss-per-token tracks active or physical
  parameters. The committed ~993M shape assumes the active-parameter answer.
- **G7 — kernels and shape economics:** benchmark E=28/h2,112 grouped GEMM in
  BF16/FP8/NVFP4 on GB202 and compare the expert path with v6's E=8 shape.
- **MTP head count:** two heads are retained. Section 15 reopens the count;
  do not slim to one, and do not expand it without the separate G8 evidence.
- **Production schedule:** choose the trunk token budget, WSD stable length,
  decay branch length, checkpoint cadence, and final early-stop thresholds
  only after G2/G3a/G7. These values are deliberately not guessed in code.

When those decisions are recorded, add a distinct paid `v7` config and Modal
registry entry, rerun this full preflight, then launch the trunk.
