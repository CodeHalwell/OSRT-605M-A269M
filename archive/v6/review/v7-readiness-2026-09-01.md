# v7 code-readiness review — 2026-09-01

## Verdict

**Preflight-ready; paid trunk NO-GO.** The committed v7 architecture now has a
tested preset and a deliberately capped GPU launch gate. The repository must
not expose a paid `v7` stage until roadmap gates G2, G3a, and G7 close and the
30-step GPU/resume rehearsal passes.

## Launch-critical findings addressed

### P0 — the committed v7 model had no executable preset

The roadmap committed E=28/h2,112/top-4, mHC off, SiTU-GLU, QB, per-head V4
Muon, and ~993M physical parameters, but `presets.py` still exposed only v6.

Resolution: added `OSRT_V7`, `build_v7_config`, exact meta-device budget
coverage, a capped `V7SanityConfig`, local `--preset v7-sanity`, and Modal
`--stage v7_sanity`. There is intentionally no paid trunk stage.

### P0 — local training silently accepted the stale 32K tokenizer

`train_main.py` defaulted to `./tokenizer` (32,768 entries), printed a warning,
then resized the model to that incompatible vocabulary. This could launch an
expensive run whose embedding/checkpoints violated the 65K contract.

Resolution: default to `./v6_tokenizer_export`; validate all 32 structural and
reserved token IDs, role IDs, and the 65,536 vocabulary size; fail closed on
any mismatch. The Modal v6 full-pretrain and v7 sanity paths use the same
validator.

### P0 — Quantile Balancing was required but absent

The existing fixed-step controller is not the QB algorithm committed by
roadmap §14. QB also requires the persistent bias to affect selection without
contaminating mixture weights.

Resolution: added per-loop, grad-accumulation-wide margin histograms; k/E
quantile extraction; next-step mean-centred bias commits; unbiased selected
mixture weights; and clean/raw routing tests. Scratch histograms are
non-persistent; the routing bias remains checkpointed. A fullgraph compile
test guards against reintroducing a controller graph break.

### P1 — v7 optimizer and schedule recipe was incomplete

Muon had only the historical five fast Newton-Schulz iterations and the v6
shape heuristic. The pretrain scheduler supported cosine only.

Resolution: opt-in 8-fast + 2-stabilising NS, update RMS 0.18, per-head Muon,
configurable AdamW epsilon/betas, and WSD. Defaults remain the v6 recipe;
`V7SanityConfig` selects the v7 recipe.

### P1 — checkpoint identity and resume behavior were unsafe for v7

The base loop hard-coded `osrt_v5_*` names and `pretrain_v5` metadata. An
optimizer mismatch was caught and silently reset, invalidating a v7 resumed
experiment while allowing it to continue.

Resolution: configurable, filename-safe prefixes; v7-isolated checkpoint
directory; atomic saves with semantic model-config metadata; and strict v7
resume that rejects missing metadata, model semantic drift, optimizer recipe
drift, group mismatch, or optimizer-state mismatch. Legacy v5/v6 behavior
remains permissive for historical recipe migrations.

### P1 — router health threshold was hard-coded to eight experts

The entropy gate used `2.079` (`ln(8)`) regardless of model shape and treated
fixed-controller bias saturation as meaningful for QB.

Resolution: derive the initialization entropy from
`ln(num_routed_experts)` and apply the saturation check only to the fixed-step
controller. The 30-step v7 smoke does not pretend that the long-run gate is
calibrated; production v7 thresholds remain a post-ladder decision.

### P2 — static checks and live-state documentation were stale

The advertised Ruff command failed across live code, while historical archive
and notebook-export artifacts were mixed into its target set. The README,
roadmap header, and handoff still described pre-GPU or “nothing implemented”
states.

Resolution: format and lint the live Python surface, exclude explicitly
historical/non-module artifacts, correct active static defects, and link all
entry docs to `docs/V7_RUNBOOK.md`.

## Verification performed

- `ruff check .`: pass.
- `ruff format --check .`: pass.
- Full `pytest`: 254 passed, 1 skipped (15 upstream/runtime warnings).
- v7 focused suite: 34 passed.
- v7 parameter budget: 993,437,571 physical / 288,004,995 active.
- v6 budget unchanged: 601,444,393 physical / 278,217,769 active.
- `sanity_overfit.py`: pass; loss 5.602 → best 0.113 (98% reduction).
- `dummy_train.py`: pass; fresh-batch task CE 4.915 → 0.001, routing balanced.
- QB `MoELayer` training forward: CPU `torch.compile(..., fullgraph=True)`
  pass with histogram accumulation.
- `app.py` import and `train_main --help`: pass.

## Remaining blockers (not papered over with guessed config)

1. G2 tokenizer bake-off and KD dry run.
2. G3a active-vs-physical token-yardstick experiment.
3. G7 E=28 grouped-GEMM plus BF16/FP8/NVFP4 benchmark on target hardware.
4. Full v7 30-step H100 run, then a second invocation proving strict resume
   and cross-session first-batch fingerprint behavior.
5. Production token budget, WSD stable/decay lengths, checkpoint cadence, and
   v7-specific health thresholds after the above evidence exists.
6. MTP head count remains open under roadmap §15; two heads are preserved.

Until all blocking items have recorded results, the absence of a paid `v7`
registry entry is a safety property, not missing work.
