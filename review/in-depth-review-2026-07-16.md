# OSRT In-Depth Project Review — 2026-07-16

**Scope:** full read of architecture (`ARCHITECTURE.md`, `docs/`), implementation
(`src/osrt/`), tests, training pipeline (`train.py`, `train_config.py`),
infrastructure (`app.py`, `scripts/`), prior reviews (`review/`), git history,
and live verification: `uv run pytest` (194 passed, 1 skipped),
`scripts/compute_budget.py` (601,444,393 physical / 278,217,769 active),
`uv run ruff check` (96 active errors), no `.github/` CI.

This is a project/​architecture/​process review, not a security audit. It does
not run Modal/GPU jobs.

---

## Overall take

This is a **seriously engineered, intellectually ambitious, but as-yet-unvalidated**
research codebase. It combines three ideas no released model puts together —
sparse MoE + depth recurrence + Muon — and wraps them in a large stack of
further techniques (KDV latent cache, mHC Birkhoff residual, HRA, MTP,
speculative decode, aux-loss-free balancing). The documentation discipline and
measurement-first philosophy (drawn from an honest v5 post-mortem) are genuinely
above par for a solo/AI-assisted research repo. The central problem: **there is
no completed training run**, so the architecture's headline claims are all
unproven, and the project is ~2 orders of magnitude short of the token budget
needed to test them (`docs/AGENT_HANDOFF.md`: ~2.2B tokens trained ≈ 0.4×
Chinchilla, GSM8K ~0.05).

---

## Strengths

1. **Documentation rigor.** README → ARCHITECTURE → LEARNINGS → RESEARCH →
   `docs/` chapters form a coherent reading path. The "code is ground truth,
   docs cite file:line" rule and `compute_budget.py` as the single source for
   param counts are good discipline. The honesty in `ARCHITECTURE.md` §6.3
   ("it is NOT accurate to say KDV loses no expressivity") and README §12
   (flagging its own 10× cost error) is commendable.
2. **Measurement-first, wired from step 1.** Per-loop CE, dead-expert count,
   loop-update norms, prebias vs clean router telemetry, OOD probe — all the
   things v5 discovered late are now permanent instrumentation
   (`monitoring.py`, `_collect_moe_metrics`). This is the single biggest
   improvement over v5.
3. **Real stability engineering, with reasons given.** QK-norm, sandwich
   RMSNorm, SwiGLU clamp, log-domain Sinkhorn (the code notes the exp form
   NaN'd), orthogonal expert init with a documented std-correction bug fix
   (`model.py:148`), aux-loop losses + loop dropout. Each is tied to a specific
   failure mode in `LEARNINGS.md`.
4. **Hard-won implementation fixes that show depth:** grouped-GEMM dispatch
   removing the only `torch.compile` graph break (`_dispatch_grouped`);
   persistent RoPE buffers to survive HF `from_pretrained` meta-init; the
   private `_osrt_grad_ckpt` gate to avoid HF's gradient-checkpoint mechanism
   colliding with the recursive forward; dropless dispatch; the
   `sigmoid(lse − sink)` log-sum-exp rescale. These are non-obvious and correct.
5. **Test suite is strong for research code:** 194 passing CPU tests covering
   KV-cache consistency across chunks, `num_loops` validation, speculative
   decode, MTP, mHC, quantization, router features, checkpoint drift.
   Checkpoint loading uses `strict=False` only to produce a *custom* mismatch
   error then raises — avoids the silent partial-load trap.
6. **Honest lineage accounting.** `LEARNINGS.md` is unusually frank about v5's
   failures (loop collapse at month 4, reward hacking after $50, system
   prompts never trained for 12 months). That honesty is the project's best
   asset.

---

## Weaknesses (prioritized)

### Critical: no validation of the core thesis

- **Zero completed training runs.** Everything rests on CPU pre-flight. The
  canonical preset itself flags the headline differentiator as broken:
  `presets.py:38-42` says `use_mhc=True` but "CPU pre-flight showed gradient
  amplification + NaN under sustained training, needs profiling on real
  hardware to see if it's a CPU-precision artifact or a real bug." The most
  novel component may not work, and it's on by default.
- **Architecture complexity vs. zero evidence of composition.** ~8 risky
  techniques are stacked simultaneously. Each is a failure surface; with no
  end-to-end run, there's no evidence they compose at scale. `LEARNINGS.md`
  §12 itself concludes "training-pipeline discipline matters more than
  architectural novelty" — yet v6 expanded the novelty surface considerably.
  That's the project's central tension.
- **Results so far are weak.** `docs/AGENT_HANDOFF.md`: SFT v2 gives
  ~0.04–0.06 GSM8K, flat with SFT v1, "fluent-but-wrong math." The diagnosis
  ("undertrained, not capacity-capped") is plausible but unproven — it could
  also be that the architecture's restrictions (KDV K-side, mHC, 3 reused
  blocks) bite at this scale.

### High: documentation drift and internal contradictions

The repo's greatest strength (docs) is also its biggest maintenance risk.
Concrete contradictions found:

- **`ARCHITECTURE.md` §5.4 vs §2.4:** §5.4 says "87 HRA injection points
  across Q/K/V/O + every expert + router" with `x + adapter_b(adapter_a(x))`;
  §2.4 and `model.py:1127` say 18 points on the attention sub-block only,
  applied as `x_in @ adapter_a @ adapter_b` (parallel, additive inside
  attention). §5.4 is stale and wrong.
- **`ARCHITECTURE.md` §10** keeps pseudocode the doc itself calls buggy ("the
  pseudocode below reads the OLD buggy form in places; treat it as a
  conceptual sketch"). Known-wrong pseudocode in the spec is worse than no
  pseudocode.
- **`ARCHITECTURE.md` §7.6** is a duplicated heading the doc keeps "to
  preserve numbering." Technical debt as policy.
- **Gated short convolutions:** README §10a.1 lists them as an LFM2 lesson;
  ARCHITECTURE.md §1 says "NOT in the architecture: gated short convolutions
  … never specified or implemented." Direct contradiction.
- **Four different test counts:** README "144 unit tests", CLAUDE.md "193
  unit tests", AGENT_HANDOFF "196 pass", actual `194 passed, 1 skipped`.
- **`presets.py` docstring** says "~607M physical / ~288M active (47.5%)" —
  wrong on all three counts (601M / 278M / 46.3%).
- **`ARCHITECTURE.md` §2.1** table is "hand-adjusted by −72 pending a clean
  regen" — manual edits to a code-generated table.
- **`hra.py` docstring** references `RecursiveOSRT(cfg)` and `load_pretrained`
  — a class that doesn't exist (only `OSRTForCausalLM`).

### High: tooling gates are absent or red

- **No CI.** No `.github/`, no pre-commit. Nothing prevents regressions on a
  codebase with 272 commits in 4 months and many AI-tool branches
  (`bolt-*` remotes).
- **Lint is red, not a gate.** `ruff check` on active code: **96 errors** (39
  auto-fixable); `archive/` adds 162. CLAUDE.md advertises ruff as a command
  but it's failing. Unused imports, E501, ambiguous names. The prior review
  (2026-06-08) found 50 active errors — it's gotten worse.
- **Type checking non-functional.** Prior review: 1003 `ty` diagnostics. No
  type gate. For code this intricate (mHC einsums, KDV latent reshape,
  18-layer recursion) types would catch real bugs.

### Medium: two parallel HRA systems

`hra.py` (`HRALinear`/`inject_hra`/`get_param_groups`) is the v5-era path
still wired into `app.py`, `train.py`, `sft_train.py`, `lm_eval_wrapper.py`,
`train_config.py` for the v5/extend resume lineage. v6 uses *native* HRA built
inline in `OSRTModel` (`adapters_a`/`adapters_b`). The handoff explicitly
warns "never `inject_hra` on v6." So two different HRA mechanisms coexist,
`hra.py`'s `get_param_groups` overlaps-but-differs from
`muon.py::build_param_groups`, and the `hra.py` docstring references a
nonexistent class. This is a real confusion surface for any resuming
contributor.

### Medium: premature research output

`paper.pdf`, `paper.tex`, `compile_expanded_paper.py`,
`review/cross-loop-results-draft.tex` exist, but there are no v6 cross-loop
results (no surviving midtrain3 checkpoint). The legitimate finding (loop
collapse discovered and fixed in v5 at 363M) is being presented under a
v6/601M architecture that hasn't demonstrated the fix works at the new scale.
Writing the paper before a validated run inverts the measurement-first
principle the project otherwise champions.

### Medium: overclaimed deployment story

- **Speculative decode is greedy-only / not distribution-preserving** (the
  code says so in a warning box, `model.py:2369`). But README §1.4 and §12.3
  advertise "~2× faster inference + rollout, ~$50-100 saved per GRPO run."
  GRPO rollouts need *sampling*, where this path is not valid. The headline
  RL benefit is overstated.
- **mHC numbers wrong and feature unproven.** `ARCHITECTURE.md` §8.6 claims
  "~720K params, ~6.7% overhead"; `compute_budget.py` reports 921,766.
  README §10b.8 rates mHC "medium" priority, yet it's on in the canonical
  preset with an acknowledged NaN risk. There's a strong case to default
  `use_mhc=False` until a GPU run proves it helps.
- **`torch._grouped_mm`** is a private internal API (`model.py:577`). The
  code notes "CPU backward broken in torch 2.10." Binding to a private torch
  symbol is a portability time bomb.

### Low: naming tax

Repo `OSRT-605M-A269M`, preset `OSRT_605M_A288M`, alias `OSRT_605M_A279M`,
model "OSRT-600M", actual 601M/278M. CLAUDE.md says "naming is stale on
purpose." Carrying three wrong numbers indefinitely is a readability cost on
every new contributor; a one-time rename + back-compat alias is cheaper than
the perpetual confusion. Also: `config.py` docstring still describes the
*363M* v5 model ("~363M physical params, ~192M active"); `model.py:22`
docstring says "362,720,259 (~363M)". The module headers describe the wrong
model.

### Low: test coverage gaps for the riskiest paths

No test that the canonical preset forwards at real scale (only tiny configs).
No integration test asserting cached+speculative greedy decode equals
non-cached greedy (the spec path's correctness rests on an untested invariant
per its own docstring). No test that grouped-GEMM and loop dispatch agree *at
the preset config* (CPU always uses the reference loop). No test that
`inject_hra` and native HRA produce equivalent behaviour where they overlap.

---

## Concrete improvements (in priority order)

1. **Get one full training run to completion before anything else.** All
   other improvements are academic until the architecture is shown to
   actually learn at scale. Per `AGENT_HANDOFF.md`, midtrain3 is the path —
   chain it to 1× Chinchilla and re-run SFT v2. Declare success = GSM8K lifts
   off the 0.05 floor. Until then, treat every architecture claim as a
   hypothesis.
2. **Default `use_mhc=False` in the canonical preset** until a GPU run proves
   it helps (the preset already admits NaN risk). Keep it as an A/B knob.
   Re-enable only with a measured win.
3. **Reconcile the docs in one sweep:** delete the buggy §10 pseudocode, fix
   §5.4 to match §2.4/code, remove the §7.6 duplicate, resolve the
   gated-convolutions contradiction, regenerate the §2.1 table without
   hand-edits, fix the `hra.py` and `config.py`/`model.py` module docstrings.
   Make `compute_budget.py` the *only* place numbers live.
4. **Add a minimal CI:** GitHub Actions running `uv run pytest` + `uv run
   ruff check` (after fixing the 96). No GPU, no Modal — just the CPU gate.
   This alone prevents most regressions for ~zero cost.
5. **Fix ruff (39 are auto-fixable) and make it a gate.** Decide `archive/`
   is excluded history (add to ruff `exclude`). Then get active code to zero.
6. **Collapse the two HRA systems.** Either delete `hra.py` (if v6 native
   fully supersedes it and the v5 resume path is dead) or clearly fence it as
   "v5-resume-only, do not use for v6" with a deprecation header. Update its
   docstring to not reference `RecursiveOSRT`.
7. **Add the two missing integration tests:** (a) canonical preset forwards a
   tiny batch on CPU without shape errors; (b) `generate(speculative=True)`
   greedy output bit-identical to `generate(speculative=False)` greedy on a
   tiny config. These pin the two riskiest untested invariants.
8. **Hold the paper until there's a result.** The cross-loop finding is real
   but it's a v5 result; presenting it under v6 branding before v6 validates
   it invites a credibility hit.
9. **One-time rename pass:** pick a single name (`OSRT-601M-A278M`), alias
   the old names, update the repo description. Stop carrying three wrong
   numbers.
10. **Reduce the config knob surface.** ~40 knobs, many off in the canonical
    preset (hash_routing, attention_sink, router_affinity alt, gumbel,
    capacity_factor, per_loop_aux_weights, fused_ce_chunks). Each is a
    maintained, untested code path. Delete or gate the ones you've decided
    against (e.g. `attention_sink` is dropped per §6.6 — consider removing
    the path entirely rather than keeping it as an A/B knob that can OOM at
    seq 8192).

---

## Bottom line

The architecture is interesting and the engineering is more careful than most
research code at this stage. But the project is in a dangerous phase: an
enormous amount of sophisticated, well-documented machinery has been built to
validate a thesis that has not yet been tested, and some of it (mHC) is
flagged broken by the author. The highest-leverage move isn't another feature
or another doc chapter — it's **one completed training run that shows the
recursive-MoE+Muon combination actually learns better than a dense baseline at
this scale**, and a doc/lint/CI cleanup to make the codebase safe for whoever
resumes it. Until the run lands, treat the architecture as a well-reasoned
bet, not a proven result.

---

## Verification commands run

```text
uv run pytest -q                    → 194 passed, 1 skipped, 15 warnings (54.73s)
uv run python scripts/compute_budget.py
                                    → 601,444,393 physical / 278,217,769 active (46.3%)
uv run ruff check app.py src scripts tests
                                    → 96 errors (39 auto-fixable)
uv run ruff check archive           → 162 errors
ls .github/                         → NO .github/ CI present
git log --oneline --all | wc -l     → 456 commits
git log --oneline --since="3 months ago" | wc -l
                                    → 272 commits in last 3 months
```

## Files inspected (non-exhaustive)

- `CLAUDE.md`, `ARCHITECTURE.md`, `README.md`, `LEARNINGS.md`
- `src/osrt/`: `config.py`, `presets.py`, `model.py`, `mhc.py`, `muon.py`,
  `hra.py`, `train.py`, `train_config.py`
- `tests/`: all 16 test files (full `pytest` run)
- `docs/`: `AGENT_HANDOFF.md`, `00-overview.md` (dir listing of all chapters)
- `review/`: `deep-dive-code-review-2026-06-08.md`, `learnings-scratchpad.md`
- `pyproject.toml`, `app.py` (entrypoints), `scripts/compute_budget.py`
- git history, branch list, checkpoint/data directories
