# Deep Dive Code Review - 2026-06-08

Repository: `/Users/danielhalwell/nano-osrt-100m`

Scope: current snapshot of the OSRT project, with emphasis on active runtime
code under `src/osrt/`, Modal orchestration in `app.py`, training data artifacts,
and local verification gates. This is not a security-scan artifact and does not
claim exhaustive adversarial coverage.

## Current Workspace State

The worktree had pre-existing changes during this review:

```text
 M .gitignore
 D review/SYNTHESIS.md
 D review/agy-plan-reviewed.md
 D review/codex-plan-review.md
 M uv.lock
?? archive/SYNTHESIS.md
?? archive/agy-plan-reviewed.md
?? archive/codex-plan-review.md
?? review/code-review.md
```

I did not modify those files. This report is saved as a separate dated artifact
so it does not overwrite the existing `review/code-review.md`.

## Verification Summary

Commands run:

```text
uv run pytest -q
python3 -m compileall -q src scripts app.py tests
uv run ruff check
uv run ruff check app.py src scripts tests --statistics
uvx ty check
```

Results:

```text
pytest: 135 passed, 14 warnings in 27.74s
compileall: passed after allowing Python bytecode-cache writes
full ruff: failed, 211 errors across active code plus archive
active ruff: failed, 50 errors in app.py/src/scripts/tests
ty: failed, 1003 diagnostics
```

The compile check initially failed because macOS Python tried to write bytecode
under `/Users/danielhalwell/Library/Caches/...`, outside the sandbox. Rerunning
with permission succeeded. One `uv run python` rollout scan similarly needed
permission to access the local `uv` cache.

## Findings

### P1 - Loop dropout under-weights router regularizers on shortened forwards

Files:
- `src/osrt/model.py:1404`
- `src/osrt/model.py:1640`
- `src/osrt/train_config.py:936`
- `src/osrt/train_config.py:989`

`OSRTModel.forward()` can shorten the loop chain during training when
`loop_dropout_prob > 0`:

```python
if self.training and loop_dropout_prob > 0 and random.random() < loop_dropout_prob:
    ...
    n_loops_to_run = random.randint(min_loops, max_loops)
```

But `OSRTForCausalLM.forward()` normalizes `balance_loss`, `z_loss`, and
`seq_balance_loss` by the configured full depth:

```python
n_moe_layers = self.config.num_blocks * self.config.recursive_loops
```

That denominator is wrong when loop dropout actually runs fewer loops. The model
sums regularizer losses for only the layers that executed, then divides by the
full configured layer count. This weakens router balance and z-loss exactly on
the stochastic-depth batches that are supposed to preserve healthy recursive
depth.

I reproduced it with a tiny model:

```text
{'loops_run': 2,
 'configured_layers': 8,
 'actual_layers': 4,
 'raw_balance': 4.0319414138793945,
 'observed_norm': 0.5039926767349243,
 'expected_norm_for_actual_layers': 1.0079853534698486,
 'configured_norm': 0.5039926767349243}
```

Impact:

This affects the stages that explicitly rely on loop dropout, including MOPD
and System SFT (`loop_dropout_prob = 0.10`). It can make balance telemetry look
smaller than it is and reduce regularization pressure on shortened-loop
training samples.

Recommendation:

Normalize by actual executed MoE applications. The wrapper already receives
`loop_rms`, whose length equals the number of loops run, so one direct fix is:

```python
n_moe_layers = self.config.num_blocks * len(loop_rms)
```

Add a regression test with `loop_dropout_prob=1.0` and a seeded tiny config that
asserts `last_balance_loss_normalised == last_balance_loss / actual_layers`.

### P2 - MBPP code reward advertises partial scoring but collapses to all-or-nothing

Files:
- `src/osrt/rewards.py:935`
- `src/osrt/rewards.py:1048`
- `src/osrt/rewards.py:1059`
- `app.py:2682`

`mbpp_test_reward()` accepts `reward_partial` and its docstring says it returns
reward based on test pass rate. In practice it concatenates all assertions into
one script and counts:

```python
passed = len(test_list) if rc == 0 else 0
```

If any assertion fails, Python exits non-zero and the function reports
`all_fail`, even if earlier assertions passed. The caller in `grpo_multi`
uses this directly for the MBPP environment.

Observed local reproduction:

```text
completion implements add(a, b) correctly
tests:
  assert add(1,2)==3
  assert add(2,2)==5

result:
(-1.5, {'verdict': 'all_fail', 'passed': 0, 'total': 2})
```

Impact:

The code environment loses useful partial reward signal. In GRPO, that increases
the chance of uniform reward groups or harsh negative groups, which this repo's
own training notes repeatedly identify as a failure mode.

Recommendation:

Run assertions one at a time inside the same hardened subprocess model, or wrap
each assertion so failures are counted instead of aborting the whole script.
Return a `partial` verdict and use `reward_partial * passed / total` or a
similar shaped reward. Update `app.py` counters to track partial MBPP successes.

### P3 - Lint is not a usable project gate yet

Files:
- `pyproject.toml:42`
- `app.py`
- `src/osrt/monitoring.py`
- `src/osrt/model.py`
- `scripts/collect_rollouts.py`

`pytest` passes, but `ruff` is red:

```text
uv run ruff check: 211 errors
uv run ruff check app.py src scripts tests --statistics:
18 E501 line-too-long
11 I001 unsorted-imports
 8 F401 unused-import
 6 E741 ambiguous-variable-name
 5 E702 semicolon multiple-statements
 2 F841 unused-variable
```

The full run includes archived code, while the active-code-only run still has
50 failures. Several are simple hygiene issues, but the net effect is that
`ruff` cannot currently protect this repo in CI without a waiver strategy.

Recommendation:

Decide whether `archive/` is linted historical source or excluded history. Then
make the active surface pass `ruff`, at least for `F` and `I` rules. This repo
has already had bugs surface through lint that tests did not catch, so keeping a
working lint gate is worth it.

### P3 - Rollout artifacts are tracked and larger than the loader comments assume

Files:
- `rollouts/mopd_v1.jsonl`
- `rollouts/system_prompt_sft.jsonl`
- `src/osrt/data.py:759`

Current tracked artifact sizes:

```text
86M  rollouts/mopd_v1.jsonl
18M  rollouts/system_prompt_sft.jsonl
```

Current MOPD scan:

```text
rows=13374
bad_json=0
empty_response=6
empty_thinking=241
duplicates=0
sources={'math': 4000, 'reasoning': 3000, 'chat': 3000, 'code': 374, 'science': 3000}
teachers={None: 5440, 'nemotron-3-ultra-free': 10, 'deepseek-v4-flash': 7924}
```

The loader correctly skips empty responses, so the six empty-response rows are
not a functional training break. The issue is repo and memory hygiene:
`RolloutDataset` loads the whole JSONL into memory per worker, while the comment
says "4K-50K rollouts, ~1MB-15MB". The current MOPD file alone is 86 MB before
Python object overhead, and each worker keeps its own decoded list.

Recommendation:

Choose an explicit policy:

- If these JSONL files are source-of-truth training artifacts, keep them tracked
  and update loader comments/docs with current size expectations.
- If Modal volumes are the source of truth, move the files out of git and add a
  `rollouts/*.jsonl` ignore rule, plus a small schema/sample fixture for tests.

Either way, add a lightweight JSONL schema/count test so future rollout files do
not silently drift.

### P3 - Type checking is too noisy to catch real regressions

`uvx ty check` reported 1003 diagnostics. The first major cluster is around
`AutoTokenizer.from_pretrained()` returning a broad/nullable tokenizer type in
`app.py`, followed by dynamic dict construction for `OSRTConfig` in tests.

This is not evidence that runtime is broken; `pytest` and compile checks pass.
But it means the type checker is currently not actionable as a quality gate.

Recommendation:

Scope `ty` initially to `src/osrt` or add targeted ignores/stubs for the
dynamic Hugging Face tokenizer surfaces. Once the active diagnostics are below a
small number, turn it into a real gate.

## Positive Signals

- The active test suite is strong for a research codebase: attention sink,
  KV-cache consistency, `num_loops`, speculative decoding, MTP, mHC,
  quantization, router features, monitoring, and checkpoint-drift checks all
  have coverage.
- Checkpoint loading uses `strict=False` only to produce a custom mismatch
  error, then raises if keys drift. That avoids the common silent partial-load
  failure.
- The GRPO MBPP execution path has a reasonable defense-in-depth baseline:
  explicit opt-in, secret-stripped environment, temp CWD, process-group timeout
  kill, and output caps. It is correctly documented as not a true sandbox.
- The rollout dataset integration is currently compatible with the mixed
  `teacher` schema because `RolloutDataset` only consumes `prompt`, `thinking`,
  `response`, and optional `system`.

## Review Boundary

I did not run Modal jobs, GPU training, external dataset streaming, or lm-eval
benchmarks. The review is grounded in local source inspection and local CPU
verification. The biggest remaining runtime risk is still distributed/Modal
behavior: dataloader workers, HF streaming reconnects, torch.compile startup,
and long-running checkpoint/resume behavior under real H100 jobs.
