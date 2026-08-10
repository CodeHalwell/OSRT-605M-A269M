# Preregistration — verification reward for GRPO v6b

**Status:** declared BEFORE implementation. Written 2026-08-10, after the wave-2
post-mortem and before any verification-reward code exists.

**Why this document exists.** Three times during the wave-2 investigation a
measurement that looked decisive was overturned by the next one: the scalar
loss-term ratio, then the raw gradient norms, then global clipping. Each error
had the same shape — reading one quantity as if it settled a question that
needed a different quantity. The defence is to name the falsifying measurement
in advance. Every threshold below is fixed now; results are scored against
these numbers, not the reverse.

---

## 0. Boundaries on the evidence this rests on

Two limits keep the existing record honest and must not be quietly widened.

**The +6.5pp wave-2 loss** (soup − step 400 acc_on, paired bootstrap
p=0.012, CI [+1.50, +11.50], discordant 19/6) is established **on the
200-question development panel**. Repeated checkpoint inspection, post-hoc soup
construction and a single training seed all prevent treating p=0.012 as a
confirmatory population claim.

**The 77–83° cross-configuration angle** describes *raw gradients on the cached
step-390 batch*:

```
cos(policy_hist,   policy_corrected)    +0.1263
cos(combined_hist, combined_corrected)  +0.2371
cos(combined_hist, policy_corrected)    +0.2346
```

Clipping is scalar and preserves these directions, but AdamW's per-parameter
preconditioning can transform them, and the angle may differ at earlier
checkpoints. So it establishes that the score-function bug materially changed
the **local** gradient. It does **not** establish that every historical
optimizer update had that angle, and it is not required to carry the stronger
claim that it fully explains the 6.5pp loss. It is sufficient on its own to
reject continuation from wave 2 and restart from the soup.

**Not established, and not to be cited as measured trends:** the OFF-up slope
(p=0.052, halves when step 400 is dropped), the delta collapse (p=0.186), and
`acc_off` overtaking the soup (p=0.602, 25 discordant items).

**The reward-misalignment hypothesis** rests on code inspection (90.9% of
maximum positive reward is the final number; `reasoning_bonus` is gated on
`correct and n_steps >= 2` but counts steps without validating them) and direct
rollout observation (whole groups at steps 250 and 350 scored the full +5.50 on
traces that computed 40 and answered 10). That is a well-motivated hypothesis,
not a measured trend.

---

## 1. Deterministic verifier correctness

Fixture-based, no model involved. **All three must hold exactly; any failure
blocks implementation.**

| # | criterion | threshold |
|---|---|---|
| 1.1 | Accuracy on constructed `valid` / `poisoned` / `corrected` / `malformed` fixtures | **100%** |
| 1.2 | Copying a poisoned step earns positive verification reward | **never** |
| 1.3 | Missing or ambiguous verification output silently passes | **never** — must route to an explicit `unparsed` label |

Fixtures are hand-written and committed alongside the verifier, covering at
minimum: correct verdict on a valid hint; correct rejection of a poisoned hint;
a corrected equation that fixes the poison; a verdict with no supporting step;
two contradictory verdicts; a verdict naming the poison value while rejecting
it (which the earlier `str(wrong_val) in final_answer` substring test would have
penalised at −7.5 — the exact behaviour the reward is meant to teach).

## 2. Shortcut resistance

| # | criterion | threshold |
|---|---|---|
| 2.1 | Valid/poison pairs drawn from the **same** underlying problems | balanced, 50/50 |
| 2.2 | `always-valid` baseline balanced verification accuracy | **≤ 50%** |
| 2.3 | `always-invalid` baseline | **≤ 50%** |
| 2.4 | `ignore-hint` / `copy-hint` baseline | **≤ 50%** |
| 2.5 | Hint validity predictable from style, length, position or formatting metadata | **AUC ≤ 0.55** from a logistic probe on those features alone |

2.5 is the one most easily missed: if poisoned hints are systematically longer,
rounder, or differently punctuated, the model learns a surface cue and 2.2–2.4
still pass. Poison must be localised and style-matched.

## 3. Frozen-policy signal viability

Measured on a cached batch via `grpo_diag_batch` + `grpo_grad_diag`, no training.

| # | criterion | threshold | wave-2 reference |
|---|---|---|---|
| 3.1 | Verification-output parser coverage | **≥ 95%** | n/a |
| 3.2 | Positive advantage on incorrect final answers, after clamping | **exactly 0** | 74/232 wrong rollouts (31.9%) had positive advantage |
| 3.3 | Live rollouts | **≥ 50%** | 87.5% unclamped, 58.6% clamped |
| 3.4 | Zero-variance prompt groups | **≤ 15%** | 7.7% all-wrong |
| 3.5 | Fraction of **final-correct** rollouts whose advantage changes under the verification term | **≥ 20%** | 0% — the current +0.3 bonus cannot discriminate among correct answers |

3.5 is the decisive one. If the verification term does not materially reorder
*correct* rollouts, it is decorative in exactly the way the existing
`reasoning_bonus` is, and the design fails regardless of how well it scores on
groups 1 and 2.

## 4. Gates — sanity wave and confirmation run are DIFFERENT

The power calculation forces this split. Paired-binary SE ≈ `sqrt(d/n)` with
`d` the discordant rate; observed `d = 25/200 = 0.125` for soup vs step 400.

```
non-inferiority power at TRUE gap = 0, one-sided alpha = 0.05, d = 0.125
  n=200 :  M=2pp 0.20   M=3pp 0.33   M=4pp 0.48   M=5pp 0.64
  n=1119:  M=2pp 0.60   M=3pp 0.88   M=4pp 0.98   M=5pp 1.00
```

**The 200-item development panel cannot support a non-inferiority claim at any
sensible margin.** So:

### 4a. Sanity wave (25–50 steps, n=200 development panel) — SCREENING ONLY

| metric | gate | rationale |
|---|---|---|
| `acc_on` | **no catastrophic drop**: upper 95% paired bound of (soup − wave) < **10pp** | detectable at n=200; screens for a wave-1-style collapse, nothing finer |
| Verification accuracy | balanced accuracy > **50%** on ≥100 poison prompts | shortcut floor from group 2 |
| `acc_off` | reported as a **control**, never optimised | |
| Format | **≥ 99%** | wave-2 achieved 100% |
| Truncation | **< 3%** | wave-2 ran 0–6/256 |
| `delta` | **diagnostic only** — must not collapse, must never be the target | it improves when `acc_off` falls, which is exactly what the wave-2 soup did |

Passing 4a licenses a full run. It does **not** license any capability claim.

### 4b. Confirmation run (n=1119, `problem_offset=200`) — CLAIMS

Declared in advance, scored once, on problems never used for selection.

| metric | gate | power |
|---|---|---|
| `acc_on` non-inferiority vs the SFT soup's 20.0% | margin **M = 3pp**, one-sided 95%: upper bound of (soup − v6b) < 3pp | **0.88** at true gap 0 |
| Verification accuracy | ≥ **30%** poison-rejection with paired 95% CI excluding zero | CI lower ≈ 21pp at n=100, so power is not binding — 30% is a **usefulness** threshold, not a detectability one |
| `acc_off` | control | |
| `delta` | diagnostic | |

**Honest limitation of the M=3pp gate:** power is 0.88 only if v6b genuinely
*equals* the soup. If the true gap is 1pp power falls to 0.60; at 2pp it is
0.24. So failing this gate will not distinguish "slightly worse" from "much
worse", and M=2pp is not achievable (0.60 power) on the available problems.
A tighter margin needs a larger evaluation set than GSM8K test provides.

---

## 5. Declared changes that are not hypotheses

Landing regardless of the above, recorded so they are not later mistaken for
findings:

- **Strict extraction is lossless on correctness** at step 390 — 248/256
  agreement, 8 disagreements all `no_answer_block`, **zero correctness flips**
  (loose 24/256, strict 24/256). It is **not** reward-neutral: those 8 (3.1%)
  move from `wrong_far_off` (−2.0) to `no_extraction` (−2.5). Declared as a
  deliberate change, not discovered afterwards.
- **Empty gold is a hard loader error**, not a filtered row.
- **`delta` is demoted** from primary to diagnostic.
- Metric hierarchy: primary `acc_on`; co-primary verification accuracy; control
  `acc_off`; diagnostic `delta`.
