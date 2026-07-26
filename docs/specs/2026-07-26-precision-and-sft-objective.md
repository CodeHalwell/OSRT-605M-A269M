# Precision, Memory, and the Group-Relative SFT Objective — Investigation Notes

**Date:** 2026-07-26
**Status:** Investigation — **nothing implemented, no config changed.** All
findings below are reads of the current tree at `claude/model-precision-sarrdk`.
**Companions:** `ARCHITECTURE.md` §14–15, `docs/08-optimizer.md`,
`docs/AGENT_HANDOFF.md` §1–2.

---

## 0. TL;DR

| question | answer |
|---|---|
| What precision runs today? | fp32 master weights + bf16 autocast, with deliberate fp32 islands (CE, router losses, softmax). §2 |
| Should we move to fp8 for speed? | **No, not now.** Blocked by `_grouped_mm`, and `dim=1536` is below the shape where fp8 pays. §3 |
| Does fp8 let us use a smaller GPU? | **Wrong tool.** fp8 mixed precision is a speed technique; gradient checkpointing already claimed the memory it would save. §4 |
| Is "group-relative SFT" worth building? | The weighting instinct is sound and **already has an exact closed form (focal loss)**. The sample-and-count step adds variance for nothing. §5 |
| Anything genuinely new in the idea? | Yes — a **consistency loss over stochastic routing passes**. Architecture-specific, real experiment. §5.5 |
| Are model-written hints for GRPO worth building? | **Yes — best of the ideas.** Targets the documented zero-gradient trap; ~66% of rollout budget currently produces no gradient. But hints must target *variance*, not success, and the simplest correct form is rejection-sampling SFT, not GRPO. §6 |
| An output-head correction layer? | **Yes — cheapest experiment here, lowest blast radius.** Fills a gap HRA structurally cannot reach (the tied LM head). ~0.8–2.4M params, zero-init so a failed run costs nothing. Fixes tying bias/calibration, not knowledge. §7 |
| Adaptive expert count (variable top-k)? | **Defer, don't shrink.** Real technique, but its payoff scales with E and this model is E=8 by deliberate choice. Breaks the fixed shapes that buy fullgraph compile, and introduces an always-max-k degenerate solution. Revisit at E≥32. §8 |
| RAG / web search into the training pipeline? | **Keep the mechanism, drop the retrieval.** The with/without-context objective is *context distillation* — real, and it supersedes the §6.6 plan. But build no retriever: apply it to the §6 hints instead. Live web search is strictly worse than the frozen dump already streaming. §9 |
| Mixture-of-LoRA with a routing classifier? | **Viable as a serving pattern at low rank** — storage is ~12% overhead, not 3.8×. Two things to design in from the start: **soft-blend with a null adapter** (not hard switching), and note §14.1 says adapters are the one component to keep at bf16. §10 |

---

## 1. Why this note exists

A session on 2026-07-26 worked through three linked questions (current
precision → fp8 → a proposed SFT objective). The conclusions took real
derivation and code-reading to reach; this note records them so they don't have
to be re-derived, and so the *rejected* options stay rejected for stated reasons
rather than getting re-litigated later.

**Stale doc found along the way:** `CLAUDE.md` claims "no completed GPU training
run yet… The model is in CPU pre-flight." `docs/AGENT_HANDOFF.md:47-63`
documents pretrain → midtrain → midtrain2 → SFT v1/v2 as *done*, with a GSM8K
result. `CLAUDE.md` should be corrected — a reader trusting it will prioritise
completely wrong. **(Open item, §11.)**

---

## 2. What precision actually runs today

### 2.1 Training — mixed precision

- **Parameters and gradients are fp32.** `OSRTForCausalLM(model_config).to(device=device)`
  (`src/osrt/train.py:698`, `:1448`) — no `.to(bfloat16)`.
- **Every forward runs under bf16 autocast:** `torch.amp.autocast("cuda", dtype=torch.bfloat16)`
  — pretrain `train.py:1125`, eval `train.py:293`, SFT `sft_train.py:298`,
  GRPO `app.py:3180` / `:3968`, lm-eval `lm_eval_wrapper.py:440`.
- **No `GradScaler` anywhere** — correct; bf16's exponent range makes loss
  scaling unnecessary (that's an fp16 concern).
- **TF32 on** for residual fp32 matmuls: `train.py:694-695`, `sft_train.py:45-46`.

### 2.2 Deliberate fp32 islands

These are load-bearing, not accidents. Do not "simplify" them away:

| site | what | why |
|---|---|---|
| `fused_ce.py:68`, `:74` | CE accumulator + logits | stable log-sum over 65k vocab |
| `model.py:802-820` | router balance loss + z-loss | under bf16 the `f·p` product loses the gradient signal |
| `model.py:1267-1277` | softmax + attention-sink rescale | sink rescale is sensitive to exp/log precision |
| `model.py:82-87` | RoPE tables (cast down at use) | stable precompute; cast so q/k aren't promoted |
| `model.py:248-283` | router state buffers, telemetry | accumulate over millions of steps |

### 2.3 Optimizer

Muon's Newton–Schulz runs in **bf16** (`muon.py:70`, 5 iterations — ~2× fp32
throughput on H100 tensor cores), but the **momentum buffer is forced fp32**
(`muon.py:161-174`) so it doesn't accumulate bf16 roundoff over millions of
steps. The orthogonalized update is cast back to param dtype before the
in-place apply.

### 2.4 Inference and deployment

- Eval casts the whole model to bf16 (`app.py:886`, `:932`); lm-eval log-probs
  upcast to fp32 for the sum (`lm_eval_wrapper.py:443`).
- Deployment plan (`ARCHITECTURE.md` §14.1) is int8 embeddings/attention/shared
  experts + **MXFP4** routed experts + bf16 for the small sensitive parts +
  **int4 KV latent** via TurboQuant (`quant.py`). Note `quant.py:11-15`: int4 KV
  is a *standalone deployment utility*, **not wired into training** and off by
  default in `generate()`.
- The `fp8` mentions at `model.py:815`, `:1053`, `config.py:114` are only
  rationale for logit-clamping bounds. **Nothing runs in fp8 today.**

---

## 3. fp8 for training speed — not now

### 3.1 The hard blocker

Routed experts are **71% of physical params** and the bulk of the FLOPs. They go
through `torch._grouped_mm`, and `model.py:570` states it plainly:

> `torch._grouped_mm` (compiled) supports only bf16/fp16.

An fp8 path would need torchao's experimental scaled-grouped-MM for MoE, or a
fallback to `_dispatch_loop` — which forfeits the **9–12% already measured** from
grouped GEMM (`presets.py:57-62`). You'd spend the fp8 win buying back a loss.

Secondary: torchao is not a dependency, and `pyproject.toml:11` pins
`torch>=2.2.0`. fp8 training tooling wants 2.5+.

### 3.2 The shape argument

`dim=1536`, `expert_hidden=3840`. fp8's 2× tensor-core peak only materialises
when the GEMM is large enough to hide the scaling overhead (dynamic per-tensor
amax = an extra full read plus a scaled cast per operand).

Rough arithmetic on the expert GEMM (M≈4096, N=3840, K=1536):

```
2 · 4096 · 3840 · 1536          = 48.3 GFLOP
bf16 @ ~600 TFLOP/s effective   ≈ 80 µs
fp8  @ ~900 TFLOP/s effective   ≈ 54 µs
cast + amax overhead            ≈ 10 µs
                                → ~20% on GEMMs that could use it
```

End-to-end, after gradient-checkpoint recompute, routing sort/scatter, 20
Sinkhorn iterations × 18 effective layers, and Muon's NS: realistically **≤10%**.

### 3.3 Architecture-specific risk

- **Depth recurrence compounds quantization error.** The same weights applied 6×
  makes the error *systematic*, not noise that averages out. No published work
  on fp8 depth-recurrent training — this would be the experiment.
- **An unresolved numerical issue already exists.** `presets.py:38-42` flags mHC
  as showing "gradient amplification + NaN under sustained training… needs
  profiling on real hardware." Stack fp8 on top and the next NaN is
  unattributable.
- **Router collapse is the documented failure mode** (`LEARNINGS.md`). The fp32
  router losses at `model.py:802-820` exist precisely to prevent it.

### 3.4 Hardware fragmentation

Per `docs/AGENT_HANDOFF.md:153`, the fleet is RTX PRO 6000 (Blackwell — fp8 ✅),
H100 (✅), and Colab **A100-40GB (Ampere — no fp8 silicon at all)**. T4 likewise
has none. A chunk of the cheapest compute gets zero benefit.

**Verdict: revisit only if a profile shows GEMMs dominating *after* the §4
levers, and only once a clean bf16 baseline exists to attribute regressions
against.**

---

## 4. fp8 for a smaller GPU — wrong tool

### 4.1 fp8 mixed precision is a speed technique

The standard recipe (torchao float8, TransformerEngine) keeps master weights,
gradients, and optimizer states in fp32/bf16 and casts to fp8 **only at the GEMM
boundary**. The only memory it saves is *saved activations for backward*.

Full gradient checkpointing is already on (`app.py:427`; required to fit per
`ARCHITECTURE.md` §15.1). Checkpointing works by **not storing** those
activations. **The two levers compete for the same bytes** — checkpointing has
already claimed them.

### 4.2 The floor fp8 cannot touch

Arithmetic on the documented param counts in `ARCHITECTURE.md` §14.2 — *not* a
fresh measurement; `compute_budget.py` is the trusted source and should be run
to confirm:

| item | size |
|---|---|
| params, fp32 (601M × 4B) | 2.40 GB |
| gradients, fp32 | 2.40 GB |
| Muon momentum, 1 buffer fp32 (~495M 2D params) | 1.98 GB |
| AdamW states, 2 buffers fp32 (~106M embed/norm) | 0.85 GB |
| **fixed floor, unchanged by fp8** | **~7.6 GB** |

Measured totals for reference (`ARCHITECTURE.md` §15.1): seq-8192/batch-2 =
**35.9 GB**; seq-4096/batch-6 = **~59 GB**. Both already post-checkpointing and
post-fused-CE.

### 4.3 What actually shrinks the footprint, ranked

1. **mHC — the big one.** `use_mhc=True, n_hc=4` makes the residual stream
   `(B, S, 4, dim)`. `app.py:418` names it as an OOM cause; `presets.py:38-42`
   says it may be buggy and has never been validated on GPU. **4× residual
   memory for an unvalidated feature.** `n_hc=2` halves it; `use_mhc=False`
   removes it.
2. **Sequence length.** With `attention_sink=False` routing through flash SDPA
   there's no S² term, so activation memory is linear in S. 2048 vs 8192 = 4×.
3. **micro-batch 1 + more grad accum.** Currently 2–3.
4. **bf16 gradients** (~1.2 GB) and **8-bit AdamW for the embedding/norm group
   only** (~0.6 GB). Keep Muon's momentum fp32 — `muon.py:164-167`.

Those four should fit 601M on **24 GB** (4090 / L4) without touching precision.

> ⚠️ **Correction recorded:** an earlier suggestion in-session to "turn on fused
> CE" was wrong. It is already on everywhere in practice — `app.py:422, 507, 676,
> 772, 879, 2381`, `train_main.py:128`, `lightning_midtrain3.py:115` all set
> `fused_cross_entropy_chunks=8`. Only the *dataclass default* is `0`
> (`config.py:180`).

---

## 5. The group-relative SFT objective

### 5.1 What was proposed

> Generate 4 predictions of the next word. Since it's non-deterministic they
> should give different distributions. If 3 of 4 match the target, weight those
> higher; if only 1 of 4 matches, weight the update much more strongly.

### 5.2 The blocking premise: the forward pass is deterministic

Same weights + same input tokens → **bit-identical logits**. The
non-determinism in LLM generation lives entirely in the *sampling* step that
draws a token **from** the distribution. In this model specifically:

| source | default | status during SFT |
|---|---|---|
| activation/attention dropout | — | **does not exist in the model** |
| loop dropout (stochastic depth), `model.py:1643` | `0.0` (`config.py:200`) | off |
| Gumbel router noise, `model.py:749-753` | `0.0` (`config.py:240`) | **off** — `PretrainExtendConfig` sets 0.0 (`train_config.py:414`) |

Gumbel is live *only* during pretrain, annealed 0.5 → 0 over 4k steps
(`train_config.py:187-189`). Loop dropout is set in `LoopFixV2Config` (0.2,
`train_config.py:858`) and `PretrainExtend3Config` (0.10, `:924`).

**So during SFT there is no source of variation. Four passes → one
distribution.**

### 5.3 What the scheme reduces to

One forward → one `p` → draw 4 tokens → count matches `k`, where
`k ~ Binomial(4, p_gold)`. But `p_gold` is already available exactly from the
softmax. This generalizes to **any** weight function `f(k)`:

```
E[f(k)] = Σ_k  C(4,k) · p^k · (1-p)^(4-k) · f(k)
```

— a degree-4 polynomial in `p_gold`. **Whatever weighting scheme you build from
4 draws, its expectation has an exact closed form computable directly from the
softmax**, with zero variance and zero extra compute.

### 5.4 The instinct is right — it's focal loss

A weight that decreases in `p_gold` is exactly focal loss. Working the
expectation:

| sampling rule | exact equivalent |
|---|---|
| `f(k) = (4-k)/4` — fraction that missed | `1 - p` → **focal loss, γ=1** |
| `f(k) = 1 if k=0 else 0` — fire only when all miss | `(1-p)⁴` → **focal loss, γ=4** |

So the number of samples and the shape of `f` just parameterise γ:

```python
# exact, no sampling — γ ≈ 1–2
logp = F.log_softmax(shift_logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
loss = -((1 - logp.exp()).pow(gamma) * logp)[mask].mean()
```

Cost of the sampling version instead: a **5-level quantized, noisy** estimate of
the same weight. At `p_gold = 0.9` it heavily upweights an already-learned token
~0.4% of the time on pure sampling noise; at `p_gold = 0.5` the weight swings
wildly between identical inputs.

### 5.5 The version that *isn't* redundant

Turn the stochastic switches on and the passes genuinely differ:

- `router_gumbel_tau_init > 0` → different experts fire → different distributions
- `loop_dropout_prob > 0` → different recursion depth → different distributions

Then "3 of 4 passes rank the gold token first" carries information CE cannot
express: the prediction is **robust across routing paths**, not merely correct on
the one path taken. This is architecture-specific — a dense model has no such
knob.

The natural use is a **consistency loss**, not a reweighting: run 2 stochastic
passes and penalize disagreement (symmetric KL, R-Drop style) alongside CE. That
trains "the answer shouldn't depend on which experts fired" — well-aimed given
this repo's router-collapse history. **2 passes, so 2× forward, not 4×.**

### 5.6 Related: the diversity/repetition angle

Separately established in-session — the repetition problem is real and measured
(`lm_eval_wrapper.py:136-149`, from `probe3.py`):

- temp 0.2 → degenerate loops (`"17*23 = 17*(2*23) = 17*(2*23)…"`, never closes `<|/think|>`)
- temp 0.7 + top_p 0.9 + top_k 50 + rep_penalty 1.2 → coherent, closes all tags

Variety is currently bought at *inference*. Moving it into the loss is possible —
**no label smoothing or entropy term exists anywhere**; `model.py:1894` and
`fused_ce.py:74` are both plain one-hot CE. Options:

- **Label smoothing** — but at V=65,536, ε=0.1 parks ~10% of mass on 65k junk
  tokens. Use ε=0.02–0.05 if at all.
- **Confidence penalty** — `L = CE − β·H(p)`, β ≈ 0.01–0.05. Preferred: resists
  sharpening without dictating where the mass goes.
- **Unlikelihood** — penalizes tokens already in context. Most targeted at the
  actual loop symptom, most code.

⚠️ `lm_eval_wrapper.py:148` already says: *"Future better-trained checkpoints
should drop temperature and repetition_penalty back toward 0.2 / 1.0 as they stop
repeating."* These techniques **flatten the distribution rather than supply the
missing knowledge.** Good if the goal is generation variety; will not move GSM8K.

---

## 6. Hint-augmented GRPO

### 6.1 The proposal

> Use a frontier model to write hints for GRPO questions. Hints guide *how* to
> answer without answering. Apply to x% of questions so the model doesn't come
> to rely on them.

**Assessment: the best-aimed of the ideas in this note.** It attacks a failure
mode the configs already document, and it buys sample efficiency rather than
attempting to create knowledge.

### 6.2 The problem it solves — quantified

`compute_group_advantages` (`rewards.py`) returns `[0.0]*n` whenever
`std < 1e-8`. `train_config.py:36-41` records run 2 hitting exactly this:

> acc 12.5 % (0) → 18.8 % (10) → 0 % (20, 30) → 3.1 % … the rest were stuck in
> GRPO's "all-rollouts-uniform-rewards" trap where group-relative advantage
> normalisation produces zero [gradient]

At GSM8K ~0.05 per-rollout (`AGENT_HANDOFF.md:54`) with `group_size=8`
(`train_config.py:56`):

```
P(all 8 wrong) = 0.95^8 ≈ 66%     ← two thirds of the rollout budget, zero gradient
P(all 8 wrong) = 0.80^8 ≈ 17%     ← if hints lift per-rollout success to 20%
                                  → ~3× more usable samples per dollar
```

Full generation cost is paid regardless (`max_gen_len=384` × 8,
`train_config.py:61`).

**This got worse when format was solved, which is easy to miss.**
`AGENT_HANDOFF.md:53` reports `format_ok_on = 1.0`. With format saturated,
`match_format_exactly_score` and `match_format_approximately_score`
(`rewards.py:195`, `:210`) are **constant within every group** and contribute no
variance. Correctness is the only remaining varying term — so when all 8 are
wrong, `std` genuinely collapses. Solving format removed the consolation-prize
gradient that used to mask this.

### 6.3 The symmetry that breaks the naive design

`std < 1e-8` fires when **all 8 are right** exactly as it does when all 8 are
wrong. GRPO needs *variance*, not success. A hint strong enough to make every
rollout land reproduces the zero gradient it was meant to escape.

Within-group Bernoulli variance `p(1-p)` peaks at **p = 0.5**. So the design
target is not "make the hint helpful":

> **Calibrate hint strength so the group lands near a 50% pass rate.**

Which argues for graded hints rather than one flavour:

| level | content | for |
|---|---|---|
| L1 | nudge — name the relevant concept | prompts already near threshold |
| L2 | strategy — the solution shape, no numbers | mid-difficulty |
| L3 | partial work — first step done, rest open | all-8-fail prompts |

### 6.4 Random x% is the wrong selection rule

The instinct that hints must not be universal is right, but random selection
spends them on problems the model already solves — *adding* zero-gradient groups
at the top end.

**Adaptive selection is strictly better and cheap:** track per-prompt pass rate,
hint only prompts sitting at 0/8, escalate L1→L3 until the group is mixed,
withdraw the hint once a prompt reaches ~50%. Every hint is then spent where a
zero-gradient group would otherwise have been.

### 6.5 The decision that determines whether it transfers

Injection is trivial — `app.py:3095` is one line:

```python
prompt_text = f"{cfg.user_tag}{question}{cfg.assistant_tag}"
```

**Option A — hint present for both sampling and scoring.** On-policy, no loss
changes. But it trains π(y | question, **hint**), and the trap is that hinted
prompts are precisely the ones producing nonzero advantage — so a
disproportionate share of the actual gradient teaches the hint-conditioned
policy. The x% mixing does **not** save this, because the unhinted remainder is
mostly the zero-gradient set. You would be optimising the mode you don't ship.

**Option B — sample with hint, score without.** Generate rollouts with the hint
so good trajectories exist, then compute log-probs on
`(unhinted prompt + completion)`. The gradient teaches "given only the question,
produce this reasoning" — the STaR-style rationalisation mechanism: use
privileged information to *find* trajectories, train without it.

⚠️ **Two concrete breakages in the current loop if you do Option B:**

1. `app.py:3202-3203` states *"importance-sampling ratio ~= 1, so PPO clipping is
   a no-op here."* Sampling with a hint and scoring without makes sampler and
   scored policy genuinely different distributions. **That ratio is not 1**, and
   the no-op clipping becomes silently wrong. Needs real importance weighting.
2. `prompt_len` (`app.py:3098`) is computed from the hinted prompt and slices
   `shift_logits` (`app.py:3186`). It must be recomputed against the unhinted
   prefix or the labels misalign.

### 6.6 Recommendation — do this outside GRPO first

The simplest correct form of Option B is **hint-assisted rejection sampling →
SFT**: generate with hints, keep only correct completions, train plain CE on
`(unhinted question → completion)`. No advantage math, no off-policy correction,
no KL anchor, no variance calibration — and it reuses the existing SFT stack.
The hint does exactly its job (surface trajectories the model can't reach alone)
and never appears in the trained policy's input.

If it must be GRPO: Option A with adaptive selection, tracking hinted-vs-unhinted
pass rate separately, adding importance weighting only if a transfer gap shows up.

> **Superseded — see §9.** Context distillation is a strictly better objective
> than rejection-sampling SFT for the same hint pipeline: it uses the full hinted
> distribution at every position (including from rollouts that were *wrong*)
> rather than discarding failures and reducing successes to one-hot targets. It
> also dissolves the §6.5 off-policy problem, since KL distillation is not a
> policy gradient and needs no importance weighting. Build §9.4, not this.

### 6.7 Two guards worth building in

**Leakage.** `LEARNINGS.md` is largely about reward hacking biting v5. A hint
like *"the answer is a multiple of 7 near 90"* is an answer. Cheap filter reusing
existing code: run every generated hint through `extract_numeric_answer`
(`rewards.py:14`) and `extract_numeric_answer_strict` (`:63`), reject any that
yields the gold value. Manual-read a sample of ~50 on top.

**Streaming join.** `app.py:2986-2989` loads the prompt dataset with
`streaming=True`, so hints cannot be joined lazily. Pre-generate offline into a
`{question_hash: [L1, L2, L3]}` mapping loaded as a dict, or build a derived HF
dataset with hint columns.

Generation cost is a one-time offline pass over the prompt set — trivial next to
a training run, and it does not recur per step.

### 6.8 Caveats

- Hints make the GRPO budget go further; they **do not add knowledge** (§13). If
  the base genuinely cannot do multi-step arithmetic, an L3 hint yields a
  completion the model could not have reached and likely cannot generalise from
  — closer to distillation than RL. **The tell is whether hinted performance
  transfers to unhinted eval; track both from step 1.**
- Eval must always run with **no hints**, on the unmodified prompt path.
- Frontier-model outputs used as training data carry provider-specific ToS
  conditions. This repo is public and ships HF artifacts — worth checking before
  release.

---

## 7. Output-head correction layer

### 7.1 The proposal

> A post-training "correction factor" layer whose sole job is to nudge the
> output distribution to better match the target — some non-linear function
> applied on top of the existing head.

**Assessment: the cheapest experiment in this note and the lowest blast radius.**
It fills a structural gap the existing adapter path cannot reach. It does not
add knowledge.

### 7.2 The shape already exists in-tree

`MTPHead` (`model.py:1353-1376`) is exactly this construct:

```python
class MTPHead(nn.Module):
    """Small projection applied to the FINAL post-norm_out hidden state before
    the WEIGHT-TIED LM head (the embedding) turns it into vocab logits."""
    def __init__(self, dim):
        self.norm = nn.RMSNorm(dim)
        self.proj = nn.Linear(dim, dim, bias=False)
```

The +2/+3 MTP heads each get an `RMSNorm + Linear(dim, dim)` before the tied
projection. **The main +1 path does not** — `model.py:1873` applies the tied
embedding directly:

```python
logits = F.linear(hidden, self.model.embedding.weight)
```

So the primary head is *less* parameterised than the auxiliary ones. The
proposal amounts to giving the +1 path what the MTP paths already have, plus a
non-linearity.

### 7.3 Be precise about the input

| variant | can it change greedy argmax? | what it is |
|---|---|---|
| `z' = f(z)` — function of logits, monotone per-coordinate | **No.** Greedy decoding is bit-identical | temperature/calibration scaling |
| `z' = F.linear(g(h), E)` — function of the hidden state | Yes | **added output-layer capacity** |

The first version cannot affect accuracy at all — worth knowing before building,
since "better match the target" sounds like an accuracy claim.

The second is the one worth building, but name it honestly: it is capacity, not
correction. **Hard ceiling: no function of `h` recovers information `h` does not
encode.** If the final hidden state doesn't represent the answer, no non-linear
map produces it.

### 7.4 Why it pays *here* — the tying gap

The LM head is **weight-tied to the embedding** (`model.py:1783-1784`, ~100M
params saved at 65K vocab). Tying forces the output projection to *be* the input
representation matrix — it cannot specialise. A learned `g(h)` before the tied
projection is the standard escape valve, and is why `MTPHead` exists at all.

The non-obvious part — **HRA structurally cannot reach the output head:**

```python
target_modules = ("q_proj", "kv_down", "v_from_k", "out_proj",
                  "w_gate", "w_up", "w_down")     # hra.py:96-99
```

`inject_hra` wraps `nn.Linear` modules (`hra.py:121`). The head is not one — it
is a bare `F.linear` call against `embedding.weight`, so no wrapper can ever
attach. The existing parameter-efficient path (used by GRPO, `app.py:2976`)
covers attention and expert FFN across every block and the output projection
**not at all**. A correction head fills that gap rather than duplicating HRA.

### 7.5 Implementation sketch

Mirror the HRA init convention (`hra.py:59-62` — A random, B zeros) so the layer
is an **exact no-op at step 0** and can be dropped onto an existing checkpoint
without perturbing it:

```python
# residual, zero-init `down` → exact identity before any training
h = h + self.down(F.silu(self.up(self.norm(h))))   # up: dim→r, down: r→dim (zeros)
```

- `r=256` → ~790K params; full `Linear(1536,1536)` → ~2.4M. Negligible vs 601M.
- Zero-init means a failed experiment costs nothing — the checkpoint is unchanged
  until the layer learns something.
- Base frozen → last-layer fine-tuning. Unfrozen alongside HRA → composes with
  the path GRPO already uses.
- Gate behind an `OSRTConfig` field defaulting to off, as with
  `fused_cross_entropy_chunks`, so the default forward stays bit-identical.

### 7.6 Measurement protocol and limits

- **Cannot add knowledge.** "Fluent but wrong math" stays wrong — same ceiling
  as §13.
- What it can plausibly fix is the systematic output bias tying imposes
  (over-favouring tokens with large embedding norm, independent of context) — a
  known effect and a plausible contributor to the repetition behaviour at
  `lm_eval_wrapper.py:136-149`.
- **Measure unhinted greedy accuracy *and* output entropy, before/after, on a
  frozen base.** If entropy moves but accuracy doesn't, what got built is a
  calibration layer: useful for sampling, not for GSM8K. That is a legitimate
  outcome — just label it correctly rather than reporting it as a reasoning win.

---

## 8. Adaptive expert count (variable top-k)

### 8.1 The proposal

> Make MoE routing fluid in the number of experts — up to x active, so in a
> model with 50 experts a token might use anywhere from 2 to 10.

**Assessment: a real technique aimed at the wrong host.** Defer to a
larger-E model rather than shrinking it to fit E=8.

### 8.2 Scale — the payoff scales with E, and E=8 was a deliberate choice

`presets.py:24-27` records the reasoning for 12 → 8 experts:

> 8 experts (not the original 12) trades sparsity for per-token capacity: each
> token sees a larger fraction of the routed knowledge base, less risk of expert
> under-utilization at this scale.

Top-2 of 8 is already **25% routing density**. Adaptive-k is a technique for
E=64/128/256, where routing is fine-grained enough that "how many experts does
this token need" is a meaningful question. At E=8 the achievable range is roughly
1–4, against a design decision that already moved *toward* density. The 50-expert
framing in the original proposal is the right scale — that is simply a different
model.

### 8.3 The fixed-shape blocker

`_dispatch_grouped` (`model.py:594-627`) is built around static shapes — its
docstring: *"Fixed-shape ops only (argsort, bincount, cumsum, index_add) so the
path is torch.compile-clean."* `presets.py:57-62` states the stakes:

> Removes the per-expert `.nonzero()` — the only torch.compile graph break — so
> the model compiles fullgraph. Validated on H100: … ~9-12% faster steady-state.

The precise distinction, easy to get wrong: **per-expert counts already vary** —
that is exactly what `bincount` → `cumsum` → `offs` exists to handle. What must
stay fixed is the **total** `N*K`:

```python
K = self.top_k
pair_expert = top_idx.reshape(-1)                       # (N*K,)
pair_token  = torch.arange(N).repeat_interleave(K)      # (N*K,)
```

Variable k makes `sum(k_i)` data-dependent → dynamic leading dimension into
grouped-GEMM → recompilation or graph break. The 9–12% is forfeit.

### 8.4 Two escapes, each with a catch

**(a) Pad to `max_k`, zero the gates on unused slots.** Shapes fixed, fullgraph
survives. But grouped-GEMM computes every padded row regardless, so a "2–10"
model costs exactly what fixed top-10 costs. **Adaptive allocation, zero compute
saving.**

**(b) Fixed global budget, variable per-token split.** Take the global
top-`(N·K)` (token, expert) pairs across the batch — confident tokens contribute
fewer, ambiguous ones more, total stays exactly `N·K`. Shapes fixed, compute
constant, allocation adaptive. But it is **batch-coupled**: a token's k depends
on which other tokens share its batch. That breaks autoregressive decode
(batch=1, one token at a time) — the same property that makes Expert-Choice
routing awkward for generation.

> Summary: per-token adaptive works at inference but breaks shapes; batch-budget
> keeps shapes but breaks decode.

### 8.5 The new degenerate solution

Given `LEARNINGS.md` is largely about router collapse, this deserves weight:
**with variable k and no explicit penalty, more experts always lowers loss**, so
the router learns to always select `max_k`. Sparsity is not self-enforcing the
way it is at fixed top-k.

Preventing it needs a compute-budget penalty on `E[k]` — another coefficient in
an objective already carrying task + balance (0.10) + z-loss (1e-3) +
seq-balance + aux-loop (0.05) + MTP (0.3), a system the configs document as
tuning-fragile.

### 8.6 Surface area

`self.top_k` is load-bearing in ~15 sites. The balance machinery in particular
assumes every token contributes exactly K pairs — **each normalizer divides by
`N * self.top_k`**:

| site | what breaks |
|---|---|
| `model.py:807, 836, 889, 938, 967` | balance `f_i`, seq-balance, prebias and clean variants — denominator becomes `sum(k_i)` |
| `model.py:785, 788` | capacity calc `capacity_factor * top_k * N / num_routed` |
| `model.py:915, 942, 971` | `if self.top_k >= 2` co-activation guards |
| `_accumulate_balance_counts` (`model.py`) | bias controller assumes K counts per token |
| `compute_budget.py:66` | `sparse_frac = top_k / num_routed` — **the headline active-param number** |

The last is not just bookkeeping: "278M active per token" becomes a
*distribution* rather than a number (~278M mean, roughly 180–520M range). That
figure appears in the repo name, the preset names, and the `ARCHITECTURE.md`
§14.2 deployment memory math, which assumes a fixed active set.

**Recursion compounds it:** 18 routing decisions per token (6 loops × 3 blocks),
each with variable k, so per-token compute variance stacks and step time becomes
unpredictable. Per-loop accounting (`balance_count_accum[loop_idx]`) needs
reworking at every site.

### 8.7 If it is tried here anyway

Route (a), padded to a small `max_k` — adaptive 1–4 padded to 4. Fullgraph
survives, the balance denominators become `sum(k_i)` (a contained change), and it
answers whether adaptive allocation helps at all.

⚠️ **Evaluate at equal FLOPs.** If the padded path pays top-4 compute, the
baseline is **fixed top-4, not fixed top-2**. Adaptive-1-to-4 beating top-2
proves nothing — it has 2× the FLOPs. The question is only whether it beats
top-4 at equal cost.

---

## 9. Retrieval and context distillation

### 9.1 The proposal

> Add RAG to the SFT and/or RL pipeline to help the model learn complex topics.
> Maybe web search too, but the model does not call these as tools. Generate
> responses with and without, and use the difference in the output distribution
> to update the weights.

**Assessment: the mechanism is correctly identified and worth building. The
retrieval infrastructure around it is not — not at this stage.**

### 9.2 The mechanism has a name: context distillation

"Generate with and without, use the difference to update the weights" is the
context-distillation objective — the standard method for internalising a context
(system prompts, few-shot examples, scratchpads, retrieved documents) into
weights.

⚠️ **The teacher branch must be stop-grad.** This is the detail that decides
whether it works at all:

```python
with torch.no_grad():
    p_ctx = F.log_softmax(model(question + docs + y).logits, -1)   # teacher, detached
p_bare = F.log_softmax(model(question + y).logits, -1)             # student, trainable
loss = F.kl_div(p_bare, p_ctx, log_target=True, reduction="batchmean")
```

Backprop through **both** branches and the objective has a trivial degenerate
solution: make the output *invariant to the context*. KL → 0 and the model has
been trained to ignore retrieval entirely — the exact opposite of the goal, and
it fails quietly because the loss curve looks excellent.

Both branches must also be scored on the **same continuation `y`**, and their
`prompt_len` differs — the same prefix-alignment problem as §6.5.

### 9.3 What this can do that the other proposals cannot

This and the hint pipeline (§6) are the **only two ideas in these notes that
bring external information into the weights**. Focal loss (§5), the correction
head (§7), and adaptive routing (§8) all rearrange what is already there.
Retrieved documents are genuinely new information, so context distillation is
not capped by the §13 "elicit, don't create" ceiling.

That makes it the first proposal here with a real knowledge-transfer channel —
and also the one most easily mistaken for a cheaper alternative to pretraining,
which it is not (§9.4).

### 9.4 Why the retrieval half is the wrong tool *for this bottleneck*

Reaching 1× Chinchilla needs ~+3.4B tokens. Per useful token:

| approach | cost |
|---|---|
| plain next-token training on the corpus | 1 forward, no questions needed |
| context distillation | 2 forwards + question generation + the teacher's context tokens |

**If the goal is facts in the weights and a corpus exists, pretrain on it** —
roughly 3× cheaper per token, and it is what §12 has been pointing at
throughout. Context distillation earns its cost when the context is *procedural*
(a format, a reasoning pattern, a system prompt you don't want to pay for at
inference), not when it is factual.

The corpus side is also already solved: `train_config.py:205, 272, 516` stream
FineWeb-Edu, with CodeParrot and Wikipedia in the knowledge phase. **There is no
retrieval infrastructure in the repo** — no faiss, no embedding model, no index —
so this would mean building a retriever, chunker and vector store from scratch
against a custom 65K tokenizer.

**On web search specifically:** for a training pipeline, live search is strictly
worse than a frozen dump — non-reproducible runs, quality variance, licensing
exposure, no determinism across resumes. FineWeb-Edu *is* the frozen web dump,
and it is already streaming.

### 9.5 A fork to decide explicitly

Keeping retrieval out of the model's hands (no tool-calling) is the right call —
tool use is a large ask for a 601M undertrained base and it changes the
deployment story. But it creates a fork:

| where retrieval lives | what to train |
|---|---|
| inference **and** training | just train with context present — standard RAG fine-tuning, **no distillation needed** |
| training only | context distillation, to internalise it — the proposal as described |

Only the second needs the with/without machinery.

### 9.6 Recommendation — keep the mechanism, drop the retrieval

**Apply context distillation to the §6 hints.** A hint *is* a context, just a
short generated one rather than a retrieved one. This gives:

- No retriever, no index, no corpus, no new subsystem
- **No sequence-budget hit.** SFT runs at `seq_len 2048`
  (`train_config.py:1215`), where retrieved documents would consume most of the
  window; hints are a line or two
- A **strictly better objective than §6.6's rejection-sampling SFT**: full-
  distribution KL uses the whole hinted distribution at every position —
  including from rollouts that were *wrong* — instead of discarding failures and
  flattening successes to one-hot targets. Far more signal per generation
- **The §6.5 off-policy problem dissolves.** KL distillation is not a policy
  gradient, so no importance weighting is needed

Cost is still 2 forward passes per step with a longer teacher branch, which
lands on the §4 memory constraints. That is a far smaller bill than a retrieval
stack.

---

## 10. Mixture-of-LoRA with a routing classifier

### 10.1 The proposal

> Fine-tune the model 50 times on 50 different subjects, then train a small
> classifier that attaches the right LoRA at inference. **The goal is a more
> generalist model in production** — adapters should not be too specific, and
> should still answer normal non-specialised questions. Low rank and low
> precision.

**Assessment: viable, because the stated goal is the one this pattern is
actually for.** Mixture-of-LoRA is a *serving* pattern (one base, many
specialisations, no merging) rather than a capability pattern. Two design
decisions matter more than the rest, and both are cheaper to build in than to
retrofit.

### 10.2 What low rank + low precision resolves

At rank 256, HRA is **14,155,776 params** across 18 injection points
(`ARCHITECTURE.md:107`) — 50 copies would be 708M, *larger than the 601M base*.
That objection dissolves at low rank:

```
rank 16:  1536 × 16 × 2 × 18 injection points  =    884,736 / adapter
          × 50 adapters                        = 44,236,800  (~44M)
          bf16 88 MB │ int8 44 MB │ int4 22 MB
          vs §14.2 deployment target ~377 MB   → 12-23% overhead
```

Note the tradeoff being accepted: `presets.py:37` calls rank 256 *"real HRA
capacity (NOT LoRA-style 16)"*. Dropping to 16 gives each adapter 1/16th the
capacity — appropriate for the deliberately-mild adapters this design wants, but
it is a real reduction, not a free win.

### 10.3 The tension to design around: specific vs generalist

"Should still answer normal non-specialised questions" is in direct tension with
the premise. Adapter strength (rank × `scale` × training steps) is a single dial
trading specialisation against generality: mild enough to preserve general
behaviour also means mild enough to deliver a small benefit.

**The fix is to blend rather than switch.** Use the classifier's *softmax* to
weight adapters, and include a **null/identity option** in the mix:

| query | behaviour |
|---|---|
| confident domain match | mostly that adapter |
| general or ambiguous | weight shifts toward null → base behaviour preserved |
| misroute | degrades gracefully instead of catastrophically |

This solves the stated requirement directly and removes the hard-selection
failure mode. It also means the classifier no longer has to be *right*, only
roughly right — a much easier target, and one that degrades sensibly
out-of-distribution.

> Worth noticing where this lands: soft-weighting a set of adapters by a learned
> gate is precisely what the native MoE router already does (`model.py:759`), at
> per-token rather than per-request granularity. The design converges on the
> existing mechanism — which is an argument for confidence in the shape, and for
> checking §10.6 before building a second one.

### 10.4 ⚠️ Low precision on adapters contradicts the current deployment spec

`ARCHITECTURE.md` §14.1 singles adapters out as the component to keep at full
precision:

| component | format | method |
|---|---|---|
| HRA adapters | **bf16** | kept full precision (**small, sensitive**) |

Two reasons that call is not obviously wrong:

1. `HRALinear.forward` (`hra.py:69+`) *adds* the adapter output to the base
   output, so quantisation error lands directly on the residual stream rather
   than being attenuated.
2. Adapters sit on the 3 physical blocks and therefore run at **all 6 recursion
   loops** — quantisation error compounds across loops, the same systematic-error
   argument that rules out fp8 in §3.3.

50 low-rank adapters is a different regime from one rank-256 adapter, so int8 may
well hold. But it contradicts the current spec, so **measure it rather than
assuming it** — and if §14.1 turns out to be wrong here, update §14.1.

### 10.5 Recursion amplification

18 injection points = 6 per physical block, and each block runs 6×. Every
adapter perturbs the same weights at all six recursion depths, where (per the
design thesis) the block is doing different work — early loops surface, late
loops abstract. **An adapter cannot be loop-specific.**

Loop collapse is the v5 lineage's headline failure (`LEARNINGS.md`). Low rank and
low `scale` reduce the risk substantially, which is what makes the mild-adapter
choice the right one here — but each adapter still wants a loop-depth probe
(`monitoring.py:104`, `loop_depth_probe`) before it ships, not just a task score.

### 10.6 Serving mechanics

Two production details that do not show up until integration:

**Mid-conversation topic shift.** Adapters change weights, so switching adapters
mid-generation makes the existing KV cache inconsistent with the new weights.
Either pin the adapter for the whole conversation (wrong after a topic turn) or
invalidate the cache on switch (a latency spike). Soft blending (§10.3) softens
this — small weight changes rather than a discrete swap — but does not remove it.
Decide the policy before building.

**Batched inference with per-sequence adapters** is the S-LoRA problem: different
sequences in a batch need different weights, which defeats naive batching. The
solution is to sort sequences by adapter and run grouped matmuls — **which is
structurally identical to `_dispatch_grouped` (`model.py:594-627`)**, already in
the tree for MoE: sort by expert, `bincount` → `cumsum` → `offs`,
`torch._grouped_mm`. The same primitive serves batched multi-LoRA. Reuse it
rather than reaching for a new dependency.

### 10.7 Do this before the 50 training runs

`monitoring.py:55` provides `moe_health` — per-block, per-loop load entropy with
`MIN_LOAD_ENTROPY = 0.55` and dead-expert detection. Run it on **domain-segmented
batches** and look at whether expert usage shifts between math, code and prose:

- **Experts already differentiate by domain** → the specialisation mechanism
  exists natively at per-token granularity. Strengthen it before paralleling it.
- **Load entropy is collapsed** → that is a router bug. A second routing layer on
  top of a broken router does not fix the first one.

Either answer costs one evaluation pass and is more informative than 50
fine-tuning runs.

**Also: start with ~5 broad adapters, not 50.** Data fragmentation is the real
cost at 0.4× Chinchilla — 50 splits starve each adapter and destroy the
cross-domain transfer that is a small model's main advantage. Five coarse domains
capture most of the serving benefit at 1/10th the training cost, and the result
tells you whether 50 is worth it.

---

## 11. Open items

Not started. Roughly in priority order:

- [ ] **Fix `CLAUDE.md`** — it claims CPU pre-flight / no GPU run; `AGENT_HANDOFF.md`
      documents pretrain → SFT v2 complete. Misleads prioritisation. (§1)
- [ ] **mHC A/B probe** — `n_hc ∈ {4, 2, off}` at fixed seq/batch: peak VRAM,
      tok/s, and whether the `presets.py:38-42` NaN reproduces on GPU. Answers
      the smaller-GPU question *and* the oldest open architecture question. (§4.3)
- [ ] **Log output entropy + top-1 probability** per step before choosing any
      diversity term — if entropy is healthy and it still loops, the problem is
      knowledge, not sharpness. Repo philosophy is measure-first. (§5.6)
- [ ] **Focal loss behind a config flag** — `model.py:1894`, threaded through
      `fused_ce.py` so `tests/test_fused_ce.py` parity holds. Apply to the **main
      task loss only**, not the 5 aux-loop heads (`model.py:1962`) or 2 MTP heads
      (`model.py:2030`) — those are regularizers against loop/router collapse. (§5.4)
- [ ] **Optional: R-Drop-style consistency term** over 2 gumbel-on passes. (§5.5)
- [ ] Profile `mhc_sinkhorn_iters=20` — Sinkhorn typically converges in 3–5;
      that's 20 bandwidth-bound passes × 18 effective layers. (§3.2)

Hint track (§6), roughly in order:

- [ ] **Instrument first** — log the fraction of GRPO groups with `std < 1e-8`,
      split into all-wrong vs all-right. Confirms the ~66% estimate before any
      hint is generated, and is the metric every later step is judged against. (§6.2)
- [ ] **Generate graded L1/L2/L3 hints** offline over the prompt set; store as
      `{question_hash: [L1, L2, L3]}` (streaming rules out a lazy join). (§6.3, §6.7)
- [ ] **Leakage filter** — reject any hint whose text yields the gold value under
      `extract_numeric_answer` / `_strict`; manual-read ~50. (§6.7)
- [ ] **Hint context distillation** — the recommended first build (supersedes the
      rejection-sampling plan in §6.6): teacher = model with hint in context
      (**stop-grad**), student = model without, KL between them on the same
      continuation. No off-policy correction, and it learns from wrong rollouts
      too. (§9.2, §9.6)
- [ ] Only if going into GRPO proper: adaptive selection targeting ~50% group
      pass rate, hinted-vs-unhinted pass rate logged separately, and the
      `prompt_len` / ratio≈1 fixes at `app.py:3098`, `:3186`, `:3202-3203`. (§6.4, §6.5)

Correction-head track (§7) — independent of the above, can run in parallel:

- [ ] **Zero-init residual correction head** behind a config flag, applied to
      `hidden` before the tied projection at `model.py:1873`. Exact no-op at
      step 0, so it drops onto an existing checkpoint safely. (§7.5)
- [ ] **Baseline first:** unhinted greedy accuracy + output entropy on the frozen
      base, so the before/after comparison exists. Shares the entropy
      instrumentation with the §5.6 item — do that one first and this is free. (§7.6)
- [ ] Label the outcome honestly: entropy moves but accuracy flat = a calibration
      layer, which is useful for sampling and **not** a reasoning result. (§7.6)

Mixture-of-LoRA track (§10) — a **serving** play, independent of the training work above:

- [ ] **`moe_health` on domain-segmented batches first** (`monitoring.py:55`). One
      eval pass; tells you whether the native per-token router already
      specialises by domain before committing to a second routing layer. (§10.7)
- [ ] **~5 broad adapters, not 50**, at rank 16. Measures the pattern without
      fragmenting scarce data. (§10.7, §10.2)
- [ ] **Soft-blend with a null/identity option** from the start — not argmax
      switching. This is what preserves general-question behaviour and makes
      misroutes graceful. (§10.3)
- [ ] Measure int8 adapters against bf16 before adopting: `ARCHITECTURE.md` §14.1
      currently specifies bf16 for adapters as "small, sensitive", and error lands
      on the residual stream × 6 loops. If int8 holds, **update §14.1**. (§10.4)
- [ ] Decide the mid-conversation switching policy (pin vs invalidate KV cache)
      before integration, and reuse `_dispatch_grouped` (`model.py:594-627`) for
      batched per-sequence adapters rather than adding a dependency. (§10.6)
- [ ] Run `loop_depth_probe` (`monitoring.py:104`) per adapter before shipping —
      a task score alone will not catch loop-dynamic damage. (§10.5)

Deferred — revisit only if a larger-E model is on the table (§8):

- [ ] Adaptive top-k. **Not queued for this model.** If a future config goes to
      E≥32, re-read §8 before designing the router: the fixed-shape constraint
      (§8.3) and the always-max-k degenerate solution (§8.5) are the two things
      that must be designed for up front, not retrofitted.

---

## 12. Rejected, with reasons — do not re-litigate without new evidence

| option | why rejected |
|---|---|
| fp8 training for speed | `_grouped_mm` is bf16/fp16-only (`model.py:570`); `dim=1536` below the payoff shape; A100 has no fp8 silicon; compounds through 6 loops. §3 |
| fp8 to fit a smaller GPU | Saves activations only, and gradient checkpointing already claimed those. Leaves the ~7.6 GB floor untouched. §4 |
| GRPO-style group-relative SFT | User explicitly out of scope; the existing GRPO stack (`rewards.py::compute_group_advantages`, `app.py:3200-3215`) already implements it if ever wanted. |
| Sampling 4 next-token candidates | Forward pass is deterministic (§5.2); any `f(k)` has an exact closed form (§5.3). Pure added variance. |
| Uniform label smoothing at ε=0.1 | V=65,536 → parks ~10% of mass on junk tokens. §5.6 |
| Hints on a fixed random x% of prompts | Spends hints on already-solved problems, *adding* all-8-right zero-gradient groups. Adaptive 0/8-triggered selection dominates it at the same cost. §6.4 |
| Single-strength hints | GRPO needs within-group variance, not success. One strength cannot land different-difficulty prompts near the p=0.5 variance peak. §6.3 |
| A correction layer applied to the *logits* (monotone `f(z)`) | Cannot change greedy argmax — accuracy is bit-identical. Only calibration/sampling moves. Use `g(h)` before the tied projection instead. §7.3 |
| Reaching the output head via HRA | `inject_hra` wraps `nn.Linear`; the head is a bare `F.linear` against `embedding.weight`, so no wrapper can attach. Needs its own module. §7.4 |
| Variable top-k at E=8 | Payoff scales with E; 25% density already, and 12→8 was a deliberate move *toward* density. Deferred to E≥32, not rejected outright. §8.2 |
| Unpadded variable-k dispatch | `sum(k_i)` is data-dependent → dynamic shape into grouped-GEMM → loses the fullgraph compile worth 9-12%. §8.3 |
| Batch-global expert budget | Keeps shapes and compute fixed, but couples a token's k to its batch-mates — breaks autoregressive decode at batch=1. §8.4 |
| Benchmarking padded adaptive-k against fixed top-2 | Padding to `max_k` pays `max_k` FLOPs; the honest baseline is fixed top-`max_k`. §8.7 |
| Building a retriever / vector store for training | ~3× the cost per token of just pretraining on the same corpus, and FineWeb-Edu + Wikipedia already stream. Use context distillation on hints instead. §9.4, §9.6 |
| Live web search in the training loop | Non-reproducible across resumes, variable quality, licensing exposure. A frozen dump is strictly better, and is what is already in use. §9.4 |
| Context distillation with a differentiable teacher | Degenerate solution: make the output invariant to the context. KL → 0 while the model learns to *ignore* retrieval, and the loss curve looks great. Teacher must be stop-grad. §9.2 |
| Mixture-of-LoRA at rank 256 | 50 × 14.16M = 708M of adapters, larger than the 601M base and ~3.8× the §14.2 deployment target. Low rank is what makes the pattern viable. §10.2 |
| Hard adapter switching on classifier argmax | A misroute is catastrophic, and it cannot preserve general-question behaviour. Soft-blend the softmax with a null/identity option instead. §10.3 |
| Starting at 50 adapters | Data fragmentation at 0.4× Chinchilla, and it destroys cross-domain transfer. ~5 broad adapters give most of the serving benefit at 1/10th the cost and tell you whether 50 is worth it. §10.7 |

---

## 13. Standing constraint

`docs/AGENT_HANDOFF.md:57-63`: the base has seen **~2.2B tokens ≈ 0.4× Chinchilla**
for 278M active params. SFT and GRPO *elicit* latent capability; they do not
*create* it. **None of the objective changes in §5 add knowledge.** They are
worth doing only for what they specifically claim (hard-token weighting, routing
robustness, generation variety) — not as a route to GSM8K.
