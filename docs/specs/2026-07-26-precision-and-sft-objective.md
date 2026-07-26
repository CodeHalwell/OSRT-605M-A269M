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
completely wrong. **(Open item, §6.)**

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

## 6. Open items

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

---

## 7. Rejected, with reasons — do not re-litigate without new evidence

| option | why rejected |
|---|---|
| fp8 training for speed | `_grouped_mm` is bf16/fp16-only (`model.py:570`); `dim=1536` below the payoff shape; A100 has no fp8 silicon; compounds through 6 loops. §3 |
| fp8 to fit a smaller GPU | Saves activations only, and gradient checkpointing already claimed those. Leaves the ~7.6 GB floor untouched. §4 |
| GRPO-style group-relative SFT | User explicitly out of scope; the existing GRPO stack (`rewards.py::compute_group_advantages`, `app.py:3200-3215`) already implements it if ever wanted. |
| Sampling 4 next-token candidates | Forward pass is deterministic (§5.2); any `f(k)` has an exact closed form (§5.3). Pure added variance. |
| Uniform label smoothing at ε=0.1 | V=65,536 → parks ~10% of mass on junk tokens. §5.6 |

---

## 8. Standing constraint

`docs/AGENT_HANDOFF.md:57-63`: the base has seen **~2.2B tokens ≈ 0.4× Chinchilla**
for 278M active params. SFT and GRPO *elicit* latent capability; they do not
*create* it. **None of the objective changes in §5 add knowledge.** They are
worth doing only for what they specifically claim (hard-token weighting, routing
robustness, generation variety) — not as a route to GSM8K.
