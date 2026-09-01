# OSRT — Architecture Optimisation Review (2026-06-08)

**Scope:** the *model architecture* code path — `src/osrt/model.py`, `src/osrt/mhc.py`,
`src/osrt/config.py`, `src/osrt/presets.py` — judged against the **headline locked
config** `OSRT_605M_A288M` (the config that actually trains), not the library
defaults. Companion to `review/code-review.md` and
`review/deep-dive-code-review-2026-06-08.md`; this one is narrowly about
*efficiency / "is it fully optimised"*, not a from-scratch correctness audit.

**Headline config that drives every number below**
(`presets.py:OSRT_605M_A288M`):

```
dim=1536  heads=24  head_dim=64  num_kv_heads=8  → kv_dim = 8·64 = 512
vocab=65536  num_blocks=3  recursive_loops=6  → 18 effective layers
routed_experts=8  top_k=2  expert_hidden=3840  shared_expert_hidden=2816
use_mhc=True (n_hc=4, sinkhorn_iters=20)  attention_sink=True
router_affinity="sqrt_softplus"  mtp_heads=2  aux_loop_loss_weight=0.05
max_position=4096 ; training curriculum seq_len 2048 → 4096 → 8192
```

---

## TL;DR verdict

The implementation is **already very well engineered**. The hot paths are
deliberate, the tradeoffs are documented, and the test suite is green
(`144 passed`). "Fully optimised" does **not** reduce to a pile of missed
micro-wins — it reduces to **one safe micro-win (applied)** plus **four
structural items that only bite on the GPU phase**, two of which are genuine
scaling walls for this config's own Phase-3 `seq_len=8192`.

Findings are split into two buckets:

- **Bucket A — provably bit-identical / structure-identical.** Safe to apply now,
  gated by tests. (1 item, **already applied**, `144 passed` before & after.)
- **Bucket B — changes numerics OR module structure.** These are
  **recommendations only**. They invalidate in-flight runs and/or break
  `load_state_dict` on existing checkpoints, so they must not be folded into a
  cleanup commit — they need explicit sign-off + a sanity run before adoption.

---

## Bucket A — applied (bit-identical, tests green)

### A1. Telemetry-only MoE stats were computed on every training step

`MoELayer.forward` computed `dispatch_one_hot`, `f`, and `p` *before* the
telemetry guard, but those three tensors feed **only** the diagnostic block
(`last_expert_fraction`, marginal/assignment entropy). The actual gradient path —
the Switch balance loss — uses the separate `raw_balance_*` tensors and was never
touched.

`set_moe_telemetry(False)` already skips the `.item()`/`.tolist()` block on
non-logging steps, but these three tensors (an `F.one_hot` over `(N, K)` plus two
reductions, **per MoE call × 18 effective layers × every non-logging step**) were
computed and discarded anyway. `p = probs.float().mean(0)` also built and threw
away an autograd node.

**Fix applied:** moved the three lines inside the existing `with torch.no_grad():`
telemetry guard. Forward outputs, all four losses, and gradients are unchanged —
verified `144 passed` before and after. This is the only change made to the tree.

---

## Implementation status — B1 + B2 built on a worktree

B1 and B2 are now **implemented, parity-tested, and isolated** on the git
worktree `/Users/danielhalwell/nano-osrt-100m-wt-b1b2` (branch
`b1b2-attn-sink-fused-ce`, based at the current HEAD). **Main is untouched.** Both
are behind config flags that **default OFF (bit-identical)**, so adopting them is a
deliberate config choice; the parity tests prove turning them on changes neither
the loss nor the gradients beyond fp tolerance.

- **B2 — `fused_cross_entropy_chunks: int = 0`** (`src/osrt/fused_ce.py`, wired into
  the aux-loop + MTP head losses). `> 0` computes those head losses with a
  gradient-checkpointed chunked linear-CE: only ~1/chunks of the `(N, vocab)`
  logits live at once. 7 parity tests (loss + d_hidden + d_weight, real-vocab
  slicing, single-chunk, all-ignored→0).
- **B1 — `attention_sink_impl: str = "manual"`** (`"flex"` opt-in). `"flex"` uses
  `flex_attention(return_lse=True)` + the identical `sigmoid(lse − sink[h])`
  rescale, with the causal `BlockMask` cached per `(S, total_len, past_len)` so it
  is built once per block and reused (steady-state zero rebuilds — a test enforces
  this). 6 parity tests (prefill / cached-decode / backward match manual; config
  validation; flex-path-taken spy; block_mask-cached-once).

Worktree result: **157 passed** (144 baseline + 13 new), authored files clean on
`ruff` + `ty`.

**What is verified vs NOT (read before enabling):**
- ✅ **Verified:** loss + gradient *numerical parity* between the new and old paths,
  in **eager mode on CPU**. Turning either flag on does not change the math.
- ❌ **NOT verified (needs the GPU box):** (a) the actual **memory/throughput
  benefit** — neither flag's win is measured here; (b) **compiled-CUDA numerics**
  for B1 — the `torch.compile(flex_attention)` path has never run; eager flex even
  warns it materialises the scores internally, so the flash win is unrealised on
  CPU by construction.
- ⚠️ **B1 must-resolve at GPU bring-up:** the original author *rejected*
  flex_attention over **compiled-GPU** behaviour — these CPU-eager parity tests do
  not retire that concern, they defer it. Also `return_lse=True` is **deprecated on
  torch ≥ 2.10** (`return_aux=AuxRequest(lse=True)`); kept as-is here because it
  works on this box, but switch it against the actual GPU torch version. Confirm the
  cached `BlockMask` survives `torch.compile` (it is built `_compile=False` and
  reused, the recommended pattern, but unproven under compile here).

Enable in a preset with e.g. `fused_cross_entropy_chunks=8` and/or
`attention_sink_impl="flex"` **only after** a GPU sanity run confirms parity under
compile and an actual memory/throughput win. B2 is the lower-risk of the two to
enable first (no kernel/compile dependency).

---

## Bucket B — recommendations (do NOT apply without sign-off + sanity run)

Ranked by impact on the upcoming GPU runs. **B1 and B2 below are now implemented on
the worktree branch (see above); B3–B5 remain recommendations.**

### B1. `attention_sink=True` disables flash attention → O(B·H·S²) fp32 scores per layer  **[HIGH]**

`presets.py` sets `attention_sink=True`, so **every** attention call routes to
`_attention_with_sink` (`model.py:1053`), not `F.scaled_dot_product_attention`.
That manual path **materialises the full score matrix** `scores` of shape
`(B, H, S, total_len)` and then `scores.float()` (a second fp32 copy) plus a
softmax copy.

At the Phase-3 curriculum point (`seq_len=8192`, `B=2`, `H=24`):

```
scores fp32:   2·24·8192·8192·4 B ≈ 12.9 GB   (per layer, before checkpoint recompute)
+ scores.float() second copy, + softmax weights, + repeat_interleave'd K/V (GQA expand to 24 heads)
```

That is per-layer attention memory in the tens of GB at Phase-3 length — it will
either OOM or force pathologically small batches, and it throws away the whole
point of flash attention. The code comment correctly notes the manual path is
"fine for our sequence lengths" — but that judgement predates committing to
`seq_len=8192` in the headline curriculum.

The sink is exactly expressible as **one extra key** whose pre-softmax score is
`sink_logit[h]` and whose value is the zero vector (it only enlarges the softmax
denominator). The flash-friendly implementations:

- **GPU phase:** `torch.flex_attention` with a `score_mod` that adds `sink_logit[h]`
  to an appended sink column, compiled under `torch.compile` — keeps the fused
  kernel. (The code comment rejected `flex_attention` on *torch 2.12 / CPU*; that
  caveat does not apply to the compiled GPU path, which is where this matters.)
- Or keep the manual path but **only** at the short Phase-1/2 lengths and switch to
  SDPA-without-sink (bit-identical to `attention_sink=False`) for Phase-3, if the
  sink turns out not to be load-bearing — that's an ablation worth running first.

**Recommendation:** before any seq_len≥4096 GPU run, replace `_attention_with_sink`
with a flash-compatible sink (flex_attention score_mod) **or** demonstrate via
ablation that the sink can be dropped at long context. This is the single biggest
"not fully optimised" item for the GPU phase.

### B2. Eight full-vocab (65536) fp32 logit materialisations per training step  **[HIGH]**

Per training step the model projects the hidden state to `vocab=65536` logits and
upcasts to fp32 for cross-entropy, **once per head**:

- main `+1` LM head (`model.py:1654` / `:1671`)
- `mtp_heads=2` MTP heads (`:1766`)
- up to `recursive_loops-1 = 5` per-loop aux LM heads (`:1719`)

= **up to 8** `(B, S, 65536)` fp32 logit tensors per step. At `B=2, S=8192` a
single such tensor is `2·8192·65536·4 B ≈ 4.3 GB`; eight of them (even staggered)
make the LM-head region the dominant activation-memory term — larger than the
transformer body.

**Recommendation:** adopt a **fused linear-cross-entropy** kernel (e.g.
Liger-Kernel `fused_linear_cross_entropy`, or a chunked-vocab CE) for all heads.
It computes CE without ever materialising the `(B, S, 65536)` logits, typically
cutting LM-head activation memory by ~4–8× and speeding the step up. This is
training-only, does not change the model's parameters or the inference path, and
preserves the math — but it changes the loss-compute code, so it needs a numerical
parity check (loss within fp tolerance) on a sanity run before adoption. High value
for fitting Phase-3 on a 3090/A100-40GB.

### B3. MLA decode recomputes `v_from_k` over the entire past every step  **[MED — profile-gated]**

The KV cache stores only the un-rotated latent `c_kv` (the MLA half-cache win).
Each decode step then does `v = self.v_from_k(c_kv)` over the **full**
`(B, total_len, 512)` latent (`model.py:1011`), even though `v_from_k` is
position-independent and the past rows never change. K's RoPE is likewise
re-applied over all positions (cheap, elementwise). Net: O(L) redundant GEMM work
per step, **O(L²) per generation**.

With `kv_dim=512` the per-layer recompute is `total_len · 512² · 2 ≈ total_len·0.52 MFLOP`,
×18 layers ≈ `total_len · 9.4 MFLOP`:

| decode context L | v_from_k recompute / step | new-token FFN+attn / step | ratio |
|---|---|---|---|
| 512  | ~4.8 GFLOP  | ~1.95 GFLOP | ~2.5× |
| 2048 | ~19 GFLOP   | ~1.95 GFLOP | ~10×  |
| 4096 | ~39 GFLOP   | ~1.95 GFLOP | ~20×  |

For a reasoning model that emits long CoT (L routinely 1–4k during a single
generation) this becomes the dominant decode-compute term.

**Caveat (why this is profile-gated, not auto-fix):** the fix — compute `v_from_k`
only on the new token and keep an incremental **V cache** — trades the MLA memory
win away. The latent-only cache stores `512/pos/layer`; caching V too stores
`1024/pos/layer` (i.e. a normal GQA KV cache). Whether the recompute or the memory
matters more is **hardware- and context-length-dependent**, and decode is often
memory-bound (the GEMM may be cheap in wall-clock). **Recommend: profile decode at
representative CoT length on the target GPU first**; only add the V cache if
`v_from_k` shows up as a real bottleneck. It also changes the cache contract
(present tuple carries V), so it's a structural change → Bucket B.

### B4. Naive per-expert Python dispatch loop  **[MED — large rewrite]**

MoE dispatch loops over experts in Python (`for ei, expert in enumerate(self.experts)`,
`model.py:698`), one expert forward per expert per call = **8 experts × 18 effective
layers = 144 separate small GEMM groups per forward**. This is the canonical naive-MoE
pattern; it's correct, but on GPU it's many small kernel launches with poor SM
utilisation when per-expert token counts are uneven.

**Recommendation (future work, not now):** a grouped/batched GEMM (megablocks-style
grouped_gemm, or `torch._grouped_mm` where available) computes all experts in one
fused call with identical math. Meaningful throughput win at the GPU phase, but it's
a real rewrite of the dispatch path and should be its own task with parity tests.
Not worth doing while the model is still in CPU pre-flight.

### B5. mHC cost is a known watch-item — keep it gated  **[track]**

`use_mhc=True` makes the residual stream `(B, S, n_hc=4, D)` — **4× the residual
activation memory** — and runs `2 sub-blocks × 3 blocks × 6 loops = 36` log-domain
Sinkhorn projections (20 iters each) per forward. The preset comment already flags
mHC as NaN-prone under sustained CPU training and "needs profiling on real hardware."

No action requested here — just confirming the cost is real and the team's existing
"profile on GPU before trusting mHC, keep `use_mhc` a clean off-switch" stance is the
right one. If GPU profiling shows the 4× residual memory is the binding constraint,
`n_hc=2` or disabling mHC are the obvious knobs.

---

## Things that look like inefficiencies but are correct/deliberate (no change)

- **Shared expert runs dense on the full `(B,S,D)` every loop** — that's the design
  (it replaces the dense FFN); not waste.
- **KV cache grows unbounded in `generate()` (no trim)** — deliberate; trimming
  would shift RoPE absolute positions and break attention. Documented at
  `model.py:1954`.
- **Balance loss computed on raw pre-bias logits** — deliberate, so the aux gradient
  pushes the *learned* router while the bias controller handles deployed load.
- **`loop_rms` reduction every loop even in eval** — low-value to guard, and its
  length is reused for the aux-loss normaliser; leave it.
- **Loop dropout uses Python `random`** — would desync across DDP ranks, but training
  is single-GPU (no DDP/FSDP in the tree), so it's a *latent* risk only. Note it if
  multi-GPU is ever added; the fix is a rank-broadcast or torch-RNG draw.

---

## Verification

- `uv run pytest` → **144 passed** (baseline) and **144 passed** (after the A1 edit).
- `uv run ruff check src/osrt/model.py` → the 5 E501 line-length warnings are all
  pre-existing (lines 1807, 1816, 2067, 2249, 2281), none introduced by the edit.
- Independent correctness pass of `model.py` (KV/RoPE indexing, causal masking,
  MoE dispatch, loop-dropout/aux interaction, speculative-decode cache math) was run
  in parallel; see the correctness section below.

---

## Correctness audit (independent parallel pass) — no material bug found

A separate read-only audit traced all five high-risk areas (including hand-tracing
the speculative-decode cache math for full-accept / partial-accept / budget-truncation
cases). **No correctness fix is warranted.** Summary of what was verified correct:

- **KV cache / RoPE indexing:** latent cached on `dim=1`; q rotates over
  `[past_len:total_len]`, k over `[:total_len]`; absolute positions preserved across
  prefill→decode. `num_loops` keys the per-effective-layer cache count consistently
  (`expected_past_layers = num_blocks·n_loops`, `idx = loop·num_blocks + block_idx`).
  Speculative `keep = cache_len + accept + 1`, `_trunc` on `dim=1`, and the full-accept
  single-`ext_toks` extend all preserve the cache invariant; the `len(new_cols) > limit`
  budget truncation only fires on the terminal round so the over-kept cache is never reused.
- **Causal masking:** SDPA `triu(diagonal=1+past_len)` and the manual-sink
  `col <= past_len + row` are the same predicate; the sink rescale
  `sigmoid(lse − sink[h])` is the algebraically exact zero-value-sink denominator.
- **MoE dispatch:** `topk` indices are distinct per token → `index_add_` has no
  within-expert double-add; capacity drop is position-uniform (`randperm`); gates
  renormalise to 1; balance loss `E·Σ f_i p_i` uses consistent raw `f`/`p`, minimised at uniform.
- **Loop-dropout / aux / normalisation:** intermediate hiddens captured pre-`norm_loop`
  with the dedicated `mhc_collapse`; `n_moe_layers = num_blocks·len(loop_rms)` tracks
  *executed* loops, so the regulariser isn't halved on truncated batches; dropout is
  train-only so eval is deterministic/chunk-stable.
- **Stale side-effects:** per-layer losses reset to `None` each `MoELayer.forward`;
  accumulation guards on `is not None`; eval is drop-free with Gumbel annealed to 0.

**Minor non-blocking observations (only one actioned):**

1. **Stale comment** at the decode no-trim block — referenced `past_key_values[0].shape[2]`
   and obsolete line numbers; the code reads `shape[1]`. **Fixed** (comment-only,
   bit-identical) as part of this review.
2. `expected_past_layers` is computed before loop-dropout can shrink `n_loops_to_run`;
   only mismatches if cached-training were ever added (forbidden today by the
   checkpoint/cache guard). Worth a guard comment if cached training is introduced.
3. `expert_ema_fraction` is a logging diagnostic, not the controller input — the bias
   delta intentionally uses the instantaneous fraction per the documented DeepSeek
   formula. Not a bug.
4. In training, applied gates derive from Gumbel-noised `probs` (noised gate *magnitude*,
   not just selection); `gumbel_tau` anneals to 0 before eval. Deliberate.
