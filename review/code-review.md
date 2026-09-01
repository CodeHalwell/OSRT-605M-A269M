# Deep-Dive Code Review: nano-osrt-100m

**Project Name:** OSRT — Optimized Sparse Recursive Transformer  
**Date:** 2026-06-08  
**Scope:** Deep-dive analysis of the codebase under `src/osrt/` and its companion modules.

---

## 1. Executive Verdict & Test Verification

A complete verification run of the test suite (`uv run pytest`) was executed:
- **Result:** `135 passed, 14 warnings in 27.05s`.
- **Package Layout:** The project metadata (`pyproject.toml`) and import paths have been successfully updated to package `src/osrt`. All test files successfully collect and execute, proving that the repository is in a cohesive, runnable state.
- **Spec vs. Code Discrepancies:** A key finding of this review is that **the codebase's implementation is significantly more mature and robust than the design proposals** (specifically `../docs/ARCHITECTURE.md` drafts). Several critical bugs flagged by Codex in the design documents are already correctly resolved in the code:
  - **mHC shape errors** are avoided using explicit `torch.einsum` contractions.
  - **Expand aliasing** is prevented by using `.repeat()` instead of `.expand()`.
  - **Final mHC collapse** is cleanly handled via a dedicated `mhc_collapse` parameter.

---

## 2. File-by-File Technical Deep Dive

### 2.1 Model Architecture & Recursion ([model.py](file:///Users/danielhalwell/nano-osrt-100m/src/osrt/model.py))

The OSRT architecture implements depth recurrence by passing token representations iteratively through 3 physical blocks for a total of 6 loops (generating 18 effective layers).

#### Strengths
* **Symmetry Breaking:** The use of `loop_embeddings` added before the first physical block in each loop breaks representation symmetry, allowing different iterations to specialize (e.g., low-level syntax vs. high-level semantics).
* **Robust MoE Layer:** The MoE block combines a single always-active shared expert (FFN size 4096) with 8 routed experts (FFN size 2048, top-2 active). This structure replaces the standard parallel FFN, focusing parameters where they matter.
* **Aux-Loss-Free Expert Balancing:** Implements a per-loop, per-expert additive load-balancing bias controller (`router_balance_bias`). The bias is updated at each step based on micro-batch load deviations and directly shifts selection logits, but is **omitted from the gating gradients**. This avoids task-gradient distortion while ensuring experts do not starve.
* **Exploration & Stability:** Integrates Gumbel top-k noise (annealed during training) to keep experts active, and utilizes QK-Norm to prevent logit explosion from depth unrolling and Muon updates.

#### Risks & Recommendations
> [!IMPORTANT]
> **Deterministic Hash Routing under Recurrence:**
> The hash routing mechanism for blocks 0 and 1 uses `(token_id + loop_idx) % num_routed_experts`. While loop-indexed hashing is a significant improvement over static hashing, it restricts routing capacity.
> * **Recommendation:** Ensure that the switch from hash to learned routing remains a clean configuration parameter (`hash_routing_blocks`) so that the model can be annealed to 100% learned routing late in pre-training.

---

### 2.2 Manifold-Constrained Hyper-Connections ([mhc.py](file:///Users/danielhalwell/nano-osrt-100m/src/osrt/mhc.py))

`mhc.py` implements the manifold-constrained residual stream, which expands the residual space to `n_hc` channels and mixes them dynamically using token-generated routing matrices.

#### Strengths
* **Einsum implementation:** The tensor contractions in `input_view` and `update` are cleanly written using `torch.einsum` (e.g., `torch.einsum("bsc,bscd->bsd", a, X)`), ensuring shape-safety.
* **Log-Domain Sinkhorn-Knopp:** The Sinkhorn projection onto the Birkhoff polytope is implemented in the log-domain using alternating `logsumexp` normalizations:
  ```python
  log_m = log_m - torch.logsumexp(log_m, dim=-1, keepdim=True)  # rows
  log_m = log_m - torch.logsumexp(log_m, dim=-2, keepdim=True)  # cols
  ```
  This is a critical stability feature. A naive `exp` followed by division is highly prone to gradient explosion and underflow; log-domain Sinkhorn preserves gradients through the 20 iterations.
* **Identity Initialization:** The static bias parameter `s_res` is initialized to a scaled identity matrix (`torch.eye(n_hc) * 4.0`), meaning that at step 0 the network behaves exactly like a standard residual connection before learning channel mixing.

#### Risks & Recommendations
> [!TIP]
> **Compilation Overhead:**
> Running 20 iterations of log-space Sinkhorn-Knopp per sub-block per loop in Python creates significant kernel launching overhead.
> * **Recommendation:** Always compile the mHC module using `@torch.compile` during training. For deployment, consider optimizing this step via a custom Triton or fused CUDA kernel if profiling reveals a bottleneck.

---

### 2.3 Muon Optimizer & Parameter Routing ([muon.py](file:///Users/danielhalwell/nano-osrt-100m/src/osrt/muon.py))

Muon (Momentum Orthogonalized by Newton-Schulz) updates 2D weight matrices by applying a quintic Newton-Schulz iteration to the momentum buffers, restricting updates to the Stiefel manifold.

#### Strengths
* **Hybrid Layout:** The `HybridMuonAdamW` wrapper and `build_param_groups` cleanly coordinate the optimization split:
  - **Muon:** 2D hidden weights (attention, MoE routed experts, router projections).
  - **AdamW:** Embeddings, LM head, norms (including QK-Norm scales), and biases.
* **Precision and Speed:** Newton-Schulz is executed in `bfloat16` and uses symmetric matrix multiplications, reducing the FLOP overhead to $<1\%$ of the forward-backward pass.
* **Decoupled Weight Decay:** Multiplies the parameters by `(1.0 - lr * wd)` directly, which is crucial for preventing spectral drift when updates are projected onto the Stiefel manifold.

---

### 2.4 Quantization & KV Cache ([quant.py](file:///Users/danielhalwell/nano-osrt-100m/src/osrt/quant.py))

`quant.py` provides a standalone deployment and rollout quantization utility compressing the key cache latent (`K_DOWN`) to symmetric 4-bit integers.

#### Strengths
* **Outlier Mitigation via Random Rotation:** Incorporates a randomized orthogonal Sylvester-Hadamard rotation with sign flips to spread out outlier channel activations across the block dimension. This lowers the peak magnitude and preserves precision on smaller channels.
* **Symmetric Grid:** Restricts quantization to the symmetric 15-level grid `[-7, 7]` (dropping the asymmetric `-8` level), which prevents introducing a half-step DC bias on zero-mean representation channels.
* **Nibble Packing:** Includes bit-shift packing/unpacking utilities to compress two int4 values into a single `uint8` byte, halving the cache storage footprint.

---

### 2.5 Reward Stack & Security ([rewards.py](file:///Users/danielhalwell/nano-osrt-100m/src/osrt/rewards.py))

`rewards.py` implements the verifiable, rule-based reward functions used in the GRPO reinforcement learning loop.

#### Strengths
* **Loopholes Closed:** The introduction of `extract_numeric_answer_strict` prevents a common GRPO exploit ("last-number-wins") where the model spammed candidate numbers hoping the final one matched. It evaluates the answer block for:
  - Uniqueness (`single_number`)
  - Formatting markup (`boxed` or backticks)
  - Concluding phrasing (e.g., `therefore, N` or `answer is N`)
  It penalizes ambiguous answers containing multiple unmarked numbers with `ambiguous_penalty` (default `-0.5`).
* **Length-Ramp Penalty:** Replaces binary truncation thresholds with a smooth linear ramp starting at $80\%$ of the token limit, capping at `-1.0` at $100\%$ usage. This discourages pathologically padding the `<think>` block.
* **Graduated Reasoning Bonus:** Only awards the reasoning step bonus if the final answer is correct, and scales the bonus relative to the number of reasoning steps (capped at 3+ steps).

---

## 3. Recommended Remediation & Action Items

### Tier 1 — High-Priority Design Decisions
1. **The GQA / MLA "V-from-K" Rank Bottleneck:**
   The current specification maps $V$ linearly from $K_{DOWN}$ ($V = W_{V\_FROM\_K} K + b_V$). If a token must act as a routing anchor (high Key affinity) but carry context-specific information (independent Value), this creates an expressivity bottleneck.
   - **Remediation Option A:** Accept the constraint (retains the lowest parameter footprint).
   - **Remediation Option B:** Widen the latent $K$ projection to 768 or 1024 dimensions, cache only this latent, and project to independent $K$ and $V$ heads at decode time. This mimics true MLA and restores representation capacity.
2. **Speculative Decoding Mode:**
   Ensure the speculative decoding path has two distinct modes:
   - `greedy_speculative`: Fast, deterministic matching.
   - `sampling_speculative`: Employs acceptance/rejection testing based on draft and target probabilities to preserve the output distribution.

### Tier 2 — Documentation & Parameter Audit
1. **Parameter & FLOP Reconciliation:**
   Write a small `compute_budget.py` script that takes the model configuration file (`OSRTConfig`) and dynamically generates:
   - Exact physical and active parameters.
   - BF16 weight memory.
   - Optimizer memory.
   - Estimated FLOPs per token.
   Use the script's output to replace hand-written totals in `README.md` and `../docs/ARCHITECTURE.md` to prevent discrepancies (e.g., the vocab dimensions and attention parameter mismatches in the docs).
2. **Clarify Training Loss Terminology:**
   To prevent implementer confusion, ensure the configuration and training logs distinguish clearly between:
   - `loop_lm_aux_loss_weight` (LM-head prediction on intermediate loops)
   - `router_balance_loss_weight` (Global Switch load balancing)
   - `router_z_loss_weight` (Log-partition regularization)
   - `loss_free_router_bias_enabled` (Per-step load bias adjustment)
