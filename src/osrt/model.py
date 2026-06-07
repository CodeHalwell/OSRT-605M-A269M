"""OSRT — Mixtral-style MoE without dense FFN.

Architecture changes from v4:
  - Dense FFN removed. Shared expert (hidden=4096) replaces it.
  - 8 routed experts × hidden=2048 (was 11 × 1024).
  - Top-2 softmax routing (Mixtral-style), renormalised gates.
  - Switch balance loss: num_experts * sum(f_i * p_i).
    NO importance loss (it enforced uniformity and killed v4's router).
    Uses a DeepSeek-style balance-bias controller plus annealed Gumbel top-k
    noise to keep experts alive while the router learns token preferences.
  - Capacity factor 2.0, tokens exceeding capacity skip that expert's branch.
  - Orthogonal per-expert initialisation breaks symmetry at step 0.

Kept from v4:
  - Recursive weight sharing (3 physical blocks × 6 loops).
  - Per-pass low-rank adapters (residual, not LoRA).
  - Causal attention with RoPE.
  - KV cache.
  - Loop embeddings for per-loop routing preferences.
  - HuggingFace PreTrainedModel compatibility.

Default config (measured on the actual model):
  Physical params      : 362,720,259 (~363M, LM head tied with embedding)
  Active / token (body): ~192M       (shared expert + 2 of 8 routed + attn + embed)
  Block applications    : 18          (num_blocks × recursive_loops)
"""

import math
import random
from contextlib import contextmanager
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from osrt.config import OSRTConfig
from osrt.mhc import ManifoldHyperConnection

# ── RoPE ────────────────────────────────────────────────────────────────


def compute_rope_freqs(
    seq_len: int,
    dim: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    scaling: dict | None = None,
) -> tuple[Tensor, Tensor]:
    """Pre-compute RoPE cos/sin tensors. Shape: (1, seq_len, 1, dim)."""
    if dim % 2 != 0:
        raise ValueError(f"RoPE requires even dimension, got dim={dim}")

    effective_theta = theta
    if scaling is not None:
        stype = scaling.get("type", "").lower()
        factor = float(scaling.get("factor", 1.0))
        if stype == "ntk" and factor > 1.0:
            effective_theta = theta * (factor ** (dim / (dim - 2)))

    freqs = 1.0 / (
        effective_theta ** (
            torch.arange(0, dim, 2, dtype=torch.float32, device=device)[: dim // 2]
            / dim
        )
    )
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    sin = sin.unsqueeze(0).unsqueeze(2)
    return cos, sin


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    # RoPE buffers are stored in fp32 for stable precompute, but attention runs
    # under bf16 autocast. Cast here so q/k do not get promoted back to fp32.
    if cos.dtype != x.dtype or cos.device != x.device:
        cos = cos.to(device=x.device, dtype=x.dtype)
    if sin.dtype != x.dtype or sin.device != x.device:
        sin = sin.to(device=x.device, dtype=x.dtype)
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    x_rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + x_rot * sin


# ── Expert FFN ──────────────────────────────────────────────────────────


class ExpertFFN(nn.Module):
    """SwiGLU feed-forward. Used for both shared and routed experts."""

    def __init__(self, dim: int, hidden: int, clamp: float | None = None) -> None:
        super().__init__()
        hidden = 64 * ((hidden + 63) // 64)  # TC-align
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        # Optional SwiGLU stability clamp (ARCHITECTURE.md §7.8): bound the
        # gate (max) and up (both sides) pre-activations so a single extreme
        # activation can't blow up the product. None → no clamp (no-op for a
        # healthy model; the bound just caps tails).
        self.clamp = clamp

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w_gate(x)
        up = self.w_up(x)
        if self.clamp is not None:
            gate = gate.clamp(max=self.clamp)
            up = up.clamp(min=-self.clamp, max=self.clamp)
        return self.w_down(F.silu(gate) * up)


def orthogonal_expert_init(expert: ExpertFFN, seed: int, gain: float = 1.0) -> None:
    """Initialise an expert's projections with orthogonal columns.

    Ensures experts start in different feature subspaces so gradients
    push them in distinct directions. Uses QR decomposition of a random
    matrix (deterministic given seed).

    w_gate, w_up: (hidden, dim) — columns span a subspace of R^hidden
    w_down: (dim, hidden) — rows span a subspace of R^dim

    `gain` scales the resulting matrix to match standard init variance
    (roughly 1/sqrt(fan_in) for nn.Linear).
    """
    gen = torch.Generator(device=expert.w_gate.weight.device)
    gen.manual_seed(seed)
    with torch.no_grad():
        for lin in (expert.w_gate, expert.w_up, expert.w_down):
            w = lin.weight  # (out, in)
            rows, cols = w.shape
            # Generate random matrix with same dtype/device
            rand = torch.randn(
                max(rows, cols), min(rows, cols),
                generator=gen, device=w.device, dtype=w.dtype,
            )
            q, _ = torch.linalg.qr(rand)
            # q is orthonormal along its shorter axis. After slicing:
            #   rows >= cols (fat): q has orthonormal columns; element
            #     std = 1/sqrt(rows), NOT 1/sqrt(cols). This was the
            #     previous bug — for w_gate/w_up (hidden > dim) the
            #     weights came out ~13% under the claimed fan_in std.
            #   rows <  cols (tall, w_down): q has orthonormal rows
            #     after transpose; element std = 1/sqrt(cols) already.
            q = q[:rows, :cols] if rows >= cols else q[:cols, :rows].T
            # Target: std = gain / sqrt(fan_in) where fan_in = cols.
            # q's native element std is 1/sqrt(max(rows, cols)), so
            # rescale by sqrt(max(rows, cols) / cols) * gain.
            scale = gain * math.sqrt(max(rows, cols) / cols)
            w.copy_(q * scale)


# ── MoE Layer (Switch-style) ────────────────────────────────────────────


class MoELayer(nn.Module):
    """Mixtral-style MoE: top-k (default 2) softmax routing, capacity-limited.

    Key differences from v4's MoELayer:
      - Switch balance loss: N * sum(f_i * p_i) — minimises at uniform without
        enforcing it on router probs.
      - No importance/z loss and no soft warmup/blend — sparse routing from
        step 0.
      - Optional persistent balance bias directly controls expert load.
      - Optional training-only Gumbel top-k noise, annealed by the trainer.
      - Dropped tokens (exceeded per-expert capacity) skip that expert's branch
        for this batch.
      - Orthogonal expert init (per-expert QR decomposition).
      - Top-k gates are renormalised so they sum to 1 — router decisions
        don't down-weight the MoE output just because k > 1.
    """

    def __init__(
        self, config: OSRTConfig, moe_seed: int = 0, block_idx: int = 0,
    ) -> None:
        super().__init__()
        self.dim = config.dim
        self.num_routed = config.num_routed_experts
        self.top_k = config.top_k_experts
        self.expert_hidden = config.expert_hidden
        self.capacity_factor = config.router_capacity_factor
        self.num_loops = config.recursive_loops
        # Save seed for deferred orthogonal init (applied after post_init).
        self._moe_seed = moe_seed
        self._orthogonal_init_requested = config.expert_orthogonal_init
        # Hash routing (ARCHITECTURE.md §7.5): this physical block uses
        # deterministic top-1 hash routing instead of the learned router iff
        # block_idx < hash_routing_blocks. Hard switch, decided at construction.
        self.block_idx = block_idx
        self.use_hash_routing = block_idx < config.hash_routing_blocks

        # Shared expert: always active, larger hidden than routed experts.
        # Replaces v4's parallel dense FFN.
        clamp = getattr(config, "swiglu_clamp", None)
        self.shared_expert = ExpertFFN(
            config.dim, config.shared_expert_hidden, clamp=clamp,
        )

        # Routed experts
        self.experts = nn.ModuleList([
            ExpertFFN(config.dim, config.expert_hidden, clamp=clamp)
            for _ in range(self.num_routed)
        ])

        # Router: projects (hidden + loop_emb) → num_routed logits
        self.loop_embeddings = nn.Embedding(config.recursive_loops, config.dim)
        self.loop_embeddings._osrt_init_std = config.loop_embedding_init_std
        self.router = nn.Linear(config.dim, self.num_routed, bias=False)
        # Affinity transform: "softmax" (historical) or "sqrt_softplus"
        # (DeepSeek-V4). See forward() for how each maps logits -> routing.
        self.router_affinity = config.router_affinity
        self.register_buffer(
            "gumbel_tau",
            torch.tensor(config.router_gumbel_tau_init, dtype=torch.float32),
        )

        # Per-loop, per-expert additive load-balancing bias. This is part of
        # the routing mechanism, not an optimizer parameter: it is applied in
        # train/eval selection and saved in checkpoints, then updated once per
        # optimizer step by the training loop via apply_balance_update().
        #
        # Capacity is enforced per MoE call, so loop-specific load imbalance
        # must be corrected per loop. A single block-level bias can look
        # balanced in aggregate while individual loop calls overflow.
        self.bias_enabled = config.router_balance_bias_enabled
        self.bias_update_rate = config.router_balance_bias_update_rate
        self.bias_ema_rate = config.router_balance_bias_ema_rate
        self.bias_max = config.router_balance_bias_max
        self.register_buffer(
            "router_balance_bias",
            torch.zeros(self.num_loops, self.num_routed, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "balance_count_accum",
            torch.zeros(self.num_loops, self.num_routed, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "balance_total_accum",
            torch.zeros(self.num_loops, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "expert_ema_fraction",
            torch.full(
                (self.num_loops, self.num_routed),
                1.0 / self.num_routed,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.balance_accum_enabled = True

        # NOTE: orthogonal expert init is NOT applied here because HF's
        # post_init() walks the module tree and calls _init_weights on every
        # nn.Linear, which would stomp the orthogonal weights. Apply via
        # apply_orthogonal_init() after post_init() has finished.

        # Per-layer losses (set during forward, read by wrapper).
        # balance_loss   — Switch global imbalance, scaled by aux coeff.
        # z_loss         — (logsumexp router_logits)^2; bounds magnitude.
        # seq_balance_loss — Per-sequence Switch; opt-in long-context safety.
        self.balance_loss: Tensor | None = None
        self.z_loss: Tensor | None = None
        self.seq_balance_loss: Tensor | None = None

        # Telemetry — plain Python lists, zero cost.
        # per_token_entropy: mean_token entropy of softmax (real sharpness signal).
        # marginal_entropy: entropy of batch-mean p vector (balance proxy).
        # assignment_entropy: entropy of hard-assignment fractions f.
        # raw_max_prob: mean top-1 softmax prob BEFORE renormalisation (router
        #   confidence — uniform 1/E means no opinion, >1/E means preferences).
        # top_margin: mean (p_rank0 - p_rank1) (confidence gap).
        # drop_rate: fraction of (token, rank) pairs dropped by capacity cap.
        self.last_per_token_entropy: list[float] = [0.0] * config.recursive_loops
        self.last_marginal_entropy: list[float] = [0.0] * config.recursive_loops
        self.last_assignment_entropy: list[float] = [0.0] * config.recursive_loops
        self.last_clean_per_token_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_clean_marginal_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_clean_assignment_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_expert_fraction: list[list[float]] = [
            [0.0] * self.num_routed for _ in range(config.recursive_loops)
        ]
        self.last_clean_expert_fraction: list[list[float]] = [
            [0.0] * self.num_routed for _ in range(config.recursive_loops)
        ]
        self.last_drop_rate: list[float] = [0.0] * config.recursive_loops
        self.last_raw_max_prob: list[float] = [0.0] * config.recursive_loops
        self.last_top_margin: list[float] = [0.0] * config.recursive_loops
        self.last_prebias_per_token_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_prebias_marginal_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_prebias_assignment_entropy: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_prebias_expert_fraction: list[list[float]] = [
            [0.0] * self.num_routed for _ in range(config.recursive_loops)
        ]
        self.last_prebias_raw_max_prob: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_prebias_top_margin: list[float] = [
            0.0
        ] * config.recursive_loops
        self.last_clean_raw_max_prob: list[float] = [0.0] * config.recursive_loops
        self.last_clean_top_margin: list[float] = [0.0] * config.recursive_loops

    def apply_orthogonal_init(self) -> None:
        """Apply orthogonal per-expert init. MUST be called after HF post_init()."""
        if not self._orthogonal_init_requested:
            return
        # nn.ModuleList iterates as Module (generic); we know ours contains
        # ExpertFFN instances because we just constructed them.
        for ei, expert in enumerate(self.experts):
            assert isinstance(expert, ExpertFFN)
            orthogonal_expert_init(
                expert, seed=self._moe_seed * 1000 + ei, gain=1.0,
            )

    @torch._dynamo.disable
    @torch.no_grad()
    def _accumulate_balance_counts(self, top_idx: Tensor, loop_idx: int) -> None:
        """Accumulate clean top-k assignment counts for the bias controller."""
        if not self.bias_enabled:
            return
        flat = top_idx.reshape(-1)
        counts = torch.bincount(flat, minlength=self.num_routed).float()
        self.balance_count_accum[loop_idx].add_(counts)
        self.balance_total_accum[loop_idx].add_(counts.sum())

    @torch.no_grad()
    def apply_balance_update(self) -> None:
        """Update per-expert routing bias once from accumulated clean load."""
        if not self.bias_enabled:
            return
        active = self.balance_total_accum > 0
        if not active.any():
            return

        current_frac = self.expert_ema_fraction.clone()
        current_frac[active] = (
            self.balance_count_accum[active]
            / self.balance_total_accum[active].unsqueeze(-1)
        )
        self.expert_ema_fraction[active] = torch.lerp(
            self.expert_ema_fraction[active],
            current_frac[active],
            self.bias_ema_rate,
        )

        target = 1.0 / self.num_routed
        delta = torch.zeros_like(self.router_balance_bias)
        delta[active] = current_frac[active] - target
        self.router_balance_bias.add_(delta, alpha=-self.bias_update_rate)
        self.router_balance_bias.clamp_(-self.bias_max, self.bias_max)

        self.balance_count_accum.zero_()
        self.balance_total_accum.zero_()

    def _hash_route(
        self,
        x: Tensor,
        x_flat: Tensor,
        shared_out: Tensor,
        token_ids: Tensor,
        loop_idx: int,
    ) -> tuple[Tensor, Tensor]:
        """Deterministic top-1 hash routing (ARCHITECTURE.md §7.5).

        Each token is sent to exactly one expert,
            expert_id = (token_id + loop_idx) % num_routed_experts
        with gating weight 1.0. The shared expert is unaffected; we return the
        same (shared_out, routed_out) contract as the learned path so the Block
        is oblivious to the routing mode. Routing is deterministic, so there is
        no balance/z/seq aux loss to learn — those are set to zero tensors (kept
        non-None so the model's accumulation stays well-defined), and the
        telemetry attributes are populated from the hard hash histogram (never
        left stale from a previous forward, so the collapse monitor stays sane).
        """
        B, S, D = x.shape
        N = B * S
        E = self.num_routed
        device = x.device

        # Loop-indexed hash assignment, one expert per token.
        assign = (token_ids.reshape(N) + loop_idx) % E  # (N,), long

        # Dispatch: gather every token routed to each expert, run it, scatter
        # back at gate weight 1.0. No capacity cap — top-1 hashing is balanced
        # in expectation and dropping tokens here would only add noise.
        moe_out = torch.zeros_like(x_flat)
        for ei, expert in enumerate(self.experts):
            token_indices = (assign == ei).nonzero(as_tuple=True)[0]
            if token_indices.numel() == 0:
                continue
            moe_out.index_add_(0, token_indices, expert(x_flat[token_indices]))
        moe_out = moe_out.view(B, S, D)

        # Aux losses: deterministic routing has nothing to balance. Keep them as
        # zero tensors (not None) so OSRTModel.forward's `is not None` checks and
        # the wrapper's normalisation see a valid contribution.
        zero = torch.zeros((), device=device)
        self.balance_loss = zero
        self.z_loss = zero
        self.seq_balance_loss = zero

        # Telemetry from the hard hash assignment. f_i = fraction of tokens on
        # expert i (top-1, so it sums to 1). Entropy of f is the only meaningful
        # signal here; per-token entropy is 0 (a one-hot assignment) and the
        # "router confidence" metrics are 1.0 by construction.
        with torch.no_grad():
            counts = torch.bincount(assign, minlength=E).float()
            f = counts / counts.sum().clamp_min(1.0)
            f_list = f.tolist()
            f_log = torch.log(f.clamp_min(1e-10))
            assign_ent = -(f * f_log).sum().item()

            self.last_per_token_entropy[loop_idx] = 0.0
            self.last_marginal_entropy[loop_idx] = assign_ent
            self.last_assignment_entropy[loop_idx] = assign_ent
            self.last_expert_fraction[loop_idx] = f_list
            self.last_drop_rate[loop_idx] = 0.0
            self.last_raw_max_prob[loop_idx] = 1.0
            self.last_top_margin[loop_idx] = 1.0
            # Mirror onto the prebias and clean diagnostic families so the
            # collapse monitor (which reads last_clean_*) sees the deterministic
            # assignment rather than stale learned-router values.
            self.last_prebias_per_token_entropy[loop_idx] = 0.0
            self.last_prebias_marginal_entropy[loop_idx] = assign_ent
            self.last_prebias_assignment_entropy[loop_idx] = assign_ent
            self.last_prebias_expert_fraction[loop_idx] = f_list
            self.last_prebias_raw_max_prob[loop_idx] = 1.0
            self.last_prebias_top_margin[loop_idx] = 1.0
            self.last_clean_per_token_entropy[loop_idx] = 0.0
            self.last_clean_marginal_entropy[loop_idx] = assign_ent
            self.last_clean_assignment_entropy[loop_idx] = assign_ent
            self.last_clean_expert_fraction[loop_idx] = f_list
            self.last_clean_raw_max_prob[loop_idx] = 1.0
            self.last_clean_top_margin[loop_idx] = 1.0

        return shared_out, moe_out

    def forward(
        self,
        x: Tensor,
        loop_idx: int,
        token_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Forward pass through MoE.

        Args:
            x: Hidden states (B, S, dim).
            loop_idx: Current recursive loop index.
            token_ids: Optional (B, S) input token ids. Only consumed when this
                block hash-routes (block_idx < hash_routing_blocks); the learned
                router never looks at them. Defaults to None so the standard
                path and signature stay backward-compatible.

        Returns:
            (shared_out, routed_out): both (B, S, dim). Caller applies
            moe_gate to routed_out only; shared_out is always full weight.
        """
        B, S, D = x.shape
        N = B * S
        x_flat = x.reshape(N, D)

        # Reset losses (prevents stale values in eval)
        self.balance_loss = None
        self.z_loss = None
        self.seq_balance_loss = None

        # Shared expert (always active, not gated by caller's moe_gate)
        shared_out = self.shared_expert(x)

        # ── Hash routing (ARCHITECTURE.md §7.5) ──
        # Deterministic top-1 dispatch: expert_id = (token_id + loop_idx) % E,
        # a loop-indexed hash. No learned router, no balance loss/z-loss/Gumbel.
        # Used to stabilise early blocks before the learned router warms up.
        if self.use_hash_routing:
            if token_ids is None:
                raise ValueError(
                    "hash routing requires token_ids at the MoE layer; "
                    "OSRTModel.forward must thread input_ids through the block."
                )
            return self._hash_route(x, x_flat, shared_out, token_ids, loop_idx)

        # Router: add loop embedding, project to expert scores
        loop_emb = self.loop_embeddings.weight[loop_idx].view(1, 1, D)
        router_input = x + loop_emb
        router_logits = self.router(router_input.reshape(N, D))  # (N, E)

        # The routing math below operates on a per-expert "probability-like"
        # view of three router states — raw (pre-bias, no Gumbel), clean (bias,
        # no Gumbel) and the noisy selection path (bias + Gumbel). Two affinity
        # transforms are supported and produce these three views differently:
        #
        #   "softmax"       — historical Mixtral/v5 path. The balance bias is
        #                     added to the LOGITS pre-softmax; each view is a
        #                     softmax over its logits. Bit-identical to before.
        #   "sqrt_softplus" — DeepSeek-V4. affinity = sqrt(softplus(logits)) is
        #                     always non-negative; the balance bias is added to
        #                     the AFFINITY (not the logits); top-k selection and
        #                     the renormalised gating weights operate on that
        #                     balanced affinity. Telemetry and the Switch balance
        #                     loss consume an affinity-normalised probability
        #                     view (affinity / affinity.sum) so every downstream
        #                     entropy/fraction/z-loss stays well-defined.
        affinity_mode = self.router_affinity
        if affinity_mode == "sqrt_softplus":
            # Non-negative per-expert affinity. softplus keeps it smooth and
            # strictly positive; sqrt compresses the tail (DeepSeek-V4).
            affinity = torch.sqrt(F.softplus(router_logits))  # (N, E)
            raw_router_probs = affinity / affinity.sum(
                dim=-1, keepdim=True,
            ).clamp_min(1e-9)
            if self.bias_enabled:
                loop_bias = self.router_balance_bias[loop_idx].view(1, -1)
                clean_affinity = affinity + loop_bias
            else:
                clean_affinity = affinity
            # The aux-loss-free balance bias can push an affinity negative;
            # clamp at 0 so the normalised view and the gating weights stay
            # non-negative (the bias still steers top-k selection via ordering).
            clean_affinity = clean_affinity.clamp_min(0.0)
            clean_probs = clean_affinity / clean_affinity.sum(
                dim=-1, keepdim=True,
            ).clamp_min(1e-9)

            # Training-time noisy exploration: Gumbel is added to the balanced
            # AFFINITY (not logits) so cold experts still get explored under the
            # affinity transform. Annealed to 0 by the trainer before eval.
            selection_affinity = clean_affinity
            if self.training:
                u = torch.rand_like(clean_affinity).clamp_(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(u))
                tau = cast(Tensor, self.gumbel_tau).to(
                    dtype=clean_affinity.dtype,
                )
                selection_affinity = (clean_affinity + tau * gumbel).clamp_min(
                    0.0,
                )
            probs = selection_affinity / selection_affinity.sum(
                dim=-1, keepdim=True,
            ).clamp_min(1e-9)
        else:
            raw_router_probs = F.softmax(router_logits, dim=-1)
            if self.bias_enabled:
                loop_bias = self.router_balance_bias[loop_idx].view(1, -1)
                clean_logits = router_logits + loop_bias
            else:
                clean_logits = router_logits

            # "Clean" means deterministic deployed routing: bias applied, no
            # Gumbel. Raw un-biased logits are diagnostic only once the
            # controller is enabled.
            clean_probs = F.softmax(clean_logits, dim=-1)  # (N, E)

            # Training-time noisy top-k exploration. This prevents experts that
            # lose the first few router updates from going permanently cold. The
            # trainer anneals gumbel_tau to zero before the 5k health gate, so
            # the final pass is evaluated on the clean router.
            selection_logits = clean_logits
            if self.training:
                u = torch.rand_like(clean_logits).clamp_(1e-6, 1.0 - 1e-6)
                gumbel = -torch.log(-torch.log(u))
                tau = cast(Tensor, self.gumbel_tau).to(dtype=clean_logits.dtype)
                selection_logits = clean_logits + tau * gumbel

            # Softmax probabilities
            probs = F.softmax(selection_logits, dim=-1)  # (N, E)

        # Top-k selection (raw probs, before renormalisation)
        raw_top_probs, top_idx = probs.topk(self.top_k, dim=-1)  # (N, K)
        clean_raw_top_probs, clean_top_idx = clean_probs.topk(
            self.top_k, dim=-1,
        )
        prebias_raw_top_probs, raw_balance_top_idx = raw_router_probs.topk(
            self.top_k, dim=-1,
        )
        if self.training and self.balance_accum_enabled:
            self._accumulate_balance_counts(clean_top_idx, loop_idx)

        # Renormalise so the K chosen gates sum to 1. Without this, the MoE
        # output would be down-weighted when K > 1 just because softmax is
        # spread across E>K experts. Renormalisation keeps the MoE branch
        # at a consistent magnitude regardless of K.
        top_probs = raw_top_probs / raw_top_probs.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-9)

        # Per-expert capacity. In training, enforce the cap to force
        # balancing pressure. In eval/inference, disable drops entirely so
        # generation is chunk-stable (prefill-then-decode must match a full
        # forward). Inference non-determinism across chunks was a documented
        # v4 failure mode — v5 makes eval drop-free by construction.
        if self.training:
            capacity = max(
                1,
                int(math.ceil(self.capacity_factor * self.top_k * N / self.num_routed)),
            )
        else:
            capacity = N * self.top_k  # effectively unlimited (one pair per slot)

        # Dispatch/noisy assignment stats. These keep the existing telemetry
        # semantics: last_expert_fraction and marginal_entropy describe the
        # actual noisy dispatch path while Gumbel exploration is enabled.
        dispatch_one_hot = F.one_hot(top_idx, num_classes=self.num_routed)
        f = dispatch_one_hot.float().sum(dim=(0, 1)) / (N * self.top_k)
        p = probs.float().mean(dim=0)

        # Switch balance loss extended to top-k. Compute it on the RAW router
        # logits, not the noisy dispatch path or the bias-corrected clean path.
        # Gumbel is exploration and bias is an external controller; the aux
        # gradient must still push the learned router itself away from collapse.
        # Dispatch below still uses bias+Gumbel top_idx/probs.
        #   f_i = fraction of token-expert pairs routed to expert i.
        #         Count each top-k membership, divide by N*K so sum(f)=1.
        #   p_i = mean softmax prob for expert i (sums to 1).
        #   loss = E * sum(f_i * p_i). Minimum at uniform = 1.0.
        raw_balance_one_hot = F.one_hot(
            raw_balance_top_idx, num_classes=self.num_routed,
        )
        # Compute balance loss in fp32. Under bf16 autocast, f·p can
        # underflow late in training when both are near 1/E (= 0.125 for
        # E=8); fp32 keeps the product and sum precise so the gradient
        # signal survives into long runs.
        raw_balance_f = (
            raw_balance_one_hot.float().sum(dim=(0, 1)) / (N * self.top_k)
        )
        raw_balance_p = raw_router_probs.float().mean(dim=0)
        self.balance_loss = self.num_routed * (
            raw_balance_f * raw_balance_p
        ).sum()

        # Router Z-loss (ST-MoE §3.2): mean_token (logsumexp(logits))^2.
        # Bounds the absolute magnitude of router logits so bf16/fp8
        # softmax exponentials don't overflow, and keeps early softmax
        # distributions flatter so cold experts retain non-zero gradient
        # through LR warmup. Computed on raw router logits (pre-bias,
        # pre-Gumbel) so the penalty acts on the learned router itself.
        # fp32 for the same precision reasons as balance_loss above.
        z = torch.logsumexp(router_logits.float(), dim=-1)  # (N,)
        self.z_loss = (z ** 2).mean()

        # Sequence-wise balance loss (DeepSeek-V3 §5.2). Penalises
        # imbalance INSIDE each individual sequence, complementing the
        # global balance_loss above. Useful at long context (Phase 3
        # seq_len=8192) where one document can dominate one micro-batch
        # even when the global batch averages to balanced. Computed
        # under no_grad would defeat the purpose — we want the per-seq
        # gradient to push the router away from intra-sequence collapse.
        # Uses the same raw (un-noised) routing decisions as
        # balance_loss for a coherent gradient signal.
        seq_one_hot = raw_balance_one_hot.float().view(
            B, S, self.top_k, self.num_routed,
        )
        f_seq = seq_one_hot.sum(dim=(1, 2)) / (S * self.top_k)  # (B, E)
        p_seq = raw_router_probs.float().view(B, S, self.num_routed).mean(
            dim=1,
        )                                                       # (B, E)
        self.seq_balance_loss = self.num_routed * (
            f_seq * p_seq
        ).sum(dim=-1).mean()

        # Dispatch: for each expert, gather every token that picked it at
        # ANY top-k rank, apply capacity, run expert, scatter-add into output
        # with the renormalised gate weights.
        moe_out = torch.zeros_like(x_flat)
        total_dropped = 0

        for ei, expert in enumerate(self.experts):
            # Where (token_idx, rank) pairs where this expert is chosen
            is_chosen = (top_idx == ei)  # (N, K), bool
            token_indices, rank_indices = is_chosen.nonzero(as_tuple=True)  # both (T,)

            if token_indices.numel() == 0:
                continue

            # Apply capacity limit. `nonzero()` returns indices in
            # token-major order, so a naive `[:capacity]` always drops
            # the tail of the sequence and keeps prefix positions —
            # under expert collapse or long-context bursts this trains
            # the model to ignore late positions. Shuffle the (token,
            # rank) pairs before truncating so every position has equal
            # survival probability when an expert overflows. In eval
            # mode capacity == N*K so this branch never triggers.
            if token_indices.numel() > capacity:
                total_dropped += (token_indices.numel() - capacity)
                perm = torch.randperm(
                    token_indices.numel(), device=token_indices.device,
                )
                keep = perm[:capacity]
                token_indices = token_indices[keep]
                rank_indices = rank_indices[keep]

            # Run expert on selected tokens (one forward per expert per batch)
            expert_input = x_flat[token_indices]  # (T, D)
            expert_output = expert(expert_input)   # (T, D)

            # Gate = renormalised softmax prob for this (token, rank) pair.
            # Router gets gradient through this gate.
            gates = top_probs[token_indices, rank_indices].unsqueeze(-1)  # (T, 1)

            # Scatter-add: a token may contribute from multiple experts
            # (different ranks), so use index_add, not direct assignment.
            moe_out.index_add_(0, token_indices, expert_output * gates)

        moe_out = moe_out.view(B, S, D)

        # Telemetry (detached, CPU-side scalars).
        # These fixes address v4 misdiagnosis:
        #   - "router_entropy" was the entropy of the batch-marginal p vector,
        #     which stays at ln(E) for any well-balanced router even when
        #     per-token routing is razor-sharp. Rename to marginal_entropy
        #     and add per_token_entropy as the real sharpness signal.
        #   - max_prob was reported AFTER top-k renormalisation, so a
        #     uniform top-2 router showed max_prob = 0.5 not 1/E. Report raw.
        #   - Add top_margin = raw top-1 prob - raw top-2 prob, which
        #     directly measures router confidence in its primary pick.
        with torch.no_grad():
            # Per-token entropy — the real sharpness metric. Uniform per-token
            # softmax => ln(E). Sharp routing => much lower. Average over tokens.
            log_probs = torch.log(probs.clamp_min(1e-10))
            per_token_ent = -(probs * log_probs).sum(dim=-1)  # (N,)
            per_token_ent_mean = per_token_ent.mean().item()

            # Marginal entropy (entropy of mean_token p). High = balanced,
            # low = some experts globally never picked. Keep as balance proxy.
            p_log = torch.log(p.clamp_min(1e-10))
            marginal_ent = -(p * p_log).sum().item()

            # Assignment entropy (hard f). Mirrors marginal but over hard picks.
            f_log = torch.log(f.clamp_min(1e-10))
            assign_ent = -(f * f_log).sum().item()

            # Drop rate (across all N*K dispatch opportunities). 0 at inference.
            drop_rate = total_dropped / max(N * self.top_k, 1)

            # Raw top-1 router confidence (before renormalisation).
            # Uniform router gives 1/E; preferences show as > 1/E.
            raw_max = raw_top_probs[:, 0].mean().item()

            # Top-1 vs top-2 margin (raw probs). Large = strong primary pick.
            if self.top_k >= 2:
                top_margin = (raw_top_probs[:, 0] - raw_top_probs[:, 1]).mean().item()
            else:
                top_margin = raw_top_probs[:, 0].mean().item()

            self.last_per_token_entropy[loop_idx] = per_token_ent_mean
            self.last_marginal_entropy[loop_idx] = marginal_ent
            self.last_assignment_entropy[loop_idx] = assign_ent
            self.last_expert_fraction[loop_idx] = f.tolist()
            self.last_drop_rate[loop_idx] = drop_rate
            self.last_raw_max_prob[loop_idx] = raw_max
            self.last_top_margin[loop_idx] = top_margin

            prebias_log_probs = torch.log(raw_router_probs.clamp_min(1e-10))
            prebias_per_token_ent = -(
                raw_router_probs * prebias_log_probs
            ).sum(dim=-1)
            prebias_p = raw_router_probs.float().mean(dim=0)
            prebias_p_log = torch.log(prebias_p.clamp_min(1e-10))
            prebias_marginal_ent = -(prebias_p * prebias_p_log).sum().item()
            prebias_one_hot = F.one_hot(
                raw_balance_top_idx, num_classes=self.num_routed,
            ).to(raw_router_probs.dtype)
            prebias_f = prebias_one_hot.sum(dim=(0, 1)) / (N * self.top_k)
            prebias_f_log = torch.log(prebias_f.clamp_min(1e-10))
            prebias_assign_ent = -(prebias_f * prebias_f_log).sum().item()
            prebias_raw_max = prebias_raw_top_probs[:, 0].mean().item()
            if self.top_k >= 2:
                prebias_top_margin = (
                    prebias_raw_top_probs[:, 0]
                    - prebias_raw_top_probs[:, 1]
                ).mean().item()
            else:
                prebias_top_margin = prebias_raw_top_probs[:, 0].mean().item()

            self.last_prebias_per_token_entropy[loop_idx] = (
                prebias_per_token_ent.mean().item()
            )
            self.last_prebias_marginal_entropy[loop_idx] = prebias_marginal_ent
            self.last_prebias_assignment_entropy[loop_idx] = prebias_assign_ent
            self.last_prebias_expert_fraction[loop_idx] = prebias_f.tolist()
            self.last_prebias_raw_max_prob[loop_idx] = prebias_raw_max
            self.last_prebias_top_margin[loop_idx] = prebias_top_margin

            clean_log_probs = torch.log(clean_probs.clamp_min(1e-10))
            clean_per_token_ent = -(clean_probs * clean_log_probs).sum(dim=-1)
            clean_p = clean_probs.mean(dim=0)
            clean_p_log = torch.log(clean_p.clamp_min(1e-10))
            clean_marginal_ent = -(clean_p * clean_p_log).sum().item()
            clean_one_hot = F.one_hot(
                clean_top_idx, num_classes=self.num_routed,
            ).to(clean_probs.dtype)
            clean_f = clean_one_hot.sum(dim=(0, 1)) / (N * self.top_k)
            clean_f_log = torch.log(clean_f.clamp_min(1e-10))
            clean_assign_ent = -(clean_f * clean_f_log).sum().item()
            clean_raw_max = clean_raw_top_probs[:, 0].mean().item()
            if self.top_k >= 2:
                clean_top_margin = (
                    clean_raw_top_probs[:, 0]
                    - clean_raw_top_probs[:, 1]
                ).mean().item()
            else:
                clean_top_margin = clean_raw_top_probs[:, 0].mean().item()

            self.last_clean_per_token_entropy[loop_idx] = (
                clean_per_token_ent.mean().item()
            )
            self.last_clean_marginal_entropy[loop_idx] = clean_marginal_ent
            self.last_clean_assignment_entropy[loop_idx] = clean_assign_ent
            self.last_clean_expert_fraction[loop_idx] = clean_f.tolist()
            self.last_clean_raw_max_prob[loop_idx] = clean_raw_max
            self.last_clean_top_margin[loop_idx] = clean_top_margin

        # Return (shared, routed) so the Block can apply moe_gate only to
        # the routed contribution. Shared expert stays at full weight.
        return shared_out, moe_out


# ── Recursive Block ─────────────────────────────────────────────────────


@contextmanager
def _balance_accumulation(moe: MoELayer, enabled: bool):
    previous = moe.balance_accum_enabled
    moe.balance_accum_enabled = enabled
    try:
        yield
    finally:
        moe.balance_accum_enabled = previous


@torch.compiler.disable
def _checkpoint_block(block_fn, *args, context_fn):
    # Dynamo's higher-order-op tracer raises NotImplementedError on
    # checkpoint(..., context_fn=...). Wrapping the call in
    # torch.compiler.disable forces an eager fallback for just the
    # checkpoint dispatch; the block_fn itself is still compiled when the
    # outer model is wrapped in torch.compile because the inner call
    # re-enters the compiled graph.
    return gradient_checkpoint(
        block_fn, *args, use_reentrant=False, context_fn=context_fn,
    )


class RecursiveBlock(nn.Module):
    """Physical transformer block: attention + MoE (no dense FFN).

    FFN path is:
        shared_expert(x)
        + moe_gate * sum_{(t, k) in top_k} gate_{t,k} * expert_{top_idx[t,k]}(x_t)
    wrapped inside MoELayer. Top-k gates are renormalised so they sum to 1
    per token, and tokens exceeding per-expert capacity are dropped from
    that expert's branch (training only; disabled at inference).
    No parallel dense path — shared expert replaces it.
    """

    def __init__(self, config: OSRTConfig, block_idx: int = 0) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.head_dim
        self.kv_heads = config.num_kv_heads
        self.kv_dim = self.kv_heads * self.head_dim
        self.group_size = self.heads // self.kv_heads

        # Attention — GQA with an MLA-style compressed K/V latent.
        #   q_proj:    full query projection (heads × head_dim)
        #   kv_down:   compress the hidden state to a single latent of
        #              kv_dim (= kv_heads × head_dim). THIS is the only
        #              thing cached — un-rotated.
        #   v_from_k:  derive V from that same latent (V = W·c + b). Both
        #              K and V are linear functions of one cached latent,
        #              the same expressivity class as DeepSeek MLA's shared
        #              c_KV, at ~half the cache of storing K and V.
        self.norm_attn = nn.RMSNorm(config.dim)
        self.q_proj = nn.Linear(config.dim, self.heads * self.head_dim, bias=False)
        self.kv_down = nn.Linear(config.dim, self.kv_dim, bias=False)
        self.v_from_k = nn.Linear(self.kv_dim, self.kv_dim, bias=True)
        # QK-Norm: per-head RMSNorm on q and k before RoPE+SDPA. Bounds
        # attention logits so they don't explode in bf16/fp8 — protects
        # the downstream MoE router from inheriting pathological hidden
        # states. Per-head (head_dim) is the standard formulation; sharing
        # the norm parameter across heads keeps the addition lightweight
        # (~head_dim params per block) and matches Gemma2/Chameleon.
        self.norm_q = nn.RMSNorm(config.head_dim)
        self.norm_k = nn.RMSNorm(config.head_dim)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)

        # Attention sink (ARCHITECTURE.md §6.6). Per-head learnable sink logits,
        # initialised to zeros (sink_logit=0 ⇒ the sink contributes exp(0)=1 to
        # the denominator). The sink adds an extra term to the softmax
        # denominator only — its "value" is zero, so it never enters the
        # numerator/output:
        #   s_{h,i,j} = exp(z_{h,i,j}) / (Σ_k exp(z_{h,i,k}) + exp(sink[h]))
        # This lets a query's weights sum to < 1 (a head can attend to
        # "nothing" when no key is relevant). The parameter is 1D (heads,) →
        # routed to AdamW by build_param_groups with no change there. None when
        # disabled so the standard SDPA path stays bit-identical.
        self.attention_sink = config.attention_sink
        if config.attention_sink:
            self.sink_logits = nn.Parameter(torch.zeros(self.heads))

        # MoE (shared + routed), pre-norm
        self.norm_moe = nn.RMSNorm(config.dim)
        self.moe = MoELayer(config, moe_seed=block_idx, block_idx=block_idx)

        # Gate on the MoE (routed) branch. Reparameterised through
        # softplus so the EFFECTIVE gate is always > 0:
        #   effective = softplus(moe_gate) = log(1 + exp(moe_gate))
        # The raw parameter is initialised to log(e - 1) ≈ 0.5413 so
        # softplus(raw) ≈ 1.0 at step 0 (matches the previous unbounded
        # 1.0 init). Without this constraint the scalar can drift
        # negative under task gradient and zero out the routed branch
        # entirely, recreating the v4 "dense crutch" failure mode where
        # the always-on shared expert does all the work and routed
        # experts receive no learning signal.
        # Read `effective_moe_gate()` (or compute F.softplus(moe_gate)
        # at use sites) to get the actual gate value.
        self.moe_gate = nn.Parameter(torch.tensor(math.log(math.e - 1.0)))

        # Manifold-Constrained Hyper-Connections (one per sub-block, shared
        # across loop iterations). None when mHC is disabled — the block then
        # uses the proven standard single-stream residual.
        self.use_mhc = config.use_mhc
        if config.use_mhc:
            self.mhc_attn = ManifoldHyperConnection(
                config.dim, config.n_hc, config.mhc_sinkhorn_iters,
            )
            self.mhc_ffn = ManifoldHyperConnection(
                config.dim, config.n_hc, config.mhc_sinkhorn_iters,
            )

    def effective_moe_gate(self) -> Tensor:
        return F.softplus(self.moe_gate)

    def _attention(
        self,
        x_in: Tensor,
        adapter_a: Tensor,
        adapter_b: Tensor,
        adapter_scale: float,
        rope_cos: Tensor,
        rope_sin: Tensor,
        past_key_value: Tensor | None,
        use_cache: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Attention sub-block contribution (pre-residual): GQA + MLA latent.

        Returns (out_proj(attn) + adapter, present_latent). The caller adds it
        into the residual (standard) or mixes it via mHC.
        """
        B, S, D = x_in.shape
        adapter_out = adapter_scale * (x_in @ adapter_a @ adapter_b)

        h = self.norm_attn(x_in)
        q = self.q_proj(h).view(B, S, self.heads, self.head_dim)
        c_kv_new = self.kv_down(h)            # (B, S, kv_dim) — un-rotated latent

        # The cache holds ONLY the un-rotated latent. K and V are recomputed
        # from the full latent every step: RoPE is positional and V-from-K
        # must operate on un-rotated K, so neither may be cached rotated.
        if past_key_value is not None:
            c_kv = torch.cat([past_key_value, c_kv_new], dim=1)  # (B, L+S, kv_dim)
        else:
            c_kv = c_kv_new
        present_kv = c_kv if use_cache else None
        total_len = c_kv.shape[1]
        past_len = total_len - S

        # Derive K and V from the same latent (both linear in c_kv).
        k = c_kv.view(B, total_len, self.kv_heads, self.head_dim)
        v = self.v_from_k(c_kv).view(B, total_len, self.kv_heads, self.head_dim)

        # QK-Norm before RoPE.
        q = self.norm_q(q)
        k = self.norm_k(k)
        # Queries rotate at the new positions [past_len:total_len]; keys were
        # just rebuilt un-rotated, so they rotate over the whole [0:total_len].
        q = apply_rope(
            q, rope_cos[:, past_len:total_len].to(q.dtype),
            rope_sin[:, past_len:total_len].to(q.dtype),
        )
        k = apply_rope(
            k, rope_cos[:, :total_len].to(k.dtype),
            rope_sin[:, :total_len].to(k.dtype),
        )
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # GQA attention: q has `heads`, k/v have `kv_heads`; enable_gqa lets
        # SDPA broadcast KV groups without materialising repeated heads.
        gqa = self.group_size > 1
        if self.attention_sink:
            # Attention sink (ARCHITECTURE.md §6.6): the sink adds an extra term
            # to the softmax denominator only, which SDPA cannot express. Use
            # the manual path so we can apply the exact log-sum-exp rescale.
            attn_out = self._attention_with_sink(
                q, k, v, S, total_len, past_len,
            )
        elif past_len > 0 and S > 1:
            attn_mask = torch.full(
                (S, total_len), float("-inf"), device=q.device, dtype=q.dtype,
            )
            attn_mask = torch.triu(attn_mask, diagonal=1 + past_len)
            attn_out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, enable_gqa=gqa,
            )
        else:
            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=(S > 1), enable_gqa=gqa,
            )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(attn_out) + adapter_out, present_kv

    def _attention_with_sink(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        S: int,
        total_len: int,
        past_len: int,
    ) -> Tensor:
        """Manual GQA attention with a per-head learnable sink (§6.6/§6.7).

        Exact log-sum-exp rescale of the standard attention output. If `out`
        and `lse = log Σ_k exp(z_{i,k})` are the usual (sink-free) attention
        output and per-query log-sum-exp of the scores, then adding exp(sink[h])
        to the denominator simply multiplies the output by
            Σexp(z) / (Σexp(z) + exp(sink)) = sigmoid(lse - sink[h]),
        because the sink's value is zero and so contributes nothing to the
        numerator. We therefore compute the masked scores, derive `out` and
        `lse` from one softmax/logsumexp, and rescale per head.

        flex_attention(return_lse=True) was investigated as the "flash + lse"
        route but on this target (torch 2.12, CPU) it (a) emits a deprecation
        warning for return_lse, (b) without torch.compile materialises the full
        score matrix anyway (no fused-kernel speedup), and (c) needs a custom
        mask_mod to express GQA broadcasting. The manual path below materialises
        the same score matrix, reuses the EXACT causal masking the SDPA path
        uses, and is fully correct; it is O(S·total_len) in memory per head —
        fine for our sequence lengths and the simplest thing that is provably
        right. Inputs q:(B,heads,S,hd), k/v:(B,kv_heads,total_len,hd).
        """
        B, H, _, hd = q.shape
        # Expand GQA groups so every query head sees its KV head. SDPA does this
        # internally via enable_gqa; here we do it explicitly with
        # repeat_interleave on the kv-head dim (groups of `group_size`).
        if self.group_size > 1:
            k = k.repeat_interleave(self.group_size, dim=1)
            v = v.repeat_interleave(self.group_size, dim=1)

        scale = 1.0 / math.sqrt(hd)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B,H,S,total_len)

        # Causal masking — identical semantics to the SDPA path. For the
        # cached-decode case the S query positions occupy [past_len:total_len],
        # so a key j is visible to query i iff j <= past_len + i. With S == 1
        # (single-token decode) all `total_len` keys are visible and no mask is
        # needed (matches SDPA's is_causal=False branch).
        if S > 1:
            row = torch.arange(S, device=scores.device).view(S, 1)
            col = torch.arange(total_len, device=scores.device).view(1, total_len)
            causal = col <= (past_len + row)  # (S, total_len) bool, True = keep
            scores = scores.masked_fill(~causal, float("-inf"))

        # Per-query log-sum-exp of the (masked) scores, then standard softmax.
        # Compute in fp32 for a stable exp/log; the sink rescale is sensitive to
        # the lse magnitude. lse: (B,H,S).
        scores_f = scores.float()
        lse = torch.logsumexp(scores_f, dim=-1)
        attn_weights = torch.softmax(scores_f, dim=-1).to(v.dtype)
        out = torch.matmul(attn_weights, v)  # (B,H,S,hd)

        # Sink rescale: multiply each head's output by sigmoid(lse - sink[h]).
        # sink_logits: (H,) → broadcast over (B,H,S). Done in fp32 then cast.
        sink = self.sink_logits.float().view(1, H, 1)
        rescale = torch.sigmoid(lse - sink).unsqueeze(-1).to(out.dtype)
        return out * rescale

    def _moe(
        self, x_in: Tensor, loop_idx: int, token_ids: Tensor | None = None,
    ) -> Tensor:
        """MoE sub-block contribution (pre-residual): shared + gated routed.

        token_ids (B, S) is forwarded to the MoE layer; it is only consumed when
        this block hash-routes, otherwise ignored."""
        h_shared, h_routed = self.moe(
            self.norm_moe(x_in), loop_idx, token_ids=token_ids,
        )
        return h_shared + self.effective_moe_gate() * h_routed

    def forward(
        self,
        x: Tensor,
        adapter_a: Tensor,
        adapter_b: Tensor,
        adapter_scale: float,
        rope_cos: Tensor,
        rope_sin: Tensor,
        loop_idx: int,
        past_key_value: Tensor | None = None,
        use_cache: bool = False,
        token_ids: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        """Run attention then MoE. `x` is (B, S, D) for the standard residual
        path, or (B, S, n_hc, D) when mHC is enabled.

        token_ids (B, S) is the optional input-token tensor used only by
        hash-routing blocks; it defaults to None and is otherwise ignored."""
        if self.use_mhc:
            # ── mHC path: residual stream carries n_hc channels ──
            a, b_mat, c_out = self.mhc_attn.generate(x)
            x_in = self.mhc_attn.input_view(x, a)
            f_attn, present_kv = self._attention(
                x_in, adapter_a, adapter_b, adapter_scale,
                rope_cos, rope_sin, past_key_value, use_cache,
            )
            x = self.mhc_attn.update(x, b_mat, c_out, f_attn)

            a2, b2, c2 = self.mhc_ffn.generate(x)
            x_in2 = self.mhc_ffn.input_view(x, a2)
            f_moe = self._moe(x_in2, loop_idx, token_ids=token_ids)
            x = self.mhc_ffn.update(x, b2, c2, f_moe)
            return x, present_kv

        # ── Standard residual path ──
        f_attn, present_kv = self._attention(
            x, adapter_a, adapter_b, adapter_scale,
            rope_cos, rope_sin, past_key_value, use_cache,
        )
        x = x + f_attn
        x = x + self._moe(x, loop_idx, token_ids=token_ids)
        return x, present_kv


# ── Main Model ──────────────────────────────────────────────────────────


class OSRTPreTrainedModel(PreTrainedModel):
    """Base class for OSRT models."""

    config_class = OSRTConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            custom_std = getattr(module, "_osrt_init_std", None)
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=custom_std if custom_std is not None else std,
            )


class OSRTModel(OSRTPreTrainedModel):
    """Core OSRT model (without LM head)."""

    def __init__(self, config: OSRTConfig) -> None:
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.dim)

        cos, sin = compute_rope_freqs(
            config.max_position_embeddings,
            config.head_dim,
            config.rope_theta,
            scaling=config.rope_scaling,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Physical blocks with distinct block-idx seeds so experts differ
        # across blocks too (not just within a block).
        self.blocks = nn.ModuleList(
            [RecursiveBlock(config, block_idx=bi) for bi in range(config.num_blocks)]
        )

        # Per-pass low-rank adapters
        total_pairs = config.num_blocks * config.recursive_loops
        self.adapters_a = nn.ParameterList(
            [nn.Parameter(torch.randn(config.dim, config.adapter_rank) * 0.01)
             for _ in range(total_pairs)]
        )
        self.adapters_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(config.adapter_rank, config.dim))
             for _ in range(total_pairs)]
        )
        self.adapter_scale = config.adapter_alpha / config.adapter_rank

        self.norm_loop = nn.RMSNorm(config.dim)
        self.norm_out = nn.RMSNorm(config.dim)

        # mHC: dedicated learnable collapse head mixing the n_hc residual
        # channels back to a single d_model vector for the LM head and for
        # every per-loop aux-head capture (resolves ARCHITECTURE.md §10 Bug 3,
        # the stale-A_l collapse). Initialized to a uniform channel average.
        self.use_mhc = config.use_mhc
        if config.use_mhc:
            self.mhc_collapse = nn.Parameter(
                torch.full((config.n_hc,), 1.0 / config.n_hc)
            )

        self.gradient_checkpointing = False

        # Side-effect storage for per-loop auxiliary LM-head losses.
        # Populated by forward() when aux_loop_loss_weight > 0 and the
        # model is in training mode. Consumed by OSRTForCausalLM
        # to compute the aux loss term, and by the train loop for
        # per-loop logging.
        self.last_intermediate_hiddens: list[Tensor] | None = None

    def _collapse(self, X: Tensor) -> Tensor:
        """mHC: mix the n_hc residual channels into one d_model vector via the
        dedicated learnable collapse head. X: (B, S, n_hc, D) -> (B, S, D)."""
        return torch.einsum("c,bscd->bsd", self.mhc_collapse, X)

    def forward(
        self,
        input_ids: Tensor,
        past_key_values: list[Tensor] | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[
        Tensor, list[Tensor], Tensor, Tensor, Tensor,
        list[Tensor] | None,
    ]:
        """Forward pass.

        Returns:
            (hidden, loop_rms, balance_loss, z_loss, seq_balance_loss, presents)
        Each loss is the SUM across all (num_blocks * recursive_loops) MoE
        applications. The wrapper normalises by that count.
        """
        x = self.embedding(input_ids)
        if self.use_mhc:
            # Expand to the n_hc-channel residual stream. .repeat (not
            # .expand) so the channels are independent storage — §10 Bug 1.
            x = x.unsqueeze(2).repeat(1, 1, self.config.n_hc, 1)
        S = input_ids.shape[1]
        expected_past_layers = self.config.num_blocks * self.config.recursive_loops

        past_length = 0
        if past_key_values is not None:
            if len(past_key_values) != expected_past_layers:
                raise ValueError(
                    f"Invalid past_key_values: expected "
                    f"{expected_past_layers} entries, "
                    f"got {len(past_key_values)}."
                )
            for idx, layer_past in enumerate(past_key_values):
                if layer_past is None:
                    continue
                if not isinstance(layer_past, torch.Tensor):
                    raise ValueError(
                        f"past_key_values[{idx}] must be a latent Tensor "
                        f"(B, seq, kv_dim)."
                    )
                layer_len = layer_past.shape[1]
                if past_length == 0:
                    past_length = layer_len
                elif layer_len != past_length:
                    raise ValueError(
                        f"KV cache length mismatch at layer {idx}."
                    )

        required_seq_len = past_length + S
        # Pass the FULL-range [0:required_seq_len] cos/sin to each block: K is
        # rebuilt un-rotated from the cached latent and must be rotated over
        # the whole span, while Q is rotated only at its new positions. The
        # block slices each internally from past_len.
        if required_seq_len <= self.rope_cos.shape[1]:
            cos = self.rope_cos[:, :required_seq_len, :, :]
            sin = self.rope_sin[:, :required_seq_len, :, :]
        else:
            rope_cos, rope_sin = compute_rope_freqs(
                required_seq_len,
                self.rope_cos.shape[-1],
                theta=getattr(self.config, "rope_theta", 10000.0),
                device=x.device,
                scaling=getattr(self.config, "rope_scaling", None),
            )
            cos = rope_cos[:, :required_seq_len, :, :].to(
                device=self.rope_cos.device, dtype=self.rope_cos.dtype
            )
            sin = rope_sin[:, :required_seq_len, :, :].to(
                device=self.rope_sin.device, dtype=self.rope_sin.dtype
            )

        loop_rms: list[Tensor] = []
        total_balance_loss = torch.tensor(0.0, device=x.device)
        total_z_loss = torch.tensor(0.0, device=x.device)
        total_seq_balance_loss = torch.tensor(0.0, device=x.device)

        # Per-loop aux-loss capture (architecture fix for loop collapse).
        # Enabled when config.aux_loop_loss_weight > 0 and training. We
        # capture the hidden state at the END of each non-final loop
        # (after the 3 blocks, BEFORE norm_loop) so the aux LM head can
        # apply norm_out to it and predict the next token from the
        # intermediate representation. Forces gradient signal into
        # loops 0..N-2 instead of letting loop N-1 absorb everything.
        intermediate_hiddens: list[Tensor] = []
        capture_aux = (
            getattr(self.config, "aux_loop_loss_weight", 0.0) > 0.0
            and self.training
        )

        # Loop dropout (stochastic depth). With probability
        # loop_dropout_prob in training, truncate the loop chain to a
        # random length in [min_loops, recursive_loops]. The main task
        # loss then flows from a shorter chain's final output, forcing
        # the truncation point loop to be standalone-useful. Complements
        # the aux loss — aux pushes intermediate loops to predict in
        # parallel, dropout makes their predictions become the actual
        # model output some fraction of the time.
        n_loops_to_run = self.config.recursive_loops
        if (
            self.training
            and getattr(self.config, "loop_dropout_prob", 0.0) > 0.0
            and random.random() < self.config.loop_dropout_prob
        ):
            min_loops = max(2, getattr(self.config, "loop_dropout_min_loops", 3))
            max_loops = self.config.recursive_loops
            if max_loops > min_loops:
                n_loops_to_run = random.randint(min_loops, max_loops)

        use_ckpt = self.gradient_checkpointing and self.training
        if use_ckpt and (use_cache or past_key_values is not None):
            raise ValueError(
                "KV caching is incompatible with gradient checkpointing."
            )
        presents: list[Tensor] | None = [] if use_cache else None

        for loop in range(n_loops_to_run):
            for block_idx, block in enumerate(self.blocks):
                idx = loop * self.config.num_blocks + block_idx
                adapter_a = self.adapters_a[idx]
                adapter_b = self.adapters_b[idx]
                layer_past = (
                    past_key_values[idx] if past_key_values is not None else None
                )

                if use_ckpt:
                    def _block_fn(
                        _x, _a, _b, _cos, _sin,
                        _block=block, _scale=self.adapter_scale, _loop=loop,
                        _tok=input_ids,
                    ):
                        # token_ids is captured (closure default), not a
                        # checkpoint input — it carries no gradient and only
                        # hash-routing blocks read it.
                        return _block(
                            _x, _a, _b, _scale, _cos, _sin, _loop,
                            token_ids=_tok,
                        )[0]

                    def _context_fn(_block=block):
                        return (
                            _balance_accumulation(_block.moe, True),
                            _balance_accumulation(_block.moe, False),
                        )

                    x = _checkpoint_block(
                        _block_fn, x, adapter_a, adapter_b, cos, sin,
                        context_fn=_context_fn,
                    )
                else:
                    x, present_kv = block(
                        x, adapter_a, adapter_b,
                        self.adapter_scale, cos, sin,
                        loop_idx=loop,
                        past_key_value=layer_past,
                        use_cache=use_cache,
                        token_ids=input_ids,
                    )
                    if presents is not None:
                        presents.append(present_kv)

                # Accumulate router auxiliary losses (Switch balance,
                # Z-loss, sequence-wise balance). Each is set as a
                # side-effect on the MoE module during forward; the
                # wrapper normalises by the number of MoE applications.
                if block.moe.balance_loss is not None:
                    total_balance_loss = total_balance_loss + block.moe.balance_loss
                if block.moe.z_loss is not None:
                    total_z_loss = total_z_loss + block.moe.z_loss
                if block.moe.seq_balance_loss is not None:
                    total_seq_balance_loss = (
                        total_seq_balance_loss + block.moe.seq_balance_loss
                    )

            # Capture pre-norm_loop hidden for aux-loss computation.
            # Only capture non-final loops of THIS forward pass — under
            # loop dropout, the "final" loop varies per batch. The
            # hidden state at position n_loops_to_run - 1 will feed
            # the main LM head, so no aux for it.
            if capture_aux and loop < n_loops_to_run - 1:
                # Collapse the mHC stream to a single vector for the aux head
                # (dedicated learnable collapse — never a stale dynamic A_l).
                intermediate_hiddens.append(self._collapse(x) if self.use_mhc else x)

            loop_rms.append(x.float().pow(2).mean().sqrt())
            if loop < n_loops_to_run - 1:
                x = self.norm_loop(x)

        if self.use_mhc:
            x = self._collapse(x)
        x = self.norm_out(x)
        # Expose intermediate hiddens to the CausalLM wrapper. Set to
        # None when not capturing so downstream code can do a cheap
        # truthy check.
        self.last_intermediate_hiddens = (
            intermediate_hiddens if capture_aux else None
        )
        return (
            x, loop_rms,
            total_balance_loss, total_z_loss, total_seq_balance_loss,
            presents,
        )


class OSRTForCausalLM(OSRTPreTrainedModel):
    """OSRT with causal LM head. HF-compatible.

    LM head is weight-tied to embeddings (via F.linear with embedding.weight).
    Saves ~50M params vs untied for 32K×1536 embedding. Matches v4.
    """

    def __init__(self, config: OSRTConfig) -> None:
        super().__init__(config)
        self.model = OSRTModel(config)
        # HF's post_init walks all nn.Linear and calls _init_weights on them,
        # which would overwrite any orthogonal init done in MoELayer.__init__.
        self.post_init()
        # Apply orthogonal per-expert init AFTER post_init so it survives.
        for block in self.model.blocks:
            block.moe.apply_orthogonal_init()

        # Last-forward loss components (for training-loop logging).
        # These are plain tensors, set during forward. The training loop
        # reads them directly instead of us extending the HF ModelOutput.
        self.last_task_loss: Tensor | None = None
        self.last_balance_loss: Tensor | None = None
        self.last_balance_loss_normalised: Tensor | None = None
        self.last_z_loss: Tensor | None = None
        self.last_z_loss_normalised: Tensor | None = None
        self.last_seq_balance_loss: Tensor | None = None
        self.last_seq_balance_loss_normalised: Tensor | None = None
        # Per-loop aux losses (when aux_loop_loss_weight > 0 + training).
        # Indexed loop 0..N-2 (final loop has no aux). Each entry is the
        # raw CE loss for predicting next-token from that loop's hidden.
        self.last_per_loop_aux_losses: list[Tensor] = []
        self.last_aux_loop_total: Tensor | None = None

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embedding

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embedding = value

    def forward(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
        past_key_values: list[Tensor | None] | None = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        (
            hidden, loop_rms,
            balance_loss, z_loss, seq_balance_loss,
            presents,
        ) = self.model(
            input_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
        )

        # Weight-tied LM head
        logits = F.linear(hidden, self.model.embedding.weight)

        # Reset loss attributes (prevent stale values in eval-without-labels)
        self.last_task_loss = None
        self.last_balance_loss = None
        self.last_balance_loss_normalised = None
        self.last_z_loss = None
        self.last_z_loss_normalised = None
        self.last_seq_balance_loss = None
        self.last_seq_balance_loss_normalised = None
        self.last_per_loop_aux_losses = []
        self.last_aux_loop_total = None

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :self.config.real_vocab_size]
            shift_logits = shift_logits.contiguous().float()
            shift_labels = labels[..., 1:].contiguous()
            task_loss = F.cross_entropy(
                shift_logits.view(-1, self.config.real_vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Normalise each aux loss by num MoE applications so the
            # coefficient matches per-layer weight (not per-whole-model sum).
            n_moe_layers = self.config.num_blocks * self.config.recursive_loops
            balance_norm = balance_loss / n_moe_layers
            z_norm = z_loss / n_moe_layers
            seq_balance_norm = seq_balance_loss / n_moe_layers

            # Per-loop aux LM-head losses (architecture fix). Captures the
            # hidden state at the END of each non-final loop, applies
            # norm_out + the (weight-tied) LM head, and computes CE
            # against the same shifted labels. Adds gradient signal to
            # intermediate loops 0..N-2.
            #
            # Per-loop weighting: if config.per_loop_aux_weights is set
            # (list of length n_intermediate_loops), each loop's CE is
            # multiplied by its weight before summing. Otherwise all
            # intermediate losses use the uniform aux_loop_loss_weight.
            aux_loop_total = torch.tensor(0.0, device=task_loss.device)
            per_loop_aux: list[Tensor] = []
            intermediate_hiddens = self.model.last_intermediate_hiddens
            aux_weight = getattr(self.config, "aux_loop_loss_weight", 0.0)
            per_loop_weights = getattr(
                self.config, "per_loop_aux_weights", None,
            )
            if (
                self.training
                and aux_weight > 0.0
                and intermediate_hiddens
            ):
                for i, h_loop in enumerate(intermediate_hiddens):
                    h_norm = self.model.norm_out(h_loop)
                    h_logits = F.linear(h_norm, self.model.embedding.weight)
                    h_shift = h_logits[
                        ..., :-1, :self.config.real_vocab_size
                    ].contiguous().float()
                    aux_l = F.cross_entropy(
                        h_shift.view(-1, self.config.real_vocab_size),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )
                    per_loop_aux.append(aux_l)
                    # Per-loop scaling. When per_loop_weights is set,
                    # use it (must be at least as long as the captured
                    # intermediates); else uniform weight 1.0 (then
                    # the overall aux_weight multiplies the sum).
                    if (
                        per_loop_weights is not None
                        and i < len(per_loop_weights)
                    ):
                        w = per_loop_weights[i]
                    else:
                        w = 1.0
                    aux_loop_total = aux_loop_total + w * aux_l

            # Total loss: add aux losses ONLY during training. Eval loss must
            # be pure task CE so eval perplexity and held-out comparisons
            # aren't polluted by hyperparameter choices. Training loops that
            # want the aux signals at eval time should read the
            # last_*_normalised attributes instead.
            if self.training:
                loss = (
                    task_loss
                    + self.config.router_aux_loss_coeff * balance_norm
                    + self.config.router_z_loss_coeff * z_norm
                    + self.config.router_seq_balance_loss_coeff
                    * seq_balance_norm
                    + aux_weight * aux_loop_total
                )
            else:
                loss = task_loss

            # Stash per-loop aux losses (detached) for telemetry.
            self.last_per_loop_aux_losses = [l.detach() for l in per_loop_aux]
            self.last_aux_loop_total = (
                aux_loop_total.detach() if per_loop_aux else None
            )

            # Expose components for logging — always set, regardless of mode.
            self.last_task_loss = task_loss.detach()
            self.last_balance_loss = balance_loss.detach()
            self.last_balance_loss_normalised = balance_norm.detach()
            self.last_z_loss = z_loss.detach()
            self.last_z_loss_normalised = z_norm.detach()
            self.last_seq_balance_loss = seq_balance_loss.detach()
            self.last_seq_balance_loss_normalised = seq_balance_norm.detach()

        # Cast to FloatTensor to satisfy HF's type stubs. The runtime types
        # are correct — logits comes from F.linear on a float hidden state,
        # and loss is a scalar float tensor from cross_entropy.
        return CausalLMOutputWithPast(
            loss=cast("torch.FloatTensor | None", loss),
            logits=cast("torch.FloatTensor", logits),
            past_key_values=presents,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 0,
        eos_token_id: int | None = None,
        stop_token_ids: list[int] | None = None,
        repetition_penalty: float = 1.0,
        **kwargs,
    ) -> Tensor:
        """Autoregressive generation with KV cache.

        The model's forward already supports past_key_values + use_cache
        (per-effective-layer KV cache, 18 layers for default v5). This
        method does the prefill + decode loop: one full forward over the
        prompt to seed the cache, then one single-token forward per step
        consuming the cache. That turns O(N) per-step attention cost into
        O(1) and is ~3x faster than the non-cached path for a 256-token
        generation on this architecture.

        Defaults are IFEval-safe (greedy, no repetition penalty). Pass
        temperature>0 to sample; top_p < 1 and top_k > 0 gate the
        sample population. Sampling reuses the standard top-k then
        top-p nucleus filtering pattern.

        Caller is expected to set the model to eval mode if they want
        KV drops disabled at the MoE layer — the training vs inference
        switch happens in MoELayer.forward via self.training, which
        .train(False) toggles.
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        # Prefill over the prompt, keeping KV cache for decode.
        # HF's CausalLMOutputWithPast types past_key_values as
        # `Cache | None`, but our forward returns a plain list of
        # per-layer latent tensors. Cast locally so ty/mypy line up.
        PastKV = list[Tensor | None]
        context = input_ids[:, -self.config.max_position_embeddings:]
        out = self.forward(context, use_cache=True)
        past_key_values = cast("PastKV | None", out.past_key_values)

        # Per-row finished mask. A row is "finished" once it has ever
        # emitted eos_token_id on any decode step. Once finished, we
        # overwrite its next-token with EOS so downstream callers can
        # cleanly truncate, and we stop updating logits for it.
        batch_size = input_ids.shape[0]
        finished = torch.zeros(
            batch_size, dtype=torch.bool, device=input_ids.device,
        )
        logits_tensor = cast(Tensor, out.logits)
        logits_last = (
            logits_tensor[:, -1, :self.config.real_vocab_size].float()
        )
        generated = input_ids.clone()

        for step_idx in range(max_new_tokens):
            if step_idx > 0:
                # Decode: pass only the newest token + existing cache.
                new_tok = generated[:, -1:]
                # Don't trim past_key_values when the cache exceeds
                # max_position_embeddings — left-truncating the cache
                # shifts the absolute RoPE indices that the forward
                # derives from past_key_values[0].shape[2] (model.py:761
                # past_length read), so cached K (rotated at original
                # absolute positions) and the new K (rotated at the
                # post-trim shifted index) end up in different
                # positional bases and attention breaks.
                # The forward already handles required_seq_len > the
                # precomputed RoPE range by recomputing on demand
                # (model.py:773-786), so letting the cache grow
                # naturally is safe. Memory cost grows with generation
                # length; if that becomes a constraint, the right fix
                # is sliding-window with re-rotation, not a naive trim.
                out = self.forward(
                    new_tok,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                past_key_values = cast("PastKV | None", out.past_key_values)
                logits_tensor = cast(Tensor, out.logits)
                logits_last = (
                    logits_tensor[:, -1, :self.config.real_vocab_size].float()
                )

            # Repetition penalty (disabled by default). Vectorised so the
            # cost stays O(B*T) on-device instead of O(B*|set|) Python-loop
            # overhead per decode step. gather pulls each previously-seen
            # token's current logit, scales it, and scatters back. When a
            # token id appears multiple times in `generated[b]` the same
            # scaled value is written for every occurrence, so last-write-
            # wins on scatter is safe — semantically identical to the
            # original "apply once per unique id" loop.
            if repetition_penalty != 1.0:
                vocab = logits_last.shape[-1]
                # Mask out-of-vocab tokens so gather doesn't touch them.
                gen_clamped = generated.clamp(max=vocab - 1)
                in_vocab = (generated < vocab)
                score = torch.gather(logits_last, 1, gen_clamped)
                penalised = torch.where(
                    score > 0,
                    score / repetition_penalty,
                    score * repetition_penalty,
                )
                # Where the gathered position was out-of-vocab, write the
                # original score back (no-op) so scatter doesn't corrupt
                # in-vocab logits with garbage from clamped indices.
                penalised = torch.where(in_vocab, penalised, score)
                logits_last.scatter_(1, gen_clamped, penalised)

            if temperature > 0:
                next_logits = logits_last / temperature
                if top_k > 0:
                    topk_vals, _ = torch.topk(
                        next_logits, min(top_k, next_logits.size(-1)),
                    )
                    next_logits[
                        next_logits < topk_vals[:, -1:]
                    ] = float("-inf")
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        next_logits, descending=True,
                    )
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumprobs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_mask = cumprobs - sorted_probs >= top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    next_logits.scatter_(1, sorted_indices, sorted_logits)
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = logits_last.argmax(dim=-1, keepdim=True)

            # Force already-finished rows to keep emitting EOS so the
            # tensor stays rectangular, their completion is stable, and
            # we stop polluting them with extra tokens.
            if eos_token_id is not None and finished.any():
                next_token = torch.where(
                    finished.unsqueeze(-1),
                    torch.full_like(next_token, eos_token_id),
                    next_token,
                )

            generated = torch.cat([generated, next_token], dim=1)

            # Per-row termination. A row is finished once it has EVER
            # emitted EOS or any stop_token_id — we track that in
            # `finished` and break only when every row has finished at
            # some point, not only when all rows happen to emit a stop
            # token on the same step.
            #
            # stop_token_ids lets callers stop on chat-template markers
            # like <|/answer|> (token 10) or <|user|> (token 11). Useful
            # because MOPD-distilled models often generate additional
            # answer blocks after the first one or try to start a new
            # user turn — stopping on those keeps inference output clean.
            nt = next_token.squeeze(-1)
            if eos_token_id is not None:
                finished = finished | (nt == eos_token_id)
            if stop_token_ids:
                for sid in stop_token_ids:
                    finished = finished | (nt == sid)
            if (eos_token_id is not None or stop_token_ids) and bool(finished.all()):
                break

        return generated
