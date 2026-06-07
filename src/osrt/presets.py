"""Canonical model presets for osrt.

Generated/validated by `scripts/compute_budget.py`. The headline preset
`OSRT_605M_A279M` is the locked target for the $350 build:

    8 routed experts (Mixtral-style, top-2), shared expert shrunk and routed
    experts widened so the model is ~608M physical / ~279M active (45.8%).
    Hitting both the physical and active targets with 8 experts requires
    h_shared < h_routed (capacity shifted into the sparsely-active path).
"""

from __future__ import annotations

from osrt.config import OSRTConfig

# Locked $350-run target. See compute_budget.py: ~607.7M physical / 278.6M active.
OSRT_605M_A279M: dict = dict(
    dim=1536,
    heads=24,
    head_dim=64,
    num_kv_heads=8,            # GQA 24/8 + MLA-style compressed-latent KV cache
    vocab_size=65536,
    real_vocab_size=65536,
    num_blocks=3,
    recursive_loops=6,
    num_routed_experts=8,
    top_k_experts=2,
    expert_hidden=3968,        # routed experts (widened)
    shared_expert_hidden=2816,  # shared expert (shrunk; shifts capacity to sparse path)
    adapter_rank=16,
    # Manifold-Constrained Hyper-Connections (ARCHITECTURE.md §8): 4-channel
    # residual stream, Birkhoff/Sinkhorn doubly-stochastic mixing.
    # DEFAULT OFF: the CPU pre-flight (scripts/ablate_features.py) showed mHC
    # robustly amplifies gradient norms 10-700x and NaNs under sustained
    # training. Code + tests are kept; stabilization (per-channel stream norm,
    # bounded/straight-through Sinkhorn, smaller alpha-init) and re-enabling are
    # deferred to the small-GPU sanity run where gradients can be measured.
    use_mhc=False,
    n_hc=4,
    mhc_sinkhorn_iters=20,
    swiglu_clamp=10.0,         # DeepSeek-style SwiGLU stability clamp (§7.8)
    attention_sink=True,       # per-head learnable softmax-denominator sink (§6.6)
    # lean-v6 training stack (all already supported by the v5 model code)
    aux_loop_loss_weight=0.05,   # on from step 1 — anti loop-collapse
    router_aux_loss_coeff=0.10,  # v5-proven balance pressure
    router_z_loss_coeff=1e-3,
    router_balance_bias_enabled=True,
    # sqrt(softplus) routing affinity (ARCHITECTURE.md §7.4, §16.3): the
    # balance bias steers TOP-K selection on the non-negative affinity, gating
    # weights renormalise the selected balanced affinities. Hash routing stays
    # off (hash_routing_blocks default 0) — reserved for later A/B testing.
    router_affinity="sqrt_softplus",
    max_position_embeddings=4096,
)


def build_config(preset: dict = OSRT_605M_A279M, **overrides) -> OSRTConfig:
    """Build a OSRTConfig from a preset, with optional overrides."""
    return OSRTConfig(**{**preset, **overrides})
