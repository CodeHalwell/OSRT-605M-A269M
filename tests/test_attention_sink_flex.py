"""Parity tests for the flex_attention sink implementation (review item B1).

The headline config runs ``attention_sink=True``, which currently routes every
attention call through ``_attention_with_sink`` — a manual path that materialises
the full ``(B, H, S, total_len)`` fp32 score matrix (no flash). At seq_len 8192
that is tens of GB per layer.

``attention_sink_impl="flex"`` instead uses ``torch.flex_attention`` with
``return_lse=True`` and applies the exact same per-head sink rescale
``sigmoid(lse - sink[h])`` to the fused-kernel output. These tests prove the flex
path is numerically equivalent to the manual path (fp32, tight tolerance) on both
the masked prefill branch (S>1) and the single-token decode branch (S==1).

NOTE: the *memory / flash* benefit only materialises under ``torch.compile`` on
CUDA; eager CPU flex_attention still materialises internally. These tests verify
correctness/parity, which is what makes the flag safe to flip on GPU.
"""

import pytest
import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM

# Eager flex_attention warns that it materialises the score matrix without
# torch.compile — that is the documented, expected behaviour on CPU (the memory
# win only lands on a compiled GPU run). The parity these tests assert is
# unaffected, so silence the expected warning to keep the suite output pristine.
pytestmark = pytest.mark.filterwarnings(
    "ignore:flex_attention called without torch.compile"
)


def _cfg(**overrides) -> OSRTConfig:
    defaults = dict(
        dim=128, heads=4, head_dim=32,
        num_kv_heads=2,                # GQA group_size=2 → exercise flex GQA
        vocab_size=256, real_vocab_size=256,
        num_blocks=2, recursive_loops=2,
        num_routed_experts=4, top_k_experts=2,
        expert_hidden=64, shared_expert_hidden=128,
        max_position_embeddings=64,
        attention_sink=True,
    )
    defaults.update(overrides)
    return OSRTConfig(**defaults)


def _set_impl(model: OSRTForCausalLM, impl: str) -> None:
    for blk in model.model.blocks:
        blk.attention_sink_impl = impl


def test_config_rejects_bad_attention_sink_impl():
    with pytest.raises(ValueError):
        _cfg(attention_sink_impl="bogus")


def test_flex_sink_path_is_actually_taken(monkeypatch):
    """When impl='flex', the block must route through the flex method (not the
    manual path). Drives the wiring so the parity tests aren't trivially equal."""
    from osrt.model import RecursiveBlock

    orig = getattr(RecursiveBlock, "_attention_with_sink_flex", None)
    calls = {"n": 0}

    def spy(self, *a, **k):
        calls["n"] += 1
        return orig(self, *a, **k)

    monkeypatch.setattr(
        RecursiveBlock, "_attention_with_sink_flex", spy, raising=False,
    )

    torch.manual_seed(0)
    model = OSRTForCausalLM(_cfg()).eval()
    _set_impl(model, "flex")
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        model(ids)
    # 2 blocks × 2 loops = 4 attention calls, all flex.
    assert calls["n"] >= 4, f"flex sink path not taken, got {calls['n']}"


def test_flex_sink_matches_manual_prefill():
    """Full forward (prefill, past_len=0) — flex sink output == manual."""
    torch.manual_seed(0)
    model = OSRTForCausalLM(_cfg()).eval()
    ids = torch.randint(0, 256, (2, 12))

    with torch.no_grad():
        _set_impl(model, "manual")
        logits_manual = model(ids).logits.clone()
        _set_impl(model, "flex")
        logits_flex = model(ids).logits.clone()

    assert torch.allclose(logits_manual, logits_flex, atol=1e-4, rtol=1e-4), (
        (logits_manual - logits_flex).abs().max().item()
    )


def test_flex_sink_matches_manual_cached_decode():
    """Single-token decode (S==1, past_len>0) — flex sink output == manual,
    given an identical (manual-computed) KV cache."""
    torch.manual_seed(1)
    model = OSRTForCausalLM(_cfg()).eval()
    ids = torch.randint(0, 256, (2, 10))

    with torch.no_grad():
        # Prefill (manual) to build an impl-independent latent cache.
        _set_impl(model, "manual")
        pre = model(ids[:, :-1], use_cache=True)
        past = pre.past_key_values
        last = ids[:, -1:]

        dec_manual = model(
            last, past_key_values=past, use_cache=True,
        ).logits.clone()
        _set_impl(model, "flex")
        dec_flex = model(
            last, past_key_values=past, use_cache=True,
        ).logits.clone()

    assert torch.allclose(dec_manual, dec_flex, atol=1e-4, rtol=1e-4), (
        (dec_manual - dec_flex).abs().max().item()
    )


def test_flex_block_mask_built_once_not_per_call(monkeypatch):
    """The causal block_mask is identical across all blocks/loops/forwards for a
    fixed (S, total_len, past_len); it must be built once and cached, not rebuilt
    on every attention call. Rebuilding per call is the flex_attention footgun
    that erases the memory/throughput win under torch.compile."""
    import torch.nn.attention.flex_attention as fa

    real = fa.create_block_mask
    calls = {"n": 0}

    def spy(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(fa, "create_block_mask", spy)

    cfg = _cfg()
    torch.manual_seed(0)
    model = OSRTForCausalLM(cfg).eval()
    _set_impl(model, "flex")
    ids = torch.randint(0, 256, (1, 8))
    with torch.no_grad():
        model(ids)               # num_blocks builds (loops reuse each block)
        n_after_first = calls["n"]
        model(ids)               # second forward — pure cache hits
    # Per-block cache: built once per block instance and reused across all loops,
    # so the first forward builds exactly num_blocks (NOT num_blocks*loops=4), and
    # the second forward rebuilds nothing. Steady-state rebuilds = 0.
    assert n_after_first == cfg.num_blocks, (
        f"first forward built {n_after_first}, expected {cfg.num_blocks} "
        f"(one per block, reused across loops)"
    )
    assert calls["n"] == n_after_first, (
        f"second forward rebuilt {calls['n'] - n_after_first} mask(s) — "
        f"must be pure cache hits"
    )


def test_flex_sink_backward_matches_manual():
    """Gradients through the flex sink path match the manual path (fp32)."""
    cfg = _cfg()
    torch.manual_seed(2)
    model = OSRTForCausalLM(cfg).train()
    ids = torch.randint(0, cfg.real_vocab_size, (2, 12))
    labels = ids.clone()

    def run(impl):
        _set_impl(model, impl)
        torch.manual_seed(7)  # pin stochastic routing
        model.zero_grad(set_to_none=True)
        out = model(ids, labels=labels)
        out.loss.backward()
        return out.loss.detach().clone(), model.model.embedding.weight.grad.clone()

    loss_m, g_m = run("manual")
    loss_f, g_f = run("flex")
    assert torch.allclose(loss_m, loss_f, atol=1e-4, rtol=1e-4)
    assert torch.allclose(g_m, g_f, atol=1e-4, rtol=1e-4)
