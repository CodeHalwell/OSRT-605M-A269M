"""Unit tests for the v6 midtrain stage: config + native-HRA load gate."""

import pytest
import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM
from osrt.train import load_model_state_or_raise
from osrt.hra import inject_hra


def tiny_config(**overrides) -> OSRTConfig:
    """Small config for fast CPU tests (mirrors tests/test_model.py)."""
    defaults = dict(
        dim=128, heads=4, head_dim=32,
        vocab_size=512, real_vocab_size=512,
        num_blocks=2, recursive_loops=2,
        num_routed_experts=8, top_k_experts=2,
        expert_hidden=64, shared_expert_hidden=128,
        adapter_rank=16, adapter_alpha=16.0,
        max_position_embeddings=64,
    )
    defaults.update(overrides)
    return OSRTConfig(**defaults)


def test_native_hra_checkpoint_loads_without_injection():
    """A model built straight from config (native HRA) round-trips its
    own state_dict with no inject_hra — the hra_native=True path."""
    cfg = tiny_config()
    src = OSRTForCausalLM(cfg)
    state = src.state_dict()

    dst = OSRTForCausalLM(cfg)  # native HRA present, NO inject_hra
    # Must not raise: keys match exactly.
    load_model_state_or_raise(dst, state, context="native-hra test")


def test_inject_hra_on_native_model_breaks_load():
    """Proves WHY the gate is needed: injecting HRALinear onto a model
    that already has native HRA changes the key namespace, so loading a
    native checkpoint then raises."""
    cfg = tiny_config()
    native_state = OSRTForCausalLM(cfg).state_dict()

    injected = OSRTForCausalLM(cfg)
    inject_hra(injected, rank=cfg.adapter_rank, scale=1.0,
               freeze_pretrained=False)

    with pytest.raises(RuntimeError, match="key mismatch"):
        load_model_state_or_raise(
            injected, native_state, context="inject-breaks-load test"
        )


def test_midtrain_config_values():
    """MidtrainConfig encodes the locked decisions (spec §2/§4.4)."""
    from osrt.train_config import MidtrainConfig

    cfg = MidtrainConfig()
    assert cfg.total_steps == 9_000
    assert cfg.peak_lr == 2e-4
    assert cfg.min_lr == 2e-5
    assert cfg.warmup_steps == 150
    assert cfg.lr_anchor_step == 0
    # native + trainable HRA
    assert cfg.hra_native is True
    assert cfg.hra_frozen is False
    assert cfg.hra_enabled is True
    # router exploration off
    assert cfg.router_gumbel_tau_init == 0.0
    # computed Muon LR: (peak_lr / foundation_peak 6e-4) * foundation muon 0.02
    assert cfg.muon_lr == 6.6e-3
    assert cfg.muon_min_lr == 6.6e-4
    # gate disabled — fully disabled (not just "high")
    assert cfg.early_stop_check_step > 9_000
    assert cfg.early_stop_check_step == 9_999_999
    # resume + prefix
    assert cfg.pretrained_checkpoint.endswith("osrt_v5_final.pt")
    assert cfg.stage_prefix == "midtrain"
    # Checkpointing OFF — throughput bet, gated by the sanity probe (if it
    # OOMs at seq 4096, flip to True before the paid run).
    assert cfg.gradient_checkpointing is False


def test_midtrain_phase_is_seq4096_math_mix():
    """The single 'extend' phase is seq 4096 with the knowledge mix."""
    from osrt.train_config import MidtrainConfig

    phase = MidtrainConfig().phases["extend"]
    assert phase["seq_len"] == 4096
    names = {d["name"] for d in phase["datasets"]}
    assert "nemotron-cc-math-4plus" in names
    assert "fineweb-edu" in names           # general anchor retained
    assert "cosmopedia-openstax" in names
    # weights: math/STEM/reasoning should dominate (~0.65)
    math_sci = sum(
        d["weight"] for d in phase["datasets"]
        if d["name"] in {
            "nemotron-cc-math-4plus", "nemotron-stem",
            "nemotron-math-textbooks", "nemotron-reasoning",
        }
    )
    assert 0.60 <= math_sci <= 0.70
    # per-phase sizing (the loop reads these, not the inherited top-level batch)
    assert phase["batch_size"] == 6
    assert phase["grad_accum_steps"] == 11
    # all dataset weights sum to ~1.0 — guards against a typo'd weight
    total_weight = sum(d["weight"] for d in phase["datasets"])
    assert abs(total_weight - 1.0) < 1e-9
    assert len(phase["datasets"]) == 7


def test_midtrain_sanity_writes_no_final():
    """Sanity config is a short probe that won't clobber a real final."""
    from osrt.train_config import MidtrainSanityConfig

    cfg = MidtrainSanityConfig()
    assert cfg.total_steps == 30
    assert cfg.save_final_checkpoint is False
    assert cfg.stage_prefix == "midtrain-sanity"
    assert cfg.compile_enabled is False
    # inherits the real seq/mix so VRAM is measured at production size
    assert cfg.phases["extend"]["seq_len"] == 4096
