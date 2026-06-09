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
