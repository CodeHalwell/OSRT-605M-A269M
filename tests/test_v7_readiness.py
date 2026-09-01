"""CPU launch gates for the committed v7 architecture and training recipe."""

from __future__ import annotations

import math

import pytest
import torch
from transformers import AutoTokenizer

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM
from osrt.presets import build_v7_config
from osrt.tokenizer_contract import validate_tokenizer_contract
from osrt.train import (
    _set_param_group_lrs,
    get_lr,
    load_checkpoint,
    save_checkpoint,
)
from osrt.train_config import V7SanityConfig


def test_v7_preset_matches_committed_shape_and_budget():
    cfg = build_v7_config(expert_orthogonal_init=False)
    assert cfg.num_routed_experts == 28
    assert cfg.top_k_experts == 4
    assert cfg.expert_hidden == 2112
    assert cfg.shared_expert_hidden == 2816
    assert cfg.use_mhc is False
    assert cfg.situ_glu is True
    assert cfg.router_balance_method == "quantile"
    assert cfg.router_seq_balance_loss_coeff == 1e-4

    with torch.device("meta"):
        model = OSRTForCausalLM(cfg)
    # 968,468,355 after gate G2 resolved the tokenizer to the OSRT-Ostinato
    # vocabulary (roadmap §16). The 25M delta against the pre-G2 993,437,571
    # is exactly the 65,536 -> 49,280 embedding change.
    assert sum(p.numel() for p in model.parameters()) == 968_468_355


def test_v7_wsd_schedule_boundaries():
    cfg = V7SanityConfig()
    assert get_lr(0, cfg) == 0.0
    assert get_lr(cfg.warmup_steps, cfg) == cfg.peak_lr
    decay_start = cfg.total_steps - cfg.wsd_decay_steps
    assert get_lr(decay_start - 1, cfg) == cfg.peak_lr
    assert get_lr(decay_start, cfg) == cfg.peak_lr
    assert math.isclose(
        get_lr(decay_start + cfg.wsd_decay_steps // 2, cfg),
        (cfg.peak_lr + cfg.min_lr) / 2,
    )
    assert get_lr(cfg.total_steps, cfg) == cfg.min_lr


def test_wsd_respects_per_group_peak_and_floor():
    cfg = V7SanityConfig()

    class _Optimizer:
        param_groups = [
            {"lr": 0.0, "_peak_lr": 2.0, "_min_lr": 0.2},
            {"lr": 0.0, "_peak_lr": 1.0, "_min_lr": 0.1},
        ]

    optimizer = _Optimizer()
    _set_param_group_lrs(optimizer, cfg.total_steps, cfg)
    assert optimizer.param_groups[0]["lr"] == 0.2
    assert optimizer.param_groups[1]["lr"] == 0.1


def test_verified_tokenizer_satisfies_contract():
    """tokenizer/ is now the OSRT-Ostinato vocabulary (gate G2, roadmap §16),
    not the v6 65,536 BPE. The contract pins its real size and structural
    token IDs so a swapped or half-built tokenizer fails before training."""
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    validate_tokenizer_contract(tokenizer)


def test_wrong_vocab_is_rejected():
    """Fail-closed is the point: a checkpoint trained at one vocab cannot load
    at another, and the v6 tokenizer is exactly the wrong one to reach for."""
    tokenizer = AutoTokenizer.from_pretrained("tokenizer")
    with pytest.raises(ValueError, match="vocab size is 49184, expected 65536"):
        validate_tokenizer_contract(tokenizer, expected_vocab_size=65_536)


def test_v7_resume_fails_closed_on_optimizer_mismatch(tmp_path):
    source = torch.nn.Linear(2, 2)
    source_optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    path = tmp_path / "mismatch.pt"
    torch.save(
        {
            "step": 7,
            "model_state_dict": source.state_dict(),
            "optimizer_state_dict": source_optimizer.state_dict(),
        },
        path,
    )

    target = torch.nn.Linear(2, 2)
    target_optimizer = torch.optim.AdamW(
        [
            {"params": [target.weight]},
            {"params": [target.bias]},
        ]
    )
    with pytest.raises(RuntimeError, match="strict resume"):
        load_checkpoint(
            target,
            target_optimizer,
            str(path),
            torch.device("cpu"),
            strict_optimizer=True,
        )


def test_v7_resume_rejects_same_shape_semantic_config_drift(tmp_path):
    common = dict(
        dim=32,
        heads=4,
        head_dim=8,
        num_kv_heads=2,
        vocab_size=64,
        real_vocab_size=64,
        num_blocks=1,
        recursive_loops=1,
        num_routed_experts=4,
        top_k_experts=2,
        expert_hidden=32,
        shared_expert_hidden=32,
        adapter_rank=4,
        max_position_embeddings=32,
        use_mhc=False,
        expert_orthogonal_init=False,
    )
    source = OSRTForCausalLM(OSRTConfig(**common, situ_glu=False))
    source_optimizer = torch.optim.AdamW(source.parameters())
    path = tmp_path / "semantic-drift.pt"
    recipe = {"total_steps": 30}
    save_checkpoint(
        source,
        source_optimizer,
        3,
        str(path),
        "pretrain_v7",
        recipe,
    )

    target = OSRTForCausalLM(OSRTConfig(**common, situ_glu=True))
    target_optimizer = torch.optim.AdamW(target.parameters())
    with pytest.raises(RuntimeError, match="configuration drift"):
        load_checkpoint(
            target,
            target_optimizer,
            str(path),
            torch.device("cpu"),
            strict_metadata=True,
            expected_training_stage="pretrain_v7",
            expected_training_recipe=recipe,
        )


def test_v7_strict_resume_accepts_exact_metadata_and_recipe(tmp_path):
    cfg = OSRTConfig(
        dim=32,
        heads=4,
        head_dim=8,
        num_kv_heads=2,
        vocab_size=64,
        real_vocab_size=64,
        num_blocks=1,
        recursive_loops=1,
        num_routed_experts=4,
        top_k_experts=2,
        expert_hidden=32,
        shared_expert_hidden=32,
        adapter_rank=4,
        max_position_embeddings=32,
        use_mhc=False,
        expert_orthogonal_init=False,
        situ_glu=True,
        router_balance_method="quantile",
    )
    recipe = {"lr_schedule": "wsd", "total_steps": 30}
    source = OSRTForCausalLM(cfg)
    source_optimizer = torch.optim.AdamW(source.parameters())
    path = tmp_path / "exact.pt"
    save_checkpoint(
        source,
        source_optimizer,
        3,
        str(path),
        "pretrain_v7",
        recipe,
    )

    target = OSRTForCausalLM(cfg)
    target_optimizer = torch.optim.AdamW(target.parameters())
    resumed_step = load_checkpoint(
        target,
        target_optimizer,
        str(path),
        torch.device("cpu"),
        strict_optimizer=True,
        strict_metadata=True,
        expected_training_stage="pretrain_v7",
        expected_training_recipe=recipe,
    )
    assert resumed_step == 4
