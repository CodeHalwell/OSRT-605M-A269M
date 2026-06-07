"""Overfit-one-batch sanity: prove the lean-v6 training path actually learns.

Exercises the real stack on a small proxy of the OSRT-605M architecture:
recursive MoE (8 experts, top-2), aux-loop LM-head loss ON, Muon+AdamW split
optimizer. A correct training loop drives the loss on a fixed batch toward ~0.

    PYTHONPATH=src python scripts/sanity_overfit.py
"""

from __future__ import annotations

import torch

from nano_osrt.config import NanoOSRTConfig
from nano_osrt.model import NanoOSRTForCausalLM
from nano_osrt.muon import HybridMuonAdamW, Muon, build_param_groups


def main() -> None:
    torch.manual_seed(0)
    # Small proxy that keeps the architecture knobs we care about.
    cfg = NanoOSRTConfig(
        dim=256, heads=4, head_dim=64,
        vocab_size=512, real_vocab_size=512,
        num_blocks=2, recursive_loops=3,
        num_routed_experts=8, top_k_experts=2,
        expert_hidden=128, shared_expert_hidden=128,
        max_position_embeddings=64,
        aux_loop_loss_weight=0.05,      # the anti-collapse fix, ON
        router_balance_bias_enabled=True,
    )
    model = NanoOSRTForCausalLM(cfg)
    model.train()

    # Fixed batch to overfit.
    B, L = 4, 32
    ids = torch.randint(0, cfg.real_vocab_size, (B, L))
    labels = ids.clone()

    # Real optimizer wiring: Muon for 2D hidden matrices, AdamW for the rest.
    muon_params, adamw_groups = build_param_groups(model.named_parameters(), weight_decay=0.01)
    muon = Muon(muon_params, lr=0.02)
    adamw = torch.optim.AdamW(adamw_groups, lr=3e-3, betas=(0.9, 0.95))
    opt = HybridMuonAdamW(muon, adamw)

    losses = []
    for step in range(60):
        opt.zero_grad()
        out = model(ids, labels=labels)
        out.loss.backward()
        opt.step()
        if step % 10 == 0 or step == 59:
            losses.append((step, out.loss.item()))
            print(f"step {step:3d}  loss {out.loss.item():.4f}")

    first, last = losses[0][1], losses[-1][1]
    drop = 100 * (first - last) / first
    print(f"\nloss {first:.3f} -> {last:.3f}  ({drop:.0f}% drop)")
    assert last < first * 0.5, "FAIL: loss did not at least halve — training path broken"
    print("PASS: lean-v6 training path learns (Muon + aux-loop loss + 8-expert MoE).")


if __name__ == "__main__":
    main()
