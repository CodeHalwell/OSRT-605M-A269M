"""Canonical parameter / active-param budget for osrt configs.

Replaces the hand-derived tables in README.md / ARCHITECTURE.md with numbers
generated from the real model on a meta device (no memory allocated).

Usage:
    PYTHONPATH=src python scripts/compute_budget.py                 # default cfg
    PYTHONPATH=src python scripts/compute_budget.py --solve 605e6   # widen experts to hit a target
"""

from __future__ import annotations

import argparse

import torch

from osrt.config import OSRTConfig
from osrt.model import OSRTForCausalLM

# Map a parameter name to a budget category.
_CATEGORIES = [
    ("embedding", lambda n: "embedding" in n),
    ("attention", lambda n: any(k in n for k in ("q_proj", "kv_down", "v_from_k", "out_proj", "norm_q", "norm_k", "norm_attn"))),
    ("shared_expert", lambda n: "shared_expert" in n),
    ("routed_experts", lambda n: ".moe.experts." in n),
    ("router", lambda n: "router" in n or "moe_gate" in n),
    ("adapters", lambda n: "adapters_" in n),
    ("loop_emb", lambda n: "loop_embeddings" in n),
    ("norms_misc", lambda n: True),  # catch-all
]


def _categorize(name: str) -> str:
    for cat, pred in _CATEGORIES:
        if pred(name):
            return cat
    return "norms_misc"


def budget(cfg: OSRTConfig) -> dict[str, int]:
    """Return per-category physical param counts (meta device, no allocation)."""
    cfg.expert_orthogonal_init = False  # QR can't run on meta tensors
    with torch.device("meta"):
        model = OSRTForCausalLM(cfg)
    cats: dict[str, int] = {}
    for name, p in model.named_parameters():
        cats[_categorize(name)] = cats.get(_categorize(name), 0) + p.numel()
    return cats


def active_per_token(cats: dict[str, int], cfg: OSRTConfig) -> int:
    """Active params per token: routed experts scaled by top_k / num_routed,
    everything else fully active (embedding counted full: LM head touches the
    whole matrix)."""
    sparse_frac = cfg.top_k_experts / cfg.num_routed_experts
    active = 0
    for cat, n in cats.items():
        active += int(n * sparse_frac) if cat == "routed_experts" else n
    return active


def report(cfg: OSRTConfig) -> tuple[int, int]:
    cats = budget(cfg)
    total = sum(cats.values())
    active = active_per_token(cats, cfg)
    print(
        f"cfg: dim={cfg.dim} vocab={cfg.vocab_size} blocks={cfg.num_blocks} "
        f"loops={cfg.recursive_loops} experts={cfg.num_routed_experts} "
        f"top_k={cfg.top_k_experts} h_routed={cfg.expert_hidden} "
        f"h_shared={cfg.shared_expert_hidden} rank={cfg.adapter_rank}"
    )
    print("-" * 64)
    for cat in ("embedding", "attention", "shared_expert", "routed_experts",
                "router", "adapters", "loop_emb", "norms_misc"):
        if cat in cats:
            print(f"  {cat:<16} {cats[cat]:>14,}")
    print("-" * 64)
    print(f"  {'TOTAL PHYSICAL':<16} {total:>14,}  (~{total/1e6:.0f}M)")
    print(f"  {'ACTIVE / TOKEN':<16} {active:>14,}  (~{active/1e6:.0f}M, "
          f"{100*active/total:.1f}% of physical)")
    return total, active


def solve_expert_hidden(target: int, base: OSRTConfig, step: int = 128) -> int:
    """Smallest expert_hidden (multiple of `step`) whose total >= target."""
    h = step
    while True:
        cfg = OSRTConfig(**{**base.to_dict(), "expert_hidden": h,
                                "expert_orthogonal_init": False})
        if sum(budget(cfg).values()) >= target:
            return h
        h += step


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solve", type=float, default=None,
                    help="target physical params; widens expert_hidden to hit it")
    ap.add_argument("--vocab", type=int, default=65536)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--h-shared", type=int, default=4608)
    ap.add_argument("--h-routed", type=int, default=2048)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    base = OSRTConfig(
        vocab_size=args.vocab, real_vocab_size=args.vocab,
        num_routed_experts=args.experts, shared_expert_hidden=args.h_shared,
        expert_hidden=args.h_routed, adapter_rank=args.rank,
    )
    if args.solve:
        h = solve_expert_hidden(int(args.solve), base)
        print(f"=> expert_hidden={h} hits target {args.solve:.0f}\n")
        base = OSRTConfig(**{**base.to_dict(), "expert_hidden": h})
    report(base)


if __name__ == "__main__":
    main()
