"""Offline GRPO gradient diagnostics on a CACHED rollout batch.

Why this is a separate path from training
-----------------------------------------
Two questions cannot be answered from the training log:

  1. Does the incorrect-answer advantage clamp help or starve the gradient?
  2. Is the KL term actually opposing the policy term, and how strongly?

Both need the SAME batch scored more than one way, and (2) needs the policy and
KL gradients as separate vectors. The training loop accumulates both terms into
one scalar before `backward()`, so it cannot produce them — an earlier claim of
"KL is 40% of the policy gradient" was derived from the ratio of the two scalar
LOSS terms, which is not a gradient-magnitude ratio and says nothing about
cancellation. That needs norms and a cosine, which is what this computes.

Everything here is deliberately outside the hot path: it loads a checkpoint and
a rollout dump, runs three backward passes with `zero_grad(set_to_none=True)`
between each, takes NO optimizer step, and writes nothing back. Diagnostic cost
does not touch training.

Norms are broken down by the two optimizer parameter groups, because they run
at different learning rates (base `peak_lr` 1.5e-6, HRA `hra_lr` 1.5e-5), so a
gradient norm alone does not tell you how far each family will actually move.

Usage:
  python scripts/grpo_grad_diag.py \
      --ckpt /vol/checkpoints/v5/grpo_v6_step_390.pt \
      --dump /vol/rollouts/diag_step390.jsonl \
      --tokenizer /vol/tokenizer
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from osrt.grpo_train import _seq_logprobs, load_rollout_dump  # noqa: E402
from osrt.presets import build_config  # noqa: E402
from osrt.rewards import compute_group_advantages  # noqa: E402
from osrt.train_config import GRPOv6Config  # noqa: E402

HRA_KEYS = ("adapters_a", "adapters_b")


def _is_hra(name: str) -> bool:
    return any(k in name for k in HRA_KEYS)


def _grad_snapshot(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone current .grad per parameter (fp32, CPU-free)."""
    return {n: p.grad.detach().clone().float()
            for n, p in model.named_parameters() if p.grad is not None}


def _norms(g: dict[str, torch.Tensor]) -> dict[str, float]:
    tot = base = hra = 0.0
    for n, v in g.items():
        s = float(v.pow(2).sum())
        tot += s
        if _is_hra(n):
            hra += s
        else:
            base += s
    return {"all": tot ** 0.5, "base": base ** 0.5, "hra": hra ** 0.5}


def _cosine(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor],
            which: str = "all") -> float:
    dot = na = nb = 0.0
    for n in a.keys() & b.keys():
        if which == "base" and _is_hra(n):
            continue
        if which == "hra" and not _is_hra(n):
            continue
        dot += float((a[n] * b[n]).sum())
        na += float(a[n].pow(2).sum())
        nb += float(b[n].pow(2).sum())
    if na <= 0 or nb <= 0:
        return float("nan")
    return dot / (na ** 0.5 * nb ** 0.5)


def _backward_policy(model, ref_model, groups, cfg, real_vocab, device, pad_id,
                     micro_batch: int, clamp: bool) -> tuple[int, int]:
    """Accumulate ONLY the policy-gradient term. Returns (live, changed)."""
    flat, changed = [], 0
    for group in groups:
        advs = compute_group_advantages([r.reward for r in group])
        for r, a in zip(group, advs):
            a = float(a)
            if clamp and not r.correct and a > 0.0:
                a = 0.0
                changed += 1
            flat.append((r, a))
    usable = [(r, a) for r, a in flat if len(r.ids) - r.prompt_len > 0]
    live = sum(1 for _, a in usable if abs(a) > 1e-8)
    n = len(usable)
    order = sorted(usable, key=lambda t: len(t[0].ids), reverse=True)
    for i in range(0, n, micro_batch):
        chunk = order[i:i + micro_batch]
        max_len = max(len(r.ids) for r, _ in chunk)
        batch = torch.full((len(chunk), max_len), pad_id, dtype=torch.long,
                           device=device)
        for k, (r, _) in enumerate(chunk):
            batch[k, : len(r.ids)] = r.ids
        p_lens = [r.prompt_len for r, _ in chunk]
        s_lens = [len(r.ids) for r, _ in chunk]
        pol = _seq_logprobs(model, batch, p_lens, s_lens, real_vocab, True,
                            temperature=cfg.temperature)
        loss = torch.zeros((), device=device)
        for lp, (r, a) in zip(pol, chunk):
            if abs(a) > 1e-8:
                loss = loss + -(lp * torch.tensor(a, device=device)).mean()
        if float(loss.detach()) != 0.0 or loss.requires_grad:
            (loss / n).backward()
    return live, changed


def _backward_kl(model, ref_model, groups, cfg, real_vocab, device, pad_id,
                 micro_batch: int) -> float:
    """Accumulate ONLY the beta*KL term, over every usable rollout."""
    usable = [r for g in groups for r in g if len(r.ids) - r.prompt_len > 0]
    n = len(usable)
    total_kl = 0.0
    order = sorted(usable, key=lambda r: len(r.ids), reverse=True)
    for i in range(0, n, micro_batch):
        chunk = order[i:i + micro_batch]
        max_len = max(len(r.ids) for r in chunk)
        batch = torch.full((len(chunk), max_len), pad_id, dtype=torch.long,
                           device=device)
        for k, r in enumerate(chunk):
            batch[k, : len(r.ids)] = r.ids
        p_lens = [r.prompt_len for r in chunk]
        s_lens = [len(r.ids) for r in chunk]
        pol = _seq_logprobs(model, batch, p_lens, s_lens, real_vocab, True,
                            temperature=cfg.temperature)
        ref = _seq_logprobs(ref_model, batch, p_lens, s_lens, real_vocab, False,
                            temperature=cfg.temperature)
        loss = torch.zeros((), device=device)
        for lp, rlp in zip(pol, ref):
            log_ratio = rlp.detach() - lp
            kl = (torch.exp(log_ratio) - log_ratio - 1).mean()
            loss = loss + cfg.kl_coeff * kl
            total_kl += float(kl.detach())
        (loss / n).backward()
    return total_kl / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument("--base-ckpt",
                    default="osrt_v5_sft_v4_soup_1200_1400_1600_1800.pt")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="override the SCORING temperature (0 = use cfg). Set "
                         "1.0 to reproduce the HISTORICAL wave-2 objective, "
                         "which sampled at 0.4 but scored log-probs at 1.0.")
    ap.add_argument("--alt-temperature", type=float, default=0.0,
                    help="ALSO score under this temperature and report "
                         "CROSS-CONFIG cosines. Clipping is a scalar, so it "
                         "cannot change a cosine — this answers how MISDIRECTED "
                         "one objective's update is relative to the other's, "
                         "which norms and clip coefficients cannot.")
    ap.add_argument("--alt-kl-coeff", type=float, default=-1.0,
                    help="beta for the --alt-temperature comparison run.")
    ap.add_argument("--kl-coeff", type=float, default=-1.0,
                    help="override beta (<0 = use cfg). Combine with "
                         "--temperature 1.0 --kl-coeff 0.15 to measure what "
                         "wave 2 ACTUALLY optimised, rather than extrapolating "
                         "a beta change at the corrected temperature.")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    device = torch.device("cuda")
    cfg = GRPOv6Config()
    if args.temperature > 0:
        cfg.temperature = args.temperature
    if args.kl_coeff >= 0:
        cfg.kl_coeff = args.kl_coeff

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    mc = build_config(vocab_size=len(tok), real_vocab_size=len(tok),
                      bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                      pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8)

    from osrt.model import OSRTForCausalLM
    model = OSRTForCausalLM(mc).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=True)
    miss, unexp = model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    assert not miss and not unexp, f"policy: {miss[:3]} {unexp[:3]}"
    del ck
    inner = model.model if hasattr(model, "model") else model
    inner._osrt_grad_ckpt = True
    if hasattr(model, "set_moe_telemetry"):
        model.set_moe_telemetry(False)

    ref = OSRTForCausalLM(mc).to(device)
    rck = torch.load(args.base_ckpt, map_location=device, weights_only=True)
    rmiss, runexp = ref.load_state_dict(rck.get("model_state_dict", rck),
                                        strict=False)
    assert not rmiss and not runexp, f"reference: {rmiss[:3]} {runexp[:3]}"
    del rck
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    if hasattr(ref, "set_moe_telemetry"):
        ref.set_moe_telemetry(False)

    groups, meta = load_rollout_dump(args.dump, device)
    n_roll = sum(len(g) for g in groups)
    print(f"replaying {len(groups)} groups / {n_roll} rollouts from {args.dump}")
    print(f"  dump meta: {meta}")
    print(f"  kl_coeff={cfg.kl_coeff} SCORING temperature={cfg.temperature} "
          f"micro_batch={args.micro_batch}\n")

    model.train()
    runs: dict[str, dict[str, torch.Tensor]] = {}

    model.zero_grad(set_to_none=True)
    live_u, _ = _backward_policy(model, ref, groups, cfg, mc.real_vocab_size,
                                 device, pad_id, args.micro_batch, clamp=False)
    runs["policy_unclamped"] = _grad_snapshot(model)

    model.zero_grad(set_to_none=True)
    live_c, changed = _backward_policy(model, ref, groups, cfg, mc.real_vocab_size,
                                       device, pad_id, args.micro_batch, clamp=True)
    runs["policy_clamped"] = _grad_snapshot(model)

    model.zero_grad(set_to_none=True)
    mean_kl = _backward_kl(model, ref, groups, cfg, mc.real_vocab_size, device,
                           pad_id, args.micro_batch)
    runs["beta_kl"] = _grad_snapshot(model)
    model.zero_grad(set_to_none=True)

    n_wrong = sum(1 for g in groups for r in g if not r.correct)
    print("── rollout accounting ─────────────────────────────────────")
    print(f"  rollouts {n_roll}   incorrect {n_wrong} ({n_wrong/n_roll:.1%})")
    print(f"  live unclamped {live_u}/{n_roll} ({live_u/n_roll:.1%})")
    print(f"  live clamped   {live_c}/{n_roll} ({live_c/n_roll:.1%})"
          f"   delta {live_c - live_u:+d}")
    print(f"  changed rollouts (wrong, adv>0 -> 0): {changed} "
          f"({changed/n_roll:.1%} of all, {changed/max(n_wrong,1):.1%} of wrong)")
    print(f"  mean approx_kl {mean_kl:.4f}\n")

    print("── gradient norms (by optimizer group) ────────────────────")
    print(f"  {'term':<18} {'all':>12} {'base':>12} {'hra':>12}")
    for k in ("policy_unclamped", "policy_clamped", "beta_kl"):
        nm = _norms(runs[k])
        print(f"  {k:<18} {nm['all']:12.4e} {nm['base']:12.4e} {nm['hra']:12.4e}")
    nu, nk = _norms(runs["policy_unclamped"]), _norms(runs["beta_kl"])
    for grp in ("all", "base", "hra"):
        print(f"  ||beta*KL|| / ||policy_unclamped||  [{grp}] = "
              f"{nk[grp]/nu[grp]:.3f}" if nu[grp] > 0 else "")

    # ── combined gradient: what the optimizer would actually see ───────
    # policy and beta*KL are gradients of separate loss terms, so they ADD.
    # Computed from the snapshots rather than a fourth backward pass.
    print("\n── combined gradient (policy + beta*KL) ───────────────────")
    import math
    for label, pk in (("unclamped", "policy_unclamped"),
                      ("clamped", "policy_clamped")):
        gp, gk = runs[pk], runs["beta_kl"]
        np_, nk = _norms(gp)["all"], _norms(gk)["all"]
        dot = sum(float((gp[k] * gk[k]).sum()) for k in gp.keys() & gk.keys())
        n_comb = (np_ ** 2 + nk ** 2 + 2 * dot) ** 0.5
        # angle between the policy term and the combined update
        cos_pc = (np_ ** 2 + dot) / (np_ * n_comb) if np_ * n_comb > 0 else float("nan")
        ang = math.degrees(math.acos(max(-1.0, min(1.0, cos_pc))))
        clip = min(1.0, cfg.grad_clip / n_comb) if n_comb > 0 else 1.0
        print(f"  {label:<10} ||policy|| {np_:.4f}  ||beta*KL|| {nk:.4f}  "
              f"||combined|| {n_comb:.4f}")
        print(f"  {'':<10} KL rotates the update {ang:.1f} deg off the policy "
              f"direction; clip coef (max_norm {cfg.grad_clip}) = {clip:.4f}")
    print("  NOTE: raw-gradient geometry only. Translating this into parameter"
          "\n  movement needs optimizer-state replay — global clipping and"
          "\n  Adam's per-parameter preconditioning both intervene.")

    print("\n── cosines ────────────────────────────────────────────────")
    for grp in ("all", "base", "hra"):
        print(f"  [{grp}] unclamped <-> clamped        "
              f"{_cosine(runs['policy_unclamped'], runs['policy_clamped'], grp):+.4f}")
        print(f"  [{grp}] unclamped <-> beta*KL        "
              f"{_cosine(runs['policy_unclamped'], runs['beta_kl'], grp):+.4f}")
        print(f"  [{grp}] clamped   <-> beta*KL        "
              f"{_cosine(runs['policy_clamped'], runs['beta_kl'], grp):+.4f}")
    # ── cross-configuration comparison ─────────────────────────────────
    if args.alt_temperature > 0:
        alt_t = args.alt_temperature
        alt_b = args.alt_kl_coeff if args.alt_kl_coeff >= 0 else cfg.kl_coeff
        prim_t, prim_b = cfg.temperature, cfg.kl_coeff
        print(f"\n── cross-config: primary (T={prim_t}, b={prim_b}) vs "
              f"alt (T={alt_t}, b={alt_b}) ──")
        cfg.temperature, cfg.kl_coeff = alt_t, alt_b
        model.zero_grad(set_to_none=True)
        _backward_policy(model, ref, groups, cfg, mc.real_vocab_size, device,
                         pad_id, args.micro_batch, clamp=False)
        runs["alt_policy"] = _grad_snapshot(model)
        model.zero_grad(set_to_none=True)
        _backward_kl(model, ref, groups, cfg, mc.real_vocab_size, device,
                     pad_id, args.micro_batch)
        runs["alt_beta_kl"] = _grad_snapshot(model)
        model.zero_grad(set_to_none=True)
        cfg.temperature, cfg.kl_coeff = prim_t, prim_b

        def _add(a, b):
            return {k: a[k] + b[k] for k in a.keys() & b.keys()}

        comb_p = _add(runs["policy_unclamped"], runs["beta_kl"])
        comb_a = _add(runs["alt_policy"], runs["alt_beta_kl"])
        for grp in ("all", "base", "hra"):
            print(f"  [{grp}] cos(policy_alt,   policy_primary)   "
                  f"{_cosine(runs['alt_policy'], runs['policy_unclamped'], grp):+.4f}")
            print(f"  [{grp}] cos(combined_alt, combined_primary) "
                  f"{_cosine(comb_a, comb_p, grp):+.4f}")
            print(f"  [{grp}] cos(combined_alt, policy_primary)   "
                  f"{_cosine(comb_a, runs['policy_unclamped'], grp):+.4f}"
                  "   <- alignment of the ALT update with the PRIMARY objective")
        print(f"  ||alt_policy|| {_norms(runs['alt_policy'])['all']:.4f}  "
              f"||alt_combined|| {_norms(comb_a)['all']:.4f}  "
              f"||primary_combined|| {_norms(comb_p)['all']:.4f}")

    print("\nNegative policy<->KL cosine = the anchor opposes the objective.")
    print("Cosines are clipping-invariant; norms are NOT what the optimizer")
    print("receives once global clipping binds. NO optimizer step was taken.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
