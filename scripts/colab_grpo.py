"""GRPO on Colab (G4 / RTX PRO 6000 Blackwell 96GB) from the SFT-v4 soup.

Mirrors scripts/lightning_midtrain3.py: local checkpoint dir + periodic push to
a private HF repo, resume-scan on restart (Colab VM disk is ephemeral and
sessions cap at ~24h).

Uses osrt.grpo_train, which generates ALL num_prompts x group_size rollouts in
ONE call and batches the log-prob passes. The Modal loop did neither, giving
~5.5 min/step; see that module's docstring for the measured batch/throughput
curve. On G4 (~63% of H100, measured during midtrain3) expect ~60s/step.

Colab recipe — all four fixes are REQUIRED (each cost a debugging round):
  1. `--auth=adc` on every `colab` CLI call. oauth2 silently drops the
     `colaboratory` scope on refresh -> keep-alive 403s -> VM reclaimed.
  2. `--num-workers 0` (here: no DataLoader at all). Spawned workers hit a
     fatal PyGILState_Release teardown race with tokenizers/pyarrow + torch.
  3. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (fragmentation).
  4. Launch detached via nohup'd boot.sh; read truth from W&B / HF, NOT
     `colab exec` (flaky websocket reports a false "step 0").

Run:
  python scripts/colab_grpo.py \
      --ckpt-dir /content/ckpt --hf-repo HallD/osrt-v6-ckpt \
      --prompts /content/grpo_prompts.jsonl --ckpt-interval 50
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from osrt.grpo_train import (  # noqa: E402
    dump_rollouts,
    ema_init,
    ema_update,
    ema_weight_of_init,
    generate_rollouts,
    lr_at_step,
    train_on_rollouts,
)
from osrt.presets import build_config  # noqa: E402
from osrt.system_prompts import get_by_name  # noqa: E402
from osrt.train_config import GRPOv6Config  # noqa: E402


def _latest_local(ckpt_dir: Path, prefix: str) -> tuple[Path | None, int]:
    best, best_step = None, -1
    for f in ckpt_dir.glob(f"{prefix}_step_*.pt"):
        m = re.search(r"_step_(\d+)\.pt$", f.name)
        if m and int(m.group(1)) > best_step:
            best, best_step = f, int(m.group(1))
    return best, best_step


def _latest_hf(repo: str, prefix: str, ckpt_dir: Path) -> tuple[Path | None, int]:
    """Newest <prefix>_step_N.pt on the HF repo, downloaded into ckpt_dir.

    REQUIRED for multi-session runs. /content/ckpt dies with the VM, so a run
    spanning several Colab sessions has NO local checkpoint at the start of each
    one — and without this it silently restarted from step 0 every time. A
    200-step run across three sessions would have produced three ~65-step runs
    and never reached 200. The docstring above claimed "else HF"; the code did
    not implement it.

    Also pulls the matching EMA sidecar when one was pushed, so the shadow
    resumes rather than restarting from the resumed weights. Optimizer state is
    deliberately local-only (4.8GB), so AdamW moments WILL be cold after a
    session boundary — a ~20-step rebuild at beta2=0.95, reported so it is not
    mistaken for a training anomaly.
    """
    if not repo:
        return None, -1
    try:
        from huggingface_hub import HfApi, hf_hub_download
        pat = re.compile(rf"^{re.escape(prefix)}_step_(\d+)\.pt$")
        best, best_step = None, -1
        for f in HfApi().list_repo_files(repo, repo_type="model"):
            m = pat.match(f)
            if m and int(m.group(1)) > best_step:
                best, best_step = f, int(m.group(1))
        if best is None:
            return None, -1
        print(f"resuming from HF: {best} (step {best_step})", flush=True)
        dest = ckpt_dir / best
        shutil.copy2(hf_hub_download(repo, best, repo_type="model"), dest)
        ema_name = f"{prefix}_ema_step_{best_step}.pt"
        try:
            shutil.copy2(hf_hub_download(repo, ema_name, repo_type="model"),
                         ckpt_dir / f"{prefix}_ema.pt")
            print(f"  also pulled {ema_name} -> EMA sidecar", flush=True)
        except Exception:
            print("  no matching EMA sidecar on HF; shadow restarts from the "
                  "resumed weights", flush=True)
        print("  NOTE: optimizer state is local-only, so AdamW moments are COLD "
              "(~20-step rebuild at beta2=0.95)", flush=True)
        return dest, best_step
    except Exception as e:  # noqa: BLE001 — a failed scan must not kill the run
        print(f"HF resume scan failed ({type(e).__name__}: {e}); "
              f"falling back to the base checkpoint", flush=True)
        return None, -1


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/content/ckpt")
    ap.add_argument("--prompts", default="/content/grpo_prompts.jsonl")
    ap.add_argument("--dump-rollouts", default="",
                    help="JSONL path to append every rollout to. DEFAULT OFF: "
                         "adds file I/O to the training loop and grows ~2MB per "
                         "step. Used to cache a fixed batch for offline A/Bs "
                         "(strict-vs-loose extraction, advantage clamp).")
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--base-ckpt",
                    default="osrt_v5_sft_v4_soup_1200_1400_1600_1800.pt")
    ap.add_argument("--ckpt-interval", type=int, default=50)
    ap.add_argument("--total-steps", type=int, default=0, help="override cfg")
    ap.add_argument("--num-prompts", type=int, default=0, help="override cfg")
    ap.add_argument("--micro-batch", type=int, default=4,
                    help="sequences per log-prob forward")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--seed", type=int, default=1234,
                    help="seeds Python, CPU torch AND CUDA RNGs, and is logged "
                         "and stored in every checkpoint. Previously NOTHING "
                         "seeded torch: only prompt choice was fixed (via "
                         "random.Random(1234 + start_step)), so rollout "
                         "sampling was unrecorded and two 'seed' runs were "
                         "irreproducible — they demonstrated sensitivity but "
                         "could not estimate between-run variance. Default 1234 "
                         "keeps prompt selection identical to earlier runs; "
                         "torch seeding is new, so a --seed 1234 run will NOT "
                         "reproduce the unseeded ones.")
    ap.add_argument("--peak-lr", type=float, default=0.0,
                    help="override cfg.peak_lr (0 = use cfg). For an LR "
                         "calibration probe, compare at STEP 30: lr_at_step "
                         "computes warmup as peak_lr*eff/warmup_steps "
                         "independently of total_steps, and at step 30 "
                         "eff == warmup_steps so the cosine term is cos(0)=1 "
                         "and returns exactly peak_lr. Steps 0-30 are therefore "
                         "schedule-matched for ANY total_steps; divergence "
                         "starts at 31.")
    ap.add_argument("--hra-lr", type=float, default=0.0,
                    help="override cfg.hra_lr (0 = use cfg). HOLD THIS FIXED "
                         "while raising --peak-lr to attack adapter dominance: "
                         "measured HRA:base movement was 49x against a 10x lr "
                         "ratio, so scaling both together preserves the "
                         "imbalance. The HRA group is driven at "
                         "scheduled_lr * (hra_lr / peak_lr).")
    ap.add_argument("--run-tag", default="",
                    help="appended to cfg.stage_prefix. REQUIRED for an LR "
                         "sweep: without it every setting writes the same "
                         "<prefix>_step_N.pt names, so HF pushes collide and "
                         "_latest_local can resume a run at one LR from a "
                         "checkpoint trained at another.")
    ap.add_argument("--kl-abort", type=float, default=0.0,
                    help="stop if approx_kl exceeds this (0 = disabled). A "
                         "calibration probe at a raised LR multiplies drift: "
                         "KL ran ~0.0024/step at beta=0.04 and peak_lr 1.5e-6, "
                         "so 3.3x LR implies ~0.008/step, nearing wave 1's "
                         "worst of 0.33 by step 30.")
    ap.add_argument("--ema-decay", type=float, default=0.99,
                    help="passive weight-EMA decay (0 disables). 0.99 gives a "
                         "~100-step memory, right for a ~900-step run; 0.999 "
                         "would average over longer than the run. The shadow "
                         "NEVER generates rollouts and is never loaded into the "
                         "training model, so it cannot affect the theta path.")
    ap.add_argument("--ema-push-interval", type=int, default=50,
                    help="push the EMA sidecar to HF this often. The fp32 "
                         "shadow is 2.4GB, so pushing it every ckpt-interval "
                         "would cost more upload than the steps themselves; the "
                         "local copy is refreshed every interval for resume.")
    # ── speed levers ─────────────────────────────────────────────────
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the policy. Helps BOTH generation and "
                         "the log-prob passes (~20%%+). Weights are not dynamo "
                         "guards, so the graph stays valid across updates. Do "
                         "NOT prepack experts here — a gradient step would "
                         "leave the packed copies stale.")
    ap.add_argument("--max-gen-len", type=int, default=0,
                    help="override cfg.max_gen_len. Batched generation runs "
                         "until the LONGEST row finishes, so one slow rollout "
                         "costs the whole batch. Responses measure ~357-430 "
                         "tok, so ~448 trims the tail with little truncation.")
    args = ap.parse_args()

    cfg = GRPOv6Config()
    import random as _random
    _random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print(f"seed: {args.seed} (python + cpu torch + cuda)", flush=True)
    if args.total_steps:
        cfg.total_steps = args.total_steps
    if args.max_gen_len:
        cfg.max_gen_len = args.max_gen_len
    if args.peak_lr:
        cfg.peak_lr = args.peak_lr
    if args.hra_lr:
        cfg.hra_lr = args.hra_lr
    if args.run_tag:
        cfg.stage_prefix = f"{cfg.stage_prefix}_{args.run_tag}"
        cfg.wandb_run_name = f"{cfg.wandb_run_name}-{args.run_tag}"
    print(f"lr: peak {cfg.peak_lr:.3e}  hra {cfg.hra_lr:.3e}  "
          f"ratio {cfg.hra_lr / cfg.peak_lr:.1f}x | prefix {cfg.stage_prefix}",
          flush=True)
    num_prompts = args.num_prompts or cfg.grad_accum_steps
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    assert torch.cuda.is_available(), "GRPO needs a GPU"
    device = torch.device("cuda")
    print(f"GPU: {torch.cuda.get_device_name(0)} | "
          f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}",
          flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    assert len(tok) == 65536, f"wrong tokenizer: {len(tok)} (v6 is 65536)"

    model_config = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )

    # ── resume: prefer the newest local step ckpt, else HF, else the base ──
    from osrt.model import OSRTForCausalLM
    model = OSRTForCausalLM(model_config).to(device)
    resume_path, start_step = _latest_local(ckpt_dir, cfg.stage_prefix)
    if resume_path is None:
        resume_path, hf_step = _latest_hf(args.hf_repo, cfg.stage_prefix, ckpt_dir)
        start_step = hf_step if resume_path is not None else 0
    if resume_path is None:
        start_step = 0
        local_base = ckpt_dir / args.base_ckpt
        if not local_base.exists():
            from huggingface_hub import hf_hub_download
            print(f"pulling {args.base_ckpt} from {args.hf_repo}...", flush=True)
            src = hf_hub_download(args.hf_repo or "HallD/osrt-v6-ckpt",
                                  args.base_ckpt, repo_type="model")
            local_base = Path(src)
        resume_path = local_base
    print(f"loading {resume_path.name} (resume_step={start_step})", flush=True)
    ck = torch.load(resume_path, map_location=device, weights_only=True)
    sd = ck.get("model_state_dict", ck)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (
        f"state mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
    )
    del sd

    # Frozen reference for the KL anchor — ALWAYS the SFT base, never the
    # resumed policy, or the anchor drifts with the policy and stops anchoring.
    print("building frozen reference model...", flush=True)
    ref_model = OSRTForCausalLM(model_config).to(device)
    base_for_ref = ckpt_dir / args.base_ckpt
    if not base_for_ref.exists():
        from huggingface_hub import hf_hub_download
        base_for_ref = Path(hf_hub_download(
            args.hf_repo or "HallD/osrt-v6-ckpt", args.base_ckpt,
            repo_type="model"))
    rck = torch.load(base_for_ref, map_location=device, weights_only=True)
    # ASSERT like the policy load does. strict=False silently tolerates a key
    # mismatch, which would leave part of the reference randomly initialised —
    # the KL term would then penalise divergence from noise, inflating KL and
    # injecting a meaningless gradient, with nothing in the logs to show it.
    r_missing, r_unexpected = ref_model.load_state_dict(
        rck.get("model_state_dict", rck), strict=False)
    assert not r_missing and not r_unexpected, (
        f"reference state mismatch: missing={r_missing[:3]} "
        f"unexpected={r_unexpected[:3]}"
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    del rck, ck

    # ── optimiser: full-parameter, with the differential HRA lr ───────────
    hra_params = [p for n, p in model.named_parameters()
                  if "adapters_a" in n or "adapters_b" in n]
    hra_ids = {id(p) for p in hra_params}
    base_params = [p for p in model.parameters() if id(p) not in hra_ids]
    # ── passive weight EMA (see grpo_train.ema_init for why full state_dict) ──
    ema_state, ema_updates = None, 0
    ema_path = ckpt_dir / f"{cfg.stage_prefix}_ema.pt"
    if args.ema_decay > 0:
        ema_state = ema_init(model)
        if start_step > 0 and ema_path.exists():
            eck = torch.load(ema_path, map_location=device, weights_only=False)
            if int(eck.get("source_step", -1)) == start_step and \
                    float(eck.get("ema_decay", -1)) == args.ema_decay:
                got = eck["model_state_dict"]
                assert set(got) == set(ema_state), "EMA sidecar key mismatch"
                for k in ema_state:
                    ema_state[k].copy_(got[k].float())
                ema_updates = int(eck.get("ema_updates", 0))
                print(f"restored EMA (decay {args.ema_decay}, "
                      f"{ema_updates} updates) from step {start_step}", flush=True)
            else:
                print(f"EMA sidecar is step {eck.get('source_step')}/decay "
                      f"{eck.get('ema_decay')}, resuming at {start_step}/"
                      f"{args.ema_decay} — RESETTING shadow to current weights",
                      flush=True)
            del eck
        print(f"weight EMA: decay {args.ema_decay}, {len(ema_state)} tensors, "
              f"fp32 shadow (passive; never samples, never trains)", flush=True)

    print(f"full-parameter GRPO: {sum(p.numel() for p in base_params):,} base "
          f"+ {sum(p.numel() for p in hra_params):,} HRA", flush=True)
    optimizer = torch.optim.AdamW(
        [{"params": base_params, "lr": cfg.peak_lr},
         {"params": hra_params, "lr": cfg.hra_lr}],
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )
    # Restore AdamW moments if a matching sidecar exists, so re-running the cell
    # does not silently restart with cold moments (~20-step rebuild at
    # beta2=0.95). Only accept it if it belongs to the step we resumed from.
    opt_path = ckpt_dir / f"{cfg.stage_prefix}_optim.pt"
    if start_step > 0 and opt_path.exists():
        try:
            ock = torch.load(opt_path, map_location=device, weights_only=False)
            if int(ock.get("step", -1)) == start_step:
                optimizer.load_state_dict(ock["optimizer_state_dict"])
                print(f"restored optimizer state from step {start_step}", flush=True)
            else:
                print(f"optimizer sidecar is step {ock.get('step')}, "
                      f"resuming at {start_step} — NOT loading (cold moments)",
                      flush=True)
            del ock
        except Exception as e:  # noqa: BLE001 — cold moments beat a dead run
            print(f"optimizer restore failed ({type(e).__name__}: {e}); "
                  f"continuing with cold moments", flush=True)
    elif start_step > 0:
        print("no optimizer sidecar — resuming with COLD moments", flush=True)

    # GRADIENT CHECKPOINTING — required, not optional. `use_ckpt` in
    # OSRTModel.forward is `self._osrt_grad_ckpt and self.training`, so it
    # applies ONLY to the log-prob training pass and never to generation
    # (which runs under eval()/no_grad anyway). Without it, activations for 18
    # effective layer applications across a micro-batch of long sequences fill
    # the card: an OOM at 94.07/94.97GB allocated on the very first policy
    # forward. SFT ran with this on throughout.
    inner = model.model if hasattr(model, "model") else model
    inner._osrt_grad_ckpt = True
    print("gradient checkpointing: ENABLED (_osrt_grad_ckpt=True)", flush=True)

    # MoE telemetry does ~21 .item() calls per layer; each is a CUDA sync AND a
    # dynamo graph break (dynamo even specialises on the scalar VALUE, which
    # blew the recompile limit on the Modal run). GRPO never reads it.
    if hasattr(model, "set_moe_telemetry"):
        model.set_moe_telemetry(False)
    if hasattr(ref_model, "set_moe_telemetry"):
        ref_model.set_moe_telemetry(False)

    if args.compile:
        print("compiling policy (cold trace is minutes; helps every step "
              "after)...", flush=True)
        model.forward = torch.compile(model.forward, dynamic=True)  # type: ignore[method-assign]

    # ── prompts (pre-built, unseen, with numeric gold) ────────────────────
    prompts_all = [json.loads(line) for line in open(args.prompts)]
    # HARD FAIL on invalid gold — deterministic data validation, not a tunable.
    # An empty/whitespace gold makes compute_reward return the
    # `no_ground_truth` tier (0.0 correctness), so EVERY rollout on that prompt
    # scores format-only (+0.20 observed at step 310) regardless of what it
    # answers: a pure format-training group with no correctness signal. The
    # builder's `gold is not None` guard does not catch it, and validating only
    # in the builder would let an older or hand-made prompt file silently
    # reintroduce it here.
    bad = [i for i, p in enumerate(prompts_all)
           if not str(p.get("answer", "")).strip()
           or not str(p.get("question", "")).strip()]
    if bad:
        raise ValueError(
            f"{len(bad)} of {len(prompts_all)} prompts in {args.prompts} have an "
            f"empty question or gold answer (first offending lines: "
            f"{[i + 1 for i in bad[:5]]}). Rebuild the prompt file; these "
            f"contribute zero correctness signal and train format only."
        )
    print(f"{len(prompts_all)} prompts from {args.prompts} (gold validated)",
          flush=True)
    rng = random.Random(args.seed + start_step)
    sys_text = get_by_name(cfg.system_persona) if cfg.system_persona else ""
    end_ans = tok.encode(cfg.answer_close, add_special_tokens=False)[0]
    stop_ids = [end_ans]

    use_wandb = not args.no_wandb and os.environ.get("WANDB_API_KEY")
    if use_wandb:
        import wandb
        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                   config={"total_steps": cfg.total_steps,
                           "num_prompts": num_prompts,
                           "group_size": cfg.group_size,
                           "temperature": cfg.temperature,
                           "kl_coeff": cfg.kl_coeff}, resume="allow")

    print(f"\n>>> GRPO | {num_prompts} prompts x {cfg.group_size} = "
          f"{num_prompts * cfg.group_size} rollouts/step | T={cfg.temperature} "
          f"| kl={cfg.kl_coeff} | steps {start_step}->{cfg.total_steps}\n",
          flush=True)

    t0 = time.time()
    for step in range(start_step, cfg.total_steps):
        lr = lr_at_step(step, cfg)
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * (cfg.hra_lr / cfg.peak_lr)

        picks = rng.sample(prompts_all, min(num_prompts, len(prompts_all)))
        batch = [(f"{cfg.system_tag}{sys_text}{cfg.user_tag}{p['question']}"
                  f"{cfg.assistant_tag}" if sys_text else
                  f"{cfg.user_tag}{p['question']}{cfg.assistant_tag}",
                  str(p["answer"])) for p in picks]

        model.eval()
        groups = generate_rollouts(model, tok, batch, cfg, device, stop_ids)
        model.train()

        if args.dump_rollouts:
            dump_rollouts(
                args.dump_rollouts, groups,
                ckpt=resume_path.name, step=step, seed=1234 + start_step,
                temperature=cfg.temperature, top_p=getattr(cfg, "top_p", 1.0),
            )

        flat = [r for g in groups for r in g]
        # Hand the generation KV back before the training pass allocates.
        torch.cuda.empty_cache()
        optimizer.zero_grad(set_to_none=True)
        loss_val, mean_kl = train_on_rollouts(
            model, ref_model, flat, cfg, model_config.real_vocab_size,
            device, tok.pad_token_id or tok.eos_token_id,
            micro_batch=args.micro_batch,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        if ema_state is not None:          # AFTER the step, observer only
            ema_update(ema_state, model, args.ema_decay)
            ema_updates += 1

        if step % cfg.log_interval == 0 or step == start_step:
            rewards = [r.reward for r in flat]
            acc = sum(r.correct for r in flat) / max(len(flat), 1)
            live = sum(1 for r in flat if abs(r.advantage) > 1e-8)
            # Truncation is the silent killer: a rollout that never closes
            # <|answer|> forfeits the +3.0 exact-format term AND takes the
            # truncation penalty, so reward sits negative and it reads as "the
            # model is bad" rather than "max_gen_len is too small".
            # TWO DISTINCT failure modes, previously conflated under one
            # counter. `no_close` counts completions missing the answer-close
            # tag; `cap` counts completions that actually hit max_gen_len, which
            # compute_reward already flags as breakdown["truncated"]
            # (completion_tokens >= max_tokens) and which this logger ignored.
            # Reading no_close as "hit the cap" made "truncation self-corrected"
            # and "1024 was required" look measured when neither was: a model
            # learning to emit <|/answer|> moves no_close without touching cap.
            no_close = sum(1 for r in flat if cfg.answer_close not in r.text)
            cap = sum(1 for r in flat if (r.breakdown or {}).get("truncated"))
            trunc = no_close  # keep the historical column comparable
            vram = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            print(f"step {step:>5d}/{cfg.total_steps} | loss {loss_val:.4f} | "
                  f"reward {sum(rewards) / len(rewards):+.3f} | acc {acc:.1%} | "
                  f"live {live}/{len(flat)} | noclose {no_close}/{len(flat)} | "
                  f"cap {cap}/{len(flat)} | "
                  f"kl {mean_kl:.4f} | lr {lr:.2e} | "
                  f"vram {vram:.1f}GB | {time.time() - t0:.0f}s", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"grpo/loss": loss_val, "grpo/accuracy": acc,
                           "grpo/mean_reward": sum(rewards) / len(rewards),
                           "grpo/approx_kl": mean_kl, "grpo/lr": lr,
                           "grpo/live_rollouts": live,
                           "grpo/truncated": trunc,
                           "grpo/no_answer_close": no_close,
                           "grpo/hit_cap": cap}, step=step)

        # Real generations — scalars cannot show the text has gone degenerate,
        # and reward hacking looks exactly like a healthy reward curve.
        if args.kl_abort and mean_kl > args.kl_abort:
            print(f"\nABORT: approx_kl {mean_kl:.4f} exceeded --kl-abort "
                  f"{args.kl_abort} at step {step}. The policy is drifting off "
                  f"the frozen reference faster than the anchor holds; wave 1's "
                  f"worst was 0.33. Checkpoint at the last interval is intact.",
                  flush=True)
            break

        if step % cfg.sample_print_interval == 0:
            print(f"  ---- rollouts @ step {step} (T={cfg.temperature}) ----",
                  flush=True)
            for r in groups[0][:3]:
                print(f"  [{'OK ' if r.correct else 'x  '}] rw {r.reward:+.2f} "
                      f"adv {r.advantage:+.2f} | {' '.join(r.text.split())[:320]}",
                      flush=True)

        if step > 0 and step % args.ckpt_interval == 0:
            out = ckpt_dir / f"{cfg.stage_prefix}_step_{step}.pt"
            # LR provenance: a calibration sweep produces checkpoints that are
            # otherwise indistinguishable from each other.
            torch.save({"step": step, "model_state_dict": model.state_dict(),
                        "seed": args.seed,
                        "peak_lr": cfg.peak_lr, "hra_lr": cfg.hra_lr,
                        "kl_coeff": cfg.kl_coeff, "temperature": cfg.temperature,
                        "top_p": getattr(cfg, "top_p", 1.0),
                        "max_gen_len": cfg.max_gen_len}, out)
            print(f"  -> saved {out.name}", flush=True)
            # Optimizer state goes to a SEPARATE, LOCAL-ONLY file. AdamW moments
            # for 601M params are ~4.8GB, so pushing them every 10 steps would
            # cost more upload time than the step itself; but losing them resets
            # both moments on resume and at beta2=0.95 that is a ~20-step
            # rebuild, i.e. a fresh discontinuity every Colab session boundary.
            # Keeping them on disk covers re-running the cell in a LIVE session;
            # --opt-state-interval controls the (optional) durable copy.
            opt_out = ckpt_dir / f"{cfg.stage_prefix}_optim.pt"
            torch.save({"step": step, "optimizer_state_dict": optimizer.state_dict()},
                       opt_out)
            if ema_state is not None:
                # Complete, strictly-loadable state_dict (all 229 keys), so the
                # eval path can load it into a SEPARATE model. Never loaded into
                # the compiled training model.
                torch.save({"step": step, "source_step": step,
                            "ema_decay": args.ema_decay,
                            "ema_updates": ema_updates,
                            "model_state_dict": ema_state}, ema_path)
                w0 = ema_weight_of_init(args.ema_decay, ema_updates)
                print(f"  -> saved EMA sidecar ({ema_updates} updates, "
                      f"{w0:.1%} residual weight on the base)", flush=True)
                if args.hf_repo and step % args.ema_push_interval == 0:
                    named = ckpt_dir / f"{cfg.stage_prefix}_ema_step_{step}.pt"
                    torch.save({"step": step, "source_step": step,
                                "ema_decay": args.ema_decay,
                                "ema_updates": ema_updates,
                                "model_state_dict": ema_state}, named)
                    try:
                        from huggingface_hub import HfApi
                        HfApi().upload_file(
                            path_or_fileobj=str(named), path_in_repo=named.name,
                            repo_id=args.hf_repo, repo_type="model",
                            commit_message=f"grpo EMA step {step}")
                        print(f"  -> pushed {named.name}", flush=True)
                    except Exception as e:  # noqa: BLE001
                        print(f"  EMA push failed ({type(e).__name__}: {e})",
                              flush=True)
                    named.unlink(missing_ok=True)
            if args.hf_repo:
                try:
                    from huggingface_hub import HfApi
                    HfApi().upload_file(
                        path_or_fileobj=str(out), path_in_repo=out.name,
                        repo_id=args.hf_repo, repo_type="model",
                        commit_message=f"grpo step {step}")
                    print(f"  -> pushed {out.name} to {args.hf_repo}", flush=True)
                except Exception as e:  # noqa: BLE001 — never kill the run
                    print(f"  HF push failed ({type(e).__name__}: {e})",
                          flush=True)

    final = ckpt_dir / f"{cfg.stage_prefix}_final.pt"
    torch.save({"step": cfg.total_steps, "model_state_dict": model.state_dict()},
               final)
    print(f"\nGRPO complete. {final}", flush=True)
    # PUSH IT. The upload above only runs inside the ckpt_interval block, so the
    # final weights previously stayed on an ephemeral Colab VM and were lost
    # when the session ended — the newest surviving checkpoint was whatever the
    # last interval happened to be (step 40 of a 50-step run). Also push the EMA
    # shadow, which is a separate evaluation candidate.
    if args.hf_repo:
        from huggingface_hub import HfApi
        to_push = [final]
        if ema_state is not None:
            ema_final = ckpt_dir / f"{cfg.stage_prefix}_ema_final.pt"
            torch.save({"step": cfg.total_steps, "source_step": cfg.total_steps,
                        "ema_decay": args.ema_decay, "ema_updates": ema_updates,
                        "model_state_dict": ema_state}, ema_final)
            to_push.append(ema_final)
        for f in to_push:
            try:
                HfApi().upload_file(
                    path_or_fileobj=str(f), path_in_repo=f.name,
                    repo_id=args.hf_repo, repo_type="model",
                    commit_message=f"grpo final ({cfg.total_steps} steps)")
                print(f"  -> pushed {f.name}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  push failed for {f.name} "
                      f"({type(e).__name__}: {e})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
