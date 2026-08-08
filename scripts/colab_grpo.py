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
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch  # noqa: E402

from osrt.grpo_train import (  # noqa: E402
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


def main() -> int:  # noqa: PLR0915
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default="/content/ckpt")
    ap.add_argument("--prompts", default="/content/grpo_prompts.jsonl")
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument("--hf-repo", default="")
    ap.add_argument("--base-ckpt",
                    default="osrt_v5_sft_v4_soup_1200_1400_1600_1800.pt")
    ap.add_argument("--ckpt-interval", type=int, default=50)
    ap.add_argument("--total-steps", type=int, default=0, help="override cfg")
    ap.add_argument("--num-prompts", type=int, default=0, help="override cfg")
    ap.add_argument("--micro-batch", type=int, default=8,
                    help="sequences per log-prob forward")
    ap.add_argument("--no-wandb", action="store_true")
    # ── speed levers ─────────────────────────────────────────────────
    ap.add_argument("--compile", action="store_true",
                    help="torch.compile the policy. Helps BOTH generation and "
                         "the log-prob passes (~20%+). Weights are not dynamo "
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
    if args.total_steps:
        cfg.total_steps = args.total_steps
    if args.max_gen_len:
        cfg.max_gen_len = args.max_gen_len
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
    ref_model.load_state_dict(rck.get("model_state_dict", rck), strict=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    del rck, ck

    # ── optimiser: full-parameter, with the differential HRA lr ───────────
    hra_params = [p for n, p in model.named_parameters()
                  if "adapters_a" in n or "adapters_b" in n]
    hra_ids = {id(p) for p in hra_params}
    base_params = [p for p in model.parameters() if id(p) not in hra_ids]
    print(f"full-parameter GRPO: {sum(p.numel() for p in base_params):,} base "
          f"+ {sum(p.numel() for p in hra_params):,} HRA", flush=True)
    optimizer = torch.optim.AdamW(
        [{"params": base_params, "lr": cfg.peak_lr},
         {"params": hra_params, "lr": cfg.hra_lr}],
        weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
    )

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
    print(f"{len(prompts_all)} prompts from {args.prompts}", flush=True)
    rng = random.Random(1234 + start_step)
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

        flat = [r for g in groups for r in g]
        optimizer.zero_grad(set_to_none=True)
        loss_val, mean_kl = train_on_rollouts(
            model, ref_model, flat, cfg, model_config.real_vocab_size,
            device, tok.pad_token_id or tok.eos_token_id,
            micro_batch=args.micro_batch,
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0 or step == start_step:
            rewards = [r.reward for r in flat]
            acc = sum(r.correct for r in flat) / max(len(flat), 1)
            live = sum(1 for r in flat if abs(r.advantage) > 1e-8)
            # Truncation is the silent killer: a rollout that never closes
            # <|answer|> forfeits the +3.0 exact-format term AND takes the
            # truncation penalty, so reward sits negative and it reads as "the
            # model is bad" rather than "max_gen_len is too small".
            trunc = sum(1 for r in flat if cfg.answer_close not in r.text)
            vram = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            print(f"step {step:>5d}/{cfg.total_steps} | loss {loss_val:.4f} | "
                  f"reward {sum(rewards) / len(rewards):+.3f} | acc {acc:.1%} | "
                  f"live {live}/{len(flat)} | trunc {trunc}/{len(flat)} | "
                  f"kl {mean_kl:.4f} | lr {lr:.2e} | "
                  f"vram {vram:.1f}GB | {time.time() - t0:.0f}s", flush=True)
            if use_wandb:
                import wandb
                wandb.log({"grpo/loss": loss_val, "grpo/accuracy": acc,
                           "grpo/mean_reward": sum(rewards) / len(rewards),
                           "grpo/approx_kl": mean_kl, "grpo/lr": lr,
                           "grpo/live_rollouts": live,
                           "grpo/truncated": trunc}, step=step)

        # Real generations — scalars cannot show the text has gone degenerate,
        # and reward hacking looks exactly like a healthy reward curve.
        if step % cfg.sample_print_interval == 0:
            print(f"  ---- rollouts @ step {step} (T={cfg.temperature}) ----",
                  flush=True)
            for r in groups[0][:3]:
                print(f"  [{'OK ' if r.correct else 'x  '}] rw {r.reward:+.2f} "
                      f"adv {r.advantage:+.2f} | {' '.join(r.text.split())[:320]}",
                      flush=True)

        if step > 0 and step % args.ckpt_interval == 0:
            out = ckpt_dir / f"{cfg.stage_prefix}_step_{step}.pt"
            torch.save({"step": step, "model_state_dict": model.state_dict()}, out)
            print(f"  -> saved {out.name}", flush=True)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
