"""v6 training paths archived with the v6 pipeline.

run_rollout_eval, _freeze_hra_params and run_pretrain_extend. All three went
dead when the v6 stage configs were archived, and run_pretrain_extend /
run_rollout_eval also called make_rollout_loader, which moved with the SFT
rollout builder. Reference only.
"""

def _freeze_hra_params(model: nn.Module) -> tuple[int, int]:
    """Freeze every HRA adapter parameter on a (possibly compiled) model.

    HRA wraps Linear layers with a parallel `adapter_a @ adapter_b`
    pair (singular, see hra.py::HRALinear). State-dict names end in
    ".adapter_a" / ".adapter_b" — e.g. "model.blocks.0.qkv.adapter_a".

    CRITICAL: do NOT use substring `"adapter_a" in name` — that also
    matches the recursive-architecture loop adapters at "model.adapters_a"
    (plural, ModuleList from config.adapter_rank=16). Endswith with the
    leading "." disambiguates: HRA uses `.adapter_a` singular, loop
    adapters are `model.adapters_a.<i>` plural. A previous version of
    this function used the substring check and only froze 884k params
    (the loop adapters!), leaving the 86.1M HRA tensors fully trainable
    despite "Frozen HRA" log messages — the bug was caught during
    validation by counting trainable params before/after.

    Setting requires_grad=False excludes them from both Muon and
    AdamW optimizer groups (build_param_groups in muon.py and the
    AdamW path in run_training both check requires_grad).

    Returns (n_frozen_params, n_frozen_tensors) for logging.
    """
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    n_params = 0
    n_tensors = 0
    for name, p in inner.named_parameters():
        if name.endswith(".adapter_a") or name.endswith(".adapter_b"):
            p.requires_grad = False
            n_params += p.numel()
            n_tensors += 1
    return n_params, n_tensors


def run_pretrain_extend(
    model_config: OSRTConfig,
    extend_cfg,
    vol,
    tokenizer_name: str,
    ckpt_dir: str = "/vol/checkpoints/v5",
) -> None:
    """Continued pre-training on top of an SFT checkpoint.

    Differs from `run_training` in four bounded ways:
      1. Initial weights come from `extend_cfg.pretrained_checkpoint`
         (an SFT ckpt with HRA params), not from a glob-scan.
      2. HRA structure is injected before load so SFT-trained HRA
         tensors land in their slots correctly.
      3. HRA params are frozen post-load if `hra_frozen=True` (the
         default for this stage).
      4. Checkpoints are written under `osrt_v5_{stage_prefix}_step_*.pt`
         so they don't collide with base pretrain checkpoints under
         the resume scanner.

    Held-out eval is OPT-IN via `eval_interval`: when set (e.g. the v6
    MidtrainConfig uses 750), the loop runs run_eval on that cadence and
    logs eval/loss + eval/perplexity. v5 extend stages set eval_interval
    to 9_999_999 to skip it — the held-out cache builder eats 10-20 min
    of first-call time, which can trip Modal's heartbeat on short runs.
    """
    device = torch.device("cuda")

    print("=" * 60)
    print("OSRT — Continued Pre-training (Extend / Mid-training)")
    print("=" * 60)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Build model ─────────────────────────────────────────────────
    model = OSRTForCausalLM(model_config).to(device=device)
    base_params = sum(p.numel() for p in model.parameters())
    print(f"Base parameters     : {base_params:>12,}")
    print(f"Resume from         : {extend_cfg.pretrained_checkpoint}")
    print(f"Stage prefix        : {extend_cfg.stage_prefix}")
    print(f"Total steps         : {extend_cfg.total_steps}")
    print(f"Peak LR             : {extend_cfg.peak_lr}")
    print(f"HRA enabled         : {extend_cfg.hra_enabled}")
    print(f"HRA frozen          : {getattr(extend_cfg, 'hra_frozen', False)}")
    print()

    # ── HRA injection (BEFORE state_dict load) ─────────────────────
    # v5 path: the foundation model had NO HRA, so this stage injects
    #   HRALinear wrappers before loading an SFT checkpoint whose
    #   state_dict contains them.
    # v6 path (hra_native=True): HRA is built inline from config
    #   (model.py adapters_a/adapters_b ParameterList) and is ALREADY in
    #   the foundation checkpoint. Injecting HRALinear here would graft a
    #   second, mismatched layout and make load_model_state_or_raise throw.
    # Key shapes disambiguate the two "HRA"s: native uses plural,
    #   dotted-index keys `adapters_a.N`/`adapters_b.N`; inject_hra's
    #   HRALinear uses singular `....adapter_a` nested per wrapped Linear.
    # NB: hra_native=True is meant to pair with hra_frozen=False (the
    #   shipped MidtrainConfig); with both True, _freeze_hra_params keys
    #   on singular `.adapter_a` and silently freezes nothing on a native
    #   model.
    hra_native = getattr(extend_cfg, "hra_native", False)
    if extend_cfg.hra_enabled and not hra_native:
        from osrt.hra import inject_hra

        print(f"Injecting HRA before load (rank={extend_cfg.hra_rank})...")
        inject_hra(
            model,
            rank=extend_cfg.hra_rank,
            scale=getattr(extend_cfg, "hra_scale", 1.0),
            freeze_pretrained=False,
        )
        with_hra_params = sum(p.numel() for p in model.parameters())
        added = with_hra_params - base_params
        print(f"  HRA injected: +{added:,} params ({added / 1e6:.1f}M)")
    elif hra_native:
        print(
            "HRA is native (built from config) — skipping inject_hra; "
            "foundation checkpoint already carries adapters_a/adapters_b."
        )

    # ── Load SFT checkpoint ────────────────────────────────────────
    ckpt_path = extend_cfg.pretrained_checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"pretrain_extend refuses to start: pretrained checkpoint "
            f"not found at {ckpt_path}. Run SFT first or correct the "
            f"`pretrained_checkpoint` path in the extend config.",
        )
    print(f"Loading initial weights from {ckpt_path}...")
    init_ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    init_state = init_ckpt.get("model_state_dict", init_ckpt)
    load_model_state_or_raise(
        model,
        init_state,
        context=f"pretrain_extend initial load from {ckpt_path}",
    )
    print("  Clean load: all keys matched.")

    # ── Freeze HRA after load ──────────────────────────────────────
    if extend_cfg.hra_enabled and getattr(extend_cfg, "hra_frozen", False):
        n_frozen_params, n_frozen_tensors = _freeze_hra_params(model)
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(
            f"Froze {n_frozen_tensors} HRA tensors "
            f"({n_frozen_params:,} params, "
            f"{n_frozen_params / 1e6:.1f}M). "
            f"Trainable: {trainable_after:,} "
            f"({trainable_after / 1e6:.1f}M)",
        )

    # `compile_enabled` opt-out lets a stage skip torch.compile entirely.
    # Eager is ~2-3× slower per step but starts emitting step events
    # immediately — useful for sanity stages and for workspaces where
    # silent compile time triggers Modal's idle-cancellation heuristic.
    if getattr(extend_cfg, "compile_enabled", True):
        print("\nCompiling model with torch.compile...")
        compile_start = time.time()
        model = torch.compile(model)
        print(f"Model compile done in {time.time() - compile_start:.1f}s")
    else:
        print("\nSkipping torch.compile (compile_enabled=False, eager mode).")

    # ── W&B ────────────────────────────────────────────────────────
    use_wandb = extend_cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb_kwargs = {
            "project": extend_cfg.wandb_project,
            "name": extend_cfg.wandb_run_name,
            "config": {
                "stage": "pretrain_extend",
                "base_params": base_params,
                "hra_enabled": extend_cfg.hra_enabled,
                "hra_frozen": getattr(extend_cfg, "hra_frozen", False),
                "pretrained_checkpoint": extend_cfg.pretrained_checkpoint,
                "peak_lr": extend_cfg.peak_lr,
                "min_lr": extend_cfg.min_lr,
                "warmup_steps": extend_cfg.warmup_steps,
                "total_steps": extend_cfg.total_steps,
                "datasets": [
                    d["name"] for d in extend_cfg.phases["extend"]["datasets"]
                ],
            },
        }
        if extend_cfg.wandb_run_id:
            wandb_kwargs["id"] = extend_cfg.wandb_run_id
            wandb_kwargs["resume"] = "allow"
        wandb.init(**wandb_kwargs)
        print("W&B logging enabled.")

    # ── Optimizer ──────────────────────────────────────────────────
    inner_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    if extend_cfg.optimizer_name.lower() == "muon":
        from osrt.muon import (
            HybridMuonAdamW,
            Muon,
            build_param_groups,
        )

        muon_params, adamw_groups = build_param_groups(
            inner_model.named_parameters(),
            weight_decay=extend_cfg.weight_decay,
            per_head_attn=getattr(extend_cfg, "per_head_muon", False),
            head_dim=model_config.head_dim,
        )
        muon_lr = getattr(extend_cfg, "muon_lr", extend_cfg.peak_lr)
        muon = Muon(
            muon_params,
            lr=muon_lr,
            momentum=0.95,
            nesterov=True,
            weight_decay=extend_cfg.weight_decay,
        )
        adamw = torch.optim.AdamW(
            adamw_groups,
            lr=extend_cfg.peak_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        muon_min = getattr(extend_cfg, "muon_min_lr", muon_lr * 0.1)
        for pg in muon.param_groups:
            pg["_peak_lr"] = muon_lr
            pg["_min_lr"] = muon_min
        for pg in adamw.param_groups:
            pg["_peak_lr"] = extend_cfg.peak_lr
            pg["_min_lr"] = extend_cfg.min_lr
        optimizer = HybridMuonAdamW(muon, adamw)
        n_muon = sum(len(g["params"]) for g in muon.param_groups)
        n_adamw = sum(len(g["params"]) for g in adamw_groups)
        per_head = getattr(extend_cfg, "per_head_muon", False)
        print(
            f"Muon+AdamW hybrid: {n_muon} matrix tensors → Muon "
            f"(lr={muon_lr}{', per-head attn' if per_head else ''}), "
            f"{n_adamw} other tensors → AdamW (lr={extend_cfg.peak_lr})",
        )
    else:
        router_params = []
        other_params = []
        for name, param in inner_model.named_parameters():
            if not param.requires_grad:
                continue
            if "router" in name or "loop_embeddings" in name:
                router_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": extend_cfg.weight_decay},
                {"params": router_params, "weight_decay": 0.0},
            ],
            lr=extend_cfg.peak_lr,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        for pg in optimizer.param_groups:
            pg["_peak_lr"] = extend_cfg.peak_lr
            pg["_min_lr"] = extend_cfg.min_lr
        print(f"AdamW (lr={extend_cfg.peak_lr}, wd={extend_cfg.weight_decay})")

    # ── Resume from prior extend checkpoints ────────────────────────
    prefix = extend_cfg.stage_prefix
    os.makedirs(ckpt_dir, exist_ok=True)
    best_step = -1
    best_ckpt: str | None = None
    for pattern in (
        f"{ckpt_dir}/osrt_v5_{prefix}_step_*.pt",
        f"{ckpt_dir}/osrt_v5_{prefix}_rescue_step_*.pt",
    ):
        for f in glob.glob(pattern):
            try:
                s = int(f.rsplit("_", 1)[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            if s > best_step or (s == best_step and "rescue" in f):
                best_step = s
                best_ckpt = f

    start_step = 0
    if best_step > 0 and best_ckpt is not None:
        print(f"Found {prefix} checkpoint at step {best_step}: {best_ckpt}")
        ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        load_model_state_or_raise(
            model,
            ckpt["model_state_dict"],
            context=f"{prefix} resume from {best_ckpt}",
        )
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, RuntimeError) as e:
            print(f"  Optimizer state mismatch, starting fresh: {e}")
        start_step = ckpt["step"] + 1
        print(f"Resumed at step {start_step}")

    # ── Training loop (single phase, no curriculum, no held-out val) ──
    start_time = time.time()
    step = start_step
    extend_phase = extend_cfg.phases["extend"]
    seq_len = extend_phase["seq_len"]
    batch_size = extend_phase["batch_size"]
    grad_accum = extend_phase["grad_accum_steps"]

    print(
        f"\n>>> Extend phase | seq_len: {seq_len} | "
        f"batch: {batch_size} | accum: {grad_accum} | "
        f"effective batch: {batch_size * grad_accum}",
    )
    print(f"    Datasets: {[d['name'] for d in extend_phase['datasets']]}")

    load_t = time.time()
    # MOPD-style override: when extend_cfg has a rollout_dataset_path,
    # train on local Gemini-rollout JSONL instead of streaming HF
    # datasets. Same (input_ids, labels) interface — every other path
    # in run_pretrain_extend stays the same (LR schedule, MoE balance,
    # aux loop loss, telemetry, ckpts).
    rollout_path = getattr(extend_cfg, "rollout_dataset_path", None)
    if rollout_path:
        from osrt.data import make_rollout_loader

        print(f"    [MOPD] using rollout dataset: {rollout_path}")
        loader = make_rollout_loader(
            jsonl_path=rollout_path,
            seq_len=seq_len,
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            step_num=step,
            num_workers=getattr(extend_cfg, "dataloader_num_workers", 2),
            prefetch_factor=getattr(extend_cfg, "dataloader_prefetch_factor", 2),
        )
    else:
        loader = make_loader(
            extend_phase["datasets"],
            seq_len,
            tokenizer_name,
            batch_size,
            step,
            num_workers=getattr(extend_cfg, "dataloader_num_workers", 4),
            prefetch_factor=getattr(extend_cfg, "dataloader_prefetch_factor", 4),
        )
    loader_iter = iter(loader)
    print(f"    DataLoader ready in {time.time() - load_t:.1f}s")

    # Activation checkpointing: foundation needs it at seq 2048 (39.5GB);
    # the v6 model (MTP + mHC 4-stream + 8 experts) is heavier than the v5
    # extend model this loop was written for. Drive from the config when
    # set, else trigger at seq>=4096 (was 8192 — too high for v6).
    # NB: the model's ONLY checkpointing gate is _osrt_grad_ckpt (model.py
    # use_ckpt); HF's `gradient_checkpointing` name is deliberately not wired
    # up (supports_gradient_checkpointing=False). Set our private gate like
    # run_training does, NOT base.gradient_checkpointing (which the model
    # never reads).
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    base = inner.model if hasattr(inner, "model") else inner
    need_ckpt = bool(getattr(extend_cfg, "gradient_checkpointing", seq_len >= 4096))
    base._osrt_grad_ckpt = need_ckpt
    print(
        f"    Gradient checkpointing: {'ENABLED' if need_ckpt else 'disabled'} "
        f"(_osrt_grad_ckpt={need_ckpt}, seq_len={seq_len})"
    )

    # Gumbel buffer fill (no-op since extend_cfg sets tau init = 0).
    set_router_gumbel_tau(model, extend_cfg.router_gumbel_tau_init)

    # Aux-loss curriculum setup. When extend_cfg has
    # aux_loop_curriculum_steps > 0, linearly ramp the model's
    # aux_loop_loss_weight from aux_loop_weight_start → the model's
    # initial aux_loop_loss_weight over the first N steps. Lets the
    # model adapt smoothly without an initial loss shock.
    base_cfg = inner.config
    final_aux_weight = getattr(base_cfg, "aux_loop_loss_weight", 0.0)
    aux_curriculum_steps = getattr(
        extend_cfg,
        "aux_loop_curriculum_steps",
        0,
    )
    aux_curriculum_start = getattr(
        extend_cfg,
        "aux_loop_weight_start",
        final_aux_weight,
    )

    while step < extend_cfg.total_steps:
        lr = _set_param_group_lrs(optimizer, step, extend_cfg)

        # Curriculum: linearly ramp aux_loop_loss_weight on the MODEL
        # config (where forward reads it). Side-effect mutation is fine
        # since each rank has its own config object.
        if aux_curriculum_steps > 0 and final_aux_weight > 0:
            if step < aux_curriculum_steps:
                ratio = step / aux_curriculum_steps
                base_cfg.aux_loop_loss_weight = aux_curriculum_start + ratio * (
                    final_aux_weight - aux_curriculum_start
                )
            else:
                base_cfg.aux_loop_loss_weight = final_aux_weight

        optimizer.zero_grad(set_to_none=True)

        # Hoist log decision out of the micro-batch loop so we can skip
        # the .item()-heavy _collect_moe_metrics on non-logging steps
        # (review/performance-loop-audit P1). Extend has no early-stop
        # gate, so the only consumer of moe_metrics/moe_summary is the
        # logging block.
        extend_should_log = (
            step % extend_cfg.log_interval == 0
            or step == 0
            or (step < 100 and step % 10 == 0)
        )

        # Gate the .item()/.tolist() MoE telemetry inside model forward
        # on whether this step will actually consume it. Extend has no
        # early-stop gate, so the only consumer is the logging block.
        inner.set_moe_telemetry(extend_should_log)

        accum_task_loss = torch.tensor(0.0, device=device)
        accum_balance_norm = torch.tensor(0.0, device=device)
        moe_snapshots: list[dict[str, float]] = []

        if step == start_step:
            print("Fetching first batch...")
            batch_t = time.time()

        for micro in range(grad_accum):
            try:
                input_ids, labels = next(loader_iter)
            except StopIteration:
                # Loader exhausted — rebuild with a different seed.
                # Honour the MOPD rollout-path override here too.
                import gc

                loader_iter = None
                del loader
                gc.collect()
                if rollout_path:
                    from osrt.data import make_rollout_loader

                    loader = make_rollout_loader(
                        jsonl_path=rollout_path,
                        seq_len=seq_len,
                        tokenizer_name=tokenizer_name,
                        batch_size=batch_size,
                        step_num=step,
                        num_workers=getattr(extend_cfg, "dataloader_num_workers", 2),
                        prefetch_factor=getattr(
                            extend_cfg, "dataloader_prefetch_factor", 2
                        ),
                    )
                else:
                    # Thread the configured worker/prefetch settings, matching
                    # the INITIAL make_loader call above. Without this the
                    # rebuild silently reverts to make_loader's 4-worker default,
                    # re-opening too many concurrent HF streams from one
                    # container → the SSL/connection-reset storm we tuned the
                    # config down to avoid (see the initial-loader comment).
                    loader = make_loader(
                        extend_phase["datasets"],
                        seq_len,
                        tokenizer_name,
                        batch_size,
                        step,
                        num_workers=getattr(extend_cfg, "dataloader_num_workers", 4),
                        prefetch_factor=getattr(
                            extend_cfg, "dataloader_prefetch_factor", 4
                        ),
                    )
                loader_iter = iter(loader)
                input_ids, labels = next(loader_iter)

            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if step == start_step and micro == 0:
                # Fingerprint the very first batch. Across resumed sessions this
                # reveals whether the stream restarts at position 0: an identical
                # hash at *different* resume steps means the drip is silently
                # re-training the stream head (oversampling that wastes the
                # token budget). Compare `first_batch_sha` across W&B runs.
                # (docs/specs/2026-07-26-precision-and-sft-objective §14.1)
                import hashlib

                fb_sha = hashlib.sha256(
                    input_ids.detach().to("cpu").numpy().tobytes()
                ).hexdigest()[:16]
                print(
                    f"First batch fetched in {time.time() - batch_t:.1f}s | "
                    f"resume_step={start_step} first_batch_sha={fb_sha}",
                    flush=True,
                )
                if use_wandb:
                    wandb.run.summary["data/first_batch_sha"] = fb_sha
                    wandb.run.summary["data/first_batch_resume_step"] = start_step

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
                loss = outputs.loss / grad_accum
            loss.backward()

            if inner.last_task_loss is not None:
                accum_task_loss += inner.last_task_loss.detach() / grad_accum
            if inner.last_balance_loss_normalised is not None:
                accum_balance_norm += (
                    inner.last_balance_loss_normalised.detach() / grad_accum
                )
            if extend_should_log:
                micro_metrics, _ = _collect_moe_metrics(model)
                moe_snapshots.append(micro_metrics)

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            extend_cfg.grad_clip,
        )
        optimizer.step()
        apply_router_balance_updates(model)

        if extend_should_log:
            moe_metrics, moe_summary = _average_moe_snapshots(moe_snapshots)
        else:
            moe_metrics, moe_summary = {}, {}

        # ── Logging ────────────────────────────────────────────────
        should_log = extend_should_log
        if should_log:
            elapsed = time.time() - start_time
            vram_gb = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            eff_batch = batch_size * grad_accum
            steps_done = max(step - start_step, 1)
            tok_per_sec = (
                eff_batch
                * seq_len
                / max(
                    elapsed / steps_done,
                    1e-8,
                )
            )

            # Recursion telemetry: per-loop aux CE loss (if enabled) and
            # per-loop adapter contribution magnitude. The per-loop aux
            # loss tells us whether intermediate loops are learning to
            # predict next-token. The adapter magnitude tells us whether
            # those loops' adapters are growing or staying tiny. Together
            # they answer "is the recursive depth being used?" in real time.
            per_loop_aux = (
                [loss.item() for loss in inner.last_per_loop_aux_losses]
                if hasattr(inner, "last_per_loop_aux_losses")
                else []
            )
            aux_total = (
                inner.last_aux_loop_total.item()
                if getattr(inner, "last_aux_loop_total", None) is not None
                else None
            )
            base_model = inner.model if hasattr(inner, "model") else inner
            n_loops = base_model.config.recursive_loops
            n_blocks = base_model.config.num_blocks
            adapter_scale_val = base_model.adapter_scale
            per_loop_adapter_norm: list[float] = []
            with torch.no_grad():
                for loop_i in range(n_loops):
                    total = 0.0
                    for blk in range(n_blocks):
                        idx = loop_i * n_blocks + blk
                        a = base_model.adapters_a[idx]
                        b = base_model.adapters_b[idx]
                        total += adapter_scale_val * (a @ b).norm().item()
                    per_loop_adapter_norm.append(total / n_blocks)

            print(
                f"step {step:>5d}/{extend_cfg.total_steps} | "
                f"task {accum_task_loss.item():.4f} | "
                f"bal {accum_balance_norm.item():.4f} | "
                f"lr {lr:.2e} | vram {vram_gb:.1f}GB | "
                f"tok/s {tok_per_sec:,.0f}",
                flush=True,
            )
            print(
                f"           moe: pte={moe_summary['per_token_H']:.3f} "
                f"assn={moe_summary['assign_H']:.3f} "
                f"drop={moe_summary['drop_rate']:.4f} "
                f"gate={moe_summary['moe_gate']:.3f}",
                flush=True,
            )
            if per_loop_aux:
                aux_str = " ".join(f"{v:.3f}" for v in per_loop_aux)
                print(
                    f"           aux: total={aux_total:.3f}  per_loop=[{aux_str}]",
                    flush=True,
                )
            adapter_str = " ".join(f"{v:.3f}" for v in per_loop_adapter_norm)
            print(
                f"           rec: adapter_||scaled_a@b||_per_loop=[{adapter_str}]",
                flush=True,
            )
            if use_wandb:
                log_dict = {
                    "extend/task_loss": accum_task_loss.item(),
                    "extend/balance_loss_normalised": accum_balance_norm.item(),
                    "extend/lr": lr,
                    "extend/vram_gb": vram_gb,
                    "extend/tok_per_sec": tok_per_sec,
                }
                if aux_total is not None:
                    log_dict["extend/aux_loop_total"] = aux_total
                    for i, v in enumerate(per_loop_aux):
                        log_dict[f"extend/aux_loop_L{i}"] = v
                for i, v in enumerate(per_loop_adapter_norm):
                    log_dict[f"extend/adapter_norm_L{i}"] = v
                log_dict.update(moe_metrics)
                wandb.log(log_dict, step=step)
        elif step < 100:
            sys.stdout.write(".")
            sys.stdout.flush()
            if step % 25 == 24:
                sys.stdout.write(f" [step {step}]\n")
                sys.stdout.flush()

        # ── Periodic held-out eval ──────────────────────────────────
        # Ported from run_training. Required for a 9k-step run: the
        # pre->midtrain gate (review/learnings-scratchpad.md) needs an
        # eval trend. No-op for v5 stages (eval_interval defaults to
        # 9_999_999 there).
        eval_interval = getattr(extend_cfg, "eval_interval", 0)
        if eval_interval and step > 0 and step % eval_interval == 0:
            # Eval is BEST-EFFORT: it runs before the checkpoint block below,
            # so an eval failure must NOT crash the run and lose the pending
            # checkpoint (a DataLoader-worker death once took the whole process
            # down at a 500-boundary, before step_*.pt was written). Catch,
            # log, and fall through to the checkpoint save.
            try:
                eval_metrics = run_eval(
                    model,
                    tokenizer_name,
                    seq_len,
                    batch_size,
                    getattr(extend_cfg, "eval_steps", 20),
                    device,
                    model_config.real_vocab_size,
                )
                print(
                    f"  EVAL step {step} | "
                    f"loss {eval_metrics['eval/loss']:.4f} | "
                    f"ppl {eval_metrics['eval/perplexity']:.1f}",
                    flush=True,
                )
                if use_wandb:
                    wandb.log(eval_metrics, step=step)
            except Exception as e:  # noqa: BLE001 — eval must never kill training
                print(
                    f"  EVAL step {step} FAILED ({type(e).__name__}: {e}) — "
                    f"skipping eval, continuing to checkpoint.",
                    flush=True,
                )

        # ── Periodic held-out SFT eval (rollout corpus) ──────────────
        # The signal that decides when to STOP an SFT run: training loss
        # plateaus early on this model while the eval metric keeps moving
        # (midtrain3 precedent). Opt-in via `rollout_eval_path`; same
        # best-effort contract as the FineWeb eval above.
        sft_eval_path = getattr(extend_cfg, "rollout_eval_path", "")
        sft_eval_interval = getattr(extend_cfg, "rollout_eval_interval", 0)
        if (
            sft_eval_path
            and sft_eval_interval
            and step > 0
            and step % sft_eval_interval == 0
        ):
            try:
                m = run_rollout_eval(
                    model,
                    sft_eval_path,
                    tokenizer_name,
                    seq_len,
                    batch_size,
                    getattr(extend_cfg, "rollout_eval_steps", 16),
                    device,
                )
                print(
                    f"  SFT-EVAL step {step} | held-out loss "
                    f"{m['sft_eval/loss']:.4f} | ppl "
                    f"{m['sft_eval/perplexity']:.2f} | "
                    f"{m['sft_eval/tokens']} tok",
                    flush=True,
                )
                if use_wandb:
                    wandb.log(m, step=step)
            except Exception as e:  # noqa: BLE001
                print(
                    f"  SFT-EVAL step {step} FAILED "
                    f"({type(e).__name__}: {e}) — continuing.",
                    flush=True,
                )
                model.train(True)  # run_eval set eval mode; restore train mode

        # ── Numbered checkpoints ───────────────────────────────────
        if step > 0 and step % extend_cfg.ckpt_interval == 0:
            path = f"{ckpt_dir}/osrt_v5_{prefix}_step_{step}.pt"
            save_checkpoint(model, optimizer, step, path)
            vol.commit()

        # ── 23h Modal safety rescue ────────────────────────────────
        if time.time() - start_time > 82_800:
            rescue_path = f"{ckpt_dir}/osrt_v5_{prefix}_rescue_step_{step}.pt"
            save_checkpoint(model, optimizer, step, rescue_path)
            vol.commit()
            print(
                f"\n23h boundary reached at step {step}. "
                f"Rescue checkpoint saved; exiting cleanly for resume.",
                flush=True,
            )
            if use_wandb:
                wandb.finish()
            return

        step += 1

    # Final checkpoint. Also expose it under a step-numbered name (hardlink — no
    # extra disk) so the HF sync daemon (which globs `_step_*`) uploads it and
    # the resume-scan can rank it by step. `_final.pt` alone is unsyncable and
    # has no trailing step to sort on, so a completed run's tail otherwise lived
    # only on the ephemeral VM disk. Downstream configs that resume from
    # `_final.pt` keep working unchanged. (docs/specs/2026-07-26-ckpt-sync §2)
    final_path = f"{ckpt_dir}/osrt_v5_{prefix}_final.pt"
    save_checkpoint(model, optimizer, step, final_path)
    step_alias = f"{ckpt_dir}/osrt_v5_{prefix}_step_{step}.pt"
    if os.path.abspath(step_alias) != os.path.abspath(final_path) and not (
        os.path.exists(step_alias)
    ):
        try:
            os.link(final_path, step_alias)
        except OSError:
            import shutil

            shutil.copy2(final_path, step_alias)
    vol.commit()
    elapsed_total = time.time() - start_time
    print(
        f"\n{prefix} complete. {step:,} steps in {elapsed_total / 3600:.1f}h",
        flush=True,
    )
    print(f"Final checkpoint: {final_path}", flush=True)
    if use_wandb:
        wandb.finish()


def run_rollout_eval(
    model: nn.Module,
    jsonl_path: str,
    tokenizer_name: str,
    seq_len: int,
    batch_size: int,
    eval_steps: int,
    device: torch.device,
) -> dict:
    """Held-out loss on an SFT rollout JSONL (assistant tokens only).

    Why this exists: SFT-v3 was sized off a TRAINING-loss plateau (flat from
    ~step 475), and midtrain3 had already shown that signal to be unreliable
    on this model — its train loss plateaued while held-out ppl kept falling
    (math 3.06 -> 2.97, fineweb 28.32 -> 26.30). run_eval() answers a
    different question (general-LM retention on FineWeb-Edu); this answers
    "is the model still learning the SFT task distribution, or overfitting?"

    Batches are fetched once and cached, so every call scores the SAME
    held-out examples and the curve is strictly comparable across steps.
    Loss is masked to the assistant span by RolloutDataset (-100 prefix), so
    this is genuinely task loss, not prompt-reconstruction loss.
    """
    from osrt.data import make_rollout_loader

    was_training = model.training
    model.train(False)

    # Telemetry MUST be off for the eval forward. Its ~21 .item() calls per
    # MoE layer graph-break the compiled model, and worse, dynamo SPECIALISES
    # ON THE SCALAR VALUE ("___as_tensor(...).item() == -2.074575662612915"),
    # so every distinct entropy value is a fresh recompile — the v4 run blew
    # config.recompile_limit (8) on the resume frame at the first eval, after
    # which dynamo abandons that frame to eager for the rest of the run.
    # The trainer leaves telemetry ON on logging steps, and eval_interval
    # lines up with those, so this fires on every single eval unless disabled.
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    _base = inner.model if hasattr(inner, "model") else inner
    _prev_telemetry = getattr(_base, "telemetry_enabled", None)
    if hasattr(inner, "set_moe_telemetry"):
        inner.set_moe_telemetry(False)

    cache_key = ("rollout", jsonl_path, tokenizer_name, seq_len, batch_size, eval_steps)
    cached = _EVAL_BATCH_CACHE.get(cache_key)
    if cached is None:
        loader = make_rollout_loader(
            jsonl_path=jsonl_path,
            seq_len=seq_len,
            tokenizer_name=tokenizer_name,
            batch_size=batch_size,
            step_num=999999,  # fixed seed; never collides with training seeds
            num_workers=0,  # same rationale as run_eval: no worker spawn
        )
        data_iter = iter(loader)
        cached = []
        for _ in range(eval_steps):
            try:
                cached.append(next(data_iter))
            except StopIteration:
                break
        _EVAL_BATCH_CACHE[cache_key] = cached
        import gc

        del data_iter
        del loader
        gc.collect()

    total_loss = 0.0
    total_tokens = 0
    # no_grad is LOAD-BEARING, not a micro-optimisation: without it the eval
    # forward builds a full autograd graph (activations retained for a backward
    # that never comes) ON TOP of the live training allocation, and the run OOMs
    # mid-eval. Observed on the sft_v4 sanity: training sat at 41GB, the first
    # eval tried to allocate into a 79GB-full GPU and threw. The eval is
    # wrapped in try/except upstream, so this failed SILENTLY apart from one log
    # line — losing exactly the signal this function exists to provide.
    with torch.no_grad():
        for input_ids, labels in cached:
            input_ids = input_ids.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, labels=labels)
            n_tokens = (labels != -100).sum().item()
            total_loss += outputs.loss.item() * n_tokens
            total_tokens += n_tokens

    if was_training:
        model.train(True)
    # Restore whatever the trainer had set, so the logging step that follows
    # still gets its MoE diagnostics.
    if _prev_telemetry is not None and hasattr(inner, "set_moe_telemetry"):
        inner.set_moe_telemetry(_prev_telemetry)
    # Release the eval activations so training's allocator doesn't have to
    # fight fragmentation for the next 100 steps.
    torch.cuda.empty_cache()

    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "sft_eval/loss": mean_loss,
        "sft_eval/perplexity": math.exp(min(mean_loss, 20.0)),
        "sft_eval/tokens": total_tokens,
    }

