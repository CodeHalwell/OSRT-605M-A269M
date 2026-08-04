"""OSRT — Modal deployment entrypoint.

~363M physical params (32K vocab + 1536 dim), ~192M active/token (top-2 of 8
routed experts + shared expert), ~1.15B effective via recursive weight sharing.
3 physical blocks × 6 loops = 18 effective layers.
Mixtral-style MoE: no dense FFN, 1 shared + 8 routed experts (top-2), Switch
balance loss, orthogonal per-expert init, eval-time drop-free capacity.

Reuses the v4 tokenizer volume (osrt-v4-tokenizer — same 32K BPE vocab and
structural tags). v5 keeps its own checkpoint volume (osrt-checkpoints)
so v4 and v5 can coexist during the transition.

Stages:
    modal run app_v5.py --stage sanity       200-step smoke test (~$1, ~20 min)
    modal run app_v5.py --stage pretrain     Full 300K-step pretrain
"""

import modal

# =============================================================================
# MODAL INFRASTRUCTURE
# =============================================================================

app = modal.App("osrt")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .env({
        "TORCH_LOGS": "perf_hints",
        "PYTHONUNBUFFERED": "1",
        # Disable HF tokenizers-rs thread pool before fork. Otherwise
        # DataLoader(num_workers=2) deadlocks when the child inherits a
        # locked mutex whose owning thread no longer exists. Confirmed
        # failure mode: sanity run stuck at "Fetching first batch..."
        # for 45 min with no output until manually stopped.
        "TOKENIZERS_PARALLELISM": "false",
        # CUDA allocator: expandable segments prevent fragmentation OOMs.
        # The recursive + checkpointed + chunked-CE training churns the
        # allocator with variable-size tensors (per-step attention scores,
        # checkpoint recompute, fused-CE chunks); over many steps the default
        # caching allocator fragments and can fail to find a contiguous block
        # even with ~15GB free (observed: OOM at step ~27 with only 60/79GB
        # actually allocated). expandable_segments grows segments in place
        # instead of fragmenting. PyTorch's own OOM message recommends this.
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # Persistent HF datasets cache. Volume mounted by SFT/eval/GRPO
        # functions; pretrain doesn't mount it, but HF datasets handles
        # a non-existent path gracefully under streaming=True (the
        # iterable doesn't touch the cache; only metadata downloads do,
        # and those mkdir the path on demand within the container's
        # writable layer if no volume is mounted there).
        "HF_DATASETS_CACHE": "/vol/hf_cache",
    })
    .pip_install(
        "torch==2.10.0+cu128",
        extra_options="--index-url https://download.pytorch.org/whl/cu128",
    )
    .pip_install(
        # Keep Modal aligned with the pinned local/Lightning lockfile. The
        # CUDA torch wheel is installed above from the PyTorch index; these are
        # the PyPI-side packages that otherwise drift when the image rebuilds.
        "transformers==5.3.0",
        "datasets==4.6.1",
        "lion-pytorch==0.2.4",
        "triton==3.6.0",
        "wandb==0.25.1",
        "tokenizers==0.22.2",
        "sentencepiece==0.2.0",
        "safetensors==0.7.0",
        # lm-eval baked into the base image so the `evaluate` function
        # doesn't need a derived image. Modal disallows .pip_install
        # after .add_local_dir; folding it in here keeps the build chain
        # linear (env → apt → pip → add_local) so any function can
        # evaluate without an image rebuild.
        #
        # [ifeval] extras: pulls in langdetect + immutabledict + nltk
        # which IFEval's instruction graders rely on. Without these,
        # `lm_eval.simple_evaluate(tasks=["ifeval", ...])` fails at
        # task-config load time with "ModuleNotFoundError: No module
        # named 'langdetect'" before the model even runs.
        "lm-eval[ifeval]==0.4.11",
    )
    .add_local_dir("src/osrt", remote_path="/root/osrt")
    .add_local_dir("scripts", remote_path="/root/scripts")
    # Bake the tokenizer into the image so workspace-portable functions (e.g.
    # ppl_eval, which pulls the checkpoint from HF) need no tokenizer volume.
    .add_local_dir("v6_tokenizer_export", remote_path="/root/v6_tokenizer_export")
)

# v5 gets its own checkpoint volume so we can run v4 and v5 in parallel.
# Tokenizer volume is shared with v4 (same 32K BPE).
vol = modal.Volume.from_name("osrt-checkpoints", create_if_missing=True)
tokenizer_vol = modal.Volume.from_name(
    "osrt-v4-tokenizer", create_if_missing=True,
)
# Persistent HF datasets cache. First run downloads dataset shards from
# the Hub into this volume; subsequent runs read from local volume
# storage, which removes Hub round-trips and the latency variance that
# caused 20→75 sec/step swings during SFT-long. Streaming mode bypasses
# the dataset cache for the iterable itself, but it still uses this
# directory for split metadata, dataset_info.json, and any non-streamed
# auxiliary downloads — small but noticeable wins.
#
# HF_DATASETS_CACHE=/vol/hf_cache is set in the base image's env block
# (above) so the datasets library auto-discovers it. SFT/eval/GRPO
# functions mount this volume; pretrain skips the mount and HF falls
# back gracefully under streaming=True.
hf_cache_vol = modal.Volume.from_name(
    "osrt-hf-cache", create_if_missing=True,
)

# MOPD rollout volume — holds Gemini teacher-rollout JSONL collected
# via scripts/collect_rollouts.py. Uploaded from local before launching
# the mopd stage. Per-workspace; create_if_missing so first launch on a
# fresh workspace works without manual setup.
rollouts_vol = modal.Volume.from_name(
    "osrt-rollouts", create_if_missing=True,
)

# v6 tokenizer volume — the 65K / 21-token contract tokenizer. Kept
# SEPARATE from the v4 32K tokenizer (osrt-v4-tokenizer) so retraining
# doesn't clobber the archived v4/v5 artifact. Pretrain switches to this
# once the tokenizer is verified.
v6_tokenizer_vol = modal.Volume.from_name(
    "osrt-v6-tokenizer", create_if_missing=True,
)


# =============================================================================
# TOKENIZER (v6 — 65K / 21-token contract)
# =============================================================================


@app.function(
    gpu="T4",
    cpu=8.0,
    memory=32768,
    image=image,
    volumes={
        "/vol/tokenizer_out": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=6 * 3600,
)
def tokenizer(vocab_size: int = 65536, sample_size: int = 3_000_000_000):
    """Train the v6 65K BPE tokenizer (21-token contract + reserved band)
    and write it to the osrt-v6-tokenizer volume.

    Runs on a cheap T4 box (~$0.59/h). The HF byte-level BPE trainer is
    CPU-bound — the T4 mostly idles — but it's the lowest-cost Modal GPU
    tier and keeps this off the H100 pool. Streams ~3GB of FineWeb-Edu +
    CodeParrot + Wikipedia from HF (with retry), trains, verifies, and
    commits the volume. ~1-3 h. See scripts/train_tokenizer.py for the
    special-token contract.
    """
    import sys

    sys.path.insert(0, "/root/scripts")
    import train_tokenizer as tt

    out_dir = "/vol/tokenizer_out"
    print(
        f"v6 tokenizer: vocab={vocab_size:,}, "
        f"sample={sample_size:,} chars (~{sample_size / 1e9:.1f} GB)"
    )
    data_path = tt.sample_training_data(sample_size)
    tt.train_with_hf_tokenizers(data_path, vocab_size, out_dir)

    v6_tokenizer_vol.commit()
    print(f"\nv6 tokenizer committed to osrt-v6-tokenizer volume ({out_dir}).")


@app.local_entrypoint()
def run_tokenizer(vocab_size: int = 65536, sample_size: int = 3_000_000_000):
    """Spawn the v6 tokenizer training (fire-and-forget).

    Launch with: modal run --detach app.py::run_tokenizer
    spawn() submits the job and returns a handle immediately; the call
    runs to completion on Modal independent of this local client (the
    right pattern for long stages — see project memory). Poll progress
    with: modal app logs <app-id>.
    """
    call = tokenizer.spawn(vocab_size=vocab_size, sample_size=sample_size)
    print(f"Spawned v6 tokenizer training — call_id={call.object_id}")
    print("Runs on a T4 independent of this client. Tail logs with:")
    print("  modal app logs <app-id>   (app id shown above)")


# ALL Modal runs go through a spawn entrypoint (project rule: only ever use
# .spawn(), never `modal run --detach app.py::<func>` directly). Launch with
# `modal run --detach app.py::run_<stage>`, then `modal app logs ap-<id>`.


@app.local_entrypoint()
def run_pretrain_sanity():
    """Spawn the full-footprint memory check (eager, fire-and-forget)."""
    call = pretrain_sanity.spawn()
    print(f"Spawned pretrain mem-check — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_pretrain_compile_check():
    """Spawn the torch.compile validation (compile on, 40 steps so tracing
    amortises and steady-state tok/s is meaningful)."""
    call = pretrain_sanity.spawn(compile_on=True, steps=40)
    print(f"Spawned pretrain compile-check — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_flash_experiment():
    """FLASH EXPERIMENT (advisor-designed decider): flip BOTH attention_sink
    and gradient_checkpointing OFF, compile on, 40 steps. Reads three outcomes:
      • fits 80GB + faster  → flash SDPA wins, ship the 1-line preset flip
      • OOMs without ckpt    → checkpointing needed either way → skip flash, do B4
      • fits, not faster     → MoE .nonzero() graph break is the ceiling → B4
    Compare vram + steady tok/s against the sink+ckpt compile-check (~68.7GB)."""
    call = pretrain_sanity.spawn(
        compile_on=True, steps=40, attention_sink=False, grad_ckpt=False,
    )
    print(f"Spawned FLASH experiment — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_grouped_check():
    """B4 acceptance run: grouped-GEMM MoE + compile, real config (sink + ckpt
    + fused-CE), 40 steps. Gates:
      • compiles fullgraph (the MoE .nonzero() break is gone)
      • loss tracks the loop baseline (11.39→7.27 over 32 steps, same config)
      • steady tok/s beats the loop+compile baseline (~5.3k)
    If loss diverges from the baseline, the grouped dispatch has a training-path
    bug (the kernel backward, untestable on CPU) — do NOT ship; revert the flag.
    """
    call = pretrain_sanity.spawn(compile_on=True, steps=40, grouped=True)
    print(f"Spawned B4 grouped-GEMM check — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_seq8192_check():
    """Long-context memory check: the REAL phase-3 footprint (seq 8192, batch 2)
    + grouped + sink + checkpointing, eager, 12 steps. Eager because the binding
    question is memory: at seq 8192 the manual sink materialises a (B,H,S,S)
    score matrix (~13-25GB transient even at batch 2) — identical compiled vs
    eager, and eager skips the slow 8192 compile trace.

    Gate: fits 80GB. If it OOMs in _attention_with_sink, the sink doesn't scale
    to long context → phase 3 needs attention_sink=False (flash SDPA, which
    never materialises scores). If 8192 fits, seq 4096 (smaller) is safe too."""
    call = pretrain_sanity.spawn(
        compile_on=False, steps=12, grouped=True, seq_len=8192, batch=2,
    )
    print(f"Spawned seq-8192 mem-check — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_seq8192_flash_check():
    """Same as run_seq8192_check but attention_sink=False (flash SDPA). The sink
    OOMs at seq 8192 (the (B,H,S,S) score recompute in backward); flash never
    materialises scores, so this confirms whether dropping the sink fixes the
    long-context fit. If it does, the preset goes attention_sink=False."""
    call = pretrain_sanity.spawn(
        compile_on=False, steps=12, grouped=True, seq_len=8192, batch=2,
        attention_sink=False,
    )
    print(f"Spawned seq-8192 FLASH mem-check — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.function(
    image=image,
    volumes={"/vol/tokenizer": v6_tokenizer_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=900,
)
def smoke_new_datasets():
    """Stream-test the candidate Nemotron/Cosmopedia datasets IN the Modal env:
    proves the hf-secret token has the gated grants AND that text extraction +
    v6 tokenization work for each. CPU-only, ~2 min. Skips Nemotron-Code-Metadata
    (no text field). Run before wiring them into the full pretrain mix."""
    import os

    from datasets import load_dataset
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")
    has_tok = bool(
        os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    )
    print(f"HF token present in container: {has_tok}", flush=True)

    candidates = [
        ("nvidia/Nemotron-CC-Math-v1", "4plus", "text"),
        ("nvidia/Nemotron-CC-Math-v1", "4plus_MIND", "text"),
        ("nvidia/Nemotron-Pretraining-Code-v2", "Synthetic-Question-Answering", "content"),
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-STEM-SFT", "text"),
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Math-Textbooks", "text"),
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-InfiniByte-Reasoning", "text"),
        ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-RQA", "text"),
        ("HuggingFaceTB/cosmopedia", "web_samples_v2", "text"),
        ("HuggingFaceTB/cosmopedia", "openstax", "text"),
    ]
    ok = 0
    for repo, cfg, field in candidates:
        short = f"{repo.split('/')[-1]}/{cfg}"
        try:
            ds = load_dataset(repo, cfg, split="train", streaming=True)
            ex = next(iter(ds))
            text = ex.get(field) or ""
            n = len(tok(text[:4000]).input_ids)
            if text:
                ok += 1
                print(
                    f"OK   {short}: field={field!r} chars={len(text)} "
                    f"tok(4k)={n} :: {text[:55]!r}",
                    flush=True,
                )
            else:
                print(f"EMPTY {short}: field {field!r} was empty", flush=True)
        except Exception as e:
            print(f"FAIL {short}: {type(e).__name__}: {str(e)[:130]}", flush=True)
    print(f"\nSMOKE RESULT: {ok}/{len(candidates)} streamed + tokenized OK", flush=True)


@app.local_entrypoint()
def run_smoke_new_datasets():
    """Spawn the gated-dataset streaming smoke test (proves Modal hf-secret
    access before wiring the new datasets into the full pretrain)."""
    call = smoke_new_datasets.spawn()
    print(f"Spawned dataset smoke test — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_final_check():
    """Final pre-launch integration check: the REAL pretrain config (flash +
    grouped-GEMM + checkpointing + fused-CE), compiled, 20 steps. One short
    end-to-end pass to confirm the whole stack trains before committing to the
    full run — exercises fullgraph compile, collapse telemetry, data streaming,
    and the checkpoint-save gate (prints 'skipped'). ~8 min."""
    call = pretrain_sanity.spawn(compile_on=True, steps=20, grouped=True)
    print(f"Spawned final pre-launch check — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_pretrain():
    """Spawn the full v6 pretraining run (fire-and-forget)."""
    call = pretrain.spawn()
    print(f"Spawned v6 pretrain — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


# =============================================================================
# PRE-TRAINING
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def pretrain():
    """Run v6 pre-training with progressive seq_len curriculum.

    Uses the v6 65K/21-token tokenizer (osrt-v6-tokenizer) + the memory
    fixes (fused linear-CE + gradient checkpointing) so the full batch/seq
    fits an 80GB H100.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.train import run_training
    from osrt.train_config import PretrainConfig

    _tok_vol = _modal.Volume.from_name("osrt-v6-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path

    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    expected_vocab = 65536
    if len(tok) != expected_vocab:
        print(f"WARNING: Expected {expected_vocab} vocab but got {len(tok)}!")
        print("  Retrain tokenizer: modal run app_v4.py --stage tokenizer")

    # Build from the canonical OSRT-605M-A279M preset (GQA, attention sink,
    # MTP, sqrt-softplus routing, the 4032/2816 expert widths, etc.) and only
    # override the tokenizer-specific fields. A bare OSRTConfig() here would
    # silently fall back to the old v5 363M shape for the expensive run.
    from osrt.presets import build_config
    model_config = build_config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        # Memory: the 8 full-vocab fp32 logit tensors (1 main + 5 aux + 2 MTP)
        # + the mHC 4× residual stream OOM an 80GB H100 at batch 8 / seq 2048
        # without these. fused CE routes the 7 aux/MTP head losses through a
        # chunked, gradient-checkpointed linear-CE; gradient checkpointing
        # recomputes the (weight-shared) recursive blocks in backward.
        fused_cross_entropy_chunks=8,
    )

    train_cfg = PretrainConfig()
    # Memory: drive gradient checkpointing from the TRAIN config (HF can't
    # reset this one — see run_training). fused-CE is on via model_config.
    train_cfg.gradient_checkpointing = True

    # Target budget (see compute_budget.py): ~601M physical / ~278M active per
    # token; ~2.5B FLOPs-equivalent via the 6 recursive loops.
    print("Target budget: ~601M physical / ~278M active per token.")

    run_training(model_config, train_cfg, vol, tokenizer_name)


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=7200,
)
def pretrain_sanity(
    compile_on: bool = False,
    steps: int = 30,
    attention_sink: bool = False,  # matches the preset (sink dropped; flash)
    grad_ckpt: bool = True,
    grouped: bool = False,
    seq_len: int = 0,
    batch: int = 0,
):
    """Full-footprint check of the pretrain path on the v6 65K tokenizer.

    Runs the REAL batch 8 / seq 2048 with fused linear-CE. compile_on=False
    (default) is the fast eager MEMORY check; compile_on=True is the
    torch.compile validation — confirms the model compiles without erroring
    (mHC Sinkhorn loop, the data-dependent MoE .nonzero() dispatch,
    gradient-checkpoint + fused-CE nesting are the risky parts) and measures
    the steady-state tok/s speedup vs eager (~4.5k). `steps` sets total_steps
    (use more for compile so tracing amortises and steady tok/s is
    meaningful). No ckpt/eval/wandb.

    attention_sink / grad_ckpt override the preset's sink (True) and the
    checkpointing flag (True) for THIS run only — the FLASH EXPERIMENT flips
    both False to test whether flash SDPA (no manual (B,H,S,S) materialise)
    still fits 80GB without checkpointing, and whether that's faster. The
    preset is untouched; we override on the config object.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from osrt.train import run_training
    from osrt.train_config import PretrainConfig

    _tok_vol = _modal.Volume.from_name("osrt-v6-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path
    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    # FULL-FOOTPRINT memory validation on the REAL v6 65K tokenizer. The
    # config enables both memory fixes (fused linear-CE on the 7 aux/MTP
    # heads + gradient checkpointing on the recursive blocks); with them on,
    # the real batch 8 / seq 2048 should fit an 80GB H100. An earlier sanity
    # had to shrink to batch 1 / seq 1024 to dodge the OOM — if this survives
    # 30 steps at full batch/seq, blocker #2 is cleared.
    from osrt.presets import build_config
    model_config = build_config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
    )
    # Override sink for this run only (preset stays True). attention_sink=False
    # routes every attention through F.scaled_dot_product_attention (flash) and
    # never builds the (B,H,S,total_len) score matrix.
    model_config.attention_sink = attention_sink
    # B4: grouped-GEMM MoE for this run only. True → fullgraph compile (the MoE
    # .nonzero() is the only break); validate the loss tracks the loop baseline.
    model_config.moe_grouped_gemm = grouped

    # Subclass so the real PretrainConfig is untouched. Real batch_size/seq
    # (8 / 2048 foundation). ckpt/eval/early-stop pushed past total_steps;
    # eager (compile off) so steps appear immediately and we isolate "does it
    # fit + learn" from compile; wandb off.
    class SanityCfg(PretrainConfig):
        total_steps = steps
        warmup_steps = 5
        grad_accum_steps = 2  # fewer micro-batches → quicker steps; peak mem
                              # is per-micro-batch so this doesn't change the
                              # memory test (still batch_size=8 at seq 2048)
        log_interval = 2
        ckpt_interval = 999_999
        eval_interval = 999_999
        early_stop_check_step = 999_999
        save_final_checkpoint = False  # don't clobber a real final with throwaway
        compile_enabled = compile_on
        wandb_log = False
        wandb_run_name = (
            "osrt-pretrain-compilecheck" if compile_on else "osrt-pretrain-memcheck"
        )
        # HF-immune gradient-checkpointing flag (run_training reads train_cfg)
        gradient_checkpointing = grad_ckpt

        # Long-context memory check: override the foundation phase to the target
        # seq_len/batch (reusing its datasets — peak memory is data-independent)
        # so we test e.g. the real phase-3 footprint (seq 8192, batch 2) from
        # step 0. grad_accum=2 keeps steps quick; peak memory is per-micro-batch
        # so it's unchanged by accum.
        if seq_len:
            _phases = {k: dict(v) for k, v in PretrainConfig.phases.items()}
            _phases["foundation"]["seq_len"] = seq_len
            _phases["foundation"]["batch_size"] = batch
            _phases["foundation"]["grad_accum_steps"] = 2
            phases = _phases
            batch_size = batch

    sanity_cfg = SanityCfg()

    mode = "torch.compile" if compile_on else "eager"
    _sq = seq_len if seq_len else 2048
    _bs = batch if batch else 8
    print(
        f"pretrain {'COMPILE-CHECK' if compile_on else 'MEM-CHECK'}: {steps} "
        f"steps, {mode}, batch={_bs} seq={_sq} | fused-CE on | "
        f"attention_sink={attention_sink} "
        f"({'flash SDPA' if not attention_sink else 'manual (B,H,S,S) sink'}) | "
        f"gradient_checkpointing={grad_ckpt} | "
        f"moe={'grouped-GEMM (fullgraph)' if grouped else 'loop dispatch'}. "
        + ("Confirming compile works + measuring tok/s speedup vs eager (~4.5k)."
           if compile_on else
           "Verifying the real footprint fits an 80GB H100 and trains.")
    )
    run_training(model_config, sanity_cfg, vol, tokenizer_name)


# =============================================================================
# PRETRAIN_EXTEND (continued pretraining / "mid-training" on top of SFT ckpt)
# =============================================================================
#
# Loads osrt_v5_sft_ultralong_final.pt, injects + freezes HRA, then runs
# 1,800 steps of continued pretraining at seq 4096 with a math/science/
# code-heavy mix plus 25 % SFT-formatted rehearsal data to combat
# chat-format forgetting. Output: osrt_v5_extend_step_N.pt and
# osrt_v5_extend_final.pt (distinct prefix so resume scans don't
# collide with base pretrain ckpts). See PretrainExtendConfig +
# train.py::run_pretrain_extend for the design rationale.
#
# Mounts the HF cache volume so the new datasets (Nemotron-CC-Math,
# RedPajama-arxiv, the-stack-smol, plus the existing FineWeb-Edu and
# Wikipedia) populate /vol/hf_cache on first run and reuse the cache
# on subsequent runs.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def pretrain_extend():
    """Continued pretraining on top of the SFT-ultralong checkpoint.

    See train_config.py::PretrainExtendConfig for the full design
    rationale (lineage decision, LR schedule, rehearsal mix, HRA
    freeze). Single phase, seq 4096, ~1,800 steps, ~$30 of H100
    time, ~485M new pretrain tokens concentrated in the
    math/science/code categories the original pretrain missed.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import PretrainExtendConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path

    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    extend_cfg = PretrainExtendConfig()
    print(
        "pretrain_extend: 1,800 steps at seq 4096, peak LR 1.5e-5, "
        "HRA frozen, 25 % SFT-formatted rehearsal mix.",
    )
    print(
        f"Resume base: {extend_cfg.pretrained_checkpoint}",
    )

    run_pretrain_extend(model_config, extend_cfg, vol, tokenizer_name)


# =============================================================================
# MIDTRAIN — v6 mid-training (continued pretraining, seq 4096, math mix)
# =============================================================================
# Generalizes run_pretrain_extend via hra_native=True (skip inject_hra: v6
# HRA is native + already in the foundation ckpt). See MidtrainConfig and
# docs/superpowers/specs/2026-06-09-v6-midtraining-design.md.


def _run_midtrain(cfg_cls):
    """Shared body for midtrain + midtrain_sanity (differ only by config)."""
    from transformers import AutoTokenizer

    from osrt.presets import build_config
    from osrt.train import run_pretrain_extend

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    model_config = build_config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
    )
    cfg = cfg_cls()
    print(
        f"{cfg.__class__.__name__}: {cfg.total_steps} steps @ seq "
        f"{cfg.phases['extend']['seq_len']}, peak LR {cfg.peak_lr}, "
        f"HRA native+trainable, resume {cfg.pretrained_checkpoint}"
    )
    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def midtrain():
    """v6 mid-training: continued pretraining from the foundation base,
    seq 4096, math-heavy mix, ~9k steps. See MidtrainConfig."""
    from osrt.train_config import MidtrainConfig
    _run_midtrain(MidtrainConfig)


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def midtrain_sanity():
    """30-step VRAM/throughput probe at real seq 4096 / batch 6 before
    the $150 launch. See MidtrainSanityConfig."""
    from osrt.train_config import MidtrainSanityConfig
    _run_midtrain(MidtrainSanityConfig)


@app.local_entrypoint()
def run_midtrain():
    """Spawn v6 mid-training (fire-and-forget)."""
    call = midtrain.spawn()
    print(f"Spawned v6 midtrain — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_midtrain_sanity():
    """Spawn the 30-step v6 midtrain VRAM/throughput sanity probe."""
    call = midtrain_sanity.spawn()
    print(f"Spawned v6 midtrain sanity — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


# =============================================================================
# SFT v2 — reasoning distillation from the v6 midtrain base
# =============================================================================
# Reuses run_pretrain_extend (the MOPD rollout-loader path) on the v6 midtrain
# base, training the long coherent teacher CoT in /vol/rollouts/sft_v2.jsonl
# with reasoning-on/off personas. Upload the corpus first (regenerated locally
# via scripts/build_sft_v2_data.py):
#   modal volume put osrt-rollouts rollouts/sft_v2.jsonl sft_v2.jsonl
# See SFTv2Config.
def _run_sft_v2(cfg_cls):
    """Shared body for sft_v2 + sft_v2_sanity (differ only by config)."""
    import os

    from transformers import AutoTokenizer

    from osrt.presets import build_config
    from osrt.train import run_pretrain_extend

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    cfg = cfg_cls()
    model_config = build_config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
        # Thread loop dropout. SFTv2Config declares loop_dropout_prob=0.10 but
        # it was never passed here, so sft_v2 silently trained at the
        # build_config default while every other post-training stage (loop_fix,
        # system_sft, mopd) threads its own. (docs/specs/2026-07-26-precision §5.2)
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    if not os.path.exists(cfg.rollout_dataset_path):
        raise FileNotFoundError(
            f"SFT-v2 corpus not found at {cfg.rollout_dataset_path}. Upload it: "
            "`modal volume put osrt-rollouts rollouts/sft_v2.jsonl sft_v2.jsonl`"
        )
    print(
        f"{cfg.__class__.__name__}: {cfg.total_steps} steps @ seq "
        f"{cfg.phases['extend']['seq_len']}, peak LR {cfg.peak_lr}, "
        f"HRA native+trainable, resume {cfg.pretrained_checkpoint}, "
        f"rollout {cfg.rollout_dataset_path}"
    )
    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
        "/vol/rollouts": rollouts_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_v2():
    """v6 SFT v2: reasoning distillation from midtrain_final, seq 4096,
    ~1500 steps on the persona-tagged teacher CoT. See SFTv2Config."""
    from osrt.train_config import SFTv2Config
    _run_sft_v2(SFTv2Config)


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
        "/vol/rollouts": rollouts_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_v2_sanity():
    """30-step SFT-v2 probe: rollout loader builds the v6 seq, native-HRA
    loads clean from midtrain_final, VRAM fits at seq 4096. See SFTv2SanityConfig."""
    from osrt.train_config import SFTv2SanityConfig
    _run_sft_v2(SFTv2SanityConfig)


@app.local_entrypoint()
def run_sft_v2():
    """Spawn v6 SFT v2 (fire-and-forget)."""
    call = sft_v2.spawn()
    print(f"Spawned v6 SFT v2 — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_sft_v2_sanity():
    """Spawn the 30-step v6 SFT-v2 sanity probe."""
    call = sft_v2_sanity.spawn()
    print(f"Spawned v6 SFT v2 sanity — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=3600,
)
def sft_eval(ckpt_name: str, n: int = 100, max_new_tokens: int = 400) -> dict:
    """Scored reasoning-on/off GSM8K eval of ONE SFT checkpoint, on GPU.

    Loads /vol/checkpoints/v5/<ckpt_name>, runs run_reasoning_eval (batched
    generation, greedy + rep_penalty 1.2 = the locked decode hygiene), returns
    the sft_eval/* metric dict. GPU-side by design — the model is far too heavy
    for local MPS/CPU inference (it froze a Mac). Cheap: ~2-3 min on H100.
    """
    import torch
    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config
    from osrt.sft_eval import run_reasoning_eval

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")
    cfg = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )
    model = OSRTForCausalLM(cfg)
    path = f"/vol/checkpoints/v5/{ckpt_name}"
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.to("cuda").to(torch.bfloat16).eval()
    print(f"loaded {path} | missing={len(missing)} unexpected={len(unexpected)}",
          flush=True)

    with torch.no_grad():
        res = run_reasoning_eval(
            model, tok, torch.device("cuda"),
            n_problems=n, max_new_tokens=max_new_tokens,
            batch_size=32, repetition_penalty=1.2,
        )
    print("=== sft_eval ===", flush=True)
    for k, v in res.items():
        print(f"  {k}: {v}", flush=True)
    return res


@app.function(
    gpu="H100", image=image,
    volumes={"/vol/checkpoints": vol, "/vol/tokenizer": v6_tokenizer_vol,
             "/vol/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret")], timeout=1800,
)
def sft_sample(ckpt_name: str, n: int = 3, max_new_tokens: int = 768) -> None:
    """Print N full ON + OFF generations from an SFT checkpoint (GPU-side).

    Qualitative failure-mode read: is the model coherent-but-wrong (GRPO can
    reinforce) or degenerate (GRPO can't)? Stops at <|/answer|> so we also see
    whether a decode-time stop fixes the no-EOS runaway.
    """
    import torch
    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config
    from osrt.sft_eval import _load_gsm8k_heldout
    from osrt.system_prompts import sample_system_prompt
    import random

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")
    cfg = build_config(vocab_size=len(tok), real_vocab_size=len(tok),
                       bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
                       pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8)
    model = OSRTForCausalLM(cfg)
    ck = torch.load(f"/vol/checkpoints/v5/{ckpt_name}", map_location="cpu",
                    weights_only=True)
    model.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    model = model.to("cuda").to(torch.bfloat16).eval()
    _, on_sys = sample_system_prompt(random.Random(0), "on")
    _, off_sys = sample_system_prompt(random.Random(0), "off")

    for i, (q, gold) in enumerate(_load_gsm8k_heldout(n), 1):
        print(f"\n{'='*70}\n[Q{i}] {q}\nGOLD: {gold}", flush=True)
        for side, sysp in (("ON", on_sys), ("OFF", off_sys)):
            ids = torch.tensor([tok.encode(f"<|system|>{sysp}<|user|>{q}<|assistant|>",
                                add_special_tokens=False)], device="cuda")
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=max_new_tokens,
                                     temperature=0.0, repetition_penalty=1.2,
                                     eos_token_id=tok.eos_token_id)
            txt = tok.decode(out[0, ids.shape[1]:], skip_special_tokens=False)
            # trim at first <|/answer|> to show the intended stop point
            cut = txt.find("<|/answer|>")
            shown = txt[:cut + len("<|/answer|>")] if cut != -1 else txt
            print(f"\n--- {side} ({len(txt)} chars raw, "
                  f"{'closed' if cut!=-1 else 'NO CLOSE'}) ---\n{shown}", flush=True)


@app.local_entrypoint()
def run_sft_sample(step: str = "final", n: int = 3):
    """Print full generations from an SFT-v2 checkpoint (qualitative read)."""
    name = ("osrt_v5_sft_v2_final.pt" if step == "final"
            else f"osrt_v5_sft_v2_step_{step}.pt")
    sft_sample.remote(name, n)


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/vol/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=3600,
)
def ppl_eval(
    step: str = "latest",
    dataset: str = "math",
    eval_steps: int = 40,
    batch: int = 6,
    skip: int | None = None,
    hf_repo: str = "HallD/osrt-v6-ckpt",
) -> dict:
    """Held-out perplexity of ONE base checkpoint, on GPU.

    Pulls the checkpoint straight from the HF repo (no volume needed) and the
    tokenizer from the image, so this runs on any fresh workspace with only
    `hf-secret`. Reads a held-out slice from a `skip` offset PAST the training
    budget, so the samples are genuinely unseen. This is the honest "is
    pretraining helping" signal for a BASE checkpoint — GSM8K/capability waits
    for the SFT stage (the base can't follow the Q->A format). Does NOT touch
    the running Colab drip. Same math as notebook cell 6b.

    `step`: "latest" (highest step/rescue on HF) or a number like "8700".
    `dataset`:
      - "math"    → Nemotron-CC-Math 4plus, skip=2M (in-distribution; the
                    dominant reasoning source; ppl ~3 is normal — low-entropy).
      - "fineweb" → FineWeb-Edu, skip=100M (general web; comparable to the
                    ~28-30 midtrain2 baseline; the skip build costs ~10-20 min).
    """
    import math
    import re

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    from osrt.data import make_loader
    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config

    files = HfApi().list_repo_files(hf_repo, repo_type="model")
    ckpts = [f for f in files
             if re.match(r"osrt_v5_midtrain3_(?:rescue_)?step_\d+\.pt$", f)]
    if step == "latest":
        name = max(ckpts, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    else:
        cand = [f for f in ckpts if re.search(rf"step_{step}\.pt$", f)]
        if not cand:
            raise FileNotFoundError(
                f"step {step} not on {hf_repo}; available: {sorted(ckpts)}"
            )
        name = cand[0]
    resolved = int(re.search(r"step_(\d+)", name).group(1))
    print(f"pulling {name} from {hf_repo}...", flush=True)
    path = hf_hub_download(hf_repo, name, repo_type="model")

    tok = AutoTokenizer.from_pretrained("/root/v6_tokenizer_export")
    cfg = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )
    device = torch.device("cuda")
    model = OSRTForCausalLM(cfg).to(device)  # eager; autocast bf16 in forward
    sd = torch.load(path, map_location=device, weights_only=True)["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (
        f"state mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
    )
    model.eval()

    if dataset == "math":
        ds_cfg = {
            "name": "nemotron-cc-math-heldout",
            "hf_id": "nvidia/Nemotron-CC-Math-v1",
            "hf_config": "4plus",
            "weight": 1.0,
            "skip": skip if skip is not None else 2_000_000,
        }
        label = "Nemotron-CC-Math (in-distribution)"
    elif dataset == "fineweb":
        # skip=5M: training has consumed only ~360k FineWeb records by step
        # ~8800 (0.15 weight), so 5M is ~14x past that = safely held-out, and
        # ~20x faster than 100M (which timed out the 60-min job on Modal's HF
        # streaming). Different held-out slice than the old 100M baseline, but
        # FineWeb is homogeneous enough that the ppl is comparable.
        ds_cfg = {
            "name": "fineweb-edu-heldout",
            "hf_id": "HuggingFaceFW/fineweb-edu",
            "weight": 1.0,
            "skip": skip if skip is not None else 5_000_000,
        }
        label = "FineWeb-Edu (general web; midtrain2 baseline ~28-30)"
    else:
        raise ValueError(f"dataset must be 'math' or 'fineweb', got {dataset!r}")

    loader = make_loader(
        dataset_configs=[ds_cfg],
        seq_len=4096, tokenizer_name="/root/v6_tokenizer_export",
        batch_size=batch, step_num=999999, num_workers=0,
    )
    it = iter(loader)
    total_loss = total_tok = 0
    with torch.inference_mode():
        for _ in range(eval_steps):
            input_ids, labels = next(it)
            input_ids, labels = input_ids.to(device), labels.to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(input_ids, labels=labels)
            n = int((labels != -100).sum())
            total_loss += out.loss.item() * n
            total_tok += n
    mean_loss = total_loss / max(total_tok, 1)
    ppl = math.exp(min(mean_loss, 20.0))
    res = {"step": resolved, "dataset": dataset, "loss": mean_loss,
           "ppl": ppl, "tokens": total_tok}
    print(f"\n== held-out {label} ppl ==\n{res}", flush=True)
    return res


@app.local_entrypoint()
def run_ppl_eval(
    step: str = "latest",
    dataset: str = "math",
    eval_steps: int = 40,
    skip: int | None = None,
):
    """Held-out perplexity of a base checkpoint on GPU (blocking).

    Pulls the checkpoint from HF, so it never touches the Colab drip.
    `step`: "latest" or a number like "8700".
    `dataset`: "math" (Nemotron-CC-Math, skip=2M) or "fineweb" (FineWeb-Edu,
      skip=5M — safely held-out and fast; general-web baseline ~28-30).
    `skip`: override the held-out offset (records) if needed.
    Re-run at a later step to read the trend (ppl should DROP as pretraining helps).
    """
    print(f"Evaluating step={step} dataset={dataset} (held-out ppl) on A100...")
    res = ppl_eval.remote(
        step=step, dataset=dataset, eval_steps=eval_steps, skip=skip)
    print("\n=== RESULT ===")
    print(f"  step:    {res['step']}")
    print(f"  dataset: {res['dataset']}")
    print(f"  loss:    {res['loss']:.4f}")
    print(f"  ppl:     {res['ppl']:.2f}")
    print(f"  tokens:  {res['tokens']:,}")


# =============================================================================
# SAMPLE — free-form generation from a BASE checkpoint (eyeball the outputs)
# =============================================================================
# This is a BASE (never-SFT'd) model, so it does raw next-token CONTINUATION,
# not Q->A / chat. Prompts are text PREFIXES the model continues. Pure greedy
# loops on a base model, so the greedy view uses a repetition penalty; a sampled
# view (temp/top-p) shows the more natural distribution. Mirrors ppl_eval's
# proven HF-pull + tokenizer-from-image loading; never touches any training run.
def _use_inductor_cache() -> None:
    """Point torch.compile's inductor cache at the persistent Modal volume.

    The fullgraph+dynamic trace of the 18-effective-layer forward costs
    ~10-15 min of A100 time. Caching the compiled artifacts on the volume
    turns every later run's compile into a warm load (~1 min). Call FIRST
    in any Modal function that compiles, before torch compiles anything.
    """
    import os

    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", "/vol/hf_cache/torchinductor_cache",
    )


_SAMPLE_PROMPTS = [
    "Problem: What is the value of 12 multiplied by 8?\nSolution:",
    "To find the area of a rectangle we multiply its length by its width. "
    "A rectangle with length 5 cm and width 3 cm has an area of",
    "Photosynthesis is the process by which plants",
    "def is_prime(n):\n    ",
    "The Industrial Revolution was a period of major",
    "In mathematics, a prime number is a natural number that",
]


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/vol/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=1800,
)
def sample_base(
    step: str = "latest",
    max_new_tokens: int = 160,
    rep_penalty: float = 1.3,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 40,
    hf_repo: str = "HallD/osrt-v6-ckpt",
    prompts: list[str] | None = None,
) -> list[dict]:
    """Generate continuations from a base checkpoint. Returns per-prompt dicts
    with a greedy(+rep_penalty) view and a sampled view."""
    import re

    _use_inductor_cache()

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config

    files = HfApi().list_repo_files(hf_repo, repo_type="model")
    ckpts = [f for f in files
             if re.match(r"osrt_v5_midtrain3_(?:rescue_)?step_\d+\.pt$", f)]
    if step == "latest":
        name = max(ckpts, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
    elif step == "final":
        name = "osrt_v5_midtrain3_final.pt"
    else:
        name = next(f for f in ckpts if re.search(rf"step_{step}\.pt$", f))
    print(f"pulling {name} from {hf_repo}...", flush=True)
    path = hf_hub_download(hf_repo, name, repo_type="model")

    tok = AutoTokenizer.from_pretrained("/root/v6_tokenizer_export")
    cfg = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )
    device = torch.device("cuda")
    model = OSRTForCausalLM(cfg).to(device)
    sd = torch.load(path, map_location=device, weights_only=True)["model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (
        f"state mismatch: missing={missing[:3]} unexpected={unexpected[:3]}"
    )
    # Idea #1: telemetry-off + torch.compile — ~3x, verified output-identical.
    model.optimize_for_inference()

    import time

    def _gen(prompt: str, *, temp: float, rp: float) -> tuple[str, int, float]:
        ids = [tok.bos_token_id] if tok.bos_token_id is not None else []
        ids += tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids], device=device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(
                inp, max_new_tokens=max_new_tokens, temperature=temp,
                top_p=top_p, top_k=top_k, repetition_penalty=rp,
                eos_token_id=tok.eos_token_id,
            )
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        gen = out[0, len(ids):].tolist()
        return tok.decode(gen, skip_special_tokens=True), len(gen), dt

    # Warmup so the first timed call isn't paying CUDA lazy-init / kernel compile.
    print("warmup generation (untimed)...", flush=True)
    _gen(_SAMPLE_PROMPTS[0], temp=0.0, rp=rep_penalty)

    prompts = prompts or _SAMPLE_PROMPTS
    results = []
    tps_all = []
    for p in prompts:
        greedy, gn, gdt = _gen(p, temp=0.0, rp=rep_penalty)
        sampled, sn, sdt = _gen(p, temp=temperature, rp=1.2)
        g_tps, s_tps = gn / gdt, sn / sdt
        tps_all += [g_tps, s_tps]
        results.append({"prompt": p, "greedy": greedy, "sampled": sampled,
                        "greedy_tok_per_sec": g_tps, "sampled_tok_per_sec": s_tps,
                        "greedy_tokens": gn, "sampled_tokens": sn})
        print(f"\n{'='*70}\nPROMPT: {p!r}\n"
              f"--- greedy(rp={rep_penalty}) [{gn} tok, {g_tps:.1f} tok/s] ---\n"
              f"{greedy}\n"
              f"--- sampled(t={temperature},p={top_p}) "
              f"[{sn} tok, {s_tps:.1f} tok/s] ---\n"
              f"{sampled}", flush=True)
    avg_tps = sum(tps_all) / len(tps_all)
    print(f"\n{'='*70}\nTHROUGHPUT: mean {avg_tps:.1f} tok/s "
          f"(batch=1, A100, {len(tps_all)} gens, max_new={max_new_tokens})",
          flush=True)
    return results


@app.local_entrypoint()
def run_sample(step: str = "latest", max_new_tokens: int = 160):
    """Eyeball free-form generations from a base checkpoint (blocking).

    `step`: "latest" (=step_12600), "final" (=osrt_v5_midtrain3_final.pt), or a
    number. It's a BASE model -> continuations of text prefixes, not chat.
    """
    print(f"Sampling base step={step} on A100 (max_new={max_new_tokens})...")
    res = sample_base.remote(step=step, max_new_tokens=max_new_tokens)
    print(f"\n=== {len(res)} prompts generated ===")


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/vol/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=1800,
)
def profile_decode(
    step: str = "latest",
    decode_steps: int = 32,
    batch_sizes: tuple[int, ...] = (1, 8, 32),
    compiled: bool = False,
    hf_repo: str = "HallD/osrt-v6-ckpt",
) -> dict:
    """Diagnose the slow single-stream decode. Two experiments, no model edits:
      (1) batch-size sweep -> per-token latency + aggregate tok/s. If tok/s
          scales ~linearly with batch, decode is launch/occupancy-bound (batching
          + torch.compile will fix it), not compute-bound.
      (2) torch.profiler over a few batch=1 decode steps -> top CUDA ops by time
          and by launch COUNT (thousands of tiny ops == launch-overhead bound).
    `compiled=True` runs optimize_for_inference() first, so the profile shows the
    POST-compile bottleneck (picks the next idea by evidence: if Sinkhorn is gone
    and cat dominates -> static cache; if Sinkhorn persists -> Triton kernel).
    """
    import re
    import time

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from torch.profiler import ProfilerActivity, profile
    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config

    files = HfApi().list_repo_files(hf_repo, repo_type="model")
    ckpts = [f for f in files
             if re.match(r"osrt_v5_midtrain3_(?:rescue_)?step_\d+\.pt$", f)]
    name = (max(ckpts, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
            if step == "latest"
            else next(f for f in ckpts if re.search(rf"step_{step}\.pt$", f)))
    print(f"pulling {name}...", flush=True)
    path = hf_hub_download(hf_repo, name, repo_type="model")
    tok = AutoTokenizer.from_pretrained("/root/v6_tokenizer_export")
    cfg = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )
    device = torch.device("cuda")
    model = OSRTForCausalLM(cfg).to(device).eval()
    sd = torch.load(path, map_location=device, weights_only=True)["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    if compiled:
        print("optimize_for_inference() (telemetry off + compile)...", flush=True)
        model.optimize_for_inference()

    prompt_ids = [tok.bos_token_id] + tok.encode(
        "The Industrial Revolution was a period of major",
        add_special_tokens=False,
    )

    def _run(bs: int) -> tuple[float, float]:
        inp = torch.tensor([prompt_ids] * bs, device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.generate(inp, max_new_tokens=4, temperature=0.0)  # warmup
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(inp, max_new_tokens=decode_steps, temperature=0.0)
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        per_tok_ms = dt / decode_steps * 1000
        agg_tps = bs * decode_steps / dt
        return per_tok_ms, agg_tps

    sweep = {}
    print("\n=== batch-size sweep ===", flush=True)
    for bs in batch_sizes:
        ms, tps = _run(bs)
        sweep[bs] = {"per_token_ms": ms, "aggregate_tok_per_sec": tps}
        print(f"  batch={bs:>3}: {ms:7.1f} ms/token-step | {tps:8.1f} tok/s aggregate",
              flush=True)

    # torch.profiler on batch=1 decode
    print("\n=== profiler: batch=1, 8 decode steps ===", flush=True)
    inp = torch.tensor([prompt_ids], device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.generate(inp, max_new_tokens=4, temperature=0.0)  # warmup
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            model.generate(inp, max_new_tokens=8, temperature=0.0)
    ka = prof.key_averages()

    def _dev_us(e):  # torch renamed self_cuda_time_total -> self_device_time_total
        return getattr(e, "self_device_time_total",
                       getattr(e, "self_cuda_time_total", 0.0))

    total_dev_us = sum(_dev_us(e) for e in ka)
    total_calls = sum(e.count for e in ka)
    print(f"  total op launches (8 steps): {total_calls}  (~{total_calls // 8}/token)",
          flush=True)
    print(f"  total self-device time: {total_dev_us / 1000:.1f} ms", flush=True)
    sort_key = ("self_device_time_total"
                if hasattr(next(iter(ka)), "self_device_time_total")
                else "self_cuda_time_total")
    print("  top ops by device time:", flush=True)
    print(ka.table(sort_by=sort_key, row_limit=12), flush=True)
    print("  top ops by launch count:", flush=True)
    print(ka.table(sort_by="count", row_limit=12), flush=True)
    return {"sweep": sweep, "launches_per_token": total_calls // 8}


@app.local_entrypoint()
def run_profile_decode(
    step: str = "latest", decode_steps: int = 32, compiled: bool = False,
):
    """Diagnose decode throughput (batch sweep + profiler) on A100 (blocking).

    `--compiled` profiles the optimize_for_inference() path to pick the next
    speedup idea by evidence.
    """
    print(f"Profiling decode for step={step} compiled={compiled} on A100...")
    res = profile_decode.remote(
        step=step, decode_steps=decode_steps, compiled=compiled)
    print("\n=== SWEEP SUMMARY ===")
    for bs, r in res["sweep"].items():
        print(f"  batch={bs}: {r['per_token_ms']:.1f} ms/step, "
              f"{r['aggregate_tok_per_sec']:.1f} tok/s aggregate")
    print(f"  launches/token: {res['launches_per_token']}")


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/vol/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=3600,
)
def ppl_sinkhorn_ab(
    step: str = "latest",
    iters_list: tuple[int, ...] = (20, 8, 5, 3),
    eval_steps: int = 24,
    batch: int = 6,
    hf_repo: str = "HallD/osrt-v6-ckpt",
) -> dict:
    """A/B: does held-out math ppl hold as we cut Sinkhorn iterations?

    The Sinkhorn iteration count is NOT a weight — it's a runtime attribute on
    every ManifoldHyperConnection. So we load once, fix a set of held-out
    batches, and re-score them at each iters setting by overriding the attr.
    If ppl@5 ~= ppl@20, the inference decode can drop 20->5 (the top hotspot,
    89% of decode wall time) for free.
    """
    import math
    import re

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    from osrt.data import make_loader
    from osrt.mhc import ManifoldHyperConnection
    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config

    files = HfApi().list_repo_files(hf_repo, repo_type="model")
    ckpts = [f for f in files
             if re.match(r"osrt_v5_midtrain3_(?:rescue_)?step_\d+\.pt$", f)]
    name = (max(ckpts, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
            if step == "latest"
            else next(f for f in ckpts if re.search(rf"step_{step}\.pt$", f)))
    print(f"pulling {name}...", flush=True)
    path = hf_hub_download(hf_repo, name, repo_type="model")
    tok = AutoTokenizer.from_pretrained("/root/v6_tokenizer_export")
    cfg = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )
    device = torch.device("cuda")
    model = OSRTForCausalLM(cfg).to(device).eval()
    sd = torch.load(path, map_location=device, weights_only=True)["model_state_dict"]
    model.load_state_dict(sd, strict=False)

    mhcs = [m for m in model.modules() if isinstance(m, ManifoldHyperConnection)]
    print(f"{len(mhcs)} mHC modules; default sinkhorn_iters="
          f"{mhcs[0].sinkhorn_iters}", flush=True)

    # Fix the SAME held-out batches for every setting (fair comparison).
    loader = make_loader(
        dataset_configs=[{
            "name": "nemotron-cc-math-heldout",
            "hf_id": "nvidia/Nemotron-CC-Math-v1", "hf_config": "4plus",
            "weight": 1.0, "skip": 2_000_000,
        }],
        seq_len=4096, tokenizer_name="/root/v6_tokenizer_export",
        batch_size=batch, step_num=999999, num_workers=0,
    )
    it = iter(loader)
    batches = [next(it) for _ in range(eval_steps)]

    results = {}
    for iters in iters_list:
        for m in mhcs:
            m.sinkhorn_iters = iters
        total_loss = total_tok = 0
        with torch.inference_mode():
            for input_ids, labels in batches:
                input_ids, labels = input_ids.to(device), labels.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids, labels=labels)
                n = int((labels != -100).sum())
                total_loss += out.loss.item() * n
                total_tok += n
        mean_loss = total_loss / max(total_tok, 1)
        ppl = math.exp(min(mean_loss, 20.0))
        results[iters] = {"loss": mean_loss, "ppl": ppl}
        print(f"  sinkhorn_iters={iters:>2}: loss={mean_loss:.4f}  ppl={ppl:.3f}",
              flush=True)
    return {"tokens": total_tok, "by_iters": results}


@app.local_entrypoint()
def run_ppl_sinkhorn_ab(step: str = "latest", eval_steps: int = 24):
    """A/B held-out math ppl vs Sinkhorn iteration count (blocking)."""
    print(f"Sinkhorn-iters ppl A/B for step={step} on A100...")
    res = ppl_sinkhorn_ab.remote(step=step, eval_steps=eval_steps)
    print(f"\n=== RESULT ({res['tokens']:,} tok) ===")
    base = res["by_iters"].get(20, {}).get("ppl")
    for iters, r in res["by_iters"].items():
        delta = f"  (Δ {r['ppl'] - base:+.4f} vs 20)" if base else ""
        print(f"  iters={iters:>2}: ppl={r['ppl']:.3f}{delta}")


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/vol/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=3600,
)
def bench_compile(
    step: str = "latest",
    ctx_len: int = 64,
    timed_steps: int = 50,
    batch_sizes: tuple[int, ...] = (1, 32),
    hf_repo: str = "HallD/osrt-v6-ckpt",
) -> dict:
    """Measure the OUTPUT-IDENTICAL decode ceiling (20 Sinkhorn iters kept).

    Isolates the static-cache steady state: prefill once to a length-`ctx_len`
    cache, then repeatedly run a 1-token decode step against that FIXED cache
    (discarding the grown cache), so shapes stay static — exactly what the
    static-KV-cache refactor would give. Compares eager vs
    torch.compile(reduce-overhead) (CUDA graphs). tok/s = batch / step_time.
    """
    import re
    import time

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config

    files = HfApi().list_repo_files(hf_repo, repo_type="model")
    ckpts = [f for f in files
             if re.match(r"osrt_v5_midtrain3_(?:rescue_)?step_\d+\.pt$", f)]
    name = (max(ckpts, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
            if step == "latest"
            else next(f for f in ckpts if re.search(rf"step_{step}\.pt$", f)))
    print(f"pulling {name}...", flush=True)
    path = hf_hub_download(hf_repo, name, repo_type="model")
    tok = AutoTokenizer.from_pretrained("/root/v6_tokenizer_export")
    cfg = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )
    device = torch.device("cuda")
    model = OSRTForCausalLM(cfg).to(device).eval()
    sd = torch.load(path, map_location=device, weights_only=True)["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    # Inference: turn OFF the MoE/loop-collapse telemetry. It does ~21 .item()
    # GPU->CPU syncs per MoE forward (x18 effective layers) that are pure
    # overhead here AND graph-break torch.compile. Training keeps it on.
    model.set_moe_telemetry(False)

    def _bench(fwd, bs: int) -> float:
        ctx = torch.randint(0, cfg.real_vocab_size, (bs, ctx_len), device=device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.forward(ctx, use_cache=True)  # prefill (eager, once)
            cache = out.past_key_values
            new_tok = torch.randint(0, cfg.real_vocab_size, (bs, 1), device=device)
            for _ in range(8):  # warmup (triggers compile + CUDA-graph capture)
                fwd(new_tok, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(timed_steps):
                fwd(new_tok, past_key_values=cache, use_cache=True)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / timed_steps * 1000  # ms/step

    results = {}
    variants = {"eager": model.forward}
    # Default mode: fuses kernels (logsumexp 5 ops -> 1, elementwise) WITHOUT
    # fragile CUDA-graph stream capture. reduce-overhead: adds CUDA graphs
    # (zero launch overhead) but capture breaks on MoE data-dependent ops / RNG.
    try:
        variants["compile_default"] = torch.compile(model, fullgraph=False).forward
    except Exception as e:  # noqa: BLE001
        print(f"compile(default) setup failed: {e}", flush=True)
    try:
        variants["compile_reduce_overhead"] = torch.compile(
            model, mode="reduce-overhead", fullgraph=False).forward
    except Exception as e:  # noqa: BLE001
        print(f"compile(reduce-overhead) setup failed: {e}", flush=True)

    for vname, fwd in variants.items():
        results[vname] = {}
        for bs in batch_sizes:
            try:
                ms = _bench(fwd, bs)
                tps = bs / ms * 1000
                results[vname][bs] = {"ms_per_step": ms, "tok_per_sec": tps}
                print(f"  {vname:>24} batch={bs:>3}: {ms:7.2f} ms/step | "
                      f"{tps:8.1f} tok/s", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  {vname} batch={bs}: FAILED {type(e).__name__}: {e}",
                      flush=True)
                results[vname][bs] = {"error": f"{type(e).__name__}: {e}"}
    return results


@app.local_entrypoint()
def run_bench_compile(step: str = "latest", ctx_len: int = 64):
    """Measure output-identical decode ceiling (eager vs compiled) on A100."""
    print(f"Benchmarking compiled decode for step={step} on A100...")
    res = bench_compile.remote(step=step, ctx_len=ctx_len)
    print("\n=== CEILING (20 Sinkhorn iters, outputs unchanged) ===")
    for vname, by_bs in res.items():
        for bs, r in by_bs.items():
            if "tok_per_sec" in r:
                print(f"  {vname} batch={bs}: {r['ms_per_step']:.2f} ms/step, "
                      f"{r['tok_per_sec']:.1f} tok/s")
            else:
                print(f"  {vname} batch={bs}: {r.get('error')}")


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/vol/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=3600,  # fullgraph+dynamic compile can take several minutes
)
def bench_generate(
    step: str = "latest",
    new_tokens: int = 64,
    hf_repo: str = "HallD/osrt-v6-ckpt",
) -> dict:
    """Authoritative end-to-end generate() bench + identity gate on the REAL
    decode path (not a fixed-cache loop). Three configs on ONE loaded model:
      - eager_telem_on:  today's shipping default (telemetry ON, no compile)
      - eager_telem_off: set_moe_telemetry(False) only
      - compiled:        optimize_for_inference() — telemetry off + compiled
                         forward routed through generate() via _fwd
    Reports batch=1 and batch=32 throughput, prefill latency separated from
    steady-state decode tok/s, and token-identity of all three vs eager_telem_on
    (must match — compile is output-identical). Confirms _compiled_forward is set
    AND (batch=1) that the compiled config is actually faster, i.e. compile is
    genuinely wired through generate() now (the earlier bug: it was not)."""
    import re
    import time

    _use_inductor_cache()

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config

    files = HfApi().list_repo_files(hf_repo, repo_type="model")
    ckpts = [f for f in files
             if re.match(r"osrt_v5_midtrain3_(?:rescue_)?step_\d+\.pt$", f)]
    name = (max(ckpts, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
            if step == "latest"
            else next(f for f in ckpts if re.search(rf"step_{step}\.pt$", f)))
    print(f"pulling {name}...", flush=True)
    path = hf_hub_download(hf_repo, name, repo_type="model")
    tok = AutoTokenizer.from_pretrained("/root/v6_tokenizer_export")
    cfg = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )
    device = torch.device("cuda")
    model = OSRTForCausalLM(cfg).to(device).eval()
    sd = torch.load(path, map_location=device, weights_only=True)["model_state_dict"]
    model.load_state_dict(sd, strict=False)

    prompt = ([tok.bos_token_id]
              + tok.encode("The Industrial Revolution was a period of major",
                           add_special_tokens=False))

    def _gen(bs: int, warmups: int = 1) -> tuple[list[int], float, float]:
        inp = torch.tensor([prompt] * bs, device=device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(warmups):  # warmup (compile trace on first call)
                model.generate(inp, max_new_tokens=4, temperature=0.0)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model.generate(inp, max_new_tokens=1, temperature=0.0)  # prefill+1
            torch.cuda.synchronize()
            t_prefill = time.perf_counter() - t0
            t0 = time.perf_counter()
            out = model.generate(inp, max_new_tokens=new_tokens, temperature=0.0)
            torch.cuda.synchronize()
            t_full = time.perf_counter() - t0
        ids = out[0, len(prompt):].tolist()
        decode_tps = bs * (new_tokens - 1) / max(t_full - t_prefill, 1e-6)
        return ids, t_prefill * 1000, decode_tps

    res = {}
    # 1) eager, telemetry ON (today's default)
    ids_on, pf_on, tps_on = _gen(1)
    _, _, tps_on32 = _gen(32)
    res["eager_telem_on"] = {"prefill_ms": pf_on, "decode_tps_b1": tps_on,
                             "decode_tps_b32": tps_on32}
    # 2) eager, telemetry OFF
    model.set_moe_telemetry(False)
    ids_off, pf_off, tps_off = _gen(1)
    _, _, tps_off32 = _gen(32)
    res["eager_telem_off"] = {"prefill_ms": pf_off, "decode_tps_b1": tps_off,
                              "decode_tps_b32": tps_off32}
    # 3) compiled (the real fix): optimize_for_inference sets _compiled_forward,
    #    generate() routes through _fwd. 2 warmups to amortise trace/recompile.
    model.optimize_for_inference()
    assert model._compiled_forward is not None, "compile not wired"
    ids_c, pf_c, tps_c = _gen(1, warmups=3)
    _, _, tps_c32 = _gen(32, warmups=3)
    res["compiled"] = {"prefill_ms": pf_c, "decode_tps_b1": tps_c,
                       "decode_tps_b32": tps_c32}

    res["identity"] = {
        "telem_off_matches_on": ids_off == ids_on,
        "compiled_matches_on": ids_c == ids_on,
    }
    for k in ("eager_telem_on", "eager_telem_off", "compiled"):
        r = res[k]
        print(f"  {k:>16}: prefill {r['prefill_ms']:7.1f} ms | "
              f"decode b1 {r['decode_tps_b1']:6.1f} tok/s | "
              f"b32 {r['decode_tps_b32']:7.1f} tok/s", flush=True)
    idn = res["identity"]
    print(f"  identity vs eager_telem_on: telem_off={idn['telem_off_matches_on']}"
          f"  compiled={idn['compiled_matches_on']}", flush=True)
    return res


@app.local_entrypoint()
def run_bench_generate(step: str = "latest", new_tokens: int = 64):
    """Authoritative generate() bench + identity gate on A100 (blocking)."""
    print(f"Benchmarking real generate() for step={step} on A100...")
    res = bench_generate.remote(step=step, new_tokens=new_tokens)
    print("\n=== REAL generate() PATH (20 Sinkhorn iters) ===")
    for k in ("eager_telem_on", "eager_telem_off", "compiled"):
        r = res[k]
        print(f"  {k}: prefill {r['prefill_ms']:.1f}ms | "
              f"decode b1 {r['decode_tps_b1']:.1f} tok/s | "
              f"b32 {r['decode_tps_b32']:.1f} tok/s")
    print(f"  compiled output-identical to eager: "
          f"{res['identity']['compiled_matches_on']}")


@app.function(
    gpu="A100-80GB",
    image=image,
    volumes={"/vol/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("hf-secret")],
    timeout=3600,  # eager ppl+gens + fullgraph trace + compiled ppl+gens
)
def verify_compile_identity(
    step: str = "latest",
    max_new_tokens: int = 64,
    hf_repo: str = "HallD/osrt-v6-ckpt",
) -> dict:
    """QUALITY GATE for compile: is the REAL compiled path (generate via
    self._fwd -> self._compiled_forward) a quality regression, or just benign
    bf16 fused-reduction noise?

    Persists the inductor cache on the volume (see _use_inductor_cache) so the
    ~15-min fullgraph trace is paid once, then warm-loaded on later runs.

    Three checks vs eager, on the SAME weights, using the REAL _compiled_forward
    (not the old model.compile()+self.forward bug):
      (A) forward logit diff — teacher-forced fixed input, eager vs compiled
          logits: max|Δlogit| + top-1 argmax agreement %. Large diff => compile
          MISCOMPILES (must not ship); ~bf16-epsilon => benign.
      (B) held-out math ppl eager vs compiled — teacher-forced (NO generation
          feedback to amplify). This is the decisive quality number: if
          ppl_compiled ~= ppl_eager, quality is preserved regardless of greedy
          token drift.
      (C) greedy first-divergence — the amplified free-gen view, for context
          only (a single late argmax flip cascades here; not a quality metric).
    """
    import math
    import re

    _use_inductor_cache()

    import torch
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    from osrt.data import make_loader
    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config

    files = HfApi().list_repo_files(hf_repo, repo_type="model")
    ckpts = [f for f in files
             if re.match(r"osrt_v5_midtrain3_(?:rescue_)?step_\d+\.pt$", f)]
    name = (max(ckpts, key=lambda f: int(re.search(r"step_(\d+)", f).group(1)))
            if step == "latest"
            else next(f for f in ckpts if re.search(rf"step_{step}\.pt$", f)))
    print(f"pulling {name}...", flush=True)
    path = hf_hub_download(hf_repo, name, repo_type="model")
    tok = AutoTokenizer.from_pretrained("/root/v6_tokenizer_export")
    cfg = build_config(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id, fused_cross_entropy_chunks=8,
    )
    device = torch.device("cuda")
    model = OSRTForCausalLM(cfg).to(device).eval()
    sd = torch.load(path, map_location=device, weights_only=True)["model_state_dict"]
    model.load_state_dict(sd, strict=False)
    model.set_moe_telemetry(False)

    def _greedy(prompt: str) -> list[int]:
        ids = [tok.bos_token_id] + tok.encode(prompt, add_special_tokens=False)
        inp = torch.tensor([ids], device=device)
        with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(inp, max_new_tokens=max_new_tokens, temperature=0.0)
        return out[0, len(ids):].tolist()

    # Fixed held-out math batches (same for eager + compiled) for the ppl gate.
    loader = make_loader(
        dataset_configs=[{
            "name": "nemotron-cc-math-heldout",
            "hf_id": "nvidia/Nemotron-CC-Math-v1", "hf_config": "4plus",
            "weight": 1.0, "skip": 2_000_000,
        }],
        seq_len=4096, tokenizer_name="/root/v6_tokenizer_export",
        batch_size=4, step_num=999999, num_workers=0,
    )
    it = iter(loader)
    ppl_batches = [next(it) for _ in range(12)]

    def _ppl() -> float:
        tot_loss = tot_tok = 0
        with torch.inference_mode():
            for input_ids, labels in ppl_batches:
                input_ids, labels = input_ids.to(device), labels.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids, labels=labels)
                n = int((labels != -100).sum())
                tot_loss += out.loss.item() * n
                tot_tok += n
        return math.exp(min(tot_loss / max(tot_tok, 1), 20.0))

    fixed = torch.tensor(
        [[tok.bos_token_id] + tok.encode(
            "The Industrial Revolution was a period of major economic change "
            "that began in Britain and spread across the world over decades.",
            add_special_tokens=False)],
        device=device,
    )
    # EAGER pass first (before compile is wired).
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        eager_logits = model.forward(fixed).logits.float()
    eager_ppl = _ppl()
    eager_gens = {p: _greedy(p) for p in _SAMPLE_PROMPTS}

    # COMPILE for real: sets _compiled_forward; _fwd + generate route through it.
    print("optimize_for_inference() (real compiled path)...", flush=True)
    model.optimize_for_inference()
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = model._fwd(fixed)  # warmup / trace the compiled forward
        comp_logits = model._fwd(fixed).logits.float()
    comp_ppl = _ppl()

    max_abs = (eager_logits - comp_logits).abs().max().item()
    top1_agree = (
        (eager_logits.argmax(-1) == comp_logits.argmax(-1)).float().mean().item()
    )

    gen_report = {}
    for p in _SAMPLE_PROMPTS:
        e, c = eager_gens[p], _greedy(p)
        first_div = next((i for i, (a, b) in enumerate(zip(e, c)) if a != b), None)
        gen_report[p[:40]] = {"first_divergence": first_div, "n": len(e)}
        tag = "IDENTICAL" if first_div is None else f"diverges @ {first_div}/{len(e)}"
        print(f"  [{tag}] {p[:48]!r}", flush=True)

    res = {
        "forward_max_abs_logit_diff": max_abs,
        "forward_top1_agreement": top1_agree,
        "eager_ppl": eager_ppl,
        "compiled_ppl": comp_ppl,
        "greedy": gen_report,
    }
    print(f"\n  forward: max|Δlogit|={max_abs:.4g}  top1_agree={top1_agree*100:.3f}%"
          f"\n  ppl: eager={eager_ppl:.4f}  compiled={comp_ppl:.4f}  "
          f"Δ={comp_ppl - eager_ppl:+.4f}", flush=True)
    return res


@app.local_entrypoint()
def run_verify_compile(step: str = "latest", max_new_tokens: int = 64):
    """Quality gate: is compile a regression or benign bf16 noise? (blocking)."""
    print(f"Verifying compile quality for step={step} on A100...")
    res = verify_compile_identity.remote(step=step, max_new_tokens=max_new_tokens)
    print("\n=== COMPILE QUALITY GATE ===")
    print(f"  forward max|Δlogit|:   {res['forward_max_abs_logit_diff']:.4g}")
    print(f"  forward top-1 agree:   {res['forward_top1_agreement']*100:.3f}%")
    print(f"  held-out math ppl:     eager={res['eager_ppl']:.4f}  "
          f"compiled={res['compiled_ppl']:.4f}  "
          f"Δ={res['compiled_ppl'] - res['eager_ppl']:+.4f}")
    ident = sum(1 for r in res["greedy"].values() if r["first_divergence"] is None)
    print(f"  greedy identical:      {ident}/{len(res['greedy'])} prompts "
          f"(free-gen amplifies bf16 flips; not a quality metric)")


@app.local_entrypoint()
def run_sft_eval(step: str = "final", n: int = 100, max_new_tokens: int = 768):
    """Scored reasoning on/off eval of an SFT-v2 checkpoint on GPU (blocking).

    `step`: "200"/"600"/"1000"/… → osrt_v5_sft_v2_step_<step>.pt, or "final".
    `max_new_tokens`: 768 default — long OpenR1-style CoT needs room to close
    into <|answer|> (400 truncated it at step_200, reading format_ok=0).
    Compare acc_delta_on_minus_off vs SFT-v1's +0.02 (acc_on 0.06/off 0.04).
    """
    name = ("osrt_v5_sft_v2_final.pt" if step == "final"
            else f"osrt_v5_sft_v2_step_{step}.pt")
    print(f"Evaluating {name} (n={n}, max_new={max_new_tokens}) on H100...")
    res = sft_eval.remote(name, n, max_new_tokens)
    print("\n=== RESULT ===")
    for k in ("sft_eval/acc_on", "sft_eval/acc_off",
              "sft_eval/acc_delta_on_minus_off", "sft_eval/format_ok_on",
              "sft_eval/format_ok_off", "sft_eval/resp_len_on",
              "sft_eval/resp_len_off", "sft_eval/n"):
        print(f"  {k}: {res.get(k)}")


# =============================================================================
# MIDTRAIN 2 — extended continued-pretraining (push the undertrained base)
# =============================================================================
# The base is ~0.3x Chinchilla (~1.7B tokens). Adds ~1.1B more on a
# reasoning/instruction-heavy reweight of the knowledge mix (full-seq LM),
# resuming from midtrain_final. Reuses _run_midtrain + run_pretrain_extend —
# streams HF datasets, no rollout volume. See MidtrainExtendConfig.
@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def midtrain2():
    """v6 midtrain phase 2: ~4000 more steps, reasoning-heavy mix, fresh
    re-warm cosine from midtrain_final. See MidtrainExtendConfig."""
    from osrt.train_config import MidtrainExtendConfig
    _run_midtrain(MidtrainExtendConfig)


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def midtrain2_sanity():
    """30-step probe: native-HRA loads clean from midtrain_final, reweighted
    mix streams, VRAM fits at seq 4096. See MidtrainExtendSanityConfig."""
    from osrt.train_config import MidtrainExtendSanityConfig
    _run_midtrain(MidtrainExtendSanityConfig)


@app.local_entrypoint()
def run_midtrain2():
    """Spawn v6 midtrain phase 2 (fire-and-forget)."""
    call = midtrain2.spawn()
    print(f"Spawned v6 midtrain2 — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_midtrain2_sanity():
    """Spawn the 30-step v6 midtrain2 sanity probe."""
    call = midtrain2_sanity.spawn()
    print(f"Spawned v6 midtrain2 sanity — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.function(
    gpu="H100", image=image,
    volumes={"/vol/checkpoints": vol, "/vol/tokenizer": v6_tokenizer_vol,
             "/vol/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("wandb-secret"),
             modal.Secret.from_name("hf-secret")],
    timeout=86400,
)
def midtrain3():
    """v6 midtrain phase 3: the LONG capability push (12.6k-step cosine →
    ~1x Chinchilla), chained across monthly workspaces. Resumes from the
    highest midtrain3 checkpoint each run. See MidtrainExtend3Config."""
    from osrt.train_config import MidtrainExtend3Config
    _run_midtrain(MidtrainExtend3Config)


@app.function(
    gpu="H100", image=image,
    volumes={"/vol/checkpoints": vol, "/vol/tokenizer": v6_tokenizer_vol,
             "/vol/hf_cache": hf_cache_vol},
    secrets=[modal.Secret.from_name("wandb-secret"),
             modal.Secret.from_name("hf-secret")],
    timeout=86400,
)
def midtrain3_sanity():
    """30-step midtrain3 probe: clean load of step_1750, mix streams."""
    from osrt.train_config import MidtrainExtend3SanityConfig
    _run_midtrain(MidtrainExtend3SanityConfig)


@app.local_entrypoint()
def run_midtrain3():
    """Spawn v6 midtrain phase 3 (fire-and-forget; resumes from last ckpt)."""
    call = midtrain3.spawn()
    print(f"Spawned v6 midtrain3 — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_midtrain3_sanity():
    """Spawn the 30-step v6 midtrain3 sanity probe."""
    call = midtrain3_sanity.spawn()
    print(f"Spawned v6 midtrain3 sanity — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


# =============================================================================
# PRETRAIN_EXTEND2 — broadened mid-training pass
# =============================================================================
# Reuses run_pretrain_extend (the training loop is config-driven). Resumes
# from osrt_v5_grpo_final.pt (canonical step-700 GRPO ckpt) with HRA frozen
# so the SFT+GRPO investment in chat/answer format stays put while the base
# weights absorb new reasoning/code/math knowledge. Output checkpoints use
# the `extend2` prefix to avoid colliding with extend1's scan.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def pretrain_extend2():
    """Second mid-training pass — broader reasoning + code + science mix.

    See train_config.py::PretrainExtend2Config for the design
    rationale (DeepSeek-R1 cold-start strategy, 30/40/15/15 mix,
    HRA freeze, tag-rewrite for R1 traces). Single phase, seq 2048,
    ~3,000 steps, ~$28 of H100 time. Resumes from GRPO step-700.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import PretrainExtend2Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path

    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    extend2_cfg = PretrainExtend2Config()
    print(
        "pretrain_extend2: 3,000 steps at seq 2048, peak LR 1e-5, "
        "HRA frozen, 30/40/15/15 code/math/reasoning/general mix.",
    )
    print(f"Resume base: {extend2_cfg.pretrained_checkpoint}")

    run_pretrain_extend(model_config, extend2_cfg, vol, tokenizer_name)


# =============================================================================
# LOOP_FIX — architecture-fix continuation with per-loop aux LM-head losses
# =============================================================================
# Recursive-loop probe (probe_recursion.py, 2026-06-05) showed loop 5 doing
# ~6.0 of the CE loss reduction while loops 1-4 contributed ~0.75 combined.
# This stage attaches the weight-tied LM head to each non-final loop's hidden
# state (after norm_out for path consistency) and adds the resulting CE
# losses to the main loss with `aux_loop_loss_weight`. Forces gradient signal
# into the intermediate loops. ~1500 step continuation from extend2_final.
# See train_config.py::LoopFixConfig for the full design.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def loop_fix():
    """Per-loop aux-loss continuation from extend2_final.

    The aux_loop_loss_weight on OSRTConfig (model config) is the
    real switch — without it the model forward is unchanged. We thread
    it through here from the training config (LoopFixConfig).
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import LoopFixConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path
    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)

    loopfix_cfg = LoopFixConfig()
    # Phase end is informational (the loop stops on total_steps), but
    # set it for accurate printed banner.
    loopfix_cfg.phases["extend"]["end"] = loopfix_cfg.total_steps
    # Aux losses materialise 5 extra logit tensors (B × T × V) per
    # forward pass at the same precision as the main logits — ~10 GB
    # extra at batch=8/seq=2048. Cut batch 8→4 and bump accum 8→16 to
    # keep effective batch=64 while halving activation memory.
    loopfix_cfg.phases["extend"]["batch_size"] = 4
    loopfix_cfg.phases["extend"]["grad_accum_steps"] = 16

    # Critical: pass aux_loop_loss_weight to the MODEL config so the
    # model's forward actually computes the aux losses. Without this,
    # the training config's aux_loop_loss_weight is a no-op.
    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=loopfix_cfg.aux_loop_loss_weight,
    )

    print(
        f"loop_fix: {loopfix_cfg.total_steps - loopfix_cfg.lr_anchor_step} "
        f"steps continuation, peak_lr={loopfix_cfg.peak_lr}, "
        f"aux_loop_loss_weight={loopfix_cfg.aux_loop_loss_weight}.",
    )
    print(f"Resume base: {loopfix_cfg.pretrained_checkpoint}")

    run_pretrain_extend(model_config, loopfix_cfg, vol, tokenizer_name)


def loop_fix_sanity_inner():
    """Body of the 50-step sanity smoke test for loop_fix.
    Shared between the real sanity stage and any future ad-hoc test.
    """
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import LoopFixConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    class SanityLoopFix(LoopFixConfig):
        # Fresh step counter — no lr_anchor_step.
        total_steps = 50
        lr_anchor_step = 0
        warmup_steps = 10
        log_interval = 5
        ckpt_interval = 999_999
        eval_interval = 999_999
        wandb_log = False
        compile_enabled = False      # fast first-step events

    sanity_cfg = SanityLoopFix()
    sanity_cfg.phases["extend"]["end"] = 50
    sanity_cfg.phases["extend"]["batch_size"] = 4
    sanity_cfg.phases["extend"]["grad_accum_steps"] = 16
    model_config = OSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=sanity_cfg.aux_loop_loss_weight,
    )
    print(f"loop_fix SANITY: 50 steps from extend2_final, "
          f"aux_loop_loss_weight={sanity_cfg.aux_loop_loss_weight}.")
    run_pretrain_extend(model_config, sanity_cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def loop_fix_sanity():
    loop_fix_sanity_inner()


# =============================================================================
# LOOP_FIX_V2 — stacked fixes (aux + dropout + curriculum + per-loop weights)
# =============================================================================
# Layered on top of loop_fix's aux LM-head loss with:
#   (1) loop dropout (stochastic depth, p=0.2)
#   (2) aux-weight curriculum (0.02 → 0.10 over 200 steps)
#   (3) per-loop aux weights biased toward earlier loops [2.0,1.5,1.0,0.7,0.5]
# See train_config.py::LoopFixV2Config for the full rationale.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def loop_fix_v2():
    """Stacked architecture fixes (aux + dropout + curriculum +
    per-loop weights) from loop_fix's final ckpt."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import LoopFixV2Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    cfg = LoopFixV2Config()
    cfg.phases["extend"]["end"] = cfg.total_steps
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16
    # Resume base: loop_fix's step_400 (we plan to stop loop_fix early
    # at step 400 — fast gains were done by step 200, diminishing
    # returns thereafter; the v2 stacked-fix run is the better use of
    # the remaining compute). Update to loopfix_final.pt if we end up
    # running loop_fix to completion.
    cfg.pretrained_checkpoint = "/vol/checkpoints/v5/osrt_v5_loopfix_step_400.pt"

    model_config = OSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        # All three architecture-fix knobs go through the model config:
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"loop_fix_v2: {cfg.total_steps} steps, peak_lr={cfg.peak_lr}, "
        f"aux_loop_loss_weight={cfg.aux_loop_loss_weight} "
        f"(curriculum: {cfg.aux_loop_weight_start} → final over "
        f"{cfg.aux_loop_curriculum_steps} steps), "
        f"loop_dropout_prob={cfg.loop_dropout_prob}, "
        f"per_loop_weights={cfg.per_loop_aux_weights}."
    )
    print(f"Resume base: {cfg.pretrained_checkpoint}")

    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def loop_fix_v2_sanity():
    """50-step sanity for loop_fix_v2 stacked-fix run."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import LoopFixV2Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    class SanityCfg(LoopFixV2Config):
        total_steps = 50
        lr_anchor_step = 0
        warmup_steps = 10
        log_interval = 5
        ckpt_interval = 999_999
        eval_interval = 999_999
        wandb_log = False
        compile_enabled = False
        # Shorter curriculum so we exercise the ramp within 50 steps.
        aux_loop_curriculum_steps = 30
        # Resume from extend2_final (loop_fix may not have a final.pt
        # yet during testing); the sanity is to validate the new code
        # paths run end-to-end, not to require a specific ckpt.
        pretrained_checkpoint = "/vol/checkpoints/v5/osrt_v5_extend2_final.pt"

    cfg = SanityCfg()
    cfg.phases["extend"]["end"] = 50
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16
    model_config = OSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"loop_fix_v2 SANITY: 50 steps, dropout={cfg.loop_dropout_prob}, "
        f"curriculum {cfg.aux_loop_weight_start}→{cfg.aux_loop_loss_weight} "
        f"over {cfg.aux_loop_curriculum_steps} steps, "
        f"per_loop_weights={cfg.per_loop_aux_weights}."
    )
    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


# =============================================================================
# PRETRAIN_EXTEND3 — first mid-training round with WORKING recursive depth
# =============================================================================
# All prior training (~30k+ steps) happened with the loop-collapsed
# architecture. extend2's 9-stream mix was absorbed at only ~6 effective
# layers of depth. With loop_fix + v2 done, the model can now actually use
# all 18 effective layers — so re-running on the same data mix lets it
# encode information it couldn't before. v2 already showed this happening
# (task CE dropped 1.80 → 1.54 in 300 steps with fix on, on data the model
# had seen 8100 steps of before).
# See train_config.py::PretrainExtend3Config for the full design.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def pretrain_extend3():
    """Third mid-training pass — first run with the architecture fix
    permanently in the loss path. Same 9-stream extend2 mix, softer
    fix knobs (aux=0.05, dropout=0.10), lower LR (peak 3e-6), 3000
    steps from loopfixv2_merged.pt."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import PretrainExtend3Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    cfg = PretrainExtend3Config()
    cfg.phases["extend"]["end"] = cfg.total_steps
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16

    model_config = OSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"pretrain_extend3: {cfg.total_steps} steps, peak_lr={cfg.peak_lr}, "
        f"aux_loop_loss_weight={cfg.aux_loop_loss_weight} "
        f"(curriculum {cfg.aux_loop_weight_start}→{cfg.aux_loop_loss_weight} "
        f"over {cfg.aux_loop_curriculum_steps} steps), "
        f"loop_dropout_prob={cfg.loop_dropout_prob}, "
        f"per_loop_weights=uniform."
    )
    print(f"Resume base: {cfg.pretrained_checkpoint}")

    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def pretrain_extend3_sanity():
    """50-step sanity for pretrain_extend3."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import PretrainExtend3Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    class SanityCfg(PretrainExtend3Config):
        total_steps = 50
        lr_anchor_step = 0
        warmup_steps = 10
        log_interval = 5
        ckpt_interval = 999_999
        eval_interval = 999_999
        wandb_log = False
        compile_enabled = False
        # Shorter curriculum to exercise the ramp within 50 steps.
        aux_loop_curriculum_steps = 30

    cfg = SanityCfg()
    cfg.phases["extend"]["end"] = 50
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16
    model_config = OSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"pretrain_extend3 SANITY: 50 steps, peak_lr={cfg.peak_lr}, "
        f"aux={cfg.aux_loop_loss_weight}, dropout={cfg.loop_dropout_prob}."
    )
    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


# =============================================================================
# MOPD — Multi-teacher On-Policy Distillation from Gemini rollouts
# =============================================================================
# Trains on a local JSONL of teacher rollouts (collected via
# scripts/collect_rollouts.py + uploaded to the osrt-rollouts volume).
# Reuses run_pretrain_extend with the rollout_dataset_path override so all
# the architecture-fix telemetry, LR schedule, MoE balance, and checkpoint
# infrastructure works unchanged. Resumes from extend3_final.pt.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
        "/vol/rollouts": rollouts_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def mopd():
    """MOPD distillation on Gemini-rollout JSONL from extend3_final.pt.

    Reads /vol/rollouts/mopd_v1.jsonl (upload from local with
    `modal volume put osrt-rollouts rollouts/mopd_v1.jsonl mopd_v1.jsonl`
    before launching). 1000 steps, peak_lr 1.5e-6, aux fix knobs at
    extend3 levels."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import MOPDConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    cfg = MOPDConfig()
    cfg.phases["extend"]["end"] = cfg.total_steps
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16
    # Shorter seq_len for rollouts — most are under 1024 tokens, so
    # 2048 is mostly wasted padding. Cuts compute ~50% per step.
    cfg.phases["extend"]["seq_len"] = 1024

    model_config = OSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"mopd: {cfg.total_steps} steps from {cfg.pretrained_checkpoint}, "
        f"peak_lr={cfg.peak_lr}, rollout_path={cfg.rollout_dataset_path}, "
        f"aux={cfg.aux_loop_loss_weight}, dropout={cfg.loop_dropout_prob}."
    )
    if not os.path.exists(cfg.rollout_dataset_path):
        raise FileNotFoundError(
            f"Rollout JSONL not found at {cfg.rollout_dataset_path}. "
            "Upload via: "
            "`modal volume put osrt-rollouts rollouts/mopd_v1.jsonl "
            "mopd_v1.jsonl`"
        )

    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
        "/vol/rollouts": rollouts_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def mopd_sanity():
    """30-step MOPD sanity validating the rollout loader path."""
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import MOPDConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    class SanityCfg(MOPDConfig):
        total_steps = 30
        lr_anchor_step = 0
        warmup_steps = 5
        log_interval = 5
        ckpt_interval = 999_999
        eval_interval = 999_999
        wandb_log = False
        compile_enabled = False
        aux_loop_curriculum_steps = 10
        # Resume from extend3 step ckpt (or loopfixv2_merged) — sanity
        # is to validate the rollout pipeline end-to-end, not to
        # depend on a specific final ckpt that may not exist yet.
        pretrained_checkpoint = "/vol/checkpoints/v5/osrt_v5_loopfixv2_merged.pt"

    cfg = SanityCfg()
    cfg.phases["extend"]["end"] = 30
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = 16
    cfg.phases["extend"]["seq_len"] = 1024
    model_config = OSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"mopd SANITY: 30 steps, peak_lr={cfg.peak_lr}, "
        f"rollouts={cfg.rollout_dataset_path}."
    )
    if not os.path.exists(cfg.rollout_dataset_path):
        raise FileNotFoundError(
            f"Rollout JSONL not found at {cfg.rollout_dataset_path}. "
            "Upload via: "
            "`modal volume put osrt-rollouts rollouts/mopd_v1.jsonl "
            "mopd_v1.jsonl`"
        )

    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


# ─────────────────────────────────────────────────────────────────────
# system_sft: teaches the model to handle <|system|>...<|user|>...
# format. Resumes from grpo_v2_step_50.pt and trains on OpenHermes
# rollouts (filtered to system-prompt-bearing rows).
# Same loader path as MOPD; the RolloutDataset auto-detects the
# `system` field and emits `<|system|>{sys}<|user|>{q}<|assistant|>{a}`
# when present.
# ─────────────────────────────────────────────────────────────────────


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
        "/vol/rollouts": rollouts_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def system_sft():
    """SFT pass that teaches the model to handle <|system|> blocks."""
    _run_system_sft(sanity=False)


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
        "/vol/rollouts": rollouts_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def system_sft_sanity():
    """30-step system_sft sanity — validates that <|system|> prefix
    appears in batches and loss flows through the assistant turn only."""
    _run_system_sft(sanity=True)


def _run_system_sft(sanity: bool = False) -> None:
    import os
    import modal as _modal
    from transformers import AutoTokenizer
    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import SystemSFTConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    cfg = SystemSFTConfig()
    if sanity:
        cfg.total_steps = 30
        cfg.warmup_steps = 5
        cfg.log_interval = 5
        cfg.ckpt_interval = 999_999
        cfg.eval_interval = 999_999
        cfg.wandb_log = False
        cfg.compile_enabled = False
        cfg.aux_loop_curriculum_steps = 0

    # Wire the rollout loader through the same phases dict the
    # pretrain_extend trainer reads. Same shape MOPD uses.
    cfg.phases["extend"]["end"] = cfg.total_steps
    cfg.phases["extend"]["batch_size"] = 4
    cfg.phases["extend"]["grad_accum_steps"] = cfg.grad_accum_steps
    # System prompts make the prefix longer. Bump seq_len 1024 → 1536
    # to fit system+user+assistant comfortably.
    cfg.phases["extend"]["seq_len"] = 1536

    model_config = OSRTConfig(
        vocab_size=len(tok), real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
    )
    print(
        f"system_sft{' SANITY' if sanity else ''}: {cfg.total_steps} steps "
        f"from {cfg.pretrained_checkpoint}, peak_lr={cfg.peak_lr}, "
        f"rollout_path={cfg.rollout_dataset_path}.",
        flush=True,
    )
    if not os.path.exists(cfg.rollout_dataset_path):
        raise FileNotFoundError(
            f"Rollout JSONL not found at {cfg.rollout_dataset_path}. "
            "Upload via: `modal volume put osrt-rollouts "
            "rollouts/system_prompt_sft.jsonl system_prompt_sft.jsonl`",
        )

    run_pretrain_extend(model_config, cfg, vol, "/vol/tokenizer")


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def pretrain_extend2_sanity():
    """50-step smoke test of pretrain_extend2 — verify all 10 streams
    connect, format functions yield clean batches end-to-end, and the
    training loop completes a few cycles before committing $28 on the
    full 3,000-step run.

    Total cost ~$1 (~10 min including compile time). Disables
    checkpoint saving so the volume isn't polluted with throwaway
    sanity ckpts. Overrides `wandb_run_name` so the smoke run is
    visually separated from real extend2 runs in the dashboard.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.train import run_pretrain_extend
    from osrt.train_config import PretrainExtend2Config

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tokenizer_name = tokenizer_path
    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)

    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    # Subclass so we don't mutate the real config. ckpt_interval set
    # past total_steps so save path is never triggered. warmup_steps
    # cut to 10 (still 20% of total) so we actually exit the warmup
    # and see a cosine-shaped LR at least once before the run ends.
    class SanityCfg(PretrainExtend2Config):
        total_steps = 50
        warmup_steps = 10
        ckpt_interval = 999_999
        log_interval = 5
        eval_interval = 999_999
        wandb_run_name = "osrt-pretrain-extend2-sanity"
        # Skip torch.compile — eager starts producing step events
        # immediately (compile takes ~10 min of silent GPU time).
        compile_enabled = False
        wandb_log = False
        # Differential diagnosis: extend1 (which worked) resumed from
        # sft_ultralong_final.pt; extend2 (which auto-cancels mid-
        # first-forward-pass) resumes from osrt_v5_grpo_final.pt.
        # Swapping to the pre-GRPO sft_math ckpt for sanity isolates
        # whether the GRPO checkpoint itself is the trigger.
        pretrained_checkpoint = (
            "/vol/checkpoints/v5/osrt_v5_sft_math_final.pt"
        )

    sanity_cfg = SanityCfg()
    sanity_cfg.phases["extend"]["end"] = 50
    # Datasets now live in PretrainExtend2Config; sanity inherits
    # the locked 9-stream working mix verified via v9-v25 bisection.

    print("pretrain_extend2 SANITY: 50 steps, no ckpts, no eval — "
          "validating all streams + format functions.")
    print(f"Resume base: {sanity_cfg.pretrained_checkpoint}")

    run_pretrain_extend(model_config, sanity_cfg, vol, tokenizer_name)


# =============================================================================
# SANITY (200-step smoke test)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=7200,
)
def sanity():
    """Short smoke test: 200 steps, verifies the full pipeline end-to-end.

    Purpose: before committing to a $30+ full pretrain run, prove that
      - torch.compile succeeds on the full model
      - Data streaming works for FineWeb-Edu + CodeParrot
      - Loss descends (sanity: should drop from ~ln(32768)=10.4 at step 0)
      - MoE telemetry populates sensibly (prob H near ln(8), balance loss ~1)
      - Eval path runs without errors (drops disabled, chunk-stable)
      - Checkpoint save + W&B logging work

    Overrides vs full pretrain:
      - total_steps 1200 (was 300k)
      - warmup_steps 3000 (same as Foundation)
      - eval / ckpt intervals 500
      - early_stop_check_step disabled (set past total_steps) — 1200 steps
        isn't enough for the 5k-calibrated gate to be meaningful.
      - W&B run name "osrt-extended-sanity" — keeps sanity separate from
        real pretrain runs in the dashboard.

    Uses a separate checkpoint dir (/vol/checkpoints/v5-sanity-gumbel1000) so
    this cold-expert experiment starts from step 0 and never collides with
    real pretrain checkpoints.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.train import run_training
    from osrt.train_config import PretrainConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    print(f"Tokenizer volume contents: {os.listdir(tokenizer_path)}")
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    class SanityConfig(PretrainConfig):
        total_steps = 1200
        warmup_steps = 3000
        log_interval = 50
        # Eval disabled for sanity: we only care about whether the new
        # architecture trains. Running eval would pay the ~15 min
        # 100M-record FineWeb skip that primes the held-out cache, and
        # sanity isn't long enough to collide with that offset anyway.
        eval_interval = 10_000_000
        ckpt_interval = 500
        # Foundation-matched schedule: LR warms for 3000 steps, and router
        # noise anneals over 4000 so exploration survives peak LR.
        router_gumbel_anneal_steps = 4000
        # Disabled: 1200 steps with Foundation LR warmup + cosine is not
        # enough for the 5k-calibrated gate to be meaningful.
        early_stop_check_step = 10_000_000
        wandb_run_name = "osrt-extended-sanity"

    train_cfg = SanityConfig()
    print("=" * 60)
    print("v5 EXTENDED SANITY — 1200 Foundation-matched steps")
    print("=" * 60)
    print(f"  total_steps         : {train_cfg.total_steps}")
    print(f"  warmup_steps        : {train_cfg.warmup_steps}")
    print(f"  ckpt_interval       : {train_cfg.ckpt_interval}")
    print(f"  eval_interval       : {train_cfg.eval_interval}")
    print(
        f"  router_gumbel_tau   : {train_cfg.router_gumbel_tau_init} -> "
        f"{train_cfg.router_gumbel_tau_final} over "
        f"{train_cfg.router_gumbel_anneal_steps} steps"
    )
    print(
        f"  early_stop_step     : {train_cfg.early_stop_check_step} "
        f"(disabled)"
    )
    print()

    run_training(
        model_config, train_cfg, vol, tokenizer_path,
        # Loop-level bias + raw-router aux validation. Bias is now
        # shaped recursive_loops × num_routed_experts (was block-level),
        # so loop-specific imbalances can't cancel in aggregate. Aux
        # regularizes pre-bias raw router probs, so bias can't mask
        # raw concentration. Fresh ckpt dir because bias buffer shape
        # changed — resume from the prior (block-level) ckpts would
        # fail the state_dict shape check.
        ckpt_dir="/vol/checkpoints/v5-sanity-biasloop",
    )


# =============================================================================
# GUMBEL SWEEP (runs B, C, D — A is the default sanity)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=14400,  # 4h max for 3 sequential runs
)
def sweep():
    """Run Gumbel schedule sweep configs B, C, D sequentially.

    A (tau=0.5, anneal=1000, aux=0.03) runs separately via --stage sanity.

    | Run | Aux  | Tau init | Anneal steps | Purpose                          |
    |-----|-----:|---------:|-------------:|----------------------------------|
    | B   | 0.03 | 0.8      | 1000         | Stronger early exploration       |
    | C   | 0.03 | 0.5      | 2000         | Same noise, slower decay         |
    | D   | 0.05 | 0.5      | 1000         | More balance pressure + explore  |
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.train import run_training
    from osrt.train_config import PretrainConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    model_config_kwargs = dict(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sweep_configs = [
        {
            "name": "B",
            "wandb_name": "osrt-sweep-B-tau0.8",
            "ckpt_dir": "/vol/checkpoints/v5-sweep-B",
            "aux_coeff": 0.03,
            "tau_init": 0.8,
            "anneal_steps": 1000,
        },
        {
            "name": "C",
            "wandb_name": "osrt-sweep-C-anneal2k",
            "ckpt_dir": "/vol/checkpoints/v5-sweep-C",
            "aux_coeff": 0.03,
            "tau_init": 0.5,
            "anneal_steps": 2000,
        },
        {
            "name": "D",
            "wandb_name": "osrt-sweep-D-aux0.05",
            "ckpt_dir": "/vol/checkpoints/v5-sweep-D",
            "aux_coeff": 0.05,
            "tau_init": 0.5,
            "anneal_steps": 1000,
        },
    ]

    for sc in sweep_configs:
        print("=" * 60)
        print(f"SWEEP RUN {sc['name']}: "
              f"aux={sc['aux_coeff']}, "
              f"tau={sc['tau_init']}→0 over {sc['anneal_steps']}")
        print("=" * 60)

        model_config = OSRTConfig(
            router_aux_loss_coeff=sc["aux_coeff"],
            **model_config_kwargs,
        )

        class SweepConfig(PretrainConfig):
            total_steps = 200
            warmup_steps = 25
            log_interval = 10
            eval_interval = 100
            ckpt_interval = 100
            early_stop_check_step = 10_000_000

        cfg = SweepConfig()
        cfg.router_gumbel_tau_init = sc["tau_init"]
        cfg.router_gumbel_anneal_steps = sc["anneal_steps"]
        cfg.wandb_run_name = sc["wandb_name"]

        os.makedirs(sc["ckpt_dir"], exist_ok=True)
        run_training(
            model_config, cfg, vol, tokenizer_path,
            ckpt_dir=sc["ckpt_dir"],
        )
        print(f"\n>>> Run {sc['name']} complete.\n")


# =============================================================================
# OPTIMIZER × ROUTING ABLATION (cells A/B/C/D, 1200 steps each)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=21600,  # 6h for 4 sequential 1200-step runs (~80-90 min total + headroom)
)
def ablate():
    """Optimizer × routing ablation, 1200 Foundation-matched steps per cell.

    Reads each cell against:
      - the four-metric clean health gate (Phase 1 success criteria)
      - the three new prebias guards (router not collapsed under bias)

    Cells:
      | Cell | Optimizer   | Aux  | Routing      | Purpose              |
      |------|-------------|-----:|--------------|----------------------|
      | A    | Lion        | 0.10 | aux + bias   | old optimizer base   |
      | B    | Lion        | 0.0  | bias only    | aux-loss isolation   |
      | C    | Muon hybrid | 0.10 | aux + bias   | production default   |
      | D    | Muon hybrid | 0.0  | bias only    | aux-free failure     |

    Reading guide:
      - If A passes the clean gate but B fails marginal_entropy below 1.5 →
        the bias controller alone can't hold balance at this scale; keep aux.
      - If A and C both pass but C reaches lower task loss at step 1200 →
        Muon is paying off on the matrix updates; keep it for full pretrain.
      - If any cell trips a prebias guard (clean passes, raw collapses) →
        the bias controller is hiding raw-router collapse and the cell is
        misleading; do NOT promote that recipe to a full run.

    Each cell runs 1200 steps with Foundation-matched warmup (3000) so the
    first ~1000 steps are LR-warmup territory — exactly when v4 saw expert
    death. The 5k clean health gate is disabled because 1200 steps isn't
    enough to calibrate it.
    """
    import os

    import modal as _modal
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.train import run_training
    from osrt.train_config import PretrainConfig

    _tok_vol = _modal.Volume.from_name("osrt-v4-tokenizer")
    _tok_vol.reload()

    tokenizer_path = "/vol/tokenizer"
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"Tokenizer loaded: vocab_size={len(tok)}")

    model_config_kwargs = dict(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    cells = [
        {
            "name": "A",
            "label": "lion+aux (baseline)",
            "wandb_name": "osrt-ablate-A-lion-aux",
            "ckpt_dir": "/vol/checkpoints/v5-ablate-A",
            "optimizer_name": "lion",
            "aux_coeff": 0.10,
        },
        {
            "name": "B",
            "label": "lion+bias-only",
            "wandb_name": "osrt-ablate-B-lion-biasonly",
            "ckpt_dir": "/vol/checkpoints/v5-ablate-B",
            "optimizer_name": "lion",
            "aux_coeff": 0.0,
        },
        {
            "name": "C",
            "label": "muon+aux",
            "wandb_name": "osrt-ablate-C-muon-aux",
            "ckpt_dir": "/vol/checkpoints/v5-ablate-C",
            "optimizer_name": "muon",
            "aux_coeff": 0.10,
        },
        {
            "name": "D",
            "label": "muon+bias-only",
            "wandb_name": "osrt-ablate-D-muon-biasonly",
            "ckpt_dir": "/vol/checkpoints/v5-ablate-D",
            "optimizer_name": "muon",
            "aux_coeff": 0.0,
        },
    ]

    for cell in cells:
        # Skip cells that already produced a final checkpoint. Lets us
        # crash-recover the ablation without paying for cells that
        # already finished — important because cell A is ~$1 of compute.
        final_ckpt = f"{cell['ckpt_dir']}/osrt_v5_final.pt"
        if os.path.exists(final_ckpt):
            print("=" * 60)
            print(
                f"ABLATE CELL {cell['name']}: SKIP — final checkpoint "
                f"already exists at {final_ckpt}"
            )
            print("=" * 60)
            print(f"\n>>> Cell {cell['name']} ({cell['label']}) skipped.\n")
            continue

        print("=" * 60)
        print(
            f"ABLATE CELL {cell['name']}: {cell['label']} "
            f"(optimizer={cell['optimizer_name']}, aux={cell['aux_coeff']})"
        )
        print("=" * 60)

        # Each cell carries the new architectural defaults from today's
        # session: Z-loss on, seq-balance off, QK-Norm always-on,
        # softplus moe_gate, bias controller on. Only optimizer + aux
        # coefficient vary across cells.
        model_config = OSRTConfig(
            router_aux_loss_coeff=cell["aux_coeff"],
            router_balance_bias_enabled=True,
            **model_config_kwargs,
        )

        class AblateConfig(PretrainConfig):
            # 1200 Foundation-matched steps — long enough to see expert
            # death during LR warmup but short enough that 4 cells fit
            # in one Modal run.
            total_steps = 1200
            warmup_steps = 3000
            log_interval = 50
            # Eval skipped — pays a 10-15 min FineWeb skip for telemetry
            # we already get from the four-metric health gate at every step.
            eval_interval = 10_000_000
            ckpt_interval = 600
            # Match the production Gumbel schedule so noise survives peak LR.
            router_gumbel_tau_init = 0.5
            router_gumbel_tau_final = 0.0
            router_gumbel_anneal_steps = 4000
            # 5k gate is calibrated for the full Foundation phase — at 1200
            # steps it would always trip, so disable it. Read the clean gate
            # plus the three prebias guards manually from W&B instead.
            early_stop_check_step = 10_000_000

        cfg = AblateConfig()
        cfg.optimizer_name = cell["optimizer_name"]
        cfg.wandb_run_name = cell["wandb_name"]

        os.makedirs(cell["ckpt_dir"], exist_ok=True)
        run_training(
            model_config, cfg, vol, tokenizer_path,
            ckpt_dir=cell["ckpt_dir"],
        )
        print(f"\n>>> Cell {cell['name']} ({cell['label']}) complete.\n")


# =============================================================================
# SFT
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft():
    """Run balanced SFT on top of the Foundation+Knowledge checkpoint.

    Loads /vol/checkpoints/v5/osrt_v5_final.pt (set by SFTConfig), injects
    HRA adapters for extra capacity, and trains on the math+code+STEM+general
    mixture with v4-style packing (inherited from v4_sft_data unchanged).
    """
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.sft_train import run_sft
    from osrt.train_config import SFTConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sft_cfg = SFTConfig()
    run_sft(model_config, sft_cfg, vol, tok)


# =============================================================================
# SFT v1 — v6 system-prompt instruction tuning (on midtrain_final)
# =============================================================================
# Builds from the OSRT_605M_A288M preset (NOT bare OSRTConfig, which falls back
# to the v5 363M shape) + the v6 tokenizer + fused-CE. Native HRA (hra_native)
# so run_sft skips inject_hra. See SFTv1Config and the SFT-v1 design spec.


def _run_sft_v1(cfg_cls):
    """Shared body for sft_v1 + sft_v1_sanity (differ only by config)."""
    from transformers import AutoTokenizer

    from osrt.presets import build_config
    from osrt.sft_train import run_sft

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")
    print(f"Tokenizer loaded: vocab_size={len(tok)}")
    model_config = build_config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
    )
    cfg = cfg_cls()
    print(
        f"{cfg.__class__.__name__}: {cfg.total_steps} steps @ seq {cfg.seq_len}, "
        f"system_tag={cfg.system_tag}, hra_native={cfg.hra_native}, "
        f"base={cfg.pretrained_checkpoint}"
    )
    run_sft(model_config, cfg, vol, tok)


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_v1():
    """v6 SFT v1: system-prompt instruction tuning on midtrain_final.
    seq 2048, reasoning-conditioned <|system|> turns, ~2000 steps."""
    from osrt.train_config import SFTv1Config
    _run_sft_v1(SFTv1Config)


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_v1_sanity():
    """30-step SFT-v1 gate: native-HRA load + <|system|> build + VRAM, no eval."""
    from osrt.train_config import SFTv1SanityConfig
    _run_sft_v1(SFTv1SanityConfig)


@app.local_entrypoint()
def run_sft_v1():
    """Spawn v6 SFT v1 (fire-and-forget)."""
    call = sft_v1.spawn()
    print(f"Spawned v6 sft_v1 — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


@app.local_entrypoint()
def run_sft_v1_sanity():
    """Spawn the 30-step SFT-v1 sanity gate."""
    call = sft_v1_sanity.spawn()
    print(f"Spawned v6 sft_v1 sanity — call_id={call.object_id}")
    print("Monitor: modal app logs <app-id>")


# =============================================================================
# SFT-LONG (long-context follow-up SFT, seq_len 4096, Nemotron-heavy mix)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_long():
    """Long-context SFT (seq_len 4096) resuming from osrt_v5_sft_final.pt.

    Configures a 1000-step run on a Nvidia-Nemotron-heavy data mix
    (math + stem + code + tool_calling = 75% Nemotron, 25% diversity)
    with HRA already loaded from the base SFT pass. Cooler LR
    (5e-6 peak) since we're fine-tuning a fine-tune.
    """
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.sft_train import run_sft
    from osrt.train_config import SFTLongConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sft_long_cfg = SFTLongConfig()
    run_sft(model_config, sft_long_cfg, vol, tok)


# =============================================================================
# SFT-ULTRALONG (seq_len 8192, resumes from sft_long_final.pt)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_ultralong():
    """Ultra-long-context SFT (seq_len 8192) resuming from sft_long_final.pt.

    500 steps at the same Nemotron-heavy mix, batch 2 × accum 32 to
    keep effective batch at 64 sequences within H100 80GB at seq 8192.
    Cooler LR (3e-6 peak) for the third successive fine-tune.
    """
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.sft_train import run_sft
    from osrt.train_config import SFTUltraLongConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sft_ultralong_cfg = SFTUltraLongConfig()
    run_sft(model_config, sft_ultralong_cfg, vol, tok)


# =============================================================================
# SFT_REFRESH (short SFT pass to re-anchor chat format after extend)
# =============================================================================
#
# Local probe on osrt_v5_extend_final.pt showed chat-format degradation
# (special tokens emitted in wrong positions, <|/answer|> never closes,
# some prompts produce immediate EOS) despite the 25 % rehearsal mix
# during pretrain_extend. This stage runs a short, low-LR SFT on top
# of the extended ckpt to re-anchor the format wrapping. The math /
# code learning the extend gave us is preserved; only the format
# placement is re-tuned.
#
# 500 steps at seq 2048 ≈ 50 min on H100 ≈ ~$3-4. Output:
# osrt_v5_sft_refresh_step_N.pt + osrt_v5_sft_refresh_final.pt.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_refresh():
    """Short SFT format-anchor pass on top of pretrain_extend ckpt.

    See train_config.py::SFTRefreshConfig for the full design
    rationale. 500 steps, peak LR 5e-6 (33 % of base SFT), HRA
    trainable, NO tool_calling in the data mix (preserves the
    anti-hallucination win from extend).
    """
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.sft_train import run_sft
    from osrt.train_config import SFTRefreshConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sft_refresh_cfg = SFTRefreshConfig()
    print(
        "sft_refresh: 500 steps, peak LR 5e-6, HRA trainable, "
        "no tool_calling. Goal: re-anchor chat-format emission "
        "after pretrain_extend.",
    )
    print(f"Resume base: {sft_refresh_cfg.pretrained_checkpoint}")
    run_sft(model_config, sft_refresh_cfg, vol, tok)


# =============================================================================
# SFT_MATH (math-only SFT pass between sft_refresh and GRPO)
# =============================================================================
#
# Math probe of sft_refresh_final.pt revealed think→answer decoupling
# (think block had correct steps, answer block ignored them and
# emitted random wrong content). 200 steps of pure math SFT trains
# the answer block to commit to the think block's conclusion.
# Cheap (~$1.30) and gives GRPO a coherent base before RL kicks in.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def sft_math():
    """Math-only SFT polish on top of sft_refresh_final.pt.

    See train_config.py::SFTMathConfig for design rationale. 200
    steps, pure math mix (GSM8K + Orca-Math + MathInstruct +
    NuminaMath-CoT, all warm-cached on gradio-winter-hack), peak
    LR 3e-6. Goal: tighten think→answer correlation before GRPO.
    """
    from transformers import AutoTokenizer

    from osrt.config import OSRTConfig
    from osrt.sft_train import run_sft
    from osrt.train_config import SFTMathConfig

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    sft_math_cfg = SFTMathConfig()
    print(
        "sft_math: 1,000 steps, peak LR 3e-6, math-only mix. "
        "Goal: tighten think→answer correlation before GRPO.",
    )
    print(f"Resume base: {sft_math_cfg.pretrained_checkpoint}")
    run_sft(model_config, sft_math_cfg, vol, tok)


# =============================================================================
# EVALUATE (lm-eval-harness pass: gsm8k + IFEval + MMLU-stem)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,  # lm-eval is in the base image's pip_install
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer_v4": tokenizer_vol,
        "/vol/tokenizer_v6": v6_tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=14400,  # 4h: comfortable headroom for full lm-eval suite
)
def evaluate(
    ckpt_name: str = "osrt_v5_sft_v2_final.pt",
    tag: str = "sft-v2",
    tasks: str = (
        # Original three: math reasoning + instruction following + STEM knowledge
        "gsm8k,ifeval,mmlu_stem,"
        # Small-model commonsense suite — gives a clean GPT-2-class
        # comparison anchor. GPT-2 medium (355M) reference numbers:
        #   hellaswag 33%, arc_easy 49%, arc_challenge 22%,
        #   piqa 63%, winogrande 52%.
        # All five are pure loglikelihood scoring (no generation),
        # so total added cost is ~$3-4 on top of the generate-heavy
        # gsm8k + ifeval. Adds ~30 min to the full pass.
        "hellaswag,arc_easy,arc_challenge,piqa,winogrande"
    ),
    limit: int | None = None,
    tokenizer_path: str = "/vol/tokenizer_v6",
    hra_native: bool = True,
    hra_enabled: bool = True,
):
    """Run lm-evaluation-harness on an OSRT checkpoint.

    Runs gsm8k + IFEval + MMLU-stem by default. Results are written to
    `/vol/checkpoints/v5/eval_<tag>.json` so pre-GRPO and post-GRPO
    runs can be diffed straightforwardly.

    Args:
        ckpt_name: filename under /vol/checkpoints/v5/. Default points at
            the active v6 SFT-v2 final ckpt. For legacy v4/v5 checkpoints,
            pass tokenizer_path="/vol/tokenizer_v4" and hra_native=False.
        tag: short label embedded in the output filename and W&B run.
            Use "pre-grpo" / "post-grpo" / "post-iter-grpo" for the
            comparison sequence.
        tasks: comma-separated lm-eval task names. Defaults match the
            three benchmarks we care about; override for ad-hoc runs.
        limit: cap problems per task for quick smoke tests. None =
            full benchmark. 50 is a reasonable smoke value.
        tokenizer_path: mounted tokenizer directory. Defaults to the v6
            65K tokenizer; legacy ckpts should use /vol/tokenizer_v4.
        hra_native: True for the v6 preset-native adapters_a/adapters_b lane.
            False for legacy HRALinear checkpoints that need injection.
        hra_enabled: Whether the checkpoint includes HRA parameters.

    Cost: ~$5 for the default three-task pass on H100 (gsm8k 1319
    problems × 256 generated tokens dominates).
    """
    import json
    import os

    try:
        import wandb
    except ImportError:
        wandb = None

    from lm_eval import simple_evaluate

    from osrt.lm_eval_wrapper import OSRTLMEval

    ckpt_path = f"/vol/checkpoints/v5/{ckpt_name}"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Available files: "
            f"{sorted(os.listdir('/vol/checkpoints/v5'))[:10]}",
        )

    print("=" * 60)
    print(f"OSRT — lm-eval-harness ({tag})")
    print("=" * 60)
    print(f"Checkpoint     : {ckpt_path}")
    print(f"Tokenizer      : {tokenizer_path}")
    print(f"HRA native     : {hra_native}")
    print(f"Tasks          : {tasks}")
    print(f"Per-task limit : {limit or 'full'}")
    print()

    wrapper = OSRTLMEval(
        ckpt_path=ckpt_path,
        tokenizer_path=tokenizer_path,
        hra_enabled=hra_enabled,
        hra_rank=256,
        hra_native=hra_native,
        batch_size=8,
        device="cuda",
    )

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]

    if wandb is not None:
        wandb.init(
            project="osrt",
            name=f"osrt-eval-{tag}",
            config={
                "stage": "evaluate",
                "ckpt_name": ckpt_name,
                "tasks": task_list,
                "limit": limit,
                "tokenizer_path": tokenizer_path,
                "hra_native": hra_native,
                "hra_enabled": hra_enabled,
            },
        )

    # Sample logging is gated on the smoke run — when limit is set
    # (i.e. iterating on the wrapper), we want every prompt+response
    # in the JSON to debug. Full eval (limit=None) skips it because
    # the transcript dump bloats the JSON ~100×.
    log_samples = limit is not None
    print(
        f"Running lm-eval on {task_list}... "
        f"(limit={limit}, log_samples={log_samples})",
        flush=True,
    )
    results = simple_evaluate(
        model=wrapper,
        tasks=task_list,
        limit=limit,
        log_samples=log_samples,
    )

    # Strip the bulky "model_dump" entry. Keep "samples" iff log_samples
    # was on (smoke runs) so we can read what the model actually emitted
    # — the whole point of running smokes.
    summary = {
        "tag": tag,
        "ckpt_name": ckpt_name,
        "tasks": task_list,
        "limit": limit,
        "results": results.get("results", {}),
        "configs": {k: v.get("task", k) for k, v in results.get("configs", {}).items()},
    }
    if log_samples and "samples" in results:
        # Cap to first 5 per task to keep JSON readable even at limit=50.
        # Five samples per task is enough to see if formatting / extraction
        # is working without burying us in transcripts.
        capped_samples = {}
        for task_name, sample_list in results["samples"].items():
            capped_samples[task_name] = sample_list[:5]
        summary["samples"] = capped_samples

    out_path = f"/vol/checkpoints/v5/eval_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults written to {out_path}", flush=True)

    # Print headline numbers for stdout/Modal log readability.
    print("\n=== Headline metrics ===")
    for task_name, task_results in summary["results"].items():
        print(f"\n[{task_name}]")
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")

    if wandb is not None:
        # Log flat metrics so they're queryable + plottable across runs.
        for task_name, task_results in summary["results"].items():
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    wandb.log({f"eval/{task_name}/{metric}": value})
        wandb.finish()

    vol.commit()


# =============================================================================
# GRPO (REINFORCEMENT LEARNING)
# =============================================================================


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def grpo():
    """Run GRPO with verifiable math rewards."""
    import copy
    import math
    import os
    import time

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer

    try:
        import wandb
    except ImportError:
        wandb = None

    from osrt.config import OSRTConfig
    from osrt.hra import get_param_groups, inject_hra
    from osrt.model import OSRTForCausalLM
    from osrt.rewards import compute_group_advantages, compute_reward
    from osrt.train import apply_router_balance_updates, load_model_state_or_raise
    from osrt.train_config import GRPOConfig

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = GRPOConfig()
    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    print("=" * 60)
    print("OSRT — GRPO Training")
    print("=" * 60)

    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
    )

    model = OSRTForCausalLM(model_config).to(device)

    # Inject HRA before loading SFT checkpoint
    hra_params = []
    if cfg.hra_enabled:
        print(f"Injecting HRA (rank={cfg.hra_rank})...")
        hra_params = inject_hra(model, rank=cfg.hra_rank)

    # Load SFT weights — GRPO MUST start from a real SFT checkpoint.
    ckpt_path = cfg.pretrained_checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"GRPO refuses to start: SFT checkpoint not found at {ckpt_path}. "
            "Run SFT first (modal run app_v5.py --stage sft)."
        )

    print(f"Loading SFT weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    load_model_state_or_raise(
        model, state_dict, context=f"GRPO SFT load from {ckpt_path}",
    )
    print("  Clean load: all keys matched.")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Group size: {cfg.group_size}")
    print(f"Total steps: {cfg.total_steps}")

    # Reference model
    print("Creating frozen reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    print("Compiling policy model...")
    model = torch.compile(model)
    # Uncompiled handle for rollout — KV-cached generate() uses
    # eager-mode forward per decode step (shape changes each step
    # would trigger recompilation anyway).
    inner_for_gen = model._orig_mod if hasattr(model, "_orig_mod") else model
    # Hold the policy in eval mode for the entire GRPO step so that the
    # rollout (generate) and the log-prob recompute (model(...)) see the
    # same routing distribution. With train(True) the MoE layer enforces
    # capacity drops (model.py:394-398), so dropped (token, expert) pairs
    # collapse to "shared expert + residual" only — different logits than
    # the no-drop rollout. That makes the assumed importance ratio ≈ 1
    # invalid and biases the policy gradient. The bias controller's
    # accumulators are gated on self.training so they simply don't update
    # during GRPO; the controller is already learned in pretrain.
    inner_for_gen.train(False)
    ref_model.train(False)

    # W&B
    use_wandb = cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={"stage": "grpo"},
        )

    # Optimizer
    if hra_params:
        param_groups = get_param_groups(
            model, hra_params, cfg.peak_lr, cfg.hra_lr, cfg.weight_decay,
        )
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.peak_lr,
                                       weight_decay=cfg.weight_decay, betas=(0.9, 0.95))

    # Prompt dataset
    print("Loading prompt dataset...")
    load_kwargs = {"split": cfg.prompt_split, "streaming": True}
    if cfg.prompt_config:
        load_kwargs["name"] = cfg.prompt_config
    prompt_ds = load_dataset(cfg.prompt_dataset, **load_kwargs)
    prompt_ds = prompt_ds.shuffle(buffer_size=2_000, seed=42)
    prompt_iter = iter(prompt_ds)

    # Resume. GRPO was previously write-only — it would reload the base
    # SFT weights on every launch and drop any partial progress. Now we
    # scan for existing grpo step and rescue checkpoints, prefer rescue
    # on ties (same logic as pretrain/sft), and start_step from there.
    ckpt_dir = "/vol/checkpoints/v5"
    os.makedirs(ckpt_dir, exist_ok=True)
    import glob as _glob
    best_grpo_step = -1
    best_grpo_ckpt: str | None = None
    for pattern in (
        f"{ckpt_dir}/osrt_v5_grpo_step_*.pt",
        f"{ckpt_dir}/osrt_v5_grpo_rescue_step_*.pt",
    ):
        for f in _glob.glob(pattern):
            try:
                s = int(f.rsplit("_", 1)[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            if s > best_grpo_step or (
                s == best_grpo_step and "rescue" in f
            ):
                best_grpo_step = s
                best_grpo_ckpt = f

    start_step = 0
    if best_grpo_step > 0 and best_grpo_ckpt is not None:
        print(
            f"Found grpo checkpoint at step {best_grpo_step}: "
            f"{best_grpo_ckpt}",
        )
        grpo_ckpt = torch.load(
            best_grpo_ckpt, map_location=device, weights_only=True,
        )
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        load_model_state_or_raise(
            inner,
            grpo_ckpt["model_state_dict"],
            context=f"GRPO resume from {best_grpo_ckpt}",
        )
        try:
            optimizer.load_state_dict(grpo_ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"  Optimizer state mismatch, starting fresh: {e}")
        # Fall back to the filename-extracted step (best_grpo_step) when
        # the ckpt itself doesn't carry a "step" field — happens when a
        # final.pt is renamed to step_N.pt (the final-save path only
        # writes model_state_dict, not the step int, so naive resume
        # crashes with KeyError: 'step'). Caught during the
        # 500→700 extension restart.
        start_step = grpo_ckpt.get("step", best_grpo_step) + 1
        # Do NOT rebuild ref_model here. ref_model was frozen from the
        # SFT-loaded policy at line 470 and must remain the SFT anchor.
        # Rebuilding it from the resumed (already-drifted) policy would
        # make KL penalize drift from the drifted policy, not the SFT
        # baseline, so restarting would silently change the objective.
        print(f"  Resumed at step {start_step}")

    # Training loop
    start_time = time.time()

    for step in range(start_step, cfg.total_steps):
        # LR schedule. lr_anchor_step lets a resumed run re-warm:
        # the warmup/cosine treats `step - anchor` as the effective
        # step, so the new phase gets a real gradient instead of
        # the near-zero LR a continued cosine would yield.
        anchor = getattr(cfg, "lr_anchor_step", 0)
        eff_step = max(step - anchor, 0)
        eff_total = max(cfg.total_steps - anchor, 1)
        if eff_step < cfg.warmup_steps:
            lr = cfg.peak_lr * eff_step / cfg.warmup_steps
        else:
            progress = (eff_step - cfg.warmup_steps) / max(
                eff_total - cfg.warmup_steps, 1,
            )
            lr = cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in optimizer.param_groups:
            if pg.get("group_name") == "hra":
                pg["lr"] = lr * (cfg.hra_lr / cfg.peak_lr)
            else:
                pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_kl = 0.0
        step_rewards = []
        step_correct = 0
        step_total = 0

        for _accum in range(cfg.grad_accum_steps):
            try:
                example = next(prompt_iter)
            except StopIteration:
                prompt_ds = load_dataset(cfg.prompt_dataset, **load_kwargs)
                prompt_ds = prompt_ds.shuffle(buffer_size=2_000, seed=42 + step)
                prompt_iter = iter(prompt_ds)
                example = next(prompt_iter)

            question = example["question"]
            ground_truth = example["answer"]

            prompt_text = f"{cfg.user_tag}{question}{cfg.assistant_tag}"
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            prompt_len = len(prompt_ids)

            # Batched rollout using KV-cached generate(). The previous
            # implementation did group_size sequential loops, each
            # feeding the full prefix back into the compiled model
            # every step — O(N^2) per token and sequential across the
            # group. Replicating the prompt group_size times and calling
            # generate() once uses the per-effective-layer KV cache
            # built into OSRTForCausalLM.generate(), decoding all
            # group_size samples in parallel at O(1) attention cost
            # per step.
            prompt_batch = prompt_tensor.expand(
                cfg.group_size, -1,
            ).contiguous()
            with torch.no_grad():
                generated_batch = inner_for_gen.generate(
                    prompt_batch,
                    max_new_tokens=cfg.max_gen_len,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    eos_token_id=tok.eos_token_id,
                )
            # generate() pads finished rows with EOS so the batch stays
            # rectangular. Truncate each row to its first EOS in the
            # completion region (inclusive) so downstream scoring and
            # policy log-prob computation don't see the EOS padding.
            comp_region_batch = generated_batch[:, prompt_len:]
            eos_hits_2d = (comp_region_batch == tok.eos_token_id)
            indices = torch.arange(eos_hits_2d.shape[1], device=generated_batch.device)
            masked_indices = torch.where(eos_hits_2d, indices, eos_hits_2d.shape[1])
            first_eos_indices = masked_indices.min(dim=1).values.tolist()

            completions = []
            for row_idx, first_eos in enumerate(first_eos_indices):
                if first_eos < eos_hits_2d.shape[1]:
                    completions.append(generated_batch[row_idx, : prompt_len + first_eos + 1])
                else:
                    completions.append(generated_batch[row_idx])

            # Score — IMPORTANT: skip_special_tokens=False so native tags
            # like <|think|>, <|answer|> survive decoding for the reward
            # scorer. And explicitly pass the v4 native tag strings so
            # the reward function doesn't fall back to v3 defaults.
            rewards = []
            for comp_ids in completions:
                comp_text = tok.decode(
                    comp_ids[prompt_len:].tolist(),
                    skip_special_tokens=False,
                )
                comp_tokens = len(comp_ids) - prompt_len
                reward, breakdown = compute_reward(
                    comp_text, ground_truth,
                    correctness_weight=cfg.correctness_reward,
                    format_weight=cfg.format_reward,
                    length_penalty=cfg.length_penalty,
                    think_open=cfg.think_open,
                    think_close=cfg.think_close,
                    answer_open=cfg.answer_open,
                    answer_close=cfg.answer_close,
                    max_tokens=cfg.max_gen_len,
                    completion_tokens=comp_tokens,
                    reasoning_bonus=cfg.reasoning_bonus,
                    truncation_penalty=cfg.truncation_penalty,
                    empty_think_penalty=cfg.empty_think_penalty,
                )
                rewards.append(reward)
                if breakdown["correct"]:
                    step_correct += 1
                step_total += 1
            step_rewards.extend(rewards)

            advantages = compute_group_advantages(rewards)

            for comp_ids, adv in zip(completions, advantages):
                if abs(adv) < 1e-8:
                    continue
                comp_ids = comp_ids[:cfg.seq_len].to(device)
                comp_len = len(comp_ids) - prompt_len
                if comp_len <= 0:
                    continue

                # Policy log probs on the sampled completion
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(comp_ids.unsqueeze(0))
                    logits = out.logits[0, :, :model_config.real_vocab_size].float()
                shift_logits = logits[prompt_len - 1:-1]
                shift_labels = comp_ids[prompt_len:]
                policy_lp = F.log_softmax(shift_logits, dim=-1).gather(
                    1, shift_labels.unsqueeze(1)
                ).squeeze(1)

                # Reference log probs (frozen, no grad)
                with torch.no_grad():
                    ref_out = ref_model(comp_ids.unsqueeze(0))
                    ref_logits = ref_out.logits[
                        0, :, :model_config.real_vocab_size
                    ].float()
                ref_shift = ref_logits[prompt_len - 1:-1]
                ref_lp = F.log_softmax(ref_shift, dim=-1).gather(
                    1, shift_labels.unsqueeze(1)
                ).squeeze(1)

                # Direct policy gradient weighted by group-normalised advantage.
                # Since we perform only one gradient step per sampled batch,
                # importance-sampling ratio ~= 1, so PPO clipping is a no-op
                # here. We keep the formulation simple and correct.
                adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
                policy_loss = -(policy_lp * adv_t).mean()

                # Schulman's unbiased non-negative KL approximation:
                #   approx_kl = exp(ref_lp - policy_lp) - (ref_lp - policy_lp) - 1
                # Always >= 0 (unlike the simple mean(policy_lp - ref_lp) which
                # can go negative and give a bogus "negative KL" penalty).
                log_ratio = ref_lp - policy_lp
                approx_kl = (torch.exp(log_ratio) - log_ratio - 1).mean()

                loss = (policy_loss + cfg.kl_coeff * approx_kl) / cfg.grad_accum_steps
                loss.backward()
                step_loss += loss.item()
                step_kl += approx_kl.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        apply_router_balance_updates(model)

        # Logging
        if step % cfg.log_interval == 0 or step == 0:
            mean_reward = sum(step_rewards) / len(step_rewards) if step_rewards else 0
            accuracy = step_correct / step_total if step_total > 0 else 0
            elapsed = time.time() - start_time
            vram = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            mean_kl = step_kl / max(step_total, 1)
            print(f"step {step:>6d}/{cfg.total_steps} | loss {step_loss:.4f} | "
                  f"reward {mean_reward:.3f} | acc {accuracy:.1%} | "
                  f"kl {mean_kl:.4f} | lr {lr:.2e} | "
                  f"vram {vram:.1f}GB | elapsed {elapsed:.0f}s")
            if use_wandb:
                wandb.log({
                    "grpo/loss": step_loss,
                    "grpo/mean_reward": mean_reward,
                    "grpo/accuracy": accuracy,
                    "grpo/approx_kl": mean_kl,
                    "grpo/lr": lr,
                }, step=step)

        # Checkpoints
        if step > 0 and step % cfg.ckpt_interval == 0:
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({"step": step, "model_state_dict": inner.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict()},
                       f"{ckpt_dir}/osrt_v5_grpo_step_{step}.pt")
            vol.commit()

        # 23h safety. Filename includes the step so the resume scanner
        # can rank it against numbered checkpoints (same convention as
        # pretrain/sft).
        if time.time() - start_time > 82_800:
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            rescue_path = (
                f"{ckpt_dir}/osrt_v5_grpo_rescue_step_{step}.pt"
            )
            torch.save({
                "step": step,
                "model_state_dict": inner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, rescue_path)
            vol.commit()
            print(f"\n23h boundary at step {step}. Rescue: {rescue_path}")
            if use_wandb:
                wandb.finish()
            return

    # Final
    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save({"model_state_dict": inner.state_dict(), "training_stage": "grpo"},
               f"{ckpt_dir}/osrt_v5_grpo_final.pt")
    vol.commit()
    elapsed_h = (time.time() - start_time) / 3600
    print(f"\nGRPO complete. {cfg.total_steps} steps in {elapsed_h:.1f}h")
    if use_wandb:
        wandb.finish()


# =============================================================================
# GRPO_MULTI — multi-env GRPO from mopd_final.pt
# =============================================================================
# Same PPO-style loop as grpo() but with:
#   - per micro-batch env sampling (math 60% / ifeval 30% / mbpp 10%)
#   - env-aware prompt fetcher + ground-truth extractor
#   - env-aware reward dispatcher (compose_template_rewards + per-env scorer)
#   - per-env wandb keys (math_acc, ifeval_constraints_hit_rate, mbpp_pass_rate)
#   - stop_token_ids during rollout for clean <|/answer|> halt
#   - RewardEMA logging
# Resumes from mopd_final.pt. See train_config.py::MultiEnvGRPOConfig.


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=86400,
)
def grpo_multi():
    """Run multi-env GRPO from mopd_final.pt."""
    _run_grpo_multi(sanity=False)


@app.function(
    gpu="H100",
    image=image,
    volumes={
        "/vol/checkpoints": vol,
        "/vol/tokenizer": tokenizer_vol,
        "/vol/hf_cache": hf_cache_vol,
    },
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("hf-secret"),
    ],
    timeout=3600,
)
def grpo_multi_sanity():
    """30-step multi-env GRPO sanity: validate env dispatch, rollout,
    reward computation, KV-cached generation with stop tokens."""
    _run_grpo_multi(sanity=True)


def _run_grpo_multi(sanity: bool = False) -> None:
    """Multi-env GRPO training loop.

    Shared by grpo_multi (full) and grpo_multi_sanity (30-step smoke).
    """
    import copy
    import math
    import os
    import random as _random
    import time

    import torch
    import torch.nn.functional as F
    from datasets import load_dataset
    from transformers import AutoTokenizer

    try:
        import wandb
    except ImportError:
        wandb = None

    from osrt.config import OSRTConfig
    from osrt.hra import get_param_groups, inject_hra
    from osrt.model import OSRTForCausalLM
    from osrt.rewards import (
        RewardEMA,
        compose_template_rewards,
        compute_group_advantages,
        ifeval_constraint_reward,
        mbpp_test_reward,
    )
    from osrt.train import apply_router_balance_updates, load_model_state_or_raise
    from osrt.train_config import MultiEnvGRPOConfig

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    cfg = MultiEnvGRPOConfig()
    if sanity:
        # Sanity overrides — small everything, no compile, no wandb
        cfg.total_steps = 30
        cfg.warmup_steps = 5
        cfg.log_interval = 2
        cfg.ckpt_interval = 999_999
        cfg.wandb_log = False
        cfg.grad_accum_steps = 2  # smaller for fast iteration
        cfg.aux_loop_curriculum_steps = 0

    tok = AutoTokenizer.from_pretrained("/vol/tokenizer")

    print("=" * 60)
    print(f"OSRT — Multi-env GRPO {'(SANITY)' if sanity else ''}")
    print("=" * 60)
    print(f"  Envs: {dict(zip(cfg.env_names, cfg.env_weights))}")
    print(f"  Resume: {cfg.pretrained_checkpoint}")
    print(f"  Steps: {cfg.total_steps}, group_size: {cfg.group_size}, "
          f"max_gen_len: {cfg.max_gen_len}, kl_coeff: {cfg.kl_coeff}")
    print(f"  Stop token ids: {cfg.stop_token_ids}")

    # Model with architecture-fix knobs
    model_config = OSRTConfig(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        aux_loop_loss_weight=cfg.aux_loop_loss_weight,
        loop_dropout_prob=cfg.loop_dropout_prob,
        loop_dropout_min_loops=cfg.loop_dropout_min_loops,
        per_loop_aux_weights=cfg.per_loop_aux_weights,
    )
    model = OSRTForCausalLM(model_config).to(device)

    hra_params = []
    if cfg.hra_enabled:
        print(f"Injecting HRA (rank={cfg.hra_rank})...")
        hra_params = inject_hra(model, rank=cfg.hra_rank)

    ckpt_path = cfg.pretrained_checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"grpo_multi refuses to start: ckpt not found at {ckpt_path}. "
            "Upload mopd_final.pt to the osrt-checkpoints volume first.",
        )
    print(f"Loading base weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    state_dict = ckpt.get("model_state_dict", ckpt)
    load_model_state_or_raise(
        model, state_dict, context=f"grpo_multi load from {ckpt_path}",
    )
    print("  Clean load: all keys matched.")

    # Frozen reference for KL anchor
    print("Creating frozen reference model...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if not sanity:
        print("Compiling policy model...")
        model = torch.compile(model)
    inner_for_gen = model._orig_mod if hasattr(model, "_orig_mod") else model
    inner_for_gen.train(False)
    ref_model.train(False)

    # W&B
    use_wandb = cfg.wandb_log and wandb is not None
    if use_wandb:
        wandb.init(
            project=cfg.wandb_project,
            name=cfg.wandb_run_name,
            config={"stage": "grpo_multi"},
        )

    # Optimizer
    # HRA-only training mode freezes the base weights so only the
    # rank-256 adapters get gradient updates. Closes the capability
    # regression failure mode the step 75→150 runs revealed (base
    # weights drifted away from MOPD distribution under
    # policy-gradient pressure, costing capabilities). See
    # MultiEnvGRPOConfig.hra_only_training for full rationale.
    hra_only = bool(getattr(cfg, "hra_only_training", False))
    if hra_only and not hra_params:
        raise ValueError(
            "hra_only_training=True requires hra_enabled=True (no HRA "
            "params to train otherwise).",
        )
    if hra_only:
        # Freeze every parameter that isn't an HRA adapter. Use id()-set
        # rather than name matching so we're robust to torch.compile()
        # renaming and to any future HRA injection-point changes.
        hra_id_set = {id(p) for p in hra_params}
        frozen_count = 0
        frozen_param_count = 0
        for p in model.parameters():
            if id(p) not in hra_id_set:
                p.requires_grad = False
                frozen_count += 1
                frozen_param_count += p.numel()
        trainable_count = sum(p.numel() for p in hra_params)
        print(
            f"HRA-only training: froze {frozen_count} tensors "
            f"({frozen_param_count:,} params). "
            f"Trainable: {len(hra_params)} HRA tensors "
            f"({trainable_count:,} params).",
        )
        # Optimizer over HRA-only — single param group with group_name
        # = "hra" so the LR schedule below applies the hra_lr/peak_lr
        # cosine scaling instead of the default peak_lr scaling.
        optimizer = torch.optim.AdamW(
            [{
                "params": hra_params,
                "lr": cfg.hra_lr,
                "weight_decay": cfg.weight_decay,
                "group_name": "hra",
            }],
            betas=(0.9, 0.95), eps=1e-8,
        )
    elif hra_params:
        param_groups = get_param_groups(
            model, hra_params, cfg.peak_lr, cfg.hra_lr, cfg.weight_decay,
        )
        optimizer = torch.optim.AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.peak_lr,
            weight_decay=cfg.weight_decay, betas=(0.9, 0.95),
        )

    # ──────────────────────────────────────────────────────────────
    # Per-env prompt streams. Each env yields (prompt_text, gt_blob).
    # gt_blob is env-shaped: math = "#### N" string; ifeval = dict
    # with instruction_id_list + kwargs; mbpp_code = list of asserts.
    # We build one streaming iterator per env, then in the training
    # loop we sample which env to draw from per micro-batch.
    # ──────────────────────────────────────────────────────────────
    print("Loading per-env prompt datasets...")
    env_iters: dict[str, object] = {}
    env_ds_factories: dict[str, callable] = {}

    def _make_env_factory(env_name: str, ds_spec: dict):
        load_kwargs = {"split": ds_spec["split"], "streaming": True}
        if ds_spec.get("hf_config"):
            load_kwargs["name"] = ds_spec["hf_config"]

        def _build(seed: int):
            ds = load_dataset(ds_spec["hf_id"], **load_kwargs)
            try:
                ds = ds.shuffle(buffer_size=1_000, seed=seed)
            except Exception:
                pass
            return iter(ds)
        return _build

    for env_name in cfg.env_names:
        ds_spec = cfg.env_datasets[env_name]
        factory = _make_env_factory(env_name, ds_spec)
        env_ds_factories[env_name] = factory
        env_iters[env_name] = factory(seed=42)
        print(f"  [{env_name}] loaded from {ds_spec['hf_id']} "
              f"(split={ds_spec['split']})")

    def _next_example(env_name: str):
        """Get the next prompt + raw row from the env's iterator,
        re-creating the iterator if exhausted."""
        while True:
            try:
                return next(env_iters[env_name])
            except StopIteration:
                # Reshuffle with a different seed and continue
                env_iters[env_name] = env_ds_factories[env_name](
                    seed=42 + int(time.time()) % 100,
                )

    def _build_prompt_and_gt(env_name: str, ex: dict):
        """Env-aware prompt construction + ground-truth extraction.
        Returns (prompt_text, gt_blob) where gt_blob's shape depends
        on the env (see reward dispatcher below)."""
        ds_spec = cfg.env_datasets[env_name]
        prompt_field = ds_spec["prompt_field"]
        question = (ex.get(prompt_field) or "").strip()
        prompt_text = f"{cfg.user_tag}{question}{cfg.assistant_tag}"

        gt_format = ds_spec.get("ground_truth_format")
        if gt_format == "gsm8k_hash":
            gt = ex.get(ds_spec["gt_field"], "")
        elif gt_format == "ifeval_constraints":
            gt = {
                "instruction_id_list": ex.get("instruction_id_list") or [],
                "kwargs": ex.get("kwargs") or [],
            }
        elif gt_format == "mbpp_tests":
            gt = ex.get(ds_spec["gt_field"]) or []
        else:
            gt = None
        return prompt_text, gt

    def _score_completion(
        env_name: str, comp_text: str, gt: object,
    ) -> tuple[float, dict]:
        """Env-aware reward dispatcher. Returns (total_reward, breakdown).
        All envs get compose_template_rewards (shared format signal);
        ifeval/mbpp add their env-specific reward on top."""
        if env_name == "math":
            return compose_template_rewards(
                comp_text, ground_truth_answer=gt,
                think_open=cfg.think_open, think_close=cfg.think_close,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
                exact_format_reward=cfg.reward_exact_format,
                approx_format_pos=cfg.reward_approx_format_pos,
                approx_format_neg=cfg.reward_approx_format_neg,
                answer_check=True,
                number_check_reward=cfg.reward_number_match,
                number_check_penalty=cfg.reward_number_miss,
                strict_template_weight=cfg.reward_strict_template_weight,
                strict_extraction=getattr(cfg, "strict_answer_extraction", False),
                ambiguous_penalty=getattr(cfg, "ambiguous_answer_penalty", -0.5),
            )
        if env_name == "ifeval":
            total, bd = compose_template_rewards(
                comp_text, ground_truth_answer=None,
                think_open=cfg.think_open, think_close=cfg.think_close,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
                exact_format_reward=cfg.reward_exact_format,
                approx_format_pos=cfg.reward_approx_format_pos,
                approx_format_neg=cfg.reward_approx_format_neg,
                answer_check=False,
                strict_template_weight=cfg.reward_strict_template_weight,
            )
            ifeval_s, ifeval_bd = ifeval_constraint_reward(
                comp_text,
                instruction_id_list=gt["instruction_id_list"] if gt else None,
                kwargs_list=gt["kwargs"] if gt else None,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
            )
            total += ifeval_s
            bd["r_ifeval"] = ifeval_s
            bd["ifeval_verdict"] = ifeval_bd.get("verdict", "")
            bd["ifeval_hits"] = ifeval_bd.get("constraints_hit", 0)
            bd["ifeval_misses"] = ifeval_bd.get("constraints_miss", 0)
            bd["total_reward"] = total
            return total, bd
        if env_name == "mbpp_code":
            total, bd = compose_template_rewards(
                comp_text, ground_truth_answer=None,
                think_open=cfg.think_open, think_close=cfg.think_close,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
                exact_format_reward=cfg.reward_exact_format,
                approx_format_pos=cfg.reward_approx_format_pos,
                approx_format_neg=cfg.reward_approx_format_neg,
                answer_check=False,
                strict_template_weight=cfg.reward_strict_template_weight,
            )
            # Sandboxed exec: minimal env (no secrets), tempdir cwd,
            # process-group kill on timeout, absolute python path.
            # Modal containers ARE the outer isolation layer; this
            # in-process hardening is defence-in-depth. See
            # rewards.py::mbpp_test_reward for the full safety model.
            mbpp_s, mbpp_bd = mbpp_test_reward(
                comp_text,
                test_list=gt if isinstance(gt, list) else None,
                answer_open=cfg.answer_open, answer_close=cfg.answer_close,
                allow_unsafe_exec=True,  # explicit opt-in
            )
            total += mbpp_s
            bd["r_mbpp"] = mbpp_s
            bd["mbpp_verdict"] = mbpp_bd.get("verdict", "")
            bd["total_reward"] = total
            return total, bd
        raise ValueError(f"Unknown env: {env_name}")

    # Resume scan
    ckpt_dir = "/vol/checkpoints/v5"
    os.makedirs(ckpt_dir, exist_ok=True)
    import glob as _glob
    best_step = -1
    best_ckpt: str | None = None
    for pattern in (
        f"{ckpt_dir}/osrt_v5_{cfg.stage_prefix}_step_*.pt",
        f"{ckpt_dir}/osrt_v5_{cfg.stage_prefix}_rescue_step_*.pt",
    ):
        for f in _glob.glob(pattern):
            try:
                s = int(f.rsplit("_", 1)[1].split(".")[0])
            except (ValueError, IndexError):
                continue
            if s > best_step or (s == best_step and "rescue" in f):
                best_step = s
                best_ckpt = f
    start_step = 0
    if best_step > 0 and best_ckpt is not None:
        print(f"Found {cfg.stage_prefix} checkpoint at step {best_step}: {best_ckpt}")
        resume_ckpt = torch.load(best_ckpt, map_location=device, weights_only=True)
        inner = model._orig_mod if hasattr(model, "_orig_mod") else model
        load_model_state_or_raise(
            inner, resume_ckpt["model_state_dict"],
            context=f"grpo_multi resume from {best_ckpt}",
        )
        try:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        except Exception as e:
            print(f"  Optimizer state mismatch, starting fresh: {e}")
        start_step = resume_ckpt.get("step", best_step) + 1
        print(f"  Resumed at step {start_step}")

    # Reward EMA per env (signal-quality monitor)
    ema_overall = RewardEMA(alpha=0.1, print_every_n_calls=cfg.log_interval)
    ema_per_env = {n: RewardEMA(alpha=0.1) for n in cfg.env_names}

    # Cumulative capability counters across all steps. Lets us read a
    # success rate (rolling), not just per-step volatility. Window of
    # last N steps via deque would be cleaner; for now we accumulate
    # from start_step and report cumulative — the rate is monotonic
    # easier to interpret than a sliding window.
    cum_outcomes: dict[str, dict[str, int]] = {
        "math": {"exact": 0, "close": 0, "miss": 0},
        "ifeval": {"constraints_hit": 0, "constraints_miss": 0,
                   "no_answer": 0},
        "mbpp_code": {"all_pass": 0, "all_fail": 0, "timeout": 0,
                      "no_answer": 0, "other": 0},
    }

    # Env sampler — weighted random, seeded so reruns are reproducible
    env_rng = _random.Random(42 + start_step)

    def _sample_env() -> str:
        return env_rng.choices(cfg.env_names, weights=cfg.env_weights, k=1)[0]

    # ── OOD probe runner ──
    # Periodically evaluate the model on a held-out set the policy is
    # NOT training on. Lets us detect reward hacking DURING the run:
    # if training reward EMA climbs but OOD score drops, the model is
    # exploiting the reward function rather than learning the skill.
    ood_prompts = list(getattr(cfg, "ood_probe_prompts", ()))
    ood_interval = int(getattr(cfg, "ood_probe_interval", 0) or 0)

    def _run_ood_probe(at_step: int) -> dict:
        """Run all OOD prompts at low temp and return (score, breakdown).

        Score = fraction of prompts whose first <|answer|>...</|answer|>
        block contains the expected_answer substring (case-insensitive).
        Substring match is intentionally lenient — we're tracking
        generalization, not exact-format compliance (which the training
        rewards already cover).
        """
        if not ood_prompts:
            return {"score": 0.0, "total": 0, "hits": 0, "details": []}
        hits = 0
        details: list[dict] = []
        # Switch to eval mode for clean probe (no aux losses computed)
        inner_for_gen.train(False)
        for prompt_text, expected in ood_prompts:
            full_prompt = f"{cfg.user_tag}{prompt_text}{cfg.assistant_tag}"
            ids = tok.encode(full_prompt, add_special_tokens=False)
            t = torch.tensor([ids], dtype=torch.long, device=device)
            with torch.no_grad():
                gen = inner_for_gen.generate(
                    t,
                    max_new_tokens=int(
                        getattr(cfg, "ood_probe_max_new_tokens", 200),
                    ),
                    temperature=float(
                        getattr(cfg, "ood_probe_temperature", 0.3),
                    ),
                    top_p=cfg.top_p,
                    eos_token_id=tok.eos_token_id,
                    stop_token_ids=list(cfg.stop_token_ids),
                )
            comp = tok.decode(
                gen[0, len(ids):].tolist(),
                skip_special_tokens=False,
            )
            ans = extract_answer_text(
                comp,
                answer_open=cfg.answer_open,
                answer_close=cfg.answer_close,
            ) or ""
            # Lenient: substring match. Strict-format is the training
            # objective; this is generalization.
            hit = expected.lower() in ans.lower()
            if hit:
                hits += 1
            details.append({
                "prompt": prompt_text,
                "expected": expected,
                "answer": ans[:120],
                "hit": hit,
            })
        return {
            "score": hits / len(ood_prompts),
            "total": len(ood_prompts),
            "hits": hits,
            "details": details,
        }

    # Import the extractor lazily inside the probe but resolve it here
    # so we get a clear error if rewards.py is missing the symbol.
    from osrt.rewards import extract_answer_text  # noqa: E402

    # ── Troubleshoot-gen runner ──
    # Prints a single completion at the TRAINING temperature every N
    # steps. Different from ood_probe (low-temp deterministic eval) —
    # this shows what rollouts ACTUALLY look like at the temperature
    # being used for GRPO. Catches reward-hacking patterns by eye
    # (multi-answer-blocks, number dumping, format drift).
    troubleshoot_interval = int(
        getattr(cfg, "troubleshoot_gen_interval", 0) or 0,
    )
    troubleshoot_prompt = getattr(
        cfg, "troubleshoot_gen_prompt", "What is 17 * 23?",
    )

    def _run_troubleshoot_gen(at_step: int) -> None:
        inner_for_gen.train(False)
        full_prompt = f"{cfg.user_tag}{troubleshoot_prompt}{cfg.assistant_tag}"
        ids = tok.encode(full_prompt, add_special_tokens=False)
        t = torch.tensor([ids], dtype=torch.long, device=device)
        with torch.no_grad():
            gen = inner_for_gen.generate(
                t,
                max_new_tokens=int(
                    getattr(cfg, "troubleshoot_gen_max_new_tokens", 200),
                ),
                temperature=cfg.temperature,  # training temp
                top_p=cfg.top_p,
                eos_token_id=tok.eos_token_id,
                stop_token_ids=list(cfg.stop_token_ids),
            )
        completion = tok.decode(
            gen[0, len(ids):].tolist(), skip_special_tokens=False,
        )
        print(
            f"\n  ── troubleshoot-gen @ step {at_step} (T={cfg.temperature}) ──",
            flush=True,
        )
        print(f"  PROMPT: {troubleshoot_prompt}", flush=True)
        # Wrap completion at ~100 chars for readability
        for line in completion[:600].splitlines() or [""]:
            print(f"  | {line}", flush=True)
        if len(completion) > 600:
            print(f"  | ... ({len(completion) - 600} more chars)", flush=True)
        print("", flush=True)

    start_time = time.time()
    print(f"\nStarting training at step {start_step}...")
    if ood_prompts and ood_interval > 0:
        print(
            f"OOD probe: {len(ood_prompts)} prompts every {ood_interval} steps",
        )

    for step in range(start_step, cfg.total_steps):
        # LR schedule (cosine with re-warm anchor)
        anchor = getattr(cfg, "lr_anchor_step", 0)
        eff_step = max(step - anchor, 0)
        eff_total = max(cfg.total_steps - anchor, 1)
        if eff_step < cfg.warmup_steps:
            lr = cfg.peak_lr * eff_step / cfg.warmup_steps
        else:
            progress = (eff_step - cfg.warmup_steps) / max(
                eff_total - cfg.warmup_steps, 1,
            )
            lr = cfg.min_lr + 0.5 * (cfg.peak_lr - cfg.min_lr) * (
                1 + math.cos(math.pi * progress)
            )
        for pg in optimizer.param_groups:
            if pg.get("group_name") == "hra":
                pg["lr"] = lr * (cfg.hra_lr / cfg.peak_lr)
            else:
                pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_kl = 0.0
        step_rewards: list[float] = []
        step_env_rewards: dict[str, list[float]] = {n: [] for n in cfg.env_names}
        step_env_counts: dict[str, int] = {n: 0 for n in cfg.env_names}
        # Per-env CAPABILITY counters (separate from reward EMA). Detects
        # reward hacking: when env reward climbs but actual task success
        # rate stays flat, the model is gaming format rewards without
        # learning the underlying skill.
        #   math.exact   = exact-string or exact-numeric verdicts
        #   math.close   = within-5pct or within-20pct (partial credit)
        #   math.miss    = wrong, non_numeric_wrong, no_extract
        #   ifeval.constraints_hit / .constraints_miss (verifiable count)
        #   mbpp.all_pass / .partial / .all_fail / .timeout / .other
        step_env_outcomes: dict[str, dict[str, int]] = {
            "math": {"exact": 0, "close": 0, "miss": 0},
            "ifeval": {"constraints_hit": 0, "constraints_miss": 0,
                       "no_answer": 0},
            "mbpp_code": {"all_pass": 0, "all_fail": 0, "timeout": 0,
                          "no_answer": 0, "other": 0},
        }

        for _accum in range(cfg.grad_accum_steps):
            env_name = _sample_env()
            step_env_counts[env_name] += 1
            ex = _next_example(env_name)
            prompt_text, gt = _build_prompt_and_gt(env_name, ex)
            prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
            prompt_tensor = torch.tensor(
                [prompt_ids], dtype=torch.long, device=device,
            )
            prompt_len = len(prompt_ids)

            # Group rollout — KV-cached, batched, with stop tokens.
            prompt_batch = prompt_tensor.expand(cfg.group_size, -1).contiguous()
            with torch.no_grad():
                generated_batch = inner_for_gen.generate(
                    prompt_batch,
                    max_new_tokens=cfg.max_gen_len,
                    temperature=cfg.temperature,
                    top_p=cfg.top_p,
                    eos_token_id=tok.eos_token_id,
                    stop_token_ids=list(cfg.stop_token_ids),
                )

            # Truncate at first EOS / stop token in completion region.
            # Vectorised membership test — earlier impl did per-token .item()
            # inside a Python comprehension which forced one CPU-GPU sync
            # per generated token (review/performance-loop-audit P3).
            stop_ids_t = torch.tensor(
                [tok.eos_token_id, *cfg.stop_token_ids],
                device=generated_batch.device,
                dtype=generated_batch.dtype,
            )
            comp_region_batch = generated_batch[:, prompt_len:]
            stop_hits_2d = (comp_region_batch[:, :, None] == stop_ids_t[None, None, :]).any(dim=2)
            indices = torch.arange(stop_hits_2d.shape[1], device=generated_batch.device)
            masked_indices = torch.where(stop_hits_2d, indices, stop_hits_2d.shape[1])
            first_hit_indices = masked_indices.min(dim=1).values.tolist()

            completions = []
            for row_idx, first_hit in enumerate(first_hit_indices):
                if first_hit < stop_hits_2d.shape[1]:
                    completions.append(generated_batch[row_idx, : prompt_len + first_hit + 1])
                else:
                    completions.append(generated_batch[row_idx])

            # Score each completion using env-aware reward dispatcher.
            rewards: list[float] = []
            for comp_ids in completions:
                comp_text = tok.decode(
                    comp_ids[prompt_len:].tolist(),
                    skip_special_tokens=False,
                )
                r, bd = _score_completion(env_name, comp_text, gt)
                rewards.append(r)
                step_env_rewards[env_name].append(r)

                # Per-env capability counters — extract from bd
                if env_name == "math":
                    tier = bd.get("check_answer_tier", "")
                    if tier in ("exact", "exact_numeric"):
                        step_env_outcomes["math"]["exact"] += 1
                    elif tier in ("within_5pct", "within_20pct"):
                        step_env_outcomes["math"]["close"] += 1
                    else:
                        step_env_outcomes["math"]["miss"] += 1
                elif env_name == "ifeval":
                    verdict = bd.get("ifeval_verdict", "")
                    if verdict == "no_answer":
                        step_env_outcomes["ifeval"]["no_answer"] += 1
                    else:
                        step_env_outcomes["ifeval"]["constraints_hit"] += (
                            bd.get("ifeval_hits", 0)
                        )
                        step_env_outcomes["ifeval"]["constraints_miss"] += (
                            bd.get("ifeval_misses", 0)
                        )
                elif env_name == "mbpp_code":
                    verdict = bd.get("mbpp_verdict", "")
                    if verdict == "all_pass":
                        step_env_outcomes["mbpp_code"]["all_pass"] += 1
                    elif verdict == "all_fail":
                        step_env_outcomes["mbpp_code"]["all_fail"] += 1
                    elif verdict == "timeout":
                        step_env_outcomes["mbpp_code"]["timeout"] += 1
                    elif verdict == "no_answer":
                        step_env_outcomes["mbpp_code"]["no_answer"] += 1
                    else:
                        step_env_outcomes["mbpp_code"]["other"] += 1
            step_rewards.extend(rewards)

            advantages = compute_group_advantages(rewards)
            for comp_ids, adv in zip(completions, advantages):
                if abs(adv) < 1e-8:
                    continue
                comp_ids = comp_ids[:cfg.seq_len].to(device)
                comp_len = len(comp_ids) - prompt_len
                if comp_len <= 0:
                    continue

                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(comp_ids.unsqueeze(0))
                    logits = out.logits[
                        0, :, :model_config.real_vocab_size,
                    ].float()
                shift_logits = logits[prompt_len - 1:-1]
                shift_labels = comp_ids[prompt_len:]
                policy_lp = F.log_softmax(shift_logits, dim=-1).gather(
                    1, shift_labels.unsqueeze(1),
                ).squeeze(1)

                with torch.no_grad():
                    ref_out = ref_model(comp_ids.unsqueeze(0))
                    ref_logits = ref_out.logits[
                        0, :, :model_config.real_vocab_size,
                    ].float()
                ref_shift = ref_logits[prompt_len - 1:-1]
                ref_lp = F.log_softmax(ref_shift, dim=-1).gather(
                    1, shift_labels.unsqueeze(1),
                ).squeeze(1)

                adv_t = torch.tensor(adv, device=device, dtype=torch.float32)
                policy_loss = -(policy_lp * adv_t).mean()
                log_ratio = ref_lp - policy_lp
                approx_kl = (torch.exp(log_ratio) - log_ratio - 1).mean()
                loss = (
                    policy_loss + cfg.kl_coeff * approx_kl
                ) / cfg.grad_accum_steps
                loss.backward()
                step_loss += loss.item()
                step_kl += approx_kl.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        apply_router_balance_updates(model)

        # Per-env EMA updates
        for n, rs in step_env_rewards.items():
            if rs:
                ema_per_env[n].update(sum(rs) / len(rs))

        # Cumulative capability counter updates
        for env_name, step_buckets in step_env_outcomes.items():
            for bucket, n_hits in step_buckets.items():
                cum_outcomes[env_name][bucket] += n_hits
        mean_reward_step = (
            sum(step_rewards) / len(step_rewards) if step_rewards else 0.0
        )
        ema_overall.update(
            mean_reward_step,
            **{
                f"env_{n}": step_env_counts[n] for n in cfg.env_names
            },
        )

        # Logging
        if step % cfg.log_interval == 0 or step == 0:
            elapsed = time.time() - start_time
            vram = torch.cuda.max_memory_allocated() / 1e9
            torch.cuda.reset_peak_memory_stats()
            n_rollouts = max(len(step_rewards), 1)
            env_breakdown = "  ".join(
                f"{n}={step_env_counts[n]}" for n in cfg.env_names
            )
            print(
                f"step {step:>5d}/{cfg.total_steps} | "
                f"loss {step_loss:.4f} | reward {mean_reward_step:+.3f} "
                f"(ema {ema_overall.value:+.3f}) | "
                f"kl {step_kl/n_rollouts:.4f} | lr {lr:.2e} | "
                f"vram {vram:.1f}GB | elapsed {elapsed:.0f}s",
                flush=True,
            )
            print(
                f"           envs: {env_breakdown}",
                flush=True,
            )
            per_env_str = "  ".join(
                f"{n}={ema_per_env[n].value:+.3f}"
                if ema_per_env[n].value is not None else f"{n}=—"
                for n in cfg.env_names
            )
            print(f"           ema_reward_per_env: {per_env_str}", flush=True)

            # Capability success rates — the bullshit-detector for reward
            # hacking. If reward EMA climbs but these don't, the model
            # is gaming format rewards without learning the actual task.
            m = cum_outcomes["math"]
            m_total = m["exact"] + m["close"] + m["miss"]
            m_rate = (m["exact"] / m_total) if m_total > 0 else 0.0
            m_partial_rate = (
                (m["exact"] + m["close"]) / m_total if m_total > 0 else 0.0
            )

            i = cum_outcomes["ifeval"]
            i_attempted = i["constraints_hit"] + i["constraints_miss"]
            i_rate = i["constraints_hit"] / i_attempted if i_attempted > 0 else 0.0

            c = cum_outcomes["mbpp_code"]
            c_total = c["all_pass"] + c["all_fail"] + c["timeout"] + c["other"]
            c_rate = (c["all_pass"] / c_total) if c_total > 0 else 0.0

            print(
                f"           hit_rate: math.exact={m_rate:.1%} "
                f"(+close={m_partial_rate:.1%}) [{m['exact']}/{m_total}]  "
                f"ifeval.constraints={i_rate:.1%} "
                f"[{i['constraints_hit']}/{i_attempted}]  "
                f"mbpp.all_pass={c_rate:.1%} [{c['all_pass']}/{c_total}]",
                flush=True,
            )

            if use_wandb:
                log_dict = {
                    "grpo_multi/loss": step_loss,
                    "grpo_multi/mean_reward": mean_reward_step,
                    "grpo_multi/ema_reward": ema_overall.value or 0.0,
                    "grpo_multi/approx_kl": step_kl / n_rollouts,
                    "grpo_multi/lr": lr,
                    "grpo_multi/vram_gb": vram,
                }
                for n in cfg.env_names:
                    log_dict[f"grpo_multi/env_{n}_count"] = step_env_counts[n]
                    if ema_per_env[n].value is not None:
                        log_dict[f"grpo_multi/env_{n}_ema_reward"] = (
                            ema_per_env[n].value
                        )
                # Capability success rates (cumulative)
                log_dict.update({
                    "grpo_multi/math_exact_rate": m_rate,
                    "grpo_multi/math_partial_rate": m_partial_rate,
                    "grpo_multi/math_total_rollouts": m_total,
                    "grpo_multi/ifeval_constraint_hit_rate": i_rate,
                    "grpo_multi/ifeval_constraints_attempted": i_attempted,
                    "grpo_multi/mbpp_all_pass_rate": c_rate,
                    "grpo_multi/mbpp_total_rollouts": c_total,
                    "grpo_multi/mbpp_timeout_count": c["timeout"],
                })
                wandb.log(log_dict, step=step)

        # ── Troubleshoot generation (every troubleshoot_interval steps) ──
        # Print a single rollout-temperature sample so we can SEE what
        # the model is producing during training. Cheaper than OOD probe.
        if (
            troubleshoot_interval > 0
            and step > 0
            and step % troubleshoot_interval == 0
        ):
            _run_troubleshoot_gen(step)

        # ── OOD probe (every cfg.ood_probe_interval steps) ──
        # Generalization check on a held-out set the policy is NOT
        # training on. Diverges from training-reward EMA when the
        # model starts reward-hacking.
        if (
            ood_interval > 0 and ood_prompts
            and step > 0 and step % ood_interval == 0
        ):
            probe = _run_ood_probe(step)
            print(
                f"           ood_probe: {probe['hits']}/{probe['total']} "
                f"({probe['score']:.1%})",
                flush=True,
            )
            for d in probe["details"]:
                mark = "✓" if d["hit"] else "✗"
                print(
                    f"             {mark} {d['prompt'][:60]:<60} "
                    f"expect={d['expected']:<8} got={d['answer'][:40]!r}",
                    flush=True,
                )
            if use_wandb:
                wandb.log({
                    "grpo_multi/ood_score": probe["score"],
                    "grpo_multi/ood_hits": probe["hits"],
                    "grpo_multi/ood_total": probe["total"],
                }, step=step)

        # Checkpoints
        if step > 0 and step % cfg.ckpt_interval == 0:
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            ckpt_out = f"{ckpt_dir}/osrt_v5_{cfg.stage_prefix}_step_{step}.pt"
            torch.save({
                "step": step,
                "model_state_dict": inner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, ckpt_out)
            vol.commit()
            print(f"  -> Checkpoint saved: {ckpt_out}", flush=True)

        # 23h safety
        if time.time() - start_time > 82_800:
            inner = model._orig_mod if hasattr(model, "_orig_mod") else model
            rescue_path = (
                f"{ckpt_dir}/osrt_v5_{cfg.stage_prefix}_rescue_step_{step}.pt"
            )
            torch.save({
                "step": step,
                "model_state_dict": inner.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, rescue_path)
            vol.commit()
            print(f"\n23h boundary at step {step}. Rescue: {rescue_path}")
            if use_wandb:
                wandb.finish()
            return

    inner = model._orig_mod if hasattr(model, "_orig_mod") else model
    final_out = f"{ckpt_dir}/osrt_v5_{cfg.stage_prefix}_final.pt"
    torch.save({
        "model_state_dict": inner.state_dict(),
        "training_stage": cfg.stage_prefix,
    }, final_out)
    vol.commit()
    elapsed_h = (time.time() - start_time) / 3600
    print(f"\n{cfg.stage_prefix} complete. {cfg.total_steps} steps in "
          f"{elapsed_h:.1f}h. Final ckpt: {final_out}")
    if use_wandb:
        wandb.finish()


# =============================================================================
# ENTRYPOINT
# =============================================================================


@app.local_entrypoint()
def main(stage: str = "pretrain"):
    """Run v5 training stages.

    --stage sanity     200-step smoke test (config A)
    --stage sweep      Gumbel schedule sweep (configs B, C, D)
    --stage ablate     Optimizer × routing ablation (cells A/B/C/D, 1200 steps each)
    --stage pretrain   Full pre-training with progressive seq_len curriculum
    --stage pretrain_extend  Continued pretraining on top of SFT-ultralong ckpt
                             (~1,800 steps, seq 4096, math/science/code mix +
                              SFT-formatted rehearsal, HRA frozen)
    --stage pretrain_extend2 Broader mid-training on top of GRPO step-700 ckpt
                             (~3,000 steps, seq 2048, 30/40/15/15
                              code/math/reasoning/general mix with R1
                              cold-start traces, HRA frozen)
    --stage pretrain_extend2_sanity  50-step smoke test of the extend2
                             pipeline; validates all 10 streams + format
                             functions before committing $28 on the full run
    --stage sft            Balanced SFT on the final pretrained checkpoint
    --stage sft_long       Long-context SFT (seq 4096) resuming from
                             sft_final.pt with Nemotron mix
    --stage sft_ultralong  Ultra-long-context SFT (seq 8192) resuming
                             from sft_long_final.pt
    --stage sft_refresh    Short format-anchor SFT on top of extend_final.pt
                             (500 steps, peak LR 5e-6, no tool_calling)
    --stage sft_math       Math-only SFT polish on top of sft_refresh_final.pt
                             (1,000 steps, peak LR 3e-6, GSM8K + Orca +
                              MathInstruct + NuminaMath)
    --stage evaluate       lm-eval-harness pass (gsm8k + IFEval + MMLU-stem). Args:
                             --ckpt-name <filename in /vol/checkpoints/v5/>
                             --tag <pre-grpo|post-grpo|...>
                             --tasks <comma-separated; default 8-task suite:
                                gsm8k, ifeval, mmlu_stem, hellaswag,
                                arc_easy, arc_challenge, piqa, winogrande>
                             --limit <int or None for full benchmark>
    --stage grpo           GRPO RL on the SFT checkpoint (verifiable math rewards)
    """
    # Central stage registry — single source of truth so --stage dispatch and
    # the help/unknown-stage message can't drift. The v6 stages (midtrain*,
    # sft_v1*, sft_v2*, midtrain2*) were previously missing here entirely, so
    # `modal run app.py --stage midtrain2` fell through to "Unknown stage".
    # mode SPAWN = long fire-and-forget training (the local entrypoint can exit
    # without cancelling the run — .remote() gets cancelled on disconnect);
    # mode REMOTE = short/blocking (eval, smoke). Each training stage also has
    # a dedicated run_* @app.local_entrypoint.
    REMOTE, SPAWN = "remote", "spawn"
    registry = {
        # ── v5 lineage ──
        "sanity": (sanity, REMOTE),
        "sweep": (sweep, REMOTE),
        "ablate": (ablate, REMOTE),
        "pretrain": (pretrain, REMOTE),
        "pretrain_extend": (pretrain_extend, REMOTE),
        "pretrain_extend2": (pretrain_extend2, SPAWN),
        "pretrain_extend2_sanity": (pretrain_extend2_sanity, SPAWN),
        "loop_fix": (loop_fix, SPAWN),
        "loop_fix_sanity": (loop_fix_sanity, SPAWN),
        "loop_fix_v2": (loop_fix_v2, SPAWN),
        "loop_fix_v2_sanity": (loop_fix_v2_sanity, SPAWN),
        "pretrain_extend3": (pretrain_extend3, SPAWN),
        "pretrain_extend3_sanity": (pretrain_extend3_sanity, SPAWN),
        "mopd": (mopd, SPAWN),
        "mopd_sanity": (mopd_sanity, SPAWN),
        "grpo_multi": (grpo_multi, SPAWN),
        "grpo_multi_sanity": (grpo_multi_sanity, SPAWN),
        "system_sft": (system_sft, SPAWN),
        "system_sft_sanity": (system_sft_sanity, SPAWN),
        "sft": (sft, REMOTE),
        "sft_long": (sft_long, REMOTE),
        "sft_ultralong": (sft_ultralong, REMOTE),
        "sft_refresh": (sft_refresh, REMOTE),
        "sft_math": (sft_math, REMOTE),
        "evaluate": (evaluate, REMOTE),
        "grpo": (grpo, REMOTE),
        # ── v6 lineage (were missing from --stage) ──
        "midtrain": (midtrain, SPAWN),
        "midtrain_sanity": (midtrain_sanity, SPAWN),
        "midtrain2": (midtrain2, SPAWN),
        "midtrain2_sanity": (midtrain2_sanity, SPAWN),
        "sft_v1": (sft_v1, SPAWN),
        "sft_v1_sanity": (sft_v1_sanity, SPAWN),
        "sft_v2": (sft_v2, SPAWN),
        "sft_v2_sanity": (sft_v2_sanity, SPAWN),
    }
    entry = registry.get(stage)
    if entry is None:
        print(f"Unknown stage: {stage!r}. Available stages:")
        for name in sorted(registry):
            print(f"  - {name}")
        return
    fn, mode = entry
    if mode == SPAWN:
        call = fn.spawn()
        print(f"Spawned {stage} as call: {call.object_id}")
    else:
        fn.remote()
