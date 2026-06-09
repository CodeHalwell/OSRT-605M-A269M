# NanoOSRT Learnings Scratchpad

Working notes for training observations, phase-gate decisions, and ideas to revisit.
This file is intentionally informal: append dated notes, hypotheses, and follow-up
checks here before promoting anything into the main docs.

## Current Run Notes

Date: 2026-06-09

- Step snapshot reviewed: foundation step 2250.
- Router balance looked healthy: no dead experts, zero drop rate, near-uniform
  expert load, and strong pre-bias expert coverage.
- Main watch item before midtraining: router specialization was improving but
  not yet at the original phase-1 gate. This run is planned for 3500 steps, so
  it will end before the configured step-5000 health gate and before the normal
  4000-step Gumbel anneal fully reaches zero.
- Secondary watch item: eval loss was much higher than train task loss. Need
  trend data across the next evals before calling it overfit or data-mix drift.

## Pre-Midtraining Gate

This run is planned for 3500 foundation steps, so use an adjusted gate instead
of the original step-5000 cold-start gate.

Before moving from foundation into midtraining, check:

- Final checkpoint is from step 3500, or a short no-Gumbel cooldown checkpoint
  if one is run.
- `moe/clean_per_token_entropy_mean` is at or below about 1.53.
- `moe/clean_raw_max_prob_mean` is at or above 0.30.
- `moe/clean_top_margin_mean` is at or above 0.10.
- `moe/clean_marginal_entropy_mean` stays above 1.80.
- `moe/prebias_expert_min_mean` stays comfortably above 0.01.
- Eval loss is flat or falling over the last 2-3 evals.
- Move forward from the best eval checkpoint, not automatically the latest
  checkpoint.

## Candidate Interventions

Use these only if the gate or eval trend says they are needed.

- Best option if budget allows: add 300-500 extra foundation-style cooldown
  steps with Gumbel tau at 0 and a lower LR, then checkpoint from that run.
- If budget does not allow extra steps, treat the step-3500 checkpoint as
  acceptable only if router metrics are already close to the original thresholds
  and eval loss is not worsening.
- If router metrics pass but eval remains noisy, run a short foundation
  consolidation tail: same data mix, Gumbel tau at 0, lower LR, 500-1500 steps.
- Avoid increasing router auxiliary loss unless router health actually degrades;
  balance is already strong, and excess aux pressure can distort the language
  modeling objective.
- If eval loss keeps separating from train loss, inspect eval data mix,
  tokenizer behavior, and whether eval is out-of-distribution before changing
  optimizer or architecture knobs.

## Ideas To Revisit

- Add an automated pre-midtraining gate report that prints the key router,
  pre-bias, loop-update, and eval-trend metrics in one place.
- Track eval loss by dataset/source instead of only aggregate eval loss.
- Keep a small fixed validation shard for foundation-to-midtraining checkpoint
  selection.
- Add a checkpoint comparison note whenever choosing a non-latest checkpoint for
  the next phase.
- Revisit MoE telemetry overhead after training stability questions are settled;
  previous reviews flagged forward-path `.item()` / `.tolist()` telemetry as a
  throughput concern.

## Scratch Notes

- 2026-06-09 step 2750 update: router health still looks strong. Clean raw
  max probability (~0.411), clean top margin (~0.224), clean marginal entropy
  (~2.073), pre-bias expert min (~0.107), dead experts (0), and drop rate (0)
  are all in good shape. Clean per-token entropy (~1.615) is still above the
  original 1.53 target, but Gumbel tau is still 0.15625 and the run is not at
  the final 3500-step checkpoint yet. Evals only run at steps 1000, 2000, and
  3000, so the step-2750 eval value is stale from step 2000. Do not judge the
  eval/train gap again until the step-3000 eval lands.
- 2026-06-09 step 3000/3200 update: step-3000 eval improved to loss 3.7605 /
  ppl 43.0, so the eval/train gap is moving in the right direction. Router
  health remains acceptable through step 3200: dead experts 0, drop 0, clean
  raw max ~0.405, clean margin ~0.220, clean marginal entropy ~2.076, clean
  expert min ~0.109, and pre-bias expert min ~0.108. The main remaining watch
  item is depth usage: loop update norms are trending down, with min falling
  from ~0.29 near step 2200 to ~0.19 at step 3200 and last falling from ~0.59
  to ~0.46. Not collapsed, but the final step-3500 snapshot should confirm the
  late loops do not continue drifting toward no-op behavior.
- 2026-06-09 step 3200 review (additions complementing the entry above):
  - Gate caveat, clean per-token entropy. `clean_per_token_entropy_mean` is
    FLAT at ~1.615 across steps 2750->3200; it is not trending toward the 1.53
    gate. The rise in the *noised* pte (1.64->1.69) is purely the Gumbel anneal
    (tau 0.156->0.100) revealing the clean gate, not the routing structurally
    sharpening. So 1.53 is unlikely to be met by step 3500. Recommend NOT
    failing the gate on that single number: the other four clean metrics
    (raw_max ~0.41 >= 0.30, top_margin ~0.22 >= 0.10, marginal_entropy ~2.076
    >= 1.80, prebias_expert_min ~0.108 >> 0.01) all pass comfortably, and the
    1.53 target was likely calibrated for the original step-4000 anneal /
    step-5000 gate, not this compressed 3500-step run. Judge on trend + the
    passing four, or run the no-Gumbel cooldown tail (already in Candidate
    Interventions) to push clean entropy down before midtraining.
  - Loop-update structure (refines the "no-op drift" watch item). The loop
    |dx|/|x| profile is three regimes, not a uniform decline: L0 ~8.2
    (rock-stable; the first block wholesale-rewrites the mHC-initialised
    streams), L1/L2 in clean monotonic decline (L2 1.60->0.89, L1 1.11->0.77
    over 2150->3200, early loops converging toward a near-fixed-point), and
    L3-L17 a stable 0.19-0.75 band. Crucially the *last* loop L17 is NOT the
    lowest, it holds ~0.46-0.62. At step 3200 the min is L16 (0.193) and the
    low loops (L7 0.25, L10 0.27, L16 0.19) are scattered, not
    monotonically-deepening. That argues against "late loops drifting to no-op":
    it reads as specific loops specialising into low-magnitude refinement roles
    while the terminal loop keeps contributing. The collapse signature
    (monotonic L13<L14<L15<L16<L17 -> 0) is absent. Still worth the 3500 confirm.
  - OPERATIONAL FLAG, checkpoint path. Checkpoints are written to
    `/vol/checkpoints/v5/osrt_v5_step_*.pt`, but this is unmistakably the v6
    architecture (3x6 recursion, mHC, KDV latent cache, 8-expert grouped-GEMM
    MoE). If midtraining globs `v6/` it will silently miss these. Confirm the
    path convention before the foundation->midtraining handoff.
  - Eval coverage gap. Step 3000 was the LAST scheduled eval (evals fire only
    at 1000/2000/3000). The final 500 steps anneal LR 9.4e-5 -> ~2e-5; that
    cosine tail usually buys more ppl that will go unmeasured, so the official
    foundation number becomes 3.76/43.0 while the *delivered* 3500 checkpoint is
    likely better. Recommend a single end-of-run eval at 3500 for an honest
    foundation number and clean checkpoint selection.
- 2026-06-09 FOUNDATION COMPLETE (step 3500, 14.0h). Final ckpt:
  /vol/checkpoints/v5/osrt_v5_final.pt (the "v5" dir is a LEGACY LABEL;
  this IS the v6 model — resolves the checkpoint-path flag above: the
  handoff uses /vol/checkpoints/v5, not v6/). Last scheduled eval (step
  3000): loss 3.7605 / ppl 43.0; W&B eval/loss history is a clean
  down-staircase across the three evals (1000->2000->3000). The
  L17-no-op watch item is RESOLVED: at step 3450 L17=0.411 and the min
  is L16=0.175 (scattered, not monotonic late decline) — terminal loop
  kept contributing, collapse signature absent. bias_abs_max settled
  0.028, dead_experts 0 throughout, clean pte plateaued ~1.63 (never hit
  the 1.53 gate, as predicted — judged on the four passing clean metrics
  + trend). No step-3500 eval was added, so the annealed checkpoint's
  true held-out loss is unmeasured (very likely a touch better than
  3.76/43.0).
- 2026-06-09 MIDTRAINING SPECCED (design + plan committed; impl NOT
  started). docs/superpowers/specs/2026-06-09-v6-midtraining-design.md +
  docs/superpowers/plans/2026-06-09-v6-midtraining.md. Decisions: ~$150 /
  ~9k steps, seq 2048->4096, knowledge-phase math mix (~65% math/STEM/
  reasoning, FineWeb anchor retained), HRA native+trainable, re-warm
  peak LR 2e-4 (cosine->2e-5, fresh cosine lr_anchor_step=0), resume
  straight from osrt_v5_final.pt. Approach: GENERALIZE run_pretrain_extend
  (new hra_native flag gates inject_hra; lower grad-ckpt trigger to
  seq>=4096; port periodic run_eval) NOT a new loop — the v5 extend loop
  would graft a mismatched HRALinear layout onto the v6 native-HRA ckpt
  and fail load_model_state_or_raise. New MidtrainConfig +
  MidtrainSanityConfig + midtrain/midtrain_sanity entrypoints. Pre-launch
  gate: run_midtrain_sanity (30 steps @ real seq4096/batch6) MUST pass
  before run_midtrain; if OOM drop to batch4/accum16. Every loop change
  is flag-gated to default to v5 behaviour (zero blast radius on legacy
  stages).
- 2026-06-09 MIDTRAIN PROBE #1 — code gates PASSED on GPU, then a data
  bug blocked it. The build-small sanity probe (seq 4096, batch 6,
  checkpointing OFF) confirmed all 5 code gates live: HRA-native skip,
  "Clean load: all keys matched" (v6 601M ckpt), "Gradient checkpointing:
  disabled (_osrt_grad_ckpt=False)", Muon+AdamW built, 7 streams
  connected. BUT it never reached a forward pass — nemotron-math-textbooks
  threw "RecursionError: maximum recursion depth exceeded in comparison"
  in the DataWorker, looping on reconnect. Stopped the run (no runaway $).
- 2026-06-09 DATA BUG ROOT-CAUSED + FIXED (commit b0edb13). Cause:
  TokenStream._cycling_iter re-shuffled the already-shuffled view each
  cycle (ds = ds.shuffle(...)), nesting a ShuffledExamplesIterable wrapper
  per cycle. Iterating an N-deep nest is super-exponential → hit Python's
  1000-frame recursion limit mid-iteration. Proven deterministically
  (scripts/repro_cycling_recursion.py, no network): nested time-to-first-
  item 0.001s@depth1 → 51.7s@depth20; base-shuffle fix flat (600 cycles /
  18k pulls in 0.45s). Fix: shuffle the UNSHUFFLED base fresh each cycle →
  one wrapper always. Shared data.py path so ALL stages benefit. This is
  the classic "data fragility v5 dodged by timing": foundation's mix never
  streamed the small Specialized-v1 configs the knowledge mix uses.
  STILL OPEN: the checkpointing-off VRAM question (probe #1 never got
  there). Probe #2 pending.
- 2026-06-09 VRAM + DATALOADER RESOLVED across 4 probes (judge by W&B
  system metrics, NOT modal logs — the CLI log tail was flaky all session:
  truncated/stale tails caused a near-miss premature kill of probe #3).
  Findings:
  * Checkpointing OFF @ seq4096/batch6: OOM (probe #2, W&B gpu mem
    99.4%/84.97GB). Does NOT fit. Reverted to ON (commit 25946de).
  * Checkpointing ON @ seq4096/batch6: FITS at 51-54GB / 80GB (~26GB
    headroom) — probe #4. This is the production config.
  * First-batch stall (probes #2/#3 stuck, 0 W&B gpu rows): caused by
    dataloader_num_workers=2 → each worker opens ALL 7 streams + fills a
    5k shuffle buffer each = 14 cold stream-opens. Foundation worked at
    1x4. Fix: num_workers 2→1 (commit e998ea8). First batch then assembled
    in 443s (one-time cold cost, amortized over 9k steps); training healthy
    (task 2.23→2.05, bal 1.02, 0 drops, tok/s ramping 569→3472).
  * SANITY GATE PASSED (probe #4): ckpt ON + 1 worker + seq4096/batch6 +
    the b0edb13 cycling fix all confirmed live on GPU. READY for the $150
    run_midtrain.
  * CC-Code-v1 swap DEFERRED: its field-inspection couldn't pull one
    example in 100s (slow first-byte) — would worsen the first-batch stall.
    Add separately after the base run is confirmed streaming. Task #91 open.
- 2026-06-09 REAL RUN launch #1 hit a TRANSIENT HF connection storm at
  cold-start (Connection reset by peer / Bad file descriptor on
  InfiniByte-Reasoning parquet shards), stalled at step 0, killed at
  retry [2/20]. Diagnosed ENVIRONMENTAL not structural: probe #4 ran the
  identical config clean <1hr earlier, and InfiniByte streamed 50/50
  locally in 28.9s. Relaunched (app ap-FywAk1zfOl4ge58H8lTRRW). Decision
  (advisor-backed): relaunch once with a real budget, judge by W&B
  _step>=1, do NOT pre-harden shared data.py retry/timeout (untestable
  transient, rabbit hole). ckpt-every-500 caps any future cold-start
  storm. IF it storms again at cold-start (then it's reproducible): the
  fix is to SNAPSHOT the small Specialized-v1 configs to the
  osrt-checkpoints volume + read locally (kills the cold-start connection
  burst for the fragile sources), NOT retry-tuning.
- KNOWN EDGE (record, not blocking): the debt-based sampler in
  TokenStream._pick_stream can starve on a PERSISTENTLY-empty stream — a
  zero-yield stream holds max deficit and keeps getting picked. Only bites
  on a sustained outage (not this transient). Worth fixing if it ever
  recurs: skip/deprioritise a stream that returns None too many times.
