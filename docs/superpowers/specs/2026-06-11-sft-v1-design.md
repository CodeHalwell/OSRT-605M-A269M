# SFT v1 (system-prompt instruction tuning) — Design

**Date:** 2026-06-11
**Status:** Approved (design); implementation plan pending
**Stage:** Roadmap Stage 1 (see review/learnings-scratchpad.md "PROJECT NORTH STAR")
**Base:** `osrt_v5_midtrain_final.pt` (v6, step 4500, ppl ~30, fully annealed)

---

## 1. Goal & scope

Produce an instruction-tuned v6 model that **follows a `<|system|>` prompt** and
emits the native `<|think|>…<|/think|><|answer|>…<|/answer|>` format. The
mid-trained base is a strong language model but does NOT follow prompts at all
(smoke test: "capital of France" → an unrelated article). SFT-v1 makes it
controllable.

**This is NOT the long-reasoning stage.** Per the roadmap, long coherent
reasoning is built later (Stage 2 seq-8192 extension → Stage 3 reasoning-CoT SFT
→ Stage 4 GRPO-verifiable). SFT-v1's job is format + instruction-following +
**establishing the reasoning-on/off control and its measurement harness** so the
later stages have a baseline to beat.

**Explicitly out of scope for v1:** long CoT data (OpenThoughts/OpenR1), seq
8192, tool-calling, multilingual, RL.

## 2. Approach — extend existing machinery, don't rebuild

The repo already has: `SFTStream` (streaming SFT builder with tag templating +
loss masking, `sft_data.py:361`), 12 `format_fn`s, a curated system-prompt pool
(`system_prompts.py`), and `SFTConfig` (HRA-trainable SFT config,
`train_config.py`). `<|system|>` is a real single token (id 13) in the v6
tokenizer. Three bounded changes:

1. **Add a `<|system|>` turn to `SFTStream`** (the core change).
2. **Reasoning-conditioned system prompts** (the key design decision — §4).
3. **New `SFTv1Config` + one new `format_tulu` + the on/off eval harness.**

## 3. Sequence template & loss masking

Every example, regardless of source dataset, is built as:

```
<|system|>{persona}<|user|>{question}<|assistant|><|think|>{reasoning}<|/think|><|answer|>{answer}<|/answer|><|end_of_text|>
```

Loss masking (extends the current `SFTStream` scheme):
- **masked (IGNORE_INDEX=-100):** `<|system|>{persona}<|user|>{question}<|assistant|>` — the whole prefix.
- **trained (real labels):** `<|think|>…<|/answer|><|end_of_text|>` — the response.

Because the system turn joins the **masked prefix**, the model learns to *attend
to and follow* the system prompt but is never trained to echo it (same rationale
already documented in `SystemSFTConfig`). The `{persona}` and the dataset content
are independent: the persona teaches format/behaviour-following, the dataset
supplies the task.

Implementation: in `SFTStream.__iter__` (currently `sft_data.py:569`), change
```python
prompt_text = f"{self.user_tag}{question}{self.assistant_tag}"
```
to
```python
prompt_text = f"{self.system_tag}{persona}{self.user_tag}{question}{self.assistant_tag}"
```
where `persona` is sampled per-example (§4). Add `system_tag="<|system|>"` to
`SFTStream.__init__` and `SFTv1Config`. Loss masking already keys off
`len(prompt_ids)`, so masking stays correct with no further change.

## 4. Reasoning-conditioned system prompts (the key decision)

**Problem this avoids:** if we sample a "think step by step" persona onto data
that has no real reasoning (general/code), we train the model to *disobey* the
system prompt (persona says reason, response doesn't) AND teach a prior that
`<|think|>` is normally empty — poisoning the Stage-3 reasoning well. Both are
the opposite of the project goal.

**Solution:** split the system-prompt pool by reasoning mode and match the
persona to the data.

- `system_prompts.py` gains two pools:
  - `REASONING_ON` — the existing 12 personas ("think step by step inside
    <|think|>…"). Used for data with real reasoning.
  - `REASONING_OFF` — ~6 NEW personas ("answer directly and concisely; put the
    answer in <|answer|>, keep <|think|> empty or brief"). Used for
    instruction/chat/code data without reasoning traces.
- Each dataset config carries a `reasoning_mode: "on" | "off"` flag.
- `SFTStream` samples from the matching pool for each example.
- For `off` data, an empty/brief `<|think|>` is now CONSISTENT with the persona
  ("you said answer directly") instead of contradictory.

**Why this is worth the extra code:** it makes "follow the system prompt"
literally true in both directions, and it BUILDS THE reasoning-on/off TOGGLE in
SFT-v1 — which is the project's north-star success metric (reasoning-on accuracy
> reasoning-off accuracy). Establishing the toggle now means every later stage
can A/B against it. (NVIDIA's IF-Chat data ships exactly this on/off field — an
established pattern.)

`sample_system_prompt(rng, mode)` gains a `mode` arg; keep the old signature
working (default "on") so MOPD/GRPO call sites don't break.

## 5. Data mix

All five sources flow through §3's template; the persona pool is chosen by the
`reasoning_mode` column. All format_fns exist except `format_tulu` (new).

| source | hf_id | format_fn | reasoning_mode | weight |
|---|---|---|---|---|
| Tülu-3 SFT | allenai/tulu-3-sft-mixture | **format_tulu (NEW)** | off | 0.30 |
| OpenHermes-2.5 | teknium/OpenHermes-2.5 | format_openhermes ✓ | off | 0.25 |
| GSM8K | openai/gsm8k (main) | format_gsm8k ✓ | on | 0.20 |
| Numina-Math | (existing numina cfg) | format_numina_math ✓ | on | 0.15 |
| Evol-Code | nickrosh/Evol-Instruct-Code-80k-v1 | format_evol_code ✓ | off | 0.10 |

→ ~55% general instruction-following (off), ~35% math (on), 10% code (off).
Plays to the midtrain-strengthened math/STEM while fixing the core
follow-the-prompt gap. Probed lengths (median resp tok): tulu ~269, openhermes
~231, gsm8k/numina moderate — all in the 200-800 target band.

**`format_tulu` (new):** Tülu-3 uses a `messages` chat-list schema none of the
existing format_fns handle. New fn: take the first user message as `question`,
the assistant message as `answer`, empty `reasoning`. Skip multi-turn rows (>1
user turn) for v1 simplicity. ~15 lines, mirrors the existing `format_openhermes`
conversations handling.

**Note on the reasoning slot for off-mode data:** the existing format_fns
(`format_openhermes`, `format_evol_code`) already heuristically split a leading
reasoning paragraph from the answer when present, and fall back to empty
reasoning otherwise. So off-mode `<|think|>` is "empty or brief", NOT strictly
empty — which is exactly what the `REASONING_OFF` personas should say ("keep
<|think|> empty or brief"). The persona wording must tolerate a short think
block, not forbid it, to stay consistent with what these fns emit.

## 6. Length controls

- Existing `SFTStream` already skips examples `> seq_len` (2048) and packs.
- ADD `min_response_tokens` (≈150) to `SFTv1Config` + a skip in `SFTStream`:
  drop examples whose response is shorter than the floor ("not too short").
- Soft upper: rely on seq_len + packing; the 200-800 band is achieved by the
  source selection (probed medians), not a hard cap.

## 7. Config — `SFTv1Config(SFTConfig)`

Overrides on the existing `SFTConfig`:
```python
class SFTv1Config(SFTConfig):
    pretrained_checkpoint = "<midtrain_final path>"   # v6 base
    stage_prefix = "sft_v1"
    seq_len = 2048
    total_steps = 2000           # batch8 × accum8 × 2048 = 131K tok/step ≈ 260M tok
    # peak_lr 1.5e-5 → min 1.5e-6, warmup 250, AdamW — inherited SFTConfig defaults
    hra_enabled = True; hra_rank = 256; hra_lr = 7.5e-5; hra_freeze_pretrained = False
    system_tag = "<|system|>"
    min_response_tokens = 150
    eval_interval = 500
    ckpt_interval = 500
    datasets = [ ... §5 table, each with reasoning_mode ... ]
    wandb_run_name = "osrt-v6-sft-v1"
```
HRA stays trainable (it was pretrained + midtrained, not a frozen SFT delta —
keep learning it; the SFTConfig default `hra_freeze_pretrained=False` is correct).

## 8. Eval — the north-star metric, built early

A `reasoning_on/off` A/B eval every `eval_interval` (500) steps on a small
held-out **GSM8K** slice (verifiable answers). NEW infra (the biggest build item):

1. For each held-out problem, generate twice: once with a `REASONING_ON`
   persona, once `REASONING_OFF` (same problem, same user turn).
2. Extract the `<|answer|>` content, compare to ground truth.
3. Report: **accuracy_on, accuracy_off**, **mean_resp_len_on/off** (confirms the
   toggle physically changes behaviour — on should be longer), and held-out loss.
4. **format-compliance rate:** fraction of generations with well-formed
   `<|think|>…<|/answer|>` tags (catches the most likely SFT failure).

**Expectation at SFT-v1:** do NOT expect accuracy_on > accuracy_off yet — the
model can barely reason. This run establishes the BASELINE the Stage-3/4 stages
must beat, and proves the toggle + harness work. The win condition
(on > off) is a later-stage target; measuring it starts here.

## 9. Pre-launch gate

`sft_v1_sanity` (30 steps) MUST pass before the paid run:
- the `<|system|>` turn builds; both persona pools sample correctly
- loss masking verified (prefix incl. system masked; response trained) — assert
  on a constructed example that label[:len(prefix)] == -100
- VRAM fits at seq 2048 / batch 8 on the target GPU
- one forward/backward + a generation runs clean
If OOM, drop batch_size. (Same discipline as the midtrain sanity gate.)

## 10. Run target & platform

- ~2000 steps ≈ 260M tokens. Lightning H100 ($3.50/hr) ≈ ~17h ≈ ~$60, or chunk.
- Resume via `stage_prefix="sft_v1"` scan (same machinery as midtrain).
- Reuse `scripts/lightning_midtrain.py` pattern for a Modal-free entry, OR the
  existing Modal `sft` entrypoint — decide at plan time.

## 11. Code-change summary (blast radius)

| change | file | risk |
|---|---|---|
| `<|system|>` turn + `system_tag` + persona sampling | sft_data.py `SFTStream` | low (additive; masking unchanged) |
| `REASONING_OFF` pool + `mode` arg on sampler | system_prompts.py | low (back-compat default) |
| `format_tulu` (new fn) | sft_data.py | low (new, isolated) |
| `min_response_tokens` skip | sft_data.py | low |
| `SFTv1Config` | train_config.py | low (new subclass) |
| reasoning-on/off GSM8K eval harness | new (sft_eval.py?) | MEDIUM — the real build item |
| sanity + entry wiring | app.py / lightning_*.py | low |

Existing MOPD/GRPO call sites of `sample_system_prompt` and `SFTStream` keep
working (back-compat defaults). The on/off eval harness is the one substantial
new piece.

## 12. Out of scope (deferred)

- Long-CoT data, seq 8192, GRPO, DPO — later roadmap stages.
- Cascade-generated data — can be added to the mix later; not needed for v1.
- Multi-turn Tülu rows (v1 uses single-turn only).
