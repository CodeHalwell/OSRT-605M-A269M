# Tokenizer & Embedding

*Part of the `docs/` architecture series for OSRT-605M. This document explains the tokenizer + embedding block — the model's input/output interface; see `ARCHITECTURE.md` §3 (tokenizer) and §4 (embedding) for the original design intent, and `src/osrt/model.py` for the implementation that ships.*

---

## 1. Purpose / summary

A language model never sees text. It sees integers, and it emits a probability distribution over integers. Two pieces of machinery bracket the entire network and turn text into those integers and back:

1. The **tokenizer** — a byte-level BPE that maps a UTF-8 string to a sequence of token IDs in `[0, 65535]`, and back.
2. The **embedding matrix** — a `65,536 × 1,536` table that maps each ID to a 1,536-dim vector (the *input* side) and, because it is **weight-tied**, the same matrix projects the final hidden state back to a logit per vocabulary entry (the *output* side).

Everything in between — attention, MoE, recursion — operates purely on those 1,536-dim vectors. The embedding is the only place IDs enter, and the (tied) embedding is the only place logits leave. It is also, at ~16.7% of the model, the single largest parameter block that is *not* doing computation, which is why its size is a deliberate budget decision and not an afterthought (§2, §7).

> **Build status (read this first).** This document describes the **target/preset** architecture: vocab 65,536, embedding `65,536 × 1,536`. That is what the locked `OSRT_605M_A288M` preset builds (`src/osrt/presets.py:27`) and what `scripts/compute_budget.py` reports. **Several on-disk artifacts still lag that target** — the shipped `tokenizer/tokenizer.json` is a 32,768-wide BPE, only 14 of the 21 contract special tokens exist, and the bare `OSRTConfig` defaults are 32,768. Each lag is flagged inline at the relevant section. Where ARCHITECTURE.md §4.2 disagrees with the code, the code wins and the discrepancy is noted (§5).

---

## 2. Byte-level BPE and why vocab = 65,536

### 2.1 What byte-level BPE is

The tokenizer is **byte-level Byte-Pair Encoding** (`ARCHITECTURE.md:213`, and the on-disk `tokenizer/tokenizer.json` reports `"model": {"type": "BPE"}` with a `ByteLevel` pre-tokenizer). Two properties matter:

- **Byte-level** means the alphabet is the 256 possible bytes, not Unicode characters. Any string — English, Arabic, Japanese, emoji, raw bytes, a corrupted file — is representable, because every string *is* a sequence of bytes. There is no true "out-of-vocabulary" failure; in the worst case a rare string falls back to its constituent bytes. (An `<|unknown|>` token exists at ID 3 as a contract slot, but byte-level coverage means it is essentially never emitted.)
- **BPE** then learns a merge table: starting from single bytes, it repeatedly merges the most frequent adjacent pair into a new token, growing the vocabulary up to the target size. Common sequences (`" the"`, `"def "`, `"```"`) become single tokens; rare ones stay fragmented.

Pre-tokenization uses the GPT-2-style regex (`ARCHITECTURE.md:218`) that splits on contractions, digit runs, and punctuation before BPE runs, so merges never cross those boundaries — `"don't"` and `"123456"` tokenize sensibly.

### 2.2 Why 65,536 specifically

65,536 is `2^16` — the largest vocabulary addressable in a 16-bit integer, a clean ceiling. But the *reason* it is exactly this big (and not bigger) is a parameter-budget argument that this project takes seriously:

> **The embedding tax.** Vocabulary is "dead weight" relative to the network body: every embedding row is a lookup table entry, not a transform that composes with depth or recursion. A small model that spends too much of its budget on vocabulary starves the blocks that actually do the reasoning. **Gemma-3-270M** is the cautionary tale — a 256K vocabulary on a 270M model means embeddings dominate (~63% of params), so the "270M" model has only ~100M params doing computation. **LFM2-700M** went the other way and proved that small models should pour params into blocks (~85%) rather than vocabulary.

OSRT-605M follows the LFM2 lesson: at 65,536 the embedding is **16.7% of physical params** (`scripts/compute_budget.py`, §7), leaving ~83% for attention, experts, recursion, and adapters. A larger vocabulary would buy slightly shorter sequences (fewer tokens per document) but at a steep param cost the body can't afford at this scale. 65,536 is the compromise: large enough for English + 6 multilingual scripts + code (`ARCHITECTURE.md:216-217`), small enough to keep the embedding tax modest.

> **Artifact lag (vocab width).** The shipped `tokenizer/tokenizer.json` is only **32,768** wide — 32,768 vocab entries, 32,498 learned merges — i.e. *half* the target. This is not just the missing special tokens (§3); the BPE merge table itself is half-size. `scripts/train_tokenizer.py` confirms the lag: its module docstring says *"Train a custom 32K BPE tokenizer for OSRT"* and it samples ~2 GB from FineWeb-Edu (55%) + CodeParrot-clean (30%) + Wikipedia (15%) — the pretrain mix — to learn merges, but at the 32K target. The 65,536 model embedding therefore has ~32,768 rows that no current tokenizer ID will ever index. The model is the source of truth for "605M" (which only exists at 65,536); the tokenizer artifact needs a rebuild at the full width before those rows mean anything.

---

## 3. Special tokens

The first stretch of the ID space is reserved for control tokens — structural markers the model is trained to emit and condition on. These are *added tokens*: registered atomically so a string like `<|assistant|>` always encodes to exactly one ID, never to its byte-BPE fragments.

### 3.1 What's actually on disk

Inspecting `tokenizer/tokenizer.json` `added_tokens` directly (not the contract, the file):

```
[(0, '<|padding|>'), (1, '<|begin_of_text|>'), (2, '<|end_of_text|>'),
 (3, '<|unknown|>'), (4, '<|fim_prefix|>'), (5, '<|fim_middle|>'),
 (6, '<|fim_suffix|>'), (7, '<|think|>'), (8, '<|/think|>'),
 (9, '<|answer|>'), (10, '<|/answer|>'), (11, '<|user|>'),
 (12, '<|assistant|>'), (13, '<|system|>')]
```

That is **14 added tokens, IDs 0–13, max id 13** — exactly matching `tokenizer/special_tokens_map.json` (`bos=<|begin_of_text|>`, `eos=<|end_of_text|>`, `pad=<|padding|>`, `unk=<|unknown|>`, plus the 10 `additional_special_tokens`) and `tokenizer/tokenizer_config.json` (`add_bos_token: true`, `add_eos_token: false`).

### 3.2 The full v6 contract — ✓ on disk, ✗ missing

`ARCHITECTURE.md:233-262` defines a 21-token contract (IDs 0–20). Only 14 are built. The ✗ rows are a real gotcha — see §4.

| token | id | role | on disk? |
|---|---|---|---|
| `<\|padding\|>` | 0 | PAD — masked filler for batching | ✓ |
| `<\|begin_of_text\|>` | 1 | BOS — prepended to every sequence (`add_bos_token: true`) | ✓ |
| `<\|end_of_text\|>` | 2 | EOS — end of document; stop signal at inference | ✓ |
| `<\|unknown\|>` | 3 | unk — contract slot; ~never emitted (byte-level fallback) | ✓ |
| `<\|fim_prefix\|>` | 4 | Fill-in-the-Middle: text before the gap | ✓ |
| `<\|fim_middle\|>` | 5 | FIM: the gap to be filled | ✓ |
| `<\|fim_suffix\|>` | 6 | FIM: text after the gap | ✓ |
| `<\|think\|>` | 7 | reasoning block — open | ✓ |
| `<\|/think\|>` | 8 | reasoning block — close | ✓ |
| `<\|answer\|>` | 9 | final-answer block — open | ✓ |
| `<\|/answer\|>` | 10 | final-answer block — close | ✓ |
| `<\|user\|>` | 11 | user turn — open | ✓ |
| `<\|assistant\|>` | 12 | assistant turn — open | ✓ |
| `<\|system\|>` | 13 | system prompt — open | ✓ |
| `<\|end_turn\|>` | 14 | turn separator (ChatML-style) | ✗ **missing** |
| `<\|tool_call\|>` | 15 | tool invocation — open | ✗ **missing** |
| `<\|/tool_call\|>` | 16 | tool invocation — close | ✗ **missing** |
| `<\|tool_result\|>` | 17 | tool result — open | ✗ **missing** |
| `<\|/tool_result\|>` | 18 | tool result — close | ✗ **missing** |
| `<\|image\|>` | 19 | reserved for a vision retrofit | ✗ **missing** |
| `<\|audio\|>` | 20 | reserved for future audio | ✗ **missing** |

IDs **21–31 are reserved** for future expansion; **real (learned) vocabulary begins at id 32** (`ARCHITECTURE.md:262`). Reserving a contiguous low-ID band like this means new control tokens can be added later without renumbering existing IDs or invalidating a trained embedding — the rows are already allocated.

### 3.3 The role tokens, and the open-only chat template

The control tokens fall into families:

- **Structural** (0–3): padding, sequence boundaries, the unk slot.
- **FIM** (4–6): mark a prefix/middle/suffix split so the model can be trained to infill — predict the *middle* given the *prefix* and *suffix*. This is how code-completion-style "fill the gap" training is framed.
- **Reasoning / answer** (7–10): separate the model's scratch reasoning (`<|think|>…<|/think|>`) from its committed answer (`<|answer|>…<|/answer|>`). Training the model to wrap its chain-of-thought in these tags lets a serving layer hide the reasoning and show only the answer.
- **Role** (11–13): mark whose turn it is — system, user, assistant.

The chat template uses the project's **open-only-tag convention**: role tags *open* a turn but have no closing partner (`<|/user|>` does not exist). The next role tag, or `<|end_turn|>`, ends the previous turn implicitly. Reasoning/answer blocks, by contrast, *are* paired (`<|think|>…<|/think|>`). The canonical single-turn template (`ARCHITECTURE.md:264-271`):

```
<|system|>{system_message}
<|user|>{user_question}
<|assistant|><|think|>{reasoning}<|/think|><|answer|>{final_answer}<|/answer|>
<|end_turn|>
```

Open-only role tags keep the template terse — one token per turn boundary instead of two — and the model learns the turn structure from the *opening* tags alone.

---

## 4. The consequence of the missing tokens (a real gotcha)

This is not cosmetic. The contract closes the loop only if a control string encodes to its reserved single ID. With IDs 14–20 *not on disk*, the tokenizer has no atomic entry for them — so it falls back to **byte-level BPE**, fragmenting each missing tag into several ordinary subword tokens.

- `<|end_turn|>` should be `[14]`. With the current tokenizer it is some multi-token sequence of `<`, `|`, `end`, `_`, `turn`, `|`, `>` fragments — and crucially, *those same fragments* can appear in ordinary text, so the model cannot cleanly distinguish a turn separator from prose.
- **Tool use** (`<|tool_call|>`, `<|/tool_call|>`, `<|tool_result|>`, `<|/tool_result|>`) and **multimodal** (`<|image|>`, `<|audio|>`) are the multi-turn / agentic templates in `ARCHITECTURE.md:280-287`. They will *silently mis-tokenize* — no error, just fragmented IDs the model was never trained to treat as control tokens.

The fix is ordered explicitly in `ARCHITECTURE.md:227-230`: add IDs 14–20, then write a `tokenizer_contract_test.py` asserting e.g. `tok("<|end_turn|>") == [14]`, **before** any tool-use or vision training. Basic chat (system/user/assistant/think/answer) works today because every token that template needs (0–13) is on disk; tool-use and multimodal do not. Note this is independent of the §2.3 width lag — even a 65,536-wide rebuild must still register these seven tokens at their reserved IDs.

---

## 5. The embedding matrix

### 5.1 Shape and construction

The embedding is a single `nn.Embedding` created once in `OSRTModel.__init__`:

```python
self.embedding = nn.Embedding(config.vocab_size, config.dim)   # model.py:1237
```

With the preset (`vocab_size = 65,536`, `dim = 1,536`) that is a **`65,536 × 1,536`** table. The forward pass turns IDs into vectors with a plain lookup — `x = self.embedding(input_ids)` (`model.py:1351`) — producing the `(B, S, 1536)` tensor that is the input to loop 0 of the recursion (§6).

### 5.2 The tied LM head (and what tying saves)

There is **no separate `lm_head` module anywhere in the model**. The output projection reuses the embedding matrix directly via `F.linear`:

```python
logits = F.linear(hidden, self.model.embedding.weight)   # model.py:1658
```

`F.linear(h, W)` computes `h @ W.T`. Because `W` is the `(65536, 1536)` embedding, this maps the `1,536`-dim final hidden state to one logit per vocabulary entry — the LM head — *using the same weights that did the input lookup*. The same matrix is reused for the auxiliary per-loop heads (`model.py:1723`) and the training-only MTP heads (`model.py:1771`). The class docstring states it plainly: *"LM head is weight-tied to embeddings (via F.linear with embedding.weight)"* (`model.py:1569`).

**Why tie?** Two reasons:

1. **Parameter saving.** An untied LM head would be a second `65,536 × 1,536 = 100,663,296`-param matrix. Tying avoids that entire matrix — a **~100M-param saving** at the 65,536 preset, roughly a sixth of the whole model.
2. **Inductive bias.** The "meaning" a token contributes on input and the "evidence" that should raise that token's output logit are closely related, so sharing weights is a sensible prior, not just a budget hack. It is standard for small models (GPT-2 onward).

> **Stale comment.** `model.py:1570` says tying *"Saves ~50M params vs untied for 32K×1536 embedding."* That figure is written against the **32,768 default**, not the preset. At the 65,536 preset the saving is ~100M (one avoided 100,663,296-param matrix). The number is stale; the mechanism is correct.

### 5.3 Initialization — verified against code (ARCHITECTURE §4.2 is wrong)

Initialization runs through `OSRTPreTrainedModel._init_weights` (`model.py:1216-1228`):

```python
def _init_weights(self, module):
    std = self.config.initializer_range            # = 0.02  (config.py:256)
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        ...
    elif isinstance(module, nn.Embedding):
        custom_std = getattr(module, "_osrt_init_std", None)
        nn.init.normal_(module.weight, mean=0.0,
                        std=custom_std if custom_std is not None else std)
```

So an embedding gets a per-module `_osrt_init_std` *if it has one*, else the default `0.02`. The only module that sets `_osrt_init_std` is `loop_embeddings` (= `0.1`, `model.py:214`). The **main token embedding has no `_osrt_init_std`**, so it initializes as **plain normal, mean 0, std 0.02**.

`ARCHITECTURE.md:299-303` disagrees three ways — flag all three as doc-vs-code discrepancies:

| ARCHITECTURE.md §4.2 claims | Code actually does |
|---|---|
| **truncated** normal | plain `nn.init.normal_` (`model.py:1224`) |
| std = `1/√1536 ≈ 0.0255` | std = `0.02` (`config.py:256`) |
| divide logits by `√1536` (μP) | no scaling — `F.linear(hidden, embedding.weight)` is unscaled (`model.py:1658`) |

The design *intent* in §4.2 (μP-flavoured init/scaling) was not what shipped; the implementation is the simpler "plain normal std=0.02, no output scaling" path. (Per `ARCHITECTURE.md:305-309`, the embedding is also routed to AdamW — not Muon — with no weight decay, to preserve representation norms.)

---

## 6. How the embedding feeds the recursion

The embedding is the **loop-0 input**. `OSRTModel.forward` does the lookup once, then hands the `(B, S, 1536)` tensor to the recursive stack:

```python
x = self.embedding(input_ids)                     # model.py:1351
if self.use_mhc:
    x = x.unsqueeze(2).repeat(1, 1, self.config.n_hc, 1)   # model.py:1355
```

When manifold-constrained hyper-connections are enabled, that vector is *replicated* into `n_hc` residual channels (`.repeat`, not `.expand`, so the channels are independent storage). From there the same physical blocks are applied repeatedly — 3 blocks × 6 loops = 18 effective layers — before `norm_out` and the tied LM head. The embedding therefore seeds a loop, not a feed-forward stack: see `docs/06-recursion.md` for the loop structure and `docs/04-mhc.md` for the multi-channel residual it feeds into. The loop *also* has its own small `loop_embeddings` table (`6 × 1536`, used by the MoE router per-loop, `model.py:213`) — distinct from this token embedding, and counted separately below.

---

## 7. Parameter cost

From `scripts/compute_budget.py` on the canonical `OSRT_605M_A288M` preset (run it yourself: `PYTHONPATH=src python scripts/compute_budget.py`):

```
cfg: dim=1536 vocab=65536 blocks=3 loops=6 ... mtp=2 mhc=True
  embedding           100,690,944
  attention            17,308,032
  ...
  TOTAL PHYSICAL      601,444,465  (~601M)
  ACTIVE / TOKEN      278,217,841  (~278M, 46.3% of physical, inference — excl. MTP)
```

- **Pure tied embedding matrix:** `65,536 × 1,536 = 100,663,296`.
- **`compute_budget.py` "embedding" bucket:** `100,690,944` — `27,648` more. That difference is exactly the `loop_embeddings` (`3 blocks × 6 loops × 1,536 = 27,648`): the budget categoriser matches the substring `"embedding"` first, so `loop_embeddings` lands in the embedding bucket (note there is no separate `loop_emb` line in the output — confirming the bundling). The pedagogically honest split is **100,663,296 token-embedding + 27,648 loop-embedding**.
- **Share of the model:** `100,690,944 / 601,444,465 = 16.7%`. This is the embedding tax in numbers — and it is the LFM2-style "blocks not vocabulary" target (§2.2), not the Gemma-3-270M ~63% catastrophe.

Because the LM head is tied, this single matrix is the *entire* cost of both the input and output interface. An untied head would add another ~100,663,296 params (~17% more model) for no new representational class. Note also that `compute_budget.py` counts the embedding as **fully active per token** at inference (`scripts/compute_budget.py:64-65`) — the tied head touches the whole matrix on every forward, so unlike the sparse routed experts there is no "active fraction" discount here.

> **Note on the equation.** A common shorthand writes the embedding as `100,690,944` params; that figure already *bundles* the 27,648 loop-embedding params. The pure tied token-embedding matrix is `100,663,296`. Both are correct for what they measure — just don't conflate "`65,536 × 1,536`" (= 100,663,296) with the bundled bucket (= 100,690,944).

---

## 8. Summary

| Aspect | Value | Source |
|---|---|---|
| Algorithm | byte-level BPE, GPT-2 pre-tok regex | `ARCHITECTURE.md:213-218` |
| Vocab (target/preset) | 65,536 (`2^16`) | `presets.py:27` |
| Vocab (on-disk artifact) | **32,768** (32,498 merges) — needs rebuild | `tokenizer/tokenizer.json` |
| Special tokens on disk | 14 (IDs 0–13) of 21 contract | `tokenizer/tokenizer.json` added_tokens |
| Missing tokens | IDs 14–20 (tool-use + multimodal) | `ARCHITECTURE.md:254-260` |
| Embedding shape | `65,536 × 1,536` | `model.py:1237` |
| LM head | **tied** via `F.linear(h, embedding.weight)` | `model.py:1658` |
| Init | plain normal, std `0.02` (no μP scaling) | `model.py:1216-1228`, `config.py:256` |
| Params (tied matrix) | 100,663,296 | `65,536 × 1,536` |
| Params (budget bucket) | 100,690,944 (+27,648 loop-emb), 16.7% | `scripts/compute_budget.py` |
