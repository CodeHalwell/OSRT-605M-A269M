"""v6 SFT rollout dataset, split out of osrt/data.py when v6 was archived."""

class RolloutDataset(IterableDataset):
    """Streaming dataset for MOPD distillation from a JSONL rollout file.

    JSONL schema (one record per line):
        {"prompt": str, "thinking": str, "response": str, ...}

    Records with empty `response` are skipped. The full sequence is:
        <|user|>{prompt}<|assistant|><|think|>{thinking}<|/think|>
        <|answer|>{response}<|/answer|>
    Sequences too long for seq_len have the assistant portion truncated
    (we keep the full prompt — if the prompt alone wouldn't leave room
    for at least min_response_tokens of answer, the example is skipped).

    Labels: -100 for the prompt prefix (and padding); real token ids
    for the assistant portion. Cross-entropy with ignore_index=-100
    yields per-token assistant loss only.
    """

    MIN_RESPONSE_TOKENS = 32

    def __init__(
        self,
        jsonl_path: str,
        seq_len: int,
        tok_name: str,
        seed: int,
    ) -> None:
        self.jsonl_path = jsonl_path
        self.seq_len = seq_len
        self.tok_name = tok_name
        self.seed = seed

    def _build_sequence(
        self,
        rec: dict,
        tok: AutoTokenizer,
        pad_id: int,
    ) -> tuple[Tensor, Tensor] | None:
        prompt = (rec.get("prompt") or "").strip()
        thinking = (rec.get("thinking") or "").strip()
        response = (rec.get("response") or "").strip()
        # NEW: optional system prompt. When present, the prefix becomes
        # <|system|>{sys}<|user|>{q}<|assistant|>. Loss is still only
        # computed on the assistant portion (everything before is in the
        # ignored prefix), so the model learns to ATTEND to the system
        # block but not regenerate it.
        system = (rec.get("system") or "").strip()
        if not prompt or not response:
            return None

        if system:
            prefix_text = f"<|system|>{system}<|user|>{prompt}<|assistant|>"
        else:
            prefix_text = f"<|user|>{prompt}<|assistant|>"
        if thinking:
            target_text = (
                f"<|think|>{thinking}<|/think|><|answer|>{response}<|/answer|>"
            )
        else:
            # Datasets without explicit `thinking` (OpenHermes, etc.)
            # are trained as direct answer-only outputs. The model should
            # learn that `<|answer|>` follows `<|assistant|>` even
            # without a `<|think|>` block.
            target_text = f"<|answer|>{response}<|/answer|>"

        prefix_ids = tok.encode(prefix_text, add_special_tokens=False)
        target_ids = tok.encode(target_text, add_special_tokens=False)

        # Terminal EOS so the model learns to STOP after <|/answer|>. Without
        # it the target ended at <|/answer|> followed by masked (-100) padding,
        # so EOS was never a training target — and at inference the model never
        # stopped, running to the length cap (observed in SFT-v2: both modes
        # maxed out, resp_len ≈ cap). Reserve a slot for it and give it a REAL
        # label so it's actually learned.
        eos_id = tok.eos_token_id
        reserve = 1 if eos_id is not None else 0

        # Truncation policy: keep full prefix, truncate target tail. Skip
        # if even after truncation the target is below the minimum.
        max_target = self.seq_len - len(prefix_ids) - reserve
        if max_target < self.MIN_RESPONSE_TOKENS:
            return None
        if len(target_ids) > max_target:
            target_ids = target_ids[:max_target]
        if eos_id is not None:
            target_ids = target_ids + [eos_id]

        seq_ids = prefix_ids + target_ids
        seq_labels: list[int] = (
            [-100] * len(prefix_ids) + list(target_ids)  # EOS is a real label
        )
        # Right-pad to seq_len
        pad_len = self.seq_len - len(seq_ids)
        if pad_len > 0:
            seq_ids = seq_ids + [pad_id] * pad_len
            seq_labels = seq_labels + [-100] * pad_len

        return (
            torch.tensor(seq_ids, dtype=torch.long),
            torch.tensor(seq_labels, dtype=torch.long),
        )

    def __iter__(self):  # noqa: ANN204
        import json

        tok = AutoTokenizer.from_pretrained(self.tok_name)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        worker_info = torch.utils.data.get_worker_info()
        seed = self.seed if worker_info is None else self.seed + worker_info.id
        n_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id
        rng = random.Random(seed)

        # Load full JSONL into memory (4K-50K rollouts, ~1MB-15MB — fits
        # easily). Filter out empty responses up front. Each worker keeps
        # its own copy of the SAME list but shuffles with a worker-specific
        # seed, then strides by num_workers so they don't yield the same
        # record in the same step.
        rollouts: list[dict] = []
        with open(self.jsonl_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("response"):
                    rollouts.append(rec)

        if not rollouts:
            raise RuntimeError(f"No usable rollouts in {self.jsonl_path}")
        print(
            f"[RolloutWorker {worker_id}/{n_workers}] loaded "
            f"{len(rollouts)} rollouts from {self.jsonl_path}",
            flush=True,
        )

        # Infinite cycle so training never runs out of data — the
        # train loop stops on total_steps, not on dataset exhaustion.
        epoch = 0
        while True:
            indices = list(range(len(rollouts)))
            rng.shuffle(indices)
            # Stride across workers so each rank sees a disjoint subset
            # per epoch (a poor-man's distributed sampler).
            for i in indices[worker_id::n_workers]:
                out = self._build_sequence(rollouts[i], tok, pad_id)
                if out is not None:
                    yield out
            epoch += 1


def make_rollout_loader(
    jsonl_path: str,
    seq_len: int,
    tokenizer_name: str,
    batch_size: int,
    step_num: int,
    num_workers: int = 2,
    prefetch_factor: int = 2,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """Drop-in replacement for make_loader() that serves MOPD rollouts.

    Same return shape: DataLoader yielding (input_ids, labels) where
    labels mask the user prompt with -100.
    """
    ds = RolloutDataset(
        jsonl_path=jsonl_path,
        seq_len=seq_len,
        tok_name=tokenizer_name,
        seed=42 + step_num,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        multiprocessing_context="spawn" if num_workers > 0 else None,
    )
