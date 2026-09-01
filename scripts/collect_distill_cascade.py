"""Distillation rollout collection with per-prompt teacher CASCADE.

Builds a new SFT/distillation dataset primarily from NVIDIA Nemotron 3 Ultra
(free, OpenRouter), falling back PER PROMPT to Nemotron 3 Super then Qwen3-Next
when Ultra is rate-limited / erroring. Every prompt gets answered; each record
notes which teacher produced it, so you can audit/filter by teacher later.

All three teachers are OpenRouter chat models that return separate
`reasoning` + `content` fields → map cleanly to osrt's <|think|>/<|answer|>
template (same schema as scripts/collect_rollouts.py / the MOPD set).

Reuses collect_rollouts.py for the prompt registry, resume logic, and the
OpenRouter call — this script only adds the cascade + a single shared client.

Output schema (JSONL, one per line) — matches collect_rollouts.py plus a
`teacher_chain` field recording the cascade that was tried:
    {"id","source","prompt","thinking","response","ts","elapsed_s",
     "input_tokens","output_tokens","teacher","teacher_chain"}

USAGE (e.g. on the Lightning box or anywhere with OPENROUTER access):
    OPEN_ROUTER_API_KEY=... PYTHONPATH=src \
    python scripts/collect_distill_cascade.py \
        --output rollouts/distill_cascade_v1.jsonl \
        --sources gsm8k open_thoughts mbpp ultrachat sciq \
        --max-per-source 2000 \
        --concurrency 8 \
        --ultra-retries 3

Resumes automatically: existing rollout IDs in --output are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
from pathlib import Path

# Reuse the proven pieces from the existing collector.
from collect_rollouts import (  # noqa: E402  (same scripts/ dir on PYTHONPATH)
    SOURCES,
    call_openrouter,
    iter_prompts,
    load_done_ids,
)

# ── Cascade: ordered list of OpenRouter model IDs. Index 0 is primary. ──
# Slugs: Ultra is the proven one already used in collect_rollouts.py; Super
# and Qwen3-Next are from the OpenRouter free catalogue (data/
# openrouter_free_models.json). All :free → $0 but rate-limited, which is
# exactly why the cascade exists.
CASCADE = [
    ("ultra", "nvidia/nemotron-3-ultra-550b-a55b:free"),
    ("super", "nvidia/nemotron-3-super-120b-a12b:free"),
    ("qwen3-next", "qwen/qwen3-next-80b-a3b-instruct:free"),
]

DEFAULT_OUTPUT = Path(__file__).parent.parent / "rollouts" / "distill_cascade_v1.jsonl"

# Errors worth falling back / retrying on (rate limits, 5xx, timeouts).
_RETRYABLE = (
    "429",
    "rate",
    "limit",
    "503",
    "500",
    "502",
    "504",
    "timeout",
    "overloaded",
    "unavailable",
    "connection",
)


def _is_retryable(exc: Exception) -> bool:
    return any(s in str(exc).lower() for s in _RETRYABLE)


async def cascade_call(
    client,
    prompt: str,
    *,
    ultra_retries: int,
    per_model_retries: int,
) -> dict:
    """Try each teacher in CASCADE order. The PRIMARY (Ultra) gets
    `ultra_retries` attempts with backoff before we fall through; each
    fallback gets `per_model_retries`. Returns the rollout dict plus
    `teacher` (the one that succeeded) and `teacher_chain` (all tried).

    Raises RuntimeError only if EVERY teacher in the cascade is exhausted.
    """
    chain: list[str] = []
    for tname, model_id in CASCADE:
        attempts = ultra_retries if tname == "ultra" else per_model_retries
        for attempt in range(attempts):
            chain.append(tname)
            try:
                # call_openrouter is sync → run in a worker thread.
                result = await asyncio.to_thread(
                    call_openrouter,
                    client,
                    prompt,
                    model_id,
                )
                # A model can return 200 with empty content (free-tier
                # truncation / refusal). Treat empty as a soft failure so
                # the cascade moves on rather than recording a blank.
                if not result["response"].strip():
                    raise RuntimeError(f"{tname}: empty response")
                result["teacher"] = tname
                result["teacher_chain"] = chain.copy()
                return result
            except Exception as e:  # noqa: BLE001
                last = attempt == attempts - 1
                if _is_retryable(e) and not last:
                    wait = 2**attempt + random.random()
                    await asyncio.sleep(wait)
                    continue
                # non-retryable, or out of attempts for this model → next teacher
                print(f"  [{tname} gave up: {str(e)[:90]}] -> next teacher", flush=True)
                break
    raise RuntimeError(f"cascade exhausted (tried {chain})")


async def collect(args: argparse.Namespace) -> None:
    import concurrent.futures as _cf
    import os

    loop = asyncio.get_running_loop()
    loop.set_default_executor(_cf.ThreadPoolExecutor(max_workers=args.concurrency + 50))
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = max(soft, args.concurrency * 4 + 256)
        if target > soft:
            resource.setrlimit(resource.RLIMIT_NOFILE, (min(target, hard), hard))
    except (ImportError, ValueError, OSError):
        pass

    # Single shared OpenRouter client for the whole cascade (all 3 teachers
    # are OpenRouter), built the same way collect_rollouts._build_client does.
    try:
        from openrouter import OpenRouter
    except ImportError:
        print("ERROR: openrouter not installed. uv add openrouter", file=sys.stderr)
        sys.exit(1)
    api_key = os.environ.get("OPEN_ROUTER_API_KEY") or os.environ.get(
        "OPENROUTER_API_KEY"
    )
    if not api_key:
        print(
            "ERROR: OPEN_ROUTER_API_KEY (or OPENROUTER_API_KEY) not set",
            file=sys.stderr,
        )
        sys.exit(1)
    client = OpenRouter(api_key=api_key)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done_ids = load_done_ids(output)
    print(f"Cascade: {' -> '.join(t for t, _ in CASCADE)}")
    print(
        f"Primary retries (ultra)={args.ultra_retries}, "
        f"per-fallback retries={args.per_model_retries}"
    )
    print(f"Output: {output}")
    print(f"Resume: {len(done_ids)} rollouts already on disk\n", flush=True)

    queue: list[tuple[str, str, str]] = []
    for src_key in args.sources:
        if src_key not in SOURCES:
            print(f"WARN: unknown source '{src_key}', skipping", file=sys.stderr)
            continue
        added = 0
        for rid, prompt in iter_prompts(src_key, args.max_per_source):
            if rid in done_ids:
                continue
            queue.append((rid, src_key, prompt))
            added += 1
        print(f"[{src_key}] queued {added} prompts", flush=True)

    print(f"\nTotal queued: {len(queue)}, concurrency={args.concurrency}", flush=True)
    if not queue:
        print("Nothing to do.", flush=True)
        return

    write_q: asyncio.Queue = asyncio.Queue(maxsize=max(64, args.concurrency * 4))

    async def writer():
        with output.open("a", buffering=1) as f:
            while True:
                rec = await write_q.get()
                if rec is None:
                    return
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                write_q.task_done()

    writer_task = asyncio.create_task(writer())
    sem = asyncio.Semaphore(args.concurrency)
    start = time.time()
    stats = {"done": 0, "failed": 0, "by_teacher": {}}

    async def worker(rid: str, src_key: str, prompt: str) -> None:
        async with sem:
            t0 = time.time()
            try:
                result = await cascade_call(
                    client,
                    prompt,
                    ultra_retries=args.ultra_retries,
                    per_model_retries=args.per_model_retries,
                )
            except Exception as e:  # noqa: BLE001
                stats["failed"] += 1
                print(f"  ✗ {rid}: {str(e)[:100]}", flush=True)
                return
            rec = {
                "id": rid,
                "source": src_key,
                "prompt": prompt,
                "thinking": result["thinking"],
                "response": result["response"],
                "ts": time.time(),
                "elapsed_s": round(time.time() - t0, 2),
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "teacher": result["teacher"],
                "teacher_chain": result["teacher_chain"],
            }
            await write_q.put(rec)
            stats["done"] += 1
            stats["by_teacher"][result["teacher"]] = (
                stats["by_teacher"].get(result["teacher"], 0) + 1
            )
            if stats["done"] % 25 == 0:
                rate = stats["done"] / max(time.time() - start, 1e-6)
                mix = " ".join(
                    f"{k}:{v}" for k, v in sorted(stats["by_teacher"].items())
                )
                print(
                    f"  {stats['done']}/{len(queue)} done "
                    f"({rate:.1f}/s, failed {stats['failed']}) | {mix}",
                    flush=True,
                )

    await asyncio.gather(*(worker(r, s, p) for r, s, p in queue))
    await write_q.put(None)
    await writer_task

    mix = " ".join(f"{k}:{v}" for k, v in sorted(stats["by_teacher"].items()))
    print(
        f"\n=== DONE: {stats['done']} rollouts, {stats['failed']} failed "
        f"in {(time.time() - start) / 60:.1f} min ==="
    )
    print(f"teacher mix: {mix}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT))
    p.add_argument(
        "--sources",
        nargs="+",
        default=["gsm8k", "open_thoughts", "mbpp", "ultrachat", "sciq"],
        help="prompt sources from collect_rollouts.SOURCES",
    )
    p.add_argument("--max-per-source", type=int, default=2_000)
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="free-tier rate limits are the real cap; start ~8",
    )
    p.add_argument(
        "--ultra-retries",
        type=int,
        default=3,
        help="attempts on Ultra (primary) before falling back",
    )
    p.add_argument(
        "--per-model-retries",
        type=int,
        default=2,
        help="attempts per fallback teacher (Super, Qwen)",
    )
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(collect(parse_args()))
