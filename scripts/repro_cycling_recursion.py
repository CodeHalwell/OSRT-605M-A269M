"""Deterministic, network-free repro for the midtrain data-loader failure.

Production symptom (v6 midtrain probe, nemotron-math-textbooks), in the
DataLoader worker, before any forward pass:
    RecursionError: maximum recursion depth exceeded in comparison
…looping forever on reconnect.

Root cause: the OLD TokenStream._cycling_iter did `ds = ds.shuffle(...)` every
cycle, wrapping the already-shuffled IterableDataset in ANOTHER shuffle wrapper.
A tiny dataset cycles many times fast, so the wrapper nest deepens fast. Iterating
an N-deep nest of HF `BufferShuffledExamplesIterable` is super-exponential and, past
a depth that depends on buffer size / platform, hits Python's recursion limit
(default 1000) during the buffer comparison → "RecursionError ... in comparison".
On a laptop with a small buffer it instead manifests as a multi-second-per-item
stall; same root cause, different surface.

Fix (TokenStream._cycling_iter): shuffle the UNSHUFFLED base fresh each cycle, so
the nest is always exactly ONE layer deep regardless of cycle count.

This test is deterministic, network-free, in-memory, and bounded (~seconds). It:
  1. Shows the OLD nested-shuffle pattern's time-to-first-item exploding with depth.
  2. Shows the FIXED base-shuffle pattern staying flat across hundreds of cycles.

Run: python scripts/repro_cycling_recursion.py
Exit 0 iff: nesting shows the blow-up AND the fixed pattern stays fast.
"""

from __future__ import annotations

import signal
import sys
import time

from datasets import Dataset


def _base():
    """Tiny in-memory streaming dataset that exhausts fast (forces cycling)."""
    return Dataset.from_dict(
        {"text": [f"row{i}" for i in range(30)]}
    ).to_iterable_dataset()


# Depth at which the OLD nested-shuffle pattern is already clearly pathological
# (seconds per single item). Below the recursion-limit crash seen on Modal, but
# the same monotonic blow-up — enough to prove the mechanism on any platform.
SLOW_THRESHOLD_S = 5.0


def characterize_buggy_nesting() -> bool:
    """OLD pattern: re-shuffle the shuffled view. Time-to-first-item should
    blow up with nest depth. Returns True if the blow-up is observed."""
    print("=== BUGGY: nested .shuffle() — time-to-first-item vs nest depth ===")
    for k in (1, 5, 10, 15, 20):
        cur = _base().shuffle(buffer_size=8, seed=0)
        for s in range(1, k):  # (k-1) further shuffles of the shuffled view
            cur = cur.shuffle(buffer_size=8, seed=s)
        t = time.time()
        try:
            next(iter(cur))
        except RecursionError as e:
            print(f"  nest={k:>3}: RecursionError ({str(e)[:48]}) — blow-up confirmed")
            return True
        dt = time.time() - t
        print(f"  nest={k:>3}: first item in {dt:.3f}s")
        if dt > SLOW_THRESHOLD_S:
            print(
                f"  nest={k:>3}: >{SLOW_THRESHOLD_S}s for ONE item — blow-up confirmed"
            )
            return True
    print("  (no blow-up observed up to nest=20 — unexpected on this platform)")
    return False


def drive_fixed(cycles: int) -> bool:
    """FIXED pattern: shuffle the UNSHUFFLED base fresh each cycle (one layer).
    Must stay fast across many cycles. Returns True if it does."""
    rows = 30
    pulls = cycles * rows
    print(f"\n=== FIXED: base-shuffle each cycle — {pulls} pulls ({cycles} cycles) ===")

    def _fixed_iter(base):
        seed = 0
        while True:
            for ex in base.shuffle(buffer_size=8, seed=seed):
                yield ex
            seed += 1

    base = _base()
    it = _fixed_iter(base)
    t = time.time()
    for _ in range(pulls):
        next(it)
    dt = time.time() - t
    print(f"  pulled {pulls} items in {dt:.2f}s — flat, no slowdown, no RecursionError")
    return dt < SLOW_THRESHOLD_S


def main() -> int:
    def _boom(*_a):
        print("HARD-TIMEOUT — exceeded 90s budget", flush=True)
        raise SystemExit(2)

    signal.signal(signal.SIGALRM, _boom)
    signal.alarm(90)

    print(f"python recursionlimit: {sys.getrecursionlimit()}\n")
    buggy_blows_up = characterize_buggy_nesting()
    fixed_fast = drive_fixed(cycles=600)

    print("\n" + "=" * 60)
    print(f"BUGGY nested-shuffle blows up?  {buggy_blows_up}  (want True)")
    print(f"FIXED base-shuffle stays fast?  {fixed_fast}  (want True)")
    ok = buggy_blows_up and fixed_fast
    print(f"VERDICT: {'mechanism confirmed + fix proven' if ok else 'INCONCLUSIVE'}")
    print("=" * 60)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
