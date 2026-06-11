"""Snapshot the fragile gated Nemotron dataset repos to a local HF cache.

WHY: across the v6 midtrain runs, live-streaming the small gated Nemotron
configs caused repeated cold-start HF-Hub connection storms (Connection reset
/ Bad file descriptor / SSL bad-record-mac on their parquet shards), stalling
or crippling the run before the first step. A Lightning box has persistent
disk, so we download these repos ONCE; then load_dataset(..., streaming=True)
with HF_HUB_OFFLINE=1 streams from the local parquet files — no network, no
storm.

The mid-train knowledge mix streams 3 nvidia repos (the storm-prone ones):
  - nvidia/Nemotron-Pretraining-Specialized-v1  (STEM-SFT, Math-Textbooks,
    InfiniByte-Reasoning configs)
  - nvidia/Nemotron-Pretraining-Code-v2         (Synthetic-Question-Answering)
  - nvidia/Nemotron-CC-Math-v1                  (4plus) — large; see note
FineWeb-Edu and Cosmopedia stream fine and are huge; leave them live.

USAGE (on the box, BEFORE training):
    HF_TOKEN=<halld token> python scripts/snapshot_gated_datasets.py \
        --cache /teamspace/studios/this_studio/hf_cache

Then train with:
    HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=/teamspace/studios/this_studio/hf_cache ...

NOTE on size: the small Specialized-v1 + Code-v2 configs are the cheap, high-
value snapshots (they're the storm source AND small). Nemotron-CC-Math-v1
'4plus' is ~52B tokens — do NOT fully download it; it streams more reliably
than the small repos anyway (more shards, less per-shard contention). This
script snapshots the SMALL fragile repos by default and SKIPS CC-Math unless
--include-ccmath is passed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

# (repo_id, config) pairs to snapshot. allow_patterns restricts to the
# config's parquet subtree so we don't pull unrelated configs.
SMALL_GATED = [
    ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-STEM-SFT"),
    ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-Math-Textbooks"),
    ("nvidia/Nemotron-Pretraining-Specialized-v1", "Nemotron-Pretraining-InfiniByte-Reasoning"),
    ("nvidia/Nemotron-Pretraining-Code-v2", "Synthetic-Question-Answering"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True,
                    help="HF_DATASETS_CACHE dir to snapshot into (persistent disk)")
    ap.add_argument("--include-ccmath", action="store_true",
                    help="also snapshot Nemotron-CC-Math-v1 4plus (~52B tokens, large)")
    args = ap.parse_args()

    if not (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        print("ERROR: no HF_TOKEN in env — gated Nemotron repos will 403.",
              flush=True)
        return 2

    os.makedirs(args.cache, exist_ok=True)
    os.environ.setdefault("HF_DATASETS_CACHE", args.cache)

    from datasets import load_dataset

    targets = list(SMALL_GATED)
    if args.include_ccmath:
        targets.append(("nvidia/Nemotron-CC-Math-v1", "4plus"))

    ok = 0
    for repo, cfg in targets:
        t = time.time()
        print(f"\n=== snapshotting {repo} / {cfg} → {args.cache} ===", flush=True)
        try:
            # streaming=False downloads + caches the parquet locally. Iterate a
            # couple rows to confirm it materialised, then it's cached for the
            # offline streaming path during training.
            ds = load_dataset(repo, name=cfg, split="train",
                              cache_dir=args.cache)
            n = ds.num_rows if hasattr(ds, "num_rows") else len(ds)
            print(f"  OK: {n:,} rows cached in {time.time()-t:.0f}s", flush=True)
            ok += 1
        except Exception as e:
            print(f"  FAIL {repo}/{cfg}: {type(e).__name__}: {str(e)[:160]}",
                  flush=True)

    print(f"\n=== snapshot result: {ok}/{len(targets)} repos cached ===",
          flush=True)
    print("Now train with HF_HUB_OFFLINE=1 HF_DATASETS_CACHE=" + args.cache,
          flush=True)
    return 0 if ok == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
