"""Paired item bootstrap + leave-one-checkpoint-out fits for the held-out panel.

Why the OLS intervals were wrong
--------------------------------
Every checkpoint is scored on the SAME 200 GSM8K questions, so the measurements
are a panel, not seven independent samples. Item difficulty is shared across
checkpoints: a question no checkpoint can solve contributes an identical 0 to
all seven means, and a question all seven solve contributes an identical 1.
An ordinary OLS interval over checkpoint means treats the residuals as
independent and therefore understates the uncertainty in the slope.

The right resampling unit is the QUESTION, not the checkpoint. This script
resamples the 200 items with replacement, preserving each item's full
seven-checkpoint trajectory, recomputes the per-checkpoint accuracies and refits
the slope inside each replicate. It also runs leave-one-checkpoint-out fits, so
a single influential checkpoint can't carry a conclusion on its own.

Reads the per-checkpoint JSON written by app.py::sft_eval_sweep (which stores
`items.on` / `items.off` as 0/1 lists indexed by problem).

Usage:
  python scripts/eval_paired_bootstrap.py --items-dir ./items --reps 10000
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import statistics as st


def _slope(xs: list[float], ys: list[float]) -> float:
    mx, my = st.mean(xs), st.mean(ys)
    sxx = sum((a - mx) ** 2 for a in xs)
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / sxx


def _step_of(name: str) -> int:
    m = re.search(r"_step_(\d+)", name)
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items-dir", required=True)
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-step", type=int, default=0,
                    help="exclude checkpoints below this step. Use 1 to drop "
                         "the pre-GRPO baseline, which is a LEVEL SHIFT rather "
                         "than part of the trend and would otherwise turn a "
                         "flat line into a spurious slope.")
    args = ap.parse_args()

    rows = []
    for p in sorted(glob.glob(os.path.join(args.items_dir, "*.json"))):
        d = json.load(open(p))
        step = _step_of(d["ckpt"])
        rows.append({"step": step, "ckpt": d["ckpt"],
                     "on": d["items"]["on"], "off": d["items"]["off"]})
    baseline = [r for r in rows if r["step"] < 0]
    rows = [r for r in rows if r["step"] >= args.min_step and r["step"] >= 0]
    rows.sort(key=lambda r: r["step"])
    if len(rows) < 3:
        raise SystemExit(f"need >=3 checkpoints with a step, found {len(rows)}")

    n_items = len(rows[0]["on"])
    assert all(len(r["on"]) == n_items and len(r["off"]) == n_items for r in rows), \
        "checkpoints scored different numbers of items — not a paired panel"
    steps = [float(r["step"]) for r in rows]
    print(f"panel: {len(rows)} checkpoints x {n_items} items  "
          f"(steps {int(min(steps))}-{int(max(steps))})")
    for b in baseline:
        print(f"  baseline (excluded from slope): {b['ckpt']} "
              f"acc_on {100*st.mean(b['on']):.1f}%")

    def series(idx: list[int], key: str) -> list[float]:
        return [100.0 * sum(r[key][i] for i in idx) / len(idx) for r in rows]

    allidx = list(range(n_items))
    point = {k: _slope(steps, series(allidx, k)) * 100 for k in ("on", "off")}
    on_s, off_s = series(allidx, "on"), series(allidx, "off")
    delta_obs = [a - b for a, b in zip(on_s, off_s)]
    point["delta"] = _slope(steps, delta_obs) * 100

    print("\nobserved per-checkpoint accuracy")
    for r, a, b in zip(rows, on_s, off_s):
        print(f"  step {r['step']:>4}  acc_on {a:5.1f}%  acc_off {b:5.1f}%  "
              f"delta {a-b:+5.1f}pp")

    # ── paired item bootstrap ──────────────────────────────────────────
    rng = random.Random(args.seed)
    boot: dict[str, list[float]] = {"on": [], "off": [], "delta": []}
    for _ in range(args.reps):
        idx = [rng.randrange(n_items) for _ in range(n_items)]
        o, f = series(idx, "on"), series(idx, "off")
        boot["on"].append(_slope(steps, o) * 100)
        boot["off"].append(_slope(steps, f) * 100)
        boot["delta"].append(_slope(steps, [a - b for a, b in zip(o, f)]) * 100)

    print(f"\npaired item bootstrap, {args.reps} reps "
          f"(resampling QUESTIONS, trajectories preserved)")
    for k in ("on", "off", "delta"):
        v = sorted(boot[k])
        lo, hi = v[int(0.025 * len(v))], v[int(0.975 * len(v))]
        # Two-sided bootstrap p for "slope != 0". Must use the CLOSED tails
        # (<=0 and >=0), not `>0` and its complement: with every replicate
        # exactly 0 — which happens when a series is flat on a deterministic
        # panel — `>0` has zero mass and the naive 2*min(p, 1-p) reports
        # p=0.000, i.e. maximal significance for a slope that is identically
        # zero. Same failure with sign flipped for an all-negative slope.
        p_le = sum(1 for x in v if x <= 0) / len(v)
        p_ge = sum(1 for x in v if x >= 0) / len(v)
        pval = min(1.0, 2 * min(p_le, p_ge))
        crosses = "crosses 0" if lo <= 0 <= hi else "EXCLUDES 0"
        print(f"  {k:<6} slope {point[k]:+6.2f} pp/100 steps   "
              f"95% CI [{lo:+.2f}, {hi:+.2f}]   p={pval:.3f}   {crosses}")

    # ── leave-one-checkpoint-out ───────────────────────────────────────
    print("\nleave-one-checkpoint-out slopes (pp/100 steps)")
    print(f"  {'dropped':>8}  {'on':>8} {'off':>8} {'delta':>8}")
    for j in range(len(rows)):
        keep = [i for i in range(len(rows)) if i != j]
        s2 = [steps[i] for i in keep]
        o2 = [on_s[i] for i in keep]
        f2 = [off_s[i] for i in keep]
        d2 = [o2[i] - f2[i] for i in range(len(keep))]
        print(f"  {rows[j]['step']:>8}  {_slope(s2, o2)*100:+8.2f} "
              f"{_slope(s2, f2)*100:+8.2f} {_slope(s2, d2)*100:+8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
