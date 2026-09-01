"""Print full generated text from a local SFT checkpoint — eyeball quality
WITHOUT spending GPU time on the scored eval.

For each of N held-out GSM8K problems it generates with the reasoning-ON and
reasoning-OFF personas and prints the raw decoded output (special tokens kept),
the gold answer, the extracted prediction, and whether it's well-formed / correct.

Run: HF_TOKEN=... PYTHONPATH=src python scripts/local_sft_sample.py \
        --ckpt checkpoints/v5/osrt_v5_sft_v1_final.pt --n 4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_env = Path(__file__).parent.parent / ".env"
if _env.exists():
    for _l in _env.read_text().splitlines():
        _l = _l.strip()
        if _l and not _l.startswith("#") and "=" in _l:
            k, _, v = _l.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/v5/osrt_v5_sft_v1_final.pt")
    ap.add_argument("--tokenizer", default="v6_tokenizer_export")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--rep-penalty", type=float, default=1.0)
    args = ap.parse_args()

    import random

    import torch
    from transformers import AutoTokenizer

    from osrt.model import OSRTForCausalLM
    from osrt.presets import build_config
    from osrt.rewards import extract_numeric_answer
    from osrt.sft_eval import _WELL_FORMED, _load_gsm8k_heldout, _norm
    from osrt.system_prompts import sample_system_prompt

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"device: {device}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    cfg = build_config(
        vocab_size=len(tok),
        real_vocab_size=len(tok),
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id,
        pad_token_id=tok.pad_token_id,
        fused_cross_entropy_chunks=8,
    )
    model = OSRTForCausalLM(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sd = ck["model_state_dict"] if "model_state_dict" in ck else ck
    model.load_state_dict(sd, strict=False)
    if args.dtype == "bf16":
        model = model.to(torch.bfloat16)
    model = model.to(device).eval()
    print(
        f"loaded {args.ckpt}  (stage={ck.get('training_stage')}, "
        f"steps={ck.get('total_steps')})\n",
        flush=True,
    )

    # Same fixed personas the scored eval uses, so this mirrors it.
    on_name, on_sys = sample_system_prompt(random.Random(0), "on")
    off_name, off_sys = sample_system_prompt(random.Random(0), "off")
    print(f"ON  persona [{on_name}]: {on_sys}")
    print(f"OFF persona [{off_name}]: {off_sys}")
    print(
        f"decode: temp={args.temperature} top_p={args.top_p} "
        f"rep_penalty={args.rep_penalty}\n"
    )

    @torch.no_grad()
    def gen(system_text: str, question: str) -> str:
        prompt = f"<|system|>{system_text}<|user|>{question}<|assistant|>"
        ids = torch.tensor(
            [tok.encode(prompt, add_special_tokens=False)],
            dtype=torch.long,
            device=device,
        )
        out = model.generate(
            ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.rep_penalty,
            eos_token_id=tok.eos_token_id,
        )
        return tok.decode(out[0, ids.shape[1] :], skip_special_tokens=False)

    problems = _load_gsm8k_heldout(args.n)
    for i, (q, gold) in enumerate(problems, 1):
        print("=" * 80)
        print(f"[Q{i}] {q}")
        print(f"  GOLD: {gold}")
        for side, sys_text in (("ON", on_sys), ("OFF", off_sys)):
            text = gen(sys_text, q)
            pred = _norm(extract_numeric_answer(text))
            ok_fmt = bool(_WELL_FORMED.search(text))
            correct = pred is not None and pred == _norm(gold)
            print(f"\n  --- {side} --- fmt_ok={ok_fmt}  pred={pred}  correct={correct}")
            print(f"  {text}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
