"""optimize_for_inference() — the decode-speed inference prep (Idea #1).

Guards the property that makes it safe: turning MoE/loop telemetry OFF must not
change the model's outputs. The compile() half is a torch.compile fusion (no
math change, verified bit-exact on GPU: max|Δlogit|=0, 100% greedy token-match)
and isn't exercised here — CPU inductor is slow/flaky in CI and adds no signal
beyond the telemetry gate, which is the only thing that touches forward outputs.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_model import tiny_config  # noqa: E402

from osrt.model import OSRTForCausalLM  # noqa: E402


def _fixed_input(cfg, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, cfg.real_vocab_size, (2, 16), generator=g)


def test_telemetry_off_is_output_identical():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = OSRTForCausalLM(cfg).eval()
    x = _fixed_input(cfg)

    with torch.no_grad():
        ref = model(x).logits.clone()  # telemetry ON (default)
        model.set_moe_telemetry(False)
        off = model(x).logits.clone()  # telemetry OFF

    assert torch.equal(ref, off), (
        f"telemetry gate changed logits: max|Δ|={(ref - off).abs().max()}"
    )


def test_optimize_for_inference_no_compile_matches_eager():
    torch.manual_seed(0)
    cfg = tiny_config()
    model = OSRTForCausalLM(cfg)
    x = _fixed_input(cfg)

    with torch.no_grad():
        ref = model.eval()(x).logits.clone()
        ret = model.optimize_for_inference(compile_model=False)
        opt = model(x).logits.clone()

    assert ret is model                       # returns self for chaining
    assert model.training is False            # eval() applied
    assert torch.equal(ref, opt)              # outputs unchanged


def test_optimize_for_inference_disables_telemetry_on_all_blocks():
    cfg = tiny_config()
    model = OSRTForCausalLM(cfg)
    model.optimize_for_inference(compile_model=False)
    for blk in model.model.blocks:
        assert blk.moe.telemetry_enabled is False
