"""StaticKVCache ("speed mode") — fixed post-RoPE K/V decode buffers.

Guards the equivalence that makes it shippable: generate(cache_impl="static")
must produce the same tokens as the latent-cache path (fp32 CPU: the GEMV vs
GEMM accumulation difference is far below argmax-flip scale on a tiny model).
GPU bf16 equivalence is gated separately via held-out ppl (per reviewer:
mathematically equivalent, not bit-identical).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from test_model import tiny_config  # noqa: E402

from osrt.model import OSRTForCausalLM, StaticKVCache  # noqa: E402


def _model(**over):
    torch.manual_seed(0)
    return OSRTForCausalLM(tiny_config(**over)).eval()


def _gen(model, ids, impl, n=12):
    with torch.no_grad():
        out = model.generate(
            ids, max_new_tokens=n, temperature=0.0, cache_impl=impl,
        )
    return out[0, ids.shape[1]:].tolist()


def test_static_generate_matches_latent():
    model = _model()
    ids = torch.randint(0, 512, (1, 8), generator=torch.Generator().manual_seed(1))
    assert _gen(model, ids, "latent") == _gen(model, ids, "static")


def test_static_generate_matches_latent_batched():
    model = _model()
    ids = torch.randint(0, 512, (3, 8), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        a = model.generate(ids, max_new_tokens=10, temperature=0.0,
                           cache_impl="latent")
        b = model.generate(ids, max_new_tokens=10, temperature=0.0,
                           cache_impl="static")
    assert torch.equal(a, b)


def test_static_generate_matches_latent_reduced_loops():
    # num_loops < recursive_loops changes the effective layer count — the
    # static cache must key off the same resolved count as the latent path.
    model = _model()
    ids = torch.randint(0, 512, (1, 8), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        a = model.generate(ids, max_new_tokens=8, temperature=0.0,
                           num_loops=1, cache_impl="latent")
        b = model.generate(ids, max_new_tokens=8, temperature=0.0,
                           num_loops=1, cache_impl="static")
    assert torch.equal(a, b)


def test_static_cache_rejects_attention_mask():
    model = _model()
    ids = torch.randint(0, 512, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    try:
        with torch.no_grad():
            model.generate(ids, max_new_tokens=4, attention_mask=mask,
                           cache_impl="static")
        raise AssertionError("expected ValueError")
    except ValueError:
        pass


def test_static_cache_cursor_device_side():
    cache = StaticKVCache(
        num_layers=2, batch=1, kv_heads=2, head_dim=8, max_len=16,
        device=torch.device("cpu"), dtype=torch.float32,
    )
    cache.cursor.fill_(3)
    cache.advance()
    assert cache.cursor.item() == 4
    view = cache.layer(1)
    assert view.k is cache.k[1] and view.cursor is cache.cursor
