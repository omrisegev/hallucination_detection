"""CPU checks for spectral_utils/rope_extension.py (run: python -m pytest tests/test_rope_extension.py -q)."""
import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spectral_utils.rope_extension import apply_dynamic_ntk  # noqa: E402

transformers = pytest.importorskip('transformers')
L0 = 64


V5 = int(transformers.__version__.split('.')[0]) >= 5


def rope_kw(scaling=None):
    if V5:
        return {'rope_parameters': {'rope_type': 'default', 'rope_theta': 10000.0, **(scaling or {})}}
    return {'rope_theta': 10000.0, **({'rope_scaling': scaling} if scaling else {})}


def tiny(scaling=None):
    cfg = transformers.Qwen2Config(vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=L0,
                                   **rope_kw(scaling))
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).eval()


class CachedQwen2RotaryEmbedding(torch.nn.Module):
    """Verbatim logic of the Qwen2.5-Math-PRM trust_remote_code Qwen2RotaryEmbedding (Mixtral style)."""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim, self.max_position_embeddings, self.base = dim, max_position_embeddings, base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype())

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        emb = torch.cat((torch.outer(t, self.inv_freq),) * 2, dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return self.cos_cached[:seq_len].to(dtype=x.dtype), self.sin_cached[:seq_len].to(dtype=x.dtype)


class Holder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = transformers.Qwen2Config(max_position_embeddings=L0, **rope_kw())
        self.rot = CachedQwen2RotaryEmbedding(16, L0, 10000.0)


def test_native_in_range_bitwise_and_beyond_matches_hf_dynamic():
    m = tiny(); ref = copy.deepcopy(m)
    rep = apply_dynamic_ntk(m, 3.0)
    assert rep['styles'] == ['native'] and rep['extended_context'] == 3 * L0
    x = torch.randint(0, 97, (1, L0))
    with torch.no_grad():
        assert torch.equal(m(x).logits, ref(x).logits)            # <= L0: original path, bit-identical
    hf = tiny({'rope_type': 'dynamic', 'factor': 3.0})
    hf.load_state_dict(ref.state_dict(), strict=False)
    for n in (L0 + 1, 2 * L0 + 7, 3 * L0):
        x = torch.randint(0, 97, (1, n))
        with torch.no_grad():
            a, b = m(x).logits, hf(x).logits
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-5), (n, (a - b).abs().max())
    with torch.no_grad():                                         # back in range: unchanged afterwards
        x = torch.randint(0, 97, (1, L0 - 5))
        assert torch.equal(m(x).logits, ref(x).logits)


def test_cached_style_matches_native_dynamic_formula():
    h = Holder(); orig = CachedQwen2RotaryEmbedding(16, L0, 10000.0)
    rep = apply_dynamic_ntk(h, 3.0)
    assert rep['styles'] == ['cached']
    v = torch.zeros(1, 2, 8, 16)
    for n in (5, L0):
        assert all(torch.equal(a, b) for a, b in zip(h.rot(v, seq_len=n), orig(v, seq_len=n)))
    hf = tiny({'rope_type': 'dynamic', 'factor': 3.0}); rot = hf.model.rotary_emb
    # same per-head dim (16) as the cached module: hidden 64 / 4 heads
    n = 2 * L0 + 3
    cos, sin = h.rot(v, seq_len=n)
    c2, s2 = rot(torch.zeros(1, n, 64), torch.arange(n)[None])
    assert torch.allclose(cos, c2[0], atol=1e-6) and torch.allclose(sin, s2[0], atol=1e-6)
    assert h.rot.max_seq_len_cached == L0                         # cache never extended with the unscaled base
    assert all(torch.equal(a, b) for a, b in zip(h.rot(v, seq_len=L0), orig(v, seq_len=L0)))


def test_refuses_existing_scaling():
    with pytest.raises(ValueError):
        apply_dynamic_ntk(tiny({'rope_type': 'dynamic', 'factor': 2.0}), 3.0)
