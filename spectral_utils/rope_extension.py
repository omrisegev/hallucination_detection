"""Dynamic NTK RoPE scaling applied in place to a loaded model (DeltaBench long-context adaptation, 2026-11-08).

Why: Qwen2.5-Math models (Qwen-PRM-7B/72B, PURE, and their standalone bases) have a 4,096-token context. 728 of
1,236 DeltaBench answers are longer (max 11,600 tokens), so those answers were counted context failures.

What: the same formula as transformers' rope_type="dynamic" (`_compute_dynamic_ntk_parameters`):
    base' = base * ((factor * L / L0) - (factor - 1)) ** (dim / (dim - 2)),   L = sequence length, L0 = 4,096
applied ONLY when L > L0. For L <= L0 every rotary module runs its original code path, so an answer that fits the
native context gets bit-identical outputs. This is an ADAPTATION of the released models: label every cell that uses it.

Two rotary styles exist here:
- "cached" (the trust_remote_code Qwen2.5-Math-PRM files: forward(x, seq_len), _set_cos_sin_cache). The original
  module would silently extend its cache with the unscaled base; the patch returns scaled cos/sin instead and never
  touches the cache, so the <= L0 path stays exactly the original one.
- "native" (transformers Qwen2RotaryEmbedding: forward(x, position_ids), inv_freq buffer). The patch swaps inv_freq for
  the scaled one during the call and restores it afterwards.
"""
from __future__ import annotations

import inspect

import torch


def scaled_inv_freq(base, dim, factor, seq_len, original_max, device=None):
    base_s = base * ((factor * seq_len / original_max) - (factor - 1)) ** (dim / (dim - 2))
    return 1.0 / (base_s ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))


def _patch_cached(mod, factor, original_max):
    orig_forward = mod.forward

    def forward(x, seq_len=None):
        if seq_len is None or seq_len <= original_max:
            return orig_forward(x, seq_len=seq_len)
        inv = scaled_inv_freq(mod.base, mod.dim, factor, seq_len, original_max, device=x.device)
        t = torch.arange(seq_len, device=x.device, dtype=torch.int64).type_as(inv)
        emb = torch.cat((torch.outer(t, inv),) * 2, dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)

    mod.forward = forward


def _patch_native(mod, factor, original_max, base):
    orig_forward = mod.forward
    dim = 2 * mod.inv_freq.shape[0]
    if getattr(mod, 'rope_type', 'default') != 'default':
        raise ValueError(f'native rotary module already has rope_type={mod.rope_type}; refusing to stack scalings')

    def forward(x, position_ids):
        seq_len = int(position_ids.max()) + 1
        if seq_len <= original_max:
            return orig_forward(x, position_ids)
        saved = mod.inv_freq
        mod.inv_freq = scaled_inv_freq(base, dim, factor, seq_len, original_max, device=saved.device).to(saved.dtype)
        try:
            return orig_forward(x, position_ids)
        finally:
            mod.inv_freq = saved

    mod.forward = forward


def rope_settings(cfg):
    """(rope_theta, existing scaling or None) for transformers 4.x (rope_theta/rope_scaling) and 5.x (rope_parameters)."""
    d = cfg.to_dict()
    params = d.get('rope_parameters') or {}
    theta = d.get('rope_theta', params.get('rope_theta'))
    if theta is None:
        raise ValueError('config has no rope_theta')
    scaling = d.get('rope_scaling') or ({k: v for k, v in params.items() if k != 'rope_theta'}
                                       if params.get('rope_type', 'default') != 'default' else None)
    if scaling and scaling.get('rope_type', scaling.get('type', 'default')) == 'default':
        scaling = None
    return float(theta), scaling


def apply_dynamic_ntk(model, factor, original_max=None):
    """Patch every rotary module of `model`. Returns a report dict; raises if no rotary module is found."""
    if factor is None or factor <= 1:
        raise ValueError('factor must be > 1')
    cfg = model.config
    original_max = int(original_max or cfg.max_position_embeddings)
    base, existing = rope_settings(cfg)
    if existing:
        raise ValueError(f'model already has RoPE scaling {existing}')
    styles = {}
    for name, mod in model.named_modules():
        if not type(mod).__name__.endswith('RotaryEmbedding'):
            continue
        params = list(inspect.signature(mod.forward).parameters)
        if hasattr(mod, '_set_cos_sin_cache') and 'seq_len' in params:
            if float(mod.base) != base: raise ValueError(f'{name}: base {mod.base} != rope_theta {base}')
            _patch_cached(mod, factor, original_max); styles[name] = 'cached'
        elif hasattr(mod, 'inv_freq') and 'position_ids' in params:
            ref = scaled_inv_freq(base, 2 * mod.inv_freq.shape[0], 1.0, original_max, original_max)
            if not torch.allclose(mod.inv_freq.float().cpu(), ref, rtol=1e-5, atol=0):
                raise ValueError(f'{name}: inv_freq does not match rope_theta {base} default RoPE')
            _patch_native(mod, factor, original_max, base); styles[name] = 'native'
        else:
            raise ValueError(f'{name}: unknown rotary style {type(mod).__name__} {params}')
    if not styles:
        raise ValueError('no rotary module found')
    kinds = sorted(set(styles.values()))
    return {'rope_extension': 'dynamic_ntk', 'factor': float(factor), 'original_max_position_embeddings': original_max,
            'extended_context': int(factor * original_max), 'rope_theta': base, 'n_rotary_modules': len(styles),
            'styles': kinds}
