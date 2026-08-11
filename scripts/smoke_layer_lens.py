#!/usr/bin/env python
"""CPU-only correctness checks for cluster/layer_lens.py — run BEFORE any GPU job.

Four of the six Step-163 pilot bugs were the kind a CPU fixture catches, and the
per-layer field has its own version of that failure mode: hooks attached to the wrong
submodule, or a lens that does not reproduce the model's own head, both produce a
plausible-looking field that is only wrong in ways nothing downstream can see.

The checks, on a tiny randomly-initialised Llama (no download, no network):

  1. RESIDUAL IDENTITY   x_l == x_{l-1} + a_l + m_l   for every layer.
     This is TriLens Eq. 1 and it is what proves ModuleTap is reading the isolated
     MHSA and FFN writes rather than, say, a normalised or pre-projection tensor.

  1b. THE hidden_states[L] TRAP (found by this file, 2026-08-11).
     HF applies the final norm BEFORE appending the last entry, so hidden_states[L]
     is Norm_final(x_L), not x_L, while entries 0..L-1 are raw pre-norm streams.
     Reading it as x_L applies the norm twice.  The lens check barely notices —
     RMSNorm is near-idempotent when its weights are ~1, as at random init, so the
     original bug showed up as a 3.6e-4 deviation — but on a trained model the norm
     weights are not 1 and the final-layer readout is genuinely wrong, corrupting the
     KL reference every other layer is measured against.  Both checks are kept
     because neither alone catches it.

  2. LENS FIDELITY       lens(x_L) == the model's own logits, on the RECONSTRUCTED
     x_L.  If the final-layer residual read through our norm+head does not reproduce
     out.logits, the lens is misconfigured and every layer's entropy is wrong.

  3. FIELD SHAPE + FINITENESS, and that lens_kl_final is ~0 at the last layer
     (KL of the reference against itself) and non-negative everywhere.

  4. ALIGNMENT           the field's token axis matches the generated-token slice
     used by backfill_views.forward_batch.

Usage:  python scripts/smoke_layer_lens.py
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "cluster"))

import numpy as np
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from layer_lens import (
    MODULES, QUANTITIES, ModuleTap, candidate_layer_field, make_hid_proj,
    resolve_stack,
)

TOL = 2e-4  # float32 accumulation over a few layers


def tiny_model(n_layers=4, hidden=64, vocab=256, heads=4):
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=vocab, hidden_size=hidden, intermediate_size=hidden * 2,
                      num_hidden_layers=n_layers, num_attention_heads=heads,
                      num_key_value_heads=heads, max_position_embeddings=128)
    return LlamaForCausalLM(cfg).eval().to(torch.float32)


def main():
    fails = []

    def check(name, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    mdl = tiny_model()
    layers, norm, head = resolve_stack(mdl)
    L = len(layers)
    plen, tgen = 7, 11
    ids = torch.randint(0, mdl.config.vocab_size, (1, plen + tgen))
    gen_ids = ids[0, plen:]

    print("1. residual identity  x_l == x_{l-1} + a_l + m_l")
    with ModuleTap(layers) as tap, torch.no_grad():
        out = mdl(input_ids=ids, attention_mask=torch.ones_like(ids),
                  use_cache=False, output_hidden_states=True)
        hs = out.hidden_states
        worst = 0.0
        for l in range(L - 1):  # hs[L] is normed — the identity stops at L-1
            recon = hs[l] + tap.attn[l] + tap.mlp[l]
            worst = max(worst, (recon - hs[l + 1]).abs().max().item())
        check("residual identity holds for layers 0..L-2", worst < TOL,
              f"max |x_l - (x_(l-1)+a_l+m_l)| = {worst:.2e}")

        print("1b. hidden_states[L] is Norm_final(x_L), not x_L")
        x_final = hs[L - 1] + tap.attn[L - 1] + tap.mlp[L - 1]
        d_normed = (norm(x_final) - hs[L]).abs().max().item()
        check("hs[L] == norm(reconstructed x_L)", d_normed < TOL,
              f"max |norm(x_L) - hs[L]| = {d_normed:.2e}")
        d_raw = (x_final - hs[L]).abs().max().item()
        check("hs[L] != raw x_L (the trap is real on this transformers version)",
              d_raw > TOL, f"max |x_L - hs[L]| = {d_raw:.2e}")

        print("2. lens fidelity  lens(reconstructed x_L) == model logits")
        dl = (head(norm(x_final)) - out.logits).abs().max().item()
        scale = out.logits.abs().max().item()
        check("final-layer lens reproduces the model head", dl / scale < 1e-3,
              f"max |lens(x_L) - logits| = {dl:.2e} (logit scale {scale:.2e})")

        print("2b. verify_residual_reconstruction agrees")
        from layer_lens import verify_residual_reconstruction
        try:
            dev = verify_residual_reconstruction(mdl, tap, hs, out.logits)
            check("runtime guard passes on a correct setup", True,
                  f"identity {dev['residual_identity_max_abs']:.2e}, "
                  f"lens {dev['lens_max_abs']:.2e}")
        except RuntimeError as e:
            check("runtime guard passes on a correct setup", False, str(e))

        print("3. field shape, finiteness, KL sanity")
        proj = make_hid_proj(mdl.config.hidden_size, torch.device("cpu"), dim=32)
        field = candidate_layer_field(mdl, tap, hs, gen_ids, plen, tgen, proj,
                                      cov_eigs_r=8, batch_index=0)

    for q in QUANTITIES:
        check(f"{q} shape == (3, L, T)",
              field[q].shape == (len(MODULES), L, tgen), str(field[q].shape))
        check(f"{q} finite", bool(np.isfinite(field[q]).all()))
    check("resid_norm shape == (L, T)", field["resid_norm"].shape == (L, tgen))
    check("cov_eigs shape == (L, r)", field["cov_eigs"].shape == (L, 8))
    check("hid_proj shape == (L, d)", field["hid_proj"].shape == (L, 32))

    kl = field["lens_kl_final"].astype(np.float64)
    check("KL >= 0 everywhere (allowing fp16 noise)", kl.min() > -1e-2,
          f"min = {kl.min():.2e}")
    resid_last = kl[MODULES.index("resid"), L - 1]
    check("KL(resid_L || resid_L) ~ 0", np.abs(resid_last).max() < 1e-2,
          f"max |KL| at final layer = {np.abs(resid_last).max():.2e}")
    ent = field["lens_H"].astype(np.float64)
    check("entropy within [0, log V]", ent.min() >= 0 and
          ent.max() <= np.log(mdl.config.vocab_size) + 1e-2,
          f"[{ent.min():.3f}, {ent.max():.3f}], log V = "
          f"{np.log(mdl.config.vocab_size):.3f}")

    print("4. alignment with backfill_views.forward_batch")
    from backfill_views import forward_batch
    item = {"prompt_ids": ids[0, :plen].tolist(), "gen_ids": gen_ids.tolist()}
    raw = forward_batch(mdl, [item])[0]
    lp = raw.float().log_softmax(-1)
    ref_tgt = lp[torch.arange(tgen), gen_ids].numpy()
    ours = field["lens_logp_tgt"][MODULES.index("resid"), L - 1].astype(np.float64)
    d = np.abs(ref_tgt - ours).max()
    check("final-layer lens_logp_tgt == forward_batch log-prob of the same token",
          d < 5e-3, f"max |d| = {d:.2e} (fp16 storage floor ~1e-3)")

    print()
    if fails:
        print(f"SMOKE FAILED: {len(fails)} check(s) — {fails}")
        return 1
    print("SMOKE PASSED — all checks green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
