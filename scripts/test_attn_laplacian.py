"""
test_attn_laplacian.py — CPU unit test for the LapEigvals attention-Laplacian
reducer (spectral_utils.model_utils.attn_laplacian_capture / _attn_lap_diag_stats).

Two layers of verification, both CPU-only:

  1. ALGEBRAIC (no model): synthetic row-stochastic lower-triangular attention.
     Checks the paper's identity eig(L) = diag(L) = d_ii - a_ii directly against a
     dense eigendecomposition of L = D - A, so the "diag is the spectrum" shortcut is
     proven, not assumed. Also checks the length-independence divisor (T-i), the
     NaN-padding when T < top_k, descending sort, and the diag-logmean stat.

  2. END-TO-END (tiny HF model, if transformers is importable): runs generate_full
     with capture_attention=True on hf-internal-testing/tiny-random-* and asserts the
     capture shapes ([L,H,top_k] / [L,H]), finiteness of the populated eigenvalues,
     and that attn_lap_meta.total_len == prompt_len + generated tokens.

Run: python scripts/test_attn_laplacian.py
Exit code nonzero on any failure.
"""
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch

from spectral_utils.model_utils import _attn_lap_diag_stats

FAILS = []


def check(cond, msg):
    print(f"  [{'ok  ' if cond else 'FAIL'}] {msg}")
    if not cond:
        FAILS.append(msg)


def random_causal_attention(H, T, seed=0):
    """[1, H, T, T] row-stochastic lower-triangular (a valid causal attention map)."""
    g = torch.Generator().manual_seed(seed)
    raw = torch.rand(H, T, T, generator=g)
    mask = torch.tril(torch.ones(T, T))
    raw = raw * mask
    raw = raw / raw.sum(dim=2, keepdim=True)     # each row sums to 1
    return raw.unsqueeze(0)


def test_algebraic():
    print("== 1. algebraic identity: eig(D-A) == diag(D-A) for causal A ==")
    H, T, K = 3, 12, 100
    att = random_causal_attention(H, T, seed=7)
    eig, dlm = (t.numpy() for t in _attn_lap_diag_stats(att, top_k=K))

    # Reference: build L = D - A per head with the paper's out-degree diagonal and
    # take a real eigendecomposition. Eigenvalues of a triangular matrix are its
    # diagonal, so the sorted real parts must match diag(L). We verify the shortcut in
    # float32 (tight — this is the actual correctness claim) and separately that the
    # stored value is float16(diag) (the reducer stores float16 by design, ~1e-3 ULP).
    a = att[0]                                          # [H, T, T]
    denom = (T - torch.arange(T, dtype=a.dtype))        # T..1
    for h in range(H):
        A = a[h]
        d = A.sum(dim=0) / denom                         # d_ii = colsum / (T-i)
        L = torch.diag(d) - A                            # lower-triangular
        dense_eig = torch.sort(torch.linalg.eigvals(L).real, descending=True).values
        diag_ref = torch.sort(d - torch.diagonal(A), descending=True).values
        got = eig[h][:T]                                 # first T are populated (float16)
        check(np.allclose(dense_eig.numpy(), diag_ref.numpy(), atol=1e-4),
              f"head {h}: dense eig(L) == diag(L) shortcut in float32 (max |Δ|="
              f"{np.max(np.abs(dense_eig.numpy() - diag_ref.numpy())):.2e})")
        check(np.allclose(diag_ref.to(torch.float16).numpy(), got, atol=1e-3),
              f"head {h}: stored float16 eigvals == float16(diag) (max |Δ|="
              f"{np.max(np.abs(diag_ref.to(torch.float16).numpy() - got)):.2e})")

    # NaN padding beyond T, descending order, diag-logmean sanity.
    check(np.isnan(eig[:, T:]).all(), f"columns >= T ({T}) are NaN-padded")
    valid = eig[:, :T]
    check(np.all(np.diff(valid, axis=1) <= 1e-6), "eigenvalues sorted descending")
    ref_dlm = torch.log(torch.diagonal(a, dim1=1, dim2=2).clamp_min(1e-12)).mean(dim=1)
    check(np.allclose(ref_dlm.numpy(), dlm, atol=1e-5), "attn_diag_logmean matches mean log(a_ii)")

    # T < top_k path (short trace).
    att_s = random_causal_attention(2, 4, seed=3)
    eig_s = _attn_lap_diag_stats(att_s, top_k=10)[0].numpy()
    check(eig_s.shape == (2, 10) and np.isnan(eig_s[:, 4:]).all() and np.isfinite(eig_s[:, :4]).all(),
          "short trace (T=4 < top_k=10): first 4 finite, rest NaN")


def test_end_to_end():
    print("\n== 2. end-to-end capture on a tiny HF model ==")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as e:
        print(f"  [skip] transformers unavailable ({type(e).__name__}) — algebraic test covers the reducer")
        return
    model_id = "hf-internal-testing/tiny-random-LlamaForCausalLM"
    try:
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
        mdl.eval()
    except Exception as e:
        print(f"  [skip] tiny model load failed ({type(e).__name__}: {e}) — needs network once")
        return

    from spectral_utils.model_utils import generate_full
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    r = generate_full(mdl, tok, "The capital of France is",
                      temperature=0.0, max_new_tokens=6,
                      logprob_top_k=0, capture_attention=True, attention_top_k=20)

    check("attn_lap_eigvals" in r and "attn_diag_logmean" in r, "capture keys present")
    eig = np.asarray(r["attn_lap_eigvals"])
    dlm = np.asarray(r["attn_diag_logmean"])
    L, H = mdl.config.num_hidden_layers, mdl.config.num_attention_heads
    check(eig.shape == (L, H, 20), f"attn_lap_eigvals shape {eig.shape} == ({L},{H},20)")
    check(dlm.shape == (L, H), f"attn_diag_logmean shape {dlm.shape} == ({L},{H})")
    check(np.isfinite(eig[~np.isnan(eig)]).all(), "populated eigenvalues finite (no inf)")
    meta = r["attn_lap_meta"]
    n_gen = len(r["gen_token_ids"])
    check(meta["total_len"] == meta["prompt_len"] + n_gen,
          f"meta total_len {meta['total_len']} == prompt {meta['prompt_len']} + gen {n_gen}")


def main():
    test_algebraic()
    test_end_to_end()
    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILURES'}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
