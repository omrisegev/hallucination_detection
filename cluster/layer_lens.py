#!/usr/bin/env python
"""Per-layer, per-module logit-lens telemetry — the white-box view source.

WHAT THIS IS FOR
----------------
Everything the repository fuses today is derived from ONE trajectory: a scalar per
generated token.  This module produces a second, orthogonal trajectory — a scalar per
generated token *per layer per module* — so that the label-free fusion family
(``upcr_fit``, ``laplacian_iu_fit``) can be run over depth instead of, or alongside,
time.

Motivated by TriLens (arXiv:2606.01033, Yang et al., May 2026), which established that
per-layer logit-lens entropy at the MHSA output, the FFN output and the residual stream
carries hallucination signal and that the three module readouts are *independently*
informative.  TriLens combines them with a SUPERVISED probe (MLP / L2 logistic
regression, 80/20 train-test split); the words "unsupervised" and "ensemble" do not
occur in that paper.  Our interest is exactly the part it leaves open — combining the
3L readouts with no labels — so this module deliberately computes the raw field and
commits to NO pooling rule, NO view definition and NO layer selection.  All three are
CPU-side decisions made later, against saved data.

NOTHING IS GENERATED.  This is a teacher-forced measurement over token ids that are
already in the cache, so the resulting views align row-for-row with every published
number on the same cell and can be fused with the existing token-trace views.

WHAT IS SAVED (per candidate)
-----------------------------
The 2D field, for each module m in (attn, mlp, resid), layer l, generated token t:

  ``lens_H[m, l, t]``          Shannon entropy of the lens distribution, full vocabulary
  ``lens_logp_tgt[m, l, t]``   lens log-prob of the token actually generated  (a
                               depth-resolved spilled energy)
  ``lens_logp_top1[m, l, t]``  max lens log-prob                             (commitment)
  ``lens_kl_final[m, l, t]``   KL(lens_l || lens_final)                      (DoLa contrast)

plus residual-stream geometry that the lens throws away, so that the subspace family
(HaloScope, INSIDE/EigenScore, effective rank) is reachable on CPU without a second
GPU pass:

  ``resid_norm[l, t]``   ||x_l,t||
  ``cov_eigs[l, :r]``    top-r eigenvalues of the centred token-covariance of x_l
  ``hid_proj[l, :d]``    token-mean of x_l under a FIXED seeded Gaussian projection

The projection seed is stored in the output.  It must never change: the projection is
shared across candidates and cells so that corpus-level subspace methods see a
consistent basis.

Storage is float16 and runs ~100-120 MB for a 500-candidate reasoning cell at L=32.

WHY FULL VOCABULARY
-------------------
The entropy is computed over the whole vocabulary, matching TriLens, not over a top-K
renormalisation.  The cached ``token_entropies`` in this repo ARE top-15 renormalised;
the two are different statistics and must not be compared to each other.  Gate B (in
``backfill_views``) is what validates the prompt reconstruction, and it uses the
top-15 form on the final layer, which is the quantity that was actually saved.
"""

import numpy as np
import torch

# Modules read at every layer, in the stored axis order.
MODULES = ("attn", "mlp", "resid")
# Quantities stored per (module, layer, token), in the stored axis order.
QUANTITIES = ("lens_H", "lens_logp_tgt", "lens_logp_top1", "lens_kl_final")

HID_PROJ_DIM = 256      # Johnson-Lindenstrauss target for the pooled hidden state
COV_EIGS_R = 16         # top eigenvalues of the per-layer token covariance
HID_PROJ_SEED = 20260811
LENS_TOKEN_CHUNK = 256  # tokens per lens matmul; caps peak memory at chunk x V floats


def resolve_stack(mdl):
    """Return (layers, final_norm, lm_head) for a Llama/Qwen-style causal LM.

    Kept explicit rather than duck-typed: a silent mismatch here produces a
    plausible-looking field measured off the wrong tensors, which is exactly the
    class of bug that is invisible downstream.
    """
    base = getattr(mdl, "model", None)
    if base is None or not hasattr(base, "layers"):
        raise RuntimeError(
            f"{type(mdl).__name__} has no .model.layers — unsupported architecture "
            "for the logit lens; add it explicitly rather than guessing")
    norm = getattr(base, "norm", None)
    if norm is None:
        raise RuntimeError("no final norm at model.model.norm")
    head = getattr(mdl, "lm_head", None)
    if head is None:
        raise RuntimeError("no unembedding at model.lm_head")
    return base.layers, norm, head


def make_hid_proj(hidden_size, device, dtype=torch.float32, dim=HID_PROJ_DIM,
                  seed=HID_PROJ_SEED):
    """Fixed seeded Gaussian projection, shared across every candidate and cell."""
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    m = torch.randn(hidden_size, dim, generator=g, dtype=torch.float32)
    m /= np.sqrt(dim)
    return m.to(device=device, dtype=dtype)


class ModuleTap:
    """Forward hooks capturing the isolated MHSA and FFN writes at every layer.

    ``output_hidden_states=True`` yields only the composed residual stream x_l.  The
    separate a_l and m_l writes — the ones TriLens found independently informative —
    are only reachable by hooking the submodules.
    """

    def __init__(self, layers):
        self.layers = layers
        self.attn = {}
        self.mlp = {}
        self._handles = []

    def __enter__(self):
        def mk(store, idx):
            def hook(_mod, _inp, out):
                store[idx] = (out[0] if isinstance(out, tuple) else out).detach()
            return hook
        for i, layer in enumerate(self.layers):
            self._handles.append(layer.self_attn.register_forward_hook(mk(self.attn, i)))
            self._handles.append(layer.mlp.register_forward_hook(mk(self.mlp, i)))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self.attn.clear()
        self.mlp.clear()
        return False

    def clear(self):
        self.attn.clear()
        self.mlp.clear()


def _lens_stats(z, norm, head, gen_ids, ref_logprobs=None, chunk=LENS_TOKEN_CHUNK):
    """Lens statistics for one (module, layer) activation slice.

    ``z`` is [T, d] — the activation at the generated-token positions.  Returns
    (H, logp_tgt, logp_top1, kl_to_ref) as float32 CPU tensors of length T, plus the
    full log-prob matrix when ``ref_logprobs`` is None (the caller keeps the final
    layer's as the KL reference and discards it afterwards).

    Chunked over tokens so peak memory stays chunk x V rather than T x V.
    """
    T = z.shape[0]
    H = torch.empty(T, dtype=torch.float32)
    lp_tgt = torch.empty(T, dtype=torch.float32)
    lp_top1 = torch.empty(T, dtype=torch.float32)
    kl = torch.empty(T, dtype=torch.float32) if ref_logprobs is not None else None
    keep = [] if ref_logprobs is None else None

    for s in range(0, T, chunk):
        e = min(s + chunk, T)
        logits = head(norm(z[s:e])).float()
        lp = logits.log_softmax(dim=-1)
        p = lp.exp()
        H[s:e] = (-(p * lp).sum(dim=-1)).cpu()
        lp_tgt[s:e] = lp[torch.arange(e - s, device=lp.device), gen_ids[s:e]].cpu()
        lp_top1[s:e] = lp.max(dim=-1).values.cpu()
        if ref_logprobs is not None:
            ref = ref_logprobs[s:e].to(lp.device)
            # KL(lens_l || lens_final), the direction DoLa contrasts on.
            kl[s:e] = (p * (lp - ref)).sum(dim=-1).cpu()
        else:
            keep.append(lp.cpu())
        del logits, lp, p
    return H, lp_tgt, lp_top1, kl, (torch.cat(keep) if keep is not None else None)


def candidate_layer_field(mdl, tap, hidden_states, gen_ids, plen, tgen, hid_proj,
                          cov_eigs_r=COV_EIGS_R, chunk=LENS_TOKEN_CHUNK,
                          batch_index=0):
    """Compute the full per-layer field for ONE candidate in a forward batch.

    ``hidden_states`` is the tuple from ``output_hidden_states=True``; entry l+1 is the
    residual stream after layer l.  Activation slice [plen-1 : plen-1+tgen] holds the
    positions whose next-token prediction IS generated token j — the same alignment
    ``backfill_views.forward_batch`` uses for the final-layer logits.

    Returns a dict of float16 numpy arrays.
    """
    layers, norm, head = resolve_stack(mdl)
    L = len(layers)
    lo, hi = plen - 1, plen - 1 + tgen

    field = {q: np.empty((len(MODULES), L, tgen), dtype=np.float16) for q in QUANTITIES}
    resid_norm = np.empty((L, tgen), dtype=np.float16)
    cov_eigs = np.zeros((L, cov_eigs_r), dtype=np.float16)
    proj = np.empty((L, hid_proj.shape[1]), dtype=np.float16)

    # ``hidden_states[L]`` IS NOT x_L.  HF applies the final norm before appending the
    # last entry, so hidden_states[L] == Norm_final(x_L) while entries 0..L-1 are the
    # raw pre-norm streams.  Reading it as x_L applies the norm twice — which the lens
    # barely notices (RMSNorm is near-idempotent when its weights are ~1, as at random
    # init) but which is genuinely wrong on a trained model, and it would corrupt the
    # KL reference that every other layer is measured against.  So the residual stream
    # is reconstructed from the taps instead, via the identity x_l = x_{l-1}+a_l+m_l,
    # and hidden_states[L] is never read.  scripts/smoke_layer_lens.py locks this down.
    resid = []
    for l in range(L):
        resid.append(hidden_states[l][batch_index, lo:hi]
                     + tap.attn[l][batch_index, lo:hi]
                     + tap.mlp[l][batch_index, lo:hi])

    # The lens at the final layer's residual stream is the KL reference for every other
    # readout, so it is computed first and its log-probs held for one pass.
    _, _, _, _, ref_lp = _lens_stats(resid[L - 1], norm, head, gen_ids, None, chunk)

    for l in range(L):
        acts = {
            "attn": tap.attn[l][batch_index, lo:hi],
            "mlp": tap.mlp[l][batch_index, lo:hi],
            "resid": resid[l],
        }
        for mi, m in enumerate(MODULES):
            z = acts[m]
            H, lp_tgt, lp_top1, kl, _ = _lens_stats(z, norm, head, gen_ids, ref_lp, chunk)
            field["lens_H"][mi, l] = H.numpy().astype(np.float16)
            field["lens_logp_tgt"][mi, l] = lp_tgt.numpy().astype(np.float16)
            field["lens_logp_top1"][mi, l] = lp_top1.numpy().astype(np.float16)
            field["lens_kl_final"][mi, l] = kl.numpy().astype(np.float16)

        # Residual-stream geometry: what the lens discards.
        x = acts["resid"].float()
        resid_norm[l] = x.norm(dim=-1).cpu().numpy().astype(np.float16)
        proj[l] = (x.mean(dim=0) @ hid_proj.float()).cpu().numpy().astype(np.float16)
        if tgen >= 2:
            xc = x - x.mean(dim=0, keepdim=True)
            # Gram spectrum == covariance spectrum up to the 1/(T-1) factor, and T is
            # far smaller than d here, so the T x T form is the cheap one.
            gram = (xc @ xc.T) / max(tgen - 1, 1)
            ev = torch.linalg.eigvalsh(gram.double()).flip(0)[:cov_eigs_r]
            cov_eigs[l, :ev.numel()] = ev.cpu().numpy().astype(np.float16)

    del ref_lp
    out = {q: field[q] for q in QUANTITIES}
    out["resid_norm"] = resid_norm
    out["cov_eigs"] = cov_eigs
    out["hid_proj"] = proj
    return out


def verify_residual_reconstruction(mdl, tap, hidden_states, logits, tol=5e-2):
    """One-shot architecture check, run on the first candidate of every cell.

    Confirms on the live model+dtype that (a) the tapped writes reconstruct the
    residual stream, and (b) reading the reconstructed x_L through norm+head
    reproduces the model's own logits.  Cheap insurance against a transformers
    version that changes what the submodules return: the field would still be
    finite and plausible, just measured off the wrong tensors.

    Returns a dict of measured deviations; raises RuntimeError on mismatch.
    """
    layers, norm, head = resolve_stack(mdl)
    L = len(layers)
    worst_id = 0.0
    for l in range(L - 1):  # hidden_states[L] is normed, so the identity stops at L-1
        recon = hidden_states[l] + tap.attn[l] + tap.mlp[l]
        worst_id = max(worst_id, (recon - hidden_states[l + 1]).abs().max().item())
    x_final = hidden_states[L - 1] + tap.attn[L - 1] + tap.mlp[L - 1]
    lens_dev = (head(norm(x_final)) - logits).abs().max().item()
    scale = logits.abs().max().item() or 1.0
    if worst_id > tol:
        raise RuntimeError(
            f"residual identity violated (max |x_l - (x_(l-1)+a_l+m_l)| = {worst_id:.3e} "
            f"> {tol:.0e}) — ModuleTap is not reading the isolated MHSA/FFN writes on "
            f"this architecture; do not trust the field")
    if lens_dev / scale > tol:
        raise RuntimeError(
            f"logit lens does not reproduce the model head (max |d| = {lens_dev:.3e}, "
            f"logit scale {scale:.3e}) — norm/unembedding mismatch")
    return {"residual_identity_max_abs": worst_id, "lens_max_abs": lens_dev,
            "logit_scale": scale}


def forward_batch_layers(mdl, items):
    """Teacher-forced forward capturing hidden states AND the isolated module writes.

    Mirrors ``backfill_views.forward_batch`` exactly — same right padding, same
    causal-LM alignment — but returns the full stack instead of the final logits.
    Yields (item, tap, hidden_states, batch_index) so the caller reduces one candidate
    at a time and the big tensors are freed on exit.
    """
    dev = mdl.device
    seqs = [it["prompt_ids"] + it["gen_ids"] for it in items]
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    attn = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, :len(s)] = 1
    layers, _, _ = resolve_stack(mdl)
    with ModuleTap(layers) as tap, torch.no_grad():
        out = mdl(input_ids=input_ids.to(dev), attention_mask=attn.to(dev),
                  use_cache=False, output_hidden_states=True)
        yield out, tap


__all__ = [
    "MODULES", "QUANTITIES", "HID_PROJ_DIM", "COV_EIGS_R", "HID_PROJ_SEED",
    "resolve_stack", "make_hid_proj", "ModuleTap", "verify_residual_reconstruction",
    "candidate_layer_field", "forward_batch_layers",
]
