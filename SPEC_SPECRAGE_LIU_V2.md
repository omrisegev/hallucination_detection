# Cross-view Agreement SpecRaGE–LIU (CA-SpecRaGE–LIU) v2

**Status:** synthetic mechanism baseline passed; real-artifact calibration is
implemented but remains development evidence until grouped evaluation completes.

**Relationship to v1:** v1 remains the negative result. It attempted to infer
semantic reliability from the unmodified SpecRaGE Rayleigh objective and passed
an alpha-weighted raw graph into IU-PCR. Debugging showed both that alpha stayed
uninformative and that the original synthetic world had essentially no IU-PCR
headroom. V2 repairs and tests those two mechanisms separately.

## 1. Scientific claim

CA-SpecRaGE–LIU tests whether label-free agreement among three or more feature
families can identify a locally unreliable family and build a sample graph that
suppresses dependent-error directions in IU-PCR.

It assumes that useful geometry is shared by a majority or plurality of views.
It cannot identify a nuisance shared coherently by every view, and with exactly
two disagreeing views it cannot determine which side is correct without another
asymmetry. The implementation returns uniform agreement targets in that
unidentifiable two-view case.

## 2. Two-link development contract

The method is not evaluated end-to-end until both links pass independently:

1. **Link A — reliability/representation:** on planted conditional corruption,
   learned weights must identify clean views above chance and remain reproducible.
2. **Link B — graph actuation:** before training a graph learner, an oracle graph
   must materially change projected IU-PCR roughness and score ranks. A world in
   which oracle and uniform scores are nearly identical is rejected.

This gate prevents a failed learner from being diagnosed in a benchmark where
even perfect geometry could not improve the fusion head.

## 3. Label-free cross-view agreement

For each provenance view (v), form a sparse affinity (W^{(v)}), row-normalize
it to a transition matrix (P^{(v)}), and use a softened diffusion profile

\[
H^{(v)}=\tfrac12 P^{(v)}+\tfrac12(P^{(v)})^2.
\]

After row-wise L2 normalization, the agreement of view (v) for sample (i) is

\[
a_i^{(v)}=\frac{1}{V-1}\sum_{u\ne v}
\left\langle H_{i:}^{(v)},H_{i:}^{(u)}\right\rangle .
\]

For (V\ge3), the label-free reliability target is

\[
\pi_i=\operatorname{softmax}
\left(\frac{a_i-\bar a_i}{\tau_a}\right).
\]

The SpecRaGE fusion network still predicts sample-specific simplex weights

\[
\alpha_i=\operatorname{softmax}(q(X_i)/\tau),
\qquad
Y_i=\sum_v\alpha_i^{(v)}Y_i^{(v)}.
\]

Training minimizes

\[
\mathcal L=\mathcal L_{\mathrm{SpecRaGE}}
+\beta\,\mathrm{KL}(\pi\Vert\alpha)
+\gamma\,\mathcal L_{\mathrm{mass}}.
\]

The mass term compares the observed weighted affinity mass

\[
\sum_{ij}W_{ij}^{(v)}\alpha_i^{(v)}\alpha_j^{(v)}
\]

with the mass expected from the marginal mean weight. It prevents the learner
from reducing the Rayleigh loss merely by assigning adjacent endpoints to
different near-one-hot views and deleting their edges.

No correctness label, target latent, AUROC, or target graph enters these
quantities.

## 4. Two candidate graph interfaces

V2 keeps the two scientifically distinct interfaces visible:

1. **Embedding interface (paper output):** build a rotation-invariant k-NN graph
   separately from every seed's fused embedding (Y), then average the graphs.
2. **Agreement-weight interface (derived extension):**

\[
W_{\alpha}=\frac1V\sum_v
\operatorname{diag}(\alpha^{(v)})W^{(v)}
\operatorname{diag}(\alpha^{(v)}).
\]

The embedding interface is closest to unchanged SpecRaGE. The agreement-weight
interface is the new CA-SpecRaGE contribution and must be named as a derived
method. They are calibrated and reported separately, together with an
end-to-end uniformly fused embedding control.

Either graph supplies the existing LIU equation

\[
w_\lambda=U\left[U^T(C+\lambda\bar R)U\right]^{-1}U^T\hat\rho,
\qquad R=FLF^T/n.
\]

At (lambda=0), ordinary IU-PCR remains an exact algebraic identity.

## 5. Optimization repair and diagnostics

The v1 adaptation performed one optimizer update per epoch. V2 iterates over
every pair of independently shuffled gradient and orthogonalization batches,
matching the released trainer's structure. It uses the released LeakyReLU
hidden activation and a fixed final checkpoint rather than selecting a noisy
small-validation checkpoint.

The released QR operation is ill-conditioned at this sample size. V2 registers
an SVD singular-value floor of (10^{-3}) of the leading singular value and
records the raw condition number, transform behavior, clipping fraction,
gradient norm, optimizer-update count, Rayleigh loss, agreement loss, and edge
mass loss. This is an explicit numerical stabilization, not a claim that the
original QR path is healthy.

## 6. Frozen synthetic development settings

- output dimension: 2;
- graph neighbours: 15;
- fusion temperature: 1;
- agreement temperature: 0.08;
- agreement coefficient: 2;
- edge-mass coefficient: 0.1;
- learning rate: (10^{-2});
- batch size: 128;
- epochs: 60;
- model seeds: 11 and 23;
- candidate (lambda): 0.3, 1, 3, 10, 30, 100;
- calibration worlds: seeds 5000–5002;
- held-out worlds: seeds 6000–6003.

Labels choose only the graph interface and (lambda) across calibration worlds.
The best calibration mean occurred at the saturating grid boundary, so the
one-standard-error rule selected the smaller (lambda=10). Held-out seeds were
then opened once.

## 7. Synthetic result

On the Link-A conditional-corruption worlds, unchanged SpecRaGE alpha achieved
0.663 clean-view AUROC, while CA-SpecRaGE achieved 0.930. Its alpha entropy was
0.985, so the improvement was not caused by one-view collapse.

On four held-out Link-B dependent-error worlds:

| method | mean AUROC | change vs IU-PCR | wins |
|---|---:|---:|---:|
| IU-PCR | 0.7262 | 0.000 pp | — |
| DUFS-LIU (`lambda=0.1`) | 0.7260 | -0.029 pp | 3/4 |
| raw-uniform LIU (`lambda=10`) | 0.7342 | +0.798 pp | 4/4 |
| uniform SpecRaGE-(Y) LIU | 0.7392 | +1.300 pp | 4/4 |
| **CA-SpecRaGE agreement graph** | **0.7411** | **+1.484 pp** | **4/4** |
| oracle target graph | 0.7431 | +1.689 pp | 4/4 |

Thus the corrected method captures most of the planted oracle headroom and
clearly separates from DUFS-LIU in the world designed to represent
family-specific dependent errors. This is mechanism evidence, not yet a real
hallucination-detection performance claim.

## 8. Real-data evaluation

The real runner fits CA-SpecRaGE independently inside every cell, evaluates the
agreement graph and embedding graph separately, and fits the uniform embedding
control end-to-end. Dataset/model families—not random samples—are the
calibration units. Labels are joined only after scores are frozen and selection
uses leave-one-family-out cross-fitting plus a one-standard-error rule.

```bash
python scripts/specrage_upcr_real.py --stage smoke
python scripts/specrage_upcr_real.py --stage development
```

The real method must not be promoted unless it beats deployed U-PCR and frozen
DUFS-LIU under the registered family-macro, orientation, numerical, graph, and
lower-tail guards. Synthetic success alone does not satisfy that condition.
