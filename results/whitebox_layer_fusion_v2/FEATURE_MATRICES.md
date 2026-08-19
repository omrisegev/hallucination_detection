# Feature and Matrix Contracts

## Saved candidate tensors

For candidate `i:j` with `T` generated tokens and architecture depth `L`, the recovered `layer-lens-v1` capture stores:

| Field | Shape | Meaning |
|---|---:|---|
| `lens_H` | `3 × L × T` | full-vocabulary logit-lens entropy |
| `lens_logp_tgt` | `3 × L × T` | log-probability of the realized target token |
| `lens_logp_top1` | `3 × L × T` | log-probability of the lens top-1 token |
| `lens_kl_final` | `3 × L × T` | KL divergence to final logits |
| `resid_norm` | `L × T` | residual-stream norm |
| `cov_eigs` | `L × 16` | top covariance eigenvalues; overflow-blocked on three cells |
| `hid_proj` | `L × 256` | seeded Gaussian JL projection of token-mean residual states |

The first tensor axis is fixed as `attn`, `mlp`, `resid`. Captures use L=32, 36, or 40. All enabled core inputs reduce the token axis with a preregistered arithmetic mean. `n_gen_tokens` is never a fusion input.

## Public matrix orientation

`FeatureMatrix.values` has shape **N candidates × M feature experts**. The solver receives its transpose **F = M experts × N candidates** after a single label-free standardization. Feature names, fixed groups, a protocol signature, and the final residual target-token NLL risk anchor are carried beside the matrix.

For every column `m`, standardization is performed within the cell:

`X[:,m] = (raw[:,m] - mean(raw[:,m])) / std(raw[:,m])`.

Only mechanically degenerate columns are removed. The kept column indices, means, scales, matrix bytes, and anchor bytes are hashed in fit diagnostics. Every method attached to a given contract sees the same standardized matrix and the same anchor. Evaluation labels never choose signs, layers, metrics, number of PCs, groups, or DUFS hyperparameters.

## Registered contracts

### `resid-core-L` — headline

One expert is constructed per residual layer. For layer `l`, the candidate-level token means are:

1. entropy `H_l` (higher risk);
2. negated target log-probability, i.e. target NLL (higher risk);
3. negated top-1 log-probability, i.e. top-1 surprisal (higher risk);
4. KL-to-final (higher risk).

Each non-degenerate component is standardized first, then the layer expert is their equal mean. Final residual KL is an exact zero and is omitted from the final layer's component mean. Matrix shape is `N × L`.

Architecture-relative layer variants:

- all layers: `0..L-1`;
- spaced eight: `round(linspace(0,L-1,8))` — L32 `[0,4,9,13,18,22,27,31]`, L36 `[0,5,10,15,20,25,30,35]`, L40 `[0,6,11,17,22,28,33,39]`;
- late eight: `L-8..L-1`.

Four hierarchical bands are `array_split(0..L-1, 4)`. Thus L32 uses four 8-layer bands and L40 four 10-layer bands.

### `lens-96` — richer secondary contract

The matrix uses the same four metrics at all three module positions over the architecture-relative spaced eight layers:

`3 modules × 4 metrics × 8 layers = 96 nominal columns`.

Exact final-residual KL is degenerate, so the usual realized shape is `N × 95`. Flat methods see all columns. Matched hierarchical methods first fuse within the twelve fixed `module × metric` groups and then fuse the twelve virtual experts.

### `trilens-entropy-3L`

This is the closest saved-data reconstruction of the TriLens feature map:

`[H_attn(layer 0..L-1), H_mlp(layer 0..L-1), H_resid(layer 0..L-1)]`, shape `N × 3L`.

TriLens describes a fixed token readout but its paper text does not specify whether that readout is last-token or token-mean. This benchmark freezes token-mean, labels the result an approximation, and compares flat and three-module hierarchical U-PCR/IU-PCR/DUFS-LIU. Its label-using L2 logistic probe is reported separately.

### `dola-kl-proxy-L`

Residual-stream token-mean `KL(layer || final)` over depth, nominal shape `N × L`; exact final KL is dropped, producing `N × (L-1)`. The literature detector compared by TriLens uses JSD, while the saved sidecar contains KL, so this is named a DoLa-KL proxy. DoLa itself is a decoding method, not this detector.

### HaloScope direct-projection proxy

The fixed middle architecture layer contributes the saved `N × 256` mean-token JL projection. After column centering, SVD gives singular values `σ_j` and right singular vectors `v_j`. With fixed `k=4`, direct membership is:

`ζ_i = (1/k) Σ_j σ_j <f_i, v_j>²`.

Its global direction is oriented only by the final-NLL anchor. The full HaloScope pipeline uses last-token full-dimensional embeddings, validation-selected hyperparameters, pseudo-labels, and a trained classifier; those unavailable/label-using stages are not claimed here.

### Raw output baselines

Shape `N × 4`:

- mean generation entropy;
- mean realized sampled-token NLL (the legacy cache name `token_spilled_energies` is not paper Spilled Energy);
- mean reconstructed Spilled Energy Eq. 8;
- minimum reconstructed Spilled Energy Eq. 8.

For token `t` whose sampled ID appears in saved raw top-K:

`raw_logit_t = raw_topk_logprob_t(sampled_id) + logsumexp_t`,

`DeltaE_t = raw_logit_t - logsumexp_(t+1)`.

Coverage is recorded per cell and is effectively 100% for eligible two-step tokens. Rows with no two-step delta (488 one-token CoQA generations) receive a label-free cell-median imputation. Pooling is over the full generated answer because exact-answer spans were not captured.

### INSIDE EigenScore appendix

Only CoQA/Llama-1 stores K=10 middle-layer last-token vectors of dimension 4096. For a question, the K embeddings are centered across dimensions; `Sigma = Z_centered Z_centered^T`; and the direct score is:

`(1/K) log det(Sigma + 0.001 I_K)`.

The question score is repeated on its ten candidate rows solely to fit the candidate-level audit format. This cell is protocol-rejected and excluded from every primary macro.

## Token-length sensitivity

The transparent token-length baseline is `log1p(n_gen_tokens)`. A separate fixed sensitivity regresses every `resid-core-L` column and the risk anchor on `[1, log1p(T)]` without labels and feeds the residual matrix to the three core solvers. It is not used to choose the primary result.

## Leakage boundary

Preparation writes 155 compressed label-free NPZ bundles. Fitting can open only those bundles; its APIs have no label-like argument. `FIT_COMPLETE.json` and `SCORE_FREEZE_MANIFEST.json` attest `labels_seen_during_fit=false` and hash every score file. Only then does evaluation reopen raw correctness labels and define `y_hallucination = 1 - label`. Reversing or permuting evaluation labels cannot change a score hash.
