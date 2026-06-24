# Fusion Pipeline — Entry Points & Flowchart

## Entry Points Summary

| Entry point | Line | Labels needed? | Binarizes? | Groups? | Purpose |
|-------------|------|---------------|-----------|---------|---------|
| `lsml_continuous_pipeline` | 563 | No — uses `FEATURE_SIGNS` | No | Yes | **Current production** |
| `sml_unsupervised` | 588 | No | Yes (median) | Yes | Paper-faithful, zero supervision |
| `best_nadler_on` | 672 | Yes — for sign + search | Optional | No (flat SML) | Historical supervised search that produced `GOOD_FEATURES` |
| `best_nadler_pseudo_label` | 822 | No (wraps `best_nadler_on`) | Optional | No | Zero-label variant of the above |

---

## Flowchart

```mermaid
flowchart TD
    classDef ep     fill:#2d6a4f,color:#fff,stroke:#1b4332,font-weight:bold
    classDef pre    fill:#457b9d,color:#fff,stroke:#1d3557
    classDef core   fill:#1d3557,color:#fff,stroke:#0a1929
    classDef branch fill:#e76f51,color:#fff,stroke:#c15b3c,font-weight:bold
    classDef flat   fill:#6d6875,color:#fff,stroke:#4a4353
    classDef out    fill:#264653,color:#fff,stroke:#1a2e38

    %% ── Entry points ──────────────────────────────────────────
    EP1("★ lsml_continuous_pipeline\n:563\nProduction"):::ep
    EP3("sml_unsupervised\n:588\nPaper-aligned"):::ep
    EP2("best_nadler_on\n:672\nSupervised search"):::ep
    EP4("best_nadler_pseudo_label\n:822\nZero-label"):::ep

    %% ── Preprocessing ─────────────────────────────────────────
    PRE1["orient: arr × sign\nzscore()"]:::pre
    PRE3["median binarize\narr → ±1  (no sign orient)"]:::pre
    PRE2["boot_auc(labels, feat)\n→ sign per feature\nzscore(arr × sign)\nSpearman ρ filter"]:::pre
    PSEUDO["seed majority vote\n→ pseudo_labels"]:::pre

    EP1 --> PRE1
    EP3 --> PRE3
    EP2 --> PRE2
    EP4 --> PSEUDO --> EP2

    %% ── Flat SML path (best_nadler_on) ────────────────────────
    subgraph FLAT["Flat SML — exhaustive subset search"]
        direction TB
        SF["sml_fuse() :134\nleading eigvec of R_off, |·|\n(binarize=True)"]:::flat
        NF["nadler_fuse() :180\nM-matrix C⁻¹ reweight\n(binarize=False, default)"]:::flat
        AVG["simple_average_fusion() :222\nequal weights — baseline"]:::flat
    end

    PRE2 -->|"binarize=True"| SF
    PRE2 -->|"binarize=False ← default"| NF
    PRE2 -.->|"also runs for Lift report"| AVG

    FLAT --> BAUC["boot_auc(labels, fused)\nbest subset wins\n→ GOOD_FEATURES"]:::out

    %% ── L-SML path ────────────────────────────────────────────
    LC["lsml_continuous() :500"]:::core
    LF["lsml_fuse() :436"]:::core

    PRE1 --> LC
    PRE3 --> LF

    LC --> DDG
    LF --> DDG

    DDG["detect_dependent_groups() :375\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n① R = cov(X)  — k×k covariance\n② s_ij = Σ |r_ij·r_kl − r_il·r_kj|  Eq.15  :275\n③ SpectralClustering on s  :296\n④ residual(R,c) Eq.14 :343 → best K\n   alt: eigengap heuristic :359"]:::core

    DDG -->|"K groups, assignment c"| WG

    subgraph WG["Within each group g"]
        SS["sml_fuse_signed() :241\nleading eigvec of R_off\nAssumption iii sign resolve:\nif Σ(v>0) < k/2 → flip v"]:::core
    end

    %% ── The divergence ────────────────────────────────────────
    SS -->|"lsml_fuse path"| BIN["np.sign(score)\n→ ±1 virtual ξ_g\n(loses continuous signal)"]:::branch
    SS -->|"lsml_continuous path"| CONT["score as-is\n→ continuous virtual ξ_g\n(+4.9pp, no Lemma 1 guarantee)"]:::branch

    %% ── Cross-group ───────────────────────────────────────────
    BIN  --> CG["sml_fuse_signed() :241\ncross-group on ξ_1 … ξ_K"]:::core
    CONT --> CG

    CG --> OUT["fused_scores  +  meta_dict\n{ K, c, group_weights,\n  cross_weights,\n  virtual_classifiers }"]:::out
```

---

## The One Divergence Point That Matters

The entire binary vs. continuous difference collapses to a single line:

```python
# lsml_fuse  (binary, paper-faithful)  — fusion_utils.py:475
xi_g = np.sign(score)   # throws away magnitude

# lsml_continuous  (project invention, +4.9pp)  — fusion_utils.py:539
xi_g = score            # keeps magnitude
```

Everything above and below that line — group detection, `sml_fuse_signed`, the cross-group step, the output `meta_dict` — is shared code.

---

## Three Paths in Plain English

**Path A — Production** (`lsml_continuous_pipeline`):
```
feats_dict + FEATURE_SIGNS
  → orient + zscore  (pre-oriented, no labels)
  → lsml_continuous → detect_dependent_groups → within sml_fuse_signed
  → keep score continuous (no np.sign)
  → cross-group sml_fuse_signed
  → fused_scores
```

**Path B — Paper-aligned** (`sml_unsupervised`):
```
feats_dict
  → median binarize to ±1  (no sign orient — Assumption iii handles it)
  → lsml_fuse → detect_dependent_groups → within sml_fuse_signed
  → np.sign(score) → ±1 virtual classifier
  → cross-group sml_fuse_signed
  → fused_scores
```

**Path C — Historical supervised** (`best_nadler_on`):
```
feats_dict + labels
  → boot_auc → sign per feature
  → zscore + Spearman ρ filter
  → exhaustive subset search (size 2..4)
     each subset: sml_fuse OR nadler_fuse → boot_auc(labels, fused)
  → best (subset, auc)   ← this is how GOOD_FEATURES was found
```
