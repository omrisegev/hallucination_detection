# Every family we have tried, on the 25 in-scope cells

Best method that picks its own subset: **U-PCR + rho-derived polarities = 0.7551**, against the hand-curated GOOD_6 bar of 0.7594 — a gap of 0.43pp that nothing has closed.

| family | method | macro AUROC | QA | math | cells | picks its own subset? | note |
|---|---|---|---|---|---|---|---|
| fixed hand-curated subsets — THE BAR, NOT METHODS | LOCO_5 (sweep consensus) | **0.7705** | 0.7437 | 0.7866 | 24 | no — fixed | hand-curated, 24/25 cells |
| fixed hand-curated subsets — THE BAR, NOT METHODS | GOOD_6 | **0.7594** | 0.7274 | 0.7807 | 25 | no — fixed | hand-curated — the standing bar |
| U-PCR (Step 204) | U-PCR + rho-derived polarities | **0.7551** | 0.0000 | 0.0000 | 25 | yes | drops the 42 hand signs, keeps the anchor bit |
| U-PCR (Step 204) | U-PCR best of 64 configurations | **0.7533** | 0.0000 | 0.0000 | 25 | yes | winner's curse — not a result, shown as a ceiling |
| L-SML + automatic feature selection | pseudo-label gated DUFS — selector of record | **0.7524** | 0.7091 | 0.7813 | 25 | yes | seeded from GOOD_6 (answer keys), label-free at runtime |
| fixed hand-curated subsets — THE BAR, NOT METHODS | GOOD_5 | **0.7519** | 0.7210 | 0.7725 | 25 | no — fixed | hand-curated |
| L-SML + automatic feature selection | DUFS, parameter-free loss (paper Eq. 7) | **0.7507** | 0.7089 | 0.7787 | 25 | yes |  |
| L-SML + automatic feature selection | DUFS gated Laplacian | **0.7502** | 0.7087 | 0.7778 | 25 | yes |  |
| residual-guided (leave-one-out / greedy on the fit) | route between fusions, subset fixed to GOOD_5 | **0.7494** | 0.7154 | 0.7721 | 25 | no — fixed | fixed subset — fusion choice only |
| L-SML + automatic feature selection | GroupFS trace selector | **0.7481** | 0.7072 | 0.7754 | 25 | yes |  |
| L-SML + automatic feature selection | same, WITHOUT the GOOD_6 seed | **0.7478** | 0.7008 | 0.7791 | 25 | yes | fully unseeded |
| L-SML + automatic feature selection | continuous L-SML, ALL 30 views | **0.7452** | 0.0000 | 0.0000 | 25 | yes | no selection at all — the honest no-subset baseline |
| U-PCR (Step 204) | U-PCR, hand polarities | **0.7405** | 0.0000 | 0.0000 | 25 | yes |  |
| U-PCR (Step 204) | U-PCR legacy configuration | **0.7392** | 0.0000 | 0.0000 | 25 | yes | what fusion_utils.upcr_fuse does today |
| L-SML + automatic feature selection | concrete autoencoder, K=3 | **0.7388** | 0.6964 | 0.7671 | 25 | yes |  |
| L-SML + automatic feature selection | mRMR, adaptive size | **0.7379** | 0.6991 | 0.7638 | 25 | yes |  |
| L-SML + selection, anchor step ablated | a7 iterative consensus, anchor step ON | **0.7378** | 0.7013 | 0.7621 | 25 | yes | picks its own subset AND its own K |
| L-SML + automatic feature selection | RANDOM subset of size 6 | **0.7360** | 0.6830 | 0.7714 | 25 | yes | the floor every selector must beat |
| L-SML + automatic feature selection | Laplacian score | **0.7261** | 0.6684 | 0.7645 | 25 | yes | classical baseline |
| residual-guided (leave-one-out / greedy on the fit) | greedy on the U-PCR projection residual | **0.7225** | 0.6726 | 0.7557 | 25 | yes |  |
| residual-guided (leave-one-out / greedy on the fit) | minimise Eq.14 residual, adaptive K | **0.6955** | 0.6574 | 0.7209 | 25 | yes |  |
| residual-guided (leave-one-out / greedy on the fit) | greedy on the relative Eq.14 residual | **0.6952** | 0.6567 | 0.7209 | 25 | yes |  |
| U-PCR (Step 204) | U-PCR + L-SML clustering (our variant) | **0.6931** | 0.0000 | 0.0000 | 25 | yes | cross-cluster pairs only — REFUTED, both gates fail |
| U-PCR (Step 204) | U-PCR fully paper-faithful | **0.6910** | 0.0000 | 0.0000 | 25 | yes | every deviation corrected |
| residual-guided (leave-one-out / greedy on the fit) | route by the residual | **0.6851** | 0.6541 | 0.7057 | 25 | no — fixed |  |
| L-SML + selection, anchor step ablated | a7 iterative consensus, anchor step OFF | **0.6524** | 0.7013 | 0.6198 | 25 | yes | no anchor, no hand signs — 3 cells fall below 0.5 |
| U-PCR (Step 204) | U-PCR hierarchical (2-level) | **0.6291** | 0.0000 | 0.0000 | 25 | yes | REFUTED |
| U-PCR (Step 204) | U-PCR, anchor step OFF | **0.2449** | 0.0000 | 0.0000 | 25 | yes | global sign is provably unidentifiable — inverts in 25/25 |
