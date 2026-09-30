Subject: Update on the last three weeks: the aligned fusion benchmark, white-box features and applications

Hi Ofir, Bracha and Amir,

Since our last meeting I worked on three fronts: the algorithmic core (label-free fusion), white-box internal features, and applications beyond completed-answer detection. Most of the work went into the algorithmic line, and I think that is where the thesis core now sits; the other two fronts are presented more briefly below.

### TL;DR

- **Aligned fusion benchmark (main result):** 13 label-free methods on the same frozen 24 cells (48,607 answers) land within 1.4 AUROC points of each other, with overlapping intervals. Added structural complexity — graphs, dependence models, nonlinear energy models — does not buy a reliable response-level gain.
- **Two results do separate:** Family-NRM improves the reserved PRMBench confirmation by +0.46pp with a positive interval (our one confirmed increment), and CIW-DEEM is the registered leading challenger (best point estimate, below the promotion threshold).
- **White-box:** a practical tie with the gray-box score on matched answers, with much broader coverage; needs one preregistered validation before it can be a central result.
- **Applications:** first-error localization beats Mind the Gap in all 8 cells (25.7% → 31.4% macro F1); prefix prediction separates cleanly at 64 and 256 tokens; LEASH stopping loses accuracy in every cell (boundary evidence); RAG transfers but supports no superiority claim.
- **To decide at the meeting:** which prospective confirmation comes first, and whether RAG stays an active direction.

### 1. The algorithmic line — what the aligned benchmark says

The route since July, briefly: I implemented IU-PCR and SU-PCR (Tenzer et al.'s extensions of the U-PCR line), integrated DUFS as a graph regularizer inside the solve rather than as a selector before it (DUFS-LIU), decomposed the fused score into six provenance families and corrected it with a residual mode (Family-NRM, architecture inspired by HARP), and adapted DEEM's nonlinear energy model to our continuous measurements (DEEM-B3, then CIW-DEEM). To compare all of this honestly I froze a 24-cell response benchmark and evaluated the 13 methods under identical conditions: two independent builds with byte-identical outputs, scores frozen before labels were opened, grouped 20,000-draw bootstrap.

| Method | What it does | Macro-24 AUROC | 95% CI |
|---|---|:---:|:---:|
| **DEEM-B3** (ours) | nonlinear family-wise energy correction | **0.7813** | [0.772, 0.790] |
| Equal six-family average | simple baseline | 0.7810 | [0.772, 0.790] |
| **DUFS-LIU** (ours) | DUFS answer-graph regularizes the IU solve | 0.7766 | [0.767, 0.786] |
| IU-PCR | projected covariance solve (anchor) | 0.7761 | [0.767, 0.785] |
| Family-NRM (within-cell) | residual family-disagreement correction | 0.7746 | [0.766, 0.784] |
| CA-SpecRaGE (atomic) | agreement-weighted multi-view graph regularizing IU | 0.7742 | [0.765, 0.783] |
| Deployed U-PCR | previous method of record | 0.7740 | [0.764, 0.783] |
| Equal 30-feature average | simple baseline | 0.7739 | [0.765, 0.783] |
| PGRD-A | residual-space graph-roughness descent | 0.7735 | [0.764, 0.783] |
| SU-PCR | sparse correlated-error correction | 0.7714 | [0.762, 0.780] |
| Continuous L-SML | earlier spectral lineage | 0.7710 | [0.762, 0.780] |
| DUFS stability + L-SML | stability-selected gates, then L-SML | 0.7703 | [0.761, 0.780] |
| DUFS parameter-free + L-SML | gate-selected features, then L-SML | 0.7674 | [0.758, 0.777] |

The table itself is the finding. Every dependence-aware extension recovers real, stable structure, but that structure follows shared confidence and response length more than correctness: the whole roster sits within 1.4 points, a trivial six-family average is second, and SU-PCR's sparse correction is inconclusive (heterogeneous across cells, and replacing the projected solve with the full structured inverse was clearly harmful). The bottleneck is not model capacity; it is that an unsupervised objective can identify dependence without knowing which side of it is correct.

Two results do separate from this picture:

- **Family-NRM** (the donor-environment version) is the one confirmed increment: on the reserved PRMBench response test it improves AUROC 0.7206 → 0.7252, +0.46pp with 95% CI [+0.07, +0.84]. It relies on a manually supplied provenance prior — which is exactly what buys the target alignment the graph methods lack.
- **CIW-DEEM** is the registered challenger from the DEEM line: it separates each measurement's shared component from its own innovation before the unchanged B3 fit, cross-fitted and label-free. It has the best point estimates (0.7820 cell-macro; 0.7492 under the stricter equal-dataset-family aggregation), but its gain over frozen B3 (+0.07pp equal-family) is below the preregistered +0.25pp promotion threshold — a challenger, not a promoted method.

The many unsuccessful attempts in between (clustered U-PCR at −4.5pp, the 111 published keep-rule variants, residual-graph DEEM, sample-dependent routers) are preserved in the algorithmic deep dive.

### 2. White-box — tie on matched answers, broader coverage, validation pending

I captured per-layer logit-lens and residual-stream measurements (TriLens-style trajectories, fused without labels) on 14 dataset-model cells across nine model families; 13 cells were scorable. The registered compact fusion lost to final-layer NLL (0.618 vs 0.730). A richer distributed-depth representation reached 0.785 macro AUROC, but its view registry was chosen after seeing these cells. On the 31,440 answers where both methods score, white-box and gray-box are a practical tie (0.7817 vs 0.7830, Δ −0.13pp, CI [−1.69, +1.23]pp); white-box does cover 42,238 answers versus 31,467 complete gray-box cases, and a post-hoc white+gray average reaches 0.790. So: real signal and a real coverage win, but no validated aggregate advantage yet. This comparison ran against the gray mixed-v2 generation and was frozen by design; one preregistered validation on new cells is the missing piece.

### 3. Applications — four panels, two worth pursuing

All four lanes ran under the then-current fusion generation (fixed IU-PCR / DUFS-LIU heads), frozen deliberately for certification. Re-running them under the current generation is scheduled follow-up, and the small deltas in the table above bound what that re-alignment can be expected to change.

| Application | Result vs comparator | Status |
|---|---|---|
| First-error localization (ProcessBench) | macro F1 25.71% → **31.36%**, better in all 8 cells; exact-step 17.8% → 21.8%; within-one-step 39.4% → 46.8% — vs reproduced Mind the Gap | positive — continue |
| Prefix prediction (fixed budgets) | AUROC 0.5955 vs 0.5629 at 64 tokens (+3.3pp [+0.4, +6.3]) and 0.6572 vs 0.6114 at 256 (+4.6pp [+1.5, +7.7]) — vs our frozen causal baseline; budgets 16/32/128 do not separate | positive — continue |
| Actual stopping (LEASH reproduction) | −38.8% generated tokens but −18.3pp pass@1, worse in all 6 eligible cells (2 Mistral cells excluded: no chat template) | negative — boundary evidence |
| RAG evidence detection | RAGTruth 0.727 / 0.689 / 0.659 (answer/sentence/token); GASP-style score 0.671 vs matched IU 0.660, CI crosses zero; supervised LettuceDetect ceiling 0.793 F1; RefChecker 0.664 / 0.640 / 0.751 across context conditions | transfer shown, no superiority — decision below |

(My previous email quoted 31.72% for localization — that was a factorial variant; 31.36% is the frozen selected pipeline that the certified lane stands behind. Our internal variants differ by only +0.33–0.58 F1 with intervals crossing zero, so the contribution is the framework's transfer to localization, not one particular graph.)

The way I read this: localization and prefix prediction are one story — detect whether a trace errs, locate the first error, and predict the outcome early — and it is the application story we are demonstrably good at, including against a published label-free comparator. LEASH stays as boundary evidence that live stopping at preserved accuracy is unsolved, which is precisely what makes the prefix result the necessary first step. RAG transfers, but shows no superiority over the simple task-specific score, and it gives up the single-pass property (evidence contrasts require rescoring under several context conditions), so I would rather we decide its status explicitly than keep it half-alive.

### What I would like to decide together

1. **The next prospective confirmation.** My preference: a new-data confirmation of the localization + prefix pipeline, re-aligned to the current fusion generation, with CIW-DEEM and the white+gray combination carried as frozen challengers inside the same run. The alternatives are a dedicated CIW-DEEM confirmation or the white-box validation.
2. **RAG:** keep investing, or record it as bounded transfer evidence?

Could we meet next week to go over these and settle the thesis and paper story?

The complete record — equations, full chronology, the negative experiments, confidence intervals, paper references and every report — is in the packet:

- [Visual HTML version of this letter](advisor_update_aug26_2026/index.html)
- [Algorithmic deep dive](advisor_update_aug26_2026/01_algorithmic_deep_dive.md)
- [White-box deep dive](advisor_update_aug26_2026/02_whitebox_deep_dive.md)
- [Applications deep dive](advisor_update_aug26_2026/03_applications_deep_dive.md)
- [Complete plot and document index](advisor_update_aug26_2026/ASSET_INDEX.md)
- [Full paper list and how each paper informed the work](advisor_update_aug26_2026/REFERENCES.md)

Thanks,
Omri
