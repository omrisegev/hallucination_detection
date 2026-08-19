# A0–A6 route survey and post-A6 recommendation

**Date:** 2026-08-15
**Status:** read-only evidence survey; not a preregistration, not a result.
**Provenance:** produced by the read-only survey subagent that
`HANDOFF_A6_S0B_TO_CLAUDE_2026_08_15.md` §8 requested, run in parallel with the
S0b continuation work. Boundary respected: no natural labels, no PopQA
content/targets, no sealed A6 results were opened; no files were edited by the
surveyor.
**Scope note:** this note informs the *post-A6* route decision only. It does
not change, rescue, or delay the frozen S0b/S1 execution.

## 1. Mechanism-level survey (36 materially distinct label-free / mechanically self-supervised routes)

Failure codes: **ID** identifiability · **LEN** length capture · **RED**
redundancy/duplicate mass · **INST** instability · **NUIS** nuisance capture ·
**NC** non-convergence · **TR** transfer · **DEP** deployment incompatibility.

### 1a. Feature-selection / per-feature-statistic era (Steps 186–225)

| # | Route | Mechanism (one line) | Scale of evidence | Outcome | Failure |
|---|---|---|---|---|---|
| 1 | Selector bench: GroupFS, Concrete-AE, Laplacian Score, SPEC, MCFS (Step 186) | six label-free selector families → same L-SML → AUROC | 19–25 cells, 6 families | best learned = GroupFS **0.7323** vs curated GOOD_6 **0.7440**; CAE 0.68–0.71; classical ≈0.70; kurtosis floor **0.72** | ID |
| 2 | Pseudo-label-anchored DUFS gates `a6` (Step 194) | fuse 4 seed views → pseudo-label → supervise gates with centered agreement term | 25 in-scope cells, 2 pre-registered gates | mechanism gate ρ(gate, view AUROC) = **+0.207** vs 0.30; performance **+0.22pp** vs +1.0pp gate. Both FAIL | ID |
| 3 | Channel ceiling pricing (Step 220) | let labels do each U-PCR step perfectly, price the room | 24 cells, 4 channels + 6 controls | only *which features are kept* has room **+1.48pp [+0.97,+2.03]**; sign channel **−0.06pp [−0.29,+0.08]**; v1/v2 blend +0.19pp; `var_y` ≈0 | — (pricing) |
| 4 | Marginal-correlation ranking, oracle and estimated (Step 221) | rank features by ρ(feature, correctness) | 24 cells × 5 split-halves | true ρ vs matched floor **+0.08pp [−0.78,+0.87], p=0.62**; a *perfect* ρ̂ is worth **+0.34pp, p=0.88** | ID |
| 5 | Label-free ranker menu, 8 arms (Step 222) | DUFS gate, principal leverage, redundancy, L-SML cluster size, pair-fit residual, cluster round-robin | 24 cells × 5 splits, Holm over 6 | **all on/below floor**; DUFS **−0.70pp** (Holm 0.36), cluster size **−1.61pp** (Holm 0.008), redundancy **−3.13pp** (Holm 0.002). Spearman(\|overlap excess\|, performance) = **−0.71** | **ID (sharpest)** |
| 6 | Set-level label-free objectives (Step 223) | composite reliability `C=λλᵀ+Ψ+Δ` (5 arms) + ℓ0-CCA | same 120 splits | best `m_eff` **+0.08pp** (Holm 0.72); `cca_gates` −0.47pp. Same greedy handed half-A labels clears floor by **+1.88pp** → *the objective fails, not the search* | ID |
| 7 | Published unsupervised FS transplanted as keep rules (Step 224) | 111 variants of LS-CAE, DPP, mmDUFS, RFAE, SCFS, CAE, Laplacian Score, SPEC, MCFS | 24 cells × 5 splits × 2 arenas, 3 sweeps | **all 111 negative** vs deployed rule: DUFS −0.96pp, CAE −2.74pp, LapScore/SPEC −3.77pp, Eq-14 residual −5.89pp; **DPP k4 −8.08pp, 0W/24L**; mmDUFS −0.12pp linear and nonlinear | ID + RED |

### 1b. Dependency / graph / repeated-measurement fusion era (Steps 226–246)

| # | Route | Mechanism | Scale | Outcome | Failure |
|---|---|---|---|---|---|
| 8 | SU-PCR sparse-error reliability (Steps 226/235; `results/dependency_fusion_study/REPORT.md`) | `C = L + S`, Dror et al. sparse-error correction of U-PCR | 24 cells, factorial 2×2, Holm-3 | H1 **+1.26 [−1.78,+6.33], p=0.73**; keep-arena **−0.07pp**. Not established | ID |
| 9 | SDSF condition-controlled dependency weights (same) | PSD condition-controlled covariance weighting | 24 cells | H2 **−5.65 [−10.19,−2.73], p=5.96e-07, 2W/22L**; `full.sdsf` 0.7104 vs `keep.iu_pcr` 0.7742 | NUIS |
| 10 | DEEM (AISTATS 2026) nonlinear dependence layers (same) | learned multinomial layers → identifiable iRBM pseudo-probabilities | 24 cells × 5 seeds | deep-hard ensemble **0.7544**, iRBM **0.7464**, both below `full.iu_pcr` 0.7542/`keep.iu_pcr` 0.7742. Deep-hard **seed std 1.32pp, max cell range 38.43pp** | INST |
| 11 | SpecRaGE view fusion: manual / balanced-atomic / LOCO micro-views (Step 227) | CA-weighted view fusion at three granularities | frozen 24-cell benchmark, fingerprint `f9bcfeed…` | atomic CA **+0.023pp** (tie); micro-view CA **−0.363pp**, worst cell −2.855pp — yet micro-partitions reproducible at **bootstrap ARI 0.84–0.94** | ID |
| 12 | Atomic-operator gating premise (Step 228 research) | frozen label-free roughness/reproducibility proxy predicts operator usefulness | 24 cells × 30 features × 9 (k,λ) × 40 subsamples | median within-cell Spearman **−0.312**; permutation p **0.690**; top-proxy atom **−0.838pp** (7W/17L); 3/15 gates pass. *Anti-correlated, not merely uninformative* | ID |
| 13 | Graph-coupled family relevance GCFR (Step 229) | per-sample family gates smoothed by a fixed 6-node family Laplacian | 24 cells + 20-seed synthetic | real **−0.135pp** (8W/16L); synthetic **+0.773pp** (inactive family inconsistent) vs **−9.272pp** (inactive family shares coherent nuisance) | NUIS |
| 14 | Repeated cross-view diffusion RCV-AD (Step 230) | alternating diffusion `(P_A P_B + P_B P_A)/2` over 16 complementary partitions | 24 cells, 3 block schemes, T∈{4,8,16} | **+0.004pp** [−0.052,+0.029]; converged (partition-score Spearman ≈1.000) but graph CKA vs AUROC Spearman **−0.240, p=0.259** | ID |
| 15 | Repeated-measurement reliability (Step 234) | synchronized moving-block bootstrap replicates → `S_signal v = λ S_within v` | GSM8K + MATH, frozen retention rule | **+0.0006 / +0.0013**, both intervals crossing zero. Generalized eigenvectors drive off-diagonal covariance fraction **0.89 → 0.03/0.07**, destroying what U-PCR needs | **DEP** + ID |
| 16 | Hard filtering / gating before fusion (Step 235) | deployed `rho_max/3` filter and 3 stricter levels | 24 cells, 2 contracts | full-pool **0.776562** → **0.774249** → **0.764153**; DUFS increment over IU-PCR **+0.048 → −0.025** | RED |
| 17 | Coupled third-moment CP k-factor deflation (Step 245) | symmetric CP on distinct-index 3rd-order tensor, deflate before IU-PCR/DUFS-LIU | 24 cells, ranks 0–4, permuted-moment control | label-free selector chose **rank 0 in 19/24**; deflated IU-PCR **0.7540** vs 0.7761; monotone collapse to **0.5734** at rank 4 | ID |
| 18 | IU-PCR-initialized latent-state HMM (Step 246) | explicit temporal latent state over fused local risk | 8 ProcessBench cells | reversible **30.03% F1 / 25.20%** vs DUFS-LIU 31.72%/26.70%; absorbing control 12.64%/8.73% | ID |

### 1c. HARP / NRM line (Steps 247–252)

| # | Route | Mechanism | Scale | Outcome | Failure |
|---|---|---|---|---|---|
| 19 | HARP supervised contribution teacher (Step 247) | decompose IU into 6 provenance-family contributions, fit a *supervised* correction | 23 cells + 4 transfer domains | within-cell **+0.721pp**; global teacher +0.410 / +0.684 / +1.191 / +0.646pp; all 8 LOFO fits sign-consistent | *(ceiling — proves target exists)* |
| 20 | Cardinality balancing CB-CS-IU (Step 248) | family IU leverage + feature cardinality as label-free nuisance proxies | 23 cells + PB + SemGrad | +0.442 / +0.864 / +1.263pp, but SemGrad **−0.767pp** equal-dataset (TruthfulQA **−1.708pp**); reverse-cardinality control *helped* | **TR** |
| 21 | **Family-NRM (NRM-CS-IU) — the one confirmed win** (Steps 249/250) | residualize 6 family contributions vs IU, take eigenvector at λ closest to 1 (λ=1.035378), fixed `1/G` trust | 23 cells + PB(2) + SemGrad + PRMBench(6,966) + HLE | LOFO **+0.277pp [+0.016,+0.533]**; PB Qwen +0.557; PB Llama +1.580; SemGrad +1.310; **PRMBench 0.720602→0.725206, +0.460pp [+0.068,+0.841], P(δ>0)=0.9892**; HLE +0.345pp [−0.898,+1.628] underpowered (68 correct) | *(passes — but manual families)* |
| 22 | Atomic NRM: group-free neutral projector (Step 251) | permutation-null band over 17 universal atoms, project inverse-dependence anchor | 23 cells + 3 transfer domains, 1,000-draw null | **−0.667 / −1.106 / −1.305 / −4.216pp**; atomic−family −0.944 [−1.654,−0.174] … −5.526 [−9.005,−2.047]. *Yet* supervised atomic ceiling **+1.298pp** vs family +0.721pp, diff **+0.577 [+0.102,+0.910]** | **ID** |
| 23 | Learned / random / refined partitions (Steps 251–252) | 5-cluster dependence partition; 50 cardinality-matched random partitions; deterministic refinements; γ3-signed refinement (G=10) | 4 domains | 5-cluster loses everywhere; 3/50 random beat family on originals; **refined-partition NRM v0 negative everywhere** (−0.29/−0.90/−1.43 banded); every label-free partition-selection criterion uninformative (**Spearman −0.13…+0.27**), best label-free pick +0.52 vs best available +1.21 | ID |
| 24 | Diagnosis of the atomic failure (Step 252; `docs/research_notes/atomic_orientation_reply_2026-08-13.md`) | measure *why* the projector points away | bit-exact reproduction of frozen calibration | permutation band holds **3.0%** of supervised target mass; **63.6%** sits on the rejected λ=2.04 mode; positive anchors **+0.302/+0.215 unprojected → −0.173 projected**; **even the supervised global atomic direction nets ≤ +0.1pp on heterogeneous originals at any trust** (per-cell coherence median cos **0.394**), while on homogeneous ProcessBench it gives +0.46/+1.31 | **TR (transport wall)** |
| 25 | b-coupled cubic orientation channel γ̂3 (Step 252) | pooled cubic Hermite coupling of residuals to IU-score nonlinearity | 23 cells + 2 PB domains, 5-reviewer pass | **cos +0.76** with supervised direction, **13/17** signs, **9/9** within-family signs the family quotient cannot represent. As a *corrector*: −0.16 / −0.22 / **+0.39pp (4/0 on PB Llama)**. Amplitude ≈ a³/48 (30–50× attenuated); per-cell noise-dominated | *(orientation instrument, not corrector)* |

### 1d. The frozen A0–A6 program

| # | Route | Mechanism | Scale | Outcome | Failure |
|---|---|---|---|---|---|
| 26 | **A0** identifiability/data audit (Steps 255/257) | mechanically derived feature DAG + 23-env missingness + exact cross-model pairing | 30 features, 23 environments | **PASS**. 17 universal features, pair coverage 8–23 cells, min bundle retention 19.8%, 3,400 exact Qwen3-4B/8B/Llama3.1-8B triples | — |
| 27 | **A1** factorial soft measurement quotient (Steps 256/257) | channel × operator crossed design, soft loadings, held-env reconstruction | 16/7 hash split, 32 matched random partitions | audit MSE **0.032009** vs pooled PCA **0.034704**, delta **−0.002695 [−0.005845, 0.000282]** (crosses zero); ρ=0.999 duplicate gets **3.009×** mass vs 1.10 gate | **RED** (+ non-material) |
| 28 | **A2** multi-environment missing-aware JBD (Step 257) | shared loadings / distinct per-env variance profiles | all 30 atoms, LOEO folds, PSD stationary null | **0.028700** vs matched PCA **0.032864**, delta **−0.004164 [−0.012164, 0.000838]**; LOEO mechanism-rank ratio **0.618** vs 0.70 gate; one fold = 30 singletons; advantage **reverses** under train-only stationary null | **INST + ID** |
| 29 | **A3** A1×A2 hybrid (Step 257) | — | — | **CLOSED BY PREMISE** (both inputs failed) | — |
| 30 | **A4** paired cross-model CorrCA (Steps 258–260) | shared/individual sources over 3 scorer views of a fixed response | 3,400 × 3 × 29 tensor, item-first folds, 2 nulls | repeatability **0.997881**, held-Llama **0.955465** — but nested baseline `single:1 = trace_length` reached **0.966908**; delta **−0.011444 [−0.016036,−0.009034]**. CorrCA loading on trace length **0.997897–0.999279** across folds | **LEN** (+ pre-registered `CLOSE_NO_TARGET_CONTRAST`) |
| 31 | **A5** IU-anchored sparse latent mixture (Steps 261–263) | item-level equal-covariance 2-component likelihood, sparse within-component precision | 100 sealed nuisance-dominant seeds, 20,000-draw CI | formal **`CLOSE_NUMERICAL_NONCONVERGENCE`** (2/100). Semantically decisive: target preference **62/98 and 25/98** vs 90/100 gate; candidate−IU **−0.038484 [−0.047495,−0.029659]**, no nonnegative draw; α=1 chosen **46/98**, losing **0.080974** | **NUIS** (formal NC) |
| 32 | **A6 / PTNI-IU** (Steps 264–268) | reciprocal 2×2 prompt-response crossover over verified typed-AST task worlds; factorial target/nuisance/interaction moments → one nuisance-whitened atomic direction, IU-orthogonal, exact α=0 fallback | S0a executed: 1,800 quartets, 6,000 natural rows, 7,200 inner folds, 36 null cells, 7,800 checkpoints | **`PASS_S0A_VERIFIED`** only (construction). S0b implemented but not frozen/run; **no detector number exists** | *(open)* |

### 1e. Adjacent intervention routes and the parallel branch

| # | Route | Mechanism | Scale | Outcome | Failure |
|---|---|---|---|---|---|
| 33 | RAG evidence-contrast EC-IU-PCR (master Step 243) | rescore a *fixed* answer under full / no-context / leave-one-chunk-out | 2,700 test responses, 16,200 condition records | EC-IU-PCR **0.7031**, EC-DUFS-LIU 0.7026 vs GASP-top50 0.6721 (**+0.0305 [0.0237,0.0378]**) — the *contract* wins. DUFS−IU **−0.00048 [−0.00061,−0.00034]**: the mechanism failed. Residualizing length/chunks: response 0.7484→0.6481 | LEN + ID (mechanism) |
| 34 | Mixed-v2 under evidence interventions (master Step 244) | same 30 features re-extracted per evidence condition | RAGTruth QA + D2T | largest task-macro gain **+0.1163 [+0.0795,+0.1544]**; DUFS−IU only **+0.0065**. A6 §1 rules this an **ambiguous adjacent target** (evidence removal changes contextual support, not necessarily the answer proposition) | *(target-ontology mismatch)* |
| 35 | K-sample answer-agreement self-consistency (Steps 158/174/181) | boxed-answer agreement over K=5 same-temperature passes, fused with 1-pass L-SML | **1 cell**, MATH-500/Qwen2.5-Math-7B, N=200 | **fused 95.2 [91.8, 98.0]** vs L-SML 85.1 and SC 82.1, ρ=+0.23 → **+10.1pp, gate PASS — the strongest fusion number in the project**. Same-T K=5 0.912 vs multi-T 0.859 | **DEP** at deployment (K passes); *never tried as a calibration-only orienter* |
| 36 | White-box per-layer depth views (`origin/whitebox/per-layer-views` Steps 243–244) | logit-lens entropy / lens-logprob / top-1 / KL(l‖final) at MHSA, FFN and residual taps + per-layer covariance eigenvalues | **14 cells / 9 model families / 4.56 GB** extracted; architecture guard **0.00e+00** on every accepted family | **No fusion result exists.** 84–126 views/model, single-layer AUROC 0.63–0.73, constant depth structure where the token axis fails (median trace 6–8 tokens on short QA vs 243 on GSM8K). Omri's Step-243 ruling: **orthogonal arm, deliberately not combined** | **DEP** vs the S1 contract (different access class, not mixed-v2) |

## 2. Failure-mode taxonomy

### Demonstrated-fundamental for this feature channel

1. **Covariance/likelihood/moment-only target identification is closed** — hit at 2nd moment (#22, #28), likelihood (#31), 3rd moment (#17), graph geometry (#12–#14), and marginal statistics (#4–#7). Formal statement: `atomic_orientation_reply_2026-08-13.md` §3 (target identifiable at most up to binary-signature factor directions; rate-matched difficulty is observationally equivalent); matching literature verdict in `atomic_nrm_null_spectrum_literature_2026-08-13.md` (a noise subspace rejects nuisance, it never names a target). Encoded in the program doc §2.
2. **Stability ≠ target relevance** — four independent measurements: ARI 0.84–0.94 yet −0.363pp (#11); partition-score Spearman ≈1.000 yet +0.004pp (#14); repeatability 0.997881 that was token count (#30); LOEO cosine 0.975505 yet negative everywhere (#22).
3. **Identifying good features ≠ performance** — redundancy ranker most-identifies and worst-performs; Spearman(\|overlap excess\|, performance) = **−0.71** (Step 222); perfect ρ̂ priced at +0.34pp p=0.88 (Step 221).
4. **Elementwise-positive anchors carry no contrast orientation; projection flips them** (+0.302/+0.215 → −0.173, Step 252).
5. **Anti-redundancy/diversity selection is actively harmful, dose-dependent** (DPP k4 −8.08pp 0W/24L; Steps 222–224).
6. **The transport wall** — on heterogeneous original cells no transported global atomic direction, supervised included, nets more than ≈+0.1pp at any trust (median per-cell cos 0.394); on homogeneous ProcessBench transport works (+0.46/+1.31). This bounds ANY single frozen global atomic corrector — including a successful PTNI direction. Family-NRM's edge = transportable 6-dim geometry + smaller trust dilution, not superior coherence.

### Merely unexplored (not closed)

- Mechanically verified target-changing interventions (A6 — in flight; first legal instance).
- Domain-conditional / per-environment adaptation (A8): measured headroom **+1.31pp supervised / +0.39pp label-free** (PB Llama, Step 252) and **+2.833pp** conditional-family headroom (p=0.0020, Step 229); no universal rule exists — the headroom is structurally conditional.
- The b-coupled sign bit applied to deployed Family-NRM (margin ≈0.56 vs the current 0.065 — 8×; `atomic_orientation_reply` §5 item 1; never built).
- Cross-model *answer-string* agreement at calibration (permission granted by Omri's Step-252 ruling; feasibility audit never run).
- K-sample self-consistency as a calibration-only orienter (named as the required next premise by Steps 228, 229, 230 independently; never executed; the one existing measurement is the largest label-free fusion gain in the repo, +10.1pp on one cell).
- White-box depth views as a fusion basis (data collected, zero fusion decisions made).

## 3. Genuinely closed vs superficially closed

**Genuinely closed** (do not reopen without a new information source): `argmin|λ−1|` single-mode selection at atomic resolution; elementwise-positive anchors into data-defined subspaces; marginal per-feature ranking (label-free *and* oracle); transplanted published FS keep rules (111/111); anti-redundancy selection; SDSF weighting; 3rd-moment deflation as a selector; missing-aware JBD *as implemented*; A5 mixture likelihood; A4 fixed-response cross-model views as a target source; γ3-signed partition refinement; random-partition search with label-free selection; sign(skew) harmonization, accuracy-proxy-weighted covariance contrast, partition model averaging, trust-scale tinkering (each closed analytically and empirically, `atomic_orientation_reply` §6).

**Superficially closed / reopenable with a different mechanism**:
1. A5's *formal* `CLOSE_NUMERICAL_NONCONVERGENCE` is a technicality — but the semantic audit is independently decisive; what is reopenable is only "an unlabeled density objective anchored by interventions", not likelihood-anchored-by-IU.
2. A2 JBD — Step 257: "closes the implemented missing-aware route, not all possible JBD algorithms"; the capacity-matched simulator passed; the reopening shape is CPC/JBD-supplies-axes + y-asymmetric channel supplies choice+sign (target-heavy λ=2.04 mode has cross-cell variance-dispersion CV 0.41 vs 0.12–0.18 in-band). Caveat: A2's detector-only gates were never run.
3. RAG evidence interventions — closed for A6 on target-ontology, not on failure evidence; the structure demonstrably carries information (+0.1163 task-macro). Reopenable only under a faithfulness/grounding target (different thesis claim).
4. A1 factorial quotient — closed on a duplicate-mass robustness gate; the retained structural finding survives; reopening buys measurement structure only, never target identification.
5. γ̂3 b-coupling — closed as a *corrector*, open as an *orientation instrument* (cos +0.76, 13/17, 9/9); its one-bit deployment was never built.
6. Cross-model answer agreement — never closed; contract-blocked, and the block was lifted.
7. **A7 (≤32-label orientation) — closed by goal change, not by evidence, and inconsistently documented.** `Research_Directions.md` (Step 268 section) demotes it out of the deployable ladder ("cannot fit, select, orient, or promote the deployable method"), while the frozen program doc §6 and the S0/S1 contract §7.7 still name A7 the automatic successor on A6 failure. **This documentation conflict should be resolved in writing before A6 reaches its registered verdict.**

## 4. Recommendation for the route after A6

**Primary: A8, re-specified as domain-conditional orientation** — `w_e = w_global + α_e·δ_e`, `α_e` from unlabelled calibration evidence only, exactly zero when evidence is insufficient (preserves the exact-IU fallback discipline).

- Every closed route died at **identification** or at **transport**. A6 attacks identification and is in flight; **nothing in A0–A6 attacks transport**, and A8 is the only registered route that does.
- The headroom is measured, not hypothesized (+1.31/+0.39pp PB Llama; +2.833pp conditional-family, p=0.0020), and it is structurally conditional — exactly what a global direction cannot capture.
- Transport demonstrably works within homogeneous domains and fails across them; a per-environment correction operates where the evidence says transport works.
- A8 **composes with A6**: if PTNI yields a valid direction, A8 supplies the per-environment trust deciding whether it survives the transport wall; if PTNI closes, A8 still has a legal α_e source below.

α_e evidence source (the key design decision), in preference order: (1) PTNI's own held-family/held-scorer mechanical statistics (if A6 reaches S2a/S2b — zero new data); (2) per-environment K-sample answer-agreement consistency used only at calibration — the "repeated generations / semantic answer consistency" premise independently named by Steps 228/229/230 and never executed; amplitude-advantaged over every channel tried; deployment stays one-pass because the K passes are calibration-time only. Prerequisites: the never-run feasibility audit (item-ID alignment across dataset-sharing pairs; where K>1 exists — currently only `edis_aime24` T-triples and the five MATH-500/Qwen-7B passes, so most environments need a cluster run) and a pre-registered falsification arm for **coherent repeatable hallucinations** (the case Step 229 names).

**Secondary, cheap insurance: build the b-coupled sign bit for deployed Family-NRM** — replace/gate the all-ones sign rule with `sign(⟨v_neutral, γ̂3_family⟩)` (margin ≈0.56 vs 0.065), plus a label-free abstain gate returning exact IU. Hours of work, label-free, deployment-neutral, does not touch the A6 boundary; protects the PRMBench confirmation whose CI floor is +0.068pp. Falsification already specified (`atomic_orientation_reply` §5).

**Explicitly not recommended now**: PTNI-guided NRM as a next route (correctly queued as conditional on A6's outcome; the neutral band held only 3.0% of target mass); A7 (excluded by the current goal statement and priced near zero by Step 220); white-box depth views as the A6 successor (most novel unexplored axis, but different access class and Omri's standing ruling keeps the arms separate — record as a separate thesis arm).

**Planning notes (not grounds to touch S0b):** (i) the handoff §5 preflight (fake NLL, one fold, gradients 1.67e-7–2.9e-7 vs the 1e-8 gate) makes an early registered close plausible, so the successor decision may be needed sooner than the S2/S3/S4 ladder implies — the recommendation above is robust to both branches; (ii) the A7 documentation conflict (§3 item 7) should be resolved before the A6 verdict.
