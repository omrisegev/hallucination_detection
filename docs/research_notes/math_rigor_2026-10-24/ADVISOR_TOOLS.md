# What the advisors' own work offers against the math review (2026-10-24)

**Question (Omri, 2026-10-24).** Do Ofir Lindenbaum or Bracha Laufer-Goldshtein have mathematical tools, or other
contributions, that can strengthen the thesis against the math peer review?

**How it was answered.** Two read-only agents built each advisor's publication list from primary sources: the
arXiv API, Semantic Scholar, OpenAlex, Google Scholar, university pages, and the proceedings pages of each venue.
They checked the repository for earlier uses and mapped each tool to the review's weaknesses (W1–W12, see
`HISTORY_AUDIT.md`). "Verified" means the agent read the claim in a source it opened. Claims marked "recalled"
were not opened.

## Short answer

- **Neither advisor has a theorem saying when label-free fusion beats the PRM.** The improvement condition (W3)
  has to come from our own model (`FRAMING_THEORY.md`, P2 and P5).
- **Laufer-Goldshtein offers a certified decision layer.** Statistical guarantees on top of our label-free score
  answer the forking-paths critique (W12), give a gate with a guarantee (W3), and give a count rule with a
  guarantee (W10). The price is a declared labeled calibration split, which is CROP's access level. It is
  therefore a separately labeled "calibrated" track, not a change to the label-free method.
- **Lindenbaum offers estimators and diagnostics that the project has not used for this purpose.**
  - Block CCA gives Fisher-type weights whose PRM share does not depend on K (W4, W5, W8).
  - mmDUFS with position/length as their own view diagnoses how much shared structure is nuisance (W1, W2).
  - Partition First comes with a recovery theorem for grouping (W7).
  - His methods were used in the repository before, but only as feature selectors, and every such use tied or
    lost.

## Bracha Laufer-Goldshtein

Assistant Professor, School of Electrical Engineering, Tel Aviv University (verified:
[TAU profile](https://en-engineering.tau.ac.il/profile/blaufer),
[Scholar](https://scholar.google.com/citations?user=1YweTFwAAAAJ)). Her publications cover two lines:
conformal prediction and risk control (2023–2026), and manifold- and simplex-based speaker localization and
source counting (2016–2025). There is no LLM-verification paper.

| Tool | Result | Weaknesses | How we would use it | Cost |
|---|---|---|---|---|
| **Pareto Testing** ([ICLR 2023](https://arxiv.org/abs/2210.07913), with Fisch, Barzilay, Jaakkola) | Prop. 5.1: every returned configuration satisfies all risk constraints with probability ≥ 1−δ. Candidates are chosen on one split and tested by fixed-sequence testing on a disjoint split; the method abstains if none passes | W12, W3, W5, W8, W11 | The design grid (bank, arm, PRM scale, shrinkage, K, rule, cap) becomes one tested family. "PRM alone" is a configuration, so the output is a certified fuse-or-not gate with a fallback | 1–2 days, CPU. Needs a labeled calibration split (about 3,700 answers to certify a 2-point gain at δ = .05; PRMBench has 6,211) |
| **Risk-controlled source counting** ([TASLP 2026](https://arxiv.org/abs/2603.17377)); adaptive set size ([Damri, ICLR 2026](https://openreview.net/forum?id=ck5T7QeiDh)) | The count threshold is calibrated so that missed detections and false alarms are both ≤ α with probability ≥ 1−δ | W10 (with W6) | "Sources" become erroneous steps: a per-answer flag count with a guarantee, possibly on the PRM's absolute scale. Caveat: PRMBench's fixed error-count prior may make it rediscover k ≈ 2 | About 1 day |
| **Multi-Dimensional CP** ([ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/57c56985d9afe89bf78a8264c91071aa-Abstract-Conference.html), with Tawachi) | Coverage under exchangeability (Prop. 2); cells ranked by a false/true ratio, with no weighted sum | W4, W5, W8 | A calibrated readout on the vector (PRM, group scores) that needs no SML weights. Steps inside an answer are not exchangeable, so calibrate on one step per answer | About 1 day |
| **Conformalized Kalman filter** ([ICASSP 2026](https://doi.org/10.1109/icassp55912.2026.11464419), [arXiv 2609.27506](https://arxiv.org/abs/2609.27506)) | Thm 1: per-time coverage. Thm 2: whole-trajectory coverage. Exchangeability is needed across trajectories only | i.i.d. steps (A5), W1/W2 in part, W6 | The answer is the exchangeable unit; per-position thresholds; a certificate that an answer is clean | A few hours |
| Semi-supervised manifold GP + transductive CP (TASLP 2016/2017, [ICASSP 2025](https://arxiv.org/abs/2409.11804)) | GP posterior with unlabeled points in the kernel; coverage only for exchangeable labeled/test pairs | W1, W10, W12 | A few labeled answers decide whether the unlabeled geometry's latent is error or position | Medium–high |
| Simplex source counting ([TSP 2018](https://doi.org/10.1109/tsp.2018.2876349)) | Number of sources from the rank of a frame correlation matrix, with a heuristic eigenvalue-ratio threshold | W10 | Low value: it needs "pure" frames and high dimension, and our label-free count estimators already failed | — |

**Already in the repository:**
- Step 21 and Extension A planned an LTT guarantee (FNR ≤ α). It was never run.
- She raised conformal calibration at the July 2026 meeting; it was "explicitly parked"
  (`Research_Directions.md:2802–2816`).
- Her earlier suggestions were all tested: Step 147 (LR vs L-SML weights, Spearman ≈ .1–.2), Step 204 §D
  (clustering inside U-PCR, −4.46 pp), and Steps 221–222 (two DUFS proposals, no gain and −0.70 pp).

**Correction.** `docs/research_notes/Bracha Laufer's Research_ LLMs and Anomaly Detection.md` (a Gemini note)
credits her with papers she did not write:
- Learn-then-Test (Angelopoulos et al.)
- CATs (Schuster et al.)
- "Fast yet Safe" early exiting (Jazbec et al.)
- COIN

## Ofir Lindenbaum

Senior Lecturer, Faculty of Engineering, Bar-Ilan University (verified:
[CRIS](https://cris.biu.ac.il/en/persons/ofir-lindenbaum)). His work covers manifold learning, feature selection
with stochastic gates, sparse CCA and LLM optimization. There is no paper on SML, crowdsourcing, unsupervised
ensembles or LLM verification.

| Tool | Result | Weaknesses | How we would use it | Cost |
|---|---|---|---|---|
| **Block-vs-rest CCA / ℓ0-CCA** ([ICLR 2022](https://arxiv.org/abs/2010.05620), with Salhov, Averbuch, Kluger); code `spectral_utils/selectors/l0cca.py` | Under a one-factor model with noise independent across blocks, block CCA weights equal inverse-noise-covariance × loadings, i.e. u/(1−u²) with diagonal noise. In the agent's toy the PRM keeps a .212 share for every K, while the current two-stage eigen method moves from .338 to .136 as K goes 1→6 (agent's own numpy derivation, not from the paper) | W4, W5, W8, W3 | Candidate estimator for the cross stage. The regression of the PRM on the telemetry gives Σ_T⁻¹u_T, the Fisher telemetry composite, even with a telemetry-wide nuisance (`FRAMING_THEORY.md` P5) | Low |
| **mmDUFS** ([UAI 2023](https://arxiv.org/abs/2303.09381), with Yang, Kluger, Jaffe); code `spectral_utils/selectors/a10_mmdufs.py` | Shared operator vs modality-specific operator between two views | W1, W2 | View X = position, step length, answer length; view Y = channels + PRM. Measures how much of the PRM–telemetry agreement is shared with X | Low–medium (subsample to about 3k steps) |
| **Partition First, Embed Later** ([ICML 2025 oral](https://proceedings.mlr.press/v267/peterfreund25a.html)) | Thm 4.4: exact recovery of feature subsets when features are many projections of [group latent; shared latent] | W7, W8 | A grouping method with a theorem, to replace Eq.15 + spectral clustering. Caveat: the theorem needs many features; we have 11–16 | Medium |
| Spectral Self-supervised Feature Selection ([TMLR 2024](https://arxiv.org/abs/2407.09061)) | Thm 2: on a product manifold, a feature driven by one latent is asymptotically orthogonal to the other latents' eigenvectors; stability across 500 subsample refits | W1, W7, W11 | Stability diagnostics; correlate stable eigenvectors with position/length vs with the PRM | Low |
| PRAE ([UAI 2024](https://proceedings.mlr.press/v244/lindenbaum24a.html)) | ℓ0-penalized inlier selection; the relaxation equals the combinatorial problem | W10, W12 | A penalized flag objective. In one dimension it reduces to a threshold | Low |

**Already in the repository.** His methods were used only as feature selectors or sample graphs, and none
helped:
- DUFS / GroupFS: Steps 186, 202, 207, 224.
- ℓ0-CCA as a selector: Step 223.
- Alternating diffusion: Step 230. It reproduced the shared confidence/length nuisance without target signal.
- c-STG: Steps 276–277.
- LOCA audit: Step 299.

None of them was used as a weight estimator or as a nuisance diagnostic, which is what the review asks for.

**Correction.** `docs/research_notes/Ofir Lindenbaum's Research Areas.md` (a Gemini note) credits the RADAR
benchmark to his lab; it is by [Gu et al.](https://arxiv.org/abs/2506.08249).

## Not covered by either advisor

- Per-answer standardization (W6), the sign rule (W9), the refit bootstrap (W11), and the improvement condition
  itself (W3).
- These follow from our own model (`FRAMING_THEORY.md`: P1 for W6, P5's abstain rule for W9, P2/P5 for W3) and
  from evaluation hygiene (Stage 4 of the plan).

## Where each tool enters the plan

| Tool | Plan stage |
|---|---|
| Block CCA | Candidate gate estimator in the synthetic selection (Stage A) |
| mmDUFS with position as a view | Diagnostic in Stage 1 |
| Partition First, spectral stability | Stage 4 (grouping stability) |
| Pareto Testing and the calibrated count rule | "Calibrated" track after the label-free gate is evaluated; compare the two gates |
