# Peer review: PRM boosting positioning handoff (2026-10-24)

**Target.** `docs/research_notes/positioning_peer_review_2026-10-23/PEER_REVIEW_HANDOFF.md` and
`PRM_BOOSTING_POSITIONING.html` (page version 4), branch `claude/prm-boosting` @ `976845d5`.
**Reviewer.** Claude, cloud session, on Omri's request. Independent of the session that wrote the handoff.
**Method.** Table re-checks from the committed CSVs; arXiv / ACL pages read with a web fetch tool
(direct PDF download was blocked by the network policy, so `pypdf` was not run). No experiment was run.

---

## 0. Verdict

| item | verdict |
|---|---|
| BR2-PRM blocks publication | **No.** Confirmed. |
| Numbers on the page vs result tables | **All rows I checked match** (Section 2). |
| CROP "tie" wording | **Correct**, but explain it with the standard error, not the SD (Q3). |
| "+2.39 under the same rule" | **Correct and same pool.** Stronger support exists: equal-count k=2 gives +2.64 (Q4). Add it. |
| CARE description | **Two claims not found in the source** ("tuned on 15% labels", "App. E.8 uses the test set"). Soften or verify (Q2). |
| New competitor after 2026-10-21 | **None found** that does label-free step-level fusion (Q5). |
| Page overstatement | **Three small fixes** (Q6). |

---

## 1. Answers to the six reviewer questions

### Q1. BR2-PRM bibliography (Nadler / Parisi / Jaffe / Kluger)

| check | result |
|---|---|
| Nadler, Parisi, Jaffe, Kluger in references | **0 hits** |
| Dawid & Skene (1979) | present; App. D says the model "draws inspiration from" it |
| "unsupervised", "label-free", "spectral", "expectation-maximization" | 0 hits ("weak-supervision" appears once, in Limitations) |
| Fit procedure (App. D.2) | posterior = prior + **counts of observed scores on calibration data with known latent states**. No EM. |

- The handoff is confirmed: supervised naive Bayes with a Dawid-Skene framing.
- Caveat: read through a web fetch tool, not a local `pypdf` grep. The command in Section 7 of the handoff
  still needs to run on a machine that can download the PDF.

### Q2. CARE references and appendix

| claim in handoff / page | what I found in arXiv HTML v2 | status |
|---|---|---|
| No Nadler, Parisi, Jaffe | not in the reference list; FUSE also absent | **confirmed** |
| Three-group partition | Alg. 2: "partition judges into three disjoint groups" with cross-group sparse entries ≤ ε. **The search method is not stated** | handoff "unverified" is correct; the paper does not specify it |
| CARE-SVD weights | leading eigenvector of the low-rank part; App. C.4 heuristic w ← λ\*u\* − Σ λᵢuᵢ for the non-orthogonal case | partly answered |
| "Hyperparameters tuned on a 15% labeled split" | **no "15%" in the text.** Labels appear only as (a) an optional small human anchor set, (b) a 10% validation split for greedy judge selection (Sec. 5.3). γ_n and τ selection is not described | **not found** |
| "App. E.8: best-factor choice uses the test set, optimistic" | the fetch tool could not reach App. E; no "test set" / "optimistic" match in the parts it read | **not verified** |

- **Action:** on the page, change CARE's access cell from "label-free core tuned on 15% labels" to
  "label-free core; optional human anchor set; 10% validation split for judge selection". Keep the E.8
  remark only if someone reads App. E directly.

### Q3. CROP seed SD and the word "tie"

| row (Qwen-7B, CROP grid) | kept % | seed SD | n seeds | SE of mean |
|---|---|---|---|---|
| L-SML, train-only fit (`CROP_C4_INDUCTIVE.csv`, `CHAIN_N16 (inductive, C4)`, fixed101_minmax) | 22.334 | 1.946 | 10 | 0.62 |
| L-SML, quantile grid | 23.182 | 1.737 | 10 | 0.55 |
| CROP Trace+PRM (published) | 21.8 | unknown | 10 (seeds 2806–2815) | unknown |

- Margin 22.33 − 21.8 = 0.53 = 0.27 × SD. The handoff value is correct.
- **"Tie" is the right word**, but the SD is the wrong yardstick for a difference of means. The SE of our
  mean is 0.62, so the margin is below 1 SE even before CROP's own noise. Suggested text:
  "0.5 points, below one standard error of our 10-seed mean (0.6); CROP's seed spread is not reported."
- **Second reason for "tie":** our reproduction is not CROP's exact pipeline (approximate subset of 4,995
  traces, our own features). Our own supervised analogue (Telemetry16+PRM LR) keeps 25.2% under the same
  protocol. So under one protocol, label-free L-SML is about 3 points **below** supervised fusion. The page
  says this under "Not supported", which is correct. The CROP section title "matched without them" is
  fine only as "matched the published number".
- 81–92% recovery claim checked: L-SML / Telemetry13+PRM LR (quantile) = 7B 0.92, 72B 0.89, PURE 0.82. OK.
  (RLHFlow 0.74, Math-Shepherd 0.84 are outside the stated range, which the page correctly limits to Qwen and PURE.)

### Q4. "+2.39 under the same rule": same answers, same evaluator?

| row (`COMP_PRISM_T1_T2.csv`, non-control 6,211 answers) | TP+FN | FP+TN | PRMScore | flag rate |
|---|---|---|---|---|
| PRM_Q80_PROB, GAP | 70,222 | 13,149 | 68.008 | 13.2% |
| CHAIN_N16, GAP | 70,222 | 13,149 | 70.399 | 18.3% |

- Same step counts, same pool, same file, same evaluator run. Difference 2.391. `STATUS_CONTRASTS.csv`
  gives the bootstrap CI [+1.90, +2.90] (707 source groups, 2,000 draws). **Confirmed.**
- **New caveat:** GAP picks a different count for each scorer. CHAIN flags 18.3% of steps, the PRM 13.2%.
  So "+2.39 under the same rule" mixes better ranking with a different flag count.
- **The clean control already exists** in `STATUS_MAIN.csv`, column `K2` (constant k=2 for every scorer,
  flag rate 14.9% for all arms). It shows the gain is not a count effect:

| PRM | PRM (best scale) K2 | L-SML (CHAIN_N16) K2 | Δ K2 | Δ GAP (page) | Δ Q80 |
|---|---|---|---|---|---|
| Qwen-7B | 68.16 | 70.80 | **+2.64** | +2.39 | +0.93 |
| Qwen-72B | 69.38 | 71.90 | **+2.51** | +1.60 | −0.14 |
| PURE | 64.56 | 69.60 | **+5.03** | +3.27 | +2.57 |
| RLHFlow | 58.92 | 67.44 | **+8.52** | +6.86 | +7.38 |
| Math-Shepherd | 59.32 | 66.77 | **+7.45** | +6.18 | +6.64 |

(Q80 column: CHAIN_N16 Q80 minus the better of PRM_Q80 / PRM_Q80_PROB under Q80.)

- **Recommendation:** add the K2 row next to GAP on the page. It answers the reviewer worry
  "your rule flags more steps for the fused score", and it is the "constant-k reference" that the page
  already lists under "Open before a paper". That item is done, not open. No bootstrap CI for the K2
  contrast exists yet; it is CPU-only from the existing score npz files.
- **Caveat that stays:** under Q80 (also equal count, 20%), Qwen-72B is −0.14. So the claim "improves
  every one of 5 PRMs under the same rule" holds for GAP and K2, not for Q80.

### Q5. New step-level label-free PRM ensembles after 2026-10-21

Searched on 2026-10-24 (web search: label-free / unsupervised step-level PRM ensemble; FUSE follow-ups).

| hit | what it is | competitor? |
|---|---|---|
| uPRM, arXiv 2605.10158 | trains a PRM without step labels | no (already in the registry) |
| VeriGate, arXiv 2605.30451 (UMD) | gates between a verifier and a PRM inside GRPO; no PRMBench; no FUSE / Nadler | no |
| "Learning Discriminative PRMs without Step Labels" (Kim, Liu, Wei; OpenReview df3p10k2kq, SPOT 2026) | trains a PRM from outcome labels with a latent-step aggregator; claims ProcessBench and PRMBench results, no numbers on the page | no (trained, one model), but **add to the registry** as a PRMBench row candidate |
| Unsupervised Multimodal Reward Modeling, 2610.12707 | already in the handoff | no |

- Nothing found that fuses several views of one step without labels. Search limits: two web searches plus
  one targeted; no Semantic Scholar citer sweep of FUSE or CARE (the API was not reachable from here).

### Q6. Does the page overstate or understate?

| # | location | issue | suggested fix |
|---|---|---|---|
| 1 | Lede and "Supported" bullet 2: "without retraining **or labels**, the same PRM reaches PRISM's PRMScore" | 68.01 is on the probability scale, which `STATUS_MAIN.csv` marks "scale picked with labels after scoring". On the logit scale the PRM under GAP is 66.37. The probability scale is the PRM's native output, so the claim is defensible, but the label-picked flag must be disclosed | add "(on the PRM's native probability scale; 66.4 on the logit scale)" |
| 2 | "Supported" bullet 1: "improves every one of 5 PRMs under the same rule" | true for GAP and K2; false for Q80 on Qwen-72B (−0.14) | "under the GAP rule and under a constant 2 flags per answer" |
| 3 | CARE row: "tuned on 15% labels" | not in the source (Q2) | see Q2 |
| 4 | CROP bullet: "a quarter of our seed SD" | correct, but SD is not the test for a mean difference | use SE (Q3) |
| 5 | "Open before a paper": constant-k reference | already computed (`K2` column) | move to "Supported" with the table in Q4 |
| 6 | PRISM section | understates slightly: the page frames fusion as adding only on top of the rule; K2 shows +2.6 at equal count | optional: one line with the K2 number |

Rows I re-checked and found correct (no change): matched-TPR FPRs for all 5 PRMs (PURE 49.82 vs 52.24,
RLHFlow 57.77 vs 77.14, Math-Shepherd 57.28 vs 81.34, Qwen-72B 45.87 vs 45.25); CROP other-PRM rows
(72B 22.25 vs 21.28, PURE 20.68 vs 22.06, RLHFlow 16.43 vs 17.81, Math-Shepherd 17.42 vs 18.58);
plain 72B PRM risk 7.41%; GAP-rule gains +1.60 / +3.27 / +6.86 / +6.18 against the README.

Not re-checked: ARES, SLIDER, Socratic and leaderboard rows; the "7 of 10 banks" PROGRESS.md entry;
FUSE's "7 of 10 settings" (FUSE PDF not read in this review).

---

## 2. Suggested next CPU steps (none run)

1. Bootstrap CI for the K2 contrast (CHAIN_N16 vs PRM, 5 PRMs), same source-group bootstrap as
   `STATUS_CONTRASTS.csv`. Needs the score npz files (Drive / local Stage-2 machine; not in git).
2. Run the handoff's `pypdf` grep on BR2-PRM on a machine with PDF access.
3. Read CARE App. E directly before the page names its label use.

## 3. Sources

- ACL 2026 BR2-PRM / PRMBench-V PDF: <https://aclanthology.org/2026.acl-long.2068.pdf>
- CARE arXiv HTML v2: <https://arxiv.org/html/2603.00039v2>
- FUSE: <https://arxiv.org/pdf/2604.18547>, ICML poster <https://icml.cc/virtual/2026/poster/64617>
- uPRM: <https://arxiv.org/pdf/2605.10158>
- VeriGate: <https://arxiv.org/pdf/2605.30451>
- Kim, Liu, Wei: <https://lacuna.tiptreesystems.com/work/learning-discriminative-process-reward-models-without-step-labels/wrk_ff54e42ab1249be04af2843c57c1b0cd>
- Unsupervised Multimodal Reward Modeling: <https://arxiv.org/abs/2610.12707>
- Repo files: `results/prm_boost_collection_v1/tables/{COMP_PRISM_T1_T2,COMP_PRISM_MATCHED_TPR,STATUS_MAIN,STATUS_CONTRASTS,CROP_C1_MAIN,CROP_C4_INDUCTIVE}.csv`, `results/error_count_label_free_v1/README.md`.
