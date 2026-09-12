# Stage 2 interim — varentropy-expansion (cross-rank) fusion v1, fast pass (Claude, 2026-09-12)

Status: **Stage 2 OPEN.** The 19 non-Joint arms are scored on all 13,769 answers with independent
replay PASS (`results/varentropy_expansion_fusion_v1/fast_pass/RESULT_REVIEW.json`: 261,611 arm
checks, 69 input/code hashes). The two Joint L-SML arms on the primary banks are running
(`joint_pass/`, ~14–20 h estimate at 4 workers); the two Joint arms on the non-selected banks are
deferred for compute (pending, not dropped). The supervised step-level diagnostic is complete (see below).
Protocol: `docs/experiments/VARENTROPY_EXPANSION_FUSION_V1.md`. Figures and independent checks:
`fast_pass/REVIEW_FIGURES/` (6 PNG, `REVIEW.md`, `CHECKS.json`; every recomputed macro and the
primary within-AUC delta match the saved metrics).

Framing: we have not yet demonstrated a consistent overall advantage from learned fusion; the
contributions of representation, optimization, normalization and readout remain partly entangled.

## Definition (Omri, 2026-09-12)

With the frozen top-15 convention q_i = p_i/(Σp+1e-12), s_i = −log(q_i+1e-12):
V = Σ_i q_i s_i² − Σ_{i,j} q_i q_j s_i s_j. Columns: D_i = q_i s_i² (15), P_ii = (q_i s_i)² (15),
P_ij = q_i q_j s_i s_j for i<j (105, each pair once, weight 2 in the identity), selected-token
block a, a², a³. Identity check Σ D − Σ P_ii − 2 Σ P_ij = varentropy15 holds on every answer
(max discrepancy 8.5e-12); the identity-weighted fixed fusion reproduces the Step-339 `k15__raw`
step scores (2.9e-12); the historical bank arms replay Step 339 exactly (0.0).

Banks: B1_hist (Step 339 contributions, unchanged), B2d_sel = D + P_ii + SEL (33), B2_sel =
D + P_ii + P_ij + SEL (138); B2d / B2 without SEL as secondary. Solvers: identity-sign equal
(anchor-oriented), oriented equal, IU-PCR (frozen defaults; P ≥ 64 uses the analytic pair path),
shrinkage IU (joint target, LW alpha), Joint L-SML (pending). Orientation anchor: the answer's own
raw varentropy15. Readout top-10 token mean, earliest argmax, external mean-entropy q=0.3 gate.

## Results (full population; PB all-8 %, PRMB within-answer AUC, PRMScore q0.8)

| Arm | PB | within | PRMScore |
|---|---:|---:|---:|
| Token entropy (reference) | 35.44 | 0.7301 | 0.625 |
| Varentropy15 raw = B1_hist raw (reference) | 35.96 | 0.7378 | 0.626 |
| B1_hist equal / IU (Step 339, reproduced) | 35.60 / 35.35 | 0.7470 / 0.7468 | 0.613 / 0.623 |
| B2d_sel: identity-sign equal / oriented equal / IU / shrink | 32.80 / 34.73 / 34.75 / 34.66 | 0.7321 / 0.7344 / 0.7336 / 0.7314 | 0.602 / 0.618 / 0.618 / 0.618 |
| B2_sel: identity-sign equal / oriented equal / IU / shrink | 34.08 / 34.38 / 33.95 / 34.06 | 0.7241 / 0.7291 / 0.7270 / 0.7274 | 0.616 / 0.618 / 0.616 / 0.617 |
| B2d (no SEL): identity-sign equal / IU | 36.06 / 34.73 | 0.7464 / 0.7332 | 0.614 / 0.617 |
| B2 (no SEL): identity-sign equal / IU | 34.07 / 33.97 | 0.7243 / 0.7270 | 0.616 / 0.616 |

Coverage 13,769/13,769 for every arm; zero declared or undeclared failures.

**Primary contrast (pre-registered, 97.5%): B2_sel__iu − B2d_sel__iu = PB −0.80 pp [−1.31, −0.30];
within −0.0066 [−0.0089, −0.0044]** (26 answers gained, 60 lost). Adding the 105 cross-rank
products lowers both endpoints under the primary solver. Same direction under shrinkage (−0.60 pp,
−0.0040), oriented equal (−0.35 pp, −0.0053) and IU without SEL (−0.76 pp, −0.0062), all with 95%
intervals excluding zero. The only positive PB point is identity-sign equal (+1.27 pp, interval
includes zero) and it loses within-AUC (−0.0080, excludes zero); that arm is also the least stable
across banks (36.06 → 32.80 → 34.08), so its rise reads as recovery from a collapsed B2d_sel point.

Bank versus reference: every B2_sel / B2d_sel arm is below entropy and varentropy15 on ProcessBench
(B2_sel arms −1.07 to −1.49 pp vs entropy with intervals excluding zero; B2d_sel IU/shrink/oriented
0.69–0.79 pp below with intervals including zero; all 1.2–3.2 pp below varentropy15, excluding
zero). On PRMBench within-AUC the diagonal solvers beat entropy (B2d_sel IU +0.0035 [+0.0005,
+0.0065]) but none reaches varentropy15 (0.7378) or the Step-339 contribution arms (0.7468–0.7470).
The highest expanded point, B2d identity-sign equal without SEL (36.06 / 0.7464), is +0.10 pp
[−0.74, +0.96] vs varentropy15 on PB and +0.0086 [+0.0060, +0.0112] on within; it is a secondary,
fixed-sign arm that re-expresses the Step-339 equal level on the diagonal bank — an observation to
carry forward, not a candidate.

Architecture within a bank: the solver moves the endpoints far less than the bank does (B2d_sel
IU, shrink and oriented equal within 0.1 pp / 0.003); shrinkage is indistinguishable from IU on the
pair banks (+0.11 pp [−0.05, +0.28]) because alpha clips to 1.0 on 98.9% of answers. IU beats the
identity-sign equal arm on B2d_sel PB (+1.95 pp [+0.37, +3.55]) and on B2_sel within (+0.0029). The
Joint-vs-IU and Joint-vs-shrink pairs are pending.

Where the fitted weight goes (B2_sel IU): the P_ij columns absorb 82% of the absolute weight, the
fused score's correlation with the varentropy anchor drops from 0.80 (B1_hist IU) to 0.40, and the
fitted P_ii / P_ij signs are positive where the identity has −1 / −2 — the expanded fit is a
different combination, not a re-weighting of varentropy. The selected-surprisal block is inert
under IU (|Δ| ≤ 0.02 pp).

## Disclosures required by the protocol

Shrinkage alpha clipped at 1.0 on 86.7% (B2d), 86.6% (B2d_sel), 98.9% (B2, B2_sel) of answers;
n<p fits (T < active columns) on 1 answer for the 33-column banks and 648 / 705 (4.7% / 5.1%) for
the 135/138-column banks; the identity-sign equal arm is anchor-flipped on 99.9% of answers on the
pair banks (mean pre-orientation anchor correlation −0.37) and is reported as "identity-sign equal
weights, anchor-oriented"; the B1_hist equal arm is the unoriented Step-339 arm; IU/shrink flip on
0.03–0.14% of answers. Fit runtime for 13,769 answers: 118 s (B1_hist IU) to 207 s (B2_sel shrink).
Fusion is answer-local and unlabeled; the gate and PRMScore calibration are external and frozen;
cached development data, not an untouched test; no localization improvement is claimed.

## Reading for the letter's cross-rank proposal

The specific claim tested — that exposing the varentropy expansion's cross-rank products to learned
fusion lets the fusion re-weight them usefully — is not supported at this benchmark under IU-PCR,
shrinkage IU-PCR or equal weights: the products reduce both ProcessBench exact localization and
PRMBench within-answer ranking relative to the diagonal terms, and no expanded bank reaches the raw
varentropy or the Step-339 contribution arms. What remains open in Stage 2: the Joint L-SML arms
(pending), the supervised matched diagnostic (pending; a diagnostic, not a ceiling), and the
secondary observation that the identity-sign diagonal arm without the selected block tracks the
contribution-equal level.

## Supervised matched diagnostic — COMPLETE (other-answer labelled access; not a ceiling)

Linear token score on the answer-locally standardized bank, the same top-10 token-mean step
readout, class-balanced step-level BCE against step labels (PB: prefix 0 / first-error 1 / later
steps excluded; PRMB step labels), per-cell 5 source-group folds, ridge 0.01; 90 fits, all FIT (67
converged, 23 at the iteration limit, 0 stalled). Same gate, readout and evaluator.

| Arm | PB all-8 % | PRMB within | PRMScore |
|---|---:|---:|---:|
| supervised B2d_sel | 36.03 | 0.7530 | 0.629 |
| supervised B2_sel | 36.30 | 0.7531 | 0.629 |

Paired (frozen evaluator, 10,000 draws; `supervised/CONTRASTS.json`): B2_sel − B2d_sel **+0.27 pp
[−0.53, +1.08], within +0.0001 [−0.0009, +0.0010]** (97.5%) — the cross-rank products add nothing
even when the coefficients are fitted with labels. Supervised B2_sel vs unsupervised B2_sel IU:
+2.35 pp [+1.17, +3.52], within +0.026 [+0.023, +0.030]; vs token entropy: +0.85 pp [−0.21, +1.94],
within +0.023 [+0.019, +0.027]; vs varentropy15: +0.34 pp [−0.80, +1.48], within +0.015 [+0.011,
+0.019]. Reading: labels recover the loss the expanded bank suffers under answer-local IU and lift
within-answer ranking above every unsupervised row, but on ProcessBench exact localization the
labelled linear score on this bank does not separate from entropy or varentropy. This is a
diagnostic with different access; it bounds neither the unsupervised methods nor the quadratic
feature class in general.

## Stage 3 (Rényi) — prototype status, design pending review

Implemented and tested (14 tests, `docs/experiments/RENYI_VIEW_FUSION_V1_DRAFT.md`, smoke 27/27,
0 failures, ~13 min projected full run). Redundancy on the smoke answers: H2, H4 and H∞ are one view
(|r| ≥ 0.99, numerically −log p₁), H1 sits between them and H0.5, and H0.5 is the only order with
distinct information (closest to the varentropy anchor, r ≈ 0.88); no view is constant; condition
number of the 5-view bank ~5e4; IU-PCR sits at its residual ceiling (g2 = var_y) on all 27 answers
for the 5-view bank. Design questions for the post-review stage: reduce the order set or use
differences (H0.5−H1, H1−H2, H2−H∞); reference arm for IU given the ceiling; SEL block handling.
