# Stage 3 — Rényi-order combination (Claude, 2026-09-13)

Protocol: `docs/experiments/RENYI_VIEW_FUSION_V2.md`. Results: `results/renyi_view_fusion_v2/fast_pass/`
(12 non-Joint arms, all 13,769 answers) and `results/renyi_view_fusion_v2/joint_pass/` (Joint L-SML arm,
running). Same frozen contract as Stages 1–2 (v3 labels, FOLDS_V2, top-10 token-mean readout, external
mean-entropy gate q=0.3, PRMScore q=0.8 held folds, 10,000-draw paired source-group bootstrap).

**Framing (unchanged).** We have not yet demonstrated a consistent overall advantage from learned fusion. The
contributions of representation, optimization, normalization and readout remain partly entangled.

## Question and design

Omri (2026-09-13): does a *combination* of several Rényi orders beat a single entropy? The v1 prototype grid
{0.5, 1, 2, 4, ∞} had three distinct directions only (H2/H4/H∞ pairwise |ρ| > 0.99 on every smoke answer).
v2 spends the grid where the views differ: α ∈ {0.1, 0.25, 0.5, 1, 2, ∞}; α = 4 dropped (duplicate), α = 0
constant (log 15). Banks R6 (six orders) and R6_sel (+ selected-token block); solvers equal / IU-PCR /
shrinkage IU / Joint L-SML (declared groups: tail orders < 1, head orders ≥ 1, selected token). Primary
contrasts (97.5 %): R6__iu, R6__equal, R6_sel__iu each against view__H1 (= frozen entropy15, reproduced
exactly on PB and within-AUC; pooled AUC differs by 8.4e-9 through rank ties, recorded), and R6__iu
against R6__equal.

Redundancy on the full population (fraction of answers with |Pearson| > 0.99): only H1–H2 (1.1 %) and
H2–H∞ (4.2 %); > 0.95: H0.5–H1 (99.9 %), H1–H2 (100 %), H2–H∞ (100 %), H0.1–H0.25 (80 %), H0.25–H0.5 (82 %).
No near-constant view; condition number of R6 median 1.4e4 (max 6.8e4), of the tail block median 441.
Coverage 100 % for every arm, no declared failures.

## Results — fast pass (PB all-8 % / PRMB within-AUC / pooled AUC / PRMScore q0.8)

| Arm | PB | within | pooled | PRMScore |
|---|---:|---:|---:|---:|
| view H0.1 | 35.37 | **0.7425** | 0.7152 | **0.6331** |
| view H0.25 | 35.52 | 0.7414 | 0.7131 | 0.6322 |
| view H0.5 | 35.55 | 0.7371 | 0.7088 | 0.6296 |
| view H1 = entropy15 (reference) | 35.44 | 0.7301 | 0.7027 | 0.6254 |
| view H2 | 35.59 | 0.7242 | 0.6972 | 0.6213 |
| view H∞ | 35.77 | 0.7178 | 0.6924 | 0.6158 |
| view sel1 (selected surprisal) | 22.84 | 0.6975 | 0.7121 | 0.6119 |
| R6 equal | 35.62 | 0.7325 | 0.6776 | 0.5968 |
| R6 IU-PCR | 35.62 | 0.7310 | 0.6777 | 0.5975 |
| R6_sel equal | 35.61 | 0.7352 | 0.6756 | 0.5949 |
| R6_sel IU-PCR | **35.78** | 0.7338 | 0.6820 | 0.6028 |
| R6_sel shrinkage IU | 35.75 | 0.7338 | 0.6814 | 0.6024 |
| R6_sel Joint L-SML | running | | | |
| varentropy15 raw (Step 339, reproduced) | 35.96 | 0.7378 | 0.7101 | 0.6258 |
| varentropy15 contributions IU (Step 339, reproduced) | 35.35 | 0.7468 | 0.7103 | 0.6227 |
| direct-probability IU (17 inputs, frozen) | 34.50 | 0.7328 | 0.7038 | 0.6205 |

Primary contrasts (97.5 %):

| Contrast | PB pp | within-AUC |
|---|---:|---:|
| R6 IU − H1 | +0.18 [−0.28, +0.65] | +0.0009 [−0.0002, +0.0020] |
| R6 equal − H1 | +0.18 [−0.24, +0.60] | +0.0024 [+0.0012, +0.0036] |
| R6_sel IU − H1 | +0.34 [−0.14, +0.82] | +0.0037 [+0.0023, +0.0051] |
| R6 IU − R6 equal | −0.00 [−0.20, +0.20] | −0.0015 [−0.0023, −0.0007] |

Secondary (95 %): single views against H1 — within-AUC is monotone in α: H0.1 +0.0124 [+0.0100, +0.0149],
H0.25 +0.0113, H0.5 +0.0070, H2 −0.0060, H∞ −0.0123; PB intervals all include zero (−0.07 … +0.33 pp).
Every fused arm is *below* the best single view on within-AUC: R6 IU − H0.1 = −0.0115 [−0.0140, −0.0091],
R6_sel IU − H0.1 = −0.0087 [−0.0110, −0.0065]; PB intervals include zero. Fused arms against varentropy15:
R6 IU −0.34 pp [−1.62, +0.93], within −0.0068 [−0.0110, −0.0025]; R6_sel IU −0.18 pp, within −0.0040
[−0.0080, +0.0001]. Shrinkage ≡ IU (alpha at 1 on 16 %; deltas ≤ 0.03 pp).

Declared post-hoc contrasts (`POSTHOC_CONTRASTS.json`, 95 %, chosen after seeing the table):
H0.1 − varentropy15: PB −0.59 [−1.72, +0.53], within +0.0047 [+0.0016, +0.0079], PRMScore +0.0073;
H0.1 − varentropy15-IU: PB +0.02 [−0.86, +0.90], within −0.0043 [−0.0067, −0.0018];
H0.1 − direct-probability IU: PB +0.87 [+0.08, +1.65], within +0.0098 [+0.0063, +0.0134];
H0.1 − H0.25: PB −0.15 [−0.40, +0.09], within +0.0011 [+0.0003, +0.0019].

## Reading

1. **Combining orders does not beat the best single order.** All fused arms sit between H1 and H0.1 on
   within-AUC (+0.001 … +0.004 over H1, −0.009 … −0.012 below H0.1) and gain nothing on PB with intervals
   including zero. Learned weights (IU) do not exceed equal weights (R6: −0.0015 within, 0.00 pp PB).
2. **The tail-sensitive single order H0.1 is the best answer-local single stream so far on within-AUC
   (0.7425) and PRMScore (0.633).** It exceeds entropy on within-AUC by +0.012 with an interval excluding
   zero, and varentropy15 by +0.005 (excluding zero), but not on PB (−0.07 / −0.59 pp, intervals include
   zero). It matches the best PRMScore in the table (varentropy50 0.633) and is above every learned row.
   This is a representation finding, not a fusion finding, and a single-endpoint lead: it is not a winner
   under the two-endpoint rule.
3. **Normalization effect, again.** The fused arms lose 0.03 pooled AUC and 0.03 PRMScore relative to their
   own inputs because per-answer z-scoring removes the between-answer scale that pooled AUC and the
   PRMScore threshold use; within-answer ranking and PB decisions are unaffected. Any comparison of fused
   rows on pooled/PRMScore against raw single views is confounded by this.
4. Joint L-SML on R6_sel (tail/head/selected groups) is scored separately (≈2.2 s/answer at 4 workers);
   its fit quality on the smoke (converged 56 %, multistart PASS 56 %, condition > 1e12 on 11 %) is the
   same pattern as Stage 2; its result will be appended to the table when complete.

Status: fast pass COMPLETE. Independent replay review PASS (`fast_pass/RESULT_REVIEW.json`: 13,769 answers
replayed, 165,228 checks, 16 metric bundles re-derived; the evaluate-only driver change is recorded and
accepted against the checkpoint's stored hash). Figure review PASS WITH CAVEATS (`fast_pass/REVIEW_FIGURES/`,
5 PNG, checks (a)–(j) all pass; caveats: Joint arm absent from this pass; near-collinear R6 columns;
shrink alpha at 1 on 16 %; fused pooled/PRMScore below the entropy row through per-answer normalization).
Joint pass RUNNING (`joint_pass/`, ≈2.1 s/answer at 4 workers; ≈8 h projected on a contended machine).
