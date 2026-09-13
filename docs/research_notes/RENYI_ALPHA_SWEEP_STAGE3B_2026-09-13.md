# Stage 3b — Rényi-order sweep and escort varentropy (Claude, 2026-09-13)

Authorized by Omri (2026-09-13): "you may sweep to find the optimal alpha", and "is there also a varentropy that
uses these weights?". Results: `results/renyi_alpha_sweep_v1/` (31 single-view arms, all 13,769 answers, frozen
contract of Stages 1–3: v3 labels, FOLDS_V2, top-10 token-mean readout, external mean-entropy gate q=0.3, PRMScore
q=0.8 held folds, 10,000-draw paired source-group bootstrap, 95 % on every contrast; no pre-registered primary).
Code: `spectral_utils/renyi_alpha_sweep.py`, `scripts/run_renyi_alpha_sweep_v1.py` (v2 harness with the sweep
module swapped in), `scripts/review_renyi_alpha_sweep_v1.py`, `scripts/renyi_alpha_sweep_selection.py`.

**This is a label-guided hyperparameter sweep on development data.** The curves and any chosen alpha are
development evidence; a selection rule must be frozen before an untouched confirmation. Framing unchanged: no
consistent overall advantage from learned fusion has been demonstrated; representation, optimization,
normalization and readout remain partly entangled.

## Definitions (frozen top-15 head: q = p/(Σp+1e-12), s = −log(q+1e-12))

* Rényi entropy `H_α = log(Σ q^α)/(1−α)`, α ∈ {0.001, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75,
  1, 1.5, 2, 3, 4, 8, ∞}, plus the analytic α→0 limit ordering `H0lim = mean_i log q_i` (from
  H_α = log K + α (mean log q + log K) + O(α²)). Natural high-is-risk sign, no flip.
* Escort varentropy `VE_α = Σ w_i s_i² − (Σ w_i s_i)²`, `w ∝ q^α` (α = 0: uniform weights over the head; α = 1:
  exactly the frozen varentropy15, asserted to 1e-9 on every endpoint; α → ∞: 0, excluded),
  α ∈ {0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 8}. The v2 smoke showed VE at small α anticorrelates with
  entropy (uniform-weighted surprisal spread is large when the head is peaked), so each VE arm is oriented per
  answer by the sign of its correlation with the answer's own varentropy15 (the label-free anchor rule of the
  fusion arms); the flip is recorded. On the full population VE_0, VE_0.1, VE_0.25 flipped on 100 % of answers
  (a fixed negative sign is equivalent), VE_0.5 on 1.7 %, VE_0.75 … VE_2 on 0 %, VE_3/4/8 on ≤ 1.4 %.

## Results (PB all-8 % / PRMB within-AUC / pooled AUC / PRMScore q0.8; coverage 100 %, no failures)

Rényi entropy family:

| α | PB | within | pooled | PRMScore |
|---|---:|---:|---:|---:|
| 0 (limit, mean log q) | 35.53 | **0.7440** | **0.7162** | 0.6334 |
| 0.001 / 0.01 / 0.02 / 0.05 | 35.53 / 35.50 / 35.48 / 35.53 | 0.7440 / 0.7438 / 0.7437 / 0.7433 | 0.7162 / 0.7161 / 0.7161 / 0.7158 | 0.6334 (all) |
| 0.1 / 0.25 / 0.5 | 35.37 / 35.52 / 35.55 | 0.7425 / 0.7414 / 0.7371 | 0.7152 / 0.7131 / 0.7088 | 0.6331 / 0.6322 / 0.6296 |
| 1 (entropy15) | 35.44 | 0.7301 | 0.7027 | 0.6254 |
| 2 / 4 / 8 / ∞ | 35.59 / 35.80 / **35.88** / 35.77 | 0.7242 / 0.7207 / 0.7192 / 0.7178 | 0.6972 / 0.6942 / 0.6930 / 0.6924 | 0.6213 / 0.6177 / 0.6160 / 0.6158 |

Escort varentropy family:

| α | PB | within | pooled | PRMScore | flipped |
|---|---:|---:|---:|---:|---:|
| 0 (uniform weights; sign −) | 35.57 | **0.7534** | **0.7231** | **0.6355** | 100 % |
| 0.1 | 35.77 | 0.7482 | 0.7201 | 0.6340 | 100 % |
| 0.25 | 34.89 | 0.7352 | 0.7089 | 0.6316 | 100 % |
| 0.5 (orientation boundary) | 36.20 | 0.6788 | 0.6341 | 0.5731 | 1.7 % |
| 0.75 | **36.76** | 0.7323 | 0.7043 | 0.6187 | 0 % |
| 1 (= varentropy15) | 35.96 | 0.7378 | 0.7101 | 0.6258 | 0 % |
| 1.5 / 2 / 3 / 4 / 8 | 35.46 / 35.01 / 34.98 / 34.06 / 34.17 | 0.7333 / 0.7259 / 0.7136 / 0.7061 / 0.6824 | 0.7049 … 0.6455 | 0.6248 … 0.5909 | ≤ 1.4 % |

References (same evaluator): entropy 35.44 / 0.7301 / 0.7027 / 0.6254; varentropy15 35.96 / 0.7378 / 0.7101 /
0.6258; varentropy15-contributions IU 35.35 / 0.7468 / 0.7103 / 0.6227; direct-probability IU 34.50 / 0.7328 /
0.7038 / 0.6205.

Paired contrasts (95 %, PB pp / within-AUC):

| Contrast | PB | within |
|---|---:|---:|
| H0lim − entropy (H1) | +0.09 [−0.62, +0.81] | +0.0139 [+0.0114, +0.0164] |
| H0lim − H0.1 | +0.16 [−0.06, +0.40] | +0.0014 [+0.0009, +0.0020] |
| H0lim − varentropy15 | −0.43 [−1.52, +0.68] | +0.0062 [+0.0031, +0.0094] |
| H0lim − varentropy15 IU | +0.19 [−0.67, +1.03] | −0.0029 [−0.0053, −0.0004] |
| H8 − H1 | +0.43 [−0.27, +1.13] | −0.0109 [−0.0131, −0.0088] |
| VE0 − varentropy15 (VE1) | −0.39 [−1.51, +0.69] | **+0.0156 [+0.0121, +0.0191]** |
| VE0 − entropy | +0.12 [−0.82, +1.07] | **+0.0233 [+0.0199, +0.0268]** |
| VE0 − H0.1 | +0.20 [−0.52, +0.88] | +0.0109 [+0.0087, +0.0130] |
| VE0 − VE0.1 | −0.21 [−0.67, +0.23] | +0.0052 [+0.0038, +0.0067] |
| VE0.75 − varentropy15 | +0.80 [−0.20, +1.81] | −0.0055 [−0.0089, −0.0020] |
| VE0.75 − entropy | +1.32 [−0.05, +2.71] | +0.0022 [−0.0034, +0.0079] |
| VE0.25 − VE0.5 / VE0.5 − VE0.75 | −1.31 [−2.90, +0.25] / −0.56 [−1.86, +0.74] | +0.0563 / −0.0535 (boundary artefact) |

Selection stability (`SELECTION.md`): H family — within-AUC argmax is the α→0 limit on every fold (H0lim or
α = 0.001; the two are identical to 4 decimals), cross-fitted within (choose on 4 folds, evaluate on the 5th) 0.7439 =
per-fold oracle; the region α ≤ 0.05 is flat (0.7433–0.7440). PB argmax is α = 8 on the whole set but the cell-wise
argmax scatters (α = 0.05, 0.15, 8, ∞) with cell differences ≤ 1.9 pp. VE family — within argmax is α = 0 on
every fold, cross-fitted 0.7534 = oracle; PB argmax α = 0.75 overall, cell-wise scattered (VE1, VE0.1, VE4,
VE0.5, VE0.75, VE0.25) with cell differences up to 2.8 pp.

## Reading

1. **There is no interior optimum for the Rényi order: within-answer ranking improves monotonically as α → 0,
   and the α → 0 limit (mean surprisal over the top-15 head, i.e. "how flat is the head", ignoring the weights)
   is the best member of the family** (0.7440; +0.0139 over entropy, +0.0062 over varentropy15, intervals
   exclude zero). PB does not move (all intervals include zero; the whole family spans 35.4–35.9).
2. **The escort varentropy answers Omri's question: with uniform weights (α = 0) the spread of surprisals across
   the head, with its sign reversed (low spread = flat head = risk), is the best answer-local single stream on
   all three PRMB endpoints** (within 0.7534, pooled 0.7231, PRMScore 0.6355), above varentropy15 by +0.0156
   within (interval excludes zero) and above every learned row of Stages 1–3 (best learned within 0.7473,
   supervised). Its PB (35.57) is not distinguishable from varentropy15 (−0.39 [−1.51, +0.69]).
3. **VE_0.75 has the highest PB of any answer-local stream measured in Stages 1–3 (36.76 %)**, +0.80 pp over
   varentropy15 and +1.32 pp over entropy, but both intervals include zero, and its within-AUC is below
   varentropy15 (−0.0055, excluding zero). A PB-only, non-significant lead with a within-AUC cost.
4. Two label-free views therefore pull in different directions: VE_0 for within-answer ranking / PRMScore,
   VE_0.75 (or varentropy15) for PB. Neither satisfies the two-endpoint rule; neither is a winner.
5. Mechanism note (descriptive): both leaders are functions of the head's surprisal spread rather than of the
   probability-weighted moments; α < 1 de-emphasises the top token. This is consistent with Stage 3's finding
   that within-AUC rises as α falls, and with the readout/gate remaining the PB bottleneck.

## Status

Scoring and evaluation COMPLETE. Independent replay review PASS (`RESULT_REVIEW.json`: 13,769 answers, 426,839
checks, 35 metric bundles re-derived, 70 manifest hashes). Figure review PASS WITH CAVEATS (`REVIEW_FIGURES/`, 5 PNG,
checks (a)–(k) pass; caveats: label-guided development sweep, nothing promoted; no H-vs-H1 PB interval excludes
zero; near-limit views are rank-equivalent to the limit; VE small-α arms rest on a 100 % anchor flip, i.e. a fixed
negative sign). Joint L-SML pass of Stage 3 still running separately.
