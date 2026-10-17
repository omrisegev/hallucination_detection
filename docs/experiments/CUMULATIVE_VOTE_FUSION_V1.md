# Cumulative-vote fusion of first-error localizers (binary L-SML over "error ≤ n")

Status: COMPLETE, retrospective, development-only. Started and finished 2026-09-20 (Step 423).
Result: `results/cumulative_vote_fusion_v1/REPORT.md`.

## Question

Omri (2026-09-20): the localizers that already work each name a step; let each answer the
binary family "is the first error at a step ≤ n?" and fuse the answers with the binary SML /
L-SML machinery. What is different on the long-chain subsets, where Step 422 found the level
readout losing to Mind-the-Gap's derivative readout?

## Encoding

- Localizer j with estimate ŝ_j votes v_j(n) = +1 if ŝ_j ≤ n else −1. The votes are monotone in
  n by construction and the true label y(n) = 1[s* ≤ n] has the same shape (Frank & Hall 2001).
- Instances are pooled (question, n) pairs over the disagreement region [min_j ŝ_j, max_j ŝ_j − 1].
  Outside it every localizer agrees and the instance carries no information about relative
  reliability; including those rows pushes every fit toward equal weights.
- Majority vote on cumulative votes = median of positions. SML = weighted median. The mode of the
  fused pmf = weighted plurality. Dawid-Skene ψ_j / η_j = "not late" / "not early" rates.
- Readout: fused CDF over the candidate grid → pool-adjacent-violators → pmf → argmax (0/1 loss)
  or median. A soft version (F_j(n) = Σ_{s≤n} softmax(r_j(s)/τ), `lsml_continuous` on 2F−1) is
  specified but not run here because per-step score profiles are not committed.

## Data and protocol

- `results/fair_paper_exact_comparisons_v1/lanes/localization/PER_QUESTION_LONG.csv`:
  ProcessBench × Llama-3.1-8B, 3,400 questions, four subsets, five source folds; five
  label-free telemetry localizers carry an ungated `locator`.
- Mind-the-Gap SLA protocol: erroneous answers only, no gate. Fits are label-free and
  out-of-fold by the file's `fold`. Labels enter only evaluation and the paired bootstrap.
- Length covariates: prefix-lane `final_length` (1,717 traces) and the max locator across the
  five as a chain-depth proxy for all 3,400.

## Fits compared

SML signed weights; L-SML with group discovery; Dawid-Skene EM initialised from SML;
bias-shift (subtract consensus-relative median offset, refit); per-subset versus pooled fit
scope. Baselines: each single localizer, median of positions, plurality.

## Decision

`FUSION_REPRODUCES_INCUMBENT`. Overall SLA: incumbent 30.12, Dawid-Skene mode 30.08
(−0.05 pp [−0.27, +0.18]), L-SML mode 29.90, median 29.40. The label-free consensus is the
incumbent (ψ 0.97 / η 0.95). Mind-the-Gap receives a negative weight (late localizer, ψ 0.26);
Unified-28 is an early localizer with a step-0 prior (η 0.29). No promotion; no candidate.

## Long-chain findings (the emphasis of the run)

Incumbent SLA falls from 45.5% (depth 0–2) to 16.4% (depth 8+) and the misses are late
(late fraction 0.22 → 0.60). Against a uniform-guess null by true position, the level
localizers carry a residual late bias of +0.5 to +0.8 steps on the long subsets and the
derivative localizer +1.1 to +1.4; the bias concentrates on errors at steps 0–2 of deep
chains. Diversity between localizers rises on long chains but in one direction, so SML has no
independent errors to exploit. Lower-quantile readouts rebalance early/late without a
significant exact-accuracy gain (descriptive sweep only).

## Boundaries and next

Llama-3.1-8B, binary votes, common-protocol Mind-the-Gap replay. Next: the soft cumulative
fusion on the Qwen OOF step scores (cluster), an onset readout at token level, and the
delta-shaped operator column in the source × operator grid. See the report for details.
