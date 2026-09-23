# Where L-SML can still act on CT7's failures: a diagnosis from the branch record (2026-09-23)

Omri's question: how to attack CT7's failure points (Section 7 of `ct7_anatomy_2026-09-23.md`)
with L-SML. This note reads the "next" lines of HISTORY Steps 397-432, the handoffs
(`HANDOFF_TOKEN_PROBABILITIES.md`, `HANDOFF_localization_2026-09-17_evening.md`,
), Codex's 2026-09-23 plan and reviews, and the branches
`codex/fusion-independence-atlas-v1`, `claude/token-probability-
fusion-v1`, `codex/cumulative-vote-fusion-v2`, `claude/competition-sync-2026-09-23`. Nothing was
recomputed.

## 1. Where L-SML is defined, and where the project has been running it

L-SML (Jaffe-Fetaya-Nadler) needs more than three conditionally independent views; below that
the eigen-stage is undetermined and the registered guard returns equal weights (Step 205).
Measured effective independent views on this population (labels for measurement only):

| representation | effective views | outcome of learned vs equal |
|---|---|---|
| step level, entropy-transform banks (20, 50, eleven, CT7) | 1.80-2.46 | learned = equal: Step 413 ladder within 1 pp; Step 429 pmf L-SML +0.19 [-0.50, +0.87]; CT7 profiles soft L-SML +0.11 [-0.60, +0.83] |
| token level, eleven channels, fused before Top10 | marginal 9.69 (not comparable) | +3.33 [+1.90, +4.75] (Step 422), of which two thirds is repair of the energy gauge pair; +1.0 to +1.2 remains (Step 425) |
| 8-token windows, moment bank (level / sd / slope), 24 answers | 3.4-3.9 | never run on the full population (atlas handoff B3, not authorized) |
| answer level, final-answer detection, 16 continuous features | not measured | L-SML +6.1 pp over flat SML at 16 features, +3.6 at 9, tie at 5 (Step 135) |
| wide redundant rosters L11-L24, step level | not measured | Joint partition + equal within/across groups holds 43.2 where continuous L-SML falls to 38.9; the active ingredient is the partition at the smallest stable K, not the eigenvector (Step 399 addendum) |

So every negative L-SML result on localization was obtained at step level on banks with fewer
than three independent views, where averaging is the method's own correct output. The three
places where fusion did add something are: many views at answer level, fusion before a
noisy readout (token level), and a partition that isolates a family (Joint, K=3, equal across
groups). Those are the levers, and each maps to a specific CT7 failure.

## 2. Failure by failure

| CT7 failure (Section 7) | fusion point | what the record says | status |
|---|---|---|---|
| **Gate**: clean accuracy .49-.72 per cell, closes on 20-25 % of erroneous answers, opens on 0 / 6,969 PRMBench answers; the largest F1 loss | **answer-level L-SML over whole-answer readouts** (mean / Top10 / max of the 50 streams, step count, log answer length, the fused locator's own max and margin), n = 6,800 answers, group discovery, label-free quantile threshold (Step 331 precedent q = 0.3) or the within-cell midrank | the gate has only ever been a single selected feature (tail15 Top10, Step 365/373; Step 397's gate ladders kept it). Answer length is orthogonal to every telemetry family (Step 414) and is gate-only material, so the answer level has at least two independent families before any transform. Answer-level detection with many features is the one regime where L-SML beat averaging by 6 pp (Step 135) | **never run**; all inputs cached; no inference |
| **Late misses** (34 %, 42 % on 11+): peaks after the error | **partition-then-equal on CT7's three families** (entropy level x5, temporal x2 = innovation + BOCPD residual, chosen-token x1): equal across groups 1/3 each instead of 5/7 : 1/7 : 1/7 | the two temporal views are early-biased (early .44 / .38), the level views late-biased (late .39-.42); CT7's balance is the accident of equal weight. Step 399 addendum: the partition, not the eigenvector, carries the Joint result; a cross-group eigen-solve costs ~3 pp. Codex's soft L-SML on the seven profiles (+0.11) is the eigen-solve, not this rule | **never run on CT7**; one arm on the frozen `profiles.npy`, report early / late by stratum |
| **Early misses** (27 %): start-of-answer excess | position-conditional null (Step 432) fixes them one-for-one against late misses | not a fusion lever; closed | closed |
| **Argmax competition** on long chains (exact .30 at 11+ with pairwise .78) | cumulative-vote L-SML / DS was the fusion designed for this decision and reproduced the incumbent (Steps 423, v2, CT7-profile run); no non-max rule beats argmax (Step 430 A3) | closed for weighting; only new views change the competitor's height | closed |
| **1.80-view ceiling** and the length-coupled Top10 blind spot (short error steps in long answers) | **window representation** (atlas B3): ~10-view moment bank on 8-token windows, 49 rows per answer, answer-local fit with shrinkage, label-free alpha | 3.4-3.9 effective views on 24 answers; the only representation where the thesis's answer-only fit is estimable | **not authorized yet** |
| **Fusion before the readout on CT7's own streams** (HANDOFF_TOKEN_PROBABILITIES 5.1, still open per Step 429) | token-level continuous L-SML on the five bank streams + token BOCPD residual (Step 420 recomputation) + per-token standardized excess, then Top10, answer-local standardization before the fit (5.2) | the only configuration in which L-SML beat equal (C1 +3.32 vs C3 +0.22) has never been applied to the strongest bank; expected size after the Step 425 correction is about +1 pp | **never run**; inputs cached |
| **PRMBench multi-error answers** (28 % have more than one error run; argmax is the wrong object) | per-step encoding (clarifications note §3): top-k-per-answer binarization or z-profiles as columns, `lsml_continuous`, fused per-step score as the ranking | within-answer ranking is the one endpoint where learned combinations moved: Step 416 L-SML +.0011*, Step 432 position evidence +.014 within-AUC | designed, not implemented |

## 3. Order, with the reason

1. **Answer-level L-SML gate.** Largest lever on the reported F1, L-SML's home regime, zero new
   inference. Measure the answer-level participation ratio first (labels for measurement only);
   then Continuous L-SML with group discovery, threshold by the frozen rule, nested folds; report
   clean accuracy, gated error accuracy and F1 per cell with CT7's locator frozen. Also the first
   gate that can open on PRMBench.
2. **CT7 family-equal (K = 3 partition, equal across groups).** One arm, minutes of CPU, targets
   the late/early balance directly. If it moves the 11+ stratum without losing the short cells,
   it is a new frozen candidate id; if not, the temporal family is not the lever for late misses.
3. **Token-level L-SML on CT7's streams before Top10**, answer-local standardized. Cheap, and it
   closes HANDOFF 5.1 on the bank that matters.
4. **Window representation measurement** (atlas B3), then answer-local L-SML with shrinkage.
5. **PRMBench per-step mode.**

Not again: step-level weighting on entropy-family banks (Steps 413, 429, CT7-profile run),
readout sweeps (Step 429), non-max rules (Step 430), consensus or triplet-statistic selectors
(Step 429; Codex 2026-09-22).

Scope note (Omri, 2026-09-23): the white-box depth line is a separate arm and is excluded from this
diagnosis; its numbers are not comparable to the gray-box localization results.

Implementation (2026-09-23): items 2, 3 and 4 are built, synthetic-tested and pre-registered on branch
`claude/lsml-ct7-levers-v1` (from competition-sync `49787d46`); Codex runs them on the data. Handoff:
`docs/HANDOFF_CODEX_LSML_CT7_LEVERS_2026-09-23.md` on that branch.
