# CT7 token-level L-SML v1 — fusion before the readout on CT7's own streams (item 3) — PRE-REGISTRATION

Written and committed 2026-09-23 before any score was computed. Branch `claude/lsml-ct7-levers-v1`.
Extraction `scripts/diagnostics/extract_ct7_token_streams_v1.py`, builders
`spectral_utils/ct7_token_streams.py`, driver `scripts/experiments/ct7_token_lsml_v1.py`, config
`configs/ct7_token_lsml_v1.json`. Development only; nothing is promoted.

## Question

Stage B (`docs/experiments/TOKEN_PROBABILITY_FUSION_V1_SESSION_REPORT.md`) found on the eleven-channel
bank that L-SML beats equal weighting only when the fusion is fitted BEFORE the Top10 step readout
(C1 pooled/before +3.32 [+1.90, +4.75]; C3 pooled/after +0.22; C4 answer-local/after +0.10), and
Step 425 traced two thirds of that margin to the energy gauge pair (about +1.0 to +1.2 pp remains).
`HANDOFF_TOKEN_PROBABILITIES.md` §5.1 is still open per Step 429, and it was never applied to the
strongest bank: CT7's own seven streams. Does token-level continuous L-SML on CT7's streams, before
the readout, beat equal weight on the same streams, and where does either stand against CT7?

## Streams (token level, CT7 view order)

0-4 the five digit-free bank streams (`digitfree_broad50.token_bank` columns 27, 28, 30, 31, 41; VE
orientation per answer toward ve1; token 0 of the prefix innovation invalid); 5 the signed BOCPD
residual (Step 387 / Step 420 recipe: z of the five streams with whole-answer mean/scale, minus the
reset-before-observation Gaussian BOCPD prior mean, hazard 1/32, averaged over the five; verbatim from
`temporal_context_data_v1` when present, recomputed from the answer's own streams otherwise);
6 the per-token standardized excess surprisal `(-log q(x) - H) / sqrt(VE + .01)`
(`chosen_token_calibration.token_calibration` column 3). CT7's seventh view is the pooled step
z-test of this statistic with step 0 neutralized; a token-level version is a new construction, so
CT7 itself is compared from its frozen scores, never rebuilt from these streams.

## Exactness gates before any fit (asserts)

(i) the masked Top10 of the five bank columns, cast to float32, equals the frozen bank extraction
`length_explicit_ct7_v1/bank/<cell>.npz['top10']` exactly; (ii) the answer-standardized masked
Top10 of the BOCPD column replays CT7's view 5 (`profiles.npy[:, 5]`) to 1e-8 (verbatim source) or
1e-6 (fallback recomputation); (iii) CT7's macro-F1 .41188745848863717 and within-AUC
.7723966352864217 replay from `CT7_DEV_SCORES.npz` (sha `9d10d2ff…b430`).

## Declared rules (the two additions to the Stage B machinery)

- Invalid tokens (token 0 of the innovation; any all-channels-invalid row) never enter the donor
  standardizer or the L-SML fit; the fused token is valid only when all its channels are. The Stage
  B code sampled every answer's token 0 into its fit (`deterministic_token_sample`), which for the
  innovation and chosen-token columns is a literal 0 and the start-of-answer spike; that inheritance
  is recorded for the eleven-bank anchor and not repeated here.
- Token analogue of CT7's step-0 rule on the chosen-token column: the tokens of the answer's first
  official step are replaced by the column's mean over the valid tokens of the other steps, before
  any standardization; applied to every T-arm and to `ct7_top10_equal7`. Sensitivity
  `T_C2_nostep0fit`: step-0 tokens excluded from the fit sample only.

## Arms (five source folds; argmax; frozen CT7 gate for the common-gate F1; every arm's step scores
answer-standardized before the pooled endpoints)

- `ct7` (frozen scores; replay), `ct7_top10_equal7` (masked Top10 per stream, answer-standardized,
  mean of seven: CT7's architecture with the token-level seventh view), `ct7_top10_equal7_nodespike`.
- `T_E1` / `T_C1`: pooled donor standardizer, equal / continuous L-SML on the seven token streams,
  masked Top10 (Stage B C1 on CT7). Pooled-donor access, declared.
- `T_E2` / `T_C2`: within-answer standardization of each stream over its valid tokens, then the
  same donor standardizer (near identity) and equal / L-SML (Stage B C2 on CT7; §5.2).
- `*_six`: the same four without the chosen-token stream.

Pre-registered chain: `ct7` → `ct7_top10_equal7` (seventh-view construction) → `T_E2` (fusion before
the readout, answer-local, equal) → `T_C2` (L-SML); the pooled branch `T_E1` / `T_C1` beside it.

## Endpoints and contrasts

- **Primary**: `T_C2 − T_E2` on macro8 gate-free SLA (the L-SML question, answer-local axis).
- One Holm family: `T_C1 − T_E1`, the six-stream companions, each T-arm minus `ct7`,
  `T_E2 − ct7_top10_equal7` (fusion order at equal weight), `T_C2 − ct7_top10_equal7`,
  `T_C* − T_C*_six` (the seventh stream), `T_C2 − T_C1`, `T_C2_nostep0fit − T_C2`,
  `ct7_top10_equal7 − ct7_top10_equal7_nodespike`; the same on common-gate F1 and PRMB within-AUC;
  depth-stratum SLA intervals; early/late; PRMScore secondary.
- Per fold: flattened weights, IPR, K, groups, any negative weight on a risk-oriented stream
  (Step 422's `chosen_surprisal` signature); the conditional participation ratio of the seven
  Top10 step views on PRMBench steps (comparable to CT7's 1.80).

## Predictions, written before scoring

1. `T_C2 − T_E2` and `T_C1 − T_E1` are small, about +1 pp or less, because the five entropy streams
   are near copies with independent token noise and the bank has no gauge pair to repair.
2. K = 2 in most folds (the five entropy streams against the rest); if so the L-SML arm is a
   token-level family-equal (item 2 at token level) and is read as such.
3. `T_E2` is below `ct7_top10_equal7` (fusion before the readout of near-copies loses the per-view
   Top10's noise reduction), and both are below CT7 because the token-level seventh view is weaker
   than the pooled z-test.
4. The despike rule matters for the seventh view (`ct7_top10_equal7 − ct7_top10_equal7_nodespike`
   positive) and little for the fused T-arms.

## Decision language

Development rows. `T_C1 − CT7` is descriptive. No arm is promoted; a favourable L-SML result is
"L-SML minus equal above zero with an interval excluding zero on the primary", recorded as such.

## Execution

```
python -B scripts/diagnostics/extract_ct7_token_streams_v1.py --source-root <repo> --temporal <temporal_context_data_v1> \
    --bank-dir <length_explicit_ct7_v1/bank> --profiles <ct7_profiles_v1/profiles.npy> \
    --out results/ct7_token_lsml_v1/CT7_TOKEN_MATRICES.npz        # ~2 h CPU, once
python -B scripts/experiments/ct7_token_lsml_v1.py --config configs/ct7_token_lsml_v1.json   # minutes
python -B scripts/experiments/ct7_token_lsml_v1.py --dry-run                                  # synthetic
```
