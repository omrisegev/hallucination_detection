# Family-tail external transfer V3: maximal-step tail rows

Date: 2026-09-24. Branch: `codex/lsml-external-generalization-v1`. This follows `FAMILY_TAIL_EXTERNAL_V2.md`.

## Contract

- **Request.** Omri asked to repeat the fixed procedure with a tail that is only the maximal step, on bank11 and on family15. There are three new rows. Each marks the tie-aware top-1 step per answer, centres the marks within the answer, z-scores them over the pooled fit rows, fits `lsml_continuous` (unit scale), and applies the weights to the continuous features:
  - `F15_tail1s_lsml`: the 15 families.
  - `B11_tail1s_lsml`: unoriented bank11, the same representation as the frozen bank11 L-SML.
  - `B11o_tail1s_lsml`: bank11 oriented by the source signs.

  Both bank11 orientations were declared in advance, so no orientation was chosen from results.
- **Source stage.** Branch `claude/ssl-pseudolabel-residual-v1`, `results/tail1_transfer_v3/`, Step 445. The protocol came before any result. It declared that all three rows go external regardless of their source result.
- **Lock.** `results/family_tail_transfer_v2/TRANSFER_LOCK_V3.json`, sha256 `0fc561b7e382814cacdaa3a6425916cdba5119d03b48f2839729459a7e58c8ee`. It is V2 unchanged plus the three rows, with deployment fits on source folds 0-3 and the q80 threshold on fold 4.
- **Exposure.** V1 and V2 external results had been read before this lock. The rows come from the request, not from external data. This is exploratory.
- **Scoring.** The rows were rescored from the sealed V1 features. Before sealing, the eleven V2 arms had to reproduce the sealed V2 records exactly, and they did: difference 0.0 on 6,190 records.
- **Analysis.** Unchanged evaluator logic. The primary family is 5 contrasts x 3 cells = 15, Bonferroni-corrected, with 100,000 source-question draws and seed 20260924. The V1 (18) and V2 (12) contrasts reproduce exactly.

## Results

Scores are percentages. The Hard2Verify and Socratic metrics are never averaged.

| Method | Hard2Verify balanced F1 | Socratic-Qwen3 PRMScore | Socratic-QwQ PRMScore |
|---|---:|---:|---:|
| bank11 max-step, unoriented | **44.48** | 62.38 | 63.65 |
| bank11 L-SML (frozen) | 43.67 | **63.22** | **64.24** |
| bank11 max-step, oriented | 43.86 | 61.82 | 63.12 |
| family15 max-step | 42.79 | 61.05 | 62.28 |
| family15 equal | 42.38 | 61.12 | 62.94 |
| family15 top-20% (V2) | 42.28 | 61.00 | 62.30 |
| bank11 equal | 40.88 | 60.79 | 61.50 |
| CT7 | 37.75 | 58.75 | 60.17 |

Differences are in points, with Bonferroni intervals over 15 contrasts.

| Contrast | Hard2Verify | Socratic-Qwen3 | Socratic-QwQ |
|---|---|---|---|
| family max-step minus top-20% | +0.52 [-1.89, +2.96] | +0.05 [-0.42, +0.51] | -0.02 [-0.50, +0.46] |
| family max-step minus family equal | +0.41 [-2.37, +3.24] | -0.07 [-0.57, +0.43] | -0.66 [-1.16, -0.16] |
| bank11 max-step minus bank11 L-SML | +0.81 [-1.11, +2.81] | -0.84 [-1.23, -0.45] | -0.59 [-0.96, -0.21] |
| bank11 oriented max-step minus bank11 L-SML | +0.19 [-2.43, +2.80] | -1.40 [-1.88, -0.95] | -1.12 [-1.57, -0.67] |
| bank11 max-step minus bank11 equal | +3.60 [+0.68, +6.66] | +1.59 [+1.02, +2.18] | +2.15 [+1.59, +2.72] |

**Source PRMScore, for comparison.** The three rows scored 63.76 (family max-step), 63.25 (bank11 max-step) and 63.75 (bank11 oriented max-step). Every one of them was below its alternatives on source, and bank11 max-step only tied bank11 equal (-0.08).

**Interpretation.**
- The frozen bank11 L-SML stays the leading candidate on the PRMScore benchmarks.
- On Hard2Verify, bank11 max-step has the highest score of any method, but the difference is not significant.
- On bank11, learning from the maximal step clearly beats equal externally, while on source it only tied equal.
- Orienting by source signs hurts in every cell.
- On family15, max-step equals the other family variants.
- As in V1 and V2, the source ranking did not predict the external ranking.

**Not done.** Codex's independent audits were not re-run on V3.

## Reproduction

```text
python -B .worktrees/ssl-pseudolabel-residual-v1/scripts/experiments/tail1_transfer_v3_run.py   # source + lock
python scripts/run_family_external_v3.py
python scripts/evaluate_family_external_v3.py --seal-only
python scripts/evaluate_family_external_v3.py
```
