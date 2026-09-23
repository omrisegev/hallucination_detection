# Token-probability line, Stage A — hold the gate fixed, both directions

Runs `scripts/diagnostics/gate_hold_ct7_vs_token_v1.py`; output
`results/token_probability_fusion_v1/GATE_HOLD_STAGE_A.json`.
Answers §5.4 of `docs/HANDOFF_TOKEN_PROBABILITIES.md`. Development-only.

## Contract

Nothing is refitted. Both locators are frozen per-step score vectors.

| item | value |
|---|---|
| population | 13,769 answers / 145,597 steps; 6,800 ProcessBench answers, 4,442 erroneous |
| CT7 scores | `CT7_DEV_SCORES.npz`, **restored from the Drive backup of 2026-09-17, not re-derived** |
| CT7 integrity | its `FROZEN_CANDIDATE_CT7.json` is byte-identical to master's |
| token scores | `claude_feature_bank_token_lsml_v1/OOF_SCORES.npz` (L-SML and equal) |
| scorer | imported from `gate_isolation_token_lsml_v1.py`, not re-implemented |
| bootstrap | 10,000 draws, unit = source group, 1,979 error groups / 2,842 PB groups, seed 20260918 |

**Replication check.** The run reproduces every number the handoff already
carried: L-SML 35.92, equal 32.59, chance 16.58, CT7-under-its-own-gate 41.19,
token-L-SML-under-LOCO-5 34.08, and the paired interval +3.32 pp [+1.90, +4.75]
against the handoff's +3.33 [+1.90, +4.75]. The scorer is therefore the same one.

## A1 — gate-free SLA, the column that was missing

Per-subset Step-level Localization Accuracy on erroneous answers, no gate.

| subset | CT7 | token L-SML | token equal | chance | MtG Shannon-Drop | MtG Shannon-Avg |
|---|---|---|---|---|---|---|
| GSM8K 4B | **48.31** | 47.83 | 44.93 | 20.84 | 43.42 | 27.94 |
| GSM8K 8B | 47.34 | **49.76** | 41.06 | 20.84 | 46.11 | 27.66 |
| MATH 4B | **35.69** | 34.85 | 30.30 | 18.10 | 32.03 | 24.17 |
| MATH 8B | **36.87** | 31.99 | 29.80 | 18.10 | 32.90 | 24.62 |
| OlympiadBench 4B | 39.03 | 30.56 | 27.53 | 13.55 | **43.06** | 24.95 |
| OlympiadBench 8B | 37.67 | 31.01 | 29.80 | 13.55 | **41.52** | 26.30 |
| Omni-MATH 4B | 37.15 | 30.96 | 29.51 | 13.84 | **38.04** | 23.67 |
| Omni-MATH 8B | 37.02 | 30.43 | 27.80 | 13.84 | **37.04** | 23.40 |
| **mean** | **39.89** | 35.92 | 32.59 | 16.58 | 39.27 | 25.34 |

Chen et al., ICML 2026, Table 3; their ProcessBench subsets are exactly ours
(400/1000/1000/1000). Their score is a derivative, CT7 and the token arms are levels.

## A2 — the gate held fixed, both directions

ProcessBench macro-F1. Rows are the gate, columns the locator.

| gate | answers opened | CT7 | token L-SML | token equal |
|---|---|---|---|---|
| CT7 frozen non-digit tail15 | 4,556 | **41.19** | 37.32 | 34.88 |
| LOCO-5 @ 0.33 | 5,391 | **36.98** | 34.08 | 32.34 |

The handoff's two headline numbers sit on the diagonal. Off-diagonal is new.

## A3 — paired source-group intervals

| contrast | difference | 95% CI | verdict |
|---|---|---|---|
| gate-free mean SLA: L-SML − equal | +3.32 pp | [+1.90, +4.75] | excludes zero |
| gate-free mean SLA: **CT7 − L-SML** | **+3.96 pp** | **[+2.28, +5.59]** | **excludes zero** |
| gate-free mean SLA: CT7 − equal | +7.29 pp | [+5.39, +9.21] | excludes zero |
| macro-F1 under tail15: CT7 − L-SML | +3.86 pp | [+2.57, +5.15] | excludes zero |
| macro-F1 under LOCO-5: CT7 − L-SML | +2.90 pp | [+1.95, +3.85] | excludes zero |
| macro-F1 under LOCO-5: L-SML − equal | +1.74 pp | [+0.96, +2.53] | excludes zero |

## What this settles

1. **The gate was not hiding a token-level win.** CT7 leads on the gate-free
   protocol and under *both* gates held fixed, every interval excluding zero.
   The 41.19-vs-34.08 comparison the handoff forbade was directionally right for
   the wrong reason: the gap is real, but roughly 4 pp of it is the locator and
   roughly 3–4 pp was the gate.
2. **The tail15 gate dominates LOCO-5 @ 0.33 for both locators** (+4.21 pp for
   CT7, +3.24 pp for token L-SML). LOCO-5 is not merely mistuned at 0.33; it is
   the worse gate even so. This raises the value of §5.6 and lowers the value of
   the label-selected 0.41 ceiling.
3. **CT7 edges the published best on the mean** (39.89 vs 39.27) as a level
   signal against a derivative — but only on the mean. Per cell it wins GSM8K
   and MATH and still loses OlympiadBench by 4.0/3.9 pp. The length asymmetry in
   §3 of the handoff is not a token-level artefact; it survives in our own
   strongest arm, in the same direction, which strengthens the case for §5.5.
4. **The L-SML-over-equal gain survives** (+3.32 pp) but it is a gain inside an
   arm that trails CT7 by a larger margin than the gain itself. It remains the
   project's first fusion-over-averaging interval excluding zero; it is not
   evidence that the token representation is the better one.

## Consequence for Stage B

The §5.1 hypothesis — that the L-SML advantage is noise-down-weighting that a
step-level readout already performs for free by averaging — gains support here,
because the step-level arm is the better one. Stage B's pre-registered
prediction should be written against that, and the energy-pair degeneracy
declared before the 2×2 is run, since it is inherited by all four cells.
