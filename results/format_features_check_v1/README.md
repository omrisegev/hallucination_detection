# Check F: CROP-style format counts as an extra DS group (2026-10-29)

**Verdict: not a candidate.** The format block is weakly above chance (within-answer AUC 0.51-0.60), and it is
strongly dependent, given the step label, on one telemetry group: max within-class |r| 0.46-0.64 (benchmark mean).
That group is mainly top15_turnover, top50_js and energy_innovation, plus dominant_freq16 and bocpd_p0 in some cells.
Under `groups_from_cov`, the format block never forms its own group: 0 of 16 cells on the continuous covariance,
0 of 16 on the GAP-vote covariance. It fails the frozen rule (max conditional |r| < 0.2), so it is not added to E1b.

Protocol: `PROTOCOL.json`, written before results. It has one amendment, A1 (a constant count gets sign +1 and adds
0). A1 was found as a crash on CRV before any CRV result existed. Run: `python format_features.py` (CPU, about 10 min).
The per-step score function is `format_block(bench, prm)`.

## Inputs and checks

- **Step texts:**
  - PRMBench: `spectral_utils.prmbench.load_prmbench` (HF cache).
  - ProcessBench: HF `Qwen/ProcessBench` (cache).
  - CRV: `hd_collection_data/crop_ares_v1/inputs_text/<bench>/answers.json`.
- **Alignment:** the text step count equals the cell step count for every answer in all 4 benchmarks, checked with an
  assert.
- **Counts:** char length, whitespace tokens, digits, letters, `=`, `+-*/^` (summed), newlines, colons, `()`, `<>`.
  This is the list from Omri's message; the CROP PDF is not in `papers/`, so it was not checked against App. B.
  `-` also counts hyphens.
- **Cleaning:** log1p of each count. Each count is then cleaned by pooled regression on a cubic in relative position;
  the 8 non-length counts are also regressed on log1p(char length). The fit is label-free.
- **Orientation:** the sign of each count's correlation with MEAN16.
- **Format block:** the plain average of the cleaned, oriented counts.

**Reviewer:** this check ran in a worker that cannot launch subagents, so no independent reviewer ran. My self-check
against the protocol is as follows:
- the text alignment assert passed;
- no label is used before the evaluation functions;
- the cleaning pools are answers with at least 2 steps;
- the orientation uses MEAN16 only.

## R1 within-answer AUC (mean over 4 PRMs; text is the same, so it differs only through orientation; R1_AUC.csv)

| count | PRMBench | ProcessBench | CRV-Arith | CRV-GSM8K |
|---|---|---|---|---|
| char length | .660 | .705 | .553 | .560 |
| whitespace tokens | .627 | .694 | .567 | .557 |
| digits | .525 | .510 | .638 | .454 |
| newlines | .664 | **.300** | .429 | .611 |
| colons | .380 | .382 | .389 | .512 |
| **format block** | **.604** | **.579** | **.584** | **.506** |
| MEAN16 telemetry | .788 | .777 | .499 | .579 |

- **Length:** most of the block's signal is step length. Position is cleaned, but the two length counts themselves
  stay in the block.
- **Orientation:** the label-free orientation picks the wrong sign for some counts. On ProcessBench, newlines (.300)
  and colons (.38) point the wrong way.
- **CRV-Arith, PURE and RLHFlow:** MEAN16 is 0.37 and 0.31 there, so the stored telemetry orientation looks inverted
  in those cells. This is outside this check's scope, but worth a look.

## R2 conditional dependence (answer-z, position-cleaned; R2_CONDCORR.csv)

| | PRMBench err / ok | ProcessBench err / ok | CRV-Arith err / ok | CRV-GSM8K err / ok |
|---|---|---|---|---|
| max \|r\| with a telemetry group | .58 / .61 | .46 / .57 | .64 / .60 | .62 / .60 |
| \|r\| with the PRM | .07 / .02 | .09 / .09 | .25 / .21 | .21 / .13 |

The format block is almost independent of the PRM. It is not independent of the telemetry: it shares a group with the
turnover, JS and energy-innovation channels (R3_CLUSTER.csv lists the group members per cell).

## R4 PRMBench by perturbation class (R4_PRMBENCH_CLASS.csv)

- **Digits:** the digit count is near chance in every class (0.42-0.59). The "PRMBench edits numbers" artifact does
  not show up as a digit signal.
- **Char length:** separates best in confidence (.73), domain_inconsistency (.74) and redundancy (.69). Those
  perturbations change step length, so part of the PRMBench format signal is perturbation-driven.
- **Natural errors:** on ProcessBench, the block (.58) is no better than on PRMBench.
