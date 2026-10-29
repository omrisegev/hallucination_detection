# answer_mixture_gate_v1: label-free clean-answer detection from raw answer-level scores. Both tracks FAIL

Pre-registration: `docs/experiments/ANSWER_MIXTURE_GATE_PREREG_2026-10-29.md` (commit 8d150169). Nothing was tuned on
outputs. The verdict uses only the primary variant: the answer maximum, raw residual, two-component Gaussian mixture.
Development data; labels were used only to score.

## Verdict against the pre-registered rule

- **Track P (PRM raw risk) FAILS.**
  - The rule needs |π̂ − π| ≤ .05 in at least 3 of 4 PRMs on ProcessBench and also on Hard2Verify.
  - **0 of 12 cells** meet it (0/4 ProcessBench, 0/4 Hard2Verify, 0/4 PRMBench). Source: `TRACK_P_VERDICT.json`, `TRACK_P.csv`.
  - The closest cells are ProcessBench rlhflow_prm at .078 and ProcessBench qwen_prm_72b at .080.
- **Track T (raw telemetry, ProcessBench × qwen_prm_7b) FAILS.**
  - Parity passed first (see below).
  - |π̂ − π| is .172 for T-MEAN, .156 for T-LSML and .105 for T-CHAIN. None is ≤ .05.
  - Track P's estimate in the same cell is off by .125, so only T-CHAIN is closer than Track P. Source: `trackT/TRACK_T_VERDICT.json`.
- **Gate usefulness (secondary; closes ≥30% of clean answers while keeping ≥85% of erroneous ones):** met in 2 of 15 primary rows.
  - ProcessBench qwen_prm_72b: closes .904 of clean answers, keeps .908 of erroneous ones.
  - ProcessBench rlhflow_prm: closes .450, keeps .911.
  - No Track T arm meets it.
- **Overall:** label-free clean-answer detection by an answer-level mixture is **not supported** on this data. The
  mixture's share estimate is off by .08–.45 everywhere.
  - Its estimate does not follow the truth. Hard2Verify has π = .768, but π̂ is .32–.54.
  - PRMBench has π = .866, but π̂ is .50–.59.
  - Synthetic check (a) shows why the estimate cannot be trusted: on unimodal heavy-tailed data, BIC prefers two
    components every time.

## Track P: the PRM's raw risk logit(1 − reward), primary variant (`TRACK_P.csv`, rows with primary = True)

"Clean closed" is the share of clean answers the gate calls clean. "Err. kept" is the share of erroneous answers it
keeps open. ΔBIC = BIC(2) − BIC(1); a negative value favours 2 components. The DS-gate π̂ is context only, from
`DS_GATE_CONTEXT.csv` (`origin/claude/ds-gate-2026-10-29`, ds_gate_v1b_fixed E1, AUTO / TELE16; "–" means no valid
fit, and there was no Hard2Verify run).

| bench | PRM | n | π (true) | π̂ | \|π̂−π\| | ΔBIC 2 vs 1 | clean closed | err. kept | AUROC s_a | DS-gate π̂ |
|---|---|---|---|---|---|---|---|---|---|---|
| ProcessBench | qwen_prm_7b | 3400 | .653 | .778 | .125 | −511.2 | .246 | .991 | .944 | .988 / .984 |
| ProcessBench | qwen_prm_72b | 3400 | .653 | .573 | .080 | −679.2 | .904 | .908 | .964 | .976 / .993 |
| ProcessBench | pure_prm | 3400 | .653 | .407 | .246 | −133.7 | .938 | .517 | .860 | .988 / .991 |
| ProcessBench | rlhflow_prm | 3400 | .653 | .731 | .078 | −303.0 | .450 | .911 | .780 | .959 / .987 |
| Hard2Verify | qwen_prm_7b | 177 | .768 | .319 | .449 | −10.6 | .732 | .199 | .639 | – |
| Hard2Verify | qwen_prm_72b | 177 | .768 | .475 | .293 | +12.8 | .707 | .507 | .661 | – |
| Hard2Verify | pure_prm | 177 | .768 | .482 | .286 | +10.7 | .585 | .522 | .539 | – |
| Hard2Verify | rlhflow_prm | 177 | .768 | .542 | .227 | +10.3 | .512 | .640 | .634 | – |
| PRMBench | qwen_prm_7b | 6969 | .866 | .510 | .356 | −51.5 | .845 | .611 | .821 | – / .916 |
| PRMBench | qwen_prm_72b | 6969 | .866 | .589 | .277 | −879.8 | .817 | .710 | .841 | .851 / .935 |
| PRMBench | pure_prm | 6969 | .866 | .501 | .365 | −42.8 | .793 | .477 | .694 | .940 / .943 |
| PRMBench | rlhflow_prm | 6969 | .866 | .566 | .300 | +30.3 | .617 | .623 | .637 | .911 / .936 |

**Reading.**
- The raw PRM maximum ranks answers well on ProcessBench (AUROC .78–.96) and less well on Hard2Verify and PRMBench (.54–.84).
- Good ranking does not make the mixture weight equal the prevalence. On ProcessBench qwen_prm_7b the AUROC is .944, yet
  π̂ is .778 against a truth of .653. The fitted mixture splits the scores at a point that closes only 25% of clean answers.
- On Hard2Verify, BIC rejects the second component in 3 of 4 cells. That is expected with n = 177.

**Secondary variants (reported, they do not replace the primary).**
- The table covers the top-2 mean answer score and the Yeo–Johnson model (`TRACK_P.csv`, primary = False).
- Only one secondary row of the 36 has |π̂ − π| ≤ .05: ProcessBench qwen_prm_72b, top-2, raw, with π̂ .668 (off by .015).
- The same cell under Yeo–Johnson gives .567.
- No secondary variant comes close on Hard2Verify or PRMBench (smallest error .266).

## Track T: raw telemetry, ProcessBench × qwen_prm_7b, primary variant (`trackT/TRACK_T.csv`)

| step score | n | π (true) | π̂ | \|π̂−π\| | ΔBIC 2 vs 1 | clean closed | err. kept | AUROC s_a |
|---|---|---|---|---|---|---|---|---|
| T-MEAN | 3400 | .653 | .481 | .172 | −64.4 | .818 | .625 | .831 |
| T-LSML | 3400 | .653 | .498 | .156 | −119.8 | .824 | .667 | .850 |
| T-CHAIN | 3400 | .653 | .548 | .105 | −99.0 | .804 | .751 | .887 |
| (Track P, same cell) | 3400 | .653 | .778 | .125 | −511.2 | .246 | .991 | .944 |

- The telemetry scores under-estimate π, and the PRM score over-estimates it.
- None of the telemetry scores ranks answers as well as the PRM alone (.83–.89 against .944).
- The secondary variants are in the same CSV. None has an error below .119.

**Parity (mandatory, passed before any Track T number was read).** Source: `trackT/PARITY.json`, `trackT/REBUILT_BANK.json`.
- **Download:** `external_telemetry.pkl` was downloaded with rclone. Its sha256 (378ca81e…) equals the AIRCC bank_v3
  manifest's `raw_sha256`.
- **Bank rebuild:** the bank was rebuilt locally with `results/prm_boost_external_v1/build_bank_external.py`, unchanged,
  through the driver `trackT/build_bank_local.py`.
  - The driver's only change is the frozen Qwen digit ids 15..24, because there is no tokenizer.json locally.
  - The rebuilt bank matches the AIRCC bank_v3 to 2.7e-9 over all 56 channels. Masks, offsets and uids are identical,
    and so are the rewards.
- **Score recomputation:** TELE_N16 and CHAIN_N16 were recomputed from the rebuilt bank. The logic is that of the scorer
  that wrote the stored file: `trackT/score_chain_pb_integration.py`, copied from `origin/claude/integration-2026-10-17`
  (guarded L-SML).
  - Both match the stored `processbench_scores/SCORES_CHAIN_qwen_prm.npz`: TELE_N16 to 4.0e-14, CHAIN_N16 to 1.8e-14
    (tolerance 1e-6). PRM matches to 8.9e-16.

## Synthetic checks, run before the real data (`SYNTHETIC.csv`, `SYNTHETIC_SUMMARY.json`, `synthetic.log`)

**Setup.**
- n = 3,000.
- The pre-registered result is replicate 0 (seed 20261029).
- 20 replicates were added to measure the spread; they do not replace replicate 0.
- In (b), t5 noise is rescaled to unit variance, so d is measured in noise standard deviations.

**(a) No true mixture: does the method invent a second group? Yes, always.**

| data | model | π̂ (replicate 0) | ΔBIC (replicate 0) | BIC prefers 2 (of 20) | π̂ range over 20 |
|---|---|---|---|---|---|
| Student-t, 3 df | raw | .335 | −466.9 | 20 | .209–.958 |
| Student-t, 3 df | Yeo–Johnson | .677 | −444.3 | 20 | .055–.874 |
| log-normal (0, 1) | raw | .205 | −3371.3 | 20 | .170–.232 |
| log-normal (0, 1) | Yeo–Johnson | .476 | −117.6 | 20 | .465–.503 |

- Heavy tails or skew alone make BIC choose two components every time, with a large margin.
- So "BIC prefers 2" on the real data is **not** evidence of a clean group and an erroneous group.
- The Yeo–Johnson transform does not remove this.

**(b) A true mixture: is π recovered? Only roughly, and only at d = 2.** Raw model, replicate 0:

| π | d | noise | π̂ | \|π̂−π\| | BIC prefers 2 (of 20) |
|---|---|---|---|---|---|
| .65 | 1 | Gaussian | .531 | .126 | 0 |
| .65 | 1 | t5 | .530 | .119 | 6 |
| .65 | 2 | Gaussian | .563 | .076 | 20 |
| .65 | 2 | t5 | .613 | .046 | 13 |
| .20 | 1 | Gaussian | .465 | .256 | 0 |
| .20 | 1 | t5 | .431 | .230 | 9 |
| .20 | 2 | Gaussian | .382 | .179 | 20 |
| .20 | 2 | t5 | .265 | .067 | 19 |

- At d = 1 the estimate is pulled to about .5 whatever the truth.
- Even at d = 2 the error is above .05 in 3 of 4 settings.
- This lines up with the real data. Each cell's real-data AUROC s_a appears in the Track P table above.

## Independent verification (`verification/VERIFY.json`)

The pre-registered verifier was a separate subagent writing its own code, which imports none of this experiment's
scripts. Every check passed:
- **Track P primary, all 12 cells:** π̂ reproduced to 1.3e-15, AUROC to 1.1e-16, and the gate rates exactly.
- **Labels:** no label enters any fit.
- **Populations:** PRMBench 6,969 / π .866; Hard2Verify 177 / .768; ProcessBench 3,400 / .653.
- **Parity:** independent L-SML re-fit, TELE_N16 4.0e-14, CHAIN_N16 2.6e-12.
- **Track T, all three arms:** reproduced to ≤1.1e-14.
- **Synthetic:** qualitatively the same with the verifier's own seeds; BIC prefers 2 on unimodal data in 5/5 replicates.
- **Not verified:** the secondary variants.

## Implementation choices where the pre-registration left room (fixed before the runs, not tuned)

- **Answer length in tokens ("where available"):**
  - PRMBench: the sum of the stored log1p step token counts.
  - Hard2Verify: steps only, because the score files have no token counts.
  - ProcessBench qwen_prm_7b (Track P and T): tokens from the raw telemetry spans (`trackT/PB_QWEN_ANSWER_TOKENS.npz`).
  - The other three ProcessBench PRMs: steps only, because there are no local token counts.
- **Error labels:**
  - Hard2Verify: an answer has an error if any included step is incorrect. The common set has no empty steps.
  - ProcessBench: label != −1, with label the 0-based first-error step (`pb_metrics.py`). All steps are non-empty for these 4 PRMs.
- **Track T standardization:** the digit channels are standardized over their active steps, with inactive steps set
  to 0. This is the global analogue of the bank's `masked_std`. In this cell every step is active (verifier).
- **T-CHAIN inputs:** the PRM risk and each T-LSML group score, each z-scored globally. This replaces the answer-z
  used in the CHAIN fit.
- **Raw check:** every score called raw was checked to have a per-answer mean spread above 0 (`per_answer_mean_sd`
  column). The stored N16 channels (answer-z) were not used as raw.

## Limits

- ProcessBench and Hard2Verify were inspected before. ProcessBench is out of the thesis focus and is used here only
  for its many clean answers.
- Hard2Verify has 177 answers, so its fits are noisy.
- Track T covers one PRM. The other backbones' raw telemetry, about 2.8 GB, was not downloaded.
- The mixture is transductive per cell and assumes a two-group structure. Check (a) shows that BIC cannot test that
  assumption on skewed or heavy-tailed scores.
- The two useful-gate cells (ProcessBench 72B and rlhflow) come from the same procedure that gives wrong prevalence
  elsewhere. A label-free user cannot tell in advance which cells these are. They are not a supported method.

## Files

- **Code:** `mixture.py`, `synthetic.py`, `track_p.py`, `trackT/track_t.py`, `trackT/build_bank_local.py`,
  `trackT/extract_answer_tokens.py`, `trackT/score_chain_pb_integration.py` (copied scorer).
- **Logs:** `synthetic.log`, `track_p.log`, `trackT/track_t.log`, `trackT/build_bank.log`, `trackT/extract_answer_tokens.log`.
- **Not in the repo:** the rebuilt bank stays outside the repo (session scratchpad). Its hashes are in `trackT/REBUILT_BANK.json`.
- **Download:** the download directory `/Users/osegev/Desktop/hd_collection_data/processbench_raw/qwen_prm/` was written only by rclone.
