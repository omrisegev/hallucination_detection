# Audit: raw_null_gate_v1 (branch claude/raw-null-gate-2026-10-30, HEAD 42f83d01)

Read-only review. No repo file was changed. Real data are not on this machine, so nothing was rerun on real data.
Run here: the 8 unit tests (pass) and synthetic probes that import the frozen module (scratchpad `probe/`).
**V** = verified from files or by running code. **S** = suspected, with the evidence given.

## Verdict

**Partly trustworthy.**
- The FAIL is real **for the pre-registered gates as frozen**. I recomputed the pass rule from RESULTS.csv: 0 of 20
  pass. The code hashes match PROTOCOL.json and CHECK.json.
- It is **not** evidence that raw step scores carry no clean-answer signal. The gates fail mostly because the
  empirical null is not identified and the thresholds land in the wrong place, not because the ranking is absent.

## Findings, ranked by severity

| # | What | Evidence | Impact on the conclusion | Conf. |
|---|---|---|---|---|
| 1 | **The empirical null is not identified in about half of all fits.** The "centre = correct steps" assumption fails. | **V** FITS.json: G1 `pi0_raw` > 1 in 31/68 cells, and `sigma0` > 1 (pooled-z units, so wider than the data) in 17/68. ProcessBench G1: pi0_raw 0.99–1.61 and sigma0 0.92–1.80. Group fits: 140/244 have pi0_raw > 1 and 141/244 sit at the .995 clip. 29/273 group fits fail outright. Within one cell, group pi0_raw differs by a median of 0.36 (max 1764), so the over-identification check fails. pi0_raw > 1 is outside the two-groups model, and the .995 clip hides that. | This is the main driver. When the fitted null is as wide as the data, lfdr ≈ 1 everywhere, so G1 closes almost every answer. ProcessBench G1: closed_clean ≈ 1.00 and kept_error ≈ 0.00 in 20/20 cells. | High |
| 2 | **Errors are not a separate tail on ProcessBench.** About 65% of answers have an error, and post-first-error steps are inside the fit pool (keep_gold is applied only when scoring). | **S/V** Synthetic probe with the module's `gate_g1`: an error shift of 1.5 SD, with 4% or 33% error steps, gives pi0_raw 1.01–1.05, sigma0 1.00–1.07, closed_clean 0.99–1.00 and kept_error 0.01–0.05. That is the same signature as real ProcessBench. The selftest passed only at shifts d = 3.0/2.8/2.5/2.3 SD (run_raw_null_gate.py:301), which are unrealistically strong. | G1 cannot work at realistic step separation. This is a design limit, not a bug. Section 7 of the prereg names the zero assumption as a risk but does not model this case. | Med-High |
| 3 | **Discrete and spiky channels break central matching.** | **V** Singleton groups: bocpd_p0 13/13 pi0_raw > 1 (median 1.36); dominant_freq16 7/16 fail and 6 have pi0_raw > 1; top15_turnover 5/6 have pi0_raw > 1; digit_spread 3/7 fail, and one has pi0_raw 22.2. Probe: an 8-level discrete channel gives pi0_raw 2.0; 4 or 16 levels, or a bimodal beta mix, give "IRLS did not converge" or "not concave". | These channels feed G2/G3 nulls that carry little meaning. The effect is worst in SUB5, which is 4/6 discrete-ish. | High |
| 4 | **G2 uses the shared median π₀ in place of each group's π₀** in LR_g. With one group at π₀_g < median, its LR is inflated by up to ×20. The product then acts as "any one group vetoes". | **V** Code raw_null_gate.py:231-237 follows the prereg 3.4 formula, but the LR should be f1/f0 = (π₀_g/(1−π₀_g))(1−lfdr_g)/lfdr_g. Probe with **no errors at all**: (gauss, gauss, t6) gives 26.4% of answers called erroneous, against 7.4% with per-group π₀. (gauss, gauss, lognormal) gives 99.9%. Real data: the median over-call of G2 (share called − true share) rises with the number of groups: 2 groups −.16, 3 +.24, 4 +.45, 5 +.79, 6–7 +.8. | This explains why G2 closes nothing on CRV-GSM8K (median closed_clean .01–.08 in 4 of 5 arms). The cause is a design or formula error plus the veto structure, not the absence of signal. Dependence between groups within an answer (difficulty is not removed) probably adds to it, but that is not tested. | High (mechanism), Med (share due to dependence) |
| 5 | **The thresholds, not the ranking, look like the bottleneck.** | **S** For each valid row I fitted an equal-variance binormal to the single operating point (d′ = z(ke) − z(1−cc)). G1b on ProcessBench gives median d′ 0.90 (implied AUC ≈ .74, n = 20), so closed_clean at kept_error = .85 would be ≈ .44, which passes the .30 bar. G2 and G3 on CRV-arith give d′ ≈ 1.16. G1b closes 94–96% of clean answers but keeps only 21–25% of erroneous ones: the α = .05 Šidák cut is far too lenient. | The rule is achievable in principle, including at CRV's 8–21% error base rates. The label-free threshold placement failed. This is an extrapolation from one point per cell and has not been measured. | Medium |
| 6 | **G3 is often invalid or inverted.** | **V** SUB5 has K = 2 in 13/28 cells, so G3 is invalid by construction ("2 group votes"). Of 218 G3 votes, 70 fire on ≤ 1% of answers and 11 on ≥ 99%, so they carry no information. ProcessBench G3 median d′ is −0.25 (worse than chance); for example SUB5 sa processbench qwen_prm_7b has cc .21 and ke .33. That points to a DS label switch or a degenerate fit. This is the known failure of answer-level vote DS seen in the earlier DS gates. | G3 results do not test the DS idea fairly. Its two "cannot work" modes are visible here. | High |
| 7 | **The verdict code credits invalid gates on PRMBench no-harm.** | **V** run_raw_null_gate.py:285 does not check `ok`. SUB5\|sa\|G3 shows "3/4" no-harm, but all 3 of those cells are invalid (clean = none, so the metric equals GAP_ONLY). The only valid cell harms: 45.7 vs 65.4. The verifier does not recompute VERDICT.json. | No effect on the outcome, because ProcessBench is 0/4 everywhere. The PRMBench criterion is also nearly trivial (97% of evaluated answers have errors, 176 clean of 6211), and a gate that closes nothing passes it. | High |
| 8 | **The 3 `ece_step` mismatches** (all G2, SUB5 standalone, all K = 2 cells with a 4-channel group containing turnover, dominant_freq16, bocpd_p0 and digit_spread). | **V** The two ECE codes (runner:111-116, verifier:976-982) put identical arrays into identical bins: both use a stable argsort with ties broken by index. So the mismatch needs float-level differences in `p_err_step` itself. Discrete channels produce exact repeated u values, which makes that plausible. **S**: the tie explanation is consistent but not proven. The decisions and rates match exactly. | None for the verdict. G2 `ece_step` is ill-defined under 29–36% exact ties and should not be read anywhere. | Medium |
| 9 | **Blind spots in the verifier.** | **V** The verifier imports the same loaders (lr_transfer, E.channels, votes_e1b.raw_matrix/clean), the same label definitions (copied from run_e1), the same metric (R1.metric), the same stage7 gap_flags, lsml_continuous_guarded and ds_em_votes. It does not compare `p_err_step`, `p_err_worst`, the G1b/G3 `p_answer` arrays or VERDICT.json. The npz files hold only clean booleans and scores. | A label, mask or metric bug shared by both sides would go undetected. Orientation is protected by the raw_matrix parity assert (answer-z(raw) = stored to 1e-4) and by fused parity in CHECK. | Medium |
| 10 | **Defined-step handling (3.2a) never activated.** | **V** share_defined = 1.000 for all 273 groups, including 7 singleton digit_spread groups. So `digit_active` is all-True in every bank, or the mask is lost. **S**: digit channels then carry point masses into central matching, which fits finding 3. | NaN→0 is moot here. The protection the prereg added (review finding 3) did nothing. | Medium |
| 11 | **Documentation errors.** | **V** The README mismatch table swaps runner and verifier: RESULTS.csv holds 0.136894 and 0.086129 (runner), while VERIFY's "mine" (verifier) holds 0.139077 and 0.085404. The README says `labelfree/` is committed, but `results/**/*.npz` is gitignored and git ls-files shows no labelfree files. | Cosmetic. The second error blocks offline AUC checks here. | High |

**Checked and found OK (V):**
- **Labels:** `has_err` / `ev` / `stepmask` match run_e1 exactly. The ProcessBench error share is .653 (2221/3400), the known ProcessBench base rate.
- **PRMBench:** excluding class `correct` leaves 6211 answers, 176 of them clean.
- **CRV base rates:** .083 (arith) and .210 (GSM8K).
- **Orientation:** σ = +1 and all orientation entries are +1 except a few −1 cross weights.
- **Fused parity** passed in CHECK, and the unit tests pass.
- **ProcessBench GAP_ONLY F1 = 0** is expected, because GAP always flags, so accuracy on clean answers is 0.

## What the numbers say

Median closed_clean / kept_error over the 4 PRMs (pass needs ≥ .30 / ≥ .85 in 3 of 4 PRMs on each of the first three benchmarks):

| Gate | Arm/mode | ProcessBench | CRV arith | CRV GSM8K | PRMBench |
|---|---|---|---|---|---|
| G1 | AUTO/prm | 1.00/0.00 | 0.53/0.78 | 0.89/0.45 | - |
| G1 | TELE16/prm | 1.00/0.00 | 0.54/0.76 | 0.92/0.41 | - |
| G1 | TELE16/sa | 1.00/0.00 | 0.53/0.77 | 0.86/0.42 | 0.66/0.59 |
| G1 | SUB5/prm | 1.00/0.00 | 0.64/0.63 | 0.96/0.19 | - |
| G1 | SUB5/sa | 1.00/0.01 | 0.59/0.69 | 0.95/0.17 | 0.91/0.16 |
| G1b | AUTO/prm | 0.96/0.23 | 0.91/0.42 | 0.88/0.45 | - |
| G1b | TELE16/prm | 0.95/0.24 | 0.92/0.41 | 0.87/0.45 | - |
| G1b | TELE16/sa | 0.95/0.22 | 0.80/0.49 | 0.75/0.43 | 0.85/0.41 |
| G1b | SUB5/prm | 0.94/0.25 | 0.91/0.44 | 0.85/0.47 | - |
| G1b | SUB5/sa | 0.94/0.21 | 0.81/0.48 | 0.74/0.47 | 0.77/0.32 |
| G2 | AUTO/prm | 0.67/0.30 | 0.20/0.97 | 0.04/0.97 | - |
| G2 | TELE16/prm | 0.91/0.08 | 0.24/0.95 | 0.02/0.95 | - |
| G2 | TELE16/sa | 0.09/0.88 | 0.24/0.91 | 0.08/0.90 | 0.28/0.92 |
| G2 | SUB5/prm | 0.48/0.57 | 0.57/0.80 | 0.01/0.99 | - |
| G2 | SUB5/sa | 0.18/0.93 | 0.35/0.95 | 0.49/0.58 | 0.78/0.41 |
| G3 | AUTO/prm | 0.78/0.28 | 0.38/0.96 | 0.98/0.26 | - |
| G3 | TELE16/prm | 0.98/0.04 | 0.41/0.97 | 0.95/0.38 | - |
| G3 | TELE16/sa | 0.97/0.15 | 0.48/0.85 | 0.99/0.10 | 0.57/0.69 |
| G3 | SUB5/prm | 0.00/1.00 | 0.59/0.88 | 0.26/0.89 | - |
| G3 | SUB5/sa | 0.00/1.00 | 0.42/0.88 | 0.00/1.00 | 0.00/1.00 |

- **Cells meeting cc ≥ .30 and ke ≥ .85:** ProcessBench 0/80, CRV-arith 24/80, CRV-GSM8K 1/80. The one GSM8K cell is SUB5/prm, qwen_prm_72b, G3, at .47/.86.
- **The gates swing between extremes.** Many cells sit at about 1/0 (closes everything) or about 0/1 (closes nothing). Which extreme a gate lands on depends on the null fit, not on separating clean from erroneous answers.
- **Calling rate vs truth.** On ProcessBench, G1's share called erroneous is about 0.00, while the true share is .65. On CRV-GSM8K, G2 calls .94–1.00, while the true share is .21.
- **Nearest misses.**
  - CRV-arith is close: several arms reach 2–3/4.
  - On ProcessBench, the best point is SUB5/sa, qwen_prm_72b, G2: .22/.91.
- **Secondary metric.** ProcessBench F1 rises from 0 to 13–33 with G1b. CRV macro-F1 rises by up to about 16 points (for example AUTO, crv_arith, qwen_prm_7b, G1b: 65.8 vs 50.2). So a gate is useful, even though none passes the rule.

## Suggested next checks (max 5)

1. **Answer-level AUC of each gate statistic, with labels, on the PC.** The statistics are raw_fused at the worst step, G2 p_err_worst and the G3 posterior. This separates "weak ranking" from "wrong threshold" (finding 5). Save the p arrays too, since the npz files lack them.
2. **Re-fit G1 with the null pool restricted to steps up to the first GAP-flagged step**, and report sigma0 and pi0_raw. This tests whether contamination makes the null as wide as the data (findings 1–2). The restriction is label-free.
3. **Treat pi0_raw > 1 or a clipped π₀ as "null not identified"**, rather than silently clipping. Exclude discrete channels from per-group nulls (finding 3).
4. **Fix G2's LR to use π₀_g** and re-score as a labelled post-hoc variant. Also measure within-answer residual correlation between groups given the label, which tests Omri's conditional-independence rule directly (finding 4).
5. **Patch the verdict to require `ok` for PRMBench no-harm, and fix the README.** Correct the swapped runner/verifier columns and the claim that `labelfree/` is committed. Count `digit_active` in each FEATURES.npz to explain share_defined = 1 (findings 7, 10, 11).
