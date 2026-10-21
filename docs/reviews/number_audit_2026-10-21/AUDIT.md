# Audit of the headline numbers in the advisor materials (2026-10-21)

Read-only audit. Every value below was opened in its source file; nothing was recomputed except trivial
differences of values already in those files (data.json, CSVs). Branches read: `claude/prm-boosting` (HEAD
bdada205), `origin/claude/integration-2026-10-17` (46b3a631), `origin/claude/prm-boost-external-2026-10-18`
(479ee146), `origin/claude/prm-boost-processbench-2026-10-19`.

Status key: VALID (current, can be sent) / STALE (true when written, superseded by a later run or fix) /
AFFECTED (computed on data or code that a later fix changes; value would move) / OUT-OF-FOCUS (ProcessBench,
Omri's decision; not invalid) / WRONG (false as stated now) / NEEDS-CHECK (could not verify).

## 0. Provenance: which rows come from which model and template

- **PRM boosting on PRMBench does not depend on the Qwen3 think-block fix.** `results/prm_boosting_v1/build_bank.py:1-16,102-133`
  takes the think-fix **mask1** source bank (AIRCC `/shared/.../thinkfix_rescore_v1/source_bank_thinkfix_mask1/FEATURES.npz`,
  recorded in `results/prm_boost_ablation5_v1/digitfix_v1/manifests/`) only as a layout, and replaces all 6,969
  PRMBench rows with features from **each PRM's own backbone** (Qwen2.5-Math-PRM-7B/72B, PURE = Qwen2.5-Math,
  RLHFlow = Llama-3.1-8B, Math-Shepherd = Mistral-7B), run on each PRM's model-card template
  (`cluster/run_prmbench_prm_telemetry.py:18,30,227`; no think block). The mask1 rule (drop forced token 0) is also
  applied to the PRM cells. The 6,800 ProcessBench rows of the base bank are Qwen3-8B/4B mask1 and are **never read**:
  `ablation5.py:47` and `lsml_bank_sweep.py` keep only `cells == prmbench`.
- Only arms that explicitly use Qwen3-8B telemetry depend on the template: `prm_boosting_v1/README.md:35,40,41`
  ("N16 Qwen3-8B" .7767/65.23 and the two mixes). These use mask1 (parity to `thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz`
  at 3e-15, `prm_boosting_v1/README.md:20-21`). Current.
- Socratic, Hard2Verify and ProcessBench boosting banks are built from each PRM's own telemetry with mask1
  (`prm_boost_external_v1/README.md:55`). No Qwen3 dependency.
- The bank rosters (N13/N16) were chosen on the old-template Qwen3 line; that is provenance, not invalidation
  (`thinkfix_rescore_v1/README.md:12-13`: N16 stays the highlight bank in all three variants).
- **Telemetry-only Qwen3 line:** old-template numbers are STALE. The only such numbers in these materials are
  Step 467 sections 3 and 5 (`sml_weight_estimator_v1/README.md:50-69,116-151`, disclosed at `:56`).
- **Digit ids:** 15..24 are correct for Qwen, PURE and Llama-3 (`digitfix_v1/README.md:6`; PURE verified on AIRCC,
  integration `prm_boost_processbench_v1/README.md:140-143`). Only Math-Shepherd (Mistral) had zero digit channels.
  Affected wherever an MS bank contains digit channels and was built before db054926: PRMBench N14/N16/S5/S6
  (bank sweep, controls; S5 and N16 rerun in job 288932, N14/S6 not), Socratic N16 (frozen eval, chain_socratic,
  pre-fix ablation, diagnosis LR, all of Step 467's Socratic runs), Hard2Verify N16 (external branch lacks db054926;
  `full_math_shepherd_hard2verify/bank_v2/MANIFEST.json` has no `digit_ids`). ProcessBench boosting used the v2
  builder with Mistral ids (commit 06d1a125) and is not affected.
- **Labels:** every PRMBench number here uses the v3 release (step_evidence_v1 OOF labels; one-based conversion in
  `ablation5.py:58-61`; `prm_label_contract.prm_error_flags` in `score_banks.py:87` and `answer_local/run.py:154`);
  PRMScore uses the official evaluator on raw one-based `error_steps`. No number found on v1/v2 labels.
  65.46 vs 65.49 for PRM-7B native is a tie rule on reward = 0.5, not a label version (`eval_lsml4/README.md:34-35`).
- **Socratic reference rebuild (Omri option 1):** old-builder rebuild differs from the bank_v2 used for every Socratic
  number by at most 5.4e-8 in features (integration `prm_boost_processbench_v1/README.md:144-152`). Negligible.
- **Constant-column guard (db054926):** changes results only when a view is constant, i.e. only the MS zero-digit
  banks above. Verified a no-op on qwen_prm_7b (49 arms, `digitfix_v1/README.md:8`). Step 467's code commit
  (3927d562) adds functions only; defaults unchanged.

## 1. Table

| # | claim / number | where | source (file:line) | variant / template | labels | affected by | status | what to do |
|---|---|---|---|---|---|---|---|---|
| I1 | PRISM metric identical; threshold-moved PRM reaches PRISM's point (68.26) | Index :17 | parity README :73-82 | PRM rewards only | v3/official | none | VALID | keep; use it to fix D7 |
| I2 | Qwen-7B PRMScore 65.49 -> 69.48 (3 label-chosen ch.) | Index :23 | eval_lsml4 README :19; bank sweep :60-61 | PRM backbone, mask1 | v3 | none (no digit ch.) | VALID | keep "label-selected" |
| I3 | CHAIN_N16 68.25; CHAIN_N13 67.82 | Index :27-28 | bank sweep :77,:69 | PRM backbone, mask1 | v3 | none (Qwen digits OK; 0 of 60 sweep fits have negative PRM weight, re-checked in SWEEP_DIAG_*.json) | VALID | keep |
| I4 | Socratic frozen: gain only on RLHFlow, 1 of 5 | Index :30 | external README :67-73,:86-88 | PRM backbones, mask1 | Socratic gold | sign rule on MS (LSML_N16_PRMG .451, LSML_CONT .485 within) | VALID (verdict) | note: fixed-½ control beats native on PURE too (+3.9) |
| I5 | Socratic CHAIN Qwen-7B 65.5 vs 68.1 | Index :33 | chain_socratic README :34 | same | gold | none (Qwen-7B: no flip, DIAG_qwen_prm.json) | VALID | keep |
| I6 | LR 0.836 vs PRM 0.824; energy_level .799/.801 vs .683 | Index :35-36 | diagnosis README :15-16,:59-60 | same | gold (label-using) | none for Qwen-7B | VALID | keep "label-using diagnostic" |
| I7 | ProcessBench "ready, not yet run" | Index :40-41 | integration PB README :116-169 | — | PB 0-based | ran: jobs 289058-67, 289297; verdict FAIL 1/5 | STALE + OUT-OF-FOCUS | update or drop |
| I8 | PRM+tele beats PRM in within-AUC on 5/5 by **+0.027 to +0.112** | Index :49; HISTORY :10637 | data.json (digit-fixed) | PRM backbone, mask1 | v3 | MS digit fix | **STALE** | lower bound .0275 is pre-fix MS (.7119-.6844, bank sweep :77/:60). Digit-fixed CHAIN_N16: **+0.037 (7B) to +0.112 (RLHFlow)** |
| I9 | L-SML over EQ +0.013 to +0.024, 5/5 on N16 | Index :49 | data.json CHAIN-EQ_N16 | same | v3 | — | VALID | note: "most of the gain is telemetry" holds on average, not on Qwen-7B (EQ-PRM .014 < CHAIN-EQ .023) |
| I10 | MS CHAIN_N16 .712 -> .778; gone after residualization | Index :50; PROGRESS :14; HISTORY :10638-39 | digitfix README :17; data.json CHAINRES | same | v3 | — | VALID | keep caveat |
| I11 | Sign bug lowered MS Socratic to 0.468; **3 of 20** Socratic fits, 0 of 60 PRMBench | Index :52; PROGRESS :5; HISTORY :10656; sml README :18-24 | DIAG_*.json | Socratic bank_v2 (pre-digit-fix) | gold | digit fix; count | **AFFECTED / NEEDS-CORRECTION** | .468 is N13 (= pre-fix N16; DIAG_ms_v2.json has identical N13/N16 weights). Current page N16 is .427, mechanism not re-diagnosed (but .427 ≈ 1-TELE .582). Anchor flip fired in **5 of 20 arm-fits** (MS CHAIN+PRMG x N13/N16, 72B CHAIN N13). Negative weights without a flip also in 72B N16 CHAIN (-0.39) and PRMG, RLHFlow N16. "0 of 60" VALID |
| I12 | unbiased fit loses: N16 PB 41.61 -> 38.61, PRMB .7958 -> .7762; nonneg helps N40/N43/N53; N16 L-SML stays best | Index :53; PROGRESS :6; HISTORY :10659; sml README :58-76 | processbench_banks/RESULTS.csv | **old Qwen3 template** | v3 | template fix | **STALE** (+PB OUT-OF-FOCUS) | On mask1 (frozen `thinkfix_mask1/cont_baselines/ABSOLUTE.csv`) N16 LSML_CONT = 40.02/.7792 and the oriented plain mean = 40.44/.7811: L-SML no longer beats the mean on N16. Rerun on AIRCC |
| D1-D5 | Deck table: PRM 65.49/.801/68.1; LSML_CONT 69.48/.850/64.7; CHAIN_N16 68.25/.838/65.5; CHAIN_N13 67.82/.829/62.7; PRMG 67.80/.830/65.1 | SLIDE :167-171 | bank sweep :60-77; external RESULTS_socratic.csv; chain_socratic :34 | PRM-7B backbone, mask1 | v3 / gold | none | VALID | keep |
| D6 | PRISM 65.5->68.0; NR 49.1->53.5; NCL 55->66; FPR 69.3->47.1; our baseline 65.49/49.0/54.9 | SLIDE :192-201 | parity README :45-52; PRM7B_BY_TYPE.csv | — | official | none | VALID | keep |
| D7 | "Without label-selected channels CHAIN_N16 gives 68.25, like PRISM's 68.0" | SLIDE :205 | parity README :73-82; data.json | — | — | — | **WRONG framing** | threshold-moved untouched PRM = 68.26; at PRISM's TPR CHAIN_N16 = 68.20; same label-free rule on the PRM = 67.32, so CHAIN adds +0.93 [+0.43,+1.42]. Contradicts Index :17 |
| D8 | "We will compute pooled AUROC / step accuracy" | SLIDE :233,:263 | parity README :113-166 | — | — | done 10-18 | STALE | replace with results (pooled AUROC CHAIN_N16 .794, kept 23.3%; SLIDER bal. acc 72.8-74.3) |
| D9 | "our macro-F1 (67.7)" | SLIDE :263 | parity METRICS.csv (NR) | — | official | — | VALID | it is LSML_CONT (label-selected) NR 67.66; CHAIN_N16 68.68, N13 69.22 |
| D10 | Socratic 68.1 -> 62.7-65.5, only RLHFlow; **"therefore we did not run Hard2Verify"** | SLIDE :275; page missing.html :12, template.html :115 | external branch HARD2VERIFY_RESULT_2026-10-19.md | — | gold | Hard2Verify ran 10-19 (AIRCC 288968, overflow rows removed) | first part VALID; **H2V sentence WRONG** | H2V CHAIN_N16 vs PRM Q80, common 177: 7B -0.009, 72B +0.011, PURE -0.100, RLHFlow +0.183, MS +0.045 (MS on zero-digit bank) |
| D11-13 | energy_level NR 67.6 / NCL 70.5; PRM Q80 NR 55.2 / NCL 63.8; slide 2 .799/.801, .683/.824 | SLIDE :276-277,:366 | PRM7B_BY_TYPE.csv; diagnosis :59-60 | PRM-7B, mask1 | v3 | none | VALID | keep |
| M1 | Same-rule PRMScore: 7B +0.93, PURE +2.57, 72B -0.14 | page missing :2 | data.json | same | v3 | — | VALID | — |
| M1b | "MS: 10.7 of 13.1 points come from the rule" | page missing :2 | take.json :30 (pre-fix) | — | — | MS digit fix | **STALE** | digit-fixed: CHAIN_N16 - native +17.35, of which 10.71 is the rule; vs same rule +6.64 |
| M2 | CHAIN_N13 +.026 to +.096; EQ-MEAN >0 in 5; PERM kills gain (287777) | missing :4 | data.json; controls README :19,:33-35 | same | v3 | — | VALID | — |
| M3 | CHAIN>EQ on N16 5/5; MS N13 .727 vs .737; RLHF HALF 60.46 vs 63.92; Socratic ½ beats L-SML 4/5 | missing :6 | data.json | same | v3/gold | sign rule (Socratic MS, 72B) | VALID (conclusion holds with nonneg weights, sml README :38-45) | — |
| M4 | N16 67.20 (backbone) / 65.93 (Instruct) / 65.23 (Qwen3-8B) vs PRM 65.46 | missing :8 | prm_boosting_v1 README :34-37 | Qwen3 arm = mask1 | v3 | — | VALID | note 65.46 uses the old tie rule; elsewhere 65.49 |
| M5 | POS .662, LEN .619; RES cost .016-.027, MS .066; vs same rule -0.05/-1.15; Socratic POS .731 | missing :10 | data.json | same | v3/gold | — | VALID | — |
| M5b | "CHAIN beats PRM+POS+LEN in 4 of 5 (not MS on N16)" | missing :10 | data.json; digitfix README :30 | — | — | MS digit fix | **STALE** | digit-fixed MS CHAIN_N16 .778/64.37 vs PRMPOSLEN .743/60.59 (+.036/+3.78): 5 of 5 on N16 |
| M6 | Socratic CHAIN 14-16 PRMScore below native on 72B and MS; MS CHAIN .427 | missing :12,:14; PROGRESS :15; HISTORY :10640 | data.json | Socratic | gold | sign rule | AFFECTED | say the fused score is inverted on MS (and 72B N13) by the sign rule; 72B N16 has a negative group weight (nonneg .613 -> .648) |
| M7 | ~208 contrasts per PRM | missing :16 | ABL5_BOOTSTRAP_*.csv | — | — | — | NEEDS-CHECK | file has 184 contrasts x 2 metrics; +26 controls = 210 |
| M8 | 72B CHAIN_N16 vs native +0.73 [+0.10,+1.40]; 7B Simplicity 52.0->68.9, same rule 60.4; MS ~6 pts | missing :16,:18 | data.json; bank sweep :113 | same | v3 | — | VALID | — |
| P | page data provenance | data.json | build_data.py :58-65 | — | — | — | VALID | all 49 MS PRMBench arms and 51 MS pairs come from digitfix; ablation5.html == template+data+missing (byte check) |
| S1 | sml §2: CHAIN N16 signed/abs/clip/nnr1, MS .468 -> .564-.599 | sml README :38-45; HISTORY :10658 | socratic_sign/*.csv | pre-fix bank_v2 | gold | MS digit fix | AFFECTED (MS column only) | label MS as N13-equivalent; digit-fixed N16 untested |
| S2 | sml §4 placement; "external benchmarks show the same picture, CHAIN near telemetry" | sml README :94-114 | RESULTS_socratic_placement.csv | pre-fix | gold | digit fix (MS); PB MS inversion | AFFECTED (MS) | on PB MS, CHAIN_N16 AUROC .312 ≈ 1 - TELE .696: near minus telemetry |
| S3 | sml §5 answer-local SML (all numbers) | sml README :124-151 | answer_local/*.csv | **old template** | v3 | template; PB focus | STALE + OUT-OF-FOCUS | grouped one-answer L-SML (.766 vs .799) is a 1,000-answer subset (LSML_LOCAL.csv 2x1,000 rows): feasibility only |
| P3 | "decide sign handling before ProcessBench Step 7" | PROGRESS :7 | integration 46b3a631 | — | — | Step 7 ran 10-20 with the signed rule | STALE | — |
| P5 | voters "Next: fetch"; registry pins 9 numbers, PASS | PROGRESS :22,:24 | configs/current_line_registry.json | mask1 + LSML_CONT | v3 | none (LSML_CONT has no digit ch.) | voters STALE; pins VALID | — |
| R1 | parity table, PRISM/CROP/SLIDER/OptAgg numbers | parity README :26-31,:45-82,:110-131,:159-166,:197-203 | METRICS.csv | PRM-7B, mask1 | v3 | none | VALID | — |
| R2 | "ProcessBench is a GAP: no PRM-backbone telemetry"; "Nothing is committed" | parity README :87-90, :6 | — | — | — | PB telemetry now exists; dir committed (419bb905) | STALE (+OUT-OF-FOCUS) | — |
| X1 | bank sweep "identical rows are real, a fitted outcome" (MS N11=N14, AUTO_N13=N16) | bank sweep README :50-54 (cited by deck) | — | — | — | digit bug | **WRONG** | cause is the constant zero digit channels; MS N14/N16/S5/S6 rows AFFECTED |
| X2 | chain_socratic "MS N13 = N16, the digit group gets zero weight"; "telemetry dominates" | chain_socratic README :38-41,:45-49 | — | — | — | digit bug; sign rule | WRONG explanation / AFFECTED | — |
| X3 | PB Math-Shepherd CHAIN AUROC .313 | integration PB README :168,:174 | eval/RESULTS_processbench.csv; scores/SCORES_CHAIN_math_shepherd.json | v2 builder, correct digits | PB 0-based | sign rule (suspected) | NEEDS-CHECK (likely AFFECTED), OUT-OF-FOCUS | evidence: CHAIN .312 / PRMG .306 / LSML_CONT .394 vs PRM .597 and TELE_N16 .696; corr(TELE,PRM) -0.113 (N13), -0.073 (N16). Verdict FAIL unaffected (Qwen PRMs positive, corr .39-.45) |

## 2. (a) Statements that are wrong and must not be sent
1. Deck :275, page missing.html :12 and template.html :115: "therefore Hard2Verify was not run". It was run (AIRCC 288968).
2. Deck :205: CHAIN_N16 68.25 presented as reaching PRISM's 68.0. The untouched PRM with a moved threshold gives 68.26, and the same label-free rule on the PRM gives 67.32.
3. Index :49 and HISTORY :10637: "+0.027 to +0.112". Use +0.037 to +0.112.
4. Index :52 / PROGRESS :5 / HISTORY :10656 / sml README :22: "3 of 20". DIAG files show the anchor flip in 5 of 20 arm-fits; and 0.468 is the N13 value, not the current N16 (.427).
5. Page missing.html :2 "10.7 of 13.1" and :10 "4 of 5 (not MS on N16)": pre-digit-fix.
6. Index :53 / PROGRESS :6 "N16 with the current L-SML remains the best": old template. On mask1 the plain mean beats L-SML on N16.
7. Index :40, parity README :87-90: ProcessBench "not run" / "GAP".
8. bank sweep README :50-54 and chain_socratic :41: zero digit weights called a fitted outcome.

## 2. (b) Fine to send as is
I1-I6, I9, I10; deck table D1-D6, D9, D11-D13; page items M1 (first part), M2, M3, M4, M5 (first part), M8; the digitfix README numbers; parity README sections 1-4 except :6 and :87-90; registry pins. All are development data on already-inspected PRMBench.

## 2. (c) What must be recomputed, and where the data is
| item | data needed | where |
|---|---|---|
| Text fixes 1-5, 7-8 above | data.json, digitfix README, H2V note | local / git |
| Step 467 sections 3 and 5 on thinkfix_mask1 (`score_banks.py`, `answer_local/run.py`) | mask1 source bank or thinkfix raw cells (PRMBench + 8 PB) | **AIRCC only** (locally: old PRMBench pkl, thinkfix manifest only). Interim: frozen mask1 `ABSOLUTE.csv` (local) already gives LSML_CONT vs ALL_EQ per bank |
| Sign diagnosis on digit-fixed MS Socratic N16 (DIAG, nonneg modes, placement) | `socratic_bank_v3` (job 288932, MANIFEST_socratic_bank_v3.json) + local gold | AIRCC (Drive if uploaded) |
| LSML_CONT sign check on Socratic MS (.485) | pre-fix bank_v2 is enough (no digit ch.) | Drive; a copy sits in an ephemeral scratchpad of session 35cf95ec |
| PB MS inversion confirmation: corr(CHAIN, TELE) per step | `scores/SCORES_math_shepherd.npz` (job 289297) | AIRCC; low priority (out of focus) |
| Hard2Verify MS rebuild with v2 builder (`--tokenizer`, guard) | H2V MS telemetry | AIRCC, CPU |
| Bank sweep MS N14/S6, controls MS N16 | digit-fixed PRMBench MS bank (job 288932) | AIRCC |
| Contrast count "~208" | ABL5/controls bootstrap CSVs | local |
