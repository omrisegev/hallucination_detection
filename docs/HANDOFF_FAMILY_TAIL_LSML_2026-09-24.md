# HANDOFF: family/tail L-SML line and transfer status (2026-09-24, end of session)

This handoff is for the next research session. Read it after `PROGRESS.md` and before planning anything on fusion or transfer. Numbers come from the files named here. Every table is full-population (13,769 source answers; 6,190 external answer/backbone records).

## 1. Where we stopped

| Branch | Commit(s) | What is there |
|---|---|---|
| `claude/ssl-pseudolabel-residual-v1` (worktree `.worktrees/ssl-pseudolabel-residual-v1`) | 62078ae50 + the V3 commit | Source stages, HISTORY Steps 441-443 and 445, locks V1/V2/V3 (source copies) |
| `lsml-ct7-levers-run` (worktree `.worktrees/lsml-ct7-levers-run`) | 1a3446b6e | Token-level tail L-SML, original run and calibration-corrected replay |
| `codex/lsml-external-generalization-v1` (main checkout) | 1d3c23681 (Codex V1), 33d0acf8d (V2), + the V3 commit | External pipeline, results V1/V2/V3, HISTORY Steps 444 and 446 |

- **Nothing is pushed.** `git push` needs Omri's terminal if the credential helper fails.
- **The main checkout contains Codex's own uncommitted, unrelated work.** It is the group-confidence experiment: `results/lsml_group_confidence_v1/`, `spectral_utils/lsml_group_confidence*.py`, and edits in HISTORY/LESSONS/PROGRESS. Claude's commits staged only Claude's hunks. Do not commit, revert or overwrite Codex's edits.

### Decisions in force

- **Frozen bank11 L-SML is the leading learned transfer method.** It is also the only representation where learned weights beat matched equal both on source (+0.84 PRMScore) and externally.
- **The family/tail L-SML line is closed** at the step level: 15 families, tail-mark learning, top-20% and top-1, common or per-family thresholds.
- **CT7 is a source reference only.** It is last externally.
- **Hard2Verify balanced F1 and Socratic PRMScore are never averaged.**

## 2. What was established (in order)

1. **Evaluation fix (Step 442, `results/family_tail_calfix_v1/`).**
   - What changed: same-model calibration (3 fit / cal (k+1)%5 / eval k), write-once score bundles, P1 = pooled q80, paired source-group PRMScore bootstrap with Holm.
   - It moved numbers by at most 0.14 pp.
   - It reproduces Codex's frozen bank11 validation to machine precision.
2. **K=2 was an input-scale artefact (Step 443, `results/tail_threshold_calibration_v1/`).**
   - Centred tail marks (variance ~0.18) were fed to `lsml_continuous` unstandardized. Its default `loading_scale='unit'` K criterion is scale-sensitive: the same continuous features multiplied by 0.4 also give K=2.
   - Standardized marks give K=4 with a stable partition.
   - Calibrating the tail threshold does not beat matched equal on source: common, per-family, label-free (FUSE-style) or label-selected.
3. **External results.** Hard2Verify balanced F1 / Socratic-Qwen3 PRMScore / Socratic-QwQ PRMScore, 100,000 source-question draws, Bonferroni within each lock family:

| Method | Lock | Source PRMScore | Hard2Verify | Socratic-Qwen3 | Socratic-QwQ |
|---|---|---:|---:|---:|---:|
| bank11 L-SML (frozen) | V1 | 64.17 | 43.67 | **63.22** | **64.24** |
| bank11 max-step tail (unoriented) | V3 | 63.25 | **44.48** | 62.38 | 63.65 |
| bank11 max-step tail (oriented) | V3 | 63.75 | 43.86 | 61.82 | 63.12 |
| family15 equal | V1 | 64.33 | 42.38 | 61.12 | 62.94 |
| family15 max-step tail | V3 | 63.76 | 42.79 | 61.05 | 62.28 |
| family15 top-20% tail, corrected | V2 | 64.17 | 42.28 | 61.00 | 62.30 |
| family15 top-20% tail, V1 (K=2) | V1 | 64.52 | 41.02 | 59.92 | 62.69 |
| bank11 equal | V1 | 63.33 | 40.88 | 60.79 | 61.50 |
| CT7 | V1 | 64.62 | 37.75 | 58.75 | 60.17 |

   - Sources: `results/family_tail_external_v{1,2,3}/METRICS.json`, `CONTRASTS.json`, and the Hebrew `REPORT_HE.md` files in v2/v3.
   - Each V2/V3 run replays every earlier arm and contrast exactly (difference 0.0).
4. **Main methodological finding: source rank did not predict external rank, three times.**
   - On source, CT7 and the family variants led and bank11 L-SML trailed.
   - Externally the order reverses: bank11 L-SML leads and CT7 is last.
   - bank11 max-step tied bank11 equal on source but beats it by +1.6 to +3.6 externally.

## 3. Recommended next research (priority order; each needs Omri's go)

1. **A source-side transfer proxy before any new method.** Source five-fold PRMScore selected the wrong methods three times. Build a split that mimics a domain shift and check whether it would have ranked bank11 L-SML above the family variants and CT7. Options:
   - leave-one-benchmark-out (fit on PB, evaluate PRMB)
   - leave-one-PB-cell-out
   - PRMB category holdout

   Checking the proxy against the known external ranking uses exposed data; label it as proxy validation, not confirmation. If no proxy tracks transfer, say so and prefer simpler, more constrained representations (bank11).
2. **Why bank11 L-SML transfers.** Look at partition and weight stability across source folds, and at how external per-channel behaviour differs from source (diagnostic only, on exposed data). Step 438/440 memory notes: its edge is de-noising (suppressing three anti-oriented channels) with one fixed K=6 partition.
3. **Independent audits of V2/V3.** Codex's three audit scripts (metric recompute, coverage, null) were run for V1 only. The recompute script lives in `results/family_tail_external_v1/independent_metrics/`.
4. **Untouched confirmation.** Any new candidate must be frozen on source and tested on data not yet exposed. MedPRMBench is deferred; the handoff there is `docs/experiments/LSML_EXTERNAL_GENERALIZATION_V1.md`. Hard2Verify/Socratic are exposed and can only give exploratory follow-ups.
5. **Guard the library contract.** Callers of `lsml_continuous` must pass z-scored inputs or use `loading_scale='complete'`. `tail_calib_common.lsml_fit_scaled` enforces this for the new code. Frozen external modules must not be edited in place; any hard guard in `spectral_utils/external_generalization` needs a new version.
6. **Do not reopen without a new idea:**
   - tail-threshold tuning
   - per-family thresholds
   - 16 families or within-family SML
   - token-level tail fusion (below equal, and fusion-before-readout costs -0.92)
   - FUSE-style consistency objectives (they improve model fit and hurt PRMScore)

## 4. Reproduction dependencies (fragile!)

- **The source pool is outside git, in a temporary folder:**
  - `pool_z.npy` (sha256 d9abccfb561f459d2f3a6c5b4346a1f6766df7fdbaddb230338aa40349077a16) and `pool_names.json`
  - Path: `C:/Users/DELL/AppData/Local/Temp/claude/c--Users-omris-TAU-hallucination-detection/2d14a8c9-8b3f-489b-b26f-812b4b84a8b3/scratchpad/`
  - All source scripts hard-code it. If the temp folder is gone, restore `reproduction/source_pool/` from Codex's verified Drive archive (`results/family_tail_external_v1/ARCHIVE.json`, `RESTORE.md`) and point the scripts at it.
- **Population:** `.worktrees/readout-quickest-detection-v1/results/step_evidence_v1/` (OOF_ANSWERS.csv, OOF_STEP_SCORES.npz, INPUT_FREEZE.json with the PRMBench metadata path).
- **External private inputs** (not in git): `scratch/external_generalization_private/`, holding inputs, pinned official evaluator sources and raw telemetry archives. Per-answer records and bootstrap arrays of v1/v2/v3 are git-ignored. v2/v3 can be regenerated from the v1 records in minutes: `scripts/run_family_external_v{2,3}.py`, then `evaluate_family_external_v{2,3}.py --seal-only`, then the same script without the flag.
- **Tests:** `scripts/experiments/test_calfix_common.py` and `test_tail_calib_common.py` on the ssl branch (known-result tests, all pass).

## 5. Rules that bit this line (see LESSONS.md on the ssl branch)

- z-score every matrix given to `lsml_continuous`. Before explaining a surprising K, rerun at another input scale.
- Test write-once guards with a flag, never by matching the text of the exception.
- Lock recipes before reading external results. Record exposure timing in every new lock. Never select arms on external data.

## 6. Paste-ready prompt for the next session

```text
תקרא את PROGRESS.md ואז את docs/HANDOFF_FAMILY_TAIL_LSML_2026-09-24.md (על branch codex/lsml-external-generalization-v1, main checkout).
קו ה־family/tail L-SML סגור; bank11 L-SML הקפוא מוביל בהעברה החיצונית, ובשלוש הנעילות (V1-V3) הדירוג במקור לא ניבא את הדירוג החיצוני.
המשימה: להציע ולתכנן (לפני כל הרצה) פרוקסי בצד המקור שמנבא העברה - למשל leave-one-benchmark-out (PB->PRMB) או leave-one-cell-out -
ולבדוק רטרוספקטיבית אם הוא היה מדרג את bank11 L-SML מעל גרסאות המשפחות ו־CT7 (מסומן כבדיקת פרוקסי על נתונים חשופים, לא אישוש).
לא לכייל שום דבר על Hard2Verify/Socratic. לא לגעת בשינויים הלא־committed של Codex ב־main checkout. לבדוק ש־pool_z.npy עדיין קיים (סעיף 4) לפני כל הרצה.
```
