# answer_gate_v1: the project's answer-level detectors as the "whether and how many to flag" decision

Omri 2026-09-30: use the original answer-level method (answers x features, label-free fusion) to decide whether and how
many steps to flag; try both the original spectral features + continuous L-SML (GOOD_5) and U-PCR on the full pool.
Step scores (S_equal) and their within-answer ranking are frozen; only the per-answer flag count changes.

- Protocol `PROTOCOL.json` frozen at f745f1834; amendment A1 (bc64af3f5) before the full run (min_spilled float32
  residue treated as constant, fixed-direction length control, AUROC on zA with 10,000 draws); pre-run review PASS.
- Features: 27 label-free answer-level views from the token telemetry (16 spectral entropy features, 4 surprisal, 4 energy,
  3 means), 13,769 answers, no missing value (`FEATURE_MANIFEST.json`). Adaptation: 3 canonical views need full top-k lists.
- Run `run_20260930` (32f795d22). Red team `RED_TEAM.md`: numbers confirmed; C1 and C3 weakened, C2 and C4 refuted as
  read, C5 confirmed. Development evidence; not untouched confirmation.

## What holds

1. **Answer-level detection works on ProcessBench** (erroneous vs correct answers, 6,800 answers, 4 answer sets x 2
   quantizations): U-PCR 0.773, equal average 0.781, L-SML GOOD_5 0.755, mean entropy 0.742 macro AUROC; far above
   label nulls (0.50). But the fused detectors' edge over mean entropy is answer length: within length deciles mean
   entropy alone is best (0.770 vs U-PCR 0.745), and entropy + log length reproduces U-PCR.
2. **L-SML on the full pool loses to the plain average of the same pool** (-0.096 AUROC, robust to length control), as
   in every earlier stage.
3. **On ProcessBench, a gate helps a lot, but mostly generically.** Official F1: frozen rule 0.062 (it flags 96.6% of
   correct answers); random gate with the same unflagged share 0.272; R2 calibrated per benchmark 0.341; the OFFSET rules
   0.337-0.344; a PURE gate (leave the least suspicious answers unflagged, frozen rule inside) 0.362 (U-PCR) / 0.363
   (mean entropy). The answer-level ranking adds about 0.09 over random. Error localization itself does not improve
   (first-error accuracy falls).
4. **On PRMBench the answer-level decision lowers PRMScore** (U-PCR -0.049, L-SML GOOD_5 -0.021 vs the frozen rule):
   97% of scored answers contain an error, so silencing answers removes true error flags, and moving flags into high-
   suspicion answers loses because error density is flat across detector quintiles.

## What does not hold (as first read)

- The erroneous-vs-multi_solutions separation (0.92-0.98) is a STYLE artefact: multi_solutions are a different
  generation process (verbose "Step N" format, twice the tokens per step); the detectors separate them almost as well
  from the correct controls. PRMBench has no style-matched correct answer.
- The length control's PRMScore 0.6822 (above R2) is a CONSTRUCTION PRIOR: PRMBench inserts about 2 errors per answer
  regardless of length, so the error share falls from 0.33 to 0.10 with length. Flagging the top-2 steps in every answer,
  with no answer-level score, gives 0.6808 (post hoc, diagnostic only; it does not transfer to ProcessBench).
