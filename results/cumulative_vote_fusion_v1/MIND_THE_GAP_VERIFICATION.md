# Verification of the Mind-the-Gap numbers used in Step 423

Omri asked (2026-09-20) to verify the Mind-the-Gap figures in `REPORT.md`, which looked odd
against the paper's own table. Three independent checks:

1. **Recomputed straight from the lane file**, bypassing the experiment script
   (`PER_QUESTION_LONG.csv`, rows with `method_id = mind_the_gap_common_replay`, `label != -1`):
   2,221 erroneous answers, SLA 23.32%, tolerance-one 50.88%. Per subset: gsm8k 28.50 (n=207),
   math 25.59 (594), olympiadbench 21.79 (661), omnimath 21.48 (759); late fraction 0.46 / 0.51 /
   0.49 / 0.55; mean signed offset +0.53 / +0.96 / +0.88 / +1.27 steps.
2. **Against the frozen package's own metric**: `REPORT.md` of
   `results/fair_paper_exact_comparisons_v1` reports the Mind-the-Gap common replay at native SLA
   0.23323 on 2,221 erroneous traces and tolerance-one SLA 0.50878. Identical to (1).
3. **Against the scoring code** (`spectral_utils/fair_comparisons/localization.py::_mind_gap_score`):
   evidence = negative Shannon entropy of the top-20 renormalized saved log-probabilities
   (`EVIDENCE_FNS["shannon"](row, 20)`), EMA span 5, `step_drop_scores` attributes flux j to the
   step holding token j+1 and scores a step by its most negative flux (`aggregation="min"`), locator
   = `nanargmax` over steps. That is the paper's Sec. 3.3 pipeline as pre-registered in
   `scripts/localization/evidence_drop.py` (`DROP_SIGN_CONVENTION`). Label and locator are both
   0-based (min label 0, min locator 0); the gated `discrete_prediction` equals the locator whenever
   the gate is open (100%), and the gate closes on 14.3% of erroneous answers.

So the numbers are what the frozen package says they are. Why they look odd:

- They are a **common-protocol replay on Llama-3.1-8B teacher-forced over ProcessBench's provided
  chains**, not the paper's own Qwen3-4B/8B generations. The paper reports 43–46% on GSM8K for
  Qwen3; Step 422's comparison table quoted those paper numbers against the project's Qwen L-SML
  arm. Nothing in the repository reproduces the paper's Qwen SLA under its own pipeline; the
  manifest `cluster/manifests/pb_llama31_8b_external_v1.json` records that the competitor was
  "NOT run on Llama-3.1-8B" by its authors.
- Under this replay the locator is systematically **late** (the most negative drop in a step is
  attributed to the token after the drop, then EMA-lagged), which is the offset pattern above. It
  is the latest of the five localizers on every subset, and the reason it receives a negative SML
  weight in the cumulative-vote fusion.

No number in `REPORT.md` changes. The caveat "common-protocol replay, not the paper's own scores"
was already stated there and is now backed by the three checks above.
