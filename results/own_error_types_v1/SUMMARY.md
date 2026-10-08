# Natural first errors on the models' own answers, by error kind (2026-10-07)

**Question.** Are the models' own first errors the same kinds of errors that PRMBench plants? Does the label-free score work on each kind?

**Data.** 417 Qwen3-4B/8B own answers (GSM8K, MATH) that contain an error. The first-error step is the consensus label of the LLM judges (Step 463). The scores are the existing label-free N16 ("B16") per-step scores, not refit. They reproduce the source exactly: BASE within-AUC 0.756624 over 388 answers, hit 0.263789 over 417.

**Classification.** Two blind passes by Claude agents. Each pass saw the problem, the reference solution, the numbered steps and the first-error step. It saw no scores. Categories are the 8 PRMBench types plus calculation_error, misread_problem, wrong_method and other (see PROTOCOL.json). Pass 1 labelled all 417. Pass 2 labelled 120 items (seed 20261007), with Cohen's kappa 0.861 (89.2% raw agreement). No human has checked the labels yet.

**Distribution.** 
- misread_problem 150 (36%)
- calculation_error 77 (18%)
- wrong_method 64 (15%)
- counterfactual 47 (11%)
- missing_condition 35 (8%)
- step_contradiction 28 (7%)
- domain_inconsistency 10
- confidence 5
- other 1
- redundancy, circular and deception: 0

70% of natural first errors are kinds that PRMBench does not plant.

**Performance.** BASE, 95% CIs over question groups. Column (a) is the source definition: the error step against the steps before it. Column (b) is the error step against all other steps.

| kind | n | within (a) | within (b) | hit |
|---|---|---|---|---|
| counterfactual | 47 | 0.877 | 0.844 | 23% |
| step_contradiction | 28 | 0.789 | 0.823 | 32% |
| domain_inconsistency | 10 | 0.862 | 0.843 | 30% |
| missing_condition | 35 | 0.679 | 0.719 | 26% |
| calculation_error | 77 | 0.766 | 0.788 | 33% |
| misread_problem | 150 | 0.736 | 0.793 | 27% |
| wrong_method | 64 | 0.720 | 0.740 | 17% |
| all PRMBench kinds | 125 | 0.796 | 0.801 | 26% |
| all natural-only kinds | 292 | 0.738 | 0.778 | 26% |
| all | 417 | 0.757 [0.726, 0.786] | 0.785 | 26% |

Step index alone scores 0.30–0.60 under (b), so position does not explain these results.

**Reading.**
- Natural errors are mostly different kinds from planted ones.
- Step ranking holds on them and is close to PRMBench where kinds overlap.
- The exact first-error hit is lower than on ProcessBench (26% vs 44%). That comparison is not paired: the answers differ and so do their lengths.
- Equivalence with ProcessBench was NOT shown in Step 463; do not describe it as equivalent.
- The weakest kinds are missing_condition (also the weakest planted type) and wrong_method.

**Caveats.**
- Only Claude agents labelled the kinds; a human spot-check of about 40 items is needed before citing the split.
- The metric definitions differ from PRMBench's.
- n is small for some kinds.
- Kind is confounded with dataset and step length.
- One pass-1 shard briefly saw the text of about 5 items from another shard. It saw no labels, and it discarded those items.

Files: PROTOCOL.json, labels_pass1.jsonl, labels_pass2.jsonl, AGREEMENT.json, PER_CATEGORY.csv, PER_CATEGORY_BY_DATASET.csv, PER_ANSWER.csv, analyze.py, figures/ (per_category_within_auc, P1_planted_natural from render_slide.py).
