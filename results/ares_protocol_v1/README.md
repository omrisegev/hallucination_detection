# Our arms under the ARES readout protocol (2026-10-29)

**This applies ARES's READOUT protocol to our stored step scores. It is not a rerun of ARES's entailment-stability
method, which needs many LLM calls per step.** CPU only, no new inference. Development data.

## Protocol (frozen in `PROTOCOL.json` before this folder's numbers were computed)

- One global threshold per (cell, arm). It is chosen by 5-fold cross-validation **on labels** to maximize pooled
  Macro-F1, then applied to the held-out fold. Out-of-fold decisions are pooled.
- Folds group by source problem. They are assigned round-robin after a permutation with seed 20261021.
  - PRMBench: `c.groups`, the cell's source groups (707 groups, `prmb_source_v2` ids, which the v3 label release keeps). This is what
    `competitors.ares_rows` already uses.
  - DeltaBench has no source-group field, so it groups by identical normalized question text (444 groups for
    447 answers).
- Threshold grid: 99 quantiles (1%–99%) of the arm's score on the training folds. The PRM is swept on its raw risk;
  fused arms on per-answer z.
- Metrics: two-class (valid / erroneous step) Macro-F1, Macro-Precision and Macro-Recall on the out-of-fold decisions,
  plus the SD of the 5 per-fold Macro-F1 values. The label-free GAP rule on the same rows is shown for contrast.
- PRMBench Macro-F1 comes from `results/prm_view_selection_v1/stage2/competitors.py::ares_rows`, unchanged. An own loop
  with identical folds and grid gives P/R and the bootstrap, and reproduces `ares_rows` exactly.
- Populations:
  - PRMBench: all 6,969 answers; metric pool = the 83,371 steps of the non-control answers.
  - DeltaBench (**partial coverage**): the 447-answer evaluate_cab common set of 1,236 answers, 3,924 sections as
    units.
- Bootstrap: paired AUTO − PRM in out-of-fold Macro-F1. 1000 draws over source groups, seed 20261029. The fold
  thresholds stay fixed; they are not re-tuned inside the bootstrap.
- **Scale:** ARES reports 0–1. Ours is 0–100. The reference rows below show ARES ×100.

## Published ARES numbers (arXiv 2507.12948v2)

- **Table 1, Macro-F1, GPT-4o-mini / Qwen3-4B entailment model:**

  | method | PRMBench | DeltaBench |
  |---|---|---|
  | ARES | .640 / .636 | .708 / .498 |
  | Entail-Prev | .596 / .641 | .699 / .506 |
  | LLM-Judge | .643 / .675 | .381 / .494 |
  | ROSCOE-LI-Source | .570 / .495 | .453 / .457 |
  | ReCEval-Intra | .568 / .554 | .416 / .528 |

- **Table A7, Qwen2.5-Math-PRM-7B backbone, PRMBench:**
  - PRM alone: R .763, P .743, F1 **.749**.
  - ARES: R .751, P .733, F1 .736.
  - Table A7 has no DeltaBench row.
- Provenance:
  - The PRMBench values match `docs/research_notes/competitor_registry_2026-10-21/REGISTRY.md` §1.4.
  - The DeltaBench column was read from arxiv.org/html/2507.12948v2 on 2026-10-29 and has not been checked against
    a second source.
  - ARES's DeltaBench unit (section or step) and its subset are not stated.

## PRMBench: out-of-fold Macro-F1 under the ARES readout (0–100)

| arm | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B |
|---|---|---|---|---|
| PRM (backbone model's reward) | **68.17** | **69.23** | 66.24 | 55.79 |
| AUTO_N16 (backbone) | 67.55 | 67.95 | 66.81 | 64.20 |
| CHAIN_N16 (backbone) | 68.10 | 69.08 | **67.47** | 64.13 |
| TELE_N16 (backbone telemetry, no PRM) | 67.18 | 67.58 | 66.56 | 64.51 |
| TELE_N16 standalone (L-SML, telemetry only) | 66.18 | 65.71 | 65.70 | **66.27** |
| MEAN_N16 standalone (plain mean, telemetry only) | 65.36 | 65.28 | 65.29 | 65.66 |
| *ref: ARES Table 1, GPT-4o-mini / Qwen3-4B* | *64.0 / 63.6* | | | |
| *ref: LLM-Judge Table 1, GPT-4o-mini / Qwen3-4B* | *64.3 / 67.5* | | | |
| *ref: Table A7, PRM-7B alone / ARES on PRM-7B* | *74.9 / 73.6* | | | |

- Fold SD: 0.8–1.5 for every arm.
- Macro-P and Macro-R are in `ARES_RESULTS.csv`. Both sit within about ±1.5 of the F1 value.
- **The same rows under our label-free GAP rule (official PRMScore):**
  - PRM: 68.01 / 69.49 / 66.24 / 59.39.
  - AUTO: 69.87 / 70.18 / 69.06 / 65.92.
  - CHAIN: 70.40 / 71.09 / 69.51 / 66.25.
  - TELE backbone: 69.26 / 69.68 / 68.71 / 66.17.
  - TELE standalone: 68.26 / 67.80 / 67.63 / 68.20.
  - MEAN standalone: 67.14 / 67.39 / 67.06 / 67.45.
  - On PRMBench, GAP is higher than the labelled CV threshold for every fused arm, by 1.7 to 2.3 points.
  - For the PRM, GAP is 0.16 lower for Qwen-7B, equal for PURE, 0.27 higher for 72B and 3.6 higher for RLHFlow.
  - **Best per column** (bold) is the best arm under the ARES readout.
- **Paired bootstrap, AUTO − PRM, ARES readout, 95% CI:**
  - Qwen-7B −0.62 [−1.27, +0.06].
  - Qwen-72B −1.27 [−1.97, −0.59].
  - PURE +0.57 [−0.16, +1.28].
  - RLHFlow +8.42 [+7.64, +9.17].

## DeltaBench (partial coverage: 447 of 1,236 answers): out-of-fold Macro-F1 under the ARES readout (0–100)

| arm | Qwen-PRM-7B | Qwen-PRM-72B | PURE-PRM-7B | RLHFlow-PRM-8B |
|---|---|---|---|---|
| PRM | 56.94 | 55.78 | **55.43** | 49.21 |
| AUTO_N16 (backbone) | 58.04 | 57.00 | 48.54 | **53.93** |
| CHAIN_N16 (backbone) | **58.86** | **58.62** | 48.18 | 53.62 |
| TELE_N16 (backbone) | 58.38 | 56.93 | 55.31 | 53.81 |
| *ref: ARES Table 1, GPT-4o-mini / Qwen3-4B* | *70.8 / 49.8* | | | |
| *ref: LLM-Judge Table 1, GPT-4o-mini / Qwen3-4B* | *38.1 / 49.4* | | | |

- Fold SD: 0.6–2.9.
- GAP on the same rows:
  - PRM: 58.60 / 57.78 / 57.74 / 50.94.
  - AUTO: 57.22 / 57.89 / 49.50 / 54.23.
  - CHAIN: 58.22 / 58.92 / 48.71 / 53.83.
  - TELE: 56.94 / 57.12 / 54.58 / 54.75.
- **Paired bootstrap, AUTO − PRM, 95% CI:**
  - Qwen-7B +1.10 [−1.53, +3.47].
  - Qwen-72B +1.22 [−1.28, +3.72].
  - PURE −6.90 [−9.17, −4.59].
  - RLHFlow +4.72 [+2.60, +6.87].
- With PURE-PRM, AUTO and CHAIN score below 50, the level of a random rule. Those fused scores are anti-oriented on
  DeltaBench, consistent with CHAIN's earlier within-AUC of .395 in SUB5_ALL.csv.

## Caveats

- **Approximate comparison only.**
  - ARES's PRMBench and DeltaBench subset sizes are not reported.
  - Its threshold is tuned on labels, and ours is tuned the same way here.
  - Their PRM-7B alone scores .749 Macro-F1, while the same model scores 65.49 official PRMScore on the full
    PRMBench (68.17 under this readout). Their subset, or their unit or class definition, therefore differs from ours.
  - So we compare **within protocol**:
    - our arms vs the PRM on our rows;
    - ARES vs the PRM on their rows (ARES −1.3 below the PRM in A7).
  - We do not compare our absolute numbers with theirs. The published values are scaled ×100 for display only.
- Under this labelled readout, AUTO and CHAIN do **not** beat the two Qwen PRMs on PRMBench (AUTO is significantly
  below 72B). Their advantage over those PRMs appears under the label-free GAP rule. The large gain holds for
  RLHFlow.
- These are development data. AUTO and CHAIN were developed on PRMBench.
- One fix after the first run: the DeltaBench fold SD had been written on a 0–1 scale and is now ×100. Nothing else
  changed. The protocol, folds and every Macro-F1/P/R value are identical across both runs.

## Files

- `ares_protocol.py`: the script.
- `PROTOCOL.json`: the frozen protocol.
- `ARES_RESULTS.csv`: per (prm, telemetry, arm, benchmark):
  - ARES-readout `macro_f1_oof`, `fold_sd`, `macro_precision_oof` and `macro_recall_oof`;
  - the GAP contrast columns;
  - n_answers, n_steps and n_groups.
- `BOOTSTRAP.csv`: the AUTO − PRM paired bootstrap.
- `PARITY.json`: 92 checks, all PASS, max abs diff 1.4e-14. It covers:
  - COMP_ARES.csv (12 PRMBench values);
  - own loop = ares_rows (24);
  - AUTO GAP vs stage8 (4);
  - PRM, TELE, CHAIN and MEAN GAP vs SUB5_ALL (20);
  - DeltaBench: own GAP = evaluate_cab.row (16), AUTO vs AUTO_NEW (4), the rest vs SUB5_ALL (12).
- `run.log`.
