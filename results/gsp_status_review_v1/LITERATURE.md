# Literature benchmarks and positioning (agent 3, 2026-11-13)

Desk work; numbers from repo competitor notes (REGISTRY.md, BENCHMARKS.md, competitor_primer) checked 2026-10-21/22 plus web 2026-11-13. U = unverified. Re-check every external number against the PDF before a paper.

## Benchmarks
| Benchmark | Cites (S2) | Official metric | Needs answer-level "no error" decision? | Our coverage |
|---|---|---|---|---|
| ProcessBench (ACL'25) | 352 | F1 of acc on erroneous + clean answers | Yes (we flag every answer, F1 ~0) | Full; SLA 38.5/38.1/31.3/34.3 (non-official MtG reading) |
| PRMBench (ACL'25) | 136 | PRMScore 0.5 F1 + 0.5 F1_neg, 9 categories | No | Full; TELE16 ~68.0 |
| Socratic-PRMBench | 3 | PRMScore | No | Full; 63-66, below PRMs |
| Hard2Verify | 22 | Balanced F1 (step), threshold tuned on 100 labelled responses | Step: no | Full; ~43-44 |
| DeltaBench (ACL'25) | 58 | Section F1 with dataset-level mu-sigma threshold (label-free), macro | No gate; global threshold | Data on Drive; must rerun official protocol |
| MR-GSM8K / MR-Ben | 77 / 54 | MR-Score incl. error reason (text) | Yes | Not run; partial only |
| CRV (ICLR'26) | 28 | Pooled AUROC | No (between-answer) | Run; telemetry fails |
| MedPRMBench | 2 | PRMScore | No | Not run; candidate untouched test |

## Comparators (F = same access, P = label-free score but labelled threshold or many calls, N = trained/supervised)
- PRMBench: Qwen-PRM-7B/72B 65.5/68.2 (N); PURE 65.3, RLHFlow 54.4 (N); PRISM 68.0 (N); PathFinder 67.7 (N); GPT-4o 66.8, o1-mini 68.8 (N). No published training-free telemetry baseline on PRMBench found.
- ProcessBench F1: Qwen-PRM-7B/72B 73.5/78.3 (U; we reproduce 72.9/78.2); o1-mini 87.9; Temporal Consistency 69-71 (P/N, many calls); uPRM 58.3/52.6/42.7/39.8 (N). Mind the Gap Qwen3-8B macro SLA 39.39 (P, closest competitor). MtG reproduced on our telemetry: 30.6 (results/mtg_reproduction_v1).
- Hard2Verify step balanced F1: Qwen-PRM-7B/72B 42.37/35.50, UniversalPRM 60.27, GPT-5 85.83 (N, label-tuned threshold).
- DeltaBench mu-sigma F1: Qwen-PRM-7B/72B 29.22/26.38, GPT-4-turbo critic 40.76.
- Training-free lenses (entropy, PPL, MaxProb, CoE, arXiv 2506.08410): F, but on GSM8K/MATH500 step AUROC; port as baselines.

## Gaps and minimum experiments
1. Freeze recipe (pick R0 or R1 now) with hash.
2. Matched baselines from same pass: mean entropy, max-prob, PPL, MtG drop, plain average, best single channel.
3. Paired bootstrap by source question.
4. ProcessBench: label-free answer gate, or report SLA + first-error AUC as explicitly non-official; MtG and ours from the same reader model.
5. DeltaBench official rerun.
6. Untouched confirmation: MedPRMBench test (same PRMScore) or MR-GSM8K / BIG-Bench Mistake.
7. Contamination statement (Qwen2.5-Math vs MATH/PRM800K, U).
Reviewer risks: standalone PB TELE16 ~ digit_spread alone (38.5); only PRM parity rows are reproduced.

## Positioning
- Training-free, label-free, PRM-free step-error localization from one forward pass; no published telemetry baseline on PRMBench.
- Headline: cheap alternative reaching supervised-7B-PRM level on PRMBench, not "beats PRMs".
- Include the negative half (answer-level decisions, between-answer metrics) with the answer-standardization theory.
- Venue: ACL/EMNLP via ARR (Findings realistic); fallback TMLR or workshop.
- Release hashes, per-step scores, evaluator ports, PRM parity tables; zero extra calls.
