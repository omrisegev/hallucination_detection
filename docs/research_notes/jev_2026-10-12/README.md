# Jev (TypeSafe AI): error detection and fit to our method

Date: 2026-10-12. Source: Claude project thread "Jev error detection" (Omri asked in Hebrew; this is the English record).
Status: literature scan only. No experiments, because development is frozen until the advisor meeting on 2026-10-15.

Files in this folder:
- `jev_error_detection.html`: the full scan page with citations (same as the artifact https://claude.ai/artifact/9xWfn9oUsLbpwFNvhbdJ6v).
- `README.md`: this record of the conversation.

## 1. What Jev is

- A closed "typed decision model" ("System One model") from TypeSafe AI. Early access since 2026-09-15, version `jev-1.13.0`. Source: https://en.wikipedia.org/wiki/Jev_(AI_model)
- Input: a text state plus typed questions (Choice, Score, yes/no "Noul"). Output: a probability distribution over the options the caller defines, in one parallel pass. It writes no text and no reasoning steps.
- No weights, no paper, no logprobs. Training method "RLCD" (RL for Calibrated Decisions) is not described.
- "No hallucination" means only that the output is always inside the schema. TypeSafe writes that the 0% number "is not empirical". Jev can still choose a valid wrong option with high probability.
- Open "Jev-like" models exist (Kev, Tev1, decider, Strands Decider, Laya). Most are Qwen3.5 fine-tunes with a pointer head, a mask-token head, or first-token option logits.

## 2. Is there error-detection work? (Omri asked for the list, and was right to doubt "a lot of work")

Correction: "a lot of work" was too strong. All items are non-refereed arXiv preprints, posted 7-14 days after release. Most are small API-only evaluations (cents per run). One survey counted 28 preprints from 19-24 September alone. All are black-box (output probabilities only). None studies internals or step-level errors. All arXiv links below were opened and confirmed on 2026-10-12.

arXiv preprints on errors / calibration of Jev or similar models:
1. JEV-as-a-Judge (CMU, 2026-09-22): https://arxiv.org/abs/2609.26550 . AUROC of confidence for error detection; cascade to GPT-6.
2. Type-Safe Is Not Error-Free (2026-09-22): https://arxiv.org/abs/2609.26758 . Option renaming (0/1 to no/yes) flips 70.4 more answers per 100.
3. Beyond Calibration: probability axioms (2026-09-29): https://arxiv.org/abs/2609.33209 . Label-free coherence checks across related questions.
4. JevAdvBench (2026-09-25): https://arxiv.org/abs/2609.31142 . One appended opinion flips 12.1% of decisions.
5. Evaluating and Benchmarking the System One Model Jev (2026-09-29): https://arxiv.org/abs/2609.37647 . Choice calibration good; binary 0.5 threshold badly placed.
6. Evaluating Decision Models for Text Annotation (2026-09-21): https://arxiv.org/abs/2609.24574 . High confidence near chance on empathy detection.
7. System One Models for Agent Security Decisions (2026-09-27): https://arxiv.org/abs/2609.33401 . Good average calibration hides confident failures.
8. From Text Decisions to Pixels (2026-09-24): https://arxiv.org/abs/2609.29283 . Accuracy gains do not give calibrated probabilities.

Surveys: Typed Decision Models: Early Evidence Audit (2026-09-26) https://arxiv.org/abs/2609.32160 ; Jev in the Wild (2026-09-24, 2,170 GitHub repos) https://arxiv.org/abs/2609.30216

Not papers: Vals AI eval https://www.vals.ai/blogs/independent-evaluation-of-jev ; GitHub audits https://github.com/scienthoon/jev-ood-calibration , https://github.com/jujumilk3/jev-calibration-audit , https://github.com/goya4140/jev-reward-model-evaluation

## 3. Jev on our benchmark

The community repo goya4140 reports Jev 1.13 at 66.38 "official PRM score" on PRMBench (6,216 records) and ProcessBench F1 69.5. Our label-free Qwen3-8B Overall is 65.4; Qwen2.5-Math-PRM-7B is 65.5. The protocol is not published, so this is a reference point, not a matched comparison. Do not put it in the advisor page as a direct comparison without a check.

## 4. Which benchmarks are used to evaluate Jev

No official benchmark. TypeSafe used internal workflows with two LLMs as reference labels. Outside studies chose their own:
- Judge / reward: RewardBench 1 and 2, JudgeBench, RM-Bench, HaluEval, PPE, RubricBench, ProcessBench, PRMBench.
- Classification / QA: OpenBookQA, CommonsenseQA, HellaSwag, MMLU-ProX, Banking77, XNLI, GoEmotions, LegalBench, UNFAIR-ToS, 18 social-science annotation tasks.
- Special: Vals 10-K claim verification, KoBBQ, spam and phishing sets.

## 5. Omri's idea: use our algorithm with Jev outputs as scorers

Claude's opinion: a good idea. L-SML / MEAN_MOM need several noisy scorers per item, and Jev can supply them. Two forms:

1. Jev as one more scorer in our PRMBench bank (recommended first). Ask one Noul per step ("is this step correct?") and add the probability to the bank. Jev errors should be very different from Qwen3 internal-signal errors, which helps the conditional-independence assumption of SML. Cost is about $1 for all of PRMBench.
2. Our algorithm on Jev's own tasks. The phishing study (https://www.beri.net/article/typesafe-jev-typed-decision-model-calibration-decomposition-shadow-eval) split one decision into 5 questions and fused them with LR trained on 1,000 labels (62.6% to 95%). We can try to do this fusion label-free. Clear pitch, but further from PRMBench.

Technical problems:
- The API rounds probabilities to 0.01, and many are exactly 0. This gives many ties in rankings.
- Different question forms (Choice vs Noul) are not coherent with each other. Do not mix them in one bank.

Also possible later: apply our label-free internal signals to an open decision model (Kev-4B or Tev1-4B, Qwen3.5) and test whether they detect wrong decisions better than the model's own probability. No published work does this.

## 6. Next steps (after 2026-10-15 only)

- Rerun Jev on PRMBench with our exact protocol and scorer (matched baseline).
- Add the Jev per-step probability to the bank and run L-SML.
- Optional: label-free fusion of decomposed Jev questions on a public Jev benchmark.
