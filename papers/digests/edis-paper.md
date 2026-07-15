---
slug: edis-paper
title: "EDIS: Diagnosing LLM Reasoning via Entropy Dynamics"
authors: "Chenghua Zhu, Siyan Wu, Xiangkang Zeng, Zishan Xu, Zhaolu Kang, Yifu Guo, Yuquan Lu, Junduan Huang, Guojing Zhou"
arxiv_id: "arXiv:2602.01288v2"
venue: "arXiv:2602.01288v2"
year: 2026
source_pdf: papers/EDIS paper.pdf
extracted_text: papers/extracted/edis-paper.md
last_digested: 2026-07-13
---

## Summary

Shows that the *temporal evolution* of per-token entropy during generation carries more diagnostic signal than static aggregate entropy. Incorrect reasoning shows two characteristic instability patterns — burst spikes (sustained entropy growth over a window of tokens) and peak-valley/rebound spikes (a sharp entropy rise from a historical minimum). EDIS (Entropy Dynamics Instability Score) formalizes both into one trajectory-level scalar and is shown to (a) improve best-of-N selection accuracy and (b) discriminate correct/incorrect responses better than mean entropy at the individual-sequence level.

## Datasets & models used

Four math reasoning benchmarks: GSM8K, MATH, AMC23, AIME24. GSM8K/MATH use 100 randomly sampled problems; AMC23/AIME24 use their full test sets (40 and 30 problems respectively). Three models: Qwen2.5-Math-1.5B (base), Qwen3-4B-Instruct, Qwen2.5-Math-7B. All experiments sample at three temperatures (0.2, 0.6, 1.0), results averaged/pooled across temperatures. §5.3's AUC analysis pools 26,356 valid (non-empty-answer) responses from Qwen2.5-Math-1.5B across all 4 benchmarks x 3 temps.

## Methods it compared itself against

Best-of-N / selection baselines (§5.2, Table 1): Mean (unweighted average accuracy, i.e. no selection), Majority Voting, Sequence Entropy (mean token-level entropy H̄ = 1/T Σ Hₜ, lower = higher confidence), Self-Certainty (SC, KL divergence between the predicted distribution and uniform, aggregated across tokens). §5.3's discrimination analysis compares EDIS directly against mean entropy as a single scalar confidence signal.

## Experiments — methodology & scores

| Setup | Metric | Score | Notes |
|---|---|---|---|
| §5.1 Best-of-N scaling, Qwen2.5-Math-1.5B, k=8, m∈{1..16} | Avg accuracy (pooled 4 benchmarks) | 29.9% (m=1) → 54.5% (m=16) | +24.6pp; EDIS-best accuracy also rises 46.3%→54.1% |
| §5.2 Table 1, m=16, Overall (pooled) | Accuracy | Mean 30.4 / MajVote 46.2 / Entropy 50.9 / SC 51.7 / **EDIS 60.6** | EDIS best on 9/12 dataset×m combos |
| §5.2 Table 1, m=16, GSM8K | Accuracy | Entropy 55.0 / SC 56.0 / **EDIS 72.3** | |
| §5.2 Table 1, m=16, AIME24 | Accuracy | Mean 6.9 / SC 18.9 / **EDIS 22.2** | Hardest cell for all methods |
| §5.3 Correctness discrimination, Qwen2.5-Math-1.5B, pooled 26,356 responses | ROC-AUC | **EDIS 0.804** vs mean entropy 0.673 | 13.1pp gap; EDIS Spearman ρ=-0.52 vs mean-H ρ=-0.30 with correctness |
| §5.3 Top-10% retention selection accuracy | Accuracy | EDIS 91.1% vs entropy 61.0% | 30pp gap |
| Eq. 7 hyperparameters | τ_b (burst), τ_r (rebound) | 1.36, 1.33 | Stable across RL training checkpoints (Step 0 → Step 500) |

EDIS(H) = S(H)·(1 + Var(H)), where S(H) = 0.5·(S_burst + S_rebound); S_burst counts positions where entropy grows past τ_b within a window w, S_rebound counts positions where entropy exceeds τ_r above its running historical minimum. (Confirmed no square root in Eq. 7 — see Notes.)

## Connection to our pipeline

`compute_edis()` in `spectral_utils/feature_utils.py` implements Eq. 7 with the paper's exact τ_b/τ_r; `scripts/score_edis.py` scores it as a standalone confidence signal against our L-SML GOOD_5 on repgrid cells (`results/repgrid/edis_scores.csv` — e.g. GSM8K/Llama-3.1-8B: EDIS AUROC 0.809, ρ(EDIS, L-SML)=0.87, i.e. largely redundant with our signal on that cell). This paper is the closest entropy-dynamics competitor to our spectral/L-SML approach — same underlying object (the token entropy trajectory H(n)) but a hand-crafted spike-counting statistic vs. our spectral/FFT features. A dedicated replication grid (base Qwen2.5-Math-1.5B x GSM8K/MATH/AMC23/AIME24, matching §5.3's exact protocol with an over-collection policy for class balance) is planned to reproduce the 0.804-vs-0.673 head-to-head directly and place L-SML GOOD_5 alongside it on identical traces.

## Notes / open questions

- **Formula bug found+fixed this session**: `compute_edis()` had `S * sqrt(1 + Var(H))` (a square root) — the paper's Eq. 7 and this project's own HISTORY.md Step 35 transcription both give `S * (1 + Var(H))`, no sqrt. Fixed in `spectral_utils/feature_utils.py`; `results/repgrid/edis_scores.csv` regenerated.
- The window size `w` for `S_burst` (Eq. 5, "entropy rises steadily across a window of w tokens") has no explicit numeric default in the main text; our implementation uses `w=1` (consecutive-token diff via `np.diff`), previously validated by a spike-ratio sanity check landing inside the paper's reported 1.7-3.6x range (HISTORY Step 41).
- Two prior local replication attempts (HISTORY Steps 36/41/42, Phase 13) failed: first on a `####`-vs-`\boxed{}` grading bug, then on class starvation (83-85% accuracy on Instruct-model GSM8K left too few negatives to test the AUC gap). The paper's own base-model accuracy is much lower (29.9-36% on GSM8K) — using the base checkpoint (not Instruct) as the paper does, plus explicit over-collection sized from Table 1's own per-dataset baseline accuracy, is the fix.
- AIME24 is a genuine floor for this model (paper's own ~7% baseline, not a pipeline bug) — any replication cell there should be scored with a FLOOR flag, not rejected outright.
