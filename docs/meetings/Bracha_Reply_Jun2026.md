# Draft reply to Bracha (cc Ofir, Amir)

*Attachment: `Spectral_LSML_Report.html`*

---

Hi all,

I want to share some results of an analysis I made in the past week.

Bracha, your second question first — I don't store the full logits. I save the per-token entropy sequence (and, in the newer runs, the top-50 log-probs per token). I used those to run the whole analysis below on last week's stored data with no new GPU time.

For your first question, yes there was a subset of features that performed better and it includes 5 features {epr, sw_var_peak, low_band_power, cusum_max, spectral_entropy}.

As I described in prior emails, I found differences when aligned to the original paper algorithm:
1. **Binary vs Continuous** — in the paper each classifier is binary (±1); in my prior implementation I was feeding the algorithm normalized continuous features.
2. **Fusion algorithm — SML vs L-SML:** SML assumes the classifiers are independent; L-SML first clusters the dependent ones, fuses within each cluster, then across clusters. (I explained this in the previous mail.)
3. **Leakage of the true labels into an "unsupervised" method** — I had been using the labels in two places: to search over all uncorrelated subsets for the best-fused one, and to set each feature's sign/direction.

So in the past week I was using the prior runs results (which I stored in drive), to run some variants of the algorithm, to try to recover the good results reported earlier in research, and understand the main cause for the performance degrading — which turned out to be binarization. When I feed the features as continuous (z-scored) values instead of binarizing them at the median, performance jumps back up by about +5pp on average, and ~+7pp on the math/reasoning datasets.

Here is the full picture — for each feature-set size, binary vs continuous × SML vs L-SML (macro AUROC over the 5 domains):

**5 features (the curated subset)**
|  | Binary | Continuous |
|---|---|---|
| **Flat SML** | 68.3 | 70.0 |
| **L-SML** | 65.2 | 70.1 |

**9 features**
|  | Binary | Continuous |
|---|---|---|
| **Flat SML** | 61.0 | 64.5 |
| **L-SML** | 65.4 | 68.1 |

**16 features (all)**
|  | Binary | Continuous |
|---|---|---|
| **Flat SML** | 60.8 | 63.1 |
| **L-SML** | 65.4 | 69.2 |

*(Continuous is higher than binary in every cell. The 9-feature set is the H(n) features that rarely saturate: {epr, low_band_power, high_band_power, hl_ratio, spectral_centroid, sw_var_peak, rpdi, pe_mean, cusum_max}. Simple-average baseline, continuous: 69.7 / 69.7 / 66.3 for 5 / 9 / 16 features.)*

What this says about the other two differences:
- **Feature selection:** continuous L-SML lands at ~69–70% whether I use 5, 9 or all 16 features — so the method doesn't depend on hand-picking the subset.
- **SML vs L-SML:** on the clean 5 features the two fusion methods are tied (and tied with a simple average); but as features are added, L-SML holds ~68–70% while flat SML collapses (70 → 63). The clustering is what lets the method run on all the features without selecting them.

So the corrected method — continuous, unsupervised (no leakage), paper-aligned L-SML — recovers to ~70% macro / ~78% on the reasoning datasets.

If we go back to the comparison tables I sent in the previous mail (all baselines run by me on the same model + dataset; ours is one forward pass). I'm showing several unsupervised variants so the spread is visible, and my earlier binary number for reference:

**MATH-500 / Qwen2.5-Math-7B**
| Method | AUROC | 95% CI | Compute |
|---|---|---|---|
| **L-SML continuous, 5-feat** | **94.4%** | [90.1, 97.7] | 1-pass |
| L-SML continuous, 9-feat | 94.4% | [90.0, 97.9] | 1-pass |
| L-SML continuous, 16-feat (no feature selection) | 94.2% | [89.7, 97.7] | 1-pass |
| Simple average, 5-feat | 93.6% | [89.4, 97.0] | 1-pass |
| L-SML binary (my earlier number) | 88.2% | [84.0, 92.0] | 1-pass |
| Semantic Entropy NLI K=10 | 87.7% | [79.7, 93.9] | K=10 |
| Self-Consistency K=10 | 87.2% | [72.1, 98.4] | K=10 |

**GSM8K / Llama-3.1-8B**
| Method | AUROC | 95% CI | Compute |
|---|---|---|---|
| Self-Consistency K=10 | 78.5% | [72.0, 84.5] | K=10 |
| Semantic Entropy NLI K=10 | 77.4% | [70.9, 83.5] | K=10 |
| **L-SML continuous, 5-feat** | **75.6%** | [72.2, 79.0] | 1-pass |
| L-SML continuous, 16-feat (no feature selection) | 73.8% | [70.4, 77.3] | 1-pass |
| L-SML continuous, 9-feat | 73.3% | [69.7, 77.2] | 1-pass |
| Simple average, 5-feat | 72.1% | [68.2, 75.7] | 1-pass |
| LapEigvals — unsupervised (spectral attention) | 72.0% | n/a | 1-pass |
| L-SML binary (my earlier number) | 70.7% | [67.2, 74.0] | 1-pass |

**GPQA Diamond / Qwen2.5-7B**
| Method | AUROC | 95% CI | Compute |
|---|---|---|---|
| Semantic Entropy NLI K=10 | 70.6% | [43.6, 93.3] | K=10 |
| Verbalized Confidence K=1 | 67.9% | [49.5, 83.3] | 1-pass |
| L-SML continuous, 9-feat | 57.4% | [48.7, 65.7] | 1-pass |
| L-SML continuous, 16-feat (no feature selection) | 56.4% | [48.2, 64.5] | 1-pass |
| Simple average, 5-feat | 56.0% | [47.3, 64.8] | 1-pass |
| L-SML binary (my earlier number) | 53.6% | [44.8, 62.2] | 1-pass |
| **L-SML continuous, 5-feat** | 52.3% | [43.4, 61.2] | 1-pass |
| Self-Consistency K=10 | 33.6% | [11.0, 58.2] | K=10 |

**RAG — L-CiteEval / Qwen2.5-7B** (95% CIs are wide here — small n; full CIs in the HTML)
| Sub-domain | L-SML cont. 5-feat | L-SML cont. 9-feat | L-SML cont. 16-feat | L-SML binary (prior) | SelfCheckGPT K=5 |
|---|---|---|---|---|---|
| NarrativeQA | **65.0%** | 60.3% | 62.8% | 64.1% | 52.4% |
| Natural Questions | 61.4% | 56.7% | **64.2%** | 57.1% | 57.1% |
| HotpotQA | 62.5% | 52.4% | **69.4%** | 54.9% | 51.4% |
| 2WikiMultiHopQA | 51.1% | 56.5% | 55.5% | 60.5% | 55.3% |

In short: a single forward pass that **beats the K=10 sampling methods on math** (at ~10× less compute), is **competitive on GSM8K**, **beats the standard RAG baseline (SelfCheckGPT) on 3 of 4 sub-tasks**, and **trails on GPQA** — short multiple-choice traces lack the token-level structure the method relies on. (SE = Semantic Entropy, SC = Self-Consistency, both K=10; VC = Verbalized Confidence. The EDIS and stronger-model GPQA comparisons aren't ready yet — I'm fixing a grading issue — I'll send those once clean.)

More details from the experiments (variant tables, feature correlations, and per-cluster AUCs) are in the attached HTML. Let's find a time next week so I can explain more, and maybe plan next steps.

Best,
Omri

---

*Internal notes (do not send):*
- 2×2 tables + spread = macro over the 5 domains, from `method_comparison_table1.csv`. Competitor "ours" rows are model-matched continuous variants; CONT (5-feat) is the headline variant (bold), others shown as spread (not best-per-domain).
- Do NOT reuse the Step-117 numbers (96.7 / 71.3 / 88.1) — supervised/leaked.
- GPQA: CONT is honestly our weakest variant there; 16-feat (56.4) does a little better but still loses to SE — both shown.
- EDIS Phase-13 invalid (7.7% acc grading bug); Phase 14 (DeepSeek-R1-8B GPQA) pending.
