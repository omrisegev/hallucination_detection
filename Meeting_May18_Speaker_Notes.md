# Advisor Meeting -- May 18, 2026
## Speaker Notes (updated for revised deck)

---

## Slide 1 -- Title

**What to say:**
> "Good morning. Today I want to walk you through everything we've done since the last meeting. The thesis has moved significantly: we now have results across five domains -- math reasoning, science MCQ, factual QA (a negative result I want to explain properly), retrieval-augmented generation, and now multi-step agent loops. I'll show plots and competitor comparisons for each domain. At the end I'll describe two new pilots we're running and what remains before the thesis is complete."

**Anticipated question from Ofir:** "Which direction has the most theoretical interest?"
> RAG and agentic -- those are the ones where the signal mechanism is different from what EPR predicts. The meta-analysis shows the feature ranking shifts: in math, EPR dominates; in RAG, rpdi (end-of-trace entropy rise) and spectral_entropy take over. That shift suggests different generative processes are at play.

---

## Slide 2 -- What Are We Detecting? (H(n) traces)

**What to say:**
> "The core idea: during text generation, each token has a probability distribution over the full vocabulary. The entropy of that distribution is H(n) = -sum_v p_v log p_v. We collect this number for every generated token -- that's the entropy trajectory. On this slide you can see 2x3 examples: correct answers on top, incorrect (hallucinating) answers on the bottom. Correct answers show smooth, lower-variance trajectories. Incorrect answers show more instability -- abrupt jumps, variance bursts.
>
> The method is gray-box: we need token probabilities from a single forward pass. We do NOT look at the output text at all. We're reading the model's internal uncertainty signal."

**Key point for Amir:** "We're reading a signal the model always produces, with zero extra computation beyond inference."

---

## Slide 3 -- PSD Shows Spectral Signature

**What to say:**
> "Here's why 'spectral' matters. This is the Average Power Spectral Density of H(n) across many examples from MATH-500. Blue line: correct answers. Orange line: incorrect. You can clearly see that incorrect outputs concentrate more power at HIGH frequencies -- meaning the entropy signal has more short-range bursts and instability. Correct outputs concentrate power at LOW frequencies -- the reasoning is smooth and structured.
>
> EPR, the baseline from the original paper, uses only the mean of H(n) -- that's the DC component of the PSD. We use the entire spectrum: band powers, spectral entropy, dominant frequency, STFT. We're recovering information that EPR discards."

---

## Slide 4 -- Feature Library (16 features)

**What to say:**
> "We have 16 spectral features in four families. The bar chart shows individual AUROC on MATH-500 / Qwen-7B. A few highlights: sw_var_peak -- peak sliding-window variance -- captures localized instability. cusum_max -- cumulative sum change-point statistic -- detects regime shifts in the entropy signal. These two are the most universal: they appear in the top 3 for every single domain we tested.
>
> The full Nadler fusion of the best subset achieves 90.0% AUROC on MATH-500. That's 16 candidate features; Nadler selects the best conditionally-independent subset automatically."

---

## Slide 5 -- Feature Correlation Topology

**What to say:**
> "This is the Spearman correlation matrix across all 7,001 samples from 5 domains. Notice the cluster structure: epr, hurst, and pe_mean are all highly correlated -- they're different ways of measuring the same thing, the global average entropy level. spectral_entropy and stft_spectral_entropy are also correlated -- same signal, different windowing.
>
> The critical observation: sw_var_peak and cusum_max are ORTHOGONAL to the EPR cluster. That's exactly why Nadler fusion gains signal -- it finds a linear combination of orthogonal views that agrees on whether the answer is correct, but disagrees in how they express it. pe_min is completely isolated from every cluster, which explains why it was ranked last (17/17) in every domain and was removed."

**For Ofir:** "The orthogonality structure here is the theoretical justification for the multi-view approach -- you're essentially doing a matrix factorization where independent components each contribute unique signal about the hidden binary variable (correct/incorrect)."

---

## Slide 6 -- Feature Importance Per Domain

**What to say:**
> "These are Random Forest Gini importance scores, one bar chart per domain, from the meta-analysis dataset (7,001 samples). The bottom-right panel summarizes the cross-domain verdict. cusum_max and sw_var_peak are top-3 in ALL five domains -- those are the universally robust features. But the rankings shift significantly: in math and GSM8K, epr is the single best feature. In factual QA and RAG, rpdi (end-of-trace entropy rise) becomes dominant. This shift is substantively interesting: rpdi detects an entropy rise at the END of the generation, which is what you'd expect when a model is about to produce ungrounded text. In math, that pattern doesn't exist -- the uncertainty is mid-reasoning."

---

## Slide 7 -- How It Works: Math Domain (Example)

**What to say:**
> "Let me make the method concrete. On the left: the structure of a MATH-500 example. The model receives a question -- say, solve x^2+3x+2=0 -- and generates a full chain-of-thought reasoning trace, then gives the final answer. H(n) covers EVERY generated token, typically 500 to 2,000 tokens total.
>
> On the right is what H(n) actually looks like for a correct answer. The orange region is the chain-of-thought; the green region is the answer. Notice: during reasoning there are occasional entropy spikes (the model is uncertain at branch points), and then the answer segment is very low-entropy and stable. cusum_max detects the regime shifts mid-reasoning. sw_var_peak captures those local variance spikes.
>
> A hallucinating model would show a different pattern: more frequent and larger spikes, less structure, higher final entropy at the answer."

---

## Slide 8 -- Math Results + Comparison Table

**What to say:**
> "The key numbers: MATH-500 with Qwen-7B at T=1.0 gives 90.0% AUROC with Nadler fusion. On GSM8K with Llama-3.1-8B, we get 76.0%. The closest published competitor using the same approach -- LapEigvals, which also uses spectral features of hidden states -- gets 72.0% on the same model and task. That's +4 pp unsupervised vs unsupervised.
>
> The comparison table shows what makes this relevant: both methods use no labels and require only a single forward pass. The difference is in what we compute: LapEigvals uses hidden-state Laplacian eigenvalues; we use the entropy trajectory spectrum, which requires less access -- just token probabilities, not hidden states.
>
> One caveat: T=1.5 gives 96.6% -- this is a ceiling artifact. At higher temperature the model makes more mistakes, so the class ratio shifts and the AUC inflates. The honest comparison is T=1.0 = 90.0%."

---

## Slide 9 -- GPQA Diamond Results

**What to say:**
> "GPQA Diamond is 198 PhD-level science multiple-choice questions with 4 options. This is the hardest task we tested. With 7B models, the task accuracy is only 30-40% -- barely above random for a 4-option question. When the model only gets one in three questions right, there simply aren't enough correct examples for the detector to discriminate.
>
> Moving to Qwen2.5-72B-AWQ improved task accuracy to 40.4% and AUC to 69.0%, a +3.6 pp gain. The trend is clear: as model quality improves, detection improves. The method is bottlenecked by the underlying model, not by our detector.
>
> We have not yet cleared the 70% gate. The direction is to try even larger or more specialized models. There's no published spectral competitor on GPQA, so we can't directly compare -- though if someone tries the same approach with a 70B+ model, our prediction is they'd see the same trend."

---

## Slide 10 -- Negative Result: Factual QA

**What to say:**
> "This is an important negative result and I want to explain it properly. TriviaQA and WebQuestions are factual recall tasks. The model is asked 'What is the capital of France?' -- the answer is just 'Paris'. Direct-answer traces are 20-50 tokens. That's completely insufficient for spectral analysis; FFT on 20 tokens has no frequency resolution.
>
> We added chain-of-thought prompting to extend the traces to 200-500 tokens. It didn't help: TriviaQA CoT gives 53.6% (near chance), WebQ gives 61.9%. Importantly, the EPR baseline on direct answers -- just the mean entropy -- gives 79.1% and 71.8%. So our extended approach is substantially WORSE than the simple baseline.
>
> The reason: chain-of-thought SMOOTHS the entropy trajectory on factual recall tasks. The model essentially repeats the same high-confidence text in different phrasings, so there's no systematic frequency structure. This is a structural difference from reasoning tasks. Factual recall is retrieval, not generation.
>
> This is not a failure -- it's a boundary condition that sharpens the thesis scope. We now know: spectral features detect generative uncertainty during REASONING, not factual uncertainty during recall. These are different cognitive processes."

**For Bracha:** "This boundary condition is actually useful for LTT -- we can now design the calibration guarantee specifically for reasoning-type tasks."

---

## Slide 11 -- RAG: How It Works

**What to say:**
> "Now let's talk about RAG -- retrieval-augmented generation. The task is: given a question, the model retrieves relevant documents and generates a response WITH citation markers [1], [2], etc. Our question is: for each sentence in the model's output, is it actually grounded in the retrieved documents, or did the model just make it up?
>
> The key innovation for RAG: we don't analyze the full trace. We SLICE H(n) at the citation boundaries. Each statement between citation markers gets its own entropy subsequence, its own spectral feature vector, and its own Nadler fusion score. Each statement is labeled GROUNDED if its cited passage appears in the gold supporting facts, UNGROUNDED otherwise.
>
> Look at the example: the two grounded statements have smooth, lower-entropy slices. The ungrounded statement -- 'Therefore, they were NOT founded in the same country' -- is a fabricated conclusion that appears in neither retrieved document. Its spectral features are different: higher rpdi (entropy is rising toward the end of the statement), higher spectral_entropy overall.
>
> This is why rpdi dominates in RAG but not in math: in RAG, the BOUNDARY between grounded and ungrounded text is where the signal is. In math, it's the STRUCTURE of the entire reasoning chain."

---

## Slide 12 -- RAG Results

**What to say:**
> "Results across 16 cells -- four models cross four datasets. The heatmap shows AUC per cell. Key numbers: best cell is llama8b / HotpotQA at 87.7%, which is the new overall best result in the thesis. Qwen-7B / 2WikiMultiHopQA is 80.5%. Median across all 16 cells is approximately 72.8%; 12 of 16 cells pass the 70% gate.
>
> The critical competitor comparison: LOS-Net (arXiv:2503.14043) is a supervised hallucination detector for RAG, trained with labeled data. On HotpotQA / Mistral-7B it achieves 72.92%. Our Qwen-7B / HotpotQA without any labels at test time gives 79.5% -- that's +6.6 pp. Our best cell (llama8b) gives 87.7% -- that's +14.8 pp. We beat a supervised method without seeing any labels.
>
> The sanity check that matters: trace_length alone gives 50.8% -- chance. So the signal is genuinely spectral, not just 'longer statements are more likely to be correct.'"

---

## Slide 13 -- RAG Sanity Check

**What to say:**
> "This slide shows the controlled comparison. Spectral features without length give the main AUC. Trace length alone gives chance (50.8%). Adding trace length to the spectral features doesn't help meaningfully. This rules out the confound that we're just detecting verbosity -- the signal comes entirely from the spectral SHAPE of H(n), not from how many tokens the statement has."

---

## Slide 14 -- RAG Score Distributions

**What to say:**
> "These are the histograms of Nadler-fused scores, split by label (grounded vs ungrounded), for each model-dataset cell. A higher Nadler score means 'more likely correct / grounded.' You can see clear separation in the cells with high AUC -- the distributions are well-separated. Cells with lower AUC show more overlap. This lets us visually confirm that the method is doing something principled, not just exploiting a distributional artifact."

---

## Slide 15 -- How It Works: Agentic Domain (Example)

**What to say:**
> "Now the newest direction: multi-step agents. Here we're using the ReAct framework -- Reasoning + Acting. The agent runs a loop: Thought, Action, Observation. At each step, the Thought is a free-text reasoning segment -- 'I need to find where The Chainsmokers are from...' -- followed by an Action like a search query or a finish action.
>
> We extract H(n) from the Thought tokens only, for each step. Each step gets its own spectral feature vector and Nadler fusion score. Then we aggregate across steps three ways: Phi_min (the weakest link -- the lowest score across all steps), Phi_avg (average), and Phi_last (the final reasoning step).
>
> On the right you can see what this looks like: step 1 scores 0.74, step 2 scores 0.61 (uncertain), step 3 scores 0.79. Phi_min = 0.61, driven by step 2. The intuition: if ANY reasoning step is unreliable, the agent is likely to reach a wrong conclusion. The chain is only as strong as its weakest link."

---

## Slide 16 -- Agentic Results + Comparison

**What to say:**
> "The competitor here is AUQ from Zhang et al. 2026. AUQ asks the model to verbalize its confidence -- 'How confident are you?' -- and uses that as the uncertainty estimate. Their best result, using Phi_min aggregation on ALFWorld, is 79.1%. Note: ALFWorld is a different environment from our multi-hop QA datasets, so this is a general-capability comparison, not an identical-setup comparison.
>
> Mid-run signal from DeepSeek-R1-7B on 2WikiMultiHopQA: our Phi_min gives 85.0% -- that's +5.9 pp over the AUQ baseline. This is not yet official; the run still needs Mistral-24B and Qwen-72B cells.
>
> The comparison table highlights the key structural differences: AUQ is white-box -- it needs the model to verbalize confidence in its OUTPUT, which means RLHF-aligned models may produce systematically overconfident text. Our method is gray-box -- token probabilities from the Thought segment, which are harder to 'lie' through alignment. Also, we require no extra prompting -- just the standard ReAct Thought output."

**For Ofir:** "The white-box vs gray-box distinction is actually quite deep. Verbalized confidence goes through the RLHF distribution shift; token-level entropy during generation does not. That's why gray-box can outperform white-box even with fewer data."

---

## Slide 17 -- What's Next

**What to say:**
> "Three tracks going forward. First, complete Phase 11a: run the remaining two models, Mistral-24B and Qwen-72B, and do the full analysis across all 8 cells. This should give us official AUC numbers and let us beat the AUQ baseline officially.
>
> Second, two domain-expansion pilots -- this is new. Pilot A is HumanEval, code generation: N=20 problems, Qwen-7B, 3 attempts per problem, labeled by whether the unit tests pass. The key question: does spectral entropy during code generation predict whether the code will execute correctly? The competitor is DSDE, an execution-based disagreement method with AUROC 0.82-0.84. Pilot B is ALFWorld, embodied navigation: N=5 tasks, the model navigates a simulated household environment. Label is task success. The competitor is AUQ on the same benchmark -- 0.791. These are pilots -- we need the go/no-go gates to pass before committing to full runs.
>
> For thesis closure: Bracha, the LTT calibration is about 50 lines of code. The data already exists from the temperature and behavioral ensemble experiments. It converts our AUROC number into a formal guarantee: hallucination recall >= 90% with 95% confidence. That's what I think closes the thesis formally.
>
> Ofir, the manifold question: do entropy trajectories from the SAME model lie on a low-dimensional manifold? Does hallucination correspond to escape from that manifold? This connects to LOCA and IMM -- intrinsic dimensionality estimation and regime-switching state estimation. That's the theoretical 'why' that would elevate the thesis from an empirical contribution to a principled one."

---

## Cheat Sheet -- All Headline Numbers

| Domain | Dataset | Model | Our AUROC | Competitor | Their AUC | Delta | Labels | Access | Passes |
|--------|---------|-------|-----------|-----------|-----------|-------|--------|--------|--------|
| Math | MATH-500 | Qwen-7B T=1.0 | **90.0%** | -- | -- | -- | None | Gray-box | 1 |
| Math | GSM8K | Llama-3.1-8B | **76.0%** | LapEigvals | 72.0% | +4.0 pp | None | Gray-box | 1 |
| Science MCQ | GPQA Diamond | Qwen-72B-AWQ | **69.0%** | -- | -- | -- | None | Gray-box | 1 |
| Factual QA | TriviaQA CoT | Falcon-3-10B | 53.6% | EPR direct | 79.1% | -25.5 pp | None | Gray-box | 1 |
| Factual QA | WebQ CoT | Falcon-3-10B | 61.9% | EPR direct | 71.8% | -9.9 pp | None | Gray-box | 1 |
| RAG | HotpotQA | Llama-8B | **87.7%** | LOS-Net | 72.92% | +14.8 pp | LOS-Net needs labels | Gray-box | 1 |
| RAG | 2Wiki | Qwen-7B | **80.5%** | -- | -- | -- | None | Gray-box | 1 |
| Agentic (mid-run) | 2Wiki | DeepSeek-7B | **85.0%*** | AUQ | 79.1% | +5.9 pp | None | Gray-box vs White-box | 1/step |

---

## Competitor Summary

| Competitor | Paper | Their Setting | Their Supervision | LLM Access |
|-----------|-------|--------------|-------------------|-----------|
| LapEigvals | arXiv:2502.17598 | GSM8K / Llama-8B | None | Gray-box (hidden states) |
| LOS-Net | arXiv:2503.14043 | HotpotQA RAG / Mistral-7B | **Required labels** | Gray-box |
| AUQ | Zhang et al. 2026 | ALFWorld agentic | None | **White-box (verbalized)** |
| DSDE | 2026 | HumanEval code | None | Black-box (exec. disagreement) |

---

## Advisor-Specific Talking Points

**Ofir (spectral methods, manifold):**
- The cross-domain feature orthogonality (slide 5) is the theoretical backbone: sw_var_peak and cusum_max are orthogonal to EPR across 7,001 samples. This isn't coincidence -- they measure different spectral modes of H(n).
- The feature importance shift between domains (rpdi dominant in RAG, epr dominant in math) suggests different generative processes produce different spectral fingerprints. A manifold model might explain this as different parts of the manifold being active for different tasks.
- Next theoretical question: does the entropy trajectory live on a low-D manifold, and is hallucination a departure from it?

**Bracha (conformal, calibration):**
- LTT calibration: the temperature + behavioral ensemble data from Phase 6 (TriviaQA/WebQ) gives us multiple views with known AUROCs. This is exactly the input LTT needs. ~50 lines.
- The deployment guarantee we want: "Given this query, the detector flags it as hallucination with hallucination_recall >= 90% at 95% confidence." That's a risk-controlling prediction set.
- The negative result on factual QA actually helps calibration: we can design the guarantee to only apply to reasoning-type inputs (classifiable by trace length > 200 tokens).

**Amir:**
- The clean negative result on factual QA is important: it scopes the thesis honestly and prevents overselling. We don't claim general hallucination detection -- we claim detection in REASONING-TYPE tasks.
- The 87.7% RAG result beating supervised LOS-Net is the strongest empirical contribution so far. Unsupervised vs supervised, same access level, +14.8 pp.
- Phase 11b pilots are deliberately small (N=5-20) to gate the decision before full runs.
