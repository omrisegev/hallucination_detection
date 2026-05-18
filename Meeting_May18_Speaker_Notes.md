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
> "The core idea: during text generation, each token has a probability distribution over the full vocabulary. The entropy of that distribution is H(n) = -sum_v p_v log p_v. We collect this number for every generated token -- that's the entropy trajectory. This slide shows six real traces from MATH-500 / Qwen-7B at T=1.0: three correct answers on the left and three hallucinated (incorrect) answers on the right.
>
> The difference is visible: incorrect answers show HIGHER overall entropy AND more variability -- abrupt jumps, variance bursts. Correct answers are smoother and lower. EPR -- the existing baseline -- just computes the mean of H(n), which is the DC component. It discards the entire SHAPE of the trajectory. Spectral features recover that shape.
>
> The method is gray-box: we need only token log-probabilities from a single forward pass. No attention maps, no hidden states, no extra sampling."

**Key point for Amir:** "We're reading a signal the model always produces, with zero extra computation beyond inference."

---

## Slide 3 -- PSD Shows Spectral Signature

**What to say:**
> "Here's why 'spectral' matters. This slide shows the average Power Spectral Density of H(n) across FOUR model/dataset combinations: MATH-500 with 1.5B and 7B models, and GPQA Diamond with Mistral-7B and Qwen-7B. Solid lines are T=1.0; dashed are T=1.5.
>
> In every panel the same pattern holds: incorrect outputs concentrate more power at HIGH frequencies, correct outputs at LOW frequencies.
>
> The x-axis is normalised frequency in cycles per token -- 0.5 means the entropy alternates every 2 tokens, 0.1 means one full oscillation per 10 tokens. Low band (0 to 0.1) captures slow structural patterns -- reasoning stages. High band (0.4 to 0.5) captures rapid instability bursts -- the hallucination signature. EPR = mean of H(n) = DC component at zero. We use the FULL spectrum above DC."

**Key question to anticipate:** "Is this just detecting model size?" No -- we see the same pattern within the 7B panel at both temperatures, and across MATH-500 (long traces) and GPQA (shorter traces). The frequency pattern is a genuine structural signal about generation quality.

---

## Slide 4 -- Feature Library (16 features)

**What to say:**
> "We have 16 spectral features in four families. The bar chart shows individual AUROC on MATH-500 / Qwen-7B at T=1.0 -- this is the Phase 5 set of 12 features. Four more were added in Phase 8 (cusum_max, pe_mean, hurst, cusum_shift_idx) for 16 total; the meta-analysis on slide 5 uses all 16.
>
> Quick point on the dataset: we ran this on Qwen2.5-Math-7B-Instruct, not the 1.5B we started with. The reason: the 1.5B model at T=1.0 was too accurate -- too few wrong answers -- so the class balance was too skewed for stable detection. The 7B achieves 68.7% accuracy, which gives roughly 31% wrong answers -- much more balanced. This came directly from the previous advisor meeting on class imbalance.
>
> The bottom-right panel shows the four key formulas. cusum_max detects regime shifts -- cumulative sum of mean-centred entropy. sw_var_peak is peak variance in a sliding window of 16 tokens. rpdi is the ratio of end-entropy to full-trace entropy -- hallucinations often have rising entropy at the end. spectral_entropy is the Shannon entropy of the PSD itself -- how spread is the energy across frequencies.
>
> Nadler fusion on the best rho-filtered subset gives 90.0% AUROC. That 4-feature subset is: trace_length + spectral_centroid + rpdi + sw_var_peak."

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
> On the right: BOTH a correct trace (green) and a hallucinating trace (red) on the same axes. These are representative simulated patterns based on the MATH-500 / Qwen-7B data. The correct trace is smoother, lower overall, and drops sharply at the answer segment (the part after the dashed vertical line). The hallucinating trace shows: higher baseline entropy, larger spikes mid-reasoning (cusum_max detects these regime shifts), and crucially -- the answer segment stays HIGH entropy rather than dropping. That's the rpdi signal: ratio of end-entropy to full-trace entropy.
>
> The two features annotated here -- cusum_max and rpdi -- are the ones that most clearly separate the two traces."

---

## Slide 8 -- Math Results + Comparison Table

**What to say:**
> "The key numbers: MATH-500 with Qwen-7B at T=1.0 gives 90.0% AUROC with Nadler fusion. On GSM8K with Llama-3.1-8B, we get 76.0%.
>
> The comparison table now has the corrected Access column. LapEigvals -- the closest competitor -- is labeled WHITE-BOX. I want to be precise about this: LapEigvals requires full attention maps from ALL layers and ALL heads, PCA-projected, fed to a supervised logistic regression probe. That's a very different access level from ours. Our method uses only token log-probabilities -- the minimum gray-box access. So +4 pp over LapEigvals ALSO comes with a strictly easier deployment profile.
>
> One caveat: T=1.5 gives 96.6% -- this is a ceiling artifact. At higher temperature the model makes more mistakes, so the class ratio shifts and the AUC inflates. The honest comparison is T=1.0 = 90.0%."

**If asked about LapEigvals being white-box:** "Yes -- I double-checked this in the paper. They extract hidden states from all transformer layers, compute pairwise graph Laplacians, then use PCA on the eigenvalue sequences as input to a logistic regression classifier. That requires full architectural access and a labeled training set. It is genuinely white-box and supervised."

---

## Slide 9 -- GPQA Diamond Results

**What to say:**
> "GPQA Diamond is 198 PhD-level science multiple-choice questions with 4 options. The top panel shows real entropy trajectories from Phase 8 -- correct answers on one side, incorrect on the other. These traces are shorter than math (100-400 tokens) and less structured. You can still see a difference but it's subtler than in math.
>
> With 7B models, task accuracy is only 30-40% -- barely above random. When the model only gets one in three questions right, there aren't enough correct examples for stable detection. Moving to Qwen2.5-72B-AWQ improved task accuracy to 40.4% and AUC to 69.0%, a +3.6 pp gain.
>
> We have not yet cleared the 70% gate. The direction is even stronger models. There is no published spectral competitor on GPQA, so we can't do a direct head-to-head -- but the trend is consistent with what we see everywhere else: better model quality means a cleaner spectral signal."

---

## Slide 10 -- Negative Result: Factual QA

**What to say:**
> "This is an important negative result. TriviaQA and WebQuestions are factual recall tasks: the model answers 'What is the capital of France?' with just 'Paris'. Direct-answer traces are 20-50 tokens -- completely insufficient for spectral analysis.
>
> We added chain-of-thought to extend traces to 200-500 tokens. The plot shows TriviaQA CoT vs direct. It didn't help: TriviaQA CoT gives 53.6%, near chance. WebQ CoT gives 61.9% (not plotted -- same dataset, not shown in the bar chart). Importantly, EPR on direct answers gives 79.1% and 71.8%, substantially better.
>
> One note: WebQ direct-answer failed entirely -- zero correct samples in our run, so AUC is undefined for that mode. The 61.9% for WebQ is CoT only.
>
> Why does CoT hurt? Chain-of-thought on factual recall SMOOTHS the entropy trajectory -- the model repeats similar high-confidence text in different phrasings. No systematic frequency structure appears. Factual recall is retrieval, not generative reasoning. These processes leave different internal signatures.
>
> This is a boundary condition, not a failure. Spectral features detect GENERATIVE UNCERTAINTY during REASONING, not factual recall."

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
> "Results across 16 cells -- four models cross four datasets. The heatmap shows AUC per cell. Key numbers: best cell is llama8b / HotpotQA at 87.7%, the new overall best in the thesis. Qwen-7B / 2WikiMultiHopQA is 80.5%. Median across all 16 cells is approximately 72.8%; 12 of 16 cells pass the 70% gate.
>
> On the competitor comparison: I want to be precise. LOS-Net is a supervised RAG hallucination detector -- it achieves 72.92% on HotpotQA. BUT: LOS-Net was trained on standard HotpotQA -- raw QA, no citation markers. Our task is different: we use L-CiteEval, where the model must generate responses WITH citation markers [n], and we test whether each cited statement is grounded in the retrieved passage. This is a novel task setting -- there is no published unsupervised method that solves exactly this.
>
> So the comparison is: closest available benchmark (LOS-Net, different task, supervised) gives 72.92%. We achieve 87.7% on our task (citation grounding on L-CiteEval), unsupervised. The +14.8 pp gap is a lower bound on the contribution -- the tasks differ. The honest framing is: citation grounding detection is a novel unsupervised task; we establish a strong baseline.
>
> Sanity check: trace_length alone gives 50.8% -- chance. The signal is spectral shape, not verbosity."

**If asked about the LOS-Net comparison:** "We acknowledge in the slide that it's a different task. We include LOS-Net as the closest available published benchmark, not as a direct comparison. The more important claim is that we're establishing the first unsupervised result on the L-CiteEval citation grounding task."

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
> "Now the newest direction: multi-step agents. Let me first explain the task clearly because it's different from RAG. In Phase 11a, the model receives a question AND a dictionary of Wikipedia passages (the retrieval corpus). It runs a ReAct loop: Thought, Action, Observation, up to 3 steps. Actions can be 'search(query)' which returns the matching passage, or 'finish(answer)' which terminates the loop. Label is trajectory_correct = does the final answer match the gold answer string.
>
> Key difference from RAG Phase 10: in RAG, we do ONE forward pass and the model generates a full cited response. We then slice H(n) at the citation markers. In agentic, we run 3 SEPARATE forward passes; each Thought step (50-150 tokens) gets its own H(n) subsequence and its own spectral feature vector.
>
> Phi_min = min(score_step1, score_step2, score_step3) -- the weakest link.
> Phi_avg = mean across steps.
> Phi_last = the score of the final Thought only.
>
> On the right: step 1 scores 0.74, step 2 scores 0.61 (uncertain), step 3 scores 0.79. Phi_min = 0.61, driven by step 2. The intuition: if ANY reasoning step is unreliable, the agent will likely reach a wrong conclusion."

**Key distinction to explain if asked:** "In RAG, the full H(n) is one continuous sequence that we slice by citation markers. In agentic, we have K separate sequences (one per Thought) and we aggregate their individual Nadler scores with Phi. This is a fundamentally different aggregation strategy -- step-level detection, not statement-level slicing."

---

## Slide 16 -- Agentic Results + Comparison

**What to say:**
> "The Phi_min / Phi_avg / Phi_last box at the top defines the aggregations. Phi_min = minimum score across all 3 steps -- the weakest link. Phi_avg = mean. Phi_last = final step only. We test all three; Phi_min tends to perform best.
>
> The competitor here is AUQ from Zhang et al. 2026. AUQ asks the model to verbalize its confidence -- 'How confident are you on a scale of 1 to 5?' -- and uses that text output as the uncertainty score. Their best result with Phi_min aggregation on ALFWorld is 79.1%.
>
> Key structural difference: AUQ is verbalized confidence, which is WHITE-BOX in the sense that it goes through the model's RLHF-trained output distribution. RLHF trains models to produce helpful, confident-sounding text. So RLHF-aligned models are systematically biased to claim high confidence even when they're uncertain. Our method reads the token entropy during generation -- that signal happens BEFORE the output distribution is shaped by alignment. It's harder to 'lie' through.
>
> Mid-run signal from DeepSeek-R1-7B on 2WikiMultiHopQA: our Phi_min gives 85.0% -- +5.9 pp. Not yet official; Mistral-24B and Qwen-72B still pending."

**For Ofir:** "The white-box vs gray-box distinction here is genuinely substantive. Verbalized confidence goes through the RLHF distribution shift. Token entropy during generation does not. That's a core theoretical reason why gray-box can outperform white-box."

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
| LapEigvals | arXiv:2502.17598 | GSM8K / Llama-8B | None (unsupervised) | **White-box (attn. maps + logistic probe)** |
| LOS-Net | arXiv:2503.14043 | Std HotpotQA / Mistral-7B (no citations) | **Required labels** | Gray-box |
| AUQ | Zhang et al. 2026 | ALFWorld agentic | None | **White-box (verbalized confidence)** |
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
