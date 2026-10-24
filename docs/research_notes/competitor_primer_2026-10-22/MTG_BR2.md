# Primer: Mind the Gap (ICML 2026) and BR2-PRM / PRMBench-V (ACL 2026)

Read in full on 2026-10-22 from the official PDFs. There was no bot challenge. Sources: the PMLR PDF (25 pages) and the ACL Anthology PDF (31 pages, pp. 44672-44702). I checked the code with the GitHub API. "Unverified" marks anything the paper does not state.

---

## 1. Mind the Gap: Catching Hallucinations via Evidence Drop (Chen, Chen, Yue, Li)

**Key idea.** A wrong chain is not uncertain on average. It shows a sudden local fall in confidence.
- **Evidence** is the negative entropy of the next-token distribution, renormalized over its top K=20 tokens (Eq. 9-10).
- **Drop** is a negative first difference of the smoothed evidence curve (Eq. 11).
- **Reasoning manifold** is a framing, not something they estimate: latent Markov "evidence states" (Eq. 1), and training data that cover only adjacent transitions (Assumptions 1-2). Theorem 3.1 argues that an off-manifold jump makes the entropy-regularized optimum uniform (Eq. 5-8).

**At test time.** One model, one pass, no PRM.
- On GSM8K and MATH, Qwen3-4B/8B generate greedily and their own top-20 probabilities are used. The output is one risk score per answer, which decides accept or reject.
- On ProcessBench, the same Qwen3 model teacher-forces the given chain in one forward pass (Sec. 4). Here it is a separate reader, not the generator. The output is the first flagged step.

**Algorithm.**
1. For each token: Ê_t = Σ P̃ log P̃ over the renormalized top-20 (Eq. 10).
2. Smooth with an EMA, span 5 (App. E.1).
3. Compute Δ_t = Ẽ_t − Ẽ_{t−1} over tokens, crossing step boundaries (Eq. 11).
4. Answer risk φ = −(1/M) × the sum of the M=5 most negative Δ (Eq. 12).
5. Localization: the first step whose drop exceeds a "step-wise threshold" (Sec. 4, SLA). The paper does not give the token-to-step mapping, the threshold value, how error-free chains are handled, or the Qwen3 template. All of these are **unverified**.
6. Threshold: split the test data 50/50. On the calibration half, set τ̂ to a quantile of the wrong-answer risk scores, with a binomial finite-sample adjustment (App. C, Eq. 42-45). **Labels are used here**, and per the Fig. 3 caption also for the step threshold (α=0.1). Nothing else is fitted, and there is no probabilistic model.

**Results.**
- MATH, Qwen3-8B, α=0.05: selective accuracy (Eq. 13) 88.26 with Shannon Drop vs 70.24 with Shannon Avg (Table 1). AURC×1000 falls from 288.8 to 190.0 (Table 2).
- ProcessBench Step-level Localization Accuracy (SLA), Qwen3-8B, Drop vs Avg (Table 3): GSM8K 46.11 vs 27.66, MATH 32.90 vs 24.62, OlympiadBench 41.52 vs 26.30, Omni-MATH 37.04 vs 23.40.
- **SLA counts only an exact match of the annotated first error.** It is not ProcessBench's official F1, which also scores correct chains. PRMBench is not used.

**Code.** github.com/QJ0114/evidence-drop, last pushed 2026-05-05. It covers GSM8K/MATH only. There is no step-level or ProcessBench code, and no calibration script (τ is passed in). Its drop signal (`utils/metrics.py`) is the top-1 log-prob passed through a cumulative running mean and then the EMA; it is not top-20 entropy. Differencing a running mean scales token t by 1/t, so early tokens dominate.

**Similar to us.** Label-free probability telemetry from one teacher-forced pass over a fixed chain. Top-K entropy and change-type features are already in our channel set.

**Different from us.** No PRM, one channel. It aggregates over time (the worst M drops), not across channels. Its thresholds use labels. It predicts only the first error. Nothing is standardized per answer.

**What a fair comparison needs.**
- (a) **As a channel.** Compute Ê_t from the PRM backbone and run the EMA over the whole answer. Per step, take the mean of the 5 or fewer most negative Δ inside the step. Add it as a 17th L-SML channel and check its dependence on our entropy channels.
- (b) **As a telemetry-only PRMBench baseline.** Compute the same step score from the PRM backbone, the standalone base model and Qwen3-8B. Flag steps with GAP, and separately with their first-drop rule under a declared label-calibrated threshold. Report PRMScore and within-answer AUC on identical IDs. Fix the Qwen3 empty-think template explicitly.

**Limitations the authors state (Sec. 6).** Passive detection only; no intervention.

**Inconsistencies (verified).**
1. The theorem defines evidence as log top-K mass (Eq. 36, 39). The method uses entropy (Eq. 10), for which a uniform distribution gives −log K, not log(K/|V|) (Eq. 7).
2. The text says "(1−α)-quantile". But Eq. 43, combined with "accept if φ≤τ" (Eq. 45), requires the α-quantile.
3. Table 5(a) is identical to 5(b).
4. Table 1 gives MATH "Pretrained" accuracy as 59.24 for both models, while Table 6 gives about 66 for 8B and about 58 for 4B.
5. Table 3 repeats cells (for example, 4B LN-S Drop equals Shannon Avg at 27.94 and 24.17).
6. Two claims are contradicted by the authors' own tables:
   - "Drop beats Avg for all measures" (Sec. 4.3) fails for LN-S on three datasets and LogTokU on one (Table 3).
   - "Consistently lower AURC" (Sec. 4.1) fails for 8B GSM8K Shannon, 41.9 vs 48.3 (Table 2).

---

## 2. Benchmarking Fine-Grained Error Detection in Multimodal Reasoning: PRMBench-V and BR2-PRM (Chan, Zhu et al.)

**Problem.** There is no benchmark for multimodal PRMs. PRMBench-V builds one:
- 907 seed questions from MMK12, MathVista, MathVision and MathVerse (Table 3).
- GPT-4o writes the chains. A chain is kept only if three models (Gemini-2.5-flash, GPT-4o and Qwen-2.5-VL-72B) all reach the answer from the truncated chain (Sec. 3.1.2).
- GPT-4o then inserts one error of each of nine types, and humans review with a checklist (Sec. 3.1.3, App. A.2). This gives 907×9 = 8,163 cases.
- The nine types: Non-Redundancy, Non-Circular Logic, Empirical Soundness, Step Consistency, Domain Consistency, Confidence Invariance, Prerequisite Sensitivity and Deception Resistance (from PRMBench), plus a new Visual Perception type.
- Whether error-free chains are included is **unverified**.

**BR2-PRM is not a trained PRM.** It is an aggregation rule. A judge gives nine per-type scores, and BR2 calibrates each score by the judge's measured reliability for that type.

**At test time.**
- A policy model (Gemma-3-12B-it or LLaVA-1.5-7B) samples N∈{2,4,8,16} answers at T=0.7 on MMK12 (Sec. 4.4).
- One MLLM judge generation per candidate (Fig. 17 prompt) returns integer scores per dimension for the whole trace. No token probabilities are used.

**Algorithm (Sec. 4.6, App. D, Alg. 1-2).**
- **Model.** For each type k there is a latent z_k ~ Bernoulli(π_k) (Eq. 2), and the score is s_k | z ~ Categorical(θ_{k,z}) (Eq. 3-5), with Dirichlet priors (Eq. 6-7).
- **Fit, using labels.** On PRMBench-V, where z is known, add counts to the prior: α′_{k,s,j} = α_{k,s,j} + count(s, z=j) (Eq. 8, Alg. 1). The prevalence π̂_k is the empirical rate or a Beta-posterior estimate.
  - There is no EM, so this is supervised naive Bayes despite the "Dawid-Skene" framing.
  - The prior values and how the calibration scores were produced are **unverified**.
- **Infer.** P(s | z=j) = α′/Σα′ (Eq. 10). P(z_k=0 | s_k) follows from Bayes' rule (Eq. 1/11).
- **Aggregate.** S_Bayes = Π_k P(z_k=0 | s_k), assuming the types are independent (Eq. 12). Select the candidate with the largest S_Bayes.
- An informal O(1/√N_calib) selection bound is given (Thm. D.4).

**Results.**
- **Metrics:** strict accuracy (all errors found, no false positives), plus macro-averaged instance-level precision and recall (Sec. 4.2). Not PRMScore.
- **PRMBench-V accuracy (Table 1):** best is Gemini-1.5-pro at 0.34. GPT-4o 0.27, URSA-RM-8B 0.16, MM-PRM 0.14.
- **Scalar PRMs get the error count from the labels.** They flag the k lowest steps with k = the ground-truth error count (App. C). Hence precision equals recall in Table 5.
- **Benchmark score vs Best-of-16 on MMK12:** r=0.863 and r=0.860, over 7 judges (Fig. 5).
- **BR2 vs averaging:** up to +4.8 pp (Fig. 6). The judge-averaged gains are 0.73-1.73 (Gemma policy) and 1.00-3.00 (LLaVA policy).
- No absolute accuracies, seeds or intervals are reported.

**Code.** None linked in the paper or on the ACL page; a CC BY-NC 4.0 license is stated (App. A.4). Searches of GitHub and Hugging Face on 2026-10-22 found nothing, so availability is **unverified**.

**Similar to us.** Dawid-Skene-style conditional independence, and fusion of noisy signals from frozen models over the PRMBench taxonomy.

**Different from us.**
- BR2 uses labels; L-SML does not.
- Each BR2 type has its own latent label. So BR2 never weights views of one shared label: it multiplies per-type calibrated probabilities. L-SML assumes all channels measure the same step label and estimates their weights from covariance.
- BR2 selects among whole traces; we localize errors within a trace.
- BR2's inputs are generated ordinal scores; ours are continuous telemetry.
- PRMBench-V is multimodal, so our text PRM cannot run on it.

**What a fair comparison with L-SML needs.** Use identical PRMBench steps and source-disjoint folds, and bin each channel (for example, per-answer quintiles). Then compare three arms, all flagged by GAP and scored with PRMScore:
- (a) Supervised naive Bayes with one shared step label, as a reference like our LR.
- (b) Dawid-Skene EM on the same bins, the label-free analogue.
- (c) L-SML.

Copying BR2's posterior product literally would count the prior once per channel and double-count our correlated channel families.

**Limitations the authors state (Sec. 7).** Moderate scale; no training or RL integration; scientific domains only.

**Inconsistencies (verified).**
- The text claims 1.2-1.8% over the weighted rule; the gaps between the judge-averaged labels in Fig. 6 are 0.27-1.67 pp.
- The text says BR2 is "consistently" highest, but some per-judge bars are negative (read off Fig. 6).
- The score scale is 1-5 in App. D but 0-5 in the Fig. 17 prompt, which also has ten dimensions.
- Eq. 11 drops the prevalence terms that Eq. 1 and Alg. 2 include.
- The text gives 0.17/0.13 for the scalar PRMs; Table 1 gives 0.16/0.14.
- MMK12 is both a calibration source and the Best-of-N test set, and the paper does not discuss overlap.

---

Sources: [PMLR page](https://proceedings.mlr.press/v306/chen26n.html), [PMLR PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/chen26n/chen26n.pdf), [Mind the Gap code](https://github.com/QJ0114/evidence-drop), [ACL Anthology page](https://aclanthology.org/2026.acl-long.2068/), [ACL PDF](https://aclanthology.org/2026.acl-long.2068.pdf).
