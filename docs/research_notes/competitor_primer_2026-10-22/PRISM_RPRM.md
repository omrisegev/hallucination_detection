# PRISM and R-PRM: how they work, and how they compare with PRM boosting

Read in full on 2026-10-22:
- PRISM: arXiv 2606.09078v1 PDF, 24 pages, the only version.
- R-PRM: the EMNLP 2025 PDF (ACL Anthology 2025.emnlp-main.679) and arXiv 2503.21295v1. The two versions differ, as noted below.
- Also checked: PRMBench Sec 3.2 (arXiv 2501.03124v5), plus both papers' GitHub and Hugging Face pages.

**Summary.** Both papers retrain a 7B model using PRM800K's human step labels. We train nothing.
- PRISM starts from our base PRM and goes from 65.5 to 68.0 PRMScore. Our GAP rule alone gives 68.01 on that same PRM, and our fusion gives 70.40.
- R-PRM is a generative judge that writes 10 critiques per step. It scores 66.8.

---

## 1. PRISM (Agrawal et al., UMD / Capital One)

**What the "hidden bias" is.** The authors call it "overcredit": PRMs score plausible but wrong steps as correct, so the false-positive rate (FPR) is high. Here "positive" means "step judged correct" (Sec 1, 2.1).
- Cause: step labels lean heavily toward "correct". PRM800K has 73.1% correct steps but only 14.2% correct full solutions (Sec 2.1, Fig 2). Pointwise cross-entropy training inherits that skew.
- Evidence: Qwen2.5-Math-PRM-7B flags only 30.66% of erroneous PRMBench steps (TNR) while passing 95.36% of correct ones (Table 1).
- Theorem 2.1 (§9.1): in Best-of-N, a false-positive rate α caps accuracy at p/(p+(1−p)α). False negatives only slow convergence.

**At test time.** Same as the base PRM. One forward pass reads the "correct"-class probability at each step. No generation. The PRMBench threshold is not stated [unverified].

**Training (Sec 3)**
1. **Pairs.** A correct and a wrong step sharing the same prefix, built from PRM800K's positive, negative and neutral labels. About 26k pairs. How neutral labels are used is not stated.
2. **Temporal-lookahead hard negatives.** A later step of the same solution serves as the negative for the current prefix: valid, but too early. This gives about 220k pairs with no new labels.
3. **Loss.** Bradley-Terry: −log σ(r_pos − r_neg) (Eq 6). It depends only on the margin between the two scores (§9.3).
4. **Curriculum.** Difficulty D = r_pos − r_neg. Pairs with D < 0 are dropped; the rest are binned and trained easy to hard (Sec 3.3). In the 132k run the bins are 75k, 23.5k, 23.5k and 10k pairs (Table 5).

What is trained: Qwen2.5-Math-PRM-7B itself is fine-tuned further (§9.3.1). ReasonEval-7B is also tried (Table 8).
- Not stated: frozen parts, learning rate, epochs, LoRA versus full fine-tuning.
- Compute: "four Nvidia RTX A6000 GPUs" (Sec 8).
- Supervision: existing PRM800K human labels. "No new human labels" means no new annotation, not label-free training.

**Results**
- **PRMBench (Table 2).** PRMScore 65.50 → 68.00. FPR 69.34 → 47.13; the abstract's "↓22%" is in percentage points. FNR 4.27 → 12.69.
- **Controls (Table 2).** Pointwise loss on the same 132k pairs scores 65.6.
- **PRMBench categories (Table 7, 132k Curriculum-4).**
  - Simplicity 52.1 → 59.7: Non-Redundancy 49.1 → 53.5, Non-Circular Logic 55.0 → 66.0.
  - Soundness unchanged at 71.0, with Confidence Invariance falling 78.5 → 75.6.
  - Prerequisite Sensitivity 57.7 → 62.4.
- **ProcessBench (Table 6).** Flat: average F1 72.95 baseline, 72.25 to 73.1 for PRISM (my arithmetic, after the column fix listed below).
- **Downstream.** Best-of-N and guided beam search are shown only as plots (Fig 6). GRPO with a 1.5B policy raises GSM8K 31.16 → 42.53 (Table 3).

**Code and checkpoints.** The linked repository github.com/Aakriti05/PRISM is empty (GitHub API, checked 2026-10-22). No checkpoint was found on Hugging Face.

**Similar to us.**
- Same base PRM, benchmark and official PRMScore. Same baseline: 65.5 for them, 65.49 for us.
- One teacher-forced pass at test time.
- Same target: the native operating point flags too few errors.

**Different from us.**
- PRISM retrains the weights on human-labelled pairs. We keep the PRM frozen and fuse 16 telemetry channels with L-SML, without labels.
- We decide per answer (GAP); PRISM uses one global threshold.

**Fair comparison? Partly.**
- **The rule effect.** Our rule alone gives +2.5 on the same PRM, the same size as PRISM's gain from retraining. PRISM never reports its baseline at that baseline's best threshold, so part of its gain may be an operating-point shift.
- **Like-for-like pairs.** Compare PRISM's 68.0 with our rule-only 68.01, and our 70.40 with PRISM plus GAP. The second cannot be computed, because no checkpoint is available.
- **Data overlap.** PRMBench is built from PRM800K's correct solutions in both its train and test splits (PRMBench Sec 3.2). PRISM trains on PRM800K and does not discuss this.
- **Our own caveats.** Our recipe was developed on PRMBench. Neither side reports intervals.

**Limitations the authors state.** Lowering FPR can raise FNR (Sec 6).

**Inconsistencies I noticed**
- **Pair counts.** 26k and 220k in Sec 3.2, but 12k and 132k in Tables 2, 5 and 7.
- **Configurations.** Curriculum bin edges differ across Tables 5, 8, 9 and 10. Table 2's PRISM row (FPR 47.13) matches no appendix row.
- **Dropped pairs.** The curriculum drops D < 0 pairs, exactly the cases where the wrong step outranks the right one. Yet Fig 1 says PRISM reuses false positives as hard negatives.
- **ProcessBench Omni-MATH (Table 6).** The reported F1 of 83.4 is impossible from TPR 54.8 and TNR 66.1, whose harmonic mean is 59.9. Swapping the TNR and F1 columns reproduces Qwen's official 66.3.
- **ProcessBench claim.** The abstract claims better macro F1 on ProcessBench, which Table 6 does not show.

---

## 2. R-PRM (She et al., Nanjing University; EMNLP 2025)

**Problem.** PRMs that output a score directly are hard to train on difficult math, use data inefficiently and give no explanation. Labelled step data is also scarce (Sec 1, 3.1).

**At test time.**
- **Prompt (App G).** For each step, a generative judge receives the question, the previous steps and the "Now Step". It writes an analysis (previous steps, data sources, consistency, calculations) and ends with "Verification: Is the step correct (Yes/No)?".
- **Reward (Sec 3.3).** The mean P(Yes) over K sampled analyses, with a softmax over the Yes and No logits. Default K = 10 (Sec 4.1).
- **Cost.** A solution with n steps needs n×K generations.
- **Decisions.** A step is correct if its reward is at least 0.5 (Sec 5.4).
- **Speed.** 0.52 s per ProcessBench example at K = 4 on one H100 with vLLM (App A; amortized).

**Training**
1. **SFT.**
   - Llama-3.3-70B-Instruct writes 4 analyses per PRM800K step case.
   - Only those whose verdict matches the human label are kept, about 289k.
   - Qwen2.5-Math-7B-Instruct, a policy model rather than a PRM, is fine-tuned on them (Sec 3.1, 4.1).
2. **DPO.**
   - Pairs: a correct-verdict analysis is chosen and a wrong-verdict one rejected, about 269k pairs. The reference model is a frozen copy of the SFT model (Sec 3.2).
   - v1 Sec 5.1 says the pairs come "from existing sampling results", meaning the teacher's samples, so this DPO is off-policy.
   - The released script balances Yes and No verdicts 50:50 (in the code, not the paper).
3. **Hyperparameters.** One epoch, batch 128, learning rate 5e-6 for SFT and 5e-7 for DPO, 20k samples held out (Sec 4.1). Not stated: DPO β, LoRA versus full fine-tuning, GPU-hours.
4. **Self-improvement.** A further round on 30k self-sampled pairs, then a second round, reaches 74.1 on ProcessBench (Sec 5.1, Fig 2). These rounds have no PRMBench numbers.

**Supervision.** PRM800K human labels filter the SFT data and define the DPO preferences, on top of distillation from the 70B teacher.

**Results (ACL version)**
- **ProcessBench (Table 1).** Average F1: DPO 70.4 (GSM8K 80.7, MATH 76.9, OlympiadBench 63.8, Omni-MATH 60.1); SFT 65.2. That is +13.9 over the PRM800K-trained baseline (56.5).
- **PRMBench (Table 2).**
  - DPO 66.8, SFT 64.9. That is +8.5 over that baseline (58.3). It ties GPT-4o (66.8) and is below o1-mini (68.8).
  - DPO by category: Simplicity 55.2 (Non-Redundancy 52.2, Non-Circular Logic 58.2), Soundness 71.2, Sensitivity 76.6 (Prerequisite Sensitivity 61.2).
  - DPO lowers Simplicity relative to SFT: 58.7 → 55.2.
- **Scaling with K.** ProcessBench F1 goes from 61.3 at K = 1 to 71.7 at K = 32 (Fig 5).
- **Search with a Qwen2.5-7B-Instruct policy.** Best-of-8 49.6 and guided search 49.4, against pass@1 41.0 (Tables 3, 4).

**Code and checkpoints.** github.com/NJUNLP/R-PRM has the evaluation and training scripts, without the training configs. Hugging Face has kevinpro/R-PRM-7B-DPO and the dataset kevinpro/R-PRM. No SFT or self-improvement checkpoints were found.

**Similar to us.** Same benchmark, metric and categories. The score is an average of token-probability readouts (Yes/No).

**Different from us.**
- **Model.** A different, generative model, not Qwen2.5-Math-PRM-7B.
- **Training.** Distillation plus filtering by human labels.
- **Compute.** About 10 generations per step, against our one pass per answer.
- **Fusion.** An equal-weight average of samples from one judge, against L-SML weights over 17 different views (the PRM plus 16 channels).
- **Decision.** A global threshold, against per-answer GAP.

**Fair comparison? Context only.** The base model, label access (PRM800K, which overlaps PRMBench's source data) and compute all differ, and the threshold is uncertain (see below). arXiv v1 Table 2 gives our base PRM 65.5. Against that, R-PRM-DPO is +1.3. On the same PRM, our rule is +2.5 and our fusion +4.9. On ProcessBench, the base PRM's 73.5 beats R-PRM-DPO's 70.4 (v1 Table 1).

**Limitations the authors state.** Not tested at 70B because of compute, and MCTS or beam search not explored.

**Inconsistencies I noticed**
- **Dropped baseline.** The ACL version removes v1's Qwen2.5-Math-PRM-7B rows (73.5 and 65.5), so its headline gains are against a weaker baseline trained on the same data.
- **Averaging error in v1.** v1 averages that baseline as 58.5, but its own subset F1s give 56.5. Hence the abstracts say 11.9 in v1 and 13.9 in the ACL version.
- **o1-mini claim.** The text says R-PRM beats o1-mini on prerequisite sensitivity, but Table 2 shows 61.2 against 61.8.
- **SFT size.** 289k in Sec 4.1, 285k in Sec 5.2.
- **Threshold.** The paper says 0.5, but the released PRMBench scorer `new_get_result.py` hard-codes `score >= 0.61` at line 405. Which threshold produced Table 2 is unverified.
- **Gap mostly closes with threshold tuning.** App D says the PRM800K-trained baseline loses 8.2 points at 0.5 compared with its best threshold. That puts it near 66.5 with the threshold chosen on the test set, close to R-PRM's 66.8.

---

**Could not verify**
- **PRISM:** its threshold, hyperparameters, and which configuration produced Tables 2 and 6.
- **R-PRM:** its DPO β, its training mode, which scorer produced Table 2, and the values in Fig 8.
- **Both:** whether Qwen2.5-Math-PRM-7B's own training data includes PRM800K.
