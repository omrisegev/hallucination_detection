# New per-step features from the PRM model itself (2026-10-26)

**Question (Omri).** Beyond the 16 telemetry channels (N16), which per-step features can we take from the PRM model
itself and feed to the label-free fusion (CHAIN_N16 / L-SML) as new views?

**Status.** Research note plus a CPU feasibility check. No labels were read. No GPU job, no new inference.
Development data only: Socratic-PRMBench, Qwen2.5-Math-PRM-7B same-pass cache, 2,995 answers, 26,052 non-empty steps.
Code and outputs: `results/prm_new_features_v1/`.

## Bottom line

1. **Everything we save is already used for one kind of signal.**
   - All 56 bank channels (N16 is a subset) are step readouts of how uncertain the backbone is about the written
     tokens: entropy, surprisal, margin, energy, turnover, and their changes over time.
   - Their noise is shared with the PRM's. Even another backbone's N16 is 78–89% explained by Qwen-7B's N16
     (linear R², table below). That suggests the shared part is driven by the text itself, not by one backbone.
   - So a useful new view has to measure something other than token uncertainty.
2. **One candidate from saved data looks clearly independent: the PRM's reward at the next step.**
   - It is almost uncorrelated with the PRM channel (r = .16) and with N16 (R² = .05, or .11 with PRM, position and
     length included).
   - Intuition: an error at step k usually continues into step k+1. 68–72% of erroneous answers have one contiguous
     error block. The PRM reads step k+1 with the wrong step k in its prefix, so its next reading is a second,
     partly independent look at the same region.
   - The catch: it measures the neighbour's label, not step k's. Under Jaffe's model it is a view of a shifted
     latent variable. It also carries some position (r = .23).
3. **The strongest candidate overall needs new inference: the PRM's reward under controlled input changes.**
   - Two examples: the reward of the same step with the question removed, and how much later rewards change when
     one step is removed.
   - A difference of two readings of the same PRM cancels the part of its noise that does not depend on the
     changed input. That is the shared nuisance that breaks conditional independence today.
   - It needs GPU forwards. Costs are in §4; the GPU jobs are proposed, not run.
4. **Some ideas are already covered or redundant.**
   - The lm_head reading at the scoring state is the same number as the existing `H1_first_token` of the next step
     (r = 1.000).
   - Running-max reward drop duplicates the PRM (r = .78).
   - Another backbone's telemetry duplicates our own (R² .78–.89).
   - Side finding: in N16 on Socratic, `digit_alternative` and `digit_alternative_innovation` correlate at
     r ≥ .997 for all four PRMs, so the digit family counts as two channels there, not three.

---

## 1. What is saved per token and per step

Source: `cluster/run_prmbench_prm_telemetry.py` (`--mode prm`), the same schema in `cluster/run_external_telemetry.py`.
The Socratic row was checked on disk: `qwen_prm_socratic/external_telemetry.pkl`, sha256 `1cc61f65…`, equal to the
bank MANIFEST.

| level | field | content |
|---|---|---|
| token (step content only) | `gen_token_ids` | written token id |
| | `token_entropies` | full-vocabulary entropy of the lm_head distribution |
| | `token_spilled_energies`, `token_logsumexp` | energy terms of the same distribution |
| | `top_k_logprobs` {ids, logprobs} | top-50 ids and log-probabilities, int32/float32 [tokens, 50] |
| step | `step_token_spans` | token span per step |
| | `rewards`, `native_labels` | the PRM's step reward (one number per step) and its 0.5 decision |
| answer | `terminal_reward` | only for adapters with a terminal sentinel (Skywork, not in our 5) |
| row | `question`, `steps`, `idx` | text; `error_steps` is `None` in the external caches (labels kept apart) |

Per adapter (`spectral_utils/prm_adapters.py`), the telemetry at token t is the lm_head distribution at position t−1.
For **Qwen-7B, Qwen-72B and PURE** the first token of step k+1 is therefore predicted from the very position whose
hidden state the reward head reads for step k (`<extra_0>`, or PURE's `\n\n`). For **RLHFlow and Math-Shepherd**
it is predicted from a template token (the next user turn, or `ки\n`). Their reward comes from the lm_head itself at
the tag position, which is not saved.

**Not saved anywhere:**
- hidden states at any layer;
- attention;
- the reward head at non-scoring tokens;
- the two-class reward logits, beyond `reward = softmax[1]` (for Qwen these carry no more information than
  logit(reward));
- the vocabulary distribution at the tag position of RLHFlow and Math-Shepherd.

**Where the caches are.**
- Socratic and ProcessBench raw telemetry for all 5 PRMs is on Drive:
  `gdrive:hallucination_detection/cluster_results/prm_boost_{external,processbench}_v1/`, about 1 GB each, with
  `bank_v2/{FEATURES,REWARDS}.npz`.
- PRMBench and Hard2Verify raw telemetry is on AIRCC only (`$R/prm_boost_v1/full_prm`, `$O/full_*`, `$X/full_*_hard2verify`).
- This Mac has no `aircc` ssh alias, so those were described from the code and manifests, not opened.

## 2. What is already covered (do not re-propose)

| idea | where it already exists |
|---|---|
| entropy / surprisal / margin / energy / Rényi / varentropy / tail mass of the top-50 | bank11, CT7, `hist_*` channels (56-channel bank) |
| sharp drops in top-20 entropy ("Mind the Gap", ICML 2026) | `evidence_drop_risk` (N56) |
| entropy at the step's first token = lm_head reading of the previous step's scoring state | `H1_first_token` (N56); checked r = 1.000 against the new `boundary_entropy` |
| digit alternatives | the digit family in N16 |
| hidden-state trajectory scalars (CoE, CoT-Kinetics) in a supervised LR | CROP's Trace+PRM (`papers/digests/crop-conformal-clean-prefix.md`) |
| per-layer logit-lens entropy, KL to final layer, residual norms | Step 243 white-box arm, generators only (`cluster/layer_lens.py`), never on a PRM |
| reversed-order PRM scoring | BiPRM (arXiv 2508.01682), trained, not label-free |

Quick web search (2026-10-26, three queries). It found no work on:
- reading a PRM's reward head at intermediate layers;
- using reflective-token mass as a label-free step-error feature;
- measuring how removing one step changes later PRM scores.

Adjacent work: self-reflection probes (arXiv 2506.12217, 2512.13979) and counterfactual step importance for credit
assignment (arXiv 2602.09331, 2604.18307). This is a search, not a survey; novelty is not claimed.

## 3. Candidate features

Dependence is what matters for L-SML. Near-duplicates of the PRM or of N16 break the grouping or just repeat a
view. "Expected dependence" is a prior; the measured numbers are in §5.

### (a) Computable now from saved data

| feature | what it measures | why it could add information | expected dependence | cost |
|---|---|---|---|---|
| **next-step PRM risk** `risk(r_{k+1})` | the PRM's verdict right after step k | an error usually continues; the PRM reads step k+1 with the wrong prefix, so this is a second reading of the same error region | low with the PRM at step k; it is a view of the neighbour's label, partly position | 0 (from `rewards`) |
| future mean PRM risk | mean risk over steps after k | the same idea, averaged | position proxy (r_pos .36) | 0 |
| reward change `risk_k − risk_{k−1}` | how sharply the verdict turns at step k | onset of an error block | moderate with the PRM (it is a linear function of it) | 0 |
| drop from the running-max reward | best reward so far minus the current one | a "first drop" rule | high with the PRM | 0 |
| **reflective-token mass** | top-50 probability on self-correction words ("Wait", "But", "However", "No", "wrong", …) at the scoring state after step k, and inside step k | the content of the alternatives, not their spread: the backbone "wants" to backtrack. A confident "Wait" has low entropy and still signals | expected low with entropy channels; likely sparse | 0 GPU; needs the backbone tokenizer vocab (Qwen2.5 tokenizer.json, about 7 MB) to map ids to words |
| lm_head reading at the scoring state (entropy, top-1) | backbone uncertainty at the reward head's own hidden state | the reward head reads this state | **already `H1_first_token`** | 0 |
| other PRMs' rewards on the same steps | a different model's verdict | different backbone and training data | depends on family (§5); a design change (multi-PRM fusion) | 0 (saved for all 5 PRMs × 3 benchmarks) |

### (b) Needing new inference (proposals only; each needs Omri's approval)

| feature | what it measures | why new | expected dependence | cost |
|---|---|---|---|---|
| **question-blind reward difference** `r_k(with question) − r_k(without)` | how much of the verdict is grounded in the problem rather than in the step looking plausible | the difference cancels the part of the PRM's noise that does not depend on the question; it targets counterfactual / domain-inconsistency errors | low with N16 (token telemetry barely changes without the question); moderate with the PRM | 1 forward per answer, the same as the original pass (Qwen-7B on PRMBench took about 15 min on one GPU, job 282802) |
| **leave-one-step-out influence** `Σ_{j>k} [r_j(without step k) − r_j]` | how much later verdicts depend on step k | a wrong step lowers later rewards; removing it raises them. This is causal, not distributional | low with N16; uses only the PRM | n_steps forwards per answer: about 9× the original pass on Socratic (8.7 steps per answer), about 13× on PRMBench; 72B is the expensive one |
| isolated-step reward (step k after the question alone) | whether the step is wrong locally or only in context | separates local errors from propagated ones | moderate with the PRM | n_steps short forwards, about 1–2× the original |
| reward head at intermediate layers (reward lens) | at what depth the verdict forms; disagreement between mid and final layers | an early-layer verdict carries different noise from the final one | high with the PRM at late layers, lower at middle layers | 1 forward with hidden states; store L × steps scalars (KB per answer). Reuse `cluster/layer_lens.py` |
| reward head at every token | where inside the step the verdict turns | finer than one number at the step end, for long steps | high with the PRM (same head) | the same forward as above |
| hidden-state geometry at the scoring state (norm, step-to-step drift, component orthogonal to the reward direction) | what the hidden state holds that the reward head ignores | the orthogonal part is by construction not in the reward | moderate with N16 (same backbone); CROP precedent (supervised) | 1 forward; about 0.5 GB per benchmark for 7B states at scoring positions |
| attention from the scoring state to the question / earlier steps / the current step | grounding, and copying for redundancy errors | targets the redundancy category | unknown; attention-ratio features exist in supervised work (Lookback-Lens style, not in our digests) | eager attention: memory-heavy on long rows; store aggregates only |
| activation-noise reward variance | how stable the verdict is | epistemic instability | moderate | K forwards; precedent: noise injection (ICLR 2026, in `papers/`) |

## 4. Ranked shortlist

1. **Counterfactual PRM rewards: the question-blind difference, then leave-one-step-out influence.**
   - This is the only family whose nuisance is not the shared token uncertainty: differencing two readings of the
     same PRM cancels it.
   - It is the best match to the diagnosis that conditional independence fails through a shared backbone nuisance.
   - Cost: 1× and about 9–13× one PRM pass for the 7B PRMs. Needs GPU approval.
2. **Next-step PRM risk** (computable now).
   - Almost independent of the PRM and of N16 (r .16, R² .11).
   - It uses the error-contiguity structure without labels.
   - In fusion it must be its own group, not the PRM's (L-SML would separate it anyway at r .16). Its position load
     must be checked on Socratic.
3. **The reward lens (reward head at intermediate layers and at every token).**
   - One cheap forward, and the infrastructure exists.
   - It is a PRM-group member rather than an independent view: useful if the final-layer verdict is the noisy
     part.
4. **A different-backbone PRM's reward** (saved for all cells).
   - RLHFlow's reward on Qwen-7B's rows: r .26 with Qwen-7B's PRM, R² .42 including N16, position and length.
   - Qwen-72B's reward duplicates Qwen-7B's (r .78).
   - This changes the method into step-level multi-PRM fusion. FUSE does this only at the response level, so it is
     a design decision for Omri, not a channel.
5. **Reflective-token mass** (saved data, but the tokenizer vocab is missing on this Mac).
   - The code path is in `feasibility.py`, with the word list fixed before any run.
   - Risk: Socratic steps have rigid prefixes ("Step", "1.", "-"), and base models rarely put mass on "Wait". The
     feature may be constant in most answers.

**Not recommended:**
- boundary entropy / top-1 at the scoring state (already `H1_first_token`);
- drop from the running max, and the reward change (functions of the PRM);
- other backbones' telemetry (text-driven, R² .78–.89);
- attention, before the counterfactuals have been tried (cost, and no label-free precedent).

## 5. Feasibility check (CPU, no labels)

`results/prm_new_features_v1/feasibility.py`, outputs `FEASIBILITY.csv`, `N16_REFERENCE.csv`,
`FEASIBILITY_META.json`, `feasibility.log`.
- **Inputs** (Drive, read-only): Qwen-7B Socratic raw cache plus `bank_v2` for Qwen-7B, Qwen-72B, PURE and
  RLHFlow. Hashes are in the README.
- **Population:** every row, 2,995 answers and 26,052 steps; this is not a subset.
- **Inputs to the measures:** all values are answer-z-scored, as fusion inputs are.
- **Measures:** r_prm is the correlation with z(logit(1−reward)), the CHAIN PRM channel. R²_all is the linear R² on
  z(PRM) + N16 + relative position + log step tokens.

| candidate | valid steps | r with PRM | max \|r\| with one N16 channel | R² on N16 | R²_all | r with position |
|---|---|---|---|---|---|---|
| next-step PRM risk | 88.5% | .16 | .17 (true_tail50) | .05 | **.11** | .23 |
| future mean PRM risk | 88.5% | .01 | .14 | .08 | .24 | .36 |
| reward change | 88.5% | .55 | .28 | .19 | .48 | −.03 |
| drop from running max | 100% | .78 | .56 | .43 | .66 | .41 |
| boundary entropy (scoring state) | 88.5% | .55 | .55 (q15_H1) | .36 | .42 | .19 |
| boundary top-1 surprisal | 88.5% | .66 | .68 | .57 | .63 | .29 |
| Qwen-72B reward | 100% | .78 | .53 | .39 | .65 | .23 |
| PURE reward | 100% | .60 | .64 | .56 | .71 | **.74** |
| RLHFlow reward | 100% | .26 | .18 | .25 | **.42** | −.32 |
| Qwen-72B backbone N16 mean | 100% | .31 | .74 | **.88** | .89 | −.18 |
| PURE backbone N16 mean | 100% | .37 | .76 | **.81** | .82 | −.21 |
| RLHFlow backbone N16 mean | 100% | .24 | .72 | **.78** | .80 | −.34 |

**Reference: the existing channels.** Each N16 channel regressed on the PRM, the other 15 channels, position and
length (`N16_REFERENCE.csv`):
- R² ranges from .35 (`dominant_freq16`) to 1.00 (the two digit-alternative channels); 13 of 16 are above .70.
- A new candidate with R²_all ≤ .5 is therefore less redundant than most current channels.

Checks:
- `boundary_entropy` equals the existing `H1_first_token` of the next step at r = 1.000. It is not new.
- `digit_alternative` vs `digit_alternative_innovation`, raw correlation: .9998 for Qwen-7B, .9998 for Qwen-72B,
  .9972 for PURE and .9999 for RLHFlow.
- The reflective-token path was smoke-tested with a synthetic vocabulary (it computes). No real numbers yet:
  the tokenizer is missing.

**Surface artefact seen while mapping step starts (not a feature, a caution).**
- Some perturbed Socratic steps begin with explicit error words: "Incorrect…" in about 30 steps, `错误…` in 44, and
  `前面似乎有些问题，让我重…` ("something above seems wrong, let me redo") in 37.
- Any text- or token-identity feature can read these. Check them before trusting such a feature on Socratic.

**What this check cannot say.**
- Low correlation is not class-conditional independence. It can also mean noise.
- These numbers are linear, from one PRM on one benchmark.
- Whether any candidate helps needs a label-using check. That check must be pre-registered first: candidate set,
  group placement, 8 cells (4 PRMs × PRMBench and Socratic), the same CHAIN code and the GAP rule.

## 6. What needs a decision

1. **Pre-register a label-using dev check** of next-step PRM risk as a new CHAIN group (CPU only), on the 8 cells.
   The PRMBench cells need the AIRCC caches (Omri's PC).
2. **Approve GPU** for the question-blind reward pass: 1× PRM pass per PRM and benchmark, the 7B PRMs first. Then,
   if it helps, leave-one-step-out.
3. **The tokenizer for reflective-token mass.** Either fetch Qwen2.5's tokenizer.json (public, about 7 MB) or run
   `feasibility.py <data> <out> $S/hf_cache/…/tokenizer.json` on AIRCC.
4. **Whether multi-PRM step-level fusion is in scope.** It is a method change, not a channel.
