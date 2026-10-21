---
source_pdf: papers/2605.30085v2_CROP.pdf
slug: crop-conformal-clean-prefix
pages: 39
extracted_on: 2026-10-21
extraction_mode: curated (key sections only)
---

# CROP: Conformal Certification of Reasoning Trace Prefixes (arXiv 2605.30085v2)

**Scope of this file.** A curated extraction of the sections this project relies on: setup, guarantee,
algorithm, risk proxies and features, protocol, and the numeric tables. Prose is condensed in close
paraphrase with page references; equations, hyperparameters and every table number are transcribed exactly
from the PDF text layer. It is not the whole paper. To regenerate the full raw page text locally (not
committed): `python skills/paper-digest/scripts/extract_pdf_text.py papers/2605.30085v2_CROP.pdf --out-dir <tmp dir>`.
The full raw extract was read in full (all 39 pages) on 2026-10-21 before this file was written.

## Title block (p.1)

- Title: "Conformal Certification of Reasoning Trace Prefixes".
- Authors: Matt Y. Cheung, Ashok Veeraraghavan (Rice ECE), Hanjie Chen (Rice CS), Guha Balakrishnan (Rice ECE).
- Title-page date: October 14, 2026. arXiv stamp: `arXiv:2605.30085v2 [cs.AI] 10 Oct 2026`.
- Venue: **not stated in the PDF**. (The project registry records "Accepted at EMNLP 2026 (Oral, Main)" from
  the arXiv abstract page; see `docs/research_notes/competitor_registry_2026-10-21/REGISTRY.md` §1.2.)
- Code: github.com/matthewyccheung/crop.

## Abstract and contributions (pp.1-2), condensed

- CROP = Conformal Reasoning Output Prefixes: a verifier-agnostic calibration procedure for clean-prefix
  certification. Given any step-level risk proxy, it selects a calibrated threshold and returns the longest
  contiguous prefix whose step risk proxies stay below it; the uncertified suffix is routed to review or repair.
- Under exchangeability, it controls the **marginal** probability that the returned prefix contains an
  annotated error.
- Six process-labeled datasets: Arithmetic and GSM8K (both from CRV [8]), ProcessBench, Math-Shepherd,
  PRMBench, PRM800K.
- Claims: step AUROC does not fully capture prefix utility; CROP balances over-withholding and unsafe overshoot;
  improves downstream repair in most settings.
- Contributions (p.2): (1) formalize clean-prefix certification; (2) finite-sample marginal, reasoning-instance
  level control of annotated prefix contamination under exchangeability; (3) empirical: AUROC insufficient,
  better stopping points, repair gains.

## Sec. 3 Problem setup (pp.4-5)

- Instance X = (q, τ), trace τ = (τ_1..τ_T). Prefix τ_{1:m}; uncertified suffix τ_{m+1:T}.
- Annotation Y = (Y_1..Y_T) ∈ {0,1}^T, Y_t = 1 = annotated error. The annotation protocol (human, programmatic,
  LLM-as-judge) must be fixed before calibration; CROP controls risk **relative to these annotations**.
  Labeled instance Z = (X, Y).
- Step-level risk proxy s: X ↦ (s_1(X)..s_T(X)) ∈ R^T, R_t = s_t(X); larger = more evidence of error.
  Examples: PRMs, likelihoods, hidden-state statistics, heuristics. Need not be calibrated or accurate, but must
  be **fixed before conformal calibration**.
- Three disjoint instance-level label roles: proxy-training labels (fit s if learned), calibration labels
  (select λ̂), test labels (retrospective evaluation only). At deployment s(X) is computed from the unlabeled
  instance; no step labels needed at deployment.
- **Assumption 1 (Reasoning-Instance Exchangeability).** Conditioning on all data used to build s and on all
  preprocessing/calibration-design choices fixed before seeing calibration instances, Z_1..Z_n and the future
  Z_{n+1} are exchangeable. Steps within a trace may be arbitrarily dependent. Holds e.g. for i.i.d. sampling
  from the same task distribution, same fixed generator/prompting, same fixed annotation protocol.

## Sec. 4 Method (pp.5-7)

Key observation: prefix contamination (retained prefix contains ≥1 annotated error) is a binary loss
monotone in the threshold, so conformal risk control (CRC, [35]) applies.

**Lemma 1 (Monotone binary-loss risk control).** W_1..W_{n+1} exchangeable; {L_λ : λ ∈ Λ} a fixed family of
binary losses on a finite ordered threshold set Λ, L_λ(w) ∈ {0,1} nondecreasing in λ for every w. For α ∈ (0,1):

  λ̂ = max { λ ∈ Λ : (1 + Σ_{i=1}^n L_λ(W_i)) / (n + 1) ≤ α }        (1)

if the feasible set is nonempty; else λ̂ = ⊥ with L_⊥(w) = 0. Then E_{W_{1:n+1}}[ L_λ̂(W_{n+1}) ] ≤ α   (2),
jointly over calibration sample and future example (λ̂ is random through W_{1:n}).
Proof idea: λ̂ admits at most ⌊α(n+1)⌋ − 1 calibration failures; if the future sample fails too it is in a pool
of at most ⌊α(n+1)⌋ failures; exchangeability gives probability ≤ ⌊α(n+1)⌋/(n+1) ≤ α. Full proof App. A
(pp.18-19) uses first-failing thresholds B_i = min{λ : L_λ(W_i) = 1}, L_λ(W_i) = 1{B_i ≤ λ}, k = ⌊α(n+1)⌋,
random tie-breaking ranks.

**CROP loss (Sec. 4.2).** Retained prefix length

  M_λ(R_{1:T}) = max { m ∈ {0..T} : R_t ≤ λ for every t ≤ m }        (3)

  L_λ(Z) = 1{ ∃ t ≤ M_λ(R_{1:T}) : Y_t = 1 }                        (4)

  F = { λ ∈ Λ : (1 + Σ_{i=1}^n L_λ(Z_i)) / (n + 1) ≤ α }             (5)

λ̂ = max F if F nonempty, else ⊥ with M_⊥ = 0 (empty prefix).

**Proposition 1 (Clean-prefix risk control).** Under Assumption 1, evaluating M_λ̂ on s(X_{n+1}):
P_{Z_{1:n+1}}( ∃ t ≤ M_λ̂ : Y_{n+1,t} = 1 ) ≤ α   (6). The paper states explicitly (p.6) that this is
finite-sample **marginal** risk control and **not a conditional guarantee** for each realized prefix.

**Algorithm 1 (p.7).** For each calibration trace: R_i ← s(X_i); for each λ: M_{i,λ} (first-crossing rule),
L_{i,λ}. Compute F; if empty return empty-prefix fallback; λ̂ ← max F; for the test trace compute R_{n+1},
M̂ = max{m : R_{n+1,t} ≤ λ̂ ∀ t ≤ m}; return τ_{n+1,1:M̂}. Validity needs instance exchangeability and s fixed
before calibration; a better proxy only improves efficiency (longer prefixes).

## Sec. 5 Experiments: risk proxies (pp.8-9) and their construction (App. B, pp.19-21)

Five step-level scoring rules:
1. **Random**: i.i.d. uniform step scores drawn before calibration.
2. **Token/format logistic**: LR over surface features (position, length, punctuation...).
3. **Trace-feature logistic**: LR combining token/format features with precomputed likelihood, confidence and
   hidden-state summaries **from Llama-3.1-8B-Instruct** (p.9).
4. **Direct PRM**: frozen Qwen2.5-Math-PRM-7B; error risk = 1 − positive-class probability.
5. **Trace features + PRM**: LR over the trace-feature inputs plus the Direct PRM score; output = predicted
   P(Y_t = 1).

LR settings (p.9): ℓ2 penalty, C = 1.0, L-BFGS, fitted on the proxy-training partition with the local step
label Y_t. The Direct PRM stays frozen and receives no CROP labels.

**Token/format inventory (App. B, p.20).** Text-only, no model internals, no PRM, no labels. Per step, counts
from the current step and the original problem: character length, whitespace-token count, digit count,
alphabetic-character count, equality signs, arithmetic operators {+, −, ∗, /, ∧}, newlines, colons,
parentheses, angle brackets. Position: normalized step position, raw step number, total number of steps.

**Trace-feature inventory (App. B, p.20).** Token/format-style text and metadata features (step number, total
trace length, relative step position, dataset/task complexity when available, lengths of the problem, current
step, previous generated prefix and the prefix after the current step; counts of digits, letters, arithmetic
symbols, parentheses, Boolean tokens, periods, commas, colons) PLUS model-derived features:
- described as precomputed non-PRM uncertainty signals "from the trace-generating language model" (p.20),
  while p.9 names Llama-3.1-8B-Instruct; all oriented so larger = higher risk;
- for each completed step, the already-written step text is scored under the model (i.e. teacher-forced
  scoring of the given trace, not generation); summaries of token confidence, sequence likelihood,
  prediction entropy, calibrated confidence, and energy-style uncertainty;
- latent-trajectory summaries from prior work [56] (Latent space chain-of-embedding, ICLR 2025) and [57]
  (CoT-Kinetics) measuring whether the internal trajectory stays relevant, consistent and stable;
- raw hidden states are NOT passed to the LR; the trajectory is reduced to scalar diagnostics beforehand;
  no further truncation of the scalar summaries.
- The exact list of model-derived features, the token-to-step aggregation operator, and the layer(s) used are
  **not specified in the PDF**.

**PRM risk proxy construction (App. B, pp.20-21).** Records: trace id, dataset/domain, problem text, ordered
step texts, step labels (labels are not given to the PRM). Input: an instruction to reason step by step, the
problem, the complete trace, with the Qwen PRM separator `<extra_0>` after every step; one pass over the
completed trace (no per-prefix rescoring). Qwen tokenizer, truncation at 8192 tokens, eval mode. h_t = two
class logits at the separator of step t; r_t^PRM = softmax(h_t)_1 (correct-step class); risk
R_t^PRM = 1 − r_t^PRM. For Trace+PRM, R_t^PRM is appended to the trace features before fitting the same LR.
The PRM is never fine-tuned; PRM outputs are fixed before the calibration split is used.

**Learned risk proxy fitting (App. B, p.21).** scikit-learn pipeline: median imputation → StandardScaler →
ℓ2 LogisticRegression; solver L-BFGS, C = 1.0, max_iter = 500, class_weight = balanced, random seed = split
seed; same settings for all learned proxies; fitted only on the proxy-training split; Trace-feature row trains
on per-step Y_t; Trace+PRM appends the PRM risk; single-class training split → prior-probability dummy
classifier; the fitted P(step error) is used directly as the risk proxy; **no post-hoc probability
calibration**. Compute: NVIDIA A100-PCIE-40GB, ≈450 GPU-hours total, dominated by PRM scoring and repair
inference.

## Evaluation metrics (p.9)

α = 0.05 primary. For test trace i: T_i steps, M_i = M_λ̂(R_{i,1:T_i}). Oracle annotated-clean prefix length
O_i = min({t − 1 : Y_{i,t} = 1} ∪ {T_i}).
1. Retained-prefix fraction M_i/T_i (per-trace fraction, averaged over test traces).
2. Prefix-contamination risk: fraction of test traces whose retained prefix contains ≥1 annotated error.
3. Boundary deviation |M_i − O_i|/T_i = over-withholding (O_i − M_i)_+/T_i + unsafe overshoot (M_i − O_i)_+/T_i.
4. Repair accuracy: normalized exact match of the parsed final answer vs gold.
Step AUROC (App. C, p.22): ranking of erroneous vs clean steps **after pooling steps together** (pooled over
the test split, not within-trace).

## Datasets, splits, seeds, grids (p.9; App. B pp.19-22)

Main benchmark: 2,819 traces / 19,311 steps / 1,006 annotated errors (Arithmetic + GSM8K from CRV);
402 traces have ≥1 error. Arithmetic: 1,500 traces, 10,316 steps, 349 step errors. GSM8K: 1,319 traces,
8,995 steps, 657 step errors. CRV traces are released annotated traces (not regenerated); Arithmetic labels =
task-specific automatic checks + stronger-model judge; GSM8K labels = stronger-model judge. A Boolean CRV
domain was excluded (0.5% error steps, 1.8% error traces; ceiling effect). CRV traces were generated by a
Llama-family model (p.11).

Supplementary datasets (ProcessBench, Math-Shepherd, PRMBench, PRM800K) are used only for fixed-risk prefix
utility; boundary and repair analyses are restricted to Arithmetic and GSM8K. Each dataset uses its native
labels and is calibrated separately.

Table 4 (p.21): Dataset provenance and annotation protocols (Err. = % of steps erroneous).

| Dataset | Traces | Steps | Err. | Label protocol | Trace origin |
|---|---|---|---|---|---|
| Target arithmetic | 1,500 | 10,316 | 3.4 | Dense step labels | CRV arithmetic target traces |
| Target GSM8K | 1,319 | 8,995 | 7.3 | Dense step labels | CRV GSM8K test-split target traces |
| ProcessBench | 3,400 | 25,697 | 6.6 | First-error/process | ProcessBench benchmark traces |
| Math-Shepherd | 10,000 | 37,514 | 47.6 | Step markup | Math-Shepherd-style public data |
| PRMBench | 5,000 | 67,445 | 15.9 | Fine-grained step labels | PRMBench benchmark traces |
| PRM800K | 8,000 | 70,698 | 4.2 | Human step labels | PRM800K process-supervision traces |

- Splits: reasoning-instance (trace) level 60/20/20 = proxy fitting / conformal calibration / test; same
  partition for all proxies within a seed; stratified by dataset/domain and by has-error.
- Seeds: target-domain prefix utility and repair: 20 seeds 2806..2825; additional datasets: 10 seeds
  2806..2815; boundary and risk sensitivity use the same 20 target splits.
- Threshold grids (p.19): target-domain, boundary, α-sweep and repair experiments use a per-source grid Λ_s of
  **201 empirical quantiles of the proxy-training-split step scores** (ties allowed; label-free). Additional
  datasets (incl. PRMBench) use the **fixed grid {0, 0.01, ..., 1.00} (101 points)**. Calibration only picks
  the largest feasible λ from the predeclared grid.
- PRMBench subset: the PDF says only "5,000 traces"; which 5,000 is not stated in the PDF. (The project's
  parity work, from their code, identifies the first 5,000 rows of `hitsmy/PRMBench_Preview`.)
- Note: Table 5 (p.22) mentions "50 target splits" for the token/format and trace-feature timing rows, which does
  not match the 20 seeds stated elsewhere.
- α values used: 0.05 primary; 0.025 and 0.10 in App. E; the fallback check also mentions 0.075 (p.12).
  Fallback (empty feasible set) never triggered in any run; a nonempty feasible set needs 1/(n+1) ≤ α. At
  α = 0.05, 0% of Arithmetic and 6.4% of GSM8K test traces got an empty prefix (p.12).

## Table 6 (p.32): fixed-risk prefix utility and step ranking at α = 0.05

Means ± 95% normal intervals over repeated splits (CRV 20, others 10). Columns: step AUROC (pooled), prefix
kept (%), prefix risk (%).

| Dataset | Labels | Score source | Step AUROC | Prefix kept | Prefix risk |
|---|---|---|---|---|---|
| Arithmetic | dense step | Random | 0.5049 ± 0.0164 | 62.0 ± 5.0 | 4.6 ± 0.6 |
| | | Token/format | 0.8168 ± 0.0116 | 77.5 ± 2.7 | 4.7 ± 0.6 |
| | | Trace features | 0.8564 ± 0.0153 | 90.3 ± 1.5 | 4.5 ± 0.6 |
| | | Direct PRM | 0.9421 ± 0.0100 | 97.5 ± 0.4 | 4.2 ± 0.4 |
| | | Trace+PRM | 0.9701 ± 0.0075 | 97.1 ± 0.4 | 4.1 ± 0.5 |
| GSM8K | dense step | Random | 0.5046 ± 0.0110 | 27.2 ± 2.6 | 4.3 ± 0.7 |
| | | Token/format | 0.7081 ± 0.0126 | 39.8 ± 2.8 | 4.9 ± 0.7 |
| | | Trace features | 0.7495 ± 0.0111 | 43.2 ± 2.7 | 4.3 ± 0.7 |
| | | Direct PRM | 0.8659 ± 0.0088 | 81.7 ± 4.3 | 5.1 ± 0.8 |
| | | Trace+PRM | 0.8812 ± 0.0113 | 63.1 ± 5.0 | 4.5 ± 0.8 |
| ProcessBench | first-error/process | Random | 0.4887 ± 0.0104 | 9.8 ± 1.6 | 4.9 ± 1.1 |
| | | Token/format | 0.7057 ± 0.0034 | 10.6 ± 1.4 | 4.8 ± 0.5 |
| | | Trace features | 0.7058 ± 0.0035 | 11.3 ± 1.0 | 4.5 ± 0.4 |
| | | Direct PRM | 0.8801 ± 0.0059 | 46.7 ± 1.0 | 4.5 ± 0.6 |
| | | Trace+PRM | 0.8935 ± 0.0081 | 42.4 ± 1.5 | 4.5 ± 0.5 |
| Math-Shepherd | step markup | Random | 0.5012 ± 0.0049 | 2.5 ± 0.2 | 4.3 ± 0.4 |
| | | Token/format | 0.6637 ± 0.0024 | 7.8 ± 0.5 | 4.8 ± 0.5 |
| | | Trace features | 0.9932 ± 0.0005 | 28.1 ± 0.3 | 5.2 ± 0.5 |
| | | Direct PRM | 0.4537 ± 0.0043 | 1.0 ± 0.1 | 0.9 ± 0.1 |
| | | Trace+PRM | 0.9944 ± 0.0004 | 28.1 ± 0.3 | 5.1 ± 0.5 |
| PRMBench | fine-grained | Random | 0.4994 ± 0.0050 | 8.7 ± 0.5 | 4.7 ± 0.4 |
| | | Token/format | 0.7751 ± 0.0034 | 16.2 ± 1.0 | 5.0 ± 0.7 |
| | | Trace features | 0.7812 ± 0.0034 | 15.9 ± 0.7 | 4.6 ± 0.5 |
| | | Direct PRM | 0.8010 ± 0.0049 | 4.8 ± 0.2 | 0.6 ± 0.1 |
| | | Trace+PRM | 0.8336 ± 0.0027 | 21.8 ± 0.9 | 5.0 ± 0.6 |
| PRM800K | human step | Random | 0.5002 ± 0.0100 | 20.5 ± 1.1 | 5.4 ± 0.5 |
| | | Token/format | 0.9919 ± 0.0014 | 89.0 ± 0.3 | 4.8 ± 0.4 |
| | | Trace features | 0.9925 ± 0.0013 | 89.4 ± 0.3 | 4.8 ± 0.4 |
| | | Direct PRM | 0.9327 ± 0.0021 | 67.2 ± 1.2 | 4.5 ± 0.5 |
| | | Trace+PRM | 0.9934 ± 0.0012 | 89.5 ± 0.4 | 4.6 ± 0.4 |

App. C (pp.22-23): the highest-AUROC and highest-utility proxies often but not always coincide (Trace+PRM best
AUROC on GSM8K and ProcessBench, Direct PRM keeps more). Reason given: pooled AUROC is global ranking; CROP stops
at the first crossing, so an early clean step with high risk stops the prefix.
**Proposition 2** (pp.23-24): two proxies with equal step AUROC can have different prefix utility. Construction:
single instance, labels (0,0,1), Λ = {0.5, 0.9}, R_A = (0.7, 0.0, 0.8), R_B = (0.0, 0.4, 0.8); both AUROC = 1;
λ̂ = 0.5 for both; M_A = 0, M_B = 2 (utilities 0 and 2/3).

## Sec. 5.1-5.3 and Table 1 (pp.8-10): boundary quality vs whole-trace abstention, α = 0.05, 20 splits

Proxies fixed before boundary/repair analyses: Trace+PRM for Arithmetic, Direct PRM for GSM8K; both
interfaces use the same proxy and are separately calibrated.

| Dataset | Method | Empirical risk | Steps kept | Over-withholding | Unsafe overshoot | Boundary deviation |
|---|---|---|---|---|---|---|
| Arithmetic | Whole-trace abstention | 4.4 ± 0.7 | 95.8 ± 0.7 | 2.4 ± 0.5 | 2.1 ± 0.3 | 4.5 ± 0.2 |
| | CROP prefix | 4.3 ± 0.4 | 97.2 ± 0.4 | 0.6 ± 0.2 | 1.7 ± 0.2 | 2.3 ± 0.3 |
| GSM8K | Whole-trace abstention | 5.0 ± 0.8 | 77.9 ± 2.0 | 13.4 ± 1.8 | 2.6 ± 0.4 | 15.9 ± 1.6 |
| | CROP prefix | 5.0 ± 0.8 | 81.6 ± 4.3 | 9.4 ± 4.2 | 2.3 ± 0.4 | 11.7 ± 4.0 |

## Table 2 (p.10): downstream repair accuracy (%), α = 0.05, 20 stratified 60/20/20 splits

| Dataset | Repair model | Question only | Full trace | Whole-trace abst. | CROP prefix | CROP − best Δ [95% CI] |
|---|---|---|---|---|---|---|
| Arithmetic | Gemma 4 E4B IT | 80.5 | 79.9 | 83.4 | 88.7 | +5.32 [4.41, 6.22] |
| | Qwen2.5-7B-Instruct | 85.5 | 90.8 | 90.3 | 92.4 | +1.02 [0.28, 1.75] |
| | DeepSeek-R1-0528-Qwen3-8B | 63.9 | 78.5 | 78.8 | 85.0 | +5.53 [4.79, 6.28] |
| | Llama3.1-8B | 73.7 | 87.3 | 90.5 | 89.4 | -1.18 [-1.84, -0.53] |
| GSM8K | Gemma 4 E4B IT | 82.3 | 88.8 | 88.6 | 88.3 | -0.91 [-1.53, -0.29] |
| | Qwen2.5-7B-Instruct | 80.5 | 85.1 | 86.3 | 87.6 | +0.95 [0.09, 1.81] |
| | DeepSeek-R1-0528-Qwen3-8B | 70.0 | 84.9 | 84.6 | 87.4 | +1.83 [0.89, 2.76] |
| | Llama3.1-8B | 71.1 | 73.3 | 74.6 | 73.9 | -1.18 [-2.02, -0.34] |

Five of eight settings improve. Repair models run via Ollama with deterministic decoding (Table 9, p.34).
Table 3 (p.12), CROP-minus-baseline repair (pp) on originally incorrect traces with an annotated error, by
CROP position relative to the oracle boundary (vs question only / full trace / whole-trace abstention):
Before +5.7 ± 2.5 / +21.9 ± 3.3 / +9.0 ± 2.9; At +5.9 ± 2.3 / +20.3 ± 1.8 / +8.7 ± 2.0;
Beyond −20.1 ± 3.0 / −1.4 ± 2.3 / −4.3 ± 2.4.

## Table 7 (p.33): CROP vs whole-trace abstention on pooled Arithmetic+GSM8K, α = 0.05, 20 splits

| Score | Object | Emp. risk | Kept | Full accept |
|---|---|---|---|---|
| Token/format | Whole | 4.8 ± 0.5 | 59.8 ± 2.7 | 62.0 ± 2.4 |
| | CROP | 4.7 ± 0.5 | 60.5 ± 2.3 | 4.7 ± 2.8 |
| Direct PRM | Whole | 4.4 ± 0.5 | 85.6 ± 1.0 | 86.5 ± 0.9 |
| | CROP | 4.4 ± 0.4 | 89.8 ± 0.6 | 83.7 ± 0.8 |
| Trace + PRM | Whole | 4.5 ± 0.4 | 86.4 ± 1.0 | 87.4 ± 0.9 |
| | CROP | 4.6 ± 0.4 | 89.9 ± 0.6 | 82.2 ± 1.3 |

"Kept" here is step-weighted; "Full accept" = % traces retained in full.

## Table 8 (p.33): common risk budgets (paired Δ = CROP − abstention)

Arithmetic: α=0.025 kept +3.0 [2.2, 3.8], boundary −3.2 [−3.9, −2.6]; α=0.05 +1.4 [0.8, 2.1], −2.2 [−2.5, −1.9];
α=0.10 exact tie (both 100.0 kept, 8.3 risk).
GSM8K: α=0.025 −2.2 [−6.4, 2.0], +2.1 [−2.0, 6.2] (not significant, favors abstention on point estimate);
α=0.05 +3.7 [0.6, 6.9], −4.2 [−7.0, −1.5]; α=0.10 +3.5 [2.6, 4.5], −4.5 [−4.9, −4.1].

## App. G / Table 16 (p.38): artifact, label and order controls (pooled CRV target splits)

Artifact matching (p.26): group traces by domain, trace-length bucket, average step-length bucket, answer
format, and whether the prompt has arithmetic/logical operators; average retained-prefix fractions equally over
nonempty groups (changes only the summary, not calibration). Label control = break label-trace association
during proxy fitting; order control = disrupt step order within traces.

Panel A (retained prefix after balancing artifacts): Token/format LR risk 3.6, kept 62.0, gain +0.0;
Trace-feature LR 4.5, 64.0, +2.0; Direct PRM 3.9, 89.5, +27.4; Trace+PRM LR 3.2, 88.1, +26.1.

Panel B (AUROC / prefix risk / prefix kept): Random 0.501 ± 0.005 / 4.5 ± 0.2 / 51.0 ± 1.5;
Token/format LR 0.827 ± 0.004 / 4.7 ± 0.2 / 78.9 ± 0.9; Trace-feature LR 0.861 ± 0.004 / 4.6 ± 0.2 / 84.0 ± 0.7;
Label-shuffled 0.724 ± 0.014 / 4.7 ± 0.2 / 76.5 ± 1.0; Trace-order shuffled 0.863 ± 0.007 / 4.7 ± 0.3 / 78.5 ± 1.1.
(No artifact control is reported for PRMBench or the other external datasets.)

## App. H (pp.27-31): groupwise diagnostics at α = 0.05 (no recalibration within groups)

Score choices: Trace+PRM for Arithmetic, Direct PRM for GSM8K; external datasets: Direct PRM for ProcessBench,
Trace+PRM for Math-Shepherd, PRMBench and PRM800K. Intervals over overlapping splits quantify split
sensitivity, not population uncertainty.
- Trace length (pooled CRV): risk 2.8% (2-5 steps), 4.8% (6-7), 4.8% (8-9), 9.7% (≥10); retention 90.7-91.7%
  for ≤9 steps, 81.8% for ≥10 (≈60 test traces/split; error prevalence 27.4% vs ≈11.3-13.5%).
- Arithmetic difficulty levels 3/5/7/10: risk 2.3/2.2/5.6/7.3%; retention ≈99/≈99/96.2/94.7%.
- Dataset: mean empirical risk 4.3-5.1% across all six; retention from 21.8% (PRMBench), 28.1% (Math-Shepherd)
  to 89.5% (PRM800K), 97.2% (Arithmetic); ProcessBench 4.5% risk with 46.7% retention. **PRMBench: 96.4% of
  held-out traces contain an annotated error; CROP kept 21.8% at 5.0% risk.**
- First-error position (erroneous traces only, annotation-defined groups): contamination 29.9/38.1/29.8%
  (early/middle/late); retention 30.6/51.3/63.2%; early group ≈12 traces per split.

## App. I / Table 17 (p.39): cross-domain transfer (exchangeability intentionally violated)

| Train/cal domain | Test domain | Prefix risk | Prefix kept | FA | Accept |
|---|---|---|---|---|---|
| Arithmetic | GSM8K | 14.8 | 69.9 | 13.3 | 62.5 |
| GSM8K | Arithmetic | 5.1 | 69.0 | 4.4 | 54.7 |

Presented as a failure diagnostic, not a validity result; recalibration is required under shift.

## Limitations stated (p.13)

Marginal, not conditional; certifies agreement with a fixed annotation protocol, not semantic truth, final-answer
correctness or causal faithfulness; step labels are costly; changes in task, prompts, generator, proxy or
annotation policy break exchangeability; post-hoc on completed traces; weak or poorly ordered proxies satisfy the
risk constraint only with short prefixes.
