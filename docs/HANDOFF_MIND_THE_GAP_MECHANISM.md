# HANDOFF — building the Mind-the-Gap mechanism properly

Claude, 2026-09-19. Branch `claude/token-probability-fusion-v1`, worktree
`.worktrees/token-probability-fusion-v1`, 27 commits, clean tree.
Everything below runs from cached data on CPU. No GPU, no new inference.

---

## 1. The one experiment worth running, and why it is the only one left

Our arm and theirs differ on **two** axes, and the session just finished tested each one
while holding the other at *our* setting. Both failed. The cell that is actually theirs
was never run.

| | argmax over all steps *(our rule)* | first step crossing a threshold *(their rule)* |
|---|---|---|
| **level** statistic *(ours)* | **35.92** — the published arm | **tested here, worse**: 40.58 / 25.64 / 30.04 on GSM8K / Olympiad / Omni at q=0.8, against argmax's 48.79 / 30.79 / 30.70 |
| **derivative** statistic *(theirs)* | **tested, 27.96** — much worse, and worse on long chains (`long−short` interaction −9.53 [−13.97, −5.28]) | **NEVER RUN — this is theirs** |

Read the two failures carefully, because they are not independent evidence against the
method:

- The derivative was tested **under our argmax**. A derivative series has "when did
  evidence fall" semantics; argmax asks "where did it fall most", which is not the same
  question and throws away the ordering.
- First-crossing was tested **on our level series**. Thresholding a level picks the first
  step that is merely *hot*, not the first that *drops*. And as `q → 1` first-crossing
  degenerates into argmax, so on a level series the whole family is dominated by its
  endpoint — which is what the pilot shows.

**The statistic and the decision rule are a matched pair.** Testing either alone against
our own counterpart is uninformative, and I did exactly that twice.

### Why this is the right thing to chase

Three measurements from this session point the same way.

1. **Their advantage grows with chain length; ours does not.** Ratio to chance:
   ours 2.34 → 1.85 → 2.27 → 2.22 across GSM8K / MATH / Olympiad / Omni; theirs
   2.15 → 1.79 → **3.12** → **2.71**. Relative to difficulty our detector is flat and
   theirs improves. An argmax over `n` candidates degrades as `n` grows (the maximum of
   more draws is larger); a fixed-threshold first-crossing does not have that geometry.
2. **Our misses are systematically LATE.** Among erroneous answers we get wrong, the
   prediction falls after the true first error more often than before it, by +9.1 pp on
   GSM8K, +24.5 on MATH, +3.0 on OlympiadBench and **+31.6 on Omni-MATH**. The target is
   the *first* error and our rule has no ordering prior at all. Theirs does, by
   construction.
3. **The per-step evidence is not the problem.** Lift at the true error step is the same
   on OlympiadBench as on GSM8K (0.718 vs 0.725 for `q15_VE1`). What changes is that the
   strongest *wrong* step climbs from 0.578 to 0.727 SD while the true one stays at
   ~0.4, so the mean margin is **negative** everywhere: −0.11 on GSM8K, −0.31 on
   OlympiadBench. No aggregation choice inside a step repairs a negative margin; a rule
   that does not have to beat every competitor might.

Their own paper agrees with the shape of this, and our digest already says so: *"Drop is
not a uniform win... The consistent wins are on the harder distributions (MATH,
OlympiadBench, Omni-MATH) — the same 'helps where the signal is hard' pattern our own
work keeps finding."*

---

## 2. Their exact recipe, and the hole in it

From `papers/digests/mind-the-gap-catching-hallucinations-via-evidence-drop.md`
(Chen, Chen, Yue, Li — Tongji + Shanghai Univ., **ICML 2026, PMLR 306**). Read the digest
before writing code; it carries **five verified traps** beyond the one below.

- **Evidence** `Ê = −H(P̃)`, the negative entropy of the **renormalized top-20**
  distribution. Note: top-**20**, not the top-15 our bank uses. Our cache stores top-50
  log-probs, so top-20 renormalization is exactly reproducible.
- **Smoothing**: EMA with **span 5**.
- **Risk**: mean of the **M = 5 most negative first-differences**.
- **Decoding**: greedy τ=0, top-p 0.95; Qwen3-4B / Qwen3-8B; teacher-forced over the
  given chains — the same access we have.
- **SLA**: "the first **step** t where Δt exceeds a step-wise threshold", measured on
  **erroneous traces only**. They never report ProcessBench F1 and never report
  clean-trace abstention.

**The hole, quoted from the digest:** *"SLA's token→step aggregation is undefined. `Δ_j`
is a token-level first-difference, but SLA is defined on the first step whose `Δt`
exceeds a step-wise threshold. How token Δ collapses to step Δ is the dominant free
parameter for Table 3 and is never given. Neither is the handling of ProcessBench rows
with no error, nor whether 'matches' allows tolerance."*

So Table 3 **cannot be reproduced exactly**. That is not a blocker, it is a constraint on
what may be claimed: any implementation is an *adaptation*, must be labelled one, and
must not be described as their method or carry their name (project rule: *tailor, never
transplant* — take the mechanism, cite the idea, do not claim the method).

Note also my earlier derivative attempt used **EMA 16 and M=3**, not their 5 and 5, and
applied the transform per channel rather than to a single evidence series. Its failure is
therefore not even a test of their constants.

---

## 3. Do not redo these — all closed this session, with numbers

| closed | evidence |
|---|---|
| "the gate was hiding the token result" | CT7 leads gate-free by +3.96 pp [+2.28, +5.59] and under **both** gates held fixed. The 7.11 pp gap decomposes exactly: 3.86 locator + 3.25 gate |
| length-calibrated readout (Step 420 tool) | cost is **uniform**: short −10.85, long −10.42, interaction +0.43 [−3.68, +4.60] |
| derivative channel under argmax | 27.96 vs 34.84; interaction −9.53 [−13.97, −5.28]; and its length coupling is +0.583, barely below the level's +0.665 — "mean of the M largest rises" is itself an order statistic |
| `sw_var_peak` as a 12th channel | hurts, −1.74 [−2.80, −0.70] on long chains. Standalone it is real (25.99 vs 16.58 chance) but correlates **+0.730** with `q15_H1`, the entropy it is built from |
| narrower fitting scope (per model, per cell, PRMBench-excluded) | all neutral-to-negative; per-cell −1.34 [−2.51, −0.17] |
| readout width — global | +0.90 [−0.47, +2.12] over K=10; flat optimum |
| readout width — per answer | binary oracle worth **0.70 pp**; every spectral and answer-local L-SML descriptor at chance (group count 0.495, residual 0.479) |
| readout width — per cell | +0.26 [−1.18, +1.29] out of sample over one global K, across 400 source-group half splits |
| fixed-quantile readouts | catastrophic (20.7–25.5 vs 35.9). Error evidence is concentrated in an **absolute** token count |

Two facts that are real and should be *kept*, not retested: the per-cell optimal K is
stable and rises with chain length (12–18 short, 28–47 long), and `n_eff/n` falls from
0.46–0.51 on GSM8K to 0.35–0.36 on the long cells.

---

## 4. What is on disk

All under `results/token_probability_fusion_v1/` unless noted.

| file | what |
|---|---|
| `TOKEN_MATRICES.npz` | **the main cache** — risk-oriented 11-channel token bank, `[6968779, 11]` float32, plus `token_offsets` and `step_spans`. 308 MB, ~95 min to rebuild |
| `STEP_VIEWS_12CH.npz` | per-channel Top-10 step readouts, answer-standardized |
| `GATE_HOLD_STAGE_A.json`, `STAGE_B_2X2.json`, `PR_NOISE_FLOOR.json` | Stage A / B and the shuffled null |
| `LENGTH_AXIS_BY_SUBSET.json`, `DERIVATIVE_CHANNEL_EVAL.json` | the two length attacks |
| `READOUT_*.json`, `ADAPTIVE_READOUT.json`, `PER_CELL_K_HONESTY.json` | the readout work |
| `FEATURE_BEHAVIOUR.json` | the good-cell/bad-cell diagnostic |
| `figures/` | six SVG+PNG figures |
| `../chosen_token_calibration_v1/CT7_DEV_SCORES.npz` | frozen CT7, restored from Drive |
| `../localization_full_benchmark_v3/evaluation/` | `JOINED.json`, `JOINED.npz`, `FOLDS_V2.json` — the roster |

Raw pickles live in the **main checkout**'s `dataset_cache/` (this worktree has LFS
pointers by design). `spectral_utils/adaptive_step_readout_v1.py` has the running-mean
trick that yields every top-K at once — reuse it; `Δ` over a ladder wants the same
treatment.

---

## 5. Disciplines that are not optional here

- **Anchor first.** Every script so far refuses to print a new number until it replays a
  known one (C1 = 35.92 exactly; Step 420's five macro numbers within 0.0354 pp). This
  caught two real bugs. Do the same.
- **Pre-register.** `docs/experiments/TOKEN_PROBABILITY_FUSION_V1_STAGE_B_PREREGISTRATION.md`
  is the template, amendments included. Write the falsification condition *before* the run.
- **Paired source-group bootstrap, 10,000 draws.** Use `SB._group_draws` from
  `scripts/diagnostics/stage_b_2x2_v1.py`. I twice reported a point estimate as a gain
  before computing the interval and had to correct it in writing. Do not repeat that.
- **Label-using versus label-free, always separated.** A threshold chosen to maximise SLA
  is a ceiling, not a candidate — exactly like LOCO-5's 0.41. **The threshold rule is the
  hard part of this experiment, not the derivative.** Candidates: a within-answer quantile
  of Δ, a within-answer z-score, or the split-half stability criterion (which did have a
  genuine interior optimum for K).
- **Gate-free SLA is primary**, per subset, never only the mean — the mean is what hid the
  length asymmetry for this whole line.
- **Handling of clean answers.** Their SLA is on erroneous traces only. A first-crossing
  rule needs an explicit no-error decision before it can touch ProcessBench macro-F1;
  keep that separate from the locator, and remember `tail15` beats LOCO-5 for both
  locators and beats LOCO-5's own label-selected optimum by 0.48 pp.

## 6. Bugs I hit, so you do not

- `cell[3:-3]` maps both `pb_gsm8k_q4` and `pb_gsm8k_q8` to `gsm8k` — silently halves the
  population. Written twice; caught by an anchor both times.
- `hit[i]` being an `int` makes `x and not y` return a mix of `bool` and `int`; NumPy then
  infers an **integer** array and `ids[mask]` becomes fancy indexing by position, not a
  boolean mask. Use `dtype=bool` explicitly.
- `step_token_spans` are already 0-based **within** the answer. Do not subtract the
  answer's token offset.
- A pure-Python EMA over 7M tokens × 11 channels dominates everything; `scipy.signal.lfilter`
  is identical to 1.4e-16 and 14× faster.
- Create worktrees with `GIT_LFS_SKIP_SMUDGE=1` — otherwise each one materialises 27 GB.

## 7. Where I would start

1. Rebuild **their evidence series exactly**: `−H` of the renormalized top-20, from the
   cached top-50 log-probs. Sanity-check it against `q15_H1` (top-15) — they should be
   highly but not perfectly correlated.
2. Implement `Δ` = first differences of the EMA(span 5) of that series, and **choose the
   token→step collapse explicitly**, since the paper does not. Try at least the mean of
   the M=5 most negative Δ inside the step and the single most negative; report both,
   labelled as our choice.
3. Run the missing 2×2 cell — derivative **and** first-crossing together — beside all
   three cells already measured, on the same roster, with the C1 anchor in the table.
4. Only then vary the threshold rule, label-free, with the falsification condition
   written down first.

The prediction worth writing before the run: if the gain lives in the **pairing**, the
missing cell beats both single-axis failures *and* shows a ratio-to-chance that rises
with chain length the way theirs does. If it does not, the pairing hypothesis is wrong
and this line is finished.

---

## 8. AMENDMENT, 2026-09-19 — measurements that weaken §1's argument, and close CUSUM

Written after §1–§7, before any of it was acted on. Omri proposed detecting the first
error by running **CUSUM or BOCPD over the sequence of step scores** instead of taking an
argmax. Testing that idea produced three measurements that change what this handoff
should say.

### The step-score signal is an isolated impulse, not a level shift

Fused step score by offset from the true first error, answer-standardized SD:

| subset | −3 | −2 | −1 | **0** | +1 | +2 | +3 |
|---|---|---|---|---|---|---|---|
| GSM8K | −0.021 | 0.045 | −0.074 | **0.464** | −0.078 | −0.206 | −0.321 |
| OlympiadBench | 0.012 | 0.014 | 0.083 | **0.413** | −0.021 | −0.081 | −0.201 |
| Omni-MATH | −0.019 | 0.009 | 0.024 | **0.399** | 0.035 | −0.031 | −0.146 |

And the mean **after** the error is *below* the mean **before** it, by −0.19 to −0.35 SD in
every subset. The error step is a local maximum in only 52.7–57.1% of answers.

So there is no contaminated-reasoning plateau. The model becomes **more** confident after
it errs, not less.

### What that does to the sequential-detector idea

- **CUSUM is closed by this.** It exists to detect a sustained shift in mean. Against a
  one-step impulse there is nothing to accumulate, and accumulation only adds lag.
- **BOCPD is better matched but aims one step off.** There *is* a real distribution
  change — the post-error decline — but its boundary is the 0→1 transition, not the error
  step. That is precisely the convention the project's own audit warns about
  (`docs/reviews/bocpd_boundary_audit_2026-09-07.md`: distinguish a boundary before the
  current observation from one after it). If BOCPD is tried, the off-by-one must be
  handled deliberately and declared, not discovered.
- **Exploiting the shape directly also fails.** The obvious label-free way to use the
  post-error decline is to score a step by `v[t] − mean(v[t+1:])`. Piloted: 29.77 mean
  against argmax's 34.95. `v − next` 25.96, `v − mean(before)` 33.63,
  `0.5v − 0.5·mean(after)` 33.06. **Every shape rule loses.** The decline is a
  population-average property; within a single answer it is smaller than the noise, so
  subtracting an estimate of it costs more variance than it buys.

### The consequence for §1

**§1's argument is weaker than it reads.** For an isolated impulse in noise, argmax is
close to the matched-filter optimum — so "the decision rule is the untouched axis" is not
the strong claim §1 makes it. If the shape is an impulse, losing on long chains is just
impulse detection against more candidates, and no alternative decision rule over *these*
step scores repairs it. Three separate attempts now agree: first-crossing on the level
series, all five shape rules, and the sequential-detector family.

**What survives:** the missing 2×2 cell is still worth running, but the hypothesis moves
from the decision rule to the **statistic**. Their evidence series is the derivative of
renormalized top-20 entropy at *token* level; ours is a top-K level readout. If their
advantage on long chains is real, it most likely lives in the token-level series having
structure ours lacks — which their first-crossing rule can then exploit. Build their
statistic first and look at its shape around the true error **before** choosing a readout
for it. If that series also shows an isolated impulse, expect argmax to be near-optimal
there too, and this line is finished.

**Concretely, the first thing to run is now a diagnostic, not an arm:** rebuild their
evidence series, and plot its profile around the true error step exactly as above. That
one figure decides whether anything downstream is worth building.
