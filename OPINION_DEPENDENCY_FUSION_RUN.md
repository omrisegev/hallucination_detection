# Opinion — the dependency-aware fusion experiment, after running it on the data

**Author:** Claude Code (data machine, `local_cache/` present).
**Date:** 2026-08-05.
**Subject:** commit `1a2254b` — `SPEC_DEPENDENCY_FUSION_EXPERIMENT.md`, `spectral_utils/dependency_fusion.py`,
`spectral_utils/deem_adapter.py`, `scripts/run_dependency_fusion_experiment.py`.
**Audience:** the source that wrote that commit, and Omri, so both can decide.

**Nothing in the experiment was changed.** The registered configuration hash is `568dc60530928f54`
and the runbook in §12 was followed. This file is opinion + the run's outcome. `HISTORY.md` and
`PROGRESS.md` are deliberately **not** touched yet — that write-up should happen after whoever reads
this decides what the conclusion is.

---

> ## REVISED 2026-08-05 after independent review of commit `64f57cd`
>
> A reviewing model checked this file against remote `master` and raised seven objections. **Six
> are correct and I have retracted them below; the seventh is the important one and I settled it
> with a new measurement rather than an argument.** Read **§8 (corrections)** and **§9 (the clean
> 2×2)** before §0–§6, which are the original text and are now partly superseded.
>
> The headline changes. The original text concluded that dependency-aware weighting is harmful.
> The correct conclusion is narrower: **the full-inverse condition-100 ridge *solver* is harmful
> (−3.74pp with the covariance held fixed), and the dependency-structured covariance was never
> given a fair test** — under the paper's own two-component solver it is −0.37pp, median −0.00pp,
> p = 0.085, i.e. indistinguishable from inert. The reviewer's diagnosis was right and my
> attribution was wrong.

---

## 0. Executive summary

*(original text; superseded on the attribution question by §8–§9)*

The experiment is well built and it answers its own question cleanly. The answer is **no**.

- **The contribution fails its own gate, and fails it by sign, not by margin.** The
  dependency-weighted structured solve measured against the published sparse-error method it is
  built on is **−5.65pp**, 2 wins / 22 losses, Holm-adjusted **p = 1.8e−06**. That is a significant
  result in the *wrong direction*. All six §8 conditions fail.
- **Both pre-registered guards fire.** The ridge-only arm is **−2.24pp** (p = 6e−05), so the
  regularized solve is harmful on its own; the factorial interaction is **−3.41pp** (p = 2e−04), so
  the sparse structure makes the new weight rule *worse* than plain ridge. The pre-registered
  reading "if the ridge explains it, the contribution is regularization" does not even apply —
  neither ingredient helps.
- **The published sparse-error correction is one cell, not an effect.** Mean +1.26pp, but the
  **median is exactly 0.00pp**, Wilcoxon p = 0.73, and 19 of 24 cells move less than 0.1pp. Delete a
  single cell and it becomes −0.90pp.
- **The one thing that came out ahead is what we already deploy.** U-PCR with `sign(ρ̂)` orientation
  reproduced **bit-identically on all 24 cells** and remains the best arm.

Two things I think should change in the spec regardless of the verdict: the primary DEEM arm is
**collapsing** on real data, and the way a partial-seed collapse is handled biases its hypothesis
in DEEM's favour (§4). And the sparse-error mechanism is **inert exactly where its own theorem
applies** and violent only where the theorem is violated (§3) — which I think is the most
scientifically interesting thing the run produced, and it is not visible in the registered tables.

---

## 1. What was run

| stage | command | outcome |
|---|---|---|
| dataset-free gate | `python scripts/test_dependency_fusion.py` | **ALL PASSED** (22 checks, 6 groups) |
| validity gate | inside the runner | **GOOD_6 macro = 0.7733442** vs 0.7733 ± 0.002 → PASS |
| spectral arms, both arenas | `--data-dir local_cache --skip-deem` | 24/24 cells, **0 failures** |
| DEEM install | `pip install -e ".[dependency-experiment]"` | `deem 0.2.0` + `entmax 1.3`; torch 2.6.0+cpu untouched |
| DEEM arms, full arena | `--data-dir local_cache --device auto --arenas full` | **incomplete — see §4** |

DEEM was restricted to the `full` arena on purpose: no registered contrast reads a `keep.deem_*`
arm, so the keep-arena DEEM sweep is 180 CPU fits that no hypothesis tests. `--arenas` is not part
of the config hash, so this resumes into the same output directory with the same registered
configuration. This changes no hypothesis.

Artifacts: `results/dependency_fusion_study/`. **The committed `REPORT.md`, `per_cell.csv`,
`arm_summary.csv`, `contrasts.csv`, `summary.json` and `sparse_diagnostics.csv` are the
spectral-only snapshot**, written when the DEEM sweep had not yet run. The runner regenerates all of
them at the end of the DEEM pass, so they will change when it completes. `records.jsonl` (the
append-only checkpoint, 11 MB and growing while the DEEM sweep runs) is **not committed** — it holds
the raw per-sample score vectors, and every number quoted anywhere in this file is derivable from
the CSVs alone.

---

## 2. The result

One grid, all arms on the same 24 cells, higher AUROC is better.

| metric (24 in-scope cells) | deployed U-PCR + sign(ρ) | DUFS + L-SML | SU-PCR reproduction | published IU-PCR | ridge only, no sparse | SDSF (the contribution) |
|---|---:|---:|---:|---:|---:|---:|
| macro AUROC | **0.7741** ← best | 0.7687 | 0.7668 | 0.7542 | 0.7318 | 0.7104 |
| QA macro (9 cells) | **0.7585** | 0.7520 | 0.7347 | 0.7580 | 0.7316 | 0.6939 |
| math macro (15 cells) | 0.7834 | 0.7787 | **0.7861** | 0.7519 | 0.7319 | 0.7202 |
| paired mean vs deployed | — | −0.54pp | −0.73pp | −1.99pp | −4.23pp | −6.37pp |

Registered contrasts, deltas in AUROC percentage points, positive means the candidate wins:

| contrast (registered id) | mean | median | 95% CI over cells | QA / math | equal-dataset | W/L | p | Holm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse reliability correction (H1) | +1.26 | **+0.00** | [−1.78, +6.33] | −2.33 / +3.42 | −0.40 | 14/8 | 0.733 | 1 |
| **dependency-aware weights (H2)** | **−5.65** | −2.85 | [−10.19, −2.73] | −4.08 / −6.59 | −3.77 | **2/22** | 6.0e−07 | **1.8e−06** |
| ridge without sparse (A1) | −2.24 | −1.71 | [−3.41, −1.20] | −2.64 / −2.00 | −1.33 | 3/21 | 6.4e−05 | — |
| factorial interaction (A4) | −3.41 | −1.00 | [−8.07, −0.79] | −1.44 / −4.58 | −2.44 | 4/20 | 2.1e−04 | — |
| keep arena, sparse reliability (K1) | −0.07 | +0.00 | [−0.20, +0.01] | −0.01 / −0.10 | −0.02 | 8/10 | 0.446 | — |
| keep arena, dependency weights (K2) | −3.82 | −2.84 | [−5.53, −2.41] | −3.11 / −4.25 | −2.60 | **0/24** | 1.2e−07 | — |
| SDSF vs deployed U-PCR (P1) | −6.37 | −3.22 | [−11.03, −3.15] | −6.46 / −6.32 | −5.41 | 2/22 | 6.0e−07 | — |
| SDSF vs DUFS + L-SML (P3) | −5.83 | −2.76 | [−10.69, −2.58] | −5.81 / −5.84 | −5.06 | 3/21 | 3.0e−06 | — |

The contribution loses in both arenas, in both domains, on the cell macro and on the equal-dataset
macro, with the bootstrap interval entirely below zero. There is no slice of this in which it wins.
**I do not think this needs a re-run or a sensitivity sweep to be believed.** 0 wins on 24 cells in
the fixed-input arena is about as clean as a refutation gets in this project.

---

## 3. The finding I think is worth keeping

The sparse-error correction is **inert exactly where Tenzer's theorem applies, and violent only
where it is violated.** Measured in the primary arena:

| regime | n cells | sparse-reliability delta | dependency-weights delta |
|---|---:|---|---|
| Tenzer support condition `‖vec(S)‖₀ < (m−1)/2` **holds** | 21 | mean −0.01pp, median +0.00pp, p = 0.42 | mean −3.55pp, 2W/19L, p = 4.8e−06 |
| condition **fails** | 3 | mean +10.17pp, median −9.69pp | mean −20.35pp, 0W/3L |

Only 3 of 24 cells move more than 1pp, and they are precisely the three that fail the condition:

| cell | recovered support | IU-PCR | SU-PCR | delta |
|---|---:|---:|---:|---:|
| `math500_qwenmath7b` | 36 pairs | 0.4187 | 0.9300 | **+51.12pp** |
| `epr_triviaqa_mistral24b` | 48 pairs | 0.7433 | 0.6464 | −9.69pp |
| `se_nq_open_llama8b` | 44 pairs | 0.7394 | 0.6302 | −10.92pp |

(Secondary, post-hoc, computed after the registered numbers were frozen; it joins `per_cell.csv` to
`sparse_diagnostics.csv` and adds no new fit. It does not replace any registered row.)

Two readings follow, and I think both belong in the write-up:

**(a) The registered `threshold_multiplier = 1.0` produces almost no support on our covariances.**
Median sparse fraction is 0.0035 in the full arena and **exactly 0.0000** in the keep arena; the
support is empty on 10 of 24 cells. Where the support is empty, the arm labelled
`su_pcr_reproduction` is not testing sparsity at all — it is testing **rank-two completion of the
off-diagonal** against the raw additive solve. The spec's H1 therefore measures two different
mechanisms on two different subsets of cells and averages them. That is not a bug in the code; it is
a gap between what the arm is named and what it does on this data.

**(b) `math500_qwenmath7b` is a rescue of a broken baseline, not a gain.** Paper-faithful IU-PCR
scores **0.4187 — below chance** — on a cell where the deployed arm scores 0.9155 and DUFS + L-SML
scores 0.9178. The difference is the **exclusion step**, which IU-PCR omits by construction
(`exclusion=False` in `IU_PAPER_FIT`, correctly, since Eq. 15 is a plain two-component rule). So the
single largest number in the whole experiment is not evidence about sparse errors. It is evidence
that dropping weak views is what protects U-PCR on our data, and that the published two-component
PCR rule without exclusion is fragile enough to invert on at least one cell. I would put that in
`Research_Directions.md` under Extension J rather than in a sparse-error paragraph.

A third, smaller observation with no gate on it: the relative off-diagonal residual — the mass
explained by **neither** the rank-two part nor the sparse part — sits at 0.081–0.232 (median 0.153)
on every cell. The §8 convergence gate passed 24/24 in 3–11 iterations, i.e. vacuously. The gate
that would have been informative is on that residual, not on convergence.

---

## 4. What I think should change in the DEEM arm before anyone trusts H3

The DEEM sweep is still running (CPU-only, ~29 s per fit, 360 fits ≈ 8 h). Two problems are already
visible in the first two cells, and they are not runtime problems.

**The primary DEEM arm is collapsing.** `deem_deep_soft` — the arm that carries H3 — has returned a
**non-finite or constant score on 8 of its first 10 fits** (2 cells × 5 seeds): 4/5 seeds failed on
`epr_triviaqa_mistral24b` and 4/5 on `losnet_hotpotqa_mistral7b`. Over the same two cells the hard
arms completed **10/10** (`deem_irbm_hard`) and **10/10** (`deem_deep_hard`) without a single failure.
So the collapse is specific to the rank-pseudo-probability input — the arm the spec argues *for* —
and it is not an occasional bad seed. At this rate no cell will have all five soft seeds succeed,
which means **H3 may end up with an empty candidate set** while its two ablations (A2, A3) survive on
the hard arms.

**A partial-seed collapse silently removes the cell from H3 instead of scoring against it.**
`collect_scores` builds the seed ensemble only when every registered seed succeeded:

```python
records = [store.latest.get((cell_key, arena, arm, int(seed))) for seed in seeds]
if not records or any(r is None or r.get("status") != "ok" for r in records):
    continue
```

`paired_stats` then intersects the reference and candidate cell sets. The consequence: a cell where
DEEM collapses on one seed **disappears from H3's paired comparison** rather than being recorded as a
loss. Collapse is not random with respect to the data, so H3 will be computed on the cells where
DEEM happened to fit — **survivorship bias in DEEM's favour**, in the direction of declaring the
external baseline better than it is. The fixed Holm family of three correctly protects against a
*total* DEEM crash; it does not protect against this partial one.

I did not change it, per the instruction not to. What I would do:

1. Report `n` per contrast prominently — it is already in `contrasts.csv`, but `REPORT.md`'s contrast
   table omits it, so a shrunken H3 is invisible in the human-readable output.
2. Decide the rule **before** the numbers land: either a collapsed seed is a **loss** (score = the
   arm's own constant, AUROC 0.5), or the whole cell is excluded from **all** arms in that contrast,
   not just the DEEM one. Silently intersecting is the one option that biases.
3. If the collapse is widespread, H3 should be reported as "the arm did not fit" — which is a real,
   publishable finding about DEEM on continuous detector ranks — and not as a macro AUROC over the
   survivors.

Related, and predicted before the run: `continuous_to_deem_hard` median-splits every view, forcing
each voter to a 50% predicted-positive rate while true positive rates across our cells span roughly
20–85%. That is a systematic mismatch with the Dawid–Skene premise that a voter's marginal reflects
the class prior. If the soft−hard ablation (A2) comes out positive, that base-rate distortion is a
competing explanation for it, distinct from "continuous rank information helps."

---

## 5. Code-level notes (none of these blocked the run)

Verified working, and worth saying explicitly because they are the parts that are easy to get wrong:

- **The label seam is real.** `evaluate_score` is the only function that receives labels; the
  estimator signatures carry no label parameter and the dataset-free gate asserts it; DEEM's
  Hungarian map aligns to majority vote, not correctness.
- **The reliability contrast is a genuinely clean single factor.** I traced
  `dependency_fusion._estimate_g2_and_rho` against `upcr._fit_block`: identical `g2` grid, identical
  eigen-projection rule, identical two-component PCR weights on the observed covariance. The one and
  only difference is `b` — cleaned `low_rank[i,j]` instead of observed `C[i,j]`. Good design.
- **The deployed reference is exact.** `deployed.upcr_signrho` matched
  `results/upcr_study/06_orientation/per_cell.csv` on all 24 cells, **max |diff| = 0.00e+00**. The
  missing reproduction assert (below) would have passed.

Things I would fix, in descending order of how much they could bite:

1. **`load_dufs_choices()` has no reproduction gate.** It is a last-row-wins dict over a 51-row CSV
   covering 24 cells. `labelfree_standing_report.py` builds the same dict and then asserts the
   rescored AUROC matches the recorded value to 5e−4; this runner does not. Per the known staleness
   pattern, a re-graded cell leaves stale rows in exactly this kind of resume-safe CSV, so the P3
   reference arm could silently be a stale row. Low stakes here — P3 is a practical reference, not a
   hypothesis — but the assert costs four lines.
2. **`--device` is inside the config hash.** `asdict(replace(DEEM_BASE, device=args.device))` feeds
   `config_hash`, so §12's own suggested workflow — spectral arms on a CPU machine, then the complete
   command on a GPU machine — only resumes if both invocations pass the *same literal string*.
   `--device auto` in both works, which is what §12 writes; `--skip-deem` followed by `--device cpu`
   is refused as a configuration mismatch. Either resolve the device before hashing, or exclude it
   from the hash: it is an execution detail, not a registered method parameter.
3. **A third copy of the cell loader.** `run_dependency_fusion_experiment.load_cells()` is
   `compare_anchor_quality.load_all_inscope_cells()` with a `data_dir` argument, rather than a call
   into `inscope_bench_common.load_cells()`. It is functionally identical — the GOOD_6 gate returns
   0.7733442, so it is provably the canonical data — and the gate is exactly the guard for this. But
   `inscope_bench_common.py`'s own header documents three scripts that each rolled their own loader
   and scored a whole Extension against a mis-computed 0.7273 baseline. Give the canonical function a
   `data_dir` argument instead.
4. **`target_condition = 100` is the least-justified registered constant.** §9 fixes it honestly and
   it is not tuned on AUROC, but it sets the ridge directly, and on every cell the solve lands
   *exactly* at the cap with γ ≈ 0.10–0.13 against a unit diagonal — a near-constant ~10% shrinkage
   everywhere. There is no dataset-free argument for 100 over 30 or 300. Normally this would make a
   null result confounded with a bad γ, and I would ask for the §9 sensitivity sweep in a separate
   output directory. **Here it does not matter**: the ridge-only arm is independently harmful
   (−2.24pp) and the interaction is negative (−3.41pp), so both ingredients are working against the
   score, not just their combination at one γ. I would not spend GPU-hours on the sweep.
5. **The `_seed0` arms are computed but no contrast reads them.** `full.deem_deep_soft_seed0` exists
   in `arm_summary.csv`; nothing tests it. Since H3 gives DEEM a five-seed ensemble against a single
   deterministic fit, the ensemble's variance reduction is part of the candidate, and the one
   contrast that would separate "ensembling" from "nonlinearity" is absent. It is a one-line addition
   to `contrast_specs` and it costs no new fits.
6. **The planted-world gate passes with 2 of 3 edges.** `test_sparse_support` asserts
   `overlap >= 2/3` and gets exactly 0.667 — it misses the smallest planted edge (0.40). That is the
   same threshold conservatism that produces empty supports on real data, so the gate is not
   mis-calibrated so much as it is documenting the behaviour at its own pass line. Worth a comment in
   the test so the next reader does not take 2/3 as headroom.

---

## 6. My opinion on what this closes and what it does not

**Closed, and I would state it plainly:** *replacing U-PCR's final projection with a structured,
condition-controlled covariance solve.* It is not neutral, it is actively harmful — −5.65pp against
the method it extends, −6.37pp against what we deploy, 0 wins on 24 cells under fixed inputs. This
is the strongest negative in the weights channel so far, and unlike the feature-selection results it
does not need a matched floor to interpret: the reference *is* the published method on identical
inputs.

**Closed, more weakly:** *the published sparse-error correction as a drop-in for our data.* Its
median effect is exactly zero, its support is empty on 10 of 24 cells at the paper's own threshold,
and its one large positive is a broken-baseline rescue. I would not implement SU-PCR on the strength
of this. But I would not call the *idea* dead either — see below.

**Not closed, and I think this is where the value is.** The run's most informative output is that
15% of the off-diagonal covariance mass is explained by neither a rank-two structure nor a sparse
correction, uniformly across all 24 cells. That is a stable, measured statement that the additive
one-factor premise is wrong in a *specific, quantified* way — and it is not a statement any of the
three registered hypotheses were built to test. `HANDOFF_FEATURE_SELECTION_AND_FUSE.md` §6 recorded
the sparse-Δ relaxation as the response to that misfit; this run says sparse-Δ is the wrong
relaxation, not that the misfit is unimportant. The residual is too large and too consistent to be
noise, and it is structured enough that a *dense low-rank* correction beyond rank two, rather than a
sparse one, is the reading I would pursue next.

This is exactly the "tailor, do not transplant" situation, and I want to be careful about which way
it cuts. The commit did tailor — SDSF is not a transplant, it keeps the paper's decomposition and
changes only the final weight equation, which is a real reshaping and the right instinct. It still
lost. So the lesson is not "we transplanted again." It is that **the weights channel took the
reshaped idea and still said no**, which is a more expensive and more informative no than Step 224's.

**What I would not do:** run the γ sensitivity sweep, retry SDSF with a different rank, or vary the
sparse threshold to find a support size that helps. All three are searches for a configuration that
scores, on a mechanism whose two independent ingredients are each independently negative. That is the
shape of the trap the repo has already documented twice.

**What I would do, in order:**
1. Let the DEEM sweep finish or kill it, but **do not report H3 as a macro over surviving cells**
   until the seed-collapse rule in §4 is decided. If soft-input DEEM collapses broadly, "the modern
   nonlinear baseline does not fit our continuous detector ranks" is a legitimate result and cheaper
   to defend than a partial macro.
2. Write the exclusion-step finding into Extension J. IU-PCR at 0.4187 on a cell where the deployed
   arm gets 0.9155 is the most actionable thing in the run and it is currently buried inside an H1
   cell delta.
3. Treat the 15% unexplained off-diagonal mass as the next question, and specify what structure it
   has *before* proposing an estimator for it — one variant, one discussion, then build.

---

## 7. Answering the obvious question: are all the run's results documented?

Yes, for everything that has finished. Every number in this file is derivable from the committed
CSVs:

| file | what it fixes |
|---|---|
| `results/dependency_fusion_study/run_config.json` | the registered configuration + hash `568dc60530928f54` + SHA-256 of the four scientific source files |
| `results/dependency_fusion_study/per_cell.csv` | every arm × every cell AUROC — the source of §2, §3 and the leave-one-out check |
| `results/dependency_fusion_study/arm_summary.csv` | macro / QA / math per arm, median fit seconds, failure counts |
| `results/dependency_fusion_study/contrasts.csv` | every registered contrast with mean, median, CI, per-domain, equal-dataset, W/L, p, Holm, and the per-cell delta list |
| `results/dependency_fusion_study/sparse_diagnostics.csv` | support size, sparse fraction, residual, convergence, threshold, µ, theorem flag per cell per arena — the source of §3 |
| `results/dependency_fusion_study/summary.json` | all of the above plus the failure records |
| `results/dependency_fusion_study/REPORT.md` | the runner's own generated human-readable table |

**Superseded by §11:** `records.jsonl` was withheld from the first commit as a live 11 MB file. It is
now pushed as `results/dependency_fusion_raw/records.jsonl.gz`, together with the per-cell feature
matrices, so nothing has to be taken on trust.

Still outstanding: the DEEM arms (H3, A2, A3, P2). They are **not** in the committed tables. The
snapshot committed here is spectral-only, and the runner will regenerate every table when the DEEM
pass completes.

---

## 8. Corrections after independent review

Six objections were raised against §0–§6 and I accept all six. Where I could check a claim
numerically I did, and the check is named.

**8.1 "All six §8 conditions fail" — wrong. Four of six fail.** Audited condition by condition:

| §8 condition | value | verdict |
|---|---|---|
| 1. mean gain ≥ +1.0pp | −5.65pp | fails |
| 2. bootstrap 95% lower bound > 0 | −10.19pp | fails |
| 3. Holm-adjusted p < 0.05 | 1.8e−06 | **passes as written** |
| 4. neither QA nor math below −0.5pp | −4.08 / −6.59 | fails |
| 5. equal-dataset macro positive | −3.77pp | fails |
| 6. ≥ 90% decompositions converge | 24/24 | **passes** |

The reviewer's framing is exactly right: the Holm p-value is significant *evidence of harm*, not of
benefit. **This is also a defect in the spec, not only in my prose:** condition 3 is written as a
two-sided Wilcoxon p-value inside a one-sided advancement gate, so an arm that loses catastrophically
satisfies it. §8 should read "Holm-adjusted p < 0.05 **and** the mean delta is positive", or use a
one-sided test. As written, four of the six conditions carry the whole gate.

**8.2 The theorem flag tests the recovered support, not the population support — my framing was
overstated.** `theorem_support_ok` is computed from `nnz` of the *estimated* sparse matrix, whereas
Tenzer's uniqueness condition constrains the unknown true `S`. So "the mechanism is inert wherever
its own theorem applies" is not justified. The defensible statement is: **the mechanism is inert on
the 21 cells where the recovered support is sparse** (−0.01pp, median +0.00pp, p = 0.42) and moves
large amounts only on the 3 cells where the recovered support is dense. The empirical pattern stands;
the appeal to the theorem does not.

**8.3 The MATH500 rescue is not attributable to exclusion — retracted.** `full.iu_pcr` and
`full.su_pcr_reproduction` use the identical 28-feature set with `exclusion=False` in both. The
sparse correction alone moves that cell 0.4187 → 0.9300. My sentence "the exclusion step, not sparse
modelling, is what protects U-PCR on our data" conflated two separate facts and got the attribution
of the H1 delta wrong. What survives is narrower and still worth recording: **two independent
interventions each repair the same pathological cell** — the sparse correction (0.9300) and the
deployed arm's exclusion step (0.9155) — which says plain two-component IU-PCR is fragile enough to
invert below chance there, not that sparsity is what fixed it.

**8.4 "15% unexplained covariance mass" is not what that number is — retracted.**
`relative_residual` is `‖residual[triu]‖_F / ‖observed[triu]‖_F`, a ratio of Frobenius norms over
off-diagonal entries. It is neither variance-explained nor a fraction of covariance mass, and it
pools model mismatch, finite-sample noise, and approximation error from the custom decomposition. The
correct phrasing is "the off-diagonal residual is 8–23% of the off-diagonal Frobenius norm
(median 15%)", and it does not license the interpretation I put on it.

**8.5 "A dense higher-rank correction is the next thing to pursue" — retracted, and now measured
against.** I flagged it as a hypothesis but should not have promoted it. It is also wrong in the
direction the data can see: the residual's **top-5 eigenvalues carry only 55% of its spectral mass**
(median over 24 cells, out of ~28 directions).

> **CORRECTION (added after review).** *"The leftover is diffuse, not low-rank"* is **retracted as
> unsupported**. 55% over 5 of ~28 directions is **3× the uniform share (17.86%)**, so the number I
> quoted points the opposite way from the word I attached to it. Without a null distribution,
> "diffuse" was an impression. A share statistic also cannot see magnitude — a tiny residual can have
> a concentrated spectrum — so the operator and Frobenius norms have to be tested too. The falsifiable
> version is `SPEC_SOLVER_MECHANISM_STUDY.md` §4: latent-preserving nulls, B = 1000, the complete
> decomposition pipeline refit on every draw. Until that runs, the shape of the residual is unknown.

**8.6 The DEEM base-rate confound — retracted.** `continuous_to_deem_soft` maps to
`(rank − 0.5)/n`, whose per-feature mean is ≈ 0.5, so the soft arm has balanced marginals too.
Balanced marginals are common to both arms and are not a competing explanation for A2; the reviewer
is right that A2 does mostly measure hard-versus-continuous information. My §4 paragraph on this is
void.

**8.7 The CRLF configuration-hash issue — confirmed, and it is theirs, not mine.** I verified it:
all four entries in `run_config.json`'s `source_sha256` match the **CRLF working-tree** bytes, not the
LF git blobs. So the registered configuration hash is line-ending dependent and an otherwise
identical Linux or macOS checkout would be **refused as a configuration mismatch** by
`main()`'s hash check. Their proposed fix — record the git commit, dirty status, input-cache hashes,
the DUFS-choice-file hash, and *normalized* source hashes — is the right one.

Two of their side-notes I also checked: **H2 without the extreme cell is −3.64pp** (their figure,
confirmed to 2 d.p.), and their independent verification that **DUFS + L-SML also reproduces on
24/24 cells** closes §5 item 1 empirically for this run, though the missing assert is still worth
adding.

On provenance: the attribution is deliberate, not incidental. The commit carries
`Co-Authored-By: Claude Opus 5` by this repo's standing convention and this file names its author in
its own header. Nothing is being concealed and nothing should be.

---

## 9. The clean 2×2: it is the solver that failed, not the dependency structure

This was the reviewer's central objection and it is correct. The registered comparison changes **two**
things at once, and the registered ablation does not separate them:

| arm | reliability ρ | solver | matrix |
|---|---|---|---|
| `su_pcr_reproduction` | SU | 2-component PCR | observed `C` |
| `sdsf` | SU | condition-100 ridge | **structured** `C` |
| `iu_ridge` | **IU** | condition-100 ridge | observed `C` |

So H2 = solver + matrix, while the ablation A1 changes solver *and* reliability estimate on the
observed matrix. The registered interaction term therefore cannot support the claim I made — "sparse
structure makes the ridge weight rule worse" — because it conflates the covariance substitution with
the solver substitution.

I ran the missing decomposition: hold ρ fixed at the SU estimate for all four arms and cross the two
remaining factors. `scripts/dependency_fusion_solver_matrix_diagnostic.py`, output in
`results/dependency_fusion_solver_matrix/`. **Wiring gate first:** the two anchor cells of the square
reproduce their committed per-cell AUROC to **1e−9 on all 24 cells**, so this is measuring the same
objects the registered run measured.

Macro AUROC, ρ held fixed at the SU estimate:

| | observed `C` | structured `C` |
|---|---:|---:|
| **2-component PCR** | 0.7668 *(= `su_pcr_reproduction`)* | 0.7631 |
| **condition-100 ridge** | 0.7294 | 0.7104 *(= `sdsf`)* |

| effect | mean | median | W/L | p |
|---|---:|---:|---:|---:|
| **solver**, PCR → ridge, observed `C` | **−3.74pp** | −2.21 | 2/22 | 6.0e−07 |
| **solver**, PCR → ridge, structured `C` | **−5.28pp** | −2.59 | 2/22 | 6.0e−07 |
| **matrix**, observed → structured, PCR | **−0.37pp** | **−0.00** | 7/14 | **0.085** |
| **matrix**, observed → structured, ridge | −1.90pp | −0.78 | 4/20 | 7.6e−05 |
| registered H2 (both at once) | −5.65pp | −2.85 | 2/22 | 6.0e−07 |

**The solver carries the loss.** Substituting the dependency-structured covariance under the paper's
own two-component rule costs −0.37pp with a median of exactly zero and p = 0.085 — the same
"statistically indistinguishable from inert" signature as H1. Swapping the solver costs −3.74pp with
the matrix held fixed.

The mechanism is the one the reviewer proposed, and it is now measured:

| diagnostic (median over 24 cells) | 2-component PCR | condition-100 ridge |
|---|---:|---:|
| share of ‖w‖² in the top-2 eigendirections, observed `C` | 1.0000 | **0.1627** |
| share of ‖w‖² in the top-2 eigendirections, structured `C` | 1.0000 | 0.2298 |
| split-half weight-vector agreement, ‖cos‖, median | 0.9896 | 0.9295 |
| split-half weight-vector agreement, ‖cos‖, **worst cell** | 0.3863 | **0.0066** |

PCR puts all of its weight in the leading two directions by construction. A condition cap of 100
still admits all 20–30 directions, so the ridge puts roughly **80% of its weight into low-eigenvalue
directions**, and on the worst cell the ridge weight vector fitted on even samples is **essentially
orthogonal to the one fitted on odd samples** (|cos| = 0.007, against 0.39 for PCR).

> **CORRECTION (added after review).** The sentence that followed — *"PCR's hard truncation is doing
> variance control, and removing it is what destroys the score"* — is **retracted**. It does not
> survive the cross-cell test. Correlating the per-cell ridge−PCR AUROC delta against the ridge's own
> split-half weight stability gives **ρ = −0.048, p = 0.82**: cells where the ridge weight vector is
> unstable are *not* the cells where the ridge loses. The two correlations that are strong —
> score agreement (+0.803) and ridge top-2 concentration (+0.735) — are partly tautological (if two
> scores agree their AUROCs agree; concentration measures similarity to PCR), so neither is
> independent evidence for a mechanism. **The mechanism is unidentified**, which is why the
> head-scaling × ridge-tail factorial in `SPEC_SOLVER_MECHANISM_STUDY.md` is an intervention rather
> than another correlation.

> **CORRECTION (added after review).** What stood here were two claims that the reviewer's
> speculations about PSD repair were "refuted". **Both were wrong**, and the reviewer was right:
>
> - **PSD projection does clip the structured matrix.** Measured over the shipped per-cell columns:
>   **6 of 24 cells, 8 negative eigenvalues, maximum distortion `‖PSD(C)−C‖/‖C‖ = 2.096%`**, with
>   minimum eigenvalues −0.08 to −0.29 against a unit diagonal — `epr_triviaqa_mistral24b`,
>   `se_nq_open_llama8b`, `truthfulqa_llama8b`, `ars_gsm8k_r1distill8b`, `noise_gsm8k_phi3mini`,
>   `math500_qwenmath7b`. My "zero clipped, 0.0000 distortion" is true of the **observed** covariance
>   only (0 clipped, distortion ≤ 2.7e-15, all 24 cells), and I generalized it to both matrices.
> - **The conditioning claim is a median that inverts on exactly those 6 cells.** "Better conditioned,
>   9.4e2 versus 5.2e5" holds at the median; on the clipped cells the projection drives the condition
>   number to 4e16–3e18. And the column name `cond_raw_*` is a misnomer — it reports `cond(PSD(C))`,
>   not `cond(C)`.
>
> This matters beyond the correction: the 6 cells needing PSD repair are almost exactly the cells
> where the sparse mechanism is live, plus the +51pp H1 outlier — so the PSD step, not the dependency
> model, may own most of the structured-matrix loss. Which of the two it is has a preregistered
> three-way test in `SPEC_SOLVER_MECHANISM_STUDY.md` §3c, and the direction I implied there was also
> backwards: if PSD-projected structured PCR *recovers* observed PCR, the projection **repairs** the
> loss and raw indefiniteness caused it.
>
> The reviewer could only find this because the raw bundle shipped the per-cell columns. The export
> did its job.

### What this does to the conclusion

**Closed as a result, open as a mechanism:** replacing U-PCR's two-component truncation with a
full-inverse, condition-100 ridge solve costs **−3.74pp** with the covariance held fixed, 2W/22L,
p = 6e-7. The loss is not in doubt.

> **CORRECTION (added after review).** I wrote that a mechanism — "variance in the discarded
> directions" — explains it, and that I would not revisit it. The first half is retracted (see the
> correction above: ρ = −0.048, p = 0.82 against ridge stability). The second half stands for a **γ
> sweep**, which is still not worth running, but not for the decomposition: the ridge changes two
> things at once — it rescales the top-two coefficients *and* it admits the low-eigenvalue tail — and
> nothing published here separates them. The factorial that does is preregistered.

**Not closed, and I was wrong to imply it was:** dependency-aware fusion. The structured covariance
has only ever been evaluated *through* a solver that independently costs 3.74pp. On the paper's own
solver it is −0.37pp with a zero median. That is not evidence against dependency modelling; it is an
absence of evidence either way, and it has the same signature as the sparse-reliability arm — a
mechanism that changes almost nothing on this data at the registered threshold.

**Which means the interesting question is now the one the reviewer's 2×2 exposes:** the dependency
structure is nearly inert under the paper's own solver, and the residual it leaves behind has an
**untested** shape.

> **CORRECTION (added after review).** The original sentence continued "…and the residual it leaves
> behind is diffuse (top-5 share 0.55), not low-rank. Both of those point the same way — that the
> off-diagonal misfit is not concentrated in a form any of these three corrections is shaped to
> capture." That inference is **retracted**: it rests on the "diffuse" reading retracted in §8.5, and
> 0.55 over 5 of ~28 directions is 3× uniform. The honest statement is that the dependency-structured
> covariance is **undemonstrated rather than disproved** — it has only ever been evaluated through a
> solver that independently costs 3.74pp — and that whether its residual carries real structure is
> now a preregistered test with an explicit abandonment condition, not an impression.

**Still not worth doing:** a γ sweep, a rank sweep, or a sparse-threshold sweep. The solver's failure
mode is variance in admitted directions and the matrix's effect is null under the good solver;
neither is a configuration-search problem.

---

## 10. DEEM: interim evidence, and what the reviewer asked for

The reviewer could not verify the collapse claim because `records.jsonl` is not pushed. Fair. Interim
export: `results/dependency_fusion_study/deem_interim_seed_records.csv` — every DEEM arm/seed record
written so far, with status, runtime, error, and for successful fits the score standard deviation,
unique-value count, class map and history keys.

State at the time of writing (the sweep is still running, 4 of 24 cells touched):

| arm | ok | failed |
|---|---:|---:|
| `deem_irbm_hard` | 20 | 0 |
| `deem_deep_hard` | 15 | 0 |
| `deem_deep_soft` | **2** | **13** |

Per cell, the soft arm is 1/5, 1/5 and **0/5** on the three completed cells. **No cell has all five
soft seeds succeed**, and `collect_scores` requires all five to build the ensemble — so on current
evidence H3's candidate set will be **empty**, not merely reduced. That is a stronger version of the
§4 warning: the survivorship-bias concern becomes moot if nothing survives, and the finding becomes
"the arm did not fit."

One artifact defect this exposed, which supports the reviewer's request: **the failure path discards
the training history.** `save_method_record` catches the exception raised by `orient_score` *after*
`score_fn()` has already computed DEEM's history dict, and the except branch stores only
`error_type`, `error` and `traceback`. So for exactly the fits we most want to diagnose, the loss
curve is thrown away. Retaining `dynamic_diag` on failure is a small change and it is the one I would
make before re-running DEEM.

Their remaining requests — regenerated tables, `deem_seeds.csv`, `deem_seed_summary.csv`, and
per-contrast `n` in `REPORT.md` — need the sweep to finish. I will push them when it does. Adding
per-contrast `n` to the report table is a one-line change to `build_reports`; `contrasts.csv` already
carries it. `records.jsonl.gz` is pushed now, as an interim snapshot — see §11.

---

## 11. Raw data, so nothing has to be taken on trust

The reviewer asked for the raw data to review independently. Pushed as
`results/dependency_fusion_raw/`, written by `scripts/export_dependency_fusion_raw.py`:

| file | what it lets you do |
|---|---|
| `cells.npz` | re-derive **every arm from scratch** — per cell the canonical z-scored `V`, the `sign(ρ̂)`-oriented `F` that every arm consumes, the hand signs, the derived polarity, the anchor, the labels, and the pool names |
| `cells_manifest.csv` | n, m, positive rate, GOOD_6 coverage, anchor name per cell |
| `records.jsonl.gz` | the runner's append-only checkpoint verbatim — every arm/seed frozen score vector and diagnostic, **including the failures** |
| `deem_interim_seed_records.csv` | flat DEEM per-seed table with `score_std` and `score_n_unique`, which is how a collapsed fit is identified |
| `RAW_DATA_README.md` | schemas, array conventions, and a runnable snippet that reproduces the GOOD_6 constant and two registered arms from the bundle alone |

Three things about it worth knowing before using it:

- **It is self-validating.** The README's snippet must print GOOD_6 = **0.7733442**. If it does not,
  the bundle is not the data the committed numbers came from and nothing downstream should be
  trusted. That is the project's standing rule and it applies to me as much as to anyone.
- **`V` and `F` are float64 verbatim, not downcast.** This review turned on agreement at 1e−9, so
  approximate reproduction would not have been good enough.
- **`records.jsonl.gz` is an interim snapshot** — the DEEM sweep was still running when it was
  written, so its per-arm counts are a point in time, not the final tally.

Scope: derived spectral features and correctness labels only. No prompts, no generated text, no model
weights — the same scope as the result CSVs the repo already tracks, and it makes the Step-225 goal
(a repo-only machine can re-derive every headline number) true for this experiment too.

One limitation I cannot export around, and it is the runner's, not the bundle's: **the failure path
discards DEEM's training history.** `save_method_record`'s `except` branch keeps only `error_type`,
`error` and `traceback`, and the exception is raised by `orient_score` *after* `score_fn()` has
already computed the history dict. So for exactly the fits worth diagnosing, the loss curve is
thrown away. Retaining `dynamic_diag` on failure is the change to make before DEEM is re-run.

---

## 12. Solver mechanism — the ridge loses entirely on the tail, and it is not a sample-size problem

Preregistered in `SPEC_SOLVER_MECHANISM_STUDY.md` §3 before any number below was read; run by
`scripts/solver_mechanism_study.py`; raw output in `results/solver_mechanism/`.

The reviewer rejected my first attempt at this decomposition because two of the proposed arms were
the same vector. The replacement is a factorial that crosses what a ridge actually does — rescale the
top-two coefficients, and admit the low-eigenvalue tail:

|  | tail absent | + `t_ridge` |
|---|---:|---:|
| **PCR head scaling** | 0.7668 *(= committed `su_pcr_reproduction`)* | 0.7297 |
| **ridge head scaling** | 0.7670 | 0.7294 *(= committed `ridge_observed`)* |

Two corners are committed arms, so the wiring gate is free; both reproduce their committed per-cell
AUROC to 1e-9 on all 24 cells, and `h_ridge + t_ridge` equals the registered ridge solution to 1e-10
relative on all cells at all five κ.

| effect (AUROC points, family-blocked 95% CI) | mean | CI | W/L | p |
|---|---:|---|---|---:|
| **head rescaling, tail absent** | **+0.02** | [−0.00, +0.06] | 10/9 | 0.904 |
| head rescaling, tail present | −0.03 | [−0.04, −0.01] | 4/19 | 6.6e-4 |
| **tail addition at the PCR head** | **−3.71** | [−5.36, −1.76] | 2/22 | 6.0e-7 |
| tail addition at the ridge head | −3.76 | [−5.40, −1.83] | 2/22 | 6.0e-7 |
| solver leg, both at once | −3.74 | [−5.38, −1.86] | 2/22 | 6.0e-7 |
| interaction | −0.05 | [−0.08, −0.02] | 3/20 | 7.7e-5 |

**The whole −3.74pp solver loss is admitting the tail. Rescaling the top-two coefficients is free.**
That is a sharper statement than "the full-inverse solver failed", and it is the statement the
earlier design could not have produced.

**The κ path agrees.** Family-weighted slope of the tail effect on log κ = **−0.745 pp per log κ**,
95% CI [−1.250, −0.269] — negative and excluding zero, which is the direction preregistered for
low-eigenvalue amplification being causal. The head-rescaling slope is −0.144, an order of magnitude
smaller.

**And it is not finite-sample noise.** Repeated unlabeled train/test at 25/50/75% (50 repetitions,
weights *and* the anchor flip frozen on train, AUROC on held-out samples only):

| train fraction | held-out ridge−PCR gap | tail cosine across repeats | tail norm CV | top-2 subspace angle |
|---|---:|---:|---:|---:|
| 0.25 | −4.40pp | 0.644 | 0.397 | 19.7° |
| 0.50 | −4.23pp | 0.855 | 0.241 | 10.5° |
| 0.75 | −4.21pp | 0.967 | 0.127 | 5.4° |

The tail estimate stabilises sharply — by 75% it is essentially reproducible across repeated splits —
and **the gap does not move**. By the preregistered reading that is *structural model mismatch*, not
estimation variance. Tripling the data does not rescue the full inverse, so a better-regularised
version of the same idea is not the fix.

**The PSD question, answered, and the answer is not the one I implied.** Raw and PSD-projected
structured PCR give **bit-identical AUROC on all 24 cells** (0W/0L): `_pcr_weights` takes the top-two
*algebraic* eigenvalues, and clipping only touches negative ones, so the PCR arm never sees the
repair. By the corrected three-way rule this is "raw ≈ PSD, both below observed ⇒ the structured
estimator causes the loss". PSD repair is irrelevant to the PCR arm — my worry that it owned most of
the structured-matrix loss is dead.

What indefiniteness does mark is *where* the structured estimator is worst: on the 6 clipped cells the
matrix leg is **−1.45pp** (0.7185 → 0.7040) against **−0.37pp** over all 24. So the
dependency-structured covariance is not uniformly inert; it is inert on 18 cells and mildly harmful on
the 6 where its own estimate goes indefinite.

## 13. Residual identifiability — a large residual, and a decision rule of mine that does not measure it

Preregistered in `SPEC_SOLVER_MECHANISM_STUDY.md` §4; run by
`scripts/residual_identifiability_study.py`; raw output in `results/residual_identifiability/`.
B = 1000 draws per cell per null, each put through the complete decomposition pipeline, every null
sample re-standardised with `prepare_cell`'s convention and gated at `max|diag(C*) − 1| ≤ 1e-8`.

**The magnitude result is unambiguous.** Global primary endpoint T = **+64.0**, p = **0.000999** (the
B = 1000 floor). All eight dataset families sit at the floor under BH q = 0.10, and
leave-one-family-out never lifts the worst p above the floor, so this is not gsm8k or math500 driving
it. The observed residual operator norm is **2–20× the independent-error null** on every cell. There
is a great deal of off-diagonal structure that rank-two-plus-sparse does not capture.

**The registered verdict is nevertheless FAILURE, and I do not think it means what it says.** The
decision required ≥ 5 of 8 families to pass a three-part stability gate; 1 did. But two of the three
criteria I preregistered — support Jaccard ≥ 0.50 and edge-sign agreement ≥ 0.80 — are properties of
the **sparse component S**, not of the **residual R** this study is about. And S is **empty on 9 of 24
cells** and has ≤ 7 edges on 8 more, so its Jaccard is undefined (NaN, which the gate counts as a
failure) or trivially near zero. The one criterion that *is* about R — the principal angle between
residual subspaces across 50 deterministic split halves — **passes on all 24 cells**: median 34.0°,
worst 59.6°, threshold 60°. Edge-sign agreement is 1.000 wherever it is defined at all.

So the honest report is: *by the rule I committed, the study fails; the rule is partly measuring the
wrong object.* I am not rewriting a preregistration after watching it fail — that is what
preregistration is for. **This needs the reviewer's decision**: either re-specify the gate on R alone
and re-run, or let the FAILURE stand. What should not happen is quoting "the residual is 2–20× the
null" as a positive result while the committed rule says otherwise.

One caveat I would attach either way: T = +64 is an enormous effect, and effects that large usually
mean the null is easier than the observation for some reason beyond the hypothesis. Null (a) preserves
the fitted latent signal and every error marginal, so the comparison is the right one in principle —
but the size of the gap deserves a sceptical second look before anyone builds on it.

## 14. DEEM — the soft arm did not fit, and a single label-free configuration change fixes it

Preregistered in `SPEC_SOLVER_MECHANISM_STUDY.md` §5; run after the registered sweep exited, by
`scripts/deem_soft_collapse_probe.py` and `scripts/deem_winner_validation.py`; output in
`results/deem_probe/`.

Final sweep tally for `deem_deep_soft`: **108 failed / 7 "ok" of 115**. All seven non-failures are
**collapsed** by the preregistered `sd < 1e-6` rule (σ from 8.9e-11 to 4.3e-07) — a definition
calibrated in advance so that the three then-visible "successes" at σ ≈ 1e-8 would not count as
successes. **No cell has all five soft seeds succeed**, so H3's candidate set is empty. That is a
finding — the arm did not fit — not a gap in the experiment.

The runner discards the evidence: the exception is raised by `orient_score` *after* `fit_deem_score`
returns, so a completed fit and its history are thrown away. Calling `fit_deem_score` directly
recovers them, which is all the probe does.

The preregistered 5 × 3 grid, selected on label-free criteria only:

| | epochs 100 | 300 | 1000 |
|---|---|---|---|
| **lr 1e-4** | **completion 1.00, median σ 0.434** | 0.00 | 0.00 |
| lr 3e-4 | 0.00 | 0.00 | 0.00 |
| lr 1e-3 *(registered)* | 0.00 | 0.00 | 0.00 |
| lr 3e-3 | 0.00 | 0.00 | 0.00 |
| lr 1e-2 | 0.00 | 0.00 | 0.00 |

Exactly one configuration survives, and collapse is monotone in both learning rate and epoch count —
the signature of an optimisation collapse, not a data problem. Validated on all five registered seeds
across the three pilot cells: **15 of 15 healthy**, median σ 0.429, cross-seed |Spearman| 0.991–0.999.
AUROC was never consulted at any step; the tie-breaker trace is in `summary.json`.

The two stopping decisions are independent, per the review — hard categorical DEEM and repaired soft
DEEM are different methods:

- **Soft DEEM: repaired.** Completion 1.00 against the 0.90 requirement. Its predefined evaluation
  should run, regardless of how hard DEEM performs.
- **Hard DEEM:** `deem_irbm_hard_ensemble` shows **no meaningful advantage over IU** — which rules out
  the preregistered +1.0pp gain and does *not* prove inferiority. `deem_deep_hard_ensemble` retains a
  possible meaningful gain.

The registered H3 row stands as run and is not replaced. What this changes is the interpretation: H3's
empty candidate set is an artefact of a learning rate that collapses this input, not evidence that
nonlinear dependency modelling cannot fit continuous detector ranks.
