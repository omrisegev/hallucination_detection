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

## 0. Executive summary

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

Not committed and why: `records.jsonl`, the append-only checkpoint, is 11 MB and was being written
by the live DEEM process at commit time. It holds raw per-sample score vectors; it is required to
rebuild the DEEM seed ensembles but not to re-derive any number quoted here. Say the word and I will
add it once the sweep finishes.

Still outstanding: the DEEM arms (H3, A2, A3, P2). They are **not** in the committed tables. The
snapshot committed here is spectral-only, and the runner will regenerate every table when the DEEM
pass completes.
