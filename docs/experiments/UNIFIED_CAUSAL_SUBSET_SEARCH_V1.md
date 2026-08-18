# Unified Causal IU-PCR subset search v1

Date: 2026-08-18  
Branch: `codex/unified-causal-subset-search-v1`  
Claim status: retrospective supervised development on opened Qwen and Llama
ProcessBench data; no unified-method promotion

## Question and outcome

The first Unified Causal implementation crossed 37 causal base streams with 28
DSP transforms, creating 1,036 coordinates. This cycle tests whether a
structured subset, supervised coordinate reweighting, or same-roster
DUFS-LIU-PCR can produce one score trajectory for Global detection,
first-error Localization, and Early detection.

The feature-subset hypothesis is supported, but the one-method replacement
hypothesis is not. The full bank is strongly harmful. A compact 28-coordinate
ordinary-IU candidate is the best opened cross-scorer compromise and improves
Localization, but it regresses the strongest access-matched Global and Early
incumbents beyond the frozen margins. DUFS and supervised reweighting do not
survive scorer transfer.

Final verdict:
`DO_NOT_PROMOTE_UNIFIED_CAUSAL_V1_REGRESSES_GLOBAL_AND_EARLY_INCUMBENTS`.

## Claim and leakage boundary

- All feature, sign, weight-prior, alpha, and lambda choices use historically
  opened ProcessBench labels. The method is supervised-developed.
- Source-question groups remain intact. Every Qwen fold refits base references,
  signs, IU-PCR, supervised reweighting, DUFS, and operating thresholds on its
  fit side only.
- Qwen discovery uses 3 repeats x 3 grouped folds on 128 balanced questions,
  32 per ProcessBench family.
- Llama-3.1-8B is not refit, but its labels were inspected while choosing the
  cross-scorer compromise after the corrected Qwen winner failed transfer.
  It is therefore opened development, not frozen validation.
- No cluster inference or Drive mutation was performed.

## Implementation and aggregation contracts

`scripts/run_unified_causal_subset_search_v1.py` caches raw base matrices and
materializes the causal DSP bank once per fold/reference. Subsets reuse column
slices. `UnifiedCausalIU.score_causal_matrix` is tested exactly equal to live
tokenwise replay, including imputation, robust scaling, signs, evidence,
Identity accumulation, localization, warnings, and terminal score.

The primary task vector is family-macro Global AUROC, ProcessBench Localization
F1, and mean Early AUROC at 64/128 tokens. Margins are 0.010, 0.010, and 0.015.

An audit before final handoff found that run-schema-v1 initially concatenated
OOF scores from separately fitted folds when constructing repeat metrics.
Those score scales are not comparable. The fold payloads and model predictions
were valid; only aggregation was wrong. The runner now computes each task
inside each fold, averages folds inside a repeat, and then averages repeats.
All eight search stages were rebuilt from their hashed checkpoints without
refitting. A regression test makes pooled-fold aggregation fail.

## Structured search

### Stage A — transform families

| roster | coordinates | Global | Localization | Early |
|---|---:|---:|---:|---:|
| `raw9_level` | 9 | 0.6720 | 0.1775 | **0.5456** |
| `all37_level` | 37 | 0.6650 | 0.1631 | 0.5356 |
| `core5_level` | 5 | 0.6913 | 0.1621 | 0.5247 |
| `all37_multiscale_sustained` | 407 | 0.7151 | 0.2393 | 0.5137 |
| `raw9_fastslow` | 27 | 0.7094 | **0.2440** | 0.5101 |
| `all37_no_bocpd` | 962 | **0.7666** | 0.1900 | 0.4879 |
| `raw9_sustained` | 27 | 0.6561 | 0.2183 | 0.4994 |
| `all37_change` | 185 | 0.4984 | 0.1189 | 0.4533 |
| `all37_window_moments` | 444 | 0.7027 | 0.1308 | 0.4955 |
| `all37_full` | 1,036 | 0.4870 | 0.0873 | 0.4489 |

The full bank is rejected. Removing BOCPD from the wide bank helps greatly but
still produces a severe three-task tradeoff. No transform family wins all
tasks.

### Stages B/C — provenance and targeted unions

Under `level`, raw9 is the best Early point, broad28 is the stronger
Global/Localization tradeoff, and joint18 is a small stable control:

| roster | coordinates | Global | Localization | Early |
|---|---:|---:|---:|---:|
| `raw9_level` | 9 | 0.6720 | 0.1775 | **0.5456** |
| `broad28_level` | 28 | **0.6882** | **0.2161** | 0.5235 |
| `joint18_level` | 18 | 0.6614 | 0.1830 | 0.5320 |

Adding `ewma16`, `positive_area`, and `persistence` to raw9 yields the useful
36-coordinate Pareto arm:

| roster | coordinates | Global | Localization | Early |
|---|---:|---:|---:|---:|
| `joint18_level` | 18 | 0.6614 | 0.1830 | **0.5320** |
| `raw9_level_sustained` | 36 | 0.6856 | **0.2991** | 0.5245 |
| `joint18_level + raw9_sustained` | 45 | **0.7118** | 0.2811 | 0.5202 |

### Stages D/E — attribution and compact rosters

Drop/add attribution motivates removing `raw::spilled` and
`raw::neg_margin`. The resulting `base7_full28` crosses these bases with
`level`, `ewma16`, `positive_area`, and `persistence`:

- `raw::entropy`;
- `raw::neg_logsumexp`;
- `raw::neg_top1`;
- `raw::topk_entropy`;
- `raw::topk_varentropy`;
- `raw::topk_renyi2`;
- `raw::topk_tail_mass`.

Corrected Qwen fold means are:

| candidate | coordinates | Global | Localization | Early | Qwen role |
|---|---:|---:|---:|---:|---|
| `base6_no_entropy18` | 18 | 0.6903 | 0.3108 | **0.5334** | maximin winner |
| `base7_full28` | 28 | **0.6914** | 0.3040 | 0.5301 | stable Pareto arm |
| `raw9_full36` | 36 | 0.6856 | 0.2991 | 0.5245 | control |
| `joint18_level` | 18 | 0.6614 | 0.1830 | 0.5320 | small control |

The Qwen-only winner `base6_no_entropy18` fails Llama transfer
(0.6426/0.2374/0.5463). `base7_full28` is retained only as the best opened
cross-scorer compromise; this choice consumes Llama as development data.

## Opened Llama transfer and internal-bank controls

| candidate | Global | Localization | Early |
|---|---:|---:|---:|
| `base7_full28` | 0.6629 | **0.2880** | 0.5587 |
| `raw9_full36` | **0.6645** | 0.2705 | **0.5616** |
| `joint18_level` | 0.6201 | 0.1749 | 0.5374 |

Against raw9-full36, paired 2,000-repeat source-question bootstrap deltas are
-0.0016 Global [-0.0041,+0.0009], +0.0175 Localization
[+0.0077,+0.0273], and -0.0029 Early [-0.0045,-0.0012]. Against joint18, all
three intervals are positive. These comparisons establish that feature
selection helps inside the new causal bank; they do not establish superiority
to taskwise incumbents.

## DUFS and supervised weights

On corrected Qwen fold means, same-roster DUFS lambda 3 is the base7 winner at
0.6964/0.3287/0.5351 versus ordinary 0.6914/0.3040/0.5301. This gain does not
transfer: on Llama, lambda 3 is 0.6512/0.2352/0.5503 versus ordinary
0.6629/0.2880/0.5587. Every lambda in {0.1, 0.3, 1, 3} and alpha=0.5
supervised reweighting fails the opened cross-scorer gate. Ordinary IU is the
research candidate, not DUFS.

## Access-matched taskwise incumbents

The classic Global contract is refit without final response length, inside the
same Qwen folds and then on the same 32 Qwen questions per family before frozen
Llama transfer. Local and Early live baselines use the same fit/calibration
questions and the same 3,400 Llama rows.

| task | base7 | strongest live incumbent | incumbent | delta [95% CI] |
|---|---:|---|---:|---:|
| Global | 0.6629 | classic mixed-v2, no length | 0.6870 | -0.0241 [-0.0407,-0.0070] |
| Localization | 0.2880 | max entropy + top-5 step | 0.2419 | +0.0461 [+0.0228,+0.0691] |
| Early | 0.5587 | max entropy | 0.5777 | -0.0189 [-0.0366,-0.0005] |

Base7 does beat IU28-without-length Early in this exact transfer by +0.0213
[+0.0131,+0.0289], but max entropy is the stronger Early incumbent here. The
taskwise gate therefore fails: Localization improves, while Global and Early
breach their noninferiority margins.

After excluding all 128 Qwen development question IDs, 3,272 Llama questions
remain. Global delta is -0.0276 [-0.0452,-0.0109], Localization delta is
+0.0417 [+0.0192,+0.0661], and Early delta versus max entropy is -0.0185
[-0.0373,+0.0008]. The conclusion does not depend on overlapping question IDs.

The exact historical 30-coordinate Global method additionally contains final
response length, so it has strictly greater end-of-trace access than the
length-free comparison above.

## Drive audit

Drive contains no new complete three-task dataset. The latest DeepConf M2
checkpoint contains 4,608/122,880 traces (3.75%); a worker status is still
`complete: false`, and M2 has no first-error localization labels. It was not
downloaded or misrepresented as validation.

The complete local Llama panel exactly matches Drive L0 by SHA-256:

- GSM8K: `6749d1b846e29288e46536f9d9d8e63715b447576950d8e2131fe3420a16cead`
- MATH: `119c95ca5abc8d6f9e71bdd595db9b847373067d9aa8e4e9b349fa2b1f174953`
- OlympiadBench: `ea2592de68100aeb140ce1d3a373e87d36e00685863e0dfb8c6721fa05a1812d`
- OmniMath: `d1f9ad59eacbba64963b763e07209bdccb570f9875382f1b38c5573678ce81e4`

## Canonical artifacts

- `spectral_utils/unified_causal_subset_search.py`
- `scripts/run_unified_causal_subset_search_v1.py`
- `scripts/run_unified_causal_subset_validation_v1.py`
- `scripts/bootstrap_unified_causal_subset_validation_v1.py`
- `scripts/reaggregate_unified_causal_subset_run_v1.py`
- `scripts/evaluate_unified_causal_classic30_v1.py`
- `scripts/summarize_unified_causal_subset_cycle_v1.py`
- `scripts/test_unified_causal_subset_search.py`
- `results/unified_causal_subset_search_v1/`
- `results/unified_causal_subset_classic30_v1/`
- the stage, attribution, compact, DUFS, and Llama result directories listed in
  the final report inventory.

## Decision

Do not launch a confirmation run for one unified replacement. The best current
use of the 28-coordinate candidate is as evidence that causal sustained
features help Localization. The next research step must either repair the
Global/Early regressions under the same incumbent gate or narrow the claim to
Localization before requesting new inference.
