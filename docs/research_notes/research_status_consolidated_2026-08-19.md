# Consolidated Research Status — 2026-08-19

This note is the canonical decision map after integrating the local `master`,
remote `master`, paper-exact acquisition, corrected white-box capture, and
local white-box analysis lineages. It distinguishes established results from
retrospective discoveries, validation blockers, and directions that have
already earned a stop decision. Historical detail remains in `PROGRESS.md` and
`HISTORY.md`; when a historical branch note conflicts with this status, this
note governs the current decision.

The consolidation task itself launched no experiment. A separate active task,
authorized independently by the user, completed a retrospective cross-dataset
manifold diagnostic while integration was in progress; its finished artifacts
and bounded conclusion are included below so that the late research channel is
not lost.

## Amendment — 2026-08-23: multi-population benchmark scope

The aligned 24-cell comparison is now one lane in a wider benchmark plan. One
inclusive method registry covers the static fusion methods, historical
selectors, cross-task representations, and application-native methods. Results
remain separated by prediction unit: complete-answer response detection,
first-error localization, causal prefix prediction, live stopping, RAG
answer/sentence/token/claim detection, and white-box access.

The registered populations include the frozen 24 cells; ProcessBench;
PRMBench; SemGrad; HLE; Evidence-Drop; AQuA/S2; RAGTruth; RefChecker;
white-box exact-common rows; repeated generations; and negative-scope panels.
Existing outcome labels have all been opened during earlier work, so these are
development or retrospective transfer/stress data, not sealed confirmation.
No cross-task macro is allowed. A future confirmation population must be
acquired or reserved with method, adapter, rows, and scores frozen before its
labels are opened.

The protocol is
`docs/experiments/MULTI_POPULATION_METHOD_BENCHMARK_V1.md`; its method,
population, and compatibility registries are under
`configs/multi_population_benchmark_v1_*`. This amendment is planning only and
does not authorize evaluation or inference.

## Amendment — 2026-08-23: Family-NRM/PGRD benchmark regimes

Family-NRM and PGRD must no longer be described as methods that inherently
require donor datasets. The next 24-cell benchmark treats data scope and
selection supervision as separate factors:

- **A:** target-cell-only construction, no donors and no hallucination labels;
- **B:** target-free donor cells may stabilize geometry, direction, or fixed
  settings without donor labels;
- **C:** donor labels may select direction, sign, graph, or strength, with the
  target dataset family held out.

The new A variants are primary benchmark arms and have not yet been run. The
completed cross-dataset Family-NRM and PGRD artifacts remain the canonical
historical methods, not results for A. Clean B means end-to-end donor selection
without labels. Historical Family-NRM is therefore legacy C because its
23-cell donor roster used a label-derived minimum-positive filter, although
the direction fit itself did not read labels. Historical PGRD is C, and the
Step-286 label-free geometry selector is legacy C because it inherited
correction strength selected with donor labels.

The aligned main roster also includes IU-PCR, deployed U-PCR, DUFS-LIU,
SU-PCR, continuous additive DEEM-B3, and balanced-atomic CA-SpecRaGE. DEEM-B3
has a completed byte-identical 24-cell run on branch `f7f7801`; its registered
decision is noninferiority. Its current 0.781815 cell-macro score must not be
ranked against the older fixed-stable graph table until the feature contract,
IU comparator, and macro are aligned. Residual-Graph DEEM remains a separate
synthetic-gate closure.

Raw A/B/C differences are diagnostic rather than causal because the source
population also changes. Matched slices must hold the candidate bank,
graph/actuator, normalization, and strength fixed while changing one selection
factor. This amendment controls the upcoming benchmark and advisor language.
It does not change any completed result or authorize a run. The working protocol is
`docs/experiments/GLOBAL_24CELL_METHOD_BENCHMARK_V2.md`.

## Amendment — 2026-08-20: A6-S0b closed by scope decision

Omri rejected the PTNI/A6 direction on 2026-08-20 on scope grounds: the thesis
is not pursuing a self-supervised identification mechanism. The gate was never
executed, so this is **not** a numerical outcome. Registered verdict:
`CLOSE_A6_S0B_DIRECTION_REJECTED`.

The record above stated that no `S0B_COMPLETE.json` or `S0B_CLOSED.json` exists.
That is correct, and the reason is now established: **the run never started.**
Slurm job `196764` (`a6s0b`) was submitted 2026-08-15T16:19:15, was never
allocated a node (`Start = None`), and was cancelled by the user at
2026-08-15T19:36:53 with `Elapsed 00:00:00`, `ExitCode 0:0`. Five independent
checks agree that no S0b artifact exists anywhere:

| where | checked | result |
|---|---|---|
| cluster output root | `$SHARED/code_a6_s0b/results/` | no `automatic_group_free_phase_a6_s0b_v1`; no `A6_S0B_BOUNDARY.json`, which stage 1 `prepare` would have written even on a partial run |
| cluster logs | `$SHARED/logs/` | no `a6s0b_*.out`; no `a6s0b_verify_ok_*` marker |
| local worktrees | all three | Pythia downloaded 2026-08-15 13:59 (`model.safetensors`, 911,373,632 B) under `.worktrees/a6-s0b/local_cache/a6_s0b_pythia_c4fc8d5`, but no output directory |
| Google Drive | title search on `S0B` | one hit: `HANDOFF_A6_S0B_TO_CLAUDE_2026_08_15.md`. No run artifact |
| consolidation manifests | `opening_state_manifest.json` (366 untracked), `pre_merge_untracked_drive_manifest.json` (225 archived) | zero `s0b` matches in either |

Preparation reached the point of a downloaded, authenticated Pythia and a frozen
source boundary; execution never began. **Do not resume the chain**, and do not
re-file this as `CLOSE_S0B_NUMERICAL_NONCONVERGENCE` — nothing was measured.

**Open scope question for the next session.** The rejection is recorded here at
its narrow, stated extent: PTNI/A6. Whether it also excludes other arms that
lean on pseudo-labels — notably `a6.pl_dufs`, the selector of record since
Step 194 — has not been decided. Do not widen or narrow it by inference.

## Executive state

The thesis has two frozen methodological anchors: ordinary IU-PCR for
label-free fusion and Unified-28 for a single causal representation spanning
global, localization, and prefix settings. Unified-28 is reproducible and
coherent, but the fair paper-exact comparison shows that unification has a
measurable performance cost relative to a dedicated method in every eligible
direct lane. It is therefore the unified method of record, not a claim of
universal superiority.

Family-NRM is a confirmed but small improvement on PRMBench. Its success does
not generalize to atomic grouping, and covariance geometry alone does not
identify the target direction. The current PTNI/A6 program is the principled
attempt to supply that missing mechanistic identification; finishing its
already-frozen S0b gate is the first research priority.

Localization and RAG demonstrate that the signal can be applied beyond
response-level detection, but the strongest claims remain bounded by their
frozen protocols. Current contextual routers do not supply a transferable
label-free routing key. White-box depth telemetry is promising because it
matches gray-box AUROC with broader coverage and supports a post-hoc hybrid
hypothesis, but it is not independently validated. Paper-exact computation and
the fair comparison package are complete; provenance closure, claim mapping,
and writing remain.

## Research-front matrix

| Front | Evidence and key numbers | Maturity | Open edge | Decision | Decisive next action | Stop condition |
|---|---|---|---|---|---|---|
| **IU-PCR** | Frozen two-component L2 fusion; ordinary implementation remains the project-wide reference and backward-compatible API. On the frozen 24-cell benchmark, ordinary IU is 0.7591 macro AUROC versus 0.7766 for mixed-v2 DUFS-LIU; Math is effectively tied (0.7869 vs 0.7862), while QA accounts for the loss (0.7128 vs 0.7604). | **Frozen anchor** | None inside the current thesis method search. | Keep as the interpretable label-free fusion baseline and solver contract. Do not tune it retrospectively. | Packaging, exposition, and external use only. | Reopen only under a preregistered, genuinely new identification mechanism and new evaluation data. |
| **Unified-28** | Accepted 61-file fair-comparison package, 148,502/148,502 records, byte-identical rebuild, tree SHA-256 `957cf08e94995d7b28143f1d53dd08062e80a8beab6c52650fb670ad1295260c`. It loses each eligible direct lane to its dedicated incumbent: Global 0.662910 vs 0.687036, delta -0.024125 [-0.041678,-0.007466]; Localization 0.284832 vs 0.326141, delta -0.041310 [-0.063480,-0.016467]; Prefix 0.578103 vs 0.606721, delta -0.028617 [-0.052068,-0.004390]. It remains above Mind-the-Gap Localization by +0.082376 [+0.059730,+0.117390]. | **Frozen anchor / accepted negative comparison** | Interpretation and paper narrative, not method selection. | Keep as the unified causal method of record. State the price of unification explicitly; never substitute it for a dedicated incumbent in a superiority claim. | Advisor interpretation and paper drafting from frozen tables. | Do not reopen its features, DUFS choice, confirmation cells, or stopping policy inside this package. |
| **Family-NRM** | PRMBench response-level AUROC improves from 0.720602 to 0.725206: **+0.460pp**, paired source-group 95% interval [+0.068,+0.841]pp, with the manual six-family provenance fixed in advance. Supporting transfers are positive but small/heterogeneous. | **Confirmed, limited achievement** | The manual family partition is an identification prior; the gain is not a general theorem about neutral covariance modes. | Retain as a bounded positive result and as motivation for PTNI. | Package the exact claim and contrast it with Atomic-NRM. | Stop expanding NRM variants unless a new mechanism identifies target versus nuisance independently of labels. |
| **Atomic-NRM / neutral geometry alone** | Atomic features contain supervised headroom, but the frozen label-free candidate loses to IU/Family-NRM and retrospective controls show that stability is not target identification. White-box organic NRM likewise reverses under LOMO/LOCO and same-model/same-dataset controls. | **Exhausted / falsified in tested form** | None for covariance-only selection. | Close. Preserve as a useful negative result explaining why PTNI is needed. | No next experiment in this family. | Reopen only if a non-label target-identification signal is specified before evaluation. |
| **PTNI / A6** | A6-S0a independently reproduced `PASS_S0A`: 1,800 reciprocal quartets, 6,000 natural rows, 7,200 inner-fold assignments, 36 null cells. S0b code is frozen at `89c414a`; 56/56 focused tests passed. The S0b gate was **never executed**: job `196764` was cancelled while pending, `Elapsed 00:00:00`, and no artifact exists on cluster, Drive, or disk. Step 270 [A6/PTNI] separately falsified the gamma-3 orientation channel that Steps B and C were premised on: pooled cos(gamma-3, g*) falls +0.7617 -> +0.3350 under the correction the source memo itself required, and the family sign-bit margin inverts, +0.5532 -> -0.1702 (bit wrong). | **Closed by scope decision** | None. The direction is rejected, not exhausted. | `CLOSE_A6_S0B_DIRECTION_REJECTED` (Omri, 2026-08-20): the thesis is not pursuing a self-supervised identification mechanism. Retain S0a, the frozen S0b source, and the Step-270 negative result as citable evidence for why covariance geometry alone does not identify the target direction. | None. Do not resume the chain. | Already stopped. Reopening requires a new scope decision by Omri, not a technical trigger. |
| **Localization / reasoning application** | Frozen GL-LIU v1 reaches 31.36% ProcessBench F1 versus 25.71% for reproduced Mind-the-Gap. The unified global/local core-five variant reaches 31.72%, but the +0.37-point change is descriptive and mixed; broad-28 falls to 29.03%. In the fair package the stronger dedicated family-six localizer is 0.326141 versus Unified-28 at 0.284832. | **Confirmed application, bounded claim** | Publication framing and clear separation of response detection, first-error localization, and stopping. | Keep GL-LIU v1 as frozen localization anchor; treat 31.72% as descriptive, not an externally confirmed replacement. | Package the reasoning/localization contribution with exact estimands and access labels. | No more local feature-pool search without new preregistered data. |
| **RAG application** | Fixed IU-PCR pipelines reach RAGTruth answer/sentence/token AUROC **0.7276 / 0.6893 / 0.6586**. Evidence-contrast work supports a useful feature contract, but labels were opened during this development and the mechanism is not a new DUFS/Laplacian result. | **Confirmed engineering/application result; exploratory research claim** | Convert the evidence into a precise, non-inflated claim and package it alongside reasoning. | Retain the frozen pipeline and disclose development access. Do not present the feature contrast as independent confirmation. | Assemble the reasoning/RAG application package after the paper-exact claim map is closed. | Stop feature search on the observed RAGTruth labels; require new frozen data for a stronger claim. |
| **Clustering and static feature-selection families** | Extensive clustering, subsets, SU-PCR, Laplacian, DUFS, and DUFS-LIU searches established useful diagnostics and the mixed-v2 incumbent, but did not yield a stable new universal mechanism. Gains are heterogeneous and repeated searches consume the same observed cells. | **Exhausted as discovery programs** | Maintenance of frozen comparators only. | Close clustering/DUFS/Laplacian as active research directions. Keep exact implementations for baselines and reproducibility. | None beyond packaging and regression tests. | Reopen only for a materially new data regime and a frozen hypothesis, not another selector sweep. |
| **Contextual IU / c-STG routers** | DSP contextual IU ends `STOP_NO_ROUTING_SIGNAL`. Local/Early c-STG ends `STOP_CONTEXT_NOT_SUFFICIENT`: Localization 0.2812 versus global LR 0.3503; Early 0.5906 versus 0.5995. Global c-STG ends `GLOBAL_ORACLE_NOT_ACCESSIBLE_BY_CSTG`: c-STG 0.7464 versus global LR 0.7506; the core-only 0.7563 point estimate is exploratory and heterogeneous. | **Exhausted in current form** | The missing object is an observable, transferable routing key, not more optimizer capacity. | Close current contextual routers and preserve their negative diagnostics. | No escalation to LTSREx/LEGO from these results. | Reopen only when an independently measurable context variable predicts which expert should dominate on held-out groups. |
| **Early / online / stopping** | Causal-prefix evidence exists, but Unified-28 loses the dedicated prefix incumbent. DeepConf remains a historical proxy unless the exact confidence statistic/window/multi-trace protocol is reproduced. LEASH saves 20.2%--49.9% realized tokens across six complete cells but lowers pass@1 in all six. | **Mixed evidence; canonical distinctions established** | Paper-exact provenance for remaining comparator rows and honest accuracy-compute framing. | Keep prefix prediction, first-error localization, and realized stopping as separate estimands. Do not claim a stopping win. | Close claim mapping in the paper-exact package; no new stopping design during consolidation. | Reopen method work only through the package's structured GPU gates. |
| **White-box depth fusion** | Pure distributed-depth U-PCR is 0.784612 macro AUROC; the strengthened atomic oracle is 0.784186, but the +0.000426 interval crosses zero. On 31,440 exact common rows, white-box is 0.781690/0.677048 versus gray mixed-v2 0.782994/0.687731; AUROC delta -0.001304 [-0.016931,+0.012300]. White coverage is 42,238 versus 31,467 gray complete cases. A post-hoc equal-z hybrid reaches 0.790203/0.690580, +0.007209 AUROC [+0.000101,+0.014105], with a near-zero lower bound. | **Promising, post-hoc, validation-blocked** | Corrected live Gate B on all intended cells and one independent, architecture-faithful frozen validation/pilot. | Preserve the corrected remote capture implementation as authoritative. Promote no superiority claim yet; the defensible result is gray-box AUROC parity plus wider coverage. | If budget is justified, preregister one new frozen hybrid/white-box validation after A6-S0b. | Stop if live capture/architecture fidelity fails, or if the frozen independent interval does not support the registered practical gain. Do not redesign on validation cells. |
| **Paper-exact acquisition and fair comparison** | K=512 acquisition, DeepConf, REFRAIN, portability/provenance repairs, and the accepted CPU comparison package are preserved. The package rebuild is exact and its direct computational claims are closed. Some historical rows remain blocked/partial and must stay labelled that way. | **Computationally complete; documentation/provenance closure pending** | Source-to-claim mapping, exact/adapter/proxy labels, unresolved blocked rows, advisor-facing tables, and manuscript text. | Close the package rather than launch more GPU work. | Build a claim ledger linking every manuscript sentence/table to protocol, population, artifact hash, and fidelity/access label. | Stop when every promoted claim has a frozen artifact and every unavailable claim is explicitly blocked/partial; GPU only through `GPU_GATES.json`. |
| **Cross-dataset hallucination geometry** | A frozen retrospective supervised diagnostic uses 24 cells and eight leave-one-dataset-family-out folds. Mean and covariance fingerprints transfer (held-family cosine 0.9650 and 0.6149; both sign-flip p=9.999e-05). Balanced logistic reaches 0.7379 family-macro AUROC [0.6791,0.7873]; shared direction 0.7364; kNN manifold 0.7353; PPCA manifold 0.6882. kNN minus logistic is -0.0026 [-0.0087,+0.0030], and PPCA is -0.0497 [-0.0838,-0.0278]. | **Retrospective diagnostic; supervised; not external confirmation** | Label-free identification remains unsolved; the existing confidence orientation was informed by prior labelled audits. | Decision `SHARED_DIRECTION_NOT_DISTINCT_NONLINEAR_MANIFOLD`: a transferable target axis exists, but there is no evidence here for a distinct useful nonlinear hallucination manifold. This does not rescue DUFS-LIU identifiability. | Use as interpretation/negative evidence. A stronger claim requires an untouched family frozen before fitting. | Do not search more manifold variants on these 24 cells. Reopen only with a preregistered model and new frozen family. |
| **Geometry/manifold literature** | The literature review sharpens interpretation of redundancy, local geometry, and subspaces, but by itself supplies no observed label-free router key. | **Diagnostic only** | Writing integration. | Use for motivation and limitations, not as a license for another search. | Incorporate into discussion after core claims are frozen. | No experiment without a concrete preregistered observable and falsifiable gate. |

## Ordered objectives

1. ~~**PTNI/A6 first.**~~ **Withdrawn 2026-08-20** by scope decision
   (`CLOSE_A6_S0B_DIRECTION_REJECTED`); see the amendment at the top of this
   note. Objectives 3 and 4 are now the leading work, and neither needs GPU.
2. **One new frozen white-box validation only if its cost is justified.** It
   must include corrected live capture, architecture fidelity, untouched
   evaluation data, and a preregistered claim/stop rule. The post-hoc equal-z
   hybrid is a candidate, not a result to be promoted.
3. **Close paper-exact provenance and claim mapping.** Computational work is
   complete; convert the accepted package into a manuscript-grade claim
   ledger and keep blocked/partial rows visible.
4. **Package the reasoning/RAG contribution.** Present response detection,
   first-error localization, prefix prediction, stopping, and RAG granularity
   as distinct estimands with their real access and validation boundaries.

These objectives were written as sequential behind objective 1. With objective 1
withdrawn, **objective 3 (paper-exact provenance and claim mapping) leads**, with
objective 4 (packaging reasoning/RAG) next; objective 2 stays conditional on a
justified budget. The consolidation itself authorizes no new experiment.

## Integration and provenance ledger

The integration branch preserves these heads as ancestors:

- local `master` base `7ad92c9`;
- `origin/master` `cd423ab`;
- `origin/paper-exact/acquisition-v1` `79ee28e`;
- `origin/whitebox/per-layer-views` `85149a0`;
- local `codex/whitebox-layer-fusion` `7cdc39a`.

Fair-comparison (`bc296ff`), Unified (`d3ca3a4`), and three-way (`ef3154e`)
lineages were already ancestors of `master` and were not merged a second time.
The opening-state inventory is
`results/research_consolidation_2026_08_19/opening_state_manifest.json`.

The 225 selected heavy intermediate files (162,152,300 bytes) were copied
non-destructively to
`gdrive:hallucination_detection/consolidated_results/integration_2026-08-19/pre_merge_untracked/`.
`rclone check --one-way --checksum` reported 225 matches and zero differences;
local source files were not deleted. Traceability lives in
`results/research_consolidation_2026_08_19/pre_merge_untracked_drive_manifest.json`
and `drive_verification.json`.

The 23 consolidation-only localization and RepGrid pickle payloads are retained
in their canonical Drive directories rather than added to Git LFS: GitHub's
exhausted LFS budget blocked the first publication attempt. The SHA-256
inventory is `dataset_cache/INVENTORY_2026_08_19.json`, and the checksum-tested
Drive map is `dataset_cache/DRIVE_BACKUP_2026_08_20.json`. Their compact
manifests remain in Git. Stashes, branches, worktrees, local copies, and a
pre-removal backup ref remain intact as recovery points.

## Canonical evidence map

- Fair comparisons: `results/fair_paper_exact_comparisons_v1/REPORT.md` and
  `docs/experiments/FAIR_PAPER_EXACT_COMPARISONS_V1.md`.
- Family-NRM: `results/neutral_residual_mode_prmbench_v1/REPORT.md`.
- PTNI/A6: `results/automatic_group_free_phase_a6_s0a_v1/` and the current A6
  S0b section of `PROGRESS.md`.
- Localization: `results/ours_only_localization_v1/REPORT.md` and
  `results/gl_liu_factorial_v2/REPORT.md`.
- RAG: `results/fixed_application_pipelines_v1/REPORT.md` and
  `results/ragtruth_evidence_contrast_v1/REPORT.md`.
- Contextual routers: `results/dsp_contextual_iu_pilot_v1/REPORT.md`,
  `results/contextual_stg_router_diagnostic_v1/REPORT.md`, and
  `results/global_contextual_stg_router_diagnostic_v1/REPORT.md`.
- White-box: `results/whitebox_depth_distributed_pure_v1/REPORT.html`,
  `results/whitebox_vs_graybox_matched_v1/REPORT.md`, and
  `docs/experiments/WHITEBOX_LAYER_FUSION_RESEARCH_RECORD.md`.
- Cross-dataset geometry:
  `results/cross_dataset_hallucination_manifold_v1/REPORT.md` and
  `docs/experiments/CROSS_DATASET_HALLUCINATION_MANIFOLD_V1.md`.
- Early/online distinctions:
  `docs/research_notes/early_online_detection_canonical_status_2026-08-19.md`.
