# Historical index: the eight uncovered refs (consolidation 2026-10-01)

Prepared 2026-10-02 by the Claude consolidation session ae2dd164. Read-only on the repository.

**What this file is.** The consolidation review (`docs/reviews/CONSOLIDATION_REVIEW_20261001.md`, line 17)
found eight historical research or report refs that hold **1,052 distinct paths** absent from all 13
proposed source tips. Omri accepted an archive with an index for this material (decision 5 in
`docs/consolidation/DECISIONS_2026-10-01.md`):

> "An archive with an index is acceptable for historical material. The index must record what each line
> accomplished, its evidence, and any unfinished or superseded next steps, not just branch names. Bring
> code, tables, and dependencies needed by the current research into the consolidated tree. Also, not all
> eight branches are from August; one contains a September review."

**Proposal status.** The facts below were checked against git. The dispositions ("archive", "bring in")
are **Claude's proposals, not Omri decisions**. Nothing here closes a research direction. The only
explicit closure so far is Joint L-SML (discontinued 2026-10-01).

**Terms used.** "The 13 tips" = the `plan[]` refs in `results/consolidation_review_20261001/INVENTORY.json`.
"Current line" = `claude/ssl-pseudolabel-residual-v1` (553de79c6) and `claude/decision-rule-v1` (cbe345a4b),
used for dependency searches. "pp" in AUROC deltas = percentage points of AUROC.

---

## Summary table

| # | Ref | Tip | Commits unique vs the 13 tips | What the line did | Its own final status | Needed by current research? | Proposed disposition |
|---|---|---|---|---|---|---|---|
| 1 | `origin/codex/reconstruction-science-results-v1` | d9827a835, 2026-08-27 | 7, dated 2026-08-24 to 08-27 | Certified evidence of the reconstruction program's application lanes (EDIS, external final answer, localization, prefix, LEASH, RAG, unified reporting) | Completed | No research code. The tracked source lock names 39 paths that exist only here | Archive with index; tag |
| 2 | `origin/codex/reconstruction-benchmark-v1` | 1780572a2, 2026-08-27 | 6, all 2026-08-27 | Frozen 24-cell benchmark (13 label-free fusion methods, 48,607 answers) plus the Aug 25/26 advisor letter packet | Completed ("approaching saturation") | No code. Advisor letter docs are a record (106,721 bytes) | Bring the letter docs in; archive the results |
| 3 | `claude/advisor-letter-aug27` | 2832b8a72, 2026-08-27 | 3, all 2026-08-27 (2 shared with ref 2) | An alternative Aug 27 restructure of the advisor letter | Superseded draft | None. Its one unique file is already on the rescue tip, same blob | Archive with index |
| 4 | `origin/codex/graph-geometry-selection-v1` | 483d4150c, 2026-08-27 (work dated 08-21 to 08-23) | 1 squashed commit | Graph line on the 24-cell panel: family-residual graph LIU, SU sidecars, pooled graph-roughness direction (PGRD), geometry selection, benchmark registries | Closed or design-only in its own words | **Yes.** `scripts/build_advisor_update_aug21_2026.py` (on 12 of the 13 tips) reads 5 files that exist only here. HISTORY Steps 283-289 [graph line] exist only here | Bring 3 result dirs + HISTORY steps; archive the rest |
| 5 | `origin/codex/iu-graph-smoothing-ablation-v1` | 8829fd8e3, 2026-08-27 | 6, dated 2026-08-24 to 08-27 | IU graph-order ablation: where in the pipeline a graph helps IU-PCR | Completed, negative | None | Archive with index; optional small restore |
| 6 | `origin/codex/deem-b3-moe-gating-v1` | 877919734, 2026-08-27 | 1 | Code for nine label-free DEEM-B3 routing challengers, plus one written ceiling diagnostic | Negative diagnostic; challengers preserved only | None on its 41 files. Run outputs live only in the second computer's `local_cache/` | Archive with index; optional ceiling note |
| 7 | `selector/a4-antigravity-unsupervised` | 51c8964e1, 2026-08-07 (branched 2026-07-18) | 1 | Late save of the Step 186 A4 "antigravity" selector bench CSVs | Superseded by Steps 187/189; selector family negative (Steps 186, 222, 224) | None. Its 4 absent files were renamed `_T1.5` with equal values; those are on the rescue tip | Archive with index |
| 8 | `origin/codex/og-sml-agent-b-v1` | **45f8b572e, 2026-09-04** | 1 (Claude's review); earlier commits 08-28 to 09-04 are on some tips | 0.3662 reasoning-localization program, Joint L-SML structural study, and Claude's pre-registration review | Phase 3 closed with no promotion; Joint L-SML later harmed localization and is now discontinued | The review file is pinned as provenance by a tracked run registry. The lost HISTORY Steps 298-353 must come from 48f9ef291 | Bring the review file in; restore HISTORY from 48f9ef291 |

**The September ref (Omri's note).** Ref 8, `origin/codex/og-sml-agent-b-v1`, has its tip on
**2026-09-04**: "review: add Claude's Joint L-SML v1 review + pre-registration concerns for Codex". Its line
also has commits from 2026-08-29 to 2026-09-04. The other seven tips are dated 2026-08-07 or 2026-08-27.
(Ref 7's work itself dates from July; its only commit is 2026-08-07.)

---

## Verified facts this index relies on

1. **HISTORY Steps 298-353 were removed by commit 9215ae69d** (2026-09-10, "Record direct probability
   fusion benchmark"). Its parent **48f9ef291** holds a full copy. Checked: at 48f9ef291 there are 54
   step headings numbered 298-353 tagged `[reasoning localization]`, `[localization]`,
   `[per-answer windows]` and `[joint-lsml-v2]` (for example "Step 347 [reasoning localization] -
   Joint L-SML structural success does not transfer"). After 9215ae69d none of them remain. The commit
   wrote a different set of steps under numbers 298-336 (Codex answer-only fusion line). None of the 13
   tips has the `[reasoning localization]` Steps 300-319.
2. **Steps 283-289 of the graph line exist only on ref 4.** At 483d4150c they are "Step 283
   [Family-residual graph]" through "Step 289 [Benchmark design]". The worktree tip of this rescue branch
   uses 283-288 for "[DEEM benchmark]" and 289 for "[reconstruction benchmark]". None of the 13 tips has
   "Step 283 [Family-residual graph]". By the tagging rule they go in as tagged duplicates, not renumbered.
3. **`scripts/build_advisor_update_aug21_2026.py` depends on graph-only files.** Lines 32-36 name
   `results/family_residual_graph_liu_v3/RESULT.json`, `.../SYNTHESIS.md`, `.../controls/RESULT.json`,
   `results/family_residual_graph_liu_prmbench_v3/RESULT.json` and
   `results/family_residual_graph_liu_hle_v3/RESULT.json`. The script exists on 12 of the 13 tips (not on
   890459866). The result files exist on none of them, only at 483d4150c.
4. **DEEM-B3 challenger outputs live only on the second computer.** The ceiling note at 877919734 gives
   `/Users/osegev/...` reproduction paths and `local_cache/deem_b3_moe_v1/...` output paths.
   `local_cache/deem_b3_moe_v1` does not exist on this Windows machine. 25 config and script files on
   the SSL tip still name that path (for example `configs/ciw_deem_v1.json` lines 31-32).
5. **CT7-era HISTORY is uneven across tips** (a merge note, not about the eight refs). Steps 390-421 have
   34 headings on `claude/readout-quickest-detection-v1` and `lsml-ct7-levers-run`, but only 7 on the SSL,
   decision-rule and rescue tips. A HISTORY union must take them from those two tips.

---

## 1. `origin/codex/reconstruction-science-results-v1`

- **Tip:** d9827a835, 2026-08-27 19:16, "results: publish compact application science evidence" (Codex).
- **Unique commits:** 7. Six code fixes dated 2026-08-24 (EDIS serialization, external evaluation A/B
  verifier, grouped bootstrap speedup, GPQA aggregate block, freeze handoff, PyTorch probe). One
  evidence publish on 2026-08-27 (345 files, about 118 MB). Per the draft, the code fixes reached the
  current line with identical content (for example `spectral_utils/reconstruction_benchmark/edis_evaluation.py`,
  blob 1cfa780f); this was not re-checked file by file.

**What it accomplished.** It packaged the certified evidence for the application lanes of the August
reconstruction program in one place: EDIS, external final-answer evaluation (ProcessBench, PRMBench, HLE,
Evidence-Drop, GPQA stress), first-error localization, fixed-prefix prediction, LEASH stopping, RAG
evidence, and the certified unified-reporting bridge.

**Evidence (all @ d9827a835, under `results/reconstruction_benchmark_v1/`):**
- `releases/2026-08-24_localization_v1/build_A/localization/evaluation/metrics_long.csv`: ProcessBench
  official macro F1 of DUFS-LIU with the token head is 0.3099 / 0.3115 / 0.3078 (Llama-3.1-8B, Qwen3-4B,
  Qwen3-8B); the token head alone 0.2970. PRMBench step AUROC: token head 0.6712 against 0.7983 for the
  supervised Qwen2.5-Math-PRM-7B.
- `derived/unified_reporting_v1_certified/build_A/tables/metrics.csv`: prefix AUROC at 64 tokens 0.5955
  [0.567, 0.624] against 0.5629 for Unified-28; at 256 tokens 0.6572 against 0.6114. `AB_VERIFICATION.json`: PASS.
- `releases/2026-08-24_external_final_answer_v3_opaque/build_A/external_final_answer/evaluation/metrics_long.csv`:
  PRMBench response level (N=6,966) DUFS-LIU 0.7228, IU-PCR 0.7209, DEEM-B3 0.6966. HLE near chance
  (IU-PCR 0.5183).
- `releases/2026-08-24_edis_aime_v1/build_A/edis/FIT_UNAVAILABLE.json`: `STATUS_ONLY_ROSTER_MISMATCH_OR_NO_RUNNABLE_CELLS`.
- `REMOTE_ARTIFACT_BOUNDARY.md`: the full local tree was about 15 GB. Private labels, fit inputs and
  per-example predictions were deliberately left out of git.

The numbers above come from the draft; only the file paths were re-checked here.

**Its own status:** completed. PROGRESS at the merge base (Step 290): "the reconstruction plan has no
remaining required experiment or integration step."

**Next steps it left:**
- "Advisor/paper interpretation or an explicitly requested new study." **Superseded** by later work:
  CIW-DEEM on these lanes (Step 292), cross-scale localization (Step 293), then the localization benchmark
  (2026-09-07), PRMBench primary (2026-09-23) and external telemetry first (2026-09-24).
- RAG lane: **out of the active claim** (CLAUDE.md 2026-09-07). Not closed as a research idea.
- 24-cell transfer: **open as "historical24"** in CLAUDE.md. It was redone independently on 2026-09-11
  (d7d1e17bf), which does not read these files.

**Needed by current research?** No research code (CT7, PRMBench, the label-free line) reads these files.
But the tracked `configs/reconstruction_benchmark_v1/unified_reporting_source_lock_v1.json` names 44
result paths; 39 exist only at d9827a835 (checked), 0 at ref 2, 5 nowhere. They are needed only to re-run
the August reporting bridge or `scripts/build_advisor_update_aug21_2026.py`'s reconstruction sections.
**Bring nothing in; keep the tag so the source lock stays resolvable.**

**Restore:**
```bash
git tag archive/reconstruction-science-results-v1 d9827a835
git restore --source=d9827a835 --worktree -- results/reconstruction_benchmark_v1
# .gitignore ignores releases/ and external_audits/: re-tracking needs git add -f
```

---

## 2. `origin/codex/reconstruction-benchmark-v1`

- **Tip:** 1780572a2, 2026-08-27 23:24, "docs: add TL;DR, aligned 13-method table, and comparator
  transparency to the letter" (Claude co-author).
- **Unique commits:** 6, all 2026-08-27: ead5cdf9e (evidence packet), 4485af7be (handoff note),
  812ec3c56, df0ff9177, 1cb7b0e78 (letter rewrites, Codex), 1780572a2 (Claude).
- Ref 1 and ref 2 are not supersets of each other. They share 22 result paths, some with different content.

**What it accomplished.** It published the 24-cell frozen benchmark: 13 label-free fusion methods on
48,607 answers, two byte-identical builds, scores frozen before labels, 20,000 grouped bootstrap draws. It
added reviewed winner contrasts, external audits and the advisor letter packet (letter, three deep dives,
asset index, references, four SVG figures). The letter's message: fusion is saturating on this feature
pool, and the clearest wins are in the applications.

**Evidence (all @ 1780572a2):**
- `results/reconstruction_benchmark_v1/releases/2026-08-24_frozen24_v1/evaluation/EVALUATION.json`,
  macro-24 AUROC (re-checked): DEEM-B3 0.7812 [0.772, 0.790]; equal-family mean 0.7810; DUFS-LIU 0.7766;
  IU-PCR 0.7761; down to DUFS parameter-free + L-SML 0.7674. All 13 methods lie within 1.4 points.
- `.../frozen24_v1/SCORE_AB_VERIFICATION.json`: pass = true.
- `.../frozen24_v1/reporting_v2/2026-08-24_frozen24_v1/07_reports/REPORT.html`: interactive report, 45.8 MB.
- `docs/meetings/Advisor_Update_Aug26_2026.md`: localization macro F1 31.36% against 25.71% for the
  reproduced Mind the Gap.
- `docs/experiments/REMOTE_WORKTREE_HANDOFF_2026-08-27.md`: index of the 11 Codex branches published that day.

**Its own status:** completed. The letter: "We are approaching saturation on the current feature pool ...
stop open-ended exploration, consolidate ... and choose one final prospective experiment."

**Next steps it left:**
- "Apply the innovation idea to the token trajectories themselves." **Done** in Step 293 and
  `token_temporal_innovation_b3` (2026-08-28); no gain (0.3083 against 0.3091 macro F1).
- "One final prospective experiment" and thesis consolidation. **Superseded** by the 2026-09-07
  localization mandate, the 2026-09-08 consolidation, PRMBench primary (2026-09-23) and external
  telemetry first (2026-09-24). Untouched confirmation remains a requirement in CLAUDE.md.
- A DeepConf-style prefix comparator. **Open, dormant.** Never run; no decision closed it. Prefix
  prediction is not in the active scope.

**Which letter was sent.** The sent version is the Aug 28 HTML at
`docs/reviews/sources/Advisor_Update_Aug28_2026.html`, described in
`docs/reviews/advisor_thread_reconciliation_2026-09-06.md`. It follows ref 2's wording. Checked: that HTML
is on 3 of the 13 tips (`claude/readout-quickest-detection-v1`, `lsml-ct7-levers-run`,
`claude/depth-feature-fusion-v1`) and not on the SSL, decision-rule or rescue tips.

**Needed by current research?** No code imports. The letter docs are the advisor record (14 files,
106,721 bytes). **Bring the docs in as a record; archive the results** (do not bring the 46 MB report or
the frozen24 release).

**Restore:**
```bash
git tag archive/reconstruction-benchmark-v1 1780572a2
git checkout 1780572a2 -- docs/meetings/Advisor_Update_Aug25_2026.md docs/meetings/Advisor_Update_Aug26_2026.md \
  docs/meetings/advisor_update_aug26_2026 docs/experiments/REMOTE_WORKTREE_HANDOFF_2026-08-27.md
# results when needed:
git restore --source=1780572a2 --worktree -- results/reconstruction_benchmark_v1
```

---

## 3. `claude/advisor-letter-aug27` (also `origin/claude/advisor-letter-aug27`)

- **Tip:** 2832b8a72, 2026-08-27 21:15, "docs: restructured advisor letter (Aug 27) - algorithmic core
  leads, certified 13-method table".
- **Unique commits:** 3, all 2026-08-27. ead5cdf9e and 4485af7be are shared with ref 2; 2832b8a72 adds
  `docs/meetings/Advisor_Update_Aug27_2026.md` (80 lines).

**What it accomplished.** An alternative letter that puts the algorithmic core first, with the 13-method
table and a decision list (next prospective confirmation; whether to keep RAG). Its Aug 26 packet is the
older ead5cdf9e version, without the four SVG figures.

**Evidence:** `docs/meetings/Advisor_Update_Aug27_2026.md` @ 2832b8a72. Same 13-method numbers as the
frozen24 `EVALUATION.json`; LEASH "-38.8% tokens, -18.3pp pass@1"; RAGTruth 0.727 / 0.689 / 0.659 at
answer, sentence and token level (numbers from the draft, not re-checked). Its result files match ref 2's.

**Its own status:** a draft. **Superseded** by ref 2's later edits; the sent Aug 28 email followed ref 2.

**Next steps it left:** "Whether RAG stays an active direction": answered (RAG outside the active claim,
2026-09-07). "Next prospective confirmation": superseded by the localization and PRMBench decisions; the
untouched-confirmation requirement stays open in CLAUDE.md.

**Needed by current research?** None. Its one unique file is already on the rescue tip with the same
blob (d7939062, checked).

**Restore:** `git tag archive/advisor-letter-aug27 2832b8a72`; `git checkout 2832b8a72 -- docs/meetings/Advisor_Update_Aug27_2026.md`.

---

## 4. `origin/codex/graph-geometry-selection-v1`

- **Tip:** 483d4150c, 2026-08-27 19:04, "research: publish graph geometry experiment snapshot" (Codex). One
  squashed commit on 0a631b28c (2026-08-21). Docs inside are dated 2026-08-23.
- **Footprint:** 265 absent paths, about 60 MB, plain git blobs (no LFS). 48 MB of it is two raw score
  folders: `results/graph_geometry_selection_research_v1/development_fit/` and `.../label_free_input/`.

**What it accomplished.** It tried to explain the earlier Family-NRM gain on the frozen 24-cell
final-answer panel. It rebuilt that correction from a graph over the six family residuals, without the
unsupported "eigenvalue closest to one" rule. Then it asked whether a better graph could be chosen without
labels. Finally it designed (did not run) a 24-cell V2 and a multi-population benchmark registry.

Sub-lines and outcomes:
- Step 283 [Family-residual graph]: graph LIU on family residuals. V1 withdrawn (bugs), V2 stopped before
  labels, V3 closed: +0.018pp over IU-PCR, no added value.
- Step 284 [SU/pooled-graph sidecar]: covariance cleaning adds +0.009pp; SU-rho hurts.
- Step 285 [Pooled graph-roughness direction] (PGRD): recovers about 91% of the Family-NRM gain
  retrospectively; transfer depends on the domain.
- Step 286 [Graph geometry selection]: there is headroom for a better graph, but no selector finds it.
- Steps 287-289 [Benchmark design / documentation]: design only, nothing ran.

**Evidence (all @ 483d4150c):**
- `results/family_residual_graph_liu_v3/RESULT.json` (re-checked): `nested_delta_vs_iu_pp` 0.0182,
  CI [-0.041, +0.080]; 4 of 8 families positive; Family-NRM reference +0.277pp; `promotion_pass: false`.
  `SYNTHESIS.md` line 5: `CLOSE_FAMILY_RESIDUAL_GRAPH_LIU_NO_INCREMENTAL_VALUE_V3`.
- `results/family_residual_graph_liu_prmbench_v3/RESULT.json`: -0.0068pp. `..._hle_v3/RESULT.json`: -0.019pp.
- `results/su_pooled_graph_adaptation_conservative_v2/REPORT_COMPLETE.json`: cleaned minus current +0.009pp, CI [-0.012, +0.037].
- `results/pooled_graph_roughness_direction_v2/SYNTHESIS.md`: +0.251pp, CI [+0.027, +0.458], 6 of 8
  families. External: ProcessBench Llama +0.588, ProcessBench Qwen +0.137, SemGrad +0.257, HLE +0.912,
  PRMBench -0.420 [-0.621, -0.226].
- `results/graph_geometry_selection_research_v1/FINAL_REPORT.md`: "Final decision:
  `GEOMETRY_SEARCH_SELECTION_OPTIMISM`". Graph search adds +0.016pp; label-free selector +0.220pp, below
  the fixed graph (+0.251pp); held-label oracle +0.452 to +0.723pp (diagnostic only); cross-gradient alone
  +0.245pp, so the mechanism is a pooled cross-gradient, not the quadratic graph solve.

Numbers other than the `RESULT.json` line come from the two drafts, which agree.

**Next steps it left:**
- "Freeze scores and policies before opening genuinely new, sealed dataset/model families." **Not done.
  Superseded in practice** by the localization focus (2026-09-07) and PRMBench primary (2026-09-23). No
  decision names this line, so the graph-geometry idea is not formally closed.
- Run Family-NRM-A and PGRD-A inside a 24-cell V2 benchmark. **Superseded** by `reconstruction_benchmark_v1`
  (Step 289 [reconstruction benchmark] on the current line). PGRD-A was re-implemented independently in
  `spectral_utils/reconstruction_benchmark/methods.py` and evaluated (ref 1 tables).
- Multi-population registry and a sealed confirmation population. **Superseded** by the localization
  benchmark V3 and PRMBench primary. The idea of separate per-task panels survives in CLAUDE.md
  ("Benchmark continuity and comparator coverage").
- "Recover source IDs for the seven 10-generation cells"; "retrieve the Evidence-Drop four-cell panel
  from LFS". **Open, dormant.** Relevant only if the historical24 transfer runs.

**Needed by current research? Yes, a small part.**
- The 5 files read by `scripts/build_advisor_update_aug21_2026.py` (lines 32-36). The three directories
  that hold them are 21 files, 518,291 bytes. Without them that builder raises `FileNotFoundError`.
- HISTORY Steps 283-289 [graph line], which exist nowhere else. Add them as tagged blocks, no renumbering.
- Not needed: the graph modules (`family_residual_graph.py`, `pooled_graph_roughness.py`,
  `graph_geometry_selection.py`), 29 scripts, the registry CSVs and the raw score folders. No current code
  imports them. If the graph code is ever re-run, it also needs a two-line addition to
  `spectral_utils/graph_topology.py::extended_graph_diagnostics` (keys `all_edge_weights_finite`,
  `minimum_edge_weight`) that exists only here.
- Do not merge the whole ref. It carries older copies of `spectral_utils/deem_adapter.py`,
  `spectral_utils/__init__.py`, `glossary.py` and the advisor pages.

**Restore:**
```bash
git tag archive/graph-geometry-selection-v1 483d4150c
git checkout 483d4150c -- results/family_residual_graph_liu_v3 results/family_residual_graph_liu_prmbench_v3 \
  results/family_residual_graph_liu_hle_v3
git show 483d4150c:HISTORY.md   # copy Steps 283-289 by hand as tagged blocks (union, never --theirs)
# optional documents (small): docs/experiments/{FAMILY_RESIDUAL_GRAPH_LIU_V3,GRAPH_GEOMETRY_SELECTION_RESEARCH_V1,
#   POOLED_GRAPH_ROUGHNESS_DIRECTION_V1}.md and results/graph_geometry_selection_research_v1/FINAL_REPORT.md
```

---

## 5. `origin/codex/iu-graph-smoothing-ablation-v1`

- **Tip:** 8829fd8e3, 2026-08-27 19:06, "results: include IU graph-order scores and plots" (Codex).
- **Unique commits:** 6. c7fcb1e9e, 1caf13975, 074e34495, 2bb8ea451 (2026-08-24); 7f984a199, 8829fd8e3 (2026-08-27).
- 117 absent paths, about 18.9 MB. The 18 paths with different blobs are older copies of
  reconstruction-benchmark files from the branch point; the current line has newer versions.

**What it accomplished.** It asked why an earlier graph method had looked helpful to IU-PCR. It applied
the same graph at different points: smooth the features (residual graph or raw-feature graph) then refit
IU-PCR; smooth the finished IU score; or an exact closed-form residual ridge correction. Six graph
strengths (0.03 to 10), k=7 neighbours, on the 24 historical cells. IU-PCR, the equal-family mean and
signed DEEM-B3 were anchors. Two independent builds had to be byte-identical before labels were read.

**Evidence (@ 8829fd8e3, under `results/iu_graph_order_ablation_v1/releases/2026-08-24_frozen24_v1/evaluation/`):**
- `EVALUATION.json` (re-checked): `headline_status: D0_MECHANISM_ABLATION_NOT_INDEPENDENT_VALIDATION`,
  `label_selection_used: false`. Macro AUROC: IU-PCR 0.7761; equal-family mean 0.7810; residual ridge at
  0.03 0.7761 (+0.001pp, CI [-0.022, +0.020]). At strength 10 every arm is worse (for example feature
  smoothing on the residual graph 0.7110).
- `REPORT.md`: residual ridge at 0.03 is 0.515 points below DEEM-B3 [-0.782, -0.246] and 0.492 below the
  equal-family mean [-0.786, -0.194].
- `SCORE_AB_VERIFICATION.json`: pass = true (26 arms x 24 cells).
- `docs/experiments/IU_GRAPH_ORDER_ABLATION_V1.md` @ 8829fd8e3: the frozen plan and its claim limit.

**Its own status:** completed, negative: "neither smoothing X before IU nor the exact constrained residual
correction explains the DEEM-B3/equal-family advantage on frozen24." It points to equal-family balancing
as the simpler explanation.

**Next steps it left:** none registered. The implied follow-up (equal-family balancing as the
explanation) was **superseded** by the 2026-09-07 move to localization. This line did not edit HISTORY or
PROGRESS, so the experiment is recorded nowhere on the 13 tips except in this index.

**Needed by current research?** None. No tip mentions `iu_graph_order` or `residual_ridge_correction`.
Optional: bring the doc and the evaluation folder (11 files, 1,438,889 bytes) and write a short tagged
HISTORY entry, so the record sits in the tree. Do not bring the 16 differing older module copies.

**Restore:**
```bash
git tag archive/iu-graph-smoothing-ablation-v1 8829fd8e3
# optional:
git checkout 8829fd8e3 -- docs/experiments/IU_GRAPH_ORDER_ABLATION_V1.md \
  results/iu_graph_order_ablation_v1/releases/2026-08-24_frozen24_v1/evaluation
# full experiment: git checkout 8829fd8e3 -- results/iu_graph_order_ablation_v1 \
#   configs/reconstruction_benchmark_v1/iu_graph_order_ablation_v1.json \
#   spectral_utils/reconstruction_benchmark/iu_graph_order_ablation.py spectral_utils/reconstruction_benchmark/iu_graph_order_evaluation.py \
#   scripts/reconstruction_benchmark/run_iu_graph_order_ablation.py scripts/reconstruction_benchmark/evaluate_iu_graph_order_ablation.py \
#   scripts/reconstruction_benchmark/report_iu_graph_order_ablation.py scripts/reconstruction_benchmark/test_iu_graph_order_ablation.py
```

---

## 6. `origin/codex/deem-b3-moe-gating-v1`

- **Tip:** 877919734, 2026-08-27 19:08, "experiment: preserve DEEM B3 routing challengers" (Codex).
- **Unique commits vs the 13 tips:** 1 (877919734). Its parent 097e341a1 (2026-08-26) is the stale
  `origin/master` pointer. 097e341a1 is in 3 of the tips (readout-quickest, depth-feature, lsml-ct7-levers)
  but not in the SSL or decision-rule tips; those carry re-hashed copies of the CIW-DEEM commits
  (ea9eaafbe, 33c9d5a97, checked).
- 41 absent paths, all from 877919734 (about 900 KB). No results directory.

**What it accomplished.** It preserved the code for nine label-free challengers that try to improve
frozen DEEM-B3 on the 24-cell panel by routing between its feature groups: group mixture-of-experts
(rounds 1-3), residual mixture, residual PGRD, IU-PGRD boost, local-descent PGRD, pair router, crossed
"rook" correction. Each config has a frozen screen rule (equal-family AUROC delta at least +0.0025, sign-flip
p at most 0.05, at least 6 of 8 families, worst cell loss at most 0.02). It also includes one written
diagnostic, the B3 residual reliability ceiling audit.

**Evidence:**
- `docs/research_notes/deem_b3_residual_reliability_ceiling_2026-08-25.md` @ 877919734 (re-checked):
  verdict `NO_TRANSFERABLE_RELIABILITY_SIGNAL_AT_REQUIRED_SCALE`. The gate chosen on 8 screen cells gained
  +0.000526 there and -0.000067 on the other 16. Supervised per-cell ceilings +0.0037 to +0.0076 (an
  upper bound, not a method).
- `configs/deem_b3_*.json` @ 877919734: every router is marked retrospective development;
  "confirmation_requires_new_unopened_dataset_families".
- The official outcome of the CIW-DEEM family is on the current line: `results/ciw_deem_v1/RESULT.json`
  and `docs/experiments/CIW_DEEM_V1.md`, +0.00073 equal-family AUROC over B3, p=0.137, below the +0.0025
  promotion bar (from the drafts).
- **The per-router run outputs are not in git.** They were written to `local_cache/deem_b3_moe_v1/...`,
  which exists only on the second computer (not on this machine, checked).

**Next steps it left:**
- Localization and early-detection versions of CIW-DEEM. **Done** on the current line (66abed7a5,
  cfc3a259c; Steps 292-293).
- The routing challengers: **preserved, not continued.** Superseded in practice by the 2026-09-07
  localization priority; DEEM-B3 is a frozen comparator. No decision closes routing ideas as such.

**Needed by current research?** None of the 41 files. One shared dependency: 25 tracked files on the SSL
tip (for example `configs/ciw_deem_v1.json`, `scripts/diagnose_ciw_*.py`) name
`local_cache/deem_b3_moe_v1/` input and output folders. Those folders exist only on the second computer.
Ask there whether they still exist before treating the challenger scores or CIW inputs as lost.
Optional: bring in the ceiling note (3,696 bytes), the only written result.

**Restore:**
```bash
git tag archive/deem-b3-moe-gating-v1 877919734
git checkout 877919734 -- docs/research_notes/deem_b3_residual_reliability_ceiling_2026-08-25.md   # optional
# all 41 files: git diff --name-only 097e341a1 877919734 | xargs git checkout 877919734 --
```

---

## 7. `selector/a4-antigravity-unsupervised` (also on origin)

- **Tip:** 51c8964e1, 2026-08-07 12:55, "Step 186 (A4) results: antigravity selector bench run, c46+h16
  pools scored". One commit on 2b377bb39 (2026-07-18). It saved CSVs and a results note that had sat
  uncommitted in the worktree (part of Step 228).

**What it accomplished.** A parallel "antigravity" label-free feature-subset selector: choose features by
affinity to an anchor, choose the number of features by intrinsic-dimension rules, or by greedy
reconstruction (CSSP). Scored on the shared selector bench. The selector code and a corrected re-run
reached master in Step 189 (2026-07-18).

**Evidence:**
- `docs/research_notes/selector_bench_results.md` @ 51c8964e1: an older leaderboard. On repgrid-c46,
  GOOD_5 0.7328; A4 variants below it except two ties (0.7325, 0.7309); reconstruction 0.6775-0.6784.
- `results/selector_bench/a4_antigravity__c46.csv` @ 51c8964e1: the flawed "c46 ran all 51 cells" run
  that Step 187 identified. The current line has a corrected re-run.
- The 4 absent paths are Step-153 `results/subset_sweep/math500__*_T1.0.manifest.json` files. Main
  deleted them on 2026-07-25 (de8620929) and replaced them with `_T1.5` copies with identical values (the
  cell was really T=1.5). The `_T1.5` copies are on the rescue tip (checked for Qwen-Math-7B).

**Its own status:** superseded. Step 186: "no learned label-free selector beats the curated subsets".
Step 189 deliberately did not bring this branch's stale files. Later Steps 222 and 224 tested the
selector-search family further, all negative.

**Next steps it left:** none. Note: per CLAUDE.md ("tailor, never transplant"), Step 224 closed
transplanting published keep rules into this channel, not the ideas behind them.

**Needed by current research?** None.

**Restore:** `git tag archive/selector-a4-antigravity-unsupervised 51c8964e1`; originals at
`git show de023b350:results/subset_sweep/math500__Qwen-Math-7B_T1.0.manifest.json`.

---

## 8. `origin/codex/og-sml-agent-b-v1` (the September ref)

- **Tip:** 45f8b572e, **2026-09-04** 13:06, "review: add Claude's Joint L-SML v1 review + pre-registration
  concerns for Codex". Branch point 59359001e (2026-08-28).
- **Commits on the line:** e2123168d (08-29, 0.3662 program), ff8cd0d81 (08-30, Phases 0-2), 35a7a3a3d
  (08-31, H3 transfers), 250e092e1 (09-01, Phase 3 freeze), c5a658a6a (09-04, Joint L-SML structural
  study), 45f8b572e (09-04, the review). Only 45f8b572e is outside all 13 tips; c5a658a6a and earlier are
  in readout-quickest, depth-feature and lsml-ct7-levers (checked). The single absent path is
  `HANDOFF_JOINT_LSML_REVIEW_TO_CODEX_2026_09_04.md` (11,525 bytes).

**What it accomplished.** Codex ran a staged reasoning-localization program anchored to the historical
ProcessBench F1 of 0.3662. Phase 3 closed with no method promoted. Codex then built Joint L-SML (one
shared factor plus group factors over 23 telemetry streams) and ran a label-free structural study. Claude
reviewed it and raised three concerns to settle before registration.

**Evidence:**
- `results/og_sml_agent_b_v1/COMPLETE.json` @ 45f8b572e (re-checked): status `COMPLETE_T0_FALSIFIED_STOP_BEFORE_STEPS_0_6`.
- `results/joint_lsml_v1_r2/REPORT.md` @ 45f8b572e: Joint fit admissibly in 16 of 18 lanes; roster keeps
  23 of 28 streams.
- HISTORY @ 45f8b572e: Step 339 H3 .372663 against .3662, delta +.0064 [-.027, +.039], "PROMISING_UNCONFIRMED";
  Step 346 "PHASE3_DEVELOPMENT_CLOSED__NO_PROMOTION".
- HISTORY Steps 347-348 @ 48f9ef291: PRMBench Joint .669063 against IU-PCR .671539; ProcessBench F1
  .269290 against .340378, delta -.071 [-.085, -.054]; verdict "HARM__NO_PROMOTION" (numbers from the draft).
- The review's three concerns (tie-break by minimum ARI; freeze the hierarchical map with a 0.50 agreement
  abort; a 95%-of-folds admissibility rule) were adopted. The lsml-ct7-levers tip pins the review commit
  in `scripts/joint_lsml_localization/run_existing_v1.py:415` (`"claude_review_commit": "45f8b572e..."`, checked).

**Status:** Phase 3 closed with no promotion. Joint L-SML then harmed localization.
**Joint L-SML is discontinued for new arms (Omri, 2026-10-01)** - the only explicit closure. Its
historical results stay as records.

**Next steps it left:** the review's optional extras (three-parameter attribution ablation, unpruned-28
lane) were declined; **superseded** by the discontinuation.

**Needed by current research? Yes, as provenance.**
- The review file: a tracked registry pins its commit. Bring it into the tree, for example at
  `docs/archive/handoffs/HANDOFF_JOINT_LSML_REVIEW_TO_CODEX_2026_09_04.md`.
- HISTORY Steps 298-353: restore from **48f9ef291**, not from this ref (this ref has only Steps 298-346
  and 330-339). Add them as tagged blocks beside the Codex answer-only fusion steps that now use 298-336.
- Do not take this ref's `spectral_utils/joint_lsml.py`, `scripts/test_token_local_fusion.py` or
  `cluster/smoke_token_temporal_innovation_b3_v1.sbatch`; the tips have newer versions.

**Restore:**
```bash
git tag archive/og-sml-agent-b-v1 45f8b572e
mkdir -p docs/archive/handoffs
git show 45f8b572e:HANDOFF_JOINT_LSML_REVIEW_TO_CODEX_2026_09_04.md > docs/archive/handoffs/HANDOFF_JOINT_LSML_REVIEW_TO_CODEX_2026_09_04.md
git show 48f9ef291:HISTORY.md   # copy Steps 298-353 as tagged blocks; check the start and end boundaries by hand
```

---

## Material to bring into the consolidated tree

These are proposals. Paths are exact; the operator runs them on the consolidation branch.

**Required (current code or provenance points at them):**

| From | Paths | Size | Why |
|---|---|---|---|
| 483d4150c | `results/family_residual_graph_liu_v3/`, `results/family_residual_graph_liu_prmbench_v3/`, `results/family_residual_graph_liu_hle_v3/` | 21 files, 518,291 B | `scripts/build_advisor_update_aug21_2026.py` lines 32-36 fail without them |
| 483d4150c | HISTORY Steps 283-289 [graph line], tagged, from `git show 483d4150c:HISTORY.md` | text | Exist nowhere else |
| 48f9ef291 | HISTORY Steps 298-353 (`[reasoning localization]`, `[localization]`, `[per-answer windows]`, `[joint-lsml-v2]`), tagged | text | Removed by 9215ae69d; the only full copy |
| 45f8b572e | `HANDOFF_JOINT_LSML_REVIEW_TO_CODEX_2026_09_04.md` (to `docs/archive/handoffs/`) | 11,525 B | Pinned as provenance in `scripts/joint_lsml_localization/run_existing_v1.py:415` |
| `claude/readout-quickest-detection-v1`, `lsml-ct7-levers-run` | HISTORY Steps 390-421 (union) | text | Only 7 of 34 headings are on the SSL, decision-rule and rescue tips |

**Optional (records only; no code reads them):**

| From | Paths | Size |
|---|---|---|
| 1780572a2 | `docs/meetings/Advisor_Update_Aug25_2026.md`, `docs/meetings/Advisor_Update_Aug26_2026.md`, `docs/meetings/advisor_update_aug26_2026/`, `docs/experiments/REMOTE_WORKTREE_HANDOFF_2026-08-27.md` | 14 files, 106,721 B |
| 8829fd8e3 | `docs/experiments/IU_GRAPH_ORDER_ABLATION_V1.md`, `results/iu_graph_order_ablation_v1/releases/2026-08-24_frozen24_v1/evaluation/` | doc + 11 files, 1,438,889 B |
| 877919734 | `docs/research_notes/deem_b3_residual_reliability_ceiling_2026-08-25.md` | 3,696 B |
| 483d4150c | graph-line experiment docs and `results/graph_geometry_selection_research_v1/FINAL_REPORT.md` | small |

**Already in the tree (nothing to bring):** `docs/meetings/Advisor_Update_Aug27_2026.md` (rescue tip,
blob d7939062); the `_T1.5` subset-sweep manifests (rescue tip); the Aug 28 sent letter HTML (on 3 tips;
make sure one of them is merged).

**Outside git, on another machine:** `local_cache/deem_b3_moe_v1/` (second computer). Needed by 25
tracked CIW-DEEM config/script files and by any rerun of the DEEM-B3 challengers.

**Tags to keep every archived ref reachable:** one `archive/<name>` tag per ref, listed in each section.
Omri pushes tags.

---

## Other uncovered refs (not among the eight; listed for completeness)

From `uncovered_refs[]` in `results/consolidation_review_20261001/INVENTORY.json`. Details are in the
part-3 draft; only tips were re-checked here.

| Ref | Tip | Absent paths | Note |
|---|---|---|---|
| `backup-pre-lfs-fix` | 59f2d3501, 2026-08-07 | 2 | Whole-file GPQA raw pickles (6.17 GB). Not on Drive per `RECONCILIATION_2026-10-01.md`; queued in pass 2. Keep until verified (decision 6). |
| `backup/cumulative-vote-fusion-v2-before-publication-20260922` | eecc1055b, 2026-09-22 | 0 | Tree identical to the published bcf5a4bd8 (from the draft). |
| `origin/codex/consolidate-research-2026-08-19-lfs-backup` | 935bdb6ad, 2026-08-20 | 23 | Named as `recovery_ref` in `dataset_cache/DRIVE_BACKUP_2026_08_20.json`; do not delete. |
| `origin/main` (= `origin/HEAD`) | 3829eb0b4, 2025-10-08 | 25 | Unrelated 2025 L-SML package skeleton; GitHub's default branch. |
| `codex/combined-fusion-v1` | none | n/a | Never created on this machine; a suggested name in a 2026-09-14 handoff. The second computer is unchecked. |

---

## Where the drafts disagreed, and how it was resolved

1. **DEEM-B3 unique commits.** One draft said 5 unique commits from f7f7801aa, with the first four
   re-hashed onto the current line. The other said 1 unique commit on top of `origin/master`, already in
   the tips. Git: `origin/master` is a stale pointer at 097e341a1. 097e341a1 is in 3 of the 13 tips but not
   in the SSL or decision-rule tips, which hold re-hashed copies. Both were right for different tips. Against
   the union of the 13 tips, 1 commit is unique (matches the inventory).
2. **IU graph-order disposition.** One draft: archive without merging. The other: bring specific paths
   and write a HISTORY entry. Resolved: no current code needs it, so the proposal is archive with index,
   with an optional small restore as a record.
3. **Graph-geometry scope.** One draft: bring only the three v3 result directories plus HISTORY. The
   other: bring docs, code and all compact results. Resolved: the required set is the three directories
   (the only files current code reads) plus HISTORY; the rest is optional.
4. **Restore commands.** One draft used a shell brace glob for scripts that exist only in the tagged
   commit; that glob expands from the working tree and would miss them. Commands here list paths or use
   `git diff --name-only` instead.
5. **Tag names** differed between drafts (`archive/codex-...` against `archive/...`). This index uses
   `archive/<branch name without codex/>`; either form works.

---

## Not verified here

- Numbers quoted from the drafts that are marked "from the draft" or not marked "re-checked", including
  the ref 1 lane numbers, the Aug 27 letter's LEASH and RAGTruth figures, the CIW-DEEM +0.00073 result and
  the og-sml Steps 347-348 numbers.
- That the six ref 1 code fixes are byte-identical on the current line (one blob was checked by a draft).
- Whether `local_cache/deem_b3_moe_v1/` still exists on the second computer.
- Whether `codex/combined-fusion-v1` exists on the second computer.
- Byte identity of the GPQA whole files against the tips' `.part-00/01` copies.
