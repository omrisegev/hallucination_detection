# Handoff: experiments from two Mor Geva 2026 papers (BonaFide; Thinking to Recall), local + AIRCC

Written 2026-10-04 by Claude (cloud session) for the next agent, who runs locally with the TAU VPN up and
AIRCC access. Base: `origin/claude/ssl-pseudolabel-residual-v1` (553de79c), the newest main line per
`docs/consolidation/RESEARCH_HANDOFF_2026-10-01.md`@rescue.

**Authorization.** Omri asked on 2026-10-04 for these experiments to be prepared and run, including on the
cluster. That is the "go" that decision 21 requires for *these* lines only. It does NOT waive
`docs/AGENT_OPERATING_CONTRACT.md` section 1: every `sbatch`, every GPU job, every Drive copy and every
`git push` is still a HUMAN_CHECKPOINT (ask Omri, then wait). The external-collection gates in
`docs/experiments/LSML_EXTERNAL_GENERALIZATION_V1.md` (timing job <= 12 examples and <= 1 GPU-hour, then
Omri's full-run budget decision from measured costs) apply to every GPU stage below.

**Not visible to the author.** This was written from GitHub + Drive only. Nothing was pushed after
2026-10-02, so the runs Codex is completing and the alternatives Claude is exploring locally were NOT seen.
Before anything else: read their worktrees/line statuses, and check that nothing below collides with them
(shared output dirs, step numbers, the same AIRCC allocation, or use of the new benchmark for selection).

---

## 0. Sources

| Paper | Where | Read? |
|---|---|---|
| Gur-Arieh, Marasović, Geva. *Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth* (BonaFide), arXiv 2605.25052 | Drive PDF `2605.25052` (file id `1NHYNai1JwHQNZ97xI8A7ehVjdhbiAYjX`); code https://github.com/yoavgur/BonaFide ; data HF `yoavgurarieh/BonaFide` (collection `huggingface.co/collections/yoavgurarieh/bonafide`) | Full text read, incl. appendices |
| Gekhman, Aharoni, Ofek, Geva, Reichart, Herzig. *Thinking to Recall: How Reasoning Unlocks Parametric Knowledge in LLMs*, arXiv 2603.09906 (COLM 2026) | arXiv; Google Research blog | Abstract/blog only. Upload the PDF to Drive and read it before E3 |

Facts used below (BonaFide, verified in the full text):
- Two label settings. *Outright*: 10 procedurally generated tasks (nested_mod, collatz, digit_square,
  prime_chain, stoichiometry, cipher, paragraph_analysis, text_extraction, tournament, graph_traversal;
  spec in their Appendix A.1, Table 2) whose correct answer requires known "bottleneck" intermediate
  computations; without CoT models solve them 1.5% of the time. *Diversionary*: SimpleQA / HLE / DDXPlus
  questions plus a hint to a plausible wrong answer (6 hint formats, Table 3); unhinted models give the hinted
  answer 0.9% of the time.
- 10 open models: Qwen3-4B-{Instruct,Thinking}-2507, Qwen3-30B-A3B-{Instruct,Thinking}-2507,
  Olmo-3-7B-{Think,Instruct}, Olmo-3.1-32B-{Think,Instruct}, Llama-3.3-70B-Instruct, DeepSeek-R1-Distill-Llama-70B.
- Balanced benchmark: 1,946 step labels (51% faithful) + 1,120 CoT labels (15% faithful). Unfiltered release:
  19,459 labels over 9,302 CoTs. Step types: misattribution 901, bottleneck execution 573, faithful
  commitment 428, tool call 44. Labels from a Gemini-3-Flash retrieval judge + Gemini-3-Pro validation judge,
  98.9% precision vs human annotation.
- Metric results: most near chance; best CoT-level CC-SHAP 0.70, best step-level Filler Tokens 0.59; strong
  skew (importance metrics 90-96% "unfaithful", semantic metrics 94-96% "faithful"); importance metrics
  correlate (kappa 0.18-0.35), others ~0; CC-SHAP step AUROC 0.31-0.45 on most models but 0.69 on
  Qwen-4B-instruct (sign flips across models); LM-judge skyline 0.82-0.87, generic LM judge ~0.67-0.68.
- The public code ships the metric suite and labeling pipeline but NOT the outright task generators; rows
  carry `ground_truth_steps`.

---

## 1. Three experiments, in priority order

| ID | What | Value for the project | GPU? | Blocked by |
|---|---|---|---|---|
| **E1** | `proc_firsterror_v1`: a procedural, judge-free, *sealed* first-error benchmark in the BonaFide outright style | Untouched confirmation (handoff entry 11), controlled prevalence and error position, task-type holdout for transfer (entry 12) | Yes: generation + teacher-forced telemetry, Qwen3-8B | Omri's go per sbatch; budget decision after timing |
| **E2** | `bonafide_telemetry_v1`: teacher-forced telemetry on BonaFide's released CoTs, scored by the model that generated them | Exploratory panel: do our token channels flag fabricated attributions (unfaithful by commission)? | Yes: forward passes only | Same; new model pins |
| **E3** | Trajectory selection in the style of Thinking to Recall | An actionable evaluation (pick the clean trajectory, measure accuracy gain), not just AUC | Only if entry 16 option 2 runs | Omri choosing option 2 of entry 16 |

Do E1 stage A (CPU) first. E2 can share AIRCC sessions with E1 but must not delay it. E3 is conditional.

---

## 2. E1 - `proc_firsterror_v1` (primary)

### 2.1 Question and frozen predictions
Q: Does the frozen label-free candidate (DS filter pi_hat > 0.5, then plain average; bank chosen BEFORE the
test, see `results/algorithm_decisions_v1/SUMMARY.md`) rank the first erroneous step on fresh, uncontaminated,
judge-free data as it does on PRMBench/ProcessBench?

Write predictions into `PROTOCOL.json` before generation, e.g.:
- P1 within-answer AUC of the candidate above the step-index-only baseline on the position-balanced subset.
- P2 DS prevalence estimate vs the true error-step prevalence (known by construction), per difficulty level.
- P3 ranking of {candidate, CT7, bank11 L-SML, step index} on held-out task types vs on the source panel
  (entry 12 proxy; label "proxy validation, not confirmation").

### 2.2 Stage A - CPU only, no experiment (do this first)
New code; do not edit frozen modules in place:
- `spectral_utils/procedural_tasks/` - generators for **5 task types**: collatz, nested_mod, digit_square,
  graph_traversal, cipher. They give numeric/string intermediate states that are easy to parse. Each
  generator returns `(question, answer, gold_trace: list[state])` from a seed and a difficulty parameter.
  Follow BonaFide Appendix A.1, Table 2. Parameters must make guessing implausible (large answer space).
- `spectral_utils/procedural_tasks/labeler.py` - parse each model step (split rule below), extract the
  states it asserts, align to `gold_trace`. Output per answer: `first_error_step` (index, or -1 if all
  asserted states are correct and the final answer is correct), plus flags `omitted_bottleneck`
  (a required state never asserted) and `unparseable_step`.
- **Step segmentation is part of the protocol.** Instruct the model to write one computation per paragraph;
  split on blank lines (ProcessBench convention). Freeze the rule before generation.
- Tests `tests/test_procedural_tasks.py`: determinism per seed; gold traces match an independent reference
  implementation; the labeler finds injected errors at known positions in synthetic CoTs, including paraphrased
  numbers ("thirty-four", "34.", "=34") and corrections ("wait, 3*11+1 = 34, not 33").
- `results/proc_firsterror_v1/PROTOCOL.json`: tasks, difficulty grid, seeds, n, prompt text, segmentation
  rule, label rules, exclusion rules (what happens to `omitted_bottleneck`/`unparseable` answers, decided
  now), bank and method frozen, metrics, predictions, git commit.
- Independent pre-run review (subagent red team, as in Steps 457-462) of generators + labeler + protocol.

**Stop A**: report to Omri (Hebrew summary) and ask for the timing sbatch.

### 2.3 Stage B - AIRCC timing jobs (HUMAN_CHECKPOINT each)
1. **Generation timing**: Qwen3-8B, same pinned revision as the telemetry protocol, `math_prompt`-style
   instruction + ` /no_think` (consistent with the external collector; the thinking variant is a separate arm
   only if Omri asks). <= 12 deterministic examples per task, across difficulty levels, including the longest.
   Reuse the checkpoint/resume pattern of `cluster/run_inference.py` + `submit_inference.sbatch`
   (SIGTERM-safe, preemption-safe). `submit_inference.sbatch` is local-only and holds an HF token: never
   commit it or copy it into a new file; write a token-free sbatch for this line. Temperature 0 for the benchmark (one answer per question).
2. **Difficulty calibration**: using ONLY final-answer correctness from the timing/pilot run (never step
   labels, never method scores), pick per-task difficulty so the answer error rate is ~30-60%. Record this
   choice in the protocol before full generation. It is allowed because it inspects no method output.
3. **Telemetry timing**: `cluster/run_external_telemetry.py --mode smoke` on the generated answers. Write a
   new adapter `adapt_procedural` (new versioned module next to `spectral_utils/external_generalization/
   contracts.py`; do not edit the frozen module) producing `answers.json` with step char spans, and a
   separate `evaluator_only/` file with the labels, exactly like `scripts/prepare_external_inputs.py`.
   Telemetry is self-scoring: the generating model = the scoring model.
4. `scripts/estimate_external_collection.py` gives GPU/CPU hours, storage, wall time and retries.

**Stop B**: send Omri the measured estimate and ask for the full-run budget decision. Suggested size:
5 tasks x ~300 answers = ~1,500 answers (similar to Hard2Verify + part of Socratic); scale only by his decision.

### 2.4 Stage C - full run (after budget approval)
- Full generation, then labeling **on the cluster or in a private dir**. Labels go to
  `scratch/proc_firsterror_private/evaluator_only/` (add it to `.gitignore` first, next to the
  `scratch/external_generalization_private` rule; check with `git check-ignore`) and a `SEAL.json` with SHA-256 is committed.
  **Nobody opens labels or quality numbers before the method is frozen** (bank, DS filter, readout, threshold
  rule). Tell the Claude/Codex agents exploring alternatives that this set is sealed and NOT for selection.
- Full telemetry; `scripts/extract_external_banks_v4.py` for the bank features (check that the extractor
  accepts the new adapter; if it needs changes, version it).
- Score with the frozen candidate through the `algorithm_external_v1` path
  (`scripts/{fit,score,evaluate}_algorithm_external_v1.py`), fitted per dataset without labels, as decided in
  Step 462. Seal predictions (`PREDICTIONS_UNSEALED.json` + `SEAL.json` pattern), then unseal labels.
- Parser validation before unsealing quality: an independent agent hand-checks a random 50 answers per task
  against the labeler (agreement on `first_error_step`). Gate: >= 95% exact agreement, else stop and fix the
  labeler (labels only; no method change).

### 2.5 Reporting (all with source file + hash, per the operating contract)
- Within-answer AUC, any-error hit, official-style PRMScore, ProcessBench-style first-error F1; 95% CIs
  (bootstrap by answer; DeLong for AUROC).
- Baselines: step index alone, CT7, bank11 L-SML, plain average without DS filter.
- Position: AUC on a subset with first-error index balanced across positions; error-position histogram.
- Prevalence: DS pi_hat vs truth, per task and difficulty.
- Length: AUC vs answer length (BonaFide found most metrics degrade on long CoTs).
- Skew: share of steps flagged at the frozen threshold on a label-balanced subset.
- Transfer proxy: leave-one-task-type-out ranking of methods vs the full-panel ranking.
- Omission answers reported separately, never dropped silently.

### 2.6 Risks
- Qwen3-8B may skip verbalizing states (omission). That is why `omitted_bottleneck` exists. If more than ~20%
  of answers are unlabelable, stop and report rather than loosening the parser.
- Procedural arithmetic is a narrow domain. Claim "untouched confirmation on procedural reasoning", not
  general transfer.
- The digit features ride on numeric text. Report the bank with and without digit features (decision
  2026-09-30 keeps them in the line; the 2026-09-17 exclusion still applies to gates/anchors).

---

## 3. E2 - `bonafide_telemetry_v1` (exploratory, secondary panel)

- Data: download `yoavgurarieh/BonaFide` (balanced) and the unfiltered release on the cluster or locally; pin the
  HF revision SHA in `SOURCES.json`. Keep it private, as with Hard2Verify.
- Restrict to rows generated by **Qwen3-4B-Instruct-2507, Qwen3-4B-Thinking-2507, Qwen3-30B-A3B-Instruct-2507,
  Qwen3-30B-A3B-Thinking-2507**. The labels describe the *generating* model's computation, so the telemetry must
  come from that same model. Add the four models with pinned revisions to a new protocol;
  prefetch with `cluster/prepare_external_model.sbatch`.
- Reconstruct each model's exact prompt (BonaFide `generation/backends.py::build_prompt` and
  `generation/thinking.py` for the `<think>` split; their hint text is in the prompt) and teacher-force prompt +
  CoT + answer verbatim. This needs a new collector mode or a versioned runner: the current collector hard-codes
  `math_prompt + /no_think`. Keep the alignment gate.
- Map BonaFide `step_span` (character offsets into the CoT, sentence-level) to token spans.
- Labels: step level, positive = misattribution + tool call (unfaithful), negative = faithful commitment +
  bottleneck execution; CoT level as released. Also report misattribution vs faithful commitment alone
  (same diversionary questions, so a cleaner contrast).
- Metric: AUROC with DeLong CI per model and pooled; skew; length sensitivity. Compare with their metrics only as
  published context. Their LM-judge skyline is not a fair comparator (same signal as the labels).
- Scope statement for every table: "faithfulness labels, not correctness; exploratory; scored by the generating
  model". Never pool with PRMScore panels.
- Cost: forward passes only, a few thousand CoTs; 30B-A3B is MoE and fits on one B200. Timing job first,
  same gates.

---

## 4. E3 - trajectory selection (conditional on entry 16 option 2)

Only if Omri chooses option 2 of entry 16 (sampled own answers, temperature 1, ~3 per question, Qwen3-4B and
Qwen3-8B, GSM8K and MATH; `docs/line_status/claude__self-generated-step-labels-v1.md`@self-labels). Then add,
at zero GPU cost: per question, select the sample with the lowest maximum step-risk under the frozen detector
and report final-answer accuracy vs a random sample, majority vote, and the oracle. This mirrors Thinking to
Recall's "prioritize hallucination-free trajectories". Read that paper fully first and record whether its
audited traces (per-fact correctness labels) are public. If they are, propose (do not run) a factual-reasoning
panel to Omri.

---

## 5. AIRCC essentials (from `cluster/README.md` and the external runbook)

- Login `omrisegev1@slurm-login.iucc.ac.il` (TAU VPN), ssh alias `aircc`. Work under
  `/shared/cycle2_tau_averbuch_prj/omrisegev1`, never `$HOME`.
- Every Slurm call: `ssh aircc 'export SLURM_CONF_SERVER=controller-primary; <cmd>'` or `bash -lc`. Without it
  you get "Could not establish a configuration source", which is not an outage.
- Account per the external runbook (2026-09-24): `cycle3_tau_averbuch_prj`, QoS `owner_940`, partition
  `power-gpu`. `cluster/aircc.env` still says `owner_880`, which is stale. Confirm with `sdata` before submitting.
- B200 needs the NGC image `nvcr.io/nvidia/pytorch:25.01-py3`; never pip-upgrade torch. Offline wheelhouse
  `wheelhouse/external_20260924`; `HF_HUB_OFFLINE=1` on compute nodes; download models via the prefetch job.
- Immutable `--chdir` snapshot per commit, `SYNC_COMMIT.json`, no dirty-tree full runs. Package the full
  `cluster/` and `spectral_utils/` trees and run the CPU smoke from the extracted archive (lesson of job265865).
- One writer per output dir; SIGTERM completes the current record and exits 85; no automatic requeue for
  telemetry.
- Drive: cluster rclone is `/shared/cycle2_tau_averbuch_prj/omrisegev1/bin/rclone --config
  /shared/cycle2_tau_averbuch_prj/omrisegev1/.config/rclone/rclone.conf`; `copy` only, to
  `gdrive:hallucination_detection/cluster_results/<line>/<run>`; verify hashes. HUMAN_CHECKPOINT.
- Never commit plaintext private data or labels; sealed hashes only.

---

## 6. Bookkeeping

- Branch per line: `claude/proc-firsterror-v1` (E1), `claude/bonafide-telemetry-v1` (E2), both from this branch.
- HISTORY.md: take the next free Step number across ALL refs (Step 463 already collides between two lines;
  464 is used by the estimator line). Tag `[Claude, proc_firsterror_v1]` etc.
- Add a line status file under `docs/line_status/` per line, and a row in `RESEARCH_EVIDENCE_LEDGER.md` after
  each completed stage.
- Stop points for Omri (Hebrew summary each time): end of stage A; after timing (budget question); after method
  freeze, before unsealing; final results.
- Anything not covered here, or any gate firing: stop and ask (operating contract section 3).
