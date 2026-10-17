# Frozen-score localization forensics — Step 313

This is a post-evaluation diagnostic on the same 110 development answers as
Steps 306–312. Correctness labels and text are deliberately inspected. It does
not fit a new detector, select a winning setting, change the benchmark, or
establish an untouched test result. IU-PCR / Joint L-SML remain the core.

Decision question: is there a concrete alignment, window-to-step, peak, or
no-error failure that should guide the next bounded fusion experiment?

## Fixed scope

- Retain all 110 original IDs, corrected source groups, and targets. Read the
  original pickle metadata for Qwen3-8B, including steps, token IDs, alignment
  spans and the three scalar telemetry streams. Discard NumPy payloads; do
  not describe this as verification of all top-K-derived measurements.
- Re-tokenize the exact double-newline-joined step text using the available
  local Qwen3-8B tokenizer. Require exact token-ID equality before treating
  its character offsets as an independent alignment check. Record tokenizer
  hash/revision; matching these answers does not establish original revision.
- Check every official span, label and the three primitive streams against
  the original extracted inputs. Record separator coverage and overlaps.
- Audit 16 frozen fusion/control trajectories: seven original routed cores,
  moment IU, context IU/equal, and IU/Joint-graph for AR/last/EMA augmentation.
  Preserve all 98 historical metric rows as context, not 98 new experiments.
- Replay the actual overlap-average token projection and maximum per step.
  Independent direct summation uses 1e-12 tolerance; serialized scores and
  decisions retain their exact values. Distinguish exact ties, 1e-12 numerical
  ties, small margins (<=0.1 original fit-score SD), and shared window support.
- Decompose PB clean decisions, raw first-error peaks, errors hidden by the
  gate, and wrong locations with an open gate. Compare original/augmented
  decisions with paired answer IDs. Log GMM BIC differences, but do not fit
  a new cutoff. Report step length and overlapping-window opportunities.
- Label any tie-membership or perfect-gate calculation as an oracle
  diagnostic, not an attainable score. No label-chosen tie-breaker is a method.
- Keep full text and per-step score arrays in a local interactive HTML. Choose
  a few illustrated cases by explicit diagnostic categories and stable IDs;
  expose all answers so examples cannot substitute for aggregate evidence.

## Review and execution

No new LLM inference, external data transfer, or Claude-worktree edits. CPU
only, single bounded pass plus review. Freeze source/input hashes before
diagnostic execution; checkpoint extracted metadata separately. Meaningful
synthetic tests cover shared boundary plateaus, disjoint equal scores,
overlapping end windows, invalid coverage, and token/character overlaps.
Review raw joins, alignment through the existing independent alignment API,
all audited score maps and PB counters, all historical metric bundles, and
the report's local links/data. Identify shared kernels in the review.

Next work depends on actual findings. A duration/step-support readout, a
boundary-aware window map, or a different no-error model is only a hypothesis
until frozen with incumbent and matched simple controls and evaluated on both
benchmarks. Preserve the full comparator, corrected-fold refit, named-method,
untouched confirmation and historical 24-cell transfer backlog.
