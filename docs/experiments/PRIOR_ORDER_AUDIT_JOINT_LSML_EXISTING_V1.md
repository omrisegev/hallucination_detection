# Prior-order audit — Joint L-SML on existing localization data

Status: `PASS_FOR_RETROSPECTIVE_DEVELOPMENT_ONLY`

This audit records the scope change explicitly authorized by the user on
2026-09-04: run one frozen Joint L-SML candidate on the existing Qwen
ProcessBench and PRMBench populations before deciding whether a new-population
generalization run is warranted.

The authorization is narrower than a new benchmark campaign:

- the result is `RETROSPECTIVE_OPENED_DEVELOPMENT`, never confirmation,
  promotion, a new-leader claim, or evidence of generalization;
- one candidate is allowed: the frozen active-23 hierarchical Joint L-SML map;
- three matched controls are allowed: active-23 IU-PCR, equal-family, and
  fixed-family continuous L-SML;
- no feature, sign, group count, weight map, reducer, threshold rule, or arm may
  be selected after outcome access;
- ProcessBench and PRMBench are fitted and reported separately;
- the score process may read only sanitized raw telemetry, opaque row IDs, and
  outcome-free step spans; labels are opened by a separate evaluator only after
  a hash-bound score freeze and independent audit;
- the ProcessBench reducer is fixed top-ten mean, not top-five and not top ten
  percent; PRMBench uses maximum token risk within each official step span;
- a structural failure remains `STRUCTURAL_NO_SCORE`; it may not trigger an
  unregistered fallback or algorithm edit;
- no generation, cluster run, model download, prevalence estimator, LAG,
  overlap, DUFS, Katz, rank transform, multi-threshold expansion, or new
  monotonic transform is in scope.

This is consistent with `CLAUDE.md` and `PROGRESS.md`: opened populations may
be used for honest retrospective development, but not for generalization or a
new leader. It also preserves the historical warning in `HISTORY.md` against
label-selected subset/reducer sweeps. The previous no-scoring plan remains the
record for the future fresh-population experiment and is superseded only for
this bounded development run.

Git lineage is frozen from Agent B commit
`c5a658a6ae24bf1063c40b8a60e0b9adaafb87b2` on the isolated branch
`codex/joint-lsml-localization-eval-v1`. No commit, push, or cluster execution is
implied by this audit.

Claude's independent review was read from remote commit
`45f8b572e221164ff6ebe3fe9fff96c25828a49d`. Before registration it motivated
three label-free refinements to the one candidate: minimum-ARI precedes smaller
K when median/mean stability ties; LOAO admissibility uses a frozen 95% held-fold
quantile rather than an all-fold intersection; and donor-score agreement across
the four structural weight maps is gated at Spearman 0.50 while the
hierarchical map remains irrevocably primary. The suggested unpruned arm, 3p
attribution ablation, alternative maps, and post-hoc decomposition were not
added because they would expand the efficacy/multiplicity question.
