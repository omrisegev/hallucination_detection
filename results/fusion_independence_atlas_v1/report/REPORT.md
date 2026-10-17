# Fusion Independence Atlas v1

Status: **DEVELOPMENT_PARTIAL** (development only; no external confirmation).

## Coverage

- Historical unique score arrays retained: 178
- Historical unique peak vectors retained: 173
- REPORT_ONLY archives: 9
- Factorial arms declared: 32
- Factorial arms evaluated: 0
- Factorial status: BLOCKED_COMPOSITION_CONTRACT
- Pairwise classifications: {'UNRESOLVED': 2648, 'INDEPENDENCE_COMPATIBLE': 230, 'REDUNDANT': 1}
- Independence-compatible groups: 0
- Dependent-complementary diagnostics: 38
- Five-point backend status: PARTIAL_FAIL_CLOSED
- PENDING_EXPANSION: FM, DiFlo, DOT, artifactless_models

## Frozen baseline replay

- original4: 37.4749% PB
- innovation5: 39.8314% PB
- digit025: 41.3300% PB / 0.776036 within
- current: 43.2546% PB / 0.776036 within
- Replay status: PASS before development labels were opened downstream.

## Candidate funnel

- Screened definitions: 5443
- Unique extraction signatures: 5345
- Exact duplicates removed: 98
- Outer-training family representatives: 114
- Representatives by insertion point: {'answer_gate': 22, 'background': 15, 'decoder': 29, 'step_post_readout': 24, 'token_pre_readout': 24}

## Independence result

- Combined compatibility edges by insertion point: {'answer_gate': 0, 'background': 0, 'decoder': 0, 'step_post_readout': 0, 'token_pre_readout': 0}
- No pair passed all required error views at any insertion point; therefore no independence-compatible clique of size 2-6 exists.
- Raw compatible pairs by individual matrix: {'decoder:prmb_pairwise_misorder': 162, 'step_post_readout:prmb_pairwise_misorder': 34, 'token_pre_readout:prmb_pairwise_misorder': 34}
- The compatible raw pairs occur only on PRMB pairwise misordering. Every PB raw-locator matrix has zero compatible pairs under the simultaneous phi and odds-ratio intervals.
- Background predictor residual correlation ranges: H0lim=0.684-0.996, VE0=0.695-0.998, VE0.75=0.821-1.000, VE1=0.838-1.000.

## Interpretation

Independent-compatible, dependent-complementary, redundant, and unresolved candidates are reported separately. Missing bulk artifacts are not silently promoted.

The frozen baseline replay passed before fusion comparisons were accepted.

## Nested OOF finalists

- step_post_readout: `['step::digit.token_clock_innovation::top1']` with `singleton`; stability 4/5, 42.0047% PB / 0.600821 within; promotion `NO_SUCCESSOR`.
- token_pre_readout: `['token::digit.token_clock_innovation::top1']` with `singleton`; stability 4/5, 42.0047% PB / 0.600821 within; promotion `NO_SUCCESSOR`.
- Unsupported points: {'roster_stability': 'no model recurred in at least 4/5 folds for: answer_gate, background, decoder'}
- No performance successor or independently justified simplification successor was found.

## Dependent-complementary research candidates

These rows failed the independence contract. They are diagnostics, not promotion-eligible winners.

- step_post_readout: `['step::digit.disagreement::top2', 'step::step395.logtail15::top10']` via `equal_rank` -> 43.3671% PB / 0.776019 within; supported folds [0, 1, 2, 3, 4]; unique held-fold successes 1445.
- answer_gate: `['answer_gate::digit_rate', 'answer_gate::raw_neglogp1::token_top10']` via `equal_rank` -> 43.3094% PB / 0.776068 within; supported folds [0, 1, 2, 3, 4]; unique held-fold successes 1591.
- answer_gate: `['answer_gate::digit_rate', 'answer_gate::q15_VE0.75::token_q90']` via `equal_rank` -> 42.9144% PB / 0.775093 within; supported folds [0, 1, 2, 3]; unique held-fold successes 1335.
- answer_gate: `['answer_gate::q15_VE0.75::token_q90', 'answer_gate::raw_neglogp1::token_top10']` via `equal_rank` -> 42.7005% PB / 0.775093 within; supported folds [0, 1, 2, 3]; unique held-fold successes 982.
- answer_gate: `['answer_gate::digit_rate', 'answer_gate::raw_neglogp1::mean_step_top10']` via `equal_rank` -> 42.6143% PB / 0.776058 within; supported folds [0, 1, 3, 4]; unique held-fold successes 1303.

Highest-PB diagnostic (explicit tradeoff): `['step::digit.token_clock_innovation::top1', 'step::q15.VE0.75.prefix_mean_innovation::top10']` -> 44.7409% PB / 0.715510 within.

## Tail15 finding

- The strongest near-incumbent simplification signal is digit-disagreement Top2 + Tail15/logtail15 Top10 equal-rank: 43.3671% PB / 0.776019 within, stable in 5/5 folds.
- Tail15 is therefore complementary to digit in this dataset, but not statistically independent under the locked screen.

## Open composition contract

The five insertion points were evaluated independently, but stable finalists were found only for token and step. The repository also does not define a type-safe rule for composing alternative locator representations into 32 cross-point arms. The factorial is therefore blocked rather than fabricated.

Missing stable finalists: `['background', 'decoder', 'answer_gate']`.
Reason: `UNRESOLVED_PIPELINE_COMPOSITION_NOT_FABRICATED`.
