# Pre-registration — change-point readouts as the first-error rule

Claude, 2026-09-19. Branch `claude/token-probability-fusion-v1`.
Written **before** any number in this experiment was computed. Development-only:
the population is source-disjoint by fold but has been inspected before.

Two questions, one machinery.

**Q1 (finishing `docs/HANDOFF_MIND_THE_GAP_MECHANISM.md`).** The 2x2 of
{level, derivative} x {argmax, first-crossing} is missing its fourth cell, the one
that is actually theirs. Run it.

**Q2 (Omri, 2026-09-19).** Instead of taking the argmax of the per-step score series,
treat that series as a signal and ask a sequential change-point detector — CUSUM or
BOCPD — where the first error is.

Q2 subsumes Q1: first-crossing at a threshold *is* the simplest change-point rule, and
CUSUM/BOCPD are the same axis with memory. So both are run as one grid:
**statistic series x granularity x decision rule**, on a fixed roster, with one anchor.

---

## 1. Fixed before the run

**Population and metric.** The `localization_full_benchmark_v3` roster: 13,769 answers,
145,597 steps, 6,800 ProcessBench answers of which **4,442 are erroneous**. Primary
endpoint is **gate-free SLA** — exact first-error step hit rate on erroneous answers
only, per ProcessBench cell, then the mean over the eight cells. This is the
Mind-the-Gap protocol and removes the gate entirely; the gate decomposition is already
closed (handoff section 3) and nothing here touches it. Secondary: `within1`, and the
long-minus-short interaction (olympiadbench+omnimath minus gsm8k+math).

**Anchor.** No new number is printed until the published arm replays: C1 = the pooled,
fuse-before-readout token L-SML arm, mean gate-free SLA **35.92**, from
`results/token_probability_fusion_v1/STAGE_B_SCORES.npz`. Its already-measured
companions from the handoff table are reprinted beside every new cell.

**Uncertainty.** Paired source-group bootstrap, 10,000 draws, `_group_draws` from
`scripts/diagnostics/stage_b_2x2_v1.py`, seed 20260919. Every comparison is reported as
a paired interval against the C1 argmax arm; a point estimate alone is not a finding.

**Two facts about this population that constrain the design, measured from the roster
before any readout was written:**

- Step series are **short**: mean 7.9 steps per erroneous answer, median 7, max 47.
  A change-point detector run on ~8 observations has very little to work with. This is
  the main a-priori reason Q2 could fail for reasons that have nothing to do with the
  idea. Token series are **long** — mean 713 tokens — so every rule is run at **both
  granularities**, and the token-level arm is the one the idea deserves.
- **12.25%** of erroneous answers have their first error at **step 0** (17.9% on GSM8K).
  A rule that structurally cannot return step 0 forfeits up to 12 pp. Every readout here
  must be able to return 0, and the ones whose natural convention resists it (BOCPD's
  reset probability at t=0 is the hazard by construction) are reported with that
  convention stated.

---

## 2. The grid

**Statistic axis — what series.**

| id | series |
|---|---|
| `level` | the published C1 token L-SML fusion, Top-10 mean per step. The 35.92 arm. |
| `drop` | **their evidence series, rebuilt exactly**: `E = -H(P~)` over the **renormalized top-20** from the cached top-50 log-probs, EMA span **5**, `D_j = E_j - E_{j-1}`; risk `= -D` so that a *drop in evidence* is a *rise in risk*. |

`drop` is at token granularity by definition. Its token-to-step collapse is **undefined
in the paper** (digest, "genuinely underspecified") and is therefore **our**
pre-registered choice, reported as such and never described as theirs: (a) mean of the
**M = 5** most negative `D` inside the step, their M; (b) the single most negative `D`.
Both reported.

Constants `top-20 / EMA 5 / M 5` are theirs and are used unchanged so that this is a
test of their mechanism and not of my constants. The earlier derivative attempt used
EMA 16 and M = 3 on all eleven bank channels and so never tested it (handoff section 2).

**Granularity axis.** `step` — the rule runs on the T~8 step scores. `token` — the rule
runs on the answer's own token series and the alarm token is mapped to the step whose
span contains it. Level at token granularity is the fused token series before the
Top-10 readout; drop at token granularity is `-D` itself.

**Rule axis — how the step is chosen.**

| id | rule |
|---|---|
| `argmax` | ours. Where the series is largest. |
| `first_q` | theirs, adapted. First index whose value reaches the within-answer quantile `q` of that answer's own series. Label-free: the threshold is computed from the answer alone. `q` in {0.5 ... 0.95}; **q = 0.90 pre-registered as the default.** As `q -> 1` this degenerates to argmax by construction. |
| `cusum_alarm` | one-sided Page CUSUM on the within-answer z-scored series: `S_t = max(0, S_{t-1} + z_t - k)`, alarm at the first `S_t >= h`. Prediction = the alarm index. |
| `cusum_onset` | same recursion, prediction = the **start of the excursion** — the last index at which `S` was zero, plus one. The textbook Page-Hinkley change-point estimator, and the one whose semantics are "where the drift began" rather than "when I was sure". |
| `bocpd_rise` | the project's verified reset-before-observation Gaussian BOCPD (`docs/reviews/bocpd_boundary_audit_2026-09-07.md`; the old `temporal_models.bocpd_gaussian` is **not** used — it mixes conventions). Onset curve `= P(reset at t) * max(z of the surprise against the continuation predictive, 0)`; prediction = its argmax. |
| `bocpd_reset` | argmax of `P(reset at t)` alone. Reported with the t=0 convention stated. |

**Pre-registered defaults, chosen before any result:** `k = 0.5` and `h = 5` — the
textbook Page constants for a one-SD shift, not tuned here; BOCPD `hazard = 1/32`,
which is the bank's own `BOCPD_HAZARD_LAMBDA = 32.0`, `observation_variance = 1`,
`prior_variance = 1` on a z-scored series. A grid over `k` in {0.25, 0.5, 1.0},
`h` in {1, 2, 3, 5, 8}, `hazard` in {1/8, 1/32, 1/128} and the full `q` ladder is run as
well and its maximum is reported as a **label-selected ceiling, not a candidate**,
exactly as the LOCO-5 threshold sweep is.

**No-alarm fallback** (pre-registered, because it is not neutral): when CUSUM never
crosses `h`, predict `argmax_t S_t` — the point of greatest accumulated evidence, which
is the statistic's own answer, not the raw series' answer. The variant that falls back
to `argmax` of the raw series is reported beside it as a sensitivity so that the
fallback cannot silently carry the result.

---

## 3. Predictions, and what falsifies them

**P1 — the pairing hypothesis** (handoff section 7, restated in its intent). If the gain
lives in the *pairing* of a derivative statistic with a first-crossing rule, then
`drop x first_q` beats **both** single-axis failures (`level x first_q` ~ 30.04 at
q=0.8; `drop x argmax` = 27.96 under the old constants) **and** beats `level x argmax` =
35.92, with the paired interval against C1 excluding zero.
**Falsified if** `drop x first_q` does not beat `level x argmax` with the interval
excluding zero. In that case the pairing hypothesis is wrong and this line is finished,
which is the outcome the handoff commits to accepting.

**P2 — the length hypothesis.** Their advantage grows with chain length and ours does
not (ratios to chance 2.34/1.85/2.27/2.22 ours, 2.15/1.79/3.12/2.71 theirs). If the
mechanism is real, the best drop/change-point cell has a **positive long-minus-short
interaction against C1**, interval excluding zero.
**Falsified if** the interaction is zero or negative — then whatever the mean does, the
mechanism is not the one their Table 3 shows.

**P3 — Omri's change-point hypothesis.** On the **same series as the published arm**,
at least one change-point rule (`cusum_alarm`, `cusum_onset`, `bocpd_rise`,
`bocpd_reset`) at its pre-registered default beats `argmax`, paired interval excluding
zero.
**Falsified if** every change-point rule at its default is at or below argmax on the
level series. A grid maximum that beats argmax does **not** rescue P3; it is a ceiling.

**P4 — the granularity hypothesis** (mine, stated so it cannot be claimed afterwards).
If the step series is too short for sequential detection, the token-granularity arm of
each change-point rule beats its own step-granularity arm.
**Falsified if** token granularity is at or below step granularity for every rule.

**Prior belief, recorded so the result can contradict it:** P3 at step granularity is
unlikely on an 8-point series; P4 is the one I expect to carry any gain there is; P1 I
give better odds than the two closed single-axis cells because the paper's own numbers
are strongest exactly where our detector is weakest.

## 4. What this experiment is not

It is not a reproduction of Table 3 — SLA's token-to-step collapse is undefined in the
paper, so any implementation is an **adaptation**, labelled as one, carrying no author's
name (project rule: *tailor, never transplant*). It is not a candidate for the method of
record: nothing here changes CT7, the gate, or any frozen release. It has no untouched
confirmation set; every number is development evidence.
