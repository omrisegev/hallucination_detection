# Thesis Pivot Options — Critical Assessment + Pilot Results

**Replies to**: [`thesis_pivot_options.md`](thesis_pivot_options.md) (Gemini research session, Jul 2026)
**Branch**: `experiment/pivot-alternatives` · **Step**: 151 · **Status**: pilot complete, results in §7–8

---

## 0. TL;DR verdict table

| Option (from the pivot doc) | Doc's framing | Audit | Verdict |
|---|---|---|---|
| 1. KalmanNet | Neural filter tracks a latent "groundedness state" | Gray-box ✓, but unsupervised innovation loss unproven for *discrimination*; risk of collapsing into "unsupervised HALT" | **Precursor first** — AR/Kalman innovation scores in Track B are the go/no-go. See §7 for the call. |
| 2. LOCA | Manifold embedding, hallucination = "escaping the manifold" | LOCA requires measurement **bursts** for local covariance — we have none; without them it degrades to a plain AE. "Train on correct traces" = weak supervision. | **Defer.** Advisor-lineage appeal is real; technical fit is forced. |
| 3. Diverging Flows | Flow on hidden states, transport cost diverges off-manifold | **White-box** — we cache zero hidden states; abandons the entropy-trace signal that differentiates us from FUSE | **Reject as written; idea salvaged** as gray-box density scoring on the feature space (Track A: gmm/kde). |
| 4. PRAE | Robust AE excludes outliers from training | Genuine fit: our unlabeled pools ARE contaminated (7–80% wrong per cell) and PRAE is built for exactly that. Applied **gray-box** to the 16-dim trace features, not hidden states. | **Piloted** (Track A). Direct L-SML replacement → resolves FUSE overlap while keeping the signal. |
| 5. IMM | Bank of filters, one per regime | IMM needs a-priori dynamics for *both* regimes — the hallucination-mode model is unknown. The doc's claim that HMMs "cannot integrate continuous inputs" is wrong: Gaussian-emission HMMs do exactly that. | **Replaced with its honest version** — 2-state Gaussian HMM (Track B). |
| Hybrid LOCA→KalmanNet→conformal | Unified online framework | Three unvalidated components stacked; each must earn its place alone first | **Reject** (as an experiment; fine as a future-work figure). |

**Additions not in the doc** (§5): BOCPD change-point detection (piloted, Track B), conformal/exchangeability martingales (proposed only).

---

## 1. Two problems, not one

The pivot doc's motivation section conflates two independent concerns:

1. **FUSE novelty pressure.** FUSE (Lee … Candès, arXiv 2604.18547) builds on the same Parisi/Jaffe–Nadler SML lineage we use for fusion. If the thesis were framed as "unsupervised spectral fusion for hallucination," the overlap would be heavy. But the established positioning (Step 147) is that our contribution is the **signal** — spectral features of one model's own token-entropy trace — and FUSE actually *de-risks* the fusion step (a Candès-group validation of this exact label-free fusion family at frontier scale). The correct hedge is therefore to show the signal survives an **aggregator swap**: if a method with no SML ancestry (robust AE, density model) matches L-SML continuous on the same features, the aggregation layer is a commodity and the signal claim stands on its own. That requires nothing "online". → **Track A**.

2. **Online/streaming detection.** The doc cites the Step-148 pilot and proposes online state estimation as the fix. But Step 148 *failed its pre-registered streaming gate* (G2: fused prefix L-SML ≤ best DeepConf window everywhere except the earliest 10% of the trace). The honest reading is that temporal-model value is **unproven**, not that a fancier temporal model is the obvious next step. Any temporal candidate must first beat DeepConf and full-trace L-SML on data we already have. → **Track B** (triage-grade: one clean cell).

Consequence: a Track A pass is thesis-relevant regardless of Track B; Track B failing does not undermine Track A.

## 2. White-box exclusion (this round)

Options 3–4 as written operate on transformer hidden states. Two reasons they are out of scope here:
- **No data**: no hidden states were ever cached; testing would need new inference infra + a Colab re-run of at least one cell.
- **Positioning**: hidden-state probes are a crowded field, and moving there abandons the gray-box entropy-trace signal that is the FUSE differentiator and the thesis identity.

PRAE and flow/density ideas survive gray-box: applied to the 16-dim trace-level feature vectors (Track A). Decision (Omri, Jul 2026): gray-box only this round.

## 3. Weak-supervision creep — a warning for any writeup

Several proposals in the doc quietly break the "unsupervised" claim:
- "Train LOCA / the flow / the AE **on correct-answer traces**" requires correctness labels for the training pool → that is (weakly) supervised.
- The doc's KalmanNet `L_weak` BCE term backpropagates final-answer labels through the filter → supervised training, full stop.

Everything piloted here is fit per cell on **all unlabeled samples**. PRAE is the principled way to get "train on the clean part" without labels — its gates infer the inlier set during training. If a future variant uses pseudo-labels (e.g. from an existing unsupervised score), that must be declared as self-training, not "unsupervised" simpliciter.

## 4. Per-option critique (details beyond the table)

**Option 1 — KalmanNet.** The proposal is sound engineering but the unsupervised story has a gap: the innovation loss `Σ‖y_t − ŷ_{t|t-1}‖²` trains the filter to *predict* the observation stream on all traces. For the resulting innovation magnitude to detect hallucination, wrong-answer traces must be systematically less predictable — an empirical question nobody has checked on our data. That check costs an afternoon with a fixed AR(2)/constant-velocity Kalman filter (no learning), which is what Track B's `ar2_mse` / `kalman_nis` rows are. If fixed-filter innovations carry ~zero signal, a learned Kalman gain has nothing to amplify and the KalmanNet investment (Colab training loops, architecture sweeps) is dead on arrival. Also note HALT (supervised GRU on top-20 logprobs, cited in the doc) is the supervised skyline for this family: KalmanNet-unsupervised should be positioned against it, and "GRU that learns a gain" vs "GRU that learns a score" is a thin architectural distinction — the model-based structure must demonstrably buy something (sample efficiency, cross-model transfer) or reviewers will read it as HALT-minus-labels.

**Option 2 — LOCA.** LOCA's defining requirement is *bursts*: clouds of short-time repeated measurements around each point, used to estimate local covariances and enforce local conformality. Our data has one trace per (question, generation); there is no natural burst structure. (K=8 sampling caches are per-question response sets — different answers, not perturbations of one state — and exist for 2 cells only.) Without bursts, LOCA reduces to a standard AE with extra machinery, which Track A tests directly. The "trajectory escapes the manifold" detection story additionally needs per-step intrinsic-dimension estimates, which are extremely noisy at trace lengths of 100–500 tokens.

**Option 3 — Diverging Flows.** Bracha's extrapolation-detection idea is attractive, and citing it as motivation is smart positioning. But on hidden states it is out of scope (§2), and on 16-dim feature vectors with n=100–1300 per cell, a normalizing flow is over-machinery: GMM/KDE negative log-likelihood is the same detection principle (density scoring) at pilot cost. If Track A's density rows look strong, upgrading kde→flow is a natural follow-up *with* an advisor-connection story.

**Option 4 — PRAE.** The strongest idea in the doc, for a reason the doc itself doesn't state: our per-cell unlabeled pools are contaminated at 7–80% (hallucinations are not rare outliers in several cells), and PRAE's differentiable inlier selection is designed for contaminated training pools. Two honest caveats: (a) where hallucinations are the *majority*, "inlier" flips meaning — the anomaly direction inverts, which is why the pilot's primary metric is label-free epr-anchored (§6); (b) at n<80 a per-sample-gated AE is underdetermined — those cells are skipped by pre-registration.

**Option 5 — IMM.** IMM assumes you can write down both regimes' dynamics and a transition matrix a priori. We can't — the "hallucination regime" model is exactly what we don't know. Estimating it from data by EM *is* a Gaussian HMM, which the doc dismisses with an incorrect claim (HMMs handle continuous observations via Gaussian emissions; that is the textbook case). The honest experiment is the 2-state Gaussian HMM on (H(t), ΔE(t)) with the hallucination score = posterior occupancy of the high-entropy regime. If that carries signal, IMM/switching-KF refinements are a follow-up; if not, they're moot.

**Hybrid framework.** LOCA→KalmanNet→conformal stacks three components, none individually validated on our data, plus a calibration layer whose guarantees (LTT/conformal) hold only under exchangeability assumptions that per-token generation violates. Pilot components separately; if two of them independently pass gates, *then* a composition is worth designing.

**Citation hygiene.** Before any advisor-facing use, verify: "Diverging Flows, Laufer-Goldshtein et al., 2025/2026" (no arXiv id given), "HALT arXiv:2602.02888", "Unsupervised Learned Kalman Filtering (EURASIP 2024)". Gemini-produced citations in this area have a nontrivial fabrication rate; the FUSE and PRAE and LOCA cites are confirmed real.

## 5. Additions not in the doc

- **BOCPD** (Adams & MacKay 2007, Bayesian online change-point detection): unsupervised, online, and it *localizes the drift token* — which is the doc's own "delayed detection" complaint against prefix L-SML. It is also the principled generalization of our `cusum_max`/`cusum_shift_idx` features. Cheapest possible pilot; included in Track B (`bocpd_*` rows).
- **Conformal / exchangeability martingales** (Vovk): online test of "are recent tokens exchangeable with earlier ones," sits on top of any per-token nonconformity score, gives distribution-free false-alarm control. Proposed only — worth a design discussion if any Track B candidate is promoted.
- **SSA / lag-embedding trajectory spectra**: Takens embedding of H(n) + singular spectrum analysis — connects to the group's spectral lineage. Memo-only; the 16 features already cover much of this space.

## 6. Pilot protocol (pre-registered; full details in script docstrings)

- **Track A** (`scripts/pivot_trackA.py`): 6 anomaly scorers × 29-cell battery × feature sets {5, 16}. Orientation is label-free, three tiers: raw (fixed convention), **epr-anchored (primary, gated)**, oracle max(p,1−p) (diagnostic only — the only tier comparable to the stored L-SML numbers, which used label-peeked orientation; our gate is therefore conservative). Transductive fit-and-score matches L-SML's information access (it estimates eigenvector weights from the same unlabeled eval cell). **Gate A**: macro(anchored) ≥ macro(L-SML continuous) − 1pp on common cells ⇒ viable hedge; ≥ +1pp ⇒ genuine improvement. Diagnostic: PRAE ≤ Mahalanobis ⇒ nonlinearity adds nothing.
- **Track B** (`scripts/pivot_trackB.py`): HMM occupancy, BOCPD change-point stats, AR/Kalman innovations on the one clean raw-trace cell (gsm8k/Llama-3.1-8B, n=200), vs recomputed DeepConf/L-SML baselines (cross-checked against Step-148 stored values). **Gate B**: candidate ≥ best DeepConf AND ≥ full-trace lsml5, paired-bootstrap 95% CI of the DeepConf delta excluding 0 ⇒ promote to Colab replication. MATH-500/Qwen-1.5B cell is secondary/non-canonical (K=8 correlated traces, cluster bootstrap), never gates.

## 7. Track A results — **Gate A: all candidates FAIL**

Run: 2026-07-03, 29 cells, ~30 min CPU. Full numbers in `results/pivot_trackA.pkl`, figures in `results/figs/pivot_trackA_*.png`.

| method (fs=16, PRIMARY) | macro anchored | L-SML cont. | Δ | verdict |
|---|---|---|---|---|
| maha | 0.513 | 0.651 | −13.8pp | FAIL |
| **gmm2** (best) | **0.553** | 0.651 | −9.8pp | FAIL |
| kde | 0.542 | 0.651 | −10.9pp | FAIL |
| iforest | 0.504 | 0.651 | −14.6pp | FAIL |
| ae | 0.535 | 0.646¹ | −11.2pp | FAIL |
| prae | 0.523 | 0.646¹ | −12.3pp | FAIL |

¹ ae/prae skipped on 4 cells with n<80, so their comparator macro differs slightly. fs=5 results are the same story (best: gmm2 0.555, −9.8pp).

**Diagnostics:**
- PRAE ≤ plain AE on both feature sets (−1.2 to −2.0pp): the robust gating **adds nothing** here.
- PRAE ≈ Mahalanobis (+1.0pp on fs=16): the nonlinearity **adds ~nothing**.
- Per-regime: only reasoning shows life (gmm2 0.701 on fs=16 reasoning — still ~8pp below L-SML's reasoning macro); gpqa and rag are at chance for every scorer.
- **Not an orientation artifact**: even the oracle tier (label-peeked max(p,1−p), the most generous possible, and the tier directly comparable to how the stored L-SML comparators were computed) tops out at 0.59–0.60 macro — still ~5pp below L-SML. The scorers lack separability, full stop; the anchored/raw gaps just add inversion on top.

**Why this negative result is informative.** Anomaly/density scorers are *direction-free*: they score distance from the bulk in any direction of feature space. The label-relevant structure in our features is a specific **oriented consensus direction** (each feature has a known sign; correct answers sit on the agreeing side). L-SML's whole mechanism is estimating that consensus direction from the unlabeled covariance — which is precisely the information density models discard. The aggregation layer is therefore **not a commodity**: consensus-weighting matters, and "any unsupervised aggregator would do" is false. Implication for the FUSE concern: the right defense is the established one (contribution = the signal, L-SML cited as an off-the-shelf tool with FUSE as validation of the family) — not swapping in a weaker aggregator to manufacture distance.

## 8. Track B results — **Gate B: no candidate promoted**

Run: 2026-07-03, primary cell `gsm8k/Llama-3.1-8B` (n=200, 20% wrong). Recomputed baselines cross-checked against stored Step-148 values. Full numbers in `results/pivot_trackB.pkl`.

Targets: best DeepConf = w32 **0.7355**; full-trace lsml5 **0.7539**.

| candidate | AUROC (anchored) | vs entropy level (Spearman) | note |
|---|---|---|---|
| hmm_occ (2-state regime posterior) | 0.719 | ρ = 0.97 | repackages mean entropy |
| ar2_mse (AR(2) innovations) | 0.717 | ρ = 0.94 | repackages mean entropy |
| kalman_nis (Kalman innovations) | 0.703 | ρ = 0.93 | repackages mean entropy |
| bocpd_ecp_l50 | 0.703 | ρ ≈ 0.03 | **orthogonal signal** |
| bocpd_ecp (λ=100) | 0.685 | ρ = −0.07 | **orthogonal signal** |
| hmm_tail / hmm_switch / bocpd_map… | 0.40–0.66 | — | weaker variants |

No candidate reaches either target, so none clears the paired-CI clause either. **Gate B: temporal models tried-and-rejected at pilot scale.** The secondary MATH-500/Qwen-1.5B cell (non-canonical, K=8 correlated traces, cluster bootstrap) shows the same picture: best temporal candidate hmm_tail 0.710 vs best DeepConf 0.672, +3.8pp with CI [−0.6, +8.2] — includes 0, and the cell never gates.

**KalmanNet go/no-go → NO-GO.** Innovations do carry label signal (0.70–0.72), but at ρ = 0.93–0.97 with plain mean entropy they are a repackaging of the level signal we already exploit, several points *below* the trivial baselines. A learned Kalman gain would have to overturn that from n=200 unlabeled traces; there is no headroom evidence. This also empirically undercuts the doc's "Markovian hallucination momentum" premise: the fitted high-entropy regime is not sticky (self-transition 0.46 vs 0.77 for the grounded state) — high-entropy visits are short bursts, not persistent regimes.

**The one genuinely new finding: BOCPD change-point count is orthogonal to entropy level** (ρ ≈ 0 vs epr) yet scores 0.685–0.703 alone. It is the only candidate that measures something our 16 features don't already encode (cusum_max correlates with level; ecp doesn't). Exploratory post-hoc check (not gated): adding `bocpd_ecp` as a 6th L-SML view on this cell gives −0.28pp [−3.1, +3.0] — null at n=200. Verdict: not a replacement, possibly a **view candidate**; re-check for free on the raw traces from the already-queued Colab re-inference (MATH-500/Qwen-7B + clean R1) before spending anything on it.

## 9. Recommendation

1. **Do not pivot.** Both pre-registered gates failed decisively. The alternatives from the pivot doc are either weaker aggregators (Track A: −10 to −15pp) or repackagings of the entropy level (Track B: ρ ≥ 0.93). L-SML continuous over the spectral features remains the best unsupervised method we have tested — now against 6 anomaly scorers and 4 temporal-model families, on top of the earlier U-PCR / flat-SML / avg comparisons.
2. **The FUSE defense stays as established** (Step 147 positioning): contribution = the entropy-trace signal; FUSE innovates on the fusion side of the same pipeline and validates the family. Track A now adds an empirical argument to that story: the aggregation layer is not swappable-at-zero-cost, and our use of L-SML is a choice that measurably matters — which is a *strength* in the narrative, not a liability.
3. **KalmanNet, LOCA, IMM, the hybrid: drop** (per §0/§4 + pilot evidence). Diverging-flow density ideas survive only as the gmm/kde rows already tested (failed).
4. **One cheap thread stays open:** BOCPD change-point count as a 17th view — orthogonal to everything we have, free to compute on the queued Colab re-inference traces. Decide after that data exists; do not schedule GPU time for it.
5. If the advisors want an "alternatives were considered" section for the thesis, this memo + `results/pivot_track{A,B}.pkl` is that section, with pre-registered gates and honest nulls.
