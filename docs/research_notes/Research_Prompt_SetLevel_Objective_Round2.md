# Prompt for Gemini, round 2 — the 1-factor hypothesis is now ours to test. Go find the *other* hypotheses.

**Paste everything below the line into Gemini. Use its sub-agents for parallel web search.**

---

## What you returned, and what happened to it

Your latent-variable survey was the right family. You correctly saw that anti-redundancy penalties
fight a consensus estimator because the inter-view covariance *is* its signal channel, and all four
objectives you gave test the single-latent-factor structure instead of penalising correlation. That
reframing is adopted and is now a pre-registered experiment. Credit where it is due.

**Four corrections you need to carry into this round.**

**1. Two of your four objectives are degenerate as written, and would have selected noise.**
`J(S) = ‖C_S − diag(C_S) − aaᵀ‖²_F` and `J(S) = Σ(σ_ij σ_kl − σ_ik σ_jl)²` are both **minimised at
exactly zero by a subset of mutually uncorrelated, uninformative features.** A raw residual is not a
goodness-of-fit measure; it is a scale measure. Every structural objective has to be normalised —
explained *fraction* (`λ₁²/‖C_off‖²_F`), or a tetrad statistic divided by its rank-1 reconstruction,
or Ahn-Horenstein's `λ₁/λ₂` ratio. When you propose an objective in this round, **state its
degenerate minimiser and its normalised form, or do not propose it.**

**2. Your confounding story is contradicted by our own measurement.** You wrote: *"subsets with the
highest marginal `rho_hat` likely formed a sub-cluster governed by a secondary latent variable ...
The consensus estimator confidently fused them, but toward the wrong latent."* Our redundancy arm
kept the **least** mutually-correlated views and scored **−3.13pp** against a random-pruning floor —
the worst of eight arms, Holm p = 0.002, negative on 19 of 24 test sets. Removing the tight cluster
is what *hurts*. The evidence points the other way: the dominant shared factor behaves like
correctness, not like a nuisance. We are testing the inverted arm (keep the **most** mutually
correlated) as a pre-registered hypothesis. Do not repeat the confounding story as established.

**3. Your identifiability answer contains a non-sequitur.** You wrote that because "all
sign/orientation work = −0.06pp", we have "already controlled for the sign-flip ambiguity". No. That
number says giving every feature its *correct* direction buys us nothing, because our fusion is
**exactly sign-invariant** on our data. That is a property of the estimator, not evidence about
global identifiability. Do not reason from our result tables to claims they do not support.

**4. You picked the wrong triplet paper for our setting.** FlyingSquid / Fu et al. 2020 solves the
**binary** labelling-function case. FUSE (Lee, Ma, Zhao, Nair, Spector, Cohen, Candès, arXiv:2604.18547)
gives the same triplet-consistency idea, `Ŝ` in Proposition 2.4, built for **continuous verifier
scores** — which is our encoding, and where we have already established continuous ≫ binary. FUSE is
already in our library and extracted. Cite Fu et al. as the binary special case, not the primary.

## Do not search these again — we already have them

Your survey converged on material we already hold, which is corroboration but not new information:

- **The tetrad statistic is already implemented in our codebase.** L-SML's Eq. 15,
  `s_ij = Σ_{k,l≠i,j} |r_ij·r_kl − r_il·r_kj|`, is the sum of absolute tetrad differences. It has
  been in `fusion_utils.py` for months, vectorised and tested. Your Kummerfeld & Ramsey lead is a
  one-line aggregation on top of code we already run.
- **FUSE's Ŝ, the vanishing-tetrad test (Bollen & Ting 1998), and the Ahn-Horenstein eigenvalue-ratio
  rank test** were all already identified in our own July literature review, with verified citations,
  as a pipeline design that was never piloted.
- Jaffe/Nadler/Kluger SML, L-SML, and the whole Parisi-lineage unsupervised-ensemble family are our
  fusion's direct ancestors and are already digested.

**So the 1-factor hypothesis is now ours to test, and we do not need more papers arguing for it.**
We need to know what to test *instead of it*, or *alongside it*, if it fails.

## The actual state of the problem

Label-free hallucination detection. ~30 scalar views of one LLM generation, fused by an unsupervised
consensus estimator that reads each view's reliability off the view-by-view covariance. Scored by
**AUROC against correctness** on 24 (dataset × model) test sets. `n` per cell ranges **198 to 8,460**;
`D ≤ 30`; one full fit costs **~10 ms**, so the subset space is enumerable.

- **The room:** +2.25pp held out, CI [+1.53, +3.04], winning on 23 of 24 test sets, and it lives
  **entirely** in which features get kept.
- **The good sets are SMALLER than what we deploy** — ~11.8 features vs ~21, smaller on **24 of 24**
  test sets.
- **The good sets do not transfer across test sets** — −0.81pp at matched size.
- **Nothing per-feature reaches the room.** The *true* correlation with correctness is +0.08pp
  (p = 0.62); a *perfect* estimate of it is +0.34pp (p = 0.88); all six label-free rankers land on or
  below the floor.
- **The estimator's own model-selection criterion does not rank feature sets** — every correlation
  magnitude under 0.16.
- One dangling clue nobody has explained: across test sets, **Spearman(label-free minus supervised
  ceiling, n) = −0.462, p = 0.020** — our gap to the ceiling *widens* with sample size.

## What we want from you now: the OTHER hypotheses

"The good set is the one where the 1-factor model holds" is **hypothesis 1**, and it is being built.
Below are **seven rival explanations** for why some feature sets fuse far better than others. For
each, we state what we believe and what we want from the literature. **Search each one. Report what
exists, what the label-free measurable quantity would be, and which hypothesis the literature
actually supports.**

Two of these we can test in-house without you (whether the good sets are even stable across splits;
whether good-set size scales with `n`). We are not asking for those. We are asking for the five where
published theory or method would change what we build.

---

**H2 — It is estimation variance, not model fit.**
With `m` features you get `m(m−1)/2` pair equations for `m` unknowns, so over-determination grows
with `m` — but so does the number of ways conditional independence can break. There may be a
bias–variance optimum in `m` itself, and **that would explain why every good set is smaller than what
we deploy** without any appeal to structure. *Wanted:* finite-sample error rates for Dawid-Skene /
spectral-meta-learner / crowdsourcing aggregation as a function of the number of sources and the
number of items; minimax rates; anything that gives an optimal or bounded `m*`. Gao & Zhou, Zhang–Chen–Zhou–Jordan
("Spectral methods meet EM"), Karger–Oh–Shah. **Does any of this literature predict an optimal number
of sources, and does the prediction depend on `n` the way our −0.462 clue suggests?**

**H3 — It is the induced weights, not the set.**
AUROC depends only on `w·f`. Two very different subsets can induce nearly the same ranking. The good
sets may simply be those where the unsupervised weight estimate happens to land near the supervised
optimum for that subset. *Wanted:* results that bound **AUROC loss as a function of weight-estimation
error** in linear ensembles, or characterise when an unsupervised weight estimate converges to the
MSE/AUC-optimal one. If this literature exists, the right objective is a **stability or conditioning**
measure on the weight solve, not a structural fit at all.

**H4 — The dependence is across ITEMS, not across features.**
Everything we and you have considered models a latent per *feature*. But question difficulty induces
dependence too: two views can agree because a question is easy, not because they share a reliability
factor. A subset is then good if its members are jointly reliable **across the difficulty range**
rather than each excellent on one slice. *Wanted:* item-response-theory-style unsupervised
aggregation — GLAD (Whitehill et al.), difficulty-aware Dawid-Skene, mixture/heterogeneous-item
crowdsourcing models. **Is there a label-free statistic that detects item-driven dependence and
distinguishes it from feature-driven dependence?** Our estimator has a difficulty gate that is
currently switched off; this is an unexplored axis.

**H5 — Our metric is a RANK statistic and all our machinery is second-moment. (We think this is the
strongest lead.)**
AUROC is invariant to any monotone transform of the fused score; covariance is not. A subset could
fit the 1-factor model perfectly in covariance and still fuse badly in rank terms, and vice versa.
Nobody has checked whether the covariance structure and the rank structure of our views even agree.
*Wanted:* **unsupervised rank aggregation** (Borda, Markov-chain / spectral ranking without labels,
Kemeny approximations), **copula-based** dependence modelling for ensembles, rank-based analogues of
the conditional-independence tests above, and anything that estimates source reliability from
**rank agreement** rather than from second moments. If a rank-domain reliability estimator exists,
that is a different fusion, not just a different selection objective — say so.

**H6 — Second moments are blind to the structure that matters.**
A subset can be 2nd-moment-consistent with one factor and 3rd-moment-inconsistent. Tensor methods
identify latent-variable models from third-order moments in regimes where second-order moments are
not identifiable. *Wanted:* Anandkumar–Ge–Hsu–Kakade–Telgarsky tensor decompositions for latent
variable models, and specifically whether there is a **cheap third-moment diagnostic** (not a full
decomposition) that scores a subset's consistency with a single latent. `D ≤ 30`, so a full
`30×30×30` third-moment tensor is trivially affordable for us — the question is what scalar to read
off it.

**H7 — Some pool members are actively harmful, conditionally on the rest.**
Not "weak", but harmful: a view whose inclusion degrades the fusion because it corrupts the
reliability estimate for everything else. That would make the problem **outlier/adversarial-source
removal**, not feature selection. *Wanted:* robust unsupervised aggregation with a fraction of
adversarial or corrupted sources, breakdown-point results, and any **label-free** test for "is this
source corrupting the estimate for the others".

**H8 — Free slot: tell us what we have not thought of.**
If the literature suggests a mechanism none of H1–H7 names — for why a strict subset of weak
predictors fuses better than the full set, without labels — that is the most valuable thing you can
return. State it as a hypothesis, name its label-free measurable, and say which paper supports it.

---

## Rules

- **Verbatim quotes or it does not count.** Exact title, authors, venue, year, and a quoted sentence
  or equation with its section/equation number for every substantive claim. If you cannot quote it,
  mark the row `UNVERIFIED`. Two previous literature backfills for this project contained invented
  authors, venues, datasets and scores; every row is spot-checked against the source.
- **Do not reason from our numbers to conclusions they do not support.** See correction 3.
- **For every objective you propose: state its degenerate minimiser and its normalised form.** See
  correction 1.
- **Nothing whose core criterion is decorrelation, diversity, redundancy-minimisation, or a
  per-feature score.** That family is measured and closed on our data.
- **Nothing whose contribution is scalability or a differentiable relaxation.** We have `D = 30` and
  a 10 ms fit. We can enumerate. We need to know *what* to optimise, never *how*.
- **Do not re-derive hypothesis 1.** It is built. More support for it is not useful; a *rival* is.

## Output format

**Section 1 — one table per hypothesis H2–H8**, only for hypotheses where you found real literature:

| paper | venue / year | what it establishes | the label-free measurable it implies, with its degenerate case | why it explains "a strict subset beats the full set" | verbatim quote + location |

**Section 2 — ranked verdict.** Of H2–H8, which two does the literature most support as an
explanation for our numbers — specifically for *smaller sets winning on 24 of 24 test sets*, and for
*our gap to the supervised ceiling widening with `n`*? Argue from the quoted results, not from
plausibility.

**Section 3 — the one experiment.** If you could run a single label-free measurement on our data to
discriminate between your top two hypotheses, what is it? Give the formula and what each outcome
would mean. Cheaper and more decisive beats elaborate.

**Section 4 — impossibility, again, and please be rigorous this time.** Is there a theorem stating
that the fusion accuracy of a *subset* is not identifiable from unlabelled data without assumptions
we cannot verify? A clean impossibility result is worth as much to us as a method, and we would
rather learn it from the literature than from another six weeks of experiments.
