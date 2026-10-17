# Independent pre-fit review

Reviewer: `independent_review`, which implemented no study arm. Scope: static
review and synthetic arithmetic only, before new real-data fitting. The frozen
advisor package and its canonical helpers were not edited.

Reviewed the new input bridge, latent-vote/count helpers, prediction driver,
sealed evaluator, protocol/amendments, and their synthetic tests. Current root:
`/Users/osegev/Desktop/hallucination_detection`; branch
`codex/claude-feature-bank-token-lsml-v1`; HEAD
`8904598661beb32aff5e997163c02e5a58e6c78d`. Existing dirty files were preserved.

## Findings raised before fitting

1. **Incomplete baseline coverage could satisfy the success gate.** Original
   `combined_success` inspected candidate coverage but ignored the common
   coverage recorded by each contrast. It now requires complete required
   contrasts and eligible Q80/legacy-MoM checkpoints as well. An otherwise
   passing candidate with an incomplete baseline is explicitly rejected.
2. **The evaluator did not authenticate the label-source bytes.** The actual
   label-file hash was originally only recorded after evaluation. The
   prediction seal now includes `INPUT_FREEZE.json`; the evaluator verifies
   the uniquely matched expected population hash before opening its arrays.
   Prediction group/fold identities are also checked against the authenticated
   population, with source groups confined to one fold.
3. **Canonical prediction failure left the fitted dependency registered.** A
   holdout prediction exception could leave `fusion[method]` present and
   bypass the missing-count branch. The driver now checks prediction shape
   and finiteness and removes the failed dependency on exception.
4. **B's report wording implied aggregation of several triples.** The actual
   estimator selects exactly one training-only triple. The report owner was
   asked to state that behavior and retain the orientation and conditional
   independence assumptions.

These findings were delivered to the root agent before code freeze and new
real-data fitting. The frozen current code resolves all four findings. The
mathematical receipt's code hashes match `CODE_FREEZE.json`; the final pre-fit
unit-test log reports success. No other must-fix scientific or mathematical
issue was found in the reviewed pre-fit implementation.

## Independent mathematical evidence

`independent_math_audit.py` uses exact latent-state enumeration, finite
differences, and explicitly repeated whole-source samples with independent
SciPy midranks/OLS residuals. Its receipt is `INDEPENDENT_MATH_AUDIT.json`.

- Three-voter moment recovery agrees with the exact planted latent
  distribution to at most `2.22e-16`.
- The zero-mixture answer log-likelihood agrees with latent-label enumeration
  exactly; the analytic gradient agrees with central finite differences to
  `6.09e-10`; its posterior count PMF agrees to `1.12e-16`.
- Poisson-binomial convolution agrees with full enumeration exactly in the
  fixture, including deterministic probabilities and distinct half-up/lower
  median ties.
- Bootstrap partial Spearman agrees with explicit source repetition and
  independent rank/OLS calculation to `1.39e-16`.

The arithmetic checks establish implementation consistency under the model
assumptions. They do not establish conditional independence, latent error
orientation, external validity, or performance on the study population. Final
saved-output reconstruction and report review remain pending.
