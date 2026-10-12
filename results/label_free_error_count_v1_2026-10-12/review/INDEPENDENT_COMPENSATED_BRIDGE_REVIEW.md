# Independent compensated-bridge review

The first attempt's numerical bridge failure must remain archived separately;
it is a failure of floating-point reconstruction, not evidence about the
count-model hypotheses. No new label-based evaluation had occurred before the
repair. The representation tolerance remains `1e-9`.

The repaired helper uses float64 high/residual pairs for donor centering,
division, inverse transformation, coefficient/intercept transfer, and group
summation. It rounds the public group score only once. The registered
mathematical coordinates `active*(raw-mu)/r`, active masks, donor float64
mean/scale definitions, and original within-group weights are preserved.
Model version 2 identifies the numerical backend.

The independent synthetic test uses a Decimal80 reference, including an
extreme donor-mean versus local-variation contrast. Compensated coordinates
agree to relative error `1.05e-32`; inverse `V` agrees to absolute error
`4.0e-15`; group scores equal the single-rounded Decimal80 oracle exactly.
Ordinary float64 inverse reconstruction fails badly in this fixture (maximum
error `22.22`), so the test would detect a reintroduced cancellation path.
Script and receipt: `independent_compensated_bridge_audit.py` and
`INDEPENDENT_COMPENSATED_BRIDGE_AUDIT.json`.

Static inspection found the high/residual path used consistently on fit and
holdout. No mathematical-definition change or material issue was found. The
tested domain is finite, normal representable arithmetic; Dekker splitting is
not a general near-overflow/underflow arithmetic library. Nonfinite moments
and coordinates remain explicit failures. No real-data fit or study-label
access was performed by this reviewer for this audit.

## Quota-diagnostic design requirement

Within-answer independent column shuffling preserves each answer's exact
`ceil(q*n)/n` quota. Even with independent step placements, pooled moments can
remain nonzero solely because that quota varies with answer length. Writing
`p_a=ceil(q*n_a)/n_a`, their analytic permutation expectations are
`Cov(v_i,v_j)=4 Var_stepweighted(p_a)` and
`E[(v_i-mu)(v_j-mu)(v_k-mu)]=8 E_stepweighted[(p_a-E(p_a))^3]` for distinct
voters. A surviving shuffled estimate therefore does not prove meaningful
shared alignment. Fit failures remain missing; closeness of estimated latent
prevalence to quota is an empirical diagnostic, not an algebraic identity.

The added q values and ten seeds must remain fixed exploratory diagnostics,
with no candidate selection or change to the seven-arm multiplicity. A
zero-center control uses a frozen donor absolute group center and threshold
zero, with no heldout quantile and no enforced positive fraction. Diagnostic
code review is pending the implementation owner's completion.
