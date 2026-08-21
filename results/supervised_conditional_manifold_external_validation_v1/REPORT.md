# Frozen supervised manifold external-to-discovery audit

**Decision: `CONDITIONAL_NULL_INELIGIBILITY_INVALIDATES_EXTERNAL_AUDIT`**

This is a retrospective external-to-discovery audit, not prospective confirmation.

Coverage: 3/3 dataset families. Conditional-null coverage=False; geometry=False; residual=False; distinct-vs-linear=False; utility=False.

The four fixed graphs are the learned metric, its one-dimensional linear score, an equal-weight feature graph, and the metric after removing the linear score.

The registered conditional tests were not eligible in enough independent dataset families. Descriptive effects below cannot be interpreted as a manifold pass or a transfer failure.

- `aqua`: null-eligible=False, geometry=False, residual=False, distinct=False, metric effect=+0.055, linear advantage=+0.073, LIU delta=-0.0010.
- `coqa`: null-eligible=False, geometry=False, residual=False, distinct=False, metric effect=+0.084, linear advantage=+0.085, LIU delta=+0.0026.
- `hle`: null-eligible=False, geometry=False, residual=False, distinct=False, metric effect=+0.003, linear advantage=-0.015, LIU delta=-0.0008.
