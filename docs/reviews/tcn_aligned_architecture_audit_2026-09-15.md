# Step388 architecture clarification, before full quality evaluation

The existing TelemetryTCN accepts16 history slots, but its last-position output
has receptive field15: three kernel3 convolutions with dilation1,2,4 give
1+2*(1+2+4)=15. Pointwise residual paths do not enlarge the field. The oldest
input slot (lag16) has no influence on its output. Width32; separate heads
predict mean and positive variance, with a relative-position input at the head.

The frozen experiment keeps this architecture unchanged, including the old
fold0 checkpoint. This is a disclosure of the existing implementation, not a
new arm or a post-quality choice. Ridge uses all16 lags; comparisons are between
the registered complete predictors, not a perfect isolation of nonlinearity
at identical effective receptive field. No claim that all16 TCN slots influence
the output. Whole-answer normalization and relative position remain offline.

An independent gradient/perturbation test confirms exact zero influence of
the oldest slot and nonzero influence in the remaining slots:
tests/test_tcn_aligned_architecture.py. No training/scoring code was changed.

The existing shuffle permutes all16 available input slots. Because only15
slots reach the output, for a full window it can also change which observation
is omitted. Thus real-versus-shuffled is not a pure permutation of an identical
visible15-token set. Keep the registered, already-scored control unchanged,
but do not attribute a difference exclusively to lag ordering. A future
matched-visible-history shuffle would be a separately declared control.
