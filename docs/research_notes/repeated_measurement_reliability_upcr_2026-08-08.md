# Repeated-measurement reliability U-PCR

Status: completed development experiment; not promoted.

The hypothesis was that repeated measurements of one answer could identify
feature variation caused by nuisance rather than by the unknown correctness
target. The implemented repeated measurements are synchronized moving-block
bootstraps of one saved token-telemetry trace. They require no new LLM pass.

The method estimates

\[
S_{\mathrm{signal}}=S_{\mathrm{total}}-S_{\mathrm{within}}
\]

and solves

\[
S_{\mathrm{signal}}v=\lambda S_{\mathrm{within}}v.
\]

The full feature pool did not satisfy this measurement model. A label-free
procedure-compatibility check retained 17/28 varying features on GSM8K and
18/28 on MATH. The restricted covariance estimate was stable and nearly PSD on
both cells.

The direct use of generalized eigenvectors as U-PCR regressors failed because
it removed the off-diagonal covariance required by U-PCR's moment equations.
Hard projection back to feature axes repaired that mechanical problem but lost
performance. A soft Wiener reconstruction matched the baseline but did not
improve it:

| Method | GSM8K | MATH |
|---|---:|---:|
| DUFS-LIU mixed-v2 | 0.7673 | 0.7188 |
| RM Wiener DUFS-LIU | 0.7679 | 0.7202 |
| Difference | +0.0006 | +0.0013 |

Both paired confidence intervals include zero. The candidate and baseline
scores correlate at about 0.98. The six held ProcessBench cells were not opened.

Decision: retain DUFS-LIU mixed-v2. Keep this experiment as evidence that
bootstrap stability is not the same as target relevance. Any future repeated-
measurement study must vary a known nuisance while preserving the generated
answer and semantic target.

Full protocol, diagnostics, plots, and commands:
`results/repeated_measurement_reliability/REPORT.md`.
