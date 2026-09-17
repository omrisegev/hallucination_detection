# Model-based outer group reliability, 2026-09-17

Step402 fixed model-inverse readout suppresses noise but worsens base within-AUC.
Do not promote it. Test one more precisely isolated mechanism: retain each
virtual score t_g=X_g v_g unchanged; replace ONLY its outer fusion weight.

From the fitted Joint factor model, a_g=||v_g||^2 is the loading of t_g on
the global factor; V_g=v_g^T Sigma_model,gg v_g is its modeled variance.
Use sum_g (a_g/V_g)t_g: equivalently standardized virtual scores weighted by
their model-derived correlation a_g/sqrt(V_g) with the global factor. This is
modeled shared-factor reliability, NOT independently verified error reliability.
Zero-variance groups get zero. No fitted threshold, graph, inverse or tuning.
Within-group weights stay exactly v_g. H1 sign orientation stays unchanged.

Use the exact saved fits/supports/failures from Step402 for original BOCPD51,
copies66 and noise66, full and automatic supports; no optimizer rerun. Same
13769 answers, source-fold hybrid contract, gate and labels. The original failed
noise fold remains a2730-answer H1 fallback, explicitly not a native success.

Five primary contrasts and99.5% source-group bootstrap intervals: candidate auto
versus historical hierarchical auto separately on each bank, plus candidate auto
under each stress versus candidate auto base. Same practical margins PB-1pp /
within-.002 AND native13769 required. Full-support and historical inverse,
Continuous/equal/innovation5/BOCPD are descriptive references. Development stage;
these intervals do not undo selection among multiple research stages. No extra
readout exponent, threshold, feature support or per-cell choice after results.

This stage addresses the normalization mechanism only. Even a positive result
does not solve noise-induced initial-fit failure or discovery dependence on
duplicates. Report those as remaining requirements, not hidden exceptions.
