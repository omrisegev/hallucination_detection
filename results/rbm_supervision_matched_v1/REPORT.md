# Matched RBM supervision: complete

Freeze afb852591; branch codex/rbm-supervision-matched-v1; base f7203a8a6.
Full13,769 answers,90 fits, identical source-group folds, labels, token bank12,
answer-specific base models,13 correction parameters, zero initialization,
penalty, optimizer, token Logit/Top10/argmax and entropy gate.
Only the learning objective and access to training step labels differ.

These are COEFFICIENT UPDATE alternatives above an identical saved RBM.
Both use other training answers for the shared correction; base weights stay
answer-specific. This is not a pure from-scratch supervision comparison or an oracle bound.

| Method | PB macro % | PRMB within AUC | Mean fold AUC | PRMScore |
|---|---:|---:|---:|---:|
| RBM + unsupervised coefficient update | 36.2712 | 0.745204 | 0.706205 | 0.622215 |
| RBM + supervised coefficient update | 37.2042 | 0.747301 | 0.689485 | 0.599189 |
| Original frozen answer-local RBM12 | 36.2712 | 0.745204 | 0.706205 | 0.622215 |

Primary supervised-minus-unsupervised: PB +0.9330pp,97.5%CI [-0.2162, 2.0836];
within AUC +0.002096,97.5%CI [-0.0002511660002848833, 0.004439146660454968].
10,000 paired source-group draws; uncertainty conditional on fitted predictions.

Full model/loss and separate metric arithmetic review PASS. Zero update reproduces
all original peaks (0 changes); batched score roundoff <=2.85e-14. No step-feature averaging.
The supervised loss is class-balanced STEP BCE after Top10. PB labels are known
correct prefixes plus the first error; later steps remain unknown. No token truth labels.
PRMScore q.8 uses other-fold training scores from the SAME fold model in both arms.
Mean held-out-fold AUC replaces pooled cross-fold AUC for both updated methods.

Seven PB cells rise,one falls.318 exact gated successes gained,271 lost;193 losses move late.
Both primary97.5% intervals include zero. No confirmed winner or large headroom claim.
Held-out balanced BCE improves in all45 folds, but remains above the constant .5 loss;
see HELDOUT_LOSS.json. This does not establish good calibration or superiority at the task.
Proposed next question only: does first-error listwise training improve PB over step BCE
in this identical correction/Top10 family? The late losses motivate this test, not prove its answer.

Fit health: {"supervised_update": {"fits": 45, "max_gradient": 0.0008696306889992766, "max_iterations": 65, "median_delta_norm": 4.9031143337062115, "nonconverged": 0, "total_fit_seconds": 213.89323839981807}, "unsupervised_update": {"fits": 45, "max_gradient": 4.756823216434114e-07, "max_iterations": 0, "median_delta_norm": 0.0, "nonconverged": 0, "total_fit_seconds": 11.747581000003265}}

See PB_CELLS.csv,FORENSICS.json,COEFFICIENTS.csv for cell results and learned updates.
No new model was launched after evaluation. Development findings, not untouched confirmation.
