# Supervised position diagnostic: complete

All13,769 answers;135 converged fits;40 result configurations, including35 unchanged historical references.
Frozen branch codex/rbm-supervised-position-diagnostic-v1; base ce008fb1d; protocol/code90e56eca8.

The first three methods fit labels from other source-group folds, separately per cell.
They use saved bank12 feature means per STEP. This differs from the existing RBM token Top10 pipeline.
The internal three-way comparison is matched; the RBM comparison is context, not an isolated test of supervision.

| Method | PB macro % | PRMB within AUC | Mean fold AUC | PRMScore |
|---|---:|---:|---:|---:|
| Supervised fixed fusion | 32.6198 | 0.715057 | 0.690736 | 0.585182 |
| Fixed fusion + position prior | 35.3807 | 0.724419 | 0.708017 | 0.600745 |
| Position-dependent fusion | 35.9587 | 0.731300 | 0.707513 | 0.596820 |
| Existing answer-local RBM12, Logit/Top10 | 36.2712 | 0.745204 | 0.706205 | 0.622215 |
| Varentropy15 contribution IU-PCR | 35.3498 | 0.746824 | 0.710475 | 0.622689 |
| Varentropy50 | 35.6755 | 0.742465 | 0.716202 | 0.632777 |
| Raw entropy, step mean | 27.6619 | 0.648105 | 0.637818 | 0.562973 |
| Equal normalized step means | 25.6099 | 0.632303 | 0.607479 | 0.534826 |

Primary: conditional minus position prior. PB +0.5779pp,97.5%CI[-0.2671,+1.4332]pp;
PRMB within +0.006881,97.5%CI[+0.003997,+0.009723].10,000 paired source-group draws.
PB152 successes gained,123 lost;76 losses move early,47 late. Six PB cells rise,two fall.
The prior alone adds2.7609pp over static. Position interactions contain PRMB ranking information,
but do not establish a PB gain or a better method than the original RBM. No general winner.

Checks: all35 reference metrics replay; independent40-method arithmetic audit;135 saved models
and training-only normalization/thresholds replay;135 objectives and gradients replay;
max absolute final gradient2.85e-6; all13,769 outputs valid,6,030 mixed-label PRMB answers.
No PB labels are invented after the first error. Step labels are not treated as token truth.

The common raw-entropy q0.3 gate is unchanged. PRMScore q0.8 uses TRAINING scores from
the same model that predicts each held-out fold. This is in-training calibration, not nested OOF.
Supervised pooled OOF AUC is deliberately absent; mean held-out-fold AUC is shown instead,
with the same fold metric added to all references. Within-answer AUC is the local-ranking endpoint.
Intervals condition on fitted predictions and do not cover refitting or earlier research choices.

Next: discuss a matched Top10 bridge before a new unlabeled model. Do not infer that a negative
or weaker step-mean model bounds RBM performance. Cross-boundary locality is a separate open
question; separate latent-state variance remains untested. DUFS continues unchanged.
No new training family, readout tuning, HTML, or deletion was launched. Development evidence only.
