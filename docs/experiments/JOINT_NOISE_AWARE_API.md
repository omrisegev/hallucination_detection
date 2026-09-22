# Reusing the noise-aware Joint selector

Implementation: `spectral_utils.joint_noise_aware`. This composes the frozen
Steps404-405 mechanisms; Step406 replays every original source fold before testing
new feature banks. The API accepts no correctness labels.

`fit_noise_aware_joint(values, offsets, source_ids, inner_folds, *, anchor_index,
seed=399170, notify=None)` takes:

- `values`: finite steps-by-features matrix, already standardized within each
  answer according to the feature-bank contract;
- `offsets`: answer boundaries, starting at0 and ending at the row count;
- `source_ids`: one source identity per answer, shared across repeated scoring
  models of the same source;
- `inner_folds`: source-fold IDs for TRAINING answers only;
- `anchor_index`: a fixed risk-direction feature, H1 in the registered benchmark.
  It can be excluded from the active support; it only orients the final score.

The returned mapping contains `valid`, training-learned `aliases`, sparse
`membership`, information `refinement`, selected canonical/original coordinates,
weights and fit diagnostics. Invalid fits retain a failure reason. The retained
row count refers to canonical measurements, so aliases do not create extra
independent parameters.

Example matching one registered outer source fold:

```python
import numpy as np
from spectral_utils.joint_noise_aware import (
    fit_noise_aware_joint,
    score_noise_aware_joint,
)

train_answers = folds != outer
train_rows = np.repeat(train_answers, np.diff(offsets))
train_offsets = np.r_[0, np.cumsum(np.diff(offsets)[train_answers])]
model = fit_noise_aware_joint(
    X[train_rows], train_offsets,
    source_ids[train_answers], folds[train_answers],
    anchor_index=feature_names.index('H1'),
    seed=399170 + outer,
)
if model['valid']:
    held_step_scores = score_noise_aware_joint(model, X[~train_rows])
else:
    # Registered benchmark fallback; count these answers as non-native.
    held_step_scores = score_noise_aware_joint(
        model, X[~train_rows], allow_anchor_fallback=True,
    )
```

Scoring expects the same original feature order and preprocessing. Exact aliases
share their common value; if aliases that matched in training diverge at scoring,
their mean is the declared shared measurement. An invalid model raises unless
the caller explicitly requests the anchor fallback. Shape/nonfinite input errors
raise rather than silently selecting another model.

The function returns a continuous step score. The answer-level gate is external
and unchanged in these experiments: answer Tail15 Top10 mean, cell midrank>=.33;
open answers use the maximum-score step. PRMB within-answer AUC is ungated.

The registered experiment is source-fold hybrid fitting, not answer-only. Exact
aliases and independent Gaussian additions were verified on the complete matched
benchmark in Steps404-405. Step406 tests near copies and correlated persistent
nuisance without changing the selector. Its findings, including failures or
regressions, determine the limits of reuse; the API itself is not a robustness
guarantee or a new accuracy claim.
