# Signal-group membership API (Step407)

`spectral_utils.joint_signal_membership.fit_signal_aware_joint` has the same
training/scoring contract as [the frozen API](JOINT_NOISE_AWARE_API.md).
Use it in place of `fit_noise_aware_joint`; the existing `score_noise_aware_joint`
accepts both model mappings. It returns a step score, not a no-error decision.

```python
from spectral_utils.joint_signal_membership import fit_signal_aware_joint
from spectral_utils.joint_noise_aware import score_noise_aware_joint

model = fit_signal_aware_joint(
    training_steps, training_offsets, training_source_ids, training_folds,
    anchor_index=h1_index, seed=399170 + outer_fold,
)
held_scores = score_noise_aware_joint(model, held_steps)
```

Invalid models raise at scoring unless the caller explicitly enables the
declared anchor fallback. Count fallback answers separately from native fits.
No label array is accepted. Inputs are standardized within answers; training
and held source identities must be disjoint in the registered hybrid experiment.

The new decision is in each sparse fitting round's
`model['membership']['rounds'][r]['sparse']['signal_membership']`:

- `global_support_union`: nonzero global coordinates across converged starts;
- `signal_groups`: groups containing any such coordinate;
- `nuisance_groups`: groups with no such coordinate;
- `retained`: old sparse feature support restricted to the signal groups.

The local nuisance loadings are saved in the starts even when their groups
leave the active scoring model. A local-only feature in a signal group remains
eligible. Group removal triggers the original group rediscovery; it does not
turn the provisional remaining two groups into an admissible Joint fit.
All original final-fit identification guards remain mandatory.

This separates nuisance membership from ordinary independent-noise membership.
It does not claim that a globally connected feature detects errors, nor does
it establish stability under approximate duplication. The complete experiment
is specified in [the frozen protocol](JOINT_SIGNAL_MEMBERSHIP_V1.md); its
results determine the method's current empirical limits.
