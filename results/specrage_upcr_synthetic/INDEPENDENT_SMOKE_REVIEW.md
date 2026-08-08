# Independent review of the SpecRaGE-LIU smoke run

## Verdict

**The smoke run is valid as a negative mechanism result. Do not advance to the
development grid or real-data calibration.**

The comparison is paired and complete: six worlds, two replicates, ten arms,
and five values of lambda produce 600 rows. All graph and ridge arms reproduce
ordinary IU-PCR at `lambda=0`.

## Verified findings

- Learned sample-specific weights remain approximately uniform. Across runs,
  alpha lies roughly between `0.241` and `0.263`, versus exact uniform weight
  `0.25`; normalized entropy lies between `0.999963` and `0.999978`.
- Clean-versus-corrupt reliability AUROC is approximately random under
  conditional corruption (`0.4945`, `0.4949`) and inverted under global
  corruption (`0.1195`, `0.1094`) and view-specific nuisance (`0.3289`,
  `0.3527`).
- At `lambda=0.1`, sample-specific and global/permuted arms are identical in
  every paired run. Uniform differs in only one run, by `0.0244` AUROC points.
  No performance change can be attributed to conditional reliability.
- Formal graph collapse did not occur: every graph has one connected component,
  no near-isolated samples, effective-edge fraction `0.595–0.735`, and
  fifth-percentile degree / mean degree `0.269–0.659`. The reliability mechanism
  instead collapsed to the uniform-weight control.
- Two replicates are smoke evidence only. Their bootstrap intervals are not
  inferential.

## Competing explanations

The first unresolved explanation is inadequate optimization rather than a
proved failure of SpecRaGE. The smoke configuration uses temperature `10`, and
each run receives only 6–12 optimizer updates. Five of twelve checkpoints select
epoch 1, and several unlabeled validation curves worsen. Seed-stability metrics
are also uninformative because smoke uses one model seed.

There is a deeper warning: even the planted oracle reliability weights provide
essentially no benefit in the conditional-, global-, and view-specific
corruption worlds at the primary lambda. Therefore learning alpha correctly may
still fail to improve IU-PCR if the weighted graph does not create a useful
roughness direction inside IU-PCR's fixed two-dimensional head.

## Only justified next diagnostic

Use one fixed conditional-corruption dataset—no data sweep and no performance
selection. Keep its data, affinities, and downstream LIU settings fixed. Train
with enough optimizer steps for the unlabeled objective to plateau and use two
seeds. Then ask, in order:

1. Does alpha leave uniformity and recover the planted clean-view pattern?
2. If it does, does the resulting graph differ materially from uniform/global
   controls?
3. If it does, does that graph alter the projected IU-PCR roughness and score?
4. Does the oracle reliability graph have enough downstream effect to make an
   improvement possible at all?

If alpha remains uniform, the bottleneck is optimization or the SpecRaGE
objective's identifiability in this construction. If alpha recovers reliability
but the oracle and learned graphs remain inert, the bottleneck is the
graph-to-LIU coupling or the synthetic world's available headroom.
