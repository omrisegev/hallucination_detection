# Staged Joint membership API

Use `spectral_utils.joint_staged_membership.fit_staged_joint` with the same
arguments as [the original API](JOINT_NOISE_AWARE_API.md). Scoring still uses
`score_noise_aware_joint`; invalid models require an explicit fallback request.
The registered experiment is hybrid source-fold fitting, with no correctness
labels in the fitting API and no digit inputs. This is not answer-only fitting.

Each saved sparse round now declares its decision stage:

- `zero_rows`: remove features whose global and local loadings are both zero
  in every converged start, then rediscover groups;
- `global_groups`: only after ordinary row support stabilizes, remove groups
  whose global loadings are zero in every converged start, then rediscover.

Read this at `model['membership']['rounds'][r]['sparse']['signal_membership']`.
Local-only features remain eligible inside globally connected groups. The
three-round cap and all checked-Joint identification guards remain unchanged.
A too-small active support returns an invalid model with its round diagnostics.

The same frozen95% information refinement follows this membership stage. A valid
fit does not prove that the selected features localize errors well. In particular,
changing this membership order is not a promised solution to near duplication.
Use the full [Step408 protocol](JOINT_STAGED_MEMBERSHIP_V1.md) and result audit
to assess the tested limits. The implementation does not reuse prior models;
all25 bank/fold combinations were scheduled as fresh fits.
