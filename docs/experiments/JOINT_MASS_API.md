# Mass-aware structure discovery inside Joint

`spectral_utils.joint_mass_membership.fit_mass_joint` takes the same standardized
training matrix, answer offsets, source identities, inner folds and orientation
anchor as [Step410](JOINT_FEASIBLE_API.md). Scoring uses the existing explicit
native/fallback contract. No labels enter fitting.

`joint_mass_groups.discover_mass_groups` replaces structure discovery during
membership and deletion refinement. Its feature measure solves a nonnegative
squared-correlation kernel objective, then sums to one. Zero grouping mass does
not remove a feature: its spectral coordinates are extended from the positive-
mass support. The ordinary staged sparse fit still learns feature membership.

Complete-covariance weighted rank-one projection is provisional group discovery
only. The checked Joint fit, its off-diagonal objective and profiled Jacobian
guards are unchanged. Affinity includes the diagonal to respect split-mass
repeated coordinates. Clustering is deterministic and mass weighted; K3/K4
stability uses explicitly named weighted NMI, not the old ARI criterion.

Each discovery records `mass_audit`, `fold_mass_audits`, candidate `parts` and
`stability`. Dense kernels are reconstructed from hashed training input by the
auditor to avoid repeating them in every proposal checkpoint. KKT failure,
insufficient spectral rank or inadmissible groups are explicit failures.

The refinement is the Step410 information-feasible current-factor ranking rule,
with the new discovery module. It is not Step411 minimax pruning. The initial
factor reference stays fixed, the95% threshold is fixed, and only three deletion
proposals are tried. No favorite feature is protected. Gate fitting is external.

The [frozen full protocol](JOINT_MASS_REFINEMENT_V1.md) evaluates five full banks
with hybrid source-fold fitting. Split-mass tests of internal operators do not
prove complete algorithmic invariance, robustness to near copies, or improved
localization; all three require their own correctly scoped evidence.
