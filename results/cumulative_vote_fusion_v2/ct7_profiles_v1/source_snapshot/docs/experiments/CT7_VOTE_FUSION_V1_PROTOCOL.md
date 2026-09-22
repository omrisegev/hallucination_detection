# CT7 fixed-profile cumulative vote comparison — 2026-09-22

Question: does the binary / soft / EM fusion experiment improve when its eleven
channels are replaced by CT7's seven frozen step profiles?

This is a fixed-representation intervention. Preserve CT7's five Top10 streams,
BOCPD residual, and de-spiked pooled chosen-token z. Do not replace these with
Top5, select new readouts, or change the original CT7 candidate. The seven views
were selected during earlier development on this population; label-free fitting
here does not remove that historical selection. No new inference.

Use the cumulative-vote-v2 implementation unchanged: nine arms (hard equal,
spectral, binary L-SML, continuous L-SML bridge, DS, hierarchical EM; soft equal,
spectral, continuous L-SML), tau=1, full threshold grid, weighted training,
training-only preprocessing/orientation/groups, PAVA, earliest mode and fallbacks.
All 6,800 PB and 6,969 PRMB answers, v3 labels and corrected source folds.
PB fits separate q4/q8 models across their four subsets. Primary fits use all
training answers; also run PB error-only fits. PRMB retains independent step
targets and within-answer median votes. Five fixed EM starts and convergence
settings are inherited unchanged. PRMScore uses q=.8 and actual nested source-fold
threshold selection with the same 50-quantile budget.

Anchors: original frozen CT7, MindGap replay, historical token fusion/equal,
the prior eleven-channel fixed and selected rosters, and each CT7 view alone.
Report SLA, common-CT7-gate F1, per-cell results, median/early/late/distance,
within-answer AUROC (6,030 eligible), fold AUROC/AP, PRMScore, coverage and fit health.

Ten thousand shared source-question bootstrap samples; planned comparisons:
each new arm vs CT7, MindGap and the corresponding fixed/selected eleven-channel
arm; soft vs hard for matching cores; learned vs equal; continuous/binary bridge;
DS/HEM vs spectral initialization and HEM vs DS; error-only vs all-answer fitting.
One Holm family across these three primary endpoints. No promotion threshold.

Profile provenance must be verified before fitting. Five bank readouts are
reconstructed from the historical length-study Top10 cache with its documented
float32 storage cast. The chosen-token view is rebuilt from sufficient statistics.
If the original BOCPD view is unavailable, recover it algebraically from the
independent historical six-view equal score (six times the mean minus the first
five views), verify against the independently recomputed BOCPD Top10 profile,
and require the resulting seven-view mean to reproduce CT7 within 1e-12 with
identical locations. Record the numerical differences explicitly; never claim
byte-exact recovery of a component obtained by subtraction.

The original eleven-channel report and CT7 remain immutable. New artifacts live
in `results/cumulative_vote_fusion_v2/ct7_profiles_v1/`. No readout search or token
shuffle is added: those would change the frozen representation and answer a
different question from this controlled bank replacement.
