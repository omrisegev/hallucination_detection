# Varentropy contribution fusion: frozen six-arm localization experiment

User authorized the discussed six arms on 2026-09-10. Base commit bab8b402.
Separate worktree/branch: varentropy-contribution-fusion-v1.
No historical24 transfer, new graph, temporal lag or parameter sweep here.

For each token t and probability rank k, use the saved descending logprobs.
K is exactly15 or50. q=exp(lp)/(sum(exp(lp))+1e-12), s=-log(q+1e-12),
H=sum(q*s), C=q*(s-H)^2, exactly matching token_feature_views epsilon rules.
Top-K is renormalized; no selected-token channel, residual-tail bucket or
unobserved full-vocabulary probabilities. These are transformed contributions,
not the earlier17 direct-probability coordinates.

Six arms: K15/K50 crossed with RAW, EQUAL, IU.
RAW=sum(C). EQUAL=mean(zscore(C)) over active columns. IU is the existing
two-component L2 IU-PCR, fitted to the same zscored contribution columns;
global sign oriented by correlation with RAW varentropy from this answer.
EQUAL has no learned sign flip. Fit all tokens of this answer only, offline;
no labels or other answers in fusion, no hidden fallback. The existing
scale>1e-10 active-column cutoff and IU_FIT_DEFAULTS are fixed. For EQUAL/IU,
fewer than3 tokens/active columns fails explicitly; RAW remains computable.
Save standardized weights, equivalent raw-coordinate weights and intercepts.
Signed IU outputs are scores, not mathematical variances.

Full frozen13769 answers (PB6800 in8 cells; PRMB6969), same canonical labels,
groups, outer folds, token-to-step boundaries, top10 mean readout, first-step
tie break and mean-entropy q=.3 PB gate from dual__iu. PRMScore uses the existing
per-method q=.8 held-source-group calibration. Thus fusion is answer-local,
gating/calibration external. All cached data are development, not untouched test.
PB failures count against full denominators; PRMB within AUC reports its eligible
mixed-label answer count and coverage; incomplete PRMScore headline is null.

Two primary comparisons: IU vs RAW at each K, paired canonical-source bootstrap
10000 draws,97.5% CI on PB macro F1 and within-answer AUC. Exploratory95%:
IU vs EQUAL and EQUAL vs RAW at each K; RAW50 vs RAW15. These decompose learned
weighting, normalization and retained-support effects. No promotion threshold.
Keep current/Delta direct-IU and entropy frozen scores beside these arms and
historical varentropy, token-IU and Mind-the-Gap references with access labels.

Preflight: uniform/degenerate/known binary distribution, exact scalar replay,
contribution sum, rank trimming, effective-coefficient reconstruction, same-
answer fitting and short/constant failures.27-answer smoke covers shortest,
median,95th-percentile traces in each cell; it gives feasibility only.
For EVERY scored answer, compare recomputed RAW50 token curve to frozen
benchmark raw varentropy, and verify canonical step spans against saved spans.
Full RAW50 summaries must reproduce Step334, and entropy must reproduce v2.
Hash raw sources against v2 audit; bind code/protocol/folds/gates and benchmark
input arrays to resumable checkpoints. No frozen output is overwritten.
After completion replay metrics separately and report exact gains/losses,
early/late errors, per-cell scores and learned rank-weight profiles in chat.
