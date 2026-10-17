# Frozen surprisal-power fusion experiment

User approved the six discussed arms on2026-09-10, explicitly including chosen-
token powers. Separate worktree/branch surprisal-power-fusion-v1, based on the
last reviewed Varentropy comparison5c0f673d. No new graphs, solvers or K sweep.

K=15 fixed, original saved log probabilities (NOT top-K renormalized).
S[t]=[-log(p_rank1),...,-log(p_rank15),chosen_token_surprisal]. Tiny negative
surprisal from float roundoff clamps to0 with existing5e-7 input tolerance.
Input for degree d is [S,S^2,...,S^d], d=1/2/3, so T x16/32/48.
Power blocks each contain15 ranked alternatives and the actual chosen token.
There is no tail channel, probability weighting q, centering around entropy,
new time window or other-answer fitting. All powers use float64.

Each answer's columns are individually zscored using all its tokens offline;
existing scale>1e-10 cutoff and >=3 varying columns/tokens. Equal averages
active standardized columns. IU uses unchanged IU_FIT_DEFAULTS (2-component
L2, no exclusions, no fallback or difficulty gate), with global score sign
oriented to saved token entropy, as in direct-probability fusion. It fits on
this answer only. Failures are recorded, never replaced by another method.
Coefficients, raw-coordinate equivalents, intercepts and active counts are saved.
Normalize AFTER taking powers. Deterministic powers expand a linear score's
function class; they do not add observations or new information. The selected
token is present at every order; its added powers are tested jointly with ranks,
so this design does not isolate their individual contribution.

Full13769 cached answers/145597 steps, same labels/canonical source groups,
folds and per-token/step alignment. Same top10 mean step readout, earliest
step tie break, fixed dual__iu mean-entropy q=.3 PB gate. PRMScore continues
per-method q=.8 cross-fold calibration, excluding held source groups.
Fusion is answer-local; gate and calibration external. Development evidence,
not untouched confirmation. No historical24 whole-answer transfer in this stage.

Primary: degree2-IU minus degree1-IU; degree3-IU minus degree1-IU.10000 paired
canonical-source bootstrap draws,97.5% intervals on PB macro and PRMB within AUC.
Exploratory95%: equal higher orders vs equal degree1; IU vs same-order equal;
all new arms vs RAW Varentropy15 (best prior PB point), vs normalized equal
Varentropy15 (best prior within-AUC point), and vs RAW Varentropy50 (best prior
gray-box PRMScore point). No automatic promotion threshold or best-of-table claim.
Keep frozen direct-IU, Delta-IU, entropy and all six Varentropy arms in the table;
carry historical references incl token-IU, Mind-the-Gap adapter and supervised
Math PRM separately with their access/fidelity labels. PRMScore contrasts are
descriptive here; bootstrap intervals are for PB and within-answer AUC only.

Validation: polynomial layout and dimensions; chosen-token changes affect only
chosen columns; truncate ranks to15; powers formed before normalization;
constant/short/missing data; explicit coefficient reconstruction.27-answer
smoke (shortest/median/95th percentile per cell) for mechanics only. At every
answer verify canonical step boundaries and raw entropy/source alignment;
replay RAW Varentropy50 against frozen benchmark token inputs as a join check.
Hash sources against v2 frozen audit; bind benchmark, reference scores, code,
protocol and input arrays to resumable checkpoints. Full references must replay
all headline metrics. Separate arithmetic result review after scoring. No HTML;
chat table, clear-name per-cell CSV, coefficients, scores, errors and research log.
