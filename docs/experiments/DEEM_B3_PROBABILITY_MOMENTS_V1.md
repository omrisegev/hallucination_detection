# Native soft DEEM and continuous B3 on existing probability/moment banks

Authorized2026-09-11. Parent490f4b6c. Isolated codex/deem-b3-probability-moments-v1.

## Inputs and meaning

Two banks from existing scorers: prob17=[1-p1,p2..p15,selected surprisal,tail];
var15=[q_k*(-log(q_k)-H)^2 for k1..15], exactly the frozen Varentropy convention.
Top-15 are ranked alternative tokens at each chronological token position.
All T rows of the SAME answer fit the model. No donor answers, row sampling,
labels, new windows, extra moments, new graph or threshold sweep.
Var15 deliberately preserves the exact leading contribution bank, which has
no separate selected-token coordinate. The prob17 bank includes chosen and tail.

B3 uses standardized continuous columns and the existing generic B3 energy
implementation. One group contains all active coordinates: this is an explicit
new-input adaptation/special case, not the old spectral-family partition.
Drop only columns with std<=1e-10, as existing scorers. No new sign fitting;
nominal high-risk input conventions remain those of the previous banks.

Native DEEM remains pinned0.2.0 with soft (T,2,P) input. For prob17, probabilities
are retained directly: [1-r,r] for ranked risk and tail, selected coordinate
uses r=1-exp(-selected surprisal). Clip only floating excess to[0,1]. These are
declared risk proxies, NOT calibrated hallucination probabilities. For var15,
reuse existing empirical rank adapter, clip1e-3, [1-r,r]. This changes magnitudes
and must not be described as feeding raw moments to the native categorical model.
Compare against mean of EXACTLY the same soft inputs to expose adapter effects.
The mean continuous and Varentropy references remain separate controls.

## Fixed models and controls

Per bank: standardized continuous mean, mean soft risk, native DEEM soft,
B3 before learning, B3 trained. Ten scored arms, four learned candidates.
B3 retains default100 epochs,lr1e-3,momentum0,width8,MALA5,delta.1,replay.05,
float64 CPU. Native DEEM uses the existing repaired soft configuration:
100 epochs,lr1e-4,momentum.9,one sparsemax preprocessing layer with identity
initialization,one hidden unit,sampler5,cd_k10,batch<=1024,CPU. No auto tuning.
One deterministic UID-derived seed per answer and method, no seed selection.
Report this single-seed limitation. Mechanical tests may use2 epochs; the
real-data feasibility smoke and full scorer always use100.

Both nonlinear models' risk orientation uses the mean standardized input as
an explicit label-free anchor. B3 retains its existing weighted-high/low rule;
native DEEM uses the existing class-permutation alignment helper, identity when
ambiguous and flags it. Never align with correctness labels. Retain finite
collapsed results in the benchmark and disclose health; nonfinite errors fail
explicitly. Save model parameters, normalization and diagnostic histories.
B3 initial control has same seed/architecture/input and no optimizer updates.

## Frozen benchmark and inference scope

Full13769 cached localization model-answer rows /145597 steps. Same source
groups,folds,v3 labels,token spans,top10 mean and earliest tie. Same saved PB
dual__iu entropy q=.3 gate and PRMScore held-group q=.8 calibration. Overall
pipeline includes external calibration; fusion fit itself is answer-local.
No full-answer24 transfer is claimed by this localization experiment.
Nine previous references replay and var15 continuous mean must replay exact
saved var15 equal scores. Report PB all8,Q4,Q8,each cell,clean/exact/early/late,
PRMB within and pooled AUC,PRMScore,coverage,failures,fit health,runtime.

Four primary contrasts: DEEM-soft-mean and B3-trained-initial, in both banks.
10000 canonical-source paired bootstrap draws;98.75% intervals (four contrasts)
for PB and within-answer AUC. Other contrasts exploratory95%. Intervals are
conditional on fixed seeds/fits/calibration, not all research choices or
optimization-seed uncertainty. Full cached development, not confirmation.

Feasibility smoke: shortest/median/95th-length answer in each of9 cells. Inspect
health/runtime, not rankings. CPU cost may be substantial; keep resumable
checkpoints and do not present partial scoring as a completed comparison.
No HTML before discussing results with Omri.
