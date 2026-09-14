# Temporal research program — execution contract

Authorized by Omri on 2026-09-15 ("RUN THE PLAN"). Based on reviewed commit
cf01849a7 and review a105b7a50. This authorization supersedes older priority
and stage-pause instructions for the bounded work below.

## Scientific contract

Use all 13,769 development answers / 145,597 steps, v3 PRMB labels and v2
source groups. No correctness labels enter feature extraction, model fitting,
weights, routing or checkpoint selection. Development labels may diagnose and
select banks/configurations; report this exposure. Other-answer unlabeled fits
are allowed with source-group exclusion; prefer answer-only when competitive.
Retain the Pareto frontier (PB vs PRMB within), PRMScore, cost and access scope.
Do not require simultaneous gains or promise that anomaly equals reasoning error.

Reference: q15 {H0lim, VE0, VE0.75, VE1}, original orientation, per-stream
step Top10 then natural-unit mean. PB gate: answer Top10 missing top15 mass,
within-cell midrank >= .33. This calibration is transductive and offline.
Reported targets: PB .3747489826, PRMB within .75343585; independent replay
must establish their status before improvement comparisons.

## Ordered work and acceptance gates

1. Audit inputs with explicit path mapping and SHA256; retrieve frozen score
   archives where available; inspect GraphTV completion before duplicating it.
   Repair 47 stale PB bundles in versioned outputs, preserve originals. Compute
   headlines, per-cell detail and suppression counts from one final prediction.
   Correct H1_native to top15 conditional entropy, covariance vs second moment,
   transductive gate language and missing/NaN diagnostics. Correct mistaken
   Diverging Flows attribution and distinguish GMM/KDE precursors from DiFlo.
2. Independently replay reference and matched-gate four singles, entropy15,
   RBM12 and IU. Keep Top10 ordering matched for weight claims. Historical
   longest/first step, first_near_max .25 and earlier VE0/VE0.75 peak are separate
   readout controls. Require complete IDs, denominators and consistent metrics.
3. Complementarity: unique/shared hits, add/drop gains, actual token/step
   position, first-error strata separately, lengths, Top10 overlap, redundancy,
   sign stability and distance/boundary bias. PB later steps are not all errors.
   Expert-peak union is only an oracle for choosing existing peaks.
4. Score all 15 nonempty subsets of the four features. DUFS selects 2/3/4 from
   the existing 31 streams, with unlabeled excluded-source fitting and selection
   stability. Add H0lim and VE0.75 prefix innovations separately (current token
   excluded; empty prefix zero plus mask). Reuse corrected q50/H1/Hinf results.
   Freeze <=3 banks: original, PB leader, within leader; ties fewer features,
   then PRMScore. No additional alpha/gate sweep.
5. First-cycle models: regularized linear predictor; small residual TCN; and
   Diverging Flows alongside standard conditional flow matching. Linear failure
   does not prevent TCN. Learn telemetry, not correctness. Preserve base scores.
   Context crosses steps; no evaluated target in its predictor's fit.
6. Remaining registered comparisons: context-bias then factored CRBM;
   multi-target covariance shrinkage (diagonal/static/position targets selected
   through unlabeled held-block prediction); simplex IU fusion/Top2/Top1;
   Network Lasso coefficients with eta .1 / zero / shuffled graph; static and
   rank1/rank2 coefficient maps with amplitude-only controls. Targeted tests,
   not a Cartesian sweep. LOCA: establish justified repeated measurements first.
   Task-based graph sampling selects fitting observations while scoring all
   tokens, compared with full and uniform same-budget fitting. KalmanNet,
   IMM and BOCPD remain later dynamics work, not fixed-RBM-score rescans.

## Diverging Flows adaptation

Source: Tsakonas, Ivaldi, Mouret, arXiv:2602.13061v2,
"Native Extrapolation Awareness in Flow-Based Conditional Generation".
Conditional flow predicts the next feature vector from current token, previous
16 tokens, masks and relative position. DOT is assigned to the current token
and uses generated paths/endpoints, never the true future observation at scoring.
Compare equal architectures: ordinary FM; FM + repel/curve penalties; prediction
error control. Synthetic negatives: source-probability PGD with derived features
recomputed, and matched-cell/position history swaps preserving token vectors.
Neither perturbation is asserted to be a semantic error.

Default bounded pilot: 3 hidden layers x128, 3 seeds, at most 50,000 updates,
50 integration steps; checkpoints chosen using held-out unlabeled objective.
Standalone DOT is a diagnostic. Step-level combination:
base + gamma * within-answer std(base) * z(DOT), gamma in {0,.25,1}; constant
streams contribute zero. Gate unchanged. Joint DOT does not identify which
feature to route. No imported conformal semantic-error guarantee.

## Verification and outputs

Separate extraction -> unlabeled fit -> selection/weights -> step readout ->
gate -> evaluation. Save token scores, weights/masks, fit health, coverage,
runtime and training groups. Verify independent PB arithmetic, zero-component
identities, label firewall, short/constant sequences and explicit failures;
synthetic switching-feature and no-time-benefit controls; flow integration
stability. Compare real vs shuffled context and scale-only effects. Normalize
for fitting separately from retaining level/scale in scoring. Whole-answer
normalization or relative position makes a method offline.

Full-population paired source-group bootstrap: 10,000 draws with declared
primary-comparison correction. Small runs establish correctness/cost only.
End with Pareto report and attribution to features/weights/context/readout.
Untouched-question confirmation follows method lock; historical24 is separate
retrospective transfer. Missing artifacts or infrastructure remain explicit
blocking conditions, never substituted with silently different data.
