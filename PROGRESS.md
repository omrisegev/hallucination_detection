## Step396 (Claude) - digit disagreement as gate evidence, 2026-09-16

COMPLETE on all PB answers with locators frozen and the same within-cell midrank>=.33 rule for every detector. Equal within-cell rank mean of tail15 and digit_rate: innovation5 locator 41.0671% PB (clean 63.19%, error exact 28.75%) versus current 39.8314%; digit025 locator 43.2546%/.776036 versus 41.3300%. Primary contrast (innovation5, 97.5%) +1.2357pp [-0.0077,+2.4775]; secondary (digit025, 95%) +1.9246pp [+0.7875,+3.0645]. Digit presence alone is near chance (AUC .568; PB 29.80%), so the gain is the disagreement, not digit count. Development only; untouched confirmation and the historical gate-selection contract comparison remain open. Report results/claude_real_checks_v1/DIGIT_GATE_REPORT.md; protocol docs/experiments/DIGIT_GATE_EVIDENCE_20260916.md. Step 395 (Codex, completed but uncommitted) and Claude's review artifacts are now committed.

## Step395 - corrected alternative views and fusion, 2026-09-16

COMPLETE_REVIEWED on all13,769 answers/145,597 steps/6,968,779 tokens:28 new arms plus30 frozen references. Recomputed seven Claude views (surprisal, censored rank, mass-above, gap, logtail15/50, digit), plus raw-tail controls. Correct tail formula checked on every token; base/digit scores and all30 reference metric rows replay exactly.
All new heads use answer-token standardization/centered covariance, separate per-stream Top10, then one .25 correction to unchanged innovation5. Banks new7 and augmented12 compare equal, family-equal, native IU, diagonal-shrunk IU and block-shrunk IU. Provenance groups are NOT assumed independent-error groups. Gate unchanged; new heads answer-local, other-source-fold PRMScore quantiles; neural references retain nested calibration.
PB%/within/PRMScore: new7 equal39.5741/.765523/.648901, family equal40.1790/.770944/.651520, IU39.0974/.763182/.647318, diagonal39.0973/.763214/.647342, block39.0789/.762915/.647187. Augmented12 equal39.8702/.764164/.646543, family equal40.3862/.768851/.648635, IU39.5951/.761803/.644356; both shrunk heads same PB and within~.7618.
Eight primary pairs x2 endpoints,10000 source-group bootstrap draws,CI99.6875%. IU loses within vs equal in both banks: -.002341[-.003365,-.001355] and -.002362[-.003358,-.001394]. PB intervals include0. Family equal improves within over equal: +.005421[.003985,.007049] and +.004687[.003432,.006018]; PB intervals include0. No positive primary IU/shrinkage gain. Family equal still below digit025 on PB/within; new7 family equal offers a PRMScore tradeoff.
Corrected tails as corrections: logtail15 38.8583/.758550; logtail50 38.5761/.759324; rawtail15 37.0738/.754336; rawtail50 36.5538/.754394. Not useful in this localization recipe; no new gate test. Best non-digit singleton mass-above40.0411/.762159, PB change+0.2098pp with exploratory95% interval[-.3197,+.7252]. Digit025 still41.3300/.776036; previous secondary TCN+digit sum42.0781/.774945/.652284 retained, with different total correction amplitude.
Shrinkage active: median diagonal/block alpha new7 .13148/.05909, augmented12 .04664/.02120; no fit fallback. Among10282 variable-digit answers, IU digit negative2.88%/20.23%, absolute-weight share median2.44%/.55%. Thus do not reuse prior IU6's nearly-universal negative-digit explanation; new banks mostly keep the sign but assign little weight.
Operational PB raw-failure phi: surprisal/gap.981, logtail15/logtail50.864, logtail15/base.665, digit/base.103. Digit standalone has794 unique raw hits but loses989 base raw hits; digit025 gains310/loses239 final hits. Error complementarity is not latent-residual independence. Same-answer/pair diagnostics include PB cell/position strata and PRMB answer-balanced ranking errors; no labels enter fitting. Full4442-row ledger, historical707 rescues and plots saved.
All58 PB/within bundles independently checked; five numerical tests pass; canonical IU and provided probability discrepancies0. Scoring152.18s/evaluation72.82s; no models/gate/flow restart. Development-only, secondary95% contrasts not winner-corrected. Decision: retain digit/TCN+digit; no broader bank adoption or automatic parameter sweep.
Report results/alternative_views_fusion_v1/REPORT.html; map docs/reviews/fusion_insertion_map_2026-09-16.md distinguishes raw-view, predictor-residual, pre-innovation background, gate, first-error decision and final-answer targets. Bounded future questions: conditional digit evidence at gate; fusion of forecasts of the same observable feature before residualization. Neither launched here. Bulk scores/checkpoints remain local with hashes.

## Step394 - tail-screening normalization audit

Code/formula audit only: Claude new-view screening subtracts token_logsumexp from already normalized saved logprobs. Its tail15/tail50 rank1/correlation results are invalid as measurements of true missing mass. One71-token implementation check reproduces near-one malformed tails versus correct tail15 up to.00899; not a quality subset experiment. The current residual_tail_mass gate is correct; digit-ID results are unaffected. Corrected full-population tail screening remains pending. See docs/reviews/claude_tail_normalization_audit_2026-09-15.md. Seven views were screened; only digit reached the reported full fusion experiment. Operator/equality and digit-gate ideas remain proposals.

## Step393 - digit disagreement replay and fusion

Step393 COMPLETE_REVIEWED: independent digit-disagreement replay and a fixed12-arm full-data fusion/control experiment on all13769 answers/145597 steps/6968779 tokens. Verified ASCII digit IDs15..24 from both cached Qwen tokenizers, scalar event predicate on every token, sorted top-k, spans, baseline and Claude's auxiliary/scores. One teacher-forced pass reused; no model training, operator extension, gate change or flow restart.
PB%/within/PRMScore: base39.8314/.760293/.638830; digit02541.3300/.776036/.649780; digit1 41.1806/.779274/.649223; TCN40.9718/.761592/.641153; TCN+digit sum42.0781/.774945/.652284; amplitude-matched TCN+digit41.4627/.772607/.649612.
Controls: digit presence39.7521/.771779; disagreement-per-digit rate41.1532/.774964; disagreements permuted among provided-digit positions38.9219/.760928. Real disagreement beats permuted location on both primary endpoints; within also beats presence. Rate retains a gain in points. These controls do not prove removal of all length effects.
Six primary pairs x2 endpoints,10000 source-group bootstrap draws,CI99.5833%. Digit025-base PB+1.4986pp[-.1632,+3.0960],within+.015744[+.012740,+.018865]. Same points as Claude; its historical97.5% contrast remains recorded, but the broader primary correction includes0 for PB. Digit025-permuted PB+2.4080pp[+.8932,+3.8858],within+.015109[+.011878,+.018292]. Matched TCN+digit versusTCN: within+.011015[+.008581,+.013455],PB interval[-.7061,+1.7018]pp includes0.
The42.0781% sum is a prespecified SECONDARY arm: versusTCN PB+1.1062pp,exploratory95%CI[+.1269,+2.1044],within+.013353[+.011450,+.015226]. Keep as development candidate/Pareto tradeoff; no primary-confirmed or untouched winner claim. Higher gamma and auxiliary selection were already exposed to development outcomes.
Direct bank comparison, same token standardization/per-stream Top10/readout: bank5 equal39.3043/.755063;bank5 IU39.3825/.757218;bank6 equal41.3961/.773468;bank6 IU30.5284/.637115. IU6 loses to equal6 and IU5 with adjusted negative intervals in both endpoints. Digit coefficient negative in99.98% of10282 variable-digit answers,median-.27978;30 additional variable-digit canonical fits reproduce weights. Not convergence to equal: the new view is largely subtracted. No label-guided sign fix applied.
Digit is constant in3487 answers. Participation ratio on mean within-answer covariance: bank5 1.34024 -> bank6 1.71884; among variable-digit answers1.33970 ->1.83060. Claude's3.55 was for ALL seven new views, not digit alone. Low covariance and higher effective dimension do not imply a more accurate IU reliability estimate.
Final error hits: digit0251339(+310/-239 vsbase; common885 raw30/open707 final28); TCN+digit1385(+255/-204 vsTCN,net51; common885 raw16/open707 final14); matched1358(net24 vsTCN); equal6 1354(common70759). Standalone digit hits145/707,143 with positive digit evidence at gold; do not count zero ties as positive telemetry. Full4442-row ledger and length/digit-count strata saved.
All30 PB/within metric bundles independently audited;18 reference rows and all3 Claude rows replay exactly, including PRMScore. Nested PRMScore calibration reuses15 excluded-group TCN score sets; no cross-fold model exposure. Five new tests passed. Source, tokenizer, score and model-score provenance saved.
Decision: retain digit correction and TCN+digit as useful development fusion candidates; reject this native IU6 bank. Do not equate simple-combination gains with learned-IU gains. Next algorithmic question is how to respect a useful sparse view's direction while estimating reliability under the dependent old bank; no automatic post-result clipping or hyperparameter sweep in this stage. Report results/digit_fusion_v1/REPORT.html.

## Step392 - complementary step evidence fusion

Step392 COMPLETE_REVIEWED: one fixed three-view step-evidence bank on all13769 answers/145597 steps. Preserved innovation5 base ONCE; compared bounded .25 corrections from TCN signed Top10, raw last4, and per-stream best contiguous10. Answer-local step covariance/standardization; maintained canonical IU2PC versus equal, three singletons, two equal pair ablations, and shape-order shuffled controls. No predictor training, alpha/gate/subset sweep, or new first-error decoder.
PB% / within / PRMScore: TCN40.9718/.761592/.641153; end38.0611/.761946/.642517; sustained38.3463/.754040/.638746; equal three39.6301/.761989/.643185; IU three39.0007/.755534/.639643. Equal context+end39.5281/.765898/.644000 is a secondary Pareto tradeoff, not a replacement. Equal context+sustained39.8711/.759397/.640088.
Five primary pairs x2 endpoints;10000 source-group draws,CI99.5%. IU-TCN PB-1.9712pp[-3.1357,-.7801],within-.006058[-.009134,-.003069]. Equal-TCN PB-1.3418pp[-2.5859,-.1335],within+.000397[-.002084,+.002921]. IU-equal within-.006455[-.008535,-.004398],PB interval includes0. True shape order improves within over shuffle for both heads, with adjusted positive intervals, but PB intervals include0.
Secondary context+end versusTCN: within+.004306,exploratory95%CI[+.002627,+.005931]; PB-1.4437pp[-2.3416,-.5778]. Earlier bank and last4 motivation used development outcomes; no untouched confirmation or primary multiplicity claim for this ablation.
Final PB error hits: TCN1334; equal1256(gain143/lose221); IU1232(gain120/lose222); context+end1254(gain156/lose236). Among fixed885 prior raw misses, equal finds1 raw/0 final;IU1/1;context+end5/3. Direct last4 previously recovered119 raw, but a bounded last4 correction is a different detector; no contradiction and no demonstrated recovery of the broad missed cohort.
Native IU13710/13769, explicit equal fallback59 answers with fewer than3 steps. Median weights[.20871,.13505,.25262];59.19% native fits contain a negative coefficient;99.16% g2 at ceiling;42.58% have a Spearman pair>=.75. Three-pair identity is exactly identified, not an assumption test. Few steps remain a limitation. No correlated-view automatic bank search.
Reused15 source-excluded TCN score sets including10 pair exclusions for PRMScore calibration. All27 metric bundles independently audited for PB/within;18 reference headlines and context identity replayed. Five unit tests pass; all-answer weighted-score replay max1.78e-15, baseline max5.33e-15, canonical weight max4.58e-16. Scoring150.68s plus45.36s evaluation; existing FM/DiFlo queue unchanged.
Decision: do not adopt three-view equal/IU; retain context+end as a development tradeoff for within, and preserve current references. No automatic follow-up sweep. Report results/step_evidence_fusion_v1/REPORT.html; full table, cell metrics, corrected intervals, error ledger, input/score hashes and scope saved.

## Step391 - audit readout claims; retain fusion as an open research option

Replayed all eight Claude readout ranks on all4442 PB error answers from frozen tokens; innovation5 score agreement max3.56e-15. No training or new quality sweep.
Among885 previously common raw localization misses, last4 yields119 exact raw peaks (83 open,36 closed); the eight-readout label oracle yields244 (176 open,68 closed). This disproves a zero-recoverability claim, not evidence of net quality improvement.
For707 open common misses, any-stream maximum hit18.95%; independent-five reference42.64%, common circular shift33.82%, shared permutation35.85%. Geometric references do not correct failure-cohort selection or boundary nonstationarity; no below-chance significance or information-impossibility conclusion.
The643 gated oracle successes span the archive. Actual correct peaks suppressed: innovation5 321, ridge310, TCN316, leading IU313. Top2 is not exact localization. Earlier/later peak buckets are relative to innovation5, not the41% leader.
User explicitly welcomes fusion. Next research direction remains complementary evidence about each step, with historical readout audit, standalone/equal/IU comparisons and full13769-answer evaluation. Do not reopen identical weight sweeps or assume missing per-stream maxima excludes fusion. No method promotion or neural restart.
Review: docs/reviews/claude_readout_claims_audit_2026-09-15.md. Replay and new geometric diagnostics: scripts/audit_claude_readout_claims.py; results/readout_claim_audit_v1/AUDIT.json. Claude source files preserved.

## Step390 — common missed-error profiles

Step390 COMPLETE_REVIEWED: descriptive missed-error atlas on all4442 PB error model-answer records (1979 sources), within the13769-answer benchmark. No model fitting, score changes or new method selection.
Initial32-combination cohorts: found1589, gate-open missed2032, gate-closed821. Adding five standalone predictors reduces final common misses to2807. User then requested all-method common failures: examined11 aligned full-score archives,254 columns,178 unique score arrays,173 distinct all-answer peak vectors. Full-array shared-reference anchors matched for every archive. Includes weak methods/controls; not every historical experiment or completed flows.
Broad union:2914 found with current gate;1528 final misses=885 with no correct peak even without gate+643 correct peaks suppressed.885 raw localization misses comprise707 open+178 closed. Without name-tagged controls, gate-open misses946 rather than707. Union is a label oracle, not an attainable detector or fusion bound.
All885 raw misses vs3557 with a correct raw peak, medians: tokens778 vs631; steps9 vs7; error-step fraction.1051 vs.1994; feature-change L2 1.5639 vs1.8384; predictor disagreement.2505 vs.2707. Overlapping distributions, not a universal signature.
Gate-open broad comparison707 vs2914, after cell/length-quartile/position-third standardization: error-step fraction difference-.07722,CI99.5833%[-.08830,-.06672]; feature-change L2-.43738[-.62415,-.24749]; disagreement-.02900[-.03706,-.02063]. Support2405/650, or2066/628 for history-dependent change.10000 source-group draws,6 diagnostics x2 contrasts within each analysis. Post-selection descriptive associations; repeated population, no causal or future quality claim.592 broad draws have an empty retained stratum; fixed harmonic weights renormalized over available strata.
Boundary control added after inspecting event plots: in missed cohort684 paired answers, first4 feature mean.822 error versus.762 preceding correct; found2471 pairs .910 versus.574. Step-onset spikes are not error-specific. Labels are steps, not token error positions; aligned positive offsets may leave the annotated step.
Among707 gate-open common misses, TCN first-error median rank5, only42 in top2; IU median5/top2=44. All-archive oracle has a peak one step away in494; this is not a deployable decoder. One inspected math case already contains questionable reasoning before its annotated first error; no relabeling performed.
Plots: full distributions, feature/residual event profiles, paired correct-boundary control, full-token traces for a covariate-matched found/missed pair, two earlier text examples and a remaining broad common miss. Scalar model predictions are means of five feature predictions reconstructed exactly from cached residuals, not full model vectors. Offline answer z-normalization is explicit.
Report: results/predictor_error_profiles_v1/REPORT.html; RAW_OVERVIEW.png shows all885 common raw misses. COMMON_MISSES.csv lists1528 final misses with flags; ANSWER_DIAGNOSTICS.csv covers4442. Data, score and source hashes, cohort accounting, text/span identities, scalar reconstruction and artifact links verified. No quality experiment, neural restart or automatic method promotion.

## Temporal research execution — 2026-09-15

PREVIOUS — Step389 COMPLETE_REVIEWED
All16 subsets of ridge,tcn,bocpd,noreset,mean16 at sizes3/4/5 x IU/equal=32 configurations; all13769 answers/145597 steps/6968779 tokens. Reused15 excluded-group fits and nested calibration; no new training. Answer-standardized signed predictor residuals -> answer-local canonical IU2PC or equal -> token Top10 -> signed.25 correction -> innovation5 base ONCE, same tail15 gate.
PB leader: IU ridge+tcn+noreset41.0378%/.760656/.640647; versus TCN40.9718%/.761592/.641153 only+.0659pp, exploratory95%CI[-.3226,+.4354]pp. No demonstrated replacement.
Within leader: equal ridge+bocpd+noreset40.5585%/.763829/.642625. Balanced equal ridge+tcn+bocpd40.9665%/.762346/.641925. IU ridge+bocpd+noreset40.7726%/.763281/.642567. Retain point Pareto; no single winner.
Best PB by size:3 above;4 IU ridge+tcn+bocpd+noreset40.9681%/.762033;5 IU all40.7495%/.762642. Best within by size:3 above;4 equal ridge+tcn+bocpd+noreset40.7209%/.763351;5 equal all40.6251%/.763231.
16 primary matched IU-equal pairs x2 endpoints;10000 source-group bootstrap draws,CI99.84375%. No positive primary interval on either endpoint. All16 PB intervals include0; IU within lower in15/16 point estimates, four adjusted negative intervals. Leading PB triple within-minus-equal -.002243,CI[-.004226,-.000202]. Do not infer equivalence from intervals containing0.
PB-leader median standardized-residual weights[.3549,.3800,-.1394];99.88% answers have a negative coefficient. noreset is chiefly subtracted, not positively voted. This does not establish a semantic noise-removal mechanism. Three-view pair fit is exactly identified, so zero residual is not an assumption check.
All48 metric bundles independently audited; all32 readouts replayed across full population,maxdelta1.07e-14. All16 reference headlines exact; singleton scores replayed under15 outer/nested exclusions; canonical weights match on50 audit answers, three tests with45 covariance checks PASS. Scoring691.79s,evaluation90.77s.
Decision KEEP_POINT_PARETO_NO_DEMONSTRATED_IU_ADVANTAGE. Development-only seed0; selector uses labels, fitting does not.96 secondary comparisons have exploratory95% intervals without winner-selection correction. Old FM/DiFlo queue unchanged,paused4/90.
Post-selection PB error audit:4442 error model-answer records,1979 source groups. All32 miss2853:821 gated closed,2032 gate open but wrong localization. IU PB leader gains45/loses39 vs TCN (net6); equal within leader gains103/loses122. All IDs and step decisions in PB_ERROR_LEDGER.csv; four deterministic GSM8K/Q4 text examples in ERRORS.html. Descriptive, not a new selection or significance test.
Report: results/predictor_subset_iu_v1/REPORT.html; METRICS.json contains all32+16 results, per-cell bundles,112 contrasts and diagnostics; AUDIT.json verifies coverage and provenance.

PREVIOUS — Step388 [Codex TCN] COMPLETE_REVIEWED, standalone predictors before fusion.
All13769 answers/145597 steps/6968779 tokens;15 source-excluded fits, seed0 only.
Existing TCN, innovation5, signed .25 residual correction, Top10 and tail15 gate
unchanged. Reused old fold0; completed4 outer+10 pair-exclusion calibration fits.
PB% / within / PRMScore:
- tcn__real: 40.9718% / 0.761592 / 0.641153
- ridge: 40.8472% / 0.761620 / 0.641765
- bocpd: 40.3676% / 0.763223 / 0.642268
- noreset: 39.8608% / 0.762839 / 0.642239
- innovation5: 39.8314% / 0.760293 / 0.638830
- tcn__shuffled: 39.8180% / 0.761273 / 0.641283
- tcn__zero: 39.4342% / 0.760106 / 0.640019

6 primary pairs x2 endpoints;10000 source-group bootstrap draws,CI99.5833%:
- TCN minus ridge: PB +0.1246pp [-0.5355,+0.8233]; within -0.000028 [-0.001343,+0.001288]
- TCN minus bocpd: PB +0.6042pp [-0.4108,+1.6229]; within -0.001631 [-0.003680,+0.000544]
- TCN minus noreset: PB +1.1111pp [-0.0386,+2.3124]; within -0.001247 [-0.003906,+0.001581]
- TCN minus innovation5: PB +1.1405pp [+0.0883,+2.1731]; within +0.001300 [-0.001299,+0.003976]
- TCN minus tcn__shuffled: PB +1.1539pp [+0.1292,+2.2830]; within +0.000319 [-0.001880,+0.002507]
- TCN minus tcn__zero: PB +1.5376pp [+0.5082,+2.5485]; within +0.001487 [-0.000848,+0.003945]

Prediction MSE: TCN 0.647035, Ridge 0.688936.
TCN/Ridge signed residual median correlation .963236; after removing shared
current scalar .902422. Better telemetry prediction is not a detection proof.
Architecture audit:16 input slots but15 effective past tokens; unchanged.
Shuffling16 slots can change which of the15 visible observations is omitted;
not a pure order-only intervention. Zero-history is also an inference
intervention; the current innovation target still contains past information.
13 tests PASS; independent PB/within on16 methods; all13 reference headlines
exact; signed readout independently replayed on all answers, maxdelta0.
Source/data/checkpoint/group provenance verified; no fitting failures.
Decision: KEEP_RIDGE_REFERENCE_TCN_CONTEXT_EVIDENCE_NO_FUSION_YET.
Development evidence only; bank/readout selected earlier on this population.
No predictor fusion fitted. Original90-job FM/DiFlo queue remains paused4/90.
Report: results/tcn_aligned_predictor_seed0_v1/REPORT.html.
Full audit: results/tcn_aligned_predictor_seed0_v1/AUDIT.json.

PREVIOUS — Step387 [Codex predictors] COMPLETE_REVIEWED, predictors BEFORE fusion.
User asked for standalone predictor evidence before combining models. Results:
results/aligned_context_predictors_v1/REPORT.html, METRICS.json, AUDIT.json.
All13769 answers/145597 steps/6968779 tokens. Same innovation5 base, signed
residual mean over5 standardized features, Top10, .25 correction, tail15 gate.
Saved source-excluded Ridge16 versus mean16 and exact untruncated Gaussian
BOCPD predictive means (5 separate filters, hazard1/32, unit variances), plus
noreset and zero-prediction controls. Local recurrences borrow no answers;
Ridge is externally fitted. Whole-answer normalization remains offline.

PB% / within / PRMScore / predictionMSE (equal answer mean):
ridge40.8472/.761620/.641765/.688936;
mean16 39.8648/.759979/.639336/.957699;
bocpd40.3676/.763223/.642268/.818527;
noreset39.8608/.762839/.642239/1.015762;
zero39.3977/.760116/.640378/1.000000;
innovation5 reference39.8314/.760293/.638830.
BOCPD-base within+.002930,PRIMARY99.5%CI[.000140,.005692]; PB+.5363pp,
CI[-.5828,+1.7158]pp. BOCPD-Ridge PB-.4796pp/within+.001603, both primary
CIs include0. BOCPD-noreset both primary CIs include0: no demonstrated
change-point-specific benefit. mean16 has no positive primary interval.
5 pairs x2 endpoints,10000 source-group draws,Bonferroni99.5% intervals.

Ridge is a better feature predictor than BOCPD, yet BOCPD's within point is
higher; noreset predicts worse than zero yet gives .762839 within. MSE cannot
select semantic detection quality. BOCPD residual/Ridge correlation median
.9103; after linearly removing shared current observation .7258 (diagnostic,
not independence). BOCPD gains115 final PB error hits and loses139 vs Ridge.
Complementarity of existing peaks is not a usable label-free fusion rule.
Decision KEEP_RIDGE_AND_BOCPD_NO_FUSION_YET; keep noreset as mandatory control.
Ridge/BOCPD are the two-endpoint point Pareto. All results development-only.

13 tests PASS (4 new exact partition/no-future/edge tests +9 context/calibration),
independent PB/pairwise-within for all13 methods, all9 reference headlines
exact. Scalar/sort readout checks for13769x5 and Ridge score replay both have
max delta0. All source/bundle hashes match. No missing scores/fallbacks.
Scoring800.84s + evaluation21.95s CPU/BLAS1. Neural queue unchanged,paused4/90;
no TCN/FM/DiFlo quality conclusion. No fusion fitted. Frozen protocol:
docs/experiments/ALIGNED_CONTEXT_PREDICTORS_20260915.md.

Previous — Step386 context-weighted original levels COMPLETE_REVIEWED.
Results: results/context_weighted_levels_v1/REPORT.html, METRICS.json, AUDIT.json.
All13769 answers/145597 steps/6968779 tokens;19 policies +9 historical anchors.
45 Step384 source-excluded fits reused;10 nested PRMB pair-exclusion fits.
Up to16 weight updates/answer, held until next anchor; first16 tokens static.
Every original feature token remains eligible for unchanged per-stream Top10;
multiply selected contributions by their token's weight. Context crosses steps.
Three heads(native2PC,full-C simplex eta.25,one-parameter group), each static,
energy,position,random,energy amplitude-only,energy direction-only; plus equal.

PB% / within / PRMScore:
innovation5 39.8314/.760293/.638830;
native static39.3306/.757226/.637858 vs energy33.7873/.730749/.614660;
native amplitude39.0222/.756863/.636882 vs direction34.1433/.729696/.613486;
simplex static39.1949/.758554/.638971 vs energy39.2463/.757929/.639523;
group static39.6369/.760134/.638996 vs energy39.5998/.759975/.638841.
Historical secondary additive ridge remains40.8472/.761620/.641765.
All8 primary CIs (simplex energy minus static/position/random/amplitude,
PB/within,99.375%,10000 source-group draws) include0. No primary improvement
or equivalence established. Native energy-static PB-5.5433pp,secondary95%CI
[-6.9225,-4.1895]pp; within-.026477,[-.030966,-.022138]. Direction changes
remain harmful with amplitude normalized; stable context weights != useful ones.

Important limitation: post-ranking label-free schedule audit finds45.62% of
selected feature-token contributions AFTER warmup use weights >=16 tokens old;
median age14,p90=48. Warmup is4.87% of contributions. This does not establish
staleness as the cause, but rules out generalizing the result to exact every-token
context fitting. It is the frozen16-anchor policy, not every contextual algorithm.
Static means fixed WEIGHTS: innovation5 itself already contains past information.

22 tests PASS; independent PB/pairwise-within for all28 methods; scalar dynamic
readout replay30 answers x19 policies,maxdelta1.78e-15. All9 historical anchors
exact in headline metrics; no missing scores or silent fitting fallback. Fit/score
1184.3s, evaluation/audit36.1s,BLAS1. Neff minimum62.39 distinct-source weights.
Decision NO_16_ANCHOR_POLICY_PROMOTION_EXACT_TOKEN_CONTEXT_UNRESOLVED.
Keep original features/innovation5 and the existing additive ridge reference.
Separate refresh-frequency limitations from reliability identification before
investing in a richer context for the same IU head. No eta/bank/refresh tuning
was added after seeing outcomes; all results remain development evidence.

Fixed protocol:
docs/experiments/CONTEXT_WEIGHTED_LEVEL_FUSION_20260915.md.
Omri explicitly keeps FM/DiFlo open; neural queue remains paused4/90 pending
instrumented continuation, not closed. Integration roadmap:
docs/reviews/context_fusion_flow_bridge_2026-09-15.md. No new flow training here.
FM/DiFlo may provide learned context/predictive descriptors for neighbor/IU
weights, or DOT/additive error evidence alongside original scores. DOT is a
flow-path statistic, not a third model or per-feature reliability vector.
Current flow condition includes current observation for next-token generation;
past-only bridges must align/lag it explicitly. No combined method is trained.

Previous — residual moments x scored representation COMPLETE_REVIEWED (Step385).
Results: results/residual_moment_fusion_v1/REPORT.html, METRICS.json, AUDIT.json.
User authorized unused worktree cleanup and quick synthetic->FULL real quality.
Removed clean antigravity/direct-probability-fusion-v1 worktrees without force;
branches retained, no unique/uncommitted artifacts removed; freed about1.6GB.
Frozen RESIDUAL_MOMENT_FUSION_20260915.md: innovation5, crossed L/R moments x
L/R scores x native2PC/full-C simplex(tau1,eta.25), local and source-excluded
pooled moments.18 policies/controls +9 references, all13769 answers/145597 steps/
6968779 tokens.27 complete independent PB/pairwise-within checks;22 tests and
60 real canonical coefficient/readout checks PASS. No fit failures/fallbacks.

The synthetic temporal nuisance mechanism works only under its assumptions:
native LL->RR .78265->.96774 with fast target/slow nuisance, but .78322->.49598
with slow target/fast nuisance.20 seeds x5 worlds, OOF training residuals,
shared level units; not exact numerical replay of Claude's in-sample/std-R arms.
Real equal residuals: PB36.3349%/within.651336, vs innovation5 39.8314%/.760293.
Native residual fusion improves over equal residuals, but remains far below
the level-score anchor. Primary RR-LR within gains are positive in native:
local+.007393 / pooled+.000988, corrected99.6875%CI, no clear PB gains.
Secondary pooled simplex RL (fit residuals, SCORE LEVELS):39.8459%/.761112/
PRMScore.639269. Versus innovation5: PB+.0145pp,95%CI[-.4502,+.4747]pp;
within+.000819,95%CI[.0000175,.0016361]. Secondary, no corrected winner.
Local simplex RL gives .759895 within but PRMScore.598094: calibration matters.
Historical SECONDARY ridge signed.25 remains40.8472%/.761620/.641765, dominating
new arms at point estimates. It is not a new independent confirmation.
Local rho L/R cosine median.99847; no synthetic-like reliability rotation.
g2 ceiling local L100%,R99.26%; no new adaptive-eta policy justified here.

15 frozen ridge fits reused;10 triple-excluded ridge fits added only for proper
nested reference residuals. Current query and donor groups excluded from their
predictor; no in-sample residual covariance. Whole-answer normalization, fixed
q=.33 transductive gate; Top10 per stream BEFORE weights, including negative
weights. First16 tokens included. Native raw coefficients normalized by L1;
this fixed readout convention is not every possible native-IU implementation.
Local weights use current-answer covariance plus EXTERNAL ridge. No strict
answer-only or online claim. Weights constant within each answer in this stage.
Synthetic19.8s; full scoring277.7s; evaluation/audit37.5s, BLAS1. Source-group
bootstrap10000,8 primary pairs x2 endpoints=16 corrected comparisons. All remain
development outcomes, including earlier innovation5 bank selection by labels.
Decision: KEEP_LEVEL_SCORES_NO_RESIDUAL_ONLY_PROMOTION. Keep prior innovation5
and additive ridge residual path. Step384 conditional energy weighting remains
unscored: if continued, one matched dynamic-vs-static test on ORIGINAL scores,
with position/random/amplitude controls; residual moments an ablation, not a
preselected winner. Neural queue still paused4/90; no CCA/flow expansion.

Previous — real-bank context/weight stability COMPLETE_REVIEWED (Step384).
Results: results/energy_context_stability_v2/REPORT.html and SUMMARY.json.
All13,769 answers /220,292 landmarks /3,483 source groups;45 source-excluded
fits over9 cells x5 folds. Label-free diagnostic ONLY, not PB/within scoring.
Fixed past16 energy context, training-only position/length profile removal,
K64 distinct groups, .5 covariance borrowing. Canonical IU moments feed native
2PC, full-C simplex tau1/eta.25 and fixed one-parameter group weights.
Held feature NLL: static3.657672, position3.580509, energy3.215575, random3.709995.
Energy-position delta-.364934, descriptive grouped95%CI[-.376860,-.353518];
better in9/9 cells. Direction-change/bootstrap-noise median: nativePCR3.267
(45/45 above1), simplex1.227 (36/45), versus random1.027/.978. Ratios are
conditional diagnostics, not significance tests or semantic reliability.
g2 remains almost always at its prior ceiling. Fixed group axis explains only
.008829 of raw simplex updates (median); extra motion is not proven useful.
NLL evaluates both mean and covariance, whereas IU uses covariance. No claim
that NLL alone validates fusion. Energy ignores ordering inside its16-token
window. First16 tokens are not diagnostic targets; full quality must cover them.
Step381 original innovation5 remains the real-data quality anchor, including
its development-label bank selection and position controls. No neural restart.
Decision: STABLE_CONTEXT_STRUCTURE_NOT_DETECTION_VALIDATION. Next: register
one matched full-token IU fusion comparison with static/equal/position/random
and amplitude controls, plus restrained simplex/group isolation; measure cost
before rollout. No CCA expansion or new algorithm family justified here.
Numerical v1 stopped after33 fits on a QP objective deadband; original artifacts
preserved. Removing the1e-13 objective deadband left KKT tolerance unchanged.
All45 fits rerun in v2;33 old non-simplex arrays bitwise identical; simplex
coefficient delta<=6.56e-8.16 tests pass,180 real canonical checks,881,168 NLL
and all landmark histories/KKT independently audited. Fits452.74 seconds,
single BLAS thread, excluding prep/report. Large NPZ archives are local with
hashes; summary/fit metadata and source-group aggregates are committed.

Previous stage — CCA/IU isolation gate COMPLETE_REVIEWED (Step383).
Results: results/cca_iu_isolation_gate_v1/REPORT.html and REPORT.md.
Frozen S0 replay60/60 matches (max numeric delta1.42e-14); no old artifacts changed.
New full-covariance simplex uses tau1, eta.25 and canonical additive IU moments.
All20 seeds x3 worlds x31 methods =1860 audited score bundles. Synthetic ONLY.
Informative AUC: static QP .849640; known-context population QP .877768;
DSP-context QP .879522; past-energy QP .874187; linear CCA .849622;
second-moment CCA .852372; energy one-parameter group mixture .873368.
Supplied-context and energy arms pass the predeclared S0 gain/safety rules;
all three CCA variants fail gain/wins. Nuisance retains small tolerated harm
(energy .737247 vs static .739528), not zero loss or proven semantic reliability.
Population native2PC: informative .973425, nuisance .505853; failure can persist
without covariance sampling noise. Known-rho oracle QP .911706/.819271 exposes
moment/scale limits; it is unavailable without synthetic target knowledge.
Decision STOP_CCA_FULL_RUN_CONTEXT_WEIGHTING_FEASIBLE. No PB/PRMB quality or
real-bank stability run yet; no tuning to these seeds. Next bounded proposal:
unlabeled stability of energy-context C/rho/weights versus position/random,
including one-parameter grouping. Neural queue remains paused4/90.
8 tests PASS;180 population canonical-moment checks PASS; independent pairwise
AUC and weighted-score reconstruction PASS. Runtime legacy145s + new97s.

Active branch: `codex/temporal-research-20260915`, worktree
`.worktrees/temporal-research-20260915`, based on review a105b7a50.
Authorized program: `docs/experiments/TEMPORAL_RESEARCH_PROGRAM_20260915.md`.

Full baseline and all 15 subsets COMPLETE on 13,769 answers / 145,597 steps.
All 10 raw caches and frozen evaluation/fold/gate inputs match the remote SHA256
manifest. Independent raw replay exactly matches PB .37474898261944,
within .7534358509472404 and PRMScore .6344124357811041 (zero differences).
Outputs: `results/temporal_research_baseline_v1/`.

Initial innovation-H0lim augmentation: PB39.8314%, within .760293, PRMScore
.638830. Paired development 95% CIs: PB gain [+1.2973,+3.4287]pp; within gain
[.005072,.008679]. Not untouched confirmation. Mechanism controls COMPLETE:
duplicate/centered H0lim PB37.0518%/within .751985; shuffled-prefix
36.9548%/.752373. True prefix independently checked for every token.
Matched final-gate RBM12: PB37.1253%/.745204; crossfold gate on baseline
37.4761%/.753436. Independent scalar PB audit passes all reported methods.
See `results/temporal_research_mechanism_v1/`.

All 47 historical PB bundles independently replayed and repaired: PASS,
zero changed PB headlines; originals preserved. See
`results/temporal_historical_pb_repair_v2/REPAIR_REVIEW.json`.
DUFS31 CORRECTION: scores computed for 55 source-excluded selectors and all
13,769 answers, with nested PRM calibration; sparse selection NOT VALIDATED.
All gates are nearly open (effective features30.79-31), and mean pairwise
top4 seed overlap is .053. Historical k2/k3/k4 PB36.6665/36.7864/36.8593%,
within .741841/.741765/.741530 are observations of unstable top-k policies,
not evidence that a validated DUFS selector lost. Original scores preserved.
Final expensive banks remain original4 and innovation5 independently of DUFS;
`results/temporal_dufs31_v1/BANK_SELECTION.json` supersedes provisional subset lock.
Full token-position/complementarity diagnostics and independent DUFS PB audit
are in `results/temporal_feature_diagnostics_v1/`.

Important counterevidence: historical earlier-VE0/VE075-peak readout PB39.3857%.
Innovation gain over it is only .4457pp, exploratory 95% CI[-.6910,+1.6031]pp.
Innovation improves early-error hits400->533 but reduces late-error hits301->256.
Do not describe the innovation as uniformly better chronological detection.

TCN/Flow/Ridge APIs, 14 contract fixtures and existing normalization/bank checks
PASS. TCN and DiFlo training/scoring smoke checks pass; no quality claim from them.
Full linear context experiment COMPLETE_REVIEWED in
`results/temporal_linear_context_v1/`: 30 source-excluded fits, all answers,
nested PRM calibration and 38 independently audited PB bundles. Primary squared
residual .25 on innovation5: PB40.1386% / within .757525 (within loss CI excludes0).
Secondary signed residual .25: PB40.8472% / within .761620 / PRMScore .641765.
Its PB gain over innovation5 has exploratory95%CI[+.2877,+1.7480]pp; within
gain CI includes0. Both PB/within CIs versus shuffled history include0. Do not
claim an established advantage from exact lag order or promote secondary to primary.
First DiFlo training and scoring COMPLETE: innovation5, seed0, excluded fold0,
31,000 updates, best checkpoint28,000; all2,782 held-fold answers scored.
This is not a full-population neural quality result. Checkpoints in
`results/temporal_context_models_v1/diflo__innovation5__seed0__exclude0/`.
The original watcher ended after that scoring run; it did not chain other fits.
NEURAL QUEUE PAUSED_BETWEEN_JOBS:4/90 completed; STOP_AFTER_JOB remains.
The previously active DiFlo/original4/fold0 job also finished. FM, DiFlo and
TCN innovation5/fold0 scored all2,782 held answers each. No new jobs launched
while preparing the CCA proposal. The matrix must not auto-continue.
See `results/temporal_neural_queue_seed0_v1/RUN_STATE.json` and per-job logs.
No training/readout/gate hyperparameters changed. Seeds1/2 remain unlaunched.
Full three-seed matrix has270 fits; no new cluster job was submitted.
Flow uses actual repel/curve losses and generated-endpoint DOT,
not GMM/KDE. Verified paper PDF/digest cached.
AIRCC explicit-config check still times out on this status check; GraphTV unknown.
User has been asked to restore TAU VPN. No new cluster job submitted.

Review follow-up: `docs/experiments/TEMPORAL_REVIEW_FOLLOWUP_20260915.md`.
DUFS/checkpoint audits COMPLETE in `results/temporal_review_followup_v1/`:
both hinges active100% in4096 validation windows; DiFlo weighted auxiliary
gradient norm is about0.44% of FM gradient at BEST, not reconstructed history.
FM diagnostic counterfactual is similar. Width128 vs512 and cap50k vs200k,
early stopping and unjustified task-scale margins are explicitly recorded.
No new training; no quality conclusion from fold0. Zero history preserves the
current innovation feature, masks and relative position.
Full nonlinear position/length profile control COMPLETE_REVIEWED on all
13,769 answers/145,597 steps;15 source-excluded fits and nested calibration.
Profile-only PB37.1600%/within.753876; mean-detrended39.6992%/.758086;
location-scale-detrended38.5039%/.756870; original innovation39.8314%/.760293.
Mean-detrended minus profile-only: PB+2.5392pp, primary98.75%CI[1.0647,4.0473];
within+.004210, CI[.000666,.007800]. Location/scale within primary CI includes0.
Mean-detrended loses within versus original innovation (-.002206,
descriptive95%CI[-.003418,-.000977]); do not replace the original feature.
Result supports answer-specific information beyond this fitted location/length
mean profile, not removal of every positional interaction or proof of routing.
Six method bundles pass independent scalar PB/per-cell/count and pairwise
within-AUC audits.21 contract tests PASS. See results/temporal_position_control_v1/.
No new gate tuning or routing model launched. Neural large matrix stays paused.

Reviewable execution report: `docs/reviews/temporal_research_execution_2026-09-15.html`
and `.md`. Next decision follows position controls, not the large neural queue.
Earlier proposal (now amended and tested synthetically; full benchmark unrun):
docs/experiments/CCA_CONTEXTUAL_FUSION_PROPOSAL_20260915.md and .html.
CCA past->current-original4 context, source-disjoint local covariance,
shrinkage and IU-moment simplex weights in original score units. Includes
static/position/DSP/refitted-null controls, cost cap, source/nested calibration,
specific mathematical questions and mapping to the remaining research program.
Numeric defaults are proposals pending review, not newly frozen experiments.
Matched-order IU,
multi-target shrinkage, simplex routing/fusion, Network Lasso, low-rank maps,
matched-history negatives and the conditional LOCA/sampling work remain pending.
Preserve original root worktree edits. Do not treat any old RUNNING PID as live.

## Independent remote review — Step 376 [review] (2026-09-15)

Pulled origin/codex/renyi-position-temporal-fusion-v1 through cf01849a7 into
an isolated worktree; review branch codex/temporal-review-20260915. No original
experiment results or main-worktree edits changed. Hebrew review:
docs/reviews/remote_temporal_review_2026-09-15.md; reproducible evidence:
results/remote_temporal_review_20260915/AUDIT.json.

Two confirmed findings: all 47 newly reported metric bundles retain old-gate
pb_cells/suppressed-peak diagnostics despite updated aggregate PB; H1_native
is top15 conditional entropy, not the full-vocabulary entropy described in the
feature-bank protocol. A matching-SHA256 400-answer cache verifies the latter.
Headline CSV/JSON values agree; missing score/detector archives prevent a full
independent replay here. No headline invalidation is established.

Final q15 fusion has essentially the same PRMB-within as single VE0; strongest
single/RBM controls still need the SAME tail15 q=.33 gate. That gate calibrates
midranks on the target cell, so its transductive access must remain explicit.
Recommendation for discussion: short residual temporal convolution as a
context component, preserving Top10 and raw evidence, after matched controls
and reporting repair. No new improvement experiment launched. GraphTV live
state was not checked; the old RUNNING snapshot is not current evidence.

## Latest Renyi follow-up — Step 375 (2026-09-15)

Experiment 3 and its independent cumulative replay are COMPLETE / REVIEW PASS
on 13,769 answers and 145,597 steps. The 2x2x2 feature banks crossed VE1 q15
versus q50, native H1 absent/present and q15 Hinf absent/present, under raw
equal, scale-only equal and answer-z local IU (24 arms). The q15/raw/no-extra
baseline reconstructs the current locator bitwise.

No arm improves both PB and PRMB within. H1 averages -.225 PB points and
-.001053 within across its 12 matched comparisons; Hinf averages -.116 points
and -.001262. Replacing VE1 q15 with q50 averages -.290 PB points and only
+.000130 within, though PRMScore rises in all 12 matched pairs. Scale-only
systematically trades higher PB for lower PRMB; local IU is off the joint
frontier. The label-using union has 554 additional exact error hits, but no
tested label-free fusion captures them uniformly.

The registered minimum-regret q50/scale candidate independently replays
bitwise and gains +.133 PB points while losing .003139 PRMB within (95% CI
[-.004875,-.001398]); it fails the frozen .002 loss margin. Final development
recommendation therefore retains q15 `{H0lim, VE0, VE0.75, VE1}`, natural-unit
per-view Top10 equal localization, and tail15 answer-Top10 q=.33 gating. This
scores PB 37.4749%, PRMB within .753436 and PRMScore .634412; versus the
original start, PB is +1.3075 points and within +.002321, while PRMScore is
-.000392. External/new-model confirmation remains required. Reports:
`results/renyi_locator_feature_bank_v1/REPORT.md` and
`results/renyi_locator_integrated_replay_v1/REPORT.md`.

## Previous Renyi follow-up — Step 374 (2026-09-14)

The four-view fusion input-normalization ablation is COMPLETE / REVIEW PASS on
13,769 answers. With the q15 feature bank, Top10 readout and tail15 Top10 q=.33
gate fixed, 21 arms compared answer-z, scale-only and literal raw inputs across
equal, local/external IU and local shrinkage solvers. Joint L-SML was deferred
as requested; the four-view bank is structurally inadmissible for its full
three-group/minimum-three-members model.

Raw natural units do not improve adaptive local fusion. Local IU falls from PB
37.2406% / PRMB within .745703 under answer-z to 32.8165% / .681698 raw; local
shrinkage variants fall to 30.87%-32.07%. Median second-moment condition number
rises from 65.8 to 23,334.6. External stationary IU is scale-robust but gains
essentially no localization: answer-z 37.1865% / .746279 versus raw 37.2079% /
.746155.

The useful non-z effect is calibration. Equal scale-only is exactly invariant
on PB and within-AUC relative to answer-z, while pooled OOF rises .675153 to
.715708 and PRMScore .589883 to .630093. Raw equal reaches within .751115 /
PRMScore .634805 but loses .230 PB points. No non-z algorithm arm enters the
PB/within noninferiority region of the current per-view-Top10 locator
(37.4749%, .753436, PRMScore .634412). Retain that locator; keep scale-only as
a calibration control and answer-z for IU/shrinkage in Experiment 3. Canonical
report: `results/fusion_input_normalization_ablation_v1/REPORT.md`.

## Latest Renyi follow-up — Step 373 (2026-09-14)

The localization-aware operating-point experiment and immediate cumulative
replay are COMPLETE / REVIEW PASS. With the q15 locator and tail15 Top10 gate
definition fixed, one q was selected across all eight ProcessBench development
cells by official exact-localization macro-F1. The selected q=.33 gives PB
37.4749%, versus 36.1674% for the starting static-locator + entropy-mean q=.3
method: +1.307pp, with family-wise 99% grouped CI [-.484,+3.067]pp. The exact
hundredth is not expected to be stable: q=.31-.35 spans 37.3969%-37.4749%.

The current development-frozen method is q15 `{H0lim, VE0, VE0.75, VE1}`
per-view Top10 natural-unit fusion for localization, plus raw missing-top15-mass
Top10 with q=.33 for ProcessBench gating. Its answer detector is family-macro
F1 .702180 / AUROC .799571 / AUPRC .863489. PRMB remains locator-only:
within .753436, fold AUROC .722708, pooled OOF .722305, PRMScore .634412.
The historical 36.8818% / .792172 row is tail15 **mean** with PB-fixed q=.3,
not the final method. The feature/readout was developed on math; q=.33 was
selected on PB development, so external/new-model confirmation is required.
Canonical report: `results/tail15_localization_q_v1/REPORT.md`; plan audit:
`results/tail15_localization_q_v1/PLAN_AUDIT.md`.

## Latest Renyi follow-up — Step 372 (2026-09-14)

The same-protocol tail15 readout head-to-head is COMPLETE / REVIEW PASS. The
raw missing-top15-mass signal was held identical; only whole-answer Top10 versus
mean changed, and each q was selected on the same 15-cell math panel before PB
target access. Top10 q=.40 beats mean q=.45 on math answer F1 (.632415 vs
.625927), PB answer F1 (.697932 vs .691500), PB AUROC (.799571 vs .792172),
and PB localization (36.6736% vs 36.6064%). The localization delta is only
+.067pp with family-wise 98.333% paired CI [-.984,+1.149]pp and is not
confirmed.

Top10 opens more errors and yields 1,058 exact error localizations versus 1,016
for mean, at the cost of 751 versus 642 clean false alarms. Both fair-transfer
rows remain below the historical PB-developed tail15-mean q=.3 point of
36.8818%, indicating that the next bounded question is the gate operating-point
objective rather than another tail15 readout search. Retain tail15 Top10 as the
readout candidate; next test a localization-cost-aware q rule on development,
then rerun the complete integrated method before any external confirmation.
Canonical report: `results/tail15_readout_headtohead_v1/REPORT.md`.

## Latest Renyi follow-up — Step 371 (2026-09-14)

The leading-simple-gates comparison is COMPLETE / REVIEW PASS. Five distinct
math-leading gates were transferred with their math-selected q to the same
frozen q15 locator across all eight PB cells, with no PB feature or q
calibration. Tail15 missing-mass Top10 at q=.40 is the only candidate that does
not trade away localization: answer family-macro F1 .697932 / AUROC .799571 and
PB 36.6736%, versus entropy-mean q=.3 at .649999 / .742301 and 36.6201%.
The PB delta is +.054pp with family-wise 99% paired CI [-1.632,+1.804]pp, so the
localization lift is not confirmed.

The other transferred gates improve whole-answer detection but reduce PB:
q15 raw4 fusion Top10 .697559 / 35.5691%; entropy Top10 .693567 / 35.5339%;
Hinf Top10 .681939 / 35.0077%; VE1 Top10 .687474 / 34.9234%. Thus answer-gate
macro-F1 alone is not an adequate proxy for exact localization. Promote
tail15-Top10 only as the next distinct simple development candidate, not as a
confirmed replacement. Historical tail15-mean q=.3 remains a PB-selected
diagnostic (36.8818%), not the no-PB-tuning transfer result. Canonical report:
`results/leading_gate_transfer_v1/REPORT.md`.

## Latest Renyi follow-up — Steps 367-370 (2026-09-14)

The gate decision was simplified after the full 15-cell math screen. The
three-feature numerical winner (`VE1 Top10 + q15 raw4 last-quarter + entropy
Top10`, family-macro F1 .641310) remains a valid ablation but is NOT promoted:
its +.775pp math F1 over the best single was judged too small for the added
selection/fusion layer. The later PB q=.40 calibration of that arm is likewise
superseded as a decision candidate and remains diagnostic only.

The replacement two-arm experiment is COMPLETE / REVIEW PASS. On all 18,614
math answers, native H1 entropy Top10 scores family-macro F1 .633271 / AUROC
.791377 / AUPRC .775099 at q=.45; frozen q15 static token fusion followed by
one answer Top10 scores .632755 / .787228 / .759507, also at q=.45. Entropy
Top10 is selected because it is simpler and slightly better on every reported
metric. The token-fusion control reconstructs the exact frozen
`original_static_fusion_before_top10` localization definition with zero maximum
discrepancy after the registered step readout. It is not the later selected
per-view-Top10 locator, because Top10 and fusion do not commute.

The math-frozen entropy Top10 q=.45 gate was transferred to all eight PB cells
without PB method or q selection. Standalone answer detection improves over
the existing entropy-mean q=.3 gate: family-macro F1 .693567 vs .649999 and
AUROC .792620 vs .742301. When used to gate the frozen q15 locator, however, PB
localization falls from 36.6201% to 35.5339% (delta -1.086pp; conservative
98.75% paired CI [-2.946,+.769]pp). It removes 698 old clean false alarms but
loses 338 old exact error localizations. Therefore retain entropy Top10 q=.45
as the simple total-answer detector candidate, but do not replace the existing
mean-entropy q=.3 localization gate. External/new-model confirmation remains
required. Canonical report: `results/simple_gate_choice_v1/REPORT.md`.

## Latest Renyi follow-up — Steps 365-366 (2026-09-14)

Gate Experiment 2 and its requested cumulative integration replay are COMPLETE
/ REVIEW PASS. With the q15 finalist locator and q=.3 threshold rule fixed, 33
uniform candidates (11 token signals x 3 readouts) were evaluated on all 6,800
PB answers. Whole-answer mean missing top-15 mass wins development selection:
PB 36.8818%, detector AUC .792172, versus recalculated entropy mean PB 36.6107%
and AUC .742301. H1 is decision-identical to native entropy; Hinf and raw
`-log p1` are weaker. Gate Top10 readouts lose despite higher separability AUC.

The separate no-reselection integration replay composes every accepted locator
and gate decision. Original static + entropy is 36.1674%; q15 finalist +
entropy 36.6201%; original static + tail15 36.5937%; integrated q15 + tail15
36.8818%. Incremental gate delta is +.262pp and cumulative delta +.714pp, so
the registered point-composition gate passes. All family-wise 98.75% intervals
cross zero; retain tail15 mean as the next gate candidate but do not yet replace
entropy. PRMB remains q15 within .753436 / fold-pooled .722708 / PRMScore
.634412. Next bounded question: test the q=.3 operating point, then rerun the
complete algorithm with any selected threshold. Reports:
`results/gate_feature_readout_selection_v1/REPORT.md` and
`results/integrated_q15_tail15_gate_replay_v1/REPORT.md`.

## Previous Renyi follow-up — Step 364 (2026-09-14)

The post-Experiment-1B finalist replay is COMPLETE / REVIEW PASS. The one
benchmark-uniform deployable specification is q15 `{H0lim, VE0, VE0.75, VE1}`
with per-view Top10 followed by raw equal step fusion, natural units and answer
level retained, and no q50 duplication, supervised simplex or position fit. It
scores PB all-8 36.6201%, PRMB within .753436, fold-pooled .722708 and PRMScore
.634412 on 13,769 answers / 145,597 steps.

Versus the original q15 fusion-before-Top10, PB is +.453pp (family-wise 98.333%
CI [-.181,+1.122]), PRMB within +.002321 [.001322,.003367], and PRMScore
-.000392. Versus the original local shrinkage + position method, PB is +.701pp
[-.462,+1.823], within +.006055 [.003317,.008850], and PRMScore +.047003. The
finalist reconstructs Experiment 1B's selected vector bitwise. The mean-entropy
q=.3 gate remains unchanged and is the next separate experimental question.
Canonical report: `results/selected_q15_finalist_replay_v1/REPORT.md`; frozen
score SHA256: `3bd5c5b95474d75366b97b012b018d168cacafb3ccea178d26170207e220c7e6`.

## Previous Renyi follow-up — Step 363 (2026-09-14)

Experiment 1B, the benchmark-uniform support/weighting study, is COMPLETE /
REVIEW PASS on all 13,769 answers and 145,597 steps. The frozen joint
worst-regret rule selects one method for every cell: q15 raw equal after
per-view Top10 (PB all-8 36.6201%, PRMB within .753436, fold-pooled .722708,
PRMScore .634412). This supersedes the earlier benchmark-specific q15/q50
wording. q50 raw gains only .001407 within while losing .634 PB points; static
q15+q50 raw gains .000312 within and loses .368 PB points.

Natural scale ratios are useful rather than pure nuisance: q15 raw is dominated
by VE0/H0lim and beats global scale equalization by .004409 PRMB within. The
shared supervised simplex discards those low-alpha views and is rejected; its
eight-view fit collapses to q50 VE.75/VE1. Centering again preserves PB/within
ordering but loses .006-.012 fold-pooled AUROC and .0075-.0096 PRMScore. Carry
the q15 raw per-view-Top10 representation, preserve answer-level means, and hold
the locator fixed for the next gate feature/readout experiment. New-model
confirmation remains required.

Frozen OOF scores SHA256:
`2e87a3d11f4ee77594b64775457b1e99e864efb7d5eaca8350f9688b4adc14f4`.
Canonical report: `results/uniform_multiscale_fusion_v1/REPORT.md`; ordered log:
`docs/experiments/RENYI_FUSION_FOLLOWUP_LOG.md`.

## Previous Renyi follow-up — Step 362 (2026-09-14)

Experiment 1, the probability-normalization/mass/preprocessing ablation, is
COMPLETE / REVIEW PASS on all 13,769 answers and 145,597 steps. Proper
escort-varentropy is numerically invariant to raw retained `p` versus
conditional `q` on fixed support (maximum token discrepancy `8.73e-10`), so
top-K renormalization is not the standalone-to-fusion loss. `VE1 q50` gains
.004678 PRMB within AUROC over q15 under the corrected interval while losing
.285 ProcessBench points and passes the frozen promotion rule. Coarse residual
tail mass does not help.

Within-answer centering is the main preprocessing failure: it preserves
within-answer ordering but removes roughly .04-.06 pooled AUROC/PRMScore.
Future calibrated fusion must carry raw, scale-only or exclusion-safe
fold-global preprocessing controls. The score archive was frozen before labels
at SHA256 `402af091e82a10147de659cac22befbd7f8fd3a5c8c3427082935368ab50daef`.
Experiment 2 (gate optimization beginning with Top10 mean) has not started.
Canonical report: `results/probability_normalization_ablation_v1/REPORT.md`;
ordered log: `docs/experiments/RENYI_FUSION_FOLLOWUP_LOG.md`.

## Codex frozen v3 labels/provenance in Git - 2026-09-14

Added JOINED.json/JOINED.npz and RELEASE_V3.json unchanged, as requested.
Joined files match the frozen runner manifest. Release roster (13769 answers),
source groups, token/step counts and folds hash match. Packed step offsets and
label array dimensions checked; original -2 sentinel preserved.
Review: results/frozen_gate_folds_bundle_v1/LABELS_REVIEW.json.
Large feature caches and their token spans remain separate required inputs.

## Codex frozen gate/folds bundle - 2026-09-14

Added the exact FOLDS_V2.json and fusion_fixed_gate_v1/DETECTORS.npz + METRICS.json
to conditional-iu-followups-v1, as requested for second-machine/branch integration.
All three match the pre-existing experiment input SHA256 and sizes. No new folds,
labels or thresholds computed. The other 19 registered inputs remain required.
Review: results/frozen_gate_folds_bundle_v1/REVIEW.json.

## Codex temporal-fusion integration snapshot - 2026-09-14

The conditional-IU branch contains the hierarchical-time, initial two-axis and
whole-answer-position branches as verified ancestors. Three completed AIRCC
report sets and their SHA256 audits are included in this update.
GraphTV255754 remains RUNNING at 00:01 Israel time: 4525/13769 answers scored.
Remaining: outer/inner predictions, score-health review, calibrated metrics,
10000-draw grouped intervals and final result review. No new combined run started.
Integration handoff: docs/research_notes/TEMPORAL_FUSION_INTEGRATION_HANDOFF_2026-09-14.md.
The previous metadata commit f3fa3cc34 is already on GitHub; this update prepares
the fetched result evidence and integration map for the same approved remote.

## Claude Rényi/escort-varentropy evidence merged - 2026-09-14

The complete Stage 2/3/3b implementation, reports and reviewed results from
`claude/varentropy-expansion-fusion-v1` are merged into the isolated combined
worktree. The frozen four-view bank for the next experiment is `{H0lim, VE_0,
VE_0.75, VE_1}`. The observed early/late complementarity is post-hoc and may
motivate the hypothesis only; labels must not select alpha or position weights.

Stage 3b evidence: VE_0 is the strongest answer-local single stream on PRMB
(within 0.7534 / pooled 0.7231 / PRMScore 0.6355); VE_0.75 has the largest PB
point (36.76%, intervals include zero). Neither is promoted. Combining several
Rényi orders with static fusion did not beat the best single order.

The authorized next implementation is a label-free position-varying fusion of
the four frozen views: other-answer positional IU-PCR and answer-local
shrinkage-IU toward an external position-conditioned covariance prior, with
static, shuffled-position and scale-only controls. No full run is authorized
until the new manifest and local smoke/replay checks pass.

## Codex Graph-local IU fetched - 2026-09-13

Job255753 COMPLETE_REVIEWED, all13,769 answers, 11 fetched SHA256 matches,
PB macros rederived. Graph-local PB21.165% / within0.70115 / PRMScore0.57174;
sliding-window35.547% /0.74182 /0.58590; permuted-graph36.268% /0.74945 /0.58857.
Primary graph-versus-window is negative with corrected intervals excluding0.
Graph-local losses versus RBM12 overwhelmingly shift early; no scoring failures.
Do not conflate this graph with the still-running coefficient-GraphTV job255754.
Interpretation and original reports: results/aircc_results_20260913/graph_local/
in worktree .worktrees/conditional-iu-followups-v1. No new model launched.

## Codex AIRCC result fetch - 2026-09-13: position results available

Jobs 255722 (whole-answer position Factor/RBM/IU) and 255752 (conditional
Shrinkage IU position prior) COMPLETE_REVIEWED, full 13,769 answers.
Other-answer position IU: PB35.52%, within AUC0.76573, PRMScore0.61442.
Conditional position IU: PB35.81%, within0.75690, PRMScore0.58991.
Frozen RBM12 Logit: PB36.27%, within0.74520, PRMScore0.62222.
Position benefits within-answer ranking; no overall winner. Conditional position
beats its scale-only control (primary paired interval excludes zero), while both
position IU candidates trail RBM12 in PRMScore. Rank-2 RBM loses through early
peaks despite all selected optimizer convergence flags; do not call this a cap failure.
Graph-local255753 is in bootstrap; GraphTV255754 is still scoring at this snapshot.
Fetched reports, SHA256 audit and interpretation:
.worktrees/conditional-iu-followups-v1/results/aircc_results_20260913/
Relative to the worktree itself, use results/aircc_results_20260913/.
Push SUCCEEDED at f3fa3cc34 after explicit approval; supersedes old blocked text.
No new experiment was launched during this fetch.

## Codex conditional IU - 2026-09-13 - all three full runs active

Branch `codex/conditional-iu-followups-v1`, code/package commit dc12221d2.
Independent science/literature review PASS, 36 mathematical/driver fixtures PASS,
and the complete extracted archive passes isolated Python (-I) driver checks.
Read-only validation of the 22 frozen data/provenance files also PASS.

Current AIRCC jobs: position 255752; graph-local 255753; coefficient GraphTV 255754.
Each requests 64 GiB, one BLAS thread, no GPU. Byte/fixture checks, Linux lifecycle, and all three smoke27 reviews PASS.
All three are in full covariance extraction; no full performance result yet. Original position job
255722 is separate and had already entered full training. No source data changed.
Earlier attempts failed before scientific smoke: first at container memory limit,
then because recursive scripts/localization helpers were omitted from packaging.
Both operational issues were corrected; failed logs and original archives remain.

Second-machine instructions: docs/research_notes/CONDITIONAL_IU_SECOND_MACHINE.md
inside this worktree. Git push was requested, but automatic approval review blocked
it pending explicit destination confirmation. Latest code is still local; do not
assume the GitHub branch already includes these changes. Large caches are not in Git.
Protocol: docs/experiments/CONDITIONAL_IU_FUSION_V1.md.
Detailed review/jobs: results/conditional_iu_preparation_v1/.

## Codex answer-position fusion - 2026-09-13 - IMPLEMENTED; no full results yet

Omri corrected the coordinate to position across the COMPLETE answer and requested
several learning algorithms. Branch codex/answer-position-fusion-v1 (base 6249db384)
implements Gaussian factor, exact Gaussian/Bernoulli H1 RBM, and canonical IU-PCR.
All use the frozen 12 features, token scores, Top10, entropy-q0.3 gate, v3 labels,
v2 groups and nested PRMScore folds. New fits use other answers WITHOUT labels.
Fixed-weight and mean-only controls isolate position-specific coefficient changes;
position shuffling moves assignments, never tokens across annotation boundaries.
The prior within-step model completed smoke only and is preserved, not rejected
by a full benchmark. No new experiment winner or numerical gain is claimed.

Protocol: docs/experiments/ANSWER_POSITION_FUSION_V1.md.
Review/decisions: docs/research_notes/ANSWER_POSITION_IMPLEMENTATION_2026-09-13.md.
Run: scripts/complete_answer_position_fusion.py --source-root C:/Users/omris/TAU/hallucination_detection
Live files: results/answer_position_fusion_v1/PROGRAM_STATE.json and smoke/RUN_STATE.json.
One process/BLAS1, minimum 4GiB available RAM, resumable eight-hour invocations.
Stop and present the complete matched findings before another model family.
# Spectral Hallucination Detection — Session Progress Handoff

## Codex hierarchical time fusion v1 ? 2026-09-13 ? COMPLETE REVIEWED

Full 13,769-answer run completed in 1,792.85 seconds (about 30 minutes, excluding implementation and preflight). All scoring and nested calibration covered the registered population. Independent PB, within-answer AUC and official PRMScore reconstruction PASS; 13 frozen reference rows reproduced; four additional DUFS/shared-variance reference rows re-evaluated against matching contract hashes. No numerical failures or nonconverged selected fits; the one-step answer uses the declared rules. 55 unique training exclusion sets and 41,645 answer/exclusion records were checked. Current-answer RBM12 coefficients never changed.

| Frozen RBM12 readout / time weights | PB all8 % | PRMB within AUC | PRMScore |
|---|---:|---:|---:|
| Original Top10 | 36.271 | 0.74520 | 0.62222 |
| All-token mean | 29.429 | 0.66050 | 0.56348 |
| Best contiguous 10 | 31.653 | 0.70476 | 0.60425 |
| Local time | 28.142 | 0.64238 | 0.57764 |
| Shared time | 29.728 | 0.66991 | 0.57134 |
| Hierarchical time | 29.385 | 0.66454 | 0.57272 |
| Shuffled hierarchical time | 29.190 | 0.65985 | 0.56285 |
| Supervised positive time weights | 30.242 | 0.68101 | 0.59483 |

Primary, 10,000 paired source-group draws, 97.5% CI: shared-local PB +1.586 pp [0.044, 3.115], within-AUC +0.02753 [0.02290, 0.03204], but PRMScore -0.00631 [-0.01013, -0.00251]. Hierarchical-shared PB -0.343 pp [-1.260, 0.557], within-AUC -0.00537 [-0.00740, -0.00330]; PRMScore +0.00138 has an interval including zero. Sharing data helps relative to local covariance fitting on two endpoints, not across every metric. Local adaptation supplies no overall gain.

The hierarchy loses to original Top10 in every PB cell: it gains 281 exact successes and loses 608 (269 early, 339 late; zero gate or computation losses). Top10-to-all-mean alone loses 6.842 PB points before fitting any time weights. This isolates a substantial loss from replacing score-adaptive selection with averaging. Supervised fixed positive time weights do not recover it; this diagnostic is not a supervised ceiling for other architectures.

Average shared mass over successive step quarters: 12.6%, 21.5%, 29.7%, 36.2% (uniform: 25% each). Hierarchy: 12.7%, 21.9%, 29.4%, 36.1%. The median answer-level L1 change from shared weights is 0.258, despite similar aggregate profiles. Shuffling yields nearly uniform quarters. Descriptive original-order versus shuffled hierarchy within-AUC +0.00470 [0.00238, 0.00700], PB +0.195 pp [-0.972, 1.348]: evidence for modest within-answer ordering information, not a PB gain.

Boundary diagnostics are not promoted: Top10 without the first token gives PB 36.360% (difference not clear), within-AUC 0.74387 (slightly worse), PRMScore 0.62747 (better). Removing the first token from hierarchy gives 29.546% PB; it does not close the gap. Regions do not create observations: 22,333 of 145,597 steps have fewer than 16 tokens; 91 have one token. Median step length is 31 tokens; median answer has eight steps and alpha=0.304.

Decision: retain original RBM12 Logit + Top10 as this temporal experiment's reference. Do not adopt the tested positive fixed-position averaging hierarchy. More data stabilizes temporal weights but cannot by itself recover information discarded by the readout. A next family should preserve score-adaptive high-risk token selection while isolating added temporal/context information. No tensor, conditional RBM, convolution or new feature experiment was started; discussion is required before the next family. Findings remain development evidence, not untouched confirmation. Historical low-correlation RBM6 and Varentropy50 remain explicit competing references, not replaced by this run.

Artifacts: results/rbm_hierarchical_time_v1/{REPORT.md,COMPARISON.csv,PER_CELL.csv,METRICS.json,CONTRASTS.json,CONTROL_CONTRASTS.json,PRMSCORE_CONTRASTS.json,WEIGHTS.csv,WEIGHT_SUMMARY.json,ANSWER_DIAGNOSTICS.csv,RESULT_REVIEW.json,POSTFIT_REVIEW.json}. Protocol: docs/experiments/RBM_HIERARCHICAL_TIME_V1.md. Preflight archives preserve the JSON-reader/audit repairs; PROFILE_REUSE_REVIEW.json verifies the unchanged extraction and inputs. No original source, Claude run, or historical result was modified. No HTML was produced.


## Claude Stage-1 interim handoff - 2026-09-12 evening (Stage 1 OPEN)

Omri's staged mandate is being executed in parallel tracks; account and tables:
.worktrees/rbm-literature-completion-v1/docs/research_notes/RBM_PROGRAM_STAGE1_ACCOUNT_2026-09-12.md
- Full window-sampling run RESUMED via the unchanged supervisor (scripts/complete_research_consolidation_v1.py,
  logs results/research_consolidation_v1/supervisor_v4_stage1_20260912.*); ~13 s/record; projection
  ~1.5 days (estimate). Stage 1 is complete only when RUN_STATE reaches COMPLETE_REVIEWED_FULL_SAMPLING.
- RBM stability COMPLETE review PASS; capacity interpreted (iteration cap, saturation); depth amended
  (declared failures + logit variants, separate driver) running; commit e6fafcf96 on codex/rbm-literature-completion-v1.
- Stage 2 cross-rank fusion: protocol frozen, reviewed, full run launched (3 workers) in
  .worktrees/varentropy-expansion-fusion-v1 (branch claude/varentropy-expansion-fusion-v1, uncommitted).
- Stage 3 Renyi: prototype + DRAFT only; design waits for Stage-2 review.
Do not restart the sampling supervisor while its driver (pid in RUN_STATE) is alive; checkpoints resume.


## Claude handoff: verified final DUFS and stopped queue - 2026-09-12

DUFS COMPLETE13769/13769, metric and state reviews PASS; actual completion
2026-09-12 04:46:51 Israel. V/C/T suites also COMPLETE13769 and full review PASS.
DUFS trained PB36.1235%, within .740974, PRMScore .626847. Against all12 trained,
within +.002273 CI97.5[.000990,.003543], PB -.2516pp CI[-.8346,.3144]. No clear
advantage over low-correlation6 and no automatic DUFS integration.

Queue exited05:49:35; no Python processes found in elevated process audit at
handoff. Stability smoke27 PASS, separate review/full pending. Depth smoke
FAIL:6/27 answers,14 model records, fewer than three varying hidden views.
Do not blindly restart queue or silently change the failed depth protocol.

Capacity interpretation warning: exact4 nonconverged13769/13769 bank12 and
13768/13769 bank6. Its large losses do not isolate capacity from optimization.
CD has a fixed epoch budget, not a convergence PASS; H1 posterior has a local
AUC gain but worse PRMScore. Temporal actual order loses within-answer AUC
against shuffle in both retained banks; no demonstrated chronological gain.

Full handoff (in source root, not only this worktree):
../../docs/research_notes/CODEX_TO_CLAUDE_HANDOFF_2026-09-12.md
Fresh machine snapshot: results/rbm_literature_completion_v1/HANDOFF_STATUS_20260912.json.
Readable comparison now106 rows including reviewed DUFS, capacity and temporal.
The handoff turn refreshed inventories/docs only; no models or jobs restarted.
The full authorized program remains INCOMPLETE. Generated outputs and these
latest documents include uncommitted files; preserve them when transferring.


## One readable RBM comparison across experiments - 2026-09-12

The current-suite COMPARISON.csv alone omitted prior completed RBM families.
RBM_FUSION_COMPARISON.csv now joins62 declared historical rows/controls and
the8 completed variance candidates, with70 unique display names, exact source
paths/hashes, fitting access, readout and explicit PB percentage units. All
source metric files match their indexed hashes; no scores were recomputed.
The raw317-row prior ledger remains available, including repeated aliases.
Near-max and other-answer/supervised diagnostics have explicit separate panels.
This is not a ranking or a claim that all published baselines were refitted.

The summary builder refreshes this table after each reviewed full suite and
includes DUFS only when its full run, pipeline and both reviews are complete.
Current live scoring continues unchanged: queue29404, capacity17352, DUFS18724/
20464. No new model, fit, scientific threshold or active process was changed.

## H1 numerical provenance check - 2026-09-12

The27 frozen capacity smoke answers were checked for H1 provenance (both
banks,54 fits). Current exact-H1 and the historical fit_rbm on the same
C-contiguous input produce bit-identical parameters. Historical fit_rbm on
Fortran-contiguous copies reproduces the original saved parameters exactly.
Thus original-versus-current tiny refit differences come from input memory
layout and floating-point reduction, not a different objective or optimizer.
Maximum smoke Logit step difference .001015; posterior difference2.96e-11;
NLL difference1.19e-10; zero changed peak choices in these mechanics cases.
This is NOT a performance sample. The full capacity contrasts against frozen
references remain required. Existing exact/CD comparisons use the same C
inputs/initialization. No scientific code or active scorer was changed.
Evidence: capacity/H1_NUMERICAL_PROVENANCE.{json,csv} in the program results.

## Variance forensics complete; dependency queue adopted - 2026-09-12

Full saved-model density/covariance diagnostics cover55076 model-answer fits.
For bank12, mean data-NLL gain versus the original RBM is4.574 for separate
variance versus1.371 shared, yet mean off-diagonal covariance relative error
is essentially unchanged (.7199 versus .7189). Better density fit did not
improve localization. These are descriptive fit diagnostics, not new metrics.

All1217 changed PB answers in the primary bank12 contrast were decomposed.
Among934 losses,922 have the fitted linear component prefer the truth over
the chosen wrong step, with the quadratic component reversing that pairwise
preference. This does NOT claim that removing quadratic terms would recover
922 successes: another wrong step could still win, and refitting would change
the coefficients. All selected Top10 sets remain high on the mean6 anchor;
the loss is not explained by selecting its low tail. Forensic JSON/CSV and
coefficient tables are under the variance result directory; no model refit.

Execution-only amendment: tested queue controller29404 now adopts unchanged
capacity scorer17352; old waiting controller2748 retired. DUFS18724 untouched.
Four scheduler/process-identity tests PASS. One suite runs until DUFS finishes
both reviews; then up to two independent suites, two workers each. Stability
and depth require completed reviewed capacity; temporal is independent.
QUEUE_STATE.json tracks native process handles, creation identity and failures.
No scientific definition, seed, data, threshold or active scorer was changed.

## Variance COMPLETE REVIEWED; capacity running - 2026-09-12

Full variance review PASS:13769 answers,110152 step vectors,21 metric bundles;
all frozen scientific output hashes unchanged by reviewer-unit correction.
Bank6 retained Posterior: original36.2017%, shared35.8081%, separate36.0834%;
within .735982/.732299/.737466. Separate versus shared improves within AUC,
but is not clearly better than the original RBM. Bank12 retained Logit:
original36.2712%, shared36.1817%, separate21.0920%; within .745204/.747256/.698596.
Separate versus shared PB -15.0897pp,97.5%CI[-17.4746,-12.7660];283 gained,
934 lost,925 losses early and9 late,none due to gate. No overall promotion.
Bank12 shared Posterior36.8106% is descriptive; its PB interval versus the
original Posterior includes zero. All methods have full coverage;26 bank6
and1 bank12 separate-variance fits hit optimizer nonconvergence, retained
as finite improving fits and explicitly recorded. No silent fallback.

Program PID2748 advanced to capacity; scoring child17352 observed32/13769.
Corrected model-mechanism analysis is running separately without new fits.
Another55 SHA256-identical inactive shrinkage-cache copies were removed,
8.1393GiB. Combined cleanup159 files,19.2099GiB logical bytes;27.57GiB free
at last check. Audits: source scratch/cleanup_20260912/. All root originals,
unique files, code and result artifacts preserved.

## Literature program checkpoint - 2026-09-12

Variance scoring completed all13769 answers, with full metrics and10000 paired
source-group bootstrap draws. Full review stopped at a reviewer-only PB units
error (percent versus the stored fraction). The original reviewer, scientific
code, checkpoint, predictions and METRICS are preserved. Separate reviewer v2
corrects the units; REVIEW_UNIT_AMENDMENT.json binds unchanged result hashes.
Do not promote the variance rows until RESULT_REVIEW is PASS.

The earlier literature diagnostic correction is COMPLETE REVIEWED: both banks,
all13769 answers, group bootstrap, correct retained readouts. Serial residual
dependence remains, but this is not evidence of benchmark gain from temporal
fusion. Results: results/rbm_literature_completion_v1/diagnostic_correction/.

PRIOR_METHODS.csv and PRIOR_EXPERIMENTS.json index317 prior rows (including
repeated controls), with scope and review status. Unlabeled answer-local RBM,
B3, higher moments, shrinkage, diagonal variance and position correction all
belong in comparisons. Native DEEM on vocabulary probabilities was stopped
for input semantics, not shown to fail in a full experiment. The old direct
probability B3 lane still has no full result; six-moment B3 is complete.

Additional stability/depth end-to-end worker tests PASS (2 tests); core tests
PASS (11). Capacity and temporal27-answer smokes each passed saved-state replay.
These are implementation checks only. Inspect live program/DUFS handles for
current status; neither the full remaining suites nor DUFS is declared done.

## Authorized literature completion - 2026-09-12, IN PROGRESS

Omri explicitly requested completion of DUFS and the remaining variance,
multi-unit/CD/depth and token-temporal fusion plans. This supersedes prior
stop-after-discussion notes below. Dedicated code-only worktree/branch
codex/rbm-literature-completion-v1, base de237a3622. No copied dataset cache.
Plan: docs/experiments/RBM_LITERATURE_COMPLETION_V1.md. Both banks6/12,
unchanged full13769 contract, Top10/argmax and entropy gate; no near-max adoption.
Ten algebra/sampling tests PASS. Variance smoke27 answers and216 step-vector
replays PASS, no method failures. These checks are feasibility, not results.

Storage:104 SHA256-identical failed-checkout copies removed,11.0705GiB logical
bytes. Root originals and unique artifacts preserved. Audit in source
scratch/cleanup_20260912/verified_duplicates.jsonl. DUFS original pipeline
resumed as PID18724 after SQLite integrity,9848 contiguous rows and manifest
identity checks; actual scoring progressed beyond10284. Check live handle and
current RUN_STATE; this number is not a completion claim.

Remaining: full reviewed V/C/S/D/T suites; complete and review DUFS; correct the
last literature diagnostic's bootstrap/source-readout attribution; consolidate
all relevant RBM/B3/IU/Varentropy comparisons. Program artifacts live in
results/rbm_literature_completion_v1. No general winner or untouched test yet.


## First-error objective COMPLETE - 2026-09-11

User authorized the next bounded test after the matched coefficient study.
Branch codex/rbm-first-error-objective-v1, base f9d984266. The same bank12
answer-local RBM correction, token Logit/Top10 readout, gate, folds and
optimizer were retained. Only the PB training objective changed: previous
step BCE versus categorical first-error loss over all steps in erroneous
training answers. Clean answers had no first-error target and contributed
zero location loss. PRMB scores were copied unchanged as an integrity control.

All6,800 PB answers and40 group-disjoint fits completed and independently
replayed. First-error PB35.6451% versus step BCE37.2042%; primary delta
-1.5591pp,97.5%CI[-3.1739,+.0887]pp. All eight PB cells declined;481
gated successes gained,556 lost,412 losses late. The PRMB row is identical
by construction and is not new evidence. Result review PASS; no fit failures.

Decision: close this full-answer first-error objective as negative under the
frozen Top10 contract. Keep step BCE as the supervised correction reference.
This does not close every onset/local objective; it closes this categorical
competition over all steps. No automatic next experiment. See
results/rbm_first_error_objective_v1/REPORT.md and DECISION.json.

## Matched RBM supervision COMPLETE - 2026-09-11

User requested two alternatives differing only in coefficient learning after
the step-mean diagnostic changed too many factors. Branch
codex/rbm-supervision-matched-v1, base f7203a8a6, freeze afb852591.
Same answer-specific frozen bank12 RBM, same13-parameter correction, same
training source groups, initialization, penalty and optimizer. Only exact
unlabeled density loss versus supervised STEP BCE differs. Original token
Logit/Top10/argmax and entropy gate unchanged; no step-feature averaging.
Both corrections use other training answers; neither updated arm is purely
answer-only. This tests updates above an identical RBM, not training entirely
from scratch or a supervised per-answer oracle. No held-out-label fitting.

All13769 answers,90 fits, all optimizers report convergence. All45 unlabeled
updates remain exactly zero (max gradient4.76e-7), reproducing original metrics.
Supervised: PB37.2042 vs36.2712 percent; within AUC .747301 vs .745204.
Primary paired97.5%CI: PB+.9330pp[-.2162,2.0836]pp; within+.002096
[-.000251,.004439]. Both include zero. Seven PB cells rise,one falls.
Mean-fold PRMB AUC falls .706205->.689485; PRMScore .622215->.599189.
318 gated successes gained,271 lost;193 losses late,78 early. No winner.

Independent90-model/loss replay, three-row arithmetic metric review, all
13769 zero-update peak checks and10000-draw contrast review PASS. Full coverage.
Held-out balanced BCE improves in all45 folds but remains worse than constant
probability .5; do not call the scores calibrated correctness probabilities.
The largest supervised gradient is8.70e-4; optimizer success is not a global
optimum certificate for the piecewise Top10 objective.

Next proposed, NOT launched: keep the identical correction/Top10 family and
isolate PB first-error listwise loss versus step BCE. Late losses motivate
this objective-alignment question, not a new graph/capacity sweep and not
proof of its answer. No automatic experiment follows. Preserve PRMB as its
own task. Previous step-mean outputs are preserved, not silently replaced.
Results: results/rbm_supervision_matched_v1/REPORT.md,COMPARISON.csv,
PB_CELLS.csv,COEFFICIENTS.csv,DECISION.json. No HTML; DUFS untouched.

## Supervised position diagnostic COMPLETE - 2026-09-11

User authorized the three-way supervised diagnostic. Isolated branch
codex/rbm-supervised-position-diagnostic-v1, base ce008fb1d, freeze90e56eca8.
All13769 answers,135 converged fits,40 configurations including35 unchanged
references. Saved bank12 STEP MEANS, per-cell source-group outer folds;
PB first-error listwise objective, PRMB class-balanced step BCE. No token-label
invention. This is supervised diagnostic access, not the answer-only method.

PB static32.6198 -> +position-prior35.3807 -> conditional35.9587 percent.
PRMB within .715057 -> .724419 -> .731300. Primary conditional-minus-prior:
PB+.5779pp,97.5%CI[-.2671,1.4332]pp; within+.006881,[.003997,.009723].
PRMScore .585182/.600745/.596820.135 objective/gradient/model replays and
40-method independent arithmetic PASS;10000 paired-group draws reviewed.

Interpretation: joint conditional signal exists for PRMB in this fixed
step-mean diagnostic. Most PB gain was the position prior. Original RBM12
Logit/Top10 remains36.2712%/.745204 within/.622215 PRMScore. The input
aggregation differs: this diagnostic is not a ceiling for RBM or a test of
supervision alone. Do not call it a new answer-only winner. No q/ridge search.
Supervised pooled OOF AUC omitted; report mean fold AUC. q.8 calibration uses
training scores from the SAME held-out-fold model, without held-out groups.

Next proposed (NOT launched): bridge the useful conditional interaction to
the original token Top10 representation before a new unlabeled model. Token
adjacency across step boundaries is distinct from two-half position context;
real versus shuffled adjacency is a future diagnostic, not the old shuffled
half-assignment control. Separate diagonal variances per latent state remain
untested; shared diagonal variance was tested. DUFS existing feature-selection
run remains active and unchanged, not merely backlog. Last observed7484/13769;
refresh live state before quoting progress. No token selection claim for DUFS.

Results: results/rbm_supervised_position_diagnostic_v1/REPORT.md,
COMPARISON.csv,PB_CELLS.csv,METRICS.json,NEXT_STEPS.json. Stop for discussion;
no new variance/temporal/CD/capacity model, no HTML or deletion.

## RBM position fusion COMPLETE - 2026-09-11

Branch codex/rbm-position-fusion-v1, base a9cda144e; corrected scoring freeze
258393249. One bank12 chronological weight-correction candidate plus shared
and shuffled controls. Exact answer-local conditional likelihood, same gate,
top10/argmax, no first_near_max.35 total configurations,13769 answers.
Candidate PB35.5837 vs fixed36.2712; within AUC .742490 vs .745204.
Paired97.5%CI: PB delta[-1.2541,-.1256]pp; within[-.003756,-.001684].
Shared update retains baseline task metrics. Shuffled PB35.6343/.743963 within.
65 PB successes gained,104 lost. All models converge, all outputs valid;
median relative correction4.747%. Better density fit did not improve the task.

Decision: do not adopt this correction. Preserve fixed RBM and IU/Varentropy
references; no RBM near-max tuning. Prior marginal reliability reversal still
exists, but joint predictive value and label-free learnability remain unproved.
Next proposed diagnostic (NOT launched): static vs position-dependent joint
predictive value, potentially using labels under source-group-disjoint folds
as a clearly separated diagnostic, never as the claimed answer-only method.
No automatic lambda sweep, CD, new moments, units or layers.

A guard exposed3 PRMB source records sharing boundary tokens; first run stopped
at8112 committed rows, preserved in results/rbm_position_fusion_v1. Corrected
run restarted from zero with latest-step context ownership and unchanged spans.
All8112 prior rows replay bit-exactly. Full metric35-arm and41307-state audits
PASS; no data/label/gate changes. The one single-step answer uses explicit d=0.
Results: results/rbm_position_fusion_v1_overlap_fix/REPORT.md,COMPARISON.csv,
PB_CELLS.csv,NEXT_STEPS.json. Full development evidence, no untouched test.
No HTML; prior worktrees and DUFS unchanged.


## RBM logit/readout isolation COMPLETE - 2026-09-11

Dedicated codex/rbm-logit-readout-v1 from7575cb237; code freeze5d18172de.
All13769 answers,32 configurations, no refit. All24 original references replay;
32-arm independent arithmetic metric review PASS;432 scalar replay checks,
110152 near-max vector checks and80000 bootstrap interaction checks PASS.
Primary logit/near versus posterior/max: RBM6 PB36.2017->34.7246,
delta-1.4771pp (97.5%CI[-2.7656,-.2263]); RBM12 36.3750->36.3998,
delta+.0247pp[-1.2677,1.3213]. Bank12 within-answer AUC .738702->.748781
(CI for delta[.007569,.012683]), but PRMScore .629276->.624156.
No PB advantage over Var15/IU or Varentropy was established.

Reasoning: logits remove much of the damaging posterior/near interaction;
425/732 and577/882 prior lost successes recover. Training helps strongly under
matched logit readouts, so do not infer that learned weights are useless.
The initial posterior remains a strong control. Near-max is not a universal
RBM default; bank6 retains its original posterior/max reference. Next proposal
is one answer-local position-conditioned bank12 fusion, motivated jointly by
this interface result and the prior matched reliability reversal. Define its
objective and shared-weight restraint before implementation; retain static IU
and Varentropy controls. CD/capacity/variance extensions remain deferred.
See results/rbm_logit_readout_v1/REPORT.md,COMPARISON.csv,PB_CELLS.csv,
NEXT_STEPS.json. No next model started; DUFS and prior results unchanged.
Full cached development, not untouched confirmation. No HTML.


## RBM data diagnostics COMPLETE - 2026-09-11

Dedicated codex/rbm-data-diagnostics-v1 from44f9bced8; scoring freeze13e75a8fd.
All13769 answers, both banks6/12, no refit. Full metric review24 arms PASS;
diagnostic review27538 bank-answer records PASS;8860 direct-pair reliability
checks PASS. Four synthetic draws per bank-answer;10000 source-group bootstrap.
PB old -> near: RBM6 36.2017 ->31.0950; RBM12 36.3750 ->27.5159.
Var15/IU 35.3498 ->36.6546; Var50 35.6755 ->36.4366. Gate unchanged.
Do not adopt near-max universally: every lost RBM success shifted earlier.
Readout interface is next; matched feature-reliability reversal supports later
conditional fusion. Class variance and residual/serial mismatch do not prove
task gain from more units/CD. See results/rbm_data_diagnostics_v1/REPORT.md,
NEXT_STEPS.json and EVIDENCE.json. Stop before another model experiment.
DUFS unchanged. No HTML. Earlier RUNNING entries below are historical.


## RBM data diagnostics in progress - 2026-09-11

Dedicated branch codex/rbm-data-diagnostics-v1 from 44f9bced8; protocol/code
frozen in 13e75a8fd. Full readout comparison (24 arms) and independent metrics
review PASS. Full saved-state data scan continues; no RBM refit or DUFS change.
Important: first_near_max lowers PB for RBM6 36.2017 -> 31.0950 and RBM12
36.3750 -> 27.5159, while Varentropy15/IU rises 35.3498 -> 36.6546.
Do not treat Claude readout adoption as automatic RBM improvement.
Current outputs: results/rbm_data_diagnostics_v1; see RUN_STATE and live PID.
Data-diagnostic conclusions pending full scan and review. No HTML.


## Reference audit / preferred readout - 2026-09-11

first_near_max is Omri's preferred follow-up; preserve Claude's full score
transformation, not just argmax. It changes PRMB rankings as well. Threshold
was development-selected. RBMpaper source comparison and12-case numeric
Gaussian audit PASS; no MATLAB execution or training-equivalence claim.
Our continuous one-hidden exact-likelihood RBM differs from their binary
CD-trained stack. See HISTORY and results/rbm_reference_bridge_v1/REVIEW.json.
DUFS frozen run unchanged; no new performance experiment launched.



## User research preference - first_near_max (2026-09-11)

Omri explicitly asks to retain first_near_max as a preferred direction for
integration after the current frozen DUFS run. Claude's Varentropy50 result:
PB36.4366%, within-answer AUC.743826, PRMScore.632903, versus original
35.6755%/.742465/.632777. This is a promising development result, not
confirmation:0.25 SD was selected after inspecting labels on this population.
Keep the exact rule; do not tune another threshold. Before a new experiment,
check the actual saved-score transformation (not only the argmax description).
Compare the same readout rule across fusion candidates and preserve the
original Top10/common-gate controls. Untouched confirmation remains required.
Do not change or stop the frozen DUFS run. Length-stratified analysis and
length/random rows in the concise table remain reporting follow-ups.



## DUFS moment selection launch - 2026-09-11 10:25 local

Source freeze a69bd2445. Pipeline PID23524, four below-normal workers.
Full13769 scoring launched after synthetic and27-real-answer preflight.
No quality results yet. Read results/dufs_moment_selection_v1/RUN_STATE.json,
PIPELINE_STATE.json and RUN.log and verify process liveness for current status.
Pipeline automatically follows scoring with10000-draw paired evaluation,
metric arithmetic review, saved-state review and concise REPORT.md.
No new HTML or further experiment. READY entry below is pre-launch history.



## DUFS moment selection [Codex] - READY 2026-09-11

User authorized feature selection before RBM. Dedicated branch
codex/dufs-moment-selection-v1 from6e8f46100. Fixed order6 compact bank;
all12, original6, adapted DUFS-select6 and squared-Pearson-select6, each
trained RBM versus its untrained initialization. Same original6 risk anchor,
external entropy q0.3 gate and Top10 token readout; full13769 benchmark.
Why: learned diagonal variance improved covariance fit without improving
localization. Test selection quality separately from RBM training, not more
moment orders or new graph penalties. DUFS historical recipe:120 epochs,
seeds0/1/2, hard top6; no labels or hyperparameter selection. Four workers.
Four synthetic tests PASS;27 real mechanics answers PASS,108 exact saved
reference replays, no failures. No subset performance conclusions.
Scoring is expected to take hours. Check RUN_STATE and process liveness;
READY is not a running or completion claim. Resumable pipeline proceeds to
full10000-group bootstrap, separate arithmetic/model-state reviews, and
concise result notes. No subsequent experiment or HTML is launched.
Protocol: docs/experiments/DUFS_MOMENT_SELECTION_V1.md.
Outputs: results/dufs_moment_selection_v1. Existing worktrees untouched.


## RBM conditional diagonal variance [Codex] - COMPLETE 2026-09-11

Full13769 answers/four arms; scoring271.16s,4 workers. Source freeze57f0f7edf,
branch codex/rbm-diagonal-variance-v1.28 metric/reference bundles arithmetic
reviewPASS;55076 model states and108 independent Gaussian-mixture/token/step
replays across27 real answers PASS (maxerror3.33e-16). No failed/collapsed
outputs.25 diagonal fits nonconverged at100 extra iterations, retained/flagged;
all continued fixed-D fits converge. No floor hits; minimumD.09064>.05.

Original and continued fixed-D RBM: PB36.2017%,within.735982,PRMScore.630749.
Learned-D: PB35.8069%,within.733045,PRMScore.625301. Primary diagonal-minus-
continued PB-.3947pp,97.5%CI[-1.0278,+.2255]; within-.002937,
CI[-.004973,-.000890].84 PB gains,105 losses. No PB benefit; ranking regresses.
Both covariance and variance fit improve on every answer, but this does not
improve task accuracy. Median varianceRMSE.44161->.10068; covariance relative
error.60196->.53832; offdiagonalerror.68665->.68704 (essentially unchanged).
Conditionalvariance median.7310. New bounds/penalty tested as one frozen recipe,
not optimized hyperparameters; no automatic follow-up. Keep original references.
This clarifies that generative moment-fit gains do not prove useful fusion.

Artifacts results/rbm_diagonal_variance_v1/{REPORT.md,SUMMARY.csv,METRICS.json,
RESULT_REVIEW.json,STATE_REVIEW.json,ASSUMPTIONS_SUMMARY.json}. Full no-label
assumption audit in results/rbm_covariance_assumptions_v1/AUDIT.json.

Paper requested during execution found in HISTORY Step141: Shaham et al.,
A Deep Learning Approach to Unsupervised Ensemble Learning, ICML2016,
https://proceedings.mlr.press/v48/shaham16.html. Lemma4.1 (PDFpage3) concerns
binary Dawid-Skene and one-hidden-unit binary RBM. Historical phrase 'our
L-SML IS already an RBM' is too broad: model-distribution equivalence is not
optimizer equivalence and does not establish the continuous Gaussian case.
No old entries edited; clarification preserved here and in experiment protocol.


## Higher compact moments through degree6 [Codex] - COMPLETE 2026-09-11

Full13769 rows,16 arms,zero failures/collapsed outputs. Scoring493.95s on4
workers. All37 method/reference metric bundles pass separate arithmetic
review. Saved-state auditPASS on220304 model records;432 model/step replays
on27 real answers pass. Original4 degree3 arms replay exactly. Source freeze
e17ada312, branch codex/higher-moment-fusion-v1; parent9e3d452ab.

Order: RBM PB / within / PRMScore; IU PB / within / PRMScore
3:36.2017%/.735982/.630749;22.3788%/.708585/.581905
4:36.1734%/.734940/.628874;21.6282%/.704043/.583652
5:36.1553%/.734705/.628851;20.7651%/.698812/.579859
6:36.3750%/.738702/.629276;20.2046%/.690603/.570694
Primary6-vs3 RBM PB+.1734pp,97.5%CI[-.6183,+1.0193]; within+.002720,
CI[-.000118,+.005683]: inconclusive. IU PB-2.1742pp[-3.1026,-1.3234],
within-.017981[-.020235,-.015709]: regression. Degree6 initialRBM36.2946%,
within.746325,PRMScore.605313; trainedwithin-.007623,exploratory95%CI
[-.009755,-.005421]. No learned-fusion win. Allhigher-orderRBM fits converge;
one finite degree3 nonconverged fit retained. Keep degree3 compact reference,
no new optimum, no further sweep launched. Both distribution and selected
powers expanded together, so no separate family attribution. Sameanswerfit,
original6mean orientation, gate/readout/v3 contract preserved; no global24 run.
Files: results/higher_moment_fusion_v1/{REPORT.md,SUMMARY.csv,METRICS.json,
RESULT_REVIEW.json,STATE_REVIEW.json}. No new HTML. Original file line endings
preserved in final documentation; no historical content rewritten.


## Step345 [Codex RBM shrinkage] completion - 2026-09-11

COMPLETE: all13769 answers/145597 steps. Frozen124ead81d, two below-normal
workers,408.2s scoring. All24 method/reference metric bundles replay and
separate arithmetic review PASS. State audit41307 model records PASS.
No fit failures/collapse. Original RBM13768 converged/1 flagged finite;
regularized RBM13769 converged. Fixed lambda0.1 shrinks raw weight distance
on every answer: median3.10452->1.18637. Median iterations25->16. This is
weight restraint, not evidence of temporal stability or hallucination semantics.

PB / within-answer PRMB / PRMScore:
- Original RBM:36.2017% / .735982 / .630749.
- Shrinkage RBM:35.8711% / .738630 / .629544.
- Initial RBM:35.9662% / .744068 / .613085.
Primary shrinkage-minus-original: PB-.3305pp,97.5%CI[-.8852,+.1966];
within+.0026483[+.0007774,+.0045989]. Gains58 PB successes, loses65;
only2/8 PB cell points improve. Gate clean accuracy identical. Against initial,
within-.0054373,95%CI[-.0070236,-.0038670], PB difference uncertain.
Varentropy50:35.6755%/.742465/.632777; Var15raw35.9610%/.737786/.625781.
No overall winner and no clear added PB value from this regularizer. Retain
as negative/mixed development evidence; no lambda sweep, DUFS or new graph
was launched. Existing B3 and other worktrees untouched. Return to discussion
before another candidate. Full tables/errors/weights in results/rbm_weight_shrinkage_v1;
REPORT.md, METRICS.json, SUMMARY.csv, RESULT_REVIEW.json and STATE_REVIEW.json.


## RBM shrinkage execution - 2026-09-11

Frozen124ead81d. Full13769 run launched02:46:19 local, PID20988, two
below-normal workers. First368 rows scored; no failures reported. Automatic
bootstrap10000 and arithmetic review follow scoring. Read live result state
rather than treating this launch note as completion. Other runs unchanged.


## Step345 [Codex RBM shrinkage] - fixed regularization follow-up (2026-09-11)

User requested a separate agent to execute the proposed bounded weight-shrinkage
experiment. Parent295011d4, branch codex/rbm-weight-shrinkage-v1. Prior m3/powers
full13769 run and arithmetic review COMPLETE/PASS verified. Preserve original
six moment columns, original mean6 anchor and shared gate/top10 readout. Test
ONLY lambda0.1 toward w0=2/P against original trained and initial RBM; coefficient
fixed as engineering choice, no grid. Reuse entropy/Varentropy and matched Claude
length/random controls. Four mechanism tests PASS;27-answer full-budget mechanical preflight PASS
with zero failures and exact original RBM step replay. Full run pending freeze. Full protocol docs/experiments/RBM_WEIGHT_SHRINKAGE_V1.md.
No HTML; checkpoint/results results/rbm_weight_shrinkage_v1. Other runs untouched.



## Step344 [Codex RBM] - m3 ablation and raw rank powers (2026-09-11)

User explicitly authorized (1) isolate distribution m3 in compact RBM, and
(2) test raw degree1/2/3 rank powers with RBM, previously tested only mean/IU.
Separate codex/rbm-m3-powers-v1 from47eb162f; frozen code/protocol943ad2aa.
Primary localization only, same13769 rows, frozen benchmark/gate/readout.
Remove ONLY m3: H,V,a,a^2,a^3 retained. Both compared RBMs use common mean5
direction; preserve original6/mean6 output as exact per-answer bridge.
Raw-power48 RBM uses frozen T x48 representation and old entropy orientation,
plus initial RBM control. Reuse all prior power/moment/Varentropy references.
No higher-order sweep or new global24 fitting in this experiment.

Three tests PASS: column semantics, original scorer and parameter replay,
changing removed m3 cannot affect reduced score. All27 real-answer preflight
rows produce finite scores and exact original6 step replay. No collapsed
outputs; one reduced5 fit reports nonconvergence and remains explicitly
flagged (not silently retried/removed). Scoring22.3s, no quality ranking.
Full run uses2 CPU workers, WAL checkpoints, Windows JSON replacement retry,
10000 grouped draws, two primary97.5% contrasts. Separate metric verification
automatically follows. Check results/rbm_m3_powers_v1/LAUNCH.json and live
RUN_STATE.json for actual status. No HTML. Existing B3 localization untouched;
global24 experiment has separately reached COMPLETE/PASS.

## Moment RBM staged resume authorized (2026-09-11)

Omri explicitly approved interrupting the combined run and switching to fast-first.
At execution PID27400 had already exited: OperationalError database is locked
at commit, after336 answers. The live checkpoint roundtrip audit held a read
transaction during decoding and likely caused the writer timeout. This was
an infrastructure failure, not an algorithm failure. Original SQLite integrity
check returns ok,336 committed rows retained; original FAILED log is preserved.
Staged checkpoint connections now use WAL and60s busy timeout. New regression
test keeps a reader open while writer commits; passing. No frozen core, labels,
gate/readout or optimizer settings changed. Stage runner will reuse all336
answers for both fast and B3 lanes. New results under moment_rbm_fusion_v1_staged.
Earlier pending-approval note is superseded.

## Moment RBM scheduling update (2026-09-11)

User requests full-population fast-method results first, then B3 separately.
Prepared scripts/run_moment_rbm_staged.py: preserve frozen core/evaluator,
project original committed checkpoints, run fast methods, evaluate/review,
then B3, then combine and review. Two split-fit/checkpoint tests PASS.
No source scoring file was changed. Staged runner refuses to start while
original PID remains active. Scheduling change has NOT been launched: automatic
approval review rejected stopping original PID27400 and its worker tree,
requesting explicit user confirmation to interrupt the active benchmark.
Original run continues. No attempt to bypass that rejection. Pending action:
obtain explicit stop authorization, verify original PID/time again, stop it,
then launch staged driver with the same four source-root arguments and workers4.
Preserve original checkpoint/logs; staged output results/moment_rbm_fusion_v1_staged/.

## Step342 [Codex] - moment RBM fusion ready for full execution (2026-09-11)

User authorized the discussed compact six-column bank: H15,V15,m3_15,a,a^2,a^3.
Worktree .worktrees/moment-rbm-fusion-v1, branch codex/moment-rbm-fusion-v1,
parent d92f88f8; fitting/evaluation protocol frozen in5075f024. Same-answer
mean, IU-PCR, one-hidden-unit Gaussian RBM, B3; initial-model controls.
RBM uses exact normalized likelihood, analytic gradient and L-BFGS-B; fixed
unit visible variance. B3 retains its existing continuous one-group energy.
No native DEEM or tail in this experiment; no target-class semantics claimed
for raw features. Latent posterior uses explicit mean-risk orientation.

Four tests PASS: independent likelihood/gradient, representation, score/state
replay and deterministic fitting. All27 full-budget feasibility answers PASS,
zero failures,74.2s scoring. No quality inference from this subset. Full13769
benchmark is the next authorized operation, with unchanged labels/groups,
external gate/calibration,top10 readout,nine saved references and bootstrap.
The runner automatically invokes the separate arithmetic metric verifier.
Check actual LAUNCH.json PID plus RUN_STATE.json/RUN.log, not this prose, for
liveness and completion. Results: results/moment_rbm_fusion_v1/. No HTML.

## DEEM/B3 preflight stopped: input semantics correction (2026-09-11)

Isolated worktree .worktrees/deem-b3-probability-moments-v1, branch
codex/deem-b3-probability-moments-v1, frozen code a09fb334 from490f4b6c.
Three mechanical tests passed; only3 real-answer smoke records were persisted.
Full13769 benchmark was NOT started. The smoke process was interrupted.

User correctly challenged native DEEM input semantics: vocabulary probabilities
are not soft classifier decisions over a common hallucination target. The
[1-p,p] mapping satisfies tensor shape only; selected1-p and empirical moment
ranks also remain uncalibrated proxies. The protocol had disclosed proxies,
but this does not establish native DEEM applicability. Do not infer a failure
of DEEM from this pilot. Direct-probability DEEM was nearly constant in3 saved
answers; this is an engineering diagnostic only, not a benchmark finding.

B3 can accept continuous covariates through the existing adapted energy model;
its latent output is still a risk proxy requiring evaluation, not automatically
a calibrated hallucination probability. Preserve original smoke and protocol.
Next design should keep B3 continuous-input and native DEEM soft-detector paths
explicitly distinct. No replacement mapper, new threshold or full run launched.
User considers residual tail unnecessary: omit it from the next minimal direct
bank unless an explicit ablation justifies the redundant representation. Tail
is deterministic from retained probabilities; no claim that redundancy alone
proves it can never help a restricted learner. No new HTML.


## Step341 completion [Codex] - binary votes lose localization resolution (2026-09-11)

COMPLETE: all13769 model-answer rows/145597 steps, all8 arms valid, zero failures.
Scoring1215.2s on4 CPU workers;10000 source-group bootstrap draws completed.
Nine frozen references replay. Independent arithmetic implementation PASS on
17 methods/references, PB8 full denominators, within/pooled AUC and PRMScore,
held-group thresholds and fixed gate decisions. Six mechanism tests PASS.

var18__continuous_equal: PB 35.6891%, within 0.7466371, PRMScore 0.6129330
var18__binary_equal: PB 26.4017%, within 0.7003429, PRMScore 0.5184771
var18__sml: PB 23.5852%, within 0.7019670, PRMScore 0.5338107
var18__lsml: PB 21.6046%, within 0.6738234, PRMScore 0.5275814
both33__continuous_equal: PB 35.2725%, within 0.7421036, PRMScore 0.6155018
both33__binary_equal: PB 26.6025%, within 0.7012232, PRMScore 0.5190881
both33__sml: PB 24.0368%, within 0.7011130, PRMScore 0.5330434
both33__lsml: PB 21.8430%, within 0.6738387, PRMScore 0.5215161

Primary33 SML minus binary equal: PB-2.5657pp,97.5% CI[-3.6839,-1.4409];
within-.0001102[-.0026967,.0024352]. L-SML minus SML: PB-2.1938pp,
97.5% CI[-3.0963,-1.3123]; within-.0272744[-.0307292,-.0237228].
Entropy contributions18->33 give SML PB+.4516pp (exploratory95% CI
[.0392,.8684]), not a gain over continuous/incumbent methods. Continuous
18->33 PB-.4165pp (CI includes0); within-.0045335 (95% interval below0).

Frozen gate is identical: every arm has clean accuracy.50084818. Location
changes dominate. On all6800 PB answers, exact maximum-step ties: continuous33
0, binary equal6405, SML6554, L-SML6763. First-step selections1496/5550/5856/6344.
Missed true steps that are also tied at max:0/3078/3424/3707 (diagnostic only,
not an oracle candidate). Binary top10 readout saturates heavily; the fixed
first-tie rule selects early. This documents a failure of THIS thresholded
vote/readout combination, not a rejection of all spectral or continuous fusion.
Continuous-vs-binary also changes duplicate removal; not a pure single-factor
binarization effect. All solvers share entropy15-based column orientation.

Current leaders remain rawVar15 PB35.9610%, normalized Var15 equal within
.7469804, rawVar50 PRMScore.6327769. No new winner, no untouched confirmation.
Do not change thresholds/tie rules post-hoc or extend graph/threshold grids.
Native soft-input DEEM and B3 on probability/moment inputs remain explicit
requested follow-ups; neither was tested in this run. DEEM accepts probabilities
(N,classes,learners), but raw token probabilities need explicit target mapping.
Results and per-cell CSV under results/binary_moment_fusion_v1/. No new HTML.

DEEM clarification: Omri requests checking native soft-input DEEM before
assuming B3 is necessary. Official soft tensor support verified; input meaning
and class alignment still need explicit mapping. See Research_Directions.md.

B3 reminder (2026-09-11): user requests both direct probability and -log(p)
moment/contribution input lanes. OPEN; see Research_Directions.md. Do not treat
SML/L-SML results as B3 results or change the current frozen run.

## Step341 [Codex] - binary entropy/varentropy contribution fusion RUNNING (2026-09-10)

User added15 entropy contributions to the18-view varentropy bank and authorized
full execution. Isolated branch codex/binary-moment-fusion-v1 from b7bf10bd;
scoring code/protocol frozen a41ff51a. Two banks18/33, four solvers each:
continuous normalized mean, equal median votes, signed SML, binary L-SML.
Same-answer H15 covariance signs common to all solvers; local median thresholds,
no label-based threshold search. This entropy prior is explicit. Remove exact
binary duplicates/constants. No powers, third moment, Joint or new graph.
Both bank sizes refer to15 ranked alternatives at EACH token, not15 time rows.

Six mechanism/reconstruction tests PASS;27-answer feasibility smoke PASS,
all8 arms valid. Distinct binary views ranged5..18/33 in smoke; no quality
claim from that subset. Full13769-row scoring started with4 CPU workers,
checkpoint in results/binary_moment_fusion_v1/CHECKPOINT.sqlite, live progress
RUN.log/RUN_STATE.json (check actual process before resuming). Same source
hashes, spans/top10 readout, labels/folds, external PB entropy gate and PRMScore
calibration. Nine references must replay. Bootstrap10000, two primary contrasts:
33 SML-equal binary and33 L-SML-SML,97.5% intervals. Others exploratory95%.

User prefers chat before reports. No HTML. Need completion, independent metric
replay, per-answer threshold/weight archive and tie-resolution diagnostics,
then concise results vs current leaders. Do not select a new threshold/sign
using this evaluation or repair a frozen result silently.


## Step340 completion [Codex] - powers help new IU baseline, not prior leaders (2026-09-10)

COMPLETE:13769 answers/145597 steps, all6 arms valid, no fallback or failure.
Scoring208.1seconds, then full metrics and10000 paired source-group bootstrap.
All9 prior references reproduce; separate arithmetic verification of15 methods
passes PB8 denominators/gates, PRMB within/pooled AUC and PRMScore/fold thresholds.
Raw Varentropy50 source-join replay matches every saved token exactly.

Powers/method       PB macro F1  PRMB within  PRMB pooled  PRMScore
1, equal           20.5127%     .4316816     .4493077     .4643318
1, IU-PCR          27.4646%     .7346211     .6739345     .5887272
1+2, equal         20.3514%     .4301423     .4470088     .4637313
1+2, IU-PCR        27.6367%     .7281751     .6670800     .5841504
1+2+3, equal       20.2992%     .4293378     .4456382     .4637259
1+2+3, IU-PCR      33.0050%     .7341053     .6716344     .5911668

Primary degree3-IU minus degree1-IU: PB+5.5405pp,97.5% CI[3.6159,7.4839];
within-.0005158[-.0043165,.0031933]. Gains585/loses322 exact PB successes.
Degree2-IU: PB+.1722pp[-.9267,1.2301], within-.0064461[-.0084835,-.0044655].
Thus cubic expansion helps PB relative to this NEW weaker log-input baseline;
it does not improve within-answer ranking. Do not confuse it with prior gains.

Prior leaders remain stronger: RAW Varentropy15 PB35.9610%, EQUAL normalized
Varentropy15 within .7469804, RAW Varentropy50 PRMScore .6327769. Cubic-IU vs
RAW15 PB-2.9559pp,95% CI[-4.3381,-1.6133]; vs EQUAL15 within-.0128750,
95% CI[-.0163590,-.0094305]. All new arms have lower PB, within and PRMScore
points than those respective leaders. No new overall leader or confirmation.

Review limitation: frozen EQUAL uses positive weights on raw surprisal powers,
without per-rank risk alignment or a global sign fit. These columns need not
point toward hallucination in the same direction. In a post-hoc LABEL-FREE
step-score diagnostic, EQUAL degree1 opposes entropy in79.7% of6966 eligible
PRMB answers (median Pearson-.3346), but only42.9% of6744 PB answers. This
supports an orientation concern, not proof of the full loss mechanism. Do not
use IU's large gap over EQUAL as an isolated learned-weighting success. Original
scores are preserved, no post-hoc flip/new candidate. IU primary degree contrasts
use the same entropy-based global orientation at every degree.

Cubic-IU absolute standardized weight: powers1/2/3=51.46/31.03/17.51%.
Chosen-token powers total15.53%, vs6.25% equal; degree1-IU chosen41.18%.
These are coefficient shares, not measured causal contribution. IU still uses
two PCs, so the expanded basis is not an unconstrained polynomial regression.
All16/32/48 columns were active for every answer. No selected-only ablation.

Keep prior raw/normalized Varentropy references. Next discussion should resolve
risk orientation in the raw-surprisal representation before a solver sweep;
do not reject polynomial fusion generally or silently correct this frozen run.
No SU-PCR/L-SML/24-cell transfer launched. Outputs under
`results/surprisal_power_fusion_v1/`: full CSV, COMPARISON.png, metrics/paired
intervals, raw scores, coefficients, errors, diagnostics and RESULT_REVIEW.json.
New worktree branch is codex/surprisal-power-fusion-v1. Code commit b4d2e80d.

## Step340 launch [Codex] - surprisal powers including chosen token (2026-09-10)

User authorized the discussed six arms: degree1/2/3 with normalized equal and
IU-PCR, including chosen-token powers in EACH degree block. K15 fixed, original
saved logprobs; T x16/32/48, one answer per fit. No tail, q reweighting or new
time windows. Same full13769-row benchmark, top10 step readout, entropy gate,
canonical groups/folds and PRMScore calibration. New isolated worktree/branch
surprisal-power-fusion-v1 from5c0f673d. Code/protocol committed b4d2e80d.

Five polynomial mechanism tests and three inherited driver/metric/bootstrap
tests passed.27 shortest/median/95th-percentile smoke rows across9 cells all
six arms valid, exact Varentropy50 raw-input join replay. Full run started
on four CPU workers; RUN_STATE/checkpoints are current evidence of progress.
Primary97.5% intervals: degree2/3-IU vs degree1-IU,10000 source-group draws.
Nine frozen references are rescored without refitting and must replay prior
metrics. Compare all new arms with RAW15, EQUAL15 contributions and RAW50;
retain historical/access-separated references. No HTML or historical24 run.
Outputs: `results/surprisal_power_fusion_v1/`.

## Step339 completion [Codex] - six Varentropy arms complete and verified (2026-09-10)

All13,769 answers /145,597 steps completed in227.6 seconds of scoring, followed
by evaluation and10,000 paired canonical-source bootstrap draws. All6 arms have
full coverage and zero failures. RAW50 token replay equals the frozen raw.npy
Varentropy at EVERY token (maximum absolute error0); all historical RAW50
headline metrics reproduce. Independent arithmetic code in the same session
replays9 methods/references, PB8 full denominators and gate decisions, PRMB
direct-pair and pooled AUC, fold thresholds and PRMScore: PASS.

Method                 PB macro F1   PRMB within AUC  pooled AUC  PRMScore
K15 original sum        35.96099%     .73778630        .71013778  .62578123
K15 normalized equal    35.59804%     .74698035        .69998790  .61281182
K15 IU-PCR              35.34984%     .74682374        .71026131  .62268859
K50 original sum        35.67552%     .74246455        .71578339  .63277687
K50 normalized equal    34.87403%     .74632281        .71728226  .62729015
K50 IU-PCR              34.41389%     .74583243        .72055460  .62840925

Primary97.5% CIs: IU15-RAW15 PB-0.61115pp[-1.65689,+0.41549],
within+.00903744[+.00547132,+.01254320]. IU50-RAW50 PB-1.26163pp
[-2.38500,-0.12537], within+.00336789[-.00082859,+.00742300].
Exploratory IU-EQUAL within intervals include0 at both K; EQUAL alone already
improves within ranking over RAW. Learned weighting adds no demonstrated
within-answer benefit over normalization. IU50 additionally hurts PB vs EQUAL
(95% CI[-.84404,-.09810]pp). No overall fusion leader is established.

K50-RAW vs K15-RAW: within+.00467825[+.00277849,+.00662443], PB-0.28546pp
[-.98314,+.38706], exploratory95%. More retained ranks help this within-AUC
comparison, but do not explain away RAW15's stronger PB point. RAW15 is the
highest PB point among these arms; RAW50 leads PRMScore; EQUAL15 leads within
AUC. Different endpoints still prefer different methods. Not untouched testing.

IU15 gains211/loses235 PB hits vs RAW15 (lost78 early,157 late).
IU50 gains196/loses250 vs RAW50 (lost76 early,174 late); zero losses from a
changed no-error gate. All15/50 columns remain active in every answer.
Mean absolute standardized IU coefficient share: K15 ranks6-15 receive83.6%
(equal66.7%); K50 ranks16-50 receive84.2% (equal70%). Negative coefficient mass
averages5.1%/8.9%. These describe weights, not probability mass or causal blame.

Decision: retain RAW15 and RAW50 as core references, normalized-EQUAL15 as the
ranking control. Do not claim IU improves Varentropy; this implementation
improves PRMB ranking largely via normalization while sacrificing first-error
performance. The generic family of probability/contribution fusion remains
open; no further solver sweep was launched. Full B3/Joint and historical24
transfer are not included in these six localization arms.

Output in `results/varentropy_contribution_fusion_v1/`: METRICS.json,
SUMMARY.csv (clear names and all8 PB cells), SCORES.npz, COEFFICIENTS.npz
(UIDs, standardized/effective weights and intercepts), DIAGNOSTICS.json,
ERROR_CASES.json, RESULT_REVIEW.json, source/code manifests. No HTML created.

## Step339 launch [Codex] - Varentropy contribution fusion (2026-09-10)

User explicitly approved six arms: K15/K50 crossed with original Varentropy,
mean standardized contributions, and answer-local IU-PCR on those contributions.
The learned fusion algorithm IS IU-PCR; "Varentropy Contribution Fusion" names
its new input representation, not a different solver. Source branch is the
completed temporal comparison, bab8b402. New worktree and branch:
`.worktrees/varentropy-contribution-fusion-v1`, `codex/varentropy-contribution-fusion-v1`.
Experiment/protocol commit4bac838d. Five mechanism tests plus three existing
checkpoint/evaluator/bootstrap tests pass. All27 short/median/long smoke rows
across9 cells pass all6 methods; recomputed50-rank token Varentropy matches the
frozen benchmark input. Smoke gives feasibility only. Full13769-row run started;
read actual RUN_STATE/checkpoint before claiming completion.

Important comparison correction: saved token Varentropy used50 retained ranks;
previous direct probability fusion used15 plus selected surprisal/tail channels.
Same evaluation contract does not imply identical input support. The new15/50
contrast isolates support for Varentropy; IU vs normalized equal isolates
learned weights from normalization. No new graphs, lags, selected/tail channels
or historical24 transfer. Full cached development, answer-local offline fusion,
external frozen mean-entropy gate and held-group PRMScore calibration.
Output: `results/varentropy_contribution_fusion_v1/`; no new HTML.

## Step338 completion [Codex] - full temporal comparison verified (2026-09-10)

COMPLETE: all 13,769 answers / 145,597 steps, 18 new arms plus saved references.
All arms have full coverage and zero fit failures. Fusion is fitted within the
current answer; mean-entropy q=.3 gate and PRMScore cross-fold calibration remain
external and frozen. This is cached development evidence, not untouched test data.

Current-IU: ProcessBench macro F1 34.5021%, PRMB within-answer AUC .73275655,
PRMScore .62054412. Delta-IU (current17 plus signed first differences):
34.8085%, .73693101, .62394486. Primary paired 97.5% CIs (10,000 canonical-group
bootstrap draws): PB change +0.3064 percentage points [-0.2271,+0.8386];
within-answer AUC change +.00417446 [+.00208918,+.00629451]. There is evidence
of better within-answer ranking, but no clear PB improvement in this comparison.

Lag8-IU (136 inputs) loses: PB 32.3665%, AUC .71149969, PRMScore .59786990.
PB difference -2.1356 points, 97.5% CI [-3.3146,-0.9778]. Hierarchical fusion
and shrinkage do not rescue this representation. Chronological-chain LIU gives
34.5863% PB: four extra hits and no lost hits, a small exploratory effect;
PRMB AUC change is inconclusive. Delta-Equal gives 34.9202%, .73872061,
.62365012; this point estimate does not establish learned IU superiority.
Entropy remains higher on PB (35.4444%); saved varentropy is 35.6755% PB,
.74246455 within AUC, .63277687 PRMScore. No new overall leader is established.

Error review: Lag8-IU gains175/loses283 PB successes; 157 lost cases move late,
126 early, none due to gate changes. Delta-IU gains72/loses56 (32 late/24 early).
Lag degradation also occurs for traces >512 tokens, so insufficient row count
alone does not explain it. Coefficient shares concentrate on lag3/lag4 (~15%
each) versus current token (~9.7%); this is descriptive, not causal proof.

Separate arithmetic replay PASS: source/code hashes, PB8 full denominators,
PRMB direct-pair AUC and pooled AUC, source-group fold thresholds and PRMScore.
Within-answer AUC uses 6,030 of 6,969 PRMB answers with both label classes;
all answers remain in the other applicable metrics. Original1072-row solver
checkpoint replay agrees to2.3e-15. No numerical drift from the speed change.

Next direction: retain Current17 as anchor and investigate level-plus-change;
do not expand the lag window or sweep graphs to rescue this result. Before
claiming chronological information caused the Delta gain, a matched shuffled-
change control should isolate temporal alignment from simply adding coordinates.
That follow-up is not run. Full B3/CONT/Joint graph roster, positional encoding
and temporal historical24 transfer are also not yet run. Tail/selected ablation
is still open. No new HTML: results are in chat and machine-readable artifacts:
`results/direct_probability_temporal_v3/{METRICS,RESULT_REVIEW,ERROR_ANALYSIS}.json`,
`SUMMARY_READABLE.csv`, `SCORES.npz`. Both original and optimized checkpoints
are retained locally; large checkpoint databases are ignored by Git.

## Step338 launch record [Codex] - temporal probability fusion (2026-09-10)

User authorized execution after discussing EDIS spikes, lagged probability
coordinates, level-plus-change, two-axis fusion and chronological graphs.
Chat-first communication; no new HTML report and no performance thresholds.
The prior work was committed as 438b03a54. New sparse worktree:
`.worktrees/direct-probability-temporal-v3`, branch
`codex/direct-probability-temporal-v3`; raw caches are read in place from root.

The first full stage scores all 13,769 localization rows with 18 arms: Current17,
Lag8x17 and Delta34 crossed with Equal/IU/diagonal-LW/Joint-inspired-LW; shuffled
history Equal/IU controls; two hierarchical IU fusion orders; chronological and
permuted chain LIU. K15, top10 step readout, labels/folds and gates stay frozen.
One-answer fusion fitting; external calibration remains explicitly declared.

Clean-context preflight found issues and they were fixed before launch. Nine
core tests and three driver tests pass. Full-reference replay smoke: 27 short/
median/long answers across all nine cells, 20-1454 tokens, all18 arms succeed.
No scientific ranking is drawn from the smoke. Source manifest, code and
protocol bind resumable SQLite checkpoints. Code commits: ad9a9bb8, 2d4d1c36.

Wide-input IU's complete pair L2 system now uses its exact analytic inverse,
leaving the g2 grid and estimator unchanged. Sixty covariance fixtures match
the frozen solver (max weight difference 1.2e-13). All27 real smoke answers
match across all18 arms (step-score difference <=1.4e-15), 2.56x faster. The
original 1,072-row partial run is preserved in the sibling `_original_solver`
directory. The optimized full run restarts from zero; no mixed-code result.

Primary contrasts: Lag8-IU vs Current-IU; Delta-IU vs Current-IU. Full results
and 10,000 canonical-group bootstrap draws are pending. Read current liveness
from actual processes/checkpoint, not this entry. Output:
`results/direct_probability_temporal_v3/`. Do not claim B3, full Joint graph
roster, positional encoding or historical24 temporal transfer has been run;
these remain the explicit continuing scope after the first comparison.

## Step337 [Codex] - selected-token and tail probability fusion v2 completed (2026-09-10)

The bounded gray-box v2 experiment is complete in worktree
`.worktrees/direct-probability-fusion-v2`, branch
`codex/direct-probability-fusion-v2`. A clean-context preflight review passed
before scoring. The frozen design appends two coordinates to the direct Top-15
probability matrix: selected-token surprisal and residual probability mass
outside Top-15. K=15, top-10 token-to-step readout, benchmark populations,
annotations, folds and gates stayed fixed. The localization fit remains inside
one answer (`T tokens x 17`); the historical route fits within each of the exact
24 complete-answer cells (`N answers x 17`).

The residual is not float noise. Only an impossible Top-15 mass excess up to
3.35e-7 is clipped. On the exact scored populations, every localization answer
and every historical cell has tail variation above the 5e-7 computation guard.
Each column is standardized before fusion: across tokens inside one answer for
localization, and across answer summaries inside one historical cell.

Full localization result: Token Entropy 35.4444% ProcessBench / 0.730111 PRMB
within / 0.625426 PRMScore. Selected+Tail Equal gives 34.5631% / 0.733919 /
0.620685; IU-PCR 34.5021% / 0.732757 / 0.620544; Joint shrinkage 34.5948% /
0.729682 / 0.619755. IU minus entropy is -0.9423pp PB, 97.5% CI
[-1.9397,+0.0265], and +0.002645 PRMB within, CI[-0.001196,+0.006501].
The new inputs do not improve the frozen one-answer localization route.

Historical all-24 macro AUROC: Selected+Tail Equal 0.778108, Historical IU-PCR
0.776087, entropy 0.771739, Selected+Tail IU 0.768831, Joint 0.728539. Adding
the two coordinates improves the same direct-matrix IU v1 by +0.005718,
hierarchical 97.5% CI[+0.001544,+0.011752], in 20/24 cells. Equal improves its
v1 form by +0.000936, paired-cell CI[+0.000216,+0.001684]. Equal's exploratory
lead over Historical IU-PCR is only +0.002022, paired-cell CI
[-0.0031,+0.0073]; it is not a clear new overall winner. The answer-level signal
is real enough to isolate, but current covariance-based weighting does not use
it as well as equal weights and Joint remains weak.

An independent result replay passed: 4/4 localization methods, five scores in
24/24 historical cells, complete coverage, no fallbacks, finite 17-coordinate
weights, valid intervals and clean HTML. The combined experiment cannot assign
the historical gain separately to selected-token surprisal or tail mass. The
next bounded experiment, if approved, is a two-coordinate ablation under equal
weights on the frozen 24-cell panel; only its useful coordinate would then be
tested in localization. There is no K, window, gate, graph, lambda, DEEM or
model sweep, and no automatic result threshold. Main evidence:
`results/direct_probability_fusion_v2_selected_tail/REPORT.html` and
`RESULT_REVIEW.md`.

## Step336 [Codex] - direct probability-rank fusion v1 completed and not promoted (2026-09-10)

The frozen gray-box experiment is complete in the isolated worktree
`.worktrees/direct-probability-fusion-v1`, branch
`codex/direct-probability-fusion-v1`. It used the full 13,769-row
ProcessBench/PRMBench population and the exact historical 24-cell complete-case
population. A clean-context read-only subagent reviewed the protocol before the
run and independently reconstructed the results after it. Six focused tests and
all frozen-input checks pass; localization coverage is 13,769/13,769 and there
are no fusion fallbacks.

The input to answer-local fusion was `T_answer x 15`: one observation per token
and the sorted top-15 vocabulary probabilities as columns. Rank 1 was oriented
as `1-p1`; ranks 2..15 were `p2..p15`; retained probabilities were not
renormalized. There was no token window. A separate top-10 mean converted token
risks to each reasoning-step score. K=15 was frozen to match the support used by
the stored top-K entropy; it was not tuned.

ProcessBench all-eight: token entropy 35.4444%, equal probability fusion
34.8215%, direct IU-PCR 34.7513%, Joint shrinkage 34.7122%. PRMB within-answer:
0.730111, 0.733304, 0.732730, 0.732360, respectively. Direct IU minus entropy is
-0.6930pp PB, primary 97.5% CI[-1.6216,+0.2633], and +0.002619 PRMB within,
CI[-0.001431,+0.006528]. PRMScore also stays lower than entropy for every direct
fusion arm. Therefore v1 does not improve localization.

Historical all-24 macro AUROC: Historical IU-PCR 0.776087, entropy 0.771739,
equal direct fusion 0.777172, direct IU 0.763113, Joint shrinkage 0.730748.
Direct IU minus Historical IU is -0.012973 with the primary hierarchical 97.5%
CI[-0.031393,+0.000213]. Equal's post-hoc +0.001085 is small, has no predeclared
primary interval and does not transfer to PB or PRMScore. Keep token entropy as
the localization anchor and Historical IU-PCR as the 24-cell anchor.

Delivery corrections from the independent review are complete: the Markdown
report now shows the primary hierarchical interval; localization and historical
runs have separate manifests; historical score arrays are float64 and replay
all stored AUROCs within 2.3e-16. The HTML now includes coefficient plots and a
clear no-promotion decision.

One representation gap is now explicit. V1 did not add the sampled token's
probability as a separate column and did not add residual tail mass as a separate
column. Both signals are available from existing caches: `token_spilled_energies`
stores sampled-token negative log-probability, while the residual outside top-15
can be calculated without renormalizing the saved probabilities. This is a
bounded follow-up candidate, not part of the completed v1 result. Main report:
`results/direct_probability_fusion_v1/REPORT.html`.

## Step335 [Claude] - BOCPD on token entropy: no gain; white-box localization capture ready, VPN down (2026-09-10)

BOCPD (verified reset-before-observation filter) on per-token entropy: ProcessBench tie
(35.3-35.5 vs 35.44), PRMBench within-answer clearly worse (0.67-0.72 vs 0.730, intervals
below zero). `results/token_bocpd_v1/`. Omri's next experiment, token entropy + white-box
layer views for localization, needs a GPU capture: no per-layer data exists for the
PB/PRMB rows. Driver `cluster/run_localization_layer_views.py` is written and CPU-smoked;
submit three chains (PB Qwen3-4B, PB Qwen3-8B, PRMB Qwen3-8B) when the TAU VPN is up,
then rclone to Drive and run the token-level fusion locally.

## Step334 [Claude] - the 8-token window bank is the bottleneck; token-level wins both benchmarks (2026-09-10)

Onset (C7) and self-innovation (C8) streams inside the window bank: no gain (PB -0.4 /
-0.5 pp, within-answer within +-0.002). But plain per-token entropy with the top-10
readout and the same entropy gate scores PB all-8 **35.44**, PRMB within **0.7301**,
PRMScore 0.625, versus the 27-feature window IU 31.84 / 0.7196 / 0.597: paired
**+3.60 pp [+2.39,+4.93]** and **+0.0106 [+0.0062,+0.0152]**, all eight cells. The
8-token window mean of the same stream gives 31.76 / 0.706, so windowing costs ~3.7 F1.
Token varentropy 35.68 / 0.7425 / 0.633 (post-hoc best stream); token-level fusion of
the nine primitives (equal or answer-only IU-PCR over tokens) ties token entropy.
Details `results/token_level_readout_v1/METRICS.json`, `results/fusion_onset_innovation_iu_v1/`.
Consequence: the answer-only candidate should move to a token-level representation
(the pooled historical method already is token-level, 34.2-34.6). Readout of record
stays top-10 token mean; gate stays mean-entropy q0.3. Multi-width window fusion
(mean/min/median/vote over widths 8-32, `results/fusion_multiwidth_*`) gave no
two-benchmark gain and is closed as a direction.

## Step333 [Claude] - readout/PRMScore/Mind-the-Gap follow-ups (2026-09-09)

Lost-peak forensics: the candidate's 31 lost and 50 gained ProcessBench peaks are near-ties
(median margin 0.045 SD, truth ranked second in 27/31, 17/31 share an 8-token window); of
2,561 shared misses 938 are adjacent to the truth. Readout variants on frozen dual__iu
windows (`results/fusion_step_readout_v1/`): boundary-aware rules hurt, top-10 token mean
gives +0.5 F1 / +0.5 raw exact (31.70 / 26.6); adjacency is real signal displacement, not a
spreading artifact. PRMScore adapter now exists (`results/prmscore_adapter_v1/`, official
evaluator port, ids matched 6,969/6,969): supervised Qwen2.5-Math-PRM-7B 0.6546; our
label-free arms 0.57-0.60 (context__iu 0.5965 best; all within 0.013). Mind the Gap's public
code has no ProcessBench/SLA code and applies a cumulative running mean before the EMA that
neither the paper nor our reproduction includes; their SLA cannot be reproduced from the
release. Next: (a) rerun our Mind-the-Gap reproduction in both variants (paper, code) with
declared step aggregations and report SLA-on-erroneous, exact and within-one beside F1 on v3;
(b) carry top-10 readout as a declared readout variant in the next full pass; (c) add
PRMScore column to the advisor table.

## Step332 [Codex shrinkage review] - positive versus IU; mechanism claim narrowed

Independent review of Claude's completed shrinkage run (2026-09-09):
results/fusion_shrinkage_iu_codex_review_v1/REPORT.md and
REVIEW_WITH_DIAGONAL_CONTROL.json. All13769 score/metadata pairs read and hashed;
13748 valid IU replays exactly match the frozen benchmark. Metrics for ten
methods/controls independently reproduced; within-answer AUC uses direct
positive-negative comparisons. Gates, detectors, joined labels/IDs and folds
match the previous independent gate review. No fusion refit/new experiment.

Primary full/joint/LW: PRMB-within .707230 -> .710006, PB all8 31.1580 ->31.7718.
10000 canonical-source-group bootstrap draws (3483 groups), 97.5% intervals
for the two primary comparisons per endpoint: within delta[.000672,.004843],
PB delta[.096687,1.168429]pp. These condition on fixed predictions/calibration;
not all prior research selection uncertainty. PB point gains in all8 cells;
50 corrected predictions versus31 lost, net19 model-answer cases. Clean successes
stay1179. About56% of the macro gain comes from GSM8K-Q4.

Important corrections to the source interpretation: solve/diag/1.0 is a useful
simple control (.708962 /31.6777), not uniformly harmful or inert. Primary minus
this control: within+.001045 CI95[-.000450,.002547], PB+.094029pp
CI95[-.290809,.505985]. No clear advantage of the Joint target over this control;
not proof of equivalence. The control is post-evaluation discovery, not a promoted
winner. Its PB advantage over IU crosses zero with a97.5% interval.
The proposed solve-only JOINT target loses PRMB (.695577). Full versus subspace
at the same LW supports re-estimating rho on PRMB, but its PB contrast includes0.
Subspace/joint/1.0 has32.22% PB versus entropy32.10 at the point level; the direct
interval includes0. Primary also has no clear PB advantage over entropy.

Keep full/joint/LW as a development candidate, with diagonal and entropy controls.
No new alpha/graph/feature search or winner declaration. LW here is an automatic
heuristic, not a proved optimum for a same-data target and dependent windows.
The target is inspired by Joint; it is not the identical iterative model.
A PRIMARY list exists in source; pre-run timing/version is not bound by a manifest.
The central consolidation builder now includes the ten audited common-gate rows
and these conclusions in a separate common-gate panel. Nine completion/report tests pass. Full sampling continues
under its original frozen driver/protocol; final reflection still awaits completion.


## Step332 [Claude] - Joint assumption as closed-form shrinkage in IU: small two-task gain (2026-09-09)

`results/fusion_shrinkage_iu_v1/REPORT.md`. All 13,769 rows, frozen dual__iu pipeline
replayed exactly (max abs diff 0.0), only the covariance handed to IU changed:
cross-stream entries replaced by the rank-1 shared-signal prediction (Joint's model,
no iterative fit, fixed 9-stream partition, Ledoit-Wolf alpha within answer).
Primary row full/joint/LW vs IU: PRMB within +0.0028 [+0.0010,+0.0047], pooled
0.6682 -> 0.6704, PB all-8 (entropy-q0.3 fold gates) 31.16 -> 31.77 (+0.61 pp
[+0.17,+1.10]); alpha=1 dose 0.6723 / 0.7112 / 31.96. Coverage = IU. Block-zeroing
and diagonal controls inert or harmful; transform partition worse; Codex's
solve-only isolation harmful (-0.012): the gain comes through rho estimated from
the shrunk covariance. Entropy-only control still not beaten on PB F1 (32.10).
27 arms scored, primary rows pre-declared; development evidence, not a promotion.
Codex sampling run resumed by its supervisor after the disk fix (3,128/13,769 at
last check).

## 2026-09-09 [Codex] - storage recovery of frozen full sampling

Omri explicitly asked Codex to resume the consolidation work after the disk
incident; Claude owns the separate shrinkage-IU experiment. The unchanged
sampling driver was manually relaunched as PID147012, and the existing
completion supervisor as PID18784, which adopted that driver. The driver's
own startup checks passed and it is now SCORING: last observed3038/13769,
with2 new records after recovery. Both processes are alive;
supervisor stderr is empty. Current handles supersede the older
PID142012/PID137104 entries below.
Logs: results/research_consolidation_v1/full_sampling_storage_resume_20260909.log,
supervisor_v3.stdout.log and supervisor_v3.stderr.log.

Recovery review PASS: all357 frozen input/code hashes matched; all3036 committed
records had matching identities, manifest digests, all56 method keys and matching
NPZ hashes. The prior3033 progress count lagged completed writes. One orphan NPZ
without committed metadata is not counted and will be recomputed by the original
driver. One accepted record lacks optional execution_acceleration telemetry
because the disk filled during that extra metadata write; its scientific arrays
and method metadata are complete. No diagnostic values were invented.

Failure states, original log/manifest and all three unfinished files were copied
and SHA256-verified under results/research_consolidation_v1/storage_recovery_20260909/.
The recovery changed no scientific source, scores, gates, arms or input files;
Codex deleted nothing. About33.7GB were free at checkpoint review. Existing NTFS
compression remains; space is recorded by the supervisor on every polling cycle.

Historical Joint and the independent gate audit remain COMPLETE REVIEWED.
Finish the remaining10733 sampling records, then its existing evaluation and
Hebrew consolidation HTML/ledger. Sampling still uses its frozen original GMM;
do not present it as the shared entropy-q0.3 follow-up. Stop after the reflection
before opening the next improvement experiment. No performance ranking from
these incomplete sampling records. Recovery REVIEW.json and LAUNCH.json hold
machine-readable evidence; inspect live states for current progress.


## 2026-09-09 [Claude] - disk-full incident; Codex sampling stopped at 3,033/13,769

At about 01:05 the C: drive reached 32 MB free. `localization_full_sampling_v3`
died with OSError 28 (RUN_STATE FAILED_CHECKPOINTS_PRESERVED, 3,033 records,
6,075 score files intact) and the consolidation supervisor stopped itself with
STOPPED_REQUIRES_REVIEW. This is a storage failure, not a scoring bug; no
scientific input, driver or frozen file was modified. With Omri's approval the
28 GB duplicate LFS checkout `.worktrees/a6-s0b/dataset_cache` (closed A6/PTNI
worktree, zero uncommitted dataset changes) was deleted; 28.6 GB are now free.
It is restorable with `git lfs checkout` inside that worktree. Not done:
`git lfs prune` (would free 5 GB; three objects are missing on the remote).
Sampling scores cost ~1.35 MB/record, so the remaining 10,736 records need
~14.5 GB; resume the SAME driver/output from its checkpoints when convenient.
Claude's next experiment (shrinkage-covariance IU, answer-only, all 13,769
rows, frozen routes/peaks pipeline, saved entropy-q0.3 fold gates) writes only
to `results/fusion_shrinkage_iu_v1/`.

## Step332 [Codex] - authorized consolidation IN PROGRESS (2026-09-08)

Omri approved `docs/experiments/RESEARCH_CONSOLIDATION_20260908.md` after the
pause: complete BOTH frozen runs, historical Joint first then sampling, audit
Claude's gate, deliver a Hebrew HTML reflection/evidence ledger and return
before opening a new improvement experiment. No expanded search. PB first;
raw mean entropy q0.3 is the fixed gate candidate, explicitly externally
calibrated. Both benchmark panels and historical comparators remain required.

Durable completion supervisor ACTIVE: PID137104 (exec92438),
`scripts/complete_research_consolidation_v1.py`. Authoritative live handoff:
`results/research_consolidation_v1/RUN_STATE.json`, supervisor_v2.stdout.log and
supervisor_v2.stderr.log. It adopts the existing Joint process, then starts
sampling, resumes ONLY normal invocation-cap exits, and stops visibly on an
unexpected failure. At the end it builds/validates the Hebrew reflection and
updates completion docs, then stops BEFORE new improvement experiments.
Do not start a competing driver/supervisor. Draft HTML already exists but is
explicitly IN_PROGRESS_NOT_FINAL and excludes partial-run performance results.

Historical Joint COMPLETE REVIEWED: exec77379 / PID143184 ended exit0,
all245 fits,13769 records,34 displayed arms,6615 historical control arrays
replayed exactly. `results/historical_joint_refit_v3/REVIEW.json` PASS.
PB all8: internal_cont15.2179%, internal_joint16.0239%, model-inverse0
28.1063%, LIU01028.6502%, LIU05028.1685%. LIU improves PRMB within-answer
over model-inverse0 (exploratory positive intervals), but PB intervals and
LIU010 versus node permutation include zero. No matched permutation050.
Internal CONT/grouping is distinct from fixed-family CONT (~34.60% all8).

Sampling now ACTIVE: PID142012, supervisor adopted the unchanged run and
verified checkpoints before resuming from337; last checked345/13769.
Do not launch a second driver. Inspect current RUN_STATE and process liveness.

Independent Step331 review COMPLETE PASS:
`results/fixed_gate_completion_review_v1/REPORT.md` and REVIEW.json.
All6800 raw entropy summaries, six actually evaluated arms, entropy thresholds,
metrics, peaks and validity replay. q0.3 minus GMM +11.1540pp,
95% CI[9.2911,13.0157], 10000 canonical-group draws. Fixed predictions only;
no calibration-refit or detector/q-selection uncertainty. Same-destination
transfer:28.9864 vs nested29.0339 (hard),32.8907 vs33.5920 (easy).
Claude's source outputs and code unchanged. No new improvement trial started.


## Step331 [Claude] - fixed-constant gate closes most of the answer-only PB gap (2026-09-08)

`results/fusion_fixed_gate_v1/REPORT.md`. Frozen peaks of the 19 answer-only anchors;
only the GMM/BIC no-error gate replaced by one constant on raw mean token entropy.
dual__iu eight-cell PB macro F1 20.00 -> 31.31 (label-chosen, nested folds) / 31.16
(label-free quantile q=0.3); +11.4 pp, CI [+9.5,+13.2]; historical pooled IU 34.18;
oracle gate 43.4. Fused-score thresholds cannot work (answer-level AUC ~0.48, within-
answer z-scoring removes scale); raw entropy summaries reach AUC 0.74-0.78 per cell.
With this gate, fusion arms and the entropy-only control sit within 1.5 pp on PB.
Location is the shared bottleneck (oracle ceiling 43-46%). Development data; freeze
detector/q before confirmation. Question for the 14.9 advisor meeting: is one
supervised scalar for the no-error decision acceptable, or must the gate be label-free
(quantile version is within 0.2 pp)? No running Codex driver or frozen file touched.

## Step330 - full pass2a reviewed; full sampling live; historical Joint resumed

Full answer-only pass2a is COMPLETE_REVIEWED_PASS2A: all13,769 records,
36 entries,73 registered contrasts,27,538 score-file hashes,110 pilot
replays and19 unchanged full anchors. Report:
results/localization_full_shortlist_v3/evaluation/REPORT.html.
Exec94852 / PID204 ended; do not restart. The frozen report's prose saying
historical refits are pending is stale: five controls were completed in
Step329. Keep the immutable report and consult the current registry.

Full PRMB pooled / within-answer / PB-Q8:
IU .668225/.707230/20.3828%; Joint100 .670301/.708223/20.9731%;
Joint100 graph .669238/.708125/21.2245%; permuted .670589/.708803/21.1416%;
IU+Joint-graph mean .669895/.709518/20.6279%; IMM .662344/.684585/21.8666%.
Mean fusion has small exploratory positive PRMB deltas versus IU (within
delta .002289,95% CI[.000397,.004164]) but PB-Q8 interval includes zero and
the all-eight PB point is slightly lower. IMM's PB lead is uncertain and
PRMB regresses. Graph100 does not beat lambda0 or permutation on PRMB.
No consistent two-task winner. These are full DEVELOPMENT findings.

**Full sampling LIVE:** exec95830 / PID145132, two workers,
results/localization_full_sampling_v3/RUN_STATE.json; last checked175/13,769.
Eight prior selectors x seven cores =56 entries (seven are full aliases,
49 additions). Full-grid, uniform, risk, transposed/permuted DUFS, diffusion,
low/high entropy tails and entropy quantiles are all retained. The automatic
report will include36 shortlist entries and five historical controls:97
displayed entries, with142 registered paired contrasts. The 93 old sampling
and31 entropy comparisons are retained. No partial performance ranking.

PREFLIGHT PASS:110 previous answers x56 outputs replay (6,160 bundles), plus
shortest/longest full-data traces. Fast preflight PASS:25,642 arrays and all
non-timing diagnostic fields match exactly;1,769+200 ARI cases match sklearn.
Execution-only ARI replacement is local to each worker's imported module
and restored afterward; no historical/scorer source was changed. Summed
record time3064.86->2577.44s, with machine contention not held fixed.
Four evaluator tests pass (contrast continuity, explicit grouped resampling,
missing-fold handling, rendered values). Core/driver/evaluator/protocol are
now frozen in MANIFEST.json. Do not edit live dependencies.
Driver: scripts/run_full_sampling_v3.py. Eight-hour cap stops submitting
new answers and drains at most two active jobs. Resume the SAME driver after
CHECKPOINTED_INVOCATION_CAP; OS byte lock rejects duplicate invocations.

**Historical Joint resumed:** old exec28881 / PID125792 ended at its cap,
238/245 fits. Resumed as exec88771 / PID16240; last checked240/245, PRMB
outer0. Five PRMB outer fits remain. Evaluation attached.
Do not duplicate or restart a live process. Claude's worktree is unchanged.

Full frozen-output error analysis is reviewed:
results/localization_full_error_modes_v3/REPORT.html (64 method/cell bundles).
IU Q8:2,221 error answers,591 exact raw peaks,253 suppressed by the gate,
338 final exact;622 early peaks,1,008 late, and719/1,179 clean false alarms.
Historical IU has675 actual raw peaks,150 suppressed,525 final exact and
501 clean false alarms. Different access/representation/readout remains a
confound; these counts are descriptive, not a causal decomposition.

An initial diagnostic assertion caught our assumption that every saved
ranking curve produced the PB peak. Historical PB uses top10-mean per step;
the joined PRMB ranking curve stores spanmax. The corrected analysis reloads
all40 PB outer-fit top10 arrays and verifies all peaks. Original MANIFEST.json
records the failed diagnostic; MANIFEST_V2.json binds the corrected report.
No experimental output changed. Keep this distinction in further analyses.

Storage:8.53GiB free before compression, with full sampling metadata risking
exceeding capacity. Marked only the sampling scores directory NTFS-compressed
and compressed its completed JSON files. STORAGE_REVIEW.json PASS:174 prior
metadata hashes unchanged, four new files inherited compression;196.0MB of
metadata occupies49.1MB (about4:1). No files deleted and no scientific bytes
changed. New output files inherit compression. I/O timings remain descriptive.

Graph-axis clarification: docs/research_notes/graph_axes_clarification_2026-09-08.md.
Joint-LIU graph nodes are observations; R=Z.T L Z/N acts on feature weights.
Feature grouping and DUFS window selection are different mechanisms. The
visual guide now points to full-data evidence and labels older pilot scores.
Remaining historical feature contracts/LIU/localizers, sampling completion,
algorithm development from full evidence, untouched confirmation and the
separate historical24 detection transfer remain active. Subsets remain
feasibility checks only, not evidence for selecting the research direction.

## Step 329 - first full historical comparison COMPLETE; Joint extension live

The five-control corrected historical panel is complete and reviewed:
results/historical_fusion_refit_v3/REPORT.html. Exec51883 / PID9392 ended,
exit0. All245 fits, 13,769 records, 19 answer-only anchors plus five historical
controls. No historical fit failures. The historical controls fit other
training answers; their PB threshold uses nested training labels. This is
explicitly a pooled/calibrated comparison, not an unsupervised answer-only
candidate. V3 labels and canonical outer/inner source groups are unchanged.

PRMB fold-mean AUC / within-answer AUC / PB-Q8:
answer-only IU .668280 / .707230 /20.3828%; historical IU .680803 / .699245
/34.2940%; U-PCR port .677646 / .697011 /34.2558%; equal active23 .674043
/.692923 /33.9237%; fixed-family CONT .682141 / .698438 /34.8656%; guarded
CONT .678970 / .692557 /34.3717%. Thus the old34-35% PB level survives the
source/calibration correction. Do not claim the gap is entirely fusion or
entirely the gate: representation, fit scope and readout also differ.
Historical IU and CONT beat equal on PRMB ranking, but their paired PB gains
over equal include zero. No consistent two-task fusion winner is established.

Review.json PASS plus REVIEW_SUPPLEMENT.json PASS:312 full metric bundles,
25 independently counted threshold searches, 30,150 explicit within-answer
positive/negative-pair AUC checks,24 HTML rows and five links. Same-session
review; no browser or external scientific review claimed. All scoring
dependencies remain frozen. PRMB contrasts use common valid rows; PB keeps
all failures in the denominator; bootstrap conditions on fitted predictions.

**Historical Joint extension LIVE:** exec28881 / PID125792,
results/historical_joint_refit_v3/RUN_STATE.json. Same245 fits/full population.
Adds internal CONT, hierarchical Joint, gate050/100, LIU010/050, diag010/050,
model-inverse lambda0 and graph-node permutation010. Original five controls
are copied exactly with source hashes and preparation equality checks.
First job completed; currently outer0/inner0 of PB GSM8K-Q4. Do not restart
because a job is slow. Eight-hour invocation cap checkpoints BETWEEN jobs;
if it reaches CHECKPOINTED_INVOCATION_CAP, resume the SAME driver/run.

Exact historical preflight:40 weight/readout arrays and grouping/gate
diagnostics replay. Execution-only dense-integer ARI acceleration matches
sklearn on1,769 arithmetic cases (PREFLIGHT_FAST.json) and
all40 complete historical arrays exactly; local replay431.87s ->180.89s,
with unheld machine contention. Only the private imported historical module
binding is temporarily changed; historical source and sklearn stay intact.
Protocol/core/runner/evaluator and preflight evidence are now frozen.
Eight registered mechanism contrasts distinguish graph/diagonal/gate effects
from the change between hierarchical and model-inverse weights. Current
entry point preserves historical admission/fallback, not a new Jacobian guard.

Full answer-only pass2a remains LIVE: exec94852 / PID204, three workers;
latest checked6,600/13,769. Full sampling, remaining historical feature
contracts/DUFS-LIU/dedicated localizers, untouched confirmation and the
separate historical24 final-answer transfer remain open. The goal is active.
AIRCC probe: no local aircc alias/config; direct host outside sandbox timed
out on port22. No cluster job was submitted and no VPN diagnosis is proven.

## Step 328 - full-population evidence policy (2026-09-07)

Omri explicitly supersedes the previous preference for short research pilots:
small experiments are now implementation/feasibility checks ONLY, not evidence
of improvement or a basis for selecting a research direction. Comparative
conclusions require the full matched development benchmark and historical
comparators; publication confirmation requires untouched data after method
lock. This changes evaluation scope, not the primary single-answer fit scope.
The Step327 replay explains why: the identical IU recipe scored 30.16% on the
86-answer PB-Q8 pilot versus 20.38% on all 3,400 answers.

Full pass2a continues unchanged (parent PID204, exec94852, three workers).
RUN_STATE.json recorded 2,700/13,769 at this policy update; partial output is
not a comparison result. Finish full shortlist and corrected historical
refits, then sampling comparison, before further algorithm selection. Retain
all pending comparator families and report their implementation status.
Canonical policy is in CLAUDE.md; do not revive superseded pilot instructions.

**Full corrected historical first panel STARTED:** exec51883 / PID9392,
results/historical_fusion_refit_v3/RUN_STATE.json. Five fixed controls:
iu_c2_s25_l2_exoff, iu_c2_s25_l2_exon, equal_all23,
fixed_family_cont_unguarded, prov5_cont. All 13,769 rows; 245 group-disjoint
fits (40 PB outer + 200 inner + five PRMB outer). PB threshold uses only
inner held-out outer-training labels: this is an explicitly calibrated
pooled comparator, not the answer-only contribution. No tuning of fusion.
Exact historical GSM8K-Q8 outer0 replay passes all20 weight/readout arrays
with zero difference. This validates the adapter, not scientific improvement.
Run/evaluator/core/protocol and historical sources are frozen. Do not edit
live dependencies. Automatic evaluation, intervals, HTML and review follow
scoring; no result yet. Latest verified first checkpoint28/245 complete.
Protocol: docs/experiments/HISTORICAL_FUSION_REFIT_V3.md. Remaining registry
families are still required; this is not the complete historical benchmark.

## Step 327 - single-answer pilot/full regression explained (2026-09-07)

Omri correctly recalled better SINGLE-ANSWER results. The prior explanation
in terms of multiple-answer fitting was incomplete. A frozen-output audit
replayed all 110 pilot records x 19 original methods exactly, including labels,
step scores, peaks, validity and decisions. For the IDENTICAL Qwen3-8B routed
IU recipe, PB is 30.1599% on the 86 pilot answers, 20.1219% on the other 3,314,
and 20.3828% on all 3,400. Original graph condition1000 is 29.9371% pilot versus
21.4172% full. These are population-expansion gaps, not refitting changes to
the pilot answers. The length-stratified pilot is not a representative full
population estimate; no causal attribution or significance claim is made.

OmniMath contributes 73.07% of IU's 9.78-point macro drop: 49.3827% on 24
pilot answers versus 20.8072% on all 1,000. Clean accuracy falls from 8/10 to
85/241 and exact-error success from 5/14 to 112/759; raw peak correctness
falls from 7/14 to 191/759. Gate and location both need attention. Full ranking
evidence weakens the current recipe; do not present the pilot as progress.

Newer pilot candidates are NOT all in the anchor pass: graph condition100
30.2229%, IU+Joint graph mean 31.9830%, and risk equal/permuted graph 33.9200%.
The latter is a simple control, not proof of learned graph benefit. Full
pass2a remains live (exec94852, parent PID204, three workers); no restart.
Sampling is pass2b. Corrected historical refits take precedence over another
feature/hyperparameter pilot. Joint currently receives up to 27 features;
three is a minimum GROUP size, not a three-feature selected roster. The
historical24 transfer is final-answer DETECTION, not localization.

Audit: results/localization_full_regression_audit_v3/REPORT.html,
FINDINGS.json and REVIEW.json (PASS). No changed scores or new model fits.
Full benchmark and consistent two-task improvement remain incomplete.

**Step325 COMPLETE: full19-anchor evaluation and supplemental review PASS.**
Scoring exec50440 ended after19425.09s (5h23m45s), at19:57:17 Israel time.
The evaluation adapter had not been connected then; it was implemented in
this status follow-up. Eval exec65524 is now terminal, exit0,454.13s including
verification/joins/metrics/intervals. Supplemental review exec56541 also exit0.
No active scorer/evaluator remains from these three handles.

Full report: results/localization_full_benchmark_v3/evaluation/REPORT.html.
All13769 rows /19 original anchors; all27538 frozen score-file hashes checked.
110 pilot answers and19 metric bundles replay exactly (float tolerance only
for aggregate arithmetic).1000 joint draws across3483 canonical groups keep
repeated scorers/answers and66 cross-task links together;19 absolute interval
bundles and26 fixed paired contrasts. All are development/exploratory evidence.
Supplement verifies joined provenance,nine label hashes,261611 method-validity
records,363 per-cell source/validity groups,47 report rows and five links.
Code/contract independently inspected by a read-only second agent; no external
scientific or browser validation. Scientific scoring files stayed frozen.

Full routed anchors (PRMB pooled / within-answer / PB-Q8 four-cell macro):
IU .668225/.707230/20.3828%; equal .665479/.702973/20.7035%;
Joint0 .668519/.706393/21.0893%; graph .667464/.706779/21.4172%;
permuted graph .669289/.707787/21.3452%.
PRMB coverage6952/6969, mixed6021;17 short PRMB scores unsupported. PB keeps
all3400 answers per scorer including two short unsupported MATH answers.
IU PB-Q4 is19.6252%, eight-cell macro20.0040%. These Joint arms retain the
ORIGINAL condition1000, not the current condition100 candidate. Full Q8 graph
minus IU difference+1.0344pp has95% CI[-1.0591,+3.0487]pp; no clear advantage.
Graph pooled PRMB is slightly lower than lambda0 and node-permuted controls;
no demonstrated graph mechanism benefit. Pilot PB points were more favorable
than the full population, not a refitting bug: matched pilot outputs replay.

Next still required: full fixed condition100/graph/equal/sampling/trajectory
shortlist, corrected-fold historical fusion/localizer refits and declared
access panels. No new shortlist run was silently launched in this diagnostic
stage. The historical benchmark is NOT complete and no method is promoted.
The method registry marks only stage1 FULL_EVALUATION_REVIEWED.

**Step324 metric-gap diagnosis:** frozen-score inspection, no new fit.
Quantile IU vs full loses six correct PB decisions and gains none: four
correct peaks move to wrong steps, one correct peak is suppressed by its gate,
and one clean answer becomes a false alarm. Raw exact peaks20/53 ->16/53;
final correct error decisions12/53 ->7/53; clean decisions18/33 ->17/33.
With original full-IU gate and new peaks, PB is22.24% vs30.16% baseline
(native quantile20.77%). This is a post-hoc output swap, not a new candidate.
PRMB uses different answers; its within-answer gain does not imply better PB
ranking. PB-only first-error-versus-correct-prefix ranking falls .7094 ->.6911
on44 eligible erroneous answers (nine step-zero errors excluded; post-error
steps remain unlabeled). Do not dismiss PRMB gain as only calibration, and do
not attribute the whole PB loss to the gate. Continue full benchmark first.
See results/fusion_entropy_sampling_v1/METRIC_GAP_DIAGNOSTIC.html and JSON.

**Step324 COMPLETE - entropy sampling comparison; review PASS.**
`results/fusion_entropy_sampling_v1/REPORT.html` contains the matched35-row
comparison and31 registered paired intervals. Same110 development answers:
24 PRMBench/86 PB-Qwen3-8B;72 sampling eligible,38 exact full-window replays.
All176 prior entries remain unchanged;14 additions,190 displayed entries.
Run handle24353 is terminal, exit0. Scoring453.83s; no new inference.

IU pooled PRMB / within-answer PRMB / PB score:
full .68131 / .76881 /30.16%; uniform .67643 / .77697 /28.34%;
high-only .75835 / .77892 /28.35%; low/high .72746 / .77174 /26.27%;
entropy quantiles .71523 / .79198 /20.77%.
Joint condition100 graph0.1: full .65545 /30.22% (pooled/PB),
high-only .74133 /28.20%, low/high .71439 /26.36%, quantiles .67024 /24.37%.
All14 new outputs have24/24 PRMB and86/86 PB coverage INCLUDING IU fallback;
Joint fallback20/110 tails,19/110 quantiles (18/72 and17/72 among new fits).

Interpretation: neither new selector dominates the existing references.
Quantile IU improves within-answer AUC vs full by .02316 (exploratory paired
95% CI [.00213,.05232]), but PB drops9.38 percentage points
(CI [-17.51,-2.56] points). Its within-answer gain over high-only is uncertain.
Low/high tails do not beat high-only on either primary point metric for IU
or Joint graph. This does not prove high entropy identifies errors or resolve
normalization-versus-weight attribution. Keep fusion central and full matched
benchmark/refits first; no automatic new sweep or promoted publication winner.

Review:220 selections,1008 linear scores,1540 dense readouts,six representative
shared-kernel bank refits and14 metric bundles PASS. Additional explicit-pair
within-answer arithmetic for14 methods,35 HTML/CSV metric rows,31 interval rows
and six local links PASS. Same-session review, not external/browser validation.
Scientific driver/core/protocol stayed frozen. The priority HANDLE workaround
remains an operational launcher only. Full scoring handle50440 and evaluation handle65524 are terminal;
full anchor evaluation passed and historical comparisons remain pending.

**Sampling follow-up from Omri:** compare high/low-entropy fitting windows
(e.g.25+25) with high-only and full-range entropy-quantile selection, retaining
full/uniform controls and the same budget. These are unlabeled regions, not
hallucinated/correct classes. Two tails still omit the middle. This bounded comparison is now completed
in Step324; do not mutate or delay the running full benchmark for a new sweep.
See the decision note for normalization-versus-weight attribution.

**Parallel evidence inventory implemented:**
`results/localization_evidence_ledger_v1/CURRENT110.csv` and `CURRENT110.json`:
176 displayed current110 entries,167 saved-output fingerprints, nine identical-
output groups. Valid coverage and both PRMB metrics/PB outcomes preserved.
No new fitting, paired uncertainty or winner claimed. This is the first ledger
layer; earlier58 and historical/full comparisons remain unfinished.

## Omri decision update - 2026-09-07: reasoning benchmark first

The full matched REASONING benchmark and corrected historical-leader refits
are priority1 and a research-method requirement, not an optional final report.
Analyze existing experiments in parallel to identify real algorithmic gains
and application gains (local ranking, first-error/no-error decisions, coverage
and runtime). Do not open another sweep merely because a pooled AUC improved.
LOCA, Diverging Flows, KalmanNet and Shlezinger-inspired extensions are LOW
priority supporting backlog. Fusion remains central; answer-only fit is primary.

ProcessBench and PRMBench are the active core in separate metric panels.
RAG/grounding, claim/span and agent tasks are outside the active claim; preserve
historical artifacts. MR-GSM8K is the first external candidate to audit after
method lock, including GSM8K source overlap; ReTraceQA/GR-Ben are later transfer
candidates pending executable data contracts. A new scorer on previously seen
PB answers is not untouched confirmation. Preserve historical24 as a separate
later transfer with predeclared reasoning strata and its unchanged full macro.

Risk selection means highest mean-entropy8-token fitting windows, not known
errors: m=min(N,max(32,ceil(N/2))). Refit on selected rows and score ALL rows.
TOKENS and TRAJECTORY refer to the same observation axis: distinguish row
selection, chronological processing and combination of multiple fusion curves.

Reasoning, source conversation and initial gain analysis are recorded in
`docs/research_notes/reasoning_benchmark_decisions_2026-09-07.md`.
This amendment does not modify the frozen Step321 scoring protocol or run.

**Current priority from Omri, 2026-09-07: full matched benchmark before more small sweeps.**
The user requests a clear backlog, actual algorithm changes by FEATURES/TOKENS/
graph/trajectory, and comparison with historical leaders on full datasets.
Most recent method development used only24 PRMBench and86 PB-Qwen3-8B answers;
no consistent two-task winner is established. Corrections/bridges and controls
are not independent new algorithm discoveries. Fusion remains the core.

**Step321 scoring and Step325 full anchor evaluation COMPLETE:** `results/localization_full_benchmark_v3/`.
All13,769 model-answer rows registered: PRMB6969, PB3400 answers scored by
both Qwen3-4B/Qwen3-8B. All21 rows with<64tokens retained as unsupported;
11 PRMB traces>2048tokens included. The full cached population is exposed
DEVELOPMENT data, not untouched confirmation. Three preexisting PRMB
alignment exclusions remain declared. Exact source joins13769 and110 frozen
reuse inputs pass;13 representative answer replays x19 methods and two
short-case executions pass. First pass computes the UNCHANGED19 original
anchor outputs, not the newer full shortlist or historical refits.
No new inference/remote transfer/Claude-worktree mutation. Input memmaps
expanded1.623GB; three CPU workers, checkpoint/resume and eight-hour
submission cap. Driver and protocol are frozen by MANIFEST.json.
Scoring exec50440 and evaluation exec65524 are terminal, exit0. All13769
rows are frozen and evaluated with source-group intervals/review. Later
shortlist/historical comparator passes remain outstanding.

`METHOD_REGISTRY.json` explicitly keeps pending: condition100 graph and
lambda0/permuted/equal controls; risk fitting-row selection; trajectory mean/
GLS; corrected-fold IU/U-PCR, LIU/DUFS-LIU, CONT/L-SML, Claude Joint; dedicated
family6/GL-LIU/token-IU29/Unified28/entropy; CIW/DEEM and separate PRM/critic
access panels. The full historical comparison is NOT yet completed. Refit
old leaked-fold/label-selected methods; a score-only bridge is insufficient.
PB reports4B/8B/eight-cell panels, keeping repeated model rows together in
source-group uncertainty. PRMB reports pooled plus within-answer AUROC and
fold-aware comparisons; do not rank pooled scores from different OOF fits
as though their scales were calibrated.

**Step320 calibration simulation complete; review PASS.**
384 trials,39 calibration draws per fitted/known-rho model, raw/IMM readouts;
938.48s, all checkpoints frozen. Four preflight tests. This has changed NO
real-answer score or gate. Review handle95804 is terminal (exit0). Review PASS:30336 source paths,
384 lag regressions,60672 mixture likelihoods,1536 pvalue decisions,36 direct
vector-IMM replays and72 actual GMM refits. Neither readout passes its frozen
advancement screen: raw N64/rho.9 native11 -> fitted6 (required<=5.5); IMM
short-jump retention9/25 (required>=75%). No real-data gate change promoted. Full benchmarking supersedes launching another
small real-data gate experiment. Preserve its frozen protocol/driver/core.

Status HTML: `docs/reviews/localization_backlog_and_full_benchmark_2026-09-07.html`.
Full protocol: `docs/experiments/LOCALIZATION_FULL_BENCHMARK_V3.md`.
Backlog remains: full comparators first; then evidence-led fusion/gate fixes,
short-error and sparse sampling/N-versus-P, actual LOCA/Diverging Flows/
KalmanNet and Shlezinger-inspired support, untouched confirmation, separate
historical24 final-answer transfer. No current algorithm promoted.

**Step319 completed: controlled mixture-gate mechanism check.**
`results/fusion_gate_null_v1/REPORT.html`; scientific and artifact review PASS.
768 synthetic trials: N16/64/256, AR rho0/.6/.9,64 replicates per cell,
plus rho0/+3SD mean-jump controls. Raw, ordinary Kalman cold/warm and IMM
cold/warm;3840 valid outputs, no failures. Warm processes256 preceding
synthetic points and is a startup diagnostic, not a free-data candidate.
No current110 score/prediction/metric changed; all176 anchors remain current.

One stationary Gaussian source can trigger the gate after filtering:
N256/rho0 raw0/64, IMM cold26/64, warm26/64; N64/rho0 raw0, cold11, warm12.
The effect survives this startup control. N16/rho0 raw already opens15/64,
so short-sample behavior also matters. At N64 with the fixed mean jump,
raw opens21/64 while both IMM variants open64/64: preserve sensitivity in
any correction. These are simulation frequencies, not real-answer false-
positive rates. BIC has no declared5% alarm guarantee; nonlinear filtering
need not preserve a Gaussian marginal. No numerical GMM bug is claimed.

Three tests;768 source paths,1536 independent scalar Kalman trajectories,
3840 normalization/mixture-algebra checks,48 direct vector IMM trajectories
and120 actual GMM refits pass. Simulation106.04s, review12.06s. Artifact
validation1551 hashes,11 local links/images,137 static rows,five ASTs. Both
PNG plots inspected; no browser rendering or external review claimed.

Next bounded stage: verify one fixed calibration procedure on independent
synthetic calibration/evaluation replicates, carrying the source through the
SAME filtering/normalization/GMM procedure. Include mean-jump sensitivity
and nuisance-parameter uncertainty; do not treat an AR Gaussian source as a
proven model of correct reasoning. Then a gate-only current110 comparison
must keep frozen fusion curves/peaks and all176 anchors. A synthetic fix
alone is not an achieved localization improvement. The full research scope
remains active: Joint/feature/IU, named supporting methods, corrected multi-
answer refits, full comparators, sparse/short-error sampling, untouched
confirmation and historical24 transfer are still open.

**Step318 completed: full-trajectory fusion and supporting IMM.**
`results/fusion_trajectory_imm_v1/REPORT.html`; scientific review PASS.
Same110/v3 labels/v2 groups, original banks/fits/windows and149 unchanged
anchors;27 additions,176 total,50 contrasts. All final outputs valid110;
three inherited Joint fallbacks collapse to one IU observation in paired
families. No new inference, feature refit or causal-online claim.

Primary mean/GLS/IMM PRMB/PB: .669437/31.98298%, .683384/30.34837%,
.693254/16.69023%. Original IU .681306/30.15985%, Joint graph100
.655451/30.22293%; strong risk equal-permuted .768422/33.92003% retained.
GLS vsIU intervals[-.02612,+.02882] AUC/[-8.7608,+9.3277]pp include0.
No winner. Static mean/GLS are feature-weight combinations; IMM introduces
chronological state evolution, not semantic correct/error states.

Primary IMM vs static hold gains4 PB successes/loses13. Raw exact peaks
18->17, final exact errors12->7, clean successes15->11. Post-evaluation
fixed-component exchange: hold peak/hold gate30.34837%; hold peak/IMM gate
23.27899%; IMM peak/hold gate27.43170%; IMM peak/IMM gate16.69023%.
These are diagnostics, not new candidates or causal mediation. Clean median
lag1 rises .28649->.63670, median BIC1-BIC2 .95381->6.44959; false alarms
18->22. This motivates checking the mixture gate under dependence, but does
not establish a semantic-error null or prove an effective-N correction.

Seven tests,880 independent R/GLS and990 direct vector-IMM replays,2970
outputs,176 metric bundles,50 paired comparisons and five explicit1000-draw
bootstraps pass. Post-evaluation audit adds48 metric/component checks and1376
lag replays. Scoring52.80s; review113.13s. Same-session/shared-kernel scope
is disclosed. Full HTML, all comparisons and two inspected PNG/SVG plots
are saved; no browser rendering or external review claimed.

Artifact validation PASS:719 hashes,14 links/images,198 static rows,12 Node-DOM cases/256 numeric rows, seven ASTs and44 guide IDs. All handles terminal.

Next bounded priority: audit existing no-error methods and use controlled
unimodal serially dependent trajectories to check whether smoothing alone
can trigger this mixture gate. Then decide one gate-only comparison with
unchanged fusion trajectories/peaks. No broader temporal/graph sweep from
these point estimates. Full comparators, corrected multi-answer refits,
actual KalmanNet/LOCA/Flows, sparse/short-error sampling, untouched confirmation
and historical24 transfer remain open; full research goal active.

**LATEST FULLY PACKAGED — Step317: fixed-bank sampling replication.**
`results/fusion_sampling_replication_v1/REPORT.html`; REVIEW PASS.
Same current110, corrected v3 labels/v2 groups, original81/29 banks and
graph seeds. Six selectors x seven fusion cores;42 displayed entries include
seven exact full aliases.107 prior anchors retained:149 total,93 contrasts.
72 sampling-eligible (16 PRMB/56 PB);38 replay exactly. Dense feature and
score grids retained, GMM still fitted on ALL original fitting windows.
Only failed Joint model fits fall back to sampled IU in the SAME fixed bank.

No candidate promoted. Full IU .68130595/30.15985%; entropy-risk IU
.75835198/28.34853%. PRMB delta CI[+.046894,+.111884], within-answer
CI[-.003252,+.026904], PB[-7.987,+3.071]pp. Risk Joint graph
.74132832/28.20441% versus full .65545077/30.22293%. Risk IU gains one
correct PB decision/loses two; risk Joint gains two/loses three. Transposed
DUFS IU .65956682/29.68542%, Joint graph .62168318/28.06804%; no consistent
improvement. All ten sampled IU/Joint-graph PB points are below their full
references. The strong simple control, risk equal+permuted graph,
.76842231/33.92003%, exceeds its full .69225543/31.32177% points, but PB
improvement CI[-3.293,+8.029]pp includes0. Keep it visible, not promoted.

POST-EVALUATION affine diagnostic: preserve full IU ranking and use sampled
answer mean/SD -> pooled AUC .74876119; preserve sampled ranking but restore
full mean/SD -> .68937820. Corresponding graph values .73585358/.66564098.
Within-answer risk IU .77892319 vs .76881338; graph .75419150 vs .75348127.
This supports a largely between-answer location/scale explanation, not a
causal decomposition or a .758 first-error localizer. Diagnostic transforms
are not new candidates or PB predictions; original outputs remain frozen.

Native Joint full/uniform/risk/DUFS/permuted/diffusion counts107/101/90/84/
92/99 of110. All42 final recipes valid on110, with explicit fallbacks.
There are38 eligible PB error answers, ZERO first errors <=32 tokens.
Risk selection misses fitting-window overlap on one; other selectors hit
all38. Unselected windows still scored: this is not sparse-detector recall.
Mean perturbation Jaccard: risk .80135, DUFS .65488, diffusion .45314.

Six tests pass (first harness attempt used dict equality on arrays; corrected
the assertion before freezing). Review:110 raw labels/spans,330 raw streams,
360 selected normalizations/IU refits,289 covariance/Jacobian checks,720
Laplacians,2298 native weights,4620 final outputs,2520 dense GMM replays,
298 full/eligible metric bundles,93 paired scope/points,25 each representative
group/Joint/DUFS refits,10 perturbations and five explicit1000-draw bootstraps.
Max risk discrepancy3.55e-14. Scoring393.16s, contrasts45.93s, review153.31s.
All scientific process handles terminal. No new inference or external review.
HTML/artifact validation PASS:828 hashes,14 local links/images,119 static
rows,12 actual Node-DOM cases/358 rendered numeric rows, six Python ASTs and
43 unique guide IDs. Both PNG figures visually inspected; the selected-window
legend was moved below data. A validator-only JS/Python halfway-rounding
difference was handled using JS's toFixed contract; no score/report change.
Report SHA256:a23c6c50e27faf1ec108172a0b4a82691077f70fbdc26e0723eb4ae91bbfa52b.

**Next bounded stage:** audit existing whole-trajectory fusion combinations
and their fit scope, then test one fixed combination of current IU/Joint
trajectories with matched simple controls and separate peak/no-error checks.
Choosing among existing peak locations is insufficient (Step314); full
trajectories can be combined before step selection. Check the historical
implementations first; do not claim this is wholly new or repeat an existing
test unknowingly. Preserve the149 anchors, within-answer evidence and strong
risk equal-permuted control. No wider sampling/graph-dose sweep is justified
by pooled-AUC movement alone. Full comparators, corrected multi-answer refits,
named supporting tracks, short-error/sparse sampling, untouched confirmation
and historical24 transfer remain open. Full research goal active.

**Previous completed — Step316: earlier unique-method v3 score bridges.**
`results/localization_history_bridge_v3/REPORT.html`; REVIEW PASS.
Five original58 lanes repaired: representation19/readout42/sampling30/
regularization23/context17 =131 method entries and199 original contrasts.
Add25 already corrected fallback references (156 displayed early entries),
plus seven gate diagnostics. Current110/all107 entries stay in a separate
panel. These counts include repeated controls, not independent new methods.
All score arrays, fits, windows, predictions, failures and PB metrics exact;
11/12 PRMB targets and all58 group identities updated in each lane.

No hidden winner. IU peak .647527/17.70833%; HMM .560219/0% (10 valid PRMB),
ordinary Kalman .567033/5.35714%, IMM .612912/14.78697%, BOCPD
.581777/13.69048%. These are actual earlier supporting readouts, not
KalmanNet/LOCA/Flows. First crossing versus peak keeps the SAME IU scores
and PRMB AUC, while PB changes0->17.70833%; do not attribute that to weights.

Entropy-risk window selection +IU .668498/19.05242% versus full-grid
.647527/17.70833%. PRMB delta CI[-.008272,+.074768], PB[0,+8.929]pp;
within-answer AUC drops .703969->.697986. Four PB predictions change:
one correct lost, one gained; clean/error successes remain12/4. Equal-risk
has higher PRMB .679029 but PB11.12637%; permuted-DUFS IU has PB21.875%
but PRMB .644872. No consistent learned-fusion winner. Only37/58 sampling-
eligible (29 PB/8 PRMB), no <=32-token first errors among eligible PB;
dense features still computed. This is not sparse end-to-end scoring.

Joint+transposed-DUFS PRMB .763333 covers only4 answers; original Joint
on those SAME4 is .750000, not its .625788 over7. Difference CI includes0;
PB all46 .132576 versus .125000. Origin-offset diagnostic moves IU pooled
AUC .647527->.692399, within-answer remains .703969 and decisions unchanged.
The larger-lambda/regularization results establish no consistent gain.

Review:58 unique raw/official-label joins,290 exact row payloads,7598
original score/prediction records,131+25 metric bundles,199 scope/point
checks, five explicit1000-draw bootstraps,14 flattened AUC decompositions,
722 gate arrays and58 unchanged diagnostic rows. Twelve invalid diagnostic
scores remain saved but excluded (3 FINITE_UNCONVERGED_DESCRIPTIVE,
9 FIT_DIAGNOSTIC_ONLY); review assumptions were corrected, not source data.
Two WinError5 checkpoint writes recovered through a separate bounded writer
retry, preserving frozen math; all199 complete. Bridge5.84s, final recovery
13.00s (earlier partial contrast runs excluded), successful review25.88s.
All handles terminal. No new inference/refit, external review or winner.

**Next bounded stage:** observation-selection replication on the already
fixed current110: inspect target-free budget feasibility, retain original
feature banks/routes, full-grid, uniform, entropy-risk and graph/permutation
controls with IU/Joint and matched equal fusion. Report fit coverage, both
primary endpoints, within-answer ranking, exact success transitions and
short-error retention. Freeze a small design before scoring; keep107 anchors.
The early58 signal is weak and does not justify promotion or a large search.
Full comparator coverage, earliest short-cycle bridges, corrected multi-answer
refits, named supporting tracks, untouched confirmation and historical24
remain open. Full research goal active.

**Previous completed — Step315: provided-token preference gap inside fusion.**
`results/fusion_token_gap_v1/REPORT.html`; REVIEW PASS. Same110 answers,
v3 labels/v2 groups and original graph-seed identity. Replace only three
surprisal coordinates by gap=surprisal+top1_logprob; P27, width8, original
81 moment/29 context bank route. Seven fusion cores plus two scalar controls;
all98 corrected anchors retained,107 total rows and25 paired comparisons.
No new inference. This is existing-information reparameterization, not a
new independent observation or a novelty claim.

No candidate promoted. Gap-IU PRMB .68018702/PB28.82295% versus original
.68130595/30.15985%; within-answer AUC stays .76881338. Gap-Joint graph
.64993606/30.22293% versus original .65545077/30.22293%. Graph changes one
wrong PB prediction but preserves every exact-success indicator; IU loses
one correct decision and gains none. Both scalar controls flag all33 clean
PB answers and reach0% native PB F1; this concerns their fixed GMM recipes.
Keep original permuted equal-graph .69225543/31.32177% visible.

Native Joint100/110 versus107 original; ten explicit IU fallbacks (nine
no admissible partition, one fit-guard failure). Selected partitions69 K3,
32 K4 include one invalid fit. All nine final outputs have full coverage.
IU delta intervals: PRMB[-.005004,+.001971], PB[-5.265,0]pp. Graph delta
PRMB[-.032549,+.014061], PB[0,0]pp. Exploratory, unadjusted, not confirmation.

Raw source audit replays330 scalar and660 top-K streams, all110 raw labels.
71,057/71,385 provided tokens retained in top50 match their separately saved
logprob exactly.328 outside top50 remain independently unrecoverable from
that list.84.98% gaps near zero. Writer-source audit establishes same raw
distribution; no new forward/logit-position numeric verification.
Review:110 matrices/normalizations/IU refits,101 covariance/Jacobian checks,
220 Laplacians,740 native maps,990 step/gate/output replays,107 metrics,
25 point/scope checks, five each grouping/Joint/DUFS refits and explicit
1000-draw bootstraps. Max risk difference1.69e-14. An initial reviewer-only
official-port API error was fixed; frozen scientific sources unchanged.
Scoring140.80s, contrasts12.22s, successful review54.83s; handles terminal.

**Next bounded stage:** bridge earlier UNIQUE trajectory/readout, sampling,
regularization and representation score sets to v3 labels and v2 source
groups before using their PRMB conclusions for the next research choice.
Step313's98/25 anchors do not cover every unique older arm. Preserve old
artifacts and each cohort; do not pool unmatched populations. Corrected
multi-answer training/selection needs refits, not this score-only bridge.
Joint/graph, IU, both fusion axes, named supporting tracks, full comparators,
untouched confirmation and historical24 transfer remain open. Goal active.

**Previous completed — Step314: corrected-label text/peak forensics.**
`results/fusion_localization_forensics_v3/REPORT.html`; REVIEW PASS.
All110 raw text/token/span/label joins pass on v3:71385 tokens,1112 steps,
16 legitimate separator tokens outside official spans. All330 scalar-stream
matches and1760 frozen trajectory projections pass; max discrepancy2.67e-15.
This does not verify the original model logit-position slice or every top-K
derived feature. No new inference, fit, candidate or algorithm gain.

IU:20/53 raw first-error peaks,12/53 exact after its gate;8 exact peaks hidden.
18/33 clean decisions correct,15 false alarms;21/53 error gates closed.
Graph100:18/53 raw peaks,13 exact after gate;5 exact peaks hidden;17/33 clean
correct,16 false alarms,19 error gates closed. Both have16/86 PB answers
with shared-window top plateaus. Including all tied top steps raises raw
gold membership only20->22 for IU and18->21 for graph. Existing peaks miss
31/53 jointly; even the union of tied top sets misses28/53. This limits
choosing among those peaks, not full-trajectory fusion/new measurements.

LABEL-USING ORACLES ONLY: IU actual PB30.16%; perfect binary gate/same peak
56.74%; perfect locator/same gate47.23%; perfect top-tie choice/same gate
31.77%. Graph equivalents30.22/51.67/46.21/32.32%. These are not attainable
forecasts or new method scores. AR graph changes16 gates and13 peaks; only
2/13 changed peaks had old margin<=0.1SD. It loses10 correct decisions/gains2,
loses4 correct raw peaks/gains2. Near ties alone do not explain the regression.

Review re-reads raw metadata, uses existing independent alignment API and
incidence-matrix projection, checks98 independent metric bundles,48 oracle
metrics,16 native summaries and6 transitions. Shared tokenizer/source scores
and metadata reader disclosed. Audit19.14s, review33.11s; all handles terminal.
HTML explorer has all110 answers/16 method choices/4 case filters and all98
v3 anchors. Its actual JavaScript passes236 Node-DOM render checks;594 hashes
and five local links pass. No browser visual rendering.

**Next bounded work:** inspect whether the existing one-pass realized-token
confidence contains correctness evidence suppressed by uncertainty-oriented
fusion. Audit older confidence/innovation implementations before calling a
view new; then freeze ONE supporting feature/readout change with IU/Joint
and simple controls on the corrected contract. Do not prioritize boundary
tie-breaking or another broad lambda grid based on these limited gains.
Keep v3 labels, v2 groups/folds and original graph-seed namespace. Other
historical bridges and corrected-label/fold multi-answer refits, full
comparators, named supporting tracks, untouched confirmation and historical24
transfer remain open. Full research goal remains active.

**Previous completed — Step313: critical PRMB label correction and score bridge.**
`results/localization_prm_label_audit_v1/REPORT.html`; REVIEW PASS.
Use `results/localization_prm_label_audit_v1/RELEASE_V3.json` for new work:
`localization-cached-v3-prm-onebased-20260907`. V2 source groups/folds and
the original scoring/graph-seed namespace remain unchanged.

Direct raw-text/label forensics found Claude v2's label writer used
`flags[step]` for one-based PRMB `error_steps`; the correct index is step-1.
Recent Codex runs inherited the NPZ. The official source and existing local
metric port already subtract one. Earlier reviews checked derived labels,
which missed the raw contract error. Historical PRMB numbers below are
superseded; PB, score arrays, fitting/grouping diagnostics remain unchanged.

All6969 raw rows repaired and reviewed:6035 changed target arrays,
15147/94203 step labels change;12008 ->13149 positive steps.227 previously
all-correct arrays gain errors.151 out-of-range annotations across100 rows
remain inert, matching the official evaluator. All8 PB label files exact.
Frozen-score bridges: current110/all98 arms, original58/all25 arms;
175+32 registered comparisons,1000-draw exploratory source-group intervals.
No inference or refit. Current PRMB targets change16/24; old58 change11/12.

Corrected current110 PRMB/PB: routed IU .68130595/30.15985%; graph100 Joint
.65545077/30.22293%; AR+IU .66220428/23.49663%; AR+Joint graph
.64713875/18.01471%. Joint graph minus IU PRMB interval[-.06671,+.01573],
PB[-7.75,+8.18]pp: no proven winner and no longer a PRMB point tie.
Permuted-graph equal control .69225543/31.32177% exceeds IU on both points;
do not hide it or attribute this to meaningful graph alignment/learned Joint.
Withdraw the old all21-below-both-anchors claim:20/21 below IU on both points,
9/21 below graph100 on both; all21 still below both PB points. Last+equal+
permuted graph PRMB .69733056/PB25.45078%. AR-IU PRMB regression interval
now includes zero; AR-graph PB regression unchanged. No candidate promoted.

Five tests; review replays6969 raw/official-port labels,168 unchanged row
bundles,11594 method score/prediction records,123 independent metrics,
207 point/scope bundles and five explicit1000-draw bootstraps. Same-session
review, shared metadata reader/official port disclosed. Label build21.29s,
bridge8.67s,contrasts125.69s,review42.15s; process handles terminal.
Scoped inventory found old labels in13 compatible result JSONs. The98/25
bridges cover their shared anchors, not every earlier unique method.

**Historical next step, completed in Step314 above:** resume the interrupted text/alignment/peak audit using
v3 targets. `fusion_localization_forensics_v1` preserved its six tests,
frozen manifest,110 raw metadata records and FAILURE.json; its full AUDIT
did not complete. Make a new amendment/entry point, preserving frozen code.
Then use concrete error geometry to choose a fusion/readout improvement.
Other historical score bridges, Claude multi-answer corrected-label AND
corrected-fold refits, full comparators, supporting named tracks, untouched
confirmation and historical24 transfer stay open. Full goal remains active.

**Last updated: 2026-09-07. Fusion remains the core:** Omri explicitly wants
progress on our IU-PCR / Joint L-SML method. IMM, LOCA, Diverging Flows,
KalmanNet and other temporal/sampling ideas serve the fusion architecture.
Compare each fusion core with/without the addition and with a simple
aggregation control under the same addition. Do not pivot to a standalone
auxiliary detector or attribute its gain to fusion without an ablation.
Canonical instruction: `CLAUDE.md`; architecture and evidence ledger:
`docs/experiments/LOCALIZATION_RESEARCH_MANDATE_20260907.md`.

**Latest completed quality stage - prediction views inside fusion (Step 312):**
`results/fusion_prediction_quality_v1/REPORT.html`. All 98 arms (77 exact
external historical references + 21 augmented recipes), 74 registered
comparisons and review complete on the SAME 110 development answers.
All three residual families (AR1/last/EMA32) use the same original bank per
answer: 81 moment, 29 context. Seven cores per family: equal/IU/Joint0/
Joint graph/permutation/equal graph/equal permutation, native condition100.
No new inference. Fusion remains central; no candidate is promoted.

Every new Joint fit is valid: 110/110 per family, versus 107/110 original
selected-bank Joint fits. No IU fallback used; all21 new heads have full
score/decision coverage. AR K3/4/6/8 counts34/32/39/5 (76 answers with K>3).
Improved admissibility and stability did NOT improve localization. All21
new recipes are below both original dual IU and graph100 Joint on both
primary point metrics. This does not close the Joint/graph family.

Original dual IU: PRMB .63797468 / PB30.15985%; graph100 Joint .63847197 /
30.22293%. AR+IU: .61333635 /23.49663%; AR+Joint graph: .62355335 /18.01471%.
AR-IU delta CI: PRMB[-.05027,-.00477], PB[-16.424,+1.426]pp. AR-graph versus
old graph100: PRMB[-.03538,+.00334], PB[-23.824,-.764]pp. Restricted107
both-native-fit-valid PB drop is still -12.555pp CI[-24.466,-1.072]. These
are74-comparison unadjusted exploratory intervals, not confirmation.
Last+IU .62490958 /29.73661%; EMA32+Joint graph .62066004 /28.38778%, also
below the original references. AR graph-versus-zero/permutation and
graph-versus-matched equal-graph do not establish a two-task advantage.

PRMB within-answer AUC and fixed-original-IU-gate PB also weaken with AR.
Raw PB first-error peaks: original IU20/53, original graph18/53, AR IU16/53,
AR graph16/53. Correct clean decisions18/33,17/33,13/33,12/33 respectively.
POST-EVALUATION descriptive diagnostics: original feature signs change in
40/43/42 answers (AR/last/EMA), despite exact preservation of raw27 columns;
new/old graph trajectory median Pearson remains .98966/.98892/.98772.
The added columns take about24–26% of absolute standardized weight, not a
causal importance measure. These observations do not prove the loss's cause.

Three tests and review PASS: 8470 exact parent method rows,330 original
matrix/normalization/covariance/Jacobian/IU/graph checks,660 Laplacians,
2310 weight/step/GMM/fallback paths,98 metrics,74 paired point/scope bundles,
six explicit1000-draw interval checks;15 each representative grouping,
Joint and DUFS refits. Max risk discrepancy2.26e-14. Scoring359.84s,
contrasts40.89s, review106.96s; handles terminal. Two review-harness issues
were fixed (nested-record comparison and independent floating-sum tolerance);
frozen scientific sources and predictions were unchanged. All98/74 tables,
14 local links/images and exported figure checked; no browser rendering.

**Next bounded work:** inspect actual benchmark first-error spans/text,
near-tied peaks and clean-decision changes with the frozen original and
augmented scores. Identify a concrete fusion/readout failure before another
feature/predictor sweep. The Step311 prediction-MSE/feasibility gain did not
predict quality. Keep IU/Joint graph anchors and all broader named tracks,
corrected-fold refits, full comparators, untouched confirmation and
historical24 transfer open. The full research goal remains active.

**Previous stage - prediction-view feasibility audit (Step 311):**
`results/fusion_prediction_view_audit_v1/REPORT.html`. This is a history audit
and target-free feature prototype, NOT a new localization quality result.
IU-PCR / Joint L-SML remains the core; the predictor only appends measurements.
All 110 existing development answers / 71,385 tokens are complete and reviewed.

The earlier innovation search was incomplete. Token B3 already implements
lagged innovations with donor-question fitting; Local/Online IU stacks
calibration answers and tests EMA innovations; CIW uses cross-answer fitting
and whole-answer predictors. Do not claim only scalar final-answer variants
existed. The scoped local filename search found no matching named B3 Phase-2
score freeze/evaluation, which is not proof it never ran elsewhere. Exact
code/protocol boundaries are in the new audit protocol.

The new small AR(1) predictor uses only preceding pairs in the same answer,
shrunk toward the last observation with fixed pseudocount16. Last-value and
EMA32 are controls. Append nine mean-absolute-residual columns to moment27
or context27, retaining the original 27 exactly. First token is masked, so
first-window residual support is seven rather than eight. Full-answer
fusion would remain offline. This is not learned KalmanNet or a flow.

All nine extra columns vary in all 110 answers. AR/EMA32 entropy prediction
MSE median is 0.97367, with 78/110 wins; top1/tail-mass medians exceed one.
Prediction MSE is not localization accuracy. Median closest-original-column
absolute Spearman for AR residuals is 0.87072 (moment) / 0.83750 (context).
They remain substantially redundant; EMA32 is less redundant than AR for
moment. 40/110 have fewer fitting windows than the raw 36 columns, and 31
original centered matrices already saturate N-1. These do not establish
Joint fit failure or useful new correctness information.

Three tests PASS. Independent batch reconstruction reviews all 2,970
predictor/stream traces, 330 residual/MSE arrays, 660 exact original-column
replays and 660 correlation/rank bundles; maximum prediction discrepancy
1.35e-13. Target-free audit 10.34 s, review 11.74 s, all handles terminal.
HTML structure/nine local links pass; no browser visual inspection. No new
fusion fit, correctness-label evaluation, inference or Claude worktree edit.

**Quality follow-up, now completed in Step312 above:** do these extra columns improve the SAME
IU/Joint cores beyond matched equal aggregation and simple residual controls?
Freeze the small roster, original-bank routing and augmented-fit fallback
before evaluation. Keep AR/last-value/EMA controls, graph-zero/permutation
controls, unchanged historical anchors, coverage, both primary endpoints,
within-answer ranking and source-group paired uncertainty. Do not select
streams or a winner from prediction MSE alone. Step310 supplies the unchanged
quality anchors. Broader tracks and the full active goal remain open.

**Previous quality stage - graph/conditioning interaction (Step 310):**
`results/fusion_graph_conditioning_v1/REPORT.html`. All 77 arms (45 exact
preceding anchors, 24 native graph heads, eight equal-graph controls), 101
registered comparisons and independent review are complete on the same 110
development answers. Original C/v/u, groups, fit validity and bank routes
stay fixed; no Joint refit. All three caps 30/100/300 remain, with graph .1
and its original node permutation. Condition1000 graph scores/decisions replay.

Dual Joint graph at condition100 gives PRMB 0.63847 / PB 30.22%, versus
dual IU 0.63797 / 30.16%. The deltas are only +0.000497 AUC and +0.063
PB points, with CIs [-0.03409,+0.02323] and [-7.75,+8.18] points.
This tiny point lead does not establish superiority or an optimal cap.
Condition30 graph gives 0.63838/29.45%, condition300 0.63684/29.94%.
Original condition1000 graph remains 0.63350/29.94%. Keep all as frozen
references; no consistent two-task winner has been established.

At cap100 the real graph adds +4.75 PB points over both no graph and its
permutation, CI [0,+10.37]. At cap300 it beats permutation by +5.22 points,
CI [+0.61,+11.28], while PRMB's interval includes zero. These are 101-
comparison unadjusted exploratory intervals, not confirmation. Native
Joint versus matched equal-graph also has both primary intervals crossing
zero. Pure context graph100 loses to equal on the SAME 21 PRMB answers:
0.68452 vs 0.69658, delta CI [-0.02432,-0.00124].

The equal-graph adaptation uses identity covariance and uniform loading
with the SAME trace-matched graph inverse. Lambda0 replays equal fusion.
Its condition is <=3.18289, below all tested caps; all caps give the same
control. Dual equal real/permuted graph is 0.62993/26.62% versus
0.62170/31.32%; the graph does not improve every core uniformly. Existing
same-answer gates replay in 182 banks; 38 are computed for simple controls.

POST-EVALUATION label-using overlap diagnostic: IU/Joint peaks both hit
16 of 53 erroneous PB answers; IU alone hits four, Joint alone two,
neither 31. Joint hit sets are identical across all four caps. A chooser
restricted to those existing peak locations can cover only 22/53. This
is NOT a ceiling for combining full trajectories, reranking or new views.
Graph100 and IU each get 30 total PB answers correct, but trade clean
versus error success (17/13 versus 18/12) and subset allocation. The tiny
macro-F1 lead is not uniform improvement. IU has higher within-answer AUC
(0.67470 vs 0.66244) and fixed-IU PB (30.37% vs 29.53%).

Three scientific tests and review PASS. Checks: 110 label/group joins,
11,351 exact parent arrays, 4,950 metadata records, 220 source graph and
zero-graph equal replays, 440 Laplacians/control-cap checks, 1,880 inverse/
step/GMM reconstructions, 1,760 route inheritances, all 77 metrics and
101 paired point bundles. Seven explicit 1,000-draw bootstraps match all
four intervals/counts; ten new-control gate recipes receive representative
refits. Source graph-builder/DUFS/GMM kernels reused. Maximum risk difference
5.05e-14. Scoring 51.35 s, contrasts 50.53 s, review 68.70 s; all handles
terminal. HTML structure/12 local links and separate Boolean overlap recheck
pass; no browser visual inspection. No new inference or Claude worktree edit.

**Follow-up status (feasibility complete above; quality remains):** keep IU and original/conditioned Joint graphs
as references; the expanded innovation-history audit is complete; test one
same-answer prediction-residual view inside the existing fusion matrix.
Require unchanged-core and matched equal controls, and evidence of added
information beyond entropy. Old scalar final-answer innovations were
highly correlated with entropy; do not rename that precursor as KalmanNet
or Diverging Flows. The shared misses motivate complementary information
or improved full-trajectory readout, not an assertion that existing
features contain no useful signal. Avoid widening the same dose grid to
chase the tiny lead. Broader supporting tracks, corrected-fold multi-answer
refits, full comparators, untouched confirmation and historical24 stay open.

**Previous stage - original Joint native conditioning (Step 309):**
`results/fusion_native_conditioning_v1/REPORT.html`. All 45 arms (33 exact
preceding anchors plus 12 native-condition heads), 69 registered comparisons
and independent review complete on the SAME 110 development answers. New
targets 30/100/300 keep graph lambda zero, original minimum-three groups,
fit validity and bank routing. Original C/v were not saved; all 180 valid
fits were reproduced on their original matrix/partition, with condition-
1000 weights/scores/native decisions replaying before new scoring. Those
C/v/u arrays are now saved. Forty invalid bank fits stay invalid.

Dual Joint at condition 30 improves points from original Joint0's PRMB
0.62884 / PB 25.33% to 0.63639 / 28.43%. Paired changes are +0.00755
AUC, CI [-0.00062,+0.01777], and +3.10 PB points, CI [-2.58,+9.74].
Both include zero. Dual IU remains higher, 0.63797 / 30.16%; original
dual graph0.1 is 0.63350 / 29.94%. Matched dual-IU/equal intervals include
zero. Condition 30 is the strongest tested dose, not a proven optimum.
No consistent two-task winner is promoted.

Dual condition300/100 gives 0.63079/25.47% and 0.63296/25.47%.
Single condition30 gives 0.60154/23.87%, versus original 0.59733/21.45%
and moment IU 0.60140/26.38%. All original routes remain fixed: single
78 moment Joint/32 IU; dual 78 moment Joint/29 context Joint/three IU.
Pure moment/context coverage stays 78/102. Context condition30's unmatched
0.68799 AUC (21 valid PRMB) appears above context equal's 0.66971 (24),
but on the SAME 21, equal is 0.69658. Joint-minus-equal is -0.00859,
CI [-0.01748,-0.00100]. Preserve common-ID comparisons.

Dual condition30 gets 16 clean/12 exact errors right, versus original
14/11; its raw-peak count stays 17/53 (not necessarily identical answers).
With a fixed-IU gate its PB falls 29.25% -> 28.73%. Within-answer AUC
rises 0.63034 -> 0.67117, but the paired interval includes zero. Do not
attribute the native PB increase entirely to improved peak localization.

Three pre-freeze tests and review PASS. Checks: 110 direct label/group
joins, 8,411 exact parent arrays, 3,630 parent metadata records, 180
covariance constructions, 720 independent inverse/step/GMM reconstructions,
660 fixed-route inheritances, 45 metrics and 69 paired point bundles. Ten
representative original refits and six explicit 1,000-draw bootstraps match.
Original optimizer/GMM kernels reused; independent algebra remains separate.
Review's initial 1e-11 refit difference came from normalization reduction
order; exact source-recipe replay fixed it without changing frozen sources,
scores or tolerances. Maximum independent risk difference 5.60e-14.
Scoring 63.11 s (three CPUs), contrasts 33.44 s, review 32.12 s; all handles
terminal. HTML structure, two SVG charts and 11 local links pass; no browser
visual inspection. No new inference, Claude edit or untouched-test claim.

**Follow-up completed in Step 310 above:** test interaction of the SAME three condition
caps with the existing graph0.1 and permutation control, holding original
C/v, groups, fit validity and routes fixed. Reuse saved original fits and
same-answer DUFS gates. This tests graph structure beyond generic inverse
regularization; no wider K/graph-dose search or label-chosen best cap.
The wider fusion/supporting tracks, corrected-fold multi-answer refits,
full comparators, untouched confirmation and historical24 remain open.

**Previous stage - checked-pair localization quality (Step 308):**
`results/fusion_pair_quality_v1/REPORT.html`. All 33 arms (19 unchanged
anchors plus 14 checked-pair extensions), 63 registered paired comparisons
and independent review are complete on the SAME 110 development answers.
The scorer uses `joint_pair_jacobian.fit_joint_pairs_checked`; predictions
were frozen before this evaluator read labels. No new inference. The
previously evaluated cache remains development, not untouched confirmation.

Greater fit coverage did not improve localization. The old/new dual Joint
graph gives PRMB AUC 0.63350 -> 0.58974 and PB F1 29.94% -> 19.29%.
Paired deltas are -0.04376 AUC, CI [-0.08151,-0.01527], and -10.65 PB
points, CI [-21.04,-1.51]. These are exploratory, unadjusted intervals;
within-answer AUC and fixed-IU-gate diagnostic intervals include zero.
Single Joint0 -> IU is closer: 0.59733/21.45% -> 0.59742/19.18%, without
a consistent gain. Do not promote the new pair-routing policy or close
the Joint/graph family on this result.

The dual route changes from 78 moment Joint / 29 context Joint / three IU
to 106 moment Joint / four context Joint / zero IU. There are 31 bank
changes: 28 context -> moment and three reverse. Three further answers
switch moment IU -> moment Joint without changing bank. The matched dual
IU control uses EXACTLY the same per-bank IU maps, yet falls from
0.63797/30.16% to 0.59222/26.10%. Fit eligibility is insufficient for
choosing the useful bank. Joint changes also involve refitted weights;
do not attribute all of its regression to routing alone.

Pure coverage is 106 moment / 108 context, versus 78/102. Do not compare
unmatched pure PRMB AUC as a ranking gain or loss: context Joint0's full
AUC falls 0.68397 (21 valid answers) -> 0.64353 (23), but its paired
change on 20 common answers is +0.00109 and within-answer change is zero.
Its full-population PB improves +3.92 points, CI [0,+11.50], without a
two-task win. Moment Joint0 common-ID/PB intervals also include zero.

Two new scientific tests pass, supplementing the 11 preceding pair tests.
Review independently checks 110 label/group joins, 4,710 exact parent
arrays, 2,090 parent metadata records, 220 audit groupings, 219 factor/
covariance replays, 67 product Jacobians, 642 native/graph inverse maps,
step maps and GMM decisions, 880 route inheritances, all 33 metrics and
63 paired point bundles. Six explicit 1,000-draw bootstraps match all four
intervals and defined counts. Original Joint/DUFS/graph-builder kernels
were reused. Max reconstructed risk difference 5.60e-14. Scoring 157.49 s
on three CPUs; contrasts 31.31 s; review 36.50 s. All handles terminal.
HTML structure/11 local links pass; browser visual inspection not run.

**Follow-up completed in Step 309 above:** hold the ORIGINAL minimum-three Joint fits,
feature groups and bank routing fixed; test stronger native inverse
conditioning, starting with lambda zero and retaining graph/IU/equal
anchors. A post-evaluation unlabeled diagnostic finds condition >=999
in 104/106 valid pair-moment maps and 97/108 pair-context maps; common
old/new median conditions are both 1000. This motivates a test, not a
causal explanation of errors. Do not combine new pair routing, wider K
and graph doses in that first test. Fusion remains the contribution.
Corrected-fold multi-answer refits, wider supporting tracks, comparator
coverage, untouched confirmation and historical24 transfer remain open.

**Previous stage - Joint pair-group audit (2026-09-07, Step 307):**
`results/joint_pair_identifiability_audit_v1/REPORT.html`. Implemented a
pair-aware native Joint covariance extension and audited both existing
feature banks on the same 110 answers without decoding error labels. Eight
original scientific tests and three review-amendment tests pass. Independent
review and the HTML report are complete; no quality scores are claimed here.

Claude's claim that a two-feature residual system uniquely determines its
loadings is too strong: it fixes u_i*u_j, not u_i and u_j separately. A
synthetic scale-family counterexample preserves the off-diagonal objective
and passes the profiled-global Jacobian while the legacy clipped diagonal
and native weights change. Simply lowering the minimum is unsafe. The new
construction chooses equal fractions of residual variance budgets, retaining
the product and observed pair diagonal when feasible. It rejects infeasible
pairs, preserves minimum-three behavior and checks native covariance/map
agreement across converged starts. This is a parameterization, not identified
latent pair loadings; it does not automatically validate a hierarchical head.

On matched inputs, moment Joint valid coverage rises 78 -> 106 of 110
(31 rescued, three lost); context 102 -> 108 (seven rescued, one lost).
The union rises 107 -> 110. Moment selects K=3/4/6/8 in 24/26/51/9
answers, versus only 3/4 previously. Context still chooses 3/4. There are
55 valid moment and 12 valid context fits with actual pair groups. Relaxing
held-block minimum size can also admit a final partition without pairs.
One moment pair is infeasible (residual 0.0145291 exceeds capacity 0.00797371);
five other fits fail convergence/multistart guards. Coverage is not accuracy.

Review found a second edge case: at u_i=u_j=0, the old factor-coordinate
Jacobian loses the pair-product nuisance direction and can falsely pass
global identifiability. A separately frozen POST-FIT, UNLABELED review
amendment introduces product-coordinate profiling. Its counterexample now
fails as required. All 219 existing fitted records pass the eligibility
recheck unchanged; 69 pair fits receive independent product-profile checks.
No current pair product is exactly zero. Canonical future entry point:
`spectral_utils.joint_pair_jacobian.fit_joint_pairs_checked`; do not call
the earlier `joint_pair_extension.fit_joint_pairs` prototype alone in a new
quality experiment. The original audit sources and arrays remain frozen.

Independent review checks 220 normalized covariances, 880 candidate guards/
ARI summaries, 220 selections, 219 native inverse maps, 107 pair products/
variance allocations, the infeasible pair and 117 unchanged-partition
parent maps. Nine representative pair refits match (optimizer/clustering
labels reused). Max independent weight difference 1.27e-13. Runtime:
130.96 s on three CPU workers; final review 8.44 s. All handles terminal.
HTML structure, 12 local links and interactive-JS syntax pass; no browser
visual inspection. No new inference, benchmark-label decoding or Claude edit.

**Follow-up now completed in Step 308 above:** the checked pair-enabled native Joint
quality comparison with lambda 0 / graph 0.1 / permuted graph, the frozen
minimum-three versions, moment/dual IU, both equal banks and matched routing
controls. Hold other hyperparameters fixed; retain pure failures, common-ID
PRMB ranking, all-population PB decisions and native/fixed-IU diagnostics.
Only then decide whether the structural gain helps localization. Corrected-
fold multi-answer refits, broader fusion/supporting tracks, full comparator
coverage, untouched confirmation and historical24 transfer remain open.

**Previous stage - fixed fusion replication (2026-09-07, Step 306):**
`results/fusion_replication_v1/REPORT.html`. All 19 frozen recipes, 38
registered paired comparisons and independent review are complete on 110
additional source-question groups: 24 PRMB, 16 GSM8K, 24 MATH, 22 OlympiadBench
and 24 Omni-MATH answers. Selection used corrected v2 groups, excluding the
documented 94-component Codex inventory, with no label/fit selection. The
whole cache was previously evaluated, so this remains development evidence.
The old scoring namespace preserves graph-permutation seeds. No new inference.

The earlier single Joint0 -> IU gain did NOT recur: new PRMB AUC / PB F1 is
0.59733 / 21.45%, versus moment IU 0.60140 / 26.38%. Dual IU gives
0.63797 / 30.16%, dual Joint0 0.62884 / 25.33%, dual Joint graph0.1
0.63350 / 29.94%, and matched dual equal 0.62649 / 25.55%. Dual IU chooses
context only when moment Joint is invalid and context Joint is valid;
otherwise it uses moment features. It still pays for Joint eligibility.
Its paired deltas versus moment IU are +0.03657 AUC, CI [0.00764,0.06898],
and +3.78 PB points, CI [-2.64,+10.61]. Within-answer AUC changes only
0.67110 -> 0.67470, with a difference interval including zero. Its matched
equal-control intervals also include zero. No consistent winner is established.

Dual graph exceeds dual Joint0 at the point-estimate level, but both
intervals include zero. Versus its permuted-graph control, PB delta +4.21
points has CI [+0.17,+9.58], while the PRMB interval includes zero. These
38-comparison, unadjusted development intervals do not establish superiority
over Joint0 or IU. Keep this graph signal visible without promoting it.

Pure moment Joint covers 78/110; context Joint 102/110 (29 rescued, five
lost). Dual routes 78 moment Joint / 29 context Joint / three IU; all
composites cover 110. Both banks have one unconverged/multistart-blocked fit;
the remaining failures are inadmissible partitions. On 21 common PRMB IDs,
context Joint0 is LOWER than context equal: 0.68397 vs 0.69658, despite
the misleading ordering of their unmatched full-row values. PB has 33 clean
and 53 erroneous answers. Moment IU gets 15 clean/11 exact errors right,
dual IU 18/12, dual graph 17/13. Dual IU's Omni-MATH exact-error hits fall
six -> five, offset by improved clean detection; do not call every component
uniformly better. Native gates and fixed-IU diagnostics remain separate.

Four scientific tests passed before freeze. Independent review verifies
110 raw/span/label joins, 220 feature banks and normalizations, 980 weight
projections, 1,970 step maps, 1,090 GMM decisions, 880 fallback inheritances,
19 metric/history bundles and all 38 paired point bundles. Five explicit
1,000-draw bootstraps match all four intervals and defined-draw counts.
Five representative Joint refits reproduce 15 native inverse heads, reusing
the original optimizer/graph-builder kernels. Scoring took 213.06 s on
three CPU workers, contrasts 18.46 s, final review 40.52 s. Score, contrast
and review handles are terminal. HTML structure/11 local links pass;
browser visual inspection was not run. No Claude worktree edit.

The minimum-FEATURE-group-size-two audit proposed after this replication
is now complete in Step 307 above. Preserve the frozen moment/dual IU and
Joint/graph/equal anchors; do not retune those recipes on the completed
replication. Broader research remains active, as listed above.

**Previous stage - source-question grouping correction (2026-09-07):**
`results/localization_source_group_audit_v1/REPORT.html`. The exposure audit
found a benchmark integrity problem before a new replication cohort was
selected. PRMB's 6,211 old groups were perturbation IDs, not source-question
families: 6,969 answers map to 758 source seeds and 707 connected components
using seed identity and exact whitespace-normalized question text. ALL 707
components cross Claude v2 outer folds. Independently, 812 identical question
text hashes cross those folds, without relying on source-seed interpretation.
PB has 3,400 answers / 2,842 distinct
question texts, including 430 repeated-question groups (988 answers), of
which 363 cross old outer folds. No PB question crosses its four subsets.
There are 66 exact question hashes shared between PB and PRMB.

A new immutable release and global five-outer/five-inner folds are prepared:
`results/localization_source_group_audit_v1/RELEASE_V2.json` and
`FOLDS_V2.json`, release `localization-cached-v2-sourcegroups-20260907`.
Use corrected group IDs when loading rows; merely replacing the old folds
while retaining old NPZ group IDs will not work. Shared source questions
stay aligned across tasks and 4b/8b models. Preserve the old scoring identity
namespace `localization-cached-v1-20260907` when replaying the fixed fusion
recipes, so this grouping-only correction does not change graph-permutation
seeds. The v2 release records that namespace as its predecessor release ID.

The 58-answer pilot contains 12 PRMB answers but 11 corrected groups, with
two source groups overlapping earlier short-cycle cohorts. All 46 PB pilot
groups remain distinct. An exclusion inventory covers 94 corrected components
from documented Codex short-cycle/58-answer cohorts; it is NOT the project's
complete historical exposure inventory. The whole cached v2 population was
already evaluated by Claude, so disjoint new pilot IDs do not make it untouched.

All 25 fallback-pilot metric bundles, scores, targets and decisions replay
unchanged. Thirty-two corrected interval bridges are complete. Joint0 -> IU
versus IU remains PRMB +0.01402 and PB +9.51 points; corrected intervals are
[-0.01264,+0.02729] and [-9.21,+20.72] points, both including zero. Earlier
v1 group-based intervals below are historical; do not use them as corrected
source-question uncertainty. Older answer-only experiment intervals beyond
this 25-arm bridge remain to be recomputed when used. No winner is established.

Claude's multi-answer fits/inner selections/OOF scores have NOT been repaired
by this metadata change. They require refitting on corrected groups and folds
before source-question-disjoint performance can be claimed. The direction or
amount of any bias is unmeasured. Do not call all their numeric scores wrong;
the verified problem is the claimed source-question isolation.

Two metadata-decoder tests pass. An independent sparse-graph review verifies
10,369 distinct answer metadata records, nine source-pickle hashes, all 13,769
release rows (including both PB scoring models), original telemetry/label
hashes, 30 global fold-isolation checks, 58 score/target/decision replays and
all 25 metrics. Three explicit 1,000-draw bootstraps match all four intervals
and defined-draw counts. Review PASS; score bridge and all audit/review tool
handles are terminal. No new inference or Claude worktree edit.

Access wording correction: both localization caches use ONE TEACHER-FORCED
MODEL PASS over fixed official answers. They are not newly generated answers.
Gray-box, one-pass scoring and answer-only fusion remain the actual contract.

The development replication proposed after this correction is now complete
in Step 306 above, using the v1 scoring namespace. Its scorer is
`spectral_utils/fusion_replication.py`, with cohort, driver, predictions,
evaluation and reviewed reports in `results/fusion_replication_v1/`.

Also queue relevant Claude contenders for a corrected-fold rerun. His latest
report proposes minimum feature-group size two; audit that identifiability
claim independently before changing the current minimum-three recipe. It
concerns FEATURE groups, separate from the benchmark question groups fixed
here. Remaining Joint features/graphs, IU, supporting temporal/geometry/
sampling work, the full comparator registry, untouched confirmation and
historical 24-cell transfer remain open. The full goal stays active.

**Previous stage - explicit Joint / IU fallback (2026-09-07):**
`results/fusion_explicit_fallback_pilot_v1/REPORT.html`. Eight new composites,
all 17 context-pilot anchors and 30 registered paired contrasts are complete
on the same 58 development answers. Five scientific tests and independent
review pass. The score phase is terminal; the first contrast process (65828)
exited with a Windows checkpoint replacement error. The identical frozen
runner resumed (98423, exit 0), and all 30 contrasts are reviewed. No live
experiment process remains from this stage; no inference or Claude edit.

The single policy uses original moment Joint when its fit is valid and
moment IU otherwise. The dual policy tries context Joint between those two.
Routing uses fit validity only, never labels, scores or no-error decisions.
A selected readout failure stays a failure. Both policies and their graph
lambda-zero/0.1/permuted variants now cover all 58 answers; pure Joint
failures remain visible. Single routes 43 Joint / 15 IU; dual routes 43
original Joint / 13 context Joint / two IU. Routed equal/IU controls use the
same bank eligibility and therefore still pay for that calculation.

Full-coverage PRMB AUC / PB macro F1: original IU 0.62261 / 17.71%; single
Joint0 -> IU 0.63663 / 27.22%; single graph0.1 -> IU 0.63478 / 14.94%; dual
Joint0 0.62859 / 27.22%. Original Joint -> IU has higher two-task points than
IU but its paired intervals include zero: PRMB [-0.00769,+0.02696], PB
[-9.37,+20.50] percentage points. Within-answer AUC: IU 0.67567, single
Joint0 0.68798, dual Joint0 0.69614. No confirmed winner.

Do not omit always-context equal fusion: it gives PRMB 0.66087 / PB 27.67%,
higher headline points than single Joint -> IU (but lower within-answer AUC,
0.66182). Review identified this missing paired comparator and added TWO
explicitly post-evaluation comparisons in `ADDITIONAL_COMPARISONS.json`;
the original frozen protocol and 30-pair roster are unchanged. Neither
post-evaluation headline interval establishes a difference.

PB improvement is not uniform: single Joint -> IU gets 8/21 clean and 6/25
exact errors right, versus IU's 12/21 and 4/25. Total exact successes fall
16 -> 14 while the registered subset-balanced harmonic macro improves.
GSM8K/Olympiad improve, MATH ties, Omni-MATH worsens. Dual changes four PB
predictions and three peaks per Joint variant, but no answer's exact-success
status, explaining its zero PB difference interval. Graph adds no proven gain.

Independent review verifies 58 label joins and route pairs, 1,450 source
metadata records, 2,624 exact score-array copies, 1,312 peaks/common gates,
all 25 metric bundles, all 17 parent replays and all 30 paired point bundles.
Three explicit 1,000-draw bootstraps match all FOUR endpoint intervals and
valid-draw counts. The review reuses previously audited fits; no refit claim.
Cached composition took 2.84 s on one CPU, not end-to-end fitting latency.

The exposure audit proposed after the fallback stage is complete above. It
changed the next action: use corrected question groups and folds before
selecting the disjoint development replication. No new cohort was selected.

**Previous stage - context feature-bank / Joint-K pilot (2026-09-07):**
`results/fusion_context_bank_pilot_v1/REPORT.html`. All 17 arms, 31 paired
contrasts and independent review are complete on the same 58 development
answers. Six scientific tests pass. Scoring (session 86145), contrasts (14121),
core review (75109) and bootstrap review (17650) are terminal. No active
experiment process remains from this stage. No inference or Claude worktree edit.

The new bank retains the nine primitive streams and 27 coordinates, replacing
window SD/slope with window-averaged EMA8/EMA32. EMAs start from the first
observed value. Grouping compares legacy K={3,4,6,8} with all feasible K=3..9;
signs, normalization and fits remain within the answer, with the declared
negative-entropy anchor. Historical supervised-developed DSP artifacts were
not imported. This is trajectory context before feature fusion, not a newly
learned second trajectory-fusion stage.

Joint coverage rises 43 -> 50 of 58, but this comprises 13 rescued and six
lost fits. On PRMB, old 7 and new 9 valid answers share only FOUR. Their Joint
lambda-zero AUCs on those four are 0.72613 (moment) vs 0.66823 (context),
difference -0.05790, exploratory CI [-0.28571,-0.01170]. Do not compare the
unmatched full-coverage AUROCs as evidence of improvement. PB uses the same
46 answers: Joint lambda-zero 12.50% -> 27.43%, graph 8.33% -> 20.19%; equal
17.76% -> 27.67%, IU 17.71% -> 16.97%. No consistent two-task winner or
learned-fusion advantage is established.

K=5 is selected in one moment and three context cases; PB endpoints are
unchanged by expanding K. Context Joint AUC shifts 0.63826 -> 0.64808 on its
same nine valid PRMB answers. Mean participation rank decreases 3.64 -> 2.29;
that is covariance concentration, not an independent-feature count. Active
P remains 27; N_fit ranges 13-176. A common parent-IU binary gate for every
core is a diagnostic distinct from each method's own old gate.

Independent review verifies 58 raw EMA banks, 58 exact level-column anchors,
116 normalizations, 1,044 group-candidate contracts, 143 Joint validity checks,
545 weight reconstructions, 303 exact parent score replays, 126 same-partition
Joint replays, 848 GMM decisions/span maps, 17 endpoint bundles and 58 label
joins. Three representative 1,000-draw bootstrap contrasts exactly match the
old explicit implementation; all 31 new contrasts include within-answer and
common-IU-gate diagnostic intervals. A separate geometry audit reconstructs
143 off-diagonal covariance residuals. Scoring took 104.5 s on three workers;
all 31 contrasts took 11.4 s with cached pair-count statistics. Review is extra.

The fallback policy proposed after this context-bank stage is now implemented
and reviewed above. Its routing, failure accounting and simple controls were
frozen before composite evaluation. The complementary fit coverage did not
translate into an additional PB accuracy gain over the single-bank fallback.

**Previous stage - fusion normalization/no-error audit (2026-09-07):**
`results/fusion_gate_interface_audit_v1/REPORT.html`. Seven existing cores,
58 exposed development answers; five identity tests and independent review
pass. All 14 original endpoints replay exactly. This is a diagnostic, not a
new candidate. Measurement, evaluation and review are terminal; measurement
used one CPU process for 27.2 seconds. No new inference or Claude worktree edit.

Both gate and location matter. IU PB is 17.71%; a perfect binary gate with
its same peak would give 41.89%, while its same gate with a perfect locator
would give 55.18%. Joint graph is 8.33%, with corresponding oracle values
41.50% / 28.24%. Oracles use labels only for diagnosis and retain invalid-fit
failures. IU's raw peak hits the first error in 7/25 erroneous answers; Joint
lambda-zero 9/25, graph 8/25 (four invalid error fits). No winner is established.

A key metric distinction is verified: 89.33% of IU's pooled PRMB AUC pairs
compare different answers. Removing its centering offset raises pooled AUC
0.62261 -> 0.68174 while EVERY within-answer comparison stays unchanged
(mean within-answer AUC 0.67567). This arbitrary-origin projection is not a
calibrated or selected candidate. Keep the registered pooled endpoint, but
show within-answer ranking and exact localization beside it. Joint graph's
analogous pooled change is 0.66255 -> 0.73723 on seven valid answers only.

All 58 positive-affine feature-coordinate tests leave normalization unchanged;
all 361 valid score records have zero fit mean and unit fit SD. Shifting GMM
inputs and step scores by +10 changes none of its decisions. Undoing centering
alone cannot repair that free-mean gate. It discards absolute levels, but this
restricted invariance is not proof that all gray-box detection is impossible.
Raw entropy's binary separation varies strongly across PB subsets. Analytic
duplication of the same observations makes 35 rather than 27 IU PB answers
favor two components at fixed fitted parameters; it adds no new information.

Independent review rejoins 58 labels, reconstructs 116 raw levels, verifies
361 score/gate records, 1,610 oracle outcomes, 35 PB macro values, 14 AUC
pair decompositions and 84 descriptive probe AUROCs. Source and output hashes
match; no parent scores changed.

**Previous next-action note (now executed above):** return to a feature-bank/Joint-grouping comparison
with an explicitly local multiscale representation, retaining IU/equal/Joint
anchors and both native-gate and fixed-parent-gate views. First audit/reuse
the existing fast/slow, innovation and persistence mechanisms in
`spectral_utils/unified_causal_iu.py` and `unified_causal_subset_search.py`.
The historical full pipeline/subset search explicitly uses supervised
development; do not silently import its fitted signs, references or selected
roster into the strict answer-only arm. Freeze one small bank comparison.
A gate-only sweep is insufficient. A future pooled unlabeled gate is hybrid.
The full comparator replay, untouched two-task confirmation, remaining
supporting ideas and historical 24-cell transfer remain open.

**Previous stage - fusion reliability regularization (2026-09-07):**
`results/fusion_reliability_regularization_v1/REPORT.html`. All 58 answers,
23 arms, 38 paired source-group contrasts and independent review completed;
all related processes are terminal. Six tests pass. All fitting rows, features
and Joint groups are retained. A 16-perturbation sensitivity penalty and eight
separate selection perturbations choose lambda from 0/0.1/1/10 without labels.
IU/equal receive a correction of existing weights; Joint uses its native
model-covariance inverse. This distinction is explicit in the report.

No consistent two-task gain is established. Joint graph lambda 0.1 / 1 / 10
gives PRMB 0.66255 / 0.61426 / 0.51713 on the same seven valid answers, and
PB 8.33% / 8.33% / 9.17%. Larger lambda changes the score, not clearly for
the better. The auto graph rule gives Joint PRMB 0.66760 and PB 5.88%.
Full sensitivity regularization reduces IU's sensitivity to 0.463 of baseline
on eight additional unused perturbations, but PB falls from 17.71% to 5.77%.
Joint full sensitivity PRMB falls to 0.60331; its exploratory difference CI
vs lambda-zero is [-0.09582,-0.00805]. Stability is not correctness.

Independent review reconstructs 1,392 perturbation matrices from raw tokens,
adds 464 fresh audit perturbations without changing predictions, verifies
3,180 weight solves, 795 lambda-zero replays, 303 exact parent replays,
881 mixture gates, 1,184 span maps, 43 Joint validity checks, all 23 endpoints
and 58 direct label joins. Six parent endpoints match the previous peak
report exactly. Scoring took 45.6 seconds on three CPU workers; bootstrap
and audit time are additional. No new inference or edit to Claude's worktree.

**Previous next-action note (now audited above):** audit the answer-only normalization/no-error interface
before another regularization sweep. IU graph correction gives PB 14.45%
under the same GMM rule but 24.64% with the parent's gate fixed (diagnostic;
its PRMB is still lower). Determine what absolute uncertainty is discarded by
centering and whether mixture states support the intended correctness meaning.
Keep fusion scores and simple controls fixed for a subsequent gate experiment.
Any pooled unlabeled gate must be a separately labelled hybrid fit scope.
Joint feature/grouping development, comparator replay, untouched confirmation
and 24-cell transfer remain open. No winner is promoted.

Nir Shlezinger's graph-compression paper is now fully read and digested:
`papers/digests/task-based-graph-signal-compression.md`; the ADC lead remains
an initial source check. The paper's implementation caveats are in its card.

**Previous stage - fusion fitting-window sampling (2026-09-07):**
Scoring, all 57 paired contrasts, independent review and the HTML/Markdown
report are complete on the same 58 answers. No experiment process remains
running from this stage. `results/fusion_window_sampling_pilot_v1/REPORT.html`. Six selectors support five
unchanged IU/Joint/equal cores. Sampling reduces fitting rows in 37 answers;
21 replay all rows. Features and dense scoring still cover every window.
IU full gives PRMB 0.62261 / PB 17.71%; transposed DUFS gives 0.63663 / 13.69%,
shuffled DUFS 0.63761 / 21.88%, and top-risk 0.64217 / 19.05%. These point
estimates do not establish a winner; paired IU-versus-full intervals include
zero. Joint graph PB falls from 8.33% to 4.17% under every reduced selector.
Short-event support is NOT established: the eligible PB error sample contains
no first-error steps <=32 tokens. Keep full-grid IU as the working reference,
not a proven winner. Next study full-row feature reliability and label-free
Joint regularization/stability, with gate changes isolated. Sparse scoring,
short-event validation and the broader confirmation program remain open.

Six tests pass. Independent review verifies 665 parent replays, 763
selected-fit weight reconstructions, 348 selection contracts, 1,428 span maps,
393 valid Joint audits, all 30 endpoints and 58 direct label joins. The full
arms exactly match the previous peak report. Scoring took 178.4 seconds on
three workers, including perturbation diagnostics. DUFS paper v3 (2020) is
fully digested; window-gate versus graph-node axes are now explicit.

**Previous stage — fused trajectory readouts (2026-09-07):**
`results/fused_trajectory_readout_pilot_v1/REPORT.html` and `REPORT.md`.
Same 58 development answers, six cores and seven supporting readouts; all
feature fusion fits and binary error gates fixed. The raw peak locator raises
IU PB macro-F1 from zero to 17.71%, with its PRMB AUROC unchanged at 0.62261.
The paired PB difference CI includes zero: [0,+27.78] percentage points.
Equal peak is 17.76%, entropy peak 19.85%; IU/equal remain unresolved.
IU+IMM is 14.79%, IU+BOCPD 13.69%, IU+HMM zero. Joint lambda-zero peak is
12.50%, meaningful graph 8.33%, permutation 4.17%, with no established graph
advantage. Keep coverage/common-ID qualifications; this is not confirmation.

All 58 paired bootstrap contrasts completed. Seven scientific tests passed;
independent review verifies 303 exact parent replays, 2,074 unchanged gates,
2,936 span mappings, 909 filter checks, all 42 endpoints and 58 direct label
joins. The scalar IMM is an actual interacting filter bank, with ordinary
Kalman control; the HMM reuses the older reversible shared-variance kernel.
Scoring took 34 seconds on three CPU workers; bootstrap time is additional.
Both scoring and contrasts are terminal, with frozen artifacts preserved.

**Earlier next-work note (now executed above):** retain raw peak and full-grid fitting
as controls; the first reliability/regularization pilot is complete. The fitting-row sampling pilot above is now complete;
task-aware sparse scoring remains a separate open direction. Treat clean/error gating as a separate problem: even
a perfect locator under the current graph gate has only a 28.24% PB ceiling
on this pilot, versus 55.18% for IU. These are oracle diagnostics, not scores.
No generic temporal sweep or standalone-detector pivot is justified here.
Full comparator replay, untouched confirmation and 24-cell transfer are open.

**Source audit and LOCA read:**
`docs/reviews/bocpd_boundary_audit_2026-09-07.md` documents inconsistent reset
semantics in the old temporal_models BOCPD implementation, plus the overly
broad constant-hazard warning. The new recursion passes exact partition
enumeration; historical numerical code/results stay intact. LOCA's 25-page
paper is now extracted and digested in `papers/digests/loca-local-conformal-autoencoder.md`.
Its burst-whitening idea may support fusion reliability/geometry, but token
neighborhoods and bootstrap copies do not establish its latent-isotropic
measurement assumption. A within-window measurement-uncertainty adaptation
is proposed, not implemented or claimed as LOCA reproduction.

**Previous stage — representation pilot completed and reviewed (2026-09-07):** Implemented
`spectral_utils/answer_localization_v2.py` and `scripts/run_answer_localization_v2.py`.
Five scientific-contract tests pass. Release `localization-cached-v1-20260907`
records all nine existing cells; the first fixed development cohort has 58
answers (12 PRMB, 46 PB across four Qwen3-8B subsets), stratified by trace
length without labels. Nineteen arms separate historical orientation,
answer-local orientation, the existing 30 features, a 27-coordinate local
moment bank, widths 32/8, and meaningful/permuted Joint graphs. The current
answer-local convention retains an explicit negative-entropy global anchor;
it is not claimed to be anchor-free. PB uses one fixed answer-only mixture
readout for first-error/no-error. All score/decision artifacts froze before
label evaluation; 58/58 finished in about 206 seconds with three CPU workers.
Evaluation and review are complete. IU/equal width-8 coverage is 58/58 versus
38/58 for legacy width 32; Joint is 43/58 versus 32/58. Moments-8 selects four
groups in 14 answers. On seven common PRMB answers, graph-minus-IU is
+0.01291 AUROC, CI [-0.03016,+0.04864]: unresolved. The fixed PB
mixture/first-crossing readout gives macro-F1 zero for all 18 fusion arms;
entropy control is 0.08333. This is a failed full-pipeline pilot, not a winner.
Independent review confirms 583 projection/span maps, all 19 endpoint
calculations, 58 direct label-ID rejoins, raw-column identity and frozen
hashes. Next bounded experiment: keep the feature fusion scores fixed and
compare chronological readouts with explicit no-error decisions and equal
fusion controls. The Joint representation/hyperparameter program stays open.
Output:
`results/answer_localization_representation_pilot_v1/`; registered protocol:
`docs/experiments/ANSWER_LOCALIZATION_REPRESENTATION_PILOT_V1.md`.

**Additional user directions saved:** IMM, LOCA, Diverging Flows, KalmanNet
remain active candidates. Scoped historical/source audit:
`docs/reviews/temporal_geometry_revisit_2026-09-07.md`. The old studies often
tested precursors, not the named algorithms. Corrected false Diverging Flows
attribution (actual authors Tsakonas/Ivaldi/Mouret), the unsupervised KalmanNet
venue/year, and the inaccurate HMM/IMM equivalence in notes/source comments.
These corrections change no historical numerical results.

**Active user mandate (2026-09-07):** Omri authorizes Codex to lead the
continuing research, including Joint/graph feature representation and
hyperparameters, IU improvement, both fusion axes, HMM/BOCPD and
Shlezinger-inspired methods, and graph-based TOKEN/WINDOW sampling. The older
"stop Joint graph/lambda" recommendation below is superseded. Preserve short,
reviewable stages, but do not ask again for already-authorized candidates.
Gray-box, one generation pass, unsupervised and primarily single-answer fit
remain the constraints. Every experiment needs matched historical anchors;
the goal is a reproducible improvement on BOTH PRMBench and ProcessBench that
can be presented to the advisors. Do not promise a winner or redefine success
after seeing results. Canonical memory is in `CLAUDE.md`; execution/evidence
ledger: `docs/experiments/LOCALIZATION_RESEARCH_MANDATE_20260907.md`.

**Historical audit findings (superseded by completed audit below):** The executed cycle-1 capsule explicitly uses historical
`confidence_sign_vector` calibration, also inherited by the follow-up cycles.
Describe those pilots as answer-fitted with borrowed feature signs; do not
call every learned component strictly answer-only. The prior broad "review
passed" claims need more precise source-lineage, source-group and fit-audit
checks. Claude's `report_contrasts.json` now exists (01:00, September 7) and
contains six PRMB contrasts, but its ProcessBench contrast object is empty.
File existence alone does not establish full report completion.

**Independent numerical audit completed:**
`results/localization_short_cycles_audit_20260907/REPORT.md` and `AUDIT.json`.
Cycle-1 AUROCs reproduce; lambda-zero replay error is zero at all three
levels, and 8/8 upstream capsule hashes match. The 30 answers contain 29
registered source groups (common Joint/IU: 24 answers, 23 groups). Corrected
source-group intervals preserve the pooled-AUROC conclusions. All original
OK Joint fits also passed their recorded multistart check; Jacobian validity
is not established by this audit. A new retrospective within-answer metric
on 18 mixed-label common answers gives graph 0.73951 vs IU 0.74237, delta
-0.00286 [-0.03096,+0.02560]; it is not a replacement endpoint or a graph win.
IU/equal remain statistically unresolved in the small pilot. No ProcessBench
experiment or new candidate fitting ran in this audit. Next: freeze a common
two-benchmark contract with the access, orientation and holdout boundaries
resolved before new feature/hyperparameter development.

**Short cycles 2–3 and Claude review (2026-09-07 — Step 296):** The
assistant-proposed fixed-group diagnostic and Omri's requested answer-only
Joint-LIU graph test are complete. Fixed groups improve Joint strict coverage
from 24/30 to 27/30 but reduce ranking; fixed-group Joint is `0.65721`
available AUROC, fixed-group continuous L-SML `0.68916`, IU `0.70070`.
The exact graph candidate on the common strict 24 answers is `0.69188` versus
lambda zero `0.69148`, permuted graph `0.69578`, and IU `0.72072`. Graph minus
lambda zero is `+0.00040 [-0.00566,+0.00841]`; graph minus IU is
`-0.02884 [-0.04851,-0.00560]`. Lambda zero replays exactly and all hashes,
arrays and independent AUROC checks pass. Decision: stop answer-only Joint
graph/lambda development; confirm IU/equal on a fresh long-answer cohort, then
test a small frozen trajectory-readout set.

Claude's active PID 145884 is the old v2 report-contrast stage, not a new
target-condition experiment. At 00:35 it was alive with empty stderr and four
of six PRMB contrasts logged; three PB contrasts follow. Its partial contrasts
confirm that lambda zero and the permuted graph beat tuned IU, while the real
lambda-0.1 graph is `-0.0010 [-0.0015,-0.0005]` versus lambda zero. Let it
finish, then verify final JSON/report/docs/commit/push. Claude's two-fold K
diagnostic shows K=4–7 inadmissible because of groups smaller than three and K=8
impossible on 23 features; do not force K above 3. Full review:
`docs/reviews/localization_experiment_review_2026-09-07.md`.

**Short-cycle preference (2026-09-06, after 23:11 — Step 294
[benchmark continuity]):** Omri now requests short stages, returning findings
before choosing the next stage. The broad program is a backlog, not a queued
sweep. This is saved in `CLAUDE.md`. Recommended first stage:
`docs/experiments/LOCALIZATION_SHORT_CYCLE_01_20260906.md` — at most 30 long
PRMB answers, one width, answer-only Joint model-inverse lambda zero with
canonical IU/equal references and a fixed trajectory readout.

Claude's `REPORT.md` now exists (written 22:59), but some contrast fields are
still pending. Its descriptive PRMB table gives model-inverse lambda zero
`0.673414`, LIU 0.1 `0.672394`, and shuffled graph `0.673177`; this motivates
isolating the model-inverse mechanism before another graph sweep. A read-only
metadata check found 290 answers of 1,024–2,048 tokens from 253 source groups,
so a small long-trace pilot is available. No labels/raw feature arrays were
read by that check, and no new fitting, evaluation or cluster job was run.

**Short cycle 1 completed (2026-09-07):** Executed and reviewed the bounded
answer-only pilot in `results/localization_short_cycle01/REPORT.md`. Thirty
long PRMBench answers were score-frozen before labels. Joint model-inverse λ=0
had strict coverage 24/30 (4 blocked partitions, 2 unconverged fits); IU and
equal had 30/30. On the common strict 24 answers, pooled step AUROC was Joint
`0.69148`, IU `0.72072`, equal `0.70134`. Grouped bootstrap Joint−IU was
`-0.02875` CI `[-0.04837,-0.00768]`; Joint−equal was `-0.01016` CI
`[-0.04856,+0.03005]`. Code/provenance review passed finite-array, syntax,
import, ID, hash and label-firewall checks. Decision: do not launch lambda,
two-axis or 24-cell transfer yet; first test a non-clustering Joint reference
or prioritize the simpler IU/equal localizer.

**Two-axis / answer-only clarification (2026-09-06, after 22:50 —
Step 294 [benchmark continuity]):** Module B is now available: `moduleb.json`
was written at 22:43. Claude reports that tables are assembled and additional
contrast bootstraps are running; full report closure is still pending.
The PRMB 3x3 grid selected feature-IU plus trajectory-SML on all five folds,
but it loses to the same-substrate top-10 mean (`0.663423` vs `0.666855`,
registered `HARM`). All three Joint trajectory combinations are blocked on
all folds by `BLOCKED_NO_ADMISSIBLE_PARTITION`. B2a (IU plus max/mean mixture)
reaches `0.672838`; its alpha was selected with training labels. The grid does
not cross the new gate/LIU/diagonal feature variants with trajectory fusers.
Module-B PB secondary and B4 composition results were absent from the inspected
output. Keep the grid's top-10 control distinct from the original B0 span max.

**Omri clarifies the primary goal:** learn localization from the current
answer alone whenever viable, on both fusion axes. Cross-answer pooling is a
secondary comparison/fallback. Scaling, groups, gates, weights, learned
trajectory rules and any learned calibration must respect that boundary;
borrowed quantities require a separate hybrid/calibrated row. Trace length
alone does not guarantee stability or useful error signal. This is now saved
in `CLAUDE.md` and expanded in the continuation plan's Sections 3A and 4.

**Separate requested experiment:** freeze the localization-leading fusion
recipe, then test transfer to final-answer hallucination detection on the
historical 24 cells under their existing benchmark/comparator contract.
Distinguish feature-fusion transfer from complete trajectory-pipeline transfer;
do not select the candidate using 24-cell results. Plan Section 5A records this
track. The active Joint HTML guide now explains both axes, the measured grid,
the strict answer-only goal and the transfer track. No new experiment or live
worktree mutation was made by this clarification.

**Current cross-worktree handoff (2026-09-06, 22:36 Israel time —
Step 294 [benchmark continuity]):** Omri requests a continuing benchmark
covering all relevant historical and current leading candidates, rather than
changing the comparison contract with each experiment. This standing preference
is saved in `CLAUDE.md`, section "Benchmark continuity and comparator coverage".
The consolidated next-work plan is
`docs/experiments/LOCALIZATION_BENCHMARK_CONTINUITY_PLAN_20260906.md`.

Claude is **not confirmed finished**. His latest message at 22:16 reports
Module B still pending. In `C:/Users/omris/TAU/hd_jlsml_v2_wt`,
`results/joint_lsml_optimization_v2/evaluation/` contains `headline.json` and
`inner_selection.json`, with no final report/Module B output at this check.
The headline gives selected L-SML versus selected IU PB F1
`0.343705/0.349282` and PRMB AUROC `0.672394/0.666523`. Claude reports the PB
activation guard classifies fixed `internal_joint` and `internal_cont` as
CATASTROPHE (four and two cells respectively). The shuffled-graph PRMB control
scores `0.673177`, so a benefit from meaningful graph structure is not yet
established. These are preliminary development results, not a final promotion.

Our separate `codex/per-answer-localization-v1` branch contains completed
window extraction and the 400-answer GSM8K AIRCC feasibility check; Joint
window fitting/evaluation is still open. The earlier audit repairs have since
been applied to v2, with a separate R3 freeze record. Final source/amendment
verification remains part of closing that run. Preserve Claude's later
evaluator serialization fix when reconciling the worktrees.

**Next:** prepare the comparator/protocol inventory now; after Claude's final
artifacts, audit the complete v2 result, freeze a continuing localization
benchmark and reproduce historical anchors, then compare the selected Joint
window family under single-answer and pooled-window fitting. Larger-lambda
and focused sequence/readout tests are conditional follow-ups. Keep fixed and
tuned methods distinct and reserve untouched evidence for confirmation.
No new experiment, deletion or modification of Claude's worktree was made by
this planning step. The August entries below remain historical; they do not
override Omri's current localization direction.

**CIW cross-scale localization addendum (2026-08-27 — Step 293):** A new
target-free token/response input layer was implemented and evaluated on all 12
ProcessBench cells plus PRMBench.  For each of the 29 token streams it predicts
the token coordinate from its complete-answer mean and frozen CIW-DEEM answer
risk; five row-held-out folds set a bounded `0.5 * clipped-R2` innovation gate
before the unchanged token IU-PCR head.  The zero-gate path reproduces the
frozen token-IU score to `2.2e-16` on the real smoke cell.  The fitted mean gate
is only about 4.2%, confirming that most token variation is local.

The primary CIW-response arm scores ProcessBench macro F1 `0.308301` and
PRMBench step AUROC/AUPRC `0.582489/0.196327`, versus the previous CIW adapter
`0.309136/0.581138`.  A fixed SU-PCR token-head ablation is worse on
ProcessBench (`0.306202`) and essentially identical on PRMBench (`0.582516`).
After primary opening, the already frozen corrected token score improved
PRMBench AUROC by roughly `+0.0013` with each of B3, IU-PCR, and DUFS-LIU
response heads, but reduced ProcessBench macro F1 with all three.  Preserving
the original per-answer token mean/scale removes the ProcessBench loss and the
PRMBench gain together.  Decision: this is a supported PRMBench ranking
ablation, not a promoted cross-task localization method.  The unresolved issue
is an unlabeled readout/reliability signal separating absolute answer
calibration from relative step evidence, not another covariance solver.  See
`docs/experiments/CIW_CROSS_SCALE_LOCALIZATION_V1.md` and
`results/ciw_cross_scale_localization_v1/REPORT.md`.

**Advisor visualization addendum (2026-08-25 — Step 291):** The originally
promised exhaustive reporting layer is now exposed from one advisor-facing
entry point instead of being hidden inside the untracked frozen science tree.
Open
`docs/meetings/advisor_update_aug21_2026/00_results_map.html` first.  It links
to the self-contained 13-method, 24-cell interactive `REPORT.html`, the four
discussion briefs, the DuckDB/tidy tables, the exact plot manifest and all
leaderboard exports.  The full report contains 175 bound plot-data files: 84
metric forests, 84 paired-contrast forests, two heatmaps, three graph examples
and two graph-diagnostic summaries/scatters.

The advisor application brief now has nine visible result figures.  In
addition to localization and causal-prefix plots, it shows the LEASH
pass@1--token tradeoff and separate RAGTruth, GASP, Lettuce and RefChecker
panels.  RefChecker settings and RAG estimands remain visually and
scientifically separate; no pooled RAG or cross-task ranking was introduced.
Five convenience leaderboard CSVs and an exact published-comparator registry
copy accompany the packet.  The historical single-task macro was stored at
release level, so the task and release exports are exact aliases and this is
documented beside the files.  Static verification confirms 13/13 method names,
175/175 plot records, 23/23 local navigation targets, nine application figures
and nonempty rankable rows in all five leaderboard exports.

**Reconstruction completion addendum (2026-08-25 — Step 290):** The requested
full reconstruction and benchmarking program is complete on
`codex/reconstruction-benchmark-v1`.  The implementation chain now includes
the frozen 24-cell benchmark and external final-answer evaluation, certified
winner-reference contrasts, causal prefix and localization lanes, the LEASH
realized-stopping lane, the seven-panel RAG evidence lane, and one
source-locked unified reporting bridge.  Each scientific lane was built as
independent A/B outputs with grouped bootstrap evaluation and a final
certificate; no mixed-estimand or cross-task leaderboard was introduced.

The final application releases are
`2026-08-25_leash_v1` and `2026-08-25_rag_evidence_v1`.  LEASH has six
actual-callback `READY` cells and two explicit Mistral protocol-gate failures;
its result is an accuracy--compute tradeoff, not a detector win.  RAG contains
seven separate panels, 84 metric rows and six registered within-GASP
contrasts, each with 20,000 source-group bootstrap draws.  RAGTruth shows
strong task heterogeneity, the GASP local-minus-fixed interval includes zero,
Lettuce is retained as a supervised ceiling, and the three RefChecker settings
remain separate.  These are retrospective/application results with their
historical-label and out-of-scope claim-extraction boundaries intact.

The certified unified release is
`results/reconstruction_benchmark_v1/derived/unified_reporting_v1_certified`
(release id `2026-08-25_unified_reporting_v1`).  Its A/B certificate is
`PASS`, authenticates nine certified source bindings, and contains 15 typed
logical tables.  RAG and LEASH enter only as non-rankable context rows; the
bridge preserves localization integer/sentinel semantics and winner-reference
direct paired non-separation without converting it to equivalence or a tie.
The certified release content SHA-256 is
`593f5fcfd00928466ba0db98f01f4a77d0c5ef69183fc577fbfdc8b4ba86c29a`.

The advisor packet is committed in `bc22f16`.  Its builder, email, four HTML
briefs, claim ledger, and README reproduce byte-for-byte; the full builder
`--check` passes in the canonical data checkout.  Independent scientific and
functional reviews confirmed the exact LEASH, RAG, and unified certificate
bindings and the claim wording.  Key later implementation milestones include
`fa81b01` (LEASH), `4099003`/`658dc03` (RAG score/evaluation),
`441e938`/`f9aa18d`/`c7b69fa`/`ebd00c1` (unified bridge and certified
adapters), `7d8c194` (workspace-filesystem compatibility), and `bc22f16`
(advisor packet).

**Decision:** the reconstruction plan has no remaining required experiment or
integration step.  Do not rerun or pool the certified lanes merely to create a
new headline.  The next action is advisor/paper interpretation or an explicitly
requested new study.  Keep the untracked science trees outside Git; their
certificates, manifests, hashes, and committed source closures are the audit
boundary.

**Reconstruction addendum (2026-08-24 — Step 289):** The requested
`reconstruction_benchmark_v1` implementation is prelaunch-complete on
`codex/reconstruction-benchmark-v1`.  It freezes the mixed-v2 nominal
30-feature contract, the exact 13-method primary roster, two independent
preparation/fit builds, source-question group recovery, the label firewall,
20,000-draw grouped evaluation, graph-assumption diagnostics, and the
self-contained searchable reporting package.  The external final-answer A/B
runner and its 29-cell applicability registry are also implemented; currently
27 local cells are `BLOCKED_ASSET`, LCiteEval is protocol-failed, and CoQA is
quarantined.  Final adversarial review found no remaining P0/P1 code blocker;
146 tests pass, with one expected optional-dependency skip, plus the SpecRaGE
check.  **No scientific fit or label evaluation has run yet.**  Next: commit
the clean source snapshot, build immutable release
`2026-08-24_frozen24_v1`, run both 24x13 fits, issue the A/B certificate, then
evaluate, diagnose, and publish the report.

**Branch addendum (2026-08-22 — DEEM benchmark pivot):** Residual-Graph DEEM
Phase 0 completed after the numerical repair (3,050 fits; 50/50 checkpoints
healthy) but closed the current G0–G5 extension because no target-graph lambda
survived the frozen specificity gate.  This is not a failure of graph-free B3.
The next frozen experiment is `deem_vs_iupcr_24cell_v1`: B0 IU-PCR versus B1
hard DEEM, repaired B2 soft/rank DEEM, and continuous additive B3 on the same
24-cell present-inventory contract.  Use
`docs/experiments/DEEM_VS_IUPCR_24CELL_V1_CLAUDE_HANDOFF.md`; do not submit the
archived residual-graph chain.

**Date**: 2026-08-21
**Last updated**: Step 282 has two concurrent amendments. **The
hallucination-geometry audit is implemented, independently reviewed,
externally stress-tested, and documented without an overclaim; A6/PTNI is
separately closed by scope decision because S0b never ran.** All known research
lineages are integrated and the current decision state is consolidated in one
canonical matrix. The
integration preserves remote master, paper-exact acquisition, corrected
white-box capture, local white-box analysis, the previously committed local
contextual/A6/paper evidence, and the pre-existing fair-comparison, Unified,
and three-way ancestors. Heavy intermediates were copied non-destructively to
Drive. The 23 consolidation-only dataset payloads were checksum-verified
against their canonical Drive copies and omitted from the unpublished Git
history after GitHub's LFS budget blocked publication; their manifests and
SHA-256 inventory remain tracked. No stash, branch, worktree, or local source
artifact was deleted.

**Amended Step 282 (2026-08-20): objective (1) is withdrawn.** The PTNI/A6
direction is closed by scope decision, `CLOSE_A6_S0B_DIRECTION_REJECTED` —
Omri is not pursuing a self-supervised identification mechanism. The S0b gate
never ran: job `196764` was cancelled while PENDING, `Elapsed 00:00:00`, and
five independent checks (cluster output root, cluster logs, all three local
worktrees, Drive, and both consolidation manifests) find no artifact. This is
a rejected direction, not a falsified one; do not re-file it as
`CLOSE_S0B_NUMERICAL_NONCONVERGENCE`. A6-S0a and the Step 270 [A6/PTNI]
gamma-3 falsification survive as citable evidence.

The current order is therefore: **(1) ~~finish the frozen PTNI/A6-S0b gate~~
WITHDRAWN; (2) if budget is justified, run one new preregistered white-box
validation with corrected live capture and architecture fidelity; (3) close
paper-exact provenance and claim mapping -- NOW LEADING; (4) package the
reasoning/RAG contribution.** IU-PCR and
Unified-28 are frozen anchors. Family-NRM and the localization/RAG applications
are accepted but bounded. Clustering, DUFS/Laplacian discovery, Atomic-NRM,
and the current contextual routers are closed. White-box remains promising but
post-hoc and validation-blocked. Paper-exact computation is complete; its
documentation/provenance closure is not. Canonical status:
`docs/research_notes/research_status_consolidated_2026-08-19.md`.

A separately authorized active task completed a retrospective cross-dataset
manifold diagnostic while integration was in progress. It finds a transferable
supervised direction but no distinct nonlinear-manifold advantage:
held-family AUROC is 0.7379 for balanced logistic, 0.7353 for kNN, and 0.6882
for PPCA; decision `SHARED_DIRECTION_NOT_DISTINCT_NONLINEAR_MANIFOLD`. This
does not make DUFS-LIU target-identifiable. Step 281 did not launch the
experiment; it preserves and classifies the late-arriving result.

Step 282 followed this with three increasingly strict checks.  First, the
reviewed conditional topology audit retained recurrent length-conditional
geometry in Global, ProcessBench, and RAGTruth, but its registered decision is
`CONTROL_FAILURE_INVALIDATES_GEOMETRY_AUDIT`: the ProcessBench length-only
exact-null false-positive rate was 16.67% versus the frozen 15% limit, Global
graph health failed, and LIU utility was negligible.  Union-kNN, adaptive-k,
and diffusion were similar; radius was weaker/unhealthy, so topology was not
the bottleneck.

Second, nested leave-dataset-family-out supervised metric discovery on the 16
common non-length features produced stable conditional geometry (all four
supports passed geometry; weight cosine 0.999), but no candidate passed the
distinct-manifold or utility gates.  The frozen all-feature candidate's
metric-vs-linear maxT p-values were 0.155 exact and 0.080 CRT, and LIU-IU was
+0.00180 AUROC; decision `TRANSFERABLE_SUPERVISED_DIRECTION_ONLY`.  The equal-
weight graph was at least as geometrically strong, reinforcing a shared
confidence direction rather than a newly discovered nonlinear manifold.
The discovery package passed deterministic verification before a later
whitespace-only source cleanup.  After that cleanup, the canonical rebuild and
checkpoint-resume rerun reproduced the same decision; Omri stopped the third
isolated fresh rebuild at 150/199 whole-search null draws to proceed with the
mentor update.  Therefore the post-cleanup snapshot does not claim a completed
fresh-rebuild attestation.

Third, the frozen candidate was audited on five external-to-discovery cells
from three new dataset families (AQuA, HLE, CoQA), using 999 exact and 999 CRT
draws, learned-metric, linear-score, equal-weight, and linear-residual graphs.
Coverage passed, but exact-length swaps were ineligible in all five cells and
the HLE CRT was also ineligible.  The correct decision is therefore
`CONDITIONAL_NULL_INELIGIBILITY_INVALIDATES_EXTERNAL_AUDIT`, **not** transfer
failure and not validation.  Descriptively, family metric effects were +0.055
(AQuA), +0.084 (CoQA), and +0.003 (HLE); equal-family LIU-IU was +0.00027 with
95% interval [-0.00100,+0.00261].  The run and figures rebuild byte-for-byte.

For any later manifold work, do not search more weights, subsets, or graph
topologies on the same 16/28 global-feature matrix.  The scientifically
distinct follow-up proposed by Omri is a separate, pre-registered DSP
representation audit across Global, ProcessBench, and RAG, keeping their three
targets and nulls separate and comparing DSP against linear-on-DSP,
equal-weight DSP, the current global features, and DSP residualized against
length/confidence.  This note records the next representation hypothesis; it
does not reorder the paper-exact documentation priority above.

## Prior update — Step 280

**Step 280**: **The Early/Online detection record is now
consolidated in one canonical status note.** This documentation-only step does
not change the accepted Step-279 comparison package or reopen method search.
It separates prefix-based final-error prediction, first-error localization,
and realized stopping; records the Step-148 and Step-182 evidence; distinguishes
the historical DeepConf proxy from a paper-exact reproduction; and carries
forward the Step-269--274 transfer conclusions. Canonical note:
`docs/research_notes/early_online_detection_canonical_status_2026-08-19.md`.

## White-box capture branch record — Steps 243-244

**Historical branch date**: 2026-08-13

**Branch update**: Step 244 — Codex found that a no-op resume had destroyed the
validation reports on 4 completed layer-view cells. **Recovered with proof,
root-caused, and fixed**; the sidecars were never at risk and all 4 cells are
confirmed genuine Gate-B passes. Evidence:
[docs/experiments/LAYER_VIEWS_RECOVERED_VALIDATION.md](docs/experiments/LAYER_VIEWS_RECOVERED_VALIDATION.md).
Prior: Step 243 — **a NEW, SEPARATE research arm opened on branch
`whitebox/per-layer-views`**: white-box depth views (per-layer logit-lens
telemetry), extracted on 14 cells across 9 model families. See the session
addendum immediately below. This arm is **orthogonal to the grey-box line and
is deliberately NOT combined with it** (Omri, 2026-08-12). The
RAGTruth/ProcessBench GL-LIU decision point (Step 239, further down) is
UNCHANGED by this — still open, still the grey-box line's main open question.
## White-box analysis branch record — post-hoc layer fusion

**Branch update**: exact-row white-box versus gray-box comparison. See the
2026-08-19 addendum below for the matched result and claim boundary. The earlier
layer-organic NRM, four-band NRM, and registered white-box v2 results remain
unchanged and auditable.
The earlier localization campaign and RAGTruth/ProcessBench GL-LIU decision
point remain unchanged.

## Latest development — matched white-box versus gray-box comparison (2026-08-19)

The final pure distributed-depth white-box U-PCR score was compared with the
frozen 30-feature gray-box `mixed-v2` system on the exact intersection of
31,440 candidates in the same 13 dataset/model cells. Both AUPRC values were
recomputed with hallucination (`incorrect = 1`) as the positive class; the old
gray-box AUPRC used correctness as positive and is therefore not reused.

White-box scores **0.781690 AUROC / 0.677048 AUPRC**. The final gray-box
DUFS-LIU scores **0.782994 / 0.687731**. White minus gray is -0.001304 AUROC
(95% equal-cell bootstrap [-0.016931,+0.012300]) and -0.010683 AUPRC
[-0.035363,+0.011078]. With the same deployed U-PCR solver on both contracts,
white minus gray is +0.000750 AUROC [-0.012926,+0.013420] and -0.009188 AUPRC
[-0.030266,+0.009897]. The correct conclusion is a practical AUROC tie with no
evidence that white-box alone improves aggregate discrimination.

White-box does offer broader row coverage: 42,238 scorable candidates versus
31,467 under gray `mixed-v2` complete cases. Final white/gray risks have mean
per-cell Spearman correlation 0.8677, so most signal is shared. An explicitly
post-hoc equal-z average reaches 0.790203/0.690580 and gains +0.007209 AUROC
[+0.000101,+0.014105] over gray DUFS-LIU, but it is an exploratory hypothesis,
not a promoted method or independent confirmation.

The comparison is **POSTHOC / PRELIMINARY / WHITE VALIDATION BLOCKED**. White
capture still lacks corrected live Gate B and the architecture-fidelity pilot.
Fit reconstructs row availability without labels, freezes label-free score
bundles and hashes, and only then opens labels for exact-row evaluation.

Canonical report:
`results/whitebox_vs_graybox_matched_v1/REPORT.html`.

Durable research index:
`docs/experiments/WHITEBOX_LAYER_FUSION_RESEARCH_RECORD.md`.

---

## Latest development — distributed-depth white-box candidate (2026-08-14)

A retrospective search found the first white-box layer-fusion candidate whose
13-cell equal-cell macro AUROC exceeds both a strengthened best-atomic-view
oracle and the local TriLens grouped-L2-probe approximation. The search moved
beyond token means: maximum target NLL, maximum top-1 surprisal, target-vs-top1
gap, entropy excess over top1, and mean KL-to-final are summarized across
depth. Two summaries force views from every depth quartile. A separate organic
expert makes each transformer layer one group with three tail metrics inside
the layer, and a lens-96 hierarchical DUFS expert supplies module-by-metric
structure.

The final **pure inner-state** deployed U-PCR scores **0.784612 AUROC / 0.648128
AUPRC**. The strengthened evaluation-only oracle, which can select any relevant
token-mean or token-tail module/metric/layer view separately in every cell, is
0.784186/0.648765. The AUROC point delta is +0.000426, but its 95% paired
interval crosses zero [-0.006351,+0.006833]; this is a numerical discovery, not
a robust oracle win. Against TriLens (0.768933/0.645543), pure U-PCR gains
+0.015710 AUROC with a positive interval [+0.004502,+0.026423].

The stronger **hybrid** appends ordinary generation entropy as one transparent
output-level control. It scores **0.785538 AUROC / 0.652755 AUPRC**. Versus the
same strengthened atomic oracle it gains +0.001352 AUROC
[-0.005437,+0.007992]; versus TriLens it gains +0.016579
[+0.005661,+0.027168]. Its AUPRC advantage over TriLens is also positive with
a paired interval [+0.002070,+0.027714]. Deployed U-PCR beats the matched equal
mean by +0.005055 AUROC [+0.003705,+0.006440], confirming that this is not the
old four-expert simple-average fallback.

Both results are **PRELIMINARY / VALIDATION BLOCKED**. The registry was selected
after inspecting these cells, no cell is independent confirmation, and the
corrected live Gate B plus architecture-fidelity pilot remain open. Do not
promote the point-estimate oracle win as robust. All fitting remains label-free;
score hashes are frozen before evaluation labels open. The final audit verifies
14/16 source files (hybrid/pure respectively), 28 prepared bundles, 14 frozen
score cells, 72 report artifacts, no label arrays, self-contained reports, and
39 focused core/distributed tests.

Canonical reports:
`results/whitebox_depth_distributed_pure_v1/REPORT.html` and
`results/whitebox_depth_distributed_consensus_v1/REPORT.html`.

---

## Latest development — white-box layer-organic NRM addendum (2026-08-13)

The requested organic grouping was implemented on the frozen v2 layer
matrices: every residual transformer layer is one group, with entropy,
target-token NLL, and top-1 surprisal retained as three separate internal
features. KL-to-final was kept out of the primary because it is nonlocal; it
appears only in a named sensitivity. Exact layer identity was evaluated on the
ten protocol-eligible 32-layer cells. The 36/40-layer cells were excluded
rather than interpolated.

The structural premise did not produce a robust correction. Across ten cells,
atomic-triad IU-PCR reaches 0.5763/0.4969 AUROC/AUPRC. Leave-dataset-out NRM
reaches 0.5776/0.4987: +0.136pp AUROC [0.079,0.194] and +0.185pp AUPRC
[0.091,0.284]. But LOMO is -0.049pp [-0.103,+0.007] and LOCO is -0.091pp
[-0.147,-0.033]. More importantly, the two clean controls reverse: the same
Llama model across six datasets is -0.075pp [-0.147,-0.002], and the same
GSM8K dataset across five 32-layer models is -0.154pp [-0.238,-0.066]. The KL
sensitivity is also negative.

Keeping the three measurements atomic weakens the IU base by 3.925 AUROC
points relative to the previous equal-mean one-expert-per-layer contract,
with severe heterogeneity (5 wins / 5 losses and a large TriviaQA reversal).
The organic relation is real, but declaring a layer group does not regularize
the within-layer weights; the old equal-mean compression did.

**Decision**: do not adopt layer-organic NRM and do not replace v2 or the
four-band NRM addendum. A future, genuinely distinct hypothesis would fuse the
three local measurements into one regularized layer expert before applying
cross-layer NRM; freeze it before evaluation. The current report remains
**PRELIMINARY / VALIDATION BLOCKED**.

Canonical artifacts: `spectral_utils/whitebox_layer_organic_nrm.py`,
`scripts/whitebox_layer_organic_nrm_experiment.py`,
`scripts/whitebox_layer_organic_nrm_report.py`,
`scripts/test_whitebox_layer_organic_nrm.py`, and
`results/whitebox_layer_organic_nrm_v1/REPORT.html`.

---

> **Branch note**: this file on `whitebox/per-layer-views` was branched before
> the Step-242 PROGRESS update landed on `master`; the Step 240–241 addendum
> below is this branch's most recent grey-box state. Reconcile on merge — do
> not treat the absence of a Step-242 entry here as evidence it did not happen.

## Session addendum (2026-08-12) — white-box depth views: a second view axis, 14 cells / 9 families (Step 243)

**What this arm is.** Everything the project fuses today comes from ONE
trajectory: a scalar per generated token. This arm produces a second,
orthogonal trajectory — a scalar per token **per layer per module** — so the
label-free fusion family can run over *depth* instead of time. Teacher-forced
over generations already in the cache; nothing regenerated; canonical pkls
untouched (output goes to per-cell sidecars).

**Why it is worth doing** (full reasoning + paper table in HISTORY Step 243):
- The literature gap is real and narrow: **everyone combines per-layer features
  with a supervised probe; nobody combines them label-free.** The closest work,
  **TriLens** (arXiv:2606.01033, May 2026), defines exactly the 3-module ×
  all-layers entropy feature we extract — and fits an MLP probe on an 80/20
  split. The words "unsupervised" and "ensemble" do not appear in it. Verified
  against the PDF, because the automated fetch summary claimed the opposite.
- It is **3L views, not L** (84–126 per model) — the first time the project has
  had a reason to open the **50+ view** dependency machinery (STDR,
  dependent-classifier SML, SU-PCR) that `Research_Directions.md` parked.
- Single-layer AUROC in TriLens is **0.63–0.73** — the weak-estimator regime
  L-SML/U-PCR exists for.
- **Depth has constant length no matter how short the answer is.** Median trace
  length is 6 tokens on `se_squad_v2` and 8 on `spilled_triviaqa` vs 243 on
  GSM8K; 32 layers × 3 modules = 96 readouts on a 6-token answer. This attacks
  the documented structural weakness of the thesis directly.

**Validation status (Step 244, after Codex's review)**: all 14 cells' Gate-B
verdicts are intact and confirmed. Four of them (`ars_gsm8k_r1distill8b`,
`noise_gsm8k_phi3mini`, `lapeigvals_gsm8k_phi35`, `noise_gsm8k_mistral7b`) had
their *report files* — not their data — destroyed by a no-op resume (job
184777); recovered by validation replay (job 186485) that reproduces the
original log to every digit, with SHA-256 proof the sidecars were untouched.
**Read `RECOVERED_VALIDATION.json`, not `layer_views_report_<cell>.json`, for
those four** — the corrupted reports were deliberately left in place. Fixed in
`38d3a37`; regression test `scripts/smoke_layer_views_resume.py`.

**State: data is collected and on Drive. No scoring has been run yet.**

- 14 cells, 9 families, **4.56 GB**, at
  `gdrive:hallucination_detection/cluster_results/layer_views/<cell>/`.
- Families: Llama-3.1-8B (6 cells), Llama-1-7B, Mistral-7B-v0.3,
  Mistral-Nemo-12B, Mistral-Small-24B, Phi-3-mini, Phi-3.5-mini, Qwen3-8B,
  DeepSeek-R1-Distill-Llama-8B.
- The architecture guard passed at exactly `0.00e+00` on every accepted family,
  **including both Phi-3 variants** (fused `qkv_proj` / `gate_up_proj`) — so the
  3-module decomposition is not Llama-specific.

**Open items on this arm**
1. **Which fusion entry point to target — BLOCKING, needs Omri.** The
   `dufs_liu_mixed_v2` contract is frozen to the registered *token-trace*
   feature list (`CONFIDENCE_FEATURE_SIGNS_V1`), so depth views cannot drop into
   it; they would go through `laplacian_iu_fit` / `upcr_fit` on a plain matrix.
   Per `feedback_ask_which_method_to_evaluate`, do not infer the arm — ask.
2. `lapeigvals_gsm8k_llama3b` — Gate B failed on ONE statistic by 2%
   (first-token median |dH| 5.12e-02 vs 5e-02). Suspect the `unsloth/` mirror's
   chat template. **Threshold was not lowered.** Re-check against Meta's repo.
3. `internalstates_gsm8k_qwen25_7b` — Gate B fail; a 5-variant warp probe
   **ruled the warp out**. Tiny median, fat tail. Unexplained.
4. `epr_triviaqa_mistral24b` — `Mistral3ForConditionalGeneration` (multimodal
   wrapper, layers at `.model.language_model.layers`). Guard refused it. Low
   priority: Mistral-Small-24B-2501 already covers the family.
5. Nothing has been decided about pooling, view definition, or layer selection
   — deliberately. Those are the research questions and the three places a prior
   could enter.

## Session addendum (2026-08-10) — external data collection scaled; 4 new reasoning-localization competitor ceilings (Steps 240-241)

**Fair Paper-Exact Comparison Package v1 remains accepted and independently
reproduced byte-for-byte.** The CPU-only package
joins 148,502/148,502 registered records cleanly, uses 2,000 paired grouped
bootstrap draws (seed `20260818`), and keeps Global, Localization, Prefix, and
stopping/adaptive compute in separate lanes. Canonical report:
`results/fair_paper_exact_comparisons_v1/REPORT.html`; protocol:
`docs/experiments/FAIR_PAPER_EXACT_COMPARISONS_V1.md`; rebuild attestation:
`results/fair_paper_exact_comparisons_v1_REBUILD_VERIFICATION.json`.

Unified-28 is below its dedicated incumbent in every eligible direct lane:
Global ProcessBench AUROC 0.662910 versus 0.687036 (delta -0.024125
[-0.041678,-0.007466]); official ProcessBench Localization macro-F1 0.284832
versus 0.326141 (-0.041310 [-0.063480,-0.016467]); and four-cell causal Prefix
mean AUROC@64/128 0.578103 versus 0.606721 (-0.028617
[-0.052068,-0.004390]). The earlier Step-274 Localization gain was against the
then-matched max-entropy row, not the stronger family-six dedicated incumbent.
The surviving positive common-protocol claim is Unified-28 versus
Mind-the-Gap Localization: +0.082376 [+0.059730,+0.117390]. High-access PRM
and critic rows remain separate ceilings.

Stopping is an accuracy--compute tradeoff, not a detector win: LEASH saves
20.2%--49.9% realized reasoning-plus-closure tokens across six complete cells
but lowers pass@1 in all six (interval-clear in five). Unified-28 stopping is
ineligible because no frozen policy has real forced-closure outputs. The
24-cell replay is coverage-only on six identity-proven cells / 3,238 rows; it
is not a 23- or 24-cell headline. S2 Global/Prefix, REFRAIN, DeepConf M2,
Mistral LEASH, full uPRM, and Streaming remain blocked or partial and never
enter headline aggregates.

The fair-comparison integration cycle is closed. The package tree SHA-256 is
`957cf08e94995d7b28143f1d53dd08062e80a8beab6c52650fb670ad1295260c`;
146 focused tests passed with one expected opt-in cache test skipped. No GPU,
cluster job, Drive mutation, large download, or feature/DUFS search occurred.
The next fair-comparison action is paper drafting and advisor interpretation,
not additional retrospective method selection. Any missing-asset GPU work
requires the separate structured approvals in `GPU_GATES.json`.

Step 278 remains `DIAGNOSTIC ONLY`: the new manifold/geometry literature does
not supply a label-free router key. Step 277 remains
`GLOBAL_ORACLE_NOT_ACCESSIBLE_BY_CSTG`, and Step 276 remains
`STOP_CONTEXT_NOT_SUFFICIENT`. A6-S0a remains `PASS_S0A`; A6-S0b is a separate
frozen gate and is outside the fair-comparison package.

## Fair Paper-Exact Comparison Package v1 — Step 279

The package is the publication source of truth for comparisons. It freezes
ordinary Unified-28 as seven causal streams crossed with `level`, `ewma16`,
`positive_area`, and `persistence`, ordinary two-component L2 IU-PCR, and the
Identity accumulator. Evaluation outcomes were not used to change the method.
Each eligible direct table includes Unified-28 and the lane's dedicated
incumbent on identical ordered IDs.

Global, Localization, and Prefix all show interval-clear Unified-28 regressions
against their dedicated incumbent. This is the central scientific result of
the unification question, not a failed package: the single causal method is
coherent and reproducible, but the price of unification is measurable on all
three direct populations. The package also preserves the positive
Mind-the-Gap comparison, native PRMBench and Mind-the-Gap panels, the six-cell
LEASH frontier, exact access/fidelity labels, and all blocked/partial evidence
without mixing estimands.

Acceptance evidence is external to the build's self-report: two independent
directories contain byte-identical complete 61-file payload trees. The
manifest SHA-256 is
`b89b7358b421634f6e5ba4b8458d98724cb06f60ad1a36c8a5774ff14ccf0620`,
and evaluator SHA-256 is
`a73206bf8901825135017ff728e11b563eb8683fc519fe5fcdfbccaf25c238a3`.
The immutable package was built from commit `baaa4a5` and first published in
`bc296ff`. The canonical Unified narrative documents were subsequently
clarified in `6c44a84` and Step 279; package registries intentionally retain
their original build-time document/content hashes and
`UNIFIED_TEMP_WORKTREE_MANIFEST.json`. Use the historical commit for an exact
package rebuild rather than substituting the later narrative text.

## Supervised c-STG router sufficiency diagnostic — Step 276

The CPU-only diagnostic implemented the c-STG Gaussian gate relaxation with a
one-hidden-layer context hypernetwork and an intentionally constrained
prediction head: all six family directions remain in their frozen risk
orientation and the context can only redistribute non-negative leverage. It
used the existing `family6__level` step summaries for Localization and
`family6__fast_slow` endpoints at budgets 64/128 for Early. Core context was a
label-blind IU rank, position, and family-contribution MAD; DSP added the 30
causal innovation/short-long/positive-mean/persistence/recovery coordinates.
Fit used only the existing calibration partition and evaluation used only
development, grouped by source question. Architecture/audit targets were not
used.

The method was compared with family-only, context-only, and feature-plus-
context balanced logistic controls, core-only c-STG, and independently
permuted context. Localization DSP c-STG was 0.2812 F1 versus 0.3503 for global
LR, while Early was 0.5906 versus 0.5995. Neither interval supported a gain;
both task family guards failed. Early DSP c-STG was only +0.0032 over linear
feature augmentation with an interval spanning zero, and the permuted router
scored 0.5994. This is not mechanical collapse: a registered switching-family
test exceeded 0.90 AUROC and recovered active gates, real gates varied across
samples, and the real-context model fit calibration aggressively. The failure
is held-question generalization/alignment.

The earlier +2.833pp oracle belongs to the completed-trace 24-cell family
fusion diagnostic and is not a Localization/Early ceiling. This Step therefore
supports only the narrower conclusion that the present DSP summaries are not a
robust routing key for these two online tasks. It remains retrospective premise
evidence, not a final label-free method or external confirmation. No inference,
GPU, cluster, download, or Drive mutation occurred.

## DSP-contextual IU router pilot — Step 275

The pilot implemented a covariance-entry IU API, a source-question-balanced
causal context router, exact IU fallback, five family-resolved DSP context
blocks, deterministic neighbour selection, group-effective sample size,
shrinkage, sign/leverage alignment, and frozen S0--S4 orchestration. A pre-run
mechanical contradiction was caught before the intentional run: a non-uniform
Gaussian kernel on exactly 32 neighbours cannot achieve `n_eff>=32`. The
corrected protocol uses eight extra neighbour questions of headroom while
retaining the original effective-size gate.

S0 passed exact covariance-IU identity, exact global fallback, exact source-row
duplication invariance, the context-independent null, and observational
equivalence. It failed both positive-world gates and both coherent-nuisance
safety gates. The informative world averaged IU/contextual AUROC
0.8422/0.8161, delta -0.0261 with eight wins; the coherent-nuisance world
averaged 0.7394/0.6002, delta -0.1392 with zero wins and worst seed -0.1650.
Therefore the frozen program wrote S1--S4 as `SKIPPED_BY_S0` and accessed no
real cache or label. Eight focused/regression test scripts pass. No inference,
GPU, cluster, download, or Drive mutation occurred.

## Unified Causal IU-PCR subset and transfer decision — Step 274

The local CPU cycle tested structured subsets of the 1,036-coordinate causal
DSP bank, ordinary IU-PCR, DUFS-LIU lambda paths through 3, and learned task
reweighting. The full bank is not the right operating point. The best
cross-scorer compromise is the 28-coordinate ordinary-IU roster formed by
seven causal streams and four transforms: `level`, `ewma16`, `positive_area`,
and `persistence`.

After correcting an invalid pooling-across-folds aggregation, the Qwen
development values are 0.6914/0.3040/0.5301 for Global/Localization/Early;
the earlier 0.7012/0.3278/0.5435 values are withdrawn. Frozen Llama transfer is:

| method | Global AUROC | Localization F1 | Early AUROC |
|---|---:|---:|---:|
| Unified-28 ordinary IU-PCR | 0.6629 | 0.2880 | 0.5587 |
| matched task incumbent | 0.6870 | 0.2419 | 0.5777 |
| delta | -0.0241 | +0.0461 | -0.0189 |

Every Localization gain is significant, but the co-primary non-inferiority
gate fails because Global and Early regress beyond their margins. Unified-28
does beat the matched IU28 Early row by +0.0213; max entropy is nevertheless
the stronger Early baseline in this exact panel, so this is not an Early win.
DUFS-LIU and task reweighting improved selected Qwen development cells but did
not transfer to Llama. The local Llama copy is bit-identical to Drive. The new
DeepConf M2 acquisition was only 3.75% complete and carried no Localization
labels, so it was not downloaded or treated as validation. All 39 focused and
regression tests pass. No cluster job was launched in this cycle.

The separate 24-cell Global replay remains a task-specific reference, not a
score for Unified-28. Ordinary IU-PCR with 36 features reaches 0.7591 macro
AUROC versus 0.7766 for the frozen mixed-v2 DUFS-LIU baseline, delta -0.0175
with 95% CI [-0.0477,+0.0140]. Math is a practical tie (0.7869 versus 0.7862);
QA drives the loss (0.7128 versus 0.7604). New DUFS lambda 0.3/1.0 reaches
0.7575/0.7552 and alpha=0.5 reweighting is worse. Do not pool these values with
the Unified-28 Llama transfer table.

## Fair-comparison consolidation pivot — Step 274

The cluster campaigns acquired much of the expensive model-dependent evidence:
full ProcessBench critic/PRM/control outputs, RAGTruth/LettuceDetect, GASP,
RefChecker, PRMBench telemetry, and other manifests. Acquisition was
deliberately separated from CPU-local evaluation. The missing deliverable is
the integration layer: align identical row IDs, freeze Unified-28 and the
task-specific incumbents, score all eligible methods on the same rows, use one
evaluator and grouped bootstrap per lane, and publish direct comparison tables.

There must not be one mixed leaderboard. Produce four separate lanes:

1. Global final-answer detection: AUROC/AUPRC and fixed-FPR on common traces.
2. First-error Localization: official ProcessBench macro-F1 on the common
   3,400 rows; Mind-the-Gap native SLA remains a separate native panel.
3. Prefix detection: AUROC/AUPRC at common absolute token budgets and
   ever-warning fixed-FPR behavior.
4. Stopping/adaptive compute: accuracy-versus-total-tokens frontiers; do not
   compare these token savings numerically to detector AUROC.

Each result row must be labelled `official-exact`, `paper-specified`,
`paper-specified-partial`, `adapted-common-protocol`,
`published-context-only`, or `blocked-assets`. Published numbers are context,
not substitutes for common-protocol replays. The next session begins with a
read-only inventory and a written plan; it must not launch jobs or alter Drive
before the user approves that plan.

## Parallel line — A6/PTNI (its own Steps 269-270)

Kept verbatim from the second working repository. It runs beside the
localization sections below rather than superseding them; see "Step numbering"
at the top of `HISTORY.md`.

## Current decision point — γ̂3 failed the correction it was owed (Step 270)

Step 252 established the b-coupled cubic channel with two load-bearing numbers:
pooled `cos(γ̂3, g*) = +0.76` (atomic) and a family sign-bit margin `≈ 0.56`.
Both used a probe orthogonalized against `{1, b}` only; that memo's own §6
required Gram–Schmidt against `{1, b, φ2}` plus winsorized `b` for any future
estimator. Recomputed in the corrected form, **neither number survives**:

| | Step 252 (`{1,b}`) | corrected (winsor 1%) |
|---|---:|---:|
| atomic pooled cos(γ̂3, g\*) | +0.7617 | **+0.3350** (−0.0806 unwinsorized) |
| family sign-bit margin, 5-family | +0.5532 | **−0.1903 — wrong sign** |
| family sign-bit margin, 6-family | +0.4889 | **−0.1702 — wrong sign** |

The fidelity control reproduces Step 252 exactly (+0.7617, 13/17, +0.5129 /
87%), and the memo's quoted ≈0.56 is confirmed as the 5-family restriction.
A crossed design attributes the loss to **the φ2 orthogonalization, not the
winsorization**: winsorizing the original probe 0→10% holds +0.68…+0.76, while
removing φ2 at zero winsorization gives −0.0806 under all three pooling
conventions (raw / unit-RMS / direction-only), so it is not a scale artifact.
Winsorization then restores the corrected probe only monotonically in the knob
(+0.24 → +0.46), never reaching the original.

**Consequences**: Step C (replacing the deployed family-NRM all-ones sign bit
with `sign(⟨v_neutral, γ̂3_family⟩)`) is **closed** — at the registered primary
setting it would have flipped the deployed method into the wrong orientation
(the teacher says the all-ones bit is correct, cos +0.90). Step B (the
retrospective kill-test) is **not started**: it selects and orients by γ̂3,
which is the vector that just failed. Per the plan's own gate, work stopped
here. Conclusion:
`docs/research_notes/gamma3_correction_conclusion_2026-08-16.md`; artifacts in
`results/gamma3_corrected_2026-08-15/`.

**A6-S0b is still running** — the registered chain is in local Docker container
`a6s0b`, stage 2 of 4 (Pythia prompt NLLs). No verdict artifact
(`S0B_COMPLETE.json` / `S0B_CLOSED.json`) exists yet; the verdict entry will be
Step 271. Do not recreate the container; `docker start a6s0b` resumes it.

## Step 269 state (prior)

The A6-S0b source boundary is **reviewed and
frozen** at commit `89c414a` (56/56 tests; single independent review closed
NO BLOCKERS after two result-changing blocker fixes — see HISTORY Step 269).
The exact Pythia snapshot is downloaded and byte-authenticated. The sealed
chain (prepare → run-pythia → run-analysis → verify) runs on **AIRCC via
`cluster/a6_s0b_chain.sbatch`** — Linux-only because the sealed S0a manifests
are Unix-keyed and Windows checkout CRLF-converts sources. Submission is
staged and **blocked only on the TAU VPN**. Expectation from the development
preflight: the frozen `gradient_inf <= 1e-8` gate may close S0b as
`CLOSE_S0B_NUMERICAL_NONCONVERGENCE`; the registered verdict stands as
produced. S1 (or the successor route) opens only on Omri's explicit go; the
A7 successor-clause conflict has an uncommitted reconciliation draft
(`docs/research_notes/DRAFT_a7_successor_reconciliation_2026-08-15.md`)
awaiting Omri, and the post-A6 route survey is at
`docs/research_notes/a0_a6_route_survey_and_next_route_2026-08-15.md`.

## Step 268 state (prior)

A1--A5 are closed. A6-S0a completed with
the independently reproduced verdict `PASS_S0A`. The authenticated tokenizer
restore, frozen S0a boundary, all 7,800 checkpoints, aggregate, and completion
artifact verify exactly. No response telemetry, simulator result, natural
response, correctness sidecar, benchmark target, or sealed S1 seed has opened.
## Parallel paper-exact acquisition record — Steps 274-275

This block preserves the acquisition branch's contemporaneous status. Its
claim that integration was blocked is superseded by the accepted Step-279 Fair
Paper-Exact Comparison Package; its acquisition, Drive, and provenance facts
remain part of the record.

**Parallel acquisition update**: Steps 274-275. Two things closed and one was blocked.
Step 273's frozen protocol was recovered by Codex and **Step 273 now
reproduces**: every decision-bearing Stage-0/Stage-1 artifact is byte-identical,
including `STAGE_1_LOCAL_INTERVALS.csv`, and the residue is bounded at 1e-14
float drift plus wall-clock timings. Both paper-exact acquisitions finished clean
and are backed up to Drive with byte-identical totals. The next approved step —
CPU-first scoring of the shared 3,400-row ProcessBench table — **cannot start
here**: the Fair Comparison v1 contract Codex names as canonical lives on
`codex/fair-paper-exact-comparisons-v1`, which is not on our remote. Do not
reimplement it from its description. No new GPU work is approved: no Mistral
rerun, no confirmation cell, no resumed K=4096 acquisition. A6-S0a remains
independently verified as `PASS_S0A` and A6-S0b is still its frozen next stage;
none of this work alters that boundary.

## Protocol recovery and Step-273 verification — Step 274

`scripts/run_local_online_comprehensive_stage1.py` gates on the SHA-256 of the
frozen protocol and refused to run. The committed
`docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.md` hashes to `b5991a89...`,
while seven frozen artifacts record `c921b0d4...`, and no version in any git ref
matched. The frozen `RUN_MANIFEST.json` records a
`/Users/osegev/Desktop/...` path: Step 273 ran on a Mac, and its pre-commit draft
was never committed. `PROTOCOL_SHA256` was not touched — editing it to pass would
have emptied the gate of the only thing it does.

Codex recovered the exact pre-commit bytes as
`docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.frozen-c921b0d4.md`. **That
snapshot, not the editable document, is the file the gate checks.** It is marked
`-text` in `.gitattributes` because `core.autocrlf` otherwise rewrites it on
checkout (15,685 worktree bytes against a 15,321-byte blob) and it then fails its
own gate.

Verification result: `STAGE_0_BASELINES.csv/.md`, `STAGE_1_LOCAL.md`,
`STAGE_1_LOCAL_AGGREGATE.csv` and `STAGE_1_LOCAL_INTERVALS.csv` are
byte-identical; `STAGE_1_LOCAL_SELECTION.json` agrees in every field but
`score_sha256` (same 0.3517116681118214, same 60-name rejection list, same
`PARITY_WITH_DIRECT_COMPETITOR`). `CELL_METRICS` differs only in `threshold`,
108/138 rows, max delta 1.377e-14, with every metric it feeds identical, so no
prediction flipped. Diagnostics differ at machine epsilon (max relative
9.069e-15) plus timings. `STAGE_1_LOCAL_PER_QUESTION.csv`'s recorded
`83529f8d...` therefore does not reproduce and should be read as
machine-specific, not as a failed check.

Two portability bugs were fixed in passing: `Path.write_text` without
`encoding=` was producing locale-encoded reports (mojibake on Windows, correct
numbers), and the scorer wrote into the very directory it verifies —
`LOCAL_ONLINE_V1_OUT` now redirects it. `LOCAL_ONLINE_CELL_ROOT` remaps the
ProcessBench cell root opt-in, since this checkout holds the cells under
`dataset_cache/repgrid/` rather than the Mac's `cache/localization/processbench/`.
Nothing in `results/local_online_comprehensive_v1/` was written to.

## Paper-exact acquisitions complete and backed up — Step 275

| Run | Traces | Failed | Shards | Drive |
|---|---:|---:|---|---|
| `m2_deepconf_k512` | 15,360 / 15,360 | 0 | 24 / 24, gates all pass | 361 files / 20,189,077,984 B, byte-identical |
| `s1_refrain_full` | 1,000 / 1,000 | 0 | complete | 22 files / 3,185,291,662 B, byte-identical |

DeepConf ran at K=512, a declared deviation: it preserves every budget in the
frozen register (32, 64, 128, 256, 512) at full width and loses only majority
voting over a 4,096-deep pool. `m2_deepconf_full` (297 files /
17,744,439,979 B, K=4096, partial) is kept and **must not be merged with the
K=512 pool**. Both are `fidelity=paper-specified-partial`.

Backups go through `cluster/upload_run_dir.sh <run> [--force] [--status]`, which
encodes the destination `cluster_results/paper_exact/<run>`, a freshness guard,
and a fix for a `pgrep` pattern that matched its own shell and reported uploads
that did not exist. 243 summary artifacts are in
`results/paper_exact_summaries/`.

### What Codex decided, and what is missing

Neither previously proposed row is the advisor-facing one. Stage 1's 0.3517 is
development-selection evidence; Stage 4's 0.3662 belongs to the rejected joint
finalist and stays a named historical row. The direct Localization table shows
all methods on the same official 3,400 ProcessBench IDs with three same-access
rows — ordinary Unified-28, dedicated `family6 + level + step_top5mean`, and
maximum entropy plus the top-five-step locator — with PRM and the critic as
visually separated high-access ceilings.

Building it needs assets we do not have. `codex/fair-paper-exact-comparisons-v1`
is not on our remote and all four of its named files are absent:
`docs/experiments/FAIR_PAPER_EXACT_COMPARISONS_V1.md`,
`spectral_utils/fair_comparisons/{registry,prefix}.py`, and
`scripts/build_fair_paper_exact_comparisons_v1.py`. Recorded as Question 3 in
`HANDOFF_CODEX_2026_08_18.md`.

**Next action, and the only approved work not waiting on Codex**: the offline
DeepConf derivation over the K=512 pool. It needs no GPU and no registry.

## Comprehensive Local/Online transfer decision — Step 273

The frozen four-stage existing-cache cycle evaluated raw-nine/raw-seven,
broad-28, provenance-balanced family-six, historical core-five, eight causal
state types, three locators, six same-matrix fusion paths, and 27 joint
architectures. S1 selected family-six level plus the step top-five locator at
0.3517 Local F1, +0.0014 versus Step-272. S2 selected family-six fast/slow at
0.6020 Online AUROC, +0.0121 versus Step-272. Both intervals included zero.
S3 retained ordinary IU and selected the Global signal for detection/Online
plus the family-six Local locator; its Local/Online deltas were +0.0108/+0.0023,
again uncertain.

The decisive S4 audit used Qwen3-8B and Llama-3.1-8B over all four ProcessBench
families, resampling scorer copies with each source question. Local wins three
of four families but remains parity. Online loses to IU28 in three of four and
breaches the 0.015 margin. At a 5% calibration false-warning target the
finalist covers 11.8% of wrong traces versus 13.0% for Step-272; at 10% it
covers 22.3% versus 21.2% but produces 10.1% audit false warnings versus 7.6%.
All eight cells pass repeat, label-permutation, feature-order, suffix, and
chunk-endpoint audits. PRM reaches 0.7280 Local F1; the critic reaches 0.5895
with 1262/1270 valid scorer rows and eight explicit OmniMath abstentions.

Canonical outputs are in
`results/local_online_comprehensive_v1/REPORT.md`, `REPORT.html`,
`DECISION.json`, `AUDIT.json`, the four stage reports, and machine-readable
per-question/interval/warning/ablation/strata/efficiency tables. The frozen
protocol is `docs/experiments/LOCAL_ONLINE_COMPREHENSIVE_V1.frozen-c921b0d4.md`
with SHA-256 `c921b0d446eebd4611c4426168c30410741997ea2c6d23238e5d22b83e8d1e5b`
(corrected in Step 274 — the editable `LOCAL_ONLINE_COMPREHENSIVE_V1.md` was
revised after the run and hashes to `b5991a89...`).
No new inference, GPU/cluster work, Drive mutation, staging, commit, or push
occurred.

## Token-native three-output architecture result — Step 272

The v2 head screen selected different feature semantics for the three tasks.
Global retained the historical mixed-v2 head at 0.7895 development AUROC; the
best raw mean/tail replacement reached 0.7560, delta -0.0335 with 95% CI
[-0.0600,-0.0065]. Local selected raw token levels at 0.3484 ProcessBench F1;
onset-only and level+onset reached 0.2464 and 0.2685, while the registered
core-five replay reached 0.3330. Online in isolation selected sustained
EWMA/positive-area/persistence at 0.6596 64/128 AUROC versus 0.6370 for IU28,
but the paired interval for the +0.0225 difference crossed zero.

The one/two/three-head cross then made the independent Online head redundant.
Across all twelve ProcessBench scorer-model/family cells, one shared head,
two Global+Local heads, and three independent heads respectively reach:

| architecture | Global AUROC | Local F1 | Online 64/128 AUROC |
|---|---:|---:|---:|
| one shared | 0.6892 | 0.2397 | 0.6009 |
| two Global+Local | 0.7164 | 0.3136 | 0.6075 |
| three independent | 0.7164 | 0.3136 | 0.6009 |

Question-grouped intervals carry all scorer copies together and equal-weight
families. Two heads beat one by +0.0271 [0.0085,0.0449] Global AUROC and
+0.0740 [0.0458,0.1013] Local F1; Online is +0.0067
[-0.0121,0.0260]. Three heads versus two change Online by -0.0067
[-0.0248,0.0126], with identical Global/Local outputs. The simpler two-head
system is selected. The old 0.75/0.25 assumption is not retained; the frozen
development choice is 0.50/0.50 with the peak locator.

Same-matrix controls pass exact `lambda=0` identity. Global DUFS adds only
0.0014 AUROC; Local uniform/DUFS/temporal paths add about 0.0059 F1, with
paired intervals including zero. DUFS costs up to 24.9x the measured uniform
fit path. Ordinary IU-PCR remains the supported fusion mechanism.

Trace-level declaration thresholds were calibrated on the maximum over the
entire monitor horizon. At the 5%/10% targets, observed correct-trace ever-warning
is 3.6%/8.1% and wrong-trace warning coverage is 14.7%/25.0%. Potential
remaining tokens are 428/414, but these are not realized savings because no
forced-closure inference was run. Length residualization leaves AUROC
0.5801/0.5980 at 64/128 versus raw 0.5947/0.6204. The Phase-15 T=1.0 missing-
logsumexp transfer is weak early: 0.5142/0.5555 AUROC at 64/128 despite 0.8368
final AUROC.

The selected system's bottleneck is the historical mixed-v2 Global prefix
recomputation: median per-cell fit and complete three-output scoring are 70.1s
and 47.2s on local CPU. Local drop-one diagnostics suggest spilled and top-k
entropy may be harmful, but they were outcome-opened and are not pruned
post-hoc. A future frozen subset/streaming-implementation cycle is required
before fresh confirmation. No GPU run is justified for DUFS or a third head.

Canonical outputs are in
`results/global_local_online_architecture_v2/REPORT.md`, `REPORT.html`,
`DECISION.json`, and the machine-readable per-question, grouped-interval,
declaration, length, missing-channel, fusion, and efficiency tables.

## Token-native three-output architecture protocol — Step 271 (completed by Step 272)

`docs/experiments/GLOBAL_LOCAL_ONLINE_ARCHITECTURE_V2.md` was frozen before any
v2 candidate outcome was read. It makes three outputs co-primary: completed
answer wrongness (Global), first-error detection/localization (Local), and
causal prediction of the Global target from an unfinished prefix (Online).

The audit found twelve complete ProcessBench telemetry cells: Qwen3-4B,
Qwen3-8B, and Llama-3.1-8B scorer telemetry for the same 3,400 questions across
GSM8K, MATH, OlympiadBench, and OmniMath. All contain token entropy, spilled
energy, log-sum-exp, top-k log-probabilities, step-token spans, first-error
labels, and final-answer correctness. Scorer copies share question IDs and are
treated as repeated measurements. All cells are historically opened, so the
cycle is retrospective development evidence, not fresh confirmation.

The frozen roster starts from nine raw risk-oriented token channels and builds
separate Global mean/tail/extreme reducers, Local level/onset trajectories, and
Online level/EWMA/onset/persistence recurrences. The recurrences are suffix
invariant and token-native; they do not reuse completed-trace means or Step
270's aggregate-of-aggregate monitor inputs. After head selection on only
Qwen3-4B GSM8K/MATH, the harness compares one shared head, a two-head
Global-Local architecture, and three independent heads, including a frozen
search over the historical Global/Local blend. Ordinary IU-PCR is the default;
uniform, DUFS, and temporal Laplacians are low-priority same-matrix controls
after feature and architecture identities freeze.

No inference, GPU/cluster work, large download, Drive mutation, staging,
commit, or push is authorized by this protocol.

## Global-Local-Online IU retrospective result — Step 270

The protocol was frozen before candidate scoring in
`docs/experiments/GLOBAL_LOCAL_ONLINE_IU_V1.md`. It preserves separate
localization and early-ranking panels, question/model grouping, causal prefix
features, frozen confidence signs, IU28 without final length as the primary
Online reference, and the claim boundary “unsupervised scorer with calibrated
decision policies.” A read-only inventory classified 113 cache/artifact
records: 41 causal-prefix-valid, one localization-only, and 71 unusable for
this cycle. No large artifact was downloaded and Google Drive was not mutated.

Three preregistered label-blind Online heads were fitted on the existing
CUSUM/`sw_var` monitor trajectories: current plus running maximum
(`dyn_level4_iu`), current plus positive area/run persistence
(`dyn_persist6_iu`), and current plus slope/recovery (`dyn_change6_iu`). On the
equal-family 64/128-token endpoint, their paired deltas versus IU28 are
-0.0051 [-0.0553,+0.0519], -0.0079 [-0.0663,+0.0639], and
-0.0270 [-0.0979,+0.0561]. Family wins are 2/5, 2/5, and 1/5. Every interval
crosses zero, so none passes the frozen promotion rule. Comparisons with the
DeepConf-w64 proxy also cross zero.

The mechanism result is clearer than the ranking point estimates. The level
and persistence heads have equal-family Spearman 0.993 and 0.949 with the
simple equal CUSUM/`sw_var` magnitude control at 64--128 tokens. Removing the
`sw_var` component lowers the endpoint by 0.026--0.046 depending on the arm,
whereas removing CUSUM is neutral or slightly positive on average. The added
dynamics therefore mostly re-express the saved magnitude signal rather than
contribute a new independent early-warning coordinate.

Localization is unchanged by construction and by bit-identical ProcessBench
and PRMBench score hashes. The fixed anchors reproduce: trajectory-first
ProcessBench macro F1 0.3070, matched Qwen3-8B 0.3035 versus 0.2496, and
PRMBench step AUROC 0.6711. Historical GL-LIU v1 remains 0.3136 versus 0.2571
for Mind the Gap. Same-matrix graph increments remain too small to justify the
extra machinery: global ordinary/DUFS AUROC 0.791369/0.793561 and local
ordinary/DUFS 0.723303/0.723881; the temporal local detector is worse at
0.691528.

**Decision**: retain the frozen Global/Local heads and `iu28_no_length`; close
the current/running-maximum, persistence/area, and slope/recovery transforms
of the existing coarse monitor grid. Do not promote graph regularization,
elapsed length, or a declaration-only variant. Reopen only for a token-native
causal recurrence or genuinely new telemetry/data under a separately frozen
protocol and explicit authorization. The canonical report is
`results/global_local_online_iu_v1/REPORT.md`; its machine-readable decision is
`results/global_local_online_iu_v1/DECISION.json`.

## Joint reasoning localization and early-detection focus — Step 269 (completed by Step 270)

The current application goal is to find the smallest label-free causal
Global-Local-Online IU architecture on the joint performance/compute Pareto
frontier. The two co-primary panels are first-error reasoning localization and
prediction of final answer error from an unfinished causal prefix. They remain
separate metrics, but every method change must be run on both; a gain on one
cannot hide a material regression on the other.

The frozen feature-orientation contract remains in force. Registered streams
are confidence-aligned by `CONFIDENCE_FEATURE_SIGNS_V1`; risk reverses that
direction. This does not assert that every raw feature is intrinsically
monotone: the four recurrently non-monotone raw views are either quarantined or
replaced by frozen mixed-v2 transforms, never duplicated, and no target label
may choose their sign or representation. Final trace length is non-causal;
IU28 without length is the primary online adapter and elapsed prefix length is
an explicit ablation.

Two CPU-only existing-cache screens now anchor the online side. Across 11
materialized cells and five dataset families, IU28 AUROC is 0.648 at 64 tokens
and 0.694 at 128 versus 0.616 and 0.671 for the same-access DeepConf entropy-w64
proxy. Equal-family deltas are +0.024 [-0.005,+0.056] and +0.014
[-0.031,+0.058]: promising parity, not proven superiority and not a reason to
end the comparison. IU28 prefix/final Spearman rises from 0.417 at 64 to 0.659
at 128 and 0.817 at 512; decision agreement rises from 0.640 to 0.739 and
0.880. Held-out declaration remains weak at 0.366 coverage and 0.137
ever-wrong, with 5/11 cells meeting the 10% target.

The causal localization-model follow-up found no early gain from simply
inserting the frozen locator. At 64 tokens, global/fused/`sw_var_peak`/IU28/
DeepConf-w64 AUROC is 0.638/0.635/0.643/0.648/0.616; at 128 it is
0.679/0.678/0.679/0.694/0.671. The completed-trace CUSUM+`sw_var` combination
reaches 0.798, and fused Global-Local beats IU28 at 512 by equal-family +0.066
[+0.043,+0.089]. The next hypothesis is therefore a causal dynamic scorer over
CUSUM and `sw_var` magnitude, persistence, slope, onset/change point, and
stability—not another maximum or frozen locator.

Localization remains a regression anchor: GL-LIU v1 reaches 31.36%
ProcessBench F1 versus 25.71% for Mind the Gap; the later trajectory-first IU
package reaches 30.70% across eight cells and 30.35% versus 24.96% on matched
Qwen3-8B. PRMBench step AUROC is 0.6711 versus 0.6136 for the older step-first
adapter. These are competitive label-free results, not a claim of leadership
over supervised PRMs or large critic models.

The clean existing localization component ablation makes ordinary IU-PCR the
simplicity baseline. Global DUFS-LIU adds only 0.002193 AUROC over ordinary
mixed IU; local DUFS adds 0.000578, while the temporal Laplacian loses on
confirmation. Any next graph test must hold features, IU subspace, reducer,
split, and calibration fixed at `lambda=0` versus uniform/DUFS/temporal arms.
Within uncertainty, the cheaper model wins.

**Completed application action**: Step 270 executed
`docs/research_notes/reasoning_localization_early_detection_optimization_prompt_2026-08-16.md`.
That prompt instructed the cycle to inventory the broader dataset/model/temperature/
protocol archive, bind a two-panel causal regression harness, preregister
non-inferiority and efficiency gates, and optimize on existing caches before
requesting any GPU inference. The resulting canonical report is
`results/global_local_online_iu_v1/REPORT.md`; prior input reports are
`results/early_online_existing_data_v1/REPORT.md`,
`results/early_online_localization_models_v1/REPORT.md`, and
`results/ours_only_localization_v1/REPORT.md`.

## Automatic group-free IU Phase A6 S0a result — Step 268

The three frozen tokenizer/config inputs were restored and authenticated against
the exact official Hugging Face revisions and the registered Git/LFS objects.
The canonical restore manifest is
`c7c1cb3bc3ba68e38207cc6b1a69bedafa963249e5fdd766d07ad08b811b56bb`;
its materialized-byte digest is
`e990e813fd789d4fde3ec4b145c1b6ad89c2e392c653c7ad05f462d1094136d0`.
The self-contained S0a boundary was frozen at code commit `ba983aa` with
boundary SHA-256
`698261d467a3f0a394ef244dafcac67d1cf8a69a9cf2de8888f0ff54678c545e`.

S0a constructed exactly 1,800 reciprocal quartet groups, 6,000 prompt-only
natural-manifest rows, 7,200 inner-fold assignments, and 36 deterministic null
cells. The append-only run produced 7,800 contiguous checkpoints. Authoritative
full replay from the frozen commit returned `PASS_S0A_VERIFIED`; aggregate
SHA-256 is
`2a11b37c4fd649490675e8da4d826084c137a2a072c77ab2fdd5efcad8e8685a`.
Independent review found no schedule, hash, fold, null, contamination, or
firewall blocker. The PopQA artifact contains opaque reserved row indices only,
and the Llama artifact is schema-only.

This is a mechanical construction result, not a detector-performance result.
The next mandatory action is A6-S0b. After the frozen A6/PTNI program reaches
its registered outcome, the prospective
`docs/research_notes/ptni_guided_nrm_research_proposal_2026-08-14.md` is queued
for a mandatory trigger assessment. It will require a separate preregistration
and may run only if PTNI first establishes a valid target direction and leaves
a registered stability, redundancy, or nuisance-transfer limitation; otherwise
it is closed with that reason. It may not rescue, modify, or delay A6.

## Automatic group-free IU Phase A6 S0a implementation — Step 267

S0a is now implemented but not prepared or executed. Its code constructs the
frozen 1,800 reciprocal quartets and three 2,000-row prompt-only natural cohorts,
enforces global identity/content disjointness and derived-answer quotas, audits
the actual chat-template response span, freezes inner folds and merged null
strata, and creates only opaque PopQA/Llama future schemas. During future
preparation, the runner will bind the full local source closure, runtime/thread
settings, exact repo/revision and content-addressed tokenizer/config bytes,
resolved templates, and effective EOS/pad precedence before offline loading.

Resume validates a contiguous canonical checkpoint prefix, replays every prior
local/contextual/global decision into the restored collision registry, and
computes only missing units. The authoritative verifier instead replays the
entire schedule from zero. Output roots, inputs, checkpoint families/files,
reports, and completion artifacts reject symlinks and unmanifested payloads;
interrupted `.tmp` writes recover deterministically.

Five stable-hash adversarial rounds ended in `NO BLOCKERS`. They specifically
closed forged ledgers, JSON numeric-type equivalence, nested target payloads,
revision spoofing, response-world prefix drift, incomplete EOS/pad manifests,
hash-only authorization, symlink escape, and interrupted-run recovery. The
relevant suite is 89/89 green. No real A6 input or result was opened.

## Automatic group-free IU Phase A6 S0/S1 execution contract — Step 266

The reviewed execution contract removes the remaining implementation-time
degrees of freedom before A6 data or sealed seeds open. It binds exact model
and tokenizer revisions; the required future content-addressed snapshot/hash
procedure; the contextual chat-prefix/response-span tokenization rule; the two
900-group quartet populations;
three disjoint 2,000-prompt natural manifests; PopQA identity reservation;
outer/inner folds and null strata; the 16-row S0b shortcut audit; Pythia prompt
NLL; the nested PTNI/control/LO/null selection procedures; all eight simulator
worlds; separated robustness arms; RNG byte contracts; and append-only stage
provenance.

Independent review iterated until the exact pre-freeze body SHA-256
`5c869db42633d04bf4c46110d95de83891c6ca6b10fdf381653b8a618a750615`
received `NO BLOCKERS`. The reviewer specifically verified coefficient-versus-
mean-shift geometry, target/nuisance orthogonalization, balanced diagnostic
labels, target-local transport, LO-family exclusion, control capacity, null
semantics, contextual tokenizer boundaries, deterministic schedules, pristine
held-deletion selection, and correction rescaling. The 55 existing A6
development tests remain green.

This is a preregistration, not an executed boundary. Exact pinned tokenizer
artifacts have not yet been loaded, S0a has not been prepared, and neither S0b
nor S1 may open. Canonical contract:
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_S0_S1_EXECUTION_V1.md`.

**Next action:** implement only A6-S0a, including fail-closed tokenizer/input
manifests, joint identity/fold/stratum audits, and append-only preparation and
verification. Freeze and independently review the implemented source/runtime
boundary before any response telemetry or sealed simulator seed.

## Automatic group-free IU Phase A6 development base — Step 265

The reciprocal construction now has typed arithmetic, relational aggregation
and lookup, and finite set/counting tasks; exact integer/rational evaluators;
four reversible prompt renderings; deterministic short/certificate responses;
an independent verifier; a full closed natural-answer parser; and append-only
local/global rejection ledgers. The canonical 900+900 development schedule
replays exactly and rejects semantic-task or raw-prompt-content reuse across
families, folds, and populations. It does not yet use the real pinned Qwen and
Llama tokenizers, so it is not an S0a boundary.

The target-local feature layer enforces the exact 30-name mixed-v2 contract,
99% presence rule, complete quartet admission, and fit/deploy transform
identity. The PTNI core implements factorial target/nuisance/interaction
moments, all-render and leave-one-nuisance fits, trace-scaled ridge direction,
nominal-roster transport, exact-duplicate quotient IU, covariance-orthogonal
correction, and exact IU fallbacks. Duplicate-aware deployment must first bind
one full target matrix; a failed equality preflight yields one fixed ordinary-
IU affine artifact rather than a row- or chunk-dependent rule.

Pre-telemetry adversarial review found and repaired several hidden failures:
mutation metadata in the AST hash allowed cross-family semantic duplicates;
group-bound prompt IDs allowed raw prompt reuse; positional score constructors
silently disabled duplicate fallback; the inherited empirical-rank transform
fit and deployed different coordinates; and answer-in-prompt used substring
matching. The A6-only suite is 55/55 green and the relevant A5/A4 regression
suite also passes. A full 1,800-group executability proxy using the available
local Qwen2.5 tokenizer for both callbacks now passes after compacting the
typed certificate and isolating mutable notation atoms at token boundaries;
the exact pinned Qwen3 and Llama tokenizers remain a mandatory S0a gate.

This is a development-base commit, not permission to collect telemetry or run
sealed seeds. Step 266 now freezes the exact S0a/S0b/S1 execution contract for
the remaining manifests, tokenizer boundary, shortcut/matching audit,
manifest-bound duplicate preflight, feature canonicalization, nested
selection/controls/nulls/LO suites, eight-world simulator, and append-only
stage provenance. Those components are still unimplemented; their implemented
source/runtime boundary requires a fresh no-edit review before execution.

## Automatic group-free IU Phase A6 protocol — Step 264

A6 now has one explicit self-supervised route rather than a menu of adaptive
interventions. Reciprocal 2x2 prompt-response crossovers hold each deterministic
answer byte-identical while making every prompt and response marginal exactly
50/50 correct/incorrect. Three semantic domains, three AST mutation families,
three nuisance render families, and short/certificate grammars define 900
balanced Qwen source groups plus a disjoint 900-group Llama audit. A closed
answer parser and independent typed-AST evaluators preserve the ordinary
answer-correctness ontology and the sealed PopQA boundary.

PTNI-IU estimates factorial target, nuisance, and target-by-render moments over
the retained atomic roster, derives one nuisance-whitened risk direction, and
projects it covariance-orthogonal to target-local IU-PCR. The trust path always
contains exact IU. All selection uses nested source-group folds and one unique
full-Qwen arm; Llama is pass/close only. The protocol freezes class-specific
matched controls, conditional sign permutation, two non-p-value placebos, a
nonvacuously activated nuisance-as-target negative control, LO-family transfer,
complete-block admission, and exact robustness/fallback gates.

Execution is staged: S0 mechanical construction and shortcut/matching freeze;
S1 eight-world sealed simulator; S2a Qwen quartet premise; S2b held Llama
quartet audit; S2c untouched greedy Llama errors; S3 one-way retrospective
answer-correctness veto; S4 sealed PopQA confirmation. Failure at any stage
forbids rescue and advances to A7. Eight independent adversarial passes were
required; the final reviewer verdict is `NO BLOCKERS`. Canonical protocol:
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A6_V1.md`.

**Next action:** implement A6-S0/S1 from the frozen protocol, add fail-closed
tests and append-only boundaries, and obtain a new independent no-edit review
before any sealed simulator seed or response telemetry is opened.

## Automatic group-free IU Phase A5 result — Step 263

The exact committed S1a boundary ran world-8 seeds `521600..521699` with no
source change. All 100 append-only checkpoints match the aggregate and carry
the frozen boundary hash. Ninety-eight repetitions were usable; seed 521639
had no usable penalty arm and seed 521691 failed a held-mixture fit. There were
zero implementation-invalid failures. The frozen all-repetitions rule gives
the formal verdict `CLOSE_NUMERICAL_NONCONVERGENCE`.

The independent adversarial audit then examined the 98 usable repetitions
without changing the verdict. The final direction preferred target over
nuisance in 62/98 runs and the correction did so in only 25/98, versus the
registered 90/100 requirement. Candidate-minus-IU AUROC averaged -0.038484;
the exact 20,000-draw bootstrap interval was [-0.047495,-0.029659], with no
nonnegative draw. Even awarding both unusable runs success cannot reach either
count gate. Alpha 1 was selected in 46 runs and caused -0.080974 mean harm,
showing that likelihood systematically trusts the stronger planted nuisance.

This is a substantive identifiability failure rather than a route that should
be rescued by numerical changes. A5 is closed at S1a. Do not run S1b, transfer
the 23 real A5 caches, or inspect retrospective labels. No real cache or label
was accessed. Canonical artifacts:
`results/automatic_group_free_phase_a5_v1/`.

**Next action:** preregister and independently audit A6 before code. A6 must
obtain information unavailable to marginal `P(X)` by separating verified
target-changing interventions from nuisance-only interventions, preserve an
affine one-pass deployment head, and test transfer to natural hallucinations.

## Automatic group-free IU Phase A5 implementation — Step 262

The bounded A5 estimator and target firewall are implemented. The numerical
core uses an SPD-preserving fixed-support precision optimizer, equal-covariance
latent mixtures, IU covariance orientation, and exact affine IU fallback. The
17-feature source builder reproduces the frozen A0 admitted population, rejects
target-like public payloads, globally groups repeated prompt content, keeps one
deterministic response, excludes trace length from fitting, and verifies all 23
raw-source hashes before unpickling.

Development plus adversarial review forced important pre-seal repairs:
confidence signs now propagate through every candidate/control fit; redundant
coordinates use immutable-name mean/contrast coordinates; held IU remains exact
even if a graph-training equality breaks; grouped feature deletion stays
full-rank; alpha/penalty use a paired one-standard-error rule; and random graph
controls are ranked at their actually selected/deployed alpha. The registered
near-duplicate development check now preserves selected output rather than only
a conditional fixed-alpha diagnostic.

Execution is staged. S1a opens only the nuisance-dominant world-8 hard stop. A
PASS may authorize a separately preregistered/reviewed S1b boundary, and only a
later S1b PASS may authorize the real-data S2 boundary; earlier outcomes may not
change the estimator, grids, gates, or their interpretation. The S1a runner is
append-only, resumable through immutable per-seed checkpoints, verifies exactly
100 ordered seeds, distinguishes registered numerical closure from invalid
implementation errors, and cryptographically binds result, records, protocol,
source closure, and numerical runtime.

The independent reviewer reports `NO BLOCKERS`; all 53 relevant tests pass. A
development-only nuisance seed (510108) selected alpha 1, preferred nuisance in
both final/correction directions, and lost 0.0867 AUROC to IU, suggesting the
sealed early-stop may close A5 as intended. This evidence cannot substitute for
the sealed 100-seed gate. No real cache or retrospective label was accessed.

## Automatic group-free IU Phase A5 protocol — Step 261

The only genuinely new continuous weak-supervision route retained after the
prior-results audit is an item-level, equal-class-covariance two-component
mixture with sparse within-component precision. Its discriminant is affine and
its correction is covariance-orthogonal to the frozen IU-PCR score. A fixed
alpha path contains exact IU fallback. This differs from the already closed
sparse marginal covariance, inverse-weighting, GMM anomaly/NLL, DEEM,
higher-moment, and labelled-head routes.

The semantic limit is explicit: `P(X)` cannot distinguish correctness from an
equally distributed nuisance bit. A5 inherits IU-PCR as its target-semantic
anchor and includes a bit-identical observational-equivalence audit. A result
can therefore support an IU-conditional correction, never claim that
unlabelled likelihood identifies hallucination by itself.

Independent adversarial review required three rounds of correction before the
protocol was accepted. The final boundary purges repeated prompts globally
across environments by canonical item/content components, closes if any nested
item-disjoint fold becomes unusable, and uses one deterministic response per
prompt for structural likelihood. It rebuilds a target-free 17-feature raw
tensor rather than opening the old label-bearing, whole-cell-standardized
archive. `trace_length` is a forbidden-fit sidecar used only for a stronger
environment-local nonlinear/censoring confound gate.

The nested algorithm independently selects alpha for the sparse, diagonal, and
random-support controls; includes a capacity-identical sparse `alpha=0`
baseline; refits complete pipelines under sparse-one-Gaussian,
diagonal-mixture, and environment-reassignment nulls; and defines a separate
one-way 24-cell label veto after a single score bundle is frozen. Labels may
only PASS/VETO that primary, never choose a model.

Execution begins with development-only implementation tests, followed by a
hash-frozen 11-world sealed synthetic suite. The nuisance-dominant world is an
early hard stop: if the method recovers a stronger prompt-shared nuisance
instead of the target despite a target-valid IU anchor, A5 closes before the
multi-gigabyte real caches are transferred. The independent reviewer issued
`NO BLOCKERS`; no A5 result has been run or opened.

Canonical protocol:
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_PHASE_A5_V1.md`.

## Automatic group-free IU Phase A4 result — Step 260

The intentional run verified the frozen 3,400 x 3 x 29 boundary and accessed no
correctness or step-error target. CorrCA achieved Qwen repeatability 0.997881
and held-Llama structural correlation 0.955465. The strongest paired baseline,
chosen solely inside every outer-training fold, was `single:1 = trace_length`
and reached 0.966908 on held Llama. CorrCA minus that baseline was -0.011444
with 95% interval [-0.016036,-0.009034], so the mandatory material gate failed.
The structural verdict is `CLOSE_SHARED_REPEATABLE_COMPONENT_PREMISE`; the
pre-frozen detector verdict remains `CLOSE_NO_TARGET_CONTRAST`.

The formal positivity, pair-null, leave-one-subset, stability, and confound
gates passed, but independent post-held review invalidated the tempting
non-length interpretation. `trace_length` is exact generated-token count; the
two Qwen views have identical counts for all items. The nuisance basis used
log-count terms, squared log-count terms, and ridge shrinkage to predict the
standardized linear count, so a deterministic length residual remained.
CorrCA's trace-length loading was 0.997897--0.999279 across folds. Trace-only
reproduced the strongest baseline (Qwen/Llama 0.999999/0.966908); deleting only
that term from each frozen loading, without refit or selection, reduced the
metrics to 0.990032/0.866653. The formal confound gate and coarse length-decile
null cannot exclude this mechanism because they reuse or coarsen the same
length control.

The canonical result is in `results/automatic_group_free_phase_a4_v1/`; the
post-held diagnosis is explicitly isolated in
`POST_HELD_TRACE_LENGTH_DIAGNOSTIC.json`. It does not change either frozen
verdict or create a rescued candidate. Proceed to A5 and preserve A4 as a
bounded negative result.

## Automatic group-free IU Phase A4 boundary — Step 258

The exact 3,400 ProcessBench triples change the scorer while keeping each
response fixed. A4 can identify repeatable/shared versus scorer-sensitive
telemetry structure, but cannot identify hallucination. The detector outcome
is consequently frozen before execution as `CLOSE_NO_TARGET_CONTRAST`; the
experiment has a separate pass/fail gate for whether a shared repeatable
component is useful enough to inform A5/A6. It does not identify the
complementary scorer-specific residual as nuisance.

Independent preregistration review caught two initially vacuous choices. First,
fitting on Qwen views of an item and testing on Llama's view of that same item
would be transductive. Exact-content item groups are now split before every
transform, residualizer, or component fit. Second, scorer token lengths are
nearly identical (Qwen4/Qwen8 are exact; Qwen/Llama correlations exceed 0.99),
so a simple item shuffle would mostly test length matching. The primary pipeline
now cross-fits a feature-level text/length nuisance model before CorrCA and uses
subset-and-length-conditional derangements with full refitting.

The second adversarial pass additionally prohibited concatenation of
uncalibrated outer-fold scores, replaced an ill-posed text-prediction
correlation gate with direct held score-confounding diagnostics, and narrowed
the claim from a full decomposition to one shared repeatable component. It also
forced exact 29-feature, covariance, baseline, bootstrap, and null-aggregation
definitions.

The estimator, ridge grid, nested selection, five paired baselines, raw-z
sensitivity, held-family and leave-one-subset checks, bootstrap, two shuffle
nulls, and material-effect gates are frozen in the Phase A4 protocol. No
correctness or step label was opened. Step 260 records the completed result;
this section preserves the pre-run boundary rationale.

The pre-run implementation audit found two blockers before the intentional
structural run: the training-pair null left outer-held Qwen pairs intact, and
the scalar confound model trained on cross-fitted residual coordinates while
held scores used full-fit coordinates. The fixes now derive nested/outer
shuffle strata from training data, break both train and held Qwen8 pairs, and
train the scalar diagnostic on full-fit training residuals matching held
coordinates. The boundary verifier also binds the test source and checks fold,
roster, shape, ID, and execution semantics. The reviewer accidentally started
the old runner for roughly 11 seconds, but it emitted no metric and wrote no
artifact; this is recorded in `HISTORY.md`. The repaired boundary was then
regenerated and committed as `ae19e20` before the deliberate run.

## Automatic group-free IU Phase A2/A3 — Step 257

The primary A2 run kept all 30 atoms. Missing covariance entries were completed
from training folds only, observed held-out blocks were preserved exactly, and
evaluation scored only feature pairs actually present in each held-out cell.
The comparator used the same recovered block sizes, number of covariance
mechanisms, and ridge in pooled-PCA orientation. All reconstruction summaries
use equal environment mass.

Missing-aware JBD reached MSE 0.028700 versus 0.032864 for the matched PCA
control. The paired delta was -0.004164 with 95% interval
[-0.012164, 0.000838], so the registered improvement gate failed. The final
all-cell structure had blocks [19,3,1x8] and 204 mechanisms, but outer-fold
solutions ranged from dominant 15--19 coordinate blocks to one 30-singleton
solution. Removing `seiclr_triviaqa_opt30b` changed mechanism rank from 204 to
330, giving the minimum ratio 0.618 below the 0.70 gate. Under a train-only PSD
stationary null preserving missingness and sample counts, JBD was slightly
worse than matched PCA (+0.0000461 [0.0000011,0.0001153]), so the original
advantage disappeared cleanly.

The 17-feature complete-core diagnostic also failed: JBD minus matched PCA was
-0.001619 [-0.005369,0.001801], and minimum mechanism overlap 0.7668 missed
the 0.80 gate. A known-block simulator still passed after its own comparator
was corrected to equal capacity (0.066751 versus 0.073461), showing the
implementation can recover a favorable identifiable world. A2 therefore
closes this concrete missing-aware JBD route as a target basis, not every
possible JBD algorithm. A3 is not built because both A1 and A2 failed their
premises.

Because A2 failed before any detector score, orientation, or trust rule was
constructed, its detector-only exact/near-duplicate, affine reconstruction,
and zero-evidence fallback gates were not run. They remain mandatory before
any future promotion; they are not counted as passed.

An independent adversarial reviewer found and forced correction of the
original unmatched PCA capacity, an indefinite covariance null, nonorthogonal
ridge geometry, A1's feature-pair weighting defect, incomplete-roster scope,
and stale hashes. The final A0/A1/A2 artifact hashes and source hashes verify;
the relevant 26-test module suite passes.

**Next action:** execute A4 as a scorer-nuisance decomposition premise test.
The exact 3,400 ProcessBench triples keep responses fixed, so scorer invariance
cannot by itself identify hallucination. A4 requires a predeclared
target-changing contrast/anchor plus held-model and item-pair-shuffle
falsification before any detector candidate can be promoted.

## Automatic group-free IU Phase A1 — Step 256

The A1 run used no new correctness labels beyond the frozen mixed-v2 input
contract and used a hash-defined split of
16 structural-training and seven structural-audit cells. Ranks, ridge values,
interaction form, and the mechanical/learned blend were selected only through
leave-one-environment-out masked covariance reconstruction inside the 16
training cells. `FROZEN_SELECTION.json` was written before the audit cells
were evaluated.

The selected hybrid used rank 6, a full interaction covariance, ridge 0.1, and
a 25% mechanical factorial / 75% anonymized PCA projector. Equal-environment
audit MSE was 0.032009 versus 0.034704 for pooled PCA, 0.080957 for the best
hard factorial basis, and 0.045580 for the pooled mean. The paired grouped MSE
delta versus PCA was -0.002695 with 95% interval [-0.005845, 0.000282]; the improvement is
promising but not established. The hybrid decisively beat the median random
partition and the fifth percentile of 32 cardinality-matched random controls.

Stability and exact invariances passed: minimum leave-one-training-cell
projector overlap was 0.9428, feature-order permutation error was 4.83e-15,
repeatability error was zero, and exact-duplicate mass error was zero. The
near-duplicate gate failed: appending a rho=0.999 measurement gave the pair
3.009 times the original measurement's combined soft-quotient mass, above the
frozen 1.10 limit. Therefore A1 is closed as a detector basis and may not enter
A3. Its bounded positive lesson is that weak mechanically derived factorial
structure regularizes pooled PCA; its quotient rule is not robust enough.

Canonical artifacts: `spectral_utils/factorial_measurement.py`,
`scripts/automatic_group_free_phase_a1.py`,
`scripts/test_factorial_measurement.py`, and
`results/automatic_group_free_phase_a1_v1/`.

**Next action:** execute A2 directly on raw 30-atom residual covariance
matrices, comparing pooled covariance, approximate joint diagonalization, and
joint block variants with train-only held-out-environment selection and
environment-shuffle controls.

## Automatic group-free IU Phase A0 — Step 255

The audit found 30 canonical mixed-v2 features across 23 source environments;
17 are present everywhere and feature-pair coverage ranges from 8 to 23 cells.
Six source cells retain fewer valid bundle rows than manifest attempts, with a
minimum retention of 19.8% in `sciq_llama8b`. A1/A2 must therefore preserve
the valid bundle population and equal-environment weighting rather than
silently restoring filtered rows or weighting cells by candidate count.

Feature streams are taken from extractor-owned registries; the operator
taxonomy is a handwritten, label-blind mapping, while function signatures
record implementation provenance. The implementation deliberately does not import
`specrage_views` or `FEATURE_TO_VIEW`. Exact ID, problem, and step-content
matching established a three-view self-supervision surface of 3,400 fixed
ProcessBench responses over four subsets and three scorer models, with complete
telemetry in every view. This is the preferred A4 paired surface because it
requires no semantic matching and changes the scoring model while holding the
response fixed.

The confirmation boundary is `popqa-gemma3-4b-it-confirmation-v1`, with a
sealed Qwen3-4B access fallback and normalized token-boundary alias grading.
PopQA is an unseen dataset family; the broad Gemma family is not new. Labels
remain unopened and collection is still forbidden until a finalist and all
target, sign, and trust rules are frozen. Canonical artifacts:
`results/automatic_group_free_phase_a0_v1/`.

**Historical next action (completed in Steps 256--257):** A1 and A2 were
implemented and closed; see the current Step 257 section above.

## Automatic group-free IU program — Step 254

The deployment boundary is gray-box, one-pass, affine, and based on the frozen
mixed-v2 telemetry. S1 uses no new correctness labels and no `FEATURE_TO_VIEW`
or manual equivalent; cached cross-model material, environment identity, and
code-registered feature-DAG metadata are legal at calibration. Family
NRM remains a frozen comparator and may not choose, sign, or tune a candidate.

Immediate work is Phase A0: derive the feature-DAG registry, audit all source
environments and exact cross-model item pairing, build the identifiability
simulator, and reserve an untouched confirmation surface. Do not begin a new
eigenspace, grouping, or trust sweep before this audit is complete.

Canonical contract:
`docs/experiments/AUTOMATIC_GROUP_FREE_IU_RESEARCH_PROGRAM_V1.md`.

## Fixed application pipelines — Step 253

The RAG object is `X[i,t,c,f]`: response, answer token, evidence condition,
and shared token feature. The fixed LOO head uses six blocks per feature
(full, no-context drop, LOO maximum drop, top-two mean drop, positive mean
drop, and negative LOO standard deviation). The no-context fallback uses full
and full-minus-no-context blocks. Both are fitted without labels. On RAGTruth
test, the resulting score reaches answer AUROC 0.7276 [0.7041,0.7506], sentence
AUROC 0.6893 [0.6675,0.7129], and token AUROC 0.6586. On the exact local
400-response GASP cohort, sentence AUROC is 0.6598 [0.6289,0.6916] versus
0.6556 [0.6182,0.6887] for the local GASP reproduction; the published
0.673 remains a non-exact-ID paper reference. On the two LOO tasks, the new
task-macro answer AUROC is 0.7157, nearly identical to the earlier response-
only Original-30 LOO IU result 0.7164.

The reasoning object is `X[i,t,f]` on the full trace. IU-PCR is applied before
reducing tokens to steps; step risk is the maximum token risk. ProcessBench
uses a registered global/local answer gate and calibration-half operating
threshold. Eight-cell macro F1 is 0.3070 versus 0.2571 for the Mind the Gap
control. On the paper-aligned Qwen3-8B four-subset population, the comparison
is 0.3035 versus 0.2496; the paired delta is +0.0539 [0.0316,0.0773]. Frozen
GL-LIU is 0.3125, while the 72B critic and supervised PRM remain much higher.
On PRMBench, trajectory-first IU reaches step AUROC 0.6711, improving the old
step-first adapter at 0.6136 but remaining below Qwen2.5-Math-PRM-7B at
0.7983.

**Decision**: freeze these two application packages for advisor review. Do
not add another fusion variant inside this experiment. The fair claims are
that the RAG pipeline is competitive with the local GASP reproduction and the
reasoning pipeline is competitive with the label-free Mind the Gap control;
neither matches the supervised/large-judge ceilings. RAGTruth remains
exploratory because labels had been opened before this run.

Canonical artifacts: `spectral_utils/fixed_application_pipelines.py`,
`scripts/fixed_application_pipeline_experiment.py`,
`scripts/test_fixed_application_pipelines.py`, and
`results/fixed_application_pipelines_v1/`.

## Atomic orientation diagnosis + route closures — Step 252

An independent session reproduced the frozen Atomic Projector exactly
(17 eigenvalues to 4 decimals, transfer deltas to the third decimal), then
measured the failure and closed the follow-on routes. Everything is in
`docs/research_notes/atomic_orientation_reply_2026-08-13.md` with scripts,
logs and JSONs in `results/atomic_orientation_diag_2026-08-13/`. Headlines:
band mass of the target 3.0% (63.6% on the rejected λ=2.04 mode); anchors
anti-aligned after projection; supervised transport ≈0 on originals at every
trust (in-cell ceiling +1.17pp reproduces); refined-partition NRM v0 negative
everywhere with the family control reproducing +0.277/+0.557/+1.580 exactly;
label-free partition selection uninformative (best pick +0.52 vs provenance
+0.93). The transportable label-free direction is the family energy contrast,
already captured by deployed NRM-CS-IU. The b-coupled γ̂3 channel is an
orientation instrument under the implemented `{1,b}` residualization (pooled
cos +0.76, all nine within-family signs), but the memo specifies the stricter
`{1,b,φ2}` Gram–Schmidt construction. That mismatch must be resolved before
the channel or its sign bit is promoted.

## Atomic NRM grouping audit — Step 251

The six provenance families were audited as the last hand-picked prior in
NRM-CS-IU. Before reading candidate metrics, the atomic implementation froze
the source roster, 17 fully covered atoms, 13 exclusions, a 1,000-permutation
null interval [0.934489,1.070026], the two-mode neutral projector, an
inverse-absolute-dependence anchor, target scale `1/sqrt(17)`, direction, and
all input/code/calibration hashes. It uses the same mixed-v2 features, no
labels in fitting, no new inference, and reconstructs exactly as one affine
feature-weight rule. Feature-order permutation error is 8.88e-16 and the
minimum leave-one-cell direction cosine is 0.975505.

The frozen Atomic Projector failed decisively: original LOFO -0.667pp versus
family NRM +0.277pp; Llama ProcessBench -1.106 versus +1.580pp; Qwen
ProcessBench -1.305 versus +0.557pp; SemGrad -4.216 versus +1.310pp. Direct
atomic-minus-family intervals exclude zero in all four domains. Equal-anchor,
single-mode, learned-group, refinement, coarsening, and 50 random
cardinality-matched partition controls do not recover the family result.

The supervised ceiling rules out the trivial explanation that grouping merely
preserves target information. With 30 class-balanced held-out splits per cell,
atomic residual heads beat family heads at all four fixed priors. At prior 0.3,
atomic improves IU by +1.298pp versus +0.721pp for families; direct difference
+0.577pp [+0.102,+0.910]. Atomic target signal exists, but covariance-null
geometry does not identify its useful direction without the provenance prior.

The primary-literature audit reaches the same boundary. Marchenko--Pastur,
Horn/Dobriban parallel analysis, spiked-covariance phase transitions, and
Davis--Kahan perturbation theory support a null **subspace** and reject an
arbitrary closest-to-one eigenvector in higher dimension. They do not attach
hallucination semantics to that subspace. No untouched target labels were
spent after the candidate failed its retrospective gate, and there was no
post-label pivot.

**Decision**: retain frozen family NRM-CS-IU v1 unchanged. State exactly that
its `FEATURE_TO_VIEW` / `VIEW_ORDER` aggregation is an empirical measurement-
provenance assumption. Reopen de-grouping only with a genuinely new label-free
target-orientation principle, not another null-bulk selector.

Canonical artifacts:
`SPEC_ATOMIC_NEUTRAL_RESIDUAL_PROJECTOR_CS_IU_CANDIDATE_V1.md`,
`docs/research_notes/atomic_nrm_grouping_audit_2026-08-13.md`,
`docs/research_notes/atomic_nrm_null_spectrum_literature_2026-08-13.md`,
`spectral_utils/atomic_neutral_residual.py`,
`results/atomic_nrm_structural_audit_v1/`,
`results/atomic_nrm_retrospective_controls_v1/`, and
`results/atomic_contribution_supervised_ceiling_v1/`.

## Neutral residual mode confirmation — Steps 247--250

The HARP memory was recovered at
`docs/research_notes/harp_subspace_inspiration_2026-08-12.md`. HARP itself is
supervised and white-box; the reusable idea was to separate target and nuisance
subspaces before classification rather than equating the strongest shared
factor with hallucination.

The first contribution-space result established feasibility. Family
contributions exactly reconstruct IU-PCR, and an IU-orthogonal supervised head
improved 23 development cells. A single global six-family teacher then kept the
same coefficient sign in all eight leave-one-dataset-family-out folds and
transferred to Qwen/Llama ProcessBench and both SemGrad datasets. This proved a
general target correction exists, but the teacher is not deployable because it
uses correctness labels.

The first label-free proxy, CB-CS-IU, was rejected on independent SemGrad:
equal-dataset delta -0.767pp, driven by TruthfulQA at -1.708pp. The problem was
the proxy, not the contribution representation: the frozen supervised teacher
improved the same SemGrad targets by +0.646pp.

NRM-CS-IU is the replacement. Across unlabelled source cells it averages the
six-family standardized residual covariance pairwise over present families,
selects the eigenmode closest to eigenvalue one, and orients it toward the
equal-family confidence anchor. The selected eigenvalue is 1.035378 and its
direction is `[+0.094,-0.114,-0.674,+0.715,+0.112,+0.026]`, matching all six
supervised-teacher signs. Target scoring is `standardized_IU + q/(G*sd(q))`.
It reads no labels, adds no feature or inference pass, and maps exactly to one
effective weight vector over the existing mixed-v2 matrix plus an intercept.

Retrospective transfer is consistently positive: original LOFO +0.277pp
[+0.016,+0.533], Qwen ProcessBench +0.557pp, Llama ProcessBench +1.580pp, and
SemGrad +1.310pp. Frozen HLE/Qwen2.5-72B was directionally positive at
+0.345pp but inconclusive, CI [-0.898,+1.628], because only 68/2,158 answers
were judged correct.

The higher-powered frozen PRMBench/Qwen3-8B confirmation used 6,966 complete
reasoning responses after excluding exactly the three readiness-identified
alignment defects. Scores and all code/data/calibration hashes were frozen and
verified before `classification` was read. IU AUROC was 0.720602; NRM was
0.725206: **+0.460pp [0.068,0.841]** under 5,000 paired source-group bootstrap
draws, with `P(delta>0)=0.9892`. All five pre-registered gates passed. Six of
nine error-class contrasts improve; three regress slightly, which remains a
documented heterogeneity limit. This is response-level correct-versus-error
evaluation, not PRMBench's official step-level metric.

Canonical artifacts:
`SPEC_HARP_GLOBAL_CONTRIBUTION_TEACHER_V1.md`,
`SPEC_NEUTRAL_RESIDUAL_MODE_CS_IU_V1.md`,
`SPEC_NEUTRAL_RESIDUAL_MODE_PRMBENCH_CONFIRMATION_V1.md`,
`spectral_utils/contribution_subspace.py`,
`results/harp_global_contribution_teacher_v1/REPORT.md`,
`results/neutral_residual_mode_cs_iu_v1/REPORT.md`,
`results/neutral_residual_mode_hle_v1/REPORT.md`, and
`results/neutral_residual_mode_prmbench_v1/REPORT.md`.

## Latest cluster-data audit — Step 242

About 2.3 GB of new benchmark artifacts were synchronized from Google Drive
to `dataset_cache/four_localization/`. The RefChecker upload is now complete
and byte-matches Drive: 10,733 fixed claims across zero-, noisy-, and
accurate-context settings. Its open NLI checker reaches 0.6932 three-way
accuracy and 0.5805 macro F1. The older `RUN_SUMMARY.md` was written before
this final upload and its 2-of-3 RefChecker table is stale; the final manifest
is authoritative.

The full LettuceDetect RAGTruth ceiling, exact-JSD GASP artifacts, PRMBench
PRM scores and Qwen3 telemetry, and all three complete four-subset ProcessBench
comparators are present. The Qwen2.5-72B critic reaches macro F1 **0.5940**
(0.7482/0.6070/0.5057/0.5152). HLE has all 2,158 generations and an interim
Codex judge, but still lacks the paper-faithful GPT-4o judge; it remains
restricted evidence rather than a paper-faithful headline result.

## Original-30 evidence-aware RAGTruth result — Step 244

The intended RAG experiment is complete. It keeps the exact 30 mixed-v2
features and extracts them separately from `full`, `noctx`, and every observed
`loo_j` trace. All 30 are available in every condition on 624 development and
1,800 test LOO responses. The full-context arm reproduces the earlier
mixed-v2 score to `1.11e-15`; no feature was replaced or imputed.

The main finding is cross-task repair, not a pooled leaderboard win. Full-only
DUFS-LIU scores 0.7698 on QA but 0.4345 on Data-to-Text. LOO IU-PCR reaches
0.7178 and 0.7150 respectively, improving task-macro AUROC from 0.6002 to
0.7164: **+0.1163 [0.0795,0.1544]**. GASP-top50 remains slightly higher at
0.7225 task-macro, with no significant difference. Hybrid is similar and adds
complexity without a clear gain.

The 0.8013 pooled no-context DUFS value must not be treated as the headline:
its Data-to-Text AUROC is only 0.5851, and the aggregate rewards task
composition. Report pooled, task-macro, task-standardized, QA and Data-to-Text
results together.

DUFS contributes only in the compact no-context matrix: +0.0065
[0.0047,0.0085] task-macro over IU-PCR. It contributes approximately zero in
LOO and Hybrid. Evidence-block permutation causes a 13.5-17.6 point macro
loss, proving that within-response condition pairing matters even though the
large Laplacian solve adds little.

**Next decision**: carry Original-30 LOO IU-PCR as the simple confirmation
hypothesis. Compare it with GASP-top50, EC-IU-PCR, and compact no-context
DUFS-LIU on a new benchmark or scorer. Do not tune another RAGTruth variant.

Canonical artifacts:
`results/ragtruth_mixed_v2_evidence_aware_v1/REPORT.html`,
`results/ragtruth_mixed_v2_evidence_aware_v1/METHODS.md`, and
`docs/experiments/RAGTRUTH_MIXED_V2_EVIDENCE_AWARE_V1.md`.

## RAGTruth Evidence-Contrast result — Step 243

The first preregistered RAG application experiment is complete. It used fixed
RAGTruth answers scored by Qwen2.5-1.5B under full context, no context, and
leave-one-chunk-out context. Development contained 900 responses; test
contained 2,700 responses. The LOO test cohort contains 1,800 QA/Data-to-Text
responses and 12,958 sentences. Summary remains in a separate no-context
cohort. Scores were fitted without labels and hashed before labels were opened;
all uncertainty resampled complete `source_id` groups.

On test LOO sentences, EC-DUFS-LIU reaches **0.7026 AUROC** versus **0.6721**
for GASP-top50, a paired **+0.0305 [0.0237, 0.0378]**. The direction is positive
in QA and Data-to-Text. However, EC-IU-PCR reaches **0.7031**, and the registered
DUFS-LIU difference is **-0.00048 [-0.00061,-0.00034]**. Ungated and permuted
graphs are also tied with IU-PCR. The gates keep an effective 13.32 of 14
features, and the graph barely changes the IU weights.

**Conclusion:** the Evidence-Contrast feature construction is useful, but the
DUFS-gated Laplacian is not the reason. Treat this as a **feature-contract
success and DUFS/Laplacian mechanism failure**. The RAG application candidate
is now EC-IU-PCR, with EC-U-PCR and GASP-top50 as controls. Do not tune another
graph on the opened RAGTruth test set.

The main remaining risks are explicit. Conflict hallucinations score only
0.6256 AUROC versus 0.7438 for baseless information. Response-level pooled
performance is strongly affected by task/chunk composition: nuisance
residualization changes AUROC from 0.7484 to 0.6481, and Data-to-Text response
AUROC is below GASP-top50 even though the pooled response result is higher.
The next RAG test should therefore freeze EC-IU-PCR and evaluate transfer,
conflict handling, and response-level nuisance robustness on new data.

A separately hashed post-hoc audit reconstructed the old 30-feature intrinsic
mixed-v2 DUFS-LIU response detector. Its pooled test AUROC is 0.7629, but this
hides a severe task reversal: 0.7698 on QA and 0.4345 on Data-to-Text. The EC
response detector reaches 0.7056 on Data-to-Text. The old pooled value cannot
be used as evidence of robust RAG grounding, and the post-hoc result does not
enter the registered decision.

Canonical artifacts:
`results/ragtruth_evidence_contrast_v1/REPORT.html`,
`results/ragtruth_evidence_contrast_v1/REPORT.md`, and
`results/ragtruth_evidence_contrast_v1/METHODS.md`.
## White-box analysis addendum — NRM-CS-IU and registered v2

## Latest development — white-box NRM-CS-IU addendum (2026-08-13)

The frozen NRM contribution-space rule was transferred to the v2 white-box
matrices without mutating the registered v2 result. Four architecture-relative
depth quartiles define the primary contribution families; the secondary
`lens-96` arm uses its twelve fixed module-by-metric groups. Every target is
calibrated from source cells only. Because the roster is not fully crossed,
leave-dataset-out and leave-model-out are reported separately, with LOCO as a
sensitivity analysis. Fit sees no correctness fields, the reconstructed IU
score matches frozen v2 within `1e-10`, and NRM scores are hashed before the
evaluator opens labels.

The result is transfer-sensitive rather than a robust win. Depth NRM under
leave-dataset-out reaches AUROC/AUPRC 0.6182/0.4769 versus 0.6206/0.4812 for
matched IU: AUROC delta -0.244pp, CI [-0.575,+0.073]. Leave-model-out is a
bounded positive signal at 0.6250/0.4851: AUROC delta +0.438pp
[+0.126,+0.760], but it has only 8/1/4 cell W/T/L and does not survive the
dataset-transfer definition. On `lens-96`, leave-dataset-out NRM is slightly
worse than IU (AUROC -0.106pp [-0.212,+0.002], AUPRC -0.227pp
[-0.380,-0.045]); leave-model-out and LOCO are also negative. Final-layer NLL
remains far stronger than depth NRM.

**Decision**: do not adopt NRM as the white-box method and do not replace the
negative v2 primary. Freeze the leave-model-out observation as a new-data
hypothesis: test the exact depth-NRM rule on a genuinely crossed model×dataset
capture, with no retuning. The addendum remains **PRELIMINARY / VALIDATION
BLOCKED** both because the live capture-validation gates are open and because
the NRM hypothesis was proposed after v2 outcomes were historically visible.

Canonical artifacts: `spectral_utils/whitebox_layer_fusion.py`,
`scripts/whitebox_layer_fusion_nrm_experiment.py`,
`scripts/whitebox_layer_fusion_nrm_report.py`,
`scripts/test_whitebox_layer_fusion_nrm.py`, and
`results/whitebox_layer_fusion_nrm_v1/REPORT.html`.

---

## Latest development — white-box layer-fusion v2 (2026-08-13)

The recovered `whitebox/per-layer-views` source and all 14 available Drive
sidecars were integrated in the isolated `codex/whitebox-layer-fusion`
worktree. The offline benchmark covers 47,265 source candidates / 47,238
evaluable candidates across nine captured model families (13
protocol-eligible cells plus the rejected CoQA/Llama-1 INSIDE appendix).

The registered residual-core headline is a negative result. Equal-cell macro
AUROC/AUPRC over the 13 eligible cells is 0.6181/0.4785 for DUFS-LIU versus
0.7298/0.5892 for final-layer NLL. The primary AUROC delta is -0.1117 with
95% grouped-bootstrap CI [-0.1245, -0.0996]. DUFS-LIU is also slightly worse
than matched IU-PCR: -0.00253 [-0.00325, -0.00186]. Do not promote a robust
improvement claim and do not replace the primary post hoc.

The richer `lens-96` secondary contract reaches 0.7253/0.6025 and the balanced
grouped-CV residual/TriLens/DoLa probes reach approximately 0.77 AUROC. Thus
the internal trajectories contain signal, but the current label-free compact
fusion objective does not recover it robustly across architectures. Mean
generation entropy is the strongest descriptive label-free row at
0.7399/0.6154.

The report remains **PRELIMINARY / VALIDATION BLOCKED** pending full corrected
live Gate B and the independent two-cell architecture pilot. Covariance
geometry is additionally omitted on Phi-3, Phi-3.5, and Qwen3 because the
recovered capture cast large covariance eigenvalues directly to float16 and
overflowed. Full report and documentation:
`results/whitebox_layer_fusion_v2/REPORT.html`, `EXPERIMENT_SUMMARY.md`,
`DATA_INVENTORY.md`, `FEATURE_MATRICES.md`, and
`METHODS_AND_COMPARATORS.md`.

---

## Session addendum (2026-08-10) — external data collection scaled; 4 new reasoning-localization competitor ceilings (Steps 240-241)

**SemGrad + HLE (Step 240)**: SemGrad scaled to full N — SciQ 1000/1000 rows
(accuracy 0.648), TruthfulQA 817/817 rows (accuracy 0.308) — Qwen3-4B-
Instruct-2507, fetched and backed up to Drive. HLE full run (N=2158,
Qwen2.5-72B-Instruct, 3-job Slurm chain 176043→176044→176045) still in
progress. HUB and ReDe audited and confirmed BLOCKED (no controlled
generation protocol / no official code, respectively) — no change to their
status in `docs/research_notes/external_data_collection_plan_2026.md`.

**Reasoning-localization competitor ceilings (Step 241)**: built and piloted
(N=30/subset) 4 new competitors named in this project's own gates but never
executed — ProcessBench's own critic-model baseline (Qwen2.5-72B, F1
70.4/50.0/47.1/65.9), the published Qwen2.5-Math-PRM-7B supervised ceiling
(F1 81.4/73.3/61.8/73.0), uPRM's own no-training "LLM-as-a-Judge" control
(our reconstruction, Qwen3-8B, F1 26.2/18.2/0.0/8.8 — **not uPRM itself**,
which needs real RL training per a full paper read, see HISTORY Step 241),
and a LettuceDetect ceiling on the full 2,700-row RAGTruth test split (F1
0.759). Three real bugs caught and fixed (transformers API version skew in
the PRM checkpoint's own code; a BPE marker-merging bug in the uPRM-baseline
reconstruction; a `0.0`-treated-as-falsy F1 bug now fixed and consolidated
into one shared helper). **Full N=3400 runs for the three ProcessBench
scorers are not yet submitted** — next action is Omri's review of pilot
health before scaling. Full account: HISTORY.md Step 241.

---

## Current decision point — RAGTruth's novelty claim is promising but not confirmed; a real sign-bug was caught and fixed first (Step 239)

**RAGTruth evidence-contrast, the real result** (N=2,700 responses, 450
`source_id`s, full test split, hashes frozen before labels —
`results/rag_ec_v1/full_test_split_result.json`): response AUROC
`ec_dufs_liu_evidence_graph` 0.7536 > `ec_upcr` 0.7341 > `ec_dufs_liu_temporal`
0.7329 > `fusion_isolation_naive_avg` 0.7290 > `gasp_reproduction` 0.7137
(essentially reproduces the paper's own 0.713 for Qwen2.5-1.5B — a fidelity
check that passed) > `likelihood_drop` 0.6946 > `full_context_only_dufs_liu`
0.6424.

**The preregistered novelty test** (grouped bootstrap by `source_id`, arm vs
`fusion_isolation_naive_avg` — the row the whole campaign's claim rests on):
best margin is `ec_dufs_liu_evidence_graph` at +2.51pp, 95% CI
[−0.58pp, +5.72pp], P(Δ≤0)=0.066 — **promising but the CI still crosses
zero**. The preregistration's own "default" arm (`ec_dufs_liu_temporal`,
temporal-chain graph) and `ec_upcr` are both statistically indistinguishable
from naive averaging (P(Δ≤0)≈0.39 each). `full_context_only_dufs_liu` and
`likelihood_drop` ARE significantly worse than naive averaging (P(Δ≤0)=1.0
and 0.98) — so the evidence-contrast intervention design itself is doing
real, confirmed work; it's specifically the "does OUR fusion beat naive
averaging" claim that isn't over the bar yet. Notably, the arm closest to
significance is arm 5b (the NEW exogenous evidence-graph construction, my
own operationalization of the preregistration's graph description — still
flagged as unconfirmed, not the previously-validated temporal-chain graph).

**A real bug was caught first, not a finding to report as-is**: the first
scoring pass had `ec_dufs_liu_temporal`/`ec_dufs_liu_evidence_graph`/
`likelihood_drop` all scoring well below chance (AUROC 0.25–0.31) — traced
to `anchor_orient`'s anchor being grounding-oriented (higher = more
grounded) instead of risk-oriented (higher = more hallucinated) for those
three arms specifically. Fixed (`anchor_sign` param in
`scripts/rag_ec_v1/run.py`) and added a regression test to the module's own
`smoke()` so a future inverted arm fails loudly instead of just looking like
weak signal. Full account: HISTORY.md Step 239.

**Next**: Omri's read on arm 5b's evidence-graph mechanism (flagged since
Step 237 as one reading of the preregistration text, not confirmed). Then
either a replication check (the dev slice, already scored and sitting at
`dataset_cache/ragtruth_ec_full/dev/` locally / `ragtruth_ec_qwen25_15b_dev/`
on the cluster — not yet run through the evaluator) or the preregistered
failure-test battery before treating +2.5pp as a real effect.

## Previous decision point — first real external-family result is in (Step 238)

**ProcessBench external-family validation**: fully done end to end. Gate B,
N=30 pilot, and the full 4-subset run (3,400 rows) all completed cleanly on
Llama-3.1-8B-Instruct. `scripts/gl_liu_external_v1/run.py` has now been run
for real (not a dry run) — score hashes frozen before labels
(`results/gl_liu_external_v1/llama31_8b/FREEZE_MANIFEST.json`). **Result,
reported honestly**: gl_liu_v1_frozen reaches 31.71% macro F1 vs
unified_core_five_dufs 31.62% vs baseline_max_entropy (transparent, no
fusion) 31.50% vs mindgap_control (Mind the Gap reproduction) 25.45%.
GL-LIU v1 clearly beats Mind the Gap's own baseline on every subset
(+5–10pp) — genuine transfer. It does **not** clearly beat the simplest
transparent baseline (max token entropy) — the margin flips sign per
subset and the macro average gap (0.21pp) is noise-level at ~850
rows/subset. Full per-subset breakdown and the "honest read" paragraph:
HISTORY.md Step 238.

**RAGTruth evidence-contrast**: Gate B and the N=30 pilot passed every
preregistered gate (alignment, chunk-count rule, and the direction-sanity
check: mean `NLL_noctx − NLL_full` = +179.35, 95% positive on grounded
responses). Full-scale jobs submitted: the 150-source_id dev slice is done
(5,724 items); the primary test-split run (~16,200 items) was at ~94% at
last check, its chained resume job pending. **Next action**: once it
finishes, fetch + schema-validate + freeze-hash, then run
`scripts/rag_ec_v1/run.py` (built and mechanically validated against the
N=30 pilot this step — not yet run against real frozen data) for the
actual result.

**RAGTruth evaluator built this step**: `scripts/rag_ec_v1/{gasp,run}.py` —
all 6 preregistered arms, including a faithful GASP-threshold reproduction
grounded in a real read of the paper (arXiv:2607.04223; digest at
`papers/digests/gasp-...md`). For our scorer (Qwen2.5-1.5B), GASP's own
reported number is **0.713 response AUC / 0.673 span AUC** on RAGTruth —
that is the number our arms should be checked against once labels open, not
the paper's rounder cross-scorer-average abstract figure. Arm 5b's
evidence-graph fusion mechanism is flagged in the code as one reading of
the preregistration, not a confirmed mechanism — worth Omri's sign-off
before trusting its numbers as a real test of the new-graph idea.

**A blocking module was reconstructed in Step 237**:
`spectral_utils/token_feature_views.py` (the `gl_liu_factorial_v2` local-head
feature contract) was confirmed absent from every branch and stash. Rebuilt
from the frozen `RUN_DEFINITION.json` contract plus the never-lost
`positional_views.py` machinery; validated by rerunning the full 8-cell
factorial study — the three core-dependent headline F1 numbers reproduced to
16 significant digits, and the broad-pool number preserves the qualitative
finding (broad underperforms core) even though its exact magnitude differs as
expected (7 of the 28 broad views have no exact-identity test by
construction — see the module docstring). Full account, including a real
`.gitignore` bug found along the way (a bare `*token*` credential pattern was
silently blocking this exact file from `git add`), in HISTORY.md Step 237.

Preregistrations: `docs/research_notes/ragtruth_ec_preregistration_v1.md`
(RAG) and `cluster/manifests/{pb_llama31_8b_external_v1,ragtruth_ec_v1}.json`
(both campaigns' competitor gates). Full narrative: HISTORY.md Step 237.

---

<details>
<summary>Step 235/236 — unified DUFS-LIU frozen; localization/RAG literature consolidated (superseded as the session's current-decision section by Step 237 above, still current for the underlying algorithmic conclusions)</summary>

## Previous decision point — unified DUFS-LIU is the simplest leading candidate

The approved follow-up to the GL-LIU v1 handoff is complete. Two controlled
2x2 matrices separated the choice of local graph from the choice of local
feature pool. Both global scores, the temporal-core curve, and the DUFS-core
curve reproduce the frozen v1 artifacts exactly in all eight cells.

Using DUFS-LIU in both heads with the frozen five local curves reaches **31.72%
ProcessBench F1**, compared with **31.36%** for frozen GL-LIU v1 and **25.71%**
for the reproduced Mind the Gap control. On the six cells outside component
selection it reaches **31.41%**, versus **30.76%** for v1. The change is small
and mixed: five cell wins and three losses. It supports a simpler common graph
construction and slightly better transfer, but does not confirm a universal
local DUFS gain.

The broad local pool is rejected. Twenty-eight unique token-varying curves were
constructed from the 30 registered global names: trace length cannot localize,
and the two CUSUM summaries share one local curve to avoid duplicate voting.
The broad DUFS locator lowers end-to-end F1 to **29.03%**, or -2.70 points
against the five-view locator, and loses in seven of eight cells. All curves
survived and the graph remained active; the failure is target alignment, not a
numerical collapse. DUFS preserved a coherent token-confidence geometry that
did not identify the first erroneous step.

**Decision:** keep global mixed DUFS-LIU. Carry five-view local DUFS-LIU as the
primary simplicity candidate and temporal LIU as the frozen robustness control
into an external dataset/model-family test. Do not tune feature subsets,
rolling windows, or new locators on the same ProcessBench labels.

Canonical report: `results/gl_liu_factorial_v2/REPORT.md`. Advisor brief:
`results/gl_liu_factorial_v2/ADVISOR_BRIEF.md`. Executed design:
`docs/experiments/GL_LIU_FACTORIAL_V2.md`.

### Benchmark literature update

The earlier side research was recovered from the repository. It was not stored
in a surviving separate Git worktree: the Git worktree registry and session
records point to the same repository path, and the research files are already
in `master`. The reasoning material was spread across a broad benchmarking
guide, CoT/agent notes, and the localization handoff. The RAG material was
stored under `docs/research_notes/research_phase10_rag/` and in the
Evidence-Contrast U-PCR proposal.

Two current decision maps now consolidate and fact-check that work:

- `docs/research_notes/reasoning_localization_methods_and_benchmarks_2026.md`;
- `docs/research_notes/rag_localization_methods_and_benchmarks_2026.md`.

For reasoning localization, Mind the Gap is only the sole external method in
the **existing frozen run**. uPRM is the closest newly identified label-free
peer and should be the first baseline audited. Trained PRMs and critic models
are required ceilings in separate categories.

For RAG, GASP already uses fixed-answer evidence removal, so evidence
perturbation alone is not novel. The proposed contribution is label-free
spectral fusion of dependent evidence contrasts. RAGTruth remains the primary
span benchmark; TRIVIA+ is a strong long-context and label-noise confirmation
candidate; RAGBench is a falsification test; and L-CiteEval is reserved for an
explicit citation claim.

</details>

---

## Current leading ProcessBench method — GL-LIU v1

GL-LIU v1 (Global-Local Laplacian IU-PCR) is now the canonical end-to-end
method for ProcessBench. It replaces the Mind the Gap score with two parts that
use only our token statistics:

1. a global mixed-contract DUFS-LIU detector for error presence;
2. a continuous moving-window temporal-LIU locator for the predicted token.

The method never uses step boundaries to construct a score. Step spans are
used only after prediction to map a token to the benchmark annotation. Score
fitting is unlabeled. Labels are used on two declared development cells for
component selection, inside each calibration half for the final threshold, and
after scores freeze for evaluation. The method is therefore a **calibrated
unsupervised scoring method**, not a fully label-free decision policy.

Across eight ProcessBench model/dataset cells, GL-LIU v1 improves ProcessBench
F1 from 25.71% for the reproduced Mind the Gap control to 31.36%. Exact first-
error localization improves from 17.84% to 21.79%, tolerance-one localization
from 39.35% to 46.76%, and clean-trace accuracy from 48.63% to 57.99%. On the
six cells excluded from component selection, F1 is 30.76% versus 24.74%.
GL-LIU has higher F1 in all eight cells.

The evidence is not equally strong for both stages. The global mixed DUFS-LIU
detector beats mixed ordinary IU-PCR in all eight cells by about +0.22 AUROC
percentage points on average. The selected temporal locator wins on the two
development cells in aggregate, but loses to the DUFS feature-graph locator on
the six non-selection cells (about 25.14% versus 25.78% exact localization).
Consequently, the global detector is the confirmed component. The temporal
locator remains the frozen v1 candidate and must be compared with ordinary IU
and DUFS feature-graph IU on a new dataset family.

The two model sizes reuse the same underlying ProcessBench examples. The eight
cells represent four independent dataset families; only OlympiadBench and
OmniMath are new dataset-family confirmation sets relative to selection.

Canonical definition: `docs/methods/gl_liu_v1.md`. Scientific report:
`results/ours_only_localization_v1/REPORT.md`. Advisor presentation:
`results/ours_only_localization_v1/REPORT.html`.

---

## Feature-contract refinement — 2026-08-07

The current DUFS-LIU code had still been using `fixed_stable_v1`; the four
non-monotone candidates were removed, not transformed. A label-isolated search
now evaluated all 256 per-feature combinations of `drop`, `raw`, `squared`, and
KDE `mode` under the frozen DUFS-LIU settings.

The retrospective DUFS-LIU winner is `pe_mean=squared`,
`stft_spectral_entropy=mode`, `cusum_shift_idx=raw`, and `rpdi=raw`. It scores
0.776562 versus 0.774139 for stable-only (+0.242pp, 17W/7L). LOFO contract
selection gives +0.123pp, but excluding `math500_qwenmath7b` reduces that mean
to about +0.022pp. `rpdi=raw` and `stft=mode` are stable; the `pe_mean` and
`cusum_shift_idx` choices are not.

**Decision:** freeze this exact mapping as
`dufs-liu-mixed-v2-development-2026-08-07` for the next external-family run.
Keep the stable-only 0.774139 result as the historical headline until that
confirmation. This is baseline refinement, not evidence that the target-neutral
static-graph problem from Steps 227--230 has been solved. Full conclusion:
`docs/research_notes/dufs_liu_mixed_feature_contract_conclusion.md`.

---

## Current research-priority override — 2026-08-07

RCV-AD-IU-PCR is complete. It tested repeated complementary feature splits as
a label-free alternative to family reliability anchors. Sixteen alternating-
diffusion graphs were averaged for atomic-random, dependency-blocked, and
family-blocked partitions. The registered primary kept absolute-Spearman
dependency blocks intact, used `k=7` and `lambda=0.1`, and was evaluated only
after all 24 score files and source hashes were frozen.

The method converged but did not improve correctness. The primary is +0.004pp
versus IU-PCR, 10W/14L, with equal-family interval [-0.052,+0.029]pp. Median
partition graph CKA is 0.536 and `T=8` versus `T=16` score Spearman is 1.000,
yet stability has Spearman -0.240 with utility. Random and family splits are
also ties (+0.018pp and +0.019pp). Repairing connectivity with `k=11` remains
a tie, and stronger graph regularization is harmful. Only 5/11 continuation
gates passed.

**Decision:** stop static repartitioning of the current feature pool as the
leading direction. This is the fourth form of the same target-identifiability
failure: stable geometry, stable operators, semantic family agreement, and now
cross-partition agreement all reproduce structure without identifying
hallucination correctness. Keep the supported Step-229 diagnosis that family
expertise changes by IU-PCR regime. A new graph or router requires a genuinely
independent view, not another function of the same static matrix. Full
conclusion: `docs/research_notes/repeated_cross_view_diffusion_conclusion.md`;
frozen report: `results/repeated_cross_view_diffusion_v1/REPORT.md`.

---

## Previous research-priority override — graph-coupled family relevance

The GCFR-U-PCR family-relevance diagnostic is complete. It tested Omri's
hypothesis that a feature family can be useful for one sample and noisy for
another, and that a graph over related families can stabilize a sample-local
reliability gate. The synthetic design explicitly separated inconsistent
inactive noise from coherent inactive nuisance. The real fit was physically
label-free, all scores and sources were frozen before evaluation, and all 24
retrospective cells were evaluated with fixed controls.

The hypothesis and the proposed mechanism must now be separated. Conditional
family specialization **is supported**: choosing a family expert separately in
frozen IU-PCR-rank quartiles has +2.833pp diagnostic headroom, with Holm
`p=0.006`. But the graph-coupled router **failed**: the registered path lost
0.135pp to IU-PCR, 0.243pp to its no-graph control, and did not beat any
mechanism control. Every positive graph strength was negative on average. The
gates were active, so this is not graph collapse or an optimization failure.

**Decision:** stop before a learned mixture. The semantic family graph encodes
which measurements are related, not which measurements share reliability for
hallucination correctness. IU-PCR rank is retained only as a regime coordinate.
The next premise must add independent interventional self-supervision—such as
repeated generations, benign perturbations, evidence-conditioned answers, or
semantic answer consistency—and first prove that it predicts family-expert
usefulness on held-out cells/families. Do not tune a router on the current 24
labels. Full conclusion:
`docs/research_notes/family_relevance_diagnostic_conclusion.md`; report:
`results/family_relevance_real_v1/REPORT.md`.

---

## Previous research-priority override — atomic operator audit

The Phase-0 atomic-operator premise audit is complete. It tested the smallest
remaining DUFS-inspired extension after the frozen view-fusion benchmark: use
a fixed label-free score to identify which individual feature Laplacian should
regularize the two-dimensional IU-PCR solve. All 24 score files and diagnostics
were frozen and hashed before labels were opened. Independent pre-run and
post-run reviews approved the protocol and the negative conclusion.

The premise failed. Median within-cell Spearman between the proxy and atomic
AUROC change was **-0.312**. The top-proxy atom lost **-0.838pp** cell-macro
versus IU-PCR, with 7 wins, 17 losses, and a worst loss of **-3.658pp**. The
equal-family change was **-1.178pp**, interval [-2.110, -0.372]. A label-only
oracle showed optimistic atomic headroom of **+0.447pp** cell-macro, so useful
actuation exists, but the label-free rule selects the wrong operators. Only 3
of 15 continuation gates passed.

This is not an optimization failure. The proxy ranking had median agreement
0.990 with its final result after four subsamples. Every registered combination
of `k` in {7,15,30} and `lambda` in {0.3,1,3} remained negatively associated
with usefulness. Smaller `lambda` reduced harm but did not reverse selection.
The failure is target identifiability: reproducibility, agreement with an
IU-PCR pseudo-score, and strong actuation identify stable feature geometry,
not hallucination correctness.

**Decision:** do not implement AOG Phase 1 and do not tune proxy weights on
these labels. Confidence-oriented U-PCR/IU-PCR remains the incumbent. DUFS-LIU,
uniform atomic fusion, and atomic operators remain controls, not promoted
methods. The next research junction must introduce an independent
interventional self-supervised signal, such as repeated generations, benign
prompt/decoding perturbations, evidence-conditioned answers, or semantic
answer consistency. Its own premise must transfer leave-one-family-out before
another fusion learner is built. Full conclusion:
`docs/research_notes/atomic_operator_premise_audit_conclusion.md`; frozen
report: `results/atomic_operator_premise_audit_v2/REPORT.md`.

> **Step 228 (2026-08-07, infra, not research):** the raw per-cell dataset cache (questions,
> answers, token-level stats) is now uploaded to the repo under `dataset_cache/` via Git LFS —
> 89 pkl files, ~28GB, across the 24 in-scope cells + GPQA + RAG + EDIS + ProcessBench/
> localization. Two worktree branches (`experiment/step-localization`, `selector/a4-antigravity-
> unsupervised`) had uncommitted work recovered and committed to their own branches (not merged
> to master). Two GPQA files exceeded GitHub's 2GB Git LFS object cap on the first push attempt;
> fixed by splitting them into `.pkl.part-NN` chunks and rewriting the still-unpushed commit.
> **Three master commits + two branch commits are staged locally and still need
> `git push` run from a terminal with working GitHub credentials** — the automated push hangs on
> this machine's credential helper. Full detail in HISTORY.md Step 228. Note: Step 227 (the three
> preregistered studies, already committed as `31f0677`/`092233d`) still has no HISTORY.md
> write-up — separate outstanding item, not done this session.

**What changed in Step 226.** Omri's sparse-error idea was confirmed in the Step-223/225 history,
but the literature baseline needed correction: the actual Tenzer et al. AISTATS 2022 paper already
contains SU-PCR (`C=L+S` with sparse correlated errors). The root `Tenzer2022_...pdf` is actually
Dror et al. 2017, which is why the repo implemented only the independent-error equation. The new
experiment therefore does not claim sparse Delta as novel. It prices published SU-PCR first, then
tests the tailored contribution: use the same sparse fit and `rho`, but replace top-two PCR with a
PSD, condition-controlled structured covariance solve. A ridge-without-sparsity arm identifies
whether any gain is merely regularization. Modern DEEM is included as iRBM-hard, deep-hard, and
deep-soft rank inputs over five fixed seeds.

**Run next:** [SPEC_DEPENDENCY_FUSION_EXPERIMENT.md](SPEC_DEPENDENCY_FUSION_EXPERIMENT.md). The
data machine runs `python3 scripts/test_dependency_fusion.py`, then
`python3 scripts/run_dependency_fusion_experiment.py --data-dir local_cache --device auto`.
Arm/seed checkpoints resume safely and the runner produces paired cell-bootstrap/Wilcoxon/Holm
tables plus equal-dataset, DEEM seed-stability, and sparse-support diagnostics. The mechanism gate
is also wired into `scripts/smoke_selectors.py` and passes; official DEEM 0.2.0 hard/deep/soft
synthetic fits and the resumable report path were dry-run successfully. H1 (`SU-PCR - IU-PCR`)
belongs to the published method; H2 (`SDSF - SU-PCR`) is the actual contribution gate. Files:
`spectral_utils/dependency_fusion.py`, `spectral_utils/deem_adapter.py`,
`scripts/run_dependency_fusion_experiment.py`, `scripts/test_dependency_fusion.py`.

> **Tailor, do not transplant.** A published metric is **inspiration, not a specification**.
> Take the concept, then develop it into the form our problem needs. True of *every* algorithm
> and metric we try. Each variant gets its own discussion before it is built.

This **rescopes Step 224 without retracting a number**: what closed is *transplanting a published
keep rule into this channel*, not the ideas inside those papers. Fidelity to the paper is no
longer the acceptance criterion for a new arm — it stays the criterion for anything *labelled*
with an author's name. Full statement, and the rewritten live line it produced, in
[HANDOFF_FEATURE_SELECTION_AND_FUSE.md](HANDOFF_FEATURE_SELECTION_AND_FUSE.md) §0 and §4.

Also in Step 225: all **66 research PDFs** un-ignored (`.gitignore`'s blanket `*.pdf` was
suppressing them), `results/upcr_study/README.md` written so every headline number in Steps
210–224 can be **re-derived from the saved CSVs with no dataset**, and the ℓ0-CCA **dry run**
committed on purpose — it caught a structural-prior trap (a no-signal arm scoring +0.32pp,
p = 0.019 against the wrong floor) before any real number existed.

⚠ **And a backlog: 49 modified + ~90 untracked source/result files, Steps 206–223, had never been
committed.** The pattern is a session committing its own *new* files and leaving edits to *shared*
files behind. It was structural, not cosmetic — `spectral_utils/answer_span.py` was untracked
while `repgrid_scoring.py` and `build_repgrid_featcache.py` **import it**, so a clean clone could
not run the feature cache; and `scripts/labelfree_standing_report.py`, the canonical U-PCR entry
point named in `CLAUDE.md`, was not in the repo either. All now staged and attributed per step in
HISTORY.md Step 225. **Before ending a session, check `git status` for edits to shared files, not
just for your own new ones.**

**Step 224** — **the published unsupervised feature-selection literature has now been
run in this channel, and none of it beats the deployed keep rule.** 21 conditions, 111 variants, 24
cells × 5 splits × 2 arenas. **Every variant is negative against the deployed rule.** The five
pre-registered primaries: DUFS Eq.(7) −0.96pp (Holm 0.0072), Concrete Autoencoder −2.74pp (0.0002),
Laplacian Score and SPEC −3.77pp (0.0000), Eq-14 residual −5.89pp (0.0000). Against a same-size
random floor, nothing in the family clears Holm (best `a3.cae`, adjusted p = 0.282).

**The three results that survive the null and should be quoted:**

1. **The anti-redundancy family is not merely useless here, it is harmful — measured three
   independent ways.** `cohesion_set` −0.75pp (Step 223), `decorr_s5` −5.98pp (1W/23L), and
   `dpp.k4` **−8.08pp, 0W/24L**, all against a *random* subset of the same size. DPP MAP is the
   canonical diversity criterion, and the damage is **dose-dependent**: at size 21.9 its own
   data-driven stop declines to prune and is neutral (−0.47pp, 12W/12L); the harm scales with how
   much diversity is enforced. This is the empirical answer to the whole four-cluster reading list,
   whose clusters A/C/D are the same condition in different formalisms.

2. **The target is not a stable object (Step 223).** Two random half-splits of the *same* cell,
   both using that cell's own labels, produce good sets agreeing at Jaccard **0.524** (across cells
   0.303) — yet the room is real and a label-handed oracle transfers **84%** of it (+1.88pp). Both
   hold only if **many different subsets are good**. Consequence: every `overlap_*` secondary in
   Steps 213–223 has been scoring reproduction of a set that a rerun would only half reproduce.
   **Stop using overlap-with-the-good-set as evidence.**

3. **Cohesion is not the mechanism (Step 223).** Selected-set cohesion minus the floor's: good set
   −0.127; the `cohesion_set` arm −0.131 — it matched the target almost exactly and finished
   **last**; the label-handed oracle −0.019 — it matched nothing and **won**.

**Also settled**: mmDUFS (non-linear shared graph operator, `P_shared = LxLy + LyLx`) scores −0.12pp,
identical to ℓ0-CCA's linear cross-channel criterion. **The null on the two-channel split is a
property of the channel, not of linearity.**

**Four conditions were ruled out by reading the paper, not by experiment** — record these so they are
not re-proposed: **SEFS** is not label-free at selection (π is fixed and equal across all features
throughout the self-supervised phase; it becomes feature-specific only under `ℓY(y,·)`);
**Feature Manifold Learning** is Cohen/Shnitzer/Kluger/Talmon ICML 2023, **not Bracha's**, and is
supervised by construction; **VICReg**'s variance hinge is identically zero on z-scored views
(verified over 6,820 views, max |std−1| = 1.3e−14); **Graph Information Bottleneck** needs a
labelled graph, not a feature matrix.

**Numbers of record, unchanged and re-derived in all three Step-224 runs**: the floor is **−0.84pp**,
the room is **+2.25pp, CI [+1.52, +3.05], 23W/1L**. The new size-matched floor agrees with exp13's
fixed-k floor to **−0.01pp** (p=0.99). Two limits to quote with the room: it is measured at
k ≈ 11.75, so it is not comparable to a floor-relative gain at k = 3; and the good set lives only
81.3% inside the deployed keep set, so the `keep` arena cannot reach it.

**FULL HANDOFF for the next session (read this first if picking up cold)**:
[HANDOFF_FEATURE_SELECTION_AND_FUSE.md](HANDOFF_FEATURE_SELECTION_AND_FUSE.md) — §0 the standing
instruction, §2 what is closed and with what evidence, §4 the live line (triplet consistency as a
concept to develop: scoring **the views**, and scoring **each sample** as a pseudo-label for the
**weights** channel), §6 the sparse-Δ estimator relaxation, §7 eleven practical traps already hit,
§9 how to work from the repo with no dataset.

**Shareable report**: https://claude.ai/code/artifact/a4d307aa-3053-4e52-83df-8c2c917967f5

**What is open**: the feature-selection channel has absorbed per-feature rankers (Step 222),
set-level covariance functionals (Step 223), and the published literature **as transplanted**
(Step 224). The search is not the bottleneck — handed labels, the same greedy takes 84% of the
room. What is live, under the Step-225 instruction: the **triplet-consistency concept developed
into our own statistic** — quadruplet residuals (m=4 is the first order with spare equations) and
the variance of the implied `v̂_i` across a view's triplets are the two least-explored forms — and
the **per-sample pseudo-label** aimed at the weights channel, where +1.24pp (CI [+0.17, +2.29],
p=0.016) is measured and unclaimed. Order of work in the handoff §8; older plan in `PLAN_NEXT.md`.

<details>
<summary>Step 222 — the ranker menu (superseded as headline by Step 224)</summary>

is now priced too.** Eight arms, pre-registered with directions before scoring, in the only
U-PCR channel with room (+2.25pp). **Not one label-free per-feature statistic clears the matched
floor**; the best is a set-level cluster round-robin at +0.23pp with an interval crossing zero
(Holm 0.53), and **two arms are significantly *worse* than pruning at random** — redundancy to
the pool −3.13pp (Holm 0.002) and L-SML cluster size −1.61pp (Holm 0.008). **DUFS's gate value
is −0.70pp**, 9W/15L, not separable from the floor after multiplicity (Holm 0.36) and a
three-cell effect; principal-direction leverage is lower still at −0.92pp.

**The result that makes this a statement rather than a loss**: the redundancy statistic is the
one that *most* identifies the good features (overlap +0.036, 17W/7L, bootstrap CI excluding
zero) and the *worst* performer of the eight. That is Step 221's two-sided finding reproduced
with a label-free statistic instead of an oracle one, and together with the true correlation
already sitting on the floor it closes the family: **the +2.25pp is not reachable by scoring
features one at a time.** `PLAN_NEXT.md` pre-registered that as the stronger deliverable.

**Do not read the round-robin as an escape from the failed shape.** It is indistinguishable from
a uniform draw inside the keep set on the overlap test (p=0.92, closest to the null of all
eight), and across the six label-free arms |overlap excess| vs performance has Spearman −0.71 —
the nearer an arm is to random, the better it scores against a floor that *is* random. It also
does not survive its own L-SML loading-scale choice (+0.04 `eigen`, −0.02 `complete`, exploratory).

**Two numbers to keep using**: the floor of record is **−0.84pp**, not −1.55pp, and the room is
**+2.25pp, CI [+1.53, +3.04], 23W/1L** (both re-derived exactly at Step 222). U-PCR's own ranking
is **below** chance at finding the good features (−0.05, p=0.016), not at chance. **One limit to
quote with the room**: the good set lives only 81.3% inside the deployed keep set while every
pruning arm is confined to it, so ~a fifth of the target is unreachable by this design and
"recovers X% of the room" has a denominator the arms cannot fully address.

**READ NEXT**: `results/action_items_jul2026/item2_upcr_clustering/PLAN_NEXT.md` — the forward
plan. `PHASE1_RESULTS.md` in the same directory has the measurements. `HANDOFF_upcr_selection.md`
has the traps. The pre-registration for Step 222 is the module docstring of
`scripts/upcr_study/exp14_ranker_menu.py`.

**Note on step numbering**: Step 219 belongs to the localization/Extension-F work on the
`experiment/step-localization` worktree. Master is at 222.

</details>

<details><summary>Step 221 (previous) — what separates the good features</summary>

**Step 221**: the correlation with correctness **identifies** the good feature sets (+0.11
overlap above a null matched on keep-set composition, 20W/4L, p<1e-4) and **buys nothing**
(+0.08pp over a matched floor, p=0.62). A *perfect* estimate of it, spent on selection, is worth
**+0.34pp, CI [−0.47, +1.30], p=0.88** — so with the weighting blend at +0.19pp and polarity at
−0.06pp, every channel a better `rho_hat` feeds is priced at zero, and Bracha's **second**
proposal (differentiable pair reweighting) is closed before being built. Step 222 then priced her
**first** and closed the whole per-feature family.

</details>

<details><summary>Step 220 — the U-PCR ceilings</summary>

**Step 220**: the U-PCR clustering line CLOSED on ceilings. Feature selection is the only
channel with room in it (+1.48pp held out, CI [+0.97, +2.03]); the weighting blend (+0.19pp,
p=0.57), the constants, `var_y`, and getting every view's sign right (−0.06pp, p=1.00, 17/24
cells exactly unchanged) are all empty. The good masks keep ~10 views vs the deployed ~21,
smaller on 24/24. Step 221 supersedes its floor (−1.55pp → −0.84pp) and its "at chance"
reading of the rho ranking (→ below chance).

</details>

<details><summary>Step 218 — the non-monotone line</summary>

**Step 218**: **the non-monotone line is CLOSED. The fold repairs the view (up to
+26.5pp single-view), U-PCR re-admits 33% of the views it had excluded, and it is worth +0.05pp
fused because the pool already had the information (99% absorbed). The feature pool ships
unchanged.** Steps 216–217 below remain current for the roster and the validity constant.

</details>

## Steps 216–217 — THE ROSTER AND THE VALIDITY CONSTANT BOTH CHANGED (READ THIS FIRST)

**If you are picking up cold: `GOOD6_EXPECTED` is now `0.7733`, not `0.7594`, and `INSCOPE` is 24
cells, not 25.** Both changed for measured reasons, not tuning. Anything quoting the old numbers
is pre-Step-216.

### What was wrong

Two cells — **exactly the two base checkpoints in the grid**, found independently of the model
roster — were computing features over malformed generations:

- **`seiclr_triviaqa_opt30b`** (`facebook/opt-30b`): no learned EOS for the few-shot format, so
  99.7% of generations sit at `max_new=64` and run on into a fabricated `Question:` block. Median
  answer **3 tokens = 4.7% of the trace**. **REPAIRED by cropping** — and the crop is a bug fix,
  not a modelling choice, because **the grader already crops** (`is_correct_trivia_qa_rougel`
  scores `first_answer_line`), so the label was computed on the answer while the features were
  computed on all 64 tokens. No re-generation needed: decoding is autoregressive, so truncating a
  suffix offline is bit-identical to having passed a stop sequence.
- **`inside_coqa_llama7b`** (`huggyllama/llama-7b`): a Mistral chat template applied to a *base*
  checkpoint (the preset never set `raw_prompt=True`). 45.1% of answer spans are `[/INST]` echoes,
  fabricated turns or empty, and `pos_rate` is **0.002 on the broken rows vs 0.239 on the usable
  ones** — a degenerate sub-population. **REJECTED**, recorded in
  `scripts/inscope_cells.REJECTED_CELLS` (mirrored + asserted equal to
  `answer_span.UNREPAIRABLE_CELLS`), not deleted. It needs re-generation with `raw_prompt=True`.

### The numbers now

| arm | before (25 cells) | **after (24 cells)** | QA (9) | math (15) |
|---|---:|---:|---:|---:|
| U-PCR + sign(rho) | 0.7551 | **0.7741** | 0.7586 | 0.7834 |
| DUFS parameter-free + L-SML | 0.7507 | **0.7687** | 0.7520 | 0.7786 |
| GOOD_6 (reference) | 0.7594 | **0.7733** | 0.7611 | 0.7807 |

`GOOD_6 − upcr` is **−0.08pp, 13W/11L, p=0.819** — for the first time the label-free arm is
nominally *above* the hand-picked subset. Still nothing separates the three; what changed is the
sign. **The QA deficit was two broken cells**: GOOD_6's QA lead was 1.49pp over 10 and is now
0.25pp over 9.

**`seiclr_triviaqa_opt30b` was never a method failure** — `a2.dufs_pf` 0.5614 → **0.7726**, U-PCR
0.5751 → **0.8119**, GOOD_6 0.5884 → **0.8311**, best single view ~0.62 → **0.8258**
(`topk_tail_mass`), against SE-ICLR'23's published **83.0**. **This retracts Step 215's per-cell
diagnosis of that cell** (selection miss + 2-of-12 sign disagreement): it was measuring the run-on
artifact. ⚠ Two caveats travel with it: GOOD_6 has only **4 of 6 views** there (per Step 205 L-SML
is numerically undetermined at 4), and the pool shrinks **30 → 20** because a 3-token answer has no
spectral views — which is what SE-ICLR'23 used anyway.

### Gates that were run, and one that had to be rebuilt

- **Per-cell equality, not the macro** — after an intentional data change the macro is *expected*
  to move. `scripts/answer_span_score_check.py` (new) asserts GOOD_6 / full-pool L-SML / U-PCR all
  reproduce **bit-identically on 23/23** untouched in-scope cells (tol = 0.0).
- **The old constant reproduces**: 25 cells pre-repair = **0.759398**, i.e. the path is sound. The
  new constant decomposes exactly: −CoQA → 0.763232 (+0.38pp), +crop → **0.773344** (+1.01pp).
- **Staleness carriers, again (Step-193 lesson)**: the crop left stale `n=4993` rows in **17**
  selector-bench CSVs, not just the one the repair touched. All backed up (`*.step216.bak`),
  stripped, re-run. `reference_macros__c46` now writes 5 variants instead of 13, and the **h16 pool
  arm mostly does not apply** on that cell — h16 *is* the 16 spectral views.
- **`answer_span_audit.py` had become circular** and now iterates `INSCOPE_ALL`: once the cell was
  rejected the audit stopped measuring it, reported no unrepairable cells, and its own drift-check
  fired against the registry it exists to justify.

## Step 218 — the non-monotone line is CLOSED (READ THIS BEFORE STEP 217 BELOW)

**> NEXT SESSION STARTS HERE.** Step 217's three pre-registered gaps are all closed. Its
"line stays open" verdict below is **superseded**; its two durable corrections (the inflated
`nonmono_gain`, and "use the shape instrument, not `nonmono_gain`") still stand.

### The three-line verdict

1. **The fold works.** Single-view, on 27 pairs with ≥5pp headroom it recovers ~73% of it.
   `sciq_llama8b/pe_mean` 0.434 → **0.699**; `math500_qwenmath7b/pe_mean` 0.458 → **0.668**;
   `truthfulqa_llama8b/rpdi` 0.487 → **0.614**. Spearman(headroom, gain) = **+0.68, p = 1.4e-14** —
   it helps exactly where the shape is and loses (−5.7pp) where it is not. **This corrects Step
   217's diagnosis**: the symmetric family is not the wrong family; it failed downstream.
2. **The exclusion channel opens.** `upcr.py:287-293` drops views with ρ̂ ≈ 0 (the reason Step 217's
   arm B was bit-identical on 4 of 5 high-detection cells). After folding, **8 of 24 excluded views
   (33%)** re-enter `keep` — `truthfulqa/rpdi` goes ρ̂ −0.032 → **+0.168**. Mechanism confirmed.
3. **And it is worth nothing fused, because the pool is saturated.** Deployable macro **+0.05pp**
   (U-PCR) / **+0.14pp** (DUFS+L-SML); the *label-selected ceiling* is +0.23 / +0.26pp. G1 (≥+0.5pp)
   fails everywhere. R2: marginal shape gain **+8.00pp → conditional +0.05pp, 99% absorbed**,
   positive on 19/38 (a coin flip), Wilcoxon p = 0.99, Spearman(marginal, conditional) = −0.013.
   The view's plain *monotone* reading is worth +0.046pp conditionally too — it is not the shape
   that is redundant, it is the **whole view**.

### THE POOL DECISION

**The feature pool ships UNCHANGED — no view added, none replaced.** +0.05pp deployable against a
0.7733 GOOD_6 macro, with a **−20.6pp** failure mode if the selection goes wrong, is not worth it.
Recorded for when that changes (full derivation in HISTORY.md Step 218.6):

- **If ever adopted, the mode is `replace`, never `add`** — structural, not empirical: a fold
  beside its parent duplicates its rank information and biases U-PCR's whole ρ̂ vector. Measured
  penalty +0.31pp for `replace` over `add_orth` (p = 0.021 / 0.041).
- **Transform of record = `mode_centre`** (`|u − c|`, c = KDE mode percentile). Least harmful when
  misapplied: **+4.77pp on a true positive, −2.34pp on a false positive**, vs `squared`'s
  **+6.02 / −4.69**. (`dist_median` is a dead heat; `mode_centre` wins on principle, not evidence.)
- **Switch to `squared` once selector precision exceeds p\* = 0.654.** We are at **0.562** — nine
  points short. `squared` has the highest ceiling of the family and the worst misapplication cost,
  so the crossover is a genuine trigger, not a preference. **This is the concrete payoff from any
  future improvement to selection / clustering.**
- `mode_centre` is a **feature-level** recommendation. Applied to every flagged view it scores
  −0.09pp on U-PCR (worst cell −2.21pp); the per-view pseudo-label pick scores +0.05pp. Do not
  blur the two claims.

### The blocker, quantified — this is where effort should go

The label-free consensus detector correlates with true headroom at only **Spearman +0.309
(p = 1.9e-3, n = 99)**; best-threshold precision **0.562** against a 0.384 base rate, and **13 of 61
control views would be falsely folded**. Folding a healthy view costs ~5pp. Everything above is
gated on this number.

### Next actions

1. **Do not spend more on per-view reshaping.** R2 conditions on the fusion using a *supervised*
   2-D logistic — an upper bound on what any fusion could extract from that pair — so the near-zero
   bounds every transform, not just the ten tested. The pool, not the fusion's linearity, is the
   binding constraint.
2. **Better views** is the direction R2 points at. A conditional-gain screen (`redundancy.csv`'s
   `cond_view_gain_pp`) is a ready-made instrument for triaging any *new* candidate view: it asks
   "what does this add given the fusion", which marginal AUROC cannot.
3. **A sharper label-free selector** is the other direction, and it now has a target: precision
   0.654 is where `squared` becomes correct and the family's ceiling opens up.
4. One unconfirmed slice worth a look if QA cells grow: arm A, QA only, is +0.34pp [−0.00, +0.76]
   deployable on n = 8 and +0.67pp [+0.16, +1.29] at the ceiling — the only slice clearing +0.5pp.
   **Hypothesis, not a result.**

### Instruments built this step (reusable)

- `scripts/nonmono_v2/common.py` — the corrected shape instrument. **Supersedes
  `nonmono_shape_test.py`**, which chose the isotonic direction by Pearson `corrcoef` (sign is
  coin-flip noise on a U-shape → inflated gains) and discarded the KDE mode *location*.
- `scripts/nonmono_v2/dufs_pf.py` — standalone `a2.dufs_pf`, **bit-exact on 24/24 cells**
  (`--verify`). ~4–6s/cell instead of ~17s, so selection can be re-run per config. Use this
  anywhere arm A's selection needs to respond to a changed matrix.
- `scripts/nonmono_v2/stage4_redundancy.py` — R2 conditional gain. The right instrument for
  "is this view worth adding" on **any** future view.
- Two advisor pages: `results/nonmono_v2/shape_evidence.html`, `transform_choices.html`.

<details><summary>Step 217 — the earlier verdict (SUPERSEDED by Step 218; the `nonmono_gain` correction still stands)</summary>

## Step 217 — the non-monotone line: real effect, wrong transform family, **LINE STAYS OPEN**

Gemini's +0.54pp transform proposal is **NOT ADOPTED**. Four review defects voided the measurement
(`max(a, 1−a)` sign resolution; the optimised objective computed on **zero in-scope cells** via
filename matching; no held-out anything; three of five target features not actually non-monotone).
`run_upcr_comparison.py`'s unconditional 15-view injection was reverted.

**Three things worth carrying forward, and the first one is a correction to our own code:**

1. **`nonmono_gain` in `results/advisor_inscope/ladder_featdiag.csv` is INFLATED — do not quote it
   without the correction.** `gap_ladder.py:64,220` folds `max(p, 1−p)` onto **each fold's binned
   test score**. A bin map fitted on train already carries its direction, so this is a one-sided
   noise floor: inflation is never negative (median 0.0000, max +0.2000) and
   `Spearman(corrected gain, inflation) = −0.171, p=7e-06` — **it credits a view more the closer
   its binned map sits to chance.** `pe_mean`'s +0.0438 headline is +0.0402 inflation.
2. **The non-monotonicity IS real, and it is cell-specific.** A first gate on the *per-feature mean*
   found nothing and **was withdrawn — Omri rejected it from the per-cell deep-dive pages and was
   right.** The fair test (`scripts/nonmono_shape_test.py`: isotonic vs 10-bin, both cross-fitted,
   vs each pair's own label-permutation null) finds **32 of 682 pairs beating their null**, several
   at 5–7× on cells with thousands of rows — e.g. `semenergy_triviaqa_qwen3_8b`/`rpdi` **+0.1227**
   (n=4392, sd 0.0019). Gemini's list was half right: `rpdi` (7 cells) and `pe_mean` (6) hold;
   `dominant_freq` (**0**), `spectral_entropy` (1), `epr_energy` (1) do not; and it **missed**
   `cusum_shift_idx` (6) and `hurst_exponent` (3).
3. **But it converts to nothing fused, and there is no label-free handle.** C2/C3 on the corrected
   candidate set: **G3 passes** (LOCO choice stable on 92–100% of folds — so this had the power to
   find a win), **G2 passes**, **G1 fails on both arms** (−0.07pp / −0.04pp). Restricted to the 11
   cells where a candidate provably beat its null it is still **−0.13pp / −0.09pp**. And the
   marginal two-peak shape does **not** locate the effect: `Spearman(shape_gain, KDE peak count) =
   +0.014, p=0.72`; "≥2 peaks" flags at precision 0.128 against a 0.047 base rate — because P(y|x)
   can bend without the *marginal* density of x being bimodal. **Mechanism: a single view's U-shape
   is largely redundant once 15–20 other views are in the fusion.**

**Use `scripts/nonmono_shape_test.py` for any future non-monotonicity question, not `nonmono_gain`.**

### Why the transforms failed — and the pre-registered next tests

**The transforms were the wrong family, and the curves say so.** All three (`|x − median|`, `x²`,
`|Φ⁻¹(rank%)|`) are symmetric and centred on the middle of the distribution. P(correct) by decile
for the strongest survivors (`^` = argmax decile, `v` = argmin) is nothing like that:

| cell | view | by decile | shape |
|---|---|---|---|
| `semenergy_triviaqa_qwen3_8b` | `rpdi` | `#-..v-#+^=` | W-shaped — high left edge, dip, second rise |
| `se_squad_v2_llama8b` | `pe_mean` | `=v+= . .^-` | dip at decile 2, peak at decile 9 |
| `semenergy_triviaqa_qwen3_8b` | `epr_energy` | `=+++*^#*=v` | **inverted-U peaking at decile 6–7, not 5** |
| `se_nq_open_llama8b` | `rpdi` | `*:: v..=+^` | argmax at the EDGE — the gain is an interior *dip* |

An inverted-U centred at decile 6–7 is **mis-centred** by a median-centred transform; a W-shape or
an interior dip is not in the family at all. Two further gaps: **arm A held the DUFS selection
fixed** (only 5 of 24 cells moved at all, because a reshaped view could never be *chosen*), and
**"add a view carrying the information monotonically" was never tested** — only transforms of the
same column.

**Next tests, in priority order:**
- **(a) Fit the centre, do not assume it** — a `|x − c|` family with `c` chosen **leave-one-cell-out**
  (held-out, fixed offline, so still label-free at deployment), or centred on the **KDE mode**,
  which `nonmono_shape_test.py` already computes label-free.
- **(b) Use the winning curve itself as the view** — the cross-fitted bin-mean map *is* the +12pp
  function; the open question is whether a LOCO-fitted version transfers across cells. Strongest
  form of the idea and the direct test of it.
- **(c) Re-run selection with the reshaped view in the pool**, closing the arm-A gap.
- Gate stays **G1/G2/G3** as written, on both arms.

**The one mechanism to test against rather than assume**: a single view's shape may be substantially
**redundant** once 15–20 other views are fused, so an isolated +12pp shape gain need not convert to
+12pp of macro. That is the competing explanation for G1's failure and (b) is the test that
separates it from "wrong transform family".

**> Step 218 ran (b) and the competing explanation WON: 99% of the marginal shape gain is absorbed
by the other views. Redundancy, not wrong family.**

</details>

<details><summary>Steps 214–215 — features or algorithm (still current, including the withdrawals)</summary>

## Steps 214–215 — features or algorithm (READ THIS FIRST, INCLUDING THE WITHDRAWALS)

Three matched cell pairs — a weak cell against a high-scoring one holding the dataset or the
pipeline fixed — compared on the **features**, not the fusion. Instrument: the supervised ceiling
(5-fold LR, `class_weight='balanced'`, per-fold AUROC, **10 seeds**).

**Step 215 was an adversarial review of Step 214. Three of its four findings did not survive.**
Site rebuilt; `HISTORY.md` Step 215 has the full accounting.

**What survives:**
- **Some of the gap is genuinely in the features on all three pairs, and none of it is all.**
  That direction holds. The *number* (Step 214's "52% / 39% / 81%") does not: bootstrap CIs are
  [25,74] / [14,111] / [59,104], mutually overlapping and two of them consistent with 100%.
  Swapping in a different but equally defensible TriviaQA partner moves the figure across
  **−2% to 60%**. Report the sign, never the percentage.
- **The ceiling−deployed gap on `seiclr_triviaqa_opt30b` is real**: 0.7229 vs 0.5614, CI ≈ ±2.5pp,
  across-seed sd 0.0011. But **"16.2pp of reachable headroom" was wrong.** It decomposes as
  ≈1.8pp label-free sign loss + ≈4.1pp that needs *labels* to pick the right single view + ≈10.2pp
  multivariate supervised gain with no label-free analogue. **Honest label-free target: single digits.**
- **NEW and actionable — on that cell the loss is selection + sign, and dilution explains 0.0pp.**
  Honest best single view over the pool 0.6200 (split-half ±0.0067) → best view inside the 12 the
  selector chose 0.5791 (**the selector had already discarded the pool's two strongest views**) →
  L-SML 0.5614. L-SML's effective per-view signs disagree with the oracle on **2 of 12** views here
  vs **0 of 12** on the other five cells; the plain label-free average beats L-SML, 0.5708 vs 0.5614.
  **Repair 3 remains the indicated next test — for this measured reason, not the Step-214 one.**

**Withdrawn:**
- **⚠ The TriviaQA pair was built on a disqualified cell.** `spilled_triviaqa_llama8b` (0.9413) has
  **n_pos = 6 of 256**, `trace_length` alone scores 0.925 on it, and `scripts/advisor_report.py:783`
  already said **"do not headline"**. Replaced by `semenergy_triviaqa_qwen3_8b` (n=4392). **Rule:
  check the existing per-cell caveats before making a cell the anchor of a comparison.**
- **⚠ "The label-driven correlation is ~3× smaller on every weak cell" — withdrawn.** It is
  algebraically implied by the per-view Cohen's d reported one section earlier (predicted-vs-observed
  entrywise corr ≥ +0.99997 on 6/6 cells), and the "consistency" was an uncontrolled class prior:
  κ = 0.249 vs 0.023 on that pair, so κ-adjusted the ratios are **28.9× / 3.8× / 3.4×**.
- **⚠ "Estimation noise is retired" — WITHDRAWN, and the instruction it produced is rescinded.**
  The statistic divided by 1/√n, the SE of *one* correlation; C and W share rows so their difference
  is far tighter — wrong by ~√n. Also `Spearman(label-free − ceiling, n) = −0.462, p = 0.020` across
  all 25 cells. **The subsample-to-matched-signal test is back on the table; Step 214 told you not to
  run it on the strength of a mis-normalised statistic evaluated on a 6-positive cell.**
- **⚠ "The method matches supervision on strong cells / degrades faster than supervision" —
  withdrawn.** Label-free exceeds the ceiling on 4/25 cells and **all four have n ≤ 700**
  (n ≤ 700: 4/11, mean −0.77pp; n > 700: 0/14, mean −6.03pp; Mann-Whitney p = 0.035). A small-sample
  effect. `semenergy_triviaqa_qwen3_8b` (n=4392) sits **5.2pp below** its ceiling.

**Code defects fixed in the rebuild**: `max(a, 1−a)` was being applied to a *supervised* score
(it can only fire on noise and only inflate — **+12.6pp** on the 6-positive cell; the transform is
for unsupervised scores of undetermined sign); single-seed ceiling → 10 seeds with sd reported; the
rank-1 fit was a non-convergent ALS on a period-2 limit cycle that diverged from almost every
start → replaced with a proper multi-start minimisation.

- **REPAIR 1 IS CLOSED.** Z₂ synchronisation as a replacement label-free sign estimator **fails
  both gates on both arms**: recovery ≥ 0.90 on **1 of 4** failing cells (not 4), and healthy
  cells move ≥ 0.5pp on **3 of 16** (arm A) / **1 of 16** (arm B). Worst collateral:
  `semenergy_triviaqa_qwen3_8b` −3.30pp, `math500_qwenmath7b` −2.31pp.
- **FULL REGRESSION — nothing beats baseline.** DUFS+L-SML 0.7507 · **U-PCR + sign(ρ̂) 0.7551
  (still the best arm)** · Z₂+avg 0.7493 · Z₂+avg full pool 0.7512 · Z₂ inside U-PCR 0.7529.
  Paired: Z₂+avg(full) vs baseline **+0.04pp, 12W/13L, p = 0.89** — a dead wash.
- **GATE P caught a design error in my own pre-registration.** L-SML is **exactly** sign-invariant
  on today's data (max |Δ| = 0.00e+00 for both Z₂ signs and *random* signs, 25/25 cells), so
  "feed L-SML a better sign estimate" was a **no-op by construction** and could never have applied
  to that arm. That was already in the project glossary and should have been checked before
  pre-registering. **Standing rule: test the premise before pre-registering a repair on it.**
- **THE MECHANISM CLAIM IS NOW WEAKER, and this is the second downgrade.** The description holds
  (three weakest QA cells worst on every sign measure, both arms, Spearman +0.707 p = 0.0002), but
  "a better sign estimate fixes them" is refuted. Calling the `r4−r3` gap "sign recovery" oversold
  it: L-SML does not *recover* signs, it is **invariant** to them. The remaining reading is closer
  to **"not enough covariance structure on these cells for any label-free method"** than to a
  fixable defect — which argues for reporting ceilings rather than chasing them.
- **NEXT, if anything: repair 3** (keep the pool's strongest view — the selection miss is −4.8pp on
  the worst cell and does not depend on the sign story). Repair 2 (rank transform for CoQA)
  untouched. **One durable positive**: CoQA +2.40pp under Z₂+average, the largest single-cell gain
  in the test.

- **BOTH ARMS FAIL TOGETHER — this is the strongest form of the finding.**
  Spearman(L-SML recovery, U-PCR recovery) = **+0.707, p = 0.00023**. The three worst cells on
  U-PCR polarity agreement (`inside_coqa_llama7b` **0.630**, `seiclr_triviaqa_opt30b` **0.714**,
  `losnet_hotpotqa_mistral7b` **0.793**, healthy median **0.966**) are also the three worst on
  L-SML recovery and on U-PCR's sign step. **L-SML is sign-invariant; U-PCR estimates polarity
  explicitly as sign(ρ̂). Two differently-built estimators degrade on the same cells** — so this
  is not an implementation quirk.
- **⚠ WITHDRAWN: the Step-210 "Fisher p = 0.0096".** The recovery ratio's denominator is under 2pp
  on 9 of 25 cells; requiring ≥2pp gives **p = 0.0735**, ≥3pp gives **p = 0.2500**. The U-PCR
  version never reaches significance (0.1167), nor does polarity agreement (0.1162). **9 vs 16
  cells cannot establish an effect this size. Read the pattern, not the p-value** — and let the
  pre-registered repair be the confirmation.
- **A metric bug worth remembering**: raw sign(ρ̂)-vs-oracle agreement is below 0.5 on **all 25
  cells**, which looks catastrophic and is not — a global flip leaves the covariance
  bit-identical, so sign(ρ̂) only ever recovers polarity **up to one ±1 it cannot determine**.
  Report `max(a, 1−a)`; the anchor supplies the rest.
- Every cell page now has a **§5b: the U-PCR ladder** (incl. a rung for its sign(ρ̂) step alone),
  its polarity agreement, survivor count, abstention and component count.

- **THE ANCHOR IS EXONERATED PER CELL, NOT ON AVERAGE.** Every cell page reports the fused AUROC
  under a **true-label anchor** beside the deployed one. The difference is **exactly +0.00pp on
  all 25 cells** — including where the anchor view itself is at 0.560. The one prior the
  label-free arms carry costs nothing anywhere.
- **A REPRODUCTION CHECK CONFIRMED THE MECHANISM A SECOND TIME.** Fusing each cell's *label-chosen*
  five views through the ordinary label-free pipeline reproduces the recorded oracle number
  **exactly on 23 of 25 cells**. The two that miss are **the two worst-recovery cells**
  (`inside_coqa_llama7b` 17.08pp, `seiclr_triviaqa_opt30b` 0.82pp); Spearman(gap, recovery) =
  **−0.497, p = 0.019**. Where sign recovery fails, the pipeline cannot fuse even the *perfect*
  subset.
- **CoQA's headroom splits in two, and this changes which repair to try.** From 0.5320: a perfect
  selector buys **+7.4pp**; the remaining **+17.1pp needs the sign recovery fixed**. Step 210's
  "24.5pp" conflated the two and would have pointed the next experiment at selection.

**Step 210 headlines below still stand** — the mechanism, the cleared suspects, the per-cell
assignments. Only the CoQA headroom figure moved.

- **THE MECHANISM.** The ladder isolates it. Not knowing the per-view signs costs a simple average
  `d_signs`; recovering it without labels is what L-SML's grouping is *for*. The recovery ratio
  `d_lsml_vs_avg / −d_signs` is **0.919–1.247 on every healthy cell** (median 1.025) and **below
  0.90 on 4 of 8 weak cells — 0 of 14 healthy cells is** (Fisher p = 0.0096 — **withdrawn in
  Step 212**, see above). A support
  difference, not a shift. The failing four: `seiclr_triviaqa_opt30b` (−1.269, L-SML makes it
  worse), `inside_coqa_llama7b` (−0.122), `ars_gsm8k_r1distill8b` (0.156), `noise_gsm8k_phi3mini`
  (0.761).
- **TWO SUSPECTS CLEARED, and clearing them was the point.** **Orientation**: the global-sign rung
  costs **exactly 0.00pp on 25/25 cells**. **K-selection**: the Step-205 degeneracy flag fires on
  **0/25**, and the eigengap helps on only 5/25 (mean −1.39pp). The Step-209 lead about
  `ars_gsm8k_r1distill8b` (K=4 → 0.364) sits on **`ALL_H16`, which we do not deploy** — withdrawn.
- **THE CONFOUND THAT NEARLY ATE THE DIAGNOSIS.** The nine weak cells are the nine lowest
  `anchor_auc`, and Spearman(anchor, deployed) = +0.981 — which reads as orientation. But
  Spearman(anchor, **best single view**) = **+0.975**: `epr` is a pooled feature, so a weak anchor
  just means every view is weak. Re-verified on freshly loaded data as a gate.
- **TWO SECONDARY MECHANISMS.** The selector **drops the pool's strongest view on 4 cells**
  (`internalstates_gsm8k_qwen25_7b` −4.81pp, `seiclr_triviaqa_opt30b` −4.57pp) — more than the
  whole gap to the ceiling, before fusion starts; two weak cells have Jaccard **exactly 0.000**
  against the label-chosen oracle-5. And **CoQA's views are non-monotone** (cross-fitted bin-mean
  gain +0.045, **z = +3.19**, max elsewhere +0.020) — signal no monotone fusion can reach, on the
  cell with the grid's largest headroom — of which only **+7.4pp is reachable by better
  selection**; the other **+17.1pp needs the sign recovery fixed** (Step 211).
- **THREE OF THE NINE HAVE NO DEFECT.** `truthfulqa_llama8b`, `lapeigvals_gsm8k_llama3b`,
  `trace_math500_qwenmath15b_k10` trip nothing — they are simply hard. **"Why do we fail here" has
  two answers**: six cells have a named fixable defect, three want an honestly reported ceiling.
- **NEXT — three repairs, pre-registered with gates in Step 210, none run.** Priority order:
  (1) a better label-free relative-sign estimator, candidate `orientation.z2_sign_recovery`
  (already written, unused on this path) — gate: recovery > 0.90 on the four, <0.5pp movement on
  the 22 healthy; (2) rank/quantile transform for CoQA — gate: no-op on the other 24;
  (3) unconditionally keep the strongest view — needs a label-free "strongest".
- **Gates**: GOOD_6 anchor 0.7594 · **both deployed arms reproduce to <5e-4 on 25/25** · ladder
  `r5 ≤ r4` everywhere · confound re-check +0.975 · 0 joined K/residual values.

</details>

<details><summary>Step 209 — the meeting record (still current)</summary>

**Last updated**: Step 209 — **the advisor meeting closed the feature-selection line.** Three
action items replace it. Full write-up in HISTORY.md Step 209.

- **THE DIRECTION IS CLOSED, and the repo already had the evidence.** L-SML over the full ~30-view
  pool, L-SML after DUFS selection, and U-PCR's own exclusion all tie: Step 207's `upcr` 0.7551 /
  `dufs_pf` 0.7507 / GOOD_6 0.7594, with **no pairwise contrast significant** (p = 0.059 / 0.191 /
  0.615). Step 206 had already closed pool composition in both directions.
- **THE THREE ACTION ITEMS**: (1) understand why we fail where we fail — per-cell deep dive;
  (2) consider clustering inside U-PCR; (3) consider adjacent applications — localization, and
  detection early in generation.
- **Item 2 is already answered, do not rebuild it.** Step 204 §D built the clustered variant
  (`spectral_utils/upcr_clustered.py`): it **failed both pre-registered gates and lost −4.46pp**
  (9W/16L, p = 0.030), and its premise was a confound — the 2.03× same-vs-cross fit gap collapses
  to 0.97–1.00× matched on |C_ij| decile. One variant untried (K-means on the (v₁,v₂) coordinates)
  and rated low: `lambda2_threshold` is inert and one-component U-PCR is exactly PC1 of the
  survivors, so the second component has nothing to cluster on.
- **Item 3 is the strongest publishable arm.** Extension E already has a replicated effect —
  `lsml16` beats the best DeepConf window by **+5.6pp [+0.9, +10.6]** at the **earliest 10% of the
  trace** — and Step 208 adopted `Online Auditing of Information Flow` for the missing stopping
  rule plus the corrected metric, **(AUROC at budget, tokens consumed)**.
- **Item 1 is the active work, scoped to DIAGNOSIS ONLY** (per Omri). Nine cells are "failing":
  `losnet_hotpotqa_mistral7b`, `inside_coqa_llama7b`, `seiclr_triviaqa_opt30b`,
  `truthfulqa_llama8b`, `internalstates_gsm8k_qwen25_7b`, `noise_gsm8k_phi3mini`,
  `trace_math500_qwenmath15b_k10`, `ars_gsm8k_r1distill8b`, `lapeigvals_gsm8k_llama3b`. Repairs
  are pre-registered and tested in a **later** step, so the diagnosis cannot be tuned to make a fix
  look good.
- **Orientation is still a genuine finding but is no longer the thing being worked on.** The
  "single open lever" framing below belongs to Steps 204–208 and is superseded as a *priority*,
  not as a result. **Step 210 closes it further**: the global bit costs 0.00pp on 25/25.

</details>

<details><summary>Step 207 + 208 headlines (still current)</summary>

**Last updated**: Step 207 — **the label-free standing page shipped, and building it exposed two
reporting errors that were in every draft.** Full write-up in HISTORY.md Step 207; page at
`results/action_items/labelfree_standing.html`.

**Also landed this session, from a parallel thread: Step 208** — the Huleihel / Oren-Loberman
publication line assessed against our open threads. Three proposed imports rejected, one adopted.
Full write-up in HISTORY.md Step 208.

- **Adopted: `Online Auditing of Information Flow`** (Oren-Loberman, Azar, Huleihel;
  arXiv:2310.14595, IEEE TSIPN 10:487-499, 2024) — digested. Sequential detection under a risk that
  prices **error and delay**; the optimal rule is a two-sided threshold on the posterior, a
  Wald-calibrated SPRT. That is exactly what **Extension E** lacks (the Step-148 pilot scores
  prefixes at fixed budgets with no stopping rule). Two caveats in the digest: the offline stage is
  **supervised**, and the graph/path machinery does not transfer to a single fully-observed decoded
  trace. **Cite it for the formulation, not the theorems.** Metric lesson: report
  **(AUROC at budget, tokens consumed)**, not AUROC alone — their accuracy is a wash (0.86 vs 0.85)
  and the entire contribution is 6.29 vs 12.75 events to decide.
- **`Inhomogeneous Submatrix Detection`** (arXiv:2603.09602) — extracted, **deliberately not
  digested**. Detection only: its tests do not localize the support, so it supplies no K\* selection
  criterion. Live angle, if any, is the variance-shift + consecutive-placement variant as a formal
  model for `sw_var_peak` window selection.
- **Two papers Omri picked out, now OBTAINED + EXTRACTED but NOT DIGESTED** — index rows are
  grounded in the extracts, but carry abstract-level claims only. **Run `/paper-digest` before
  citing anything deeper**; nobody has read either in full:
  - **`Detection and Recovery of Hidden Submatrices`** (Dadon, Huleihel, Bendory; arXiv:2306.06643v2,
    IEEE TSIPN 10:69-82, 2024) — the **recovery/localization** companion, and the answer to the
    objection that sank the 2026 inhomogeneous paper: it locates the planted support, with low-degree
    lower bounds and an impossible/hard/easy partition. **If the submatrix -> feature-selection idea
    is pursued at all, this is the entry point** — but it is *homogeneous* (one common elevated mean,
    mean-shift only), so it is a weaker model than the paper it corrects, and the idea it serves is
    still rated low because pool composition is closed in both directions.
  - **`Einstein from Noise: Statistical Analysis`** (Balanov, Huleihel, Bendory; arXiv:2407.05277v3,
    IEEE T-SP 74:1751-1766, 2026) — a formal analysis of an estimator manufacturing the structure it
    was told to look for: align pure noise to a template and average, and the **Fourier phases of the
    estimator converge to the template's phases** ("phase locking"). **Framing and citable
    methodological support, not a source of method**: Steps 203-206 are an empirical rediscovery of
    the same class of artifact. Companion, not obtained: `Confirmation Bias in Gaussian Mixture
    Models` (T-IT 2025, same authors).
  - **Standing is unchanged by having obtained them.** `Online Auditing` is still the only one of the
    four that touches an open thread.
- **`AdaRankGrad` (ICLR'25) is co-authored by Huleihel and O. Lindenbaum** — there is already a warm
  path to that group through Ofir.
- **No roadmap change.** Orientation remains the single open lever (Steps 204/206); none of these
  papers speaks to it. Extension E gains a formulation and a corrected metric definition for a
  future re-run.

- **NEW DELIVERABLE: `results/action_items/labelfree_standing.html`** (78 KB, self-contained, zero
  external references), built by `scripts/labelfree_standing_report.py`. One page replacing
  `item3_qa_evaluation.html` + `item4_benchmarking.html` for the two arms that need nothing
  hand-picked beyond the anchor bit, on the 25 in-scope cells. Nothing copied from the old pages:
  every AUROC recomputed through the canonical path with bootstrap CIs, behind two gates that abort
  the build — the GOOD_6 validity anchor at 0.7594, and per-arm reproduction within 5e-4 of the
  recorded value. Both pass 25/25.
- **ERROR 1 — `upcr.rho_polarities` keeps 21 of ~29 views, NOT 12.** `comparison.csv` prints
  `size_mean = 11.7` on *every* `upcr.*` row because `build_comparison.py:495-502` computes one
  shared `kept_on` from the 64-config factorial, and **every config there is hand-oriented**.
  Re-orienting by `sign(rho)` makes every rho positive, so far fewer views trip Algorithm 1's
  exclusion. Measured on the deployed `FIT`: hand arm **0.416 → 12.0 views**, `sign(rho)` arm
  **0.731 → 21.0**. "U-PCR keeps ~12 of 30, so it is itself a selector" describes the arm we do
  **not** deploy. The "found a selector" reading survives on Step 204's mechanism (one-component
  U-PCR is exactly PC1 of the survivors, so exclusion is the only live part) rather than on the drop
  rate, which is 8 of 29.
- **ERROR 2 — Bar B is NOT our cost class.** The headline "+8.7pp on 11 cells, p = 0.042" is **Bar
  B** (unsupervised, one pass, *any* access, i.e. it includes white-box competitors). **Bar A**, our
  exact grey-box class, is **+6.17pp / +6.56pp over 5 cells, p = 0.312**. Both source pages, both
  letter drafts and `benchmark_standing.py`'s section-3 heading called the Bar B number "our own
  cost class". Fixed in the new page and in `benchmark_standing.py` (regenerated).
- **Also corrected: GroupFS is 0.7481, not 0.7502** (`a2.select` vs `a2.dufs`) — the drafts gave
  DUFS's number to both.
- **Measured on the 25 cells, with CIs, for the first time on these arms:**

  | Arm | macro | QA (10) | math (15) | in-band (19) | kept |
  |---|---:|---:|---:|---:|---:|
  | U-PCR + sign(rho) | 0.7551 | 0.7126 | 0.7834 | 0.7593 | 21.0 |
  | DUFS parameter-free + L-SML | 0.7507 | 0.7089 | 0.7787 | 0.7532 | 16.9 |
  | GOOD_6 (reference) | 0.7594 | 0.7274 | 0.7807 | 0.7604 | 6 |

  Paired: `upcr − dufs_pf` +0.43pp 16W/9L p=0.059; `GOOD_6 − dufs_pf` +0.87pp p=0.191;
  `GOOD_6 − upcr` +0.43pp p=0.615. **Nothing separates the three.**
- **The QA deficit is ONE cell.** GOOD_6 leads QA by 1.49pp and trails math by 0.27pp. **CoQA alone
  contributes 13.19pp**; drop it and the QA gap over the remaining nine is **0.18pp**. Base model,
  14.7% positive rate, both label-free arms near chance (53.5 / 53.2) where GOOD_6 is not (66.7).
- **Step-155's QA gate re-run label-free: 4 of 4** (SQuAD v2 81.0, TruthfulQA 66.3, SciQ 74.1,
  NQ-Open 75.5). CoQA is deliberately not in that gate and was Item 3's top-priority dataset — the
  page says so, because the first draft did not and it read as a clean sweep of short-form QA.
- **Trivial-baseline floor: 10W/2T/7L over 19 cells, +1.14pp, p = 0.182** vs seq-logprob on our own
  traces. Ahead on balance, not significantly. Appendix, per the published-roster rule.
- **Flags now come from the scored-label positive rate, not task accuracy.** They differ >5pp on 3
  cells (SciQ 0.877/0.662, SQuAD v2 0.606/0.280, spilled TriviaQA 0.320/0.023) where only part of
  the traces carry every field. This re-flags cells vs the old pages: SciQ was CEILING, now in-band;
  `math500_dsmath7b` is now FLOOR.
- **HAZARD, now fixed: `scripts/build_glossary.py` was silently DELETING hand-written GLOSSARY.md
  content.** The file and its generator had diverged in both directions — running the generator
  destroyed the whole `## Grouping determinacy (Step 205)` section (6 terms), the `prior tiers` row,
  the a1 Step-205 caveat and the a6 anchor-only-tier text, none of which existed in
  `spectral_utils/glossary.py`. All of it is ported into the source now, plus a
  `GROUPING_TERM_NOTES` dict and its renderer section, and the regenerated GLOSSARY.md is verified a
  strict superset of HEAD (0 rows/sections lost, 8 Step-206 rows gained). **The rule "GLOSSARY.md is
  generated, hand-edits are overwritten" was being violated in practice — do not hand-edit it.**
- **Open follow-up (small):** `benchmark_standing.py`'s SHORTLIST still carries `a2.dufs`, not
  `a2.dufs_pf`, so `BENCHMARK_STANDING.md` does not show the arm the advisor letter now leads with.
  `cell_method_matrix.csv` has no `a2.dufs_pf` column; the per-cell values are in
  `results/selector_bench/a2_groupfs__c46.csv` (`variant == 'a2.dufs_pf'`).
- **Advisor letter**: reviewed against the data, three numbers corrected before sending (U-PCR keep
  count, the cost-class attribution, GroupFS's macro). The two drafts under `docs/meetings/`
  (`Advisor_Update_Jul2026_long.md` / `_short.md`) still carry the pre-correction wording and were
  deliberately not edited — treat them as superseded by the version Omri is sending.

</details>

<details><summary>Step 206 headlines (still current)</summary>

**Last updated**: Step 206 — **pool composition is closed as a lever, in both directions.** Full
write-up in HISTORY.md Step 206; pages at `results/upcr_study/08_pool_lovo/index.html` and
`results/upcr_study/09_add_test/index.html`.

- **REMOVING views is significantly HARMFUL on U-PCR** (`exp08`), where on L-SML it was a coin
  flip. Omri's objection to WS3 was correct and verified: `pipeline_lovo.py:95-96` defaults to
  `fusion='lsml'`, so all 775 WS3 runs used L-SML, and it ran 2026-07-23 — before Steps 204/205.
  Re-run LOCO-honest on `upcr.rho_polarities`: **−0.50pp, 7W/18L, p = 0.0096** at threshold 0.0pp;
  −0.046pp (p = 0.18) at 0.1pp; and at **≥ 0.2pp no view qualifies for removal at all**, on either
  path. Anchor gate: the FULL condition reproduces exp06's 0.7551 to 4dp.
- **THE MECHANISM — a pre-registered check failed, and that is the finding.** A view U-PCR already
  excludes (`w_i = 0`) should be a no-op to remove. It is not: where removal leaves the survivor
  set unchanged, **90.5% are exact no-ops** (mean |Δ| 0.035pp); where removal **changes which other
  views survive** (56 of 193 pairs), **0% are no-ops** (mean |Δ| 0.656pp). **U-PCR's Algorithm-1
  exclusion is data-dependent — exclusion and removal are different operations.** Pruning hurts
  because you are perturbing the estimator that decides what counts as dead, not deleting dead
  weight.
- **THE ADD TEST IS ALSO NEGATIVE** (`exp09`). `topk_tail_mass` / `renyi_entropy_2` rank #1 and #5
  of 30 by informativeness and had never been in any scored subset. Six variants pre-registered
  together; **all six land below GOOD_6**. Best is `ref.GOOD_6+topk` **0.7587** (−0.07pp vs GOOD_6,
  p = 0.426; −0.72pp vs LOCO_5). Worst is `ref.ENTROPY_6` **0.7462** (−1.32pp) — six readings of
  one quantity. Reproduced by `run_eval_pipeline.py` to 4dp on every row.
- **The shared lesson: high individual informativeness ≠ additive value.** `topk_tail_mass` is
  genuinely strong — `ref.LOCO_5` contains it, picked independently by the Step-195 exhaustive LOCO
  search. Adding it to a subset that already covers that direction buys nothing. Adding to GOOD_5
  helps (+0.39pp renyi, +0.15pp topk); adding to GOOD_6 hurts, because `varentropy` holds the slot.
- **Pool composition now has FOUR independent negatives** (WS3 LOCO, pool-size, inclusion audit,
  exp08) plus exp09 on the add side. **Orientation remains the single open lever** — the NEXT item
  below is unchanged.
- **A THIRD STALENESS CARRIER, beyond Step 193's three: a cached ERROR row.**
  `a6.pruned_dufs` carried `{"error": "name 'mu3' is not defined"}` on 11/25 cells, falling back to
  the full pool (size 27-30 against a declared `k_max=15`). **`mu3` exists nowhere in the current
  codebase** — cached from a code version that no longer exists and kept alive by resume-skip,
  which only stale-gates on row `n`. Dropped + re-benched: **the true value is 0.7514 macro /
  0.7117 QA / 0.7779 math**, uniform size 17.0, 0 errors, 0 fallbacks — still below `a6.pl_dufs`
  (0.7524) and GOOD_5 (0.7519), so the old verdict holds even though all four previously-quoted
  numbers were wrong (Step 197 claimed 0.7596, GLOSSARY said 0.7537, the postfix CSV 0.7487, the
  contaminated bench 0.7456). **General point: the stale-gate cannot see code changes.** Every bench CSV predates the
  Step-205 grouping fix; only the 430 size-4 rows can actually move, and no headline row is among
  them.
- **`a6.adaptive_pl_mrmr` surfaced as a new bench row at 0.7569** — above the selector of record
  `a6.pl_dufs` (0.7524), and 6th on the scoreboard. Not yet investigated.
- Also corrected: the GLOSSARY coverage gate was **failing** on two pre-existing gaps
  (`a7.iter_consensus`, `a6.adaptive_pl_mrmr`) — the "0 gaps currently" note below was stale.
  Both entries added to `spectral_utils/glossary.py` (**GLOSSARY.md is generated — hand-edits are
  overwritten**).

</details>

<details><summary>Step 205 headlines (still current)</summary>

**Last updated**: Step 205 — **every published number replayed through today's code; the one real
defect found, fixed exactly, and gated.** Full write-up in HISTORY.md Step 205; the advisor-facing
page is `results/upcr_study/comparison.html`.

- **UNEXPLAINED ROWS = 0**, and **the instability is gone, not just documented.** All 169 published
  (variant, pool) rows were replayed through current code and cross-tabulated against their own
  numerical noise. Re-running the jitter audit *after* the fix is the pass/fail test, and it passes
  outright: rows moving <0.1pp go **134/165 → 165/165**, rows ≥0.5pp go **22 → 0**, the size-4 band's
  median spread goes **0.439pp → 0.000pp** (18 unstable rows → 0), and Spearman(size, spread) goes
  −0.492 → −0.072 — the size dependence disappears because there is no instability left to predict.
  The worst remaining row moves a single *cell* by 0.02pp.
- Page verdicts: **63 verified / 49 within their own noise / 34 `code fix: Step-205 exact small-m
  solve` / 8 lookup-table / 4 not replayable / 2 Step-189 K clamp**. The exact count fell (75 → 63)
  *because* the fix moved the size-4 rows and the post-fix noise floor is ~0, so any drift above
  0.02pp must now be named rather than absorbed. Those old values were tie-breaks; today's code
  returns a defined answer.
- **THE ANSWER TO "did the fixes help, or is it just instability?" — neither fix improved any
  algorithm, and the instability is real but sharply localised.** The m<4 short-circuit restores old
  numbers by design; the corrected loading scale is +0.08pp (10W/15L); the new exact small-m solve
  is +0.03pp (15W/10L, p=0.696). **Determinacy was the only thing on offer.** Under a 1e-10 jitter,
  134 of 165 rows move <0.1pp; the whole problem sits at **mean size 4** (median spread 0.439pp,
  18 of 36 rows ≥0.5pp) against 0.000pp at every other size. **Size 3 is degenerate but
  deterministic** (Eq.15 is exactly zero → constant tie-break); **size 4 is meaningful but
  undetermined** (Eq.15 has two terms; K ∈ {2,3} decided in the last bits).
- **THE DEFECT, AND THE FIX.** Spectral clustering is a *heuristic* for "partition minimising the
  Eq.14 residual"; at small m it ties and float noise settles it. Demonstrated at 5.55e-17: `np.cov`
  on a non-contiguous column slice vs a contiguous copy of the same numbers flips the m=4 partition
  and moves one cell 9.7pp. Now: **canonical covariance input** at every m, **exact enumeration of
  all partitions at m ≤ 4**, and a **near-degeneracy flag** (`return_diag=True` →
  `meta['grouping_diag']`) that fires at any m when the winner beats its rival by less than noise.
  **Every headline anchor moves 0.00pp** (GOOD_6 0.7594, GOOD_5 0.7519, LOCO_5 0.7705, `a6.pl_dufs`
  0.7524, `a2.dufs` 0.7502); only the m=4 reference `ref.consensus_4` moves, +0.60pp.
- **THE STANDING MECHANISM: gate U5.** Asserts the grouping is invariant to (i) memory layout,
  (ii) feature-order permutation, (iii) a 1e-12 jitter, on real cells at m=3..8. **An invariance
  failure is the signature of an answer decided by rounding, not by data.** It already found one
  near-tie above the exact-solve cutoff: `m=8 math500_r1distill8b` is relabel-dependent. Gate U6
  covers two smaller determinism/robustness holes (`sml_fuse_signed`'s even-k tie, `zscore` on
  non-finite input) — both verified inert on current data before being fixed.
- **STEP 204's HEADLINE (P2) SURVIVES, re-measured on fixed code.** All three size-3-inclusive
  pruning studies were re-run. `exp06`: Spearman(misfit, AUROC) unit **+0.222** (published +0.223,
  23/25 positive) → complete **−0.022** (published −0.006, 10/25), shift **−0.243, p = 0.0006**
  (published −0.228, p = 0.0015) — **P2 holds, slightly strengthened**, so the loading-scale
  correction and the decision *not* to build Extension I1 rest on current numbers. `exp01_grouping`:
  grouping OFF beats ON at **13/13 sizes, 12 at p<0.05** — unchanged. `exp03_preflight`: typical
  accuracy rises with size in **25/25 cells**, and *only the k=3 endpoints moved* (typical
  0.6928→0.6881 at k=3 but 0.7450→0.7450 at k=21; best-found 0.7740→0.7726 at k=3 but
  0.7634→0.7634 at k=25) — a clean confirmation of the fix's blast radius.
- **STEP 204's B1 IS NARROWED.** "The g2 search range never binds" is true of the **pre-exclusion**
  fit (which is the only one exp01 draws) and **false** of the g2 the pipeline returns: Algorithm 1
  excludes weak experts and recalculates g2 on the ~12 survivors of ~29, where it lands **exactly on
  the ceiling in 24/25 cells**. The conclusion survives — un-pinning it is −0.28pp, 12W/13L, a wash.
  And g2 is **not** the component-count dial: `auto_components` keys off `lambda2_frac > 0.1`,
  computed before g2 and independently of it.
- **STEP 204's R2 IS QUALIFIED, and this one matters for how we report.** The 2-eigenvector rule at
  "−3.67pp, p=9.1e-05" is a **factorial main effect** — each cell's delta averaged over the 32
  combinations of the other five factors, most of which we never run. At the **deployed**
  configuration the identical switch is **−0.43pp mean / +0.07pp median, 15W/9L, p=0.16** (new
  `exp07`, verified two independent ways). The reversal of Step 142's *sign* stands; the *magnitude*
  does not transfer. **Do not quote a factorial main effect as a deployed cost.**
- **`lambda2_threshold` — the most promising open lead — is dead.** Sweeping it 0.05→0.25 moves the
  component count on 24/25 cells and buys **+0.43pp (9W/15L, p=0.16)**. Consistent with Step 204's
  finding that one-component U-PCR is exactly PC1 of the surviving features: the estimation
  machinery is inert on our data, and what mattered was orientation and exclusion.
- **NEXT** (unchanged from Step 204): build the **orientation** result into a proper experiment —
  `sign(ρ̂)` polarity against `ALL_SIGNS` on the L-SML path (a free no-op) and on every
  sign-sensitive consumer (where it is not), plus a decision on the 15 mis-signed entries. Do **not**
  build Extension I1. **Step 206 reinforces this**: pool composition is now closed in both
  directions (four negatives on removal, six on addition), so orientation is the only lever left.

</details>

<details><summary>Step 204 headlines (still current except where Step 205 narrows them above)</summary>

- **STEP 203's HEADLINE IS SUPERSEDED.** The "+0.223, every selector minimised what should have been
  maximised" finding is an **artifact of the L-SML loading scale**. `_estimate_von_voff` returned the
  unit-norm eigenvector where Lemma 1 requires the loadings to reproduce the covariance, so misfit was
  inflated by group size. Corrected: Spearman(misfit, AUROC) goes **+0.223 (24/25 positive) → −0.006
  (12/25)**, shift −0.228, **p = 0.0015**. The `unit` arm reproduces Step 203 exactly, so the harness
  is sound. **→ Extension I1 (sign-flip the selectors) is the wrong remedy and should not be built.**
- **The SPEC's own proposed fix was not the right one.** `SPEC_residual_scaling_fix.md` pre-registered
  `eigen` (`sqrt(λ₁)·v`); it **fails the SPEC's own U1 check** (0.2500 at m=2, identical to the broken
  path) and drops 6/25 cells to K=2. A masked rank-one **completion** estimator is exact (~1e-25) and
  keeps every cell at K≥3. Per Omri, the loading scale is now **reported three ways everywhere rather
  than chosen** (`common.SCALES`), because `complete` is also the convenient answer.
  Anchors hold: GOOD_6 = 0.7594 and K=4 / residual 88.455 / sizes [5,7,7,11] on flag-off.
- **U-PCR: every claimed deviation from the paper was real, and none of them helps.** All seven
  verified against the paper. On fixed code **two previously-settled results reverse**: the
  2-eigenvector rule is **−3.67pp** (3W/21L, p=0.0001), not Step 142's +0.5pp (that measurement was
  confounded by the g2 range cap — g2 *is* the 1-vs-2 dial); **→ QUALIFIED IN STEP 205**: −3.67pp is
  a factorial main effect averaged over 32 other-factor combinations; at the deployed config the
  same switch is −0.43pp / +0.07pp median, 15W/9L, p=0.16. And g2 is *not* the 1-vs-2 dial —
  `lambda2_threshold` is, and it is inert (exp07). The sign reversal stands; the magnitude does not
  transfer. And the absolute loss is a **wash**
  (+0.07pp median, 13W/12L, p=0.615), never actually measured before. Being faithful *loses*
  (0.6910 vs 0.7392, but only −0.18pp median, p=0.173). No configuration of 64 beats GOOD_6.
- **The g2 search range — the headline suspect — never binds.** Chosen g2 sits at q≈0.01–0.08 and
  widening the range 16× moves it in **0/25 cells**. `var_y` acts entirely through the **abstain gate**
  (cells declared too hard 3.9 → 17.6 of 25): a routing knob, not a weight-estimation knob.
  **→ NARROWED IN STEP 205**: true of the pre-exclusion fit only. The g2 `upcr_fit` returns comes
  from the post-exclusion refit and **is** at the ceiling in 24/25 cells. Un-pinning it is a wash
  (−0.28pp, 12W/13L), so the conclusion holds and the mechanism sentence does not.
- **The clustered variant is refuted, and its premise was a confound.** Fit error is essentially pair
  correlation (Spearman **0.870**); the raw 2.03× same-vs-cross gap collapses to **0.97–1.00** when
  matched on |C_ij| decile, a random partition gives the same, and **magnitude-only clustering
  separates it *better* (3.06–3.81)**. The variant fails both pre-registered gates and loses
  **−4.46pp** (9W/16L, p=0.030). Identifiability requirement derived and enforced: **K ≥ 3** (the
  cross-cluster pair graph is complete multipartite; at K=2 it is bipartite and rho is unidentifiable).
- **THE FINDING WORTH CARRYING FORWARD — orientation.** Deriving per-feature polarity from `sign(ρ̂)`
  **beats the 42 hand signs**: 0.7551 vs 0.7405, **+1.46pp, 20W/5L, p<0.001**. And **15 of 30 pool
  features carry the wrong hand sign** (`epr_spilled` 0.277, `cusum_max_spilled` 0.281 oriented AUROC,
  both below 0.5 in 25/25 cells) — though correcting them moves GOOD_6/LOCO_5 by **exactly 0.0000pp**
  (sign-gauge invariance, re-verified). Structure recovers the **empirical** direction on **91.8%** of
  features (p<0.001). **The global ±1 is provably NOT recoverable** — a global flip leaves ρ̂
  bit-identical (max|Δρ| = 0.000e+00), so the anchor bit cannot be derived from covariance structure.
- **An independent adversarial review found 17 defects in this session's own work**, several changing a
  conclusion — including a `pairs=None` that made the clustered variant measure an all-pairs fit, two
  anti-regression gates that were no-ops (`bool((ok, macro))` is always True), and main-effect p-values
  pseudo-replicated to n=800. All fixed and re-run; withdrawn claims are named in HISTORY Step 204 §F.
- **The honest headline**: one-component U-PCR is *exactly* PC1 of the surviving features (cosine
  deviation 7e-12), so the whole ρ/g²/Eq.-20 apparatus enters only through the exclusion mask.
  **U-PCR's estimation machinery is inert on our data; what mattered was orientation and exclusion.**
- **NEXT**: build the orientation result into a proper experiment — `sign(ρ̂)` polarity against
  `ALL_SIGNS` on the L-SML path (where it is a free no-op) and on every sign-sensitive consumer (where
  it is not), plus a decision on correcting the 15 mis-signed entries. Do **not** build Extension I1.

</details>

<details><summary>Step 203 headlines (superseded above, kept for the record)</summary>

- **HEADLINE — every selector we built minimised a quantity that should have been maximised.**
  Two independent measurements, both on live current data:
  | Evidence | Number | Consistency |
  |---|---|---|
  | Within-size Spearman(misfit, AUROC), 30-view pool, 6,756 subsets | **+0.223** mean / +0.185 median | **24/25 cells positive** |
  | Repair worst-fitting group vs repair **random** group | **−2.22pp** | W/L 7/18, **p = 0.032** |
  Misfit is *lower = better fit*, so **worse-fitting subsets score higher**. Mechanism: the
  worst-fitting group is reliably the near-duplicate confidence cluster (`epr`, `epr_spilled`,
  `epr_energy`, `mean_top1_logprob`) — the *strongest* views, which break the rank-one model
  **because** they are several readings of one quantity. **Poor fit marks where the signal is, not
  where the junk is.** → new **Extension I** in Research_Directions.md, with I1–I4 and theorems T1–T4.
- **Omri's cluster-localized idea had never actually been tested.** The prototype's fit score
  (`test_iterative_lsml_pruning.py`) is `‖Cov·v₁ − λ₁·v₁‖` = **zero by construction** (2e-15); it
  ranked removals by rounding error, so the recorded 0.7004 is void. None of its three arms was the
  proposed algorithm. Now run properly — and refuted, but for an informative reason (above).
- **~1.03M cached subset scores are STALE.** Only **5/19** repgrid cells in `results/subset_sweep/`
  still reproduce; disagreements to **0.374 AUROC** (cells re-graded after the sweep). The npz files
  look healthy — only re-scoring detects it. *An earlier pass of this study used that cache and
  reported the correlation as ≈ −0.02; superseded by +0.223.* Audit:
  `results/pruning_study/03_size_and_criterion/cache_staleness_audit.csv`.
- **No interior best size, on live data and the full pool**: typical-subset accuracy rises
  monotonically with size in **25/25** cells (0.6928@k=3 → 0.7450@k=21) while best-found-at-size falls
  (0.7740@k=3 → 0.7634@k=25). Trimming has a **high ceiling and a poor average** — all value is in
  choosing well, none in being small. No turn exists for a stopping rule to find.
- **Weight estimation (R3−R2 = +1.45pp) is measured but not closed** (→ **Extension J**). A 2×4×2
  factorial spans only **0.7434–0.7555**. Main effects: triplets +0.21pp, low-rank+sparse +0.11pp,
  RMT cleaning +0.14pp, robust-IRLS −0.33pp, **precision weighting −0.13pp** (predicted a priori at
  +0.5…+1.2pp). Diagnostic: second factor at **0.312** of the first, rank agreement +0.186, **top-5
  overlap 1/5** — the label-free estimator and the supervised model disagree about *which* views
  matter, so this is not a calibration problem.
- **Grouping step does not earn its keep**: OFF beats ON at **every** size (13/13, 12 at p<0.05);
  full pool 0.7457 → 0.7533 (p=0.024, 17/25). Effect <1pp — it fails at its own job rather than being
  idle (near-duplicates exist in every cell, max |ρ| 0.996–1.000).
- **Shared-code speedups, all output-identical**: `_score_matrix_lsml` vectorised (**34×**),
  `_residual_lsml`/`_estimate_von_voff` vectorised, and new
  `lsml_continuous(..., compute_score_matrix=False)` skips the unused O(m⁴) matrix on the `groups=`
  path (**103×** at m=30). Regression anchor held (K=4, residual 88.455, sizes [5,7,7,11]); GOOD_6 =
  0.7594 asserted at the top of every experiment.
- **Reference-table corrections**: `ref.LOCO_5` (0.7705, 24 cells) was missing — GOOD_6 is *not* the
  selection ceiling. And **`a6.pl_dufs` (0.7524) is label-free at runtime but seeded from GOOD_6**
  (chosen with answer keys), and is selector of record *by default, not by merit* (both gates failed).
- **MECHANISM FOUND (post-commit, 2026-07-26)** — Omri: *"L-SML clustering is supposed to cluster the
  dependent features together — isn't that the assumption?"* It is, and the clustering works. The
  **residual scoring it is mis-scaled**: `_estimate_von_voff` returns the **unit-norm** eigenvector,
  but Lemma 1 requires `v_i·v_j = r_ij` (i.e. `a = √λ₁·v`). A perfect `m`-duplicate block — the ideal
  the clustering exists to produce — scores misfit/pair **0.25 → 0.83 as m goes 2 → 11**. So misfit is
  inflated by **group size × coupling strength**, and "repair the worst group" means "dismantle the
  biggest tight cluster": **the selection step optimises against the clustering step.** This supersedes
  the "redundancy and informativeness travel together" explanation (that was the symptom). It also sits
  in the **deployed detector** — K is chosen by minimising this residual, and **15/25 cells are pinned
  at K ≥ 7**, the predicted upward bias. Written up with a full test plan in
  **`SPEC_residual_scaling_fix.md`** (nothing changed in code yet).
- **NEXT**: **`SPEC_residual_scaling_fix.md` (Extension I0) before anything else** — scale `v_on` by
  `√λ₁` behind a flag, then read predictions P1 (K falls), P2 (the +0.223 sign weakens/flips), P3 (the
  −2.22pp localizer deficit shrinks/reverses), with anchors R1 (GOOD_6 = 0.7594 on flag-off) and R2
  (K=4, residual 88.455, sizes [5,7,7,11]). **If the sign flips, I1 below is the wrong remedy.**
- **THEN (fallback)**: **Extension I1 — sign-flip the existing selectors** (`a1.residual`, `a6.pl_dufs`, the
  Step-203 localized arm) to maximise misfit instead of minimising it. Hours of work on code that
  already exists, and decisive. Bar to clear is **0.7524** (the automatic-picker bar), not 0.7594.
  Per Omri: **report effect sizes with W/L + Wilcoxon; do not gate on 1–2pp differences.**
  *(Step 204 resolution: P1 HOLDS, P2 HOLDS — the +0.223 goes to −0.006, p = 0.0015 — so **I1 is the
  wrong remedy and is not to be built**. P3 was not measured separately: it asks the same question as
  P2 and P2 answered it.)*

</details>

<details><summary>Step 202 headlines (superseded by 203 above)</summary>

</details>

<details><summary>Step 201 headlines (the audit — superseded by 202 above)</summary>

**Step 201** — **Audit of the Step-200 Extension H build.** Full write-up in HISTORY.md Step 201. Headlines:
- **R6 RAN AND IS THE HEADLINE: `R6 = 0.7676` = +0.82pp over GOOD_6 → DEAD** by the pre-registered
  ≥+1.0pp gate (`ladder_gates.json` records `"verdict": "DEAD"`). A *perfect, label-derived* consensus
  target still lands inside noise of GOOD_6, so **target quality is NOT the cap** — the constraint is
  downstream, in fusion / weight estimation. This closes H3's premise.
- **The gap-ladder is now trustworthy.** The `StratifiedGroupKFold` fix worked: R3 = 0.7809 vs LR
  oracle 0.7810 ✅, R0@GOOD_6 = 0.7594 ✅, and R4 fell 0.7938 → 0.7659 (now −0.015, p = 0.004) — the
  k=10 leakage is gone.
- **H1 is settled by measurement, and there is no headroom.** L-SML is *exactly gauge-invariant* to
  input feature signs (1150/1150 sign vectors incl. 20 random per cell — bit-identical, worst dev
  `0.000e+00`). So `ALL_SIGNS` (42 hand-derived polarities) is a **free no-op** to remove; the
  ladder's `dominant_term: sign_recovery` is an **artifact** (R2−R0 confounds sign with fusion
  method — sign alone is worth 0.0000pp); and the prior-free skew tiebreaker is **refuted** (only
  9/25 cells have pos_rate > 0.5; it costs −13pp and inverts 6 cells).
- **Step 200 is banner-annotated SUPERSEDED IN PART** — four of its claims are contradicted by the
  files it cites: it calls the DEAD R6 gate "viable"; "100% sign accuracy" is the gauge, not accuracy;
  "adaptive K" is the constant `{3: 25}`; and a7's "prior-free 0.6840" actually uses the epr anchor
  (the genuinely prior-free arm is **0.5103**, chance).
- **8 code defects catalogued (Step 201 §C); the prior-free numbers are void pending fixes.** Most
  consequential: `sweep_dufs_groupfs.py` **never runs GroupFS** — it is `sklearn
  AgglomerativeClustering`, and λ1 is written to the CSV but never used in any computation (λ1 changed
  nothing in **0/350** configs). **GroupFS grouping therefore remains untested.** Fixes + corrected
  numbers land in Step 202.

</details>

<details><summary>Step 200 headlines as originally written (superseded — see Step 201)</summary>

- **Phase 0 R6 Ceiling**: Gemini's refreshed run verified on disk (`ladder_gates.json`). Reached **$0.7676$ macro AUROC** (+0.82pp over `GOOD_6`), confirming target quality as a real lever.
- **Phase 1 H1 Orientation ($Z_2$ Synchronization)**: Built `spectral_utils/orientation.py`. $Z_2$ eigenvector sign recovery (`z2_sign_recovery`) achieves **100% relative sign accuracy** on `GOOD_6` ($0.7594$ macro AUROC), solving relative feature orientation. Pure feature-free skewness (`distributional_orient`) drops to $0.5103$ macro AUROC because Math cells are symmetric/left-skewed (1 anchor view still required for global $\pm 1$ sign).
- **Phase 2 H2 Signal Dimension (Adaptive $K^*$)**: Added participation ratio $(\sum \lambda)^2 / \sum \lambda^2$ (`eff_rank`) and Marchenko-Pastur noise floor (`mp_floor`) to [spectral_utils/selectors/adaptive_k.py](file:///C:/Users/omris/TAU/hallucination_detection/spectral_utils/selectors/adaptive_k.py). Dynamically sizes $K^* \approx 3..6$ per cell, solving the fixed $K=15$ budget requirement.
- **Phase 3 H3 Iterative Selector**: Registered `a7.iter_consensus` in [spectral_utils/selectors/a7_iter_consensus.py](file:///C:/Users/omris/TAU/hallucination_detection/spectral_utils/selectors/a7_iter_consensus.py) with Z2-synchronized fusion. Smoke test passed; achieved $0.6840$ macro AUROC prior-free.
- **Phase 4 & 5 GroupFS & Integrated Benchmark**: Built [scripts/sweep_dufs_groupfs.py](file:///C:/Users/omris/TAU/hallucination_detection/scripts/sweep_dufs_groupfs.py) and [scripts/prior_free_bench.py](file:///C:/Users/omris/TAU/hallucination_detection/scripts/prior_free_bench.py). GroupFS $C=3$ latent clustering reaches **$0.7063$ macro AUROC** prior-free, demonstrating data-driven feature selection.

</details>

---

### Step 199 (Summary) — Gap-decomposition ladder & prior-free pivot

**Standing from Step 198 (still current):** GOOD_6 0.7594 / QA 0.7274 / math 0.7807 unbeaten by any label-free selector; D1 adaptive-K refuted (r_s=+0.007 vs oracle-K); D2 PL-mRMR bounded (beats GOOD_5 p=0.037, under GOOD_6); supervised LR oracle 0.7810 / QA 0.7524 is a stationary linear model, so the gap is label-free estimation not model capacity; QA deficit concentrated in `inside_coqa` (estimation) + `seiclr_triviaqa` (feature coverage).

---

<details><summary>Step 197 headlines (superseded above, kept for the record)</summary>

**Last updated**: Step 197 — **Feature selection pruning, multi-anchor audit, honest LOCO CV tuning, pure unsupervised control, and advisor update letter (Joint with Antigravity AI).** Full write-up in HISTORY.md Step 197. Headlines:
- **`a6.pruned_dufs` registered**: Reaches **0.7596 Macro AUROC** with `logprob_margin` anchor (matching `GOOD_6` baseline 0.7594 label-free) and **0.7741 Macro AUROC on Math cells** under honest LOCO CV (beating `GOOD_6` 0.7594 and `GOOD_5` 0.7519).
- **Hyperparameter Pruning**: Enforcing target size cap $K_{max}=15$ raises AUROC from 0.7524 to 0.7549 while saving 2.6 features per cell.
- **Label-Free Structural Diagnostics**: Proved that L-SML covariance residual ($r = +0.648$) and Spectral Gap ($r = +0.423$) continuously estimate optimal subset size $K_{cell}^*$ label-free via closed-form formula $K_{cell}^* = \arg\max_k (\varepsilon(k+1) - \varepsilon(k))$.
- **Pure Unsupervised Control (`task-354`)**: Evaluated pure unsupervised DUFS ($\lambda_3=0$), proving pseudo-label agreement ($\lambda_3$) is essential (+1.60pp overall, +2.89pp on QA cells).
- **Advisor Handoff Drafted**: `HANDOFF_advisor_letter.md` fully updated with "I" voice, Mermaid pipeline diagram, mathematical formulas, and explicit DUFS paper citation.
</details>

- New `scripts/run_eval_pipeline.py` is now the "one checkpoint" — run it instead of trusting
  stale per-script numbers; writes `results/checkpoints/scoreboard_latest.csv` with a `role`
  column and a dynamically-computed best-ref delta (the old hardcoded `delta_vs_good5` column,
  not CLAUDE.md, was why everything defaulted to comparing against GOOD_5).
- **`cell_oracle_vs_chosen.py`: mean per-cell oracle ceiling 0.7998 vs `a6.pl_dufs` 0.7524 (our
  actual selector) = +4.74pp gap, only 0.169 feature-overlap Jaccard** — real room left on the
  table, and the selector reaches for different features than the oracle, not just fewer of the
  same ones.
- Orientation (`anchor_orient`), K-selection (residual over eigengap), and the a6 seed choice
  were all re-audited on the current 25-cell in-scope grid and all held up — no pipeline change
  needed from any of these three.
- GroupFS + DUFS verified **term-by-term faithful** to their source papers (4 documented,
  non-bug deviations); added the paper's missing Eq.7 parameter-free DUFS loss as `a2.dufs_pf`
  (ties `a2.dufs`, 0.7507 vs 0.7502).
- **`GLOSSARY.md`** (repo root) now decodes every subset/selector/variant/pool-mode nickname AND
  documents all 30 features (formula, paper origin, empirical best-domain AUROC, HISTORY
  pointer) — build has a hard coverage gate. **CORRECTED (Step 206): it was FAILING** on two
  pre-existing gaps (`a7.iter_consensus`, `a6.adaptive_pl_mrmr`); both now have entries and it
  passes again. **GLOSSARY.md is generated from `spectral_utils/glossary.py` — hand-edits to the
  .md are overwritten on the next build.**
- ~~**WS3 (exhaustive pipeline-level LOVO redundancy test) was STILL RUNNING at session end**~~
  **CORRECTED (Step 206): WS3 FINISHED 2026-07-23.** `pipeline_lovo_loco.csv` has all 100 rows
  (4 thresholds × 25 cells). Verdict: **negative** — mean held-out Δ −0.22pp (11W/12L/2T) at
  threshold 0.0pp, +0.04pp (14W/7L/4T, p = 0.23) at 0.1pp, and **nothing qualifies for removal at
  ≥ 0.2pp**. Re-run on U-PCR in Step 206 (`exp08`), where it is significantly harmful. Do not
  re-launch.
- **Nothing from this session is committed to git yet** — `GLOSSARY.md`, all new
  `scripts/*.py`, `spectral_utils/glossary.py`, and the modified selector/reference-subset
  modules are new/modified and uncommitted.
- **Coverage-matched delta fix (post-hoc, same session)**: Omri asked why `ref.LOCO_5` only
  scores 24/25 cells and whether that's inflating its lead. Answer: the missing cell
  (`inside_coqa_llama7b`) is genuinely one of GOOD_6's weaker cells (0.667, 4th-weakest of 25),
  so GOOD_6's own macro rises +0.38pp once it's excluded — but LOCO_5 still beats GOOD_6 by a
  real +0.73pp on the identical 24-cell set (already Step 195's comparison, not new). The
  actual bug this surfaced: `scoreboard()`'s `delta_vs_current_best_ref` was a raw `macro_all`
  subtraction with NO coverage check, so a lower-coverage variant could look like the leader
  with no warning. Fixed in `run_eval_pipeline.py` — read `delta_vs_current_best_ref_MATCHED`
  (computed on the shared-cell intersection) in `scoreboard_latest.csv`, not the raw column,
  whenever `n_cells_shared_with_best_ref` is less than a row's own `n_cells`.

**Previously (Step 194)** — a6 pseudo-label gates built + benched: both pre-registered gates
FAIL, yet `a6.pl_dufs` is adopted as the SELECTOR OF RECORD** (supersedes `a2.dufs`). Omri's idea
implemented as `spectral_utils/selectors/a6_pseudolabel_gates.py`: 4 seed views (`epr`,
`low_band_power`, `spectral_entropy`, `cusum_max` — same set resolved on all 25 cells) fused by
continuous L-SML into a pseudo-label that supervises the DUFS gates via a **centered** agreement
term (centering = redistribution of the sparsity budget, not relaxation; lam2 chosen by the same
stability rule as the control BEFORE lam3 enters). Verdicts on 25/25 cells, 0 fallbacks:
**mechanism gate FAIL** (rho(gate, view AUROC) +0.207 vs threshold 0.30; a2 baseline +0.149),
**performance gate FAIL** (+0.22pp vs a2.dufs, 14W/7L, Wilcoxon **p = 0.0273** — significant but
below the +1.0pp bar sized to the GOOD_6 gap). Still: macro **0.7524 = best label-free selector,
first to nominally edge GOOD_5** (0.7519, +0.05pp, p = 0.173 n.s.); GOOD_6 0.7594 remains the
headline detector. Ablations: gates > pseudo-label ranking +0.37pp; seeds +0.46pp; `a6.dufs`
control reproduces `a2.dufs` +0.06pp (harness sane). **The gates govern the claim, not the tool**
— report as "best selector, parity with GOOD_5, gap to GOOD_6 not closed". Also Step 194:
c46 subset sweep (sizes 3-5) running — see sweep section below; three item4-idiom SVG figures on
the advisor_inscope pages; `scripts/reconcile_competitors.py` → 59 MATCH / 5 DELTA (all five =
known Step-193b LapEigvals corrections, 3 hand-verified vs papers/extracted) / 15 coverage-only;
NOTE `report_figs.OVERRIDE_Y` is still load-bearing (`scores_lsml_upcr.csv` still says 0.925).

**Previously (Step 193b)** — **LapEigvals located and verified; all 5 of its anchors were mislabeled.** The paper was in `papers/` all along under its *title*, `Hallucination Detection in LLMs Using Spectral Features of Attention.pdf` (Binkowski et al., Wroclaw/UTS) — not the method name, which is why the Step-193 audit left it UNVERIFIED. Table 1 (temp=1.0, cols `CoQA | GSM8K | HaluevalQA | NQOpen | SQuADv2 | TriviaQA | TruthfulQA`): **4 of our 5 stored "LapEigvals" anchors are actually the paper's `AttentionScore` baseline** (0.717/0.666/0.630/0.576 — values right, method name and supervision tag wrong), and **`lapeigvals_gsm8k_llama8b`'s 0.925 is Mistral-Small-24B's LapEigvals**, a different model (its own model gives 0.720/0.872). **Substantive point: LapEigvals is SUPERVISED** — a logistic-regression probe (`max_iter=2000`, `class_weight='balanced'`) over Laplacian eigenvalues of attention maps; the caption itself says *"We mark results for AttentionScore in gray as it is an unsupervised approach, not directly comparable to the others."* So `AttentionScore` is the correct like-for-like comparator for our label-free detector, and LapEigvals belongs against our LR oracle. Correctly paired: **label-free — GOOD_6 beats AttentionScore on 4/5 cells** (loses only Llama3.2-3B 0.703 vs 0.717); **supervised — LapEigvals beats our LR@30 on all 5 by 6–12pp** (0.870–0.925 vs 0.752–0.869), though not a strict head-to-head (their signal is attention maps / white-box internals, ours is the entropy-logprob trace; different generation + grading + split). Verified anchors **10/18 → 15/18**. **Trace-based selector vs GroupFS**: `a2.dufs` 0.7502 vs `a2.select` 0.7481 — **+0.20pp, 16W/9L, Wilcoxon p=0.173** → nominally better and structurally cleaner (0/25 saturated vs 12/25) but **not significant**, and both below GOOD_5 0.7519. **Coverage audit: NO cell was excluded** — all 25 present in every analysis artifact; the only 19/25 is `competitors_verified.csv` (the 6 new cluster cells have no published paper) and those same 6 lack the split-half `fulloracle` column (never exhaustively swept).

---

## NEXT SESSION — cluster results are WAITING; 2 GPQA long-answer jobs stalled

**Cluster state as of 2026-07-21 (checked via cluster-ops; `squeue` is EMPTY).** The jobs Omri
submitted 2026-07-20 have finished. 40 slurm jobs, all `spectral_infer`; the 21 "FAILED" entries
are exit-85 preemption/requeue links, not real failures.

**9 of 11 presets COMPLETE — pkl ready to fetch** from
`/shared/cycle2_tau_averbuch_prj/omrisegev1/results/regen/<preset>/`:
`gpqa_qwen72b` (1.2G), `gpqa_llama70b` (1.2G), `gpqa_llama8b` (1023M), `gpqa_r1distill8b` (3.2G),
`gpqa_mistral7b` (803M), `trace_gpqa_r1qwen7b` (3.2G), `trace_gsm8k_llama8b_k10` (1.1G),
`trace_math500_qwenmath15b_k10` (1.2G), `math500_r1distill8b_mn4096` (654M). All 198/198 (GPQA) or
full-N, each ending `ALL TEMPS COMPLETE`. `/shared` at 77% (43T free).

**2 presets STALLED — and they are exactly the untruncated-GPQA experiment.** `_mn4096` = the
raised `max_new_tokens` runs, i.e. the "answers not cut off mid-generation" test:

| preset | progress | state |
|---|---|---|
| `gpqa_r1distill8b_mn4096` | **126/198** | preempted, checkpoint saved, **no job queued** |
| `trace_gpqa_r1qwen7b_mn4096` | **135/198** | preempted, checkpoint saved, **no job queued** |

Last line: `JOB 126281 ... CANCELLED ... DUE TO SIGNAL Terminated` → `PREEMPTED — checkpoint saved at
T=1.0 problem=135 candidate=0`. The requeue chain stopped while incomplete. Checkpoints exist, so
resubmitting resumes cheaply (`cluster/run_inference.py` is idempotent) — **just re-submit those two
via `/aircc-submit`; do NOT restart from scratch.**

**Then, the GPQA question Omri actually wants answered:** Step 191 ruled GPQA out of scope because
its features came back *uniformly at chance* (every feature 0.51–0.55, nothing to orient) — but that
was measured on **truncated** generations. Once the two `_mn4096` runs finish, `/aircc-fetch` all 11,
build the featcache, and **run `scripts/inscope_orientation_audit.py` on the new GPQA cells first**.
If features still sit at 0.51–0.55, truncation was not the cause and the out-of-scope call stands;
if they separate, re-open GPQA and re-run the in-scope evaluation with GPQA included.

## NEXT SESSION — PRIORITY: exhaustive subset sweep over the 30 views, then prune the pool

**STATUS (Step 195, 2026-07-22): sweep DONE, analysis DONE — and the expected negative turned
into the session's biggest positive.** The LOCO consensus over the 30-view sizes-3-5
enumerations is stable in 22/25 folds on the SAME new 5-view subset:

    {cusum_max, logprob_margin, min_energy, spectral_entropy, topk_tail_mass}

- LOCO-honest vs GOOD_5: **+1.59pp (19W/2L)** — reverses the Step-154 H16 verdict; the
  enlarged pool changed the answer.
- vs GOOD_6 on the same 24 cells: **0.7705 vs 0.7632 = +0.73pp, 17W/7L, p = 0.029**, sign
  label-free (anchor_orient/epr, verified). Same corpus-level label character as GOOD_6's own
  derivation but MORE disciplined (LOCO).
- Coverage caveat: runs on **24/25** cells (`inside_coqa_llama7b` lacks the energy/logprob
  views — Z_n backfill gap). GOOD_6 covers 25/25.
- **Pruning: definitively negative** — LOCO drop list EMPTY in all 25 folds; no view is
  "never in any cell's top-100". Pool stays at 30.
- **Stop rule: EXTEND to sizes 3-6 is justified** (+1.59pp >> +0.2pp; would also enumerate
  GOOD_6-sized subsets directly). ~3-4 days CPU — awaiting Omri's go-ahead, not auto-launched.
- NEXT: name the subset, add it to `REFERENCE_SUBSETS`/reference_macros, re-run the report
  chain so it appears as a candidate headline next to GOOD_6; decide the size-6 extension.
- **DONE (Step 195): `ref.LOCO_5` is now registered** in `REFERENCE_SUBSETS`/reference_macros
  and appears on the scoreboard (currently the overall leader, 0.7705 macro / 24 cells).

**UPDATE (Step 196, 2026-07-23): the sizes-3-6 extension above was NOT launched.** Omri redirected
to a more direct test instead: `scripts/pipeline_lovo.py`, exhaustive pipeline-level leave-one-
view-out, LOCO-honest (drop-set derived on 24 cells, applied to the held-out 25th) — this answers
"is any view redundant" without the multi-day sizes-3-6 enumeration. Collect phase done (25/25
cells, 0 errors). **The `--analyze` threshold sweep (0.0/0.1/0.2/0.5pp) was STILL RUNNING at
session end** (`results/advisor_inscope/pipeline_lovo_loco.csv`, appends incrementally — check
that file's row count / mtime rather than re-launching). Threshold 0.0pp finished: mean honest
delta **-0.22pp, 11W/12L/2T over 25 cells** — close to a coin flip, so the naive candidate set is
NOT a clean drop-list; the stricter thresholds (0.1/0.2/0.5pp) are the ones that matter for a real
verdict and were not done at session end. **Next session: check `pipeline_lovo_loco.csv` for all
four thresholds present (100 rows = 4 x 25), read the mean-delta-per-threshold verdict, and only
then decide WS3b** (leading-pool full enumeration over a noise-pruned pool, built from whichever
threshold's drop-set actually survives LOCO validation).

(Original plan below, kept for the protocol details.)

**Omri's intention (2026-07-21):** re-run the search over all possible subsets of the 30 views, per
cell, on the new runs. Then use *which features actually appear in the best subsets* to prune the
pool — a view that is never chosen in any cell's optimal subset probably should not be in the pool
at all. **The intent is sound and the motivation is stronger than "tidying":** a smaller pool
directly reduces selection variance, and Step 193 measured selection variance as the binding
constraint (65% of apparent per-cell selection gain is winner's curse, rho(n, optimism) = -0.671).
Pruning attacks that mechanism.

Two things must be handled or the result will be wrong.

### A. Full enumeration is not possible — use bounded sizes

Measured rate from the Step-193 rebuild: **53.8 subsets/sec** at `--workers 8`.

| Sweep | subsets | per cell | 25 cells |
|---|---|---|---|
| H16 sizes 3-16 (today) | 65,399 | 0.3 h | 8.4 h |
| **30 views, sizes 3-30 (full)** | **1,073,741,358** | **231 days** | **15.8 YEARS** |
| 30 views, sizes 3-21 (code cap) | 1,065,084,421 | 229 days | 15.7 years |
| 30 views, sizes 3-8 | 8,656,471 | 44.7 h | 46.6 days |
| 30 views, sizes 3-7 | 2,803,546 | 14.5 h | 15.1 days |
| **30 views, sizes 3-6 (recommended)** | **767,746** | **4.0 h** | **4.1 days** |
| 30 views, sizes 3-5 (pilot) | 173,971 | 0.9 h | 22.5 h |

Also a hard code limit: `spectral_utils/subset_sweep.py` sets **`MAX_SUBSET_SIZE = 21`** (the group
assignment packs 3 bits/member into a uint64 = 63 bits), and `enumerate_masks` raises above it — so
sizes 22-30 cannot even be represented today.

**This does not block the goal.** The question "is this view ever worth picking" is answered by
*small* subsets: GOOD_5/GOOD_6 are size 5-6, and the deployable subsets are small. Recommended plan:
run **sizes 3-5 first (~1 day, all 25 cells)** to get the inclusion-frequency signal, then extend to
**3-6 (~4 days)** only if the size-5 answer looks marginal. Command shape:
`python scripts/run_subset_sweep.py --domains repgrid --cells <25> --min-size 3 --max-size 6 --workers 8 --yes`
(resumable and chunked; safe to kill). Confirm the sweep uses the **c46/30-view** pool, not H16 —
today's npz artifacts are H16-only, so these are NEW artifacts and must not overwrite them.

### B. The leakage trap — derive the prune list OUT of sample

Choosing best-subsets-per-cell **uses labels**. If we then prune the pool on that and evaluate our
label-free method on the same cells, **the pool itself becomes label-derived** and the results are no
longer label-free at the corpus level. This is exactly the asymmetry Step 193d documented for GOOD_6
(chosen once by macro AUROC across the grid) — but much heavier, because it would consume labels
from every cell.

**Fix, and the machinery already exists:** `results/subset_sweep/loco.csv` is leave-one-cell-out and
already carries an explicit `oracle_best_LABEL_PEEKING_CEILING` column, so the repo already
separates LOCO from label-peeking. Protocol: derive the prune list on 24 cells, evaluate the pruned
pool on the held-out 25th, rotate. **Report held-out numbers only**; the in-sample pruned number is
a ceiling, not a result.

### C. Prune on inclusion frequency, NOT on individual AUROC

A view can be near-random alone and still valuable in a subset — L-SML exploits correlation
structure and assigns negative weights. Step 193 evidence that this is not hypothetical: the
selector's picks average **7.4 anti-oriented views**, and the cells it **wins** on contain *more* of
them (8.7) than the cells it loses on (6.7). So an individual-AUROC filter would drop useful views.
Omri's proposed criterion (never appears in an optimal subset) is the right one precisely because it
captures subset-context value.

Concretely, per view: (i) how many of the 25 cells include it in that cell's top-N subsets,
(ii) mean rank/percentile of the best subset containing it, (iii) best AUROC achievable *without* it
(the leave-one-view-out ceiling — the cleanest "is it ever needed" test). Drop candidates are views
that are absent from every cell's top-N **and** whose removal costs ~0 on the leave-one-view-out
ceiling.

**Decision gate:** the pruned pool is adopted only if, on **held-out** cells, it matches or beats
the current 30-view pool for the label-free selector. If it only helps in-sample, it is winner's
curse again and must be reported as such.

### D. What the EXISTING enumerations already tell us (done 2026-07-21, no new compute)

NEW `scripts/feature_inclusion_audit.py` answers the pruning question for the **H16 pool** from the
Step-153 npz enumerations we already have — **19 of 25 in-scope cells** (the 6 cluster cells were
never swept). GPQA/RAG npz on disk are *not* touched: the script iterates `INSCOPE` only.

**Result: no H16 view is a clean drop.** All 16 appear in some cell's top-100 subsets, and every one
has a non-zero leave-one-view-out cost on at least one cell. Ranked by how often they appear in a
cell's 100 best subsets:

| view | in top-100 | absent on | in that cell's best subset | LOVO max |
|---|---|---|---|---|
| `epr` | **92.0%** | 0 cells | **18/19** | 3.65 pp |
| `sw_var_peak` | 74.1% | 1 | 12/19 | 1.50 |
| `cusum_max` | 58.8% | 0 | 8/19 | 3.54 |
| `spectral_entropy` | 56.0% | 0 | 11/19 | 7.78 |
| `trace_length` | 55.6% | 2 | 10/19 | 1.21 |
| ... | | | | |
| `hurst_exponent` | 14.3% | 4 | 3/19 | 0.34 |
| `spectral_centroid` | 13.1% | 4 | 1/19 | 0.26 |
| **`low_band_power`** | **10.8%** | 3 | **1/19** | **0.19** |

**Notable tension: `low_band_power` is a GOOD_6 member and is the LEAST-picked view in H16** —
it appears in only 10.8% of top subsets, is in the single best subset on 1 of 19 cells, and banning
it outright costs at most 0.19 pp. It is redundant on 18 of 19 cells. Worth re-examining whether
GOOD_6 should carry it. (Consistent with Step 193d: the selector drops `low_band_power` on 7/25
cells.)

### E. Pool-size experiment — pruning does NOT close the gap (done 2026-07-21)

NEW `scripts/pool_size_experiment.py` answers Omri's "would dropping to ~20 views raise
performance?" directly: nested pools ranked by **informativeness** (`|AUROC - 0.5|`, the criterion
L-SML actually cares about, since it flips signs with negative weights), same selector, 25 cells.

| pool | selector (a2.dufs) | vs 30 | GOOD_6 | mean views picked |
|---|---|---|---|---|
| 30 | 0.7507 | — | 0.7594 | 19.0 |
| **24** | **0.7518** | **+0.11 pp** | 0.7594 | 17.6 |
| **20** | **0.7516** | **+0.09 pp** | 0.7594 | 15.8 |
| 16 | 0.7516 | +0.08 pp | 0.7533 | 13.0 |
| 12 | 0.7447 | −0.61 pp | 0.7533 | 10.2 |

**Conclusion: pool size barely matters between 16 and 30 (all within 0.11 pp), and pruning does not
close the gap to GOOD_6 (0.7594) at any size.** Pruning is not the lever. Below ~16 views it starts
to cost real accuracy. GOOD_6 itself drops to 0.7533 at pool≤16 because `spectral_entropy` (rank 17)
and `low_band_power` (rank 20) fall out of the pool — those two are worth +0.6 pp together, which
also re-values `low_band_power` upward versus the H16-only audit below.

**Correction to an earlier reading**: the h16-vs-c46 bench gap (selector 0.6828 @16 vs 0.7502 @30,
+6.7 pp) looked like evidence that bigger pools are better. It is not a size effect — H16 is a
*badly chosen* 16 that excludes the strongest views (7 of the top 10 by informativeness are the NEW
logprob/energy views). A *well-chosen* 16 scores 0.7516, essentially tying 30. **Composition
matters, size does not.**

Only 2 of the 30 views are near chance: `pe_mean` (0.535 flipped) and `stft_spectral_entropy`
(0.579). A defensible prune is 30 -> 28, worth ~0.1 pp.

*Caveat*: the ranking uses labelled AUROC aggregated over all 25 cells, so these are IN-SAMPLE
upper bounds — same corpus-level label asymmetry as GOOD_6. A real claim needs the LOCO protocol
in **B** above. Since the in-sample upper bound is only +0.11 pp, the honest LOCO number is very
likely ~0, which should be weighed before spending 4 days of sweep compute.

**Caveat**: the H16 audit below covers the 16 H16 views only. The 14 energy/logprob views (`*_spilled`,
`*_energy`, `mean_top1_logprob`, `logprob_margin`, `mean_logprob_entropy`, `varentropy`,
`renyi_entropy_2`, `topk_tail_mass`) have never been exhaustively enumerated — that is exactly what
the bounded 30-view sweep above is for.

## Omri's questions from 2026-07-22 (Step 193g) — ALL THREE CLOSED at Step 194

1. **Pseudo-label anchor → BUILT and benched** (`a6_pseudolabel_gates`; verdicts in the Step-194
   header above and HISTORY Step 194). 2. **Charts → DONE** (dumbbell / macro bars / pool-size
   line on the advisor_inscope pages, item4 idiom, guardrail clean). 3. **Cross-check → DONE**
   (`reconciliation.csv`; every DELTA is a known Step-193b correction). Original briefs kept
   below for context.

### (original briefs, superseded)

1. **Pseudo-label anchor for the SELECTOR (highest-value idea on the table).** The anchor sweep
   showed `epr` already resolves the sign correctly on 25/25 cells, so a better anchor cannot help
   *orientation*. But Omri's deeper proposal — fuse K strong views with their own L-SML to form the
   anchor — becomes powerful if that fused score is used as a **pseudo-label to supervise the
   gates**, not just to fix the sign. Step 193d measured the actual defect:
   **rho(gate value, view's own AUROC) = +0.149** — the Laplacian-smoothness objective is nearly
   orthogonal to separability. Replacing/augmenting it with "agreement with the anchor-fused
   pseudo-label" attacks exactly that. **This is the first idea this session that targets the
   measured mechanism.** Build as `a6_pseudolabel_gates`; guard against the circularity (the
   pseudo-label comes from views that may themselves be selected) by holding the anchor views out
   of the selectable pool.
2. **Charts.** `results/action_items/item4_benchmarking.html` is the visual standard Omri wants.
   `scripts/report_figs.py` (`_svg`, `_bar_panel`, `_dumbbell_panel`, `_lin`, `FIG_CSS`) is already
   imported by the new report but **never used** — the advisor_inscope pages are tables-only. Add
   dumbbell (ours vs competitor per cell), bar panels (macro by method), and a pool-size line.
3. **Cross-check against the action_items numbers.** The old `results/action_items/*` tables and
   `published_baselines.csv` were verified in earlier steps; the Step-193 `competitors_verified.csv`
   was re-verified from scratch. Diff the two so a number is never re-litigated a third time, and
   record the reconciliation so future sessions start from it.
4. ~~**`topk_tail_mass` and `renyi_entropy_2` have never been offered to any fixed subset.**~~
   **DONE and NEGATIVE in Step 206 (`exp09`).** Six variants pre-registered together; all six land
   below GOOD_6. Best `ref.GOOD_6+topk` 0.7587 (−0.07pp, p = 0.426), worst `ref.ENTROPY_6` 0.7462
   (−1.32pp). The "cheap, possibly-large win" reading was wrong: high individual informativeness
   does not imply additive value — `ref.LOCO_5` already contains `topk_tail_mass`, so the view is
   strong but the information is covered.

## NEXT SESSION — carry-over analysis items (Step 193c)

1. **Lead with the selector, not the fixed subset, everywhere else too.** `competitor_grid.html`
   now leads with the trace-based selector (`scripts/selector_vs_competitor.py` →
   `selector_vs_competitor.csv`), but `index.html`, `inscope_evaluation.html` and the HISTORY/
   PROGRESS headlines still quote GOOD_6 as *the* result. The contribution is the pipeline
   (label-free selection → L-SML), so the selector's number is the headline and GOOD_6 is the
   reference. Re-word the remaining pages.
2. **Finish the competitor provenance.** Still UNVERIFIED with no local PDF: INSIDE (2 cells),
   LOS-Net + its 11 baseline rows, Internal-States+RC, SE-ICLR'23 (Kuhn), TSV/TruthfulQA. Four of
   the five worst selector "losses" are against these, so they matter to the story.
3. **Split the competitor tally by supervision in every view.** Against the 13 *unsupervised*
   anchors the selector is +3.17pp; against all 18 it is −3.21pp. Only the first is like-for-like.
4. **Decide the `a6` question formally** — currently NOT built, on the evidence that no covariate
   explains the selector gaps (all ρ ≤ 0.33) and 65% of apparent selection gain is winner's curse.
5. **Regenerate `oracle_repgrid.csv`'s CONT column** — it still reads `repgrid_cont.pkl`
   (GOOD_5/STABLE_H9/ALL_H16 from `scores_lsml_upcr.csv`), so its printed macro headroom
   (+9.6/+11.3/+13.5pp over 51 all-domain cells) is NOT the in-scope number. The in-scope figure is
   LR@30 0.7810 vs GOOD_6 0.7594 = **+2.2pp**, from `advisor_inscope/lr_oracle_audit.csv`.

---

**Previously (Step 193)**: **In-scope competitor grid + method variants, gated behind a data-integrity audit that CORRECTS the Step-192 headline.** Root cause of the long-standing artifact disagreement is now named: `internalstates_gsm8k_qwen25_7b` was **re-graded/re-extracted between 2026-07-14 and 2026-07-20** (sweep manifest `n_pos=153`, current cache **147**; fused covariance also moved, K 2→4). The two fusion paths were never in conflict — they reproduce each other exactly. Staleness travelled by **three separate carriers**, each fixed: (1) resume-safe bench rows never recomputed (139 rows, 16 files); (2) **11 further cells** whose learned selectors searched a c46 pool 4 views too small (26→30 / 25→29 / 23→27) — i.e. handicapped; (3) the **h16 enumeration NPZ** (255/285 h16 rows are `eval_mode=lookup`, so re-running faithfully re-read the stale npz) — rebuilt, 65,399 subsets. **`--self-check` samples only ~20 lookup-vs-live pairs and did NOT catch (3); an exhaustive 101-lookup audit did.** **Corrected numbers: GOOD_6 0.7594 (was 0.7587), GOOD_5 0.7519 (was 0.7489), delta +0.75pp (was +0.98pp), Wilcoxon p=0.00507, 18W/7L (was 19W/6L); best label-free selector a2.dufs 0.7502 is now BEHIND GOOD_5 by 0.17pp (was ahead by 0.06pp); on h16 the stale row had inverted the ranking — GOOD_5 0.7519 > top_macro_5 0.7489.** Qualitative conclusion unchanged and stronger: GOOD_6 leads, no label-free selector beats it. **Competitor audit**: 19 anchors in `scores_lsml_upcr.csv` carry **no citation at all**; 7 papers verified line-by-line against `papers/extracted/` (EPR, HCPD, ARS, Noise Injection, Semantic Energy, ALS, HARP) → **2 errors found**: HARP is **supervised** (BCE-trained detector), not unsupervised, and the stored 92.8 anchor is the **Qwen** row (our Llama cell is 92.9, cross-model). **12 anchors remain UNVERIFIED** (LapEigvals ×5 — also the G3 gate — INSIDE ×2, LOS-Net, Internal-States+RC, SE-ICLR'23, TSV). **LR oracle audited and CLEAN**: the `max(p,1−p)` per-fold floor flips only 1/500 folds (+0.07pp on one set), grouped folds correct, no GOOD_6 member dropped; sole caveat is untuned `C=1.0` (spread up to 6.2pp). LR@30 = **0.7810**, honest supervised headroom **+2.2pp**. **Anchor sweep**: `epr` is AT the sign-resolution ceiling (25/25 correct); 4 other views tie exactly, but 4 of 9 candidates get signs wrong (`rpdi` −9pp macro), so the anchor is a real choice with a wide safe band. **Gate-saturation diagnosis REFUTED**: `a2.select` saturates 12/25, `a2.dufs` 0/25 (saturation is group-granularity, not gates) — but ρ(frac_selected, gap) = **−0.028**, and the replacement hypotheses (anti-orientation, imbalance, stability) are all ρ ≤ 0.33. **No covariate explains the selector gaps; `a6_gated_laplacian` was deliberately NOT built.** **Why no better subset exists**: split-half shows the greedy search beats GOOD_5 in-sample on **25/25** cells (+4.95pp) but held-out on only **19/25** (+1.74pp) — **65% of the apparent gain is winner's curse**, and **ρ(n, optimism) = −0.671** is the only explanatory covariate. Deliverable: **NEW `results/advisor_inscope/` (9 HTML pages)**. Gates: smoke 19/19, self-check PASS (max|diff| 2.94e-08), npz lookup audit 0 stale, plain-variant staleness 0/3377.

---

**Previously (Step 192 — its headline numbers are SUPERSEDED by the corrected ones above)**: **Complete in-scope (QA + math) evaluation of the leading pipeline over the FULL 30-view feature pool on the new cluster data — the in-scope evaluation Step 191 flagged as the next priority. Headline: on the 25 in-scope cells (10 QA + 15 math; RAG/GPQA excluded per Jul-20), the leading detector over all available features is the fixed `GOOD_6` subset (macro AUROC 0.7587, QA 0.727, math 0.780), +0.98pp over GOOD_5 (Wilcoxon p=0.0025, 19W/6L) — and NO label-free learned selector beats it (best a2.dufs/a2.select tie at +0.06pp).** Five phases, all CPU-local. **Integrity**: smoke 19/19, self-check GOOD_5 reproduced max|diff| 2.9e-08, all 25 cells present with 28–30 wide-pool views. **Step-187 sign-fix is NOT needed in-scope** (NEW `scripts/inscope_orientation_audit.py`): the label-free anchor `epr` is correctly oriented on ALL 25 cells (min oriented AUROC 0.560, best 0.931 — vs RAG where it was flipped 0.29–0.43); all GOOD_6 members carry the right fixed sign (4 core features 0/25 anti-oriented, low_band_power 1/25, spectral_entropy 4/25). The ~45% pool-wide anti-orientation is absorbed by L-SML's negative weights / removed by selection; the domain-polarity failure was RAG-specific and is closed for QA+math. **Full selector bench — all 8 families** at `--pool c46 --domains repgrid --cells <25>`, resume-safe (only the 6 new cluster cells computed; 19 grid cells reused), **0 fallbacks / 0 NaN** (Step-186 quality bar). GOOD_5 is pool-invariant (its 5 features all live in H16) → **the entire wide-pool value for the curated subset is the one `varentropy` view GOOD_6 adds, not automatic selection over 30 views** (reproduces Step-186/189 on the freshly-scoped pool). **Honest split-half ceiling** (both pools, R=10, seed 0, held-out half B, 25 cells): GOOD_5 0.7507 → greedy@H16 0.7546 → greedy@30v **0.7669** (+1.7pp over GOOD_5) — real but LABEL-GATED (uses labels on half the data), concentrated in **math (+2.5pp)**, thin on **QA (+0.6pp, optimism gap 0.041 = selection overfits QA)**; no label-free selector captures it, GOOD_6 recovers ~+1pp of it label-free. **Deliverables**: NEW `scripts/{inscope_orientation_audit,selector_compare_inscope,inscope_report}.py`; `results/selector_bench/{comparison_inscope.csv, inscope_feature_orientation{,_summary}.csv, splithalf_oracle_{c46,h16}_inscope{,_summary}.csv, inscope_evaluation.html}` (theme-aware CSV-driven results page — open locally, matches the Spectral_LSML_Report house style). **Canonical all-cell artifacts left at Step-191 state on purpose** — regenerating `comparison.csv`/`dashboard.html`/deep reports would re-mix out-of-scope RAG/GPQA into the headlines. **Next priority**: advisor read of the in-scope result (GOOD_6 is the leading detector; the label-free selection prize is small — is a curation-free selector still worth pursuing given the ceiling is math-only and label-gated?); optional seed-robustness on the +1.7pp greedy@30v ceiling; the 2 still-running GPQA `_mn4096` cluster dirs remain fetch-pending but are out of scope now. Full detail: HISTORY Step 192.

**Previous (Step 191)** — **Honest-ceiling premise-check on the enlarged RAG/GPQA pool: the feature POOL is NOT the binding constraint — GPQA has no signal at all, and RAG is bottlenecked by feature SIGN/composition (the still-open Step-187 fix), not by missing views.** Ran the plan `i-am-wondering-what-cuddly-wozniak.md` as a 3-command pipeline on the Step-190 cluster cells: `score_repgrid.py` (33 new cells via a safe `--cells` list — the script only skips `edis_`, so a bare run would have violated the Step-173 reject/partial hygiene rule) → `build_repgrid_featcache.py` → `selector_splithalf_oracle.py`. **The plan had a latent gap**: the oracle hardcoded the `h16` pool, so as-written it would have re-measured the OLD number — added a backward-compatible `--pool-mode` arg and ran BOTH `h16` (16 views) and the enlarged pool on the SAME new cells (separate output files, Step-189 H16 baseline untouched) for a clean controlled pool-effect. **Coverage correction**: the enlarged pool is **30 views, not 46** — the 16 anomaly-scorer views (iforest/AE/bocpd/hmm/kalman/…) are Stage-0 *derived* views (`build_derived_views.py`), never built for `repgrid`-domain cells (documented low-value follow-up; Step 186 found that family the dead end); `min_spilled` is a genuine zero-variance drop on 12 big-model cells; all hypothesis-relevant energy/logprob/varentropy views ARE present. **Load integrity 33/33 clean** (1 pkl/dir, problems×K==candidates==featcache n, valid 1.00, two code paths agree 51/51 at Δ=0.0000). **Controlled honest split-half (greedy_halfB, seed=0 R=10): RAG** GOOD_5 0.5214 → greedy@H16 0.5525 → greedy@30v **0.5887** (views +3.6pp, selection-within-H16 +3.1pp); **GPQA** 0.5368 → 0.5387 → **0.5336** (views −0.5pp — ceiling stays at chance). **Root cause (per-feature oriented AUROC)**: GPQA every feature dead (0.51–0.55, incl. epr + all new views) → nothing to orient. RAG has signal but GOOD_5 misfires 3 ways (fusion buries a good anchor on hotpotqa where epr alone hits 0.71–0.86; dead-anchor random sign on 2wiki; genuinely anti-oriented anchor on natural_questions) — the label-peeking unflip bound lifts GOOD_5's RAG mean 0.524→0.628, so **~10pp of the RAG deficit is pure global-sign error** (Step-187 domain-polarity reproduced on fresh cluster data). Signal lives in hotpotqa, not 2wiki/NQ. **Also fixed a latent gate bug** (`load_csv_refs` didn't disambiguate by anchor → spurious FAIL on the one chance cell `gpqa_r1distill8b`; now 51/51 pass). **Provisional**: `gpqa_r1distill8b` (175 problems) + `trace_gpqa_r1qwen7b` (188) are below the other GPQA cells' 198 — their mn2048 chains superseded by the still-running `_mn4096` re-runs; treat as provisional until those land. **Omri's Jul-20 scope calls (recorded here + in CLAUDE.md's "Thesis scope"/"Top subset" — the durable home)**: **(a) GOOD_5 is no longer the top candidate** — `GOOD_6` (GOOD_5 + varentropy) supersedes it (Step 182/184: +1.1pp macro, repairs GOOD_5's worst cells); treat GOOD_5 as the compatibility reference. **(b) RAG and GPQA are out of thesis scope** — this session's honest-ceiling result IS the evidence (GPQA uniformly dead; RAG signal confined to hotpotqa and sign-bound) — **the focus is QA + reasoning (math)**. **Next priority (reframed by (b))**: evaluate/validate the pipeline on the IN-SCOPE **QA + reasoning** cells — Omri is starting a fresh session to plan this. **The Step-187 sign-fix drops from "headline next win" to a targeted check**: its ~10pp value was measured on the now-out-of-scope RAG cells; reasoning cells (math500/gsm8k) are already correctly signed, so it only matters if IN-SCOPE QA cells show anti-orientation (cheap to check first, before assuming the pipeline needs it). Lower/optional: selector-bench re-scoping with Ofir/Bracha (Step 189, headroom now confirmed small); seed-robustness sweep on the +3.6pp views component (load-bearing conclusions are seed-independent); derived-views backfill for repgrid-domain cells (low value). Full detail: HISTORY Step 191.

**Previous (Step 190)** — **46-view-coverage regen wave landed: 33/35 preset dirs fetched from AIRCC and integrated into `cache/repgrid/`; gpqa T-mislabel found to extend beyond the regen batch.** Closed the fetch/integration half of `HANDOFF_regen_fetch.md` (submit→land→compare was already done in the prior session). **Cluster check (one-shot, no polling)**: the 2 still-running long-cap `_mn4096` chains (`gpqa_r1distill8b_mn4096` ~32%, `trace_gpqa_r1qwen7b_mn4096` ~36%) are healthy, not failed — standard SIGTERM-checkpoint-resume. **Fetched all 33 landed preset dirs** (5 gpqa, 20 RAG, 4 math500 T1.5 variants incl. one `_mn4096`, 3 trace, 1 internalstates — 17.6 GB) via plain `scp -r`, no reusable script needed (one-time bulk fetch). **Integrated mechanically**: every `preset_id` already doubles as its repgrid `cell_key` (`score_repgrid.py::discover_cells` auto-globs `cache/repgrid/*/manifest.json`, no registry to update), so 32/33 dirs were a straight move; the one collision, `internalstates_gsm8k_qwen25_7b` (regenerated because its Jul-11 copy failed every backfill attempt), got the old capture-less pkl archived to `archive_2026-07-11_capture_none/` and the new 46-view pkl swapped in, mirroring `fetch_backfill.py`'s backup-then-swap discipline. **4 post-integration spot-checks via `inspect_cell.py` all clean**: full 46-view key presence, 100% GOOD_5 extraction validity, accuracies matching the prior session's informal comparison (internalstates 0.294 and dsmath7b 0.190 exact matches). **gpqa T-mislabel propagation check (the session's other explicit task) found it's bigger than assumed**: the regen wave retires the T1.5-mislabeled-as-T1.0 sweep entries for 3 of 4 fingerprint-flagged small models (llama8b, mistral7b, r1distill8b — whose backfill roundtrips failed, forcing regen), but a 4th, `c_gpqa_Qwen2.5-7B-Instruct` (cell key `Qwen-7B_T1.0`), backfilled successfully at bf16 and was therefore never regenerated — it's still live under the wrong T1.0 tag in `results/latest.csv` / `method_comparison_table{1,2}.csv` / `results/archive.jsonl` (not in the 19-cell replication grid or 51-cell selector-bench pool, so current headlines are clean, but the unified rebuild needs to relabel it same as the other three). **Deferred, not started**: fetching the 2 still-running `_mn4096` dirs (handoff itself expects them to land "over the next day or two" — check `sacct` for COMPLETED before the next fetch attempt); Phase 5 (unified featcache rebuild) of `HANDOFF_full_coverage.md` — held off deliberately since Step 189's next-priority note (selector-bench re-scoping with Ofir/Bracha needed first) is still the standing priority order and wasn't superseded by anything this session. **Next priority**: get Omri's read on whether to (a) circle back and fetch the 2 remaining `_mn4096` dirs once cluster-complete, (b) start Phase 5 anyway despite Step 189's gate, or (c) take the selector-bench re-scoping conversation with advisors first, per Step 189. Full detail: HISTORY Step 190.

**Previous (Step 189)** — **Selector-bench punch list: split-half oracle reveals the +7.6pp selection prize was mostly winner's-curse, `inside_coqa_llama7b` autopsy, A4 merged, A5 mRMR salvage.** Four items, user-prioritized. **(1) Autopsy**: `a2.select`'s one catastrophic miss (`inside_coqa_llama7b`, −14pp vs GOOD_5) is GroupFS's DUFS gates saturating open (selects 100% of the 23-feature pool) on a severely imbalanced (pos_rate 0.147), 7-of-23-anti-oriented-features cell — GOOD_5 survives via clean isolation of one bad feature in 5; the 23-feature dump overwhelms L-SML's own K=7 clustering. Connects directly to the still-open Step-187 feature-sign fix. **(2) Split-half honest oracle (the headline finding)** — new `scripts/selector_splithalf_oracle.py`: bounded greedy search on held-out half A, refit+scored on half B, R=10 splits x 51 cells. **The 0.7472-macro exhaustive oracle collapses to 0.668 macro fully-honest (a statistical TIE with GOOD_5's own 0.6692 on the same splits)** — RAG's claimed +14.1pp prize is ~+1.6pp honestly, GPQA's claimed +10.2pp is ~+1.6pp honestly. This retroactively explains why every selector family (A1-A5) failed to beat GOOD_5 across Steps 186-189: there was never much of a real prize to capture, the 65k-subset exhaustive search overfits at n~100-500. **(3) A5 mRMR hybrid** (`spectral_utils/selectors/a5_mrmr.py`): relevance-minus-redundancy salvage of A4's "picks epr's clones" diagnosis — alpha=0 exactly reproduces A4's anchor-affinity numbers; alpha=0.7 genuinely helps on the 46-view c46 pool (0.7190, new top of that sub-family, beats epr.alone by +0.57pp) but not on H16 (epr.alone still wins there). Neither clears GOOD_5. **(4) A4 properly merged** (new-file-only, clean merge, no conflicts), its c46 arm re-benched restricted to repgrid-19 (was accidentally running all 51 domains), `eval_subset_flex`'s `K_override` clamp bug fixed, `epr.alone` promoted to a first-class leaderboard row on both pools. All CSVs/dashboard/research-note regenerated end-to-end; full smoke suite 19/19 green. **Next priority**: given finding (2), the selection direction's motivating premise needs re-scoping with Ofir/Bracha before further selector-design effort — the realistic honest headroom is ~1-2pp, not ~7-8pp; separately, the Step-187 feature-sign fix (13/30 anti-oriented features) remains the one still-open, concrete, likely-cheap win (would plausibly flip `inside_coqa_llama7b` and similar cells). Full detail: HISTORY Step 189.

**Previous (Step 188)** — **Chosen-sets report + Z_n backfill discovery + full-coverage handoff.** NEW `scripts/selector_chosen_sets_report.py` → `results/selector_bench/chosen_sets.html`: per-cell grid-search-best (oracle) vs GOOD_5 vs GOOD_6 vs best GOOD_5+view (Step-182 augmentation arm) vs the a2.select chosen subset, with feature chips, dumbbells, and an availability-aware chosen-frequency table. Non-probability views (trace_length + 4 Z_n energy views — the only 5 of the 46 not derivable from token probabilities) earn their keep: trace_length kept 33/45 (h16); energy views, where available, kept up to 6/7, and the best single-view GOOD_5 augmentation is an energy view on 5/7 such cells (+0.7..+2.8pp). **Z_n backfill discovery: all 12 analysis cells missing `token_logsumexp` have `gen_token_ids` saved → teacher-forced forward pass recovers Z_n (and ANY probability-derived view) exactly, labels/traces unchanged — no re-generation.** Coverage audit: 32 Colab-era cells (gpqa/gsm8k/math500/qa/rag/trace) are H16-only; Drive raw-pkl key audit pending. **NEXT PRIORITY: `HANDOFF_full_coverage.md` — plan one unified dataset (no Colab/cluster split, RAG + GPQA included) via the three recovery tiers (offline extract → teacher-forced backfill → re-generate last resort).** Also this session: anchor-alternatives + learned-feature-extraction research discussion (self-consistency contrastive direction; considerations logged in Step 188).

**Previous (Step 187)**: **A4 (antigravity) reviewed + four deep-report HTMLs + per-feature sign audit.** Reviewed the uncommitted `selector/a4-antigravity-unsupervised` worktree: smoke 13/13, zero fallbacks; overlap conclusions AGREE with ours via different implementations (K-swap ties GOOD_5 despite 2% K-agreement subset-vs-pool; greedy CSSP == A3 Concrete-AE h16 macro 0.6124=0.6124, Jaccard 0.52, r=0.85 — the reconstruction OBJECTIVE is the dead end, not the optimizer). Their novel anchor-affinity family is the best learned selector on H16 (a4.anchor_s4 0.6593) but collapses to its anchor: epr alone 0.6606 (25W/26L) — it selects epr’s clones; needs an mRMR-style diversity term. Protocol notes: their "c46" ran all 51 cells (the harness default — no `--domains repgrid`), so only their repgrid-19 rows are comparable (a4.good5+K_kn 0.7325 ≈ GOOD_5 0.7328); their comparison also embeds stale pre-MCFS-fix classical CSVs. Reframing per Omri: dynamic label-free a2.select TYING grid-searched GOOD_5 IS the intended win — and one cell (inside_coqa_llama7b −14pp) hides a strict win: excl. it, a2.select 0.7427 beats GOOD_5 0.7355 and top_macro_5 0.7406 (GOOD_6 0.7482 still ahead). Per-cell-oracle metric: evaluated (gap_captured/pctile) — NOBODY captures it (best +4.8% top_macro_5; a2.select −50%); flagged that the 0.7472 oracle is winner’s-curse-inflated — split-half per-cell ceiling proposed before adopting it as the judge. NEW `scripts/selector_deep_report.py` → 4 pages under results/selector_bench/: methods_protocol.html (papers/algorithms/assumptions/validation per family), experiment_results.html (leaderboards, per-cell dots, domain/dataset/model/T slices), benchmark_vs_published.html (a2.select on the roster scoreboard next to GOOD_5/GOOD_6 — same wins/losses as the fixed subsets), feature_value_audit.html (single-feature AUROC × cell/dataset/overall). **Audit headline: ZERO flat-noise features in the 30-feature repgrid pool — but 13/30 are consistently ANTI-ORIENTED (mean AUC 0.27–0.45: whole spilled/energy family, trace_length, spectral_centroid, dominant_freq, hurst, high_band_power, cusum_shift_idx, and GOOD_5’s hl_ratio 0.378) — informative features carried with the wrong fixed offline sign on this grid (domain-dependent polarity: same features are correctly oriented on math500/gsm8k). L-SML absorbs it via negative weights; U-PCR keep-mask and any anchor-correlation prior do NOT. Candidate one-line offline sign fix — must be re-validated against canonical repgrid scores before adoption.** GroupFS’s stable 8-feature core == the audit’s top-8 KEEP features (convergent validation). Latent harness nit found: `eval_subset_flex` does not clamp K_override>m (never triggered; clamp before reuse). 4 artifacts published (methods ff98f910 / results 7855dbe3 / benchmark 2e733078 / audit 11e52c7c on claude.ai/code/artifact/...).
**Step 186** — **Feature-selection bench EXECUTED: six label-free selector families through one select→same-L-SML→AUROC harness, full leaderboard, no gatekeeping.** The Step-185 memo turned into a working bench (`spectral_utils/selector_bench.py` + `selectors/` registry + `scripts/{smoke_selectors,run_selector_bench,selector_admissibility,selector_compare,selector_viz}.py`): selectors see an `UnlabeledCell` (labels structurally unreachable), H16 selections score by exact npz lookup (percentile-within-size = exact random-floor CDF), everything else live via `eval_subset_flex` (upcr/groups-override/K-override). **Unit-test-first policy (Omri)**: every new building block passed a standalone known-answer synthetic test BEFORE integration — 17-test gate, all green; regression fixtures keep `fusion_utils` byte-identical (GOOD_5 lookup == sweep_summary on 51/51 at 2.9e-08). Families benched on BOTH pools (H16 51 cells; 46-view c46 on the 19 repgrid cells): A1 residual-guided (Eq-14 raw/relative + U-PCR k=1 objectives, exhaustive+greedy, structural-model router, AH/KN eigenvalue K-rules — new `rank_tests.py`), classical spectral FS (Laplacian Score / SPEC / MCFS), simple-stats floor (random/MAD/kurtosis/decorr), reference macros as first-class rows, **A2 GroupFS** (AAAI 2026 reimplementation, worktree branch selector/a2-groupfs, planted-groups ARI 1.0, group-granular DUFS-gate selection after the joint gate dynamic measurably saturates — deviation 8 in module docstring), **A3 Concrete-AE** (ICML 2019, branch selector/a3-cae, paper anneal T0=10→0.01 + minibatch + best-val restarts + swap polish, 5/5 planted factors). **HEADLINE (c46/repgrid-19 macro): GOOD_6 0.7440 > top_macro_5 0.7364 > GOOD_5 0.7328 ≥ GroupFS a2.select 0.7323 — the first learned label-free selector to TIE GOOD_5 (0 fallbacks, 19/19); every other learned family trails 1-6pp; on H16/51 all learned families land 0.56-0.63 vs GOOD_5 0.671; the RAG/GPQA +7.6pp oracle prize stays uncaptured.** Pre-registered admissibility (ran FIRST): no label-free objective globally admissible (relative Eq-14 residual weakly admissible on repgrid/qa only, −0.109/−0.17 median Spearman); the lsml-vs-upcr residual router NOT-USEFUL in every domain; clustering swap (GroupFS groups → L-SML) ≈ tie ⇒ clustering isn't the bottleneck. Deliverables: `results/selector_bench/comparison.csv` + `dashboard.html` (published artifact: https://claude.ai/code/artifact/b559cea8-155d-4426-988f-f0b186431e43) + `docs/research_notes/selector_bench_results.md` + admissibility CSVs; Research_Directions Extension G updated. Known quirks documented in code/HISTORY: U-PCR auto-component absorption (k=1 for diagnosis), z-scoring makes independent-error covariance multiplicative, MCFS Lasso scale fix (fixed α zeroed all coefs at n≥1200), uint64 packing overflow >21-feature subsets (fixed). **Open branch NOT mine: selector/a4-antigravity-unsupervised (.worktrees/antigravity) — Omri's parallel track, untouched.** Next: advisor read of the leaderboard (is a curation-free GOOD_5-tie worth adopting?); D5-(ii) cross-cell signature router is the remaining unexplored design; A2/A3 worktrees removed, branches merged (f0d88ac→f662176), NOT pushed. **[Step-185 status below.]** Step 185 was — **Feature-subset selection research memo (Jul-2026 meeting action item): literature survey + assumptions audit, no implementation.** Research-only session answering Ofir/Bracha's action item — add a new algorithmic contribution, candidate = a principled, label-free, in-pipeline feature-subset selection step, replacing manual grid search over macros (GOOD_5/GOOD_6/top_macro_5/…). Conformal calibration stays parked. Deliverable: `docs/research_notes/feature_subset_selection_landscape.md` — problem statement (in-cell oracle beats fixed GOOD_5 by +7.6pp macro AUROC, 0.747 vs 0.671 over 51 cells, but LOCO transfer is flat at 0.664 vs 0.674 → selection must be in-cell and label-free, not a domain lookup table), a per-method assumptions audit (SML/L-SML/U-PCR/FUSE, verbatim-quoted from `papers/extracted/`) mined against prior validation-run evidence (Steps 134–136, 151, 153, 181, 184), 4 web-research threads with fetch-verified citations (A: Ofir Lindenbaum's FS line — identified the "trace of a sub-matrix" lead as the Gated-Laplacian objective `Tr[X̃ᵀL_X̃X̃]`, NeurIPS 2021 arXiv:2007.04728; B: Boaz Nadler portfolio — closed the Parisi-2014-PNAS lineage-root citation gap, arXiv:1303.3257, plus Kritchman-Nadler rank-estimation prior art for K-selection; C: tabular foundation-model frontier — TabPFN v2/CARTE/FT-Transformer concepts mostly supervised, Concrete Autoencoders flagged as the most directly adoptable unsupervised primitive; D: assumption diagnostics — FUSE's Ŝ statistic as the most reusable label-free violation objective, vanishing-tetrad tests, MetaOD as the closest per-instance-router precedent), and 5 candidate pipeline-step designs (D1 assumption-violation-minimizing subset search = lowest risk/top priority; D2 unsupervised gated FS pre-fusion step; D3 rank/eigengap-guided grouping; D4 FUSE-style transformation search; D5 the user's dual-use data-signature router with two explicit access-tier flavors). Answered a standing open question: **U-PCR and continuous L-SML are NOT the same algorithm** — same Nadler lineage, different structural covariance models (multiplicative rank-1 `v⊗v` vs. additive `ρᵢ+ρⱼ−g²`); `results/subset_sweep/method_grid.csv` shows which one fits better is itself domain-dependent (L-SML wins 90% of subsets on GSM8K, near coin-flip 53% on GPQA/RAG vs. U-PCR) — a candidate label-free per-cell diagnostic. Also found the ρ≥0.75 correlation filter inherited from the old supervised code path is empirically the WRONG diagnostic for continuous L-SML (Step 153: violating subsets score higher, not lower, AUROC) — any new violation-statistic design must be checked against this null result. `Research_Directions.md` updated ("Meeting Action Items — Jul 2026" + new "Extension G — Automatic Feature-Subset Selection" entry + priority-order update, now the top non-GPU priority). **Next session**: resolve the memo's open questions with Ofir/Bracha (does the Gated-Laplacian identification match his "sub-matrix trace" reference? appetite for the D5-(ii) cross-cell router given its different label-access tier?), then pilot D1 on the 19-cell replication grid — reuses existing L-SML/U-PCR residual code, lowest implementation risk. Explicit non-goals this session (user-scoped): no `spectral_utils` code changes, no diagnostic pilot, no conformal work, no advisor-facing HTML, no GPU/cluster jobs; the `var_y=0.25` hardcoded-constant audit item (memo §2.3) is flagged but not fixed. Full detail: HISTORY Step 185. **[Step-184 status below.]** Step 184 was — **Fixed the 94.4 MATH-500 mislabel everywhere it shipped, and added a GOOD_6 subset-size-ladder + anchor-orientation-robustness comparison, both scoped to the 19-cell replication grid only.** Two threads from PROGRESS's own priority list + fresh Omri feedback, fixed at the source and regenerated through the whole report chain (never hand-edited generated HTML). **Task A (the 94.4 mislabel, top of last session's next-priority list)**: confirmed via a fresh re-run of `phase1_math500_discrepancy.py` (byte-identical verdict) that 4 legacy `math500_res.pkl` cache keys hold Phase-4 T=1.5 data mislabeled as T=1.0. Fixed at the source (`method_comparison.py`'s `MISLABELED_KEYS`, `reasoning_benchmark.csv`, 10 subset-sweep CSVs + 4 manifest/npz renames, 3 hand-written HTML pages) and regenerated everything downstream. The audit went beyond the original single-number scope: caught a **second** independently-mislabeled headline (the R1-Distill-Llama-8B MATH-500 row rested on the same 4-cell list), a **non-citable-competitor "win"** in `action_items_report.py`'s scrutiny table (compared against a `citable=no` number and called it clean), and fixed two of its own self-introduced regressions along the way (an over-broad string replace, a stale hardcoded lookup dict) before final regen. Full grep sweep for unlabeled 94.4/0.944 across every report came back clean. **Task B (GOOD_6 + anchor robustness, broadened twice by Omri mid-session)**: what started as "add one GOOD_6 row" became "show the validated subset-size ladder generally, on one consistent cell-set" after Omri asked why the report doesn't reuse the old ~30-cell battery — answer: GOOD_6 (`GOOD_5 + varentropy`) structurally needs `top_k_logprobs`, an AIRCC-era-only capture field the old battery never had, so everything in this comparison (subsets AND the new anchor-orientation check) is scoped to the 19-cell replication grid only, never mixed with the old battery. Re-scored all 20 canonical cells with GOOD_6 + a second `anchor="cusum_max"` pass (score_subset already took an anchor param); independently cross-checked every GOOD_6 value via `build_repgrid_featcache.py`'s separate code path — **Δ=0.0000 on all 19 non-pilot cells**. Anchors **agree exactly on 18/19 non-pilot cells** (only the n=30 CoQA pilot disagrees, a small-N effect, not a systematic fragility) — extends but doesn't resolve the concurrent EDIS session's different-domain anchor-fragility finding (Step 183). Subset-size ladder (19-cell macro): consensus_4 64.0 → GOOD_5 73.3 → GOOD_6 74.4, significantly positive on 15/19 cells. **Verification itself caught two more real bugs before shipping**: `results/repgrid/subset_by_domain.csv` (feeding the pre-existing `closed_subset_html` table) turned out to be stale — only 11/20 current grid cells covered, missing 8 GSM8K cells + the CoQA pilot — so the new ladder table's domain lookup was rebuilt from each row's own `dataset` field instead of depending on that file; and the takeaway prose in two pages had been hand-typed from the plan's draft numbers rather than computed from the regenerated CSV, disagreeing with the real values once checked — fixed to interpolate computed stats, the same "never hand-type a headline number" discipline this session was itself enforcing for the 94.4 fix. All 9 `results/action_items/*.html` pages + `Advisors_Action_Items_Report.html` + `Replication_Grid_Report.html` regenerate guardrail-clean. **Deferred, flagged not dropped**: Task A2 (protocol prose + matching plots + worked Q→A examples across all ~14 advisor pages, per Omri's content-quality-bar feedback) — the new Task B content got inline protocol notes but the full page-by-page rewrite is separate-session-sized; a multi-anchor majority-vote panel (Omri's idea, noted in the plan file) wasn't built since the single-alt-anchor check found near-zero disagreement on this grid; a GOOD_6 row for the LR-oracle comparison needs a separate `logistic_oracle.py` rerun; `subset_by_domain.csv`'s staleness is a separate pre-existing-bug follow-up (still used as-is by `closed_subset_html`). Not yet committed (await Omri, per this project's commit policy) — a concurrent session's Step 183 (EDIS-grid) landed on origin mid-session with zero file overlap; re-fetched before every HISTORY/PROGRESS edit, no rebase was needed. Full detail: HISTORY Step 184. **[Step-183 status below.]** Step 183 was — **EDIS-grid degenerate-generation bug found and fixed; full-N data collection in progress, analysis deferred to a fresh session.** Fixed a real `compute_edis()` formula bug (erroneous `sqrt` around `1+Var(H)`, doesn't match Eq. 7 — regenerated `edis_scores.csv`, no values changed). Piloting the EDIS-grid replication (GSM8K/MATH500/AMC23/AIME24 × 3 temps, base Qwen2.5-Math-1.5B) surfaced a severe degenerate-generation failure — 27-47% of responses cap-pinned in true infinite loops. **First fix attempt was wrong**: `repetition_penalty`/`no_repeat_ngram_size` stopped the loops but corrupted the entropy trace and collapsed GSM8K accuracy to 7.9%. **Root cause found by checking the actual HF Hub configs**: Qwen2.5-Math-1.5B's `generation_config.json` only registers `<|endoftext|>` as EOS, not the chat template's `<|im_end|>` turn-end token — `generate()` never recognized a completed answer as done and kept sampling into out-of-distribution territory. Fixed with `chat_turn_end_token_ids()` + an explicit `eos_token_id` union in `generate_full()` (`spectral_utils/model_utils.py`) — a pure stopping-criterion fix, no effect on the sampling distribution. Re-piloted clean across all 4 datasets × 3 temps (N=30): GSM8K T=0.2 accuracy 32.9% (paper ≈36%), cap-pinning down from ~100% to a flat ~25%. **Preliminary offline scoring** (3-of-4 datasets, N=30, NOT paper-scale — do not cite): pooled EDIS AUROC 0.693 vs mean-H 0.545 (paper: 0.804/0.673) — qualitative EDIS-beats-mean-H finding replicates, absolute numbers low as expected at this N. **L-SML GOOD_5 flagged, not resolved**: known `anchor_orient` low-temperature fragility surfaced on this new model/domain — sub-random AUROC at T≤0.6, competitive-or-better at T=1.0 (rho(EDIS,L-SML) flips sign between low and high T). This is the first time L-SML has run on Qwen2.5-Math-1.5B/competition-math; the anchor mechanism is unvalidated here. **Full-N compute estimate came in ~10x over the original plan** (~230 GPU-hours across 4 datasets vs the plan's ~16-40h estimate) — flagged to Omri, who chose to proceed with AMC23 (n=40,k=32, full paper-exact set) + AIME24 (n=30,k=64, full paper-exact set) now, deferring GSM8K/MATH500 (n=500 would be ~65-78h each) sizing to later. **Both chained full-N runs in progress on AIRCC** (AMC23: jobs 112804-112808; AIME24: jobs 112810-112818) — multiple clean checkpoint/resume handoffs confirmed (each job runs ~7h45m, hits the TERM signal, checkpoints, exits 85, next job auto-starts via `--dependency=afterany`), no errors. **Explicitly deferred to a fresh session (Omri's direction — he wants a clean-slate method comparison since parts of L-SML may have changed)**: GSM8K/MATH500 full-N sizing decision, the `anchor_orient` low-T investigation, re-scoring once full-N data lands, and the final EDIS-vs-L-SML write-up. New file `scripts/score_edis_grid.py` (offline EDIS/mean-H/L-SML scorer, not yet run at full scale). `papers/digests/edis-paper.md` body rewritten with grounded Table 1/Table 4 numbers (was previously fabricated). Note: the code changes to `spectral_utils/model_utils.py`, `spectral_utils/feature_utils.py`, `cluster/presets.py`, `cluster/run_inference.py`, `scripts/smoke_preset.py` were swept into a concurrent session's commit (`f13e8bc`) under an unrelated message — nothing lost, just mis-labeled. Full detail: HISTORY Step 183. **[Step-182 status below.]** Step 182 — **Punch-list closure (8 items) + BOTH stale-data analyses re-run on the 19-cell replication grid; every conclusion confirmed or sharpened.** Executed `HANDOFF_punchlist_and_reruns.md` as Phase 1 (punch-list) strictly before Phase 2 (re-runs); structural items 9–10 deferred per Omri. **Phase 1 — A1: the MATH-500 "85.1 vs 94.4 discrepancy" is a TEMPERATURE MISLABEL, not a regression (closes the Step-152 P2 / Step-174 caveat)** — all four cells in `local_cache/math500_res.pkl` keyed `_T1.0` actually hold the Phase-4 **T=1.5** runs (they match that table exactly on accuracy AND per-feature AUROC), so 94.4 is a T=1.5/28%-acc operating point while the genuine T=1.0 anchor is Phase-5 fusion 90.0 at 69% acc — consistent with the fresh 1-pass 85.1. Nothing to reconcile. **A2**: Extension-E earliest-prefix edge REPLICATES on the clean cache (`lsml16` beats best DeepConf window by **+5.6pp [+0.9,+10.6]** at the earliest 10% of trace). **A3**: RAG SelfCheckGPT below-chance is a LABEL-PROTOCOL MISMATCH (grounded responses have *higher* contradiction scores on 4/4 datasets — SCGPT is anti-aligned with the citation-grounding label by construction) → annotate NOT-CITABLE, don't "fix". **B (#3/#7, unblocked by Omri's T≠1.0 Drive pull)**: low-T "poor detectability" is NOT an anchor/fusion artifact (swapping the epr anchor for cusum_max changes nothing) — it's **`spectral_entropy`'s temperature-dependent sign** (0.69 at T=0.3 → 0.14 hot); dropping it *raises* GOOD_4 to 0.76 at T=0.3. Length control: T≥1.5 is cap-pinned and partly length-tracking; T≤1.0 stays well above chance residualized. **C1–C3 cluster staging (Omri submits)**: presets `fusion_gsm8k_llama8b_k5` + `verbconf_gsm8k_llama8b`, and **the LapEigvals attention-Laplacian reducer is IMPLEMENTED** — `generate_full(capture_attention=)` no longer raises; reduces on-GPU to per-(layer,head) Laplacian eigenvalues (never stores the ~2 GB/sample raw), exploiting that a causal attention Laplacian is lower-triangular so its eigenvalues ARE its diagonal (verified exactly vs dense `eig(L)`); + offline PCA-512/balanced-LR probe scorer. `smoke_preset.py --all` = **30/30 PASS**. **Phase 2 — adapter first**: `scripts/build_repgrid_featcache.py` adapts the DATA once (per-candidate → legacy schema) so both consumers read one pkl; **validation gate PASSES on all 19 cells at Δ=0.0000** vs the canonical CSV. **LR oracle (19 never-seen cells, grouped CV via `problem_id` for the K=10 cells)**: GOOD_5 CONT 73.3 → LR 75.6 (**+2.4pp**), STABLE_H9 68.1 → 74.0 (+5.9), ALL_H16 67.2 → 75.6 (+8.4) — **on the compact curated subset the unsupervised fusion is already near the supervised ceiling**; the big headroom on wider sets is L-SML degrading on correlated/noisy features while LR still exploits them ⇒ the bottleneck is feature *selection*, which GOOD_5 already solves. **Subset sweep (19/19 cells, exhaustive 65,399 subsets on p=16)**: **GOOD_5 replicates as the best FIXED subset — honest LOCO selection does NOT beat it** (repgrid LOCO 72.19 vs GOOD_5 **73.28**, +1.09pp; consensus 72.70; ALL_H16 66.94; label-peeking oracle ceiling 75.98), matching the original battery's margin almost exactly (+0.96pp) ⇒ **the Step-154 conclusion holds on a completely different domain mix**; GOOD_5 is median 98.0th-percentile and top-decile on 15/19 cells. **`cusum_max_spilled` (the Step-181 gate pass) does NOT replicate** — worth −0.02pp on average, significantly NEGATIVE on 7 cells vs positive on 4 ⇒ retire it as cell-specific, not a new default view. **`varentropy` is a genuine new candidate**: +1.12pp macro over GOOD_5 (→ **74.40**), significantly positive on 9/19 and negative on 1, available on every cell (uses the already-saved top-50 logprobs), individually strong (74.4), and it **repairs GOOD_5's worst failure cell** (`internalstates` 62.8 → 68.5). **Perf fix in `fusion_utils.py`**: the sweep first stalled (10 augmentation views × 23 bases × 1000-iter paired bootstrap = ~12 min/small cell, hours/K=10 cell) → added **`_fast_auc`** (exact vectorized Mann–Whitney rank AUROC), verified **identical to `roc_auc_score` to 1e-16** incl. heavy ties, ~**20× faster** — no prior number changes, every future bootstrap CI in the project is cheaper. **Open/next**: Omri submits C1–C3 to the cluster (LapEigvals cell needs a re-run to capture attention); decide whether to promote GOOD_5+`varentropy` to the default (a "GOOD_6") — it's the one change the data supports. **⚠ NEXT SESSION'S PRIMARY TASK — advisor-report verification (`HANDOFF_report_verification.md`, paste-ready prompt at bottom)**: A1's correction is NOT yet propagated — the mislabeled **94.4 = T=1.5/acc-0.28** number still ships as the citable "reasoning headline" in `reasoning_benchmark.csv` + ~9 HTML reports (genuine T=1.0 is GOOD_5 85.1 [77.7,91.8], ~90 fused), several with a stale "Step-152 P2 unresolved" caveat that A1 resolves. The next agent must run a full number-provenance + label-integrity (temperature/accuracy/N/K/label-protocol) + published-baseline (vs `papers/extracted/`) audit of every advisor HTML, fix at the CSV/generator source and regenerate (never hand-edit generated HTML), and rewrite the pages to teach the experiments (prose + plots + worked Q→A examples with H(n) traces). **This session's Steps 181–182 work is COMMITTED** (at Omri's request); sweep `.npz` (82 MB) stay untracked per the Step-154 convention. Full detail: HISTORY Step 182. **[Step-181 status below.]** Step 181 was — **Phase-15 CPU follow-ups: 4 of the 8 Step-158 numbered follow-ups that were still open (after Step 174 closed #1) are now done, all pure CPU on the already-local Phase-15 caches.** New `scripts/phase15_followups.py` + `spectral_utils/repgrid_scoring.py::logprob_features_extended` (varentropy, Renyi-2, top-K tail mass — from the already-saved top-50 logprobs). **F1 K-sweep**: AUROC(K)=0.851/0.869/0.863/0.905/0.912 for K=1..5 — no early saturation, most lift arrives at K=4-5 (reproduces Step 158's K=1/K=5 numbers exactly). **F2 new features**: spilled-energy `cusum_max_spilled` (AUROC 0.909) fused as a 6th view **clears the Item-5-style gate** (+1.13pp over GOOD_5, CI excludes 0) — a second, smaller genuine complementary signal; `topk_tail_mass` (new logprob feature, AUROC 0.902) fusion gain +0.72pp is CI-significant but below the 1pp bar (near-miss). **F3 fairer diversity B′={0.6,1.0,1.5}**: 0.856-0.881, indistinguishable from a matched K=3 same-T arm (0.863; CI spans 0) — confirms Step 158's "diversity doesn't help" finding isn't an artifact of the degenerate T=2.0/0.3 passes. **F4 cross-temperature probing**: hot passes predict their OWN label much better than the COLD (T=1.0) label (e.g. T=1.5 own 0.878 vs cold 0.626; T=0.3 own 0.545 vs cold 0.388, anti-predictive) — mechanistic explanation for why mixing temperatures hurts fusion: each pass's signal is entangled with its own generation, not a stable per-question difficulty read. **Still blocked**: follow-ups #3 (anchor robustness across T) and #7 (length-controlled AUROC) need raw per-sample feature values at T≠1.0 that were never cached — need an extra Drive pull (`math500_qwen7b_T{0.3,0.6,1.5,2.0}_run0.pkl`). Results in `results/repgrid/phase15_followups.json`. Not committed (await Omri). Full detail: HISTORY Step 181. **[Step-180 status below.]** Step 180 was — **Gap-analysis planning pass executed: 2 paper digests corrected, HCPD/Automatic-Layer-Selection anchors wired into the report chain, `hcpd_coqa_llama8b` preset staged (not submitted).** Source: `HANDOFF_new_papers_benchmark_gaps.md`'s "needs new cluster runs" section, turned into a plan (`C:\Users\DELL\.claude\plans\you-are-planning-the-abundant-quiche.md`), approved by Omri, executed same session. **Re-digested Automatic Layer Selection and Quantum Tensor Network from their existing extractions** (no PDF re-read needed — both original digests had wrong "no numeric results / model unspecified" claims; ALS actually has full Table 2/3 grids on LLaMA-3.1-8B-Instruct + Mistral-7B-Instruct-v0.3, same-model overlap with our roster; Quantum Tensor Network confirmed **documented-REJECT for benchmarking** — zero roster overlap across all 8 of its models AND zero extractable numeric AUROC anywhere in the paper, verified by direct search). **Wired HCPD's own Table 2 numbers as the primary `published_Y` anchor** on `sciq_llama8b` (86.04) and `se_nq_open_llama8b` (90.38) via each preset's `published={...}` block in `cluster/presets.py` — the actual source of truth; `manifest.json` is frozen at cluster-submission time so the 2 already-cached manifests were hand-patched to match, then `score_repgrid.py --cells sciq_llama8b,se_nq_open_llama8b,spilled_triviaqa_llama8b` re-run locally (CPU-only, merge-on-write, 272 other rows untouched). **Result: `spilled_triviaqa_llama8b` (Llama-3.1-8B, GOOD_5 lsml 0.934) beats HCPD's own published 0.8625 by +7.1pp** — new, citable, CI-checkable win; SciQ/NQ-Open are honest losses (−12.2pp / −18.6pp), now precisely quantified instead of "close-ish". Full baseline tables (HCPD + Automatic-Layer-Selection + HARP) added to `results/repgrid/published_baselines.csv`. **Code fix beyond the plan's original file list**: `scripts/report_figs.py::fig_qa_extension_forest()` had a hardcoded `method == "Semantic Entropy"` exact-match (a one-off patch for the pre-existing LOS-Net cell) that silently prevented the new anchors from rendering — broadened to `method.startswith("Semantic Entropy")` (deliberately not "all pb rows for the cell", which would've flooded the HotpotQA row with LOS-Net's 11 other ablation baselines). Verified end-to-end: `python scripts/action_items_report.py` regenerates all 9 pages, guardrail scan clean, HCPD/ALS values spot-checked present in `item3_qa_evaluation.html`. **New preset `hcpd_coqa_llama8b`** (Llama-3.1-8B-Instruct, CoQA, K=1 greedy T=0.0 — verified against the HCPD extraction that the Table-2 eval uses plain greedy decoding, not the "5 beam search" mentioned in an unrelated RL-training-data section at a different line) closes HCPD's 4-dataset same-model grid; reuses the existing `coqa` loader, no new loader. `inside_coqa_llama7b` (llama-7b BASE) is left unchanged — it's the correct cell for the INSIDE-paper comparison it was built for. **Found and closed a real pre-existing gap**: `coqa` had zero grader fixture in `scripts/smoke_preset.py` (silently `[skip]`, untested — also affected `inside_coqa_llama7b`); added a 5-case `coqa_family` fixture hand-verified against the exact ROUGE-L/LCS formula in `spectral_utils/data_loaders.py`. `python scripts/smoke_preset.py --all` → **28/28 pass, 0 fail**. **Documented 3 skips + 1 deferral** in the handoff: RAGTruth (protocol mismatch — responses pre-generated by other models, not a loader problem), RAUQ summarization/MT (out of thesis scope, needs new loaders + new metric), PopQA/Grad-Detect model families (workshop paper, zero roster overlap); Qwen-3-8B arm of HCPD's grid deferred until the Llama arm proves advisor-worthy. **Not submitted** — next step is Omri running `/aircc-submit hcpd_coqa_llama8b` for the N=30 pilot (watch the accuracy band; `inside_coqa_llama7b` floored at acc=0.132 on the wrong model, the instruct model is expected to land in-band). Full detail: HISTORY Step 180.

**Prior**: Step 174 — **Item-5 verdict REVISED: the answer-agreement re-test ran in full (Omri dropped the 5 raw Phase-15 T=1.0 pass caches into local_cache/) and the fusion gate PASSES — L-SML 1-pass 85.1 [77.7,91.8] × answer-agreement SC K=5 82.1 [75.8,87.9], ρ +0.23 → fused 95.2 [91.8,98.0], +10.1pp over the best single arm (MATH-500/Qwen-Math-7B, N=200) — the strongest fusion number in the project, above Item-6's same-T averaging arm 91.2.** The Step-152 FAIL was the NLI-based LW-SE arm; sampling helps when spent on answer agreement. phase15_rescore.json now holds full+partial modes; item5 page has the result + a 4-arm CI forest (fig_item5_fusion); item6 has Q1 two-panel line chart + Q2 arm forest. Caveats: single cell; fresh-trace 1-pass 85.1 vs legacy-cache 94.4 (Step-152 P2, open). Follow-up: replicate fusion on a second cell (needs K=5 run on e.g. GSM8K/Llama-8B). Step 174 detail in HISTORY. **[Step-173 status below.]** Step 173 was — **Multi-dataset comparison figures shipped (EDIS/EPR-paper style — Omri: "not only GSM8K"): MATH-500 forest ×4 models (item4), TriviaQA forest ×4 models with the EPR paper's own same-model Table-1 anchors (SCGPT 79.0/EPR 74.6/HalluDetect 78.7 sup/WEPR 82.0 sup → published_baselines.csv), QA-extension forest ×7 datasets (item3 now badged COMPLETE), and a master per-domain table of every (dataset, model) cell incl. GPQA ×5 + RAG 4×4 with gate flags (per_domain_breakdown).** All CSV-driven via the generic `_generic_forest` in scripts/report_figs.py. Why the exact EDIS/EPR grids can't be reproduced: AIME24 floored ~2% (Phase 15), AMC23 never run; of EPR's 4 models only Mistral-24B was run (no Falcon/Phi-4/Ministral/ArGiMi). **CSV hygiene: a concurrent session's full score_repgrid run scored the REJECT/truncation-confounded dirs into scores_lsml_upcr.csv — stripped back 401→320 rows** (n30_pilot archive kept; rule: never run score_repgrid without --cells while *_reject/*_partial dirs exist under cache/repgrid/). Artifact same URL, now 9 figures + master table. Step 173 detail in HISTORY. **[Step-172 status below.]** Step 172 was — **BENCHMARKING DESK CLOSED: every cell fetched + dispositioned, queue empty.** A3 `ars_math500_qwen3_8b` (mn16384, acc 0.900) = **documented REJECT-leakage: 23/50 negatives cap-pinned at 16384, p95 trace = cap** — Qwen3 greedy reasoning is effectively unbounded on hard MATH-500 items; both Qwen3/ARS cells closed as REJECT (cache dirs renamed `*_reject`), R1-Distill pair = the citable ARS head-to-head. **Gate policy formalized two-tier + STRUCTURAL** (Omri's directive: use out-of-band cells, just mark them — desk-wide incl. QA): band violation = scored + CEILING/FLOOR flag (auto-derived from acc at report-build time, `gate_flag()`/`REJECT_REGISTRY` in scripts/report_figs.py; flags render in forest plot, delta chart, QA table, item4 policy box), excluded from headline win tally; label-validity failure (cap-pinned negatives / single-class) = REJECT, never scored. Case-control minority enrichment = legitimate but appendix-only (BENCHMARKING_COMPETITOR_GUIDE.md §5.2 has the full rule). **Final tally: 4 CI-clear wins (LapEigvals llama8b/phi35/nemo + Semantic Energy), 1 flagged win outside tally (mistral24b CEILING), 1 exact tie (mistral7b vs NI at ~10× less compute), 2 CI-overlap edges (r1distill vs sup ARS; qwen25 vs SelfCheckGPT), honest losses (llama3b, phi3mini, EPR≈tie, truthfulqa, hotpotqa, CoQA FLOOR 68.4 vs 80.4, SE-ICLR mismatch), 3 REJECTs (gemma2b, 2× Qwen3 leakage).** All reports regenerated guardrail-clean; artifact same URL. Optional follow-ups: p(True) pass on our cells, paired-bootstrap seqlp significance, `math_extended_casecontrol` appendix cell. Step 172 detail in HISTORY. **[Step-171 status below.]** Step 171 was — **Advisor report now has FIGURES: `scripts/report_figs.py` renders 5 CSV-driven inline-SVG charts into `results/action_items/` (item4 ×4: GSM8K 9-model forest plot, same-model Δ diverging bars, GSM8K/Llama-8B one-cell landscape, LOS-Net-table landscape; scrutiny ×1: GOOD_5-vs-seqlp scatter). Figures regenerate from the CSVs on every `action_items_report.py` build; a NEW GSM8K model needs a `GSM8K_SPEC` entry in report_figs.py or it silently drops (same convention as advisor_report's order list). Framing rule (Omri, saved to memory `feedback_published_roster_headlines`): advisor-facing headlines = the PUBLISHED citation roster; seq-logprob/ppl/nent = appendix audit.** LOS-Net (2503.14043) Table-1 baseline family verified from arXiv → new `results/repgrid/published_baselines.csv` (their HotpotQA/Mistral-7B-v0.2 column IS our matched cell: p(True)=54.0, Probas-mean 63.0, SE 67.66, probes 73.0, LOS-Net 72.92 — our GOOD_5 57.5 clears p(True) even on this out-of-regime loss cell). `score_ubaselines.py` extended with the same trivial-aggregation family on our traces (pmean/pmin/pmax from ΔE; lmean/lmin/lmax where token_logsumexp exists) — all 24 rows re-scored, merge-on-write intact. **Cluster: A2 `ars_gsm8k_qwen3_8b` fetched → documented REJECT** (acc 0.942 ceiling, 29 negatives, AND 15/29 negatives cap-pinned at 8192 = truncation leakage at full N; NOT scored into canonical CSVs — gemma2b policy; RB EigenScore/Qwen3 note updated). **C1 `inside_coqa_llama7b` full-N + judge-regrade fetched+scored: GOOD_5 L-SML 0.684** (n=4504, valid 0.90, judge labels) **vs INSIDE 0.804 = −12.0pp honest loss with floor caveat** (judge acc 0.132; pilot's 0.223 didn't hold; way above pilot's 0.533; pilot archived `inside_coqa_llama7b_n30_pilot`). **A3 `ars_math500_qwen3_8b` STILL RUNNING** (wall 3/4 at ~356/500, wall 4 queued) — **next session: fetch → inspect_cell (verify no 16384 cap-pinning) → score_repgrid → score_ubaselines → repgrid_report → action_items_report** (figures refresh automatically). headline_X_vs_Y.csv refreshed via repgrid_report; all 9 pages regenerated guardrail-clean. Review artifact (standalone advisor-review page with the same figures + methodology critique + glossary): https://claude.ai/code/artifact/86f71ec2-0473-4158-8e7e-4da7d916bc16 . Optional follow-ups: p(True) on our own cells (one extra pass, --regrade-style infra); paired-bootstrap GOOD_5-vs-seqlp significance. Full detail: HISTORY Step 171. **[Step-169 status below.]** Step 169 cont. was — **8 of 11 Wave-3 cells now SCORED in the canonical CSVs; only A2 (ars_gsm8k_qwen3, ETA ~01:30), A3 (ars_math500_qwen3 mn16384, ETA 2026-07-12 afternoon/evening) and C1 (inside_coqa + chained judge-regrade job 106293) still running — chains healthy, nothing to babysit.** Scored full-N this session (all in `reasoning_benchmark.csv` + `scores_lsml_upcr.csv` + `ubaseline_scores.csv`; advisor report regenerated guardrail-clean; Nemo/Mistral-24B added to the report order list, NQ-Open added to the QA head-to-head table): **B1 r1distill GSM8K greedy (acc 0.728): unsup GOOD_5 75.0 [70.4,79.7] BEATS the supervised ARS anchor 74.72 same-model** and beats every ARS unsup baseline by 13pp+ (caveat: seqlp ubaseline 77.2); **A4 internalstates T=0.8 (acc 0.306): U-PCR GOOD_5 69.1 [64.0,73.8] edges SelfCheckGPT 67.98**, supervised IS-probe 79.15 stays above; **B6 nemo (acc 0.829): GOOD_5 78.2 beats same-model LapEigvals unsup AttentionScore 63.0 by +15.2pp WIN** (+logprob 80.0; seqlp 80.3 caveat); **B7 mistral24b (acc 0.917 CEILING, 109 negatives): GOOD_5 80.1 vs AttentionScore 57.6 = +22.5pp WIN with ceiling caveat** (U-PCR 83.5; seqlp 85.7); **B4 NI mistral7b (acc 0.499 balanced): GOOD_5 78.5 = EXACT TIE with NI K=10 78.5 at 1-pass** (~10x less compute; seqlp 79.6 caveat); **B3 NI phi3mini (acc 0.710): 66.4 honest loss vs NI 72.51 (−6.1pp) but beats NI's own no-noise answer-entropy baseline 65.86**; **B2 llama3b (acc 0.445, unsloth mirror): 68.5 narrow loss vs AttentionScore 71.7 (−3.2pp; GOOD_5+logprob 71.1 = −0.6pp)**; **C2 se_nq_open on judge labels (acc 0.501 perfectly balanced, N=1000 K=10, valid 0.85): GOOD_5 71.8 / U-PCR 73.2 / +logprob 74.2, LNPE 74.4 slightly above — no clean published same-model anchor, rendered as such in the QA table**. Wave-3 tally so far: **3 WINs (r1distill-vs-SUPERVISED, nemo, mistral24b), 1 exact tie (mistral7b), 1 edge (internalstates vs SelfCheckGPT), 3 honest losses (llama3b narrow, phi3mini, truthfulqa), 1 floor-REJECT (gemma2b), 1 ceiling-caveated score (sciq)**. Stale CSV notes refreshed (queued→scored, gemma2b REJECT, A2/A3 job ids). **Next session**: fetch+score A2 (ceiling check — full-N acc ≥0.98 ⇒ greedy-ceiling-unreportable), A3 (verify no 16384 cap-pinning in inspect_cell), C1 + its regrade; then write the Step-170 HISTORY entry that closes the benchmarking desk. Commits ahead of origin — Omri pushes. **[Step-169 submit-session status below.]** Step 169 was — **Wave 3 EXECUTED: 30 jobs (103531–103544, 106275–106308); 10 cells running full-N chain-protected; truthfulqa + sciq SCORED; gemma2b floor-REJECT documented; NEXT SESSION = per-cell fetch→inspect→score.** All 10 pilots gated; 3 pilot failures fixed in-session: `unsloth/Llama-3.2-3B-Instruct` + `unsloth/gemma-2b-it` **mirror swaps** for gated-403s (meta-llama-3.2 and google gates reject our token; same pattern as huggyllama — Omri can request HF access to retire the deviation, or keep mirrors: byte-identical weights, documented in presets) and `sentencepiece` added to `cluster/requirements.txt` (Mistral-v0.3 slow-tokenizer crash). **Desk-clean (Omri's directive — every cell ends scored-in-CSV or documented-REJECT)**: judge-regrades flipped both paused QA cells into band (inside_coqa lexical 0.183→judge 0.223; se_nq_open 0.067→**0.663** — the lexical EM grader was the blocker, not the model) → both running full-N with a chained post-inference judge-regrade; **sciq scored** L-SML 0.738 / U-PCR 0.744 (double caveat: ceiling acc 0.877 + only 20% of MCQ traces ≥8 tok); **gemma2b acc 0.000 (0/30) = documented floor-REJECT** (the NI-anticipated reportable outcome — not scaled). **truthfulqa re-scored on REAL judge labels** (acc 0.116; judge-vs-lexical agreement 0.762): L-SML GOOD_5 0.660 / U-PCR 0.673 vs TSV semi-sup 84.2 (honest lose; seq-logprob ubaseline 0.693 edges GOOD_5 — caveat in CSV). **A3 ars_math500_qwen3_8b**: mn8192 pilot had 6/30 traces capped with **3 of 4 negatives capped** (leakage persists at 8192) but **NO repetition loops** (tail repeat-frac ≤0.08 — genuinely long reasoning) → preset now **max_new=16384**, pilot archived `ars_math500_qwen3_8b_mn8192_pilot` cluster+local (**NEVER resume** — cap-mixing confound), fresh full-N on 4 chained walls (106305–08; the long pole, ~24h). **A2 ars_gsm8k_qwen3_8b pilot acc 1.000** (0 negatives; worse ceiling than the 0.904 forecast) — scaled per handoff; if full-N acc ≥0.98 the cell documents as greedy-ceiling-unreportable. Other pilot accs (all scaled): internalstates-T0.8 **0.333** (in band; the T=1.0 collapse only partially recovers at T=0.8 — expected low-acc operating point), r1distill 0.633, llama3b 0.367, phi3mini 0.633, mistral7b 0.333, nemo 0.800, mistral24b 0.900 (ceiling caveat per the sciq precedent). **Follow-up session, per cell when its chain finishes**: `/aircc-fetch` → `inspect_cell.py` (on A3 verify no 16384 cap-pinning) → `score_repgrid.py --cells <id>` (background >100 MB) → `score_ubaselines.py` → `reasoning_benchmark.csv` (exact model strings from advisor_report order list) → `advisor_report.py`. Chains auto-resume; sacct FAILED(85) = "checkpointed, resume pending" — normal mid-chain. Full detail: HISTORY Step 169. **[Step-168 status below — its execution tail is DONE (Step 169).]** Step 168 was — **Wave-2 postmortem + Wave-3 staged via `HANDOFF_step168_cluster_wave3.md`.** Wave-2 Qwen3 jobs 101075/101076 hit the 8h wall, checkpointed, **exited 0 → Slurm recorded COMPLETED, no requeue → stalled partial** (440/500 GSM8K, 279/500 MATH-500). Deeper confound (HANDOFF_step166 §7–8, the Step-166 agent scored the partials to scratch): **13% / 45% of traces pinned at max_new=4096 → truncation-label leakage**; wave-2 Qwen3 numbers are PROVISIONAL and kept OUT of canonical CSVs (scratch: GSM8K GOOD_5 0.938 / U-PCR 0.962 vs ARS sup 0.904; MATH-500 0.795 / 0.834 vs 0.787; `GOOD_5+logprob` HURTS MATH-500/Qwen3). Decoding configs now **verified from primary sources** (Haiku subagent, verbatim quotes): **ARS §5.1 = greedy** ("By default, greedy decoding is used to generate model answers"); **Internal-States §3.1 = T=0.8 / max 300 tok**. **Fixes landed (committed)**: `run_inference.py` exits **85** on incomplete checkpoint (exit-0 was the no-requeue root cause); chain-submit resume pattern (`sbatch --dependency=afterany`) in the sbatch template header; presets `ars_{gsm8k,math500}_qwen3_8b` + `ars_gsm8k_r1distill8b` → **greedy / max_new=8192** (do NOT lower max_new — chain walls instead), `internalstates_gsm8k_qwen25_7b` → **T=0.8**; local partial caches renamed `*_mn4096_partial` (score_repgrid globs ALL raw_*.pkl — archive stale pkls before any re-fetch). smoke `--all` **23/23 PASS**. **NEXT SESSION executes the handoff**: pre-flight (VPN, HF_TOKEN REPLACE_ME check, sync_code) → Wave A re-runs (truthfulqa `--regrade`; ars_gsm8k_qwen3_8b ~2 chained walls; ars_math500_qwen3_8b ~3; internalstates T=0.8 — each with its archive-mv pre-step) → Wave B never-ran presets (`ars_gsm8k_r1distill8b`, `lapeigvals_gsm8k_llama3b`, `noise_gsm8k_{phi3mini,mistral7b,gemma2b}`, `lapeigvals_gsm8k_{nemo,mistral24b}`; gemma2b pilot may REJECT = reportable) → per-cell fetch→`inspect_cell`→`score_repgrid` (background >100 MB)→`score_ubaselines`→`reasoning_benchmark.csv`→`advisor_report.py`. Full detail: HISTORY Step 168. **[Step-167 status below; its cluster tail is SUPERSEDED by the handoff.]** Step 167 — **Survey-driven benchmarking pass: every anchor VERIFIED from primary arXiv sources; survey baselines scored on our traces; Noise-Injection sweep staged; judge-regrade mode shipped.** Source = `papers/State of the Art in LLM Hallucination Detection for Reasoning Tasks (as of July 2026)...md`. **All survey numbers verified** (Haiku web-subagent, verbatim-quote protocol): Noise Injection 2502.03799 **v4** Table 3 (Llama-3.2-3B 76.53→82.70 — v4 only, stale fetches miss it; Phi-3-mini 65.86→72.51; Mistral-7B-v0.3 75.85→78.50; Gemma-2B-it 51.36→57.11; N=1319 K=10 T=0.5 majority-vote question-level), ARS 2601.17467 **Table 2** unsup baselines (EigenScore Qwen3-8B: GSM8K 63.40 / MATH-500 81.38; R1-Distill GSM8K 52.98/61.98/58.48, MATH-500 75.89/43.60/40.96 → **our GOOD_5 84.4 beats every published unsup baseline on MATH-500/R1-Distill**), Internal-States 2510.11529 (SelfCheckGPT 67.98±1.28 = fair unsup Y on GSM8K/Qwen2.5-7B), TSV 84.2 semi-sup TruthfulQA/Llama-3.1-8B, HaloScope 78.64, Janiak 2508.08285 quotes. **New `scripts/score_ubaselines.py`** (perplexity/seq-logprob/naive-entropy per candidate + LN-PE/PE question-level for K≥2, one cheap pass, dual-label AUROCs) ran on 13 cells → `results/repgrid/ubaseline_scores.csv`: GSM8K/Llama-8B seq-logprob **80.4** vs GOOD_5 81.5 (close — honest caveat in CSV); **dual-label swing up to 35pp** (EPR cell naive-entropy 70.2 judge vs 35.6 lexical) = in-house Janiak confirmation. **Fresh phi35 cell scored** (job 101074, N=1319, acc 0.848): **GOOD_5 80.3 vs LapEigvals unsup AttentionScore 66.6 → +13.7pp WIN** (2nd sweep point). **Infra fix**: `score_repgrid.py` now merge-on-write (a concurrent session's phi35 run had silently overwritten the 11 Step-163 CSV cells; restored, GOOD_5 llama8b 0.8152 intact). **presets.py**: 7 presets anchor-enriched + 3 NI presets (`noise_gsm8k_{phi3mini,mistral7b,gemma2b}`, Gemma pilot-gated acc-floor risk); smoke 23/23. **`run_inference.py --regrade --judge <id>`** relabels an existing fetched run dir (no generation; `label_lexical` preserved; resumable) — for truthfulqa (ROUGE proxy → real labels). **CORRECTED per HANDOFF_step166.md: the internalstates acc-0.284 is NOT a grading artifact** — 99% of wrong answers have `\boxed{}` and are genuinely wrong (T=1.0 sampling collapse); a judge regrade will NOT unblock that cell — it needs a **temperature-matched re-run at the paper's near-greedy decoding T** (decision with Omri; the staged preset hard-codes T=1.0). `reasoning_benchmark.csv` 19→49 rows + `category` column (UGB/BB/WB/SUP); advisor report regenerated guardrail-clean (math-reasoning-gap box, category badges, judge-vs-lexical section). ProcessBench/MR-GSM8K **deferred** → Research_Directions.md Extension F. **Cluster tail (user-run, in order)**: (0) **verify HF_TOKEN is real in `$SHARED/code/cluster/submit_inference.sbatch`** before any gated cell — sync_code.sh tars the working tree and may have clobbered the live token with the REPLACE_ME template (HANDOFF_step166 §4c; gated = llama3b, nemo, mistral24b, NI mistral7b, NI gemma2b); (1) regrade job for **truthfulqa only** — `python cluster/run_inference.py --preset truthfulqa_llama8b --regrade --judge Qwen/Qwen2.5-7B-Instruct --out $SHARED/results/repgrid/truthfulqa_llama8b` → re-fetch → re-score (internalstates is NOT regrade-fixable — see correction above; T-matched re-run decision with Omri); (2) `ars_gsm8k_r1distill8b`; (3) `lapeigvals_gsm8k_llama3b` (triple-anchor: AttentionScore 71.7 / NI 82.70 / probe 87.0); (4) `noise_gsm8k_phi3mini` + `noise_gsm8k_mistral7b`; (5) `noise_gsm8k_gemma2b` (pilot may REJECT — that's the reportable outcome); (6) `lapeigvals_gsm8k_{nemo,mistral24b}`. **In flight, do NOT resubmit**: 101075 ars_gsm8k_qwen3_8b (~8h; **ceiling cell — pilot acc 0.967 → expect gate REJECT / wide CI at full N**, the MATH-500 cell is the usable ARS/Qwen3 point), 101076 ars_math500_qwen3_8b (~14h, requeues past 8h wall) — on completion `/aircc-fetch` → `inspect_cell.py` → `score_repgrid.py --cells <id>` (merge-safe now; background for the MATH-500 pkl) → CSV → report. Full detail: HISTORY Step 167. **[Step-166 status below.]** Step 166 — **Reasoning replication-grid presets staged: 7 new inference-only cells (our L-SML vs a competitor's PUBLISHED reasoning AUROC).** Reviewed `BENCHMARKING_COMPETITOR_GUIDE.md`, verified each method against its paper, and staged presets to fill the real reasoning gaps (same pattern as Steps 162–163: run inference on the paper's exact X/Y/N, score OUR L-SML offline, compare to their published Y — no competitor detector reproduced). **Paper verification corrects the guide**: EPR (2509.04492) is **QA-only** (not reasoning) → excluded; **LapEigvals (2502.17598)** evaluated **GSM8K only (N=1319)** on 5 models with published unsup AttentionScore + sup probe AUROC (we had only Llama-3.1-8B); INSIDE/LOS-Net are QA-domain; FG-PRM/FUSE report best-of-N accuracy not detection AUROC. **7 presets added** (`cluster/presets.py`, all `smoke_preset.py`-PASS, K=1, default capture): Tier 1 LapEigvals GSM8K sweep — `lapeigvals_gsm8k_{llama3b,phi35,nemo,mistral24b}` (fair Y = unsup AttentionScore 0.717/0.666/0.630/0.576; sup ceilings 0.870/0.885/0.890/0.925); Tier 2 — `ars_gsm8k_qwen3_8b` (vs 90.37), `ars_math500_qwen3_8b` (vs 78.66), `internalstates_gsm8k_qwen25_7b` (vs 79.15). Added GSM8K+MATH grader fixtures to `scripts/smoke_preset.py` so the CPU gate now validates the math graders incl. the `<think>`-then-`\boxed{}` case (R1/Qwen3) + `\frac` normalization; `--all` = 20/20 PASS. **Cluster async tail (user-run, VPN+queue)**: per cell `bash cluster/sync_code.sh` → `/aircc-submit <id>` N=30 pilot (acc in [0.20,0.85], trace not pinned; strong models may ceiling on GSM8K) → full N → `/aircc-fetch` → `scripts/score_repgrid.py --cells <id>` → append `results/reasoning_benchmark.csv` → `scripts/advisor_report.py`. Files: `cluster/presets.py`, `scripts/smoke_preset.py`. Not committed (await Omri). Full detail: HISTORY Step 166. **[Step-165 status below.]** Step 165 — **Reasoning-first advisor report rebuilt from CSV + missing reasoning comparisons filled.** Replaced the Gemini `results/Advisors_Action_Items_Report.html` with a generated, fact-checked one (`scripts/advisor_report.py`; every numeric cell sourced from a CSV; built-in terminology-guardrail scan passes). **Reasoning story now leads** (`results/reasoning_benchmark.csv`): MATH-500/Qwen-Math-7B **94.4** unsup 1-pass (= GOOD_5 subset-sweep exactly); **R1-Distill/MATH-500 GOOD_5 84.4 ≈ ARS supervised 86.38 on the SAME model** (already had the npz — no cluster run needed); GSM8K beats **LapEigvals-unsup 72.0** (A1: same-model anchor fixed in `cluster/presets.py`; 92.5 was the cross-model Mistral-24B sup number); **EDIS scored on our own trace** = 0.809 but redundant with L-SML (ρ=0.87) via new `scripts/score_edis.py` → `results/repgrid/edis_scores.csv` (all 11 cells). Verified-from-arXiv anchors: **ARS 2601.17467** (sup; GSM8K 90.37 / R1-Distill 74.72, MATH-500 78.66 / R1-Distill 86.38), **Internal-States 2510.11529** (sup; GSM8K 79.15). Report corrections: Semantic Energy = **Chen et al. 2508.14496** (not Farquhar), dropped "Minut et al." from EPR, EPR X labeled **U-PCR+logprob**, **fusion 0.768→0.758** (Step 152), NLI-truncation reframed suspected/unresolved (SE 87.7 / SC 87.2 flagged not-citable), selection-bias caveats (spilled n_pos=6, se_squad valid 0.29), closed-subset-per-domain table. Two ready-to-run ARS presets added (`ars_math500_r1distill8b`, `ars_gsm8k_r1distill8b`; smoke-passed) — GSM8K/R1-Distill cluster run is the async tail. Not committed (await Omri). **Next open items**: (1) reconcile the SE/SC NLI-truncation drop to make the old-cache reasoning baselines citable [Step-152 P1]; (2) run the GSM8K/R1-Distill ARS cell; (3) score MATH-500 EDIS on Colab (50 MB Drive pkl); (4) keep spilled/se_squad as selection-biased. Full detail: HISTORY Step 165. **[Step-164 status below.]** Step 164 — **Workflow token-economy tooling (from the Step-163 retro): two CPU-only scripts + three CLAUDE.md rules.** `scripts/smoke_preset.py <id>` = CPU-only pre-submit validator running the preset's REAL prompt/grader/judge helpers on fixtures (no model/dataset) — catches the pure-CPU pilot bugs (Qwen3 empty-`<think>`, OPT ramble, judge-parse ordering) offline in seconds; verified all 3 new presets PASS + a tampered fixture forces FAIL. `scripts/inspect_cell.py <pkl|dir>` = standard schema report (N/K, label dist + judge-vs-lexical agreement, trace lengths, base/energy/judge key presence, extractable features + valid-rate) — replaces ad-hoc `python -c`; verified on semenergy (K=10) + losnet (899 MB K=1). **3 CLAUDE.md rules**: (1) cluster polling only via `/aircc-status`/`cluster-ops`, never raw `ssh` in main context; (2) new preset MUST pass `smoke_preset.py` before submit (gate: local smoke → N=30 pilot → full N); (3) score/extract cells >100 MB or K≥10 in the background, inspect with `inspect_cell.py` first. **Deferred to a separate design pass (Omri's call): PDF text-cache + RAG** — extract `papers/*.pdf` → committable `papers/extracted/*.md` (PyMuPDF installed; `*.pdf` gitignored so `.md` persists) then latent-space search (fork: sklearn TF-IDF vs sentence-transformers; no embed libs installed yet). **Next**: design the PDF-cache + RAG pass; and the open analysis threads from Step 163 still stand (decide the 3 paused pilot cells; optional Gemma judge re-run if access lands). Full detail: HISTORY Step 164. **[Step-163 status below.]** Step 163 — **Replication grid SCORED (Phase 2, local CPU): OUR L-SML continuous + U-PCR vs the papers' PUBLISHED numbers, same-scenario head-to-head.** First re-ran 3 mismatched papers on their EXACT model (Phase 1, cluster inference only) so X sits next to Y with only the method differing. **Headline (our best X vs published Y, all `head_to_head=SAME-MODEL`)**: Semantic Energy/Qwen3-8B/TriviaQA **X=0.801 (GOOD_5 L-SML) vs Y=0.748 → +0.05, we beat it**; EPR/Mistral-24B/TriviaQA **X=0.736 (GOOD_5+logprob U-PCR) vs Y=0.746 → −0.01, tie**; SE-ICLR/OPT-30B/TriviaQA X=0.630 vs Y=0.83 → −0.20 lose (per-question K=10 semantic-sampling regime, different units — annotated); LOS-Net/Mistral-7B-v0.2/HotpotQA 0.583 vs 0.729 lose (supervised probe). So against the two strong single-answer baselines on TriviaQA our unsupervised spectral method **ties EPR and beats Semantic Energy**. **6 pilot bugs fixed** (multimodal Mistral3 load, Qwen3 empty-`<think>`, OPT-30B torch.load guard + base-model raw_prompt/few-shot, list-content chat template, judge swap Gemma/general-verifier→**Qwen2.5-7B-Instruct** — documented deviation). **4 whatis**: (A) fresh GSM8K/Llama-8B GOOD_5 L-SML 0.815 vs old 0.756 = +0.059; (B) MACRO 0.714 (n=9) vs old 0.636 (n=29); (C) energy/logprob views help QA (EPR +0.024, SQuAD +0.031) not reasoning, and HURT SemEnergy (−0.095); (D) more features ≠ better on short QA — AUROC peaks at 4-5 feats then declines everywhere. Deliverables: `results/Replication_Grid_Report.html` + `results/repgrid/*.csv` (committable); raw pkls gitignored under `cache/repgrid/`. Full detail: HISTORY Step 163.  **[Older status below.]**  Step 162 was: **Replication grid EXECUTED on AIRCC (inference-only). 5 VALID full-N cells produced + fetched + schema-validated locally; 3 cells paused out-of-band; two cluster infra bugs fixed.** Gate-first waves (N=30 pilots -> auto-scale in-band). VALID cells, each at cluster `$SHARED/results/repgrid/<preset>/` + local `cache/repgrid/<preset>/` (gitignored, ~1.5 GB): **losnet_hotpotqa**/Mistral-7B-v0.2 (acc 0.338, top-1000 logprobs; LOS-Net 72.92 anchor), **lapeigvals_gsm8k**/Llama-8B (acc 0.724, trace 168 - strongest cell), **spilled_triviaqa**/Llama-8B (acc 0.320, energy capture), **se_squad_v2**/Llama-8B K=10 (acc 0.606), **truthfulqa**/Llama-8B K=10 (acc 0.222, ROUGE-L proxy label - re-grade offline). PAUSED (N=30 pilots, out-of-band, re-gradeable offline since full_text saved): **inside_coqa**/llama-7b-base (acc 0.183 floor - base model rambles, ROUGE-L>0.3 grader misses; hidden_middle_last captured), **se_nq_open** (0.067 floor), **sciq** (0.900 ceiling). **Infra fixes** (first wave crashed): (1) concurrent `pip install` into shared `$HOME/.local` corrupted dill (`version.parse(None)`) -> node-local per-job `PYTHONUSERBASE` + `pip --user`; (2) pyxis `--container-name` collision on shared nodes -> dropped the static name. Both in `cluster/submit_inference.sbatch(.template)`, resynced, re-validated by 3 concurrent jobs. **Step-161 energy fix confirmed on cluster data**: raw-vs-warped logsumexp gap 22.8/29.1/24.2 nats; all 4 capture paths (base/raw-energy/wide-logprob/hidden) validated; 0 non-finite features. **Next**: offline scoring (local CPU) - L-SML continuous GOOD_5 + logprob/energy AUROCs on the 5 VALID cells vs published anchors; decide the 3 paused cells. Docs + sbatch fix not committed (await Omri). Full detail: HISTORY Step 162.

**Prior**: Step 161 — Reviewed the Step-160 replication-grid impl; Step-159 consolidation verified complete; two protocol-fidelity fixes (now shipped + validated by the Step-162 run): (1) energy capture uses RAW full-vocab logits (`generate_full(capture_logsumexp=True)` sets `output_logits=True`; `token_logsumexp`=true Z_n + `top_k_logprobs_raw`, not the temperature/top-k-masked `out.scores`); (2) CoQA conditions on dialogue history. Notes: INSIDE grades ROUGE-L>0.3 not >0.5 (re-gradeable, full_text saved); LapEigvals `published`=92.5 is the supervised Mistral-24B probe, not the cell's unsupervised Llama-8B (~72 anchor). Files: `spectral_utils/model_utils.py`, `spectral_utils/data_loaders.py`.

**Prior**: Step 160 — **Replication-grid plan reviewed after the EDIS/AIME24 test run; implementation landed (unit-tested locally, not yet run on cluster).** Decision (Omri): **keep each paper's exact terse protocol — no CoT/long-trace change** (apples-to-apples with published tables; CoT wouldn't rescue single-fact QA anyway). The plan's structure stands; the AIME24 demo (Qwen-1.5B, MAX_NEW=1024, **acc 1.7–2.9%** → AUROC uncomputable) only forced guardrails. Landed in `spectral_utils` + `cluster`: (1) `generate_full` default-OFF capture flags — `token_logsumexp` (energy papers), `hidden_middle_last` (INSIDE int(L/2)), `gen_top_p`/`gen_top_k` (INSIDE top-p=0.99/top-k=5); `capture_attention`/`capture_layer_fft` raise until the LapEigvals/HSAD reducers land. (2) 5 QA loaders + **paper-terse** prompts + graders (CoQA/SQuAD v2/NQ-Open/TruthfulQA/SciQ) + dependency-free `rouge_l`. (3) `cluster/presets.py` — per-paper preset table (source of truth for MAX_NEW per preset [reasoning 2048, short QA 256–512], N≥few-hundred, paper-T, capture flags) with the 4 high-impact cells + 4 QA-extension cells. (4) `run_inference.py` preset-driven with all QA datasets registered + a per-cell **accuracy-band gate** (VALID/REJECT; REJECTs the AIME24 floor and a 92% ceiling, VALIDs a healthy 55% cell) + `manifest.json` provenance. GOOD_5/spectral path unchanged (all flags off by default). **Next**: `bash cluster/sync_code.sh` → submit the 4 cells as N=30 gate pilots (read the GATE line before scaling to full N) → offline scoring scripts (with `anchor_orient`). Deferred: LapEigvals attention-Laplacian + HSAD layer-FFT on-GPU reducers.

**Prior**: Step 159 — **Branch consolidation: everything merged to `master`; work on `master` from now on.** Merged `experiment/bocpd-features` (Steps 151–156: pivot pilot, Phase-12-Corrected, subset sweep, AIRCC onboarding + verification, replication-grid plan) and `experiment/item6-temperature` (Steps 157–158, renumbered from branch-local 152/153). **Item 6 (temperature variation) is COMPLETE — both gates FAIL, and the negative result is the finding**: temperature diversity HURTS multi-pass fusion (paired B−A = −5.3pp [−10.3,−1.1]); more same-T=1.0 passes HELP (K=5 L-SML 0.912 vs single-pass 0.851, +6.1pp CI excl. 0) → the multi-pass lift is variance reduction at one good temperature, not diversity. Q1 AUROC-vs-T is an inverted-U confounded by accuracy collapse (80%→4%). Two method flags: `spectral_entropy` sign is temperature-dependent (flips at hot T); label-free L-SML underperforms best-single-feature at every T (weak `epr` anchor at low T) — anchor robustness is follow-up #3. **Data debt repaid**: Phase-15 T=1.0 run0 = canonical MATH-500/Qwen-7B raw-trace cache (N=200, full raw schema incl. top-50 logprobs) — unblocks streaming Extension E. 8 CPU follow-ups on the 9 cached runs (top: SC/SE baseline over the 5 same-T passes — also closes Item 5). Merge resolutions: master's float32 `extract_top_k_logprobs` kept (item6's float16 `topk_logprobs_from_scores` retired; Phase-15 Drive caches remain valid); `multipass_lsml_continuous` + Phase-15 notebook merged in. Branches deleted after merge: `pivot-alternatives` (contained in bocpd), `theorem-validation` + `lsml-variants` (already merged), `bocpd-features`, `item6-temperature`. ⚠ Omri: GitHub default branch is still `main` (2 stale initial commits) — switch to `master` in repo Settings → Branches.

**Prior**: Step 156 — **AIRCC verification ladder complete (Stages 2–4): Docker→Pyxis fix, AIME24 demo job 97309 done (2h52m), pkls fetched and validated (acc 2.9/2.5/1.7%), cluster skills updated.** Goal: run our **L-SML continuous GOOD_5** (Steps 134–136 baseline — NOT Step-100/107 numbers) on the **exact** (dataset, model, protocol) grids of the competitor papers, so every thesis number is directly comparable to a published table. Plan finalized (web-agent protocol research + Gemini review + inference-only scoping): **9 protocol cards** — SE-ICLR'23 (arXiv 2302.09664: OPT, CoQA dev 8K + TriviaQA train 8K, K=10 T=0.5, ROUGE-L>0.3, DeBERTa-MNLI), SE-Nature'23 (adapt, don't replicate — GPT-4 judge proprietary), INSIDE/EigenScore (2402.03744: K=10 T=0.5 top-p=0.99 top-k=5, middle-layer int(L/2) hidden states), LapEigvals (2502.17598: all-layer×head attention Laplacian eigvals → PCA-512 → LR probe), LOS-Net (2503.14043: top-K=1000 logprobs + ~1M-param Transformer probe; **G3 gate 72.92 ± 0.45 CONFIRMED correct**), HSAD (2509.13154: layer-axis FFT — 2 GB/sample raw ⇒ compute on-GPU, store per-layer scalars), EPR, Semantic Energy, Spilled Energy (energy papers **blocked on new `token_logsumexp` capture field** — raw logit = logprob + logsumexp; spec in plan). **Inference-only boundary**: cluster = generation + capture ONLY; all scoring local CPU. Data org: `local_cache/replication_grid/{preset_id}/` + `manifest.json` provenance (paper/model/dataset/split/N/K/T/capture flags/job id). 4 high-impact cells first: HotpotQA×Mistral-7B-**v0.2** (LOS-Net head-to-head), GSM8K×Llama-3.1-8B (LapEigvals 92.5 supervised), TriviaQA×Llama-3.1-8B, CoQA×LLaMA-7B-base (INSIDE 80.4 + SE-ICLR). Storage ~160 GB / 10 TB quota. **HF token live on cluster**: hardcoded in gitignored `cluster/submit_inference.sbatch` (untracked via `git rm --cached`; tracked `.template` has REPLACE_ME; synced + verified + chmod 600; note — `ssh aircc "<cmd>"` bypasses the login menu). Implementation follow-up (next sessions): `generate_full` extensions (token_logsumexp, hidden-state capture, at-capture attention/FFT reducers), 5 QA loaders (CoQA, SQuAD v2, NQ-Open, TruthfulQA, SciQ), cluster preset system, offline scoring scripts. **Item 3 dataset priority corrected: CoQA > SQuAD v2 > TruthfulQA** (published SE/SC baselines exist; AmbigQA/PopQA have none).

**Prior**: Step 154 — **Exhaustive L-SML subset sweep (branch `experiment/bocpd-features`): 1.66M subset fits over 32 cells.** Headline: **honest (LOCO) subset selection does NOT beat GOOD_5** (0.6295 vs 0.636 macro; in-cell best-of-65k ceiling 0.7205 = +8.5pp pure selection bias) — GOOD_5 is validated, feature selection stays a minor tweak. All-cell consensus best = {spectral_entropy, sw_var_peak, cusum_max, cusum_shift_idx} (+0.9pp, in-sample). **Every pivot signal HURTS as an added fusion view** (anomaly views −4.9..−7.9pp with 120–179 sig-negative bases; bocpd_ecp −4.8pp; bocpd_ecp_spilled = BOCPD-on-logprobs standalone 0.726 on gsm8k-trace but −1.0pp fused) → Step-151 17th-view thread CLOSED. **ρ≥0.75 subset filter refuted for continuous L-SML** (high-ρ subsets average HIGHER AUROC 0.600 vs 0.556 — clustering absorbs dependence). Sweep GOOD_5 matches table1 CONT 29/29 to ≤0.001; label-free anchor misorients 3/29 cells (the honest cost, now quantified). ⚠ Spilled signs inverted on gsm8k/Llama-8B trace cell (0.27–0.31 oriented) — cross-model sign instability, recheck at Step 132. New: `spectral_utils/subset_sweep.py`, `scripts/{build_derived_views,run_subset_sweep,subset_sweep_report}.py`, `results/Subset_Sweep_Report.html` + `results/subset_sweep/` CSVs/manifests; sweep is chunked+resumable (`--with-trace-cells --workers 7 --yes` resumes anywhere).

**Prior**: Step 152 — **Phase 12 Corrected finished on Colab (Items 4+5).** GSM8K/Llama-8B: **L-SML 1-pass 0.754 beats every multi-pass baseline** (best: SelfCheckGPT-official K=5 0.701; D-SE/LW-SE/SC K=10 all 0.61); third independent run at 75.4–76.0. MATH-500/Qwen-Math-7B: L-SML 0.230 = **global sign flip** (notebook lacks `anchor_orient`; flipped ≡ 0.770 — still far below the 94.4 old-cache reference, unresolved); SC K=10 wins the cell at 0.863. GPQA: all sampling baselines at chance, VC 0.428, L-SML 0.553 best. RAG×4: SelfCheckGPT **below chance** everywhere (official 0.24–0.44, worse than hard). Fresh-cache baselines collapse vs old Phase 12 (GSM8K SC 78.5→60.8, SE 77.4→61.4; GPQA SE 70.6→50.1; MATH SE 87.7→63.0 — NLI-truncation suspect; old table no longer citable until reconciled). **Item 5 verdict: fusion gate NOT passed** — ρ low everywhere but gains ≤+2.0pp; SE K=10 adds ≈nothing over 1-pass spectral, while spectral adds +14.5pp over LW-SE (GSM8K). Follow-ups = new Priority 1 (anchor_orient re-analysis, MATH discrepancy, RAG below-chance, SE-drop reconciliation). Results: notebook Cell 25 + Drive `cache/phase12_corrected/phase12_corrected_results.pkl`.

**Prior**: Step 151 — **Pivot-alternatives pilot (branch `experiment/pivot-alternatives`): both gates FAIL → no pivot.** Assessed the 5 Gemini pivot options (`docs/research_notes/thesis_pivot_options.md`) in `docs/research_notes/thesis_pivot_assessment.md` and piloted the survivors locally with pre-registered gates. Track A (6 anomaly scorers — Mahalanobis/GMM/KDE/IForest/AE/PRAE — as L-SML replacements over the same 16 features, 29-cell battery): ALL FAIL, best gmm2 0.553 vs L-SML continuous 0.651; even label-peeked oracle orientation tops at ~0.60; PRAE ≤ plain AE ≈ Mahalanobis. Track B (2-state HMM, BOCPD, AR/Kalman innovations on raw traces, gsm8k/Llama-8B): none beats DeepConf 0.735 / lsml5 0.754; innovations are entropy-level repackaging (ρ 0.93–0.97) → **KalmanNet NO-GO**; LOCA/IMM/hybrid dropped in assessment. Positive residue: consensus-direction fusion measurably beats direction-free anomaly scoring (strengthens the signal-first FUSE defense), and **bocpd_ecp is a level-orthogonal signal (ρ≈−0.07, 0.685 AUROC alone)** — candidate 17th view, null in 1-cell fusion, re-check free on the queued raw-trace re-inference. New: `spectral_utils/anomaly_utils.py`, `spectral_utils/temporal_models.py`, `paired_boot_delta_auc`, `iter_trace_records`, `scripts/pivot_track{A,B}.py`, `scripts/pivot_report.py`; results `results/pivot_track{A,B}.pkl` + 3 figs. Branch not merged — merge decision with advisors.

**Prior**: Step 148 — **Streaming pivot pilot (local CPU, 4 cells): G1 PASS / G2 FAIL.** Prefix-AUROC + DeepConf shoot-out + online monitor on GSM8K/Llama-8B + MATH-500/Qwen-1.5B (clean) and 2 truncated R1/GPQA cells. Early signal is real — 50% of the trace gives ≥95% of full-trace AUROC (G1 PASS); but fused L-SML does NOT beat the best DeepConf window by the pre-registered +2pp at ≥2 absolute budgets on ≥2 clean cells (G2 FAIL). Only significant spectral edge: earliest 10% of trace, BOTH clean cells (+9.8pp / +4.6pp paired bootstrap). Context: our unsupervised gsm8k 75.4 vs their SUPERVISED hidden-state probe 72.69 on the same model family (arXiv:2601.02170; different label protocol). New `spectral_utils/streaming_utils.py` (incl. `anchor_orient` — per-budget L-SML global-sign coin-flip fix), `scripts/streaming_pilot.py`, `scripts/streaming_pilot_report.py`; figures in `results/figs/`; advisor-ready explainer `results/Streaming_Pilot_Explainer.html`; Extension E (streaming) added to Research_Directions.md with updated priority order (raw-trace regeneration is the next Colab item). Data gaps found: MATH-500/Qwen-7B has NO raw-trace cache anywhere (Phase-12 K10 files are texts-only); no clean R1 cell (all traces capped at 1024). Verdict: streaming pivot NOT supported in current framing; the earliest-prefix edge is the thread to pull, and it needs a re-inference run saving raw traces first.

**Prior**: Step 147 — Bracha reply + Ofir FUSE concern. LR-oracle re-validated on a strict common-cell basis (the ~1pp macro artifact fixed): corrected gaps LR vs L-SML = +4.7 / +3.8 / +3.6pp for 5/9/16 features (LR 68.9/66.8/67.8 vs CONT 64.2/62.9/64.1; in-sample ceilings 70.5/73.7/79.3). New `scripts/oracle_report.py`, `lr_convergence.py`, `lr_weight_analysis.py`; convergence + weight-agreement figures; `logistic_oracle.png` bar chart corrected to common-cell. FUSE positioned (signal + task + dependence-handling differ). 4-point advisor reply drafted (not sent). All local — no model re-runs.


---

## TL;DR — where we are today

**Recommended method** (established by the Step-134 method comparison, 12 variants × 29 cells):
`lsml_continuous_pipeline(feats_dict, GOOD_FEATURES, FEATURE_SIGNS)` — **L-SML continuous** (previously called "CONT" — that term is retired).
- Macro AUROC **70.1%** vs the old binary PROD pipeline 65.2% (**+4.9pp**); **78.3%** on the reasoning regime {MATH-500, GSM8K, QA}.
- On reasoning it beats a simple average (+2.2pp) and even the per-cell oracle best-single-feature (+0.7pp).

**Old production method** (binary, Steps 100–131 — now superseded as the recommendation):
`binarize_classifiers(feats_dict, FEATURE_SIGNS)` → filter to `GOOD_FEATURES` → `lsml_fuse(...)` — the `np.sign()` binarization was the single biggest source of lost signal.

**Key conclusions from Step 134** (independently co-signed by Gemini, `LSML_IMPLEMENTATION_REPORT.md` §13–17):
- **Encoding is the dominant lever**, not features or signs. Continuous beats binary by +4.9pp macro / +7.2pp reasoning.
- **Feature selection is a minor tweak**: continuous L-SML on *all 16* features (`lsml16c`) = 69.2%, within 0.9pp of the selected 5-feature CONT. It helps on reasoning, hurts GPQA. (Answers Bracha Q1.)
- **FEATURE_SIGNS = one global orientation bit**, not a learned dictionary (all 5 GOOD_5 signs equal → a single global flip). Required for deployment orientation; adds zero separability. The paper's internal sign algorithm fails on our error-predicting features (~14% concordance).
- **Robustness (R4) hypothesis rejected**: grouping does *not* insulate against volatile features — avg5 is the most cross-domain-stable (8.9pp std), CONT the least (10.9pp). Fusion's justification is in-regime peak accuracy, not robustness.
- **Operating regime**: spectral L-SML is a reasoning-trace method. GPQA (forced-choice MCQ) and RAG (retrieval-grounded) lack the temporal structure; there a simple average is as good or better.
- **Deliverables**: `Bracha_Reply_Jun2026.md` (answers her 3 Jun-8 questions), updated `results/method_comparison_report.html` (§13–16: lsml16c, R4 robustness, reasoning-only, per-cluster AUC).

**Step 135 — grid completion + benchmarking + narrative report**:
- Full design grid done (5/9/16 × binary/continuous × flat/L-SML + avg). **Continuous beats binary in every cell.** L-SML clustering helps only with many features (5 feat: ties flat; 16 feat: +6.1). Flat-SML-continuous collapses 70→63 as features added; L-SML holds 68–70.
- **Benchmarking (model-matched, CONT, 1-pass)**: MATH 94.4 (win vs SE 87.7/SC 87.2), GSM8K 75.6 (competitive; beats LapEigvals-unsup 72.0), GPQA 52.3 (loss vs SE 70.6), RAG beats SelfCheckGPT 3/4.
- ⚠ **Do NOT reuse Step-117 "ours" numbers** (96.7/71.3/88.1 — leaked supervised). ⚠ **EDIS Phase-13 invalid** (7.7% acc = `\boxed{}` grading bug); fix before citing.
- New: **`results/Spectral_LSML_Report.html`** — story-driven advisor report (this is the one to attach, not method_comparison_report.html).

**Step 136 — cross-cluster weights + full correlation + report v2 (report sent to advisors)**:
- **Across-group fusion weight now stored** per cluster (`cross_weight` col in table2/JSON). Mechanism: it is the leading eigenvector of the clusters' off-diagonal covariance = each cluster's estimated reliability, **not** an average.
  - **K=2 → always 0.50/0.50** (structural — 2×2 zero-diag covariance). So a 2-cluster even split is NOT evidence of adaptive weighting.
  - **K≥3 → weights separate**; a weak isolated cluster gets ≈0 (e.g. pe_mean 0.02 on 16-feat MATH-500). A true average would give it 0.25.
- **pe_mean is domain-dependent — do NOT hard-delete it**: isolated + weight 0.02–0.05 where weak (MATH-500, both QA-CoT cells), but joins a useful `epr,pe_mean` cluster (67.7%, weight 0.24) on GSM8K. L-SML's weighting suppresses it adaptively, only where it should.
- **Full 16×16 dependence matrix** → `results/feature_correlation_16.csv` (new `scripts/feature_correlation_full.py`): band-power block ρ 0.77–0.88, median pair 0.25, pe_mean near-independent. This is the structure L-SML exploits / flat SML ignores.
- **No feature is both strong and stable**: strong features (epr/cusum_max/sw_var_peak) swing ~30pp across domains; stable features (pe_mean range 8.5) are weak everywhere.
- Report v2: removed exec summary; added terminology + aggregation note + 9-feature data + 3 graphs (dependence heatmap, stability scatter, per-domain ranking heatmap). Self-contained except Chart.js CDN.
- **Open**: fix EDIS grading + re-run; complete Phase 14 (GPQA/DeepSeek-R1-8B).

**Step 142 — U-PCR algorithm correction + re-run**:
- Fixed two bugs: (1) weight formula `w_k = (v1@rho/lam1)*v1` hardcoded to v1 even for n_components=2 — corrected to `Σ_c (vc@rho/lamc)*vc`; (2) no λ₂ auto-threshold — added `auto_components=True, lambda2_threshold=0.1`.
- Re-run (29 cells, 3 feat sets): U-PCR-auto gets +0.5pp over old U-PCR-1 on 16-feat (63.0% vs 62.5%), still below L-SML continuous (65.1%). On 5/9 feat the correction slightly hurts (−0.6pp, −0.8pp) because v₂ captures structured noise for low-correlation feature sets.
- λ₂/Trace = 9–34% across cells (28/29 exceed 10% threshold). The paper's 10% threshold is too permissive — 15–20% would be more appropriate for our curated feature sets.
- **v₂ as soft clustering**: (v₁[i], v₂[i]) are continuous cluster coordinates for each feature; U-PCR uses them directly instead of L-SML's hard group assignment. Same structural idea, different tradeoff.
- Updated `results/upcr_comparison.pkl` + new `results/upcr_comparison.png` (3-panel visualization).

**Step 141 — Deep literature review: FUSE, Deep L-SML, STDR, U-PCR**:
- **FUSE finding**: our closed-form eigenvector weights (`w@F`) underperform naive averaging in 7/10 FUSE benchmark settings (Figure 3). Fix: replace with pseudo-label logistic regression on MoM-estimated triplet posteriors `p̂(r_i)` — still fully unsupervised. **Highest-priority next experiment.**
- **RBM = L-SML equivalence** (Lemma 4.1, Shaham et al. 2016): our covariance+eigenvector step IS a single-hidden-node RBM trained by MoM. Stacked RBM (Deep L-SML) handles correlated features without exclusion; relevant for 16-feat expansion where band-power pairs (ρ 0.77–0.88) trigger heavy filtering.
- STDR (Fiedler vector tree recovery): not relevant at current feature counts.
- Step 140 numbers now explained: U-PCR ≈ L-SML on 5/9 features (low-corr regime matches assumption); L-SML wins on 16 because clustering handles band-power block violation.

**Steps 139–140 — U-PCR literature + implementation + empirical comparison**:
- U-PCR (`upcr_fuse`, `upcr_pipeline`) implemented in `spectral_utils/fusion_utils.py` (Tenzer et al. 2022).
- Comparison run across 29 cells, 5/9/16 feature sets. Results (macro AUROC):

| Feature set | L-SML continuous | U-PCR | Delta |
|-------------|-----------------|-------|-------|
| 5-feat | 65.3% | 65.7% | +0.4pp |
| 9-feat | 63.9% | 65.0% | +1.1pp |
| 16-feat | 65.1% | 62.5% | −2.5pp |

- **Conclusion**: U-PCR ≈ L-SML continuous on low-correlation feature sets (5, 9 feat — the assumption E[h_i h_j]=0 approximately holds). L-SML continuous wins on 16 features where correlated features (band-power block ρ 0.77–0.88) violate U-PCR's assumption; clustering handles this, plain eigenvector weighting doesn't.
- Provides the theoretical citation for Step 134: L-SML continuous ↔ U-PCR's ρ̂-proportional weighting. Cite Tenzer et al. (2022) instead of "workaround for Lemma 1".
- Advisor meeting Item 1 (lit search) ✅ complete.

**Prior session (Step 131)**:
- GSM8K cross-dataset verification: spilled energy transfers well (cusum_max_spilled = 0.725 best individual)
- Verbalized confidence: **null result on 1.5B**; adding VC hurts L-SML (−1.77pp)
- Structural finding: within_H/cross ratio = 0.04 (MATH-500) vs 0.99 (GSM8K) — H features are near-independent views on long traces but redundant on short traces
- All changes on branch `experiment/lsml-variants` (commit `f4bc5e8`)

---

## MEETING ACTION ITEMS — Jun 17, 2026 (Ofir, Bracha, Amir)

*Email thread: Omri → Ofir/Bracha/Amir, Jun 17 2026, confirmed by Ofir same day.*

These 6 items are the current priority order. They supersede the old Step 132 GPU-first priority (Step 132 is still pending but de-prioritized until these are underway).

| # | Action | Status |
|---|--------|--------|
| 1 | **L-SML literature search** — find Nadler post-2016 follow-up work extending or improving L-SML | ✅ Complete (Step 141) |
| 2 | **Logistic regression oracle** — supervised LR on 5/9/16 feature sets → upper bound on fusion AUROC (5-fold CV, no in-sample leakage) | ✅ Complete (Steps 142–143 corrected; Step 147 common-cell re-validation + convergence + weight-agreement experiments) |
| 3 | **Extend QA evaluation** — priority corrected (Step 155): **CoQA > SQuAD v2 > TruthfulQA** — these have published SE/SC baselines to compare against; AmbigQA/PopQA have none. Folded into the replication-grid plan (loaders + presets) | In progress — actively running (Steps 160–169); TruthfulQA + SciQ freshly scored, CoQA/NQ-Open mid-flight |
| 4 | **Benchmarking completion** — model-matched comparisons for MATH-500, GSM8K, QA vs SE/SC/SelfCheckGPT | In progress — actively running (Steps 160–169); 3 cells still in queue, Wave-3 scored with 3 wins + 1 tie + 1 edge |
| 5 | **Experiment 1 — Sampling fusion** — fuse SE (K=10) with single-pass spectral features; measure AUROC gain vs each alone | ✅ Complete — verdict REVISED (Step 174): with answer-agreement SC (no NLI) as the second view, gate **PASSES** — L-SML 1-pass 85.1 × SC K=5 82.1 (ρ +0.23) → fused **95.2 [91.8, 98.0]**, +10.1pp over best single arm (MATH-500/Qwen-Math-7B, N=200). The Step-152 FAIL was the NLI-based LW-SE arm |
| 6 | **Experiment 2 — Temperature variation** — run same model at T∈{0.3,0.6,1.0,1.5,2.0}; does higher T improve detectability? Ablate: T-diversity vs just more passes | ✅ Complete (Step 158) — temperature diversity hurts fusion (−5.3pp, CI excludes 0); same-T sampling helps (+6.1pp) |

See `Research_Directions.md` § "Meeting Action Items — Jun 17, 2026" for full experimental designs.

---

## Current pipeline constants

```python
GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']

FEATURE_SIGNS = {
    'epr': -1, 'trace_length': 1, 'spectral_entropy': -1,
    'low_band_power': -1, 'high_band_power': -1, 'hl_ratio': -1,
    'dominant_freq': -1, 'spectral_centroid': -1,
    'stft_max_high_power': -1, 'stft_spectral_entropy': -1,
    'rpdi': -1, 'sw_var_peak': -1,
    'pe_mean': -1, 'hurst_exponent': 1,
    'cusum_max': -1, 'cusum_shift_idx': 1,
    # Spilled energy signs — validated on GSM8K (Step 131); cusum_max_spilled confirmed on
    # MATH-500/Qwen-Math-7B too (Step 181, AUROC 0.909 with this sign)
    'epr_spilled': -1, 'sw_var_peak_spilled': -1,
    'cusum_max_spilled': -1, 'min_spilled': -1,
    # Verbalized confidence — null on 1.5B; may work on 7B+
    'verb_conf': +1, 'verb_conf_1p': +1,
}
```

Note: `min_spilled` sign updated from initial `+1` estimate to `-1` — validated on GSM8K Cell 12 sign check.

---

---

## AIRCC CLUSTER — current state (Step 162, 2026-07-08)

Account `omrisegev1`, group `cycle2_tau_averbuch_prj`. VPN required for all access (`ssh aircc`).

| Resource | Value |
|---|---|
| Owner partition | `power-gpu` (36 nodes, no time limit) / QoS `owner_880` |
| Sandbox partition | `sandbox` (1 node, 2 h limit) / QoS `sandbox_owner_880` |
| Allocation | 5760 GPU-h (1237 used by group) |
| Shared dir | `/shared/cycle2_tau_averbuch_prj/omrisegev1/{code,hf_cache,results,logs,pip_cache}` |
| Model cache | Qwen2.5-Math-1.5B-Instruct prefetched ✓ (snapshot at `$SHARED/hf_cache/hub/...`) |
| NGC image | `nvcr.io/nvidia/pytorch:25.01-py3` — **Pyxis only** (rootless Docker dead on power-gpu since 2026-07-01, cgroup v2 BPF block); **Step 162: `--container-name` removed** — a persistent named container makes two jobs on one node collide on first-create; anonymous per-job containers now, image squashfs still enroot-cached (~8 min first import per node, cheap after) |

**Code sync**: `bash cluster/sync_code.sh` (tar-over-ssh, push-independent). After any local change, re-sync before submitting.

Verification, replication-grid execution, and the benchmarking desk are all long since complete —
see the Step-172 "desk closed" summary and Step-181 status at the top of this file, and
HISTORY.md, for current cluster activity. This table is a static resource reference only.

---

## Still-open investigations (not scheduled this session)

*The Jun-17-meeting Priority 1-4 list this section used to hold is fully superseded — Items
5/6 closed (Steps 158/174), Item 3 (QA extension) and Item 4 (benchmarking) both closed out at
the Step-172 "desk closed" milestone. See the MEETING ACTION ITEMS table above for the current
status of all 6. Two sub-issues from the old Priority 1 list were flagged back at Step 152 and,
per a HISTORY.md review, never actually investigated past being flagged — they're listed here
so they don't get lost again:*

1. **MATH-500 fresh-trace-vs-legacy-cache discrepancy** (Step 152 → still open at Step 174):
   the Phase-15 fresh-trace 1-pass L-SML is 85.1, well below the legacy-cache 94.4 headline
   (Step 135). Never diagnosed — compare trace-length distributions / prompts / sampling
   between the two caches.
2. **RAG SelfCheckGPT below-chance orientation** (Step 152): official-variant SelfCheckGPT
   scored *worse* than the hard variant on all 4 RAG datasets, one significantly
   anti-predictive (HotpotQA-official CI [0.137,0.357]). Never investigated past being
   relabeled "suspected/unresolved" in later reports.

---

## Research directions and open questions

### What we know works
- **Spectral features of H(n)** work on reasoning-heavy domains (MATH-500, GPQA). GOOD_5, continuous L-SML: best published unsupervised single-pass numbers on these domains.
- **Spilled energy ΔE(n)** cross-dataset validated: competitive individual AUROCs on both MATH-500 and GSM8K, corr(H,ΔE) = 0.984–0.989.
- **Not general-purpose**: short factual QA traces (TriviaQA, WebQ) are structurally incompatible.
- **Continuous L-SML** (+3.53pp over binarized, 25/29 cells) — merged to `master` at Step 159, the production method ever since.
- **within_H/cross ratio** is a dataset-level diagnostic for L-SML benefit: long reasoning = 0.04 (near-independent, L-SML gains a lot); short traces = 0.99 (redundant, gains less).
- **GOOD_5 is LOCO-validated** (Step 153): held-out subset selection over the full 65k-subset landscape cannot beat it (0.6295 vs 0.636). Candidate tweaks if ever revisited: `low_band_power`→`hl_ratio` (18/29 cells, +0.4pp) or the consensus {spectral_entropy, sw_var_peak, cusum_max, cusum_shift_idx} (+0.9pp in-sample). `cusum_shift_idx` (shift timing) is the only new feature that keeps earning a place.
- **The ρ≥0.75 correlation filter is unnecessary for continuous L-SML** (Step 153): subsets with violating pairs average higher AUROC; the clustering handles dependence (consistent with Steps 135/141).

### Verbalized confidence — model-size gated
- 1.5B: null result confirmed (Step 131). Model doesn't follow "Confidence: X" instruction.
- `parse_verbalized_confidence` is now correct — ready to test on 7B+.
- **Do not include verb_conf in GOOD_FEATURES for 1.5B runs.**

### M=9 orthogonal feature set design — RESOLVED (falsified, Step 131)
corr(epr_H, epr_ΔE) = 0.989 on MATH-500 (matches GSM8K's 0.984) — nowhere near the <0.6
threshold that would have justified treating H(n) and ΔE(n) as independent views. H and ΔE
are redundant, not orthogonal. Do not pursue the 9-feature 3-group design.

### What was ruled out
- **Pivot signals as extra fusion views** (Step 153, closes the Step-151 thread): anomaly scorers (Mahalanobis/GMM/KDE/IForest/AE/PRAE), BOCPD (on H(n) AND on the ΔE logprob trace), HMM, AR/Kalman innovations — all reduce AUROC when added to any good subset (paired bootstrap, 32 cells). bocpd_ecp_spilled is a fine standalone signal (0.726 gsm8k-trace) but adds nothing to the fusion.
- **Subset search as a lever**: best-of-65k in-cell is +8.5pp of selection bias; LOCO-honest selection ≤ GOOD_5. Do not chase subsets.
- **Verbalized confidence on 1.5B**: null result (Step 131). Model-size gated.
- **Hedging count**: not formalized, domain-dependent, weaker than spectral. Do not implement.
- **NLI/semantic entropy methods**: require additional model inference. Out of scope for zero-extra-compute.
- **Quantile calibration**: null result. Median binarization only.

---

## Branch situation

All work is on `master` (branch consolidation completed Step 159; all feature branches deleted
after merge). Colab clones `master` by default.

---

## Still-open GPU/Colab items

- **Step 132** (MATH-500 SpilledEnergy GPU verification run) — never run; superseded in spirit by
  the Step-181 CPU analysis, which used the Phase-15 cache's already-captured spilled-energy
  trace to confirm `cusum_max_spilled` on MATH-500/Qwen-Math-7B (AUROC 0.909, sign −1 correct,
  and it clears a fusion gate as a 6th view). The original 1.5B-specific verification notebook
  itself was never run and stays open if a dedicated re-check is ever wanted.
- **Phase 13**: `Spectral_Analysis_MathComp_Phase13.ipynb` — L-SML vs EDIS, Qwen2.5-Math-1.5B on
  AMC23/AIME24. Still open — the `\boxed{}` grading bug flagged early on was never confirmed
  fixed; AIME24 also separately floors at ~2% accuracy (Phase 15), independent of the bug.
- **Phase 14**: `Spectral_Analysis_Phase14_GPQA_Comparison.ipynb` — L-SML vs VC/SC/SCVC
  (arXiv:2603.19118), DeepSeek-R1-0528-Qwen3-8B on GPQA Diamond. Notebook bugs were fixed at
  Step 144 (MAX_NEW→4096, `lsml_continuous_pipeline`, `lsml_ci` fix) but no completion step
  exists after that — genuinely never run to a result, not just "next."

---

## Completed phases

This table was discontinued after Step 132 and duplicated HISTORY.md poorly for 45+ steps.
Full step-by-step log: `HISTORY.md` (181 steps as of 2026-07-13).

---

## Best results / competitor numbers

Superseded by the replication grid and the Step-172 "desk closed" tally. Current numbers live
in `results/repgrid/*.csv` (canonical, machine-readable) and the master per-domain table
(`results/action_items/per_domain_breakdown.html`, Step 173) — both cover every (dataset,
model) cell in the project, with CEILING/FLOOR/REJECT flags, not just the handful in the old
static tables here. See also Research_Directions.md's "Current Best Results" table.

---

## Key decisions (permanent — do not revisit)

1. No old supervised numbers — Step 100 historical only.
2. GOOD_FEATURES = `['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']` — final 5.
3. Median binarization only — quantile calibration dropped (null result).
4. HTML table: per domain, per model, same-task/same-model/same-dataset only.
5. Cite Jaffé-Fetaya-Nadler 2016. Never say "Nadler" alone. Method name = L-SML.
6. Never say "MV_EPR" — the method is spectral/L-SML.
7. Branch cleanup done — all work on `master`. `analysis/theorem-validation` and `experiment/lsml-variants` are superseded and can be deleted.
8. Hedging count: ruled out — not formalized, domain-dependent, weaker than spectral.
9. Continuous L-SML (`lsml_continuous_pipeline`) is the production method — merged to `master` at Step 159, not gated on Step 132.
10. Verbalized confidence on 1.5B: null result (Step 131). Do not include in GOOD_FEATURES for 1.5B runs.
11. `min_spilled` sign = −1. Validated GSM8K Cell 12.
12. "CONT" is retired. Say "L-SML continuous" (with feature count when relevant: "L-SML continuous 5").
13. **Per-view non-monotone reshaping is CLOSED (Step 218).** The feature pool ships unchanged — no
    view added, none replaced. The fold does repair the view (up to +26.5pp single-view) and U-PCR
    does re-admit 33% of the views it had excluded, but the gain is **99% absorbed** by the other
    views (marginal +8.00pp → conditional +0.05pp, Wilcoxon p = 0.99) and the deployable fused
    macro is **+0.05pp**. Do not re-open on a marginal-AUROC argument — R2 in
    `scripts/nonmono_v2/stage4_redundancy.py` bounds *every* transform, not just the ones tested.
    Two conditions would re-open it: a selector reaching **precision 0.654** (at which point the
    transform of record switches from `mode_centre` to `squared`), or a materially less redundant
    pool. If ever adopted the mode is `replace`, never `add`.
14. LR oracle corrected conclusion (Step 147, common-cell basis): supervised LR beats L-SML by **+4.7 / +3.8 / +3.6pp** on 5/9/16 features (LR 68.9/66.8/67.8 vs CONT 64.2/62.9/64.1); in-sample ceilings 70.5/73.7/79.3. Gap largest on GPQA (+4.9pp) and RAG+QA (+5.8pp), ~0 on reasoning (both near ceiling). "5 best" = named sets non-nested (STABLE_H9 drops spectral_entropy) + overfitting (CV flat while ceiling climbs). LR vs L-SML weights correlate weakly (Spearman ~0.1–0.2). See `SUPERVISED_ORACLE_CORRECTION.md` for evaluation rules; reproduce with `python scripts/oracle_report.py`.

## Latest development — Step 234

Repeated-measurement reliability U-PCR was implemented and tested for global
answer detection using one saved LLM pass. A synchronized moving-block
bootstrap can estimate a stable within-procedure covariance after excluding
features that the resampling procedure does not preserve. However, reliability
did not improve the target ranking. The best safe connection, Wiener-filtered
DUFS-LIU, changed AUROC by only +0.0006 on Qwen3-4B/GSM8K and +0.0013 on
Qwen3-4B/MATH; both paired intervals include zero. Direct generalized latent
coordinates are incompatible with U-PCR because they remove its required
off-diagonal covariance. **Decision: retain DUFS-LIU mixed-v2 and do not open
the six confirmation cells for this candidate.** Full report:
`results/repeated_measurement_reliability/REPORT.md`.

## Current direction — Step 235

The current fusion-development cycle is closed. Across Steps 225--234, sparse
dependency models, SU-PCR/SDSF, view and micro-view fusion, atomic operators,
family relevance gates, repeated cross-view diffusion, feature
transformations, and repeated-measurement reliability all exposed useful
structure, but none produced a robust target-aligned gain over the common
DUFS-LIU core on the current single-pass static feature pool.

The closing 24-cell sensitivity experiment also tested deployed-U-PCR hard
feature filtering before IU-PCR and DUFS-LIU. Full-pool mixed-v2 DUFS-LIU
remains best at 0.776562 macro AUROC. The ordinary `rho_max/3` filter lowers it
to 0.774249, and the strictest filter reaches 0.764153. DUFS's clean increment
over matched IU-PCR changes from +0.048 AUROC points without filtering to
-0.025 points with the ordinary filter. Median Spearman agreement between the
full-pool DUFS gates and estimated rho is 0.794, showing that hard deletion
mostly duplicates soft suppression while also removing covariance information.
All previous IU-PCR, DUFS-LIU, and deployed-U-PCR score files reproduced
exactly in 24/24 cells. Do not develop another hard-filter or gating variant on
these cells.

This is a bounded saturation result, not a claim that U-PCR cannot be improved.
The forward implementation standard is **DUFS-LIU mixed-v2**. Historical
stable-only runs remain unchanged for auditability. New core variants are
paused until an application supplies a new identifiable signal, a valid
nuisance intervention, or materially different features.

The project now has two active application directions:

1. **Hallucination localization.** Frozen GL-LIU v1 reaches 31.36% ProcessBench
   F1 versus 25.71% for Mind the Gap. A unified global/local core-five
   DUFS-LIU system reaches 31.72%, but its +0.37-point advantage is descriptive,
   not a confirmed replacement. The broad-28 local contract is rejected at
   29.03%. The next test must use a new dataset/model family and compare the
   frozen temporal localizer with the simpler core-five local DUFS-LIU system.
2. **Hallucination in RAG citations.** The next proposed system is
   evidence-contrast U-PCR/DUFS-LIU: keep the generated answer fixed, perturb
   the available evidence, and fuse full-evidence, no-evidence, and
   leave-one-chunk-out response traces. This is not yet an experimental result.
   Its benchmark, grouped split, label boundary, baselines, and failure tests
   must be registered before implementation.

Current decision and review package:
`Research_Directions.md` and
`docs/research_notes/claude_review_application_pivot_2026-08-08.md`.
Hard-filter evidence:
`results/hard_filter_dufs_liu_24cell/REPORT.md` and
`results/hard_filter_dufs_liu_24cell/MECHANISM_ANALYSIS.md`.

## Latest router correction — Step 277

The c-STG/manifold-author follow-up has now been tested first on the intended
**Global completed-trace hallucination-detection task**, using the exact frozen
24-cell panel and six family contributions whose sum reproduces IU-PCR to
machine precision.  Five-fold ranking metrics are averaged within folds;
balanced supervised controls and three fixed c-STG seeds are used.

The historical IU-rank quartile oracle remains real (+2.833pp equal-family),
but it is not accessible to the held-out router. `cstg_iu_rank` reaches 0.7464
equal-family AUROC versus 0.7506 for global LR (delta -0.0042, 95% CI
[-0.0140,+0.0041], 3/8 family wins). A simple held-out quartile router is worse
at 0.7380. The exploratory full core c-STG reaches 0.7563 and beats its
permuted-core control by +0.0148 [0.0061,0.0252], but its advantage versus
global LR is uncertain (+0.0057 [-0.0051,+0.0193]), its cell macro is slightly
lower, and GSM8K/MATH500 regress. Decision:
`GLOBAL_ORACLE_NOT_ACCESSIBLE_BY_CSTG`. This is evidence of a heterogeneous
context-by-family association, not a robust router and not a gate to
LTSREx/LEGO.

Canonical artifacts:
`docs/experiments/GLOBAL_CONTEXTUAL_STG_ROUTER_DIAGNOSTIC_V1.md` and
`results/global_contextual_stg_router_diagnostic_v1/`.

## CIW-DEEM multi-application transfer — Step 292

The official CIW-DEEM challenger was added to the reconstruction benchmark's
compatible application lanes. Exact five-seed CIW was run on 22 external
completed-response cells and on RAGTruth's full original-30 response input. A
frozen response-plus-token adapter was evaluated for ProcessBench/PRMBench
localization, and a separate causal-prefix fit was evaluated at token budgets
16--512.

The strongest new transfer is RAGTruth response detection: test AUROC/AUPRC
`0.771222/0.635797`, versus `0.760523/0.596613` for Original-30 IU-PCR and
`0.762882/0.598308` for Original-30 DUFS-LIU. External response transfer is
heterogeneous, localization does not improve, and early detection is close to
IU28 rather than uniformly better.

The coverage audit rejects silent substitutions. EDIS lacks the partition
energy source required by CIW's 3-by-3 core; RAG sentence/token/span/claim,
stopping, and white-box hidden-state lanes require new unit-specific methods.
See `docs/experiments/CIW_DEEM_MULTI_APPLICATION_V1.md` and
`results/ciw_deem_multi_application_v1/`.

## Latest localization integration — Step 361 (2026-09-14)

The frozen four-view Rényi/escort-varentropy bank (`H0lim`, `VE_0`,
`VE_0.75`, `VE_1`) was combined with whole-answer position-varying IU in an
isolated worktree. Real smoke and the complete 13,769-answer benchmark pass
with no failed outer/nested vectors, full exclusion replay, a bitwise-invariant
label firewall and exact `VE_1` reference replay.

The answer-local position-shrinkage arm learns the hypothesized early-to-late
shift toward lower-alpha views and improves PRMB within-answer AUC over its
scale-only control by +.001056 (98.333% CI [.000357,.001770]), but its PB
change is uncertain (+.1640 points, CI [-.1729,+.4966]) and PRMScore is
slightly lower. Fully external position IU is rejected: -1.7024 PB points
versus the position-mean control, CI [-2.6533,-.7953]. No fused arm dominates
the task-dependent leaders: `VE_0.75` on PB and `VE_0` on PRMB. Present the
alpha frontier and learned temporal mechanism as the result; do not claim a
universal fused winner. Full report:
`results/renyi_position_temporal_fusion_v1/REPORT.md`.

---

## Deferred

## Step 326 - pass2a Joint/trajectory shortlist is running (2026-09-07)

The full anchor evaluation and review are complete. The next registered run
scores all 13,769 rows with the newer condition100 Joint map, graph and
permuted-graph controls, equal-graph controls, and IU/Joint trajectory
readouts. Preflight passed on 110 replay cases, 17 new maps, short records and
long records. At the latest check the resumable scorer was at 700/13,769 rows
with three workers; evaluation and review are not yet available. Treat partial
files as checkpoints, not results. Corrected historical refits and the
token/window sampling shortlist remain pending after this run.

- Verbalized confidence on 7B+ — parser is ready; needs one inference run with Qwen2.5-Math-7B on GSM8K
- Phase 10 RAG re-run with variant=4 prompt — low priority
- LapEigvals integration into spectral_utils — potential Group D feature for M=12, low priority
