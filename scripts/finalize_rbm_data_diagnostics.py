"""Record the reviewed diagnosis and a bounded next-step recommendation."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base,csv_write
OUT=ROOT/'results/rbm_data_diagnostics_v1'


def insert_after_title(path,text):
    original=path.read_bytes();end=original.find(b'\n')+1
    path.write_bytes(original[:end]+text.encode('utf8')+original[end:])


def main():
    for filename in ('RESULT_REVIEW.json','DIAGNOSTIC_REVIEW.json','UNIT_TEST_REVIEW.json','RELIABILITY_REVIEW.json'):
        assert json.loads((OUT/filename).read_text())['status']=='PASS',filename
    assert json.loads((OUT/'DIAGNOSTICS.json').read_text())['n_answers']==13769
    plans=[
      dict(priority=1,direction='RBM score and readout interface',
           finding='Near-max harms RBM6 and RBM12 through earlier selections, with the gate unchanged.',
           effect='PB -5.1066 and -8.8591 percentage points; 732/882 lost successes, all early.',
           uncertainty='97.5% PB intervals [-7.113,-3.154] / [-11.211,-6.522] pp.',
           coverage='All 6800 PB answers; all 13769 models replayed.',
           alternative='Large near-max sets are observed; sigmoid compression is a hypothesis, not a proven sole cause. No lost case required exact numerical saturation at 1.',
           smallest_followup='Same saved RBM weights: top10 mean of oriented token logits versus top10 mean of posteriors, with the frozen near-max rule and gate. No refit or threshold tuning.',
           status='Recommended next experiment; NOT run. Do not promote near-max as the RBM default.'),
      dict(priority=2,direction='Regime-dependent fusion reliability',
           finding='The more useful feature changes with entropy regime and answer position on matched answers.',
           effect='Selected surprisal minus entropy AUC: early -0.1061, late +0.04236; interaction +0.14846. Low entropy -0.07109, high +0.04659.',
           uncertainty='95% interaction CI [0.12459,0.17245] for position; [0.09416,0.14142] for entropy. Descriptive development contrasts.',
           coverage='1672 answers eligible in BOTH early/late; 2401 in BOTH entropy regimes; identical step-mean feature aggregation.',
           alternative='This does not establish a label-free rule for learning adaptive weights. Subgroup coverage is conditional; a feature advantage is not a trained fusion gain.',
           smallest_followup='After fixing the score interface, one prespecified answer-local conditional-weight mechanism versus the same static fusion, with equal-weight controls. No feature or regime sweep.',
           status='Measured justification for discussing adaptive fusion; NOT implemented.'),
      dict(priority=3,direction='Separate conditional variances',
           finding='Error/correct step-mean feature variances differ, including within fixed length and position strata. Error-step variance is often LOWER.',
           effect='Entropy mean log(error/correct variance) -0.8046; geometric mean ratio about 0.447. This is not mean token entropy and not a ratio of pooled variances.',
           uncertainty='95% log-ratio interval [-0.8802,-0.7290]; all length/position stratum intervals remain below zero.',
           coverage='4030 PRMB answers with >=2 steps of each class; stratum counts 865-1937 for length/position. Zero variances excluded and counted.',
           alternative='Step labels do not label individual tokens. Heteroscedasticity is not proof of an unsupervised semantic two-state model; shared-diagonal variance already improved fit without improving the task.',
           smallest_followup='Only after linking the variance mismatch to the task failure: two Gaussian components with separate restrained diagonal variances versus shared variance. Keep readout fixed.',
           status='Statistical support; task mechanism and latent class recovery remain open.'),
      dict(priority=4,direction='Serial fusion',
           finding='Residual step dependence remains after one hidden unit and in coarse length/position strata.',
           effect='Lag1 correlation excess over permutation: +0.15710 for bank6 and +0.13656 for bank12; decreases at lags2/3.',
           uncertainty='95% intervals [0.14944,0.16497] / [0.12991,0.14335].',
           coverage='13364 eligible answers at lag1; 12388 at lag2; 11032 at lag3. Within-stratum coverage is smaller and explicit.',
           alternative='Dependence also occurs on successful answers; coarse strata do not remove all position trends or semantic continuity. It is not proof of useful onset information.',
           smallest_followup='One trajectory component only after the readout issue is isolated, versus a matched permutation control; no sequential-model sweep.',
           status='Real serial structure; causal task value unproven. Defer.'),
      dict(priority=5,direction='More hidden units or shared-noise modeling',
           finding='Large residual correlations exceed the saved model simulation, but are not a useful general failure discriminator in this diagnosis.',
           effect='Mean absolute residual correlation .4218 vs simulated .0536 (bank6), .4329 vs .0430 (bank12). Bank6 excess in PB hits .34677 vs misses .34652.',
           uncertainty='Four synthetic draws per answer condition on estimated parameters. No full model-test p-value or parameter-identification conclusion.',
           coverage='All 13769 answers per bank; 110152 synthetic traces total.',
           alternative='Powers/moments are structurally dependent; a richer model may spend capacity on harmless feature redundancy.',
           smallest_followup='Four hidden units with exact likelihood only if a task-linked residual factor is identified; retain one-unit control before CD or a second layer.',
           status='Do not expand capacity merely to reduce residual correlation.'),
      dict(priority=6,direction='Restarts, CD and a second layer',
           finding='Saved optimizer diagnostics show convergence for 13768/13769 bank6 and all bank12 fits. Likelihood improves, but task gains depend on metric/readout.',
           effect='With near-max, learning gains/losses are 228/390 (bank6) and 212/573 (bank12). Original readout learning improves PRMScore but reduces within-answer AUC.',
           uncertainty='One saved initialization cannot test multimodality or restart sensitivity.',
           coverage='All 27538 saved learned models; median iterations25/28, median max gradient about4e-6/5e-6.',
           alternative='Converging to the generative objective does not guarantee task improvement. CD changes optimization, not the objective by itself.',
           smallest_followup='If optimization becomes the measured bottleneck: restart check, then one-unit exact versus CD; later four-unit exact versus CD; second layer only after useful multi-unit representation.',
           status='Defer; no evidence for making CD the immediate next step.')]
    base.atomic_json(OUT/'NEXT_STEPS.json',dict(status='RECOMMENDATIONS_NOT_EXECUTED',rows=plans,
        scope='Full cached development evidence; no untouched confirmation and no automatic next model.'))
    csv_write(OUT/'NEXT_STEPS.csv',plans)
    discussion='''

The first-near-max rule is not a safe default for the current RBM scores.
Claude's entropy/Varentropy result replays; transfer of that improvement to RBM
fails. Preserve the original RBM readout and all negative results. Keep near-max
as a candidate for Varentropy and contribution IU, not a universal replacement.
Every lost RBM success moved earlier; no gate decision changed. Broad near-max
sets cover about44%/53% of PB steps for RBM6/12 versus21% for Varentropy50.
The sigmoid may compress useful score separation, but that mechanism needs a
saved-weight logit/readout ablation; it was NOT run in this diagnosis.

The most task-relevant new modeling evidence is a reversal of feature reliability.
On the SAME1672 eligible PRMB answers, selected surprisal trails entropy in the
early half by0.1061 AUC, then exceeds it in the late half by0.04236. Both features
use step means; no aggregation mismatch. The paired reversal is0.14846,
95%CI[0.12459,0.17245]. This motivates conditional fusion, not a claim that such
a label-free learner already exists or improves the benchmark.

Class variance is different on4030 eligible PRMB answers, often LOWER on error
steps. Residual correlation and serial dependence are also real model mismatches.
However, residual dependence is present in successes too. Do not equate fitting
these statistics better with better hallucination localization. The observed
near-zero training residual mean is expected from fitted location parameters;
its difference from non-refitted synthetic means is not evidence for more units.

| Priority | Smallest useful direction | Current decision |
|---|---|---|
| 1 | Same weights, inspect pre-sigmoid scoring/readout | Next proposal; no new fit |
| 2 | A single regime-dependent fusion mechanism | Supported by matched reliability reversal; discuss after1 |
| 3 | Separate conditional variances | Statistical evidence; task link and label-free recovery open |
| 4 | One sequential fusion component | Serial structure exists; task value not isolated |
| 5 | Four hidden units/shared noise | Residual mismatch alone is insufficient |
| 6 | Restarts/CD, then deeper layers | Current fits mostly converge; no restart evidence yet |

NEXT_STEPS.json/CSV records effects, uncertainty, coverage and alternative
explanations. EVIDENCE.json/CSV contains232 descriptive diagnostic contrasts;
RELIABILITY_REGIMES.json contains the same-aggregation matched follow-up.
All results remain development evidence. No new model was trained, no HTML
was generated, and the separate DUFS run was left unchanged.
'''
    report=OUT/'REPORT.md'
    if 'The first-near-max rule is not a safe default' not in report.read_text():
        with report.open('a',encoding='utf8') as f:f.write(discussion)
    note='''
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

'''
    p=ROOT/'PROGRESS.md'
    if '## RBM data diagnostics COMPLETE - 2026-09-11' not in p.read_text(encoding='utf8'):
        insert_after_title(p,note)
        insert_after_title(ROOT/'Research_Directions.md',note.replace('Earlier RUNNING entries below are historical.','Original readout references remain available.'))
        history='''

### Step 346 [Codex RBM data diagnostics] - completed 2026-09-11

**What**: Implemented the user-approved saved-data diagnosis in a dedicated
worktree/branch from44f9bced8, frozen13e75a8fd. No model refit, restarts or CD.
All13769 answers/145597 steps; original6 and order6/all12 banks;24 old/new
readout/reference arms. Exact same-input Claude first_near_max replay; original
independently saved float reductions differ by<1e-15 with identical metrics.
Failed empty preflight preserved separately; full run unchanged thereafter.

**Why**: Test whether Claude's development-selected readout gain transfers to
RBM before adding model capacity, and link proposed extensions to measured
failures. User asked to analyze both banks equally and leave DUFS untouched.

**Result**: Near-max harms RBM6 PB by5.1066pp (97.5%CI[-7.113,-3.154]) and
RBM12 by8.8591pp[-11.211,-6.522]. Old/new PB36.2017/31.0950 and36.3750/27.5159.
No gate changes. Gains/losses456/732 and448/882; all losses select too early.
Var15/IU improves35.3498->36.6546; Var50 reproduces Claude35.6755->36.4366.
All13k fits replay; metric review24 arms and diagnostic review27538 states PASS;
8860 direct-pair feature AUC spot checks PASS. Audit RAM caching changes I/O only.

PRMB4030 answers support within-answer class-variance comparison: error-step
entropy variance is often lower (mean log-ratio-.8046), including length/position
strata. Label unit remains STEP, not token; no post-first-error PB relabeling.
Residual correlations greatly exceed four same-model samples but also occur in
successes; bank6 PB excess.34677 on hits vs.34652 on misses. Serial lag1 excess
.1571/.1366 exists but does not by itself establish useful error-onset information.
Matched feature reliability reverses: selected-surprisal minus entropy early
-.1061, late+.04236 on1672 same answers; reversalCI[.12459,.17245]. No adaptive
model trained; shared bank coordinates are not independent replications.

**Decision/reasoning**: Recommend original RBM readout rather than automatic
near-max adoption. Next smallest proposal is a saved-weight pre-sigmoid readout
ablation, then discuss conditional fusion. Separate variances, sequential
fusion, four units and CD remain conditional proposals, not launched jobs.
Convergence flags alone cannot exclude alternate optima; there is no multistart
evidence from these single saved fits. Full cached development, no untouched
confirmation. CSV/JSON/conciseMD in results/rbm_data_diagnostics_v1; no HTML.
'''
        with (ROOT/'HISTORY.md').open('ab') as f:f.write(history.encode('utf8'))
    supplemental=[ROOT/'scripts'/n for n in ('analyze_rbm_readout_failures.py',
        'verify_rbm_data_diagnostics_cached.py','review_rbm_reliability_regimes.py',Path(__file__).name)]
    base.atomic_json(OUT/'ANALYSIS_PROVENANCE.json',dict(hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in supplemental},
        note='Supplemental descriptive analysis and I/O-only audit cache; frozen scoring/diagnostic code unchanged.'))
    base.atomic_json(OUT/'RUN_STATE.json',dict(status='COMPLETE',completed=13769,expected=13769,
        review='PASS',banks=[6,12],n_model_refits=0,next_experiment_started=False))
    print('COMPLETE: reviewed full diagnosis; next steps recorded, no new model launched.')


if __name__=='__main__':main()
