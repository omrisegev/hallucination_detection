"""Record audited outcomes and preserve the distinction from an RBM ceiling."""
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base


def main():
    out=ROOT/'results/rbm_supervised_position_diagnostic_v1'
    for n in ('RESULT_REVIEW.json','OBJECTIVE_REVIEW.json'):
        assert json.loads((out/n).read_text())['status']=='PASS'
    data=json.loads((out/'METRICS.json').read_text());m=data['metrics'];contrasts=data['contrasts']
    z=np.load(out/'BOOTSTRAP_DRAWS.npz');names=z['names'].tolist()
    for key,v in contrasts.items():
        k=names.index(key);level=v['ci_level'];quantiles=[(1-level)*50,100-(1-level)*50]
        np.testing.assert_allclose(np.percentile(z['pb'][:,k],quantiles),v['pb_ci'],atol=1e-14)
        np.testing.assert_allclose(np.percentile(z['prm'][:,k],quantiles),v['prm_within_ci'],atol=1e-14)
        assert z['pb'].shape[0]==z['prm'].shape[0]==10000
    for field in ('pb','prm'):
        np.testing.assert_allclose(z[field][:,names.index('conditional_vs_static')],
            z[field][:,names.index('conditional_vs_prior')]+z[field][:,names.index('prior_vs_static')],atol=1e-14)
    base.atomic_json(out/'BOOTSTRAP_REVIEW.json',dict(status='PASS',draws=10000,contrasts=len(contrasts),
        checks=['reported intervals from all saved draws','nested contrast identity in every draw'],
        limitation='Conditional on fitted models; does not include training/refit variability.'))
    recommendations=[
        dict(priority=1,finding='PRMB conditional weights add .006881 within-answer AUC beyond position prior; 97.5% CI [.003997,.009723].',
             implication='Joint predictive signal exists in this step-mean diagnostic; marginal reversal alone is no longer the only evidence.',
             limitation='Conditional remains below original RBM; PRMScore falls versus prior. Not evidence of unlabeled learnability.',
             next_question='Before another unlabeled model, bridge this diagnostic to the original token-Top10 representation to test whether the useful interaction survives without step averaging.'),
        dict(priority=2,finding='PB prior adds2.761pp; conditional adds .578pp with97.5%CI[-.267,1.433]pp.',
             implication='Most observed PB gain over static is a position-prior effect. No clear incremental PB benefit from conditional weights.',
             next_question='Token-neighbor information across step boundaries is a distinct diagnostic; compare matched feature evidence with real versus shuffled adjacency, before temporal smoothing.'),
        dict(priority=3,finding='Shared diagonal variance was tested earlier; state-specific diagonal variance is not tested.',
             next_question='Retain two latent Gaussian components with regularized separate diagonal variances as a candidate only; step-label variance differences do not prove token-level latent state recovery.'),
        dict(priority=3,finding='DUFS full run is active and separate from this completed diagnostic.',
             next_question='Await its feature-selection results and reviews without modifying the frozen run; it is not token selection or cross-step temporal fusion.')]
    base.atomic_json(out/'NEXT_STEPS.json',dict(status='stop_for_discussion',recommendations=recommendations,
        no_automatic_next_experiment=True,not_adopted=['new conditional supervised scorer as an answer-only method','first_near_max for RBM']))
    names={'supervised_static':'Supervised fixed fusion','supervised_prior':'Fixed fusion + position prior',
           'supervised_conditional':'Position-dependent fusion','rbm12__logit_old':'Existing answer-local RBM12, Logit/Top10',
           'var15_iu__old':'Varentropy15 contribution IU-PCR','var50__old':'Varentropy50',
           'entropy_step_mean':'Raw entropy, step mean','equal_step_mean':'Equal normalized step means'}
    lines=['# Supervised position diagnostic: complete',
      '', 'All13,769 answers;135 converged fits;40 result configurations, including35 unchanged historical references.',
      'Frozen branch codex/rbm-supervised-position-diagnostic-v1; base ce008fb1d; protocol/code90e56eca8.',
      '', 'The first three methods fit labels from other source-group folds, separately per cell.',
      'They use saved bank12 feature means per STEP. This differs from the existing RBM token Top10 pipeline.',
      'The internal three-way comparison is matched; the RBM comparison is context, not an isolated test of supervision.',
      '', '| Method | PB macro % | PRMB within AUC | Mean fold AUC | PRMScore |',
      '|---|---:|---:|---:|---:|']
    for k,label in names.items():
        v=m[k];lines.append(f"| {label} | {100*v['pb_all8']:.4f} | {v['prm_within']:.6f} | {v['prm_fold_auc']:.6f} | {v['prmscore_q08']:.6f} |")
    lines += ['', 'Primary: conditional minus position prior. PB +0.5779pp,97.5%CI[-0.2671,+1.4332]pp;',
      'PRMB within +0.006881,97.5%CI[+0.003997,+0.009723].10,000 paired source-group draws.',
      'PB152 successes gained,123 lost;76 losses move early,47 late. Six PB cells rise,two fall.',
      'The prior alone adds2.7609pp over static. Position interactions contain PRMB ranking information,',
      'but do not establish a PB gain or a better method than the original RBM. No general winner.',
      '', 'Checks: all35 reference metrics replay; independent40-method arithmetic audit;135 saved models',
      'and training-only normalization/thresholds replay;135 objectives and gradients replay;',
      'max absolute final gradient2.85e-6; all13,769 outputs valid,6,030 mixed-label PRMB answers.',
      'No PB labels are invented after the first error. Step labels are not treated as token truth.',
      '', 'The common raw-entropy q0.3 gate is unchanged. PRMScore q0.8 uses TRAINING scores from',
      'the same model that predicts each held-out fold. This is in-training calibration, not nested OOF.',
      'Supervised pooled OOF AUC is deliberately absent; mean held-out-fold AUC is shown instead,',
      'with the same fold metric added to all references. Within-answer AUC is the local-ranking endpoint.',
      'Intervals condition on fitted predictions and do not cover refitting or earlier research choices.',
      '', 'Next: discuss a matched Top10 bridge before a new unlabeled model. Do not infer that a negative',
      'or weaker step-mean model bounds RBM performance. Cross-boundary locality is a separate open',
      'question; separate latent-state variance remains untested. DUFS continues unchanged.',
      'No new training family, readout tuning, HTML, or deletion was launched. Development evidence only.', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    base.atomic_json(out/'RUN_STATE.json',dict(status='COMPLETE',answers=13769,converged_fits=135,configurations=40,
        frozen_commit='90e56eca8',reviews=['RESULT_REVIEW.json','OBJECTIVE_REVIEW.json','BOOTSTRAP_REVIEW.json']))
    print('COMPLETE: full scores, independent arithmetic and loss review, fixed-draw interval review.')


if __name__=='__main__':main()
