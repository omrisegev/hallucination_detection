"""Write compact results only after full model, readout and metric review."""
import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_data_diagnostics import base,csv_write


def main():
    out=ROOT/'results/rbm_supervision_matched_v1'
    for name in ('RESULT_REVIEW.json','MODEL_REVIEW.json','ZERO_UPDATE_REVIEW.json','BATCHED_ZERO_REVIEW.json'):
        assert json.loads((out/name).read_text())['status']=='PASS'
    data=json.loads((out/'METRICS.json').read_text());metrics=data['metrics'];contrasts=data['contrasts'];z=np.load(out/'BOOTSTRAP_DRAWS.npz');names=z['names'].tolist()
    for key,v in contrasts.items():
        i=names.index(key);q=[50*(1-v['ci_level']),100-50*(1-v['ci_level'])]
        for arr,field in (('pb','pb_ci'),('prm','prm_within_ci')):
            assert z[arr].shape[0]==10000
            np.testing.assert_allclose(np.percentile(z[arr][:,i],q),v[field],atol=1e-14)
    for arr in ('pb','prm'):
        np.testing.assert_allclose(z[arr][:,names.index('supervised_vs_frozen')],z[arr][:,names.index('supervised_vs_unsupervised')]+z[arr][:,names.index('unsupervised_vs_frozen')],atol=1e-14)
    base.atomic_json(out/'BOOTSTRAP_REVIEW.json',dict(status='PASS',draws=10000,checks=['reported intervals','contrast identity in every draw']))
    health=json.loads((out/'FIT_HEALTH.json').read_text());assert len(health)==90
    summary={m:dict(fits=sum(r['method']==m for r in health),nonconverged=sum(r['method']==m and not r['converged'] for r in health),
        max_gradient=max(r['gradient_max'] for r in health if r['method']==m),
        max_iterations=max(r['iterations'] for r in health if r['method']==m),
        total_fit_seconds=sum(r['seconds'] for r in health if r['method']==m),
        median_delta_norm=float(np.median([r['delta_norm'] for r in health if r['method']==m]))) for m in ('unsupervised_update','supervised_update')}
    base.atomic_json(out/'FIT_SUMMARY.json',summary)
    rows=[]
    for path in sorted(out.glob('fit_*.npz')):
        info=json.loads(path.with_suffix('.json').read_text())
        with np.load(path) as z:
            for method in summary:
                for j,v in enumerate(z[method]):rows.append(dict(cell=info['cell'],fold=info['fold'],method=method,coefficient=j,value=float(v)))
    csv_write(out/'COEFFICIENTS.csv',rows)
    primary=contrasts['supervised_vs_unsupervised']
    base.atomic_json(out/'DECISION.json',dict(status='stop_for_discussion',primary=primary,
        scope='Two matched shared coefficient updates above identical answer-local RBM; learning loss and training-label access differ.',
        not_claimed=['pure from-scratch supervised versus unsupervised RBM','answer-only updated method','oracle bound','untouched confirmation'],
        finding='PB and within-AUC point estimates improve, but both primary97.5% intervals include zero. PRMScore and mean-fold AUC fall. No confirmed winner or large attainable margin.',
        next='Proposed only: isolate step-classification versus first-error training loss in this SAME coefficient/readout family for PB.193 of271 lost exact successes move late; this motivates, but does not prove, an objective-mismatch hypothesis. Preserve PRMB separately. No automatic model expansion.',
        heldout_loss_caution='BCE improves on every held-out fold, but remains above the constant .5 predictor loss; not proof of calibrated correctness probabilities.'))
    labels={'unsupervised_update':'RBM + unsupervised coefficient update','supervised_update':'RBM + supervised coefficient update','rbm12__logit_old':'Original frozen answer-local RBM12'}
    lines=['# Matched RBM supervision: complete','',
        'Freeze afb852591; branch codex/rbm-supervision-matched-v1; base f7203a8a6.',
        'Full13,769 answers,90 fits, identical source-group folds, labels, token bank12,',
        'answer-specific base models,13 correction parameters, zero initialization,',
        'penalty, optimizer, token Logit/Top10/argmax and entropy gate.',
        'Only the learning objective and access to training step labels differ.',
        '', 'These are COEFFICIENT UPDATE alternatives above an identical saved RBM.',
        'Both use other training answers for the shared correction; base weights stay',
        'answer-specific. This is not a pure from-scratch supervision comparison or an oracle bound.',
        '', '| Method | PB macro % | PRMB within AUC | Mean fold AUC | PRMScore |',
        '|---|---:|---:|---:|---:|']
    for k,label in labels.items():
        m=metrics[k];lines.append(f"| {label} | {m['pb_all8']*100:.4f} | {m['prm_within']:.6f} | {m['prm_fold_auc']:.6f} | {m['prmscore_q08']:.6f} |")
    lines+=['',f"Primary supervised-minus-unsupervised: PB {100*primary['pb_delta']:+.4f}pp,97.5%CI {[round(100*v,4) for v in primary['pb_ci']]};",
        f"within AUC {primary['prm_within_delta_common']:+.6f},97.5%CI {primary['prm_within_ci']}.",
        '10,000 paired source-group draws; uncertainty conditional on fitted predictions.',
        '', 'Full model/loss and separate metric arithmetic review PASS. Zero update reproduces',
        'all original peaks (0 changes); batched score roundoff <=2.85e-14. No step-feature averaging.',
        'The supervised loss is class-balanced STEP BCE after Top10. PB labels are known',
        'correct prefixes plus the first error; later steps remain unknown. No token truth labels.',
        'PRMScore q.8 uses other-fold training scores from the SAME fold model in both arms.',
        'Mean held-out-fold AUC replaces pooled cross-fold AUC for both updated methods.',
        '', 'Seven PB cells rise,one falls.318 exact gated successes gained,271 lost;193 losses move late.',
        'Both primary97.5% intervals include zero. No confirmed winner or large headroom claim.',
        'Held-out balanced BCE improves in all45 folds, but remains above the constant .5 loss;',
        'see HELDOUT_LOSS.json. This does not establish good calibration or superiority at the task.',
        'Proposed next question only: does first-error listwise training improve PB over step BCE',
        'in this identical correction/Top10 family? The late losses motivate this test, not prove its answer.',
        '', 'Fit health: '+json.dumps(summary,sort_keys=True),
        '', 'See PB_CELLS.csv,FORENSICS.json,COEFFICIENTS.csv for cell results and learned updates.',
        'No new model was launched after evaluation. Development findings, not untouched confirmation.', '']
    (out/'REPORT.md').write_text('\n'.join(lines),encoding='utf8')
    base.atomic_json(out/'RUN_STATE.json',dict(status='COMPLETE',answers=13769,fits=90,freeze='afb852591',nonconverged=sum(not r['converged'] for r in health)))
    print('COMPLETE: matched two-arm scores,90 model/loss replays, metrics and paired intervals reviewed.')


if __name__=='__main__':main()
