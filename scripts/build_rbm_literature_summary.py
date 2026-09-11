"""Concise CSV/JSON evidence ledger; no HTML or automatic winner promotion."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_literature_completion import PROGRAM,SUITES,METRIC_KEYS,base,csv_write


def display_name(method):
    references={
        'rbm6__old':'RBM6, trained, posterior',
        'rbm12__old':'RBM12, trained, posterior',
        'rbm6__logit_old':'RBM6, trained, Logit',
        'rbm12__logit_old':'RBM12, trained, Logit',
        'initial6__old':'RBM6, before training, posterior',
        'initial12__old':'RBM12, before training, posterior',
        'entropy__old':'Token entropy, saved reference',
        'var15__old':'Varentropy, top 15 probabilities',
        'var50__old':'Varentropy, top 50 probabilities',
        'var15_iu__old':'Varentropy15 contributions, IU-PCR fusion',
        'var15_equal__old':'Varentropy15 contributions, equal fusion',
        'shrinkage__old':'RBM6 with weight shrinkage, posterior',
        'diagonal__old':'RBM6 with shared diagonal variance, previous experiment',
    }
    if method in references:return references[method]
    bank,rest=method.split('_',1);variant,score=rest.rsplit('_',1)
    descriptions={
        'variance_shared':'Gaussian fusion, shared state variance',
        'variance_separate':'Gaussian fusion, separate state variances',
        'exact1':'RBM, 1 hidden unit, exact training',
        'exact4':'RBM, 4 hidden units, exact training',
        'cd1':'RBM, 1 hidden unit, CD-10 training',
        'cd4':'RBM, 4 hidden units, CD-10 training',
        'best_exact1':'RBM, 1 hidden unit, best density fit of 3 starts',
        'best_exact4':'RBM, 4 hidden units, best density fit of 3 starts',
        'layer2_exact':'Stacked RBMs, 4 to 1 hidden units, exact second layer',
        'layer2_cd':'Stacked RBMs, 4 to 1 hidden units, CD-10 second layer',
        'chain_full':'RBM with token sequence fusion across steps',
        'chain_step_reset':'RBM with token sequence fusion reset at each step',
        'chain_shuffled':'RBM with shuffled token order, control',
    }
    return f"{descriptions[variant]}, {bank[1:]} features, {score}"


def main():
    evidence=[];rows=[];contrasts=[]
    for suite in SUITES:
        out=PROGRAM/suite;state=out/'RUN_STATE.json'
        status=json.loads(state.read_text()) if state.exists() else dict(status='NOT_STARTED')
        review=out/'RESULT_REVIEW.json'
        checked=review.exists() and json.loads(review.read_text()).get('status')=='PASS'
        entry=dict(suite=suite,status=status['status'],review=checked,path=str(out))
        if status['status']=='COMPLETE' and checked:
            data=json.loads((out/'METRICS.json').read_text());health=json.loads((out/'FIT_HEALTH.json').read_text())
            names={m for r in health for m in r['models']};fit_summary={}
            for m in sorted(names):
                fits=[r['models'][m] for r in health if m in r['models']]
                fit_summary[m]=dict(finite_fits=len(fits),failures=sum(m in r['failures'] for r in health),
                    nonconverged=sum(d.get('converged') is False for d in fits),
                    fit_seconds=sum(d.get('seconds',0) for d in fits))
            entry['fit_summary']=fit_summary
            for m,v in data['metrics'].items():
                rows.append(dict(suite=suite,method=m,display_name=display_name(m),**{k:v[k] for k in METRIC_KEYS},
                                 valid_answers=v['valid_answers'],prm_within_n=v['prm_within_n']))
            for key,c in data['contrasts'].items():contrasts.append(dict(suite=suite,comparison=key,**c))
            text=['# '+suite+': full development result','',
                'All 13,769 answers scored; saved-state and separate metric review PASS.',
                'Fusion fits each answer without labels. The entropy gate and PRMScore calibration use external folds.',
                'Posterior and logit scores use the same Top10/argmax; no first_near_max.',
                '', '| Method | PB macro % | PRMB within AUC | PRMScore | Valid answers |',
                '|---|---:|---:|---:|---:|']
            for m,v in data['metrics'].items():
                fmt=lambda x:'NA' if x is None else f'{x:.6f}'
                text.append(f"| {display_name(m)} | {fmt(100*v['pb_all8'])} | {fmt(v['prm_within'])} | {fmt(v['prmscore_q08'])} | {v['valid_answers']} |")
            text+=['','Primary planned contrasts (97.5% source-group intervals; 10,000 draws):']
            for key,c in data['contrasts'].items():
                if c['primary']:
                    a,b=key.split('_minus_')
                    text.append(f"- {display_name(a)} minus {display_name(b)}: PB {100*c['pb_delta']:+.6f} pp, CI {[100*x for x in c['pb_ci']]}; within {c['prm_within_delta_common']}, CI {c['prm_within_ci']}; gained/lost {c['gained']}/{c['lost']}.")
            text+=['','No automatic promotion. Other comparisons are descriptive95%; intervals do not cover all prior research choices.',
                    'See FIT_HEALTH.json for convergence, failures and method details; PB_CELLS.csv and CHANGED_SUCCESSES.csv preserve case-level differences.']
            (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')
        evidence.append(entry)
    base.atomic_json(PROGRAM/'EXPERIMENT_LEDGER.json',evidence)
    base.atomic_json(PROGRAM/'CONTRASTS.json',contrasts)
    csv_write(PROGRAM/'COMPARISON.csv',rows)
    from scripts.build_rbm_fusion_comparison import main as build_fusion_comparison
    build_fusion_comparison()
    print('[summary]',[(e['suite'],e['status']) for e in evidence],flush=True)


if __name__=='__main__':main()
