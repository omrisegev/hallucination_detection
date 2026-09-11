"""Concise CSV/JSON evidence ledger; no HTML or automatic winner promotion."""
from pathlib import Path
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.run_rbm_literature_completion import PROGRAM,SUITES,METRIC_KEYS,base,csv_write


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
                rows.append(dict(suite=suite,method=m,**{k:v[k] for k in METRIC_KEYS},
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
                text.append(f"| {m} | {fmt(v['pb_all8'])} | {fmt(v['prm_within'])} | {fmt(v['prmscore_q08'])} | {v['valid_answers']} |")
            text+=['','Primary planned contrasts (97.5% source-group intervals; 10,000 draws):']
            for key,c in data['contrasts'].items():
                if c['primary']:text.append(f"- {key}: PB {c['pb_delta']:+.6f} pp, CI {c['pb_ci']}; within {c['prm_within_delta_common']}, CI {c['prm_within_ci']}; gained/lost {c['gained']}/{c['lost']}.")
            text+=['','No automatic promotion. Other comparisons are descriptive95%; intervals do not cover all prior research choices.',
                    'See FIT_HEALTH.json for convergence, failures and method details; PB_CELLS.csv and CHANGED_SUCCESSES.csv preserve case-level differences.']
            (out/'REPORT.md').write_text('\n'.join(text)+'\n',encoding='utf8')
        evidence.append(entry)
    base.atomic_json(PROGRAM/'EXPERIMENT_LEDGER.json',evidence)
    base.atomic_json(PROGRAM/'CONTRASTS.json',contrasts)
    csv_write(PROGRAM/'COMPARISON.csv',rows)
    print('[summary]',[(e['suite'],e['status']) for e in evidence],flush=True)


if __name__=='__main__':main()
