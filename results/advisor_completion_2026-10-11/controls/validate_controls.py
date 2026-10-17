"""Independent arithmetic checks using sklearn and repository PRMBench evaluator."""
import csv, json, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT=Path(__file__).resolve().parent.parent
REPO=ROOT.parents[1]
sys.path.insert(0,str(REPO))
from spectral_utils.prmbench import prmbench_evaluate

def main():
    inputs=ROOT/'inputs';out=ROOT/'controls'
    z=np.load(inputs/'bank_N16.npz');ids=z['idx'].astype(str);off=z['offsets']
    meta={r['idx']:r for r in json.loads((inputs/'prmbench_meta.json').read_text())['rows']}
    y=np.concatenate([np.isin(np.arange(1,b-a+1),meta[u]['error_steps']) for u,a,b in zip(ids,off[:-1],off[1:])])
    saved=np.load(out/'ALL_BANK_PER_ANSWER.npz');rng=np.random.default_rng(20261011)
    eligible=[i for i,(a,b) in enumerate(zip(off[:-1],off[1:])) if y[a:b].any() and not y[a:b].all()]
    sample=rng.choice(eligible,80,replace=False)
    worst_auc=0.;nchecks=0;tie_changes=0
    for bank in ['N11','N13','N14','N16','N22','N25','N40','N43','N53','N56']:
        zz=np.load(inputs/f'bank_{bank}.npz')
        assert np.array_equal(zz['idx'],ids) and np.array_equal(zz['offsets'],off)
        for key in [k for k in zz.files if k.startswith('S__')]:
            method=key[3:];s=zz[key]
            for i in sample:
                a,b=off[i:i+2];expected=roc_auc_score(y[a:b],s[a:b]);actual=saved[f'{bank}__{method}__auc'][i]
                worst_auc=max(worst_auc,abs(expected-actual));nchecks+=1
                assert abs(expected-actual)<1e-12
            exact=saved[f'{bank}__{method}__position']
            for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
                canonical=int(np.flatnonzero(s[a:b]>=s[a:b].max()-8*np.finfo(float).eps)[0])
                tie_changes+=int(canonical!=exact[i])
    assert tie_changes==0
    rows=list(csv.DictReader((out/'EXACT_DECISION_RULES.csv').open()))
    ref={(r['method'],r['rule'],r['classification']):r for r in rows}
    flags=np.load(out/'DECISION_FLAGS.npz');worst_f1=0.;nf1=0
    for key in flags.files:
        if '__' not in key:continue
        method,rule=key.split('__',1);fl=flags[key]
        pred=[{'idx':u,'labels':(~fl[a:b]).astype(int).tolist()} for u,a,b in zip(ids,off[:-1],off[1:])]
        result=prmbench_evaluate(pred,list(meta.values()),use_redundancy_head=False)
        expected=(result['total']['f1']+result['total']['negative_f1'])/2
        actual=float(ref[method,rule,'ALL_NONCONTROL']['category_score'])
        worst_f1=max(worst_f1,abs(expected-actual));assert abs(expected-actual)<1e-12;nf1+=1
        for kind in sorted(set(r['classification'] for r in meta.values())-{'correct'}):
            valid=result['by_classification']['f1'][kind];error=result['by_classification']['negative_f1'][kind]
            expected=valid if kind=='multi_solutions' else (valid+error)/2
            actual=float(ref[method,rule,kind]['category_score'])
            worst_f1=max(worst_f1,abs(expected-actual));assert abs(expected-actual)<1e-12;nf1+=1
    # Distinct independent dataset labels/offsets vs parent's persisted population.
    pop=np.load(inputs/'population.npz')
    assert np.array_equal(pop['idx'],ids) and np.array_equal(pop['offsets'],off) and np.array_equal(pop['y'],y)
    nu=np.load(out/'NUISANCE_INPUTS.npz');assert np.array_equal(nu['length'],pop['token_lengths'])
    qties={m:int((z['S__'+m]==np.quantile(z['S__'+m],.8)).sum()) for m in ['LSML_CONT','IU_PCR','ALL_EQ','DS_FILTER_EQ']}
    assert all(v==0 for v in qties.values())
    result=dict(status='PASS',auc_sklearn_checks=nchecks,max_auc_difference=worst_auc,
        exact_vs_canonical_tolerant_top1_selection_differences=tie_changes,
        official_evaluator_prmscore_checks=nf1,max_f1_difference=worst_f1,
        population_identity='exact idx/offsets/y agreement with parent population',
        token_span_lengths='exact agreement with independent parent lengths',q80_boundary_ties=qties,
        out_of_range_rows=sum(any(e<1 or e>r['n_steps'] for e in r['error_steps']) for r in meta.values()),
        warning='Arithmetic validation is not an independent methodological review; final independent review remains required.')
    (out/'ARITHMETIC_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
