"""Saved-score development diagnostics; no method fitting or threshold promotion.

All global thresholds include synthetic controls; pooled PRMScore excludes them.
MS follows PRMBench's official table generator: valid-step F1, not half that F1.
Best threshold and true-count quotas use labels and are oracle diagnostics only.
"""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
INPUT = OUT.parent / 'thinkfix_rescore_review_20261011' / 'sources' / 'prmscore_inputs'
OFFICIAL = OUT / 'sources' / 'OFFICIAL_PRMSCORE_N16.csv'
METHODS = ['LSML_CONT', 'IU_PCR', 'ALL_EQ', 'DS_FILTER_EQ']
CATS = {'simplicity': ['redundency', 'circular'],
        'soundness': ['counterfactual', 'step_contradiction', 'domain_inconsistency', 'confidence'],
        'sensitivity': ['missing_condition', 'deception', 'multi_solutions']}

def f1(tp, fp, fn):
    denominator = 2 * np.asarray(tp) + fp + fn
    return np.divide(2 * np.asarray(tp, dtype=float), denominator,
                     out=np.zeros_like(denominator, dtype=float), where=denominator != 0)

def metrics(y, flag):
    tp = int(np.sum(y & flag)); fp = int(np.sum(~y & flag))
    fn = int(np.sum(y & ~flag)); tn = int(np.sum(~y & ~flag))
    fe = float(f1(tp, fp, fn)); fv = float(f1(tn, fp, fn))
    return dict(tp_error=tp, fp_error=fp, fn_error=fn, tn_valid=tn,
                f1_error=fe, f1_valid=fv, prm_score=(fe + fv) / 2)

def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, keys); writer.writeheader(); writer.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bp = INPUT / 'bank_N16.npz'; mp = INPUT / 'prmbench_meta.json'
    z = np.load(bp, allow_pickle=False); ids = z['idx'].astype(str); off = z['offsets']
    meta = {r['idx']: r for r in json.loads(mp.read_text())['rows']}
    assert len(ids) == len(set(ids)) == len(meta) and set(ids) == set(meta)
    n = np.diff(off); assert np.all(n > 0)
    classes = np.array([meta[u]['classification'] for u in ids])
    ctrl = classes == 'correct'; answer = np.repeat(np.arange(len(ids)), n)
    pool = ~ctrl[answer]; y = np.zeros(int(off[-1]), bool)
    for i, u in enumerate(ids):
        assert int(n[i]) == meta[u]['n_steps']
        y[off[i]:off[i+1]] = np.isin(np.arange(1, n[i] + 1), meta[u]['error_steps'])
    truth_counts = np.add.reduceat(y.astype(int), off[:-1])
    official = list(csv.DictReader(OFFICIAL.open(newline='')))
    references = {(r['method'], r['rule']): r for r in official}
    gflag = z['G'] >= np.quantile(z['G'], .8)
    gcounts = np.add.reduceat(gflag.astype(int), off[:-1])
    summary = dict(research_commit='080877eb9a8369341d647f4115a3b475c040b46e',
                   reviewed_v10_commit='53ef62e36e80b90de30b83d3fee33e1fb85b82c3',
                   n_answers=len(ids), n_steps=len(y), n_pooled_answers=int((~ctrl).sum()),
                   n_pooled_steps=int(pool.sum()), n_synthetic_controls=int(ctrl.sum()),
                   true_error_steps=int(y.sum()), true_error_rate_all=float(y.mean()),
                   true_error_rate_noncontrol=float(y[pool].mean()),
                   g_definition=str(z['G_definition']), g_survivors=z['G_survivors'].tolist(),
                   input_hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in [bp, mp, OFFICIAL]},
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), methods={})
    curves=[]; aggregates=[]; per_answer=[]
    for method in METHODS:
        score=z['S__'+method]; assert np.isfinite(score).all()
        q80=float(np.quantile(score, .8)); qflag=score >= q80
        order=np.argsort(-score, kind='stable')
        sy=y[order]; sp=pool[order]
        tp=np.cumsum(sy & sp); fp=np.cumsum(~sy & sp)
        # A threshold cannot split ties. Include every scored step tied at its boundary.
        stops=np.r_[np.flatnonzero(score[order][:-1] != score[order][1:]),len(score)-1]
        k=stops+1; p=int(y[pool].sum()); v=int((~y[pool]).sum())
        fe=f1(tp[stops],fp[stops],p-tp[stops]); fv=f1(v-fp[stops],fp[stops],p-tp[stops])
        val=(fe+fv)/2
        # Include the reject-nothing boundary in the exhaustive global-threshold sweep.
        none=metrics(y[pool],np.zeros(pool.sum(),bool))['prm_score']
        oracle_idx=int(np.argmax(val))
        oracle_threshold=float(score[order][stops[oracle_idx]]) if val[oracle_idx]>=none else float('inf')
        oracle_flag=score>=oracle_threshold
        qmetric=metrics(y[pool],qflag[pool]); ometric=metrics(y[pool],oracle_flag[pool])
        assert abs(ometric['prm_score']-max(float(val.max()),none))<1e-12
        assert abs(qmetric['prm_score']-float(references[method,'R0']['overall']))<1e-12
        # Frozen global G counts, with each method's within-answer ranking.
        r2flag=np.zeros(len(score),bool); count_oracle=np.zeros(len(score),bool)
        for a,b,gc,tc in zip(off[:-1],off[1:],gcounts,truth_counts):
            ranking=np.argsort(-score[a:b],kind='stable')
            r2flag[a+ranking[:gc]]=True
            count_oracle[a+ranking[:tc]]=True
        r2metric=metrics(y[pool],r2flag[pool])
        assert abs(r2metric['prm_score']-float(references[method,'R2']['overall']))<1e-12
        for rule,flag in [('ANSWER_Z_Q80',qflag),('R2_FROZEN_G_RECONSTRUCTION',r2flag),
                          ('GLOBAL_THRESHOLD_ORACLE',oracle_flag),('TRUE_COUNT_ORACLE',count_oracle)]:
            count=np.add.reduceat(flag.astype(int),off[:-1])
            row=dict(method=method,rule=rule,**metrics(y[pool],flag[pool]),
                     step_flag_rate_all=float(flag.mean()),
                     controls_false_flag_rate=float((count[ctrl]>0).mean()),
                     all_error_free_false_flag_rate=float((count[truth_counts==0]>0).mean()))
            subtype={}
            for c in sorted(set(classes)-{'correct'}):
                ai=classes==c; si=ai[answer]; mm=metrics(y[si],flag[si])
                subtype[c]=mm['f1_valid'] if c=='multi_solutions' else mm['prm_score']
                aggregates.append(dict(method=method,rule=rule,classification=c,
                                       n_answers=int(ai.sum()),n_steps=int(si.sum()),
                                       mean_true_count=float(truth_counts[ai].mean()),
                                       mean_flag_count=float(count[ai].mean()),
                                       count_bias=float((count[ai]-truth_counts[ai]).mean()),
                                       count_mae=float(np.abs(count[ai]-truth_counts[ai]).mean()),
                                       category_score=subtype[c]))
            for category,members in CATS.items():
                row[category]=float(np.mean([subtype[c] for c in members]))
            for c in ['correct']:
                ai=classes==c
                aggregates.append(dict(method=method,rule=rule,classification=c,
                                       n_answers=int(ai.sum()),n_steps=int(n[ai].sum()),
                                       mean_true_count=float(truth_counts[ai].mean()),
                                       mean_flag_count=float(count[ai].mean()),
                                       count_bias=float((count[ai]-truth_counts[ai]).mean()),
                                       count_mae=float(np.abs(count[ai]-truth_counts[ai]).mean())))
            summary['methods'].setdefault(method,{})[rule]=row
            for i,u in enumerate(ids):
                per_answer.append(dict(method=method,rule=rule,idx=u,classification=classes[i],
                                       n_steps=int(n[i]),true_count=int(truth_counts[i]),flag_count=int(count[i])))
        summary['methods'][method]['oracle_diagnostic']=dict(
            threshold=None if not np.isfinite(oracle_threshold) else oracle_threshold,
            flag_rate=float(oracle_flag.mean()),gain_over_q80=ometric['prm_score']-qmetric['prm_score'],
            prm_native_minus_q80=float(references['Qwen2.5-Math-PRM-7B (supervised)','native 0.5']['overall'])-qmetric['prm_score'],
            status='Label-selected resubstitution oracle, not a deployable method or held-out gain')
        # Grid is a descriptive view; the optimum above searched all tie-preserving cuts.
        for rate in np.linspace(0,.5,501):
            threshold=float('inf') if rate==0 else float(np.quantile(score,1-rate))
            flag=score>=threshold
            curves.append(dict(method=method,nominal_rate=float(rate),threshold=threshold,
                               actual_flag_rate=float(flag.mean()),**metrics(y[pool],flag[pool])))
    summary['limits']=['No source-group confidence interval without authenticated source-group mapping.',
                       'No PRM/LR per-step score export; competitor values are saved reference reports.',
                       'Frozen G rule matches saved R2 arithmetic; not a certified replay of registered R2_COUNT inputs.',
                       'No corrected entropy gate, feature/group moment estimation, channel refit or new model inference.',
                       'Oracle diagnostic uses evaluation labels; it neither selects a deployed threshold nor proves decision is the only bottleneck.']
    (OUT/'DECISION_ANALYSIS.json').write_text(json.dumps(summary,indent=2)+'\n')
    write_csv(OUT/'threshold_curves.csv',curves);write_csv(OUT/'count_by_type.csv',aggregates)
    write_csv(OUT/'per_answer_counts.csv',per_answer)
    fig,ax=plt.subplots(figsize=(8.8,4.5))
    labels={'LSML_CONT':'Continuous L-SML','IU_PCR':'IU-PCR'}
    colors={'LSML_CONT':'#246c9c','IU_PCR':'#278577'}
    for method in labels:
        rows=[r for r in curves if r['method']==method]
        ax.plot([100*r['actual_flag_rate'] for r in rows],[r['prm_score'] for r in rows],label=labels[method],color=colors[method])
        for rule,marker in [('ANSWER_Z_Q80','o'),('GLOBAL_THRESHOLD_ORACLE','*')]:
            r=summary['methods'][method][rule]
            ax.scatter(100*r['step_flag_rate_all'],r['prm_score'],color=colors[method],marker=marker,s=70,zorder=3)
    prm=float(references['Qwen2.5-Math-PRM-7B (supervised)','native 0.5']['overall'])
    ax.axhline(prm,color='#a75e70',ls='--',lw=1,label='PRM-7B native 0.5 · reported reference')
    ax.axvline(100*y.mean(),color='#777777',ls=':',lw=1,label='True error prevalence · label diagnostic')
    ax.set(xlabel='Flagged steps (% of all 94,203 steps, including controls)',ylabel='Pooled PRMScore (macro-F1 on 6,211 noncontrol answers)',
           title='N16 corrected + mask1: decision-threshold sensitivity',xlim=(0,50))
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8,loc='lower right')
    fig.text(.01,.01,'Source: frozen N16 score export · 6,969 answers. Circles: Q80; stars: label-selected global-cut oracle. Development diagnostics.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ['svg','png']:fig.savefig(OUT/f'threshold_curve.{ext}',dpi=180)
    plt.close(fig)
    print(json.dumps({m:summary['methods'][m]['oracle_diagnostic'] for m in METHODS},indent=2))

if __name__=='__main__': main()
