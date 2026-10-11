"""Reproducible v11 CPU controls on frozen score exports and corrected telemetry.

Run with main .venv/bin/python. --stage exports can run before raw telemetry arrives.
"""
from __future__ import annotations
import os
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
os.environ.setdefault('MPLCONFIGDIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mpl_cache'))
import argparse, ast, csv, hashlib, json, pickle, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from spectral_utils.advisor_controls import (answer_z, global_z, top_flags, metrics,
    pair_stats, repetition_risk, confusion_by_group, score_from_confusion,
    bootstrap_answer_mean, f1)

BANKS = ['N11','N13','N14','N16','N22','N25','N40','N43','N53','N56']
METHODS = ['LSML_CONT','IU_PCR','ALL_EQ','DS_FILTER_EQ','PRM_NATIVE']
CATEGORIES = {'simplicity':['redundency','circular'],
    'soundness':['counterfactual','step_contradiction','domain_inconsistency','confidence'],
    'sensitivity':['missing_condition','deception','multi_solutions']}

def dump(path, obj):
    def default(v):
        if isinstance(v, np.generic): return v.item()
        if isinstance(v, np.ndarray): return v.tolist()
        raise TypeError(str(type(v)))
    path.write_text(json.dumps(obj, indent=2, default=default, allow_nan=False)+'\n')

def clean(v):
    if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [clean(x) for x in v]
    if isinstance(v,np.ndarray): return clean(v.tolist())
    if isinstance(v,(float,np.floating)) and not np.isfinite(v): return None
    if isinstance(v,np.generic): return v.item()
    return v

def csvout(path, rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(clean(rows))

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''): h.update(chunk)
    return h.hexdigest()

class Data:
    def __init__(self, root):
        self.root=root;self.inputs=root/'inputs';self.out=root/'controls'
        self.z=np.load(self.inputs/'bank_N16.npz',allow_pickle=False)
        self.ids=self.z['idx'].astype(str);self.off=self.z['offsets'];self.n=np.diff(self.off)
        mm=json.loads((self.inputs/'prmbench_meta.json').read_text())['rows']
        meta={str(x['idx']):x for x in mm}
        assert len(meta)==len(mm)==len(self.ids)==6969 and set(meta)==set(self.ids)
        self.classes=np.array([meta[u]['classification'] for u in self.ids])
        self.answer=np.repeat(np.arange(len(self.ids)),self.n)
        self.y=np.concatenate([np.isin(np.arange(1,int(n)+1),meta[u]['error_steps']) for u,n in zip(self.ids,self.n)])
        assert self.off[-1]==len(self.y)==94203
        self.pool=self.classes[self.answer]!='correct'
        self.truth=np.add.reduceat(self.y.astype(int),self.off[:-1])
        gm=json.loads((self.inputs/'CANONICAL_GROUPS.json').read_text())['rows']
        groups={str(x['row_id']):x['canonical_group_id'] for x in gm}
        self.ug,self.ag=np.unique([groups[u] for u in self.ids],return_inverse=True)
        assert len(self.ug)==707
        self.sg=self.ag[self.answer]
        self.weights=np.random.default_rng(20261011).multinomial(707,np.full(707,1/707),size=5000).astype(float)
        pm=pickle.load((self.inputs/'prmbench_prm.pkl').open('rb'))
        records=list(pm.values()) if isinstance(pm,dict) else pm
        pm={str(x['idx']):x for x in records}
        assert len(pm)==len(records)==len(self.ids) and set(pm)==set(self.ids)
        for u,n in zip(self.ids,self.n):
            assert pm[u]['n_steps']==n and pm[u]['classification']==meta[u]['classification']
            assert pm[u]['error_steps']==meta[u]['error_steps']
        self.risk=1-np.concatenate([np.asarray(pm[u]['rewards'],float) for u in self.ids])
        self.native=self.risk>.5
        assert np.isfinite(self.risk).all()
        self.meta=meta;self.prm_records=pm

    def scores(self):
        sc={m:self.z['S__'+m] for m in METHODS if m!='PRM_NATIVE'}
        sc['PRM_NATIVE']=self.risk
        return sc

    def summaries(self, score, strata=None):
        auc=np.full(len(self.ids),np.nan);top=np.zeros(len(self.ids),int);pos=np.zeros(len(self.ids),int)
        pairs=np.zeros(len(self.ids),int);wins=np.zeros(len(self.ids))
        for i,(a,b) in enumerate(zip(self.off[:-1],self.off[1:])):
            st=pair_stats(score[a:b],self.y[a:b],None if strata is None else strata[a:b])
            auc[i],pairs[i],wins[i]=st['auc'],st['pairs'],st['wins']
            pos[i]=np.argmax(score[a:b]);top[i]=self.y[a+pos[i]]
        return dict(auc=auc,top1=top,position=pos,pairs=pairs,wins=wins)

    def summary_rows(self, method, values, bank='N16',variant='thinkfix_mask1',control='unrestricted'):
        rows=[]
        for kind in ['ALL_NONCONTROL']+sorted(set(self.classes)):
            select=self.classes!='correct' if kind=='ALL_NONCONTROL' else self.classes==kind
            aucmask=select&np.isfinite(values['auc']);errors=select&(self.truth>0)
            r=dict(bank=bank,method=method,variant=variant,control=control,classification=kind,
                n_answers=int(select.sum()),n_error_answers=int(errors.sum()),n_auc_answers=int(aucmask.sum()),
                within_auc=float(np.mean(values['auc'][aucmask])) if aucmask.any() else None,
                top1_on_error=float(np.mean(values['top1'][errors])) if errors.any() else None,
                top1_step0_rate=float(np.mean(values['position'][select]==0)) if select.any() else None,
                n_pairs=int(values['pairs'][select].sum()),
                pair_pooled_auc=float(values['wins'][select].sum()/values['pairs'][select].sum()) if values['pairs'][select].sum() else None)
            rows.append(r)
        return rows

def exports(d):
    rows=[];per={};status={}
    for bk in BANKS:
        path=d.inputs/f'bank_{bk}.npz'
        z=np.load(path,allow_pickle=False)
        assert np.array_equal(z['idx'],d.ids) and np.array_equal(z['offsets'],d.off)
        available=[k[3:] for k in z.files if k.startswith('S__')]
        for m in available:
            s=z['S__'+m];assert s.shape==d.y.shape and np.isfinite(s).all()
            vals=d.summaries(s);rows.extend(d.summary_rows(m,vals,bank=bk))
            per[f'{bk}__{m}__top1']=vals['top1'];per[f'{bk}__{m}__position']=vals['position'];per[f'{bk}__{m}__auc']=vals['auc']
        status[bk]=dict(n_defined_methods=len(available),methods=available,source_sha256=sha(path))
        print('exports',bk,len(available),flush=True)
    for m,s in [('PRM_NATIVE',d.risk),('STEP_INDEX',np.arange(len(d.y))-d.off[d.answer])]:
        vals=d.summaries(s);rows.extend(d.summary_rows(m,vals,bank='REFERENCE'))
    csvout(d.out/'ALL_BANK_TOP1_AUC.csv',rows)
    np.savez_compressed(d.out/'ALL_BANK_PER_ANSWER.npz',idx=d.ids,offsets=d.off,**per)
    dump(d.out/'ALL_BANK_STATUS.json',status)
    return rows

def bank_scores(d):
    rows=[]
    for bk in BANKS+['REFERENCE']:
        if bk=='REFERENCE':sources={'PRM_NATIVE':d.native}
        else:
            z=np.load(d.inputs/f'bank_{bk}.npz',allow_pickle=False)
            assert np.array_equal(z['idx'],d.ids) and np.array_equal(z['offsets'],d.off)
            sources={k[3:]:z[k]>=np.quantile(z[k],.8) for k in z.files if k.startswith('S__')}
        for m,flag in sources.items():
            subtype={};counts=np.add.reduceat(flag.astype(int),d.off[:-1])
            for kind in ['ALL_NONCONTROL']+sorted(set(d.classes)):
                ai=d.classes!='correct' if kind=='ALL_NONCONTROL' else d.classes==kind
                si=ai[d.answer];mm=metrics(d.y[si],flag[si]);value=mm['f1_valid'] if kind=='multi_solutions' else mm['prm_score']
                subtype[kind]=value
                rows.append(dict(bank=bk,method=m,rule='NATIVE_0_5' if m=='PRM_NATIVE' else 'ANSWER_Z_Q80',classification=kind,
                    n_answers=int(ai.sum()),n_steps=int(si.sum()),category_score=value,**mm,
                    flag_rate_all=float(flag.mean()),controls_false_flag_rate=float((counts[d.classes=='correct']>0).mean())))
            for category,types in CATEGORIES.items():
                rows.append(dict(bank=bk,method=m,rule='NATIVE_0_5' if m=='PRM_NATIVE' else 'ANSWER_Z_Q80',classification=category,
                    category_score=float(np.mean([subtype[c] for c in types])),
                    convention='equal subtype mean; multi_solutions = valid F1 alone'))
        print('Q80matrix',bk,flush=True)
    csvout(d.out/'ALL_BANK_Q80_PRMSCORE.csv',rows)
    return rows

def prefix(d):
    rows=[];per={};sc={}
    position=np.arange(len(d.y))-d.off[d.answer]
    pbin=np.floor(5*(position+.5)/d.n[d.answer]).astype(int)
    for variant in ['old','thinkfix','thinkfix_mask1']:
        path=d.inputs/('bank_N16.npz' if variant=='thinkfix_mask1' else f'bank_N16_{variant}.npz')
        z=np.load(path,allow_pickle=False)
        assert np.array_equal(z['idx'],d.ids) and np.array_equal(z['offsets'],d.off)
        for method in [k[3:] for k in z.files if k.startswith('S__')]:
            vals=d.summaries(z['S__'+method]);rows.extend(d.summary_rows(method,vals,variant=variant))
            per[f'{variant}__{method}__auc']=vals['auc'];per[f'{variant}__{method}__top1']=vals['top1'];per[f'{variant}__{method}__position']=vals['position']
            if method in METHODS:
                sc[variant,method]=z['S__'+method]
                matched=d.summaries(z['S__'+method],pbin)
                rows.extend(d.summary_rows(method,matched,variant=variant,control='position_only'))
                per[f'{variant}__{method}__position_auc']=matched['auc']
                # Contributions of first five positions: pair outcomes partition the full pair-AUC numerator.
                totals=np.zeros((6,2))
                auc_without0=[];without=np.full(len(d.ids),np.nan)
                for i,(a,b) in enumerate(zip(d.off[:-1],d.off[1:])):
                    y=d.y[a:b];s=z['S__'+method][a:b]
                    if y.all() or not y.any():continue
                    bad=np.flatnonzero(y);good=np.flatnonzero(~y);dif=s[bad,None]-s[good][None,:]
                    weight=1/(len(bad)*len(good))
                    bins=np.minimum(np.minimum(bad[:,None],good[None,:]),5)
                    for k in range(6):
                        selected=bins==k
                        totals[k,0]+=weight*(np.count_nonzero(dif[selected]>0)+.5*np.count_nonzero(dif[selected]==0))
                        totals[k,1]+=weight*selected.sum()
                    v=pair_stats(s[1:],y[1:])['auc']
                    without[i]=v
                    if np.isfinite(v):auc_without0.append(v)
                per[f'{variant}__{method}__without0_auc']=without
                for k in range(6):
                    rows.append(dict(bank='N16',method=method,variant=variant,control='pair_earliest_position_contribution',classification='ALL_NONCONTROL',
                        earliest_pair_position=k if k<5 else '5+',weighted_pair_win_sum=totals[k,0],weighted_pair_mass_sum=totals[k,1],
                        pair_contribution_auc=totals[k,0]/totals[k,1] if totals[k,1] else None,
                        without_step0_auc=float(np.mean(auc_without0)),without_step0_n=len(auc_without0)))
        print('prefix',variant,'analysed',flush=True)
    contrasts=[]
    for m in ['LSML_CONT','IU_PCR','ALL_EQ','DS_FILTER_EQ']:
        for new,old in [('thinkfix','old'),('thinkfix_mask1','thinkfix'),('thinkfix_mask1','old')]:
            for metric in ['auc','top1','position_auc','without0_auc']:
                left=per[f'{new}__{m}__{metric}'];right=per[f'{old}__{m}__{metric}']
                sel=(d.classes!='correct')&(np.isfinite(left))&(np.isfinite(right))
                if metric=='top1':sel&=d.truth>0
                boot=bootstrap_answer_mean(left-right,d.ag,sel,d.weights)
                lo,hi=np.nanquantile(boot,[.025,.975])
                contrasts.append(dict(method=m,new=new,old=old,metric=metric,n_answers=int(sel.sum()),difference=float(np.mean((left-right)[sel])),ci_low=float(lo),ci_high=float(hi)))
    csvout(d.out/'PREFIX_MASK_ATTRIBUTION.csv',rows);csvout(d.out/'PREFIX_MASK_CONTRASTS.csv',contrasts)
    np.savez_compressed(d.out/'PREFIX_PER_ANSWER.npz',idx=d.ids,**per)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,metric in zip(axes,['within_auc','top1_on_error']):
        for m in ['LSML_CONT','IU_PCR','ALL_EQ','DS_FILTER_EQ']:
            rr=[next(r for r in rows if r['method']==m and r['variant']==v and r['classification']=='ALL_NONCONTROL' and r['control']=='unrestricted') for v in ['old','thinkfix','thinkfix_mask1']]
            ax.plot(range(3),[r[metric] for r in rr],marker='o',label=m)
        ax.set_xticks(range(3),['Old','Prefix fixed','Prefix + mask1']);ax.set_ylabel(metric);ax.grid(alpha=.2)
    axes[1].legend(fontsize=8);fig.suptitle('Input ablation on the same N16 bank and answer population')
    fig.tight_layout()
    for ext in ['svg','png']:fig.savefig(d.out/f'prefix_mask_ablation.{ext}',dpi=180)
    plt.close(fig)

def telemetry(d):
    path=d.inputs/'prmbench_telemetry.pkl';cache=d.out/'NUISANCE_INPUTS.npz'
    schema_path=d.out/'TELEMETRY_SCHEMA.json'
    if cache.exists() and schema_path.exists():
        schema=json.loads(schema_path.read_text())
        current_identity={'sha256':sha(path),'spans_sha256':sha(d.inputs/'step_spans_thinkfix_mask1.npz'),
                          'metadata_sha256':sha(d.inputs/'prmbench_meta.json')}
        if all(schema.get(k)==v for k,v in current_identity.items()):
            z=np.load(cache);assert np.array_equal(z['idx'],d.ids)
            return {k:z[k] for k in z.files}
        print('nuisance cache identity changed or lacks hashes; rebuilding from authenticated inputs',flush=True)
    print('loading corrected telemetry',path,flush=True)
    obj=pickle.load(path.open('rb'))
    records=list(obj.values()) if isinstance(obj,dict) else obj
    mapping={str(x['idx']):x for x in records}
    assert len(mapping)==len(records)==len(d.ids) and set(mapping)==set(d.ids)
    print('telemetry schema',list(records[0]),flush=True)
    spans=np.load(d.inputs/'step_spans_thinkfix_mask1.npz',allow_pickle=False)
    spanarr=spans['step_spans']
    if len(spanarr)!=len(d.y):
        pop=np.load(d.inputs/'population.npz',allow_pickle=False)
        spanarr=spanarr[pop['feature_rows']]
    assert spanarr.shape==(len(d.y),2)
    length=spanarr[:,1]-spanarr[:,0]
    entropy=np.zeros(len(d.ids));repeat=np.zeros(len(d.y));examples=[]
    for i,u in enumerate(d.ids):
        row=mapping[u]
        h=np.asarray(row['token_entropies'],float);assert h.ndim==1 and np.isfinite(h).all()
        entropy[i]=h.mean() if len(h) else 0.
        steps=row.get('steps',row.get('step_texts'))
        if steps is None:
            # Original dedicated PRM cache carries exact dataset steps; no tokenizer/model is needed.
            steps=d.prm_records[u].get('steps',d.prm_records[u].get('step_texts'))
        if steps is None:raise ValueError(f'Step texts unavailable; schema {list(row)}, PRM {list(d.prm_records[u])}')
        if isinstance(steps,dict):steps=list(steps.values())
        assert len(steps)==d.n[i],u
        a,b=d.off[i:i+2];repeat[a:b]=repetition_risk(steps)
        if d.classes[i] in ['redundency','circular'] and len([r for r in examples if r['classification']==d.classes[i]])<3:
            examples.append(dict(idx=u,classification=str(d.classes[i]),question=row.get('question',d.prm_records[u].get('question')),
                steps=[str(t) for t in steps],error_steps=d.meta[u]['error_steps'],repetition=repeat[a:b].tolist(),
                token_length=length[a:b].tolist(),example_selection='first three by immutable bank order; not selected by model success'))
    assert np.all(length>=0) and np.isfinite(entropy).all()
    arr=dict(idx=d.ids,length=length,repetition=repeat,answer_entropy=entropy)
    np.savez_compressed(cache,**arr)
    dump(d.out/'DATASET_EXAMPLES.json',examples)
    dump(d.out/'TELEMETRY_SCHEMA.json',dict(path=str(path),sha256=sha(path),record_keys=list(records[0]),
        spans_sha256=sha(d.inputs/'step_spans_thinkfix_mask1.npz'),metadata_sha256=sha(d.inputs/'prmbench_meta.json'),
        n_records=len(records),step_text_source='telemetry steps if present; else aligned PRM cache steps',
        length_source='corrected mask1 half-open token spans',entropy_source='mean over corrected mask1 token_entropies per answer'))
    del obj,records,mapping
    return arr

def controls(d,nu):
    position=np.arange(len(d.y))-d.off[d.answer]
    relative=(position+.5)/d.n[d.answer]
    pbin=np.floor(5*relative).astype(int)
    cuts=np.quantile(nu['length'],[1/3,2/3]);lbin=np.searchsorted(cuts,nu['length'],side='right')
    rcuts=np.quantile(nu['repetition'],[1/3,2/3]);rbin=np.searchsorted(rcuts,nu['repetition'],side='right')
    strata={'unrestricted':None,'position_only':pbin,'length_only':lbin,'repetition_only':rbin,
            'position_length':3*pbin+lbin,'position_length_repetition':9*pbin+3*lbin+rbin}
    scores={**d.scores(),'STEP_INDEX':position.astype(float),'TOKEN_LENGTH':nu['length'].astype(float),'TEXT_REPETITION':nu['repetition']}
    rows=[];per={};contrasts=[]
    for name,bins in strata.items():
        values={m:d.summaries(s,bins) for m,s in scores.items()}
        for m,v in values.items():
            rows.extend(d.summary_rows(m,v,control=name));per[f'{name}__{m}__auc']=v['auc'];per[f'{name}__{m}__pairs']=v['pairs']
        for kind in ['ALL_NONCONTROL']+sorted(set(d.classes)-{'correct'}):
            selected=d.classes!='correct' if kind=='ALL_NONCONTROL' else d.classes==kind
            matched=selected&np.isfinite(values['LSML_CONT']['auc'])
            for m in METHODS:
                a=values[m]['auc'];original=d.summaries(scores[m])['auc']
                boot=bootstrap_answer_mean(a,d.ag,matched,d.weights)
                lo,hi=np.nanquantile(boot,[.025,.975]) if matched.any() else [np.nan,np.nan]
                rows.append(dict(bank='N16',method=m,variant='thinkfix_mask1',control=name+'_same_eligible',classification=kind,
                    n_answers=int(selected.sum()),n_auc_answers=int(matched.sum()),
                    within_auc=float(np.mean(a[matched])) if matched.any() else None,
                    unrestricted_same_eligible_auc=float(np.mean(original[matched])) if matched.any() else None,
                    ci_low=float(lo),ci_high=float(hi)))
            for left,right in [('LSML_CONT','ALL_EQ'),('IU_PCR','ALL_EQ'),('LSML_CONT','DS_FILTER_EQ'),('LSML_CONT','PRM_NATIVE'),('IU_PCR','PRM_NATIVE')]:
                difference=values[left]['auc']-values[right]['auc'];sel=selected&np.isfinite(difference)
                boot=bootstrap_answer_mean(difference,d.ag,sel,d.weights)
                lo,hi=np.nanquantile(boot,[.025,.975]) if sel.any() else [np.nan,np.nan]
                contrasts.append(dict(control=name,classification=kind,left=left,right=right,metric='within_answer_matched_auc',
                    n_answers=int(sel.sum()),difference=float(np.mean(difference[sel])) if sel.any() else None,ci_low=float(lo),ci_high=float(hi)))
        print('controls',name,flush=True)
    csvout(d.out/'NUISANCE_CONTROLS.csv',rows);csvout(d.out/'PAIRED_AUC_CONTRASTS.csv',contrasts)
    np.savez_compressed(d.out/'CONTROL_PER_ANSWER.npz',idx=d.ids,**per)
    dump(d.out/'CONTROL_PROTOCOL.json',dict(position_bins=5,length_cuts=cuts,repetition_cuts=rcuts,
        length_bins='global label-free tertiles; tied boundaries retained',repetition_bins='global label-free tertiles; tied boundaries retained, can be empty',
        repetition_definition='casefold unicode word 3-gram max Jaccard to earlier steps; short-pair unigram fallback; first zero',
        primary='position_length',supplementary='position_length_repetition',nuisance_fit='label-free global development population',
        metric='pool matched error-valid pairs within answer then equal-answer mean; same-eligible unrestricted reported',
        bootstrap='5000 canonical-source-group resamples seed20261011 pointwise95percentile; no selection/multiplicity adjustment',
        limits=['coarse bins leave residual nuisance variation','matched eligibility and pair weights differ from unrestricted AUC','repetition risk can itself encode the error construct']))
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharey=True)
    for ax,kind in zip(axes,['ALL_NONCONTROL','redundency','circular']):
        names=['unrestricted','position_only','position_length','position_length_repetition']
        for m in ['LSML_CONT','IU_PCR','ALL_EQ','PRM_NATIVE']:
            rr=[next(r for r in rows if r['classification']==kind and r['method']==m and r['control']==n+'_same_eligible') for n in names]
            ax.plot(range(4),[r['within_auc'] for r in rr],marker='o',label=m)
        ax.set_title(kind);ax.set_xticks(range(4),['Raw','Position','Pos+length','+ repetition'],rotation=25);ax.grid(alpha=.2)
    axes[0].set_ylabel('Matched-pair within-answer AUC');axes[-1].legend(fontsize=8)
    fig.suptitle('Development nuisance controls · each point reports its own matched population')
    fig.tight_layout()
    for ext in ['svg','png']:fig.savefig(d.out/f'nuisance_controls.{ext}',dpi=180)
    plt.close(fig)

def decisions(d,nu):
    # Canonical registered decisions are loaded verbatim by parent into sources.
    source=d.root/'sources'/'research/results/monday_fusion_analysis_v1/code/spectral_utils/monday_fusion_decision.py'
    if not source.exists():raise FileNotFoundError('Need pinned monday_fusion_decision.py for exact registered rules')
    metric_source=source.with_name('monday_fusion_metrics.py')
    # Execute the three pinned pure functions verbatim, excluding unrelated legacy
    # Windows-path imports. This is an adapter, not a rewrite of registered rules.
    env={'np':np}
    for path,nameset in [(metric_source,{'answer_z','top_flags'}),(source,{'rules'})]:
        tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in nameset]
        assert {n.name for n in nodes}==nameset
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
    bv=np.load(d.inputs/'bank_V_N16.npz',allow_pickle=False)
    names=list(bv['columns'].astype(str));surv=list(d.z['G_survivors'].astype(str))
    raw=bv['raw'][:,[names.index(c) for c in surv]]
    reconstructed=global_z(raw).mean(axis=1)
    delta=float(np.max(np.abs(reconstructed-d.z['G'])))
    assert delta<1e-9,('R2 raw survivor parity failure',delta)
    bank={'off':d.off,'answers':np.arange(len(d.ids))}
    allflags={};rows=[];counts=[];details={};per={}
    for m,s in d.scores().items():
        if m=='PRM_NATIVE':
            flags={'NATIVE_0_5':d.native};detail={'valid_tie_0_5':'valid'}
        else:
            flags,_,detail=env['rules'](s,bank,raw,nu['answer_entropy'],d.ids)
            flags={k:flags[k] for k in ['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80']}
            for k in [1,2,3]:flags[f'TOP_{k}_DIAGNOSTIC']=top_flags(s,d.off,np.minimum(k,d.n))
            assert np.array_equal(flags['ANSWER_Z_Q80'],s>=np.quantile(s,.8))
        details[m]=detail
        for rule,flag in flags.items():
            flag=np.asarray(flag,bool);assert flag.shape==d.y.shape
            allflags[m+'__'+rule]=flag;count=np.add.reduceat(flag.astype(int),d.off[:-1])
            per[m+'__'+rule]=count
            subtype={}
            for kind in ['ALL_NONCONTROL']+sorted(set(d.classes)):
                ai=d.classes!='correct' if kind=='ALL_NONCONTROL' else d.classes==kind;si=ai[d.answer]
                mm=metrics(d.y[si],flag[si]);score=mm['f1_valid'] if kind=='multi_solutions' else mm['prm_score']
                subtype[kind]=score
                row=dict(method=m,rule=rule,classification=kind,n_answers=int(ai.sum()),n_steps=int(si.sum()),
                    category_score=score,**mm,step_flag_rate=float(flag[si].mean()),
                    answer_flag_rate=float((count[ai]>0).mean()),true_count_mean=float(d.truth[ai].mean()),
                    flagged_count_mean=float(count[ai].mean()),count_bias=float((count[ai]-d.truth[ai]).mean()),
                    count_mae=float(np.abs(count[ai]-d.truth[ai]).mean()))
                if kind=='ALL_NONCONTROL':
                    row['controls_false_flag_rate']=float((count[d.classes=='correct']>0).mean())
                    row['all_errorfree_false_flag_rate']=float((count[d.truth==0]>0).mean())
                rows.append(row)
            for category,types in CATEGORIES.items():
                rows.append(dict(method=m,rule=rule,classification=category,category_score=float(np.mean([subtype[c] for c in types])),
                    convention='equal mean of subtype scores; MS valid F1 alone'))
            for i,u in enumerate(d.ids):counts.append(dict(method=m,rule=rule,idx=u,group=d.ug[d.ag[i]],classification=d.classes[i],
                n_steps=d.n[i],true_count=d.truth[i],flagged_count=count[i]))
    csvout(d.out/'EXACT_DECISION_RULES.csv',rows);csvout(d.out/'PER_ANSWER_COUNTS.csv',counts)
    np.savez_compressed(d.out/'DECISION_FLAGS.npz',idx=d.ids,offsets=d.off,**allflags)
    np.savez_compressed(d.out/'DECISION_COUNTS.npz',idx=d.ids,**per)
    dump(d.out/'EXACT_RULE_DIAGNOSTICS.json',clean(dict(canonical_rule_module_sha256=sha(source),canonical_metrics_module_sha256=sha(metric_source),
        adapter='AST extraction executes pinned rules/answer_z/top_flags verbatim; excludes unrelated legacy filesystem imports',raw_survivors=surv,
        raw_survivor_G_parity=delta,corrected_entropy=True,rules=details,
        warning='Development replay of registered rules; R2 thresholds global raw survivor z; entropy reads corrected mask1 telemetry')))
    paired=[]
    for rule in ['ANSWER_Z_Q80','R2_COUNT','ENTROPY_GATE_Q80']:
        for kind in ['ALL_NONCONTROL','redundency','circular']:
            selected=d.pool if kind=='ALL_NONCONTROL' else d.classes[d.answer]==kind
            point={};boot={}
            for m in METHODS:
                fl=allflags[m+'__'+('NATIVE_0_5' if m=='PRM_NATIVE' else rule)]
                c=confusion_by_group(d.y,fl,d.sg,selected,707)
                point[m]=metrics(d.y[selected],fl[selected])['prm_score'];boot[m]=score_from_confusion(c,d.weights)
                explicit=d.weights[0,d.sg[selected]].astype(int)
                assert abs(boot[m][0]-metrics(np.repeat(d.y[selected],explicit),np.repeat(fl[selected],explicit))['prm_score'])<1e-12
            for left,right in [('LSML_CONT','ALL_EQ'),('IU_PCR','ALL_EQ'),('LSML_CONT','DS_FILTER_EQ'),('LSML_CONT','PRM_NATIVE'),('IU_PCR','PRM_NATIVE')]:
                lo,hi=np.quantile(boot[left]-boot[right],[.025,.975]);paired.append(dict(rule=rule,classification=kind,left=left,right=right,
                    metric='pooled_macro_f1',difference=point[left]-point[right],ci_low=float(lo),ci_high=float(hi)))
    csvout(d.out/'PAIRED_PRMSCORE_CONTRASTS.csv',paired)
    return rows

def budgets(d):
    """Fixed-score budget curves and labelled diagnostic ceilings; no promotion."""
    rows=[];oracle={}
    for m,s in d.scores().items():
        for rate in np.linspace(0,.5,501):
            threshold=np.inf if rate==0 else np.quantile(s,1-rate)
            flag=s>=threshold
            count=np.add.reduceat(flag.astype(int),d.off[:-1])
            rows.append(dict(method=m,nominal_budget=rate,actual_budget=flag.mean(),
                threshold=float(threshold) if np.isfinite(threshold) else None,
                controls_false_flag_rate=float((count[d.classes=='correct']>0).mean()),
                **metrics(d.y[d.pool],flag[d.pool])))
        # Exact oracle includes no-rejection cut and all tie-preserving thresholds.
        order=np.argsort(-s,kind='stable'); selected=d.pool[order]
        tp=np.cumsum(d.y[order]&selected);fp=np.cumsum(~d.y[order]&selected)
        stops=np.r_[np.flatnonzero(s[order][:-1]!=s[order][1:]),len(s)-1]
        p=int(d.y[d.pool].sum());v=int((~d.y[d.pool]).sum())
        vals=(f1(tp[stops],fp[stops],p-tp[stops])+f1(v-fp[stops],fp[stops],p-tp[stops]))/2
        none=metrics(d.y[d.pool],np.zeros(int(d.pool.sum()),bool))['prm_score']
        best=int(np.argmax(vals));cut=s[order[stops[best]]] if vals[best]>=none else np.inf
        cutflag=s>=cut
        frozen=d.native if m=='PRM_NATIVE' else s>=np.quantile(s,.8)
        counts=np.add.reduceat(frozen.astype(int),d.off[:-1])
        truequota=top_flags(s,d.off,d.truth)
        perfect=top_flags(d.y.astype(float),d.off,counts)
        oracle[m]=dict(frozen=metrics(d.y[d.pool],frozen[d.pool]),
            global_threshold=metrics(d.y[d.pool],cutflag[d.pool]),
            true_count=metrics(d.y[d.pool],truequota[d.pool]),
            perfect_ranking_fixed_counts=metrics(d.y[d.pool],perfect[d.pool]),
            global_threshold_cut=float(cut) if np.isfinite(cut) else None,
            global_threshold_flag_rate=float(cutflag.mean()),
            limits='All three ceilings use evaluation labels. They intervene on different components and their gains are not additive.')
        assert abs(oracle[m]['global_threshold']['prm_score']-max(none,float(vals.max())))<1e-12
    csvout(d.out/'BUDGET_CURVES.csv',rows)
    dump(d.out/'DIAGNOSTIC_CEILINGS.json',oracle)
    fig,ax=plt.subplots(figsize=(9,4.5))
    colors={'LSML_CONT':'#246c9c','IU_PCR':'#278577','PRM_NATIVE':'#a75e70'}
    for m in colors:
        rr=[r for r in rows if r['method']==m]
        ax.plot([100*r['actual_budget'] for r in rr],[r['prm_score'] for r in rr],label=m,color=colors[m])
        s=d.scores()[m];fl=d.native if m=='PRM_NATIVE' else s>=np.quantile(s,.8)
        ax.scatter(100*fl.mean(),oracle[m]['frozen']['prm_score'],color=colors[m],s=45)
        ax.scatter(100*oracle[m]['global_threshold_flag_rate'],oracle[m]['global_threshold']['prm_score'],marker='*',s=90,color=colors[m])
    ax.axvline(100*d.y.mean(),color='#777',ls=':',lw=1,label='True prevalence (label diagnostic)')
    ax.set(xlim=(0,50),xlabel='Flagged steps (% of all steps, including controls)',ylabel='Pooled noncontrol PRMScore',
        title='Ranking and decision both constrain performance')
    ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.text(.02,.01,'Circles: frozen Q80 / PRM native. Stars: label-selected global threshold; development oracle, not a deployable gain.',fontsize=8)
    fig.tight_layout(rect=(0,.04,1,1))
    for ext in ['svg','png']:fig.savefig(d.out/f'budget_curves.{ext}',dpi=180)
    plt.close(fig)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parent.parent)
    ap.add_argument('--stage',choices=['exports','bank_scores','prefix','telemetry','controls','decisions','budgets','all'],default='all');a=ap.parse_args()
    d=Data(a.root);d.out.mkdir(parents=True,exist_ok=True)
    start=time.time();status={'stage':a.stage,'n_answers':len(d.ids),'n_steps':len(d.y),'n_source_groups':707,'started':time.strftime('%Y-%m-%dT%H:%M:%S')}
    dump(d.out/f'STATUS_{a.stage}.json',status)
    if a.stage in ['exports','all']:exports(d)
    if a.stage in ['bank_scores','all']:bank_scores(d)
    if a.stage in ['prefix','all']:prefix(d)
    if a.stage in ['budgets','all']:budgets(d)
    if a.stage in ['telemetry','controls','decisions','all']:
        nu=telemetry(d)
        if a.stage in ['controls','all']:controls(d,nu)
        if a.stage in ['decisions','all']:decisions(d,nu)
    status.update(status='PASS',elapsed_s=time.time()-start,finished=time.strftime('%Y-%m-%dT%H:%M:%S'),script_sha256=sha(Path(__file__)))
    dump(d.out/f'STATUS_{a.stage}.json',status);print(status,flush=True)

if __name__=='__main__':main()
