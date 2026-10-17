"""Independent CPU replay from downloaded Drive step arrays, never slide totals.

The archive's channel atlas is float32 and label-oriented. Its documented signs
are undone before fitting; selected original channels all have sign +1. This
checks scores/metrics, not token feature extraction or untouched generalization.
Two larger banks are fixed here before computing their metrics. They are
development diagnostics selected from previously inspected feature families.
"""
from __future__ import annotations

import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '2'
from pathlib import Path
import csv
import hashlib
import json
import pickle
import sys
import time
import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.fusion_utils import lsml_continuous, sml_fuse_signed

RAW = HERE / 'inputs/unpacked/results'
S3 = ['energy_level', 'hist_entropy_cusum_abs_series', 'realized_drv']
BANKS = {'S3': S3,
         'S5': S3 + ['ct7_H0lim_prefix_innovation', 'digit_spread'],
         'S6': S3 + ['ct7_H0lim_prefix_innovation', 'digit_spread', 'hist_topk_varentropy_series']}
SEED = 20261013
DRAWS = 2000


def write_json(name, obj):
    (HERE / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if isinstance(x, np.generic) else x.tolist())+'\n')


def write_csv(name, rows):
    with (HERE / name).open('w', newline='') as f:
        w = csv.DictWriter(f, list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader(); w.writerows(rows)


def answer_z(score, offsets):
    out = np.zeros_like(score, dtype=float)
    for a, b in zip(offsets[:-1], offsets[1:]):
        s = score[a:b]
        out[a:b] = (s - s.mean()) / max(s.std(), 1e-8)
    return out


def q80(score, offsets, inclusive=False):
    z = answer_z(score, offsets)
    cut = float(np.quantile(z, .8))
    return (z >= cut if inclusive else z > cut), cut


def auc(err, risk):
    n1 = int(err.sum()); n0 = len(err)-n1
    if not n0 or not n1:
        return np.nan
    return float((rankdata(risk)[err].sum()-n1*(n1+1)/2)/(n1*n0))


def metric(counts, benchmark='prmbench'):
    # [true error predicted error, true valid predicted error,
    #  true valid predicted valid, true error predicted valid].
    tp, fp, tn, fn = np.moveaxis(np.asarray(counts, float), -1, 0)
    def div(a,b):
        return np.divide(a,b,out=np.zeros_like(np.asarray(a,float)),where=b!=0)
    if benchmark == 'hard2verify':
        re, rv = div(tp,tp+fn), div(tn,tn+fp)
        return div(2*re*rv,re+rv)
    return .5*(div(2*tp,2*tp+fp+fn)+div(2*tn,2*tn+fp+fn))


def measurements(score, flags, y, offsets, eligible_steps=None):
    mask = np.ones(len(y), bool) if eligible_steps is None else eligible_steps
    counts = np.zeros((len(offsets)-1,4),np.int64)
    au = np.full(len(counts), np.nan)
    for j,(a,b) in enumerate(zip(offsets[:-1],offsets[1:])):
        yy, ff, mm = y[a:b], flags[a:b], mask[a:b]
        counts[j] = [(yy&ff&mm).sum(), (~yy&ff&mm).sum(),
                     (~yy&~ff&mm).sum(), (yy&~ff&mm).sum()]
        au[j] = auc(yy[mm],score[a:b][mm])
    return counts, au


def orient(s, anchor):
    return -s if np.corrcoef(s,anchor)[0,1]<0 else s


def equal_cluster_score(V, groups, l2=True):
    pieces=[]
    for g in np.unique(groups):
        x=V[:,np.flatnonzero(groups==g)]
        pieces.append(x.sum(1)/(np.sqrt(x.shape[1]) if l2 else x.shape[1]))
    return np.column_stack(pieces).sum(1)/(np.sqrt(len(pieces)) if l2 else len(pieces))


def primary_replay():
    folder=RAW/'prm_boosting_v1'
    with np.load(folder/'CHANNEL_CONDCORR_prm_backbone.npz',allow_pickle=True) as z:
        names=z['names'].astype(str).tolist(); X=z['S'].astype(float)
        y=z['y'].astype(bool); offsets=z['offsets']; ids=z['idx'].astype(str)
    signs={r['channel']:float(r['sign']) for r in csv.DictReader((folder/'CHANNEL_ATLAS_prm_backbone.csv').open())}
    X *= np.array([signs[c] for c in names])
    with np.load(folder/'SCORES_prm_backbone.npz',allow_pickle=True) as z:
        assert np.array_equal(z['idx'].astype(str),ids) and np.array_equal(z['offsets'],offsets)
        prm=z['prm_risk']
    with np.load(ROOT/'results/advisor_completion_2026-10-11/inputs/population.npz') as z:
        assert np.array_equal(z['idx'],ids) and np.array_equal(z['offsets'],offsets)
        assert np.array_equal(z['y'],y)
        cls=z['classification']; groups=z['group']
    meta=json.loads((ROOT/'results/advisor_completion_2026-10-11/inputs/prmbench_meta.json').read_text())['rows']
    lookup={r['idx']:r for r in meta}; reconstructed=np.zeros_like(y)
    for j,i in enumerate(ids):
        a,b=offsets[j:j+2]
        reconstructed[a:b]=np.isin(np.arange(1,b-a+1),lookup[i]['error_steps'])
    assert np.array_equal(reconstructed,y)
    write_json('FROZEN_BANKS.json',{'banks':BANKS,'selection':'Previously inspected, label-using development candidates; no search for a winning set. All from the same PRM checkpoint.',
                                  'same_clustering_control':'replace unit-L2 eigenvectors by positive unit-L2 equal weights at both levels; keep the fitted partition fixed'})
    scores={'PRM_native':prm,'PRM_Q80':prm}; diagnostics={}; fits={}
    for bank, channels in BANKS.items():
        features=X[:,[names.index(c) for c in channels]]
        for head in [False,True]:
            label=bank+('_HEAD' if head else '_ONLY')
            V=np.column_stack([answer_z(prm,offsets),features]) if head else features
            fit,d=lsml_continuous(*V.T)
            anchor=V[:,0] if head else V.mean(1)
            scores[label+'_LSML']=orient(fit,anchor)
            scores[label+'_CLUSTER_EQUAL_L2']=orient(equal_cluster_score(V,d['c'],True),anchor)
            scores[label+'_CLUSTER_MEANS']=orient(equal_cluster_score(V,d['c'],False),anchor)
            scores[label+'_FLAT_MEAN']=V.mean(1)
            flat,_=sml_fuse_signed(*V.T);scores[label+'_FLAT_SML']=orient(flat,anchor)
            w=np.linalg.lstsq(np.column_stack([V,np.ones(len(V))]),scores[label+'_LSML'],rcond=None)[0][:-1]
            diagnostics[label]={'names':(['prm_risk'] if head else [])+channels,'K':d['K'],'groups':d['c'],
                                'group_weights':[w.tolist() for _,w in d['group_weights']], 'cross_weights':d['cross_weights'],
                                'weights_l1':w/np.abs(w).sum(),'degenerate':d['degenerate'],
                                'max_abs_lsml_minus_cluster_equal_l2':float(np.max(np.abs(scores[label+'_LSML']-scores[label+'_CLUSTER_EQUAL_L2'])))}
            fits[label]=(V,d)
        if bank=='S3':
            for j,c in enumerate(channels): scores['CH_'+c]=features[:,j]
            V=np.column_stack([answer_z(prm,offsets),features])
            scores['FIXED_HALF']=.5*V[:,0]+.5*V[:,1:].mean(1)
    # Evaluate only after all fitting/control predictions have been fixed.
    include_answer=cls!='correct'; sums={}; rows=[]; bytype=[]
    unique,inv=np.unique(groups,return_inverse=True)
    rng=np.random.default_rng(SEED)
    W=rng.multinomial(len(unique),np.full(len(unique),1/len(unique)),size=DRAWS)
    for name,s in scores.items():
        flags,cut=(prm>=0,0.) if name=='PRM_native' else q80(s,offsets)
        c,au=measurements(s,flags,y,offsets)
        c[~include_answer]=0
        rows.append({'arm':name,'prmscore':100*float(metric(c.sum(0))), 'within_auc':float(np.nanmean(au)),
                     'n_auc':int(np.isfinite(au).sum()),'n_answers':len(ids),'n_official_answers':int(include_answer.sum()),
                     'flag_rate_all_steps':float(flags.mean()),'threshold_on_answer_z':cut})
        for cl in sorted(set(cls)-{'correct'}):
            cc=c[cls==cl].sum(0)
            if cl=='multi_solutions':
                value=100*2*cc[2]/(2*cc[2]+cc[1]+cc[3])
                kind='valid_step_f1'
            else:
                value=100*float(metric(cc));kind='macro_f1_valid_error'
            bytype.append({'arm':name,'classification':cl,'category_score':value,'metric_definition':kind})
        G=np.zeros((len(unique),6))
        np.add.at(G,inv,np.column_stack([c,np.where(np.isfinite(au),au,0),np.isfinite(au)]))
        boot=W@G
        sums[name]=(100*metric(boot[:,:4]),boot[:,4]/boot[:,5])
        print(name,round(rows[-1]['prmscore'],5),round(rows[-1]['within_auc'],7),flush=True)
    rowmap={r['arm']:r for r in rows}; cons=[]
    wanted=[('S3_HEAD_LSML','PRM_native'),('S3_HEAD_LSML','PRM_Q80'),('S3_HEAD_LSML','S3_ONLY_LSML'),
            ('S3_HEAD_LSML','S3_HEAD_CLUSTER_EQUAL_L2'),('S3_HEAD_LSML','S3_HEAD_FLAT_MEAN'),('S3_HEAD_LSML','FIXED_HALF')]
    for bank in BANKS:
        for part in ['ONLY','HEAD']:
            a=bank+'_'+part+'_LSML'
            wanted += [(a,bank+'_'+part+'_CLUSTER_EQUAL_L2'),(a,bank+'_'+part+'_FLAT_MEAN')]
        if bank!='S3':wanted += [(bank+'_HEAD_LSML','S3_HEAD_LSML')]
    for a,b in dict.fromkeys(wanted):
        for j,mt in enumerate(['prmscore','within_auc']):
            diff=sums[a][j]-sums[b][j]
            lo,hi=np.quantile(diff,[.025,.975])
            cons.append({'contrast':a+' - '+b,'metric':mt,'delta':rowmap[a][mt]-rowmap[b][mt],'ci_low':lo,'ci_high':hi,'draws':DRAWS,'source_groups':len(unique),'pointwise':True})
    write_csv('PRM7B_METRICS.csv',rows);write_csv('PRM7B_CONTRASTS.csv',cons);write_csv('PRM7B_BY_TYPE.csv',bytype)
    write_json('FUSION_DIAGNOSTICS.json',diagnostics)
    reported=list(csv.DictReader(Path('/tmp/advisor_remote_review_20261015/external/results/advisor_page_v19_2026-10-15/sources/eval_slide12/SLIDE12_RESULTS_qwen_prm_7b.csv').open()))
    mapping={'LSML_CONT':'S3_HEAD_LSML','LSML3':'S3_ONLY_LSML','EQ4':'S3_HEAD_FLAT_MEAN','S3':'FIXED_HALF','SML4':'S3_HEAD_FLAT_SML'}
    checks=[]
    for r in reported:
        key=mapping.get(r['arm'],r['arm'])
        if key in rowmap:
            checks.append({'arm':r['arm'],**{mt+'_difference':rowmap[key][mt]-float(r[mt]) for mt in ['prmscore','within_auc']}})
    write_json('REPORTED_VS_RECOMPUTED.json',{'input_precision':'Float32 exported step channels; PRM risk is float64. Not a feature-extraction replay.',
                                           'label_parity':True,'identity_parity':True,'comparisons':checks})


def external_replay():
    results=[]
    for bench in ['socratic','hard2verify']:
        panel=RAW/'monday_external_panel_v1'/f'{bench}_qwen3_8b/PREDICTIONS.npz'
        gold={g['uid']:g for g in json.loads((HERE/'inputs'/f'{bench}_gold.json').read_text())}
        with np.load(panel) as z:
            uids=z['uids'].astype(str); off=z['starts']; ne=z['nonempty'].astype(bool)
        assert set(uids)==set(gold)
        y=np.concatenate([~np.asarray(gold[u]['correct'],bool) for u in uids])
        include=np.concatenate([np.asarray(gold[u]['include'],bool) for u in uids])
        assert len(y)==off[-1] and all(len(gold[u]['correct'])==b-a for u,a,b in zip(uids,off[:-1],off[1:]))
        cache=pickle.load((RAW/'external_lr_prm_v1/prm_raw'/bench/'external_prm.pkl').open('rb'))
        assert set(cache)==set(uids)
        reward=np.concatenate([np.asarray(cache[u]['rewards'],float) for u in uids]); risk=1-reward
        neo=np.r_[0,np.cumsum(np.add.reduceat(ne.astype(int),off[:-1]))]
        arms={'PRM_native':(risk,reward<.5),'PRM_Q80':(risk,None)}
        for size in ['8B','4B']:
            f=RAW/'external_lr_prm_v1/scores'/f'LR_noncontrol_{size}_{bench}.npy'
            s=np.load(f);assert len(s)==ne.sum()
            full=np.full(len(y),np.nan);full[ne]=s;arms['LR_'+size]=(full,None)
        saved=RAW/'monday_cont_baselines_v1/ext'/f'{bench}_qwen3_8b/PREDICTIONS.npz'
        with np.load(saved) as z:
            assert np.array_equal(z['uids'].astype(str),uids)
            assert np.array_equal(z['nonempty'].astype(bool),ne)
            assert np.array_equal(z['lengths'],np.diff(off))
            for method in ['LSML_CONT','IU_PCR']:
                s=z['N16__'+method+'__score'];assert len(s)==len(y)
                arms[method+'_8B_DRIVE_BASELINE_EXPORT']=(s,None)
        step_index=np.concatenate([np.arange(1,b-a+1,dtype=float) for a,b in zip(off[:-1],off[1:])])
        arms['STEP_INDEX']=(step_index,None)
        for name,(s,flags) in arms.items():
            if flags is None:
                flags=np.ones(len(y),bool);flags[ne]=q80(s[ne],neo,True)[0]
            c,unused=measurements(s,flags,y,off,include)
            au=np.array([auc(y[a:b][include[a:b]&ne[a:b]],s[a:b][include[a:b]&ne[a:b]]) for a,b in zip(off[:-1],off[1:])])
            row={'benchmark':bench,'arm':name,'official_metric':float(metric(c.sum(0),bench)),
                 'within_auc':float(np.nanmean(au)),'n_answers':len(uids),'n_auc':int(np.isfinite(au).sum()),'n_steps':int(include.sum())}
            results.append(row);print('EXTERNAL',row,flush=True)
    write_csv('EXTERNAL_METRICS.csv',results)


if __name__=='__main__':
    t=time.time()
    primary_replay();external_replay()
    tracked=[HERE/'replay.py',ROOT/'spectral_utils/fusion_utils.py']
    tracked+=list((HERE/'inputs').glob('*.tar'))+list((HERE/'inputs').glob('*gold.json'))
    tracked+=[ROOT/'results/advisor_completion_2026-10-11/inputs'/s for s in ['population.npz','prmbench_meta.json']]
    write_json('REPLAY_MANIFEST.json',{'seconds':time.time()-t,'interpreter':sys.executable,'seed':SEED,'bootstrap_draws':DRAWS,
                                     'scope':'PRM7B and external PRM/LR scores only. Other four PRM raw banks not located on Drive.',
                                     'files':[{'path':str(p),'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in tracked]})
